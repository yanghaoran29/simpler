/*
 * Copyright (c) PyPTO Contributors.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 * -----------------------------------------------------------------------------------------------------------
 */

#pragma once

#include <dlfcn.h>

#include <cstddef>
#include <memory>
#include <mutex>
#include <new>
#include <utility>

namespace hbg {

// Each live lease exclusively owns its bytes. The pool must outlive its leases.
class GraphPinnedPool {
public:
    using Allocate = int (*)(void **, size_t);
    using Free = int (*)(void *);
    static constexpr size_t arena_bytes = 16ull * 1024 * 1024;

private:
    struct Arena {
        std::byte *data = nullptr;
        Free free_pinned = nullptr;
        std::unique_ptr<Arena> next;

        ~Arena() {
            if (data == nullptr) return;
            if (free_pinned != nullptr) {
                (void)free_pinned(data);
            } else {
                ::operator delete[](data, std::align_val_t{64});
            }
        }
    };

public:
    class Lease {
    public:
        Lease() = default;
        Lease(const Lease &) = delete;
        Lease &operator=(const Lease &) = delete;
        Lease(Lease &&other) noexcept :
            pool_(other.pool_),
            arena_(std::move(other.arena_)) {}
        Lease &operator=(Lease &&other) noexcept {
            if (this != &other) {
                release();
                pool_ = other.pool_;
                arena_ = std::move(other.arena_);
            }
            return *this;
        }
        ~Lease() { release(); }

        std::byte *data() const { return arena_ ? arena_->data : nullptr; }
        size_t capacity() const { return arena_ ? pool_->bytes_ : 0; }

    private:
        friend class GraphPinnedPool;
        Lease(GraphPinnedPool *pool, std::unique_ptr<Arena> arena) :
            pool_(pool),
            arena_(std::move(arena)) {}

        void release() {
            if (!arena_) return;
            std::lock_guard<std::mutex> lock(pool_->mutex_);
            arena_->next = std::move(pool_->idle_);
            pool_->idle_ = std::move(arena_);
        }

        GraphPinnedPool *pool_ = nullptr;
        std::unique_ptr<Arena> arena_;
    };

    GraphPinnedPool() {
        handle_ = dlopen("libascendcl.so", RTLD_NOW | RTLD_LOCAL);
        if (handle_ != nullptr) {
            allocate_ = reinterpret_cast<Allocate>(dlsym(handle_, "aclrtMallocHost"));
            free_ = reinterpret_cast<Free>(dlsym(handle_, "aclrtFreeHost"));
            // A pinned allocation is usable only with its matching deallocator.
            if (free_ == nullptr) allocate_ = nullptr;
        }
    }

    GraphPinnedPool(Allocate allocate, Free free, size_t bytes = arena_bytes) :
        allocate_(free != nullptr ? allocate : nullptr),
        free_(free),
        bytes_(bytes) {}

    GraphPinnedPool(const GraphPinnedPool &) = delete;
    GraphPinnedPool &operator=(const GraphPinnedPool &) = delete;

    ~GraphPinnedPool() {
        while (idle_) {
            auto arena = std::move(idle_);
            idle_ = std::move(arena->next);
        }
        if (handle_ != nullptr) dlclose(handle_);
    }

    Lease acquire() {
        {
            std::lock_guard<std::mutex> lock(mutex_);
            if (idle_) {
                auto arena = std::move(idle_);
                idle_ = std::move(arena->next);
                return {this, std::move(arena)};
            }
        }
        auto arena = std::unique_ptr<Arena>(new (std::nothrow) Arena);
        if (!arena || bytes_ == 0) return {};
        if (allocate_ != nullptr) {
            void *data = nullptr;
            if (allocate_(&data, bytes_) != 0 || data == nullptr) return {};
            arena->data = static_cast<std::byte *>(data);
            arena->free_pinned = free_;
        } else {
            arena->data = static_cast<std::byte *>(::operator new[](bytes_, std::align_val_t{64}, std::nothrow));
            if (arena->data == nullptr) return {};
        }
        return {this, std::move(arena)};
    }

private:
    void *handle_ = nullptr;
    Allocate allocate_ = nullptr;
    Free free_ = nullptr;
    size_t bytes_ = arena_bytes;
    std::mutex mutex_;
    std::unique_ptr<Arena> idle_;
};

inline GraphPinnedPool &graph_pinned_pool() {
    static GraphPinnedPool pool;
    return pool;
}

}  // namespace hbg
