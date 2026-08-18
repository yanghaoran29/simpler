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

#include <gtest/gtest.h>

#include <condition_variable>
#include <cstring>
#include <thread>
#include <type_traits>

#include "host_build_graph/graph_pinned_pool.h"

namespace {
using hbg::GraphPinnedPool;
static_assert(!std::is_copy_constructible_v<GraphPinnedPool::Lease>);
static_assert(std::is_nothrow_move_constructible_v<GraphPinnedPool::Lease>);

TEST(GraphPinnedPoolTest, ConcurrentLeasesKeepIndependentBytes) {
    GraphPinnedPool pool(nullptr, nullptr, 1024);
    std::mutex mutex;
    std::condition_variable cv;
    int arrived = 0;
    std::byte *addresses[2]{};
    auto run = [&](int index) {
        auto lease = pool.acquire();
        EXPECT_NE(lease.data(), nullptr);
        addresses[index] = lease.data();
        EXPECT_EQ(reinterpret_cast<uintptr_t>(lease.data()) % 64, 0);
        if (lease.data() != nullptr) std::memset(lease.data(), index + 1, lease.capacity());
        {
            std::unique_lock<std::mutex> lock(mutex);
            ++arrived;
            cv.notify_all();
            cv.wait(lock, [&] {
                return arrived == 2;
            });
        }
        for (size_t i = 0; i < lease.capacity(); ++i) {
            EXPECT_EQ(lease.data()[i], std::byte(index + 1));
        }
    };
    std::thread first(run, 0);
    std::thread second(run, 1);
    first.join();
    second.join();
    EXPECT_NE(addresses[0], addresses[1]);
}

TEST(GraphPinnedPoolTest, MoveAndEarlyExitReturnExactlyOneLease) {
    GraphPinnedPool pool(nullptr, nullptr, 1024);
    auto first = pool.acquire();
    auto *first_address = first.data();
    auto second = pool.acquire();
    auto *second_address = second.data();
    second = std::move(first);
    EXPECT_EQ(first.data(), nullptr);
    EXPECT_EQ(second.data(), first_address);
    auto returned = pool.acquire();
    EXPECT_EQ(returned.data(), second_address);
    returned = {};
    try {
        auto temporary = pool.acquire();
        EXPECT_EQ(temporary.data(), second_address);
        throw 17;
    } catch (int) {}
    auto reused = pool.acquire();
    EXPECT_EQ(reused.data(), second_address);
    EXPECT_NE(reused.data(), second.data());
}

int allocations;
int frees;

int allocate_pinned(void **data, size_t bytes) {
    ++allocations;
    *data = ::operator new[](bytes, std::align_val_t{64}, std::nothrow);
    return *data == nullptr ? -1 : 0;
}

int free_pinned(void *data) {
    ++frees;
    ::operator delete[](data, std::align_val_t{64});
    return 0;
}

TEST(GraphPinnedPoolTest, RetainsUntilPoolDestructionWithMatchingFree) {
    allocations = frees = 0;
    {
        GraphPinnedPool pool(allocate_pinned, free_pinned, 1024);
        for (int round = 0; round < 5; ++round) {
            auto first = pool.acquire();
            auto second = pool.acquire();
            EXPECT_NE(first.data(), second.data());
        }
        EXPECT_EQ(allocations, 2);
        EXPECT_EQ(frees, 0);
    }
    EXPECT_EQ(frees, 2);
}

TEST(GraphPinnedPoolTest, AllocationFailureLeavesAnEmptyLeaseAndCanRecover) {
    allocations = frees = 0;
    GraphPinnedPool pool(
        [](void **data, size_t bytes) {
            if (allocations++ == 0) return -1;
            *data = ::operator new[](bytes, std::align_val_t{64}, std::nothrow);
            return *data == nullptr ? -1 : 0;
        },
        free_pinned, 1024
    );
    auto failed = pool.acquire();
    EXPECT_EQ(failed.data(), nullptr);
    EXPECT_EQ(failed.capacity(), 0);
    auto recovered = pool.acquire();
    EXPECT_NE(recovered.data(), nullptr);
    EXPECT_EQ(recovered.capacity(), 1024);
}
}  // namespace
