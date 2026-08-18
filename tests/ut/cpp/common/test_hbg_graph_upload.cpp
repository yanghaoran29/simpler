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

#include "../host/runtime_maker.cpp"

namespace {
thread_local PTO2Runtime *bound_runtime;

struct UploadApi {
    std::vector<std::unique_ptr<std::byte[]>> allocations;
    int copies = 0;
    int fail_copy = 2;

    void *allocate(size_t bytes) {
        allocations.push_back(std::make_unique<std::byte[]>(bytes));
        return allocations.back().get();
    }

    static const HostApiOps &ops() {
        static const HostApiOps result = [] {
            HostApiOps ops{};
            ops.acquire_graph_definition_buffer = [](void *ctx, uint32_t, uint64_t, size_t bytes, size_t) {
                return static_cast<UploadApi *>(ctx)->allocate(bytes);
            };
            ops.acquire_graph_execution_buffer = ops.acquire_graph_submission_buffer =
                [](void *ctx, uint32_t, uint64_t, uint32_t, size_t bytes, size_t) {
                    return static_cast<UploadApi *>(ctx)->allocate(bytes);
                };
            ops.copy_to_device = [](void *ctx, void *dst, const void *src, size_t bytes) {
                auto &api = *static_cast<UploadApi *>(ctx);
                if (++api.copies == api.fail_copy) return -1;
                std::memcpy(dst, src, bytes);
                return 0;
            };
            return ops;
        }();
        return result;
    }
};

class GraphUploadTest : public ::testing::TestWithParam<bool> {};

TEST_P(GraphUploadTest, OnlySuccessfulOrchestrationUploadsTheSmImage) {
    Runtime runtime;
    runtime.set_worker_count(PLATFORM_CORES_PER_BLOCKDIM);
    uint64_t heap_sizes[PTO2_MAX_RING_DEPTH];
    uint64_t window_sizes[PTO2_MAX_RING_DEPTH];
    std::fill_n(heap_sizes, PTO2_MAX_RING_DEPTH, 4096);
    std::fill_n(window_sizes, PTO2_MAX_RING_DEPTH, 128);
    DeviceArena host_arena;
    const auto layout = runtime_reserve_layout(host_arena, window_sizes, heap_sizes);
    ASSERT_NE(host_arena.commit(), nullptr);
    std::vector<std::byte> sm(PTO2SharedMemoryHandle::calculate_size_per_ring(window_sizes));
    std::vector<std::byte> heap(4096 * PTO2_MAX_RING_DEPTH);
    std::vector<std::byte> device_arena(layout.arena_size);
    auto *rt = runtime_init_data_from_layout(
        host_arena, layout, PTO2_MODE_EXECUTE, sm.data(), sm.size(), heap.data(), heap_sizes
    );
    ASSERT_NE(rt, nullptr);
    runtime_wire_arena_pointers(host_arena, layout, rt);
    UploadApi fake;
    fake.fail_copy = GetParam() ? 2 : 0;
    HostApi api(&fake, 0, 0, &UploadApi::ops());
    HostTensorAccessor accessor(&api);
    HostOrchEntryPoints entries;
    entries.bind = [](PTO2Runtime *runtime) {
        bound_runtime = runtime;
    };
    entries.entry = [](const ChipTaskArgs &) {
        uint32_t storage[4]{};
        uint32_t shape[] = {4};
        ChipTensor tensor = make_tensor_external(storage, shape, 1);
        CoreTaskArgs args;
        args.add_input(tensor);
        auto &orch = bound_runtime->orchestrator;
        EXPECT_TRUE(orch.graph_begin(17, args, 23).recording);
        EXPECT_TRUE(orch.submit_dummy_task(args).task_id().is_valid());
        orch.graph_end();
    };
    ChipTaskArgs args;
    const int result = run_host_orchestration(
        &runtime, &api, accessor, rt, host_arena, layout, sm.data(), sm.size(), device_arena.data(), heap.data(),
        heap_sizes, window_sizes, &entries, args
    );
    if (GetParam()) {
        EXPECT_LT(result, 0);
        EXPECT_EQ(fake.copies, 2);
    } else {
        EXPECT_EQ(result, 1);
        EXPECT_EQ(fake.copies, 6);
    }
    EXPECT_EQ(rt->orchestrator.fatal, GetParam());
    EXPECT_EQ(rt->orchestrator.graph_host_state, nullptr);
    rt->orchestrator.destroy();
    rt->scheduler.destroy();
}
INSTANTIATE_TEST_SUITE_P(UploadResult, GraphUploadTest, ::testing::Bool());
}  // namespace
