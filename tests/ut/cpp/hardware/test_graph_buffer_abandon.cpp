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

#include <cstdlib>
#include <string>

#include "acl/acl.h"
#include "device_runner.h"

namespace {
class GraphBufferRunner : public DeviceRunner {
public:
    using DeviceRunnerBase::abandon_graph_execution_buffers;
    using DeviceRunnerBase::release_graph_execution_buffers;

    size_t allocations() const { return mem_alloc_.get_allocation_count(); }
    bool graph_caches_empty() const {
        for (const auto &cache : graph_execution_buffers_)
            if (!cache.empty()) return false;
        for (const auto &cache : graph_definition_buffers_)
            if (!cache.empty()) return false;
        for (const auto &cache : graph_submission_buffers_)
            if (!cache.empty()) return false;
        return true;
    }
};

class GraphBufferAbandonTest : public ::testing::Test {
protected:
    int device = -1;
    void SetUp() override {
        const char *resource = std::getenv("CTEST_RESOURCE_GROUP_0_NPUS");
        ASSERT_NE(resource, nullptr) << "Run with CTest NPU resource allocation";
        const std::string spec(resource);
        const auto id = spec.find("id:");
        ASSERT_NE(id, std::string::npos);
        device = std::stoi(spec.substr(id + 3));
        ASSERT_EQ(aclInit(nullptr), ACL_SUCCESS);
        ASSERT_EQ(aclrtSetDevice(device), ACL_SUCCESS);
    }
    void TearDown() override {
        if (device >= 0) aclrtResetDevice(device);
        aclFinalize();
    }
};

TEST_F(GraphBufferAbandonTest, ClearsAllSlotsWithoutFreeingAndReacquiresFreshStorage) {
    GraphBufferRunner runner;
    const uint64_t keys[] = {0x100000017ULL, 0x200000017ULL};
    for (uint32_t slot = 0; slot < PTO_PIPELINE_MAX_DEPTH; ++slot) {
        for (uint64_t key : keys) {
            ASSERT_NE(runner.acquire_graph_definition_buffer(slot, key, 128, 64), nullptr);
            for (uint32_t occurrence = 0; occurrence < 2; ++occurrence) {
                ASSERT_NE(runner.acquire_graph_execution_buffer(slot, key, occurrence, 128, 64), nullptr);
                ASSERT_NE(runner.acquire_graph_submission_buffer(slot, key, occurrence, 128, 64), nullptr);
            }
        }
    }
    const auto before = runner.allocations();
    EXPECT_EQ(before, PTO_PIPELINE_MAX_DEPTH * 10u);
    runner.abandon_graph_execution_buffers();
    EXPECT_TRUE(runner.graph_caches_empty());
    // Allocations stay live in this test so any accidental free is observable.
    EXPECT_EQ(runner.allocations(), before);
    ASSERT_NE(runner.acquire_graph_submission_buffer(0, keys[0], 0, 128, 64), nullptr);
    EXPECT_EQ(runner.allocations(), before + 1);
    runner.release_graph_execution_buffers();
    EXPECT_TRUE(runner.graph_caches_empty());
    EXPECT_EQ(runner.allocations(), before);
}
}  // namespace
