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

#include "pto_orchestration_api.h"

namespace {
bool fatal;
bool fail_begin;
bool fail_end;
bool end_success;
int body_calls;
int commits;
PTO2Runtime runtime{};

class GraphFallbackTest : public ::testing::Test {
protected:
    void SetUp() override {
        fatal = fail_begin = fail_end = end_success = false;
        body_calls = commits = 0;
        static const PTO2RuntimeOps ops = [] {
            PTO2RuntimeOps result{};
            result.is_fatal = [](PTO2Runtime *) {
                return fatal;
            };
            result.graph_begin = [](PTO2Runtime *, uint64_t, const CoreTaskArgs &) {
                GraphScopeResult graph;
                if (fail_begin) {
                    fatal = true;
                } else {
                    graph.recording = true;
                    graph.execute_block = false;
                }
                return graph;
            };
            result.graph_end = [](PTO2Runtime *) {
                fatal = fail_end;
                return end_success;
            };
            result.graph_commit = [](PTO2Runtime *) {
                ++commits;
            };
            return result;
        }();
        runtime.ops = &ops;
    }

    void submit(bool cacheable = true) {
        CoreTaskArgs args;
        uint32_t storage[4]{};
        uint32_t shape[] = {4};
        ChipTensor tensor = make_tensor_external(storage, shape, 1);
        if (cacheable) args.add_input(tensor);
        rt_submit_graph_impl(17, args, [] {
            ++body_calls;
        });
    }
};
}  // namespace

extern "C" PTO2Runtime *framework_current_runtime() { return &runtime; }

TEST_F(GraphFallbackTest, FatalDuringRecordingEndDoesNotRepeatBody) {
    fail_end = true;
    submit();
    EXPECT_TRUE(fatal);
    EXPECT_EQ(body_calls, 1);
    EXPECT_EQ(commits, 0);
}

TEST_F(GraphFallbackTest, FatalCacheHitDoesNotInvokeBody) {
    fail_begin = true;
    submit();
    EXPECT_TRUE(fatal);
    EXPECT_EQ(body_calls, 0);
}

TEST_F(GraphFallbackTest, FatalUncacheableSubmissionDoesNotInvokeBody) {
    fatal = true;
    submit(false);
    EXPECT_EQ(body_calls, 0);
}

TEST_F(GraphFallbackTest, UnsupportedRecordingFallsBack) {
    submit();
    EXPECT_FALSE(fatal);
    EXPECT_EQ(body_calls, 2);
    EXPECT_EQ(commits, 1);
}

TEST_F(GraphFallbackTest, SuccessfulRecordingDoesNotRepeatBody) {
    end_success = true;
    submit();
    EXPECT_FALSE(fatal);
    EXPECT_EQ(body_calls, 1);
    EXPECT_EQ(commits, 1);
}
