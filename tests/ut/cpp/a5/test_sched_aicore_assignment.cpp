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

#include <cstdint>
#include <vector>

#include "common/platform_config.h"
#include "common/sched_aicore_assignment.h"

namespace {

std::vector<int32_t> clusters_for_logical(
    int32_t logical, int32_t cluster_count, int32_t active_threads, int32_t assignment_mode
) {
    std::vector<int32_t> owned;
    for (int32_t ci = 0; ci < cluster_count; ++ci) {
        if (logical_sched_for_cluster(ci, cluster_count, active_threads, assignment_mode) == logical) {
            owned.push_back(ci);
        }
    }
    return owned;
}

std::vector<int32_t> clusters_for_pthread(
    int32_t pthread_tidx,
    int32_t cluster_count,
    int32_t active_threads,
    int32_t assignment_mode,
    const int32_t *pthread_to_logical,
    const int32_t *logical_to_pthread
) {
    std::vector<int32_t> owned;
    for (int32_t ci = 0; ci < cluster_count; ++ci) {
        if (cluster_owned_by_pthread(
                ci, pthread_tidx, cluster_count, active_threads, assignment_mode, pthread_to_logical
            )) {
            owned.push_back(ci);
        }
    }
    (void)logical_to_pthread;
    return owned;
}

}  // namespace

TEST(A5AicoreClusterDie, LowerHalfDie0UpperHalfDie1) {
    EXPECT_EQ(aicore_cluster_die(0, 36), 0);
    EXPECT_EQ(aicore_cluster_die(17, 36), 0);
    EXPECT_EQ(aicore_cluster_die(18, 36), 1);
    EXPECT_EQ(aicore_cluster_die(35, 36), 1);
    EXPECT_EQ(aicore_cluster_die(13, 28), 0);
    EXPECT_EQ(aicore_cluster_die(14, 28), 1);
}

TEST(A5SchedAicoreAssignment, ContiguousBlocksForThirtySixClustersAndFourSchedulers) {
    constexpr int32_t kClusterCount = 36;
    constexpr int32_t kActiveSched = 4;

    EXPECT_EQ(clusters_for_logical(0, kClusterCount, kActiveSched, kSchedAicoreAssignmentDieAware),
              (std::vector<int32_t>{0, 1, 2, 3, 4, 5, 6, 7, 8}));
    EXPECT_EQ(clusters_for_logical(1, kClusterCount, kActiveSched, kSchedAicoreAssignmentDieAware),
              (std::vector<int32_t>{9, 10, 11, 12, 13, 14, 15, 16, 17}));
    EXPECT_EQ(clusters_for_logical(2, kClusterCount, kActiveSched, kSchedAicoreAssignmentDieAware),
              (std::vector<int32_t>{18, 19, 20, 21, 22, 23, 24, 25, 26}));
    EXPECT_EQ(clusters_for_logical(3, kClusterCount, kActiveSched, kSchedAicoreAssignmentDieAware),
              (std::vector<int32_t>{27, 28, 29, 30, 31, 32, 33, 34, 35}));
}

TEST(A5SchedAicoreAssignment, RoundRobinBaselineMatchesMainline) {
    constexpr int32_t kClusterCount = 36;
    constexpr int32_t kActiveSched = 4;
    constexpr int32_t kMode = kSchedAicoreAssignmentRoundRobin;

    for (int32_t t = 0; t < kActiveSched; ++t) {
        std::vector<int32_t> expected;
        for (int32_t ci = 0; ci < kClusterCount; ++ci) {
            if (ci % kActiveSched == t) expected.push_back(ci);
        }
        EXPECT_EQ(clusters_for_logical(t, kClusterCount, kActiveSched, kMode), expected);
    }
}

TEST(A5SchedAicoreAssignment, RttDieRankMapsMostDie0AffinitiveToLogicalZeroOne) {
    constexpr int32_t kClusterCount = 36;
    constexpr int32_t kActiveSched = 4;
    int32_t pthread_to_logical[kActiveSched];
    int32_t logical_to_pthread[kActiveSched];
    const int64_t deltas[kActiveSched] = {
        100,   // pthread 0: strong die1 bias
        -50,   // pthread 1: die0
        -80,   // pthread 2: strongest die0
        20,    // pthread 3: die1
    };
    assign_rtt_die_logical_order(deltas, kActiveSched, pthread_to_logical, logical_to_pthread);

    EXPECT_EQ(pthread_to_logical[2], 0);
    EXPECT_EQ(pthread_to_logical[1], 1);
    EXPECT_EQ(pthread_to_logical[3], 2);
    EXPECT_EQ(pthread_to_logical[0], 3);
    EXPECT_EQ(logical_to_pthread[0], 2);
    EXPECT_EQ(logical_to_pthread[1], 1);
    EXPECT_EQ(logical_to_pthread[2], 3);
    EXPECT_EQ(logical_to_pthread[3], 0);

    EXPECT_EQ(clusters_for_pthread(2, kClusterCount, kActiveSched, kSchedAicoreAssignmentRttDieAware, pthread_to_logical,
                                   logical_to_pthread),
              clusters_for_logical(0, kClusterCount, kActiveSched, kSchedAicoreAssignmentDieAware));
    EXPECT_EQ(clusters_for_pthread(0, kClusterCount, kActiveSched, kSchedAicoreAssignmentRttDieAware, pthread_to_logical,
                                   logical_to_pthread),
              clusters_for_logical(3, kClusterCount, kActiveSched, kSchedAicoreAssignmentDieAware));
}

TEST(A5SchedAicoreAssignment, RttDieRankTieBreaksOnPthreadIndex) {
    constexpr int32_t kActiveSched = 4;
    int32_t pthread_to_logical[kActiveSched];
    int32_t logical_to_pthread[kActiveSched];
    const int64_t deltas[kActiveSched] = {0, 0, 0, 0};
    assign_rtt_die_logical_order(deltas, kActiveSched, pthread_to_logical, logical_to_pthread);

    EXPECT_EQ(pthread_to_logical[0], 0);
    EXPECT_EQ(pthread_to_logical[1], 1);
    EXPECT_EQ(pthread_to_logical[2], 2);
    EXPECT_EQ(pthread_to_logical[3], 3);
}
