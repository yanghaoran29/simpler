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

std::vector<int32_t>
clusters_for_logical(int32_t logical, int32_t cluster_count, int32_t active_threads, int32_t assignment_mode) {
    std::vector<int32_t> owned;
    for (int32_t ci = 0; ci < cluster_count; ++ci) {
        if (logical_sched_for_cluster(ci, cluster_count, active_threads, assignment_mode) == logical) {
            owned.push_back(ci);
        }
    }
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

    EXPECT_EQ(
        clusters_for_logical(0, kClusterCount, kActiveSched, kSchedAicoreAssignmentDieAware),
        (std::vector<int32_t>{0, 1, 2, 3, 4, 5, 6, 7, 8})
    );
    EXPECT_EQ(
        clusters_for_logical(1, kClusterCount, kActiveSched, kSchedAicoreAssignmentDieAware),
        (std::vector<int32_t>{9, 10, 11, 12, 13, 14, 15, 16, 17})
    );
    EXPECT_EQ(
        clusters_for_logical(2, kClusterCount, kActiveSched, kSchedAicoreAssignmentDieAware),
        (std::vector<int32_t>{18, 19, 20, 21, 22, 23, 24, 25, 26})
    );
    EXPECT_EQ(
        clusters_for_logical(3, kClusterCount, kActiveSched, kSchedAicoreAssignmentDieAware),
        (std::vector<int32_t>{27, 28, 29, 30, 31, 32, 33, 34, 35})
    );
}

TEST(A5SchedAicoreAssignment, ContiguousBlocksForTwentyEightClustersAndFourSchedulers) {
    constexpr int32_t kClusterCount = 28;
    constexpr int32_t kActiveSched = 4;

    EXPECT_EQ(
        clusters_for_logical(0, kClusterCount, kActiveSched, kSchedAicoreAssignmentDieAware),
        (std::vector<int32_t>{0, 1, 2, 3, 4, 5, 6})
    );
    EXPECT_EQ(
        clusters_for_logical(1, kClusterCount, kActiveSched, kSchedAicoreAssignmentDieAware),
        (std::vector<int32_t>{7, 8, 9, 10, 11, 12, 13})
    );
    EXPECT_EQ(
        clusters_for_logical(2, kClusterCount, kActiveSched, kSchedAicoreAssignmentDieAware),
        (std::vector<int32_t>{14, 15, 16, 17, 18, 19, 20})
    );
    EXPECT_EQ(
        clusters_for_logical(3, kClusterCount, kActiveSched, kSchedAicoreAssignmentDieAware),
        (std::vector<int32_t>{21, 22, 23, 24, 25, 26, 27})
    );
}

TEST(A5SchedAicoreAssignment, UnevenCoreCountUsesBalancedContiguousRanges) {
    constexpr int32_t kClusterCount = 30;
    constexpr int32_t kActiveSched = 4;

    EXPECT_EQ(
        clusters_for_logical(0, kClusterCount, kActiveSched, kSchedAicoreAssignmentDieAware),
        (std::vector<int32_t>{0, 1, 2, 3, 4, 5, 6, 7})
    );
    EXPECT_EQ(
        clusters_for_logical(1, kClusterCount, kActiveSched, kSchedAicoreAssignmentDieAware),
        (std::vector<int32_t>{8, 9, 10, 11, 12, 13, 14})
    );
    EXPECT_EQ(
        clusters_for_logical(2, kClusterCount, kActiveSched, kSchedAicoreAssignmentDieAware),
        (std::vector<int32_t>{15, 16, 17, 18, 19, 20, 21, 22})
    );
    EXPECT_EQ(
        clusters_for_logical(3, kClusterCount, kActiveSched, kSchedAicoreAssignmentDieAware),
        (std::vector<int32_t>{23, 24, 25, 26, 27, 28, 29})
    );
}

TEST(A5SchedAicoreAssignment, UnknownDieFallbackUsesGlobalDynamicBoundaries) {
    constexpr int32_t kClusterCount = 29;
    constexpr int32_t kActiveSched = 4;

    EXPECT_EQ(
        clusters_for_logical(0, kClusterCount, kActiveSched, kSchedAicoreAssignmentSequential),
        (std::vector<int32_t>{0, 1, 2, 3, 4, 5, 6, 7})
    );
    EXPECT_EQ(
        clusters_for_logical(1, kClusterCount, kActiveSched, kSchedAicoreAssignmentSequential),
        (std::vector<int32_t>{8, 9, 10, 11, 12, 13, 14})
    );
    EXPECT_EQ(
        clusters_for_logical(2, kClusterCount, kActiveSched, kSchedAicoreAssignmentSequential),
        (std::vector<int32_t>{15, 16, 17, 18, 19, 20, 21})
    );
    EXPECT_EQ(
        clusters_for_logical(3, kClusterCount, kActiveSched, kSchedAicoreAssignmentSequential),
        (std::vector<int32_t>{22, 23, 24, 25, 26, 27, 28})
    );
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

TEST(A5SchedAicoreAssignment, DependencyDieModesPreserveTheirBaseOwnership) {
    constexpr int32_t kClusterCount = 28;
    constexpr int32_t kActiveSched = 4;

    EXPECT_TRUE(sched_assignment_uses_dependency_die_affinity(kSchedAicoreAssignmentRoundRobinDependencyDieAware));
    EXPECT_TRUE(sched_assignment_uses_dependency_die_affinity(kSchedAicoreAssignmentHybridDependencyDieAware));
    EXPECT_FALSE(sched_assignment_uses_dependency_die_affinity(kSchedAicoreAssignmentRoundRobin));

    for (int32_t t = 0; t < kActiveSched; ++t) {
        EXPECT_EQ(
            clusters_for_logical(t, kClusterCount, kActiveSched, kSchedAicoreAssignmentRoundRobinDependencyDieAware),
            clusters_for_logical(t, kClusterCount, kActiveSched, kSchedAicoreAssignmentRoundRobin)
        );
        EXPECT_EQ(
            clusters_for_logical(t, kClusterCount, kActiveSched, kSchedAicoreAssignmentHybridDependencyDieAware),
            clusters_for_logical(t, kClusterCount, kActiveSched, kSchedAicoreAssignmentHybridDieAware)
        );
    }
}

TEST(A5SchedAicoreAssignment, ContiguousDataAwareModePreservesDieAwareOwnership) {
    constexpr int32_t kClusterCount = 28;
    constexpr int32_t kActiveSched = 4;

    EXPECT_TRUE(sched_assignment_uses_explicit_data_affinity(kSchedAicoreAssignmentContiguousDataAware));
    EXPECT_EQ(sched_assignment_base_mode(kSchedAicoreAssignmentContiguousDataAware), kSchedAicoreAssignmentDieAware);
    for (int32_t t = 0; t < kActiveSched; ++t) {
        EXPECT_EQ(
            clusters_for_logical(t, kClusterCount, kActiveSched, kSchedAicoreAssignmentContiguousDataAware),
            clusters_for_logical(t, kClusterCount, kActiveSched, kSchedAicoreAssignmentDieAware)
        );
    }
}

TEST(A5SchedAicoreAssignment, HybridDieAwareTwentyEightClustersMatchesRequestedLayout) {
    constexpr int32_t kClusterCount = 28;
    constexpr int32_t kActiveSched = 4;
    constexpr int32_t kMode = kSchedAicoreAssignmentHybridDieAware;

    EXPECT_EQ(clusters_for_logical(0, kClusterCount, kActiveSched, kMode), (std::vector<int32_t>{0, 1, 2, 3, 4, 5, 6}));
    EXPECT_EQ(
        clusters_for_logical(1, kClusterCount, kActiveSched, kMode), (std::vector<int32_t>{7, 8, 9, 10, 14, 15, 16})
    );
    EXPECT_EQ(
        clusters_for_logical(2, kClusterCount, kActiveSched, kMode), (std::vector<int32_t>{11, 12, 13, 17, 18, 19, 20})
    );
    EXPECT_EQ(
        clusters_for_logical(3, kClusterCount, kActiveSched, kMode), (std::vector<int32_t>{21, 22, 23, 24, 25, 26, 27})
    );
}

TEST(A5SchedAicoreAssignment, HybridDieAwareThirtySixClustersKeepsMiddleSchedulersBalanced) {
    constexpr int32_t kClusterCount = 36;
    constexpr int32_t kActiveSched = 4;
    constexpr int32_t kMode = kSchedAicoreAssignmentHybridDieAware;

    EXPECT_EQ(
        clusters_for_logical(0, kClusterCount, kActiveSched, kMode), (std::vector<int32_t>{0, 1, 2, 3, 4, 5, 6, 7, 8})
    );
    EXPECT_EQ(
        clusters_for_logical(1, kClusterCount, kActiveSched, kMode),
        (std::vector<int32_t>{9, 10, 11, 12, 13, 18, 19, 20, 21})
    );
    EXPECT_EQ(
        clusters_for_logical(2, kClusterCount, kActiveSched, kMode),
        (std::vector<int32_t>{14, 15, 16, 17, 22, 23, 24, 25, 26})
    );
    EXPECT_EQ(
        clusters_for_logical(3, kClusterCount, kActiveSched, kMode),
        (std::vector<int32_t>{27, 28, 29, 30, 31, 32, 33, 34, 35})
    );
}

TEST(A5SchedAicoreAssignment, HybridDieAwareThirtyClustersUsesDynamicBalancedQuotas) {
    constexpr int32_t kClusterCount = 30;
    constexpr int32_t kActiveSched = 4;
    constexpr int32_t kMode = kSchedAicoreAssignmentHybridDieAware;

    EXPECT_EQ(clusters_for_logical(0, kClusterCount, kActiveSched, kMode).size(), 8U);
    EXPECT_EQ(clusters_for_logical(1, kClusterCount, kActiveSched, kMode).size(), 7U);
    EXPECT_EQ(clusters_for_logical(2, kClusterCount, kActiveSched, kMode).size(), 8U);
    EXPECT_EQ(clusters_for_logical(3, kClusterCount, kActiveSched, kMode).size(), 7U);
    EXPECT_EQ(
        clusters_for_logical(1, kClusterCount, kActiveSched, kMode), (std::vector<int32_t>{8, 9, 10, 15, 16, 17, 18})
    );
    EXPECT_EQ(
        clusters_for_logical(2, kClusterCount, kActiveSched, kMode),
        (std::vector<int32_t>{11, 12, 13, 14, 19, 20, 21, 22})
    );
}

TEST(A5SchedAicoreAssignment, HybridFallsBackToBalancedContiguousWithoutFourSchedulers) {
    constexpr int32_t kClusterCount = 28;
    constexpr int32_t kActiveSched = 3;

    for (int32_t ci = 0; ci < kClusterCount; ++ci) {
        EXPECT_EQ(
            logical_sched_for_cluster(ci, kClusterCount, kActiveSched, kSchedAicoreAssignmentHybridDieAware),
            logical_sched_for_cluster(ci, kClusterCount, kActiveSched, kSchedAicoreAssignmentSequential)
        );
    }
}
