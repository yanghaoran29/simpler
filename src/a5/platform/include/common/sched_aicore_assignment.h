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

#include <cstdint>

#include "common/platform_config.h"

constexpr int32_t kSchedAicoreAssignmentSequential = 0;
constexpr int32_t kSchedAicoreAssignmentDieAware = 1;
constexpr int32_t kSchedAicoreAssignmentRoundRobin = 2;
// Schedulers 0/3 stay on die0/die1 respectively. Schedulers 1/2 split
// the remaining clusters across both dies, biased toward their local die.
constexpr int32_t kSchedAicoreAssignmentHybridDieAware = 3;
// Experimental soft affinity modes. They preserve the underlying cluster
// ownership layout while preferring the die of the producer that made a
// consumer ready. Dispatch falls back to any owned idle core immediately.
constexpr int32_t kSchedAicoreAssignmentRoundRobinDependencyDieAware = 4;
constexpr int32_t kSchedAicoreAssignmentHybridDependencyDieAware = 5;
// Contiguous scheduler ownership with task-declared data affinity routed before
// schedulers compete for ready work.
constexpr int32_t kSchedAicoreAssignmentContiguousDataAware = 6;

inline bool sched_assignment_uses_dependency_die_affinity(int32_t assignment_mode) {
    return assignment_mode == kSchedAicoreAssignmentRoundRobinDependencyDieAware ||
           assignment_mode == kSchedAicoreAssignmentHybridDependencyDieAware;
}

inline bool sched_assignment_uses_explicit_data_affinity(int32_t assignment_mode) {
    return assignment_mode == kSchedAicoreAssignmentContiguousDataAware;
}

inline int32_t sched_assignment_base_mode(int32_t assignment_mode) {
    if (assignment_mode == kSchedAicoreAssignmentContiguousDataAware) {
        return kSchedAicoreAssignmentDieAware;
    }
    if (assignment_mode == kSchedAicoreAssignmentRoundRobinDependencyDieAware) {
        return kSchedAicoreAssignmentRoundRobin;
    }
    if (assignment_mode == kSchedAicoreAssignmentHybridDependencyDieAware) {
        return kSchedAicoreAssignmentHybridDieAware;
    }
    return assignment_mode;
}

inline bool sched_assignment_uses_contiguous_blocks(int32_t assignment_mode) {
    assignment_mode = sched_assignment_base_mode(assignment_mode);
    return assignment_mode != kSchedAicoreAssignmentRoundRobin &&
           assignment_mode != kSchedAicoreAssignmentHybridDieAware;
}

inline int32_t balanced_sched_boundary(int32_t logical_sched, int32_t cluster_count, int32_t active_threads) {
    return static_cast<int32_t>(
        (static_cast<int64_t>(logical_sched) * cluster_count + active_threads - 1) / active_threads
    );
}

inline int32_t hybrid_middle_die0_count(int32_t cluster_count) {
    const int32_t die0_count = cluster_count / 2;
    const int32_t die1_count = cluster_count - die0_count;
    const int32_t sched0_count = balanced_sched_boundary(1, cluster_count, 4);
    const int32_t sched1_count =
        balanced_sched_boundary(2, cluster_count, 4) - balanced_sched_boundary(1, cluster_count, 4);
    const int32_t sched2_count =
        balanced_sched_boundary(3, cluster_count, 4) - balanced_sched_boundary(2, cluster_count, 4);
    const int32_t sched3_count = cluster_count - balanced_sched_boundary(3, cluster_count, 4);
    const int32_t remaining_die0 = die0_count - sched0_count;
    const int32_t remaining_die1 = die1_count - sched3_count;

    // x is Scheduler 1's die0 share. Search the valid range for the split
    // that keeps both middle schedulers closest to half on each die. Ties
    // favor a larger x so Scheduler 1 remains biased toward die0.
    int32_t lower = sched1_count - remaining_die1;
    if (lower < 0) lower = 0;
    int32_t upper = remaining_die0 < sched1_count ? remaining_die0 : sched1_count;
    int32_t best_x = lower;
    int32_t best_score = 0x7fffffff;
    for (int32_t x = lower; x <= upper; ++x) {
        const int32_t sched1_die1 = sched1_count - x;
        const int32_t sched2_die1 = remaining_die1 - sched1_die1;
        const int32_t sched1_skew = 2 * x - sched1_count;
        const int32_t sched2_skew = 2 * sched2_die1 - sched2_count;
        const int32_t score =
            (sched1_skew < 0 ? -sched1_skew : sched1_skew) + (sched2_skew < 0 ? -sched2_skew : sched2_skew);
        if (score <= best_score) {
            best_score = score;
            best_x = x;
        }
    }
    return best_x;
}

inline int32_t hybrid_logical_sched_for_cluster(int32_t ci, int32_t cluster_count) {
    const int32_t die0_count = cluster_count / 2;
    const int32_t sched0_count = balanced_sched_boundary(1, cluster_count, 4);
    const int32_t sched1_count =
        balanced_sched_boundary(2, cluster_count, 4) - balanced_sched_boundary(1, cluster_count, 4);
    const int32_t sched3_count = cluster_count - balanced_sched_boundary(3, cluster_count, 4);
    const int32_t sched1_die0 = hybrid_middle_die0_count(cluster_count);
    const int32_t sched1_die1 = sched1_count - sched1_die0;

    if (ci < die0_count) {
        if (ci < sched0_count) return 0;
        if (ci < sched0_count + sched1_die0) return 1;
        return 2;
    }

    const int32_t die1_index = ci - die0_count;
    const int32_t die1_count = cluster_count - die0_count;
    if (die1_index < sched1_die1) return 1;
    if (die1_index < die1_count - sched3_count) return 2;
    return 3;
}

inline int32_t
logical_sched_for_cluster(int32_t ci, int32_t cluster_count, int32_t active_threads, int32_t assignment_mode) {
    if (active_threads <= 0 || cluster_count <= 0) return 0;
    assignment_mode = sched_assignment_base_mode(assignment_mode);
    if (assignment_mode == kSchedAicoreAssignmentRoundRobin) return ci % active_threads;
    if (assignment_mode == kSchedAicoreAssignmentHybridDieAware && active_threads == 4 && cluster_count >= 4) {
        return hybrid_logical_sched_for_cluster(ci, cluster_count);
    }

    // Compute balanced contiguous ranges from the discovered cluster count.
    // This fallback needs no die metadata; an offline calibration may reorder
    // scheduler CPUs so these logical ranges line up with physical dies.
    int32_t logical = static_cast<int32_t>((static_cast<int64_t>(ci) * active_threads) / cluster_count);
    if (logical >= active_threads) logical = active_threads - 1;
    return logical;
}

inline bool cluster_owned_by_logical_sched(
    int32_t ci, int32_t logical_sched, int32_t cluster_count, int32_t active_threads, int32_t assignment_mode
) {
    return logical_sched_for_cluster(ci, cluster_count, active_threads, assignment_mode) == logical_sched;
}

inline bool cluster_owned_by_pthread(
    int32_t ci, int32_t pthread_tidx, int32_t cluster_count, int32_t active_threads, int32_t assignment_mode
) {
    if (active_threads <= 0) return ci == 0 && pthread_tidx == 0;
    return logical_sched_for_cluster(ci, cluster_count, active_threads, assignment_mode) == pthread_tidx;
}
