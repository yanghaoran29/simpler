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

#include <algorithm>
#include <cstdint>

#include "common/platform_config.h"

constexpr int32_t kSchedAicoreAssignmentSequential = 0;
constexpr int32_t kSchedAicoreAssignmentDieAware = 1;
constexpr int32_t kSchedAicoreAssignmentRoundRobin = 2;
// Device-side RTT preflight: each scheduler thread probes die0/die1 AICore MMIO
// latency (physical core index), ranks threads by avg(die0)-avg(die1), then maps
// the two most die0-affinitive threads to logical exec 0/1 and the two most
// die1-affinitive to logical exec 2/3 before contiguous cluster assignment.
constexpr int32_t kSchedAicoreAssignmentRttDieAware = 3;

inline bool sched_assignment_uses_contiguous_blocks(int32_t assignment_mode) {
    return assignment_mode != kSchedAicoreAssignmentRoundRobin;
}

inline bool sched_assignment_uses_rtt_die_preflight(int32_t assignment_mode, int32_t active_threads) {
    return assignment_mode == kSchedAicoreAssignmentRttDieAware && active_threads == 4;
}

inline int32_t logical_sched_for_cluster(int32_t ci, int32_t cluster_count, int32_t active_threads, int32_t assignment_mode) {
    if (active_threads <= 0) return 0;
    if (assignment_mode == kSchedAicoreAssignmentRoundRobin) return ci % active_threads;
    const int32_t span = (cluster_count + active_threads - 1) / active_threads;
    int32_t t = ci / span;
    if (t >= active_threads) t = active_threads - 1;
    return t;
}

inline bool cluster_owned_by_logical_sched(
    int32_t ci, int32_t logical_sched, int32_t cluster_count, int32_t active_threads, int32_t assignment_mode
) {
    return logical_sched_for_cluster(ci, cluster_count, active_threads, assignment_mode) == logical_sched;
}

inline bool cluster_owned_by_pthread(
    int32_t ci,
    int32_t pthread_tidx,
    int32_t cluster_count,
    int32_t active_threads,
    int32_t assignment_mode,
    const int32_t *pthread_to_logical
) {
    if (active_threads <= 0) return ci == 0 && pthread_tidx == 0;
    if (sched_assignment_uses_rtt_die_preflight(assignment_mode, active_threads)) {
        const int32_t logical = pthread_to_logical[pthread_tidx];
        return cluster_owned_by_logical_sched(ci, logical, cluster_count, active_threads, assignment_mode);
    }
    return logical_sched_for_cluster(ci, cluster_count, active_threads, assignment_mode) == pthread_tidx;
}

struct RttDieRankEntry {
    int32_t pthread_tidx;
    int64_t die_delta_ticks;
};

// Sort scheduler pthreads by die0-minus-die1 average RTT (ascending). The two
// smallest deltas become logical exec 0/1 (die0 managers); the two largest
// become logical exec 2/3 (die1 managers). Tie-break on pthread_tidx for stability.
inline void assign_rtt_die_logical_order(
    const int64_t *die_delta_ticks,
    int32_t active_threads,
    int32_t *pthread_to_logical_out,
    int32_t *logical_to_pthread_out
) {
    for (int32_t i = 0; i < active_threads; ++i) {
        pthread_to_logical_out[i] = i;
        logical_to_pthread_out[i] = i;
    }
    if (active_threads != 4) return;

    RttDieRankEntry entries[4];
    for (int32_t i = 0; i < active_threads; ++i) {
        entries[i] = {i, die_delta_ticks[i]};
    }
    std::sort(entries, entries + active_threads, [](const RttDieRankEntry &a, const RttDieRankEntry &b) {
        if (a.die_delta_ticks != b.die_delta_ticks) return a.die_delta_ticks < b.die_delta_ticks;
        return a.pthread_tidx < b.pthread_tidx;
    });

    for (int32_t logical = 0; logical < active_threads; ++logical) {
        const int32_t pthread = entries[logical].pthread_tidx;
        pthread_to_logical_out[pthread] = logical;
        logical_to_pthread_out[logical] = pthread;
    }
}
