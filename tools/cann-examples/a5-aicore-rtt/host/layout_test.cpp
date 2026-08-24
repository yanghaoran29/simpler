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

#include <algorithm>
#include <array>
#include <cassert>
#include <cstdint>

#include "../shared/rtt_types.h"

// Validate every topology supported by the fixed-capacity host/device ABI.
int main() {
    // Check dynamic logical-core numbering, visit order, and record indexing.
    for (uint32_t cluster_count = a5_rtt::kSchedulerCount; cluster_count <= a5_rtt::kMaxClusterCount; ++cluster_count) {
        const uint32_t logical_core_count = cluster_count * a5_rtt::kSubcoresPerPhysicalCluster;
        std::array<bool, a5_rtt::kMaxLogicalCoreCount> seen_cores{};
        for (uint32_t position = 0; position < logical_core_count; ++position) {
            const uint32_t logical = a5_rtt::CoreAtVisitPosition(position, cluster_count);
            assert(logical < logical_core_count);
            assert(!seen_cores[logical]);
            seen_cores[logical] = true;
            const uint32_t cluster = position / a5_rtt::kSubcoresPerPhysicalCluster;
            const uint32_t lane = position % a5_rtt::kSubcoresPerPhysicalCluster;
            assert(a5_rtt::CoreCluster(logical, cluster_count) == cluster);
            assert(a5_rtt::CoreLane(logical, cluster_count) == lane);
            assert(
                a5_rtt::LogicalCoreKind(logical, cluster_count) ==
                (lane == 0 ? a5_rtt::CoreKind::kAic : a5_rtt::CoreKind::kAiv)
            );
        }
        for (uint32_t logical = 0; logical < logical_core_count; ++logical) {
            assert(seen_cores[logical]);
        }

        std::array<bool, a5_rtt::kMaxRecordCount> seen_records{};
        for (uint32_t round = 0; round < a5_rtt::kRoundCount; ++round) {
            for (uint32_t scheduler = 0; scheduler < a5_rtt::kSchedulerCount; ++scheduler) {
                for (uint32_t logical = 0; logical < logical_core_count; ++logical) {
                    const uint32_t index = a5_rtt::ResultIndex(scheduler, round, logical, logical_core_count);
                    assert(index < a5_rtt::kMaxRecordCount);
                    assert(!seen_records[index]);
                    seen_records[index] = true;
                }
            }
        }

        // Check that both derived ownership policies cover every dynamic cluster exactly once.
        std::array<uint32_t, a5_rtt::kSchedulerCount> modulo_counts{};
        std::array<uint32_t, a5_rtt::kSchedulerCount> contiguous_counts{};
        for (uint32_t cluster = 0; cluster < cluster_count; ++cluster) {
            const uint32_t modulo_owner = a5_rtt::ModuloOwner(cluster, cluster_count);
            const uint32_t contiguous_owner = a5_rtt::ContiguousOwner(cluster, cluster_count);
            assert(modulo_owner < a5_rtt::kSchedulerCount);
            assert(contiguous_owner < a5_rtt::kSchedulerCount);
            ++modulo_counts[modulo_owner];
            ++contiguous_counts[contiguous_owner];
        }
        const auto modulo_range = std::minmax_element(modulo_counts.begin(), modulo_counts.end());
        const auto contiguous_range = std::minmax_element(contiguous_counts.begin(), contiguous_counts.end());
        assert(*modulo_range.second - *modulo_range.first <= 1);
        assert(*contiguous_range.second - *contiguous_range.first <= 1);
    }

    // Check that the capacity-sized physical mapping covers every register slot once.
    std::array<bool, a5_rtt::kRegisterSlotCount> register_slots{};
    for (uint32_t cluster = 0; cluster < a5_rtt::kPhysicalClusterCount; ++cluster) {
        const std::array<uint32_t, a5_rtt::kSubcoresPerPhysicalCluster> slots = {
            a5_rtt::PhysicalAicRegisterSlot(cluster),
            a5_rtt::PhysicalAiv0RegisterSlot(cluster),
            a5_rtt::PhysicalAiv1RegisterSlot(cluster),
        };
        for (const uint32_t slot : slots) {
            assert(slot < register_slots.size());
            assert(!register_slots[slot]);
            register_slots[slot] = true;
            assert(a5_rtt::PhysicalCoreDie(slot) == cluster / a5_rtt::kAicPerDie);
        }
    }
    for (const bool mapped : register_slots) {
        assert(mapped);
    }

    // Check that warmup and measured tokens never alias across scheduler turns.
    constexpr uint32_t kTokensPerTurn = a5_rtt::kMaxWarmup + a5_rtt::kMaxSamples;
    constexpr uint32_t kTokenCapacity = a5_rtt::kSchedulerCount * a5_rtt::kRoundCount * kTokensPerTurn;
    std::array<uint32_t, kTokenCapacity> tokens{};
    uint32_t token_count = 0;
    for (uint32_t round = 0; round < a5_rtt::kRoundCount; ++round) {
        for (uint32_t scheduler = 0; scheduler < a5_rtt::kSchedulerCount; ++scheduler) {
            for (uint32_t iteration = 0; iteration < a5_rtt::kMaxWarmup; ++iteration) {
                tokens[token_count++] = a5_rtt::MakeToken(scheduler, round, false, iteration);
            }
            for (uint32_t iteration = 0; iteration < a5_rtt::kMaxSamples; ++iteration) {
                tokens[token_count++] = a5_rtt::MakeToken(scheduler, round, true, iteration);
            }
        }
    }
    assert(token_count == tokens.size());
    for (uint32_t i = 0; i < token_count; ++i) {
        assert(tokens[i] != a5_rtt::kAicoreExitSignal);
        assert(tokens[i] != a5_rtt::kAicpuIdleTaskId);
        for (uint32_t j = i + 1; j < token_count; ++j) {
            assert(tokens[i] != tokens[j]);
        }
    }
    return 0;
}
