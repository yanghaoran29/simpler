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

#include <cstddef>
#include <cstdint>
#include <type_traits>

namespace a5_rtt {

constexpr uint32_t kMaxClusterCount = 36;
constexpr uint32_t kMaxLogicalCoreCount = kMaxClusterCount * 3;
constexpr uint32_t kSchedulerCount = 4;
constexpr uint32_t kRoundCount = 5;
constexpr uint32_t kMaxRecordCount = kSchedulerCount * kRoundCount * kMaxLogicalCoreCount;
constexpr uint32_t kDefaultSamples = 50;
constexpr uint32_t kDefaultWarmup = 10;
constexpr uint32_t kMaxSamples = 50;
constexpr uint32_t kMaxWarmup = 100;
constexpr uint32_t kPhysicalClusterCount = kMaxClusterCount;
constexpr uint32_t kSubcoresPerPhysicalCluster = 3;
constexpr uint32_t kRegisterSlotCount = kPhysicalClusterCount * kSubcoresPerPhysicalCluster;
constexpr uint32_t kDieCount = 2;
constexpr uint32_t kAicPerDie = 18;
constexpr uint32_t kRegisterSlotsPerDie = kAicPerDie * kSubcoresPerPhysicalCluster;
constexpr uint32_t kAllowedCpuCount = 5;

constexpr uint32_t kRegDataMainBaseOffset = 0xD0;
constexpr uint32_t kRegCondOffset = 0x5108;
constexpr uint64_t kSubcoreRegisterStride = 0x100000ULL;
constexpr uint32_t kPhysicalClusterMapSize = 0x300000;

constexpr uint32_t kTaskIdMask = 0x7FFFFFFFu;
constexpr uint32_t kTaskStateMask = 0x80000000u;
constexpr uint32_t kAicoreExitSignal = 0x7FFFFFF0u;
constexpr uint32_t kAicoreExitTaskId = 0x7FFFFFFEu;
constexpr uint32_t kAicpuIdleTaskId = 0x7FFFFFFDu;
constexpr uint32_t kAicoreIdleTaskId = 0x7FFFFFFFu;
constexpr uint32_t kAicoreExitedValue = kAicoreExitTaskId | kTaskStateMask;
constexpr uint32_t kAicoreIdleValue = kAicoreIdleTaskId | kTaskStateMask;

constexpr uint32_t kOutputMagic = 0xA5845254u;
constexpr uint32_t kReadyMagic = 0xA5C0E001u;
constexpr uint32_t kReadyBootstrapMagic = 0xA5C00001u;

enum ErrorCode : int32_t {
    kOk = 0,
    kInvalidArguments = -1,
    kUnexpectedCpu = -2,
    kDuplicateCpu = -3,
    kBarrierTimeout = -4,
    kReadyTimeout = -5,
    kInvalidReadyRecord = -6,
    kRttTimeout = -7,
    kExitTimeout = -8,
};

enum class CoreKind : uint32_t { kAic = 0, kAiv = 1 };

// Convert a cluster index to its logical cube-core ID.
constexpr uint32_t AicLogicalId(uint32_t cluster) { return cluster; }

// Convert a cluster index to its first logical vector-core ID.
constexpr uint32_t Aiv0LogicalId(uint32_t cluster, uint32_t cluster_count) { return cluster_count + 2 * cluster; }

// Convert a cluster index to its second logical vector-core ID.
constexpr uint32_t Aiv1LogicalId(uint32_t cluster, uint32_t cluster_count) {
    return Aiv0LogicalId(cluster, cluster_count) + 1;
}

// Return the logical core visited at a cluster-major position.
constexpr uint32_t CoreAtVisitPosition(uint32_t position, uint32_t cluster_count) {
    const uint32_t cluster = position / kSubcoresPerPhysicalCluster;
    const uint32_t lane = position % kSubcoresPerPhysicalCluster;
    return lane == 0 ? AicLogicalId(cluster) :
                       (lane == 1 ? Aiv0LogicalId(cluster, cluster_count) : Aiv1LogicalId(cluster, cluster_count));
}

// Recover the cluster index from a logical core ID.
constexpr uint32_t CoreCluster(uint32_t logical_core_id, uint32_t cluster_count) {
    return logical_core_id < cluster_count ? logical_core_id : (logical_core_id - cluster_count) / 2;
}

// Recover the lane within a cluster: zero for AIC and one or two for AIV.
constexpr uint32_t CoreLane(uint32_t logical_core_id, uint32_t cluster_count) {
    return logical_core_id < cluster_count ? 0 : 1 + ((logical_core_id - cluster_count) & 1u);
}

// Classify a logical core as cube or vector.
constexpr CoreKind LogicalCoreKind(uint32_t logical_core_id, uint32_t cluster_count) {
    return logical_core_id < cluster_count ? CoreKind::kAic : CoreKind::kAiv;
}

// Assign a cluster to the scheduler selected by its modulo-four index.
constexpr uint32_t ModuloOwner(uint32_t cluster, uint32_t cluster_count) {
    (void)cluster_count;
    return cluster % kSchedulerCount;
}

// Split an arbitrary cluster count into four balanced contiguous ranges.
constexpr uint32_t ContiguousOwner(uint32_t cluster, uint32_t cluster_count) {
    return ((cluster + 1) * kSchedulerCount - 1) / cluster_count;
}

// Recover the die index from the physical register-slot ID.
constexpr uint32_t PhysicalCoreDie(uint32_t physical_core_id) { return physical_core_id / kRegisterSlotsPerDie; }

// Map a physical cluster to its cube-core register slot.
constexpr uint32_t PhysicalAicRegisterSlot(uint32_t physical_cluster) {
    const uint32_t die = physical_cluster / kAicPerDie;
    const uint32_t local_cluster = physical_cluster % kAicPerDie;
    return die * kRegisterSlotsPerDie + local_cluster;
}

// Map a physical cluster to its first vector-core register slot.
constexpr uint32_t PhysicalAiv0RegisterSlot(uint32_t physical_cluster) {
    const uint32_t die = physical_cluster / kAicPerDie;
    const uint32_t local_cluster = physical_cluster % kAicPerDie;
    return die * kRegisterSlotsPerDie + kAicPerDie + 2 * local_cluster;
}

// Map a physical cluster to its second vector-core register slot.
constexpr uint32_t PhysicalAiv1RegisterSlot(uint32_t physical_cluster) {
    return PhysicalAiv0RegisterSlot(physical_cluster) + 1;
}

// Locate one scheduler/round/core record in the shared output array.
constexpr uint32_t
ResultIndex(uint32_t scheduler, uint32_t round, uint32_t logical_core_id, uint32_t logical_core_count) {
    return (round * kSchedulerCount + scheduler) * logical_core_count + logical_core_id;
}

// Encode the active scheduler and round so an AICore never sees a reused token.
constexpr uint32_t MakeToken(uint32_t scheduler, uint32_t round, bool measured, uint32_t iteration) {
    return 1u + (scheduler << 24) + (round << 20) + (static_cast<uint32_t>(measured) << 16) + iteration;
}

struct alignas(64) CoreReadyRecord {
    volatile uint32_t ready_magic;
    uint32_t logical_core_id;
    uint32_t physical_core_id;
    uint32_t core_kind;
    uint32_t cluster_id;
    uint32_t lane;
    uint32_t reserved[10];
};
static_assert(sizeof(CoreReadyRecord) == 64);

struct alignas(64) SharedState {
    uint64_t register_addrs[kRegisterSlotCount];
    CoreReadyRecord ready[kMaxLogicalCoreCount];
};

struct alignas(64) CoreRttResult {
    uint32_t logical_core_id;
    uint32_t physical_core_id;
    uint32_t core_kind;
    uint32_t cluster_id;
    uint32_t lane;
    uint32_t scheduler_index;
    int32_t aicpu_cpu_id;
    uint32_t round_index;
    uint32_t visit_position;
    uint32_t completed_samples;
    int32_t error_code;
    uint32_t reserved0;
    uint64_t window_ticks;
    uint64_t sample_ticks[kMaxSamples];
};

struct alignas(64) RttOutput {
    uint32_t magic;
    int32_t consumer_rc;
    uint32_t scheduler_count;
    uint32_t cluster_count;
    uint32_t logical_core_count;
    uint32_t samples_requested;
    uint32_t warmup_requested;
    uint32_t round_count;
    uint32_t completed_records;
    uint32_t failed_records;
    uint64_t counter_frequency_hz;
    int32_t allowed_cpus[kAllowedCpuCount];
    int32_t observed_cpus[kAllowedCpuCount];
    uint32_t entry_count;
    uint32_t cpu_claim_mask;
    uint32_t launch_arrivals;
    int32_t first_error_scheduler;
    int32_t first_error_logical_core;
    uint32_t scheduler_ready_counts[kSchedulerCount];
    uint32_t measurement_turn;
    CoreRttResult records[kMaxRecordCount];
};

struct alignas(8) RttDeviceArgs {
    uint64_t reserved_pre[12];
    uint64_t output_addr;
    uint64_t shared_state_addr;
    uint32_t samples;
    uint32_t warmup;
    uint32_t round_count;
    uint32_t cluster_count;
    uint32_t logical_core_count;
    uint32_t allowed_cpu_count;
    uint32_t scheduler_count;
    int32_t allowed_cpus[kAllowedCpuCount];
    uint32_t reserved_post[2];
};

struct alignas(8) AicoreKernelArgs {
    uint64_t shared_state_addr;
};

static_assert(std::is_standard_layout_v<CoreReadyRecord> && std::is_trivially_copyable_v<CoreReadyRecord>);
static_assert(std::is_standard_layout_v<SharedState> && std::is_trivially_copyable_v<SharedState>);
static_assert(std::is_standard_layout_v<CoreRttResult> && std::is_trivially_copyable_v<CoreRttResult>);
static_assert(std::is_standard_layout_v<RttOutput> && std::is_trivially_copyable_v<RttOutput>);
static_assert(std::is_standard_layout_v<RttDeviceArgs> && std::is_trivially_copyable_v<RttDeviceArgs>);
static_assert(std::is_standard_layout_v<AicoreKernelArgs> && std::is_trivially_copyable_v<AicoreKernelArgs>);
static_assert(sizeof(SharedState) == 7808);
static_assert(offsetof(SharedState, ready) == 896);
static_assert(sizeof(CoreRttResult) == 512);
static_assert(sizeof(RttOutput) == 1106048);
static_assert(offsetof(RttOutput, records) == 128);
static_assert(sizeof(RttDeviceArgs) == 168);
static_assert(offsetof(RttDeviceArgs, output_addr) == 96);
static_assert(offsetof(RttDeviceArgs, shared_state_addr) == 104);
static_assert(offsetof(RttDeviceArgs, samples) == 112);
static_assert(sizeof(AicoreKernelArgs) == 8);

}  // namespace a5_rtt
