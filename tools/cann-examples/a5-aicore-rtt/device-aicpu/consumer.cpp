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

#include <sched.h>

#include <atomic>
#include <cstdint>

#include "../shared/rtt_types.h"

namespace {

struct KernelArgs {
    uint64_t reserved[5];
    void *device_args;
};

constexpr uint64_t kReadyTimeoutMilliseconds = 10000;
constexpr uint64_t kRttTimeoutMilliseconds = 10;
constexpr uint64_t kBarrierTimeoutMilliseconds = 1000;

std::atomic<uint32_t> s_cpu_claims{0};
std::atomic<uint32_t> s_entry_count{0};
std::atomic<uint32_t> s_launch_arrivals{0};
std::atomic<uint32_t> s_registers_initialized{0};
std::atomic<uint32_t> s_barrier_arrivals{0};
std::atomic<uint32_t> s_barrier_generation{0};
std::atomic<uint32_t> s_measurement_turn{0};
std::atomic<int32_t> s_error{a5_rtt::kOk};

// Read the AICPU virtual counter around the timed MMIO round trip.
inline uint64_t ReadCounter() {
    uint64_t value;
    __asm__ volatile("mrs %0, cntvct_el0" : "=r"(value) : : "memory");
    return value;
}

// Read the virtual-counter frequency used to convert ticks to time.
inline uint64_t ReadCounterFrequency() {
    uint64_t value;
    __asm__ volatile("mrs %0, cntfrq_el0" : "=r"(value) : : "memory");
    return value;
}

// Convert a timeout in milliseconds to counter ticks.
inline uint64_t MillisecondsToTicks(uint64_t milliseconds) { return ReadCounterFrequency() * milliseconds / 1000; }

// Preserve the first cross-thread error for host-side diagnostics.
void RecordGlobalError(int32_t error) {
    int32_t expected = a5_rtt::kOk;
    s_error.compare_exchange_strong(expected, error, std::memory_order_acq_rel);
}

// Synchronize the four scheduler entries between protocol phases.
bool WaitAtSchedulerBarrier() {
    const uint32_t generation = s_barrier_generation.load(std::memory_order_acquire);
    if (s_barrier_arrivals.fetch_add(1, std::memory_order_acq_rel) + 1 == a5_rtt::kSchedulerCount) {
        s_barrier_arrivals.store(0, std::memory_order_release);
        s_barrier_generation.fetch_add(1, std::memory_order_acq_rel);
        return true;
    }

    const uint64_t deadline = ReadCounter() + MillisecondsToTicks(kBarrierTimeoutMilliseconds);
    uint32_t polls = 0;
    while (s_barrier_generation.load(std::memory_order_acquire) == generation) {
        if ((++polls & 0xFFu) == 0 && ReadCounter() > deadline) {
            RecordGlobalError(a5_rtt::kBarrierTimeout);
            return false;
        }
    }
    return true;
}

// Wait until all four schedulers and the orchestrator entry have launched.
bool WaitAtLaunchBarrier() {
    s_launch_arrivals.fetch_add(1, std::memory_order_acq_rel);
    const uint64_t deadline = ReadCounter() + MillisecondsToTicks(kBarrierTimeoutMilliseconds);
    uint32_t polls = 0;
    while (s_launch_arrivals.load(std::memory_order_acquire) != a5_rtt::kAllowedCpuCount) {
        if ((++polls & 0xFFu) == 0 && ReadCounter() > deadline) {
            RecordGlobalError(a5_rtt::kBarrierTimeout);
            return false;
        }
    }
    return true;
}

// Map the current AICPU CPU ID to its stable scheduler/orchestrator index.
int CpuIndex(const a5_rtt::RttDeviceArgs &args, int cpu_id) {
    for (uint32_t i = 0; i < args.allowed_cpu_count; ++i) {
        if (args.allowed_cpus[i] == cpu_id) {
            return static_cast<int>(i);
        }
    }
    return -1;
}

// Wait for one AICore to publish its physical identity and readiness record.
bool WaitForReady(const volatile a5_rtt::CoreReadyRecord *ready) {
    const uint64_t deadline = ReadCounter() + MillisecondsToTicks(kReadyTimeoutMilliseconds);
    uint32_t polls = 0;
    while (ready->ready_magic != a5_rtt::kReadyMagic) {
        if ((++polls & 0xFFu) == 0 && ReadCounter() > deadline) {
            return false;
        }
    }
    return true;
}

// Poll one MMIO word until the expected acknowledgement or its deadline.
bool WaitForValueUntil(const volatile uint32_t *address, uint32_t expected, uint64_t deadline) {
    uint32_t polls = 0;
    while (*address != expected) {
        if ((++polls & 0xFFu) == 0 && ReadCounter() > deadline) {
            return false;
        }
    }
    return true;
}

// Check that an AICore readiness record matches its logical launch position.
bool ValidateReadyRecord(
    const volatile a5_rtt::CoreReadyRecord &ready, uint32_t logical_core_id, uint32_t cluster_count
) {
    return ready.logical_core_id == logical_core_id && ready.physical_core_id < a5_rtt::kRegisterSlotCount &&
           ready.core_kind == static_cast<uint32_t>(a5_rtt::LogicalCoreKind(logical_core_id, cluster_count)) &&
           ready.cluster_id == a5_rtt::CoreCluster(logical_core_id, cluster_count) &&
           ready.lane == a5_rtt::CoreLane(logical_core_id, cluster_count);
}

// Copy live launch state into the shared output for failure diagnosis.
void PublishDiagnostics(a5_rtt::RttOutput *output) {
    output->consumer_rc = s_error.load(std::memory_order_acquire);
    output->entry_count = s_entry_count.load(std::memory_order_acquire);
    output->cpu_claim_mask = s_cpu_claims.load(std::memory_order_acquire);
    output->launch_arrivals = s_launch_arrivals.load(std::memory_order_acquire);
    output->measurement_turn = s_measurement_turn.load(std::memory_order_acquire);
}

// Initialize metadata for one scheduler/round/logical-core result record.
void InitializeResult(
    a5_rtt::CoreRttResult *result, const volatile a5_rtt::CoreReadyRecord &ready, uint32_t scheduler_index, int cpu_id,
    uint32_t round, uint32_t position
) {
    result->logical_core_id = ready.logical_core_id;
    result->physical_core_id = ready.physical_core_id;
    result->core_kind = ready.core_kind;
    result->cluster_id = ready.cluster_id;
    result->lane = ready.lane;
    result->scheduler_index = scheduler_index;
    result->aicpu_cpu_id = cpu_id;
    result->round_index = round;
    result->visit_position = position;
    result->completed_samples = 0;
    result->error_code = a5_rtt::kOk;
    result->window_ticks = 0;
}

// Measure consecutive RTT samples for one logical core without including setup.
int MeasureCore(
    volatile uint32_t *data_main_base, volatile uint32_t *cond, a5_rtt::CoreRttResult *result, uint32_t warmup,
    uint32_t samples, uint32_t scheduler_index, uint32_t round
) {
    const uint64_t timeout_ticks = MillisecondsToTicks(kRttTimeoutMilliseconds);
    *data_main_base = a5_rtt::kAicpuIdleTaskId;

    // Prime the same MMIO path before opening the measured window.
    for (uint32_t iteration = 0; iteration < warmup; ++iteration) {
        const uint32_t token = a5_rtt::MakeToken(scheduler_index, round, false, iteration);
        const uint64_t deadline = ReadCounter() + timeout_ticks;
        *data_main_base = token;
        if (!WaitForValueUntil(cond, token, deadline)) {
            result->error_code = a5_rtt::kRttTimeout;
            return a5_rtt::kRttTimeout;
        }
    }

    uint64_t first_tick = 0;
    uint64_t last_tick = 0;
    // Record every measured request/acknowledgement latency in its raw tick form.
    for (uint32_t iteration = 0; iteration < samples; ++iteration) {
        const uint32_t token = a5_rtt::MakeToken(scheduler_index, round, true, iteration);
        const uint64_t begin = ReadCounter();
        *data_main_base = token;
        if (!WaitForValueUntil(cond, token, begin + timeout_ticks)) {
            result->error_code = a5_rtt::kRttTimeout;
            return a5_rtt::kRttTimeout;
        }
        const uint64_t end = ReadCounter();
        if (iteration == 0) {
            first_tick = begin;
        }
        last_tick = end;
        result->sample_ticks[iteration] = end - begin;
        result->completed_samples = iteration + 1;
    }
    result->window_ticks = last_tick - first_tick;
    return a5_rtt::kOk;
}

// Wait until the global round-robin ticket selects this scheduler entry.
bool WaitForMeasurementTurn(uint32_t expected_turn) {
    const uint64_t deadline = ReadCounter() + MillisecondsToTicks(kReadyTimeoutMilliseconds);
    uint32_t polls = 0;
    while (s_measurement_turn.load(std::memory_order_acquire) != expected_turn) {
        if ((++polls & 0xFFu) == 0 && ReadCounter() > deadline) {
            RecordGlobalError(a5_rtt::kBarrierTimeout);
            return false;
        }
    }
    return true;
}

// Visit every logical core while schedulers take strict S0-to-S3 turns.
int RunScheduler(
    const a5_rtt::RttDeviceArgs &args, a5_rtt::SharedState *state, a5_rtt::RttOutput *output, uint32_t scheduler_index,
    int cpu_id
) {
    uint64_t register_bases[a5_rtt::kMaxLogicalCoreCount] = {};

    // Resolve all logical-to-physical mappings before any scheduler is timed.
    for (uint32_t logical_core_id = 0; logical_core_id < args.logical_core_count; ++logical_core_id) {
        const volatile a5_rtt::CoreReadyRecord &ready = state->ready[logical_core_id];
        if (!WaitForReady(&ready)) {
            RecordGlobalError(a5_rtt::kReadyTimeout);
            output->first_error_scheduler = static_cast<int32_t>(scheduler_index);
            output->first_error_logical_core = static_cast<int32_t>(logical_core_id);
            return a5_rtt::kReadyTimeout;
        }
        if (!ValidateReadyRecord(ready, logical_core_id, args.cluster_count)) {
            RecordGlobalError(a5_rtt::kInvalidReadyRecord);
            output->first_error_scheduler = static_cast<int32_t>(scheduler_index);
            output->first_error_logical_core = static_cast<int32_t>(logical_core_id);
            return a5_rtt::kInvalidReadyRecord;
        }
        register_bases[logical_core_id] = state->register_addrs[ready.physical_core_id];
        if (register_bases[logical_core_id] == 0) {
            RecordGlobalError(a5_rtt::kInvalidReadyRecord);
            output->first_error_scheduler = static_cast<int32_t>(scheduler_index);
            output->first_error_logical_core = static_cast<int32_t>(logical_core_id);
            return a5_rtt::kInvalidReadyRecord;
        }
        output->scheduler_ready_counts[scheduler_index] = logical_core_id + 1;
    }

    // Start the ticketed measurement only after every scheduler sees all cores.
    if (!WaitAtSchedulerBarrier()) {
        return a5_rtt::kBarrierTimeout;
    }

    // Execute five cycles of S0, S1, S2, S3 with no scheduler overlap.
    for (uint32_t round = 0; round < args.round_count; ++round) {
        const uint32_t expected_turn = round * a5_rtt::kSchedulerCount + scheduler_index;
        if (!WaitForMeasurementTurn(expected_turn)) {
            return a5_rtt::kBarrierTimeout;
        }
        for (uint32_t visit_position = 0; visit_position < args.logical_core_count; ++visit_position) {
            const uint32_t logical_core_id = a5_rtt::CoreAtVisitPosition(visit_position, args.cluster_count);
            const volatile a5_rtt::CoreReadyRecord &ready = state->ready[logical_core_id];
            a5_rtt::CoreRttResult *result =
                &output->records[a5_rtt::ResultIndex(scheduler_index, round, logical_core_id, args.logical_core_count)];
            InitializeResult(result, ready, scheduler_index, cpu_id, round, visit_position);

            const uint64_t register_base = register_bases[logical_core_id];
            auto *data_main_base =
                reinterpret_cast<volatile uint32_t *>(register_base + a5_rtt::kRegDataMainBaseOffset);
            auto *cond = reinterpret_cast<volatile uint32_t *>(register_base + a5_rtt::kRegCondOffset);
            const int rc = MeasureCore(data_main_base, cond, result, args.warmup, args.samples, scheduler_index, round);
            if (rc != a5_rtt::kOk) {
                RecordGlobalError(rc);
            }
        }
        s_measurement_turn.store(expected_turn + 1, std::memory_order_release);
    }

    // Let S0 terminate each persistent producer exactly once after all turns.
    if (!WaitAtSchedulerBarrier()) {
        return a5_rtt::kBarrierTimeout;
    }
    if (scheduler_index == 0) {
        for (uint32_t logical_core_id = 0; logical_core_id < args.logical_core_count; ++logical_core_id) {
            const uint64_t register_base = register_bases[logical_core_id];
            auto *data_main_base =
                reinterpret_cast<volatile uint32_t *>(register_base + a5_rtt::kRegDataMainBaseOffset);
            auto *cond = reinterpret_cast<volatile uint32_t *>(register_base + a5_rtt::kRegCondOffset);
            const uint64_t deadline = ReadCounter() + MillisecondsToTicks(kReadyTimeoutMilliseconds);
            *data_main_base = a5_rtt::kAicoreExitSignal;
            if (!WaitForValueUntil(cond, a5_rtt::kAicoreExitedValue, deadline)) {
                RecordGlobalError(a5_rtt::kExitTimeout);
            }
        }
    }

    // Publish final counts only after producer shutdown is visible to all schedulers.
    if (!WaitAtSchedulerBarrier()) {
        return a5_rtt::kBarrierTimeout;
    }

    if (scheduler_index == 0) {
        uint32_t completed_records = 0;
        uint32_t failed_records = 0;
        const uint32_t record_count = a5_rtt::kSchedulerCount * args.round_count * args.logical_core_count;
        for (uint32_t i = 0; i < record_count; ++i) {
            if (output->records[i].error_code == a5_rtt::kOk && output->records[i].completed_samples == args.samples) {
                ++completed_records;
            } else {
                ++failed_records;
            }
        }
        output->consumer_rc = s_error.load(std::memory_order_acquire);
        output->scheduler_count = a5_rtt::kSchedulerCount;
        output->cluster_count = args.cluster_count;
        output->logical_core_count = args.logical_core_count;
        output->samples_requested = args.samples;
        output->warmup_requested = args.warmup;
        output->round_count = args.round_count;
        output->completed_records = completed_records;
        output->failed_records = failed_records;
        output->counter_frequency_hz = ReadCounterFrequency();
        for (uint32_t i = 0; i < a5_rtt::kAllowedCpuCount; ++i) {
            output->allowed_cpus[i] = args.allowed_cpus[i];
        }
        output->measurement_turn = s_measurement_turn.load(std::memory_order_acquire);
        std::atomic_thread_fence(std::memory_order_release);
        output->magic = a5_rtt::kOutputMagic;
    }
    return s_error.load(std::memory_order_acquire);
}

}  // namespace

extern "C" {

// Reset process-global synchronization before launching concurrent AICPU entries.
__attribute__((visibility("default"))) int simpler_aicpu_init(void *args) {
    // Reset every process-global coordination word before the five entries launch.
    (void)args;
    s_cpu_claims.store(0, std::memory_order_release);
    s_entry_count.store(0, std::memory_order_release);
    s_launch_arrivals.store(0, std::memory_order_release);
    s_registers_initialized.store(0, std::memory_order_release);
    s_barrier_arrivals.store(0, std::memory_order_release);
    s_barrier_generation.store(0, std::memory_order_release);
    s_measurement_turn.store(0, std::memory_order_release);
    s_error.store(a5_rtt::kOk, std::memory_order_release);
    return 0;
}

// Claim a stable CPU role and run either one scheduler or the idle orchestrator entry.
__attribute__((visibility("default"))) int simpler_aicpu_run(void *args) {
    // Validate the host ABI before any entry claims a stable CPU role.
    if (args == nullptr) {
        return a5_rtt::kInvalidArguments;
    }
    auto *kernel_args = reinterpret_cast<KernelArgs *>(args);
    auto *device_args = reinterpret_cast<a5_rtt::RttDeviceArgs *>(kernel_args->device_args);
    if (device_args == nullptr || device_args->output_addr == 0 || device_args->shared_state_addr == 0 ||
        device_args->samples == 0 || device_args->samples > a5_rtt::kMaxSamples ||
        device_args->warmup > a5_rtt::kMaxWarmup || device_args->round_count != a5_rtt::kRoundCount ||
        device_args->cluster_count == 0 || device_args->cluster_count > a5_rtt::kMaxClusterCount ||
        device_args->logical_core_count != device_args->cluster_count * a5_rtt::kSubcoresPerPhysicalCluster ||
        device_args->allowed_cpu_count != a5_rtt::kAllowedCpuCount ||
        device_args->scheduler_count != a5_rtt::kSchedulerCount) {
        RecordGlobalError(a5_rtt::kInvalidArguments);
        return a5_rtt::kInvalidArguments;
    }

    auto *output = reinterpret_cast<a5_rtt::RttOutput *>(device_args->output_addr);

    // Claim one of CPUs 1-5 so scheduler identity remains stable across rounds.
    const int cpu_id = sched_getcpu();
    const uint32_t entry_index = s_entry_count.fetch_add(1, std::memory_order_acq_rel);
    if (entry_index < a5_rtt::kAllowedCpuCount) {
        output->observed_cpus[entry_index] = cpu_id;
    }
    const int cpu_index = CpuIndex(*device_args, cpu_id);
    if (cpu_index < 0) {
        RecordGlobalError(a5_rtt::kUnexpectedCpu);
        PublishDiagnostics(output);
        return a5_rtt::kUnexpectedCpu;
    }
    const uint32_t claim_bit = 1u << static_cast<uint32_t>(cpu_index);
    if ((s_cpu_claims.fetch_or(claim_bit, std::memory_order_acq_rel) & claim_bit) != 0) {
        RecordGlobalError(a5_rtt::kDuplicateCpu);
        PublishDiagnostics(output);
        return a5_rtt::kDuplicateCpu;
    }
    if (!WaitAtLaunchBarrier()) {
        PublishDiagnostics(output);
        return a5_rtt::kBarrierTimeout;
    }

    if (static_cast<uint32_t>(cpu_index) == a5_rtt::kSchedulerCount) {
        // The fifth entry reserves the production orchestrator CPU but performs no RTT work.
        PublishDiagnostics(output);
        return a5_rtt::kOk;
    }

    auto *state = reinterpret_cast<a5_rtt::SharedState *>(device_args->shared_state_addr);
    if (cpu_index == 0) {
        // A5 does not let the mixed producer reach its GM readiness publication until DATA_MAIN_BASE has been
        // initialized. Physical IDs are part of that publication, so open every mapped slot once to break the
        // discovery cycle; inactive physical slots are never used again. This work is outside the timed loop.
        for (uint32_t slot = 0; slot < a5_rtt::kRegisterSlotCount; ++slot) {
            if (state->register_addrs[slot] != 0) {
                *reinterpret_cast<volatile uint32_t *>(state->register_addrs[slot] + a5_rtt::kRegDataMainBaseOffset) =
                    a5_rtt::kAicpuIdleTaskId;
            }
        }
        s_registers_initialized.store(1, std::memory_order_release);
    } else {
        const uint64_t deadline = ReadCounter() + MillisecondsToTicks(kBarrierTimeoutMilliseconds);
        uint32_t polls = 0;
        while (s_registers_initialized.load(std::memory_order_acquire) == 0) {
            if ((++polls & 0xFFu) == 0 && ReadCounter() > deadline) {
                RecordGlobalError(a5_rtt::kBarrierTimeout);
                PublishDiagnostics(output);
                return a5_rtt::kBarrierTimeout;
            }
        }
    }
    (void)RunScheduler(*device_args, state, output, static_cast<uint32_t>(cpu_index), cpu_id);
    PublishDiagnostics(output);
    return a5_rtt::kOk;
}

}  // extern "C"
