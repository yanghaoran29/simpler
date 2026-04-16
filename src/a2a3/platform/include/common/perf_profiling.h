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

/**
 * @file perf_profiling.h
 * @brief Performance profiling data structures
 *
 * Architecture: Fixed header + per-core/thread buffer states + optional phase profiling region
 *
 * Memory layout (shared memory between Host and Device):
 * ┌─────────────────────────────────────────────────────────────┐
 * │ PerfDataHeader (fixed header)                               │
 * │  - ReadyQueue (FIFO, capacity=PLATFORM_PROF_READYQUEUE_SIZE)│
 * │  - Metadata (num_cores, flags)                              │
 * ├─────────────────────────────────────────────────────────────┤
 * │ PerfBufferState[0] (Core 0)                                 │
 * │  - free_queue: SPSC queue of available buffer pointers      │
 * │  - current_buf_ptr, current_buf_seq                         │
 * ├─────────────────────────────────────────────────────────────┤
 * │ PerfBufferState[1] (Core 1)                                 │
 * ├─────────────────────────────────────────────────────────────┤
 * │ ...                                                         │
 * ├─────────────────────────────────────────────────────────────┤
 * │ PerfBufferState[num_cores-1]                                │
 * ├─────────────────────────────────────────────────────────────┤
 * │ AicpuPhaseHeader (optional, present when phase profiling)   │
 * │  - magic, num_sched_threads, records_per_thread             │
 * │  - orch_summary                                             │
 * ├─────────────────────────────────────────────────────────────┤
 * │ PhaseBufferState[thread0]                                   │
 * │  - free_queue: SPSC queue of available buffer pointers      │
 * │  - current_buf_ptr, current_buf_seq                         │
 * ├─────────────────────────────────────────────────────────────┤
 * │ PhaseBufferState[thread1]                                   │
 * ├─────────────────────────────────────────────────────────────┤
 * │ ...                                                         │
 * └─────────────────────────────────────────────────────────────┘
 *
 * Actual PerfBuffer / PhaseBuffer are allocated dynamically by Host
 * and pushed into the per-core/thread free_queue.
 *
 * Base size = sizeof(PerfDataHeader) + num_cores * sizeof(PerfBufferState)
 * With phases = Base + sizeof(AicpuPhaseHeader) + num_threads * sizeof(PhaseBufferState)
 */

#ifndef SRC_A2A3_PLATFORM_INCLUDE_COMMON_PERF_PROFILING_H_
#define SRC_A2A3_PLATFORM_INCLUDE_COMMON_PERF_PROFILING_H_

#include <cstdint>
#include <vector>

#include "common/core_type.h"
#include "common/platform_config.h"

// Maximum number of successor tasks per PerfRecord (matches Task::fanout)
#ifndef RUNTIME_MAX_FANOUT
#define RUNTIME_MAX_FANOUT 128
#endif

// =============================================================================
// PerfRecord - Single Task Execution Record
// =============================================================================

/**
 * Single task execution record
 */
struct PerfRecord {
    // Timing information (device clock timestamps)
    uint64_t start_time;  // Task start timestamp (get_sys_cnt)
    uint64_t end_time;    // Task end timestamp
    uint64_t duration;    // Execution duration (end - start)

    // AICPU-side timestamps (written by AICPU, not AICore)
    uint64_t dispatch_time;  // AICPU timestamp: when task was dispatched to AICore (task_status set to 1)
    uint64_t finish_time;    // AICPU timestamp: when AICPU observed task completion (task_status back to 0)

    // AICore writes the register dispatch token (low 32 bits only) zero-extended into task_id.
    // For multi-ring runtimes (tensormap_and_ringbuffer, aicpu_build_graph), AICPU overwrites
    // with the full PTO2 encoding (ring_id << 32) | local_id after FIN/perf row match.
    // For host_build_graph, task_id stays as the plain integer task index (ring_id = 0).
    uint64_t task_id;
    uint32_t func_id;    // Kernel function identifier
    CoreType core_type;  // Core type (AIC/AIV)

    // Dependency relationship (fanout only)
    uint64_t fanout[RUNTIME_MAX_FANOUT];  // Successor task task_id array
    int32_t fanout_count;                 // Number of successor tasks
} __attribute__((aligned(64)));

static_assert(sizeof(PerfRecord) % 64 == 0, "PerfRecord must be 64-byte aligned for optimal cache performance");

// =============================================================================
// PerfBuffer - Fixed-Size Record Buffer with WIP Staging Slots
// =============================================================================

/**
 * Fixed-size performance record buffer
 *
 * Capacity: PLATFORM_PROF_BUFFER_SIZE (defined in platform_config.h)
 * Allocated dynamically by Host, pushed into per-core free_queue.
 *
 * WIP protocol: AICore writes timing to wip[reg_task_id & 1], AICPU copies
 * it into records[count] at completion. Dual-slot parity ensures no overlap.
 */
struct PerfBuffer {
    PerfRecord wip[2];                              // AICore WIP staging slots (index = reg_task_id & 1)
    PerfRecord records[PLATFORM_PROF_BUFFER_SIZE];  // Committed records (AICPU writes)
    volatile uint32_t count;                        // Current committed record count
} __attribute__((aligned(64)));

// =============================================================================
// PerfFreeQueue - SPSC Lock-Free Queue for Free Buffers
// =============================================================================

/**
 * Single Producer Single Consumer (SPSC) lock-free queue for free buffer management
 *
 * Producer: Host (ProfMemoryManager thread) pushes newly allocated buffers
 * Consumer: Device (AICPU thread) pops buffers when switching
 *
 * Queue semantics:
 * - Empty: head == tail
 * - Full: (tail - head) >= PLATFORM_PROF_SLOT_COUNT
 * - Capacity: PLATFORM_PROF_SLOT_COUNT buffers
 *
 * Memory ordering:
 * - Device pop: rmb() → read tail → read buffer_ptrs[head % COUNT] → rmb() → write head → wmb()
 * - Host push: write buffer_ptrs[tail % COUNT] → wmb() → write tail → wmb()
 */
struct PerfFreeQueue {
    volatile uint64_t buffer_ptrs[PLATFORM_PROF_SLOT_COUNT];  // Free buffer addresses
    volatile uint32_t head;                                   // Consumer read position (Device increments)
    volatile uint32_t tail;                                   // Producer write position (Host increments)
    uint32_t pad[13];                                         // Pad to 128 bytes (aligned to cache line)
} __attribute__((aligned(64)));

static_assert(sizeof(PerfFreeQueue) == 128, "PerfFreeQueue must be 128 bytes for cache alignment");

// =============================================================================
// PerfBufferState - Per-Core/Thread Buffer State (Unified for PerfRecord and Phase)
// =============================================================================

/**
 * Per-core or per-thread buffer state for dynamic profiling
 *
 * Contains:
 * - free_queue: SPSC queue of available buffer addresses
 * - current_buf_ptr: Currently active buffer being written (0 = no active buffer)
 * - current_buf_seq: Monotonic sequence number for ordering
 *
 * Used in two contexts:
 * - Per-core PerfRecord profiling (current_buf_ptr → PerfBuffer)
 * - Per-thread Phase profiling (current_buf_ptr → PhaseBuffer)
 *
 * Writers:
 * - free_queue.tail: Host writes (pushes new buffers)
 * - free_queue.head: Device writes (pops buffers)
 * - current_buf_ptr: Device writes (after pop), Host reads (for flush/collect)
 * - current_buf_seq: Device writes (monotonic counter)
 */
struct PerfBufferState {
    PerfFreeQueue free_queue;           // SPSC queue of free buffer addresses
    volatile uint64_t current_buf_ptr;  // Current active buffer (0 = none)
    volatile uint32_t current_buf_seq;  // Sequence number for ordering
    uint32_t pad[13];                   // Pad to 192 bytes (aligned to cache line)
} __attribute__((aligned(64)));

static_assert(sizeof(PerfBufferState) == 192, "PerfBufferState must be 192 bytes for cache alignment");

// Type alias for semantic clarity in Phase profiling context
using PhaseBufferState = PerfBufferState;  // Per-thread Phase profiling

// =============================================================================
// ReadyQueueEntry - Queue Entry for Ready Buffers
// =============================================================================

/**
 * Ready queue entry
 *
 * When a buffer on a core/thread is full, AICPU adds this entry to the queue.
 * Host memory manager retrieves entries from the queue.
 *
 * Entry types (distinguished by is_phase flag):
 * - PerfRecord entry: core_index = core ID, is_phase = 0
 * - Phase entry:      core_index = thread_idx, is_phase = 1
 */
struct ReadyQueueEntry {
    uint32_t core_index;  // Core index (0 ~ num_cores-1), or thread_idx for phase entries
    uint32_t is_phase;    // 0 = PerfRecord, 1 = Phase
    uint64_t buffer_ptr;  // Device pointer to the full buffer
    uint32_t buffer_seq;  // Sequence number for ordering
    uint32_t pad;         // Alignment padding
} __attribute__((aligned(32)));

// =============================================================================
// PerfDataHeader - Fixed Header
// =============================================================================

/**
 * Performance data fixed header
 *
 * Located at the start of shared memory, contains:
 * 1. Per-thread ready queues (FIFO Circular Buffers)
 * 2. Metadata (core count)
 *
 * Ready queue design:
 * - Per-thread queues: Avoid lock contention between AICPU threads
 * - Capacity per queue: PLATFORM_PROF_READYQUEUE_SIZE (full capacity for each thread)
 * - Implementation: Circular Buffer
 * - Producer: AICPU thread (adds full buffers to its own queue)
 * - Consumer: Host memory manager thread (reads from all queues)
 * - Queue empty: head == tail
 * - Queue full: (tail + 1) % capacity == head
 */
struct PerfDataHeader {
    // Per-thread ready queues (FIFO Circular Buffers)
    // Each AICPU thread has its own queue to avoid lock contention
    ReadyQueueEntry queues[PLATFORM_MAX_AICPU_THREADS][PLATFORM_PROF_READYQUEUE_SIZE];
    volatile uint32_t queue_heads[PLATFORM_MAX_AICPU_THREADS];  // Consumer read positions (Host modifies)
    volatile uint32_t queue_tails[PLATFORM_MAX_AICPU_THREADS];  // Producer write positions (AICPU modifies)

    // Metadata (Host initializes, Device read-only)
    uint32_t num_cores;             // Actual number of cores launched
    volatile uint32_t total_tasks;  // Total tasks (AICPU writes after orchestration)
} __attribute__((aligned(64)));

// =============================================================================
// AICPU Phase Profiling - Scheduler and Orchestrator Records
// =============================================================================

/**
 * AICPU phase identifier
 *
 * Scheduler phases (0-3): four phases in each scheduler loop iteration.
 * Orchestrator phases (16-24): sub-steps within each pto2_submit_task() call.
 */
enum class AicpuPhaseId : uint32_t {
    // Scheduler phases (0-3)
    SCHED_COMPLETE = 0,     // Process completed tasks (fanout traversal)
    SCHED_DISPATCH = 1,     // Dispatch ready tasks to idle cores
    SCHED_SCAN = 2,         // Incremental scan for root tasks
    SCHED_IDLE_WAIT = 3,    // Idle/spinning (no progress)
    SCHED_PHASE_COUNT = 4,  // Sentinel: number of scheduler phases
    // Orchestrator phases (16-24)
    ORCH_SYNC = 16,      // tensormap sync
    ORCH_ALLOC = 17,     // task_ring_alloc
    ORCH_PARAMS = 18,    // param copy
    ORCH_LOOKUP = 19,    // tensormap lookup + dep
    ORCH_HEAP = 20,      // heap alloc
    ORCH_INSERT = 21,    // tensormap insert
    ORCH_FANIN = 22,     // fanin + early-ready
    ORCH_FINALIZE = 23,  // scheduler init + SM
    ORCH_SCOPE_END = 24  // scope_end
};

/**
 * Single AICPU scheduler phase record (32 bytes)
 *
 * Records one phase within one loop iteration of a scheduler thread.
 * No thread_id field: identity is derived from array index (position = identity).
 */
struct AicpuPhaseRecord {
    uint64_t start_time;    // Phase start timestamp
    uint64_t end_time;      // Phase end timestamp
    uint32_t loop_iter;     // Loop iteration number
    AicpuPhaseId phase_id;  // Phase type
    union {
        uint64_t task_id;          // Multi-ring runtimes (tensormap_and_ringbuffer, aicpu_build_graph):
                                   // full PTO2 encoding (ring_id << 32) | local_id for cross-view correlation.
        uint64_t tasks_processed;  // Scheduler phases: number of tasks processed in this batch
    };
};

/**
 * Access counter (mirrors PTO2CsvAccessCounters in the runtime layer):
 * read/write/atomic + atomic-read/atomic-write split + lock/cas.
 * POD-safe: usable in shared memory and on the host without runtime headers.
 */
struct AicpuCsvCounters {
    uint64_t read_events;
    uint64_t write_events;
    uint64_t atomic_ops;
    uint64_t atomic_read_ops;
    uint64_t atomic_write_ops;
    uint64_t lock_ops;
    uint64_t cas_ops;
} __attribute__((packed));

#ifndef AICPU_CSV_GLOSSARY_BUCKET_MAX
#define AICPU_CSV_GLOSSARY_BUCKET_MAX 32
#endif

/**
 * Task-kind glossary key mirrored from runtime-side PTO2CsvGlossaryTaskKindKey.
 * Packed to keep POD layout stable across host/device shared-memory exchange.
 */
struct __attribute__((packed)) AicpuCsvGlossaryTaskKindKey {
    int32_t kernel_aic;
    int32_t kernel_aiv0;
    int32_t kernel_aiv1;
    uint8_t active_mask;
    uint8_t ring_id;
    int8_t kind_tag;   // 0=mixed_incore, 1=alloc_tensors
    int16_t scope_depth;
    int32_t P_fanin_producers;
    int32_t C_fanout_minus_scope;
    int32_t S_subtasks;
    int32_t N_ring_acquire_proxy;
    int16_t N_in;
    int16_t N_out;
    int16_t tensor_count;
    int16_t scalar_count;
};

struct AicpuCsvGlossaryTaskKindBucket {
    AicpuCsvGlossaryTaskKindKey k;
    uint32_t submit_count;
};

struct AicpuCsvGlossaryStats {
    uint32_t bucket_count;
    AicpuCsvGlossaryTaskKindBucket buckets[AICPU_CSV_GLOSSARY_BUCKET_MAX];
};

/**
 * Per-thread scheduler profiling summary transferred via shared memory.
 *
 * Mirrors the fields of PTO2SchedProfilingData that are printed by DEV_ALWAYS.
 * Written by AICPU after pto2_scheduler_get_profiling(); read by host in
 * collect_phase_data() and printed by print_profiling_summary().
 */
struct AicpuSchedProfilingSummary {
    // Phase cycle breakdown (non-CSV fields)
    uint64_t lock_cycle;
    uint64_t fanout_cycle;
    uint64_t fanin_cycle;
    uint64_t self_consumed_cycle;
    uint64_t lock_wait_cycle;
    uint64_t push_wait_cycle;
    uint64_t pop_wait_cycle;
    uint64_t lock_atomic_count;
    uint64_t fanout_atomic_count;
    uint64_t fanin_atomic_count;
    uint64_t self_atomic_count;
    uint64_t pop_atomic_count;
    int64_t  complete_count;
    // CSV module counters ②~⑦
    AicpuCsvCounters csv_m2_pto2_task_slot_state;
    AicpuCsvCounters csv_m2_pto2_task_payload;
    AicpuCsvCounters csv_m2_pto2_dep_list_entry;
    AicpuCsvCounters csv_m2_pto2_ready_queue;
    uint64_t csv_m2_pto2_ready_queue_spin_retry_ops;
    AicpuCsvCounters csv_m4_pto2_task_slot_state;
    AicpuCsvCounters csv_m4_pto2_task_payload_meta;
    AicpuCsvCounters csv_m4_pto2_task_payload_tensors;
    AicpuCsvCounters csv_m4_pto2_task_payload_scalars;
    AicpuCsvCounters csv_m4_pto2_task_descriptor;
    AicpuCsvCounters csv_m4_pto2_dispatch_payload;
    AicpuCsvCounters csv_m4_pto2_ready_queue;
    uint64_t csv_m4_pto2_ready_queue_spin_retry_ops;
    uint64_t csv_m4_pto2_ready_queue_empty_poll_ops;
    AicpuCsvCounters csv_m4_pto2_ready_queue_pop_hit;
    AicpuCsvCounters csv_m4_pto2_ready_queue_pop_miss;
    AicpuCsvCounters csv_m5_pto2_task_slot_state;
    AicpuCsvCounters csv_m5_pto2_dispatch_payload;
    AicpuCsvCounters csv_m5_tensor;
    AicpuCsvCounters csv_m6_pto2_task_slot_state;
    AicpuCsvCounters csv_m6_pto2_dep_list_entry;
    AicpuCsvCounters csv_m6_pto2_ready_queue;
    AicpuCsvCounters csv_m7_pto2_task_slot_state;
    AicpuCsvCounters csv_m7_pto2_task_payload;
    AicpuCsvCounters csv_m7_pto2_ring_flow_control;
    AicpuCsvCounters csv_m7_pto2_fanin_spill_entry;
    AicpuCsvCounters csv_m7_ring_sched_state_advance_lock;
    // Explicit spin/poll/idle counters (avoid deriving by subtraction)
    uint64_t complete_poll_probe_count;
    uint64_t complete_poll_hit_count;
    uint64_t ready_queue_pop_hit_task_count;
    uint64_t ready_queue_pop_miss_round_count;
    uint64_t idle_no_progress_loop_count;
    uint32_t magic;  // AICPU_PHASE_MAGIC when valid
    uint32_t padding;
} __attribute__((aligned(64)));

/**
 * AICPU orchestrator cumulative summary
 *
 * Contains accumulated cycle counts from the orchestrator thread.
 * Written once after orchestration completes.
 */
struct AicpuOrchSummary {
    uint64_t start_time;       // Orchestrator start timestamp
    uint64_t end_time;         // Orchestrator end timestamp
    uint64_t sync_cycle;       // sync_tensormap phase
    uint64_t alloc_cycle;      // task_ring_alloc phase
    uint64_t args_cycle;       // param_copy phase
    uint64_t lookup_cycle;     // lookup+dep phase
    uint64_t heap_cycle;       // heap_alloc phase
    uint64_t insert_cycle;     // tensormap_insert phase
    uint64_t fanin_cycle;      // fanin+ready phase
    uint64_t scope_end_cycle;  // scope_end phase
    int64_t submit_count;      // Total tasks submitted
    AicpuCsvGlossaryStats csv_glossary;  // Task-kind buckets (P/C/S/N_in/N_out...)
    uint32_t magic;            // Validation magic (AICPU_PHASE_MAGIC)
    uint32_t padding;          // Alignment padding
} __attribute__((aligned(64)));

constexpr uint32_t AICPU_PHASE_MAGIC = 0x41435048;        // "ACPH"
constexpr int PLATFORM_PHASE_RECORDS_PER_THREAD = 16384;  // ~512KB per thread

/**
 * Fixed-size phase record buffer (analogous to PerfBuffer)
 *
 * Capacity: PLATFORM_PHASE_RECORDS_PER_THREAD
 * Allocated dynamically by Host, pushed into per-thread free_queue.
 */
struct PhaseBuffer {
    AicpuPhaseRecord records[PLATFORM_PHASE_RECORDS_PER_THREAD];
    volatile uint32_t count;
} __attribute__((aligned(64)));

/**
 * AICPU phase profiling header
 *
 * Located after the PerfBufferState array in shared memory.
 * Contains metadata and per-thread tracking.
 */
struct AicpuPhaseHeader {
    uint32_t magic;                             // Validation magic (AICPU_PHASE_MAGIC)
    uint32_t num_sched_threads;                 // Number of scheduler threads
    uint32_t records_per_thread;                // Max records per PhaseBuffer
    uint32_t num_cores;                         // Total number of cores with valid assignments
    int8_t core_to_thread[PLATFORM_MAX_CORES];  // core_id → scheduler thread index (-1 = unassigned)
    AicpuOrchSummary orch_summary;              // Orchestrator cumulative data
    AicpuSchedProfilingSummary sched_summary[PLATFORM_MAX_AICPU_THREADS];  // Per-thread scheduler CSV data
} __attribute__((aligned(64)));

// =============================================================================
// Helper Functions - Memory Layout
// =============================================================================

#ifdef __cplusplus
extern "C" {
#endif

/**
 * Calculate total memory size for performance data (buffer states only, no buffers)
 *
 * Formula: Total size = Fixed header + Dynamic tail
 *                     = sizeof(PerfDataHeader) + num_cores × sizeof(PerfBufferState)
 *
 * @param num_cores Number of cores (block_dim × PLATFORM_CORES_PER_BLOCKDIM)
 * @return Total bytes for header + buffer states
 */
inline size_t calc_perf_data_size(int num_cores) {
    return sizeof(PerfDataHeader) + num_cores * sizeof(PerfBufferState);
}

/**
 * Get header pointer
 *
 * @param base_ptr Shared memory base address (device_ptr or host_ptr)
 * @return PerfDataHeader pointer
 */
inline PerfDataHeader *get_perf_header(void *base_ptr) { return reinterpret_cast<PerfDataHeader *>(base_ptr); }

/**
 * Get PerfBufferState array start address
 *
 * @param base_ptr Shared memory base address
 * @return PerfBufferState array pointer
 */
inline PerfBufferState *get_perf_buffer_states(void *base_ptr) {
    return reinterpret_cast<PerfBufferState *>(reinterpret_cast<char *>(base_ptr) + sizeof(PerfDataHeader));
}

/**
 * Get PerfBufferState for specified core
 *
 * @param base_ptr Shared memory base address
 * @param core_index Core index (0 ~ num_cores-1)
 * @return PerfBufferState pointer
 */
inline PerfBufferState *get_perf_buffer_state(void *base_ptr, int core_index) {
    return &get_perf_buffer_states(base_ptr)[core_index];
}

/**
 * Calculate total memory size including phase profiling region (buffer states only)
 *
 * @param num_cores Number of AICore instances
 * @param num_sched_threads Number of phase profiling threads (scheduler + orchestrator)
 * @return Total bytes needed for header + all buffer states
 */
inline size_t calc_perf_data_size_with_phases(int num_cores, int num_sched_threads) {
    return calc_perf_data_size(num_cores) + sizeof(AicpuPhaseHeader) + num_sched_threads * sizeof(PhaseBufferState);
}

/**
 * Get AicpuPhaseHeader pointer (located after PerfBufferState array)
 *
 * @param base_ptr Shared memory base address
 * @param num_cores Number of AICore instances
 * @return AicpuPhaseHeader pointer
 */
inline AicpuPhaseHeader *get_phase_header(void *base_ptr, int num_cores) {
    return reinterpret_cast<AicpuPhaseHeader *>(reinterpret_cast<char *>(base_ptr) + calc_perf_data_size(num_cores));
}

/**
 * Get PhaseBufferState array start address (located after AicpuPhaseHeader)
 *
 * @param base_ptr Shared memory base address
 * @param num_cores Number of AICore instances
 * @return PhaseBufferState array pointer
 */
inline PhaseBufferState *get_phase_buffer_states(void *base_ptr, int num_cores) {
    return reinterpret_cast<PhaseBufferState *>(
        reinterpret_cast<char *>(get_phase_header(base_ptr, num_cores)) + sizeof(AicpuPhaseHeader)
    );
}

/**
 * Get PhaseBufferState for specified thread
 *
 * @param base_ptr Shared memory base address
 * @param num_cores Number of AICore instances
 * @param thread_idx Thread index
 * @return PhaseBufferState pointer
 */
inline PhaseBufferState *get_phase_buffer_state(void *base_ptr, int num_cores, int thread_idx) {
    return &get_phase_buffer_states(base_ptr, num_cores)[thread_idx];
}

#ifdef __cplusplus
}
#endif

#endif  // SRC_A2A3_PLATFORM_INCLUDE_COMMON_PERF_PROFILING_H_
