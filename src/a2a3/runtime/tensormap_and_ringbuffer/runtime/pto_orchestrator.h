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
 * PTO Runtime2 - Orchestrator Interface
 *
 * The Orchestrator is responsible for:
 * 1. Executing the orchestration function (Turing-complete control flow)
 * 2. Allocating intermediate buffers from the heap
 * 3. Submitting tasks via async InCore function calls
 * 4. Building the dependency graph using TensorMap
 * 5. Managing buffer scopes for lifecycle control
 *
 * The Orchestrator can run on either:
 * - Host CPU (lower latency for complex control, easier debugging)
 * - Device AI_CPU (lower latency for task submission)
 *
 * Based on: docs/RUNTIME_LOGIC.md
 */

#ifndef PTO_ORCHESTRATOR_H
#define PTO_ORCHESTRATOR_H

#include "pto2_csv_glossary_stats.h"
#include "pto_ring_buffer.h"
#include "pto_runtime2_types.h"
#include "pto_submit_types.h"
#include "pto_scheduler.h"
#include "pto_shared_memory.h"
#include "pto_tensormap.h"
#include "pto_types.h"

// =============================================================================
// Orchestrator State
// =============================================================================

/**
 * Orchestrator state structure (private to Orchestrator)
 *
 * Contains all state needed for task graph construction and buffer management.
 */
struct PTO2OrchestratorState {
    // === SHARED MEMORY ACCESS ===
    PTO2SharedMemoryHeader *sm_header;

    // === PER-RING RESOURCES ===
    PTO2RingSet rings[PTO2_MAX_RING_DEPTH];

    // === TENSOR MAP (Private) ===
    PTO2TensorMap tensor_map;  // Producer lookup

    // === SCOPE STACK (Private) ===
    // Single contiguous buffer of task IDs, partitioned by scope level.
    // scope_begins[i] is the index into scope_tasks where scope i starts.
    // Tasks for the top scope occupy [scope_begins[top], scope_tasks_size).
    PTO2TaskSlotState **scope_tasks;  // Flat buffer of taskSlotState (all scopes concatenated)
    int32_t scope_tasks_size;         // Number of task IDs currently in the buffer
    int32_t scope_tasks_capacity;     // Allocated capacity of scope_tasks
    int32_t *scope_begins;            // scope_begins[i] = start index of scope i in scope_tasks
    int32_t scope_stack_top;          // Current top of stack (-1 = no scope open)
    uint64_t scope_stack_capacity;    // Max nesting depth (PTO2_MAX_SCOPE_DEPTH)

    // === SCHEDULER REFERENCE ===
    // Note: In simulated mode, orchestrator and scheduler share address space
    // In real mode, they communicate via shared memory only
    PTO2SchedulerState *scheduler;  // For simulated mode only

    // Total core counts set once at executor init; used for submit-time deadlock detection.
    int32_t total_cluster_count{0};  // AIC cores = MIX clusters
    int32_t total_aiv_count{0};      // AIV cores (= 2 × clusters on standard hardware)
#if PTO2_PROFILING
    // Runtime profiling switch copied from Runtime::enable_profiling.
    bool enable_profiling;
#endif

    // === GM HEAP (for output buffers) ===
    void *gm_heap_base;     // Base address of GM heap
    uint64_t gm_heap_size;  // Total size of GM heap (all rings)

    // === FATAL ERROR ===
    // Fatal error flag (single-thread access by orchestrator, no atomic needed)
    // Cross-thread notification uses shared memory orch_error_code (atomic)
    bool fatal;

    // Hidden alloc tasks complete synchronously inside the orchestrator and
    // therefore bypass the executor's normal worker-completion counter path.
    // The executor adds this count into its completed_tasks_ progress counter
    // after orchestration finishes so shutdown/profiling totals remain closed.
    int64_t inline_completed_tasks{0};

    // === STATISTICS ===
#if PTO2_PROFILING
    int64_t tasks_submitted;
    int64_t buffers_allocated;
    int64_t bytes_allocated;
#endif
#if PTO2_ORCH_PROFILING
    /**
     * module-struct-access.csv 行 1–9：按任务形状聚合（与 submit 传入的 orch 同址）。
     * 放在此结构中而非文件静态变量，避免编排 .so 与 aicpu 主模块各有一份静态区导致 bucket 始终为空。
     */
    PTO2CsvGlossaryStats csv_glossary;
#endif

    /**
     * Get current ring index from scope depth.
     * Maps scope depth to ring_id: min(scope_depth, PTO2_MAX_RING_DEPTH - 1)
     */
    uint8_t current_ring_id() const {
        int32_t depth = scope_stack_top;
        if (depth < 0) depth = 0;
        return depth < PTO2_MAX_RING_DEPTH ? static_cast<uint8_t>(depth) : PTO2_MAX_RING_DEPTH - 1;
    }
};

// =============================================================================
// Orchestrator API
// =============================================================================

/**
 * Initialize orchestrator state
 *
 * @param orch       Orchestrator state to initialize
 * @param sm_header  Shared memory header
 * @param gm_heap    GM heap memory for output buffers
 * @param heap_size  Size of GM heap
 * @return true on success
 */
bool pto2_orchestrator_init(
    PTO2OrchestratorState *orch, PTO2SharedMemoryHeader *sm_header, void *gm_heap, uint64_t heap_size,
    int32_t dep_pool_capacity = PTO2_DEP_LIST_POOL_SIZE
);

/**
 * Destroy orchestrator state and free resources
 */
void pto2_orchestrator_destroy(PTO2OrchestratorState *orch);

/**
 * Set scheduler reference (for simulated mode)
 */
void pto2_orchestrator_set_scheduler(PTO2OrchestratorState *orch, PTO2SchedulerState *scheduler);

// =============================================================================
// Fatal Reporting
// =============================================================================

void pto2_orch_report_fatal(PTO2OrchestratorState *orch, int32_t error_code, const char *func, const char *fmt, ...);

// =============================================================================
// Scope Management
// =============================================================================

/**
 * Begin a new scope
 *
 * Pushes a new empty task list onto the scope stack.
 * Tasks submitted while this scope is at the top of the stack are
 * owned by it and have their fanout_count initialized to 1.
 */
void pto2_scope_begin(PTO2OrchestratorState *orch);

/**
 * End current scope
 *
 * Pops the top scope and increments fanout_refcount for each task
 * directly owned by that scope.
 * May trigger buffer release for tasks that are now fully consumed.
 */
void pto2_scope_end(PTO2OrchestratorState *orch);

// =============================================================================
// Task Submission
// =============================================================================

/**
 * Submit a task with InCore function and parameters
 *
 * This is the main API for building the task graph:
 * 1. Allocates task slot + packed output buffer via TaskAllocator (blocks until available)
 * 2. Looks up inputs in TensorMap to find dependencies
 * 3. Updates producer's fanout_count/list (with spinlock)
 * 4. Registers outputs in TensorMap
 * 5. Initializes task state in scheduler
 *
 * @param orch        Orchestrator state
 * @param mixed_kernels  Kernel IDs for AIC/AIV0/AIV1 slots
 * @param args      Aggregated tensor and scalar parameters
 */
TaskOutputTensors
pto2_submit_mixed_task(PTO2OrchestratorState *orch, const MixedKernels &mixed_kernels, const Arg &args);

/**
 * Allocate fresh tensors by creating one hidden runtime-owned output task.
 *
 * The returned tensors are already materialized and bound to the same creator
 * task id for scope lifetime and future creator-retention dependencies.
 */
TaskOutputTensors pto2_alloc_tensors(PTO2OrchestratorState *orch, const Arg &args);

// =============================================================================
// Flow Control
// =============================================================================

/**
 * Mark orchestration as complete
 *
 * Signals to scheduler that no more tasks will be submitted.
 */
void pto2_orchestrator_done(PTO2OrchestratorState *orch);

// =============================================================================
// Debug Utilities
// =============================================================================

/**
 * Print orchestrator statistics
 */
void pto2_orchestrator_print_stats(PTO2OrchestratorState *orch);

/**
 * Print scope stack state
 */
void pto2_orchestrator_print_scope_stack(PTO2OrchestratorState *orch);

// =============================================================================
// Orchestrator Profiling Data
// =============================================================================

#if PTO2_ORCH_PROFILING
struct PTO2OrchProfilingData {
    uint64_t sync_cycle;        // 周期：TensorMap sync_tensormap
    uint64_t alloc_cycle;       // 周期：统一 alloc（任务槽 + heap）
    uint64_t args_cycle;        // 周期：参数/GM 批量写
    uint64_t lookup_cycle;      // 周期：TensorMap lookup + 依赖收集
    uint64_t insert_cycle;      // 周期：TensorMap insert
    uint64_t fanin_cycle;       // 周期：fanin 元数据 + wiring_queue 入队
    uint64_t scope_end_cycle;   // 周期：pto2_scope_end 整段
    int64_t submit_count;       // 已提交任务数（编排统计，非 CSV 五列）
    uint64_t alloc_wait_cycle;  // alloc 内自旋等待周期（流控/heap 反压）
    uint64_t fanin_wait_cycle;  // fanout_lock 等等待周期（若路径启用）
    uint64_t alloc_atomic_count;     // alloc 路径原子累计（实现口径）
    uint64_t args_atomic_count;      // 参数阶段原子累计（如 fanout_lock.store 等）
    uint64_t fanin_atomic_count;     // fanin/ready 阶段原子累计
    uint64_t finalize_atomic_count;  // finalize 路径原子累计
    uint64_t scope_end_atomic_count; // scope_end→release_producer 链上原子累计（用于 CSV ① SlotState atomic 近似）
    /** CSV ① 行「PTO2TaskSlotState」五列；由 pto2_orchestrator_get_profiling() 从 g_orch_* 原始事件填出 */
    PTO2CsvAccessCounters csv_m1_pto2_task_slot_state;
    /** CSV ① 行「PTO2TaskPayload」五列（整段 payload 批量写等） */
    PTO2CsvAccessCounters csv_m1_pto2_task_payload;
    /** CSV ① 行「PTO2TaskDescriptor」五列 */
    PTO2CsvAccessCounters csv_m1_pto2_task_descriptor;
    /** CSV ① 行「Tensor」五列（INPUT/INOUT 查表、OUTPUT 写 owner 等） */
    PTO2CsvAccessCounters csv_m1_tensor;
    /** CSV ① 行「PTO2ReadyQueue」五列（wiring_queue 入队等） */
    PTO2CsvAccessCounters csv_m1_pto2_ready_queue;
    /** CSV ①/③ 行「PTO2RingFlowControl」在编排侧 alloc 路径上的五列 */
    PTO2CsvAccessCounters csv_m1_pto2_ring_flow_control;
    /** module-struct-access.csv 行 1–9 符号：按任务形状聚合的 submit 次数（与下述 CSV 五列同一次 flush） */
    PTO2CsvGlossaryStats csv_glossary;
};

/**
 * Get and reset orchestrator profiling data.
 * Returns accumulated profiling data and resets counters.
 */
PTO2OrchProfilingData pto2_orchestrator_get_profiling(PTO2OrchestratorState *orch);
#endif

#endif  // PTO_ORCHESTRATOR_H
