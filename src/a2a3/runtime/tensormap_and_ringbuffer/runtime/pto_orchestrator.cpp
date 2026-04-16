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
 * PTO Runtime2 - Orchestrator Implementation
 *
 * Implements orchestrator state management, scope handling, and task submission.
 *
 * Based on: docs/RUNTIME_LOGIC.md
 */

#include "pto_orchestrator.h"

#include <assert.h>
#include <inttypes.h>
#include <stdio.h>
#include <stdarg.h>
#include <stdlib.h>
#include <string.h>

#include "common/unified_log.h"
#include "pto_runtime2_types.h"
#include "pto_shared_memory.h"
#include "pto_tensormap.h"
#include "pto_types.h"
#include "tensor.h"

// =============================================================================
// Orchestrator Profiling (compile-time toggle)
// =============================================================================
#if PTO2_ORCH_PROFILING
#include "aicpu/device_time.h"
#include "aicpu/performance_collector_aicpu.h"
// Weak fallback for builds that don't link device_time.cpp (e.g. host).
// The strong symbol from platform/.../device_time.cpp wins in the AICPU build.
//
// IMPORTANT: visibility("hidden") is required to prevent the HOST .so from
// exporting this weak fallback into the global dynamic symbol table via
// RTLD_GLOBAL. Without it, when the AICPU .so is loaded and its PLT entry
// for get_sys_cnt_aicpu is resolved, the dynamic linker finds the HOST .so's
// weak definition first (already in global table) and uses it — returning 0.
// With hidden visibility, the HOST .so does not export this symbol globally,
// so the AICPU .so's PLT resolves to its own strong definition from
// device_time.cpp.
__attribute__((weak, visibility("hidden"))) uint64_t get_sys_cnt_aicpu() { return 0; }
// Weak fallback for builds that don't link performance_collector_aicpu.cpp.
// The strong symbol from the AICPU build wins when profiling is available.
// Also hidden to prevent HOST .so from polluting the global symbol table.
__attribute__((weak, visibility("hidden"))) void
perf_aicpu_record_orch_phase(AicpuPhaseId, uint64_t, uint64_t, uint32_t, uint64_t) {}
// ---------- 编排器周期量（CPU cycle，非 CSV 五列）----------
static uint64_t g_orch_sync_cycle = 0;
static uint64_t g_orch_alloc_cycle = 0;
static uint64_t g_orch_args_cycle = 0;
static uint64_t g_orch_lookup_cycle = 0;
static uint64_t g_orch_insert_cycle = 0;
static uint64_t g_orch_fanin_cycle = 0;
static uint64_t g_orch_scope_end_cycle = 0;
static int64_t g_orch_submit_count = 0;   // 成功 submit 路径累计任务数
static uint32_t g_orch_submit_idx = 0;    // 泳道/phase 记录用单调序号
/** alloc 在 TaskAllocator 自旋等待上消耗的 CPU cycle（PTO2OrchProfilingData.alloc_wait_cycle） */
uint64_t g_orch_alloc_wait_cycle = 0;
/** fanin+ready 阶段若统计等待则写入（PTO2OrchProfilingData.fanin_wait_cycle；当前热路径多为 0） */
uint64_t g_orch_fanin_wait_cycle = 0;
/** alloc 子阶段原子/自旋 load 累计（PTO2OrchProfilingData.alloc_atomic_count；含 record_wait 中 spin+1） */
uint64_t g_orch_alloc_atomic_count = 0;
/** 参数阶段 fanout_lock/fanout_count 等原子写累计（PTO2OrchProfilingData.args_atomic_count） */
uint64_t g_orch_args_atomic_count = 0;
/** fanin/ready 路径原子累计（PTO2OrchProfilingData.fanin_atomic_count） */
uint64_t g_orch_fanin_atomic_count = 0;
/** finalize 路径原子累计（预留；PTO2OrchProfilingData.finalize_atomic_count） */
uint64_t g_orch_finalize_atomic_count = 0;
/** scope_end→scheduler::release_producer 链上累计的原子/RWM 次数（汇入 CSV ① PTO2TaskSlotState.atomic_ops） */
uint64_t g_orch_scope_end_atomic_count = 0;
uint64_t g_orch_scope_end_atomic_read_count = 0;
uint64_t g_orch_scope_end_atomic_write_count = 0;
// ---------- 原始事件计数：在 pto2_orchestrator_get_profiling(orch) 中映射为 CSV ① 各「读/写/atomic/锁/CAS」----------
/** CSV ① PTO2TaskSlotState 写口径之一：各 producer 上 fanout_count++ 次数，累加为 ∑P */
static uint64_t g_orch_fanout_increment_count = 0;
/** CSV ① scope_end 行：pto2_scope_end 内 release_producer 调用次数，即 ∑N_scope */
static uint64_t g_orch_scope_end_release_count = 0;
/** CSV ① PTO2TaskSlotState 写：本任务 SlotState 字段批量初始化次数（每提交任务 1） */
static uint64_t g_orch_slot_state_init_count = 0;
/** CSV ① PTO2TaskPayload 写：payload->init 批量写次数（每任务 1） */
static uint64_t g_orch_payload_init_count = 0;
/** CSV ① PTO2TaskDescriptor 写：task 描述符批量写次数（每任务 1） */
static uint64_t g_orch_descriptor_write_count = 0;
/** CSV ① Tensor 读：INPUT/INOUT 进入 TensorMap lookup 前计数，对应 N_in 维度的查表次数 */
static uint64_t g_orch_tensor_input_read_count = 0;
/** CSV ① Tensor 写：OUTPUT/INOUT/OUTPUT_EXISTING 写 payload 内 owner_task_id 次数，与 N_out 槽位计数口径一致 */
static uint64_t g_orch_tensor_output_write_count = 0;
/** CSV ①③ RingFlowControl：TaskAllocator::alloc 成功返回次数（含 commit 写流控字） */
static uint64_t g_orch_ring_fc_alloc_count = 0;
/** CSV ①③ RingFlowControl 读：对 last_task_alive 的 memory_order_acquire 次数（含 alloc 入口与 spin） */
uint64_t g_orch_ring_fc_last_alive_acquire_reads = 0;
/** CSV ① PTO2ReadyQueue 写：wiring_queue.push 成功次数 */
static uint64_t g_orch_m1_wiring_queue_push_done = 0;

/** 见 g_orch_m1_wiring_queue_push_done：每成功 push 调用一次 */
void pto2_orch_profile_csv_m1_wiring_queue_push_done() { g_orch_m1_wiring_queue_push_done++; }

// ---------- module-struct-access.csv 行 1–9：聚合在 PTO2OrchestratorState::csv_glossary（与 dlopen 编排 .so 共用 orch 指针）----------
static bool pto2_csv_glossary_key_equal(const PTO2CsvGlossaryTaskKindKey &a, const PTO2CsvGlossaryTaskKindKey &b) {
    return memcmp(&a, &b, sizeof(a)) == 0;
}

static void pto2_csv_glossary_record(PTO2OrchestratorState *orch, const PTO2CsvGlossaryTaskKindKey &key) {
    PTO2CsvGlossaryStats *gs = &orch->csv_glossary;
    for (uint32_t i = 0; i < gs->bucket_count; i++) {
        if (pto2_csv_glossary_key_equal(gs->buckets[i].k, key)) {
            gs->buckets[i].submit_count++;
            return;
        }
    }
    if (gs->bucket_count < PTO2_CSV_GLOSSARY_BUCKET_MAX) {
        const uint32_t idx = gs->bucket_count++;
        gs->buckets[idx].k = key;
        gs->buckets[idx].submit_count = 1;
    }
}

static void pto2_csv_glossary_count_n_in_n_out(const Arg &args, int16_t *n_in, int16_t *n_out) {
    *n_in = 0;
    *n_out = 0;
    for (int32_t i = 0; i < args.tensor_count(); i++) {
        const TensorArgType ptype = args.tag(i);
        if (ptype == TensorArgType::INPUT) {
            (*n_in)++;
        } else if (ptype == TensorArgType::OUTPUT || ptype == TensorArgType::OUTPUT_EXISTING) {
            (*n_out)++;
        } else if (ptype == TensorArgType::INOUT) {
            (*n_in)++;
            (*n_out)++;
        }
    }
}

/** 每次成功 pto2_submit_mixed_task：采样 P/C/S、N_in/out、tensor/scalar_count、scope 深度（见 CSV 注释） */
static void pto2_csv_glossary_record_mixed_submit(
    PTO2OrchestratorState *orch, const PTO2TaskDescriptor &task, const PTO2TaskSlotState &slot,
    const PTO2TaskPayload &payload, const Arg &args, int32_t fanin_producers
) {
    int16_t n_in = 0;
    int16_t n_out = 0;
    pto2_csv_glossary_count_n_in_n_out(args, &n_in, &n_out);
    PTO2CsvGlossaryTaskKindKey k{};
    k.kernel_aic = task.kernel_id[static_cast<int>(PTO2SubtaskSlot::AIC)];
    k.kernel_aiv0 = task.kernel_id[static_cast<int>(PTO2SubtaskSlot::AIV0)];
    k.kernel_aiv1 = task.kernel_id[static_cast<int>(PTO2SubtaskSlot::AIV1)];
    k.active_mask = slot.active_mask;
    k.ring_id = slot.ring_id;
    k.kind_tag = 0;
    k.scope_depth = static_cast<int16_t>((orch->scope_stack_top >= 0) ? (orch->scope_stack_top + 1) : 0);
    k.P_fanin_producers = fanin_producers;
    k.C_fanout_minus_scope = (slot.fanout_count > 0) ? (slot.fanout_count - 1) : 0;
    k.S_subtasks = slot.total_required_subtasks;
    k.N_ring_acquire_proxy = 0;
    k.N_in = n_in;
    k.N_out = n_out;
    k.tensor_count = static_cast<int16_t>(payload.tensor_count);
    k.scalar_count = static_cast<int16_t>(payload.scalar_count);
    pto2_csv_glossary_record(orch, k);
}

/** pto2_alloc_tensors 等无 InCore 的隐藏任务 */
static void pto2_csv_glossary_record_alloc_hidden(
    PTO2OrchestratorState *orch, const PTO2TaskSlotState *slot, const PTO2TaskPayload &payload, const Arg &args,
    uint8_t ring_id
) {
    if (slot == nullptr) {
        return;
    }
    int16_t n_in = 0;
    int16_t n_out = 0;
    pto2_csv_glossary_count_n_in_n_out(args, &n_in, &n_out);
    PTO2CsvGlossaryTaskKindKey k{};
    k.kernel_aic = INVALID_KERNEL_ID;
    k.kernel_aiv0 = INVALID_KERNEL_ID;
    k.kernel_aiv1 = INVALID_KERNEL_ID;
    k.active_mask = 0;
    k.ring_id = ring_id;
    k.kind_tag = 1;
    k.scope_depth = static_cast<int16_t>((orch->scope_stack_top >= 0) ? (orch->scope_stack_top + 1) : 0);
    k.P_fanin_producers = 0;
    k.C_fanout_minus_scope = (slot->fanout_count > 0) ? (slot->fanout_count - 1) : 0;
    k.S_subtasks = slot->total_required_subtasks;
    k.N_ring_acquire_proxy = 0;
    k.N_in = n_in;
    k.N_out = n_out;
    k.tensor_count = static_cast<int16_t>(payload.tensor_count);
    k.scalar_count = static_cast<int16_t>(payload.scalar_count);
    pto2_csv_glossary_record(orch, k);
}

#define CYCLE_COUNT_START() uint64_t _t0 = get_sys_cnt_aicpu(), _t1
#define CYCLE_COUNT_LAP(acc)       \
    do {                           \
        _t1 = get_sys_cnt_aicpu(); \
        acc += (_t1 - _t0);        \
        _t0 = _t1;                 \
    } while (0)
#define CYCLE_COUNT_LAP_RECORD(acc, phase_id, tid)                                    \
    do {                                                                              \
        _t1 = get_sys_cnt_aicpu();                                                    \
        acc += (_t1 - _t0);                                                           \
        perf_aicpu_record_orch_phase((phase_id), _t0, _t1, g_orch_submit_idx, (tid)); \
        _t0 = _t1;                                                                    \
    } while (0)
#elif PTO2_PROFILING
#include "aicpu/device_time.h"
#include "aicpu/performance_collector_aicpu.h"
__attribute__((weak, visibility("hidden"))) uint64_t get_sys_cnt_aicpu() { return 0; }
__attribute__((weak, visibility("hidden"))) void
perf_aicpu_record_orch_phase(AicpuPhaseId, uint64_t, uint64_t, uint32_t, uint64_t) {}
// submit_idx needed for swimlane task_id tagging (no cycle accumulation at this level)
static uint32_t g_orch_submit_idx = 0;
#define CYCLE_COUNT_START()                     \
    bool _prof_active = orch->enable_profiling; \
    uint64_t _t0 = _prof_active ? get_sys_cnt_aicpu() : 0, _t1 = 0
#define CYCLE_COUNT_LAP(acc) \
    do {                     \
    } while (0)
#define CYCLE_COUNT_LAP_RECORD(acc, phase_id, tid)                                        \
    do {                                                                                  \
        if (_prof_active) {                                                               \
            _t1 = get_sys_cnt_aicpu();                                                    \
            perf_aicpu_record_orch_phase((phase_id), _t0, _t1, g_orch_submit_idx, (tid)); \
            _t0 = _t1;                                                                    \
        }                                                                                 \
    } while (0)
#else
#define CYCLE_COUNT_START()
#define CYCLE_COUNT_LAP(acc)
#define CYCLE_COUNT_LAP_RECORD(acc, phase_id, tid)
#endif

static void *pto2_aligned_zalloc(size_t size, size_t alignment) {
    void *ptr = nullptr;
    if (posix_memalign(&ptr, alignment, size) != 0) {
        return nullptr;
    }
    memset(ptr, 0, size);
    return ptr;
}

static int32_t pto2_orch_mark_fatal(PTO2OrchestratorState *orch, int32_t error_code) {
    always_assert(orch != nullptr);
    orch->fatal = true;
    if (error_code == PTO2_ERROR_NONE || orch->sm_header == nullptr) {
        return PTO2_ERROR_NONE;
    }

    int32_t expected = PTO2_ERROR_NONE;
    std::atomic<int32_t> &orch_error_code = orch->sm_header->orch_error_code;
    if (orch_error_code.compare_exchange_strong(expected, error_code, std::memory_order_acq_rel)) {
        return error_code;
    }
    return expected;
}

static void pto2_orch_report_fatal_v(
    PTO2OrchestratorState *orch, int32_t error_code, const char *func, const char *fmt, va_list args
) {
    int32_t latched_code = pto2_orch_mark_fatal(orch, error_code);

    if (fmt == nullptr || fmt[0] == '\0') {
        if (latched_code != PTO2_ERROR_NONE && latched_code != error_code) {
            unified_log_error(func, "FATAL(code=%d, latched=%d)", error_code, latched_code);
        } else {
            unified_log_error(func, "FATAL(code=%d)", error_code);
        }
        return;
    }

    char message[1024];
    vsnprintf(message, sizeof(message), fmt, args);
    if (latched_code != PTO2_ERROR_NONE && latched_code != error_code) {
        unified_log_error(func, "FATAL(code=%d, latched=%d): %s", error_code, latched_code, message);
        return;
    }
    unified_log_error(func, "FATAL(code=%d): %s", error_code, message);
}

void pto2_orch_report_fatal(PTO2OrchestratorState *orch, int32_t error_code, const char *func, const char *fmt, ...) {
    va_list args;
    va_start(args, fmt);
    pto2_orch_report_fatal_v(orch, error_code, func, fmt, args);
    va_end(args);
}

struct PTO2FaninBuilder {
    PTO2FaninBuilder(PTO2FaninPool &spill_pool) :
        count(0),
        spill_start(0),
        spill_pool(spill_pool) {}
    int32_t count{0};
    int32_t spill_start{0};
    PTO2FaninPool &spill_pool;
    PTO2TaskSlotState *inline_slots[PTO2_FANIN_INLINE_CAP];

    template <typename Fn>
    PTO2FaninForEachReturn<Fn> for_each(Fn &&fn) const {
        return pto2_for_each_fanin_storage(inline_slots, count, spill_start, spill_pool, static_cast<Fn &&>(fn));
    }

    bool contains(PTO2TaskSlotState *prod_state) const {
        bool found = false;
        for_each([&](PTO2TaskSlotState *slot_state) {
            if (slot_state == prod_state) {
                found = true;
                return false;
            }
            return true;
        });
        if (found) {
            return true;
        }
        return false;
    }
};

static bool pto2_append_fanin_or_fail(
    PTO2OrchestratorState *orch, PTO2TaskSlotState *prod_state, PTO2FaninBuilder *fanin_builder, uint8_t ring_id
) {
    if (fanin_builder->contains(prod_state)) {
        return true;
    }

    if (fanin_builder->count < PTO2_FANIN_INLINE_CAP) {
        fanin_builder->inline_slots[fanin_builder->count++] = prod_state;
        return true;
    }

    PTO2FaninPool &fanin_pool = fanin_builder->spill_pool;
    fanin_pool.ensure_space(orch->sm_header->rings[ring_id], 1);
    int32_t spill_idx = fanin_pool.top;
    PTO2FaninSpillEntry *entry = fanin_pool.alloc();
    if (entry == nullptr) {
        pto2_orch_mark_fatal(orch, PTO2_ERROR_DEP_POOL_OVERFLOW);
        return false;
    }
    if (fanin_builder->count == PTO2_FANIN_INLINE_CAP) {
        fanin_builder->spill_start = spill_idx;
    }
    entry->slot_state = prod_state;
    fanin_builder->count++;
    return true;
}

static void scope_tasks_push(PTO2OrchestratorState *orch, PTO2TaskSlotState *task_slot_state);

struct PTO2PreparedTask {
    PTO2TaskId task_id = PTO2TaskId::invalid();
    PTO2TaskAllocResult alloc_result = {-1, 0, nullptr, nullptr};
    PTO2TaskDescriptor *task = nullptr;
    PTO2TaskPayload *payload = nullptr;
    PTO2TaskSlotState *slot_state = nullptr;
};

static PTO2OutputLayout pto2_calculate_output_layout(const Arg &args) {
    PTO2OutputLayout layout;
    for (int32_t i = 0; i < args.tensor_count(); i++) {
        if (args.tag(i) != TensorArgType::OUTPUT) {
            continue;
        }
        layout.offsets[i] = layout.total_output_size;
        layout.buffer_sizes[i] =
            PTO2_ALIGN_UP(args.tensor(i).create_info->buffer_size_bytes(), PTO2_PACKED_OUTPUT_ALIGN);
        layout.total_output_size += layout.buffer_sizes[i];
    }
    return layout;
}

static bool
pto2_check_scope_can_accept_task(PTO2OrchestratorState *orch, PTO2TaskAllocator &allocator, uint8_t ring_id) {
    always_assert(orch->scope_stack_top >= 0 && "Cannot submit task outside a scope");

    int32_t scope_task_count = orch->scope_tasks_size - orch->scope_begins[orch->scope_stack_top];
    if (scope_task_count < allocator.window_size() - 1) {
        return true;
    }

    int32_t active_count = allocator.active_count();

    LOG_ERROR("========================================");
    LOG_ERROR("FATAL: Scope Deadlock Detected! (ring %d)", ring_id);
    LOG_ERROR("========================================");
    LOG_ERROR("Tasks in current scope (%d) >= task_window_size (%d).", scope_task_count, allocator.window_size());
    LOG_ERROR("  scope_depth:        %d", orch->scope_stack_top + 1);
    LOG_ERROR("  ring_id:            %d", ring_id);
    LOG_ERROR("  scope_task_count:   %d", scope_task_count);
    LOG_ERROR("  active_tasks:       %d / %d", active_count, allocator.window_size());
    LOG_ERROR("Root Cause:");
    LOG_ERROR("  Tasks within a scope hold a fanout_count reference that is only");
    LOG_ERROR("  released at scope_end. When scope task count >= window_size,");
    LOG_ERROR("  no slots can be reclaimed -> deadlock.");
    LOG_ERROR("Solution:");
    LOG_ERROR("  1. Reduce tasks per scope (use batching/unroll)");
    LOG_ERROR("  2. Increase task window (current: %d)", allocator.window_size());
    LOG_ERROR("     Compile-time: PTO2_TASK_WINDOW_SIZE in pto_runtime2_types.h");
    LOG_ERROR("     Runtime env:  PTO2_RING_TASK_WINDOW=<power-of-2>");
    LOG_ERROR("  3. Split work across multiple scopes");
    LOG_ERROR("========================================");
    pto2_orch_mark_fatal(orch, PTO2_ERROR_SCOPE_DEADLOCK);
    return false;
}

static void pto2_prefetch_payload(PTO2TaskPayload *payload, int32_t tensor_count, int32_t scalar_count) {
    for (int32_t i = 0; i < tensor_count; i++) {
        __builtin_prefetch(&payload->tensors[i], 1, 3);
        __builtin_prefetch(reinterpret_cast<char *>(&payload->tensors[i]) + 64, 1, 3);
    }
    for (int32_t i = 0; i < scalar_count; i += 8) {
        __builtin_prefetch(&payload->scalars[i], 1, 3);
    }
    __builtin_prefetch(payload, 1, 3);
    __builtin_prefetch(reinterpret_cast<char *>(payload) + 64, 1, 3);
    __builtin_prefetch(reinterpret_cast<char *>(payload) + 128, 1, 3);
}

static bool pto2_prepare_task(
    PTO2OrchestratorState *orch, const Arg &args, int32_t total_output_size, uint8_t active_mask, PTO2PreparedTask *out
) {
    uint8_t ring_id = orch->current_ring_id();
    auto &allocator = orch->rings[ring_id].task_allocator;

    if (!pto2_check_scope_can_accept_task(orch, allocator, ring_id)) {
        return false;
    }

    out->alloc_result = allocator.alloc(total_output_size);
    if (out->alloc_result.failed()) {
        pto2_orch_mark_fatal(orch, PTO2_ERROR_HEAP_RING_DEADLOCK);
        return false;
    }
#if PTO2_ORCH_PROFILING
    g_orch_ring_fc_alloc_count++;  // 见 g_orch_ring_fc_alloc_count：每任务一次成功 alloc+commit
#endif

    out->task_id = PTO2TaskId::make(ring_id, static_cast<uint32_t>(out->alloc_result.task_id));
    out->slot_state = &orch->sm_header->rings[ring_id].get_slot_state_by_slot(out->alloc_result.slot);
    out->task = &orch->sm_header->rings[ring_id].task_descriptors[out->alloc_result.slot];
    out->payload = &orch->sm_header->rings[ring_id].task_payloads[out->alloc_result.slot];

    pto2_prefetch_payload(out->payload, args.tensor_count(), args.scalar_count());

    // Fields already reset by advance_ring_pointers (eager reset after CONSUMED):
    //   fanout_lock=0, fanout_count=1, fanout_head=nullptr,
    //   fanin_refcount=0, fanout_refcount=0, completed_subtasks=0, next_block_idx=0
    // Fields immutable after RingSchedState::init():
    //   payload, task, ring_id
    // task_state left as CONSUMED by eager reset (safe for stale wait_for_tensor
    // observers); set to PENDING here when orchestrator actually reuses the slot.
    out->slot_state->task_state.store(PTO2_TASK_PENDING, std::memory_order_relaxed);
    int16_t block_num = args.launch_spec.block_num();
    out->slot_state->total_required_subtasks =
        static_cast<int16_t>(block_num * __builtin_popcount(pto2_core_mask(active_mask)));
    out->slot_state->logical_block_num = block_num;
    out->slot_state->active_mask = active_mask;
    // fanin_count is set by scheduler during wiring
    scope_tasks_push(orch, out->slot_state);
#if PTO2_ORCH_PROFILING
    g_orch_slot_state_init_count++;  // 见 g_orch_slot_state_init_count
#endif

    return true;
}

// =============================================================================
// Orchestrator Initialization
// =============================================================================

bool pto2_orchestrator_init(
    PTO2OrchestratorState *orch, PTO2SharedMemoryHeader *sm_header, void *gm_heap, uint64_t heap_size,
    int32_t dep_pool_capacity
) {
    *orch = PTO2OrchestratorState{};

    orch->sm_header = sm_header;
    orch->gm_heap_base = gm_heap;
    orch->gm_heap_size = heap_size * PTO2_MAX_RING_DEPTH;
    orch->fatal = false;

    // Initialize per-ring resources
    for (int r = 0; r < PTO2_MAX_RING_DEPTH; r++) {
        void *ring_heap_base = reinterpret_cast<char *>(gm_heap) + r * heap_size;
        auto &ring = sm_header->rings[r];

        // Initialize unified task allocator
        orch->rings[r].task_allocator.init(
            ring.task_descriptors, ring.task_window_size, &ring.fc.current_task_index, &ring.fc.last_task_alive,
            ring_heap_base, heap_size, &sm_header->orch_error_code
        );

        size_t fanin_pool_bytes =
            PTO2_ALIGN_UP(static_cast<size_t>(dep_pool_capacity) * sizeof(PTO2FaninSpillEntry), PTO2_ALIGN_SIZE);
        PTO2FaninSpillEntry *fanin_entries =
            reinterpret_cast<PTO2FaninSpillEntry *>(pto2_aligned_zalloc(fanin_pool_bytes, PTO2_ALIGN_SIZE));
        if (!fanin_entries) {
            for (int j = 0; j < r; j++) {
                free(orch->rings[j].fanin_pool.base);
            }
            return false;
        }
        orch->rings[r].fanin_pool.init(fanin_entries, dep_pool_capacity, &sm_header->orch_error_code);
    }

    // Initialize TensorMap with per-ring task window sizes
    int32_t task_window_sizes[PTO2_MAX_RING_DEPTH];
    for (int r = 0; r < PTO2_MAX_RING_DEPTH; r++) {
        task_window_sizes[r] = sm_header->rings[r].task_window_size;
    }
    if (!orch->tensor_map.init_default(task_window_sizes)) {
        for (int r = 0; r < PTO2_MAX_RING_DEPTH; r++) {
            free(orch->rings[r].fanin_pool.base);
        }
        return false;
    }
    orch->tensor_map.orch = orch;

    // Initialize scope stack: one flat buffer for task IDs + one array for begin offsets
    uint64_t max_depth = PTO2_MAX_SCOPE_DEPTH;
    int32_t init_cap = PTO2_SCOPE_TASKS_INIT_CAP;
    orch->scope_tasks = reinterpret_cast<PTO2TaskSlotState **>(malloc(init_cap * sizeof(PTO2TaskSlotState *)));
    orch->scope_begins = reinterpret_cast<int32_t *>(malloc(max_depth * sizeof(int32_t)));
    if (!orch->scope_tasks || !orch->scope_begins) {
        free(orch->scope_tasks);
        free(orch->scope_begins);
        for (int r = 0; r < PTO2_MAX_RING_DEPTH; r++) {
            free(orch->rings[r].fanin_pool.base);
        }
        orch->tensor_map.destroy();
        return false;
    }
    orch->scope_tasks_size = 0;
    orch->scope_tasks_capacity = init_cap;
    orch->scope_stack_top = -1;
    orch->scope_stack_capacity = max_depth;

    return true;
}

void pto2_orchestrator_destroy(PTO2OrchestratorState *orch) {
    orch->tensor_map.destroy();

    for (int r = 0; r < PTO2_MAX_RING_DEPTH; r++) {
        free(orch->rings[r].fanin_pool.base);
        orch->rings[r].fanin_pool.base = NULL;
    }

    free(orch->scope_tasks);
    orch->scope_tasks = NULL;
    free(orch->scope_begins);
    orch->scope_begins = NULL;
}

void pto2_orchestrator_set_scheduler(PTO2OrchestratorState *orch, PTO2SchedulerState *scheduler) {
    orch->scheduler = scheduler;
}

// =============================================================================
// Scope Management
// =============================================================================

static void scope_tasks_push(PTO2OrchestratorState *orch, PTO2TaskSlotState *task_slot_state) {
    if (orch->scope_tasks_size >= orch->scope_tasks_capacity) {
        int32_t new_cap = orch->scope_tasks_capacity * 2;
        PTO2TaskSlotState **new_buf =
            reinterpret_cast<PTO2TaskSlotState **>(realloc(orch->scope_tasks, new_cap * sizeof(PTO2TaskSlotState *)));
        assert(new_buf && "Failed to grow scope task buffer");
        orch->scope_tasks = new_buf;
        orch->scope_tasks_capacity = new_cap;
    }
    orch->scope_tasks[orch->scope_tasks_size++] = task_slot_state;
}

void pto2_scope_begin(PTO2OrchestratorState *orch) {
    if (orch->fatal) {
        return;
    }
    assert(orch->scope_stack_top < static_cast<int32_t>(orch->scope_stack_capacity - 1) && "Scope stack overflow");

    ++orch->scope_stack_top;
    orch->scope_begins[orch->scope_stack_top] = orch->scope_tasks_size;
}

void pto2_scope_end(PTO2OrchestratorState *orch) {
    if (orch->fatal) {
        return;
    }
    assert(orch->scope_stack_top >= 0 && "Scope stack underflow");

#if PTO2_ORCH_PROFILING
    uint64_t _se0 = get_sys_cnt_aicpu();
#endif

    int32_t begin = orch->scope_begins[orch->scope_stack_top--];
    int32_t count = orch->scope_tasks_size - begin;

    if (orch->scheduler && count > 0) {
        orch->scheduler->on_scope_end(&orch->scope_tasks[begin], count);
    }

    // Rewind the task buffer — these entries are no longer needed
    orch->scope_tasks_size = begin;

#if PTO2_ORCH_PROFILING
    uint64_t _se1 = get_sys_cnt_aicpu();
    g_orch_scope_end_cycle += (_se1 - _se0);
    // perf_aicpu_record_orch_phase(AicpuPhaseId::ORCH_SCOPE_END, _se0, _se1, g_orch_submit_idx, -1);
    if (count > 0) {
        // 见 g_orch_scope_end_release_count：本 scope 内待 release 的槽位数 = N_scope
        g_orch_scope_end_release_count += static_cast<uint64_t>(count);
    }
#endif
}

// =============================================================================
// Task Submission
// =============================================================================
TaskOutputTensors
pto2_submit_mixed_task(PTO2OrchestratorState *orch, const MixedKernels &mixed_kernels, const Arg &args) {
    CYCLE_COUNT_START();

    TaskOutputTensors result;

    // Orchestration API should short-circuit after fatal, but keep this entry
    // robust as a no-op in case a caller reaches it directly.
    if (orch->fatal) {
        return result;
    }

    // Validate Arg construction (errors recorded by add_input/add_output/etc.)
    if (args.has_error) {
        LOG_ERROR("========================================");
        LOG_ERROR("FATAL: Invalid Arg Detected!");
        LOG_ERROR("========================================");
        LOG_ERROR("Error: %s", args.error_msg ? args.error_msg : "(unknown)");
        LOG_ERROR("  tensor_count: %d, scalar_count: %d", args.tensor_count(), args.scalar_count());
        LOG_ERROR("This is a bug in the orchestration code.");
        LOG_ERROR("========================================");
        pto2_orch_mark_fatal(orch, PTO2_ERROR_INVALID_ARGS);
        return result;
    }

    always_assert(orch->scheduler != nullptr);
    // === Validate submit inputs ===
    uint8_t active_mask = pto2_mixed_kernels_to_active_mask(mixed_kernels);
    always_assert(active_mask != 0 && "MixedKernels must have at least one active slot");

    int16_t block_num = args.launch_spec.block_num();
    always_assert(block_num >= 1 && "block_num must be >= 1");

    // Normalize single-AIV tasks: if only aiv1 is set (no aic, no aiv0), move
    // it to the aiv0 slot.  This guarantees the dispatch path can always use
    // PTO2SubtaskSlot::AIV0 for single-AIV shapes without inspecting active_mask.
    // Mixed tasks (AIC+AIV) keep their original AIV identity so the correct
    // hardware channel (AIV0→AIC vs AIV1→AIC) is used at dispatch time.
    MixedKernels normalized = mixed_kernels;
    bool has_aic = (active_mask & PTO2_SUBTASK_MASK_AIC) != 0;
    bool has_aiv0 = (active_mask & PTO2_SUBTASK_MASK_AIV0) != 0;
    bool has_aiv1 = (active_mask & PTO2_SUBTASK_MASK_AIV1) != 0;
    if (!has_aic && has_aiv1 && !has_aiv0) {
        normalized.aiv0_kernel_id = normalized.aiv1_kernel_id;
        normalized.aiv1_kernel_id = INVALID_KERNEL_ID;
        active_mask = pto2_mixed_kernels_to_active_mask(normalized);
    }

    // Encode require_sync_start into active_mask bit 3 (only meaningful for tasks with block_num > 1)
    if (block_num > 1 && args.launch_spec.require_sync_start()) {
        // Deadlock check: block_num >= total available slots of the required type.
        // For MIX/AIC: limit is total_cluster_count (one AIC per cluster).
        // For AIV:     limit is total_aiv_count.
        PTO2ResourceShape shape = pto2_active_mask_to_shape(active_mask);
        int32_t limit = (shape == PTO2ResourceShape::AIV) ? orch->total_aiv_count : orch->total_cluster_count;
        if (limit > 0 && block_num > limit) {
            pto2_orch_report_fatal(
                orch, PTO2_ERROR_REQUIRE_SYNC_START_INVALID, __FUNCTION__,
                "require_sync_start block_num=%d > limit=%d (deadlock guaranteed)", block_num, limit
            );
            return result;
        }
        active_mask |= PTO2_SUBTASK_FLAG_SYNC_START;
    }
    PTO2OutputLayout layout = pto2_calculate_output_layout(args);
    PTO2PreparedTask prepared;
    if (!pto2_prepare_task(orch, args, layout.total_output_size, active_mask, &prepared)) {
        return result;
    }
    uint8_t ring_id = prepared.task_id.ring();
    PTO2SchedulerState *sched = orch->scheduler;
    PTO2RingFlowControl &fc = orch->sm_header->rings[ring_id].fc;
    PTO2TaskId task_id = prepared.task_id;
    PTO2TaskSlotState &cur_slot_state = *prepared.slot_state;
    PTO2TaskDescriptor &task = *prepared.task;
    PTO2TaskPayload &payload = *prepared.payload;
    result.set_task_id(task_id);

    PTO2FaninBuilder fanin_builder(orch->rings[ring_id].fanin_pool);

    CYCLE_COUNT_LAP_RECORD(g_orch_alloc_cycle, AicpuPhaseId::ORCH_ALLOC, task_id.raw);

#if PTO2_PROFILING
    if (layout.total_output_size > 0) {
        orch->buffers_allocated++;
        orch->bytes_allocated += layout.total_output_size;
    }
#endif

    // === STEP 2: Sync TensorMap validity and optional cleanup ===
    // Read current last_task_alive from shared memory for this ring
    int32_t sm_last_task_alive = fc.last_task_alive.load(std::memory_order_acquire);

    orch->tensor_map.sync_tensormap(task_id, sm_last_task_alive);

    CYCLE_COUNT_LAP_RECORD(g_orch_sync_cycle, AicpuPhaseId::ORCH_SYNC, task_id.raw);

    // === STEP 3: Lookup inputs + materialize runtime-created outputs ===
    for (int i = 0; i < args.tensor_count(); i++) {
        TensorArgType ptype = args.tag(i);
        if (ptype == TensorArgType::OUTPUT) {
            // Runtime-created OUTPUT tensors are not looked up in the TensorMap since they have no dependencies.
            continue;
        }

        const Tensor *tensor = args.tensor(i).ptr;

        // Step A: creator retention — all existing tensors extend their creator lifetime.
        PTO2TaskId owner = tensor->owner_task_id;
        if (owner.is_valid()) {
            PTO2TaskSlotState *prod_state =
                &orch->sm_header->rings[owner.ring()].get_slot_state_by_task_id(owner.local());
            if (!pto2_append_fanin_or_fail(orch, prod_state, &fanin_builder, ring_id)) {
                return result;
            }
        }

        // Step B: only INPUT/INOUT need modifier dependency lookup.
        if (ptype != TensorArgType::INPUT && ptype != TensorArgType::INOUT) {
            continue;
        }
        if (tensor->manual_dep) {
            continue;
        }
#if PTO2_ORCH_PROFILING
        g_orch_tensor_input_read_count++;  // 见 g_orch_tensor_input_read_count
#endif

        PTO2LookupResult lookup_result;
        orch->tensor_map.lookup(*tensor, lookup_result);

        for (int r = 0; r < lookup_result.count; r++) {
            PTO2TensorMapEntry &entry = *lookup_result.entries[r].entry;
            auto overlap_status = lookup_result.entries[r].overlap_status;
            auto prod_ring = entry.producer_task_id.ring();
            auto prod_local = entry.producer_task_id.local();
            PTO2TaskSlotState *prod_state = &orch->sm_header->rings[prod_ring].get_slot_state_by_task_id(prod_local);
            if (!pto2_append_fanin_or_fail(orch, prod_state, &fanin_builder, ring_id)) {
                return result;
            }
            if (ptype == TensorArgType::INOUT && overlap_status == OverlapStatus::COVERED) {
                orch->tensor_map.remove_entry(entry);
            }
        }
    }

    CYCLE_COUNT_LAP_RECORD(g_orch_lookup_cycle, AicpuPhaseId::ORCH_LOOKUP, task_id.raw);

    // === STEP 4: Register outputs/inouts in TensorMap (must be separate from lookup) ===
    {
        for (int i = 0; i < args.tensor_count(); i++) {
            TensorArgType ptype = args.tag(i);
            if (ptype == TensorArgType::INOUT || ptype == TensorArgType::OUTPUT_EXISTING) {
                if (!args.tensor(i).ptr->manual_dep) {
                    orch->tensor_map.insert(*args.tensor(i).ptr, task_id);
                }
            }
        }
    }

    CYCLE_COUNT_LAP_RECORD(g_orch_insert_cycle, AicpuPhaseId::ORCH_INSERT, task_id.raw);

    // === STEP 5: Batch-write to GM (single cache line burst) + Record fanin metadata ===
    // Deferred from allocation phase to avoid scattered GM writes that get
    // evicted by TensorMap lookup/insert cache pressure.
    __builtin_prefetch(&task, 1, 1);
    task.task_id = task_id;
    task.kernel_id[static_cast<int>(PTO2SubtaskSlot::AIC)] = normalized.aic_kernel_id;
    task.kernel_id[static_cast<int>(PTO2SubtaskSlot::AIV0)] = normalized.aiv0_kernel_id;
    task.kernel_id[static_cast<int>(PTO2SubtaskSlot::AIV1)] = normalized.aiv1_kernel_id;
    task.packed_buffer_base = prepared.alloc_result.packed_base;
    task.packed_buffer_end = prepared.alloc_result.packed_end;

    // Increment fanout_count on each producer (no lock — only orch writes this field).
    // Prevents premature CONSUMED: scope_end's release_producer checks fanout_refcount == fanout_count.
    pto2_for_each_fanin_storage(
        fanin_builder.inline_slots, fanin_builder.count, fanin_builder.spill_start, fanin_builder.spill_pool,
        [](PTO2TaskSlotState *producer) {
            producer->fanout_count++;
#if PTO2_ORCH_PROFILING
            g_orch_fanout_increment_count++;  // 见 g_orch_fanout_increment_count（每 producer 一次）
#endif
        }
    );

    int32_t inline_count = std::min(fanin_builder.count, PTO2_FANIN_INLINE_CAP);
    // Store fanin metadata in payload for scheduler to iterate
    payload.fanin_actual_count = fanin_builder.count;
    payload.fanin_spill_start = fanin_builder.spill_start;
    payload.fanin_spill_pool = &fanin_builder.spill_pool;
    for (int i = 0; i < inline_count; i++) {
        payload.fanin_inline_slot_states[i] = fanin_builder.inline_slots[i];
    }

    payload.init(args, result, prepared.alloc_result, layout);
#if PTO2_ORCH_PROFILING
    g_orch_payload_init_count++;     // 见 g_orch_payload_init_count
    g_orch_descriptor_write_count++; // 见 g_orch_descriptor_write_count
    // Write owner_task_id into payload tensors for every output-side slot (OUTPUT / INOUT /
    // OUTPUT_EXISTING) so creator-only dependency tracking matches CSV ① Tensor「写 N_out」.
    for (int i = 0; i < args.tensor_count(); i++) {
        const TensorArgType ptype = args.tag(i);
        if (ptype == TensorArgType::OUTPUT || ptype == TensorArgType::INOUT ||
            ptype == TensorArgType::OUTPUT_EXISTING) {
            g_orch_tensor_output_write_count++;  // 见 g_orch_tensor_output_write_count
        }
    }
#endif

    CYCLE_COUNT_LAP_RECORD(g_orch_args_cycle, AicpuPhaseId::ORCH_PARAMS, task_id.raw);
#if PTO2_ORCH_PROFILING
    // g_orch_args_atomic_count：当前实现将 consumer 侧 fanout_lock/fanout_count 初始化记 2 次原子写
    g_orch_args_atomic_count += 2;
#endif

    // === STEP 6: push to wiring queue ===
    // Deferred wiring: orchestrator only stores dependency metadata and increments
    // fanout_count. The actual fanout_head wiring (lock + dep_pool + early_finished)
    // is handled asynchronously by scheduler thread 0 via the wiring queue.
    // Push to global wiring queue — scheduler sets fanin_count, wires fanout, checks readiness
    while (!sched->wiring.queue.push(&cur_slot_state)) {
        SPIN_WAIT_HINT();
    }
    // module-struct-access.csv 行 1–9：本任务 P/S、N_in/out、tensor/scalar_count 等（写入 orch->csv_glossary）
    pto2_csv_glossary_record_mixed_submit(orch, task, cur_slot_state, payload, args, fanin_builder.count);
#if PTO2_ORCH_PROFILING
    // 预留：submit 热路径不向 g_orch_fanin_atomic_count 累加（保持与历史行为一致）
    g_orch_fanin_atomic_count += 0;
#endif

    CYCLE_COUNT_LAP_RECORD(g_orch_fanin_cycle, AicpuPhaseId::ORCH_FANIN, task_id.raw);

#if PTO2_PROFILING
    orch->tasks_submitted++;
#if PTO2_ORCH_PROFILING
    g_orch_submit_count++;  // 见 g_orch_submit_count：pto2_submit_mixed_task 成功路径
#endif
    g_orch_submit_idx++;
#endif
    return result;
}

TaskOutputTensors pto2_alloc_tensors(PTO2OrchestratorState *orch, const Arg &args) {
    // Orchestration API should short-circuit after fatal, but keep this entry
    // robust as a no-op in case a caller reaches it directly.
    if (orch->fatal) {
        return TaskOutputTensors{};
    }

    if (args.tensor_count() <= 0) {
        pto2_orch_report_fatal(
            orch, PTO2_ERROR_INVALID_ARGS, __FUNCTION__, "alloc_tensors requires at least one TensorCreateInfo"
        );
        return TaskOutputTensors{};
    }
    if (args.scalar_count() != 0) {
        pto2_orch_report_fatal(
            orch, PTO2_ERROR_INVALID_ARGS, __FUNCTION__, "alloc_tensors only accepts output TensorCreateInfo args"
        );
        return TaskOutputTensors{};
    }
    for (int32_t i = 0; i < args.tensor_count(); i++) {
        if (args.tag(i) != TensorArgType::OUTPUT) {
            pto2_orch_report_fatal(
                orch, PTO2_ERROR_INVALID_ARGS, __FUNCTION__, "alloc_tensors only accepts output TensorCreateInfo args"
            );
            return TaskOutputTensors{};
        }
    }

    CYCLE_COUNT_START();

    if (args.has_error) {
        pto2_orch_report_fatal(
            orch, PTO2_ERROR_INVALID_ARGS, __FUNCTION__, "%s",
            args.error_msg ? args.error_msg : "alloc_tensors failed to construct output-only Arg"
        );
        return TaskOutputTensors{};
    }

    PTO2OutputLayout layout = pto2_calculate_output_layout(args);
    PTO2PreparedTask prepared;
    if (!pto2_prepare_task(orch, args, layout.total_output_size, 0, &prepared)) {
        return TaskOutputTensors{};
    }

    PTO2TaskDescriptor &task = *prepared.task;
    PTO2TaskPayload &payload = *prepared.payload;

    CYCLE_COUNT_LAP_RECORD(g_orch_alloc_cycle, AicpuPhaseId::ORCH_ALLOC, prepared.task_id.raw);

#if PTO2_PROFILING
    if (layout.total_output_size > 0) {
        orch->buffers_allocated++;
        orch->bytes_allocated += layout.total_output_size;
    }
#endif

    task.task_id = prepared.task_id;
    task.kernel_id[static_cast<int>(PTO2SubtaskSlot::AIC)] = INVALID_KERNEL_ID;
    task.kernel_id[static_cast<int>(PTO2SubtaskSlot::AIV0)] = INVALID_KERNEL_ID;
    task.kernel_id[static_cast<int>(PTO2SubtaskSlot::AIV1)] = INVALID_KERNEL_ID;
    task.packed_buffer_base = prepared.alloc_result.packed_base;
    task.packed_buffer_end = prepared.alloc_result.packed_end;

    TaskOutputTensors outputs;
    outputs.set_task_id(prepared.task_id);
    payload.init(args, outputs, prepared.alloc_result, layout);
    payload.fanin_actual_count = 0;
    payload.fanin_spill_start = 0;
    payload.fanin_spill_pool = &orch->rings[prepared.task_id.ring()].fanin_pool;
#if PTO2_ORCH_PROFILING
    for (int32_t i = 0; i < args.tensor_count(); i++) {
        g_orch_tensor_output_write_count++;  // alloc 路径仅 OUTPUT，与 N_out 一致
    }
#endif

    CYCLE_COUNT_LAP_RECORD(g_orch_args_cycle, AicpuPhaseId::ORCH_PARAMS, prepared.task_id.raw);

    if (prepared.slot_state != nullptr) {
        // Hidden alloc tasks complete inline in the orchestrator before any
        // consumer can exist, so they have no fanout to notify and no worker
        // subtasks to retire. Running the full on_mixed_task_complete path
        // would only pay unnecessary fanout_lock / traversal overhead here.
        // The generic slot initialization done in pto2_prepare_task() is still
        // required so scope_end can release the producer-side reference and
        // drive the slot to CONSUMED, but worker dispatch fields are never
        // observed for hidden alloc tasks.
        prepared.slot_state->task_state.store(PTO2_TASK_COMPLETED, std::memory_order_release);
    }
    orch->inline_completed_tasks++;

    CYCLE_COUNT_LAP_RECORD(g_orch_fanin_cycle, AicpuPhaseId::ORCH_FANIN, prepared.task_id.raw);

    pto2_csv_glossary_record_alloc_hidden(
        orch, prepared.slot_state, payload, args, prepared.task_id.ring()
    );

#if PTO2_PROFILING
    orch->tasks_submitted++;
#if PTO2_ORCH_PROFILING
    g_orch_submit_count++;  // 见 g_orch_submit_count：pto2_alloc_tensors / hidden alloc 成功路径
#endif
    g_orch_submit_idx++;
#endif

    return outputs;
}

// =============================================================================
// Flow Control
// =============================================================================

void pto2_orchestrator_done(PTO2OrchestratorState *orch) {
    for (int r = 0; r < PTO2_MAX_RING_DEPTH; r++) {
        int32_t total_tasks = orch->rings[r].task_allocator.active_count();
        if (total_tasks > 0) {
            LOG_INFO("=== [Orchestrator] ring %d: total_tasks=%d ===", r, total_tasks);
        }
        auto &fanin_pool = orch->rings[r].fanin_pool;
        if (fanin_pool.top > 1) {
            LOG_INFO(
                "=== [FaninPool %d] top=%d tail=%d used=%d high_water=%d capacity=%d ===", r, fanin_pool.top,
                fanin_pool.tail, fanin_pool.top - fanin_pool.tail, fanin_pool.high_water, fanin_pool.capacity
            );
        }
    }
    orch->sm_header->orchestrator_done.store(1, std::memory_order_release);
#if !PTO2_ORCH_PROFILING && PTO2_PROFILING
    g_orch_submit_idx = 0;
#endif
}

// =============================================================================
// Debug Utilities
// =============================================================================

void pto2_orchestrator_print_stats(PTO2OrchestratorState *orch) {
    LOG_INFO("=== Orchestrator Statistics ===");
#if PTO2_PROFILING
    LOG_INFO("Tasks submitted:     %" PRId64, orch->tasks_submitted);
    LOG_INFO("Buffers allocated:   %" PRId64, orch->buffers_allocated);
    LOG_INFO("Bytes allocated:     %" PRId64, orch->bytes_allocated);
#endif
    LOG_INFO("Current scope depth: %d", orch->scope_stack_top + 1);
    for (int r = 0; r < PTO2_MAX_RING_DEPTH; r++) {
        int32_t active = orch->rings[r].task_allocator.active_count();
        if (active > 0) {
            LOG_INFO("Ring %d task active:  %d", r, active);
            LOG_INFO(
                "Ring %d heap used:    %" PRIu64 " / %" PRIu64, r, orch->rings[r].task_allocator.heap_top(),
                orch->rings[r].task_allocator.heap_capacity()
            );
            LOG_INFO(
                "Ring %d fanin pool:   %d / %d", r, orch->rings[r].fanin_pool.used(), orch->rings[r].fanin_pool.capacity
            );
        }
    }
    LOG_INFO("TensorMap valid:     %d", orch->tensor_map.valid_count());
    LOG_INFO("===============================");
}

void pto2_orchestrator_print_scope_stack(PTO2OrchestratorState *orch) {
    LOG_INFO("=== Scope Stack ===");
    LOG_INFO("Depth: %d", orch->scope_stack_top + 1);

    for (int i = 0; i <= orch->scope_stack_top; i++) {
        int32_t begin = orch->scope_begins[i];
        int32_t end = (i < orch->scope_stack_top) ? orch->scope_begins[i + 1] : orch->scope_tasks_size;
        LOG_INFO("  [%d] tasks_owned = %d", i, end - begin);
    }

    LOG_INFO("==================");
}

#if PTO2_ORCH_PROFILING
PTO2OrchProfilingData pto2_orchestrator_get_profiling(PTO2OrchestratorState *orch) {
    PTO2OrchProfilingData d{};
    d.sync_cycle = g_orch_sync_cycle;
    d.alloc_cycle = g_orch_alloc_cycle;
    d.args_cycle = g_orch_args_cycle;
    d.lookup_cycle = g_orch_lookup_cycle;
    d.insert_cycle = g_orch_insert_cycle;
    d.fanin_cycle = g_orch_fanin_cycle;
    d.scope_end_cycle = g_orch_scope_end_cycle;
    d.submit_count = g_orch_submit_count;
    d.alloc_wait_cycle = g_orch_alloc_wait_cycle;
    d.fanin_wait_cycle = g_orch_fanin_wait_cycle;
    d.alloc_atomic_count = g_orch_alloc_atomic_count;
    d.args_atomic_count = g_orch_args_atomic_count;
    d.fanin_atomic_count = g_orch_fanin_atomic_count;
    d.finalize_atomic_count = g_orch_finalize_atomic_count;
    d.scope_end_atomic_count = g_orch_scope_end_atomic_count;

    const uint64_t p_fan = g_orch_fanout_increment_count;
    const uint64_t n_scope = g_orch_scope_end_release_count;
    const uint64_t slot_init = g_orch_slot_state_init_count;
    const uint64_t pay_init = g_orch_payload_init_count;
    const uint64_t desc_w = g_orch_descriptor_write_count;
    const uint64_t tin = g_orch_tensor_input_read_count;
    const uint64_t tout = g_orch_tensor_output_write_count;
    const uint64_t fc_alloc = g_orch_ring_fc_alloc_count;
    const uint64_t fc_reads = g_orch_ring_fc_last_alive_acquire_reads;
    const uint64_t wq_push = g_orch_m1_wiring_queue_push_done;

    // --- CSV ① PTO2TaskSlotState：读≈P+N_scope，写≈1+P+N_scope；atomic 用 scope_end 链上原子累计近似 ---
    d.csv_m1_pto2_task_slot_state.read_events = p_fan + n_scope;
    d.csv_m1_pto2_task_slot_state.write_events = slot_init + p_fan + n_scope;
    d.csv_m1_pto2_task_slot_state.atomic_ops = g_orch_scope_end_atomic_count;
    d.csv_m1_pto2_task_slot_state.atomic_read_ops = g_orch_scope_end_atomic_read_count;
    d.csv_m1_pto2_task_slot_state.atomic_write_ops = g_orch_scope_end_atomic_write_count;
    d.csv_m1_pto2_task_slot_state.lock_ops = 0;
    d.csv_m1_pto2_task_slot_state.cas_ops = 0;

    // --- CSV ① PTO2TaskPayload：读 0、写 1（每任务一次 init）---
    d.csv_m1_pto2_task_payload.read_events = 0;
    d.csv_m1_pto2_task_payload.write_events = pay_init;
    d.csv_m1_pto2_task_payload.atomic_ops = 0;
    d.csv_m1_pto2_task_payload.atomic_read_ops = 0;
    d.csv_m1_pto2_task_payload.atomic_write_ops = 0;
    d.csv_m1_pto2_task_payload.lock_ops = 0;
    d.csv_m1_pto2_task_payload.cas_ops = 0;

    // --- CSV ① PTO2TaskDescriptor：写 1 / 读 0 ---
    d.csv_m1_pto2_task_descriptor.read_events = 0;
    d.csv_m1_pto2_task_descriptor.write_events = desc_w;
    d.csv_m1_pto2_task_descriptor.atomic_ops = 0;
    d.csv_m1_pto2_task_descriptor.atomic_read_ops = 0;
    d.csv_m1_pto2_task_descriptor.atomic_write_ops = 0;
    d.csv_m1_pto2_task_descriptor.lock_ops = 0;
    d.csv_m1_pto2_task_descriptor.cas_ops = 0;

    // --- CSV ① Tensor：读 N_in、写 N_out ---
    d.csv_m1_tensor.read_events = tin;
    d.csv_m1_tensor.write_events = tout;
    d.csv_m1_tensor.atomic_ops = 0;
    d.csv_m1_tensor.atomic_read_ops = 0;
    d.csv_m1_tensor.atomic_write_ops = 0;
    d.csv_m1_tensor.lock_ops = 0;
    d.csv_m1_tensor.cas_ops = 0;

    // --- CSV ① PTO2ReadyQueue（wiring_queue）：写=push 次数；atomic≈3×push；CAS≈push（Vyukov 单生产者近似）---
    d.csv_m1_pto2_ready_queue.read_events = 0;
    d.csv_m1_pto2_ready_queue.write_events = wq_push;
    d.csv_m1_pto2_ready_queue.atomic_ops = wq_push * 3;
    d.csv_m1_pto2_ready_queue.atomic_read_ops = wq_push * 2;
    d.csv_m1_pto2_ready_queue.atomic_write_ops = wq_push;
    d.csv_m1_pto2_ready_queue.lock_ops = 0;
    d.csv_m1_pto2_ready_queue.cas_ops = wq_push;

    // --- CSV ①③ PTO2RingFlowControl：读=last_task_alive acquire 总次数；写=成功 alloc 次数；atomic 近似 2×alloc + 额外 spin 读 ---
    d.csv_m1_pto2_ring_flow_control.read_events = fc_reads;
    d.csv_m1_pto2_ring_flow_control.write_events = fc_alloc;
    d.csv_m1_pto2_ring_flow_control.atomic_ops = fc_alloc * 2 + (fc_reads > fc_alloc ? fc_reads - fc_alloc : 0);
    d.csv_m1_pto2_ring_flow_control.atomic_read_ops = (fc_reads > fc_alloc ? fc_reads - fc_alloc : 0);
    d.csv_m1_pto2_ring_flow_control.atomic_write_ops = fc_alloc * 2;
    d.csv_m1_pto2_ring_flow_control.lock_ops = 0;
    d.csv_m1_pto2_ring_flow_control.cas_ops = 0;

    if (orch != nullptr) {
        d.csv_glossary = orch->csv_glossary;
        orch->csv_glossary.bucket_count = 0;
    }

    // 清零：以下变量已全部汇入 d.csv_m1_* / 周期字段，下一窗口重新累计
    g_orch_sync_cycle = g_orch_alloc_cycle = g_orch_args_cycle = 0;
    g_orch_lookup_cycle = g_orch_insert_cycle = 0;
    g_orch_fanin_cycle = g_orch_scope_end_cycle = 0;
    g_orch_submit_count = 0;
    g_orch_submit_idx = 0;
    g_orch_alloc_wait_cycle = 0;
    g_orch_fanin_wait_cycle = 0;
    g_orch_alloc_atomic_count = 0;
    g_orch_args_atomic_count = 0;
    g_orch_fanin_atomic_count = 0;
    g_orch_finalize_atomic_count = 0;
    g_orch_scope_end_atomic_count = 0;
    g_orch_scope_end_atomic_read_count = 0;
    g_orch_scope_end_atomic_write_count = 0;
    g_orch_fanout_increment_count = 0;
    g_orch_scope_end_release_count = 0;
    g_orch_slot_state_init_count = 0;
    g_orch_payload_init_count = 0;
    g_orch_descriptor_write_count = 0;
    g_orch_tensor_input_read_count = 0;
    g_orch_tensor_output_write_count = 0;
    g_orch_ring_fc_alloc_count = 0;
    g_orch_ring_fc_last_alive_acquire_reads = 0;
    g_orch_m1_wiring_queue_push_done = 0;
    return d;
}
#endif
