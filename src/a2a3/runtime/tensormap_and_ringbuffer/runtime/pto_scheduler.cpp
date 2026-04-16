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
 * PTO Runtime2 - Scheduler Implementation
 *
 * Implements scheduler state management, ready queues, and task lifecycle.
 *
 * Based on: docs/RUNTIME_LOGIC.md
 */

#include "pto_scheduler.h"
#include <inttypes.h>
#include <stdlib.h>
#include "common/unified_log.h"

// =============================================================================
// Scheduler Profiling Counters
// =============================================================================

#if PTO2_SCHED_PROFILING
#include "common/platform_config.h"

// ---------- 周期 / 等待 / 子阶段原子（PTO2SchedProfilingData 非 CSV 五列字段的数据源）----------
uint64_t g_sched_lock_cycle[PLATFORM_MAX_AICPU_THREADS] = {};
uint64_t g_sched_fanout_cycle[PLATFORM_MAX_AICPU_THREADS] = {};
uint64_t g_sched_fanin_cycle[PLATFORM_MAX_AICPU_THREADS] = {};
uint64_t g_sched_self_consumed_cycle[PLATFORM_MAX_AICPU_THREADS] = {};
uint64_t g_sched_lock_wait_cycle[PLATFORM_MAX_AICPU_THREADS] = {};
uint64_t g_sched_push_wait_cycle[PLATFORM_MAX_AICPU_THREADS] = {};
uint64_t g_sched_pop_wait_cycle[PLATFORM_MAX_AICPU_THREADS] = {};
uint64_t g_sched_lock_atomic_count[PLATFORM_MAX_AICPU_THREADS] = {};
uint64_t g_sched_lock_atomic_read_count[PLATFORM_MAX_AICPU_THREADS] = {};
uint64_t g_sched_lock_atomic_write_count[PLATFORM_MAX_AICPU_THREADS] = {};
uint64_t g_sched_fanout_atomic_count[PLATFORM_MAX_AICPU_THREADS] = {};
uint64_t g_sched_fanout_atomic_read_count[PLATFORM_MAX_AICPU_THREADS] = {};
uint64_t g_sched_fanout_atomic_write_count[PLATFORM_MAX_AICPU_THREADS] = {};
uint64_t g_sched_fanin_atomic_count[PLATFORM_MAX_AICPU_THREADS] = {};
uint64_t g_sched_fanin_atomic_read_count[PLATFORM_MAX_AICPU_THREADS] = {};
uint64_t g_sched_fanin_atomic_write_count[PLATFORM_MAX_AICPU_THREADS] = {};
uint64_t g_sched_self_atomic_count[PLATFORM_MAX_AICPU_THREADS] = {};
uint64_t g_sched_self_atomic_read_count[PLATFORM_MAX_AICPU_THREADS] = {};
uint64_t g_sched_self_atomic_write_count[PLATFORM_MAX_AICPU_THREADS] = {};
uint64_t g_sched_pop_atomic_count[PLATFORM_MAX_AICPU_THREADS] = {};
uint64_t g_sched_pop_atomic_read_count[PLATFORM_MAX_AICPU_THREADS] = {};
uint64_t g_sched_pop_atomic_write_count[PLATFORM_MAX_AICPU_THREADS] = {};
/** 每线程 on_task_release 完成路径次数累计 → PTO2SchedProfilingData.complete_count */
uint64_t g_sched_complete_count[PLATFORM_MAX_AICPU_THREADS] = {};
// ---------- 原始事件：pto2_scheduler_get_profiling() 中映射为 CSV ②~⑦ 的 read/write/atomic/lock/cas ----------
/** CSV ② wire_task 调用次数 wt（每 wired 任务 1） */
uint64_t g_sched_wire_task_count[PLATFORM_MAX_AICPU_THREADS] = {};
/** CSV ② ∑P：所有 wire_task 上 fanin 边总数 wf */
uint64_t g_sched_wire_fanin_count[PLATFORM_MAX_AICPU_THREADS] = {};
/** CSV ② DepListEntry 分配次数（每 live producer 1） */
uint64_t g_sched_wire_dep_entry_count[PLATFORM_MAX_AICPU_THREADS] = {};
/** CSV ② ReadyQueue(wiring push) 底层原子读/写原始累计（thread 0） */
uint64_t g_sched_m2_ready_queue_atomic_read_raw[PLATFORM_MAX_AICPU_THREADS] = {};
uint64_t g_sched_m2_ready_queue_atomic_write_raw[PLATFORM_MAX_AICPU_THREADS] = {};
uint64_t g_sched_m2_ready_queue_spin_retry_raw[PLATFORM_MAX_AICPU_THREADS] = {};
/** CSV ④⑤：子任务 dispatch 次数之和 ∑S（与 PTO2DispatchPayload 写次数线性相关） */
uint64_t g_sched_dispatch_subtask_count[PLATFORM_MAX_AICPU_THREADS] = {};
/** CSV ④：从就绪队列实际 pop 到的任务数（每任务读 Payload 元数据+Descriptor 各 1 次的数据源） */
uint64_t g_sched_dispatch_task_count[PLATFORM_MAX_AICPU_THREADS] = {};
/** CSV ⑤：completed_subtasks.fetch_add 调用次数 = ∑S */
uint64_t g_sched_subtask_complete_count[PLATFORM_MAX_AICPU_THREADS] = {};
/** CSV ⑥：fanout 边遍历条数 = ∑C（与 dep 链表遍历同阶） */
uint64_t g_sched_fanout_edge_count[PLATFORM_MAX_AICPU_THREADS] = {};
/** CSV ⑥：因 fanin 就绪而入队 consumer 次数 */
uint64_t g_sched_tasks_enqueued_count[PLATFORM_MAX_AICPU_THREADS] = {};
/** CSV ⑥：沿 DepListEntry 读 next 的遍历时次数（此处与 fanout_edges 同值） */
uint64_t g_sched_dep_list_traverse_count[PLATFORM_MAX_AICPU_THREADS] = {};
/** CSV ⑦：on_task_release() 调用次数 */
uint64_t g_sched_task_release_count[PLATFORM_MAX_AICPU_THREADS] = {};
/** CSV ⑦：每次 release 上 fanout_refcount.fetch_add 总次数 = 各任务 ∑P */
uint64_t g_sched_fanin_release_count[PLATFORM_MAX_AICPU_THREADS] = {};
/** CSV ⑦：PTO2FaninSpillEntry 读次数（spill 槽条数累计） */
uint64_t g_sched_fanin_spill_read_count[PLATFORM_MAX_AICPU_THREADS] = {};
/** CSV ⑦：advance_ring_pointers() 成功调用次数 */
uint64_t g_sched_advance_count[PLATFORM_MAX_AICPU_THREADS] = {};
/** CSV ④：调度循环内对全局就绪队列 pop_batch/get_ready 的调用次数 */
uint64_t g_sched_ready_queue_pop_count[PLATFORM_MAX_AICPU_THREADS] = {};
/** CSV ④：就绪队列 pop 命中轮次（count>0 的 pop 尝试次数） */
uint64_t g_sched_ready_queue_pop_hit_round_count[PLATFORM_MAX_AICPU_THREADS] = {};
/** CSV ④：就绪队列 pop 空转轮次（count==0 的 pop 尝试次数） */
uint64_t g_sched_ready_queue_pop_miss_round_count[PLATFORM_MAX_AICPU_THREADS] = {};
/** CSV ④：命中轮次对应的 pop 原子操作累计（按每次 pop 调用 atomic 增量归因） */
uint64_t g_sched_ready_queue_pop_hit_atomic_count[PLATFORM_MAX_AICPU_THREADS] = {};
/** CSV ④：空转轮次对应的 pop 原子操作累计（按每次 pop 调用 atomic 增量归因） */
uint64_t g_sched_ready_queue_pop_miss_atomic_count[PLATFORM_MAX_AICPU_THREADS] = {};
uint64_t g_sched_ready_queue_pop_hit_atomic_read_count[PLATFORM_MAX_AICPU_THREADS] = {};
uint64_t g_sched_ready_queue_pop_hit_atomic_write_count[PLATFORM_MAX_AICPU_THREADS] = {};
uint64_t g_sched_ready_queue_pop_miss_atomic_read_count[PLATFORM_MAX_AICPU_THREADS] = {};
uint64_t g_sched_ready_queue_pop_miss_atomic_write_count[PLATFORM_MAX_AICPU_THREADS] = {};
uint64_t g_sched_m4_ready_queue_pop_retry_count[PLATFORM_MAX_AICPU_THREADS] = {};
uint64_t g_sched_m4_ready_queue_pop_empty_poll_count[PLATFORM_MAX_AICPU_THREADS] = {};
/** CSV ⑦ advance_lock：compare_exchange_strong 尝试次数（成功+失败均计） */
uint64_t g_sched_advance_lock_cas_count[PLATFORM_MAX_AICPU_THREADS] = {};
uint64_t g_sched_m6_ready_queue_atomic_read_count[PLATFORM_MAX_AICPU_THREADS] = {};
uint64_t g_sched_m6_ready_queue_atomic_write_count[PLATFORM_MAX_AICPU_THREADS] = {};
/** CSV ④ Payload CL3~34：各 dispatch 上 tensors[] 下标遍历次数之和（每 tensor 索引一次 +1） */
uint64_t g_sched_m4_payload_tensor_lane_reads[PLATFORM_MAX_AICPU_THREADS] = {};
/** CSV ④ Payload CL35~50：各 dispatch 上 scalars[] 遍历次数之和 */
uint64_t g_sched_m4_payload_scalar_lane_reads[PLATFORM_MAX_AICPU_THREADS] = {};

PTO2SchedProfilingData pto2_scheduler_get_profiling(int thread_idx) {
    PTO2SchedProfilingData d;
    d.lock_cycle = std::exchange(g_sched_lock_cycle[thread_idx], 0);
    d.fanout_cycle = std::exchange(g_sched_fanout_cycle[thread_idx], 0);
    d.fanin_cycle = std::exchange(g_sched_fanin_cycle[thread_idx], 0);
    d.self_consumed_cycle = std::exchange(g_sched_self_consumed_cycle[thread_idx], 0);
    d.lock_wait_cycle = std::exchange(g_sched_lock_wait_cycle[thread_idx], 0);
    d.push_wait_cycle = std::exchange(g_sched_push_wait_cycle[thread_idx], 0);
    d.pop_wait_cycle = std::exchange(g_sched_pop_wait_cycle[thread_idx], 0);
    d.lock_atomic_count = std::exchange(g_sched_lock_atomic_count[thread_idx], 0);
    const uint64_t lock_ar = std::exchange(g_sched_lock_atomic_read_count[thread_idx], 0);
    const uint64_t lock_aw = std::exchange(g_sched_lock_atomic_write_count[thread_idx], 0);
    d.fanout_atomic_count = std::exchange(g_sched_fanout_atomic_count[thread_idx], 0);
    const uint64_t fanout_ar = std::exchange(g_sched_fanout_atomic_read_count[thread_idx], 0);
    const uint64_t fanout_aw = std::exchange(g_sched_fanout_atomic_write_count[thread_idx], 0);
    d.fanin_atomic_count = std::exchange(g_sched_fanin_atomic_count[thread_idx], 0);
    const uint64_t fanin_ar = std::exchange(g_sched_fanin_atomic_read_count[thread_idx], 0);
    const uint64_t fanin_aw = std::exchange(g_sched_fanin_atomic_write_count[thread_idx], 0);
    d.self_atomic_count = std::exchange(g_sched_self_atomic_count[thread_idx], 0);
    const uint64_t self_ar = std::exchange(g_sched_self_atomic_read_count[thread_idx], 0);
    const uint64_t self_aw = std::exchange(g_sched_self_atomic_write_count[thread_idx], 0);
    d.pop_atomic_count = std::exchange(g_sched_pop_atomic_count[thread_idx], 0);
    const uint64_t pop_ar = std::exchange(g_sched_pop_atomic_read_count[thread_idx], 0);
    const uint64_t pop_aw = std::exchange(g_sched_pop_atomic_write_count[thread_idx], 0);
    d.complete_count = std::exchange(g_sched_complete_count[thread_idx], 0);

    // 本窗口快照（exchange 清零全局数组，变量名与 get_profiling 内公式一致）
    const uint64_t wt = std::exchange(g_sched_wire_task_count[thread_idx], 0);
    const uint64_t wf = std::exchange(g_sched_wire_fanin_count[thread_idx], 0);
    const uint64_t wdep = std::exchange(g_sched_wire_dep_entry_count[thread_idx], 0);
    const uint64_t m2_rq_ar_raw = std::exchange(g_sched_m2_ready_queue_atomic_read_raw[thread_idx], 0);
    const uint64_t m2_rq_aw_raw = std::exchange(g_sched_m2_ready_queue_atomic_write_raw[thread_idx], 0);
    const uint64_t m2_rq_spin_retry = std::exchange(g_sched_m2_ready_queue_spin_retry_raw[thread_idx], 0);
    const uint64_t dsub = std::exchange(g_sched_dispatch_subtask_count[thread_idx], 0);
    const uint64_t dtask = std::exchange(g_sched_dispatch_task_count[thread_idx], 0);
    const uint64_t scc = std::exchange(g_sched_subtask_complete_count[thread_idx], 0);
    const uint64_t fe = std::exchange(g_sched_fanout_edge_count[thread_idx], 0);
    const uint64_t teq = std::exchange(g_sched_tasks_enqueued_count[thread_idx], 0);
    const uint64_t dlt = std::exchange(g_sched_dep_list_traverse_count[thread_idx], 0);
    const uint64_t trel = std::exchange(g_sched_task_release_count[thread_idx], 0);
    const uint64_t finrel = std::exchange(g_sched_fanin_release_count[thread_idx], 0);
    const uint64_t spill = std::exchange(g_sched_fanin_spill_read_count[thread_idx], 0);
    const uint64_t adv = std::exchange(g_sched_advance_count[thread_idx], 0);
    std::exchange(g_sched_ready_queue_pop_count[thread_idx], 0);
    const uint64_t rqh = std::exchange(g_sched_ready_queue_pop_hit_round_count[thread_idx], 0);
    const uint64_t rqm = std::exchange(g_sched_ready_queue_pop_miss_round_count[thread_idx], 0);
    const uint64_t rqha = std::exchange(g_sched_ready_queue_pop_hit_atomic_count[thread_idx], 0);
    const uint64_t rqma = std::exchange(g_sched_ready_queue_pop_miss_atomic_count[thread_idx], 0);
    const uint64_t rqhar = std::exchange(g_sched_ready_queue_pop_hit_atomic_read_count[thread_idx], 0);
    const uint64_t rqhaw = std::exchange(g_sched_ready_queue_pop_hit_atomic_write_count[thread_idx], 0);
    const uint64_t rqmar = std::exchange(g_sched_ready_queue_pop_miss_atomic_read_count[thread_idx], 0);
    const uint64_t rqmaw = std::exchange(g_sched_ready_queue_pop_miss_atomic_write_count[thread_idx], 0);
    const uint64_t m4_rq_spin_retry = std::exchange(g_sched_m4_ready_queue_pop_retry_count[thread_idx], 0);
    const uint64_t m4_rq_empty_poll = std::exchange(g_sched_m4_ready_queue_pop_empty_poll_count[thread_idx], 0);
    const uint64_t adv_cas = std::exchange(g_sched_advance_lock_cas_count[thread_idx], 0);
    const uint64_t m6_rq_ar = std::exchange(g_sched_m6_ready_queue_atomic_read_count[thread_idx], 0);
    const uint64_t m6_rq_aw = std::exchange(g_sched_m6_ready_queue_atomic_write_count[thread_idx], 0);
    const uint64_t m4_tn = std::exchange(g_sched_m4_payload_tensor_lane_reads[thread_idx], 0);
    std::exchange(g_sched_m4_payload_scalar_lane_reads[thread_idx], 0);

    // ===== CSV ② 依赖构建（wire_task / fanin / dep / readyQ）=====
    // PTO2TaskSlotState：read≈1+P→wt+wf；write≈3+P→3*wt+wf；atomic≈1+3P→wt+3*wf；锁/CAS≈P→wf（与 CSV 符号行近似对应）
    d.csv_m2_pto2_task_slot_state.read_events = wt + wf;
    d.csv_m2_pto2_task_slot_state.write_events = wt * 3 + wf;
    d.csv_m2_pto2_task_slot_state.atomic_ops = wt + wf * 3;
    d.csv_m2_pto2_task_slot_state.atomic_read_ops = wf * 2;
    d.csv_m2_pto2_task_slot_state.atomic_write_ops = wt + wf;
    d.csv_m2_pto2_task_slot_state.lock_ops = wf;
    d.csv_m2_pto2_task_slot_state.cas_ops = wf;
    // PTO2TaskPayload：CSV② 读 fanin 元数据 1 次/任务→read=wt
    d.csv_m2_pto2_task_payload.read_events = wt;
    d.csv_m2_pto2_task_payload.write_events = 0;
    d.csv_m2_pto2_task_payload.atomic_ops = 0;
    d.csv_m2_pto2_task_payload.atomic_read_ops = 0;
    d.csv_m2_pto2_task_payload.atomic_write_ops = 0;
    d.csv_m2_pto2_task_payload.lock_ops = 0;
    d.csv_m2_pto2_task_payload.cas_ops = 0;
    // PTO2DepListEntry：CSV② 写 P 条→write=wdep
    d.csv_m2_pto2_dep_list_entry.read_events = 0;
    d.csv_m2_pto2_dep_list_entry.write_events = wdep;
    d.csv_m2_pto2_dep_list_entry.atomic_ops = 0;
    d.csv_m2_pto2_dep_list_entry.atomic_read_ops = 0;
    d.csv_m2_pto2_dep_list_entry.atomic_write_ops = 0;
    d.csv_m2_pto2_dep_list_entry.lock_ops = 0;
    d.csv_m2_pto2_dep_list_entry.cas_ops = 0;
    // PTO2ReadyQueue：wiring 侧读/写近似 wt；原子用本窗口 fanout_atomic；CAS 近似 wt
    d.csv_m2_pto2_ready_queue.read_events = wt;
    d.csv_m2_pto2_ready_queue.write_events = wt;
    d.csv_m2_pto2_ready_queue.atomic_read_ops = m2_rq_ar_raw;
    d.csv_m2_pto2_ready_queue.atomic_write_ops = m2_rq_aw_raw;
    d.csv_m2_pto2_ready_queue.atomic_ops = m2_rq_ar_raw + m2_rq_aw_raw;
    d.csv_m2_pto2_ready_queue.lock_ops = 0;
    d.csv_m2_pto2_ready_queue.cas_ops = wt;
    d.csv_m2_pto2_ready_queue_spin_retry_ops = m2_rq_spin_retry;

    // CSV ③：调度器未单独拆 RingFC / SharedMemoryHeader 事件，此处五列置零占位
    d.csv_m3_pto2_ring_flow_control = {};
    d.csv_m3_pto2_shared_memory_header = {};

    // ===== CSV ④ Dispatch：dsub=子任务下发；dtask=取出的任务数；m4_tn/m4_sc=build_payload 张量/标量槽读；rqp=pop 轮次；pop_at=pop 路径原子 =====
    d.csv_m4_pto2_task_slot_state.read_events = dsub;
    d.csv_m4_pto2_task_slot_state.write_events = dsub;
    d.csv_m4_pto2_task_slot_state.atomic_ops = dsub;
    d.csv_m4_pto2_task_slot_state.atomic_read_ops = dsub;
    d.csv_m4_pto2_task_slot_state.atomic_write_ops = dsub;
    d.csv_m4_pto2_task_slot_state.lock_ops = 0;
    d.csv_m4_pto2_task_slot_state.cas_ops = 0;

    // 理论口径：CSV④ 的 PTO2TaskPayload 次数按每个 subtask 读取一次计
    d.csv_m4_pto2_task_payload_meta.read_events = dsub;
    d.csv_m4_pto2_task_payload_meta.write_events = 0;
    d.csv_m4_pto2_task_payload_meta.atomic_ops = 0;
    d.csv_m4_pto2_task_payload_meta.atomic_read_ops = 0;
    d.csv_m4_pto2_task_payload_meta.atomic_write_ops = 0;
    d.csv_m4_pto2_task_payload_meta.lock_ops = 0;
    d.csv_m4_pto2_task_payload_meta.cas_ops = 0;

    // 字段级细分读（tensors/scalars）不并入理论口径主行，保留为 0 防止与 xlsx 次数口径重复累计
    d.csv_m4_pto2_task_payload_tensors.read_events = 0;
    d.csv_m4_pto2_task_payload_tensors.write_events = 0;
    d.csv_m4_pto2_task_payload_tensors.atomic_ops = 0;
    d.csv_m4_pto2_task_payload_tensors.atomic_read_ops = 0;
    d.csv_m4_pto2_task_payload_tensors.atomic_write_ops = 0;
    d.csv_m4_pto2_task_payload_tensors.lock_ops = 0;
    d.csv_m4_pto2_task_payload_tensors.cas_ops = 0;

    d.csv_m4_pto2_task_payload_scalars.read_events = 0;
    d.csv_m4_pto2_task_payload_scalars.write_events = 0;
    d.csv_m4_pto2_task_payload_scalars.atomic_ops = 0;
    d.csv_m4_pto2_task_payload_scalars.atomic_read_ops = 0;
    d.csv_m4_pto2_task_payload_scalars.atomic_write_ops = 0;
    d.csv_m4_pto2_task_payload_scalars.lock_ops = 0;
    d.csv_m4_pto2_task_payload_scalars.cas_ops = 0;

    d.csv_m4_pto2_task_descriptor.read_events = dtask;
    d.csv_m4_pto2_task_descriptor.write_events = 0;
    d.csv_m4_pto2_task_descriptor.atomic_ops = 0;
    d.csv_m4_pto2_task_descriptor.atomic_read_ops = 0;
    d.csv_m4_pto2_task_descriptor.atomic_write_ops = 0;
    d.csv_m4_pto2_task_descriptor.lock_ops = 0;
    d.csv_m4_pto2_task_descriptor.cas_ops = 0;

    d.csv_m4_pto2_dispatch_payload.read_events = 0;
    // xlsx「次数」口径按一次dispatch记1次；19条cacheline体现在“每次读/写cacheline(B)”而非次数列
    d.csv_m4_pto2_dispatch_payload.write_events = dsub;
    d.csv_m4_pto2_dispatch_payload.atomic_ops = 0;
    d.csv_m4_pto2_dispatch_payload.atomic_read_ops = 0;
    d.csv_m4_pto2_dispatch_payload.atomic_write_ops = 0;
    d.csv_m4_pto2_dispatch_payload.lock_ops = 0;
    d.csv_m4_pto2_dispatch_payload.cas_ops = 0;

    // 理论口径：CSV④ ReadyQueue(pop) 以 subtask 次数 S 为主尺度。
    // 自旋重试使用 pop_batch 路径显式计数，不再用 atomic 差值反推。
    uint64_t rq_spin_retry = m4_rq_spin_retry;
    d.csv_m4_pto2_ready_queue.read_events = dsub;
    d.csv_m4_pto2_ready_queue.write_events = (dsub > 0) ? (dsub - 1) : 0;
    d.csv_m4_pto2_ready_queue.atomic_ops = dsub * 3 + rq_spin_retry;
    d.csv_m4_pto2_ready_queue.atomic_read_ops = pop_ar;
    d.csv_m4_pto2_ready_queue.atomic_write_ops = pop_aw;
    d.csv_m4_pto2_ready_queue.lock_ops = 0;
    d.csv_m4_pto2_ready_queue.cas_ops = dsub;
    d.csv_m4_pto2_ready_queue_spin_retry_ops = m4_rq_spin_retry;
    d.csv_m4_pto2_ready_queue_empty_poll_ops = m4_rq_empty_poll;

    // ④ ReadyQueue(pop) 拆分：按每次 pop 调用是否命中拆成两行（命中/空转）
    d.csv_m4_pto2_ready_queue_pop_hit.read_events = rqh;
    d.csv_m4_pto2_ready_queue_pop_hit.write_events = 0;
    d.csv_m4_pto2_ready_queue_pop_hit.atomic_ops = rqha;
    d.csv_m4_pto2_ready_queue_pop_hit.atomic_read_ops = rqhar;
    d.csv_m4_pto2_ready_queue_pop_hit.atomic_write_ops = rqhaw;
    d.csv_m4_pto2_ready_queue_pop_hit.lock_ops = 0;
    d.csv_m4_pto2_ready_queue_pop_hit.cas_ops = rqh;

    d.csv_m4_pto2_ready_queue_pop_miss.read_events = rqm;
    d.csv_m4_pto2_ready_queue_pop_miss.write_events = 0;
    d.csv_m4_pto2_ready_queue_pop_miss.atomic_ops = rqma;
    d.csv_m4_pto2_ready_queue_pop_miss.atomic_read_ops = rqmar;
    d.csv_m4_pto2_ready_queue_pop_miss.atomic_write_ops = rqmaw;
    d.csv_m4_pto2_ready_queue_pop_miss.lock_ops = 0;
    d.csv_m4_pto2_ready_queue_pop_miss.cas_ops = rqm;

    // ===== CSV ⑤ AICore：scc=fetch_add completed_subtasks；DispatchPayload 读次数按1×scc（cacheline体量另算）=====
    d.csv_m5_pto2_task_slot_state.read_events = scc;
    d.csv_m5_pto2_task_slot_state.write_events = scc;
    d.csv_m5_pto2_task_slot_state.atomic_ops = scc;
    d.csv_m5_pto2_task_slot_state.atomic_read_ops = 0;
    d.csv_m5_pto2_task_slot_state.atomic_write_ops = scc;
    d.csv_m5_pto2_task_slot_state.lock_ops = 0;
    d.csv_m5_pto2_task_slot_state.cas_ops = 0;

    d.csv_m5_pto2_dispatch_payload.read_events = scc;
    d.csv_m5_pto2_dispatch_payload.write_events = 0;
    d.csv_m5_pto2_dispatch_payload.atomic_ops = 0;
    d.csv_m5_pto2_dispatch_payload.atomic_read_ops = 0;
    d.csv_m5_pto2_dispatch_payload.atomic_write_ops = 0;
    d.csv_m5_pto2_dispatch_payload.lock_ops = 0;
    d.csv_m5_pto2_dispatch_payload.cas_ops = 0;

    d.csv_m5_tensor.read_events = m4_tn * 2;
    d.csv_m5_tensor.write_events = 0;
    d.csv_m5_tensor.atomic_ops = 0;
    d.csv_m5_tensor.atomic_read_ops = 0;
    d.csv_m5_tensor.atomic_write_ops = 0;
    d.csv_m5_tensor.lock_ops = 0;
    d.csv_m5_tensor.cas_ops = 0;

    // ===== CSV ⑥：complete_count=混合完成次数；fe=fanout 边；teq=入队；dlt=dep 遍历；公式为 CSV 行 1+C、2+C 等之运行期近似 =====
    d.csv_m6_pto2_task_slot_state.read_events = d.complete_count + fe;
    d.csv_m6_pto2_task_slot_state.write_events = d.complete_count * 2 + fe;
    d.csv_m6_pto2_task_slot_state.atomic_ops = d.complete_count * 3 + fe;
    d.csv_m6_pto2_task_slot_state.atomic_read_ops = lock_ar + ((fanout_ar > m6_rq_ar) ? (fanout_ar - m6_rq_ar) : 0);
    d.csv_m6_pto2_task_slot_state.atomic_write_ops = lock_aw + ((fanout_aw > m6_rq_aw) ? (fanout_aw - m6_rq_aw) : 0);
    d.csv_m6_pto2_task_slot_state.lock_ops = d.complete_count;
    d.csv_m6_pto2_task_slot_state.cas_ops = d.complete_count;

    d.csv_m6_pto2_dep_list_entry.read_events = dlt;
    d.csv_m6_pto2_dep_list_entry.write_events = 0;
    d.csv_m6_pto2_dep_list_entry.atomic_ops = 0;
    d.csv_m6_pto2_dep_list_entry.atomic_read_ops = 0;
    d.csv_m6_pto2_dep_list_entry.atomic_write_ops = 0;
    d.csv_m6_pto2_dep_list_entry.lock_ops = 0;
    d.csv_m6_pto2_dep_list_entry.cas_ops = 0;

    d.csv_m6_pto2_ready_queue.read_events = 0;
    d.csv_m6_pto2_ready_queue.write_events = teq;
    d.csv_m6_pto2_ready_queue.atomic_ops = d.fanin_atomic_count;
    d.csv_m6_pto2_ready_queue.atomic_read_ops = m6_rq_ar;
    d.csv_m6_pto2_ready_queue.atomic_write_ops = m6_rq_aw;
    d.csv_m6_pto2_ready_queue.lock_ops = 0;
    d.csv_m6_pto2_ready_queue.cas_ops = teq;

    // ===== CSV ⑦：trel=on_task_release 次数；finrel=∑P fetch_add；spill=spill 读；adv=advance_ring；adv_cas=advance_lock CAS 尝试 =====
    d.csv_m7_pto2_task_slot_state.read_events = trel + finrel;
    d.csv_m7_pto2_task_slot_state.write_events = trel + finrel;
    d.csv_m7_pto2_task_slot_state.atomic_ops = trel * 2 + finrel;
    d.csv_m7_pto2_task_slot_state.atomic_read_ops = fanin_ar + self_ar;
    d.csv_m7_pto2_task_slot_state.atomic_write_ops = fanin_aw + self_aw;
    d.csv_m7_pto2_task_slot_state.lock_ops = 0;
    d.csv_m7_pto2_task_slot_state.cas_ops = finrel;

    d.csv_m7_pto2_task_payload.read_events = trel;
    d.csv_m7_pto2_task_payload.write_events = 0;
    d.csv_m7_pto2_task_payload.atomic_ops = 0;
    d.csv_m7_pto2_task_payload.atomic_read_ops = 0;
    d.csv_m7_pto2_task_payload.atomic_write_ops = 0;
    d.csv_m7_pto2_task_payload.lock_ops = 0;
    d.csv_m7_pto2_task_payload.cas_ops = 0;

    d.csv_m7_pto2_ring_flow_control.read_events = 0;
    d.csv_m7_pto2_ring_flow_control.write_events = adv;
    d.csv_m7_pto2_ring_flow_control.atomic_ops = adv * 2;
    d.csv_m7_pto2_ring_flow_control.atomic_read_ops = 0;
    d.csv_m7_pto2_ring_flow_control.atomic_write_ops = adv * 2;
    d.csv_m7_pto2_ring_flow_control.lock_ops = 0;
    d.csv_m7_pto2_ring_flow_control.cas_ops = 0;

    d.csv_m7_pto2_fanin_spill_entry.read_events = spill;
    d.csv_m7_pto2_fanin_spill_entry.write_events = 0;
    d.csv_m7_pto2_fanin_spill_entry.atomic_ops = 0;
    d.csv_m7_pto2_fanin_spill_entry.atomic_read_ops = 0;
    d.csv_m7_pto2_fanin_spill_entry.atomic_write_ops = 0;
    d.csv_m7_pto2_fanin_spill_entry.lock_ops = 0;
    d.csv_m7_pto2_fanin_spill_entry.cas_ops = 0;

    d.csv_m7_ring_sched_state_advance_lock.read_events = adv_cas;
    d.csv_m7_ring_sched_state_advance_lock.write_events = adv_cas;
    d.csv_m7_ring_sched_state_advance_lock.atomic_ops = adv_cas;
    d.csv_m7_ring_sched_state_advance_lock.atomic_read_ops = adv_cas;
    d.csv_m7_ring_sched_state_advance_lock.atomic_write_ops = adv_cas;
    d.csv_m7_ring_sched_state_advance_lock.lock_ops = adv_cas;
    d.csv_m7_ring_sched_state_advance_lock.cas_ops = adv_cas;

    return d;
}
#endif

// =============================================================================
// Task State Names
// =============================================================================

const char *pto2_task_state_name(PTO2TaskState state) {
    switch (state) {
    case PTO2_TASK_PENDING:
        return "PENDING";
    case PTO2_TASK_READY:
        return "READY";
    case PTO2_TASK_RUNNING:
        return "RUNNING";
    case PTO2_TASK_COMPLETED:
        return "COMPLETED";
    case PTO2_TASK_CONSUMED:
        return "CONSUMED";
    default:
        return "UNKNOWN";
    }
}

// =============================================================================
// Ready Queue Implementation
// =============================================================================

bool pto2_ready_queue_init(PTO2ReadyQueue *queue, uint64_t capacity) {
    queue->slots = (PTO2ReadyQueueSlot *)malloc(capacity * sizeof(PTO2ReadyQueueSlot));
    if (!queue->slots) {
        return false;
    }

    queue->capacity = capacity;
    queue->mask = capacity - 1;
    queue->enqueue_pos.store(0, std::memory_order_relaxed);
    queue->dequeue_pos.store(0, std::memory_order_relaxed);

    for (uint64_t i = 0; i < capacity; i++) {
        queue->slots[i].sequence.store((int64_t)i, std::memory_order_relaxed);
        queue->slots[i].slot_state = nullptr;
    }

    return true;
}

void pto2_ready_queue_destroy(PTO2ReadyQueue *queue) {
    if (queue->slots) {
        free(queue->slots);
        queue->slots = NULL;
    }
}

// =============================================================================
// Scheduler Initialization
// =============================================================================

bool PTO2SchedulerState::RingSchedState::init(PTO2SharedMemoryHeader *sm_header, int32_t ring_id) {
    ring = &sm_header->rings[ring_id];
    last_task_alive = 0;
    advance_lock.store(0, std::memory_order_relaxed);

    // Initialize all per-task slot state fields.
    // bind() sets payload, task, ring_id — immutable after init, bound once
    // to their fixed shared-memory addresses.
    // reset_for_reuse() sets dynamic fields to reclaim defaults (fanout_count=1,
    // rest zero) so the first submit needs no reset.
    for (uint64_t i = 0; i < ring->task_window_size; i++) {
        ring->slot_states[i].bind(&ring->task_payloads[i], &ring->task_descriptors[i], static_cast<uint8_t>(ring_id));
        ring->slot_states[i].reset_for_reuse();
        ring->slot_states[i].fanin_count = 0;
        ring->slot_states[i].active_mask = 0;
        ring->slot_states[i].subtask_done_mask.store(0, std::memory_order_relaxed);
    }

    return true;
}

void PTO2SchedulerState::RingSchedState::destroy() { ring = nullptr; }

bool pto2_scheduler_init(PTO2SchedulerState *sched, PTO2SharedMemoryHeader *sm_header, int32_t dep_pool_capacity) {
    sched->sm_header = sm_header;
#if PTO2_SCHED_PROFILING
    sched->tasks_completed.store(0, std::memory_order_relaxed);
    sched->tasks_consumed.store(0, std::memory_order_relaxed);
#endif

    // Initialize per-ring state
    for (int r = 0; r < PTO2_MAX_RING_DEPTH; r++) {
        if (!sched->ring_sched_states[r].init(sm_header, r)) {
            for (int j = 0; j < r; j++) {
                sched->ring_sched_states[j].destroy();
            }
            return false;
        }
    }

    // Initialize ready queues (one per resource shape, global)
    for (int i = 0; i < PTO2_NUM_RESOURCE_SHAPES; i++) {
        if (!pto2_ready_queue_init(&sched->ready_queues[i], PTO2_READY_QUEUE_SIZE)) {
            // Cleanup on failure
            for (int j = 0; j < i; j++) {
                pto2_ready_queue_destroy(&sched->ready_queues[j]);
            }
            for (int r = 0; r < PTO2_MAX_RING_DEPTH; r++) {
                sched->ring_sched_states[r].destroy();
            }
            return false;
        }
    }

    // Initialize per-ring wiring queues and dep pools (exclusively managed by scheduler thread 0)
    for (int r = 0; r < PTO2_MAX_RING_DEPTH; r++) {
        PTO2DepListEntry *dep_entries =
            reinterpret_cast<PTO2DepListEntry *>(calloc(dep_pool_capacity, sizeof(PTO2DepListEntry)));
        if (!dep_entries) {
            for (int j = 0; j < r; j++) {
                free(sched->ring_sched_states[j].dep_pool.base);
            }
            for (int i = 0; i < PTO2_NUM_RESOURCE_SHAPES; i++) {
                pto2_ready_queue_destroy(&sched->ready_queues[i]);
            }
            sched->wiring.queue.destroy();
            for (int rr = 0; rr < PTO2_MAX_RING_DEPTH; rr++) {
                sched->ring_sched_states[rr].destroy();
            }
            return false;
        }
        sched->ring_sched_states[r].dep_pool.init(dep_entries, dep_pool_capacity, &sm_header->orch_error_code);
    }

    // Initialize global wiring queue (SPSC: orchestrator pushes, scheduler thread 0 drains)
    if (!sched->wiring.queue.init(PTO2_WRIRING_QUEUE_SIZE)) {
        for (int r = 0; r < PTO2_MAX_RING_DEPTH; r++) {
            free(sched->ring_sched_states[r].dep_pool.base);
        }
        for (int i = 0; i < PTO2_NUM_RESOURCE_SHAPES; i++) {
            pto2_ready_queue_destroy(&sched->ready_queues[i]);
        }
        for (int rr = 0; rr < PTO2_MAX_RING_DEPTH; rr++) {
            sched->ring_sched_states[rr].destroy();
        }
        return false;
    }
    sched->wiring.batch_count = 0;
    sched->wiring.batch_index = 0;
    sched->wiring.backoff_counter = 0;

    return true;
}

void pto2_scheduler_destroy(PTO2SchedulerState *sched) {
    for (int r = 0; r < PTO2_MAX_RING_DEPTH; r++) {
        sched->ring_sched_states[r].destroy();
        free(sched->ring_sched_states[r].dep_pool.base);
        sched->ring_sched_states[r].dep_pool.base = nullptr;
    }

    sched->wiring.queue.destroy();

    for (int i = 0; i < PTO2_NUM_RESOURCE_SHAPES; i++) {
        pto2_ready_queue_destroy(&sched->ready_queues[i]);
    }
}

// =============================================================================
// Debug Utilities
// =============================================================================

void pto2_scheduler_print_stats(PTO2SchedulerState *sched) {
    LOG_INFO("=== Scheduler Statistics ===");
    for (int r = 0; r < PTO2_MAX_RING_DEPTH; r++) {
        if (sched->ring_sched_states[r].last_task_alive > 0) {
            LOG_INFO("Ring %d:", r);
            LOG_INFO("  last_task_alive: %d", sched->ring_sched_states[r].last_task_alive);
            auto &dp = sched->ring_sched_states[r].dep_pool;
            if (dp.top > 0) {
                LOG_INFO(
                    "  dep_pool: top=%d tail=%d used=%d high_water=%d capacity=%d", dp.top, dp.tail, dp.top - dp.tail,
                    dp.high_water, dp.capacity
                );
            }
        }
    }
#if PTO2_SCHED_PROFILING
    LOG_INFO("tasks_completed:   %lld", (long long)sched->tasks_completed.load(std::memory_order_relaxed));
    LOG_INFO("tasks_consumed:    %lld", (long long)sched->tasks_consumed.load(std::memory_order_relaxed));
#endif
    LOG_INFO("============================");
}

void pto2_scheduler_print_queues(PTO2SchedulerState *sched) {
    LOG_INFO("=== Ready Queues ===");

    const char *shape_names[] = {"AIC", "AIV", "MIX"};

    for (int i = 0; i < PTO2_NUM_RESOURCE_SHAPES; i++) {
        LOG_INFO("  %s: count=%" PRIu64, shape_names[i], sched->ready_queues[i].size());
    }

    LOG_INFO("====================");
}
