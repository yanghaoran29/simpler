# PTO Runtime2 Profiling 机制分析

本文档从数据结构、开关层级、数据流和 Host/Device 分工四方面分析代码中的 profiling 机制。

---

## 1. 总体架构

Profiling 分为两类：

1. **任务级（PerfRecord）**：每条记录对应一次 AICore 上的任务执行，含时间戳、task_id、fanout 等，用于时间线/泳道图。
2. **阶段级（Phase）**：每条记录对应 Scheduler 或 Orchestrator 的一个阶段（complete/dispatch/scan/idle 或 sync/alloc/lookup 等），用于分析调度与编排开销。

两者共用 **Host–Device 共享内存**：固定头 + 每 core/每 thread 的 buffer 状态 + 由 Host 分配并注入的 **PerfBuffer / PhaseBuffer**。Device（AICPU）写记录，Host 通过 **ReadyQueue** 取满的 buffer 做后续分析或落盘。

---

## 2. 宏开关层级

（见 `pto_runtime2_types.h` 与 `profiling_levels.md`）

| 宏 | 默认值 | 含义 |
|----|--------|------|
| **PTO2_PROFILING** | 1 | 总开关：为 0 时无任何 profiling 计数与 buffer 写入。 |
| **PTO2_SCHED_PROFILING** | 0 | Scheduler 细粒度：per-phase 周期、原子操作数、lock/push/pop 等待。 |
| **PTO2_ORCH_PROFILING** | 0 | Orchestrator 细粒度：各子阶段周期、wait cycle、原子数。 |
| **PTO2_TENSORMAP_PROFILING** | 0 | 依赖 PTO2_ORCH_PROFILING，TensorMap 相关统计。 |
| **PTO2_PROFILING_BEGINEND** | 0 | 仅打首尾点：Orchestrator 每 submit 只记录 ORCH_BEGIN/ORCH_END；Scheduler 每轮循环只记录 SCHED_LOOP_BEGIN/SCHED_LOOP_END。 |

约束：`PTO2_ORCH_PROFILING` / `PTO2_SCHED_PROFILING` / `PTO2_PROFILING_BEGINEND` 必须与 `PTO2_PROFILING=1` 同时开启，否则编译报错。

运行时还有 **enable_profiling**：控制是否分配/使用 perf 共享区、是否写 PerfRecord/Phase；**不影响** 已编译的日志输出（例如 scheduler summary 在 PTO2_PROFILING=1 时仍会打）。

---

## 3. 共享内存布局（perf_profiling.h）

```
┌─────────────────────────────────────────────────────────────┐
│ PerfDataHeader                                               │
│   - queues[thread][slot]     per-thread ReadyQueue (FIFO)    │
│   - queue_heads/tails       Host 消费 / AICPU 生产           │
│   - num_cores, total_tasks                                   │
├─────────────────────────────────────────────────────────────┤
│ PerfBufferState[0..num_cores-1]  每 core 一条                │
│   - free_queue (SPSC)         Host 推入空闲 buffer，AICPU 弹出│
│   - current_buf_ptr           当前写入的 PerfBuffer 地址      │
│   - current_buf_seq           序列号                         │
├─────────────────────────────────────────────────────────────┤
│ AicpuPhaseHeader (phase 开启时)                              │
│   - magic, num_sched_threads, records_per_thread            │
│   - core_to_thread[], orch_summary                           │
├─────────────────────────────────────────────────────────────┤
│ PhaseBufferState[0..num_sched_threads+orch-1]  每线程一条    │
│   - 结构同 PerfBufferState，current_buf_ptr → PhaseBuffer    │
└─────────────────────────────────────────────────────────────┘
```

- **PerfBuffer**、**PhaseBuffer** 由 Host 分配，指针放入各 `free_queue`；Device 在需要时从 free_queue 弹出，写满后通过 **ReadyQueue** 入队，Host 从 ReadyQueue 取回并处理。
- 计算大小：`calc_perf_data_size(num_cores)` 仅头+PerfBufferState；带 phase 时用 `calc_perf_data_size_with_phases(num_cores, num_sched_threads)`。

---

## 4. 任务级 Profiling（PerfRecord）

### 4.1 数据结构（perf_profiling.h）

```cpp
struct PerfRecord {
    uint64_t start_time, end_time, duration, kernel_ready_time;  // AICore 写
    uint64_t dispatch_time, finish_time;                         // AICPU 写
    uint32_t task_id, func_id;
    CoreType core_type;
    int32_t fanout[RUNTIME_MAX_FANOUT];
    int32_t fanout_count;
};
```

- **AICore**（或仿真等价路径）：写 `start_time`、`end_time`、`duration`、`kernel_ready_time`、`task_id`；写入当前 core 的 `Handshake::perf_records_addr` 指向的 PerfBuffer，按 `count` 递增追加。
- **AICPU**：在 `drain_completed_cores` 中检测到某 core 完成任务时，取该 core 的 `PerfBuffer*` 和最后一条 record，补写 `dispatch_time`、`finish_time`、`func_id`、`core_type`（`perf_aicpu_record_dispatch_and_finish_time`）。
- **Buffer 满时**：`perf_aicpu_switch_buffer` 被调用，在切换前对当前 PerfBuffer 调用 **Runtime::complete_perf_records**，对每条 record 根据 task_id 找到对应 slot_state，遍历 **fanout 链表** 填 `fanout[]` 和 `fanout_count`；然后将该 buffer 入队 ReadyQueue，从 free_queue 再弹一块作为新 current buffer。

因此 **fanout 的填写在 AICPU 侧、且发生在 buffer 切换时**，依赖 PTO2 共享内存中的 slot_states 与 fanout 链表（见 runtime.cpp）。

---

## 5. 阶段级 Profiling（Phase）

### 5.1 Phase 类型（AicpuPhaseId）

- **Scheduler**：`SCHED_COMPLETE`、`SCHED_DISPATCH`、`SCHED_SCAN`、`SCHED_IDLE_WAIT`。
- **Orchestrator**：`ORCH_SYNC`、`ORCH_ALLOC`、`ORCH_PARAMS`、`ORCH_LOOKUP`、`ORCH_HEAP`、`ORCH_INSERT`、`ORCH_FANIN`、`ORCH_FINALIZE`、`ORCH_SCOPE_END`。

### 5.2 写入路径

- **Scheduler**：在 `aicpu_executor.cpp` 的 `resolve_and_dispatch_pto2` 循环中，每轮按阶段打时间戳，在 complete/dispatch/idle 等段落末尾调用  
  `perf_aicpu_record_phase(thread_idx, AicpuPhaseId::SCHED_*, start, end, loop_iter, tasks_processed)`。
- **Orchestrator**：在 `pto_orchestrator.cpp` 中通过宏 `CYCLE_COUNT_LAP_RECORD(acc, phase_id, task_id)` 调用 `perf_aicpu_record_orch_phase`，内部再调 `perf_aicpu_record_phase(s_orch_thread_idx, ...)`。  
  `s_orch_thread_idx` 由 Host 在切到编排线程时设为当前 thread_idx，保证 phase 记录挂在正确的线程下。

### 5.3 PhaseBuffer 与 ReadyQueue

- `perf_aicpu_record_phase` 向当前线程的 **PhaseBuffer** 追加一条 `AicpuPhaseRecord`（start_time, end_time, loop_iter, phase_id, tasks_processed）。
- PhaseBuffer 满时内部会 `switch_phase_buffer`，将满 buffer 入队 ReadyQueue（is_phase=1），再从该线程的 phase free_queue 取新 buffer。
- 初始化：`perf_aicpu_init_phase_profiling(runtime, num_sched_threads, num_orch_threads)` 在 `perf_aicpu_init_profiling` 之后由线程 0 调用，建立 AicpuPhaseHeader 和各线程 PhaseBufferState，并给每个线程从 free_queue 弹出第一块 PhaseBuffer。

---

## 6. Scheduler 细粒度统计（PTO2_SCHED_PROFILING）

当 `PTO2_SCHED_PROFILING=1` 时，scheduler 内部不仅打 phase，还累加 **per-thread 的周期与原子计数**（pto_scheduler.h/cpp）：

- **lock_cycle / lock_wait_cycle / lock_atomic_count**：`pto2_fanout_lock` + state 更新 + unlock 的时间与 CAS/load 次数。
- **fanout_cycle / fanout_atomic_count**：遍历 fanout、对消费者做 `release_fanin_and_check_ready` 及 push 的耗时与原子数。
- **fanin_cycle / fanin_atomic_count**：`on_task_release` 中 release_producer 等。
- **self_consumed_cycle / self_atomic_count**：`check_and_handle_consumed`。
- **push_wait_cycle / pop_wait_cycle / pop_atomic_count**：ready queue 的 push/pop 竞争与原子数。
- **complete_count**：该线程本统计周期内完成的 mixed-task 数。

这些通过全局数组 `g_sched_*` 按 thread_idx 累加，由 **pto2_scheduler_get_profiling(thread_idx)** 一次取走并清零，供 `pto2_print_sched_profiling` 或 aicpu_ut 的 phase breakdown 打印（例如 device log 中的 complete/dispatch/idle 占比及 fanout/fanin 边数、原子数）。

---

## 7. Orchestrator 细粒度统计（PTO2_ORCH_PROFILING）

在 `pto_orchestrator.cpp` 中，各子阶段用 `CYCLE_COUNT_LAP_RECORD` 打 phase 的同时，把周期累加到静态变量：

- **g_orch_sync_cycle**、**g_orch_alloc_cycle**、**g_orch_params_cycle**、**g_orch_lookup_cycle**、**g_orch_heap_cycle**、**g_orch_insert_cycle**、**g_orch_fanin_cycle**、**g_orch_scope_end_cycle**；
- **g_orch_alloc_wait_cycle**、**g_orch_heap_wait_cycle**、**g_orch_fanin_wait_cycle**、**g_orch_finalize_wait_cycle**（在 task_ring_alloc、heap_ring_alloc、fanout_lock、ready push 等处自旋等待的周期）；
- **g_orch_*_atomic_count**：各阶段涉及的原子操作数。

`pto2_orchestrator_get_profiling()` 汇总为 **PTO2OrchProfilingData** 并清零；`pto2_print_orch_profiling` 或 aicpu_ut 的 orch 输出会打印这些周期与原子数。

---

## 8. 数据流小结

| 数据类型 | 生产者 | 写入位置 | 消费者 |
|----------|--------|----------|--------|
| PerfRecord | AICore 写时间戳/task_id；AICPU 补 dispatch/finish/func_id/core_type；switch_buffer 时 Runtime 补 fanout | PerfBuffer（按 core） | Host 从 ReadyQueue 取满 buffer，可生成 timeline/swimlane |
| AicpuPhaseRecord | AICPU 各线程（scheduler 循环 / orch 宏） | PhaseBuffer（按 thread） | Host 从 ReadyQueue 取（is_phase=1），做 phase 统计/device log |
| PTO2SchedProfilingData | pto_scheduler 内 g_sched_* | 进程内全局数组 | pto2_scheduler_get_profiling → 打印或 test 汇总 |
| AicpuOrchSummary | Orchestrator 结束时 | AicpuPhaseHeader::orch_summary | perf_aicpu_write_orch_summary 写共享头，Host 可读 |
| PTO2OrchProfilingData | pto_orchestrator 内 g_orch_* | 进程内静态变量 | pto2_orchestrator_get_profiling → 打印或 test 汇总 |

---

## 9. 相关文件索引

| 内容 | 文件 |
|------|------|
| PerfRecord / PhaseBuffer / 共享内存布局 / AicpuPhaseId | `platform/include/common/perf_profiling.h` |
| perf_aicpu_init_profiling、switch_buffer、record_phase、flush、orch_summary | `platform/src/aicpu/performance_collector_aicpu.cpp` |
| complete_perf_records（填 fanout） | `runtime/runtime.cpp` |
| Scheduler 内 g_sched_*、PTO2SchedProfilingData、get_profiling | `runtime/pto_scheduler.{h,cpp}` |
| Orchestrator 内 g_orch_*、CYCLE_COUNT_LAP_RECORD、get_profiling | `runtime/pto_orchestrator.{h,cpp}` |
| resolve_and_dispatch_pto2 中 phase 记录与 PTO2_PROFILING 分支 | `aicpu/aicpu_executor.cpp` |
| 宏层级与日志说明 | `docs/profiling_levels.md` |

---

本文档从“数据结构 + 开关 + 数据流 + 角色分工”四方面概括当前 profiling 机制，便于扩展或排查数据来源与写入时机。
