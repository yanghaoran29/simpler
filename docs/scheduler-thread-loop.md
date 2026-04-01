# Scheduler 线程主循环分析

## 概述

文件：`src/a2a3/runtime/tensormap_and_ringbuffer/aicpu/aicpu_executor.cpp`
函数：`AicpuExecutor::resolve_and_dispatch_pto2`

Scheduler 线程在此函数中运行一个紧密的主循环，直到所有任务完成或发生致命错误。
循环整体由 x23/x24 标记（`sched_loop` 阶段）括起，内部分为四个子阶段。

---

## 主循环结构

```
[x23] ← 循环迭代开始 (sched_loop start)
  │
  ├─ 0. 退出 / 错误检测
  ├─ 1. 核心转让检测
  │
[x25] ← complete 阶段开始
  ├─ 2. 完成轮询
  │     └─ [x17→x18] task_complete 打点（mixed_complete==true 时）
[x26] ← complete 阶段结束
  │
[x27] ← dispatch 阶段开始
  ├─ 3. 任务下发
  │     └─ [x21→x22] task_dispatch 打点（next_block_idx==0 时）
[x28] ← dispatch 阶段结束
  │
  ├─ 4. idle 阶段（无进展时）
  │     [x29] ← idle 开始
  │       └─ [x19→x20] idle_spin 打点
  │     [x30] ← idle 结束
  │
[x24] ← 循环迭代结束 (sched_loop end)
```

---

## 各阶段详细说明

### 阶段 0：退出 / 错误检测（无 marker）

每次迭代开始时，首先检查是否满足退出条件：

- **正常退出**：`orchestrator_done_ == true` 且 `completed_tasks >= total_tasks` → 设置 `completed_ = true`，break
- **致命错误退出**：`orch_error_code != PTO2_ERROR_NONE` → 调用 `emergency_shutdown()`，向所有核心发送 EXIT_SIGNAL，break

此检测仅在"没有任何核心正在运行"时触发（`!tracker.has_any_running_cores()`），避免提前退出。

### 阶段 1：核心转让检测（无 marker）

仅在 `orch_to_sched_` 模式下生效（Orchestrator 与 Scheduler 共享同一批 AICPU 线程）。

Orchestrator 完成提交阶段后，会设置 `transition_requested_`。Scheduler 线程检测到后进入自旋等待，
直到 `reassigned_` 被设置（Orchestrator 确认核心已移交）。每个线程只执行一次。

### 阶段 2：完成轮询（x25→x26，complete 阶段）

对本线程管理的所有"正在运行"核心（AIC + AIV），逐一检查完成状态。

```
for 每个 running core (AIC):
    reg_val = read_reg(reg_addr, RegId::COND)   ← 内存映射寄存器读
    done = (reg_task_id == expected) && (reg_state == TASK_FIN_STATE)
    if done:
        on_subtask_complete()                    ← 原子减计数，判断 mixed_complete
        if mixed_complete:
            [x17] task_complete start
            on_mixed_task_complete()             ← fanout：后继任务 refcount-1，满足依赖则入 ready_queue
            deferred_release_slot_states[] += slot_state  ← 延迟释放，避免频繁堆操作
            [x18] task_complete end
        tracker.change_core_state()              ← 标记该核心为 idle
同上处理 AIV 核心
```

**`task_complete` 打点（x17→x18）覆盖的工作：**
- `on_mixed_task_complete()`：遍历 fanout 边，对每条后继边的 fanin_refcount 做原子 fetch_add，
  满足条件的后继任务入 ready_queue
- deferred_release 管理：将 slot_state 加入延迟释放数组（满 256 时批量调用 `on_task_release()`）

### 阶段 3：任务下发（x27→x28，dispatch 阶段）

从 ready_queue 弹出就绪任务并下发给空闲核心。

```
get_dispatch_order(thread_idx)         ← 确定 MIX/AIC/AIV 资源形状优先顺序
for shape in dispatch_order:
    valid_cluster_states = tracker.get_valid_cluster_offset_states(shape)
    while valid_cluster_states 有空闲 cluster:
        pop_ready_tasks_batch()        ← 批量从 ready_queue 弹出任务
        for 每个 slot_state in batch:
            if next_block_idx == 0:
                [x21] task_dispatch start  ← 首次下发标记
            do:
                cluster_offset = valid_cluster_states.pop_first()
                dispatch_subtask_to_core() ← 写寄存器，将任务下发给 AIC/AIV0/AIV1
                tracker 标记该 core 为 running
                slot_state->next_block_idx++
            while block 未全部下发 && 还有空闲 cluster
            if 仍有剩余 block → ready_queues[shape].push(slot_state)  ← 重入队等待下次调度
            [x22] task_dispatch end
requeue local_buf 溢出任务回全局 ready_queue
```

**`task_dispatch` 打点（x21→x22）覆盖的工作：**
- do-while 循环：对任务的每个 block，计算核心地址，调用 `dispatch_subtask_to_core()`
  写 GM 寄存器（含 task_id、param_addr 等），并更新 CoreTracker 位图
- 剩余 block 重入队检查

**注意**：x21 仅在 `next_block_idx == 0` 时触发（首次下发），x22 对每个 slot 均触发。
对于再次下发的 slot（`next_block_idx > 0`），x22 触发时栈中无匹配的 x21 session，
QEMU 插件将其静默忽略，因此 `sessions(task_dispatch) == 任务总数`。

### 阶段 4：idle 阶段（x29→x30，无进展时）

当 complete 和 dispatch 均无进展（`made_progress == false`）时进入 idle 阶段：

```
idle_iterations++
[x29] idle start

每 1024 次迭代：
    检查 orch_error_code → fatal error → emergency_shutdown

thread_idx==0 且每 N 次（无进展）：
    打印 stall 日志（completed/total）
    扫描所有 task slot，统计 STUCK-READY / STUCK-WAITING / in-flight
    打印各 cluster 核心状态（idle/busy）

    [x19] idle_spin start（spin-wait 窗口）
批量处理 deferred_release（调用 on_task_release()，归还 slot_state 内存给分配器）
SPIN_WAIT_HINT()
    [x20] idle_spin end

if idle_iterations > MAX_IDLE_ITERATIONS → 超时，DEV_ERROR，return -1

[x30] idle end
```

idle 阶段的存在是为了在"所有任务都在硬件上运行，没有可下发的新任务"时，
用最小开销等待某个核心完成。

---

## 各阶段指令占比关系

| 阶段 marker | 阶段名 | 主要开销 | 与打点的关系 |
|---|---|---|---|
| x23→x24 | sched_loop | 一次完整迭代 | 所有子阶段之和 + 阶段间开销 |
| x25→x26 | complete | `read_reg` × 每个 running core | 包含 x17→x18 |
| x17→x18 | task_complete | `on_mixed_task_complete` + deferred_release | x25→x26 的子集，sessions = 任务数 |
| x27→x28 | dispatch | `pop_ready_tasks_batch` + `dispatch_subtask_to_core` × block | 包含 x21→x22 |
| x21→x22 | task_dispatch | do-while 写寄存器循环 + 重入队检查 | x27→x28 的子集，sessions = 任务数 |
| x29→x30 | idle | 自旋等待 + deferred_release 批处理 + stall 日志 | complete+dispatch 均无进展时 |
| x19→x20 | idle_spin | 纯 SPIN_WAIT_HINT 自旋 | x29→x30 的子集 |

典型负载下，大部分循环时间消耗在 **complete 轮询**（每次循环都读寄存器）
和 **idle 自旋**（所有任务都在硬件上跑、没有新的完成或就绪）上；
真正执行 task_dispatch / task_complete 代码的迭代只占少数。

---

## 打点数据解读

在 `scheduler_submit_report_*.md` 的 `## 1.5)` 节中：

- `task_dispatch.sessions ≈ task_complete.sessions ≈ 总任务数` → 调度逻辑正常
- `task_dispatch.avg_insns` 反映每次"首次下发"的实际指令代价（含 do-while 写寄存器）
- `task_complete.avg_insns` 反映每次 fanout + deferred_release 的实际指令代价
- `dispatch.avg_insns / task_dispatch.avg_insns` 的比值 > 1 时，
  说明 dispatch 阶段除首次下发外还有大量 pop_miss / requeue 开销
- `complete.sessions >> task_complete.sessions` 时，说明大量 complete 迭代在轮询未完成的核心（`read_reg` 但 `done==false`）
