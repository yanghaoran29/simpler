# Scheduler 任务下发/回收打点说明

## 目的

验证 Scheduler 线程对任务的下发次数与回收次数均等于总任务数。

## 打点方式

使用 `PTO2_SPECIAL_INSTRUCTION`（`orr xN, xN, xN` AArch64 NOP 指令）作为标记对，
由 QEMU TCG 插件（`libinsn_count.so`，`sched_phases=1`）统计标记对的触发次数（`sessions`）。
`sessions` 即等于各打点位置的执行次数。

## 打点位置

文件：`src/a2a3/runtime/tensormap_and_ringbuffer/aicpu/aicpu_executor.cpp`

### 打点 1：task_dispatch（x21/x22）

**函数**：`AicpuExecutor::resolve_and_dispatch_pto2`

**位置**：dispatch 循环内，`for (int bi = 0; bi < got; bi++)` 中，`next_block_idx == 0` 分支。

```cpp
if (slot_state->next_block_idx == 0) {
    PTO2_SPECIAL_INSTRUCTION(21, PTO2_SPECIAL_INS_PLAIN);  // task_dispatch start
    PTO2_SPECIAL_INSTRUCTION(22, PTO2_SPECIAL_INS_PLAIN);  // task_dispatch end
}
```

**判据**：`next_block_idx{0}` 在 `PTO2TaskSlotState` 中初始化为 0。
只有首次从 ready queue 弹出的任务满足此条件；若任务有多个 block 被重新入队，
再次弹出时 `next_block_idx > 0`，不重复计数。

### 打点 2：task_complete（x17/x18）

**函数**：`AicpuExecutor::check_running_cores_for_completion<CT>`

**位置**：`if (mixed_complete)` 分支开头。

```cpp
if (mixed_complete) {
    PTO2_SPECIAL_INSTRUCTION(17, PTO2_SPECIAL_INS_PLAIN);  // task_complete start
    PTO2_SPECIAL_INSTRUCTION(18, PTO2_SPECIAL_INS_PLAIN);  // task_complete end
    ...
}
```

**判据**：`mixed_complete` 由 `on_subtask_complete` 在任务的**全部** subtask 均已完成时返回 true，
每个任务精确触发一次，因此此计数以任务为粒度。

## 标记编码

| 打点名 | 寄存器对 | start 编码 | end 编码 | plugin phase_id |
|--------|----------|-----------|---------|-----------------|
| `task_dispatch` | x21/x22 | `0xaa1502b5` | `0xaa1602d6` | 16 |
| `task_complete` | x17/x18 | `0xaa110231` | `0xaa120252` | 17 |

编码公式：`orr xN, xN, xN` → `0xaa<N>02<N_lo>` (AArch64 shifted-register ORR)。

## 插件输出格式

`libinsn_count.so`（`sched_phases=1`）在运行结束时输出：

```
QEMU_TCG phase_insns: phase_id=16 name=task_dispatch sessions=1024 avg=1 max=1 min=1
QEMU_TCG phase_insns: phase_id=17 name=task_complete sessions=1024 avg=1 max=1 min=1
```

`sessions` 字段即为该标记对触发次数（等于任务数）。

## 验证方法

运行 `count_scheduler_loop_markers.sh`（需 `PTO2_INSTR_COUNT_SCHEDULER_LOOP_ENABLE=1` 构建 + 插件重编译）：

```bash
make -C tests/aicpu_ut/plugins   # 重编译插件
TEST_NAME=test_batch_paged_attention TEST_IDX=0 \
  PTO2_INSTR_COUNT_SCHEDULER_LOOP_ENABLE=1 \
  ./tests/aicpu_ut/tools/count_scheduler_loop_markers.sh
```

汇总文件（`scheduler_loop_summary_*.txt`）和 markdown 报告（`scheduler_submit_report_*.md`）
均包含 `## 1.5) 任务下发/回收计数` 节，显示 `task_dispatch.sessions` 与 `task_complete.sessions`
及其是否相等。

## 注意事项

- 打点使用 `PTO2_SPECIAL_INSTRUCTION`，仅在 QEMU+plugin 环境下统计；on-board 运行无额外开销。
- 标记对 x21/x22 和 x17/x18 已加入 `insn_count.c` 的 `g_marker_pairs_sched_phases[]`，
  需重新编译插件（`make -C tests/aicpu_ut/plugins`）方可识别。
- 两打点均以**任务**为粒度：task_dispatch 仅在首次下发时触发，task_complete 仅在全部 subtask 完成时触发。
