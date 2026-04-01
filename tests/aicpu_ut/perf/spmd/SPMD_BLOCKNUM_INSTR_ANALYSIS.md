## 6.SPMD Multi-Block（MIX / AIV）：`block_num` 与 Orchestrator、Scheduler 指令分析

本文档合并原 **MIX** 与 **AIV** 两份分析，基于 `test_spmd_multiblock_mix.cpp`、`test_spmd_multiblock_aiv.cpp` 中 **idx=1…4** 的 workload，归纳编排器与调度器指令统计与 **`block_num`、任务数 N** 的关系，并专门解释：**为何 MIX 的 `task_dispatch` 会随 `block_num` 明显上升，而 AIV 在较小 `block_num` 下会「贴住」常数工作量 N·B**。

---

### 6.1. 共通实验设计与样例定义

两组用例采用 **相同的 (N, `block_num`) 四档**：1×8192、2×4096、4×2048、8×1024，且均满足 **N·B = 8192**（B = `block_num`），即 **总 block 步数** 相同。

差异在于 **每个 block 占用的子槽数**（下发形态不同）：

| PERF_CASE_IDX | B | N | 总 block 步 N·B | MIX：子任务完成次数 (N·B×3) | AIV：子任务完成次数 (N·B×1) |
|---------------|---|---|-----------------|------------------------------|------------------------------|
| 1 | 1 | 8192 | 8192 | 24576 | 8192 |
| 2 | 2 | 4096 | 8192 | 24576 | 8192 |
| 3 | 4 | 2048 | 8192 | 24576 | 8192 |
| 4 | 8 | 1024 | 8192 | 24576 | 8192 |

- **MIX**：每 block 下发 **AIC + AIV0 + AIV1**，调度 resource shape 为 **`MIX`**，一次内层迭代占满 **整簇** 上三个子任务。源码：`test_spmd_multiblock_mix.cpp`。
- **AIV**：仅 **AIV** 子槽，shape 为 **`AIV`**，一次内层迭代只向 **一个** AIV 核派发。源码：`test_spmd_multiblock_aiv.cpp`。

idx=0 均为各自「5 个大任务」原始图，下文 **idx 专指 1–4**。

---

### 6.2. 数据来源与报告结构

指令计数：`run_tests.sh --count-orch-and-sched-submit-instructions`。

合并报告路径示例：

- MIX：`outputs/log/test_spmd_mix_<idx>_orch_sched_submit_report_<timestamp>.md`（示例 `20260403_21*`）
- AIV：`outputs/log/test_spmd_aiv_<idx>_orch_sched_submit_report_<timestamp>.md`（示例 `20260403_222733` / `222906` / `223019` / `222938`）

报告内：**第一部分** Orchestrator（`pto2_submit_mixed_task` / `build_graph`）；**第二部分** Scheduler（`resolve_and_dispatch_pto2`）及 `task_dispatch` / `subtask_complete` / `task_complete`。

---

### 6.3. Orchestrator（`pto2_submit_mixed_task` / `build_graph`）

#### 6.3.1 MIX（示例跑批）

| idx | B | N | submit 总指令 | 平均每次 submit | build_graph 总指令 |
|-----|---|---|-----------------|-----------------|---------------------|
| 1 | 1 | 8192 | 4 874 810 | ~595.07 | 6 153 033 |
| 2 | 2 | 4096 | 2 437 690 | ~595.14 | 3 072 861 |
| 3 | 4 | 2048 | 1 219 130 | ~595.28 | 1 541 023 |
| 4 | 8 | 1024 | 609 850 | ~595.56 | 751 688 |

#### 6.3.2 AIV（示例跑批）

| idx | B | N | submit 总指令 | 平均每次 submit | build_graph 总指令 |
|-----|---|---|-----------------|-----------------|---------------------|
| 1 | 1 | 8192 | 4 874 795 | ~595.07 | 6 153 016 |
| 2 | 2 | 4096 | 2 437 675 | ~595.14 | 3 072 826 |
| 3 | 4 | 2048 | 1 219 115 | ~595.27 | 1 539 052 |
| 4 | 8 | 1024 | 609 835 | ~595.54 | 770 508 |

#### 6.3.3 结论（两组共性）

单次 submit 各阶段（alloc/sync/lookup/insert 等）几乎不随 B 变；总量 **≈ 595×N**。AIV 相对 MIX **`fanin` 平均约多 2 条指令**（118 vs 116），与仅挂 AIV 子槽的依赖路径有关，量级很小。

---

### 6.4. Scheduler：汇总数据

#### 6.4.1 MIX

| idx | B | N | 主循环总指令 | `task_dispatch`（次数 / 总指令） | `subtask_complete` | `task_complete` |
|-----|---|---|--------------|-----------------------------------|----------------------|-----------------|
| 1 | 1 | 8192 | 18 596 259 | 8192 / 2 228 224 | 24576 / 344 064 | 8192 / 278 528 |
| 2 | 2 | 4096 | 7 012 934 | 12229 / 2 421 342 | 24576 / 344 064 | 4096 / 208 896 |
| 3 | 4 | 2048 | 6 079 327 | 14299 / 2 516 624 | 24576 / 344 064 | 2048 / 172 032 |
| 4 | 8 | 1024 | 5 631 431 | 15279 / 2 566 872 | 24576 / 344 064 | 1024 / 89 088 |

#### 6.4.2 AIV

| idx | B | N | 主循环总指令 | `task_dispatch`（次数 / 总指令） | `subtask_complete` | `task_complete` |
|-----|---|---|--------------|-----------------------------------|----------------------|-----------------|
| 1 | 1 | 8192 | 16 941 458 | 8192 / 884 736 | 8192 / 114 688 | 8192 / 278 528 |
| 2 | 2 | 4096 | 10 201 541 | **8192** / 917 504 | 8192 / 114 688 | 4096 / 139 264 |
| 3 | 4 | 2048 | 3 048 027 | 9836 / 1 003 272 | 8192 / 114 688 | 2048 / 96 256 |
| 4 | 8 | 1024 | 2 106 679 | 11 233 / 1 078 368 | 8192 / 114 688 | 1024 / 54 272 |

#### 6.4.3 `subtask_complete` 与 `task_complete`

- **`subtask_complete`**：MIX 恒 **24576 = N·B×3**；AIV 恒 **8192 = N·B×1**。与每 block 子槽数一致。
- **`task_complete`**：次数均为 **N**（提交的图任务个数）。

---

### 6.5. `task_dispatch` 打点在代码里代表什么

合并报告中的 **`task_dispatch`** 对应工具对 **`dispatch_block`（`PTO2_SPECIAL_INSTRUCTION` x31/x32）** 的统计：**每进入一次**「按 block 下发」的 **`do-while` 内层循环体**（从 x31 到 x32）记 **一次 session**。

调度器在 `aicpu_executor.cpp` 中对每个 ready 任务执行：

```text
do {
    x31;
    … 从当前 valid_cluster_states 取一个 cluster，按 shape 下发子任务，next_block_idx++
    x32;
} while (next_block_idx < block_num && valid_cluster_states 仍可用);
```

若在 x31 之后 **拿不到 cluster** 或 bitmask 变空，会 **`break` 且本轮不 `next_block_idx++`**，任务 **重新入 ready 队列**；再次调度时会多一次 x31/x32 —— 这是 **S >「成功 block 步数」** 的根源。

**成功推进 block 的次数**恒为 **N·B**（本实验 N·B=8192）。记 **S** = `task_dispatch` sessions，则 **S ≥ N·B**；超出部分来自 **无进展的重试 / 再入队**。

---

### 6.6. 为何 MIX 的 `task_dispatch` 随 `block_num` 明显上升？

1. **MIX 每步资源更重**  
   `shape == MIX` 时，内层一次迭代要在 **同一 cluster** 上连续下发 **AIC、AIV0、AIV1**（`active_mask` 含三个子槽）。等价于 **每个 block 步需要「整簇」协同就绪**。

2. **多任务争用下易「半截退出」**  
   当 **B > 1** 时，同一任务要在 **连续多步**内反复占满整簇。其它 ready 任务同样抢 cluster，`valid_cluster_states` 常在 **尚未跑完当前任务的全部 B 步** 就被掏空 → `do-while` **提前 `break`** → **未递增的 block 步** 稍后以 **新的 session** 重试 → **S 累加**。

3. **B 越大，单任务「连贯跑完 B 步」越难**  
   因此 **在 N·B 固定为 8192 的前提下**，B 从 1 增到 8，**额外 session（S − N·B）** 单调增大：实测由 0 → 约 4037 → 6107 → 7087（相对 N·B）。

4. **经验拟合（仅 MIX、本四档）**  
   可用 **\(S \approx N(2B-1) = N·B + N(B-1)\)** 描述数量级；**不是定理**，详见下节备注。

#### 6.6.1 MIX 经验式 \(N(2B-1)\) 的含义与边界

| (N, B) | N(2B−1) | 实测 S | 偏差 |
|--------|---------|--------|------|
| (8192, 1) | 8192 | 8192 | 0 |
| (4096, 2) | 12288 | 12229 | −59 |
| (2048, 4) | 14336 | 14299 | −37 |
| (1024, 8) | 15360 | 15279 | −81 |

- **\(S \ge N·B\)** 恒成立。  
- **\(N(2B-1)\)** 来自 **B=1 时 S=N**、**S 对 N 近似线性**、**f(B) 取一次式并用 B=2 标定** 等假设下的 **最简拟合**；**系数 2 勿外推到任意拓扑**。  
- **N(B−1)** 项可叙述为：相对「每任务除首块外」的续块/重试带来的 **额外 dispatch_block** 量级解释，**不是严格计数**。

---

### 6.7. 为何 AIV 的 `task_dispatch` 在 B=1、2 上「贴住」N·B，看起来像「恒定」？

说明：下述 **「恒定」** 指 **在 idx=1 与 idx=2 两档，S 均等于 N·B（同为 8192）**，并不是说四档 B 下 S 全程不变 —— **idx=3、4 时 S 已大于 8192**。

#### 6.7.1 AIV 分支：单步只用一个 AIV 核

`shape == AIV` 时，内层循环 **每次只** `dispatch_subtask_to_core` **一个** AIV（在 AIV0/AIV1 间按 idle 选择），然后 `next_block_idx++`。**不要求** 同一时刻占满 AIC+双 AIV。

#### 6.7.2 更易在同一次 `do-while` 里连跑多步

源码在 **`next_block_idx < block_num`** 时会 **重新 `get_valid_cluster_offset_states(shape)`**，让调度器在 **同一 cluster 的两个 AIV** 之间切换。结果是：**对 cluster 的「瞬时占用」更碎、更短**，在 **B 较小**、并发压力尚未极端时，任务更常 **一口气跑完 B 步** 而 **不触发**「无 cluster → break → 再入队」。

#### 6.7.3 与 MIX 同场对比（关键一行）

| 后端 | (N,B)=(4096,2) | N·B | 实测 S |
|------|----------------|-----|--------|
| MIX | idx2 | 8192 | **12229**（显著 > N·B） |
| AIV | idx2 | 8192 | **8192**（= N·B） |

同一 **N·B**，**MIX 已出现大量额外 dispatch session**，**AIV 仍无超额** —— 差在 **每步资源粒度（整簇 vs 单 AIV）** 与 **AIV 路径的 cluster 状态刷新**。

#### 6.7.4 B 较大时 AIV 也会 S > N·B

idx=3、4 实测 **9836、11233**，仍符合 **争用加剧 → 提前 break → 重试** 的同一机制，但 **超额幅度小于 MIX**（MIX 在 B=4、8 时 S 更大）。

#### 6.7.5 勿把 MIX 的 \(N(2B-1)\) 套到 AIV

| (N, B) | N·B | AIV 实测 S | MIX 经验式 N(2B−1) |
|--------|-----|------------|---------------------|
| (8192, 1) | 8192 | 8192 | 8192 |
| (4096, 2) | 8192 | 8192 | 12288（严重高估 AIV） |
| (2048, 4) | 8192 | 9836 | 14336 |
| (1024, 8) | 8192 | 11233 | 15360 |

---

### 6.8. 一句话总览

| 现象 | MIX | AIV |
|------|-----|-----|
| `subtask_complete` 次数 | N·B×**3** | N·B×**1** |
| `task_dispatch` S 相对 N·B | **B 略增即明显 S > N·B** | **B=1、2 时常有 S = N·B**；B 更大才温和上升 |
| 主因 | 每 block **整簇**三连发，易半截退出与重试 | **单 AIV** 步进 + **刷新 cluster**，易连跑多 block |
| 经验闭式 | 本四档可试 **\(S \approx N(2B-1)\)**（拟合） | **勿套用** MIX 闭式；**S ≥ N·B** 恒真 |

---

### 6.9. 复现实验

```bash
cd simpler/tests/aicpu_ut
bash run_tests.sh --count-orch-and-sched-submit-instructions --test test_spmd_mix --idx <0-4>
bash run_tests.sh --count-orch-and-sched-submit-instructions --test test_spmd_aiv --idx <0-4>
```

输出目录：`outputs/log/`。

---

*若修改 `test_spmd_multiblock_mix.cpp` / `test_spmd_multiblock_aiv.cpp` 中的 N、B 组合，请同步更新本文表格与结论适用范围。*
