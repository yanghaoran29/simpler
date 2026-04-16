# module-struct-access.csv 与 Paged Attention Unroll 样例（理论对照）

本文档说明仓库根目录 `module-struct-access.csv` 中的符号与各「模块」行含义，并结合  
`simpler/tests/st/a2a3/tensormap_and_ringbuffer/paged_attention_unroll` 编排源码做**理论量级**归纳，便于与 AICPU 侧 `DEV_ALWAYS` / profiling 或 CSV 五列对照。

---

## 1. 文档范围

- **CSV 来源**：`/module-struct-access.csv`（与 `simpler/…/tensormap_and_ringbuffer` 运行时统计口径对齐的设计表）。
- **编排样例**：`paged_attention_unroll/kernels/orchestration/paged_attention_orch.cpp`  
  - 测试入口：`paged_attention_unroll/test_paged_attention_unroll.py`（`CASES` 中 Case1~Case3）。
- **说明**：下表为**静态推导**；若 `context_lens` 非均匀、或运行时与实现细节（如 hidden alloc 是否计入某计数器）有差异，以当前 `pto_orchestrator.cpp` / `aicpu_executor.cpp` 为准。

---

## 2. CSV 行 1–9 符号（与 glossary / 注释一致）

| 符号 | 含义（CSV 注释摘要） |
|------|----------------------|
| **P** | fanin / 依赖路径上的 producer 条数（本 consumer 挂接的 producer 数） |
| **C** | 解依赖 fanout 链上的 consumer 近似；submit 瞬间常用 `fanout_count-1`，编排刚结束多为 **0** |
| **S** | 一次 dispatch / AICore 路径上的子任务数或轮次（本样例单簇单块下常取 **1**） |
| **N** | Ring 自旋等路径尝试上界（与 `0~N` 连用） |
| **N_in** | 本任务参与访问的**输入侧** tensor 槽位数（INPUT + INOUT 的输入侧） |
| **N_out** | 本任务参与访问的**输出侧** tensor 槽位数（OUTPUT / OUTPUT_EXISTING + INOUT 的输出侧） |
| **N_scope** | 与 `scope_end`、`fanout_refcount` 等配对的 scope 边界量级（glossary 中亦用 submit 时 scope 深度近似） |
| **tensor_count / scalar_count** | 当前 task payload 中 tensor / scalar 槽位个数（遍历上界） |

宏：**PTO2_FANIN_INLINE_CAP**、**PTO2_NUM_RESOURCE_SHAPES** 等见 `pto_runtime2_types.h` 与 CSV 行 10–11。

---

## 3. Paged Attention Unroll 编排拓扑

### 3.1 源码常量与循环

- `N_UNROLL`：**64**（`paged_attention_orch.cpp` 中 `#define N_UNROLL 64`；文件头注释中的 8 已过时）。
- 外层：`for (b_idx in batch)`，`for (q_idx in q_loop)`，内层 **`PTO2_SCOPE()`**。
- 每个 `PTO2_SCOPE` 内：
  1. **`alloc_tensors(tile2d_ci, scalar_ci, scalar_ci)`** — 1 次 hidden alloc（3 个 OUTPUT）。
  2. **`for (bn; bn < bn_this_batch; bn += N_UNROLL)`**，每组最多 **4** 个 mixed 任务：  
     **QK（AIC）→ Softmax（AIV）→ PV（AIC）→ OnlineUpdate（AIV）**。

### 3.2 记号

| 记号 | 定义 |
|------|------|
| \(B\) | `batch` |
| \(H\) | `num_heads` |
| `q_tile` | `min(H, 128)` |
| \(L_q\) | `q_loop = ⌈H / q_tile⌉` |
| \(N_{blk}\) | 每条请求 \(\lceil cur\_seq / block\_size \rceil\)；默认 `context_lens` 全为 `context_len` 时 \(N_{blk}=\lceil context\_len / block\_size\rceil\) |
| \(G\) | \(\lceil N_{blk} / N\_UNROLL \rceil\)（内层块组迭代次数） |

### 3.3 成功 submit 总数（含 hidden alloc）

每个 `(b_idx, q_idx)` 对应一个 `PTO2_SCOPE`，其内：

- **1** 次 `alloc_tensors`
- **\(4G\)** 次 mixed submit（每组 4 个任务）

\[
T_{\mathrm{total}} = B \cdot L_q \cdot (1 + 4G)
\]

---

## 4. 测试 Case 参数与 \(T_{\mathrm{total}}\)

`max_num_blocks_per_req = max_model_len // block_size`（golden 生成逻辑）；编排内  
`bn_this_batch = ⌈cur_seq / block_size⌉`（与 `context_lens` 一致时即 \(N_{blk}\)）。

| Case | batch | num_heads | block_size | context_len | \(N_{blk}\) | \(G\) | \(L_q\) | 每 scope<br>\(1+4G\) | \(T_{\mathrm{total}}\) |
|------|------:|----------:|-------------:|------------:|------:|------:|---------------------:|------------------------:|
| Case1 | 256 | 16 | 128 | 8192 | 64 | 1 | 1 | 5 | **1280** |
| Case2 | 64 | 64 | 64 | 8192 | 128 | 2 | 1 | 9 | **576** |
| Case3 | 64 | 64 | 64 | 8192 | 128 | 2 | 1 | 9 | **576** |

Case3 仅 `head_dim` 变大，**计数与 Case2 相同**。

**可变长 `context_lens`**：对每个 `b_idx` 单独算 \(N_{blk}(b)\)、\(G(b)\)，再求和  
\(B \cdot L_q + 4 \sum_{b,q} G(b)\)（在 \(L_q\) 与 batch 正交时）。

---

## 5. 单任务形状表（N_in / N_out / tensor_count / scalar_count / 典型 P）

与 `pto2_csv_glossary_count_n_in_n_out` 一致：**INOUT 同时计入 N_in 与 N_out**。

| 步骤 | 类型 | N_in | N_out | tensor_count | scalar_count | 典型 P（同 scope 内稳态） |
|------|------|-----:|------:|---------------:|---------------:|----------------------------|
| alloc_tensors | hidden | 0 | 3 | 3 | 0 | 0 |
| QK | AIC | 3 | 1 | 4 | 2 | 3 |
| Softmax | AIV | 1 | 3 | 4 | 3 | 1 |
| PV | AIC | 3 | 1 | 4 | 2 | 3 |
| OnlineUpdate | AIV | 7 | 4 | 7 | 2 | 3 |

**C**：submit 当口多为 **0**（见 CSV 行 2 与 glossary 说明）。

---

## 6. CSV「模块」列与全图量级（理论对照）

以下用 **Case1**（\(T_{\mathrm{total}}=1280\)，\(\sum P\) 按每 scope 一轮 \(0+3+1+3+3=10\) 计，共 \(256\times10=2560\)）作示意；Case2/3 将 \(B\)、\(G\) 代入 §3、§4 即可。

| CSV 模块 | 数据结构（CSV 行） | 理论量级（符号） | Case1 示意 |
|----------|-------------------|------------------|-------------|
| ① Payload | PTO2TaskSlotState | 读 \(\sim \sum P + N_{\mathrm{scope\_rel}}\)，写 \(\sim T_{\mathrm{total}} + \sum P + \cdots\) | \(\sum P=2560\)；槽位初始化 \(\sim T_{\mathrm{total}}\) |
| ① Payload | PTO2TaskPayload | 写 ≈ 每成功 payload init **1** 次 | **1280** |
| ① Payload | PTO2TaskDescriptor | 写 ≈ 每 mixed **1** 次（hidden 是否计入依实现） | **1024** 或 **1280** |
| ① Payload | Tensor | 读 \(\sum (N_{in}\times\text{次数})\)，写 \(\sum (N_{out}\times\text{次数})\) | 需按上表 × 各任务重复次数（×\(B L_q G\)）展开 |
| ① Payload | PTO2ReadyQueue | 写 ≈ wiring **push** 成功次数 | 与「是否对 alloc push」一致时 \(\sim T_{\mathrm{total}}\) 或 \(\sim\) mixed 数 |
| ① / ③ | PTO2RingFlowControl | 读/写/atomic 与每任务 **alloc** 及 spin **N** 相关 | 与 \(T_{\mathrm{total}}\) 同阶 |
| ② | SlotState / DepList / ReadyQueue | 公式中含 **P**、锁、CAS、wiring pop/push | \(\sum P\) 与 mixed 批次数同阶 |
| ④–⑦ | Dispatch / AICore / 解依赖 / 释放 | **S**、**C**、释放侧 **P** 与 spill 等 | 本样例 **C** 在运行期解依赖阶段累积；**S** 与任务数同阶 |

---

## 7. 相关路径索引

| 内容 | 路径 |
|------|------|
| CSV 设计表 | 仓库根 `module-struct-access.csv` |
| 编排实现 | `simpler/tests/st/a2a3/tensormap_and_ringbuffer/paged_attention_unroll/kernels/orchestration/paged_attention_orch.cpp` |
| 测试与参数 | `…/paged_attention_unroll/test_paged_attention_unroll.py` |
| Golden / 输入形状 | `simpler/simpler_setup/goldens/paged_attention.py` |
| Glossary 键与 N_in/N_out | `…/runtime/pto_orchestrator.cpp`（`pto2_csv_glossary_*`） |
| ① 五列快照打印 | `…/aicpu/aicpu_executor.cpp`（`PTO2_ORCH_PROFILING`） |

---

## 8. 修订记录

- 初版：对齐 `N_UNROLL=64` 与 `module-struct-access.csv` 符号，给出 Case1~3 的 \(T_{\mathrm{total}}\) 与单任务形状表。
