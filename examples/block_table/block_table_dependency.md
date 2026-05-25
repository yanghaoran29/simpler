# `block_table` 是什么 / Task 1 ↔ Task 2 依赖关系

本文档基于 `paged_consumer_block_table` 样例 (DSL: `paged_consumer_block_table_pypto_syntax.py`, 编译产物: `orchestration/paged_consumer_block_table.cpp` + `kernels/{aic,aiv}/`), 解释 `block_table` 的语义, 并列出此样例中 Task 2 对 Task 1 的逐项依赖。

---

## 1. `block_table` 是什么

`block_table` 是一张 **逻辑页号 → 物理页号** 的映射表, 数据类型 `int32`, 形状 `[NUM_PAGES]` (这里 `NUM_PAGES = 16`)。

- **物理页**: 中间张量 `paged_y` (FP32, `[NUM_PAGES * PAGE_M, N1] = [256, 256]`) 在内存里按 `PAGE_M = 16` 行一页连续摆放, 第 `pid` 个物理页对应 `paged_y[pid * 16 : (pid+1) * 16, :]`。Stage 1 (AIC `paged_proj`) 顺序写入: 第 `ob` 次迭代写到第 `ob` 个物理页。
- **逻辑页**: Stage 2 (AIV `paged_rmsnorm`) 的 `ob` 维度是 **输出** 的页号 (输出张量 `out` 也是 `[256, 256]`, 按 16 行一页)。第 `ob` 个输出页**不一定**来自第 `ob` 个物理页 —— 由 `block_table[ob]` 决定。

模仿 Qwen3 / DeepSeek paged-KV-cache decode 里 `block_table` 的角色: 逻辑 token 位置 → 物理 KV-cache 页, RMSNorm 作用在 gather 后的乱序数据上。

### 在 orchestrator 中的处理位置

Stage 2 orchestrator 在每个 `ob` 迭代里读一次 `block_table[ob]` 然后切片 `paged_y`:

```cpp
// orchestration/paged_consumer_block_table.cpp:59-66
uint32_t indices_t[1] = {static_cast<uint32_t>(ob)};
int32_t  t        = get_tensor_data<int32_t>(ext_block_table, 1, indices_t);
int64_t  page_id  = static_cast<int64_t>(t);
int64_t  src_row  = (page_id * 16);                     // page_id * PAGE_M
Tensor   y_src    = paged_y.view(y_src_shapes,           // [16, 256]
                                  y_src_offsets);        // [src_row, 0]
```

随后 `y_src` 作为 Task 1 (AIV) 的输入提交。注意 `block_table` 本身**只在 orchestration 里读**, 并不进入 AIV kernel 的参数列表。

---

## 2. 任务图

每个 `pl.parallel(0, NUM_PAGES)` 展开成 16 个 task:

- **Stage 1 (Task 0 系列, AIC `paged_proj__windowed`)**: `T1[ob]`, `ob ∈ [0, 16)`
  - 输入: `x_page = x[ob*16 : ob*16+16, :]` ([16, 2048] bf16) + `w1` (整张, [2048, 256] bf16, 所有 `T1[*]` 共享)
  - 输出: `paged_y[ob*16 : ob*16+16, :]` ([16, 256] fp32) —— **`ob` 个物理页**

- **Stage 2 (Task 1 系列, AIV `paged_rmsnorm__windowed`)**: `T2[ob]`, `ob ∈ [0, 16)`
  - 输入: `gamma` (整张 [1, 256] fp32, 所有 `T2[*]` 共享) + `y_src = paged_y[block_table[ob] * 16 : block_table[ob] * 16 + 16, :]`
  - 输出: `out[ob*16 : ob*16+16, :]` ([16, 256] fp32)

### 依赖语义 (一般式)

```text
T2[ob]  依赖  T1[ block_table[ob] ]      ∀ ob ∈ [0, 16)
```

也就是说**第 ob 个 Stage 2 任务依赖第 `block_table[ob]` 个 Stage 1 任务** —— `block_table` 在 Stage 1 全单射的前提下, 是一组 16→16 的 **一对一依赖**, 不是全连接。

---

## 3. 实测样例的依赖明细

本次 `build_output/_jit_paged_consumer_block_table_20260525_173546/data/in/block_table.pt` 实际抽到的随机置换为:

```text
block_table = [4, 2, 9, 11, 10, 15, 3, 0, 14, 8, 12, 6, 7, 1, 5, 13]
                ob=0 1  2   3   4   5  6  7   8  9  10 11 12 13 14 15
```

逐项展开 `T2[ob] → T1[block_table[ob]]`:

| `ob` (Stage 2 任务) | `block_table[ob]` | 依赖的 Stage 1 任务 | Stage 1 写入 `paged_y` 行范围 | Stage 2 读取 `paged_y` 行范围 | Stage 2 写入 `out` 行范围 |
| :---: | :---: | :---: | :---: | :---: | :---: |
| `T2[0]`  |  4 | `T1[4]`  | `[64, 80)`   | `[64, 80)`   | `[0, 16)`    |
| `T2[1]`  |  2 | `T1[2]`  | `[32, 48)`   | `[32, 48)`   | `[16, 32)`   |
| `T2[2]`  |  9 | `T1[9]`  | `[144, 160)` | `[144, 160)` | `[32, 48)`   |
| `T2[3]`  | 11 | `T1[11]` | `[176, 192)` | `[176, 192)` | `[48, 64)`   |
| `T2[4]`  | 10 | `T1[10]` | `[160, 176)` | `[160, 176)` | `[64, 80)`   |
| `T2[5]`  | 15 | `T1[15]` | `[240, 256)` | `[240, 256)` | `[80, 96)`   |
| `T2[6]`  |  3 | `T1[3]`  | `[48, 64)`   | `[48, 64)`   | `[96, 112)`  |
| `T2[7]`  |  0 | `T1[0]`  | `[0, 16)`    | `[0, 16)`    | `[112, 128)` |
| `T2[8]`  | 14 | `T1[14]` | `[224, 240)` | `[224, 240)` | `[128, 144)` |
| `T2[9]`  |  8 | `T1[8]`  | `[128, 144)` | `[128, 144)` | `[144, 160)` |
| `T2[10]` | 12 | `T1[12]` | `[192, 208)` | `[192, 208)` | `[160, 176)` |
| `T2[11]` |  6 | `T1[6]`  | `[96, 112)`  | `[96, 112)`  | `[176, 192)` |
| `T2[12]` |  7 | `T1[7]`  | `[112, 128)` | `[112, 128)` | `[192, 208)` |
| `T2[13]` |  1 | `T1[1]`  | `[16, 32)`   | `[16, 32)`   | `[208, 224)` |
| `T2[14]` |  5 | `T1[5]`  | `[80, 96)`   | `[80, 96)`   | `[224, 240)` |
| `T2[15]` | 13 | `T1[13]` | `[208, 224)` | `[208, 224)` | `[240, 256)` |

### 反向视角: 每个 Stage 1 任务被谁消费

由于 `block_table` 是 `torch.randperm(16)` 给的双射, 每个 `T1[pid]` 恰好被 **唯一一个** `T2[ob]` 消费, 其中 `ob = block_table⁻¹[pid]`:

| `T1[pid]` 写入物理页 | 被 `T2[ob]` 唯一消费 |
| :---: | :---: |
| `T1[0]`  → `T2[7]`  | `T1[8]`  → `T2[9]`  |
| `T1[1]`  → `T2[13]` | `T1[9]`  → `T2[2]`  |
| `T1[2]`  → `T2[1]`  | `T1[10]` → `T2[4]`  |
| `T1[3]`  → `T2[6]`  | `T1[11]` → `T2[3]`  |
| `T1[4]`  → `T2[0]`  | `T1[12]` → `T2[10]` |
| `T1[5]`  → `T2[14]` | `T1[13]` → `T2[15]` |
| `T1[6]`  → `T2[11]` | `T1[14]` → `T2[8]`  |
| `T1[7]`  → `T2[12]` | `T1[15]` → `T2[5]`  |

---

## 4. 逻辑依赖 vs 运行期 TensorMap 实际识别的依赖

把 `merged_swimlane.json` 里 `name == "dependency"` 的 flow event 全部解出来, 按 task name 中的 `r2t<id>` 反推 producer / consumer (Stage 1 = `r2t0..r2t15`, Stage 2 = `r2t16..r2t31`), 去重后得到 **16 条唯一边**, 与 §3 表格 100% 一致:

```text
T1[ 4] → T2[ 0]    T1[ 2] → T2[ 1]    T1[ 9] → T2[ 2]    T1[11] → T2[ 3]
T1[10] → T2[ 4]    T1[15] → T2[ 5]    T1[ 3] → T2[ 6]    T1[ 0] → T2[ 7]
T1[14] → T2[ 8]    T1[ 8] → T2[ 9]    T1[12] → T2[10]    T1[ 6] → T2[11]
T1[ 7] → T2[12]    T1[ 1] → T2[13]    T1[ 5] → T2[14]    T1[13] → T2[15]
```

也就是 windowed orchestration 把 per-page sub-slice view (`paged_y.view([m0, 0], [16, 256])` / `paged_y.view([src_row, 0], [16, 256])`) 喂进 `register_task_outputs` 之后, [`pto_tensormap.h`](../../src/a5/runtime/tensormap_and_ringbuffer/runtime/pto_tensormap.h) 里 L1 byte-range intersection 已经能用 `start_offset / extent` 把 `[m0*1024, (m0+1)*1024)` 与 `[src_row*1024, (src_row+1)*1024)` 直接 disjoint, **运行期实测就是最小的 16 条 producer→consumer 边**, 不存在 16×16 退化。

> swimlane 里出现的 32 条 flow event (16 条边各 2 份) 来自 [`pto_dep_compute.h`](../../src/a5/runtime/tensormap_and_ringbuffer/runtime/pto_dep_compute.h) 的 Step A (creator retention) + Step B (tensormap lookup) 命中同一个 producer, 同一对 (prod, cons) 在 swimlane 里画了两条流, 属于多渠道叠加, 逻辑边只有 16 条。

---

## 5. 文件索引

- DSL 源: `paged_consumer_block_table_pypto_syntax.py`
- 生成的 orchestrator: `orchestration/paged_consumer_block_table.cpp` (`get_tensor_data<int32_t>(ext_block_table, ...)` 在第 60 行)
- AIC kernel: `kernels/aic/paged_proj.cpp` (函数 `paged_proj__windowed`, 签名 `(x_page, w1, paged_y_view, m0)`)
- AIV kernel: `kernels/aiv/paged_rmsnorm.cpp` (函数 `paged_rmsnorm__windowed`, 签名 `(gamma, y_src, out_view, out_m0)`)
- 运行期驱动: `test_paged_consumer_block_table.py`
- 本次 swimlane: `merged_swimlane.json`
