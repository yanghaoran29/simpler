# PR12：Main-style release + Qwen3 Die-locality

报告时间戳：`2026-08-27 20:22:44 +08:00`

## 更新目的

本报告合并原 `20260827_200653` release 对齐实验与本轮 Qwen locality 实验，完整记录从
带 16-task deferred-release 微批的 PR 版本到 PR12 当前版本的变化：先移除难以解释的
release 微批，再恢复
[`20260827_193536_qwen3-die-locality.md`](20260827_193536_qwen3-die-locality.md)
最终版本的 Qwen3 Die 归属规则：

1. deferred release 从“每轮最多 16 个 task并批内聚合 producer”改为触发后逐 task 排空，
   对齐 Main 的 drain 语义；
2. OutProj producer 与对应的 `residual_rms_cast` consumer 固定在同一 Die；
3. Q/K/V seed 按 projection half 拆分并固定到同一 Die，使一个 half 不必等待另一 Die 的 seed。

合并后的 Before 是 `200653` 文档原先使用的 Before，即带 16-task 微批的 `629c7cde`；
After 是 PR12 当前 `9dc809ec`。7 个非 Qwen After 使用 release 对齐快照 `4629184d` 的
100 轮结果，Qwen After 使用当前版本新跑的 5 轮结果。8 份最终代码对应的泳道统一收敛到
本报告 artifact 目录：7 份复用 release 对齐快照，Qwen 重新采集。

## 代码身份

| 数据集 | 身份 | 说明 |
| --- | --- | --- |
| Main | `80dd3cd96e568e6f3ded9c11b68e7c267a31343d` | 历史 Main 基线 |
| Before | `629c7cdef4e83b039fe83de1ec87d948ee4bb185` | 带 16-task release 微批的 PR 版本 |
| 中间快照 | `4629184d23b51beefe6cff098da4664bf4444087` | 移除 release 微批、尚未恢复 Qwen 专项 locality |
| After 源码 | `9dc809ec84eb27e1fa0f17283872b05387af58fc` | 性能测量对应的源码提交；随后仅 amend 本报告、skill 与实验 artifact |
| 历史 Qwen 参考 | `cd5eefefc1b4994dd3d7670a120e6abed2559aea` | `20260827_193536` 最终实验版本 |

- PR：[yanghaoran29/simpler#12](https://github.com/yanghaoran29/simpler/pull/12)，base 为 `main`。
- After 分支：`perf/a5-main-style-deferred-release-20260827`。
- PR 发布 head 是包含本报告和 artifact 的 amended commit；其三个 Qwen 源文件与上述
  `9dc809ec` 完全相同，因此性能代码身份没有变化。
- release 对齐提交相对 Before：`4 files changed, 57 insertions, 229 deletions`；Qwen locality
  提交：`3 files changed, 107 insertions, 111 deletions`。
- 提交前，本次 3 个 Qwen 文件与历史 `cd5eefef` 对应文件逐字一致；相对中间快照的稳定
  Qwen-only patch-id 为 `03f981735d0ba6e6a6f4cc2bbb21a24c5385d3cc`。
- PTO-ISA pin：`be5ccb765a4ce5d14ca5da8b0e2f182d7f003369`，构建 metadata 的实际
  checkout 与 pin 一致。

## 本次 Qwen3 改动

### OutProj 与 residual/RMS consumer 同 Die

- 新增 `out_proj_shard_domain(layer, shard)`。
- 50 个 OutProj block 按每 10 个 block 一个 hidden shard，共 5 个 shard；shard 奇偶映射到
  两个 Die，相邻 layer 交换物理 Die，避免持续偏向同一侧。
- direct OutProj block 0～25 按 `out_idx / 10` 确定 domain。
- 原来的 block 26～49 是两个 12-block SPMD half，会跨越 10-block consumer shard 边界；
  现在按 `4 + 10 + 10` 拆为 block `[26,30)`、`[30,40)`、`[40,50)` 三次提交。
- 5 个 `residual_rms_cast` consumer 使用完全相同的 shard-domain 函数，因此对应 producer
  完成后可沿 Scheduler-local continuation 在同一 Die 继续执行。

### Q/K/V seed 与 projection half 同 Die

- Q seed 从一个清零 5120 列的任务拆成两个任务，各清零 2560 列，即 5 个 512-wide tile。
- K/V seed 从一个同时清零完整 1024 列的任务拆成两个任务，各处理 512 列；K/V 同一 half
  仍共用一个 seed task。
- 每个 seed task 使用与其 Q/K/V projection half 相同的 `spmd_half_domain(layer, half)`。
- Q projection 的两个 25-core half 分别只依赖自己的 Q seed；K/V 的两个 5-core half也只
  依赖自己的 KV seed，不再形成跨 Die seed barrier。
- `q_seed.cpp`、`kv_seed.cpp` 增加 `half_idx` 标量，kernel 仅清零相应 half。

### 有意保持 GLOBAL 的 Qwen 节点

- 24-block、要求 sync-start 的 CANN paged-attention 保持一个 GLOBAL task；
- `mlp_out_seed` 保持 GLOBAL，因为它同时初始化多个跨 Die accumulator；
- 5-core `dcr_xgamma` 保持一个 GLOBAL task，避免复制完整 85-edge fanin 和增加 join task；
- dummy join 始终放在 GLOBAL ring。

## PR12 当前方案与 Main 的详细区别

下面比较的是 Main `80dd3cd9` 与本报告 After `9dc809ec` 的实际代码，而不是只比较本次
3 个 Qwen 文件。PR12 的核心是显式双 Die locality；deferred release 的 drain 语义则已
对齐 Main。

### 总体数据流

```text
Main
task -> 4 个 scope-depth ring -> 每种 shape 一个共享 ready queue
     -> Scheduler 按 round-robin cluster 拾取 -> completion/release -> 共享 reclaim mask

PR12
task + GLOBAL/DIE0/DIE1 -> 3 domain × 4 depth = 12 个物理 ring
     -> sync queue / Scheduler-local continuation / 分 domain ready queue
     -> 固定 Die 的 Scheduler + 连续物理 cluster -> completion/release -> 分 domain reclaim mask
```

### Runtime 与调度结构

| 维度 | Main | PR12 当前方案 | 直接影响 |
| --- | --- | --- | --- |
| task placement | task 没有显式 Die 属性，全部走公共调度域 | `TaskDomain::{GLOBAL,DIE0,DIE1}`，默认 GLOBAL，编排代码显式设置 | locality 是图构建契约，不根据完成顺序猜测 |
| 物理 ring | 4 个，仅按 scope depth 选择 | 12 个，`domain × 4 depth`；GLOBAL 0～3、DIE0 4～7、DIE1 8～11 | 固定 Die 的任务有独立 task/heap/dep-pool watermark |
| ring id | `min(scope_depth, 3)` | `domain * 4 + min(scope_depth, 3)` | slot 的 ring id 同时编码 domain，无需额外发布 placement 字段 |
| ready queue | AIC/AIV/MIX 各一个共享 queue | 每种 shape 各有 GLOBAL、DIE0、DIE1 三个 queue；sync queue 仍独立 | 固定 Die task 不会被另一 Die 的 Scheduler 抢走 |
| continuation | newly-ready consumer 回共享 queue | 每个 Scheduler、每种 shape 有 256-entry local continuation queue | 同 Die依赖链优先留在完成 producer 的 Scheduler；满时回退共享 domain queue，不影响正确性 |
| Scheduler core ownership | cluster 按 Scheduler id round-robin 分配 | 按规范化物理 cluster ordinal 划分平衡的连续区间 | 每个 Scheduler 只拥有一个 Die 上的 cluster |
| 普通调度优先级 | sync queue 后直接处理共享普通 queue | sync queue → local continuation → 本 Die queue/GLOBAL queue；两者同时有任务时轮换先后 | 优先延续依赖链，同时避免 GLOBAL 或 Die-local 饥饿 |
| completion routing | fanout ready 后进入共享 queue | completion、dummy、async-wait 都携带释放线程/domain；同 Die consumer 可进 local continuation | 减少跨 Scheduler/跨 Die 重新排队 |
| reclaim 请求 | 一个全局 pending/request/ack mask | GLOBAL、DIE0、DIE1 各有 cache-line 隔离的 pending/request/ack mask | 每个 Scheduler 服务自己的 Die；thread 0 额外服务 GLOBAL |
| scope backstop | 查当前 scope-depth ring 的 oldest-open task | 为 12 个 ring 分别维护 `oldest_open_tasks_by_ring` 和 scope-owned mask | 同一 scope 内多 domain 分配时仍能正确判断 ring/dep-pool backpressure |
| routing 激活 | 无此状态 | 首个非 GLOBAL task 提交时一次性打开 `die_routing_active` | 没有显式 placement 的图仍保留 GLOBAL 路径 |
| public ring 配置 | 4 个 depth 值配置 4 个 ring | public ABI 仍是 4 个 depth 值，每个值广播到三个 domain ring | 配置接口不扩成 12 项，但默认物理资源总量按 3 个 domain 复制 |
| 默认 ring 资源 | 4 × 16384 task、4 × 256 MB heap、4 × 16384 dep entries | 12 份同样的 per-ring 默认值，总 heap 从 1 GiB 增至 3 GiB | 用更多共享内存隔离两个 Die 的资源与 watermark |

`CHIP_LOCAL_CONTINUATION_QUEUE_SIZE=256` 只是 locality cache 容量：push 失败会回退到普通
domain queue。它不是 fanin、graph width 或 forward-progress 上限，也不同于已经移除的
deferred-release 16-task 微批。

### Deferred release：哪些相同，哪些仍不同

| 项目 | Main | PR12 当前方案 |
| --- | --- | --- |
| completed backlog 容量 | 256 | 256 |
| backlog 满时 | 逐 task release，排空当前 backlog | 同 Main |
| Scheduler idle 时 | 逐 task release，排空当前 backlog | 同 Main |
| dummy/async-wait/退出路径 | 逐 task release，必要时排空 | 同 Main |
| 单轮最多 16 个 task | 无 | 无；`DEFERRED_RELEASE_BATCH=16` 已删除 |
| 批内 producer 聚合 | 无 | 无；64-entry 聚合表和 batch-local ring mask 已删除 |
| release 后 consumer 去向 | 共享 ready queue | 根据 domain 进入 local continuation 或对应 domain queue |

因此“触发 drain 后如何排空 backlog”已经与 Main 对齐；但 `on_task_release()` 解锁 consumer
之后的路由仍属于双 Die 方案。代码中的 `PUBLISH_INTERVAL_K=16` 仍存在，它只控制 ring
watermark 向共享内存发布的间隔，与“一次 release 多少 task”无关。

### Workload 编排差异

| Workload | Main | PR12 当前方案 |
| --- | --- | --- |
| Alternating | 所有 matmul/add task 为 GLOBAL | 500 个独立 group 前半固定 DIE0、后半固定 DIE1；同组 AIC/AIV 在同一 Die |
| BGEMM | 所有 reduction chain 为 GLOBAL | 按 `group_idx` 奇偶在 DIE0/DIE1 交替；GEMM 与随后 tile-add 同 Die |
| PA unroll | QK→Softmax→PV→Update chain 为 GLOBAL | 按 `(batch, q-loop)` chain 奇偶拆 Die，整条 chain 保持同一 domain |
| Manual-scope PA | 同上，GLOBAL | 每个 manual scope 对应一条固定 Die chain |
| Batch PA | 每个 chunk 内任务为 GLOBAL | 按 `(q_idx, chunk_idx)` chain 奇偶拆 Die，嵌套 scope 不隐式继承，任务逐个显式标记 |
| Qwen attention projection | 单个多核 SPMD task 可跨两个 Die | 非 sync-start SPMD 按两个逻辑 block 区间拆开，两个 half 固定相反 Die，相邻 layer 交换 |
| Qwen MLP | Gate/Up、SiLU、Down 通过公共队列调度 | 每个 MLP shard 从 Gate/Up 经 SiLU 到 Down 使用同一 domain |
| Qwen OutProj/cast | producer/consumer 没有 shard-level Die 契约 | 10-block hidden shard 与对应 residual/RMS consumer 同 Die |
| Qwen Q/K/V seed | 一个完整 seed task，两个 projection half 都等待它 | seed 按 half 拆分，每个 projection half 只等待本 Die seed |
| Qwen sync/fanin 汇合 | GLOBAL | CANN PA、`mlp_out_seed`、`dcr_xgamma` 仍刻意保持 GLOBAL |

为了让拆开的 SPMD half 保持 Main 中原有的逻辑 block 编号，Qwen 的 15 个 AIC kernel 和
`x_gamma0` AIV kernel 从额外标量读取 logical block offset，并与 runtime `get_block_idx()`
相加。拆分改变的是 task 边界和调度 domain，不改变每个逻辑 block 计算的输出区域。

### API、内存布局和测试覆盖

- TMR `CoreTaskArgs`/`ArgWithDeps` 新增 `set_task_domain()` 与 getter；`clear()` 恢复 GLOBAL。
- HBG 类型和 scope API 增加同名 domain 字段以保持接口/ABI 表达能力；实际 Die placement
  由 onboard TMR runtime 消费。
- shared-memory layout 从 3 个普通 ready queue 扩为 9 个 domain queue，并增加每
  Scheduler 的 continuation queue；初始化、reset、pointer wiring、destroy 都覆盖这些结构。
- 单测增加 domain/ring 映射、queue routing、连续 cluster ownership、跨 domain fanin、分域
  reclaim mask、scope oldest task 和 SPMD wiring 等覆盖。
- PR diff 中的泳道 JSON/汇总文件是实验 artifact，不参与 runtime 执行语义。

## 数据来源与实验方法

Main 与 Before 均沿用 `2026-08-27` 的历史干净构建，没有重新运行：

- [Main/原 PR 历史汇总](../../../../simpler-08191/simpler3-mode9-swimlane-20260825/outputs/clean-latest-main-vs-rebased-pr-card1-20260827/summary.csv)
- [Main 原始目录](../../../../simpler-08191/simpler3-mode9-swimlane-20260825/outputs/clean-latest-main-vs-rebased-pr-card1-20260827/main/)
- [Before 原始目录](../../../../simpler-08191/simpler3-mode9-swimlane-20260825/outputs/clean-latest-main-vs-rebased-pr-card1-20260827/rebased_pr/)

After 按两个代码阶段组合，但每个数值都来自对应代码的已保存原始实验：

- A5，Ascend950PR_9579 卡 1，CANN 9.2.0，TMR level 2；
- 7 个非 Qwen After 来自 `4629184d` 的 release 对齐实验，各 100 轮算术平均；
- Qwen After 在最终 `9dc809ec` 上新跑 5 轮，按 Device 删除最快和最慢轮，再对保留的
  相同 3 轮计算全部指标；
- [After 非 Qwen 原始性能与泳道](../pr10-main-release-card1-20260827/)；
- [After Qwen 原始性能](20260827_202244_qwen3-die-locality-main-style-release/performance/qwen3_stress_batch16_seq3500_performance.log)；
- 在本轮范围明确前，7 个非 Qwen 样例曾完成一次额外 100 轮运行，但按后续指示不纳入正式
  对比；本次 [performance/](20260827_202244_qwen3-die-locality-main-style-release/performance/)
  中对应文件已替换为 `4629184d` 的正式历史日志和 summary，避免混用；
- Qwen 单独新采 1 轮 level-4 泳道；其余 7 份最终代码适用的泳道从 `4629184d` artifact
  复制到本次目录。

## Main / Before / After 三方结果

单位均为 `us`。负数表示后一个版本更快。完整长表见
[`summary.csv`](20260827_202244_qwen3-die-locality-main-style-release/summary.csv)。

### Device

| Case | Main | Before | After | Before/Main | After/Main | After/Before |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| alternating_matmul_add Case1 | 1,495.2 | 1,332.9 | 1,338.1 | -10.85% | -10.51% | +0.39% |
| benchmark_bgemm Case0 | 1,567.6 | 1,418.7 | 1,386.3 | -9.50% | -11.57% | **-2.28%** |
| paged_attention_unroll Case1 | 1,876.0 | 1,512.5 | 1,543.7 | -19.38% | -17.71% | **+2.06%** |
| paged_attention_unroll Case2 | 1,075.5 | 871.1 | 886.2 | -19.01% | -17.60% | +1.73% |
| paged_attention_unroll_manual_scope Case1 | 1,898.5 | 1,495.5 | 1,515.3 | -21.23% | -20.18% | +1.32% |
| paged_attention_unroll_manual_scope Case2 | 1,066.6 | 869.7 | 880.4 | -18.46% | -17.46% | +1.23% |
| batch_paged_attention Case1 | 6,974.1 | 5,326.3 | 5,484.5 | -23.63% | -21.36% | **+2.97%** |
| qwen3_14b_decode StressBatch16Seq3500 | 36,113.7 | 35,322.9 | **34,865.2** | -2.19% | **-3.46%** | **-1.30%** |
| **8-case 几何平均** | — | — | — | **-15.79%** | **-15.16%** | **+0.75%** |

### Effective

| Case | Main | Before | After | Before/Main | After/Main | After/Before |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| alternating_matmul_add Case1 | 1,462.1 | 1,298.9 | 1,303.6 | -11.16% | -10.84% | +0.36% |
| benchmark_bgemm Case0 | 1,535.7 | 1,385.6 | 1,352.4 | -9.77% | -11.94% | **-2.40%** |
| paged_attention_unroll Case1 | 1,844.9 | 1,480.7 | 1,509.3 | -19.74% | -18.19% | +1.93% |
| paged_attention_unroll Case2 | 1,044.4 | 839.6 | 851.7 | -19.61% | -18.45% | +1.44% |
| paged_attention_unroll_manual_scope Case1 | 1,867.7 | 1,463.7 | 1,482.6 | -21.63% | -20.62% | +1.29% |
| paged_attention_unroll_manual_scope Case2 | 1,037.0 | 837.2 | 848.4 | -19.27% | -18.19% | +1.34% |
| batch_paged_attention Case1 | 6,940.8 | 5,289.3 | 5,446.4 | -23.79% | -21.53% | **+2.97%** |
| qwen3_14b_decode StressBatch16Seq3500 | 36,077.8 | 35,286.1 | **34,827.4** | -2.19% | **-3.47%** | **-1.30%** |
| **8-case 几何平均** | — | — | — | **-16.17%** | **-15.59%** | **+0.69%** |

### Orchestrator

| Case | Main | Before | After | Before/Main | After/Main | After/Before |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| alternating_matmul_add Case1 | 1,457.3 | 1,294.9 | 1,299.5 | -11.14% | -10.83% | +0.36% |
| benchmark_bgemm Case0 | 1,437.9 | 1,302.2 | 1,265.7 | -9.44% | -11.98% | **-2.80%** |
| paged_attention_unroll Case1 | 1,410.3 | 1,254.7 | 1,288.3 | -11.03% | -8.65% | +2.68% |
| paged_attention_unroll Case2 | 715.1 | 605.9 | 623.2 | -15.27% | -12.85% | +2.86% |
| paged_attention_unroll_manual_scope Case1 | 1,279.5 | 1,154.6 | 1,171.1 | -9.76% | -8.47% | +1.43% |
| paged_attention_unroll_manual_scope Case2 | 604.6 | 527.3 | 535.3 | -12.79% | -11.46% | +1.52% |
| batch_paged_attention Case1 | 5,921.2 | 3,950.4 | 4,084.8 | -33.28% | -31.01% | **+3.40%** |
| qwen3_14b_decode StressBatch16Seq3500 | 10,269.8 | 10,510.8 | **10,657.8** | +2.35% | **+3.78%** | **+1.40%** |
| **8-case 几何平均** | — | — | — | **-13.07%** | **-11.91%** | **+1.34%** |

### Scheduler

| Case | Main | Before | After | Before/Main | After/Main | After/Before |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| alternating_matmul_add Case1 | 1,455.1 | 1,295.7 | 1,300.3 | -10.95% | -10.64% | +0.36% |
| benchmark_bgemm Case0 | 1,528.2 | 1,382.6 | 1,349.7 | -9.53% | -11.68% | **-2.38%** |
| paged_attention_unroll Case1 | 1,836.2 | 1,477.9 | 1,506.5 | -19.51% | -17.96% | +1.94% |
| paged_attention_unroll Case2 | 1,037.1 | 836.8 | 849.0 | -19.31% | -18.14% | +1.46% |
| paged_attention_unroll_manual_scope Case1 | 1,860.0 | 1,460.8 | 1,479.7 | -21.46% | -20.45% | +1.29% |
| paged_attention_unroll_manual_scope Case2 | 1,028.6 | 834.6 | 845.4 | -18.86% | -17.81% | +1.29% |
| batch_paged_attention Case1 | 6,933.7 | 5,286.7 | 5,444.0 | -23.75% | -21.48% | **+2.98%** |
| qwen3_14b_decode StressBatch16Seq3500 | 36,074.6 | 35,285.4 | **34,826.5** | -2.19% | **-3.46%** | **-1.30%** |
| **8-case 几何平均** | — | — | — | **-15.97%** | **-15.39%** | **+0.69%** |

### Host

| Case | Main | Before | After | Before/Main | After/Main | After/Before |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| alternating_matmul_add Case1 | 100,861.0 | 98,119.0 | 114,255.5 | -2.72% | +13.28% | **+16.45%** |
| benchmark_bgemm Case0 | 48,216.7 | 50,959.2 | 50,401.2 | +5.69% | +4.53% | -1.09% |
| paged_attention_unroll Case1 | 165,196.7 | 154,381.6 | 151,664.7 | -6.55% | -8.19% | -1.76% |
| paged_attention_unroll Case2 | 38,633.5 | 36,887.5 | 38,134.4 | -4.52% | -1.29% | +3.38% |
| paged_attention_unroll_manual_scope Case1 | 162,932.6 | 157,458.3 | 123,668.6 | -3.36% | -24.10% | **-21.46%** |
| paged_attention_unroll_manual_scope Case2 | 38,798.5 | 38,042.8 | 34,039.3 | -1.95% | -12.27% | **-10.52%** |
| batch_paged_attention Case1 | 168,397.1 | 156,578.4 | 127,494.4 | -7.02% | -24.29% | **-18.57%** |
| qwen3_14b_decode StressBatch16Seq3500 | 5,097,471.1 | 5,667,545.4 | **6,954,739.0** | +11.18% | **+36.44%** | **+22.71%** |
| **8-case 几何平均** | — | — | — | **-1.33%** | **-3.72%** | **-2.43%** |

## Qwen 五轮筛选

| 数据集 | 五轮 Device 原始值（r0～r4） | 删除 | 保留 |
| --- | --- | --- | --- |
| Main | `36830.5, 35627.6, 35793.9, 36244.5, 36302.7` | r1 最快、r0 最慢 | r2、r3、r4 |
| Before | `35787.7, 34867.8, 35413.7, 35271.3, 35283.7` | r1 最快、r0 最慢 | r2、r3、r4 |
| After | `35192.2, 34800.9, 35138.8, 34655.8, 34594.2` | r4 最快、r0 最慢 | r1、r2、r3 |

After 保留轮次对应均值：Host `6,954,739.0 us`、Device `34,865.2 us`、Effective
`34,827.4 us`、Orchestrator `10,657.8 us`、Scheduler `34,826.5 us`。

## 效果分析

- 合并两个代码阶段后，相对带 16-task 微批的 Before，Qwen Device、Effective、Scheduler
  均改善 `1.30%`，Orchestrator 回退 `1.40%`。全部仍落在 ±2% 噪声判断区间，方向支持
  locality 调整，但单次五轮不足以宣称稳定显著提升。
- 相对 Main，Qwen Device/Effective/Scheduler 分别改善 `3.46%`、`3.47%`、`3.46%`。
- Host 相对 Before 回退 `22.71%`，必须保留为风险项；但 Host 五轮为 5.75～8.83 秒，
  波动远大于 Device 且方向不同，不把它归因于 device-side locality。Host 若是验收指标，需
  单独进行交叉轮次实验。
- 历史 `cd5eefef` 在带 16-task release 微批的 runtime 上测得 Qwen Device `34,576.6 us`；
  本轮相同 Qwen 三文件配合 Main-style release 为 `34,865.2 us`，跨时间窗口且 runtime 不同，
  只能作为参考，不能把 `+0.83%` 直接归因于 release。
- 相对 Before，8-case Device 几何平均回退 `0.75%`：release 微批移除在大多数样例上带来
  小幅代价，BGEMM 改善 `2.28%`，Qwen locality 将 Qwen 拉回 `-1.30%`，最大风险仍是
  Batch PA `+2.97%`。
- 相对 Main，After 的 Device 几何平均仍改善 `15.16%`，Effective 改善 `15.59%`，
  Scheduler 改善 `15.39%`。

结论：合并方案以 `0.75%` 的 Device 几何平均代价换取 Main-style release 的可解释性，同时
保留 Qwen 的确定 Die locality。Qwen 的直接方向为正，但证据强度仍是噪声区间内的小幅结果。

## 验证

- 本次 3 个 Qwen 文件在修改后与历史 `cd5eefef` 对应文件逐字一致；
- editable package 重新构建成功，PTO-ISA metadata 与 pin 一致；
- Qwen golden PASS，5/5 性能轮完成；
- Qwen level-4 泳道 PASS，raw/Perfetto 均非空；
- `git diff --check` 通过；
- 在范围明确前顺带执行的其余 7 个 golden 也全部 PASS，但其性能数据不进入正式 After。

## 泳道

7 个非 Qwen 泳道从 release 对齐快照 `4629184d` 复制；Qwen 为最终代码新采。全部 raw JSON 的
`chip_swimlane_level=4`，AICore/AICPU task 非空，Perfetto trace 非空。

| Case | 来源 | AICore tasks | AICPU tasks | Perfetto events | 泳道图 | Artifact |
| --- | --- | ---: | ---: | ---: | --- | --- |
| alternating_matmul_add Case1 | 复制 `4629184d` | 1,000 | 1,000 | 13,757 | [Perfetto](20260827_202244_qwen3-die-locality-main-style-release/alternating_matmul_add_case1/merged_swimlane.json) | [目录](20260827_202244_qwen3-die-locality-main-style-release/alternating_matmul_add_case1/) |
| benchmark_bgemm Case0 | 复制 `4629184d` | 1,000 | 1,000 | 14,280 | [Perfetto](20260827_202244_qwen3-die-locality-main-style-release/benchmark_bgemm_case0/merged_swimlane.json) | [目录](20260827_202244_qwen3-die-locality-main-style-release/benchmark_bgemm_case0/) |
| paged_attention_unroll Case1 | 复制 `4629184d` | 1,024 | 1,024 | 15,215 | [Perfetto](20260827_202244_qwen3-die-locality-main-style-release/paged_attention_unroll_case1/merged_swimlane.json) | [目录](20260827_202244_qwen3-die-locality-main-style-release/paged_attention_unroll_case1/) |
| paged_attention_unroll Case2 | 复制 `4629184d` | 512 | 512 | 7,590 | [Perfetto](20260827_202244_qwen3-die-locality-main-style-release/paged_attention_unroll_case2/merged_swimlane.json) | [目录](20260827_202244_qwen3-die-locality-main-style-release/paged_attention_unroll_case2/) |
| paged_attention_unroll_manual_scope Case1 | 复制 `4629184d` | 1,024 | 1,024 | 15,126 | [Perfetto](20260827_202244_qwen3-die-locality-main-style-release/paged_attention_unroll_manual_scope_case1/merged_swimlane.json) | [目录](20260827_202244_qwen3-die-locality-main-style-release/paged_attention_unroll_manual_scope_case1/) |
| paged_attention_unroll_manual_scope Case2 | 复制 `4629184d` | 512 | 512 | 7,523 | [Perfetto](20260827_202244_qwen3-die-locality-main-style-release/paged_attention_unroll_manual_scope_case2/merged_swimlane.json) | [目录](20260827_202244_qwen3-die-locality-main-style-release/paged_attention_unroll_manual_scope_case2/) |
| batch_paged_attention Case1 | 复制 `4629184d` | 4,096 | 4,096 | 58,041 | [Perfetto](20260827_202244_qwen3-die-locality-main-style-release/batch_paged_attention_case1/merged_swimlane.json) | [目录](20260827_202244_qwen3-die-locality-main-style-release/batch_paged_attention_case1/) |
| qwen3_14b_decode StressBatch16Seq3500 | 本次新采 | 19,288 | 19,288 | 244,012 | [Perfetto](20260827_202244_qwen3-die-locality-main-style-release/qwen3_stress_batch16_seq3500/merged_swimlane.json) | [目录](20260827_202244_qwen3-die-locality-main-style-release/qwen3_stress_batch16_seq3500/) |

复制/新采来源逐项记录在
[`swimlane_status.tsv`](20260827_202244_qwen3-die-locality-main-style-release/swimlane_status.tsv)。

## 最终结论

PR12 当前版本同时具备两点：deferred release 使用 Main 可解释的“触发即排空”语义；Qwen
恢复经过单独实验的明确 Die-local 图划分。相对带 16-task 微批的 Before，8-case Device
几何平均回退 `0.75%`，但相对 Main 仍改善 `15.16%`；Qwen Device 从 `35,322.9 us` 降到
`34,865.2 us`，相对 Main 改善 `3.46%`。代码已经提交并推送到直接指向 `main` 的 PR12。
