# A5 双 Die 任务局部性方案：与最新 Main 对比

## 方案

本版本在连续物理 cluster 分核的基础上，为每个任务显式标记 `GLOBAL`、`DIE0` 或 `DIE1`：

- Scheduler 管理连续的物理 cluster，S0/S1负责Die0，S2/S3负责Die1。
- AIC、AIV和MIX分别维护GLOBAL、DIE0、DIE1 ready queue，Scheduler优先消费本Die任务。
- `DIE_AFFINE`和`AUTO_DIE_AFFINE` scope分别支持显式依赖和TensorMap自动依赖，使局部依赖链尽量留在同一个Die。
- completion、async-wait和deferred-release传播任务domain；同Die producer/consumer可以通过Scheduler-local continuation续接。
- BGEMM、PA unroll、manual PA和Batch PA按group、batch或chunk划分任务domain；Alternating根任务在两个Die间均分。
- Qwen3让Gate/Up、SiLU和Down按MLP分片保持同Die。非sync-start多核SPMD拆成两个逻辑block区间并分到两个Die；24核sync-start attention保持GLOBAL；85路fanin的`dcr_xgamma`保持单个5核任务。

## Device latency

正数表示本PR更慢，负数表示本PR更快。Main与本PR均在同一张卡上重新测试，但为避免反复安装运行环境，执行顺序为完整Main测试后再完整测试本PR，并非逐case交叉执行。

非Qwen每个case运行100轮并使用全部100轮均值。Qwen运行5轮，分别按Device latency删除最快和最慢的一轮，再对中间3轮取平均。

| Case | Main (us) | 本PR (us) | 变化 |
|---|---:|---:|---:|
| alternating_matmul_add Case1 | 1,621.5 | 1,488.7 | **-8.19%** |
| benchmark_bgemm Case0 | 1,540.3 | 1,550.3 | +0.65% |
| paged_attention_unroll Case1 | 1,730.8 | 2,024.4 | **+16.96%** |
| paged_attention_unroll Case2 | 974.8 | 1,119.9 | **+14.89%** |
| paged_attention_unroll_manual_scope Case1 | 1,667.7 | 1,996.3 | **+19.70%** |
| paged_attention_unroll_manual_scope Case2 | 941.9 | 1,143.5 | **+21.40%** |
| batch_paged_attention Case1 | 6,814.1 | 6,685.5 | -1.89% |
| qwen3_14b_decode StressBatch16Seq3500 | 42,501.2 | 40,107.1 | **-5.63%** |
| **8 case几何平均** |  |  | **+6.62%** |

当前结果不能支持直接合入为默认策略：Alternating和Qwen有明确收益，Batch PA有小幅收益，BGEMM基本持平；但4个PA unroll/manual case出现14.89%到21.40%的显著回退，使整体几何平均回退6.62%。

## 分阶段时间

| Case | Effective变化 | Orchestrator变化 | Scheduler变化 |
|---|---:|---:|---:|
| alternating_matmul_add Case1 | -8.15% | -8.34% | -8.48% |
| benchmark_bgemm Case0 | +0.90% | -1.79% | +0.59% |
| paged_attention_unroll Case1 | +17.38% | -4.74% | +17.14% |
| paged_attention_unroll Case2 | +15.60% | -3.90% | +15.14% |
| paged_attention_unroll_manual_scope Case1 | +20.12% | +2.19% | +19.94% |
| paged_attention_unroll_manual_scope Case2 | +22.19% | -1.61% | +21.85% |
| batch_paged_attention Case1 | -1.85% | -3.94% | -1.91% |
| qwen3_14b_decode StressBatch16Seq3500 | -5.75% | +8.30% | -5.99% |

PA的主要回退位于Scheduler侧，而不是Orchestrator提交侧：4个PA case的Scheduler时间增长15.14%到21.85%，其中3个case的Orchestrator反而缩短。这说明rebase后当前domain队列、completion/release和局部续接路径与最新Main的Scheduler热路径组合后仍需继续优化。

Qwen表现相反：Scheduler缩短5.99%，覆盖了Orchestrator增加8.30%的代价，最终Device改善5.63%。

## 验证与泳道

- Main与本PR的8个benchmark均运行成功。
- 本PR在卡1为8个case各采集一次4级泳道，状态8/8 PASS，详见`status.tsv`。
- 每个case目录内的原始`merged_swimlane.json`可以直接导入Perfetto；`name_map_*.json`提供kernel名称映射。
- 泳道采集会扰动细粒度任务并发，因此上表使用关闭泳道的多轮benchmark；泳道仅用于检查任务推进和Die分布。

## 已知限制

- 测试采用Main整组后本PR整组的顺序，没有逐case交叉运行，仍可能包含时间窗口偏差。
- 当前策略依赖样例选择合适的任务domain；标注不完整可能导致单Die空闲，粒度过细则会增加提交和汇合成本。
- 多ready queue、domain传播及local-continuation判断增加固定指令开销，对Scheduler-bound的PA短任务链影响尤其明显。
- 全量CTest中120/122个目标通过；两个HBG graph-submit失败表现为异步时序断言不稳定。同一最新Main工作树曾通过这两个目标，但本PR重复运行仍失败，尚未完成根因收敛。
