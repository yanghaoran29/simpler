# A5 双 Die 任务局部性最终方案：与 Main 对比

## 方案

本版本在连续物理 cluster 分核的基础上，为任务显式标记 `GLOBAL`、`DIE0` 或 `DIE1`：

- Scheduler 只管理连续的物理 cluster；S0/S1位于Die0，S2/S3位于Die1。
- AIC、AIV和MIX分别维护GLOBAL、DIE0、DIE1 ready queue，Scheduler优先消费本Die任务。
- `DIE_AFFINE`与`AUTO_DIE_AFFINE` scope分别支持显式依赖和TensorMap自动依赖，并让一条局部依赖链留在同一Die。
- completion/deferred-release传播任务domain；同Dieproducer/consumer可走Scheduler-local continuation，减少共享队列交接。
- BGEMM、PA unroll、manual PA与Batch PA按group、batch或chunk划分任务domain；Alternating根任务在两个Die间均分。
- Qwen3按MLP分片保持Gate/Up、SiLU和Down的Die一致；非sync-start多核SPMD拆为两个逻辑block区间并分到两个Die。24核sync-start attention保持GLOBAL，具有85路fanin的`dcr_xgamma`保持单个5核任务。

## Device latency

负数表示当前方案更快。非Qwen当前方案每个case运行100轮；Qwen运行5轮，按Device删除最快和最慢轮次后取中间3轮。Main使用卡1历史数据，基线提交为`3b578e30a2a9e2859d19908b2647393ffdecc543`。

| Case | Main (us) | 当前方案 (us) | 变化 |
|---|---:|---:|---:|
| alternating_matmul_add Case1 | 1,441.850 | 1,314.125 | **-8.858%** |
| benchmark_bgemm Case0 | 1,458.300 | 1,398.427 | **-4.106%** |
| paged_attention_unroll Case1 | 1,558.700 | 1,540.087 | -1.194% |
| paged_attention_unroll Case2 | 901.100 | 881.411 | **-2.185%** |
| paged_attention_unroll_manual_scope Case1 | 1,629.500 | 1,496.358 | **-8.171%** |
| paged_attention_unroll_manual_scope Case2 | 925.550 | 876.091 | **-5.344%** |
| batch_paged_attention Case1 | 7,434.600 | 5,444.691 | **-26.766%** |
| qwen3_14b_decode StressBatch16Seq3500 | 36,083.248 | 35,081.800 | **-2.775%** |
| **8 case几何平均** |  |  | **-7.787%** |

8个case均快于历史Main；其中7个改善超过2%，PA unroll Case1改善1.194%。

## 验证与泳道

- 8个case均完成benchmark；Qwen拆分版本通过完整golden。
- 当前最终代码在卡1对8个case各采集一次4级泳道，均为PASS。
- 每个case目录内的`merged_swimlane.json`可直接导入Perfetto；`name_map_*.json`提供kernel名称映射。
- 泳道采集会扰动细粒度任务并发，因此性能结论使用上面的关闭泳道多轮benchmark，泳道只用于验证任务推进和Die分布。

## 已知限制

- Main为同卡历史数据，不是本轮同窗口交叉A/B。
- 当前策略依赖样例选择合适的任务domain；错误或不完整的domain标注可能导致单Die空闲或增加跨Die依赖。
- 多ready queue、domain传播及local-continuation判断增加固定指令开销，极短且无依赖的任务对热路径布局较敏感。
- Qwen拆分SPMD会增加任务提交数与汇合依赖；`dcr_xgamma`不能直接拆分，否则会复制两份85路fanin并显著增加Orchestrator开销。
