# Batch Paged Attention：AUTO Die-affine chunk 实验

## 结论

方案可行。Batch PA Case1 的 16 个 chunk 以完整依赖单元在 Die0/Die1 间轮转，且仍保留
TensorMap 自动依赖推导。原先全部落在 Die0 的 1,024 个 UP 任务变为严格的 `512/512`；
QK、SF、PV、UP 四个阶段和 4,096 个任务总量都实现两个 Die 各一半。

卡1关闭泳道运行100轮，Device latency 为 `7,054.295 us`：

- 相对修改前同方案历史值 `7,200.981 us` 改善 `2.04%`；
- 相对卡1 Main历史值 `7,434.600 us` 改善 `5.12%`；
- p50为`7,046.493 us`，p95为`7,171.663 us`。

历史对照不是同一时间窗口的重跑，因此百分比适合作为本轮粗测结论；若决定合入，仍应补同窗口交叉测试。

## 算法

1. 新增`PTO2ScopeMode::AUTO_DIE_AFFINE`，Die affinity与依赖语义正交；
2. 进入最外层AUTO Die-affine scope时，以scope计数在Die0/Die1间轮转；
3. scope内所有可在单Die执行的root和live-fanin任务，在提交时绑定到该Die；
4. 内层普通AUTO scope继承外层Die，但仍通过TensorMap自动生成依赖；
5. 退出最外层scope后清除Die绑定，下一scope轮转到另一Die；
6. Batch PA只把每个外层chunk scope改为AUTO Die-affine，内层每个block的scope和算子代码不变。

Case1中`batch=256`、每chunk处理16个batch，所以共有16个chunk。偶数chunk固定Die0，
奇数chunk固定Die1；每个chunk中的64组QK/SF/PV/UP及其UP累积链都留在同一Die。

## 卡2泳道任务分布

| Kernel | Die0 | Die1 |
|---|---:|---:|
| QK | 512 | 512 |
| SF | 512 | 512 |
| PV | 512 | 512 |
| UP | 512 | 512 |
| 合计 | 2,048 | 2,048 |

Scheduler分布符合连续分核：S0/S1只执行Die0任务，S2/S3只执行Die1任务。

| Scheduler | QK | SF | PV | UP | Die |
|---|---:|---:|---:|---:|---:|
| S0 | 271 | 206 | 253 | 247 | 0 |
| S1 | 241 | 306 | 259 | 265 | 0 |
| S2 | 263 | 154 | 260 | 217 | 1 |
| S3 | 249 | 358 | 252 | 295 | 1 |

## AICore窗口（单张4级泳道）

泳道采集具有观察者效应，下表只分析任务结构，不能替代关闭泳道的100轮Device benchmark。

| 指标 | 修改前 | 新方案 | 变化 |
|---|---:|---:|---:|
| 首任务开始到末任务结束 | 7,054.969 us | 6,586.657 us | -6.64% |
| 90%任务完成时间 | 5,253.553 us | 5,219.627 us | -0.65% |
| 后10%收尾 | 1,801.416 us | 1,367.030 us | -24.11% |

收益主要来自尾部：前90%完成点变化较小，而两个Die共同承担UP后，最后10%的收尾缩短约24%。

## 验证

- `test_a5_orchestrator_fanin`：12/12通过；
- A5 host/AICPU/AICore runtime构建通过；
- 卡2带golden的Case1运行通过；
- 卡2单张4级泳道：4,096/4,096任务完整，四阶段均严格512/512；
- 卡1关闭泳道100轮通过。

文件：

- `card2_swimlane/merged_swimlane.json`：新方案4级泳道；
- `card2_swimlane/chip_swimlane_records.json`：原始泳道记录；
- `card1_100r.log`：卡1正式100轮日志。
