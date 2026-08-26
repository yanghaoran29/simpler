# A5 Mode9当前完整状态：功能、样例与Main对比

## 版本功能

当前版本以连续物理cluster分核为基础：S0/S1只管理Die0，S2/S3只管理Die1。运行时为每种资源
维护GLOBAL、DIE0、DIE1三类ready queue；任务提交、completion、async-wait和deferred-release均传递
`TaskReadyDomain`，Scheduler消费本Die队列及GLOBAL队列，不跨Die窃取另一侧的本地任务。

在原Mode9之上，本版本新增两种显式scope locality：

- `DIE_AFFINE`：保留MANUAL依赖语义，把最外层scope内可单Die运行的任务固定到轮转选择的Die；
- `AUTO_DIE_AFFINE`：保留TensorMap自动依赖推导，同时把最外层scope作为一个Die locality单元。

root和live-fanin任务都会在提交时预绑定scope Die，避免最后完成fan-in的Scheduler覆盖producer/consumer
locality。DUMMY、sync-start、early-resolve或单Die容量不足的任务仍走GLOBAL。

## 当前采用locality的样例

| 样例 | locality单元 | 目的 |
|---|---|---|
| BGEMM Case0 | 每个GEMM+ADD group使用`AUTO_DIE_AFFINE` | 保留TensorMap ADD链并让同一C_view留在同一Die |
| PA manual scope | 原有最外层manual batch scope改为`DIE_AFFINE` | 保留显式依赖并固定整条batch链 |
| PA unroll | 每个`b_idx/q_idx` scope使用`AUTO_DIE_AFFINE` | 保留TensorMap依赖并固定QK/SF/PV/UP链 |
| Batch PA | 每个16-batch chunk使用`AUTO_DIE_AFFINE` | 让chunk内64组任务和UP累积链留在同一Die |
| Qwen3 | 继续使用普通`MANUAL` | 避免把整层任务限制到单Die造成并行度损失 |
| alternating | 不增加scope affinity | 独立任务继续使用Mode9根任务轮转 |

## 卡1完整benchmark

当前版本7个非Qwen case各运行100轮。Qwen运行5轮，按Device删除最快和最慢后取中间3轮，
同一组轮次用于所有指标。Main来自此前卡1的2x100轮历史数据；正数表示当前版本更慢。

| Case | Main Device | 当前Device | Device变化 | Effective变化 | Scheduler变化 |
|---|---:|---:|---:|---:|---:|
| alternating Case1 | 1,441.850 us | 1,615.180 us | **+12.02%** | +12.32% | +12.40% |
| BGEMM Case0 | 1,458.300 us | 1,419.827 us | **-2.64%** | -2.62% | -2.72% |
| PA unroll Case1 | 1,558.700 us | 1,735.035 us | **+11.31%** | +11.52% | +11.61% |
| PA unroll Case2 | 901.100 us | 987.058 us | **+9.54%** | +9.87% | +10.05% |
| PA manual Case1 | 1,629.500 us | 1,733.783 us | **+6.40%** | +6.49% | +6.53% |
| PA manual Case2 | 925.550 us | 1,011.739 us | **+9.31%** | +9.55% | +9.71% |
| Batch PA Case1 | 7,434.600 us | 7,066.457 us | **-4.95%** | -4.98% | -4.98% |
| Qwen3 | 36,083.248 us | 35,014.167 us | **-2.96%** | -2.96% | -2.96% |

7个非Qwen case的Device几何平均变化为`+5.66%`。旧Mode9对应值约为`+10.25%`，scope locality
修复了部分依赖链跨Die问题，但尚未抵消多队列路由和严格本Die消费的固定开销。

## AICore任务90%完成时间

该指标由Main和当前Mode9各一次卡2四级泳道计算。以第一个AICore任务的`start_time_us`为起点，
将全部AIC/AIV任务按`end_time_us`升序排列，取第`ceil(任务数 * 90%)`个任务的结束时间；表中
正数表示当前Mode9更慢。

| Case | AICore任务数 | Main完成90% | 当前Mode9完成90% | 变化 |
|---|---:|---:|---:|---:|
| alternating Case1 | 1,000 | 804.149 us | 670.445 us | **-16.63%** |
| BGEMM Case0 | 1,000 | 944.140 us | 963.411 us | **+2.04%** |
| PA unroll Case1 | 1,024 | 1,124.307 us | 1,099.019 us | **-2.25%** |
| PA unroll Case2 | 512 | 580.328 us | 526.421 us | **-9.29%** |
| PA manual Case1 | 1,024 | 1,254.948 us | 1,159.881 us | **-7.58%** |
| PA manual Case2 | 512 | 622.908 us | 600.419 us | **-3.61%** |
| Batch PA Case1 | 4,096 | 4,818.439 us | 5,219.627 us | **+8.33%** |
| Qwen3 | 19,208 | 34,263.859 us | 33,104.489 us | **-3.38%** |

8个样例中有6个在前90% AICore任务完成时间上更快。alternating的该指标改善`16.63%`，但
完整Device benchmark回退`12.02%`，说明其主要回退不在前90%的AICore执行窗口。PA系列也普遍
表现为前90%改善而完整Device latency回退，符合泳道中观察到的最后10%任务和调度尾部较长。
BGEMM和Batch PA则分别在该单次泳道指标上回退`2.04%`和`8.33%`。

由于逐任务泳道存在观察者效应，该表用于比较任务推进结构，不能代替前述关闭泳道的多轮Device
benchmark。机器可读结果见`aicore_completion_p90.csv`。

## 相对旧Mode9的变化

| Case | 当前Device相对旧Mode9 |
|---|---:|
| alternating Case1 | +1.56% |
| BGEMM Case0 | **-13.57%** |
| PA unroll Case1 | +0.22% |
| PA unroll Case2 | **-7.34%** |
| PA manual Case1 | -0.56% |
| PA manual Case2 | **-5.40%** |
| Batch PA Case1 | **-3.17%** |
| Qwen3 | +0.07% |

BGEMM、PA Case2、manual PA Case2和Batch PA均从依赖链locality获得明显改善；Qwen不使用新scope模式，
性能保持旧Mode9水平。

## 泳道结构验证

- BGEMM：ADD从Die0/Die1=`500/0`修复为`250/250`；
- PA unroll Case2：QK/SF/PV/UP全部为`64/64`，总任务`256/256`；
- PA manual Case2：64条batch链各32条落在两个Die，四类kernel均`64/64`；
- Batch PA：QK/SF/PV/UP全部为`512/512`，总任务`2048/2048`；
- PA unroll Case2修改后两个Die最后任务结束差从`401.747 us`降至`6.015 us`；
- Batch PA后10% AICore收尾由`1,801.416 us`缩短至`1,367.030 us`。

## 现有缺陷

1. 相对历史Main仍有5个小case回退超过2%，BGEMM、Batch PA和Qwen获得总体收益；
2. 多队列选择、ready-domain原子状态以及严格本Die消费增加细粒度任务的固定调度成本；
3. 不跨Die stealing会在locality标注不足或负载天然不均时造成一侧空闲；
4. locality修复后仍会在scope尾部形成百项级release批次，PA Case2的前90%已快于Main，但尾部仍更长；
5. 当前策略由样例显式选择，尚无运行时自动判断“局部性收益”与“双Die并行度损失”的机制；
6. Main为同卡历史数据而非同窗口交叉A/B；四级泳道有观察者效应，只用于结构验证。

## 数据索引

- 本目录`*_summary.log`：逐轮解析表；
- 本目录`*.log`：原始benchmark日志；
- 本目录`summary.csv`：机器可读Main对比；
- 本目录`aicore_completion_p90.csv`：Main与当前Mode9的AICore任务90%完成时间；
- `outputs/bgemm-auto-die-affine-card2-20260826/`：正确保留TensorMap依赖后的BGEMM泳道；
- `outputs/mode9-scope-affinity-case2-card2-20260826/`：manual PA Case2泳道；
- `outputs/pa-unroll-case2-balanced-auto-die-affine-20260826/`：AUTO PA Case2泳道；
- `outputs/batch-pa-auto-die-affine-20260826/`：Batch PA泳道和100轮数据。

## 验证环境

- 设备：Ascend950PR_9579，benchmark固定卡1，泳道固定卡2；
- CANN 9.2.0；
- 单卡`task-submit`加锁，不执行RTT preflight；
- 当前全量benchmark 8/8通过；
- BGEMM、PA unroll Case2、manual PA Case2、Batch PA各1轮golden校验通过；
- `test_a5_orchestrator_fanin` 12/12通过；
- A5 host/AICPU/AICore runtime构建通过。
