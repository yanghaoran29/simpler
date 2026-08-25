# A5 Mode9：按 Die 分离 Ready Queue 的设计、修复与性能数据

## 1. 方案目标

Mode9 试图同时解决两类问题：

1. A5 有两个 Die，Scheduler 访问远端 AICore 的 FIN、参数和派发寄存器存在额外代价；
2. producer 完成后，如果 consumer 被任意 Scheduler 抢走并派到另一个 Die，会增加跨 Die 的依赖交接和数据访问。

方案基于 Mode7 的连续物理 cluster 分配：28 个 cluster 按标准化物理 cluster 序号连续切成四段，S0/S1 只拥有 Die0，S2/S3 只拥有 Die1。Mode9 在此基础上把普通 ready task 按执行 Die 分流。

## 2. 具体修改方式

### 2.1 三类 Ready Queue

每种资源 shape（AIC、AIV、MIX）各维护三类队列：

- `GLOBAL`：没有 Die 约束或无法在单 Die 内执行的任务；
- `DIE0`：要求在 Die0 执行的任务；
- `DIE1`：要求在 Die1 执行的任务。

每个 task slot 增加 `ready_domain`。该字段使用 first-writer CAS，只允许从 `UNASSIGNED` 转移到 `GLOBAL/DIE0/DIE1`，避免多个 producer 同时完成时重复改变 consumer 的归属；slot reset 时恢复为 `UNASSIGNED`。

### 2.2 根任务路由

能够在单 Die 容量内运行的普通根任务，按 resource shape 在 Die0、Die1 之间轮转；DUMMY、sync-start、early-dispatch 和超过单 Die 容量的任务进入 GLOBAL。

旧实现用引用和后缀自增更新 `root_ready_domain_turn`。在 AICPU 目标构建中实际观察到轮转状态没有可靠持久化，导致所有 alternating 根任务进入 Die0，最终只有 S0 工作。修复后改为显式：

1. 计算 shape 数组下标；
2. 显式加载轮转字节；
3. 显式写回 `turn + 1`；
4. 使用加载到的旧值决定 Die0/Die1。

修复后的 alternating 四级泳道中，1,000 个独立任务恰好按 Die0/Die1=`500/500` 分布，S0/S1/S2/S3=`254/246/236/264`。

### 2.3 依赖任务路由

- live fan-in：最后一个完成 fan-in 的 Scheduler 使用自己的 Die 原子认领 consumer 的 `ready_domain`；
- 提交时依赖已经完成：优先继承 task-id 最大的本地 producer domain；没有可继承 domain 时使用根任务轮转；
- completion、async-wait 和 deferred-release 路径都把当前 Scheduler 的 Die 继续传给 consumer release，避免不同完成路径丢失 locality。

### 2.4 Scheduler 取队列规则

Scheduler 只访问自己所在 Die 的本地队列和 GLOBAL 队列，不跨 Die stealing。两类队列同时有任务时使用轮转选择，避免 GLOBAL 永久饥饿。AIC/AIV/MIX 的资源检查、同步队列和残留 MIX 检测同步扩展到本地队列。

### 2.5 内存布局和初始化

共享 Scheduler state 增加 `2 × resource_shape` 组队列及 slot 存储；初始化、布局计算、reset、日志打印和单元测试一并更新。每个 Scheduler 在线程/core 分配完成后，根据自己拥有的标准化物理 cluster 判断所在 Die；如果一个线程同时拥有两个 Die 的 cluster，握手直接失败。

## 3. 验证范围

- 代码提交：`5d3aa98755cf`，父提交为原 Mode9 `52e7d1ef`；
- C++ 单元测试：`test_a5_scheduler_state`、`test_a5_task_state` 通过；
- 卡2四级泳道：8 个 case 全部成功生成，所有 case 的 S0～S3 均有任务；
- 卡1 benchmark：7 个非 Qwen case 各 100 轮，Qwen3 运行 5 轮，8/8 全部通过；
- 未使用 Scheduler assignment 实验环境变量；
- 性能结论不使用包含编译和 Python 开销的 Host latency。

## 4. 卡1性能数据

非 Qwen Main 基线来自同卡已有 clean `origin/main`（`3b578e30`）两次 100 轮数据，共 200 个样本。正数表示 Mode9 更慢。

| Case | Main Device | Mode9 Device | Device变化 | Effective变化 | Scheduler变化 | p50变化 | p95变化 |
|---|---:|---:|---:|---:|---:|---:|---:|
| alternating Case1 | 1,441.850 us | 1,590.404 us | **+10.30%** | +10.56% | +10.69% | +11.34% | +8.41% |
| BGEMM Case0 | 1,458.300 us | 1,642.829 us | **+12.65%** | +13.03% | +13.11% | +8.26% | +8.96% |
| paged attention Case1 | 1,558.700 us | 1,731.239 us | **+11.07%** | +11.25% | +11.30% | +10.39% | +15.64% |
| paged attention Case2 | 901.100 us | 1,065.257 us | **+18.22%** | +18.99% | +19.19% | +17.44% | +26.67% |
| manual-scope Case1 | 1,629.500 us | 1,743.515 us | **+7.00%** | +7.08% | +7.11% | +6.53% | +13.59% |
| manual-scope Case2 | 925.550 us | 1,069.516 us | **+15.56%** | +16.17% | +16.32% | +15.63% | +20.83% |
| batch paged attention Case1 | 7,434.600 us | 7,297.594 us | **-1.84%** | -1.86% | -1.87% | -1.94% | -1.34% |

7 个非 Qwen case 的几何平均变化：

- Device：`+10.25%`；
- Effective：`+10.56%`；
- Orchestrator：`+5.53%`；
- Scheduler：`+10.65%`。

p50 与均值方向一致，因此这些回退不能解释为少数长尾。Paged Attention Case2 的平均回退和尾部回退最大。

## 5. Qwen3结果

Mode9 的 5 轮 Device 原始数据为：

`35,384.6, 35,058.5, 34,940.5, 34,692.5, 34,966.0 us`

按 Device 删除一个最快和一个最慢样本后，选择第2、3、5轮，并使用相同轮次计算所有指标：

| Metric | 历史Main | Mode9 | 变化 |
|---|---:|---:|---:|
| Device | 36,083.248 us | 34,988.333 us | **-3.034%** |
| Effective | 36,046.949 us | 34,953.400 us | **-3.034%** |
| Orch | 10,279.340 us | 10,122.467 us | -1.526% |
| Scheduler | 36,043.855 us | 34,951.933 us | **-3.029%** |

Qwen Main 基线来自 clean `3b578e30` 在卡1的已有5轮数据，同样删除最快和最慢后取中间3轮。

## 6. 当前优化效果

可以确认的正向结果：

- Qwen3 Device、Effective、Scheduler 均改善约 `3.03%`；
- batch paged attention Device 改善 `1.84%`，Orchestrator 改善 `8.57%`；
- 根任务轮转恢复后，所有 Scheduler 都能参与工作，不再出现 S0 独占全部任务的错误；
- 对带明确 producer-consumer 链的任务，ready domain 能跨 completion/deferred-release 路径传递，为减少跨 Die 依赖提供机制基础。

## 7. 现有缺陷与风险

1. **不适合作为默认全局优化。** 六个非 Qwen case 的 Device 回退超过 2%，7-case 几何平均回退 `10.25%`。
2. **Die locality 与负载均衡仍会冲突。** BGEMM 四级泳道中任务分布为 Die0/Die1=`750/250`；PA Case2 为`317/195`，batch PA为`2560/1536`。consumer 继承完成者 Die 可以减少迁移，但也可能把长链集中到单侧。
3. **多队列引入固定成本。** 每次 ready/release 都需要读取或 CAS domain，并在本地/GLOBAL 队列之间仲裁；对 alternating、PA 这类细粒度任务，固定开销可能大于减少远端访问的收益。
4. **不允许跨 Die stealing。** 一侧存在 ready task 而本侧 core 饱和时，另一侧空闲 Scheduler 不能帮助执行，可能降低全芯片利用率。
5. **根任务轮转不等于依赖链均衡。** 根节点数量可以做到 50:50，但 downstream consumer 的最终归属由依赖完成顺序决定，仍可能偏向某一 Die。
6. **历史基线不是同窗口交叉 A/B。** 当前 Mode9 与 Main 均在卡1，但运行日期不同；此前同一卡上已观察到明显性能档位漂移。因此表中数值满足指定历史基线口径，但不能把全部差值严格归因于代码。
7. **泳道存在观察者效应。** 四级逐任务记录用于验证结构，不能替代关闭泳道的 benchmark 性能数据。
8. **平台假设较强。** 当前实现要求 active cluster 可被两个 Die 均分，并要求每个 Scheduler 的连续 cluster 全部位于单个 Die；其他 A5 SKU 需要重新验证 topology 和容量判断。

## 8. 随提交保存的数据

- `summary.csv`：Main/Mode9 均值、变化率、p50/p95和样本数；
- `*_summary.log`：每个 case 的逐轮 Host/Device/Effective/Orch/Scheduler 解析结果；
- 本文：算法、实验方法、结论和缺陷说明。

原始 CANN plog 和重复的完整 stdout 日志体积较大且不参与统计，未纳入提交。
