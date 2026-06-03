# AIC:AIV 1:1 改造方案（a2a3 真机）

## 目标

把 a2a3 的 cube(AIC):vector(AIV) 调度比例从 **1:2** 改为 **1:1**：

- 每组(cluster)只使用一个 AIV，舍弃 AIV1；
- MIX 任务由 1 AIC + 2 AIV 退化为 **1C1V**(1 AIC + 1 AIV)。

适用场景：真机 A2/A3。仿真 `a2a3sim` 可复用同一套调度层改动。

## 实现方案

建立一个与a2a3并列的1c1v文件夹，在1c1v文件夹中修改，同时增加编译选项使得可以编译其中的相关代码，后续运行时默认走1c1v文件夹下的代码。


## 核心约束：硬件必然拉起 2× vector 核

真机 A2/A3 的 1:2 是硅片物理耦合属性：launch `block_dim` 个 cube，硬件
必然耦合拉起 `2*block_dim` 个 vector 子核(`get_subblockdim()` 恒为 2)，
无法只拉起一个。这些核启动后立即运行常驻 kernel。

**关键推论**：被拉起的核要么被驱动到干净退出，要么挂死拖垮整个 stream
(`aclrtSynchronizeStreamWithTimeout` 超时)。"放着不管"不是一个选项。

因此本方案的本质是 **两级核数分离**：

> 物理层仍按 1:2 拉核、握手、收尾；调度层只把每组 AIV0 纳入可调度集合，
> AIV1 被驱动到退出后永不派发任务。

## 设计前提（AIV0-per-cluster 配对）

每组保留 **AIV0**、舍弃 **AIV1**，即 `aiv_worker_ids_[2*ci]` 保留、
`[2*ci+1]` 丢弃。AIC 与同一个 cube 自带的 AIV0 配对，保证 TPUSH/TPOP
与 cache 局部性正确。这恰好使用一半 AIV。

> 不采用"字面前一半 AIV"(`aiv_worker_ids_[0..block_dim)`)的读法，
> 那会让 cluster i 的 AIC 配到别的 cube 的 AIV，破坏局部性。

## AIV1 退出：两种物理策略

被硬件拉起的 AIV1 核启动后会在 `aicore_execute`(aicore_executor.cpp)
的 Phase 1 自旋等待握手信号，且 **Phase 1 没有退出分支**；退出信号只在
Phase 4 经寄存器下发。所以不能单纯"握手时跳过 AIV1"——那样它会永远卡在
Phase 1，stream 永不完成 → 挂死。

| | P1 停泊(推荐先做) | P2 跳过握手 |
| --- | --- | --- |
| device kernel 改动 | 否 | 是(AIV1 入口即返回) |
| AICPU 握手改动 | 仅追加退出 | 是(跳过 AIV1 槽) |
| 死锁风险 | 低 | 中(两端契约须一致) |
| worker 槽/profiling | 仍占用 | 可释放 |

- **P1 停泊**：全部核照常走完握手(AICPU 因此拿到 AIV1 的 `reg_addr`)，
  随后立即对 AIV1 发 `AICORE_EXIT_SIGNAL` 使其 Phase 4 立即 return。
  改动锁在 AICPU 单侧，kernel 不动，死锁面最小。
- **P2 跳过握手**：device 侧让 AIV1(`get_subblockid()==1`)入口即返回；
  AICPU 侧 `handshake_all_cores` 按 worker 下标奇偶跳过 AIV1 槽
  (`(i - block_dim)` 为奇数即 AIV1)。两端契约必须严格一致，否则死锁。

方案设计：选择P2，aiv的逻辑地址只给BLOCK_DIM个槽（只保留aiv0的位置），物理地址保留2*BLOCK_DIM个槽。后续计划根据这个调整。

## 保持不变（物理层）

| 位置 | 原因 |
| --- | --- |
| `platform/include/common/platform_config.h` `CORES_PER_BLOCKDIM=3` 等 | 决定物理拉核数与握手槽数；改小会导致 AIV1 无槽越界/挂死 |
| `platform/onboard/host/device_runner.cpp` 拉核 + `core_type` 赋值 | 全部 `2*block_dim` 个 AIV 仍需 launch；AIV1 由 kernel 早返回自行退出 |
| `RUNTIME_MAX_WORKER=72`、`PLATFORM_MAX_CORES=72`、profiling buffer 尺寸 | 仍需覆盖全部物理核 |

P2 已实现，下列两处**确有改动**（不属于"保持不变"）：

- `platform/{onboard,sim}/aicore/kernel.cpp`：AIV1(`get_subblockid()==1` /
  sim `subblock_id==1`)入口即返回，不进 executor 握手自旋。
- `handshake_all_cores`：按 worker 下标跳过停泊的 AIV1
  (`(i-block_dim)` 为奇数)，故只发现 `block_dim` 个 AIV0。

> 建议新增文档性常量 `PLATFORM_AIV_CORES_PER_CLUSTER_USED = 1`(逻辑用量)，
> 与物理 `PLATFORM_AIV_CORES_PER_BLOCKDIM = 2` 并存，避免后人误读。

## 需要修改（逻辑层）

### 改动 1 — CoreTracker 位布局：每 cluster 3 bit → 2 bit

文件：`runtime/tensormap_and_ringbuffer/runtime/scheduler/scheduler_types.h`

| 项 | 现在 (1:2) | 改为 (1:1) |
| --- | --- | --- |
| 每 cluster 位数 | 3(AIC=i*3, AIV0=+1, AIV1=+2) | 2(AIC=i*2, AIV=i*2+1) |
| `MAX_CLUSTERS` | `63/3=21` | `63、、`42/2=21` |
| `init()` | `aic\|=1<<(i*3); aiv\|=6<<(i*3)` | `aic\|=1<<(i*2); aiv\|=2<<(i*2)` |
| `set_cluster` | 写 3 个 `core_id_map_[ci*3+{0,1,2}]` | 去掉 aiv1 参数，写 `[ci*2+{0,1}]` |
| `get_valid_cluster_offset_states` | AIV: `(s>>1)\|(s>>2)&aic_mask`；MIX: `(s>>1)&(s>>2)&s&aic_mask` | AIV: `(s>>1)&aic_mask`；MIX: `(s>>1)&s&aic_mask` |
| accessors | `get_aiv0/aiv1_*` | 合并为单 `get_aiv_*`(offset = cluster+1) |
| 两阶段 dispatch 查询 | 含 AIV1 的 `>>2` 项 | 去掉 |
| `core_num()` | `cluster*3` | `cluster*2` |

### 改动 2 — cluster 组装：丢弃 AIV1

文件：`runtime/tensormap_and_ringbuffer/runtime/scheduler/scheduler_cold_path.cpp`
(`assign_cores_to_threads` 与 `reassign_cores_for_all_threads`)

```cpp
int32_t aic_wid  = aic_worker_ids_[ci];
int32_t aiv0_wid = aiv_worker_ids_[ci];   // dense: discovery skipped parked AIV1
core_trackers_[t].set_cluster(cl_idx, aic_wid, aiv0_wid);  // 去掉 aiv1 参数
```

- `thread_cores_num = max_clusters_per_thread * 3` → `* 2`
- `cluster_offset = c * 3` → `* 2`
- reassign 中恢复运行态的 `cl_idx*3 + {0,1,2}` → `cl_idx*2 + {0,1}`，删 aiv1 段

> P2 已实现：`handshake_all_cores` 在发现阶段就跳过停泊的 AIV1
> (worker 下标 `(i-block_dim)` 为奇数者)，因此 `aiv_count_ == block_dim`、
> `aiv_worker_ids_` 紧凑只含 AIV0，cluster 组装用 `[ci]` 而非 `[2*ci]`。
> 命名上保留 `AIV0` / `aiv0_kernel_id`（不重命名为 AIV），以保持 orchestration
> API 与 examples 不变。

### 改动 3 — 提交契约：删 AIV1 槽，MIX = 1C1V

文件：`runtime/tensormap_and_ringbuffer/runtime/pto_submit_types.h`

- `PTO2_SUBTASK_SLOT_COUNT = 3 → 2`
- `enum PTO2SubtaskSlot { AIC=0, AIV=1 }`(删 AIV1)
- 删 `PTO2_SUBTASK_MASK_AIV1`；`SYNC_START` 位 `1<<3` 顺延为 `1<<2`
- `MixedKernels` 删 `aiv1_kernel_id`；`to_active_mask()` 去掉对应分支
- `to_shape()`：`bit_count>=2 → MIX` 不变，现 MIX 最大即 AIC+AIV = 1C1V

### 改动 4 — orchestrator 与 dispatch

- `runtime/.../pto_orchestrator.cpp`：`submit_task_impl` 去掉 `aiv1_kernel_id`
  形参；删掉 "aiv1-only → aiv0 归一化"段；`task.kernel_id[]` 只写 2 槽
- `runtime/.../scheduler/scheduler_dispatch.cpp`：`dispatch_mix_block_to_cluster`
  删掉 AIV1 派发块；batch 数组 `MAX_CLUSTERS * 3 → * 2`
- `runtime/.../scheduler/scheduler_completion.cpp`：审计所有按 3 槽/AIV1
  遍历的循环，改 2 槽

### 改动 5 — AIV1 退出(防止 hang)

按所选策略实现 P1 或 P2(见上节)。这是正确性必做项，不可遗漏。

## 对外影响面

- orchestration API `pto_orchestration_api.h`：`rt_submit_aiv_task` 已用
  `aiv0_kernel_id`，无需改；直接构造 `MixedKernels.aiv1_kernel_id` 的用户
  代码会编译失败
- `examples/a2a3`：文档示例用 `aic_kernel_id + aiv0_kernel_id`，未直接用
  aiv1，影响小；需 grep 确认无 example 设 `aiv1_kernel_id`
- `python/bindings/`：需确认无 `PTO2SubtaskSlot::AIV1` / `aiv1_kernel_id` 引用
- 文档一致性：`scheduler_types.h` 头注释、`runtime.h` 的 `// 24 AIC + 48 AIV`、
  各 `docs/*.md` 中 "3 bits per cluster / 1 AIC + 2 AIV" 描述需同步更新

## 测试计划

1. 失败复现：a2a3sim 跑 MIX(AIC+AIV) example，断言 cluster 只占 2 核、
   AIV1 不被派发
2. `pytest examples tests/st --platform a2a3sim` 全绿
3. C++ 单测 `tests/ut/cpp`：CoreTracker 位运算必须补单测
   (mask、`get_valid_cluster_offset_states` 三种 shape、`pop_first`)
4. 真机 `--platform a2a3 --device 4-7`：重点验证收尾不 hang(AIV1 退出)、
   无死锁；vector 算力减半，吞吐预期下降属正常
5. benchmark 对比，确认延迟/吞吐符合"vector 减半"预期而非异常退化

## 风险与权衡

- **算力浪费**：每个 cube 自带的 AIV1 被永久闲置，真机 vector 吞吐砍半，
  是 1C1V 调度模型的固有代价
- **CoreTracker 位运算最易出错**，移位掩码错一位即静默错派，务必配单测
- **reassign 的运行态恢复**与 `assign_cores_to_threads` 共用索引，须一起改
- **P2 的两端契约**：device 与 AICPU 对"谁是 AIV1"的判定须完全对齐
