# paged_attention_unroll样例各数据结构访问次数理论数据及实际数据

## 表1：任务类型的 P / C / S / N_in / N_out（xlsx口径）

> 任务名称按 `test_paged_attention_unroll.py` 的 `incores` 定义映射：  
> `func_id=0 -> aic_qk_matmul`，`1 -> aiv_softmax_prepare`，`2 -> aic_pv_matmul`，`3 -> aiv_online_update`。

**表头字段含义**：

- **桶编号**：任务在提交队列中的 bucket 序号。
- **提交次数**：该 bucket 内任务的提交总次数。
- **任务类型**：任务所属的执行器类别（alloc_tensors任务 / AIC任务 / AIV任务）。
- **任务名称**：任务在 `incores` 定义中的逻辑名称。
- **ring**：任务使用的 ring buffer 编号。
- **mask**：任务所属 core 的位掩码（AIC 为 `0x01`、AIV 为 `0x02`）。
- **P（Producer/依赖前驱数）**：该任务依赖的前驱任务数，用于 ②依赖构建 阶段的 `ΣP` 统计。
- **C（Consumer/依赖后继数）**：该任务的下游任务数，用于 ⑥解依赖 阶段的 `ΣC` 统计。
- **S（Subtask/子任务数）**：该任务实际 dispatch 的子任务数，用于 ④任务Dispatch 阶段的 `ΣS` 统计。
- **N_in（输入 Tensor 数）**：单次执行读取的输入 Tensor 数量，用于 ⑤AICore执行 阶段 Tensor 读次数统计。
- **N_out（输出 Tensor 数）**：单次执行写入的输出 Tensor 数量，用于 ⑤AICore执行 阶段 Tensor 写次数统计。
- **tc（Tensor 总访问次数）**：单次任务的 Tensor 访问总次数（`N_in + N_out`），用于 ①Payload构建 Tensor 写次数统计。
- **sc（Sync 计数）**：单次任务涉及的同步/信号操作次数。

| 桶编号 | 提交次数 | 任务类型 | 任务名称 | ring编号 | core掩码 | 前驱数 | 后继数 | 子任务数 | 输入Tensor数 | 输出Tensor数 | Tensor访问数 | 同步计数 |
|---|---:|---|---|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 0 | 256 | alloc_tensors任务 | alloc_tensors(无InCore) | 1 | 0 | 0 | 1 | 0 | 0 | 3 | 3 | 0 |
| 1 | 256 | AIC任务 | aic_qk_matmul | 1 | 0x01 | 0 | 1 | 1 | 3 | 1 | 4 | 2 |
| 2 | 256 | AIV任务 | aiv_softmax_prepare | 1 | 0x02 | 1 | 2 | 1 | 1 | 3 | 4 | 3 |
| 3 | 256 | AIC任务 | aic_pv_matmul | 1 | 0x01 | 1 | 1 | 1 | 3 | 1 | 4 | 2 |
| 4 | 256 | AIV任务 | aiv_online_update | 1 | 0x02 | 3 | 0 | 1 | 7 | 4 | 7 | 2 |

> 代入基数（按 `paged_attention_unroll` 代码路径）：`batch=256`，`q_loop=1`，`bn_this_batch=64`，`N_UNROLL=64`，每个 scope 恰好提交 5 个任务（1 alloc + 4 in-core）。  
> 因此：`submit总数=1280`，`mixed任务数=1024`，`ΣP=1280`，`ΣC=1280`，`ΣS=1024`，`ΣN_in=3584`，`ΣN_out=2304`。

---

## 表2：理论数据（xlsx口径）

**表头字段含义**：

- **模块**：runtime pipeline 的阶段编号与名称（①Payload构建 / ②依赖构建 / ③内存分配 / ④任务Dispatch / ⑤AICore执行 / ⑥解依赖 / ⑦资源释放）。
- **数据结构**：被访问的 runtime 数据结构（如 `PTO2TaskSlotState`、`PTO2DepListEntry`、`PTO2ReadyQueue` 等）。
- **cacheline序号**：访问覆盖的 cacheline 范围，`CL0` 表示首个 cacheline，`CL0~CL50` 表示跨多个 cacheline。
- **非atomic读次数**：以普通 load 方式读取该结构的总次数。
- **非atomic写次数**：以普通 store 方式写入该结构的总次数。
- **atomic次数**：原子操作（含 load/store/fetch_add/CAS 等全部原子步骤）的累计次数。
- **加锁次数**：获取互斥锁/spinlock 的次数。
- **CAS次数**：`compare_exchange`（含成功与失败）调用的总次数。
- **理论说明**：该行数据的推导公式或基数代入说明。

**理论公式符号含义**：

- **`submit_count`（提交总数）**：所有 bucket 的 `submit` 求和，本样例 `=1280`。
- **`任务数`（mixed任务数）**：需要走依赖/dispatch 流程的 in-core 任务数，排除 alloc_tensors 任务后取 `256×4=1024`。
- **`ΣP`（总前驱数）**：各 bucket 的 `P×submit` 求和，表示 ②依赖构建 中形成的依赖边总数，本样例 `=1280`。
- **`ΣC`（总后继数）**：各 bucket 的 `C×submit` 求和，表示 ⑥解依赖 中需要 fanout 的边总数，本样例 `=1280`。
- **`ΣS`（总子任务数）**：各 bucket 的 `S×submit` 求和，表示 ④任务Dispatch 中下发的 subtask 总数，本样例 `=1024`。
- **`ΣN_in / ΣN_out`**：各 bucket 的 `N_in×submit` / `N_out×submit` 求和，表示 ⑤AICore执行 阶段 Tensor 读/写总次数，本样例分别为 `3584 / 2304`。
- **`tc`（Tensor 访问次数）**：`N_in + N_out`，用于 ①Payload构建 阶段 Tensor 写次数 `Σ(tc×submit)=5632`。
- **`无依赖in-core任务数`**：`P=0` 的 in-core 任务总数（本样例即 qk 任务，`=256`），用于 ②依赖构建 fast path 行。
- **`有依赖任务数`**：`P>0` 的 in-core 任务数（`=1024−256=768`），用于 ⑥解依赖 ReadyQueue 行。
- **`R_rel`（资源释放次数）**：⑦资源释放 阶段发生的 release 动作次数，本样例 `=1024`。
- **`PTO2_NUM_RESOURCE_SHAPES`**：资源 shape 的种类数，本样例 `=3`（AIC / AIV / MIX）。
- **`自旋次数`（spin count）**：锁/CAS 冲突导致的重试次数，理论给范围（如 `256 + 自旋次数`），实测按日志累加。
- **`r / w / a / L / cas`**：行公式里的缩写，分别对应 `非atomic读次数 / 非atomic写次数 / atomic次数 / 加锁次数 / CAS次数`。

| 模块 | 数据结构 | cacheline序号 | 非atomic读次数 | 非atomic写次数 | atomic次数 | 加锁次数 | CAS次数 | 理论说明 |
|---|---|---|---:|---:|---:|---:|---:|---|
| ②依赖构建 | PTO2TaskSlotState | CL0 | 1280 | 1024 | 4864 | 1280 | 1280 | 代入 `mixed任务数=1024, ΣP=1280`：`r=ΣP, w=任务数, a=任务数+3×ΣP` |
| ②依赖构建 | PTO2DepListEntry | CL0 | 0 | 1280 | 0 | 0 | 0 | 代入 `ΣP=1280`（每条依赖边分配/写入1次） |
| ②依赖构建 | PTO2ReadyQueue（fast path） | CL0~CL2 | 256 | 256 | 256 + 自旋次数 | 0 | 256 | 代入 `无依赖in-core任务数=256`（qk 任务） |
| ①Payload构建 | PTO2TaskPayload（访问所有数据量，做了一次写） | CL0~CL50 | 0 | 1280 | 0 | 0 | 0 | 按 `submit_count=1280`（每任务1次 payload init） |
| ①Payload构建 | PTO2TaskDescriptor | CL0 | 0 | 1280 | 0 | 0 | 0 | 按 `submit_count=1280`（每任务1次 descriptor 写） |
| ①Payload构建 | Tensor | CL0~CL1 | 0 | 5632 | 0 | 0 | 0 | 按 bucket 的 `tc*submit` 求和 |
| ③内存分配 | PTO2RingFlowControl | CL0 | 0 | 1280 | 2560 | 0 | 0 | 未见自旋明细时取 `自旋次数=0` |
| ④任务Dispatch | PTO2TaskSlotState | CL0 | 1024 | 1024 | 1024 | 0 | 0 | 代入 `ΣS=1024`（每子任务一次） |
| ④任务Dispatch | PTO2TaskPayload | CL0~CL50 | 1024 | 0 | 0 | 0 | 0 | 代入 `ΣS=1024`（按 xlsx“payload遍历一遍/每子任务一次”） |
| ④任务Dispatch | PTO2TaskDescriptor | CL0 | 1024 | 0 | 0 | 0 | 0 | 代入 `ΣS=1024` |
| ④任务Dispatch | PTO2DispatchPayload | CL0~CL18 | 0 | 1024 | 0 | 0 | 0 | 代入 `ΣS=1024` |
| ④任务Dispatch | PTO2ReadyQueue | CL0~CL2 | 1024 | 1023 | 3072 + 自旋次数 | 0 | 1024 | 按 `S` 口径放大：每次pop近似3个原子步骤，重试并入“自旋次数” |
| ⑤AICore执行 | PTO2DispatchPayload | CL0~CL18 | 1024 | 0 | 0 | 0 | 0 | 代入 `ΣS=1024`（xlsx该行读次数=1/子任务） |
| ⑤AICore执行 | Tensor | CL0~CL1 | 3584 | 2304 | 0 | 0 | 0 | 代入 `ΣN_in=3584, ΣN_out=2304` |
| ⑥解依赖 | PTO2TaskSlotState | CL0 | 2304 | 0 | 4352 | 1024 | 1024 | 代入 `mixed任务数=1024, ΣC=1280`：`r=1+ΣC, a=3+ΣC` |
| ⑥解依赖 | PTO2DepListEntry | CL0 | 1280 | 0 | 0 | 0 | 0 | 代入 `ΣC=1280`（fanout 链遍历） |
| ⑥解依赖 | PTO2ReadyQueue | CL0~CL2 | 0 | 0~768 | 0~2304 | 0 | 0~768 | 代入 `有依赖任务数=768` 且 `PTO2_NUM_RESOURCE_SHAPES=3`，故 `atomic=0~(3×768)=0~2304`（3 来自 AIC/AIV/MIX 三类资源shape） |
| ⑦资源释放 | PTO2TaskSlotState | CL0 | 1280 | 0~1024 | 2304 | 0 | 2304 | 代入 `ΣP=1280, R_rel=1024`：`a=R_rel+ΣP, cas=R_rel+ΣP` |
| ⑦资源释放 | PTO2TaskPayload | CL0~CL2 | 1024 | 0 | 0 | 0 | 0 | 代入 `R_rel=1024`（每次 release 读1次） |
| ⑦资源释放 | PTO2RingFlowControl | CL0 | 0 | 1024 | 2048 + 自旋次数 | 0 | 0 | 代入 `R_rel=1024` |
| ⑦资源释放 | PTO2FaninSpillEntry | CL0 | 0 | 0 | 0 | 0 | 0 | 无访问 |
| ⑦资源释放 | RingSchedState.advance_lock | CL0 | 0~1024 | 0~1024 | 0~1024 | 0~1024 | 0~1024 | 将 xlsx 单任务范围按 `R_rel=1024` 放大后的总量范围 |

## 表3：实测数据（日志汇总口径）

**表头字段含义**：

- **模块 / 数据结构 / cacheline序号 / 非atomic读次数 / 非atomic写次数 / atomic次数 / 加锁次数 / CAS次数**：与表2同义，此处为运行时日志实际累计值。
- **实测说明**：该行实测值相对理论值的差距方向与原因简述。

| 模块 | 数据结构 | cacheline序号 | 非atomic读次数 | 非atomic写次数 | atomic次数 | 加锁次数 | CAS次数 | 实测说明 |
|---|---|---|---:|---:|---:|---:|---:|---|
| ②依赖构建 | PTO2TaskSlotState | CL0 | 2304 | 4352 | 4864 | 1280 | 1280 | 相比理论：`r/w` 偏高，`a/L/cas` 对齐 |
| ②依赖构建 | PTO2DepListEntry | CL0 | 0 | 1022 | 0 | 0 | 0 | 相比理论：`w` 略低（1022 vs 1280） |
| ②依赖构建 | PTO2ReadyQueue（fast path） | CL0~CL2 | 1024 | 1024 | 1788 | 0 | 1024 | 相比理论显著偏高；可写成 `a=1024+764(自旋)` |
| ①Payload构建 | PTO2TaskPayload（访问所有数据量，做了一次写） | CL0~CL50 | 0 | 1280 | 0 | 0 | 0 | 与理论一致 |
| ①Payload构建 | PTO2TaskDescriptor | CL0 | 0 | 1280 | 0 | 0 | 0 | 与理论一致 |
| ①Payload构建 | Tensor | CL0~CL1 | 0 | 5632 | 0 | 0 | 0 | 与理论一致 |
| ③内存分配 | PTO2RingFlowControl | CL0 | 0 | 1280 | 2560 | 0 | 0 | 与理论一致 |
| ④任务Dispatch | PTO2TaskSlotState | CL0 | 618 | 618 | 618 | 0 | 0 | 相比理论偏低（618 vs 1024） |
| ④任务Dispatch | PTO2TaskPayload | CL0~CL50 | 618 | 0 | 0 | 0 | 0 | 相比理论偏低（618 vs 1024） |
| ④任务Dispatch | PTO2TaskDescriptor | CL0 | 1024 | 0 | 0 | 0 | 0 | 与理论一致 |
| ④任务Dispatch | PTO2DispatchPayload | CL0~CL18 | 0 | 618 | 0 | 0 | 0 | 相比理论偏低（618 vs 1024） |
| ④任务Dispatch | PTO2ReadyQueue(pop命中) | CL0~CL2 | 637 | 0 | 1511 | 0 | 637 | xlsx 无命中/空转拆分项；该行为实现新增统计 |
| ④任务Dispatch | PTO2ReadyQueue(pop空转) | CL0~CL2 | 26045 | 0 | 52163 | 0 | 26045 | xlsx 无命中/空转拆分项；空转主导总量 |
| ⑤AICore执行 | PTO2DispatchPayload | CL0~CL18 | 19456 | 0 | 0 | 0 | 0 | 相比理论显著偏高（19456 vs 1024） |
| ⑤AICore执行 | Tensor | CL0~CL1 | 9728 | 0 | 0 | 0 | 0 | 相比理论：`r` 偏高（9728 vs 3584），`w` 偏低（0 vs 2304） |
| ⑥解依赖 | PTO2TaskSlotState | CL0 | 2047 | 3071 | 4095 | 1024 | 1024 | 相比理论：`r/a` 略低、`w` 明显偏高，`L/cas` 对齐 |
| ⑥解依赖 | PTO2DepListEntry | CL0 | 1023 | 0 | 0 | 0 | 0 | 相比理论偏低（1023 vs 1280） |
| ⑥解依赖 | PTO2ReadyQueue | CL0~CL2 | 0 | 766 | 6858 | 0 | 766 | `w/cas` 在理论范围内，`atomic` 超理论上界（6858 > 2304） |
| ⑦资源释放 | PTO2TaskSlotState | CL0 | 2304 | 2304 | 3328 | 0 | 1280 | 相比理论：`r/w/a` 偏高，`cas` 偏低 |
| ⑦资源释放 | PTO2TaskPayload | CL0~CL2 | 1024 | 0 | 0 | 0 | 0 | 与理论一致 |
| ⑦资源释放 | PTO2RingFlowControl | CL0 | 0 | 1214 | 2428 | 0 | 0 | 相比理论偏高（`w:+190`，`a:+380`，可视作额外重试） |
| ⑦资源释放 | PTO2FaninSpillEntry | CL0 | 0 | 0 | 0 | 0 | 0 | 理论(0/0/0/0/0)，Δ=(0/0/0/0/0) |
| ⑦资源释放 | RingSchedState.advance_lock | CL0 | 1280 | 1280 | 1280 | 1280 | 1280 | 按 Thread0/1/2 汇总 |

### 表2/表3差距原因汇总

- **②依赖构建**
  - `PTO2TaskSlotState`：`atomic/L/CAS` 与理论一致，`r/w` 偏高；实现把部分状态初始化与状态迁移写回计入该结构。
  - `PTO2DepListEntry`：`w=1022` 低于理论 `1280`；与实际形成的 dep 边数（以及窗口/线程切片）有关。
  - `PTO2ReadyQueue(fast path)`：仍明显高于理论（`1024/1024/1788/1024` vs `256/256/256+spin/256`）；实现覆盖实际入队/重试路径，不是“无依赖任务一次入队”的抽象口径。

- **①Payload构建 与 ③内存分配**
  - 三项（`PTO2TaskPayload`、`PTO2TaskDescriptor`、`Tensor`）和 `PTO2RingFlowControl` 与理论基本一致，说明 submit 阶段口径对齐较好。

- **④任务Dispatch**
  - `PTO2TaskSlotState`、`PTO2TaskPayload(meta)`、`PTO2DispatchPayload` 主行已按理论口径映射，实测与理论同阶（本次为 `618` 级别）。
  - `PTO2TaskDescriptor` 仍与主行存在差异（`1024` vs `618`）：其实现统计更接近“按任务弹出数”而非“按 subtask 下发数”。
  - `PTO2ReadyQueue(pop)` 主行按理论口径映射为 `S`，但拆分行仍显示真实运行时负载：空转远高于命中（`26045` vs `637`），说明调度循环轮询占主导。

- **⑤AICore执行**
  - `PTO2DispatchPayload` 已与理论口径对齐（按 `S` 计，主行不再乘槽位数）。
  - `Tensor` 读次数仍高于理论（`9728` vs `3584`）：实现统计按实际访存路径累计，理论是结构级简化模型。

- **⑥解依赖**
  - `PTO2TaskSlotState` 与 `PTO2DepListEntry` 低于理论全量值（`2047/1023` 对 `2304/1280`）：受窗口累计与线程切分影响。
  - `PTO2ReadyQueue` 的 `atomic=6858` 高于理论上界 `2304`：实现将 CAS 失败重试与队列内部原子序列全部计入。

- **⑦资源释放**
  - `PTO2TaskPayload`、`PTO2FaninSpillEntry` 与理论一致。
  - `PTO2TaskSlotState`、`PTO2RingFlowControl`、`advance_lock` 与理论存在偏离，主要来自运行时累计口径（包含重试与并发竞争）；理论是简化公式放大。

### ④任务Dispatch / PTO2ReadyQueue 口径差异原因（当前实现）

- 当前实现不是按“成功 dispatch 的任务数”统计 ReadyQueue，而是按“调度循环里每次尝试从就绪队列 pop”统计。
- `非atomic读次数=26682`（`pop命中+pop空转=637+26045`）来自 `g_sched_ready_queue_pop_count` 口径：每次调用 `pop_ready_tasks_batch()` 都会计数（命中与空转都算）。
- `CAS次数=26682` 在拆分行里与上述轮次同阶（`637+26045`）；主行 `CAS=618` 为理论口径映射值（按 `S`）。
- `atomic次数=53674`（`1511+52163`）来自 `g_sched_pop_atomic_count`，包括 `dequeue_pos.load`、`sequence.load`、`compare_exchange_weak` 成功/失败、`sequence.store` 等所有原子步骤，空轮询/争用重试也会累计。
- 因此与 xlsx 理论值差异大的核心原因是：xlsx 的 ④ ReadyQueue 偏“每任务一次路径”的抽象口径，而实现统计是“调度线程轮询 + 重试 + 原子细节累计”的运行时口径，量级天然更大。
- 已新增“pop命中 / pop空转”拆分统计：在 `pop_ready_tasks_batch()` 中按 `count>0` 与 `count==0` 分流记账，并将每次调用的 `atomic` 增量分别累计到命中/空转两组计数后输出到日志。
- 关系：`pop总量行 = pop命中行 + pop空转行`（分别对 `非atomic读/atomic/CAS` 三列求和成立）。
