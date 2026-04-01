# Orchestrator 线程指令数量汇总及性能分析——Qwen3

## 0. 测试环境信息

本方案在 host CPU 上通过 QEMU user-mode + insn\_count 插件统计 guest 动态指令数；测试样例为 qwen3，索引 0，模拟 AICore 在接到任务后立即返回。 

## 1. 指令数量汇总

### 1.1 总指令数

| 项目 | 指令数 |
|---|---:|
| build\_graph | 1 082 815 620 |
| pto2\_submit\_mixed\_task（合计） | 902 841 735 |
| 其他部分（build\_graph − submit 合计） | 179 973 885 |

### 1.2 平均每个任务的指令数

| 指标 | 数值 |
|---|---:|
| 任务总数 | 738 048 |
| 平均每任务指令数（含 build\_graph 摊销口径，与报告一致） | 1 467.13 |
| 平均每次 pto2\_submit\_mixed\_task 指令数 | 1 223.28 |

### 1.3 平均每阶段的指令数

下表展示 pto2\_submit\_mixed\_task 各阶段的指令数量统计。其中 alloc、sync、lookup、insert、params、fanin 六段与 Profiling 打点区间一致；others 表示上述六段之外、仍计入 submit 窗口内的指令。

others 典型包含：快速返回路径（如 alloc 前 `args.has_error`）、scope 死锁预检（`scope_tasks_size` 与 `window_size`）、计时器 `CYCLE_COUNT_LAP_RECORD(...)`、以及为统计插入的 marker 指令等。

| 阶段 | 平均指令数 | 最小值 | 最大值 |
|---|---:|---:|---:|
| alloc | 160.359 | 151 | 192 |
| sync | 71.683 | 35 | 10 379 |
| lookup | 368.476 | 93 | 1 077 |
| insert | 134.096 | 91 | 475 |
| params | 189.262 | 102 | 589 |
| fanin | 182.924 | 105 | 262 |
| others | 116.482 | 103 | 231 |
| submit\_total | 1 223.283 | 684 | 11 676 |

## 2. 各步骤指令数量分析

- **alloc**：执行 `TaskAllocator::alloc(total_output_size)`（槽位 + 打包输出区，可能自旋等待槽位）；随后按 `tensor_count / scalar_count` 对 payload 与 dispatch\_args 做 `__builtin_prefetch`；若有 scheduler，初始化 `PTO2TaskSlotState`（fanin/fanout 初值、scope\_tasks\_push），并分配 `fanin_states[] / fanin_count=0`。

- **sync**：包含 `last_task_alive` 的 acquire 读、`tensor_map.sync_tensormap`、以及（有 scheduler 时）`dep_pool.reclaim`。
  - 短路径（约 34 条）：`sync_tensormap` 内先 `sync_validity`（把共享内存里的阈值写入 `last_task_alives[ring_id]`），若 `sm_last_task_alive - last_cleanup[ring_id] < PTO2_TENSORMAP_CLEANUP_INTERVAL`（64）则不进入 `cleanup_retired`；再加调用/分支与通常很轻的 reclaim，总指令落在几十条量级。
  - 长路径（可达数千条）：当进入 `cleanup_retired` 时，会对 `[old_last, new_last)` 内每个退役 `local_id` 沿 `task_entry_heads` 链表 `free_entry`，从 hash 桶双向链摘除并回收入池；退役任务挂的 TensorMap 条目越多，单次窗口内指令越多。

- **lookup**：对 `args.tensor_count` 做一轮 switch：INPUT/INOUT（且非 `manual_dep`）调用 `tensor_map.lookup`，在结果上去重写入 `fanin_states[]`，并对 INOUT + COVERED 且 `!with_alloc` 调用 `remove_entry`；OUTPUT 分支只做 `materialize_output` 与 offset 累加（不写 TensorMap）。
  - 差异来源：INPUT/INOUT 个数、lookup 桶链长度、`lookup_result.count`、fanin 去重内层循环、是否触发 `remove_entry`（日志中 UNKNOWN / 多 INOUT 类任务路径更重）；纯 OUTPUT 多的任务此段相对短。

- **insert**：遍历 tensor，对 OUTPUT / INOUT（且非 `manual_dep`）取 Tensor 后调用 `tensor_map.insert`（`needs_alloc[i]` 区分是否带 alloc 语义）。
  - 差异来源：可插入条目数 ≈ OUTPUT + 参与依赖的 INOUT；每次 insert 含 hash、new\_entry、桶头插入、task\_entry\_heads 链表维护（见 `pto_tensormap.h::insert`），故 SOFTMAX 等多输出任务明显高于 QK 等少输出任务。

- **params**：写 `PTO2TaskDescriptor`（`task_id`、三槽 `kernel_id`、`packed_buffer` 边界等）；可选 `__builtin_prefetch` 当前槽与各 `fanin_states[i]`；核心为 `payload->init(args, result)`（按 tensor/scalar 把参数拷入 GM 侧 payload）。
  - 差异来源：`tensor_count + scalar_count` 决定的 Arg 拷贝规模；与 fanin 相关的仅 prefetch 次数（随 `fanin_count`），主体仍是 init 内字段写入与可能的张量元数据拷贝。

- **fanin**：初始化本任务 `task_state / fanout_refcount`，`dep_pool.ensure_space`（必要时内部 reclaim 自旋），对每个 producer 执行 `pto2_fanout_lock` → 维护 `fanout_count / 读 task_state` → `dep_pool.prepend` 或早完成分支 → `pto2_fanout_unlock`，最后 `fanin_refcount.fetch_add` 与可能的 `ready_queues` 入队，并写 `dep_pool_mark`。
  - 差异来源：`fanin_count` 线性放大 锁 + 链表 + 原子；producer 已完成的早完成路径略省 prepend；`ensure_space` 遇压力时与 alloc/sync 类似会拉长窗口。

## 3. 主要汇编指令分布

完整指令数量分布见指令数量统计文档。

| 指令 | 次数 | 占比（相对 submit\_total 总指令） | 功能说明 |
|---|---:|---:|---|
| ldr | 194 514 177 | 20.64% | 通用加载：从内存读取数据到寄存器 |
| add | 133 026 736 | 14.12% | 加法运算：地址偏移、计数累加与结果计算 |
| cmp | 81 829 911 | 8.68% | 比较运算：更新条件标志供后续分支判断 |
| str | 69 122 687 | 7.34% | 通用存储：将寄存器数据写回内存 |
| mov | 63 368 780 | 6.73% | 寄存器赋值/搬移：参数准备与临时值传递 |
| cbnz | 32 077 647 | 3.40% | 非零分支：寄存器不为 0 时跳转 |
| b.eq | 24 617 001 | 2.61% | 条件跳转：相等时分支（Z 标志置位） |
| cbz | 24 175 623 | 2.57% | 零分支：寄存器为 0 时跳转 |
| ldrb | 22 417 975 | 2.38% | 字节加载：读取 8-bit 数据并零扩展 |
| ldp | 21 480 904 | 2.28% | 成对加载：一次读取两个寄存器（常见于栈帧恢复） |
| stp | 21 365 248 | 2.27% | 成对存储：一次写回两个寄存器（常见于栈帧/批量写） |
| b.gt | 19 803 946 | 2.10% | 条件跳转：大于时分支（有符号比较） |
| sub | 16 168 260 | 1.72% | 减法运算：地址回退、计数递减、边界计算 |
| b.ne | 15 805 496 | 1.68% | 条件跳转：不相等时分支（Z 标志清零） |
| b | 14 882 017 | 1.58% | 无条件跳转：改变控制流 |
| b.le | 13 210 321 | 1.40% | 条件跳转：小于等于时分支（有符号比较） |
| ubfx | 12 100 405 | 1.28% | 位域提取：提取指定位域并零扩展 |
| prfm | 10 782 965 | 1.14% | 预取提示：提前拉取内存到缓存，降低访存延迟 |
| movk | 10 110 336 | 1.07% | 立即数拼接：写入寄存器高位常量（与 movz/movn 配合构造常量） |
| b.hi | 10 048 449 | 1.07% | 条件跳转：无符号"大于"时分支 |

## 4. 具体各任务类型各阶段指令数量

统计项为各任务类型在总体与各阶段上的平均 / 最小 / 最大指令数。

- submit 任务名优先使用 `aiv0_kernel_id`（即 `pto2_rt_submit_aiv_task(kernel_id, ...)` 的 kernel_id）。
- 非 AIV-only 提交会回退显示为 AIC/MIXED 内核标识。

| submit 任务名 | 任务描述 | 样本数 | 参数类型（input, output, inout, scalar） |
|---|---|---:|---|
| `AIV_KERNEL_0` | RMS 统计 + 逆均方根准备（prefill 前置） | 128 | `(1, 2, 0, 3)` |
| `AIV_KERNEL_1` | RMSNorm（输入） | 128 | `(3, 1, 0, 3)` |
| `AIV_KERNEL_3` | Q 投影结果整理/写回（AIV） | 1 280 | `(2, 1, 0, 1)` |
| `AIV_KERNEL_5` | K/V 投影结果整理/写回（AIV） | 256 | `(3, 2, 0, 1)` |
| `AIV_KERNEL_6` | 初始化注意力输出 tile | 128 | `(0, 1, 0, 0)` |
| `AIV_KERNEL_7` | 提取当前 token 的 K group | 2 048 | `(1, 1, 0, 1)` |
| `AIV_KERNEL_8` | RoPE + 写 KV Cache | 2 048 | `(6, 0, 2, 3)` |
| `AIV_KERNEL_9` | 初始化单行注意力累加器 | 2 048 | `(0, 1, 1, 0)` |
| `AIV_KERNEL_10` | 提取当前 token 的 Q group | 16 384 | `(1, 1, 0, 2)` |
| `AIV_KERNEL_11` | Q 旋转编码（RoPE） | 16 384 | `(5, 1, 0, 0)` |
| `AIV_KERNEL_12` | 初始化在线 softmax 状态（li/mi/oi） | 16 384 | `(0, 6, 0, 0)` |
| `AIC_KERNEL_13` | QK MatMul（分块上下文） | 24 576 | `(2, 1, 0, 2)` |
| `AIV_KERNEL_14` | Softmax 准备（max/sum/exp） | 24 576 | `(1, 3, 0, 1)` |
| `AIC_KERNEL_15` | PV MatMul（分块上下文） | 24 576 | `(2, 1, 0, 2)` |
| `AIV_KERNEL_16` | 在线 softmax 状态更新（融合） | 24 576 | `(3, 0, 3, 1)` |
| `AIV_KERNEL_17` | 回写注意力行（按头组） | 16 384 | `(2, 0, 1, 1)` |
| `AIV_KERNEL_18` | 取 hidden 残差分支切片 | 128 | `(1, 1, 0, 3)` |
| `AIV_KERNEL_20` | WO 后处理 + residual add（AIV） | 1 280 | `(3, 1, 0, 1)` |
| `AIV_KERNEL_21` | RMS 统计 + 逆均方根准备（post-attn） | 128 | `(1, 2, 0, 0)` |
| `AIV_KERNEL_22` | Post-RMSNorm + 初始化 down\_proj | 128 | `(3, 2, 0, 0)` |
| `AIV_KERNEL_24` | FFN 激活与 chunk 融合（AIV） | 51 200 | `(3, 3, 0, 1)` |
| `AIV_KERNEL_26` | FFN down 累加回写（AIV） | 512 000 | `(2, 0, 1, 2)` |
| `AIV_KERNEL_27` | 最终输出回写（residual + down\_proj） | 1 280 | `(2, 0, 1, 3)` |

### 4.1 逐任务阶段统计明细



#### AIV_KERNEL_0 (样本数=128)

- 对应 submit 任务名: `AIV_KERNEL_0`
- kernel_id集合: `aiv0=[0], aiv1=[-1], aic=[-1]`
- 参数类型覆盖(input,output,inout,scalar): `[(1, 2, 0, 3)]`

| 指标 | 平均 | 最小 | 最大 |
|---|---:|---:|---:|
| submit_total | 1375.141 | 982 | 5620 |
| alloc | 171.914 | 161 | 172 |
| sync | 425.164 | 35 | 4670 |
| lookup | 197.000 | 197 | 197 |
| insert | 171.062 | 171 | 179 |
| params | 165.000 | 165 | 165 |
| fanin | 105.000 | 105 | 105 |
| others | 140.000 | 140 | 140 |

#### AIV_KERNEL_1 (样本数=128)

- 对应 submit 任务名: `AIV_KERNEL_1`
- kernel_id集合: `aiv0=[1], aiv1=[-1], aic=[-1]`
- 参数类型覆盖(input,output,inout,scalar): `[(3, 1, 0, 3)]`

| 指标 | 平均 | 最小 | 最大 |
|---|---:|---:|---:|
| submit_total | 1063.141 | 1052 | 1077 |
| alloc | 176.742 | 166 | 177 |
| sync | 35.000 | 35 | 35 |
| lookup | 255.367 | 255 | 269 |
| insert | 115.031 | 115 | 119 |
| params | 193.000 | 193 | 193 |
| fanin | 157.000 | 157 | 157 |
| others | 131.000 | 131 | 131 |

#### AIV_KERNEL_3 (样本数=1280)

- 对应 submit 任务名: `AIV_KERNEL_3`
- kernel_id集合: `aiv0=[3], aiv1=[3], aic=[2]`
- 参数类型覆盖(input,output,inout,scalar): `[(2, 1, 0, 1)]`

| 指标 | 平均 | 最小 | 最大 |
|---|---:|---:|---:|
| submit_total | 1030.545 | 972 | 2768 |
| alloc | 168.795 | 161 | 172 |
| sync | 85.627 | 35 | 1820 |
| lookup | 228.086 | 228 | 239 |
| insert | 107.038 | 107 | 111 |
| params | 165.000 | 165 | 165 |
| fanin | 155.000 | 155 | 155 |
| others | 121.000 | 121 | 121 |

#### AIV_KERNEL_5 (样本数=256)

- 对应 submit 任务名: `AIV_KERNEL_5`
- kernel_id集合: `aiv0=[5], aiv1=[5], aic=[4]`
- 参数类型覆盖(input,output,inout,scalar): `[(3, 2, 0, 1)]`

| 指标 | 平均 | 最小 | 最大 |
|---|---:|---:|---:|
| submit_total | 1269.523 | 1259 | 1281 |
| alloc | 181.355 | 171 | 182 |
| sync | 35.000 | 35 | 35 |
| lookup | 332.086 | 332 | 343 |
| insert | 187.082 | 187 | 195 |
| params | 225.000 | 225 | 225 |
| fanin | 155.000 | 155 | 155 |
| others | 154.000 | 154 | 154 |

#### AIV_KERNEL_6 (样本数=128)

- 对应 submit 任务名: `AIV_KERNEL_6`
- kernel_id集合: `aiv0=[6], aiv1=[-1], aic=[-1]`
- 参数类型覆盖(input,output,inout,scalar): `[(0, 1, 0, 0)]`

| 指标 | 平均 | 最小 | 最大 |
|---|---:|---:|---:|
| submit_total | 684.031 | 684 | 688 |
| alloc | 151.000 | 151 | 151 |
| sync | 35.000 | 35 | 35 |
| lookup | 93.000 | 93 | 93 |
| insert | 91.031 | 91 | 95 |
| params | 102.000 | 102 | 102 |
| fanin | 105.000 | 105 | 105 |
| others | 107.000 | 107 | 107 |

#### AIV_KERNEL_7 (样本数=2048)

- 对应 submit 任务名: `AIV_KERNEL_7`
- kernel_id集合: `aiv0=[7], aiv1=[-1], aic=[-1]`
- 参数类型覆盖(input,output,inout,scalar): `[(1, 1, 0, 1)]`

| 指标 | 平均 | 最小 | 最大 |
|---|---:|---:|---:|
| submit_total | 7171.936 | 766 | 10277 |
| alloc | 166.984 | 156 | 167 |
| sync | 6432.011 | 35 | 9539 |
| lookup | 121.405 | 120 | 209 |
| insert | 99.016 | 99 | 103 |
| params | 133.037 | 133 | 137 |
| fanin | 105.482 | 105 | 157 |
| others | 114.000 | 114 | 114 |

#### AIV_KERNEL_8 (样本数=2048)

- 对应 submit 任务名: `AIV_KERNEL_8`
- kernel_id集合: `aiv0=[8], aiv1=[-1], aic=[-1]`
- 参数类型覆盖(input,output,inout,scalar): `[(6, 0, 2, 3)]`

| 指标 | 平均 | 最小 | 最大 |
|---|---:|---:|---:|
| submit_total | 1629.091 | 1427 | 3415 |
| alloc | 181.172 | 181 | 192 |
| sync | 35.934 | 35 | 1947 |
| lookup | 313.374 | 232 | 555 |
| insert | 213.004 | 213 | 221 |
| params | 584.191 | 483 | 589 |
| fanin | 156.416 | 105 | 183 |
| others | 145.000 | 145 | 145 |

#### AIV_KERNEL_9 (样本数=2048)

- 对应 submit 任务名: `AIV_KERNEL_9`
- kernel_id集合: `aiv0=[9], aiv1=[-1], aic=[-1]`
- 参数类型覆盖(input,output,inout,scalar): `[(0, 1, 1, 0)]`

| 指标 | 平均 | 最小 | 最大 |
|---|---:|---:|---:|
| submit_total | 1053.571 | 971 | 1133 |
| alloc | 156.000 | 156 | 156 |
| sync | 35.000 | 35 | 35 |
| lookup | 224.865 | 192 | 293 |
| insert | 164.768 | 161 | 173 |
| params | 201.938 | 156 | 205 |
| fanin | 157.000 | 157 | 157 |
| others | 114.000 | 114 | 114 |

#### AIV_KERNEL_10 (样本数=16384)

- 对应 submit 任务名: `AIV_KERNEL_10`
- kernel_id集合: `aiv0=[10], aiv1=[-1], aic=[-1]`
- 参数类型覆盖(input,output,inout,scalar): `[(1, 1, 0, 2)]`

| 指标 | 平均 | 最小 | 最大 |
|---|---:|---:|---:|
| submit_total | 761.678 | 761 | 776 |
| alloc | 156.000 | 156 | 156 |
| sync | 35.000 | 35 | 35 |
| lookup | 120.661 | 120 | 131 |
| insert | 99.018 | 99 | 103 |
| params | 133.000 | 133 | 133 |
| fanin | 105.000 | 105 | 105 |
| others | 113.000 | 113 | 113 |

#### AIV_KERNEL_11 (样本数=16384)

- 对应 submit 任务名: `AIV_KERNEL_11`
- kernel_id集合: `aiv0=[11], aiv1=[-1], aic=[-1]`
- 参数类型覆盖(input,output,inout,scalar): `[(5, 1, 0, 0)]`

| 指标 | 平均 | 最小 | 最大 |
|---|---:|---:|---:|
| submit_total | 1484.367 | 1386 | 1544 |
| alloc | 176.000 | 176 | 176 |
| sync | 35.000 | 35 | 35 |
| lookup | 310.116 | 309 | 369 |
| insert | 131.018 | 131 | 135 |
| params | 529.234 | 432 | 530 |
| fanin | 156.999 | 139 | 157 |
| others | 146.000 | 146 | 146 |

#### AIV_KERNEL_12 (样本数=16384)

- 对应 submit 任务名: `AIV_KERNEL_12`
- kernel_id集合: `aiv0=[12], aiv1=[-1], aic=[-1]`
- 参数类型覆盖(input,output,inout,scalar): `[(0, 6, 0, 0)]`

| 指标 | 平均 | 最小 | 最大 |
|---|---:|---:|---:|
| submit_total | 1738.104 | 1738 | 1764 |
| alloc | 176.000 | 176 | 176 |
| sync | 35.000 | 35 | 35 |
| lookup | 478.000 | 478 | 478 |
| insert | 451.104 | 451 | 477 |
| params | 262.000 | 262 | 262 |
| fanin | 105.000 | 105 | 105 |
| others | 231.000 | 231 | 231 |

#### AIC_KERNEL_13 (样本数=24576)

- 对应 submit 任务名: `AIC_KERNEL_13`
- kernel_id集合: `aiv0=[-1], aiv1=[-1], aic=[13]`
- 参数类型覆盖(input,output,inout,scalar): `[(2, 1, 0, 2)]`

| 指标 | 平均 | 最小 | 最大 |
|---|---:|---:|---:|
| submit_total | 1092.555 | 1074 | 5980 |
| alloc | 161.000 | 161 | 172 |
| sync | 35.198 | 35 | 4912 |
| lookup | 320.337 | 320 | 344 |
| insert | 107.020 | 107 | 111 |
| params | 169.000 | 169 | 169 |
| fanin | 182.999 | 165 | 183 |
| others | 117.000 | 117 | 117 |

#### AIV_KERNEL_14 (样本数=24576)

- 对应 submit 任务名: `AIV_KERNEL_14`
- kernel_id集合: `aiv0=[14], aiv1=[-1], aic=[-1]`
- 参数类型覆盖(input,output,inout,scalar): `[(1, 3, 0, 1)]`

| 指标 | 平均 | 最小 | 最大 |
|---|---:|---:|---:|
| submit_total | 1321.593 | 1303 | 11676 |
| alloc | 166.000 | 166 | 177 |
| sync | 35.421 | 35 | 10379 |
| lookup | 355.115 | 355 | 378 |
| insert | 243.058 | 243 | 257 |
| params | 201.000 | 201 | 201 |
| fanin | 156.999 | 139 | 168 |
| others | 164.000 | 164 | 164 |

#### AIC_KERNEL_15 (样本数=24576)

- 对应 submit 任务名: `AIC_KERNEL_15`
- kernel_id集合: `aiv0=[-1], aiv1=[-1], aic=[15]`
- 参数类型覆盖(input,output,inout,scalar): `[(2, 1, 0, 2)]`

| 指标 | 平均 | 最小 | 最大 |
|---|---:|---:|---:|
| submit_total | 1092.348 | 1074 | 1125 |
| alloc | 161.000 | 161 | 161 |
| sync | 35.000 | 35 | 35 |
| lookup | 320.328 | 320 | 352 |
| insert | 107.020 | 107 | 112 |
| params | 169.000 | 169 | 169 |
| fanin | 182.999 | 165 | 204 |
| others | 117.000 | 117 | 117 |

#### AIV_KERNEL_16 (样本数=24576)

- 对应 submit 任务名: `AIV_KERNEL_16`
- kernel_id集合: `aiv0=[16], aiv1=[-1], aic=[-1]`
- 参数类型覆盖(input,output,inout,scalar): `[(3, 0, 3, 1)]`

| 指标 | 平均 | 最小 | 最大 |
|---|---:|---:|---:|
| submit_total | 1860.677 | 1740 | 2152 |
| alloc | 167.000 | 167 | 167 |
| sync | 35.000 | 35 | 35 |
| lookup | 805.663 | 695 | 1077 |
| insert | 253.014 | 253 | 265 |
| params | 254.333 | 253 | 257 |
| fanin | 217.668 | 209 | 262 |
| others | 128.000 | 128 | 128 |

#### AIV_KERNEL_17 (样本数=16384)

- 对应 submit 任务名: `AIV_KERNEL_17`
- kernel_id集合: `aiv0=[17], aiv1=[-1], aic=[-1]`
- 参数类型覆盖(input,output,inout,scalar): `[(2, 0, 1, 1)]`

| 指标 | 平均 | 最小 | 最大 |
|---|---:|---:|---:|
| submit_total | 1556.467 | 1447 | 1637 |
| alloc | 152.000 | 152 | 152 |
| sync | 35.000 | 35 | 35 |
| lookup | 680.782 | 621 | 758 |
| insert | 108.750 | 105 | 109 |
| params | 240.938 | 195 | 244 |
| fanin | 234.998 | 218 | 235 |
| others | 104.000 | 104 | 104 |

#### AIV_KERNEL_18 (样本数=128)

- 对应 submit 任务名: `AIV_KERNEL_18`
- kernel_id集合: `aiv0=[18], aiv1=[-1], aic=[-1]`
- 参数类型覆盖(input,output,inout,scalar): `[(1, 1, 0, 3)]`

| 指标 | 平均 | 最小 | 最大 |
|---|---:|---:|---:|
| submit_total | 763.672 | 763 | 789 |
| alloc | 156.000 | 156 | 156 |
| sync | 35.000 | 35 | 35 |
| lookup | 120.641 | 120 | 146 |
| insert | 99.031 | 99 | 100 |
| params | 133.000 | 133 | 133 |
| fanin | 105.000 | 105 | 105 |
| others | 115.000 | 115 | 115 |

#### AIV_KERNEL_20 (样本数=1280)

- 对应 submit 任务名: `AIV_KERNEL_20`
- kernel_id集合: `aiv0=[20], aiv1=[20], aic=[19]`
- 参数类型覆盖(input,output,inout,scalar): `[(3, 1, 0, 1)]`

| 指标 | 平均 | 最小 | 最大 |
|---|---:|---:|---:|
| submit_total | 1849.907 | 1164 | 8876 |
| alloc | 175.883 | 166 | 177 |
| sync | 711.014 | 35 | 7736 |
| lookup | 341.000 | 341 | 341 |
| insert | 115.010 | 115 | 116 |
| params | 197.000 | 197 | 197 |
| fanin | 181.000 | 181 | 181 |
| others | 129.000 | 129 | 129 |

#### AIV_KERNEL_21 (样本数=128)

- 对应 submit 任务名: `AIV_KERNEL_21`
- kernel_id集合: `aiv0=[21], aiv1=[-1], aic=[-1]`
- 参数类型覆盖(input,output,inout,scalar): `[(1, 2, 0, 0)]`

| 指标 | 平均 | 最小 | 最大 |
|---|---:|---:|---:|
| submit_total | 1108.008 | 1108 | 1109 |
| alloc | 161.000 | 161 | 161 |
| sync | 35.000 | 35 | 35 |
| lookup | 278.000 | 278 | 278 |
| insert | 171.008 | 171 | 172 |
| params | 166.000 | 166 | 166 |
| fanin | 157.000 | 157 | 157 |
| others | 140.000 | 140 | 140 |

#### AIV_KERNEL_22 (样本数=128)

- 对应 submit 任务名: `AIV_KERNEL_22`
- kernel_id集合: `aiv0=[22], aiv1=[-1], aic=[-1]`
- 参数类型覆盖(input,output,inout,scalar): `[(3, 2, 0, 0)]`

| 指标 | 平均 | 最小 | 最大 |
|---|---:|---:|---:|
| submit_total | 1375.172 | 1375 | 1386 |
| alloc | 171.000 | 171 | 171 |
| sync | 35.000 | 35 | 35 |
| lookup | 418.172 | 418 | 429 |
| insert | 187.000 | 187 | 187 |
| params | 226.000 | 226 | 226 |
| fanin | 183.000 | 183 | 183 |
| others | 155.000 | 155 | 155 |

#### AIV_KERNEL_24 (样本数=51200)

- 对应 submit 任务名: `AIV_KERNEL_24`
- kernel_id集合: `aiv0=[24], aiv1=[24], aic=[23]`
- 参数类型覆盖(input,output,inout,scalar): `[(3, 3, 0, 1)]`

| 指标 | 平均 | 最小 | 最大 |
|---|---:|---:|---:|
| submit_total | 1486.257 | 1471 | 3244 |
| alloc | 181.398 | 176 | 187 |
| sync | 44.753 | 35 | 1797 |
| lookup | 409.067 | 409 | 435 |
| insert | 259.039 | 259 | 262 |
| params | 257.000 | 257 | 257 |
| fanin | 155.000 | 155 | 155 |
| others | 180.000 | 180 | 180 |

#### AIV_KERNEL_26 (样本数=512000)

- 对应 submit 任务名: `AIV_KERNEL_26`
- kernel_id集合: `aiv0=[26], aiv1=[26], aic=[25]`
- 参数类型覆盖(input,output,inout,scalar): `[(2, 0, 1, 2)]`

| 指标 | 平均 | 最小 | 最大 |
|---|---:|---:|---:|
| submit_total | 1128.276 | 911 | 2970 |
| alloc | 156.841 | 152 | 163 |
| sync | 59.338 | 35 | 1928 |
| lookup | 347.566 | 180 | 492 |
| insert | 105.000 | 105 | 106 |
| params | 166.405 | 161 | 169 |
| fanin | 190.126 | 155 | 228 |
| others | 103.000 | 103 | 103 |

#### AIV_KERNEL_27 (样本数=1280)

- 对应 submit 任务名: `AIV_KERNEL_27`
- kernel_id集合: `aiv0=[27], aiv1=[-1], aic=[-1]`
- 参数类型覆盖(input,output,inout,scalar): `[(2, 0, 1, 3)]`

| 指标 | 平均 | 最小 | 最大 |
|---|---:|---:|---:|
| submit_total | 1076.246 | 919 | 2551 |
| alloc | 160.138 | 152 | 163 |
| sync | 46.687 | 35 | 1508 |
| lookup | 312.820 | 178 | 435 |
| insert | 108.500 | 108 | 109 |
| params | 164.347 | 161 | 169 |
| fanin | 178.755 | 157 | 209 |
| others | 105.000 | 105 | 105 |
