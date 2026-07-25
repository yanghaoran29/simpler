# A5 跨 cluster `last_task_alive` 优化计划（衍生自 PR 906，待上机测量）

**Date**: 2026-07-25
**Verdict**: **D 已合入（正确性修复，实测无回归）**；**B 已实测 → orch 退步 ~13% → dropped**（把 orch 搬进 sched 所在 die 引入的 cache 争用 > 省下的跨 die 读开销；现有跨 die 隔离是有意为之）；**C 前提被 B 动摇，存疑、暂不做**。原 M5「跨 die `last_task_alive` 是瓶颈」的假设在 A5 本机负载上不成立。要重启 B/C 需先有 B-0 隔离微基准 + full-SKU（cube≥36）跑高任务量负载。

目标：消除 A5 默认布局下 orch 每任务跨 die acquire-load `last_task_alive` 的开销。

> **测量通用约定**（每方案复用）：`benchmark` skill 自动在 `merge-base upstream/main HEAD` 建 worktree 跑「优化前」；headline = **orch_cost**，watch = **sched_max / wire / effective**；噪声 ±2%，>5% 标记；`-n 100`，整组对比重复 3 次。命令里 `<N>` = 设备号。

```bash
# headline 时序（无需 profiling flag）
./tools/benchmark_rounds.sh -p a5 -d <N> -n 100 -r tensormap_and_ringbuffer
# 或用 skill 自动 worktree 对比：/benchmark -p a5 -d <N> -n 100
# 解析 trim-10 均值
python -m simpler_setup.tools.strace_timing <run.log> --rounds-table
```

---

## Scheme D — `cleanup_retired` 选择性释放（✅ 已实现 a5 + 单测；防御性正确性修复）

**根因（实现时查清，比计划里的「范围 cap」描述更准）**：slot 的 task 链 `task_entry_heads[slot]` 在 cleanup 落后于 ring 推进时会**混入「复用同 slot 的更晚任务」的 entry**——因为 `link_entry` 只在 per-run epoch 首次触碰 slot 时重置 head，slot 复用时并不重置。于是 `cleanup_retired` 退役某个 `local_id` 时，会把整条链（含仍存活的更晚任务 entry）一起 `free_entry`；唯一护栏是 `pto_tensormap.h` 原来的 `debug_assert(producer_task_id == local_id)`，release 下被编掉 → 静默损坏 tensormap。

**PR 906 的「范围 cap」不够**：即便把单次 `[old,new)` cap 到 `< task_window_size`，**第一次访问到这条已混合的链时仍会把存活 entry 释放掉**——cap 只防「同一 slot 在一次调用里被访问两次」，不防「链本身已被污染」。

**实际改法**：`cleanup_retired` 改为按 `producer_task_id == retired_task` 选择性释放——只 `free_entry` 属于当前 `local_id` 的 entry，并在释放时正确维护 task 双链（`prev_in_task`/`next_in_task`）与 head 指针；其它任务的 entry 留在链里。**无需改** `reclaim_retired_all` / `sync_tensormap` 的调用契约。

**回归屏障**：`tests/ut/cpp/a5/test_tensormap.cpp` 新增 `CleanupRetiredSparesLaterTaskReusingSlot`（task 0 与 task 0+WINDOW_SIZE 同 slot，退役 task 0 后断言 task 0+WINDOW_SIZE 的 entry 存活）。改前 debug 下抛 `AssertionError`、release 下静默 free → 失败；改后通过。`test_a5_tensormap` 全 33 例通过。

**a2a3 同病（✅ 已镜像修复）**：`src/a2a3/.../pto_tensormap.h:605` 有完全相同的 `cleanup_retired`，已做同样改动 + 镜像单测；`test_a2a3_tensormap` 33 例通过、a2a3sim 重编通过。

- **优化前/优化后（✅ 已上机实测）**：本机设备是 partial-good（cube=28），`paged_attention_unroll/Case1`（block_dim=36）放不下，改用同走 tmr 退役/cleanup 路径的 `spmd_multiblock_mix`（block_dim=24, aicpu_thread_num=4），100 rounds、固定 device 0、`task-submit` 锁卡，优化前/后各跑一次（stash 改动重编 onboard runtime 作为 before）。trim-10 均值对比：

  | 指标 | before | after | Δ |
  |---|---|---|---|
  | Device(trim) | 141.6us | 142.5us | +0.6% |
  | Effective | 117.3us | 117.1us | −0.2% |
  | Orch | 85.8us | 84.7us | −1.3% |
  | Sched | 116.0us | 115.3us | −0.6% |

  全部 |Δ| < 2%，方向互有正负 → **噪声内，无回归、无收益**，与「D 是正确性 no-op 修复」的预期一致。
- **判据（已满足）**：正确性——`test_a5_tensormap` 33 例全过 + `a5sim mixed_example` 场景通过 + 上机 `spmd_multiblock_mix` 跑通；性能——三项运行时指标 ±2% 内。注：`paged_attention_unroll/Case1` / 全套 `TMR_EXAMPLE_CASES` 需 full-SKU（cube≥36）设备才能跑，本机暂无；留作换到 full-SKU 机器时的补充验证。

## Scheme B-0 — 跨 die 硬件 ceiling 微基准（skipped）

用户决定跳过微基准、直接做 B+实测（认为 B before/after 更接近真问题）。**事后看是教训**：B-0 能在隔离环境高 SNR 量出跨 die 硬件成本，本可避免下面 B 在低 SNR 工作负载上「同配置 ~10% 抖动、差点判错」的反复。下次类似场景先做隔离微基准。

## Scheme B — placement 修复（❌ 实测 orch 退步 ~13%，dropped）

**改（已实现后丢弃）**：在 `compute_allowed_cpus`（`aicpu_topology_probe.cpp`）现有 sched-first 逻辑后加一个 rebalance swap——若 orch 落在与 sched 多数 die 不同的 die，就把它和 sched 多数 die 里的一个 sched 交换。Scenario A：`ALLOWED_CPUS` 从 `{5,6,7,8,3}`（orch=cpu3 die0）变成 `{3,6,7,8,5}`（orch=cpu5 die1，与 3 个 sched 同 die）。最小改动、复用全部现有逻辑、orch 已同 die 时 no-op（full-SKU 单 cluster 放得下时不触发）。

**实测（a5 onboard, `spmd_multiblock_mix` Case1, 100 rounds, device 0, task-submit）**，每组 2 次重复：

| config | Orch run1 | Orch run2 | Orch mean |
|---|---|---|---|
| main placement `{5,6,7,8,3}` | 78.1us | 70.5us | ~74.3us |
| Scheme B `{3,6,7,8,5}` | 86.4us | 81.6us | ~84.0us |

B 的 orch_cost **高 ~13%、两次重复区间不重叠**（main ≤78.1 < B ≥81.6），不是噪声。Sched/Device 同向小幅退步。

**为什么变差**：把 orch 搬进 sched 所在 die，省下了 orch 跨 die 读 `last_task_alive` 的开销，但**引入了 orch 与 sched 的 L3/cache 争用**（orch 自己的 tensormap 查找 + 依赖 wiring 与 sched 抢 die1 资源），后者 > 前者。说明现有「orch 跨 die 隔离」是**有意为之**（隔离 orch 工作集），不是疏忽。同配置 ~10% 的 run-to-run 抖动也提示：本机 PG 设备 + `spmd_multiblock_mix`（Orch ~70-86us）SNR 偏低，但 B 的 gap 大于该抖动。

**判据触发「不合入」**（orch 退步 >5%）。代码已丢弃、不提交（rebalance swap 的描述留在此处，供 full-SKU 上复核时快速重写）。

**局限**：只测了本机能放的 `spmd_multiblock_mix`；高任务量负载（`paged_attention` Case1，需 full-SKU cube≥36）上 co-location 争用 vs 跨 die 读谁占优未测。但 co-location 争用机制是 workload-general 的，B 大概率在各负载上都偏负。

## Scheme C — 软件单写者 lazy publish（⚠️ 存疑，前提被 B 动摇）

原计划（依赖 D+B）：host→device 加 `publisher_sched_idx`；非 publisher 跳过 `sync_to_sm` 的 SM store；publisher 每 iter 条件 publish；orch 侧靠 `ensure_tensormap_capacity` 的 500ms backstop 容忍旧值。publish/read 仍 `stlr`/`ldar`，不需新屏障。

**现状**：B 已证「把 `last_task_alive` 的生产/消费搬到同 die 会因 cache 争用变慢」，C 的前提（跨 die 写是瓶颈）因此存疑。在拿出反证前**不做 C**。要重启 C，先得回答两件事：(1) 跨 die `last_task_alive` 往返的真实硬件成本（B-0 微基准）；(2) 高任务量负载（`paged_attention` Case1，需 full-SKU）上 co-location 争用 vs 跨 die 读谁占优。

---

## 背景（一句话）

PR 906「[WIP] A5 performance opt」整体不可直接合（WIP、自 6/11 停更、被 main 走偏、A5 无实测收益）。其 M5 思路（限制 `last_task_alive` 的 SM 写者为单一与 orch 同 cluster/die 的线程）**看起来**正中要害——A5 默认 `ALLOWED_CPUS={5,6,7,8,3}` 让 orch 与全部 sched 跨 die，每任务一次跨 die cache line 往返。但 Scheme B 的实测（见下）表明：在 A5 本机负载上，把 orch 搬到与 sched 同 die 引入的 cache 争用 > 省下的跨 die 读开销，前提并不成立。

## 不做（main 已有或已移除，避免重复踩坑）

- wiring 线程 + scheduler 侧 SPSC wiring queue（#1263 已把 fanin wiring 移到 orchestrator 并删除该 queue）
- `PTO2FaninPool` / `PTO2FaninBuilder`（`pto_orchestrator.cpp:230-261`、`pto_ring_buffer.h:500-570` 已有）
- `new_entry()` spin / assert 加固（`ensure_tensormap_capacity` `pto_orchestrator.cpp:708-793` 已 spin + 500ms backstop）
- `cluster_mode` env（main 不存在，不复活）
- `g_only_thread0_advances` 原样（A5 publisher 应是「与 orch 同 die 的 sched」，非固定 thread0）
