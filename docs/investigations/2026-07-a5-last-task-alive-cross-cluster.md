# A5 跨 cluster `last_task_alive` 优化计划（衍生自 PR 906，待上机测量）

**Date**: 2026-07-25
**Verdict**: deferred-pending-A5-onboard-measurement（4 个方案按 D→B-0→B→C 推进，每项落地前先跑「优化前 vs 优化后」真机对比）。**Scheme D 已实现 a5 + 单测通过 + 上机 before/after 实测（无回归，见下）**；B-0/B/C 仍待上机测量。

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

## Scheme B-0 — 跨 die 硬件 ceiling 微基准（不改运行时代码）

**做**：`tools/cann-examples/`（与 `aicpu-thread-spread/` 同级）加两线程 acquire-load/release-store 往返微基准，按 `allowed_cpus` 钉 same-cluster / same-die-other-cluster / other-die 三种放置，量往返 cycle。先量出 B 能恢复的硬件上限。

- **看**：同 die vs 跨 die 往返 cycle 差 = Scheme B 的可恢复上限。
- **判据**：跨 die 显著贵（>数十 ns）→ 继续 B；几乎无差 → 跳过 B、直接评估 C。

## Scheme B — placement 修复（affinity 层，最便宜）

**改**：`aicpu_topology_probe.cpp:294-364` `compute_allowed_cpus` 改两遍——先在 sched 多数 die `D*` 用 `pick_lowest_for_orch` 给 orch 预留槽，再用剩余（保 SMT 配对排序）+ `fit_spread` 填 sched。输出仍按 `[sched…, orch]` 契约（`:359-362`）。结果：orch + 3 sched 在 die1，1 sched 溢出 die0。无条件合入。

- **优化前**：merge-base baseline（skill 自动）。
- **优化后**：两遍 placement 改动后重装。
- **看**：orch_cost（headline）+ sched_max（抓溢出 sched 的跨 die CAS 回归）。
- **判据**：orch_cost 改善 >5% 且 sched_max ±2% → 合入；sched_max 退步 >5% → 不合入。

## Scheme C — 软件单写者 lazy publish（M5 适配，依赖 D+B）

**改**：
1. host→device 新字段 `publisher_sched_idx`（`device_runner.cpp:250-253` 附近下发，= 与 orch 同 die 的 sched 的 `allowed_cpus[]` index）。
2. `aicpu_executor.cpp:486` `run()` 里算 `is_publisher_sched`（函数局部 bool）。
3. `pto_scheduler.h:464/479`：非 publisher 跳过 `sync_to_sm` 的 SM store，但仍推进共享 local `last_task_alive`（把现有 `thread_idx` 传参改无条件）。
4. `scheduler_dispatch.cpp:~888`（completion 后）：publisher 每 iter 条件 publish（每 ring relaxed load + 变化时一个 release store）。

orch 侧无需新增容忍逻辑（`ensure_tensormap_capacity` 已有 500ms backstop）。publish/read 仍 `stlr`/`ldar`，不需新屏障；MMIO COND 读取不在 publish 路径，`dmb ishld` 不适用。

- **优化前**：**B tip**（隔离变量，非 merge-base）。
- **优化后**：D+C 重装；额外用 `CXXFLAGS="-DSIMPLER_TENSORMAP_PROFILING=1"` 重装看查找链是否变长。
- **看**：orch_cost（headline）+ wire/effective（抓 pool 耗尽回归）。
- **诊断**：`dfx-analyze` skill + 两次独立运行（`--enable-l2-swimlane`、`--enable-dep-gen`）→ `sched_overhead_analysis` 看 `sync_tensormap` 阶段是否缩小。
- **判据**：orch_cost >5% 改善且 wire/effective ±2% → 无条件合入；wire/effective 退步 >2% → 申请 `SIMPLER_LAZY_LAST_TASK_ALIVE_PUBLISH` opt-in env（env-macro-gating 需用户许可）落暗版。

---

## 背景（一句话）

PR 906「[WIP] A5 performance opt」整体不可直接合（WIP、自 6/11 停更、被 main 走偏、A5 无实测收益）；但其 M5 思路（限制 `last_task_alive` 的 SM 写者为单一与 orch 同 cluster/die 的线程）在 A5 上正中要害——A5 默认 `ALLOWED_CPUS={5,6,7,8,3}` 让 orch 与全部 sched 跨 die，每任务一次跨 die cache line 往返，目前未测量、未记录。

## 不做（main 已有或已移除，避免重复踩坑）

- wiring 线程 + scheduler 侧 SPSC wiring queue（#1263 已把 fanin wiring 移到 orchestrator 并删除该 queue）
- `PTO2FaninPool` / `PTO2FaninBuilder`（`pto_orchestrator.cpp:230-261`、`pto_ring_buffer.h:500-570` 已有）
- `new_entry()` spin / assert 加固（`ensure_tensormap_capacity` `pto_orchestrator.cpp:708-793` 已 spin + 500ms backstop）
- `cluster_mode` env（main 不存在，不复活）
- `g_only_thread0_advances` 原样（A5 publisher 应是「与 orch 同 die 的 sched」，非固定 thread0）
