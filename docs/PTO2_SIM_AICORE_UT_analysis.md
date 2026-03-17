# PTO2_SIM_AICORE_UT 宏必要性分析

`PTO2_SIM_AICORE_UT` 是 aicpu_ut 的构建选项，用于启用**零核心仿真**：无真实 AICore 硬件时，调度器通过 `sim_aicore` 的 COND 寄存器数组模拟任务完成。以下按文件逐处分析每个 `#if defined(PTO2_SIM_AICORE_UT)` 或 `#if !defined(PTO2_SIM_AICORE_UT)` 是否必要。

---

## 1. aicpu_executor.cpp

| 位置 | 内容概要 | 必要性 | 说明 |
|------|----------|--------|------|
| **40-56** | `#include "sim_aicore.h"`、`cpu_affinity`、`s_sched_cpus[]`、`s_actual_sched_cpu[]`、`pto2_sim_set/clear_current_core`、`s_sched_prof_snapshot` 等 | **必要** | 这些符号仅在 sim 构建中存在；非 UT 构建不包含 sim_aicore，会编译失败。 |
| **59-67** | `struct SimCoreGuard` | **必要** | 依赖 `pto2_sim_set_current_core` / `pto2_sim_clear_current_core`，仅 UT 有实现。 |
| **268-271** | `setup_after_host_orch()`、`run_resolve_and_dispatch_pto2()` 声明 | **必要** | 仅被 `aicpu_sim_run_pto2()` 等 UT 路径调用，非 UT 不链接这些入口。 |
| **319-321** | `check_running_cores_for_completion` 的 `bool sim_accumulate` 参数 | **必要** | 非 UT 无此参数，调用处也不传；若去掉宏会破坏非 UT 的调用签名。 |
| **329-331** | `SimCoreGuard guard(core_id, sim_accumulate && reg_addr == 0)` | **必要** | `SimCoreGuard` 仅在本宏下定义，非 UT 无此类型。 |
| **383-385** | `if (sim_accumulate) pto2_sim_accumulate_fanin(fe)` | **必要** | `pto2_sim_accumulate_fanin` 为 sim_aicore 专有 API。 |
| **539-570** | `cores_total_num_ == 0` 时的 drain 循环（不轮询 reg，只走 sim 完成） | **必要** | 零核 + sim 模式专用路径；非 UT 要么有真实核要么不会进此逻辑。 |
| **571-574** | `cores_total_num_ == 0` 时设 `aic_count_=0` 等 | **必要** | 零核初始化，仅 UT 会出现 `cores_total_num_==0`。 |
| **885-888** | `pto2_sim_aicore_init_all_idle()` | **必要** | sim 专有，初始化仿真核 COND 状态。 |
| **946-949** | `#if !defined`：`reg_addr == 0` 时 `DEV_ERROR` | **必要** | 非 UT 下 `reg_addr==0` 视为非法；UT 下 sim 核允许 `reg_addr==0`，不应报错。 |
| **1089-1104** | deferred_release 循环内 `pto2_sim_accumulate_fanin` | **必要** | sim 统计用，非 UT 无此 API。 |
| **1185-1187, 1209-1211** | 传 `(runtime && runtime->get_sim_aicore_mode())` 作 `sim_accumulate` 实参 | **必要** | 与上面 `sim_accumulate` 参数对应，仅 UT 需要传入。 |
| **1214-1216** | sim 模式下 `notify_edges_total` 等 profiling 统计 | **必要** | sim 路径的 profiling 逻辑，依赖 sim 专有状态。 |
| **1427-1430** | `pto2_sim_accumulate_rounds(1, made_progress ? 1 : 0)` | **必要** | sim 专有 API。 |
| **1446-1448** | `pto2_sim_accumulate_fanin(fe)` | **必要** | 同上。 |
| **1501-1503, 1521-1523** | 两处 `SimCoreGuard guard(...)` | **必要** | 与 329-331 同理，类型与语义仅 UT 存在。 |
| **1561-1579** | sim 下 `deferred_release_count` 循环中 accumulate | **必要** | sim 收尾统计。 |
| **1580-1606** | sim 下 `pto2_sim_accumulate_cycles` 等 | **必要** | sim 专有 profiling。 |
| **1607-1609** | 取 `s_sched_prof_snapshot[thread_idx]` 或 `pto2_scheduler_get_profiling(thread_idx)` | **必要** | sim 用 snapshot；非 UT 用 scheduler 实时数据，数据源不同。 |
| **1619-1622** | sim 时 `DEV_ALWAYS` 提示 otc_* 等 | **必要** | 行为区分（可视为文档/调试），保留无妨。 |
| **1752-2101** | `#if !defined`：device orchestration（dlopen SO、创建 runtime 等）；`#else` 报错 "Device orchestration not supported in PTO2_SIM_AICORE_UT" | **必要** | UT 只做 host 编排 + `aicpu_sim_run_pto2`，不支持设备侧 dlopen/device orch；非 UT 才需要整段设备编排逻辑。 |
| **2140-2146** | `#if !defined`：最后线程 `pto2_runtime_destroy`、`dlclose(orch_so_handle_)`、`unlink` | **必要** | UT 无 `orch_so_handle_`，且 runtime 由测试管理，不在此 destroy。 |
| **2258-2260** | `SimCoreGuard guard(...)` | **必要** | 同前 SimCoreGuard。 |
| **2355-2426** | `setup_after_host_orch` 实现、`run_resolve_and_dispatch_pto2`、`aicpu_sim_run_pto2`、`aicpu_sim_get_actual_sched_cpu` 等 | **必要** | 仅 UT 使用的入口与实现，非 UT 不链接 sim 运行路径。 |

**结论**：`aicpu_executor.cpp` 内所有 `PTO2_SIM_AICORE_UT` 相关分支均**必要**，用于严格区分「仿真零核 + host 编排」与「真实设备 + 可选 device 编排」两条构建与运行路径。

---

## 2. platform_regs.cpp

| 位置 | 内容概要 | 必要性 | 说明 |
|------|----------|--------|------|
| **22-24** | `#include "sim_aicore.h"` | **必要** | 仅 UT 有 sim_aicore，非 UT 无此头文件。 |
| **36-53** | `SimCoreContext`、`pto2_sim_set_current_core`、`pto2_sim_clear_current_core` | **必要** | `read_reg`/`write_reg` 通过 `g_sim_core_ctx` 判断是否走 sim 分支；非 UT 不需要这些符号。 |
| **57-61** | `read_reg` 中 `reg_base_addr==0 && reg==COND` 时调 `pto2_sim_read_cond_reg` | **必要** | sim 核无 MMIO，必须从 sim 状态读 COND。 |
| **76-85** | `write_reg` 中对 DATA_MAIN_BASE 的 sim 分支（`pto2_sim_aicore_set_idle` / `on_task_received`） | **必要** | sim 核写寄存器改为更新 sim 状态，非 UT 走真实 MMIO。 |
| **97-100** | `platform_init_aicore_regs`：`reg_addr == 0` 时直接 return | **必要** | sim 核无硬件可初始化。 |
| **102-108** | `#if defined` 里 `write_reg(FAST_PATH_ENABLE/DATA_MAIN_BASE)`；`#else` 相同两行 | **可简化** | 两分支代码完全一致，可去掉本段 `#if/#else`，只保留一份 `write_reg` 调用。 |
| **112-115** | `platform_deinit_aicore_regs`：`reg_addr == 0` 时 return | **必要** | sim 核无硬件可反初始化。 |
| **116-121** | `#if defined` 里写 EXIT_SIGNAL + FAST_PATH_CLOSE；`#else` 相同 | **可简化** | 同上，可合并为一份实现。 |

**结论**：除 **96-108** 和 **111-121** 中「同逻辑重复的 `#if/#else`」可合并为单份代码外，其余均为**必要**（区分 sim 寄存器语义与真实硬件）。

---

## 3. platform_regs.h

| 位置 | 内容概要 | 必要性 | 说明 |
|------|----------|--------|------|
| **47-54** | `pto2_sim_read_cond_reg`、`pto2_sim_aicore_on_task_received`、`pto2_sim_aicore_set_idle`、`pto2_sim_set/clear_current_core` 的 extern 声明 | **必要** | 非 UT 不链接 sim_aicore/platform_regs 的 sim 实现，声明若暴露会导致链接或未使用符号问题；仅 UT 需要这些 API。 |

---

## 4. test_common.cpp

| 位置 | 内容概要 | 必要性 | 说明 |
|------|----------|--------|------|
| **13-15** | `#include "sim_aicore.h"` | **必要** | 下面用到 `aicpu_sim_get_run_prof` 等，仅 UT 有。 |
| **519-538** | `print_sched_profiling` 里用 `aicpu_sim_get_run_prof` 回填 `g_sched_prof_data` | **必要** | `aicpu_sim_run_pto2` 路径不写 `g_sched_prof_data`，需用 sim 运行结果回填后 fanout/fanin 等才正确。 |
| **548-550** | `#if PTO2_SCHED_PROFILING && !defined(PTO2_SIM_AICORE_UT)` 时调用 `pto2_print_sched_profiling(0)` | **必要** | 非 UT 用 scheduler 的 per-thread profiling 打印；UT 已用上面回填 + `pto2_print_sim_sched_summary`，不再调 `pto2_print_sched_profiling`。 |
| **578-592** | `#if !defined`：P2 检查（每调度线程 dispatch>0）；`#else` 注释 "P2 unavailable: no per-thread profiling" | **必要** | UT 使用 `aicpu_sim_run_pto2`，无 per-thread dispatch 统计，无法做 P2 检查。 |

---

## 5. 测试用例与驱动

| 文件 | 内容概要 | 必要性 | 说明 |
|------|----------|--------|------|
| **test_concurrent.cpp, test_sched_prof_only.cpp** | `#if defined(PTO2_SIM_AICORE_UT)` 内调 `aicpu_sim_run_pto2` | **必要** | `aicpu_sim_run_pto2` 仅在此宏下存在，非 UT 构建不会编译到该调用。 |
| **test_deg2.cpp, test_latency.cpp, test_batch_paged_attention.cpp 等 perf cases** | `#if defined(PTO2_SIM_AICORE_UT)` 内 `#include "sim_aicore.h"` | **必要** | 使用 `aicpu_sim_run_pto2` 等，必须限定在 UT 构建内包含头文件。 |

---

## 6. sim_aicore.h

| 位置 | 内容概要 | 必要性 | 说明 |
|------|----------|--------|------|
| **16-17** | 整个头文件内容包在 `#if defined(PTO2_SIM_AICORE_UT)` 内 | **必要** | 非 UT 不应看到 `s_sim_core_cond_value`、`aicpu_sim_run_pto2` 等 API，避免误用或链接错误。 |

---

## 7. CMakeLists.txt（aicpu_ut）

| 位置 | 内容概要 | 必要性 | 说明 |
|------|----------|--------|------|
| **55, 74-76, 108-110, 172-175** | `option(PTO2_SIM_AICORE_UT ...)`、include 路径、`COMMON_COMPILE_DEFS`、`RUNTIME_SRCS` 中 aicpu_executor 等 | **必要** | 构建配置：仅在 aicpu_ut 且开启该 option 时定义宏并链接 sim 相关实现。 |

---

## 总结表

| 类别 | 必要 | 可简化 | 说明 |
|------|------|--------|------|
| **aicpu_executor.cpp** | 全部 | 0 | 严格区分 UT 仿真路径与正式设备路径。 |
| **platform_regs.cpp** | 除两处重复分支外 | 2 处 | 96-108、111-121 中「同逻辑的 #if/#else」可合并。 |
| **platform_regs.h** | 全部 | 0 | 仅 UT 暴露 sim 声明。 |
| **test_common.cpp** | 全部 | 0 | 回填 profiling、P2 条件、include 均依赖宏。 |
| **perf 驱动/用例** | 全部 | 0 | 仅 UT 有 `sim_aicore.h` 与 `aicpu_sim_run_pto2`。 |
| **sim_aicore.h** | 全部 | 0 | 仅 UT 暴露 sim API。 |
| **CMakeLists.txt** | 全部 | 0 | 构建开关。 |

**建议**：  
- 保留所有「必要性」列中的宏，用于正确区分 UT 仿真构建与正式构建。  
- 仅对 **platform_regs.cpp** 中两处「两分支代码相同」的 `#if defined(PTO2_SIM_AICORE_UT) / #else` 做合并，保留「reg_addr==0 提前 return」的 sim 分支，其余 init/deinit 写寄存器逻辑可共用一个实现。
