# hardware_test 分支相对 main 的 src 修改分析报告

**比较范围**: `simpler/src` 目录  
**基准**: `origin/main` → **当前**: `hardware_test`  
**统计**: 15 个文件变更，+792 行，-49 行

---

## 一、修改概览与动机

整体上，`hardware_test` 在 main 基础上做了两类改动：

1. **PTO2_SIM_AICORE_UT 仿真 AICore 支持**  
   在无真实 AICore 环境下（如 aicpu_ut 单测/性能测试），用 CPU 线程模拟“收到任务即立刻 FIN”的 AICore 行为，使调度器逻辑可测、可 profiling。

2. **PTO2_PROFILING_BEGINEND 与 Phase 扩展**  
   增加“仅记录首尾”的 profiling 模式，并扩展 Scheduler/Orchestrator 的 Phase 枚举（SCHED_LOOP_BEGIN/END、ORCH_BEGIN/END），便于轻量级性能分析。

下面按文件逐项分析每一处修改的**必要性**与**风险**。

---

## 二、按文件逐行分析

### 1. `src/a2a3/platform/include/aicpu/platform_regs.h`

| 修改内容 | 必要性 | 说明 |
|----------|--------|------|
| 在 `#if defined(PTO2_SIM_AICORE_UT)` 下声明 5 个 `extern "C"` 函数：`pto2_sim_read_cond_reg`、`pto2_sim_aicore_on_task_received`、`pto2_sim_aicore_set_idle`、`pto2_sim_set_current_core`、`pto2_sim_clear_current_core` | **必要** | 仿真模式下，`platform_regs.cpp` 的 `read_reg`/`write_reg` 需要调用这些接口拦截对“模拟核”的寄存器访问；仅在 PTO2_SIM_AICORE_UT 下声明，对 main 线无影响。 |

**结论**: 与 sim_aicore 功能强绑定，必要且隔离良好。

---

### 2. `src/a2a3/platform/include/common/perf_profiling.h`

| 修改内容 | 必要性 | 说明 |
|----------|--------|------|
| 注释更新：Scheduler phases 0–5（原 0–3），Orchestrator 16–26（原 16–24） | **必要** | 与新增 Phase ID 一致，避免文档与实现不一致。 |
| 新增 `SCHED_LOOP_BEGIN=4`、`SCHED_LOOP_END=5`，`SCHED_PHASE_COUNT=6` | **必要** | PTO2_PROFILING_BEGINEND 下每轮调度循环只打首尾点，需要独立 Phase ID。 |
| 新增 `ORCH_BEGIN=25`、`ORCH_END=26` | **必要** | 同上，Orchestrator 每次 submit 只记录开始/结束。 |

**结论**: 纯枚举与注释扩展，为 BEGINEND 模式服务，必要且无行为回退。

---

### 3. `src/a2a3/platform/include/common/platform_config.h`

| 修改内容 | 必要性 | 说明 |
|----------|--------|------|
| `PLATFORM_MAX_BLOCKDIM`、`PLATFORM_AIC_CORES_PER_BLOCKDIM`、`PLATFORM_AIV_CORES_PER_BLOCKDIM`、`PLATFORM_MAX_AICPU_THREADS` 用 `#ifndef` 包裹，允许编译时 `-D` 覆盖 | **有用** | 便于测试/不同板型在不改代码下调整平台常量；默认值不变，main 行为不变。 |
| 新增 `PLATFORM_CORES_PER_BLOCKDIM = AIC + AIV`，原 `PLATFORM_CORES_PER_BLOCKDIM = 3` 改为由两常量推导 | **必要** | 若将来 AIC/AIV 数可配，单一常量无法表达；当前 1+2=3 与 main 一致。 |
| `EXTRACT_TASK_STATE(regval)` 在 `PTO2_SIM_AICORE_UT` 下固定为 `TASK_FIN_STATE` | **必要** | 仿真核无真实 COND 寄存器，调度器需要“读到的就是完成态”；仅 sim 路径使用，不影响真机。 |

**结论**: 平台可配置化 + 仿真语义，均必要且作用域清晰。

---

### 4. `src/a2a3/platform/src/aicpu/platform_regs.cpp`

| 修改内容 | 必要性 | 说明 |
|----------|--------|------|
| `#if defined(PTO2_SIM_AICORE_UT)` 下 `#include "sim_aicore.h"` | **必要** | 使用 sim 接口与 SimCoreGuard。 |
| `read_reg`: 当 `pto2_sim_is_current_sim() && reg_base_addr==0 && reg==COND` 时走 `pto2_sim_read_cond_reg`，否则走原有内存读 | **必要** | 仿真核的“寄存器”在进程内数组，必须拦截，否则会访问空指针或错误地址。 |
| `write_reg`: 当 sim 且 `reg_base_addr==0 && reg==DATA_MAIN_BASE` 时，根据 value 调用 `pto2_sim_aicore_set_idle` 或 `pto2_sim_aicore_on_task_received`，否则写真实寄存器 | **必要** | 仿真核不写真实硬件，必须在此把“下发的 task_id”转成 sim 状态机。 |
| `platform_init_aicore_regs(reg_addr)` / `platform_deinit_aicore_regs(reg_addr)` 在 `reg_addr==0` 时直接 return | **必要** | 仿真核 reg_addr 传 0，不应写任何硬件。 |
| 删除两处注释 “Read/Write the register value” | **可选** | 仅注释，不影响功能；保留或删除均可。 |

**结论**: 所有逻辑修改均为仿真路径服务，真机路径不变，必要且安全。

---

### 5. `src/a2a3/runtime/tensormap_and_ringbuffer/aicpu/aicpu_executor.cpp`

本文件改动最多，分块说明。

#### 5.1 头文件与静态数据（PTO2_SIM_AICORE_UT）

| 修改内容 | 必要性 | 说明 |
|----------|--------|------|
| `#include <thread>` | **必要** | `aicpu_sim_run_pto2` 中用 `std::thread` 启动调度线程。 |
| `#include "sim_aicore.h"`、`"cpu_affinity.h"`，以及 `s_sched_cpus[]`、`s_actual_sched_cpu[]` | **必要** | sim 运行时绑定 CPU 并记录实际绑核结果，供测试校验。 |

#### 5.2 成员与模板签名

| 修改内容 | 必要性 | 说明 |
|----------|--------|------|
| `setup_after_host_orch(int32_t total_task_count)`、`run_resolve_and_dispatch_pto2(Runtime*, int)` | **必要** | Host 侧已建图、已设 orchestrator_done 时，由测试直接设 total_tasks 并只跑调度循环，不跑 dlopen 等设备编排路径。 |
| `check_running_cores_for_completion` 增加参数 `bool sim_accumulate`（仅 PTO2_SIM_AICORE_UT） | **必要** | 用于在访问“模拟核”前设置 SimCoreGuard，区分 sim/hw。 |
| 在 complete 分支内 `(void)mixed_task_id;` | **必要** | 某些编译配置下该变量仅 debug 使用，避免未使用变量告警。 |

#### 5.3 写寄存器下发任务

| 修改内容 | 必要性 | 说明 |
|----------|--------|------|
| 在 `dispatch_subtask_to_core` 写 DATA_MAIN_BASE 处，用 `SimCoreGuard` + 条件判断，sim 时走 sim 路径，否则 `write_reg(..., task_id+1)` | **必要** | 与 platform_regs 的 write_reg 拦截一致，保证 sim 核只更新内存状态不写硬件。 |

#### 5.4 handshake_all_cores

| 修改内容 | 必要性 | 说明 |
|----------|--------|------|
| `cores_total_num_ == 0` 时提前设置 aic_count_/aiv_count_=0 并 return 0 | **必要** | Host 建图、0 核的 sim 场景需要合法退出。 |
| `get_sim_aicore_mode()` 为真时：按 PLATFORM_MAX_AIC_PER_THREAD 等填 Handshake、core_id_to_reg_addr_[i]=0，return 0 | **必要** | 仿真不握手真实设备，仅填充调度器需要的 core 元数据，reg_addr=0 与 platform_regs 的拦截约定一致。 |
| 原 “cores_total_num_ == 0 \|\| > MAX” 改为 “> RUNTIME_MAX_WORKER \|\| > MAX_CORES_PER_THREAD”，并单独判断 `cores_total_num_==0` 报错 | **必要** | 先处理 sim 的 0 核与合法多核，再对非法值统一报错，逻辑更清晰；若 RUNTIME_MAX_WORKER 与 main 一致则行为等价。 |

#### 5.5 init / shutdown

| 修改内容 | 必要性 | 说明 |
|----------|--------|------|
| init 末尾 sim 模式下调用 `pto2_sim_aicore_init_all_idle()` | **必要** | 所有 sim 核 COND 初始化为 idle，否则调度器第一次读会出错。 |
| `shutdown_aicore` 中 `(void)thread_idx/cur_thread_cores/core_num` | **必要** | 某些路径未使用，消除告警。 |
| `reg_addr==0` 时只调 `platform_deinit_aicore_regs`，不报 DEV_ERROR；仅在非 PTO2_SIM_AICORE_UT 时对 reg_addr==0 报错 | **必要** | 仿真核 reg_addr 为 0 是合法设计，不应报错。 |

#### 5.6 resolve_and_dispatch_pto2 主循环

| 修改内容 | 必要性 | 说明 |
|----------|--------|------|
| `PTO2_LOCAL_DISPATCH_TYPE_NUM` 改为局部常量 `kLocalDispatchTypeNum = 2` | **可选** | 与 pto_scheduler.h 中删除的静态常量对应，避免依赖头文件中的已删符号；功能等价。 |
| 引入 `sched_dispatch_setup_cycle`、`loop_start_prev`、`_t_loop_end`、`loop_has_prev`（PTO2_PROFILING_BEGINEND） | **必要** | 用于每轮记录 SCHED_LOOP_BEGIN/SCHED_LOOP_END。 |
| 循环开始处：若 BEGINEND 且 `loop_has_prev` 则记录 SCHED_LOOP_END；再记录 SCHED_LOOP_BEGIN | **必要** | 实现“每轮只打首尾”的语义。 |
| 提前计算 `task_count`、`orch_done`；`task_count==0` 时记录 LOOP_END 并 break | **必要** | 空图时 orchestrator 已 done 但无任务，应直接退出并打点，避免死循环。 |
| 完成条件满足时，在 PTO2_SIM_AICORE_UT 下先 flush `deferred_release_slot_states`（调用 on_task_release），再 `completed_.store(true)` | **必要** | sim 模式下延迟 release 需在退出前处理完，否则依赖与统计不完整。 |
| 多处 `reassigned_` 等待与 break 前增加 SCHED_LOOP_END 记录 | **必要** | 保证每轮循环要么有 LOOP_END 要么以 LOOP_END 结束，profiling 时间线完整。 |
| Scan 阶段：`t_scan0/t_scan1`、`sched_scan_cycle` 累加（与 main 的 test_common 对齐） | **必要** | 保持 Phase 1 窗口与 main 一致，便于对比。 |
| `try_completed` / `try_pushed` 用 `#if PTO2_PROFILING` 包裹 | **必要** | 仅 profiling 使用，避免未使用变量告警。 |
| `check_running_cores_for_completion` 调用处增加最后一参 `(runtime && runtime->get_sim_aicore_mode())` | **必要** | 传入 sim_accumulate，内部才能正确设 SimCoreGuard。 |
| SCHED_COMPLETE / SCHED_DISPATCH 段落：PTO2_PROFILING_BEGINEND 时只更新 `_t_loop_end`、`_t0_phase` 并清零 phase 计数，不在这里打 phase 点 | **必要** | BEGINEND 模式只在循环首尾打点，中间阶段不再写 phase。 |
| `overflow_ptrs` 与 for 循环用 `kLocalDispatchTypeNum` | **必要** | 与上面局部常量一致。 |
| `ResourceCount rc = shape_resource_count(shape);` 缩进修正 | **风格** | 无功能影响。 |
| `if (made_progress)` 与 `if (!made_progress)` 拆成两个 if | **必要** | 原 else 里做 deferred release；拆开后逻辑不变，便于后续在中间加 BEGINEND 等逻辑。 |
| 循环结束后 PTO2_SIM_AICORE_UT 下再次 flush deferred_release | **必要** | 与完成时 flush 对称，保证 sim 退出时所有 release 已处理。 |
| Sched summary 中 sim 模式打印 “otc_*, perf, scan expected 0” | **有用** | 提示在 sim 下这些统计为 0 是预期。 |
| diagnose_stuck_state 中读 COND 前对 `reg_addr` 设 SimCoreGuard | **必要** | 诊断时若包含 sim 核，读寄存器必须走 sim 路径。 |

#### 5.7 run() 与设备编排

| 修改内容 | 必要性 | 说明 |
|----------|--------|------|
| 整段 dlopen/device orchestration 用 `#if !defined(PTO2_SIM_AICORE_UT)` 包裹；else 分支报错 “Device orchestration not supported in PTO2_SIM_AICORE_UT” | **必要** | sim 测试由 Host 建图并调用 `aicpu_sim_run_pto2`，不应走设备侧 dlopen。 |
| 增加注释：dlopen 在静态链接 aicpu_ut 时可能因 glibc 依赖产生告警，属预期 | **有用** | 减少误判。 |
| 最后线程收尾时，在非 PTO2_SIM_AICORE_UT 下 destroy runtime、dlclose、unlink | **必要** | sim 路径未 dlopen，不应 dlclose。 |

#### 5.8 新增 API（文件末尾）

| 修改内容 | 必要性 | 说明 |
|----------|--------|------|
| `AicpuExecutor::setup_after_host_orch`、`run_resolve_and_dispatch_pto2` | **必要** | 供 aicpu_sim_run_pto2 调用。 |
| `aicpu_sim_run_pto2(pto2_rt, num_sched_threads)`：构造 Runtime（worker_count=PLATFORM_MAX_CORES、orch 建图在 Host、sim_aicore_mode=true）、init、setup_after_host_orch、创建 num_sched_threads 个线程跑 run_resolve_and_dispatch_pto2、join、shutdown_aicore | **必要** | 无设备时完整跑一遍调度器，用于 perf/单测。 |
| `aicpu_sim_get_actual_sched_cpu(thread_idx)` | **必要** | 测试校验 CPU 亲和性。 |

**结论**: aicpu_executor 的修改全部服务于 sim 模式与 BEGINEND profiling，无多余逻辑；需确认 `RUNTIME_MAX_WORKER` 与 main 一致，以及非 sim 路径的边界（如 cores_total_num_）与 main 一致。

---

### 6. `sim_aicore.cpp` / `sim_aicore.h`（新文件）

| 内容 | 必要性 | 说明 |
|------|--------|------|
| 全局 `s_sim_core_cond_value[]`、线程局部 `SimCoreContext`、`pto2_sim_set/clear_current_core`、`pto2_sim_get_current_core_id`、`pto2_sim_is_current_sim` | **必要** | 为 platform_regs 的 read_reg/write_reg 提供“当前是哪个 sim 核”的上下文，以及每核 COND 内存。 |
| `pto2_sim_read_cond_reg`、`pto2_sim_aicore_on_task_received`、`pto2_sim_aicore_set_idle`、`pto2_sim_aicore_init_all_idle` | **必要** | 实现“下发即 FIN”的 sim 语义，与 platform_regs 的拦截一一对应。 |
| `SimCoreGuard`（RAII 设/清 current core） | **必要** | 保证在 read_reg/write_reg 执行期间 current core 正确，避免多核交错。 |
| 非 PTO2_SIM_AICORE_UT 时提供空实现 `SimCoreGuard(int,bool){}` | **必要** | 使 aicpu_executor 等调用处在未定义 PTO2_SIM_AICORE_UT 时仍可编译，无运行时开销。 |

**结论**: 纯新增、仅 PTO2_SIM_AICORE_UT 使用，必要且封装清晰。

---

### 7. `docs/PROFILING_MECHANISM.md`（新增）

| 内容 | 必要性 | 说明 |
|------|--------|------|
| 描述 PerfRecord/Phase、宏层级、共享内存布局、数据流、Scheduler/Orchestrator 细粒度统计 | **有用** | 文档性质，便于后续维护与排查 profiling 问题；不改变代码行为。 |

---

### 8. `docs/profiling_levels.md`

| 修改内容 | 必要性 | 说明 |
|----------|--------|------|
| 增加 PTO2_PROFILING_BEGINEND 说明及与 PTO2_PROFILING 的依赖 | **必要** | 与 pto_runtime2_types.h 的 #error 及 perf_profiling.h 枚举一致。 |

---

### 9. `runtime/pto_orchestrator.cpp`

| 修改内容 | 必要性 | 说明 |
|----------|--------|------|
| 新增 `g_orch_finalize_wait_cycle` | **必要** | 与其它 orch 细粒度统计一致，若 finalize 阶段有等待需单独统计。 |
| `CYCLE_COUNT_LAP_RECORD` 中：若 PTO2_PROFILING_BEGINEND 且当前 phase 不是 ORCH_FANIN，则写入 Phase 时用 ORCH_END 代替原 phase_id；否则保持原 phase_id | **必要** | BEGINEND 模式下只在 submit 结束时打 ORCH_END，中间子阶段不写 phase（或仅在 ORCH_FANIN 等关键点保留一处）。 |
| `pto2_submit_mixed_task` 开头在 PTO2_PROFILING_BEGINEND 下记录 ORCH_BEGIN | **必要** | 实现“每次 submit 只打 begin+end”的语义。 |

**结论**: 均为 BEGINEND 与 orch 统计扩展，必要且仅影响 profiling 输出。

---

### 10. `runtime/pto_runtime2_types.h`

| 修改内容 | 必要性 | 说明 |
|----------|--------|------|
| `#ifndef PTO2_PROFILING_BEGINEND` 默认 0；且 `#if PTO2_PROFILING_BEGINEND && !PTO2_PROFILING` 则 `#error` | **必要** | 保证 BEGINEND 仅在打开 PROFILING 时生效，避免配置错误。 |

---

### 11. `runtime/pto_scheduler.cpp`

| 修改内容 | 必要性 | 说明 |
|----------|--------|------|
| `#if PTO2_SCHED_PROFILING` 拆成两段，中间加 `#include "common/platform_config.h"` | **必要** | 避免在未定义 PTO2_SCHED_PROFILING 时引用 PLATFORM_MAX_AICPU_THREADS 等，减少依赖与编译问题。 |
| 文件末尾多一空行 | **风格** | 无功能影响。 |

---

### 12. `runtime/pto_scheduler.h`

| 修改内容 | 必要性 | 说明 |
|----------|--------|------|
| 注释从 “Two buffers per scheduling thread, one per CoreType” 改为 “One buffer per scheduling thread (mixed worker types)” 等 | **必要** | 与当前“每线程一组 local_bufs、按 CoreType 分槽”的实现一致；避免误导。 |
| 删除 `static constexpr int PTO2_LOCAL_DISPATCH_TYPE_NUM = 2;` | **必要** | 该常量已移至 aicpu_executor 内部 `kLocalDispatchTypeNum`，避免重复定义与头文件耦合。 |
| `PTO2SchedulerState` 增加 `PTO2DepListPool* dep_pool` | **必要** | 若调度器需要访问依赖表池（如 release/fanin 等），需持有指针；与 main 是否已使用 dep_pool 需核对，避免 main 未初始化导致空指针。 |
| `on_task_release` 中 `new_refcount == slot_state.fanin_count` 改为先赋 `ready` 再 `if(ready)` | **风格/可读** | 逻辑等价，便于加日志或断点。 |
| `on_mixed_task_complete` 签名去掉多余空行，并加注释“完成任务时同时调用 release_fanin...” | **有用** | 注释有助于理解与 main 行为一致。 |
| `pto2_scheduler_get_profiling` 声明后加空行 | **风格** | 无功能影响。 |

**风险点**: `dep_pool` 若在 main 或测试中未赋值，可能为 nullptr，需确认所有使用处做了判空或保证初始化。

---

### 13. `runtime/runtime.cpp` / `runtime.h`

| 修改内容 | 必要性 | 说明 |
|----------|--------|------|
| 构造函数中 `#if defined(PTO2_SIM_AICORE_UT)` 下 `sim_aicore_mode_ = false` | **必要** | 保证未设置时默认为 false。 |
| 新增成员 `bool sim_aicore_mode_`（仅 PTO2_SIM_AICORE_UT）及 getter/setter | **必要** | aicpu_executor 与 handshake/init 需要查询是否为 sim 模式。 |
| 非 PTO2_SIM_AICORE_UT 时 get/set 为无操作（return false / 空实现） | **必要** | 调用方统一写 get_sim_aicore_mode() 即可，无需到处 #ifdef。 |

**结论**: 与 sim 模式开关一致，必要且对非 sim 零影响。

---

## 三、必要性汇总

| 类别 | 修改条数（约） | 必要性结论 |
|------|----------------|------------|
| PTO2_SIM_AICORE_UT：platform_regs、sim_aicore、runtime、aicpu_executor | 约 40+ 处 | **全部必要**：无仿真路径则 aicpu_ut 无法在无设备下跑调度与 profiling。 |
| PTO2_PROFILING_BEGINEND：Phase 枚举、循环首尾打点、Orchestrator begin/end | 约 25+ 处 | **全部必要**：实现“仅首尾”的轻量 profiling 语义。 |
| 平台可配置（platform_config.h 的 #ifndef）、EXTRACT_TASK_STATE(sim) | 约 5 处 | **必要**：测试/多板型/仿真语义需要。 |
| 注释、文档、缩进、空行、(void) 消警、kLocalDispatchTypeNum 替代常量 | 约 15 处 | **有用或必要**：可读性、编译清洁度、与 pto_scheduler.h 解耦。 |
| pto_scheduler.h 的 dep_pool 成员 | 1 处 | **需确认**：若 main 或调用方未初始化 dep_pool，存在空指针风险，建议核对并保证初始化或判空。 |

---

## 四、合并到 main 的建议

1. **保留所有 PTO2_SIM_AICORE_UT 与 PTO2_PROFILING_BEGINEND 相关改动**  
   功能隔离清晰（宏 + 运行时 sim_aicore_mode），对 main 默认构建无影响。

2. **确认常量与上限一致**  
   - `RUNTIME_MAX_WORKER` 与 main 是否一致（handshake 校验用）。  
   - `PLATFORM_MAX_CORES` 与 sim 使用的 `worker_count` 是否与测试预期一致。

3. **dep_pool 生命周期**  
   在 main 与测试路径中确认 `PTO2SchedulerState::dep_pool` 在 use 前已赋值；若 main 尚未使用，可先保留成员并设为 nullptr，在扩展依赖表功能时再使用。

4. **回归与测试**  
   - 关闭 PTO2_SIM_AICORE_UT 与 PTO2_PROFILING_BEGINEND 的 build 与 main 行为一致。  
   - 打开 PTO2_SIM_AICORE_UT 时跑 aicpu_ut 及现有 perf 用例，确认无回归。  
   - 若启用 PTO2_PROFILING_BEGINEND，做一次小规模 profiling 采集，确认 Phase 时间线正确。

5. **文档**  
   保留并合入 `PROFILING_MECHANISM.md` 与 `profiling_levels.md` 的更新，便于后续维护。

---

**报告生成说明**: 基于 `git diff origin/main..hardware_test -- src` 逐块对照代码与注释整理，结论针对“每行/每块修改是否必要”及合并风险给出，便于 code review 与合并决策。
