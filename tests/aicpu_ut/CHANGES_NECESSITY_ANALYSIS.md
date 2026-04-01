# hardware_test9 分支：tests/aicpu_ut 之外的代码修改必要性分析

> 分析范围：`git diff HEAD~1..HEAD` 中除 `tests/aicpu_ut/` 目录以外的所有改动。
> 目标：判断每处修改对于 `tests/aicpu_ut` 测试可编译、可运行的必要程度。

---

## 总览

| 文件 / 目录 | 改动规模 | 必要性 | 类型 |
|---|---|---|---|
| `src/a2a3/sim/` (4 个新文件) | +603 行 | **必须** | 新增仿真基础设施 |
| `src/a2a3/platform/src/aicpu/platform_regs.cpp` | +27 行 | **必须** | 生产代码中植入仿真钩子 |
| `src/a2a3/runtime/tensormap_and_ringbuffer/aicpu/aicpu_executor.cpp` | +255 行 | **必须**（部分可选） | 仿真模式 + Profiling 标记 |
| `src/a2a3/runtime/tensormap_and_ringbuffer/runtime/runtime.h / .cpp` | +14 行 | **必须** | 并发仿真所需状态字段 |
| `src/a2a3/runtime/tensormap_and_ringbuffer/runtime/tensor_factory.h` | +29 行（新文件） | **必须** | 测试用辅助工厂函数 |
| `src/a2a3/runtime/tensormap_and_ringbuffer/runtime/pto_orchestrator.cpp` | +128 行 | **条件必须** | 指令计数标记 + 调试日志 |
| `python/toolchain.py` | +42 行 | **构建必须** | Conda/系统 CC/CXX 多段命令处理 |
| `examples/.../multi-round-paged-attention/golden.py` | 符号链接 → 实体文件 | **独立** | 与测试无直接依赖 |

---

## 逐文件详细分析

---

### 1. `src/a2a3/sim/`（全新目录，4 个新文件）

**文件列表**

| 文件 | 行数 | 作用 |
|---|---|---|
| `sim_aicore.h` | 35 | AICore 仿真接口声明 |
| `sim_aicore.cpp` | 236 | AICore 仿真执行实现（可选定时轮询线程） |
| `msgq_cpu_sim.h/.cpp` | 174 + 194 | 消息队列硬件寄存器 CPU 侧仿真 |
| `hscb_cpu_sim.h/.cpp` | 90 + 83 | HSCB 高速控制总线仿真 |

**依赖关系**

```
tests/aicpu_ut/common/sim_run_pto2.cpp
  └── #include "sim_aicore.h"
      └── pto2_sim_aicore_start_poller()
          pto2_sim_aicore_set_task_duration_ns()

tests/aicpu_ut/CMakeLists.txt
  └── target_sources(...sim_aicore.cpp msgq_cpu_sim.cpp hscb_cpu_sim.cpp)
```

**必要性：必须**

这 4 个文件是整个 AICPU UT 仿真栈的核心。`sim_run_pto2.cpp`（测试公共基础设施）直接 `#include "sim_aicore.h"` 并调用其接口，`CMakeLists.txt` 将这些 `.cpp` 显式加入编译目标。缺少任何一个文件都会导致编译失败。

**对生产代码的影响：无**

这些文件仅被 `PTO2_SIM_AICORE_UT` 编译宏保护的路径引用，不影响正常 target。

---

### 2. `src/a2a3/platform/src/aicpu/platform_regs.cpp`（+27 行）

**改动内容**

在 `read_reg()`、`write_reg()`、`platform_init_aicore_regs()`、`platform_deinit_aicore_regs()` 四个函数中，各增加一个 `#if defined(PTO2_SIM_AICORE_UT)` 分支：

```cpp
// read_reg(): 地址 < PTO2_SIM_REG_ADDR_MAX 时，转发给 sim_aicore 读条件寄存器
// write_reg(): 地址 < PTO2_SIM_REG_ADDR_MAX 时，通知仿真 AICore 收到任务或空闲
// init/deinit: 仿真地址直接 return，跳过硬件操作
```

**必要性：必须**

`aicpu_executor.cpp` 在调度循环中通过 `write_reg(reg_addr, RegId::DATA_MAIN_BASE, ...)` 下发任务，通过 `read_reg(reg_addr, RegId::COND)` 轮询完成。在没有真实 AICore 硬件的测试环境中，若不拦截这两个调用并转发到仿真层，程序将对非法地址（0、1、2……）进行裸指针访问，立即崩溃。

**对生产代码的影响：受控**

所有改动都在 `#if defined(PTO2_SIM_AICORE_UT)` 宏保护内，正常构建不编译这些分支。条件判断 `reg_base_addr < PTO2_SIM_REG_ADDR_MAX` 保证仿真路径仅对"虚拟地址"（值为 0..N 的 core_id）生效。

---

### 3. `src/a2a3/runtime/tensormap_and_ringbuffer/aicpu/aicpu_executor.cpp`（+255 行）

本文件改动分三个独立功能块：

#### 3a. 仿真模式核心逻辑（`PTO2_SIM_AICORE_UT`）

**改动内容**

- `handshake_all_cores()`：跳过真实握手，合成 AIC/AIV core 状态（直接写 `Handshake` 结构、设置 `reg_addr = core_id`）。
- `init()`：读取 `runtime->get_orch_deferred_on_host()` 决定是否延迟设置 `orchestrator_done_`。
- `dispatch_to_core()` 中的 `build_payload()` 调用：仿真模式下跳过（payload 不需要真实地址），改为直接写 `write_reg(DATA_MAIN_BASE, task_id + 1)`，触发 `platform_regs.cpp` 中的仿真钩子。
- 任务完成后的延迟 `on_task_release()` 处理（`deferred_release_count` 清零）。
- 新增方法 `setup_after_host_orch()` 和 `run_resolve_and_dispatch_pto2()`，供并发仿真驱动调用。

**必要性：必须**

没有这些改动，`sim_run_pto2.cpp` 中的 `aicpu_sim_run_pto2()` 和 `aicpu_sim_run_pto2_concurrent()` 无法完成一次完整的调度循环。具体阻断点：
1. `handshake_all_cores()` 在没有真实 workers 的情况下会访问空 Handshake 数组，返回错误或死循环。
2. `dispatch_to_core()` 中若仍调用 `build_payload()`，会向 `s_pto2_payload_per_core` 写入含有无效函数指针的 payload，随后对 null/随机地址发起 `write_reg`，崩溃。
3. 并发模式依赖 `setup_after_host_orch()` 在 Orchestrator 线程完成后通知 Scheduler 线程开始。

#### 3b. 指令计数 QEMU 标记宏（`PTO2_SPECIAL_INSTRUCTION`）

**改动内容**

定义宏 `PTO2_SPECIAL_INSTRUCTION(reg, mode[, flag])`，展开为 `orr xN, xN, xN` 汇编 NOP，作为 QEMU TCG 插件的定界标记，插入调度循环各阶段。

**必要性：条件必须**

- 若测试仅需功能正确性验证（functional tests），这些标记**不必须**，删除后不影响编译和运行。
- 若需要指令计数性能分析（`--count-submit-task-instructions` 模式），则标记**必须存在**，否则 QEMU 插件找不到 markers，分析输出为空。
- 标记本身是 AArch64-only，x86_64 下宏展开为 `((void)0)`，不影响 x86 构建。

#### 3c. Profiling 时间戳调整

`dispatch_timestamp` 从 `if (profiling_enabled)` 内提到外层，使得在不启用 profiling sink 的情况下也能计算任务延迟统计。

**必要性：必须（针对延迟测试）**

`test_latency.cpp` 需要准确的延迟数据，依赖 `pto2_sim_record_task_latency_cycles()` 的输出。若 `dispatch_timestamp` 仅在 profiling 开启时赋值，延迟计算将得到 0 或错误值。

**对生产代码的影响：受控**

所有仿真逻辑在 `#if defined(PTO2_SIM_AICORE_UT)` 内，指令标记在 AArch64 以外为空操作，`dispatch_timestamp` 的提升改动无功能影响（多执行一次 `get_sys_cnt_aicpu()`）。

---

### 4. `src/a2a3/runtime/tensormap_and_ringbuffer/runtime/runtime.h / runtime.cpp`（+14 行）

**改动内容**

在 `Runtime` 类中增加一个布尔成员：

```cpp
#if defined(PTO2_SIM_AICORE_UT)
    bool orch_deferred_on_host_{false};
public:
    bool get_orch_deferred_on_host() const;
    void set_orch_deferred_on_host(bool v);
#endif
```

**必要性：必须**

并发仿真驱动 `test_orchestrator_scheduler.cpp` 的执行模型：Orchestrator 与 Scheduler 在两个独立线程中并发运行，共享同一个 `Runtime` 实例。`aicpu_executor.cpp::init()` 中需要知道 Orchestrator 是否已完成以决定是否预设 `orchestrator_done_`。没有这个标志，并发模式下 Scheduler 线程会因为 `orchestrator_done_ = true`（主机 Orch 路径的默认行为）而在 Orchestrator 完成前就判定工作结束，提前退出，导致任务丢失或 hang。

**对生产代码的影响：无**

整块改动在 `#if defined(PTO2_SIM_AICORE_UT)` 内，不影响正常目标的内存布局或行为。

---

### 5. `src/a2a3/runtime/tensormap_and_ringbuffer/runtime/tensor_factory.h`（新文件，+29 行）

**改动内容**

提供 `make_tensor_external()` 辅助函数，根据外部已分配内存地址和 shape 构造一个 `Tensor` 对象。

**必要性：必须**

`tests/aicpu_ut/perf/cases/` 下的多个测试用例（`test_paged_attention.cpp`、`test_latency.cpp` 等）通过 `#include "tensor_factory.h"` 使用此函数来构造输入输出张量。缺少该文件会导致编译错误 `'make_tensor_external' was not declared`。

**对生产代码的影响：无**

这是一个纯 header-only 工具函数，不修改任何现有代码路径。

---

### 6. `src/a2a3/runtime/tensormap_and_ringbuffer/runtime/pto_orchestrator.cpp`（+128 行）

本文件改动分两个功能块：

#### 6a. 指令计数标记（`orr xN, xN, xN`）

在 `pto2_submit_mixed_task()` 内各阶段（alloc、fanin-lookup、sync、cleanup、input-lookup、output-materialize、output-register、tensormap-insert、batch-write-GM）前后插入 AArch64 汇编 NOP 标记。

**必要性：条件必须**

与 `aicpu_executor.cpp` 中的标记相同：
- 功能测试不依赖这些标记。
- 指令计数性能分析（`run_tests.sh --count-submit-task-instructions`）依赖这些标记作为 QEMU 插件的阶段边界。
- 在 x86_64 平台上直接使用内联 `__asm__` 而非宏，**仅在 `defined(__aarch64__)` 分支内生效**，x86 构建不受影响。

#### 6b. 分配调试日志（`AICPU_UT_PTO2_ALLOC_DEBUG`）

通过环境变量 `AICPU_UT_PTO2_ALLOC_DEBUG=N` 控制最多打印 N 次分配详情日志。

**必要性：不必须（辅助调试功能）**

这是一个运行时诊断工具，默认关闭（环境变量未设置时直接返回 0）。删除这部分代码不影响任何测试的通过与否，仅影响调试体验。

**对生产代码的影响**

6a 中的汇编标记在非 AArch64 平台下不编译。6b 中 `pto2_alloc_debug_take_ticket()` 在 env 未设置时开销为一次 `getenv` + 静态缓存，可忽略不计。但**这是本次改动中唯一一处在非仿真宏保护下修改生产代码逻辑的位置**，需要关注。

---

### 7. `python/toolchain.py`（+42 行）

**改动内容**

新增 `_cmake_host_compiler_from_env()` 函数，将 `CC`/`CXX` 环境变量中可能包含的多段命令（如 Conda 设置的 `gcc -pthread -B /path/to/compiler_compat`）拆分为编译器路径和额外 flags，分别映射到 `CMAKE_C_COMPILER` 和 `CMAKE_C_FLAGS_INIT`。

**必要性：构建必须（特定环境）**

`tests/aicpu_ut/CMakeLists.txt` 通过 `run_tests.sh` 调用 `python/toolchain.py` 提供的 CMake 参数来配置宿主编译器。在 Conda 环境或部分 Linux 发行版中，`CC` 被设置为多段字符串，原有代码直接将整段字符串传给 `CMAKE_C_COMPILER`，CMake 会报错 `Specified C compiler 'gcc -pthread ...' is not valid`。这一改动使测试在这类环境下能正常构建。

**对生产代码的影响：无负面影响**

改动只是更健壮地解析已有环境变量，不改变行为语义。在标准环境（`CC=gcc`）下，`extra` 为空字符串，`args` 内容与原来完全一致。

---

### 8. `examples/a2a3/tensormap_and_ringbuffer/multi-round-paged-attention/golden.py`（符号链接 → 实体文件）

**改动内容**

将原来指向 `../paged_attention/golden.py` 的符号链接替换为一个独立的实体文件，内容为包含 4 个 Case 的 paged attention golden 测试配置。

**必要性：与 tests/aicpu_ut 无关**

`tests/aicpu_ut/` 中没有任何代码引用该文件。该改动属于 `examples/` 下独立示例的维护，可能与某些 CI/packaging 流程（符号链接在 Windows 或特定打包工具下不兼容）有关。

---

## 必要性分类汇总

### 第一类：硬性依赖（缺失则编译/运行失败）

| 改动 | 阻断原因 |
|---|---|
| `src/a2a3/sim/` 全部新文件 | `sim_run_pto2.cpp` 直接 `#include` 并调用，缺失导致链接错误 |
| `platform_regs.cpp` 仿真钩子 | 无钩子时对 core_id 裸地址访问，立即 segfault |
| `aicpu_executor.cpp` 仿真模式逻辑 | `handshake_all_cores` 失败、`dispatch_to_core` 崩溃 |
| `runtime.h/.cpp` deferred_orch 标志 | 并发模式 Scheduler 提前退出，任务丢失 |
| `tensor_factory.h` | 测试用例 `#include` 此头，缺失则编译错误 |

### 第二类：功能性依赖（缺失则特定测试模式失效）

| 改动 | 影响范围 |
|---|---|
| `aicpu_executor.cpp` dispatch_timestamp 提升 | `test_latency.cpp` 延迟统计为零值 |
| `aicpu_executor.cpp` 指令计数标记 | `--count-submit-task-instructions` 分析输出为空 |
| `pto_orchestrator.cpp` 指令计数标记 | 同上，Orchestrator 阶段标记缺失 |
| `python/toolchain.py` CC/CXX 处理 | Conda 环境下 CMake 配置失败，测试无法构建 |

### 第三类：辅助功能（可删除，不影响测试通过）

| 改动 | 说明 |
|---|---|
| `pto_orchestrator.cpp` 分配调试日志 | 仅诊断用途，默认关闭 |
| `examples/.../golden.py` 符号链接变实体 | 与 `tests/aicpu_ut` 无任何依赖关系 |

---

## 生产代码侵入性评估

所有对生产代码（非 `tests/`、非 `sim/`）的修改均采用了以下保护手段之一：

1. **`#if defined(PTO2_SIM_AICORE_UT)`**：仿真路径，正常构建不编译。
2. **`#if defined(__aarch64__)`**：平台限定，x86 构建不编译。
3. **运行时环境变量检查**：默认关闭（`pto2_alloc_debug_take_ticket()` 在 env 未设置时立即返回 false）。

唯一值得关注的**无宏保护的生产代码改动**是 `pto_orchestrator.cpp` 中 `dispatch_timestamp` 的提升（从 `if (profiling_enabled)` 内移到外层）。这在非 profiling 构建中多执行一次 `get_sys_cnt_aicpu()` 读取，属于单个寄存器读操作，实际性能影响可忽略。

---

*生成日期：2026-04-01*
