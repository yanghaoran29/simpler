# 支持静态编译的AICPU的profiling测试样例

## 概览

测试样例的核心作用为：

1. 新增一套无需昇腾硬件即可在 Host CPU 上运行的编排调度单元测试套件（`tests/aicpu_ut/`）。
2. 对 `platform_config.h` 中的平台常量增加 `#ifndef` 编译期覆盖保护，使测试套件可以在构建时注入自定义平台参数。

### 如何运行测试

**前提条件：** `g++`（支持 C++17）、`cmake >= 3.15`、`make`

**快速开始：**

```bash
cd tests/aicpu_ut

# 运行全部测试（自动完成 CMake 配置 → 编译 → 执行 → 结果汇总）
./run_tests.sh

# 仅运行功能测试
./run_tests.sh --func

# 仅运行性能测试
./run_tests.sh --perf

# 查看所有可用测试
./run_tests.sh --list
```

**运行指定测试：**

```bash
# 运行 test_batch_paged_attention 的全部 5 个参数组（索引 0~4）
./run_tests.sh --test test_batch_paged_attention

# 只运行第 2 组参数（CaseVarSeq2）
./run_tests.sh --test test_batch_paged_attention --idx 2

# 运行单请求 paged attention
./run_tests.sh --test test_paged_attention --idx 0
```

**自定义平台参数（通过环境变量覆盖，脚本自动转发给 CMake）：**

```bash
# 模拟不同硬件规格，关闭 profiling
PLATFORM_MAX_BLOCKDIM=32 PLATFORM_AIV_CORES_PER_BLOCKDIM=3 \
    ./run_tests.sh --no-profiling --perf

# 自定义 CPU 绑核 + 超时
ORCH_CPU=4 SCHED_CPU0=5 SCHED_CPU1=6 TIMEOUT=120 ./run_tests.sh --perf
```

**仅构建不运行，之后手动执行二进制：**

```bash
./run_tests.sh --build-only

# 构建产物位于 build/bin/，可直接运行
./build/bin/test_cpu_affinity
./build/bin/test_platform_config
./build/bin/test_paged_attention_0
./build/bin/test_batch_paged_attention_0   # Case1: batch=64
./build/bin/test_batch_paged_attention_3   # CaseVarSeq2: batch=2
./build/bin/test_batch_paged_attention_4   # CaseVarSeq4: batch=4
```

> 详细参数说明参见 [`tests/aicpu_ut/run_tests.md`](tests/aicpu_ut/run_tests.md)。

---

## 一、测试样例设计

### 2.1 整体架构

```
tests/aicpu_ut/
├── CMakeLists.txt              # CMake 构建系统
├── run_tests.sh                # 构建+运行的 Shell 入口
├── run_tests.md                # 使用文档
├── common/
│   ├── test_common.h/cpp       # 测试框架基础设施
│   ├── test_log_stubs.cpp      # 日志桩函数
│   ├── cpu_affinity.h/cpp      # CPU 绑核工具
│   └── json_cases.h            # JSON 测试用例解析（header-only）
└── tests/
    ├── functional/
    │   ├── test_cpu_affinity.cpp    # 功能测试：CPU 绑核
    │   └── test_platform_config.cpp # 功能测试：平台常量
    └── perf/
        ├── test_paged_attention.cpp       # 性能测试：单请求 Paged Attention
        └── test_batch_paged_attention.cpp # 性能测试：批量 Paged Attention
```

**核心设计原则**：

- 每个测试文件包含独立的 `main()`，编译为独立二进制。
- 所有二进制**静态链接** a2a3 运行时、平台库和 C++ 标准库，无动态依赖（`ldd` 显示 `not a dynamic executable`）。
- CPU 仿真跳过实际 AICore/AICPU kernel 执行，仅验证任务图构建、依赖跟踪和状态转换逻辑。

### 2.2 CPU 仿真机制

测试通过三个核心函数模拟调度运行：

| 函数 | 说明 |
|------|------|
| `make_runtime()` | 在 Host 堆内存中分配并零初始化 `PTO2Runtime`，设置 orchestrator、scheduler、共享内存句柄和就绪队列 |
| `sim_drain_one_pass()` | 单轮驱动：遍历所有 `PTO2WorkerTypes`，依次调用 `get_ready_task()` → `mark_running()` → `on_task_complete()`，不执行实际 kernel |
| `sim_run_all()` | 循环调用 `sim_drain_one_pass()` 直到无任务为止（默认最多 1000 轮），返回总调度任务数 |

这套仿真可以验证：
- 任务图的正确构建
- 依赖关系（fanin/fanout 引用计数）
- 任务状态流转：`PENDING → READY → RUNNING → CONSUMED`

---


## 一、运行时修改

### `src/a2a3/platform/include/common/platform_config.h`

#### 修改动机

测试套件需要在构建时注入自定义平台规格（如减小 `PLATFORM_MAX_BLOCKDIM` 以模拟不同硬件），原来的 `constexpr` 常量是硬编码的，无法通过编译器 `-D` 标志覆盖。

#### 具体改动

所有平台常量均用 `#ifndef / #endif` 包裹，使编译命令行 `-DXXX=N` 的值优先于头文件默认值：

| 常量 | 默认值 | 说明 |
|------|--------|------|
| `PLATFORM_MAX_BLOCKDIM` | 24 | 平台支持的最大 block 维度 |
| `PLATFORM_AIC_CORES_PER_BLOCKDIM` | 1 | 每 blockdim 的 AIC cube 核心数 |
| `PLATFORM_AIV_CORES_PER_BLOCKDIM` | 2 | 每 blockdim 的 AIV cube 核心数 |
| `PLATFORM_CORES_PER_BLOCKDIM` | 自动推导 | 改为 `AIC + AIV` 的派生表达式 |
| `PLATFORM_MAX_AICPU_THREADS` | 4 | AICPU 调度线程最大并行数 |

**关键设计点**：`PLATFORM_CORES_PER_BLOCKDIM` 原来是硬编码 `3`，现在改为：

```cpp
#ifndef PLATFORM_CORES_PER_BLOCKDIM
constexpr int PLATFORM_CORES_PER_BLOCKDIM =
    PLATFORM_AIC_CORES_PER_BLOCKDIM + PLATFORM_AIV_CORES_PER_BLOCKDIM;
#endif
```

这样当 AIC/AIV 单独被覆盖时，总数自动保持一致，避免手动维护误差。

---

## 三、新增文件详解

### 3.1 `CMakeLists.txt`（275 行）

**构建系统设计要点：**

**静态链接策略：**
```cmake
-nodefaultlibs -Wl,--start-group <static-libs> -Wl,--end-group
```
产出完全自包含的可执行文件，无动态库依赖。

**libstdc++.a 三级查找优先级：**
1. CANN toolkit aarch64 交叉编译器路径：`~/Ascend/cann-*/tools/hcc/aarch64-target-linux-gnu/lib64/libstdc++.a`
2. 编译器查询：`g++ -print-file-name=libstdc++.a`
3. 文件系统 glob：`/usr/lib[64]/gcc/*/*`

**CMake 缓存变量（均可命令行覆盖）：**

| 变量 | 默认值 | 说明 |
|------|--------|------|
| `ORCH_CPU` | 0 | Orchestrator 线程绑定核心 |
| `SCHED_CPU0`~`SCHED_CPU7` | 1~8 | 8 个 Scheduler 线程绑定核心 |
| `PLATFORM_MAX_BLOCKDIM` | 24 | 最大 block 维度 |
| `PLATFORM_AIC_CORES_PER_BLOCKDIM` | 1 | AIC 核心数/blockdim |
| `PLATFORM_AIV_CORES_PER_BLOCKDIM` | 2 | AIV 核心数/blockdim |
| `PLATFORM_MAX_AICPU_THREADS` | 4 | AICPU 最大线程数 |
| `PTO2_PROFILING` | ON | 是否启用 profiling 插桩 |

**编译选项：**
```
-O0 -ggdb3 -fno-omit-frame-pointer -Wall -Wextra
-Wno-unused-parameter -Wno-class-memaccess
```
- `-ggdb3`：生成完整 DWARF 信息（含宏展开），支持 `perf annotate` 精确符号解析
- `-fno-omit-frame-pointer`：保留帧指针寄存器，使 `perf` 调用栈展开不产生 `[unknown]` 帧

**CTest 集成：**所有目标通过 `add_test()` 注册，标签为 `"functional"` 或 `"perf"`，超时 600 秒。

**性能测试多参数构建：**通过 `PERF_CASE_IDX` 编译期宏选择测试用例，CMake 遍历索引集生成独立二进制：
- `test_paged_attention_0`
- `test_batch_paged_attention_{0,1,2,3,4}`

**共享 OBJECT 库 `runtime_common`：** 以下源文件编译一次，被所有测试二进制共用：

```
Runtime:
  pto_runtime2.cpp, pto_orchestrator.cpp, pto_scheduler.cpp,
  pto_ring_buffer.cpp, pto_shared_memory.cpp, pto_tensormap.cpp,
  orchestration/common.cpp,
  platform/sim/aicpu/device_time.cpp   (通过 std::chrono 模拟 50 MHz 的 get_sys_cnt_aicpu())

Common 测试支持:
  test_common.cpp, test_log_stubs.cpp, cpu_affinity.cpp
```

---

### 3.2 `common/test_common.h` / `test_common.cpp`（66 + 193 行）

测试框架基础设施，提供：

**宏定义：**

| 宏 | 说明 |
|----|------|
| `CHECK(cond)` | 断言宏，含 `__FILE__`/`__LINE__` 报告，维护全局 `g_pass`/`g_fail` 计数器 |
| `TEST_BEGIN(name)` | 打印测试段标题 |
| `TEST_END()` | 打印 PASS/FAIL 统计 |

**工具函数：**

| 函数 | 说明 |
|------|------|
| `make_runtime()` | 堆分配并初始化 `PTO2Runtime`（orchestrator、task_ring、ready_queues、fanin/fanout 数组、共享内存句柄）|
| `sim_drain_one_pass(rt)` | 单轮任务排空 |
| `sim_run_all(rt, max_rounds)` | 完整排空循环，返回总调度数 |
| `print_orch_profiling()` | 打印 orchestrator 9 子阶段周期数（sync/alloc/params/lookup/heap/insert/fanin/finalize/scope）及每次 submit 平均微秒 |
| `print_sched_profiling(rt)` | 打印调度器统计（各 worker 类型任务数、fanout/fanin 边总数及最大度、轮次统计、dispatch/complete 周期数）|

**`SchedProfilingData` 结构体字段：**
- `tasks_dispatched[4]`：按 worker 类型（CUBE/VECTOR/AI_CPU/ACCELERATOR）分类的调度任务数
- `fanout_edges_total` / `fanout_max_degree`：完成时遍历的 fanout 边总数及最大度
- `tasks_enqueued_by_completion`：因 `on_task_complete` 变为 READY 的任务数
- `fanin_edges_total` / `fanin_max_degree`：fanin 边总数及最大度
- `rounds_total` / `rounds_with_progress`：总轮次及有进展的轮次
- `dispatch_cycle` / `complete_cycle`：调度各阶段周期数

---

### 3.3 `common/test_log_stubs.cpp`（43 行）

提供 AICPU 日志运行时的桩实现，使测试可以不依赖真实日志库链接成功：

- `unified_log_{error,warn,info,debug,always}` → 重定向到 `stderr`/`stdout`
- `perf_aicpu_record_orch_phase()` → 空操作（no-op），满足 `PTO2_PROFILING=1` 时引用的符号（生产实现写入共享内存阶段缓冲区，Host 仿真环境不存在）

---

### 3.4 `common/cpu_affinity.h` / `cpu_affinity.cpp`（83 + 329 行）

CPU 绑核工具，封装 `pthread_setaffinity_np` / `sched_getaffinity`：

| 函数 | 说明 |
|------|------|
| `bind_to_cpu(core)` | 将当前线程绑定到指定核心 |
| `unbind_from_cpu()` | 移除 CPU 亲和性限制（允许所有核心） |
| `get_bound_cpu()` | 返回当前绑定核心编号，未绑定返回 `-1` |
| `verify_cpu_binding(core)` | 断言当前亲和性与期望核心一致 |

CPU 核心编号由 CMake 注入的编译期宏 `ORCH_CPU`、`SCHED_CPU0`…`SCHED_CPU7` 提供。

---

### 3.5 `common/json_cases.h`（181 行）

Header-only 内联 JSON 解析器，用于 `PerfTestCase` 数组：

- 支持字段类型：`string`、`int`、`float`、`int[]`
- 跳过未知字段
- 提供从外部 JSON 文件加载测试参数的扩展能力（当前测试二进制内置 `PERF_CASES[]`）

`PerfTestCase` 字段：`name`、`batch`、`num_heads`、`kv_head_num`、`head_dim`、`block_size`、`block_num`、`scale_value`、`context_lens[]`、`context_lens_count`

---

### 3.6 `tests/functional/test_cpu_affinity.cpp`（41 行）

对 CPU 绑核工具的功能测试：

- `bind_to_cpu()` 将线程绑定到指定核心，通过 `sched_getaffinity()` 和 `/proc/self/status` 双重验证
- `unbind_from_cpu()` 正确恢复完整核心掩码
- `verify_cpu_binding()` 仅在亲和性匹配时通过

---

### 3.7 `tests/functional/test_platform_config.cpp`（148 行）

验证 `platform_config.h` 中 `#ifndef` 保护机制：

- 平台常量以正确默认值可见
- 通过 `-DPLATFORM_MAX_BLOCKDIM=N` 等方式注入的编译期覆盖值能正确传播到编译出的代码中

---

### 3.8 `tests/perf/test_paged_attention.cpp`（360 行）

单请求 Paged Attention 的端到端编排测试：

**适配方式**（相比生产代码 `paged_attention_orch.cpp`）：
- `pto2_rt_init_tensor_pool(rt)` → `TensorPool::set_instance(&rt->orchestrator.tensor_pool)`
- `pto2_rt_submit_task(rt, ...)` → `pto2_submit_task(&rt->orchestrator, ...)`
- 去除 `extern "C"` / `visibility("default")`（直接静态调用不需要）
- 将 `BFLOAT16` 替换为 `FLOAT32`（数据类型不影响任务图结构验证）

`PERF_CASE_IDX` 在编译时选择 `PERF_CASES[]` 中的用例，生成目标：`test_paged_attention_0`（共 1 个用例）。

启用 `PTO2_PROFILING` 时，每次运行后输出 orchestrator 9 阶段周期分解和调度器 fanout/fanin 统计。

---

### 3.9 `tests/perf/test_batch_paged_attention.cpp`（409 行）

批量 Paged Attention 测试，覆盖 5 个生产规模用例（与 `golden.py` 的 `ALL_CASES` 对齐，每个用例生成 >192 个调度任务）：

| 索引 | 名称 | batch | num_heads | head_dim | block_size | context_len |
|------|------|-------|-----------|----------|------------|-------------|
| 0 | Case1 | 64 | 16 | 128 | 128 | 8193 |
| 1 | Case2 | 64 | 64 | 128 | 64 | 8192 |
| 2 | Case3 | 64 | 16 | 128 | 128 | 8192 |
| 3 | CaseVarSeq2 | 2 | — | — | — | [8193, 4096] |
| 4 | CaseVarSeq4 | 4 | — | — | — | [8193, 4096, 1024, 256] |

将 `FLOAT16` 替换为 `FLOAT32`，生成目标：`test_batch_paged_attention_{0,1,2,3,4}`。

---

### 3.10 `run_tests.sh`（282 行）

构建与运行的 Shell 入口，自动完成 CMake 配置 → 并行构建 → 测试执行 → 结果汇总四个步骤。

**命令行参数：**

| 参数 | 说明 |
|------|------|
| *(无参数)* | 构建（如需）+ 运行全部测试 |
| `--func` | 仅运行功能测试 |
| `--perf` | 仅运行性能测试 |
| `--test <name>` | 运行指定测试的全部参数组 |
| `--test <name> --idx <n>` | 运行指定测试的第 n 组参数 |
| `--build-only` | 仅构建，不运行 |
| `--no-profiling` | 向 CMake 传入 `-DPTO2_PROFILING=OFF` |
| `--list` | 列出全部可用测试后退出 |
| `--help` / `-h` | 显示帮助后退出 |

**环境变量覆盖：** `TIMEOUT`、`BUILD_DIR`、`ORCH_CPU`、`SCHED_CPU0`~`SCHED_CPU7`、`PLATFORM_MAX_BLOCKDIM` 等均可通过环境变量传入，脚本转发给 CMake。

示例：
```bash
TIMEOUT=120 ORCH_CPU=4 SCHED_CPU0=5 PLATFORM_MAX_BLOCKDIM=32 ./run_tests.sh --perf
```

---

### 3.11 `run_tests.md`（162 行）

中文使用文档，涵盖：构建前提条件、CMake 手动构建步骤、`run_tests.sh` 参数参考、平台/CPU 覆盖示例、新增测试的注册方式。

---

## 四、关键设计决策总结

| 决策 | 原因 |
|------|------|
| 静态链接 | 消除运行时动态库依赖，测试二进制在任何环境直接可运行 |
| 每个测试独立 `main()` | 独立进程隔离，一个测试崩溃不影响其他测试 |
| `PERF_CASE_IDX` 编译期选择用例 | 性能测试需要精确的 perf 归因，运行时分支会污染测量结果 |
| `#ifndef` 保护平台常量 | 测试套件需要模拟不同硬件规格，同时保持生产代码默认值不变 |
| `PLATFORM_CORES_PER_BLOCKDIM` 改为派生表达式 | 防止 AIC/AIV 单独覆盖时总数与分量不一致 |
| `device_time.cpp` 用 `std::chrono` 模拟 50 MHz | 消除对 AICPU 硬件计时器的依赖，使时序相关代码在 Host 上可编译运行 |
| `perf_aicpu_record_orch_phase()` 提供 no-op 桩 | 避免 `PTO2_PROFILING=1` 时链接失败（生产实现依赖不存在的共享内存区域）|

---

## 五、可用测试一览

| 测试名 | 类型 | 参数索引 | 对应二进制 |
|--------|------|----------|-----------|
| `test_cpu_affinity` | functional | — | `test_cpu_affinity` |
| `test_platform_config` | functional | — | `test_platform_config` |
| `test_paged_attention` | perf | 0 | `test_paged_attention_0` |
| `test_batch_paged_attention` | perf | 0 1 2 3 4 | `test_batch_paged_attention_{n}` |
