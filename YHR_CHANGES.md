# 1. 静态链接改造

## 1.1 概述

将 a2a3sim 平台从动态共享库 (.so) 加载改为静态链接 (.a 静态库)，消除运行时 dlopen 调用。

## 1.2 修改的文件

### 1.2.1 CMake 配置

#### 1.2.1.1 Host CMakeLists.txt

**文件**: `src/platform/a2a3sim/host/CMakeLists.txt`

**修改内容**:
- 添加 `STATIC_ORCH_LINK` 编译定义（第 78 行）
- 移除 `--whole-archive` 链接标志（第 95-102 行）
- 添加 `-Wl,--allow-multiple-definition` 处理共享符号（第 94 行）

**关键代码**:
```cmake
# 定义 STATIC_ORCH_LINK：用直接调用替换 dlopen
target_compile_definitions(host_runtime PRIVATE STATIC_ORCH_LINK)

# 链接静态库（不使用 --whole-archive）
if(DEFINED AICPU_STATIC_LIB AND DEFINED AICORE_STATIC_LIB)
    target_link_libraries(host_runtime PRIVATE
        "${AICPU_STATIC_LIB}" "${AICORE_STATIC_LIB}"
    )
endif()

# 允许多重定义
target_link_libraries(host_runtime
    PRIVATE
        pthread
        dl
        -Wl,--allow-multiple-definition 
)
```

#### 1.2.1.2 AICPU CMakeLists.txt

**文件**: `src/platform/a2a3sim/aicpu/CMakeLists.txt`

**修改内容**:
- 添加 `STATIC_ORCH_LINK` 编译定义（第 66 行）
- 排除 `unified_log_device.cpp` 避免符号冲突（第 35 行）

**关键代码**:
```cmake
# 排除 unified_log_device.cpp
file(GLOB COMMON_SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/../../src/aicpu/*.cpp")
list(FILTER COMMON_SOURCES EXCLUDE REGEX ".*unified_log_device\\.cpp$")

# 定义 STATIC_ORCH_LINK
target_compile_definitions(aicpu_kernel PRIVATE STATIC_ORCH_LINK)
```

#### 1.2.1.3 AICore CMakeLists.txt

**文件**: `src/platform/a2a3sim/aicore/CMakeLists.txt`

**修改内容**:
- 添加 `STATIC_ORCH_LINK` 编译定义（第 62 行）

**关键代码**:
```cmake
target_compile_definitions(aicore_kernel PRIVATE __CPU_SIM STATIC_ORCH_LINK)
```

### 1.2.2 Runtime Maker 修改

#### 1.2.2.1 host_build_graph Runtime Maker

**文件**: `src/runtime/host_build_graph/host/runtime_maker.cpp`

**修改内容**:
- 添加静态链接模式支持（第 100-161 行）
- 使用 `dlsym(RTLD_DEFAULT)` 查找已链接的编排函数
- 保护 `dlclose(handle)` 调用（第 175-177 行）

**关键代码**:
```cpp
#ifdef STATIC_ORCH_LINK
    // 静态链接模式：编排函数已链接到此库中
    dlerror();
    orch_func = reinterpret_cast<OrchestrationFunc>(dlsym(RTLD_DEFAULT, orch_func_name));
    const char* dlsym_error = dlerror();
    if (dlsym_error != nullptr || orch_func == nullptr) {
        LOG_ERROR("dlsym(RTLD_DEFAULT) failed for '%s': %s",
                  orch_func_name, dlsym_error ? dlsym_error : "symbol not found");
        return -1;
    }
    LOG_INFO("Static link mode: resolved orchestration function '%s'", orch_func_name);
#else
    // 动态加载模式：从二进制数据加载编排 SO
    // ... dlopen 逻辑 ...
#endif

// 清理时保护 dlclose
#ifndef STATIC_ORCH_LINK
    dlclose(handle);
#endif
```

#### 1.2.2.2 aicpu_build_graph Runtime Maker

**文件**: `src/runtime/aicpu_build_graph/host/runtime_maker.cpp`

**修改内容**:
- 添加静态链接模式支持（第 127-148 行）
- 在静态模式下跳过 SO 存储
- 修复常量名称：`RUNTIME_MAX_AICPU_ORCH_SO_SIZE`（第 142 行）

**关键代码**:
```cpp
#ifdef STATIC_ORCH_LINK
    // 静态链接模式：跳过 SO 存储，仅存储函数名供 AICPU 执行器解析
    std::cout << "Static link mode: orchestration function '" << orch_func_name
              << "' will be resolved by AICPU executor\n";
#else
    // 动态加载模式：存储编排 SO 供 AICPU 加载
    if (!runtime->try_set_aicpu_orch_so(orch_so_binary, orch_so_size)) {
        std::cerr << "Error: Failed to store AICPU orchestration SO "
                     "(size=" << orch_so_size << " bytes, max="
                     << RUNTIME_MAX_AICPU_ORCH_SO_SIZE << ")\n";
        return -1;
    }
#endif
```

### 1.2.3 AICPU Executor 修改

**文件**: `src/runtime/aicpu_build_graph/aicpu/aicpu_executor.cpp`

**修改内容**:
- 保护 `write_bytes_to_file()` 函数（第 41-58 行）
- 添加静态链接模式支持（第 78-154 行）
- 使用 `dlsym(RTLD_DEFAULT)` 解析编排函数
- 修复变量作用域问题（第 846, 888 行）

**关键代码**:
```cpp
// 仅在动态模式下需要此函数
#ifndef STATIC_ORCH_LINK
int write_bytes_to_file(const char* path, const uint8_t* data, size_t size) {
    // ... 实现 ...
}
#endif

// 在 build_graph_via_aicpu_plugin 中：
#ifdef STATIC_ORCH_LINK
    ensure_current_so_is_global(thread_idx);
    func = reinterpret_cast<AicpuBuilderFunc>(::dlsym(RTLD_DEFAULT, sym));
    DEV_INFO("Thread %d: Resolved orchestration function '%s' from static link", thread_idx, sym);
#else
    // 动态加载模式：从嵌入的二进制数据加载 SO
    const void* so_data_v = runtime->get_aicpu_orch_so_data();
    // ... dlopen 逻辑 ...
#endif
```

## 1.3 技术要点

### 1.3.1 静态链接 vs 动态加载

**动态加载模式**:
- 编排函数编译为 .so 文件
- 运行时通过 dlopen/dlsym 加载
- 需要临时文件或内存中的 SO 数据

**静态链接模式**:
- 编排函数编译为 .o 文件
- 链接时直接合并到最终二进制
- 使用 `dlsym(RTLD_DEFAULT)` 查找符号
- 无需临时文件或 SO 数据传递

### 1.3.2 符号解析

- `RTLD_DEFAULT`: 在当前进程的全局符号表中查找
- `ensure_current_so_is_global()`: 将当前 .so 提升为 RTLD_GLOBAL 模式
- 内核入口点通过 `objcopy --redefine-sym` 重命名为 `kernel_entry_N`

### 1.3.3 编译流程

1. 编译内核 → kernel_N.o (重命名入口点)
2. 编译编排函数 → orch.o
3. 生成 kernel_dispatch.cpp (分发表)
4. 构建 aicpu.a 和 aicore.a (包含运行时 + 内核 + 编排)
5. 构建 host_runtime.so (链接 aicpu.a 和 aicore.a)

## 1.4 保留符号信息支持 perf 分析
### 1.4.1 修改文件
**文件路径**：`tests/orchestration_ut/Makefile`

### 修改内容
将 `CXXFLAGS` 编译选项调整，增强调试与性能分析能力：
```makefile
# 修改前
CXXFLAGS := -std=c++17 -O0 -g \
            -Wall -Wextra -Wno-unused-parameter \
            -DPTO2_PROFILING=0

# 修改后
CXXFLAGS := -std=c++17 -O0 -ggdb3 -fno-omit-frame-pointer \
            -Wall -Wextra -Wno-unused-parameter \
            -DPTO2_PROFILING=0
```

### 1.4.2 关键选项说明
| 编译选项 | 核心作用 |
|----------|----------|
| `-ggdb3` | 生成最完整的 GDB/DWARF 调试信息，包含宏展开记录，确保 `perf annotate` 符号解析精准 |
| `-fno-omit-frame-pointer` | 保留帧指针寄存器（RBP/FP），避免 `perf` 调用栈展开时出现截断 |

### 1.4.3 使用方式
```bash
cd tests/orchestration_ut
make build
perf record -g ./build/test_orchestration  # 采集性能数据
perf report                                 # 分析性能报告
```

## 1.5 静态链接设计（tests/orchestration_ut）
### 1.5.1 原始链接方式
```makefile
$(CXX) ... -o $@ -ldl -pthread
```

### 1.5.2 改造后（完全静态链接）
```makefile
# 静态库路径配置
STATIC_LIBSTDCXX := $(HOME)/Ascend/cann-8.5.0/tools/hcc/aarch64-target-linux-gnu/lib64/libstdc++.a
STATIC_LIBGCC := $(shell g++ -print-file-name=libgcc.a)
STATIC_LIBGCC_EH := $(shell g++ -print-file-name=libgcc_eh.a)
STATIC_LIBM := /usr/lib64/libm.a
STATIC_LIBC := /usr/lib64/libc.a
STATIC_LIBDL := /usr/lib64/libdl.a
STATIC_LIBPTHREAD:= /usr/lib64/libpthread.a

# 静态链接命令
$(CXX) ... -o $@ \
-nodefaultlibs \
-Wl,--start-group \
$(STATIC_LIBSTDCXX) $(STATIC_LIBM) $(STATIC_LIBPTHREAD) $(STATIC_LIBDL) \
$(STATIC_LIBC) $(STATIC_LIBGCC) $(STATIC_LIBGCC_EH) \
-Wl,--end-group
```

### 1.5.3 关键技术点
| 链接选项 | 作用 |
|----------|------|
| `-nodefaultlibs` | 禁用编译器默认附加的库，改为手动指定静态库 |
| `-Wl,--start-group/--end-group` | 允许链接器在组内多次扫描，解决静态库循环依赖问题 |
| `libstdc++.a` | 来源：Ascend CANN 工具链（适配 aarch64 架构） |
| `libm/libc/libdl/libpthread.a` | 来源：EulerOS 系统 `/usr/lib64/` 目录 |

### 1.5.4 验证结果
```bash
$ file build/test_orchestration
ELF 64-bit LSB executable, ARM aarch64, statically linked, with debug_info
$ ldd build/test_orchestration
not a dynamic executable
$ ./build/test_orchestration
All Tests Summary: PASS=15, FAIL=0
```

## 1.6 注意事项

1. **符号冲突处理**:
   - 使用 `-Wl,--allow-multiple-definition` 允许多重定义
   - 排除 `unified_log_device.cpp` 避免日志函数冲突
   - 移除 `--whole-archive` 让链接器自然解析符号

2. **条件编译**:
   - 所有平台特定代码使用 `#ifdef STATIC_ORCH_LINK` 保护
   - 确保 CMakeLists.txt 中定义了该宏

3. **向后兼容**:
   - a2a3 硬件平台仍使用动态加载模式
   - 仅 a2a3sim 使用静态链接

# 2. Orchestration单元测试改造
本次改造围绕两个样例展开，分别完成了paged_attention和batch_paged_attention的Orchestration单元测试改造工作，核心是将依赖硬件的设备测试改造为可在CPU环境独立运行的单元测试，并对测试框架进行了重构优化。

## 2.1 改造概述
以paged_attention对应的源文件为基础，构建出可独立编译、使用CPU模拟AICPU操作的测试脚本；同时基于该改造经验，进一步完成了batch_paged_attention测试文件的创建，并对整个测试框架进行重构，提取公共代码、优化目录结构。

### 2.1.1 改造目标
针对paged_attention，以其源文件为基础改造出的测试脚本具备以下特点：
1. 只模拟 AICPU 部分的操作，跳过 AICore 部分
2. 链接静态库
3. 添加了单独的main函数，使其可以独立运行

针对batch_paged_attention，除实现上述基础目标外，还需支持批处理分块架构，验证不同批次大小下的分块逻辑有效性。

### 2.1.2 改造范围
- **paged_attention相关文件**: 
  - 源文件: `simpler/tests/device_tests/tensormap_and_ringbuffer/paged_attention/kernels/orchestration/paged_attention_orch.cpp`
  - 目标文件: `simpler/tests/orchestration_ut/test_paged_attention.cpp`
- **batch_paged_attention相关文件**:
  - 改造来源: `simpler/examples/tensormap_and_ringbuffer/batch_paged_attention/kernels/orchestration/paged_attention_orch.cpp`
  - 目标文件: `simpler/tests/orchestration_ut/tests/test_batch_paged_attention.cpp`
- **公共关联文件**: 
  - `simpler/tests/orchestration_ut/Makefile` (已包含构建规则)
  - `simpler/tests/unit/Makefile` (已移除 test_sim_orch_sched 相关内容)
  - `simpler/ci.sh` (已移除上板测试调用)

## 2.2 改造前后对比

### 2.2.1 文件结构对比

#### 2.2.1.1 改造前（设备测试）
```
simpler/tests/device_tests/tensormap_and_ringbuffer/paged_attention/
├── kernels/
│   ├── orchestration/
│   │   └── paged_attention_orch.cpp  ← paged_attention源文件
│   ├── aic/
│   ├── aiv/
│   └── kernel_config.py
└── golden.py
```

**特点**:
- 作为 orchestration 函数，由 AICPU executor 动态加载
- 需要真实硬件（Ascend 设备）执行
- 执行实际的 AICore kernels
- 无独立的 main 函数

#### 2.2.1.2 改造后（单元测试）
```
simpler/tests/orchestration_ut/
├── common/                          # 公共代码目录
│   ├── test_common.h               # 测试框架宏和函数声明
│   ├── test_common.cpp             # 公共辅助函数实现
│   └── test_log_stubs.cpp          # 统一日志桩函数
├── tests/                           # 测试用例目录
│   ├── test_paged_attention.cpp    # paged_attention测试
│   └── test_batch_paged_attention.cpp  # batch_paged_attention测试
├── build/                           # 构建输出目录
│   └── test_orchestration          # 可执行文件
├── main.cpp                         # 统一测试入口
└── Makefile                         # 构建配置

simpler/tests/unit/
├── Makefile  ← 已移除 test_sim_orch_sched 相关内容
```

**特点**:
- 独立的单元测试程序，同时包含两个样例的测试用例
- 可在任何 CPU 上运行（不需要硬件）
- 跳过 AICore kernel 执行
- 包含完整的 main 函数和测试框架
- 使用 C++ chrono 库进行计时（替代硬件 cycle counting）
- 公共代码与测试用例分离，目录结构更清晰

## 2.2.2 Orchestration 单元测试 Profiling 集成
### 2.2.2.1 背景
`simpler/tests/orchestration_ut` 是 PTO Runtime 编排调度层的单元测试，可在无 AICore 硬件依赖的情况下，验证任务图构建、依赖关系及 Scope 生命周期。

### 2.2.2.2 改动文件清单
#### 1. `tests/orchestration_ut/Makefile`
| 配置项 | 修改前 | 修改后 |
|--------|--------|--------|
| `CXXFLAGS` | `-DPTO2_PROFILING=0` | `-DPTO2_PROFILING=1` |
| `RUNTIME_SRCS` | 无 `device_time.cpp` | 新增 `src/platform/a2a3sim/aicpu/device_time.cpp` |

> 补充说明：`device_time.cpp` 提供 `get_sys_cnt_aicpu()` 的模拟实现（基于 `std::chrono`），是 `pto_orchestrator.cpp` 中 `CYCLE_COUNT_START()` 宏的底层计时函数。

#### 2. `tests/orchestration_ut/common/test_log_stubs.cpp`
新增空实现桩函数，避免链接失败：
```cpp
#include "common/perf_profiling.h"
// 单元测试无共享内存 profiling buffer，用空实现替代
void perf_aicpu_record_orch_phase(AicpuPhaseId, uint64_t, uint64_t, uint32_t) {}
```
> 原因：启用 `PTO2_PROFILING=1` 后，`pto_orchestrator.cpp` 中的 `CYCLE_COUNT_LAP_RECORD` 宏会调用该函数，而完整 AICPU 运行时中该函数负责写入共享内存，单元测试场景需规避此依赖。

#### 3. 测试用例文件（2个）
`tests/orchestration_ut/tests/test_paged_attention.cpp`、`tests/orchestration_ut/tests/test_batch_paged_attention.cpp` 做了相同结构改动：
- 新增 `#include "common/platform_config.h"`（提供 `cycles_to_us()` 转换函数）；
- 图构建完成后，调用 `pto2_orchestrator_get_profiling()` 打印 orchestrator 各子步骤 cycle 分解；
- 模拟执行完成后，调用 `pto2_scheduler_print_stats()`/`pto2_scheduler_print_queues()` 打印调度器统计；
- 所有新增 profiling 代码均包裹在 `#if PTO2_PROFILING` 编译守卫中，支持编译期关闭。

### 2.2.2.3 Profiling 机制原理
#### 1. 两层 profiling 架构
```
┌─────────────────────────────────────────────────────┐
│  Layer 1: Orchestrator 子步骤 cycle 分解             │
│  数据来源：pto_orchestrator.cpp 全局累积计数器       │
│  接口：pto2_orchestrator_get_profiling()             │
│                                                      │
│  核心子步骤：                                         │
│    sync    — TensorMap 同步等待                      │
│    alloc   — TaskRing slot 分配                      │
│    params  — 参数拷贝                                │
│    lookup  — TensorMap producer 查找                 │
│    heap    — HeapRing buffer 分配                    │
│    insert  — TensorMap producer 注册                 │
│    fanin   — 依赖边建立 (fanin/fanout)               │
│    finalize — 任务状态初始化 + SM 更新               │
│    scope   — scope_end 开销                         │
├─────────────────────────────────────────────────────┤
│  Layer 2: Scheduler 统计                             │
│  接口：pto2_scheduler_print_stats()                  │
│        pto2_scheduler_print_queues()                 │
└─────────────────────────────────────────────────────┘
```

#### 2. Cycle 计时逻辑
- `CYCLE_COUNT_START()`：在 `pto2_submit_task()` 入口调用 `get_sys_cnt_aicpu()` 记录计时起点；
- `CYCLE_COUNT_LAP_RECORD(acc, phase_id)`：在子步骤边界记录时间增量，并累加到对应全局计数器；
- `cycles_to_us(cycles)`：按 `PLATFORM_PROF_SYS_CNT_FREQ = 50 MHz` 将 cycle 数转换为微秒；
- 仿真环境：`a2a3sim` 下 `get_sys_cnt_aicpu()` 通过 `std::chrono::high_resolution_clock` 模拟 50 MHz 计数器，输出微秒值反映真实 CPU 执行时间。

### 2.2.2.4 典型输出示例
```
=== test_paged_attention_basic ===
  Profiling enabled
  batch = 2, num_heads = 4, head_dim = 8
  Total tasks submitted: 18
  === Orchestrator Profiling (18 submits) ===
    sync:      1.220 us  ( 4.2%)
    alloc:     2.040 us  ( 7.1%)
    params:    2.660 us  ( 9.2%)
    lookup:    6.640 us  (23.0%)   ← TensorMap 查找占比最高
    heap:      2.400 us  ( 8.3%)
    insert:    5.480 us  (19.0%)   ← TensorMap 注册占比次高
    fanin:     3.620 us  (12.6%)
    finalize:  3.580 us  (12.4%)
    scope:     1.180 us  ( 4.1%)
    total:     28.820 us
  Tasks submitted: 18
  Tasks executed: 18
  === Scheduler Statistics ===
  ...
  PASS: 5, FAIL: 0
```
> 关键结论：`lookup`（TensorMap 查找）和 `insert`（依赖注册）合计占比约 42%，是 orchestrator 优化的核心方向，与硬件侧实测规律一致。

### 2.2.2.5 编译与运行命令
```bash
cd simpler/tests/orchestration_ut
make          # 编译并运行（输出 profiling 数据）
make build    # 仅编译
make run      # 仅运行
make clean    # 清理编译产物
```
> 关闭 profiling：将 `Makefile` 中 `-DPTO2_PROFILING=1` 改回 `-DPTO2_PROFILING=0` 即可。

### 2.2.2.6 单元测试 vs 硬件侧 Profiling 对比
| 对比项 | 单元测试（本文档） | 硬件侧（a2a3 平台） |
|--------|--------------------|---------------------|
| `get_sys_cnt_aicpu()` | `std::chrono` 模拟 50 MHz | 读取硬件系统计数器寄存器 |
| `perf_aicpu_record_orch_phase()` | no-op 桩函数 | 写入共享内存 phase buffer |
| Orchestrator 数据获取 | 直接读取进程内全局变量 | host 通过设备 log 收集 `AicpuOrchSummary` |
| Scheduler 数据获取 | `pto2_scheduler_print_stats()` | AICPU 侧 `DEV_ALWAYS` 日志 |
| Swimlane 可视化 | 不适用 | `tools/swimlane_converter.py` + `--enable-profiling` |

## 2.3 详细改造内容

### 2.3.1 API 调用方式改造

#### 2.3.1.1 Orchestration API → 直接 API
paged_attention和batch_paged_attention均将间接的orchestration API调用改为直接调用内部函数：

**改造前** (使用 orchestration API):
```cpp
#include "pto_orchestration_api.h"

extern "C" {
__attribute__((visibility("default"))) 
void aicpu_orchestration_entry(PTO2Runtime* rt, uint64_t* args, int arg_count) {
    pto2_rt_init_tensor_pool(rt);  // 通过函数指针表调用
    // ...
    pto2_rt_submit_task(rt, FUNC_AIV_HUB, PTO2_WORKER_VECTOR, params_inplace, 3);
    // ...
}
}
```

**改造后** (直接调用):
```cpp
#include "pto_runtime2.h"

// paged_attention的图构建函数
static void build_paged_attention_graph(PTO2Runtime* rt, uint64_t* args, int arg_count) {
    TensorPool::set_instance(&rt->orchestrator.tensor_pool);  // 直接设置
    // ...
    pto2_submit_task(&rt->orchestrator, FUNC_AIV_HUB, PTO2_WORKER_VECTOR, params_inplace, 3);
    // ...
}

// batch_paged_attention的图构建函数
static void build_batch_paged_attention_graph(PTO2Runtime* rt, uint64_t* args, int arg_count) {
    TensorPool::set_instance(&rt->orchestrator.tensor_pool);
    // ... 批处理分块逻辑 ...
    pto2_submit_task(&rt->orchestrator, FUNC_AIV_HUB, PTO2_WORKER_VECTOR, params_hub, 3);
    // ...
}
```

**变化说明**:
| 原 API | 新 API | 原因 |
|--|--||
| `pto2_rt_init_tensor_pool(rt)` | `TensorPool::set_instance(&rt->orchestrator.tensor_pool)` | 直接访问，无需函数指针 |
| `pto2_rt_submit_task(rt, ...)` | `pto2_submit_task(&rt->orchestrator, ...)` | 直接调用 orchestrator |
| `LOG_ALWAYS(rt, ...)` | `printf(...)` | 简化日志，使用标准输出 |

**原因**:
- 单元测试环境允许直接访问 runtime 内部结构
- 不需要通过函数指针表（orchestration API 的设计目的）
- 更易于调试和跟踪

#### 2.3.1.2 移除 extern "C" 和 visibility 属性
paged_attention和batch_paged_attention均移除了动态加载所需的编译属性：

**改造前**:
```cpp
extern "C" {
__attribute__((visibility("default"))) 
PTO2OrchestrationConfig aicpu_orchestration_config(...) { ... }

__attribute__((visibility("default"))) 
void aicpu_orchestration_entry(...) { ... }
}
```

**改造后**:
```cpp
// 普通 C++ 函数，不需要 extern "C"
static void build_paged_attention_graph(...) { ... }
static void build_batch_paged_attention_graph(...) { ... }
```

**原因**:
- 原代码需要被动态加载（dlopen），需要 C 链接和 visibility 控制
- 单元测试是静态链接，不需要这些特性

### 2.3.2 数据类型改造
paged_attention和batch_paged_attention均将硬件相关的数据类型替换为通用类型：

**改造前**:
```cpp
// paged_attention原代码
DataType data_type = DataType::BFLOAT16;
// batch_paged_attention原代码
DataType data_type = DataType::FLOAT16;
```

**改造后**:
```cpp
// paged_attention改造后
DataType data_type = DataType::FLOAT32;  // Use FLOAT32 for simulation instead of BFLOAT16
// batch_paged_attention改造后
DataType data_type = DataType::FLOAT32;  // Use FLOAT32 for simulation instead of FLOAT16
```

**原因**:
- BFLOAT16/FLOAT16 在模拟环境中可能不完全支持
- FLOAT32 在所有平台上都有完整支持
- 对于验证 orchestration 逻辑，数据类型不影响测试
- 简化测试数据准备

### 2.3.3 添加测试框架
为paged_attention和batch_paged_attention构建了统一的测试框架，提取公共代码避免重复：

#### 2.3.3.1 添加模拟执行函数
**新增公共代码**:
```cpp
static int sim_drain_one_pass(PTO2Runtime* rt) {
    int executed = 0;
    for (int wt = 0; wt < PTO2_NUM_WORKER_TYPES; ++wt) {
        int32_t task_id;
        while ((task_id = rt->scheduler.get_ready_task((PTO2WorkerType)wt)) >= 0) {
            rt->scheduler.mark_running(task_id);
            // 关键：跳过实际 kernel 执行，直接标记完成
            rt->scheduler.on_task_complete(task_id);
            ++executed;
        }
    }
    return executed;
}

static int sim_run_all(PTO2Runtime* rt, int max_rounds = 1000) {
    int total = 0;
    for (int r = 0; r < max_rounds; ++r) {
        int n = sim_drain_one_pass(rt);
        total += n;
        if (n == 0) break;
    }
    return total;
}
```

**关键设计**:
- **跳过 AICore 执行**: 不调用实际的 kernel 函数，对两个样例均适用
- **模拟完成**: 直接调用 `on_task_complete()` 标记任务完成
- **验证依赖**: 仍然验证任务依赖关系和状态转换

**原因**:
- 单元测试只关注任务图构建，不关注计算结果
- 不需要实现或模拟复杂的 AICore kernels
- 测试更快速、更简单

#### 2.3.3.2 添加日志桩函数
**新增公共代码**:
```cpp
extern "C" {
void unified_log_error(const char* func, const char* fmt, ...) {
    va_list ap; va_start(ap, fmt);
    fprintf(stderr, "[ERR]  %s: ", func); vfprintf(stderr, fmt, ap); fputc('\n', stderr);
    va_end(ap);
}
// ... 其他日志函数 ...
}
```

**原因**: 
- 替换完整的日志系统，避免依赖平台特定实现
- 简化测试环境，同时服务于两个测试样例

#### 2.3.3.3 添加测试宏
**新增公共代码**:
```cpp
#define CHECK(cond)                                                          \
    do {                                                                     \
        if (!(cond)) {                                                       \
            fprintf(stderr, "  FAIL [%s:%d]  %s\n",                         \
                    __FILE__, __LINE__, #cond);                              \
            g_fail++;                                                        \
        } else {                                                             \
            g_pass++;                                                        \
        }                                                                    \
    } while (0)

#define TEST_BEGIN(name) printf("\n=== %s ===\n", (name))
#define TEST_END() printf("  PASS: %d, FAIL: %d\n", g_pass, g_fail)
```

**原因**: 提供标准的测试框架，便于两个样例统一验证和报告结果

### 2.3.4 添加独立的测试逻辑和统一入口

#### 2.3.4.1 各自的测试数据准备与执行
- **paged_attention测试逻辑**:
```cpp
void test_paged_attention_basic() {
    TEST_BEGIN("test_paged_attention_basic");
    // 创建模拟运行时
    PTO2Runtime* rt = make_runtime();
    
    // 准备测试参数
    const uint64_t batch = 2;
    const uint64_t num_heads = 4;
    const uint64_t head_dim = 8;
    const uint64_t block_size = 4;
    const uint64_t block_num = 2;
    const float scale_value = 0.125f;
    
    // 分配测试缓冲区、初始化数据、准备参数数组
    // ...
    
    // 执行测试
    build_paged_attention_graph(rt, args, 10);
    int executed = sim_run_all(rt);
    // 验证结果
    CHECK(executed == rt->orchestrator.tasks_submitted);
    
    // 清理
    free(...);
    pto2_runtime_destroy(rt);
    TEST_END();
}
```

- **batch_paged_attention测试逻辑**:
```cpp
void test_batch_paged_attention_basic() {
    TEST_BEGIN("test_batch_paged_attention_basic");
    PTO2Runtime* rt = make_runtime();
    
    // 批处理测试参数
    const uint64_t batch = 2;
    const uint64_t num_heads = 16;
    const uint64_t head_dim = 16;
    const uint64_t q_tile = 16;
    const uint64_t IN_CORE_BATCH = 16;
    
    // 分配缓冲区、初始化数据
    // ...
    
    // 执行测试（包含分块逻辑）
    build_batch_paged_attention_graph(rt, args, 10);
    int executed = sim_run_all(rt);
    CHECK(executed == 9);  // 验证任务数量
    
    free(...);
    pto2_runtime_destroy(rt);
    TEST_END();
}

void test_batch_paged_attention_chunked() {
    TEST_BEGIN("test_batch_paged_attention_chunked");
    // batch=32，验证分块（2个chunk）逻辑
    // ...
}
```

#### 2.3.4.2 统一测试入口
```cpp
#include "test_common.h"

// 定义全局测试计数器
int g_pass = 0;
int g_fail = 0;

// 外部测试函数声明
extern void test_paged_attention_basic();
extern void test_batch_paged_attention_basic();
extern void test_batch_paged_attention_chunked();

int main() {
    printf("========================================\n");
    printf("Orchestration Unit Tests\n");
    printf("========================================\n");

    // 运行所有测试
    printf("\n[1/3] Running Paged Attention Tests\n");
    test_paged_attention_basic();

    printf("\n[2/3] Running Batch Paged Attention (Basic)\n");
    test_batch_paged_attention_basic();

    printf("\n[3/3] Running Batch Paged Attention (Chunked)\n");
    test_batch_paged_attention_chunked();

    // 最终汇总
    printf("\n========================================\n");
    printf("All Tests Summary: PASS=%d, FAIL=%d\n", g_pass, g_fail);
    printf("========================================\n");

    return (g_fail == 0) ? 0 : 1;
}
```

### 2.3.5 补充关键调用
对paged_attention和batch_paged_attention均补充了缺失的收尾调用：

**改造前**:
```cpp
// 原代码中缺少此调用
// 函数直接返回，没有标记 orchestration 完成
```

**改造后**:
```cpp
pto2_orchestrator_done(&rt->orchestrator);  // 显式标记完成
```

**原因**: 
- 确保调度器知道不再有新任务
- 符合最佳实践（其他示例都调用了此函数）

## 2.4 编译和构建

### 2.4.1 编译命令
```bash
cd simpler/tests/orchestration_ut
make clean && make build    # 编译
make run                    # 运行
```

### 2.4.2 测试输出示例
```
========================================
Orchestration Unit Tests
========================================

[1/3] Running Paged Attention Tests

=== test_paged_attention_basic ===
  batch = 2, num_heads = 4, head_dim = 8
  Total tasks submitted: 18
  Timing breakdown:
    TensorPool init:     0 us
    Orchestration:        29 us
    Total:                29 us (0.029 ms)
  Tasks submitted: 18
  Simulation execution:  9 us (0.009 ms)
  Tasks executed: 18
  Test timing summary:
    Graph building:     34 us (0.034 ms)
    Task simulation:    10 us (0.010 ms)
  Overall timing:
    Runtime creation:     8602 us (8.602 ms)
    Total test time:      10889 us (10.889 ms)
  PASS: 5, FAIL: 0

[2/3] Running Batch Paged Attention (Basic)

=== test_batch_paged_attention_basic ===
  batch = 2, num_heads = 16, head_dim = 16, q_tile = 16
  Total tasks submitted: 9
  Expected tasks: 9 (num_chunks=1, max_bn=2, IN_CORE_BATCH=16)
  Timing breakdown:
    TensorPool init:     0 us
    Orchestration:        18 us
    Total:                18 us (0.018 ms)
  Tasks submitted: 9
  Simulation execution:  5 us (0.005 ms)
  Tasks executed: 9
  Test timing summary:
    Graph building:     22 us (0.022 ms)
    Task simulation:    6 us (0.006 ms)
  Overall timing:
    Runtime creation:     8403 us (8.403 ms)
    Total test time:      8453 us (8.453 ms)
  PASS: 10, FAIL: 0

[3/3] Running Batch Paged Attention (Chunked)

=== test_batch_paged_attention_chunked ===
  batch = 32, num_heads = 16, head_dim = 16, q_tile = 16
  Total tasks submitted: 18
  Expected tasks: 18 (num_chunks=2, max_bn=2, IN_CORE_BATCH=16)
  Timing breakdown:
    TensorPool init:     0 us
    Orchestration:        28 us
    Total:                29 us (0.029 ms)
  Tasks submitted: 18
  Simulation execution:  9 us (0.009 ms)
  Tasks executed: 18
  Test timing summary:
    Graph building:     33 us (0.033 ms)
    Task simulation:    10 us (0.010 ms)
  Overall timing:
    Runtime creation:     3673 us (3.673 ms)
    Total test time:      3739 us (3.739 ms)
  PASS: 15, FAIL: 0

========================================
All Tests Summary: PASS=15, FAIL=0
========================================
```

## 2.5 AICPU 线程绑核设计文档
### 2.5.1 背景
PTO Runtime 在模拟模式（a2a3sim）下通过多个 host 线程模拟 AICPU 的执行：当 `sche_cpu_num == 4` 时，Thread 0-2 充当 scheduler，Thread 3 充当 orchestrator。默认情况下这些线程可在任意 CPU 核上自由调度，引入绑核配置后可将其固定到指定核，降低跨核迁移开销并提升调度确定性。

### 2.5.2 线程角色与默认核分配
```
sche_cpu_num == 4 时：
  Thread 0  (scheduler)    → 默认绑 CPU 核 1
  Thread 1  (scheduler)    → 默认绑 CPU 核 2
  Thread 2  (scheduler)    → 默认绑 CPU 核 3
  Thread 3  (orchestrator) → 默认绑 CPU 核 0
sche_cpu_num < 4 时：
  所有线程均为 scheduler，orchestration 在 host 侧完成
  默认绑核规则：Thread i → CPU 核 i+1
```

### 2.5.3 配置接口
绑核配置通过 `Runtime` 类的三个字段控制：
```cpp
class Runtime {
public:
    // 是否启用绑核（默认 false，不影响现有行为）
    bool cpu_affinity_enabled;
    // orchestrator 线程绑定的 CPU 核编号
    // -1 表示使用默认值（核 0）
    int orch_cpu_core;
    // scheduler 线程绑定的 CPU 核编号，下标对应 thread_idx
    // -1 表示使用默认值（核 thread_idx + 1）
    int sched_cpu_cores[PLATFORM_MAX_AICPU_THREADS];
};
```
**字段初始值**（`Runtime` 构造函数）：
| 字段 | 初始值 | 含义 |
|------|--------|------|
| `cpu_affinity_enabled` | `false` | 默认不绑核 |
| `orch_cpu_core` | `-1` | 启用时默认绑核 0 |
| `sched_cpu_cores[i]` | `-1` | 启用时默认绑核 i+1 |

### 2.5.4 使用方式
#### 2.5.4.1 启用默认绑核
```cpp
Runtime runtime;
runtime.cpu_affinity_enabled = true;
// orch_cpu_core 和 sched_cpu_cores 保持 -1，使用默认分配
// 结果：Thread0→核1，Thread1→核2，Thread2→核3，Thread3→核0
```

#### 2.5.4.2 自定义核分配
```cpp
Runtime runtime;
runtime.cpu_affinity_enabled = true;
runtime.orch_cpu_core       = 4;   // orchestrator 绑核 4
runtime.sched_cpu_cores[0]  = 5;   // scheduler 0 绑核 5
runtime.sched_cpu_cores[1]  = 6;   // scheduler 1 绑核 6
runtime.sched_cpu_cores[2]  = 7;   // scheduler 2 绑核 7
```

#### 2.5.4.3 混合配置（部分使用默认，部分指定）
```cpp
Runtime runtime;
runtime.cpu_affinity_enabled = true;
runtime.orch_cpu_core       = -1;  // 默认：绑核 0
runtime.sched_cpu_cores[0]  = 2;   // 指定：绑核 2
runtime.sched_cpu_cores[1]  = -1;  // 默认：绑核 2（thread_idx=1，1+1=2，但实际为 2）
runtime.sched_cpu_cores[2]  = 6;   // 指定：绑核 6
```

### 2.5.5 实现机制
绑核在 `DeviceRunner::run()` 启动 AICPU 线程时，于线程内部最先执行：
```cpp
// device_runner.cpp（a2a3sim）
for (int i = 0; i < launch_aicpu_num; i++) {
    aicpu_threads.emplace_back([&runtime, i, launch_aicpu_num]() {
        if (runtime.cpu_affinity_enabled) {
#ifdef __linux__
            bool is_orchestrator = (launch_aicpu_num == 4 && i == 3);
            int target_core;
            if (is_orchestrator) {
                target_core = (runtime.orch_cpu_core >= 0)
                              ? runtime.orch_cpu_core : 0;
            } else {
                target_core = (runtime.sched_cpu_cores[i] >= 0)
                              ? runtime.sched_cpu_cores[i] : (i + 1);
            }
            cpu_set_t cpuset;
            CPU_ZERO(&cpuset);
            CPU_SET(target_core, &cpuset);
            pthread_setaffinity_np(pthread_self(), sizeof(cpu_set_t), &cpuset);
#endif
        }
        aicpu_execute(&runtime);
    });
}
```
**关键设计点：**
- 绑核在 `aicpu_execute()` 之前完成，保证整个执行周期都在目标核上
- `pthread_setaffinity_np` 失败时仅打印警告，不中断执行
- 仅在 `__linux__` 下编译绑核代码，非 Linux 平台打印警告后继续

### 2.5.6 绑核工具库
绑核相关的基础函数封装在 [`tests/orchestration_ut/common/cpu_affinity.h`](../tests/orchestration_ut/common/cpu_affinity.h) 中，供所有测试文件共用。

#### 2.5.6.1 ThreadReport 结构体
记录单个线程的绑核测试结果：
```cpp
struct ThreadReport {
    int  thread_idx;  // 线程编号
    int  target_cpu;  // 期望绑定的核（-1 = 未设置）
    int  bound_cpu;   // pthread_getaffinity_np 读回的核（-1 = 未特意绑核）
    int  actual_cpu;  // sched_getcpu() 实测运行核
    bool bind_ok;     // bind_to_cpu() 是否成功
};
```

#### 2.5.6.2 工具函数
| 函数 | 说明 |
|------|------|
| `int num_cpus_online()` | 返回系统在线 CPU 核数，非 Linux 返回 1 |
| `int current_cpu()` | 返回当前线程运行的 CPU 核编号（`sched_getcpu`），失败返回 -1 |
| `int bind_to_cpu(int cpu_core)` | 将当前线程绑定到指定核（`pthread_setaffinity_np`），成功返回 0 |
| `int unbind_from_cpu()` | 清除亲和性限制，允许在所有在线核上调度，成功返回 0 |
| `int get_bound_cpu()` | 读取亲和性掩码中第一个绑定核；若覆盖所有在线核则返回 -1（视为未绑核） |
| `bool verify_cpu_binding(int target, const char* ctx)` | 验证当前线程是否在目标核上运行，打印结果并返回是否通过 |
| `void print_separator(int n)` | 打印 n 个 `-` 的缩进分隔线 |

### 2.5.7 单元测试
#### 2.5.7.1 测试结构
测试分为两类，均位于 [`tests/orchestration_ut/`](../tests/orchestration_ut/)：
| 类型 | 文件 | 说明 |
|------|------|------|
| 功能测试 | `tests/test_cpu_affinity.cpp` | 验证绑核行为是否正确，报告 PASS/FAIL |
| 性能测试 | `tests/test_paged_attention.cpp`<br>`tests/test_batch_paged_attention.cpp` | 测量任务图构建与模拟执行的吞吐量，只输出数据 |
**运行顺序**（`main.cpp`）：功能测试在前，性能测试在后，最终只汇报功能测试的 PASS/FAIL。

#### 2.5.7.2 功能测试（test_cpu_affinity.cpp）
在 host CPU 上通过 `pthread_create` 直接创建线程，调用 `pthread_setaffinity_np` + `sched_getcpu()` 验证。
| 测试函数 | 验证内容 |
|---------|---------|
| `test_cpu_affinity_without_binding` | 各线程先调用 `unbind_from_cpu()` 清除继承的亲和性，验证 `get_bound_cpu()` 返回 -1 |
| `test_cpu_affinity_with_binding_default` | 默认分配（orch→核0，sched[i]→核i+1），验证 bound_cpu == target_cpu == actual_cpu |
| `test_cpu_affinity_with_binding_custom` | 自定义分配（orch→最后一核，sched[i]→核i），核数不足时取模回绕，均验证绑定成功 |
| `test_cpu_affinity_comparison` | 对比场景A（不绑核）与场景B（默认绑核），展示线程分布差异 |

#### 2.5.7.3 Makefile 绑核参数
编译时可通过 Makefile 变量指定默认绑核配置，参数以 `-D` 宏注入测试代码：
```bash
# 使用默认值（orch=0，sched0=1，sched1=2，sched2=3）
make
# 自定义绑核配置
make ORCH_CPU=4 SCHED_CPU0=5 SCHED_CPU1=6 SCHED_CPU2=7
```
| 变量 | 默认值 | 说明 |
|------|--------|------|
| `ORCH_CPU` | `0` | orchestrator 线程绑定的 CPU 核 |
| `SCHED_CPU0` | `1` | scheduler 线程 0 绑定的 CPU 核 |
| `SCHED_CPU1` | `2` | scheduler 线程 1 绑定的 CPU 核 |
| `SCHED_CPU2` | `3` | scheduler 线程 2 绑定的 CPU 核 |

### 2.5.8 平台支持
| 平台 | 支持情况 |
|------|---------|
| Linux | 完整支持，使用 `pthread_setaffinity_np` |
| 非 Linux | 跳过绑核，打印 `LOG_WARN`，不影响功能 |

### 2.5.9 注意事项
- `cpu_affinity_enabled` 默认为 `false`，对不使用该特性的调用方完全透明
- 指定的核编号若超出系统可用范围，`pthread_setaffinity_np` 会返回错误，运行时打印警告但继续执行
- 多个线程可配置到同一个核（合法，但可能增加竞争）
- 绑核配置仅对 a2a3sim 平台的 AICPU 线程生效；真实硬件（a2a3）的线程由 CANN 驱动管理，不受此字段影响
- `pthread_create` 子线程继承父线程的亲和性掩码，测试"不绑核"场景时需在线程入口调用 `unbind_from_cpu()` 显式清除

## 2.6 核心问题解答

### 2.6.1 直接调用 on_task_complete 跳过 AICore 的原因解析
PTO2SchedulerState 是纯状态机，仅维护任务状态逻辑，不依赖硬件交互。

#### 2.6.1.1 真实硬件执行路径（aicpu_executor.cpp）
AICPU 调度器的主循环是一个持续轮询的过程，完整依赖硬件寄存器状态：
```
AICPU 调度器主循环 while(true):
    │
    ├─ [完成检测] read_reg(COND)          ← 轮询 AICore SPR 寄存器（硬件层面）
    │   reg_state == TASK_FIN_STATE ?
    │   └─ scheduler.on_task_complete()   ← 硬件状态满足后，通知调度器更新状态
    │
    └─ [派发] get_ready_task()
        └─ write_reg(DATA_MAIN_BASE)      ← 写寄存器触发 AICore 真正执行 kernel（硬件层面）
               ↑
               AICore 读到该寄存器值后才开始执行任务逻辑
```
在真实路径中，`on_task_complete()` 的触发完全依赖**硬件寄存器 COND 的状态变为 TASK_FIN_STATE**，是硬件→软件的被动通知。

#### 2.6.1.2 模拟测试执行路径（sim_drain_one_pass）
测试代码中直接跳过了所有硬件交互环节，强制触发任务完成状态：
```cpp
// test_common.cpp:37
int sim_drain_one_pass(PTO2Runtime* rt) {
    for (int wt ...) {
        while ((task_id = rt->scheduler.get_ready_task(...)) >= 0) {
            rt->scheduler.mark_running(task_id);
            // ↓ 直接跳到「AICore已执行完」的状态，完全绕过硬件交互流程
            rt->scheduler.on_task_complete(task_id);
        }
    }
}
```
被跳过的硬件交互环节包括：
1. 写 DATA_MAIN_BASE 寄存器触发 AICore 执行 kernel
2. AICore 实际运行 kernel 逻辑
3. AICore 完成后改写 COND 寄存器为 TASK_FIN_STATE
4. AICPU 轮询到 COND 寄存器的 FIN 状态

#### 2.6.1.3 为何直接调用合法？（调度器的无感知特性）
`PTO2SchedulerState::on_task_complete()` 的内部逻辑（pto_scheduler.h:314）仅处理**任务状态管理**，完全不涉及硬件：
1. 计数器更新：`tasks_completed++`（统计完成的任务数）
2. 消费者处理：遍历 fanout 链表 → 对每个消费者减 `fanin_refcount` → 若计数归零则将任务加入 ready queue
3. 生产者处理：遍历 fanin 链表 → 对每个生产者加 `fanout_refcount` → 若满足条件则标记任务为 CONSUMED

调度器只关心「某个任务 ID 被告知完成」，不感知这个“告知”的来源（是硬件寄存器轮询触发，还是测试代码直接调用）。

#### 2.6.1.4 关键前提：PTO2_MODE_SIMULATE + init_task_on_submit=true
`make_runtime()` 创建测试运行时环境时（test_common.cpp:28），会触发两个关键配置：
```cpp
pto2_runtime_create_custom(PTO2_MODE_SIMULATE, ...)
```
1. 模式设置：`PTO2_MODE_SIMULATE`（模拟模式）
2. 状态初始化：`orch->init_task_on_submit = true`（提交任务时立即初始化）

这两个配置的作用：
- 每次调用 `pto2_submit_task()` 时，Step 6 会立刻执行 `scheduler.init_task()`，将任务直接加入 ready queue
- orchestrator（编排器）和 scheduler（调度器）在**同一进程、同一线程**中操作，无需真实 AICPU 多线程调度器轮询 `current_task_index`
- 测试代码可直接从 ready queue 中取任务，通过调用 `on_task_complete` 模拟“执行完成”

---
