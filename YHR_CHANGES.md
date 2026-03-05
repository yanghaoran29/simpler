# YHR 代码改造文档

本文档记录了代码改造的主要内容，包括静态链接改造、单元测试移植等内容。

---

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

## 注意事项

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

# 2. Paged Attention Orchestration 单元测试改造

## 2.1 改造概述

### 2.1.1 改造目标
以 `paged_attention_orch.cpp` 为基础，形成可独立编译、基于CPU模拟AICPU操作的Paged Attention Orchestration测试脚本。脚本仅模拟AICPU相关操作，跳过AICore部分，采用静态库链接方式，内置独立main函数，支持直接运行。

### 2.1.2 改造范围
- **源文件**: `simpler/tests/device_tests/tensormap_and_ringbuffer/paged_attention/kernels/orchestration/paged_attention_orch.cpp`
- **目标文件**: `simpler/tests/orchestration_ut/test_paged_attention.cpp`
- **相关文件**: 
  - `simpler/tests/orchestration_ut/Makefile` 包含对应构建规则
  - `simpler/tests/unit/Makefile` 移除 test_sim_orch_sched 相关内容

## 2.2 改造前后对比

### 2.2.1 文件结构对比

#### 2.2.1.1 改造前（设备测试）
```
simpler/tests/device_tests/tensormap_and_ringbuffer/paged_attention/
├── kernels/
│   ├── orchestration/
│   │   └── paged_attention_orch.cpp
│   ├── aic/
│   ├── aiv/
│   └── kernel_config.py
└── golden.py
```
文件以orchestration函数形式存在，由AICPU executor动态加载，依赖Ascend真实硬件，执行实际AICore kernels，无独立main函数。

#### 2.2.1.2 改造后（单元测试）
```
simpler/tests/orchestration_ut/
├── test_paged_attention.cpp
└── Makefile

simpler/tests/unit/
└── Makefile
```
文件为独立单元测试程序，可在通用CPU环境运行，跳过AICore kernel执行，包含完整main函数与测试框架，采用C++ chrono库完成计时工作。

## 2.3 详细改造内容

### 2.3.1 移除硬件特定代码

#### 2.3.1.1 移除 ARM 系统计数器指令
删除ARM架构专属的`cntvct_el0`系统计数器读取指令及相关宏定义，替换为C++标准chrono库实现跨平台高精度计时，支持微秒、毫秒级时间统计与打印。

#### 2.3.1.2 移除性能分析相关代码
移除基于硬件周期计数的性能统计变量与打点逻辑，采用chrono库对TensorPool初始化、编排流程、总耗时等环节做分段计时，输出格式化的时间与任务统计信息。

### 2.3.2 API 调用方式改造

#### 2.3.2.1 Orchestration API → 直接 API
将原通过函数指针表的orchestration API调用，改为直接访问runtime内部结构与orchestrator接口，日志输出从平台日志接口替换为标准printf输出，简化调用链路与调试难度。

#### 2.3.2.2 移除 extern "C" 和 visibility 属性
删除用于动态加载的`extern "C"`声明与默认可见性属性修饰，代码以普通静态链接C++函数形式存在，适配单元测试编译链接形态。

### 2.3.3 数据类型改造
将数据类型从BFLOAT16统一切换为FLOAT32，依托FLOAT32的全平台兼容性，降低模拟环境数据处理复杂度，不影响编排逻辑验证。

### 2.3.4 添加测试框架

#### 2.3.4.1 添加模拟执行函数
新增`sim_drain_one_pass`与`sim_run_all`模拟执行函数，直接标记任务完成状态，跳过真实AICore核函数执行，保留任务依赖与状态流转校验能力。

#### 2.3.4.2 添加日志桩函数
实现`unified_log_error/warn/info/debug/always`等日志桩函数，替代平台原生日志系统，消除底层日志依赖。

#### 2.3.4.3 添加测试宏
定义`CHECK`断言宏、`TEST_BEGIN`/`TEST_END`用例宏，维护全局通过/失败计数，形成标准化测试断言与结果上报能力。

### 2.3.5 添加独立的 main 函数
新增完整main函数，完成模拟运行时创建、测试参数与缓冲区初始化、测试用例执行、结果校验、资源释放与返回值判定，覆盖全测试生命周期。

### 2.3.6 添加 pto2_orchestrator_done 调用
在编排流程末尾显式调用`pto2_orchestrator_done`，向调度器标记任务提交结束，保证调度流程完整。

# 3. Batch Paged Attention 单元测试与代码重构

## 3.1 概述
在`test_paged_attention.cpp`基础上，新增`test_batch_paged_attention.cpp`测试文件，完成批处理场景编排验证，并对测试框架做统一重构，抽取公共代码、优化目录结构。

## 3.2 新增测试文件

### 3.2.1 test_batch_paged_attention.cpp
文件位于`orchestration_ut/tests`目录，源自batch版paged attention编排代码，支持IN_CORE_BATCH=16的批分块逻辑，提供基础功能与分块功能两个测试用例，采用FLOAT32模拟，接口形态与基础测试保持一致。

## 3.3 代码重构

### 3.3.1 目录结构重组
测试工程划分为`common`公共代码目录、`tests`测试用例目录、`build`构建输出目录，`main.cpp`作为统一入口，Makefile管理整体构建，实现公共逻辑与用例分离。

### 3.3.2 提取公共代码
在`common`目录下维护测试宏、通用辅助函数、日志桩实现，`test_common.h/cpp`提供运行时创建、任务模拟、数据转换等通用能力，`test_log_stubs.cpp`提供统一日志桩。

### 3.3.3 统一测试入口
`main.cpp`集中声明并顺序执行所有测试用例，全局统计用例通过/失败数量，输出整体测试汇总结果，以返回值标识测试整体状态。

### 3.3.4 Makefile 更新
Makefile按目录分组管理源文件，指定头文件路径，将可执行文件输出至`build`目录，支持一键编译、运行与清理。

## 3.4 测试结果
编译过程无硬件依赖，可在通用x86环境完成构建与执行。所有用例任务提交数量符合预期，任务依赖与分块逻辑正常，任务模拟执行完整，全部断言通过，测试流程稳定可控。