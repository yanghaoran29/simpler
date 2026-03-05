# YHR 代码改造文档

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

## 2.5 核心问题解答

### 2.5.1 直接调用 on_task_complete 跳过 AICore 的原因解析
PTO2SchedulerState 是纯状态机，仅维护任务状态逻辑，不依赖硬件交互。

#### 2.5.1.1 真实硬件执行路径（aicpu_executor.cpp）
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

#### 2.5.1.2 模拟测试执行路径（sim_drain_one_pass）
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

#### 2.5.1.3 为何直接调用合法？（调度器的无感知特性）
`PTO2SchedulerState::on_task_complete()` 的内部逻辑（pto_scheduler.h:314）仅处理**任务状态管理**，完全不涉及硬件：
1. 计数器更新：`tasks_completed++`（统计完成的任务数）
2. 消费者处理：遍历 fanout 链表 → 对每个消费者减 `fanin_refcount` → 若计数归零则将任务加入 ready queue
3. 生产者处理：遍历 fanin 链表 → 对每个生产者加 `fanout_refcount` → 若满足条件则标记任务为 CONSUMED

调度器只关心「某个任务 ID 被告知完成」，不感知这个“告知”的来源（是硬件寄存器轮询触发，还是测试代码直接调用）。

#### 2.5.1.4 关键前提：PTO2_MODE_SIMULATE + init_task_on_submit=true
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
