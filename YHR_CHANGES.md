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
以 `paged_attention_orch.cpp` 为基础进行改造，构建出可以独立编译的、使用CPU模拟AICPU操作的Paged Attention Orchestration测试脚本。该脚本特点如下：
1. 只模拟 AICPU 部分的操作，跳过 AICore 部分
2. 链接静态库
3. 添加了单独的main函数，使其可以独立运行

### 2.1.2 改造范围
- **源文件**: `simpler/tests/device_tests/tensormap_and_ringbuffer/paged_attention/kernels/orchestration/paged_attention_orch.cpp`
- **目标文件**: `simpler/tests/orchestration_ut/test_paged_attention.cpp`
- **相关文件**: 
  - `simpler/tests/orchestration_ut/Makefile` (已包含构建规则)
  - `simpler/tests/unit/Makefile` (已移除 test_sim_orch_sched 相关内容)
  - `simpler/ci.sh` (已移除上板测试调用)

---

## 2.2 改造前后对比

### 2.2.1 文件结构对比

#### 2.2.1.1 改造前（设备测试）
```
simpler/tests/device_tests/tensormap_and_ringbuffer/paged_attention/
├── kernels/
│   ├── orchestration/
│   │   └── paged_attention_orch.cpp  ← 源文件
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
├── test_paged_attention.cpp  ← 新文件
└── Makefile

simpler/tests/unit/
├── Makefile  ← 已移除 test_sim_orch_sched 相关内容
```

**特点**:
- 独立的单元测试程序
- 可在任何 CPU 上运行（不需要硬件）
- 跳过 AICore kernel 执行
- 包含完整的 main 函数和测试框架
- 使用 C++ chrono 库进行计时（替代硬件 cycle counting）

---

## 2.3 详细改造内容

### 2.3.1 移除硬件特定代码

#### 2.3.1.1 移除 ARM 系统计数器指令

**改造前**:
```cpp
inline uint64_t get_sys_cnt_aicpu() {
    uint64_t ticks;
    asm volatile("mrs %0, cntvct_el0" : "=r"(ticks));  // ARM 特定指令
    return ticks;
}

#define CYCLE_COUNT_START() uint64_t _t0 = get_sys_cnt_aicpu(), _t1
#define CYCLE_COUNT_LAP(acc) do { _t1 = get_sys_cnt_aicpu(); acc += (_t1 - _t0); _t0 = _t1; } while(0)
```

**改造后**:
```cpp
// 使用 C++ chrono 库替代硬件 cycle counting
#include <chrono>

auto t_start = std::chrono::high_resolution_clock::now();
// ... 执行代码 ...
auto t_end = std::chrono::high_resolution_clock::now();
auto duration = std::chrono::duration_cast<std::chrono::microseconds>(t_end - t_start);
printf("  Execution time: %lld us (%.3f ms)\n", 
       (long long)duration.count(), duration.count() / 1000.0);
```

**原因**: 
- `mrs cntvct_el0` 是 ARM 架构特定的系统寄存器读取指令
- 在 x86 或其他架构上无法编译或执行
- 使用标准 C++ chrono 库提供跨平台的计时功能
- 仍然可以测量性能，但使用标准库而非硬件特定指令

#### 2.3.1.2 移除性能分析相关代码

**改造前**:
```cpp
uint64_t prof_param_extract = 0;
uint64_t prof_ext_tensor    = 0;
uint64_t prof_make_tensor   = 0;
uint64_t prof_tensor_view   = 0;
uint64_t prof_param_setup   = 0;
uint64_t prof_submit_task   = 0;
int      prof_submit_count  = 0;
int      prof_make_count    = 0;
int      prof_view_count    = 0;

CYCLE_COUNT_START();
// ... 代码 ...
CYCLE_COUNT_LAP(prof_param_extract);
// ... 更多性能分析代码 ...

LOG_ALWAYS(rt, "=== PagedAttn Orch Profiling: %d submits, %d makes, %d views, total=%.3fus ===",
    prof_submit_count, prof_make_count, prof_view_count, cycles_to_us(total));
```

**改造后**:
```cpp
// 使用 chrono 进行性能分析
#include <chrono>

auto t_start = std::chrono::high_resolution_clock::now();
// ... 代码执行 ...
auto t_end = std::chrono::high_resolution_clock::now();
auto duration = std::chrono::duration_cast<std::chrono::microseconds>(t_end - t_start);

printf("  Total tasks submitted: %d\n", total_tasks);
printf("  Timing breakdown:\n");
printf("    TensorPool init:     %lld us\n", (long long)duration_tensor_pool.count());
printf("    Orchestration:        %lld us\n", (long long)duration_orchestration.count());
printf("    Total:                %lld us (%.3f ms)\n", 
       (long long)duration_total.count(), duration_total.count() / 1000.0);
```

**原因**:
- 使用标准 C++ chrono 库替代硬件特定的 cycle counting
- 提供跨平台的性能测量
- 仍然可以分析性能，但使用标准库
- 输出格式更易读（微秒和毫秒）

### 2.3.2 API 调用方式改造

#### 2.3.2.1 Orchestration API → 直接 API

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

static void build_paged_attention_graph(PTO2Runtime* rt, uint64_t* args, int arg_count) {
    TensorPool::set_instance(&rt->orchestrator.tensor_pool);  // 直接设置
    // ...
    pto2_submit_task(&rt->orchestrator, FUNC_AIV_HUB, PTO2_WORKER_VECTOR, params_inplace, 3);
    // ...
}
```

**变化说明**:
| 原 API | 新 API | 原因 |
|--------|--------|------|
| `pto2_rt_init_tensor_pool(rt)` | `TensorPool::set_instance(&rt->orchestrator.tensor_pool)` | 直接访问，无需函数指针 |
| `pto2_rt_submit_task(rt, ...)` | `pto2_submit_task(&rt->orchestrator, ...)` | 直接调用 orchestrator |
| `LOG_ALWAYS(rt, ...)` | `printf(...)` | 简化日志，使用标准输出 |

**原因**:
- 单元测试环境允许直接访问 runtime 内部结构
- 不需要通过函数指针表（orchestration API 的设计目的）
- 更易于调试和跟踪

#### 2.3.2.2 移除 extern "C" 和 visibility 属性

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
```

**原因**:
- 原代码需要被动态加载（dlopen），需要 C 链接和 visibility 控制
- 单元测试是静态链接，不需要这些特性

### 2.3.3 数据类型改造

#### 2.3.3.1 BFLOAT16 → FLOAT32

**改造前**:
```cpp
DataType data_type = DataType::BFLOAT16;  // 用例是float32的，这个考虑要如何扩展成其他类型
```

**改造后**:
```cpp
DataType data_type = DataType::FLOAT32;  // Use FLOAT32 for simulation instead of BFLOAT16
```

**原因**:
- BFLOAT16 在模拟环境中可能不完全支持
- FLOAT32 在所有平台上都有完整支持
- 对于验证 orchestration 逻辑，数据类型不影响测试
- 简化测试数据准备

### 2.3.4 添加测试框架

#### 2.3.4.1 添加模拟执行函数

**新增代码**:
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
- **跳过 AICore 执行**: 不调用实际的 kernel 函数
- **模拟完成**: 直接调用 `on_task_complete()` 标记任务完成
- **验证依赖**: 仍然验证任务依赖关系和状态转换

**原因**:
- 单元测试只关注任务图构建，不关注计算结果
- 不需要实现或模拟复杂的 AICore kernels
- 测试更快速、更简单

#### 2.3.4.2 添加日志桩函数

**新增代码**:
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
- 简化测试环境

#### 2.3.4.3 添加测试宏

**新增代码**:
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

**原因**: 提供标准的测试框架，便于验证和报告结果

### 2.3.5 添加独立的 main 函数

#### 2.3.5.1 测试数据准备

**新增代码**:
```cpp
int main() {
    // 创建模拟运行时
    PTO2Runtime* rt = make_small_runtime();
    
    // 准备测试参数
    const uint64_t batch = 2;
    const uint64_t num_heads = 4;
    const uint64_t head_dim = 8;
    const uint64_t block_size = 4;
    const uint64_t block_num = 2;
    const float scale_value = 0.125f;
    
    // 分配测试缓冲区
    void* query_buf = malloc(query_size);
    void* key_cache_buf = malloc(key_cache_size);
    // ... 更多缓冲区分配 ...
    
    // 初始化测试数据
    memset(query_buf, 0, query_size);
    // ...
    
    // 准备参数数组
    uint64_t args[10] = { ... };
    
    // 执行测试
    {  // Scope for Tensors
        build_paged_attention_graph(rt, args, 10);
        int executed = sim_run_all(rt);
        // 验证结果
        CHECK(executed == rt->orchestrator.tasks_submitted);
    }
    
    // 清理
    free(...);
    pto2_runtime_destroy(rt);
    
    return (g_fail == 0) ? 0 : 1;
}
```

**特点**:
- 完整的测试生命周期管理
- 内存分配和清理
- 测试数据初始化
- 结果验证和报告

### 2.3.6 添加 pto2_orchestrator_done 调用

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

---

## 2.4 代码行数对比

| 项目 | 改造前 | 改造后 | 变化 |
|------|--------|--------|------|
| **总行数** | 288 | 499 | +211 |
| **核心逻辑** | ~200 | ~200 | 基本不变 |
| **性能分析** | ~80 | ~60 | -20 (使用 chrono) |
| **测试框架** | 0 | ~150 | +150 |
| **计时功能** | 0 | ~60 | +60 (chrono) |
| **硬件特定** | ~30 | 0 | -30 |

**说明**: 
- 核心 orchestration 逻辑基本保持不变
- 移除了性能分析和硬件特定代码
- 添加了完整的测试框架

---

## 2.5 编译和构建

### 2.5.1 Makefile 集成

**文件位置**: `simpler/tests/orchestration_ut/Makefile`

**配置内容**:
```makefile
TEST_PA_SRCS := test_paged_attention.cpp
TARGET_PA := test_paged_attention

$(TARGET_PA): $(ALL_SRCS) $(TEST_PA_SRCS)
	$(CXX) $(CXXFLAGS) $(INCLUDES) $^ -o $@ -ldl
	@echo "编译完成: $(TARGET_PA)"

run-pa: $(TARGET_PA)
	@echo ""
	@./$(TARGET_PA)
```

### 2.5.2 编译命令

```bash
cd simpler/tests/orchestration_ut
make build-pa    # 编译
make run-pa      # 运行
make clean       # 清理
```

**注意**: 文件位于 `simpler/tests/orchestration_ut/` 目录，而不是 `simpler/tests/unit/` 目录。

### 2.5.3 测试输出示例

```
=== test_paged_attention_basic ===
  batch = 2, num_heads = 4, head_dim = 8
  Total tasks submitted: 18
  Timing breakdown:
    TensorPool init:     0 us
    Orchestration:        28 us
    Total:                29 us (0.029 ms)
  Tasks submitted: 18
  Simulation execution:  9 us (0.009 ms)
  Tasks executed: 18
  Test timing summary:
    Graph building:     35 us (0.035 ms)
    Task simulation:    10 us (0.010 ms)
  Overall timing:
    Runtime creation:     5367 us (5.367 ms)
    Total test time:      6345 us (6.345 ms)
  PASS: 5, FAIL: 0
```
