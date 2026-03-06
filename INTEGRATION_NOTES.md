# Integration Notes - CPU Affinity Support

## 整合日期
2026-03-06

## 整合来源
从 simpler-03053 项目整合 CPU 亲和性支持功能

## 整合内容

### 1. 新增文件
- `tests/orchestration_ut/common/cpu_affinity.h` - CPU 亲和性工具库头文件
- `tests/orchestration_ut/common/cpu_affinity.cpp` - CPU 亲和性工具库实现
- `tests/orchestration_ut/tests/functional/test_cpu_affinity.cpp` - CPU 亲和性功能测试

### 2. 修改文件
- `tests/orchestration_ut/main.cpp` - 移除 platform_config 测试，保留 CPU 亲和性测试
- `tests/orchestration_ut/Makefile` - 已包含 CPU 亲和性编译参数（ORCH_CPU, SCHED_CPU0-7）

### 3. 已存在的支持
- `src/runtime/tensormap_and_ringbuffer/runtime/runtime.h` - 已包含 CPU 亲和性字段定义
- `src/runtime/tensormap_and_ringbuffer/runtime/runtime.cpp` - 已包含初始化逻辑
- `src/platform/a2a3sim/host/device_runner.cpp` - 已包含线程绑核实现

## 功能特性

### CPU 亲和性配置
通过 Runtime 类的三个字段控制：
- `cpu_affinity_enabled` - 是否启用绑核（默认 false）
- `orch_cpu_core` - orchestrator 线程绑定的 CPU 核编号（-1 = 使用默认值 0）
- `sched_cpu_cores[PLATFORM_MAX_AICPU_THREADS]` - scheduler 线程绑定的 CPU 核编号数组（-1 = 使用默认值 i+1）

### 编译时配置
通过 Makefile 变量指定默认绑核配置：
```bash
make ORCH_CPU=4 SCHED_CPU0=5 SCHED_CPU1=6 SCHED_CPU2=7
```

支持的变量：
- `ORCH_CPU` (默认: 0)
- `SCHED_CPU0` (默认: 1)
- `SCHED_CPU1` (默认: 2)
- `SCHED_CPU2` (默认: 3)
- `SCHED_CPU3` (默认: 4)
- `SCHED_CPU4` (默认: 5)
- `SCHED_CPU5` (默认: 6)
- `SCHED_CPU6` (默认: 7)
- `SCHED_CPU7` (默认: 8)

### 平台配置测试
通过 Makefile 变量指定平台参数：
- `PLATFORM_MAX_BLOCKDIM` (默认: 24)
- `PLATFORM_AIC_CORES_PER_BLOCKDIM` (默认: 1)
- `PLATFORM_AIV_CORES_PER_BLOCKDIM` (默认: 2)
- `PLATFORM_MAX_AICPU_THREADS` (默认: 4)

支持的配置测试目标：
```bash
make test-config1  # 默认配置
make test-config2  # 中等规模
make test-config3  # 大规模（8线程）
make test-config4  # 平衡配置（8线程）
make test-all-configs  # 运行所有配置测试
```

支持的性能测试目标：
```bash
make perf-config1  # 默认配置性能测试
make perf-config2  # 中等规模性能测试
make perf-config3  # 大规模性能测试
make perf-config4  # 平衡配置性能测试
make perf-all-configs  # 运行所有性能配置测试
```

## 测试验证

### 功能测试
```bash
cd tests/orchestration_ut
make run-func
```

测试内容：
1. CPU Affinity - Without Binding（观察 OS 自由调度）
2. CPU Affinity - With Binding (Default)（默认绑核：orch→0, sched[i]→i+1）
3. CPU Affinity - With Binding (Custom)（自定义绑核）
4. CPU Affinity - Comparison（对比有无绑核的差异）

### 性能测试
```bash
cd tests/orchestration_ut
make run-perf
```

测试内容：
1. Paged Attention Basic
2. Batch Paged Attention Basic
3. Batch Paged Attention Chunked
4. Batch Paged Attention Large (block_num=16/128/256)

## 验证结果
- 编译：成功
- 功能测试：PASS=13, FAIL=0
- 性能测试：正常运行，输出 profiling 数据

## 注意事项
1. CPU 亲和性功能仅在 Linux 平台有效
2. 非 Linux 平台会跳过绑核，打印警告但不影响功能
3. 默认情况下 `cpu_affinity_enabled = false`，不影响现有行为
4. 绑核配置仅对 a2a3sim 平台的 AICPU 线程生效
