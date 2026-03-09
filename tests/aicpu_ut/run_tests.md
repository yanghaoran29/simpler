# run_tests.sh

`run_tests.sh` 是编排单元测试的构建与运行入口，位于 `tests/aicpu_ut/`。脚本自动完成 CMake 配置、并行构建、测试执行、结果汇总四个步骤，支持灵活的测试范围筛选。

---

## 快速上手

```bash
# 运行全部测试
./run_tests.sh

# 查看所有可用测试及其参数索引
./run_tests.sh --list

# 查看帮助
./run_tests.sh --help
```

---

## 命令行参数

### 测试范围

| 参数 | 说明 |
|------|------|
| *(无参数)* | 运行全部测试（功能测试 + 性能测试） |
| `--func` | 仅运行功能测试 |
| `--perf` | 仅运行性能测试 |
| `--test <name>` | 运行指定测试的全部参数组 |
| `--test <name> --idx <n>` | 运行指定测试的第 `n` 组参数 |

`--test` 指定后，`--func` / `--perf` 过滤无效，直接运行所指定的测试。

### 构建控制

| 参数 | 说明 |
|------|------|
| `--build-only` | 仅构建，不运行任何测试 |
| `--no-profiling` | 关闭 `PTO2_PROFILING` 宏（默认开启） |

### 辅助

| 参数 | 说明 |
|------|------|
| `--list` | 列出所有可用测试及其类型、参数索引，然后退出 |
| `--help` / `-h` | 显示帮助信息，然后退出 |

---

## 可用测试

| 测试名 | 类型 | 参数索引 | 对应二进制 |
|--------|------|----------|-----------|
| `test_cpu_affinity` | functional | — | `test_cpu_affinity` |
| `test_platform_config` | functional | — | `test_platform_config` |
| `test_paged_attention` | perf | 0 | `test_paged_attention_0` |
| `test_batch_paged_attention` | perf | 0 1 2 | `test_batch_paged_attention_{n}` |

功能测试（functional）每个测试对应一个二进制，没有参数索引。
性能测试（perf）每组参数对应一个独立二进制，在编译时通过 `PERF_CASE_IDX` 宏选定。

---

## 使用示例

```bash
# 运行全部功能测试
./run_tests.sh --func

# 运行全部性能测试
./run_tests.sh --perf

# 运行 test_batch_paged_attention 的所有参数组（0~2）
./run_tests.sh --test test_batch_paged_attention

# 只运行 test_batch_paged_attention 的第 1 组参数
./run_tests.sh --test test_batch_paged_attention --index 1

# 只运行 test_cpu_affinity（功能测试，无参数索引）
./run_tests.sh --test test_cpu_affinity

# 仅构建，不运行
./run_tests.sh --build-only

# 关闭 profiling 后运行全部测试
./run_tests.sh --no-profiling
```

---

## 环境变量

所有环境变量均为可选，不设置时使用括号内的默认值。

### 运行控制

| 变量 | 默认值 | 说明 |
|------|--------|------|
| `TIMEOUT` | `600` | 单个测试的超时时间（秒） |
| `BUILD_DIR` | `<脚本目录>/build` | CMake 构建目录 |
| `PTO2_PROFILING` | `ON` | 是否开启 profiling 宏（`ON` / `OFF`） |

### CPU 绑核

| 变量 | 默认值 | 说明 |
|------|--------|------|
| `ORCH_CPU` | `0` | Orchestrator 线程绑定的 CPU 核心 |
| `SCHED_CPU0` ~ `SCHED_CPU7` | `1` ~ `8` | 8 个 Scheduler 线程依次绑定的 CPU 核心 |

### 平台参数

| 变量 | 默认值 | 说明 |
|------|--------|------|
| `PLATFORM_MAX_BLOCKDIM` | `24` | 最大 block 维度 |
| `PLATFORM_AIC_CORES_PER_BLOCKDIM` | `1` | 每个 blockdim 的 AIC 核心数 |
| `PLATFORM_AIV_CORES_PER_BLOCKDIM` | `2` | 每个 blockdim 的 AIV 核心数 |
| `PLATFORM_MAX_AICPU_THREADS` | `4` | AICPU 最大线程数 |

**示例：**
```bash
TIMEOUT=120 ORCH_CPU=4 SCHED_CPU0=5 PLATFORM_MAX_BLOCKDIM=32 ./run_tests.sh --perf
```

---

## 执行流程

```
1. CMake 配置   cmake -S ... -B build  (传入所有平台参数和 CPU 绑核参数)
        ↓
2. 并行构建     cmake --build build --parallel <nproc>
        ↓
3. 运行测试     按参数范围逐个执行二进制，记录 PASS / FAIL
        ↓
4. 汇总         输出 Passed / Failed 计数及失败列表，非零退出码表示有失败
```

任意步骤失败（`set -e`）会立即终止脚本，除非失败来自测试本身（测试失败会被捕获并计入 FAIL_COUNT，不中断后续测试）。

---

## 新增测试

在脚本的测试注册表（`Test Registry` 区块）中添加两行：

```bash
# 功能测试（单二进制）
TEST_TYPE["test_my_func"]="func"   ; TEST_INDICES["test_my_func"]="-"

# 性能测试（多参数，索引为 0 1 2）
TEST_TYPE["test_my_perf"]="perf"   ; TEST_INDICES["test_my_perf"]="0 1 2"
```

同时将测试名追加到 `ALL_TESTS` 数组（控制 `--list` 和全量运行的顺序）：

```bash
ALL_TESTS=(...existing... test_my_func test_my_perf)
```

CMakeLists.txt 中相应添加构建目标即可，脚本无需其他改动。
