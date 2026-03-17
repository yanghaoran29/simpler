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
| `--sched-threads N` | 设置 Scheduler 线程数（仅 perf 测试生效，默认 3，范围 1～PLATFORM_MAX_AICPU_THREADS） |
| `--no-early-return` | （已废弃）旧版通过编译时定义 `PTO2_SIM_NO_EARLY_RETURN` 在 break/return 前先 drain `deferred_release`；当前分支已移除相关分支代码，该选项不再生效，仅保留参数占位以兼容旧脚本 |

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
| `test_batch_paged_attention_orch_only` | perf | 0 1 2 | `test_batch_paged_attention_orch_only_{n}` |
| `test_batch_paged_attention_sched_prof_only` | perf | 0 1 2 | `test_batch_paged_attention_sched_prof_only_{n}` |

功能测试（functional）每个测试对应一个二进制，没有参数索引。
性能测试（perf）每组参数对应一个独立二进制，在编译时通过 `PERF_CASE_IDX` 宏选定。

**test_batch_paged_attention 与 --no-early-return（已废弃）：** 早期版本在启用 `--no-early-return` 时会定义宏 `PTO2_SIM_NO_EARLY_RETURN` 并在 Scheduler Profiling 结束后额外校验 `tasks_completed == tasks_consumed` 及 `fanout == fanin`。当前分支已移除该宏及相关分支逻辑，因此该选项不再影响执行结果，仅用于兼容旧命令行参数，不再进行额外校验。

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

# 开启 no-early-return，校验 completed==consumed、fanout==fanin（test_batch_paged_attention 会据此判 fail）
./run_tests.sh --test test_batch_paged_attention --idx 0 --no-early-return

# 4 个 Scheduler 线程运行指定用例
./run_tests.sh --test test_batch_paged_attention --idx 0 --sched-threads 4
```

---

## 环境变量与脚本参数

所有环境变量均为可选，不设置时使用括号内的默认值。Scheduler 线程数、是否跳过 early return 等通过脚本参数 `--sched-threads N`、`--no-early-return` 指定（见上文「构建控制」）。

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
| `PLATFORM_MAX_AICPU_THREADS` | `4` | AICPU 最大线程数（编译期） |

**示例：**
```bash
TIMEOUT=120 ORCH_CPU=4 SCHED_CPU0=5 PLATFORM_MAX_BLOCKDIM=32 ./run_tests.sh --perf
./run_tests.sh --test test_batch_paged_attention --idx 0 --sched-threads 4
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

---

# perf_sched.sh

`perf_sched.sh` 是基于 `perf record` 的单二进制性能采样脚本，位于 `tests/aicpu_ut/`。它在测试进程完成初始化后才挂载 `perf record`，使采样窗口精确覆盖工作阶段，排除 init 噪音。

---

## 原理

`test_batch_paged_attention*` 系列二进制均支持 `PERF_WAIT_AFTER_INIT=1` 环境变量：init 完成后进程调用 `raise(SIGSTOP)` 自我暂停，脚本检测到进程状态变为 `T` 后挂载 `perf record -p`，再发送 `SIGCONT` 恢复执行；进程退出后脚本向 `perf` 发送 `SIGINT` 结束采样。

其他二进制直接通过 `perf record -- <bin>` 全程跟踪。

---

## 快速上手

```bash
# 采样 sched_prof_only（最常用，调度器专项）
./perf_sched.sh --bin test_batch_paged_attention_sched_prof_only_0

# 先构建再采样
./perf_sched.sh --bin test_batch_paged_attention_sched_prof_only_0 --build

# 采样后直接打开交互式报告
./perf_sched.sh --bin test_batch_paged_attention_sched_prof_only_0 --report

# 查看帮助
./perf_sched.sh --help
```

---

## 命令行参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--bin <name>` | **必填** | 目标二进制名（不含路径，在 `build/bin/` 下查找） |
| `--build` | 不构建 | 采样前先调用 `run_tests.sh --build-only` 完成构建 |
| `--sched-threads N` | `3` | 传给测试二进制的 Scheduler 线程数（环境变量 `AICPU_UT_NUM_SCHED_THREADS`） |
| `--output <path>` | `build/perf.<name>.data` | 采样输出文件路径 |
| `--report` | 不打开 | 采样完成后立即执行 `perf report -i <output>` |
| `--profiling` | 关闭 | 构建时开启 profiling 插桩（`PTO2_PROFILING`）；默认关闭，避免插桩影响 perf 结果 |
| `--call-graph <mode>` | `dwarf` | 调用栈展开方式：`dwarf`（默认）、`fp`（需各库保留帧指针）、`lbr`（仅 x86） |
| `--dwarf-size <bytes>` | `65528` | dwarf 栈快照大小（字节），perf 最大值为 65528 |
| `--help` / `-h` | — | 显示帮助信息，然后退出 |

---

## 使用示例

```bash
# 调度器专项采样（orch 已完成，perf 窗口只覆盖 scheduler）
./perf_sched.sh --bin test_batch_paged_attention_sched_prof_only_0

# 编排专项采样
./perf_sched.sh --bin test_batch_paged_attention_orch_only_0

# 全流程采样（orch + sched 并发）
./perf_sched.sh --bin test_batch_paged_attention_0

# 5 个 Scheduler 线程，先构建再采样
./perf_sched.sh --bin test_batch_paged_attention_sched_prof_only_0 --build --sched-threads 5

# 指定输出路径并立即打开报告
./perf_sched.sh --bin test_batch_paged_attention_sched_prof_only_0 --output /tmp/sched.data --report

# 使用帧指针展开（各库须保留帧指针编译，栈解析更快）
./perf_sched.sh --bin test_batch_paged_attention_sched_prof_only_0 --call-graph fp
```

---

## 与 run_tests.sh 的关系

| | `run_tests.sh` | `perf_sched.sh` |
|---|---|---|
| 用途 | 构建 + 批量运行所有测试，输出 PASS/FAIL | 单二进制 perf 采样，输出 `.data` 文件 |
| 构建 | 每次自动构建 | 默认跳过，`--build` 触发 |
| 测试范围 | 全部注册测试 | 单个指定二进制 |
| 输出 | 控制台日志 + `sim_run.log` | `perf.data` 文件（可接 `perf report`） |
