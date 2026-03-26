# run_tests.sh --profiling 1 时「计时周期内」的写文件操作

本文说明在 `bash run_tests.sh --profiling 1` 下，**计时周期内**（即测试二进制从启动到退出的执行过程）会发生哪些**写文件**操作。

---

## 1. 口径说明

- **计时周期**：指每个测试用例的 `run_binary` 中 `timeout "$TIMEOUT" "$bin" "$@"` 的执行时间，即测试二进制进程从启动到退出的整段过程。
- **--profiling 1**：`PTO2_PROFILING=ON`，`PTO2_SCHED_PROFILING` 与 `PTO2_ORCH_PROFILING` 为 OFF；脚本若走 perf 分支会设置 `SIM_LOG`、`AICPU_UT_PHASE_LOG`、`AICPU_UT_SWIMLANE_DIR` 等（见下）。

---

## 2. 计时周期内的写文件操作

### 2.1 测试二进制（C++）内部

| 位置 | 条件 | 操作 | 说明 |
|------|------|------|------|
| **test_log_stubs.cpp**（`dev_log_always`） | `PTO2_PROFILING` 已开启 **且** 环境变量 **`AICPU_UT_PHASE_LOG`** 已设置 | 每次调用时：`fopen(phase_log, "a")` → `fprintf(f, "%s\n", buf)` → `fclose(f)` | 将 DEV_ALWAYS 的每行（线程释放、PTO2 进度、Scheduler Phase Breakdown 等）**追加**到指定文件。默认由 run_tests.sh 设为 `$PROJECT_ROOT/outputs/aicpu_ut_phase_breakdown.log`（仅在 `$RUN_PERF && [ -z "${AICPU_UT_QUIET:-}" ]` 时设置）。 |
| **sim_swimlane.h**（`export_sim_swimlane`） | 仅 **test_orchestrator_scheduler** 用例在**跑完后**调用 | `fopen(filepath, "w")` → 多次 `fprintf(fp, ...)` → `fclose(fp)` | 在 `AICPU_UT_SWIMLANE_DIR`（默认 `outputs`）下写入 **perf_swimlane_YYYYMMDD_HHMMSS.json**，发生在调度已结束、进程尚未退出时，仍属同一「计时周期」内。 |

说明：

- **AICPU_UT_PHASE_LOG**：只有在脚本设置了该环境变量时才会写文件；未设置时 `dev_log_always` 只做 `vprintf`/`putchar` 到 stdout，不写盘。
- **perf_swimlane_*.json**：只有运行到 `test_orchestrator_scheduler` 且调用 `export_sim_swimlane(rt)` 时才会写；其他用例不会写该文件。

### 2.2 脚本 / 内核在计时周期内的写文件

| 位置 | 条件 | 操作 | 说明 |
|------|------|------|------|
| **run_tests.sh**（`run_binary`） | 环境变量 **`SIM_LOG`** 已设置 | `timeout "$TIMEOUT" "$bin" "$@" >"$tmp_log" 2>&1` | 测试进程的 **stdout + stderr** 被重定向到 `mktemp` 生成的临时文件；在进程运行期间由**内核**持续写入该临时文件。`SIM_LOG` 在 `$RUN_PERF && [ -z "${AICPU_UT_QUIET:-}" ]` 时被设为 `$PROJECT_ROOT/outputs/aicpu_ut_sim_run.log`，且会先清空该 log。 |

注意：`cat "$tmp_log" >> "$SIM_LOG"`、`cp "$tmp_log" "$log_file"` 等是在**测试进程退出之后**执行，不属于「计时周期内」的写操作。

---

## 3. 不在此周期内的写操作（对照）

- **format_profiling_output.py**：仅从 stdin 读、向 stdout 写，不打开或写入磁盘文件。
- **perf_aicpu_record_phase / perf_aicpu_write_orch_summary / perf_aicpu_write_core_assignments**：写入的是**共享内存**（phase buffer、phase_header），不是普通文件。
- **device_runner / aicpu_executor 中写 .so 的路径**：在 aicpu_ut 构建中由 `PTO2_SIM_AICORE_UT` 排除，仿真路径不会执行该写 .so 逻辑。
- **performance_collector.cpp**（host）中写 **perf_swimlane_*.json**：用于设备侧采集场景，由 Host 写；aicpu_ut 仿真用的是 **sim_swimlane.h** 的 `export_sim_swimlane`，见上表。

---

## 4. 小结（--profiling 1 且默认 RUN_PERF 为真时）

在计时周期内会发生的写文件操作有：

1. **outputs/aicpu_ut_phase_breakdown.log**  
   多次**追加**写入（每次 `dev_log_always` 一行），来自 `test_log_stubs.cpp`。

2. **临时文件 tmp_log**（如 `/tmp/tmp.XXXXXX`）  
   测试进程的 stdout+stderr 被重定向到该文件，由内核在进程运行期间写入；脚本在进程结束后再根据 `SIM_LOG`/`LOG_DIR` 做追加或 cp。

3. **outputs/perf_swimlane_YYYYMMDD_HHMMSS.json**（仅 test_orchestrator_scheduler）  
   用例结束时由 `export_sim_swimlane()` **一次**写入。

若未设置 `AICPU_UT_PHASE_LOG` 或未设置 `SIM_LOG`（例如未走 perf 分支），则上述 1 和 2 中对应的文件写入不会发生。
