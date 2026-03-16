# AICPU Simulation Unit Test Framework

无需昇腾硬件，在 Host Linux CPU 上运行 PTO2 编排器与调度器的单元测试与性能分析套件。

---

## 快速上手

**前提条件：** `g++`（C++17）、`cmake >= 3.15`、`make`

```bash
cd tests/aicpu_ut

# 构建 + 运行全部默认测试（idx 0）
./run_tests.sh

# 构建 + 运行全部参数组
./run_tests.sh --all

# 仅构建
./run_tests.sh --build-only

# 查看所有可用测试
./run_tests.sh --list

# 对调度器做 perf 采样（需先构建）
./perf_sched.sh --bin test_batch_paged_attention_sched_prof_only_0
```

详细参数见 [`tests/aicpu_ut/tests.md`](tests/aicpu_ut/tests.md)。

---

## 目录结构

```
tests/aicpu_ut/
├── CMakeLists.txt                   # 构建系统（静态链接，每 case 独立二进制）
├── run_tests.sh                     # 构建 + 批量运行入口
├── perf_sched.sh                    # 单二进制 perf record 采样脚本
├── tests.md                         # 使用文档
├── common/
│   ├── test_common.h/cpp            # make_runtime()、sim_run_*()、profiling 打印
│   ├── test_log_stubs.cpp           # 日志桩（DEV_DEBUG / DEV_INFO 等 → stdout/stderr）
│   ├── cpu_affinity.h/cpp           # bind_to_cpu() / current_cpu()
│   └── json_cases.h                 # PerfTestCase 结构体（编译期用例选择）
└── tests/
    ├── functional/
    │   ├── test_cpu_affinity.cpp    # 验证 CPU 绑核正确性
    │   └── test_platform_config.cpp # 验证平台常量及 #ifndef 覆盖机制
    └── perf/
        ├── test_paged_attention.cpp                     # 单请求 PA（orch + sched 并发）
        ├── test_batch_paged_attention.cpp               # 批量 PA（orch + sched 并发）
        ├── test_batch_paged_attention_orch_only.cpp     # 仅编排，不启动调度线程
        └── test_batch_paged_attention_sched_prof_only.cpp  # 编排完成后再启动调度线程
```

---

## 仿真机制

### 零核心仿真（PTO2_SIM_AICORE_UT）

CMake 选项 `PTO2_SIM_AICORE_UT=ON`（默认开启）启用零核心仿真路径：

- `aicpu_executor.cpp` 中当 `cores_total_num_ == 0` 时跳过硬件寄存器轮询，改为调用 `sim_aicore` 接口
- `sim_aicore.cpp` 维护每个仿真核心的 COND 寄存器状态数组 `s_sim_core_cond_value[]`
- 调度器通过 `write_reg()` 向仿真核心"派发"任务时，`pto2_sim_aicore_on_task_received()` 立即将该核心 COND 设为 `MAKE_FIN_VALUE(task_id)`，模拟 AICore 即时完成
- 调度器在下一轮 `read_reg()` 时读取仿真 COND，触发 `on_task_complete()`，驱动依赖链向前推进

整个过程不执行任何 kernel，仅验证任务图依赖解析与状态流转：`PENDING → READY → RUNNING → COMPLETED → CONSUMED`。

### 调度线程启动（aicpu_sim_run_pto2）

`aicpu_sim_run_pto2(rt, num_sched_threads)` 是仿真运行入口（`sim_aicore.h`）：
1. 初始化所有仿真核心为 IDLE
2. 为每个线程绑定到 `SCHED_CPU{n}` 核心（由 CMake 宏注入）
3. 每个线程循环调用 `AicpuExecutor::resolve_and_dispatch_pto2()`
4. 达到 `MAX_IDLE_ITERATIONS`（100 万次空转）后线程退出
5. 汇总各线程 profiling 数据到 `AicpuSimRunProf`

---

## 测试变体

### test_batch_paged_attention（并发）

编排线程与调度线程**同时运行**：orchestrator 在一个线程上持续 `pto2_submit_task()`，多个 scheduler 线程并发消费就绪队列。最接近真实设备行为，但 init/memset 开销会混入 perf 采样窗口。

### test_batch_paged_attention_orch_only（仅编排）

只调用 `build_batch_paged_attention_graph()`，**不启动**调度线程。用于单独测量编排阶段（建图、依赖注册、Scope、Tensor 分配）的开销。启用 `PERF_WAIT_AFTER_INIT` 后，perf 窗口精确覆盖编排过程。

### test_batch_paged_attention_sched_prof_only（分阶段）

**先**完整执行编排（单线程，等待 `pto2_orchestrator_done()` 返回），**再**启动调度线程。SIGSTOP 断点在编排结束后、调度启动前插入，perf 窗口仅覆盖调度阶段，消除编排 init 噪音。任务生命周期完整（CONSUMED 状态正常触发）。

---

## SIGSTOP / perf 采样机制

所有 `test_batch_paged_attention*` 二进制支持 `PERF_WAIT_AFTER_INIT=1` 环境变量：

1. init（`make_runtime()` + 编排，视二进制类型而定）完成后调用 `raise(SIGSTOP)`
2. `perf_sched.sh` 轮询 `/proc/<pid>/stat` 等待进程状态变为 `T`
3. 附加 `perf record --call-graph dwarf -p <pid>`
4. 等待 `PERF_ATTACH_WAIT`（默认 0.3 s）后发送 `SIGCONT`
5. 进程退出后脚本向 `perf` 发送 `SIGINT` 结束采样

采样窗口仅覆盖工作阶段，排除 SM memset（~49 MB）、dep pool calloc、ready queue 原子初始化等 init 开销。

```bash
# 典型用法（不构建，直接采样已有二进制）
./perf_sched.sh --bin test_batch_paged_attention_sched_prof_only_0

# 先构建再采样，指定 5 个调度线程
./perf_sched.sh --bin test_batch_paged_attention_sched_prof_only_0 --build --sched-threads 5

# 采样完成后直接打开交互式报告
./perf_sched.sh --bin test_batch_paged_attention_sched_prof_only_0 --report
```

---

## Profiling 体系

三个独立开关（均可通过 `run_tests.sh` 参数控制）：

| 宏 | CMake 选项 | 说明 |
|----|-----------|------|
| `PTO2_PROFILING` | `PTO2_PROFILING` | 总开关；两个子开关均关闭时自动置 OFF |
| `PTO2_ORCH_PROFILING` | `PTO2_ORCH_PROFILING` | 编排器 9 子阶段周期分解（sync/alloc/params/lookup/heap/insert/fanin/finalize/scope） |
| `PTO2_SCHED_PROFILING` | `PTO2_SCHED_PROFILING` | 调度器线程级周期分解（见下） |

### PTO2SchedProfilingData（per-thread，`PTO2_SCHED_PROFILING`）

`on_task_complete()` 内各子阶段的 CPU 周期数：

| 字段 | 说明 |
|------|------|
| `lock_cycle` | fanout_lock 获取 + 状态写入 + 释放 |
| `fanout_cycle` | fanout 链表遍历（通知消费者） |
| `fanin_cycle` | fanin 引用计数递减 |
| `self_consumed_cycle` | 自身 check_and_handle_consumed |
| `lock_wait_cycle` | fanout_lock 自旋等待 |
| `push_wait_cycle` | ready queue push CAS 竞争 |
| `pop_wait_cycle` | ready queue pop CAS 竞争 |
| `complete_count` | 本线程完成的任务数 |

通过 `pto2_scheduler_get_profiling(thread_idx)` 读取并重置。

### PTO2SimSchedSummary（全局汇总，`PTO2_PROFILING`）

| 字段 | 说明 |
|------|------|
| `tasks_dispatched[4]` | 按 worker 类型（CUBE/VECTOR/AI_CPU/ACCELERATOR）分类的派发数 |
| `fanout_edges_total` / `fanout_max_degree` | 完成时遍历的 fanout 边总数及最大出度 |
| `tasks_enqueued_by_completion` | 因 `on_task_complete` 变为 READY 的任务数 |
| `fanin_edges_total` / `fanin_max_degree` | fanin 边总数及最大入度 |
| `rounds_total` / `rounds_with_progress` | 调度循环总轮次 / 有进展轮次 |
| `dispatch_cycle` / `complete_cycle` | 派发与完成阶段累计周期数 |

由 `pto2_print_sim_sched_summary()` 以表格形式输出。

---

## 平台与运行时修改

### platform_config.h — `#ifndef` 覆盖保护

所有平台常量用 `#ifndef / #endif` 包裹，允许 CMake `-DXXX=N` 注入自定义值：

| 常量 | 默认值 | 说明 |
|------|--------|------|
| `PLATFORM_MAX_BLOCKDIM` | 24 | 最大 block 维度 |
| `PLATFORM_AIC_CORES_PER_BLOCKDIM` | 1 | 每 blockdim AIC 核心数 |
| `PLATFORM_AIV_CORES_PER_BLOCKDIM` | 2 | 每 blockdim AIV 核心数 |
| `PLATFORM_CORES_PER_BLOCKDIM` | AIC + AIV（派生） | 防止分量覆盖时总数不一致 |
| `PLATFORM_MAX_AICPU_THREADS` | 4 | AICPU 调度线程上限 |

### platform_regs.h / platform_regs.cpp

新增仿真模式寄存器路径：`reg_base_addr == 0` 时 `write_reg()` 触发 `pto2_sim_aicore_on_task_received()`，`read_reg()` 调用 `pto2_sim_read_cond_reg()`，不操作真实硬件寄存器。

### performance_collector_aicpu

新增 `perf_aicpu_switch_buffer(runtime, core_id, thread_idx)` 及 per-core dispatch 时间戳数组，用于调度器派发时间线记录（`PTO2_PROFILING=1` 时启用）。`test_log_stubs.cpp` 提供 `perf_aicpu_record_orch_phase()` 的空桩，满足仿真环境的链接需求。

### pto_scheduler.cpp / pto_scheduler.h

- 新增 `PTO2SchedProfilingData` 结构体及 `pto2_scheduler_get_profiling()` / `pto2_print_sched_profiling()`（`PTO2_SCHED_PROFILING`）
- 新增 `PTO2SimSchedSummary` 及 `pto2_print_sim_sched_summary()`（`PTO2_PROFILING`）

### runtime.h / runtime.cpp

- 新增 `pto2_runtime_create_custom(mode, task_window_size, gm_heap_size)`（测试用，参数可控）
- 新增 `get_sim_aicore_mode()` 访问器

---

## 构建系统（CMakeLists.txt）

**静态链接：** 所有二进制以 `-nodefaultlibs -Wl,--start-group ... -Wl,--end-group` 完全静态链接 `libstdc++.a`、`libm.a`、`libc.a`、`libpthread.a`、`libdl.a`、`libgcc.a`，无动态库依赖。

**libstdc++.a 查找优先级：**
1. CANN toolkit 交叉编译路径（`~/Ascend/cann-*/tools/hcc/aarch64-target-linux-gnu/lib64/`）
2. `g++ -print-file-name=libstdc++.a`
3. 文件系统 glob（`/usr/lib[64]/gcc/*/*`）

**编译选项：** `-O0 -ggdb3 -fno-omit-frame-pointer`（保留帧指针、完整 DWARF，支持 `perf annotate` 和 `--call-graph dwarf`）

**共享 OBJECT 库 `runtime_common`：** Runtime 源文件和 common 测试源文件各编译一次，被所有测试二进制共用，避免重复编译。

**每 case 独立二进制：** `PERF_CASE_IDX` 宏在编译期选定用例，生成 `test_batch_paged_attention_{0,1,2}` 等独立可执行文件，perf 归因精确、无运行时分支干扰。

---

## 可用测试

| 测试名 | 类型 | 索引 | 说明 |
|--------|------|------|------|
| `test_cpu_affinity` | functional | — | 验证 `bind_to_cpu()` / `current_cpu()` |
| `test_platform_config` | functional | — | 验证平台常量及 `#ifndef` 覆盖 |
| `test_paged_attention` | perf | 0 | 单请求 paged attention |
| `test_batch_paged_attention` | perf | 0 1 2 | 批量 PA，orch + sched 并发 |
| `test_batch_paged_attention_orch_only` | perf | 0 1 2 | 仅编排，无调度线程 |
| `test_batch_paged_attention_sched_prof_only` | perf | 0 1 2 | 编排完成后再启调度线程 |

**批量 PA 用例（索引 0/1/2）：**

| 索引 | batch | num_heads | head_dim | block_size | context_lens |
|------|-------|-----------|----------|------------|--------------|
| 0 | 64 | 16 | 128 | 128 | 8193 |
| 1 | 2 | 16 | 128 | 128 | [8192, 4096] |
| 2 | 4 | 16 | 128 | 128 | [8192, 4096, 1024, 256] |

---

## 工具

### tools/sched_overhead_analysis.py

解析调度器开销日志，输出两部分：
- **Part 1**：从 `perf_swimlane_*.json` 读取每任务派发延迟（Kernel Exec / Head OH / Tail OH）
- **Part 2**：从 `outputs/aicpu_ut_phase_breakdown.log`（优先）或 device log 读取 Scheduler 循环阶段分解（get_ready / resolve_deps / dispatch_setup / idle）

```bash
# sim 运行后分析
python3 tools/sched_overhead_analysis.py --sim-log outputs/aicpu_ut_phase_breakdown.log
```

### tools/print_task_stats.py

解析 `outputs/aicpu_ut_sim_run.log`，按 worker 类型汇总派发任务数统计。
