# hardware_test 文档（AICPU 仿真/硬件相关测试）

本文档用于说明 `tests/aicpu_ut` 这套“无需昇腾硬件、在 Host CPU 上仿真 PTO2 编排器/调度器”的测试体系（也可理解为项目内的 `hardware_test`：覆盖寄存器协议、调度循环、profiling 汇总等硬件相关路径）。

## 适用范围

- **适用**：本地 Linux（aarch64/x86_64 均可），验证 PTO2 任务图依赖流转与调度器行为；对调度器做 `perf` 采样。
- **不适用**：真实 kernel 执行正确性（仿真不执行 kernel，只模拟“派发→完成”）。

## 快速运行

在仓库根目录下：

```bash
cd tests/aicpu_ut

# 构建 + 运行默认组（idx 0）
./run_tests.sh

# 构建 + 运行全部参数组
./run_tests.sh --all

# 仅构建
./run_tests.sh --build-only

# 列出可运行的测试二进制
./run_tests.sh --list
```

更完整的参数说明见 `tests/aicpu_ut/tests.md`。

## 核心机制概览

### 1) 零核心仿真（`PTO2_SIM_AICORE_UT` 下 AICore 的处理方式）

默认 CMake 打开 `PTO2_SIM_AICORE_UT=ON`，在 **运行 `tests/aicpu_ut` 下的所有样例时，AICore 统一按“零核心仿真”处理**：

- **不依赖真实昇腾 AICore 硬件**
  - 不向物理设备写寄存器，也不会触发真实 kernel 执行；
  - 所有逻辑全部在 Host Linux CPU 上的普通线程中完成。

- **AICore 被抽象为一组内存中的 COND 寄存器状态**
  - `sim_aicore.cpp` 维护全局数组 `s_sim_core_cond_value[]`，每个元素对应一个“仿真 core”的 COND 寄存器；
  - 平台寄存器封装 `write_reg/read_reg` 在发现 `reg_base_addr == 0` 时，不再访问真实寄存器，而是：
    - `write_reg` → 调用 `pto2_sim_aicore_on_task_received(core_id, task_id)` 等仿真接口，更新 `s_sim_core_cond_value`；
    - `read_reg` → 调用 `pto2_sim_read_cond_reg(core_id)` 读取对应 core 的 COND 值。

- **AICore 执行被简化为“立即完成”的模型**
  - 调度器将任务派发到某个 core 时，仿真层会在 `pto2_sim_aicore_on_task_received` 中**直接把该 core 的 COND 设为 `FIN(task_id)`**；
  - 下一轮调度循环中，executor 通过 `read_reg` 读到 `FIN(task_id)`，匹配到当前正在运行的 `reg_task_id` 后：
    - 调用 `on_subtask_complete` / `on_mixed_task_complete` 更新依赖、入 READY 队列；
    - 驱动任务状态沿着 `PENDING → READY → RUNNING → COMPLETED → CONSUMED` 流转。

- **编排 / 调度全部在 Host 线程内完成**
  - 调度线程由 `aicpu_sim_run_pto2` / `aicpu_sim_run_pto2_concurrent` 使用 `std::thread` 启动，循环调用 `AicpuExecutor::resolve_and_dispatch_pto2`，只与 shared memory + 仿真寄存器交互；
  - 编排（`build_graph`、`pto2_submit_mixed_task`、`pto2_orchestrator_done` 等）也由 Host 线程执行（部分样例单独使用 orch 线程，部分直接在主线程上运行），将任务描述与依赖写入 PTO2 shared memory。

综上，`tests/aicpu_ut` 下的样例实际上验证的是：

- **真实的 PTO2 任务图 / 依赖 / 调度逻辑**；
- **仿真的 AICore 行为（立即完成，不执行实际 kernel）**。

适用场景是：验证任务状态流转 + 压测调度器性能 + `perf` 分析；不适用于验证具体算子的数值正确性或真实设备上的 kernel 性能。

### 2) 任务计数：Multi-ring（`PTO2_MAX_RING_DEPTH`）

PTO2 shared memory 采用 multi-ring 结构，任务发布计数位于：

- `PTO2SharedMemoryHeader::rings[r].fc.current_task_index`

**总任务数 = 所有 ring 的 `current_task_index` 之和**。在仿真入口或 executor 初始化阶段，若仍读取旧的 `header->current_task_index`（单 ring 时代字段），会导致编译失败或逻辑错误。

### 3) `PTO2TaskId` 与 “reg_task_id”

`PTO2TaskDescriptor::mixed_task_id` 的类型为 `PTO2TaskId`（64-bit raw 编码：`(ring_id << 32) | local_id`），它不是普通整数：

- 需要用 `pto2_task_id_local(task_id)` 提取 local id（32-bit）用于日志/统计或与旧协议兼容
- “寄存器协议”中的 `reg_task_id` 是 **per-core 单调计数**（或在仿真模式下用 local id 映射），与 `mixed_task_id` 的 raw 值不同

---

## 线程创建方式（tests/aicpu_ut 详解）

`tests/aicpu_ut` 里有两套线程创建与使用方式：**仿真调度入口**使用 C++ `std::thread`，**CPU 绑核功能测试**使用 `pthread`。下面分别说明。

### 1. 仿真调度入口中的线程（`std::thread`）

实现位于 `common/sim_run_pto2.cpp`，在宏 `PTO2_SIM_AICORE_UT` 下编译。

#### 1.1 仅调度线程：`aicpu_sim_run_pto2`

- **用途**：编排已在 Host 上完成（`build_graph` + `pto2_orchestrator_done` 已执行），只启动调度线程消费共享内存中的任务。
- **线程数量**：`num_sched_threads`（由用例或 `get_num_sched_threads()` 决定，通常 3～8）。
- **创建方式**：
  - 使用 `std::vector<std::thread> threads`，循环 `for (i = 0; i < num_sched_threads; i++)` 中 `threads.emplace_back([&runtime, i]() { ... })`，即**按索引 `i` 捕获的 lambda** 作为线程入口。
- **每个调度线程在入口依次执行**：
  1. **绑核**（可选）：若 `i < sizeof(s_sched_cpus)/sizeof(s_sched_cpus[0])`，则调用 `bind_to_cpu(s_sched_cpus[i])`。`s_sched_cpus[]` 为编译期常量数组 `{ SCHED_CPU0, SCHED_CPU1, ... }`，由 CMake 通过 `-DSCHED_CPU0=…` 等注入。
  2. **记录实际运行核**：若 `0 <= i < PLATFORM_MAX_AICPU_THREADS`，则 `s_actual_sched_cpu[i] = current_cpu()`（-1 表示获取失败），供后续 `aicpu_sim_get_actual_sched_cpu(i)` 查询。
  3. **调度循环**：调用 `aicpu_executor_sim_run_resolve_and_dispatch_pto2(&runtime, i)`，内部循环执行“解析依赖 → 派发任务到仿真 core → 轮询完成”，直到达到退出条件（如空转轮数上限）。
- **主线程**：在创建完所有线程后，对 `threads` 中每个线程执行 `t.join()`，再调用 `aicpu_executor_sim_shutdown_aicore(&runtime)`。
- **特点**：无单独“编排线程”；编排已在调用 `aicpu_sim_run_pto2` 之前由当前进程（主线程或调用方）完成。

#### 1.2 编排 + 调度并发：`aicpu_sim_run_pto2_concurrent`

- **用途**：模拟真实设备上编排与调度同时进行——一个**编排线程**在 Host 上执行 `build_graph` 等，多路**调度线程**并发消费任务；用于“并发”类 perf 测试（如 `test_*_concurrent`）。
- **线程构成**：
  - **调度侧**：`std::vector<std::thread> sched_threads`，数量与创建方式与 1.1 相同；每个线程同样先按 `s_sched_cpus[i]` 绑核并记录 `s_actual_sched_cpu[i]`，再执行 `aicpu_executor_sim_run_resolve_and_dispatch_pto2(&runtime, i)`。
  - **编排侧**：单独一个 `std::thread orch_thread`，入口为 lambda `[pto2_rt, sm_base, &orch_fn]()`，其中 `orch_fn` 由调用方传入（例如 perf 用例里执行 `build_graph(r, ctx.args, 10)` 并可能包一层 `bind_to_cpu(ORCH_CPU)` 和 `orch_timing_begin/end`）。
- **执行顺序**：
  1. 主线程先创建并启动**所有调度线程**（不 join），再创建并启动 **orch_thread**。
  2. **orch_thread** 内执行：`orch_fn(pto2_rt)`（完成建图并调用 `pto2_orchestrator_done()`），然后从 shared memory 读取各 ring 的 `current_task_index` 求和得到 `total`，再调用 `aicpu_executor_sim_setup_after_host_orch(total)`，使调度线程能正确看到“编排已完成、任务总数已知”。
  3. 主线程先 `orch_thread.join()`，再对 `sched_threads` 中每个线程 `join()`，最后 `aicpu_executor_sim_shutdown_aicore(&runtime)`。
- **注意**：初始化时设置 `runtime.set_orch_deferred_on_host(true)`，表示“编排在 Host 上延后执行”，因此 **不会**在 `aicpu_executor_sim_init` 里调用 `setup_after_host_orch`，而是等 orch_thread 跑完建图后再调用，避免调度线程在编排未完成时就开始消费。

#### 1.3 仅编排（主线程，无调度线程）：`test_orchestrator.cpp`

- **用途**：“仅编排”性能测试——只测量建图与任务提交阶段，**不启动任何调度线程**。对应各后端的 `*_orch_only` 二进制（如 `test_linear_orch_only_0`）。
- **实现方式**：**不创建任何 `std::thread`**，编排由 **主线程** 直接完成：
  1. 主线程调用 `bind_to_cpu(ORCH_CPU)` 绑到编排核；
  2. `setup_run(tc, ctx)` 创建 runtime（不调用 `aicpu_sim_run_pto2`，因此不会创建调度线程）；
  3. 主线程执行 `build_graph(rt, ctx.args, 10)`，完成建图与 `pto2_submit_mixed_task` 等，将任务写入 PTO2 shared memory；
  4. **不调用** `aicpu_sim_run_pto2` / `aicpu_sim_run_pto2_concurrent`，因此没有任何调度线程被创建；
  5. 建图结束后直接 `pto2_runtime_destroy(rt)`，进程退出。
- **与 1.1 / 1.2 的区别**：1.1 和 1.2 的入口在 `sim_run_pto2.cpp` 中创建 `std::thread`；而“仅编排”**没有**进入 `sim_run_pto2.cpp`，编排逻辑全部在主线程内执行，且**没有**单独的“编排线程”对象——主线程本身就是唯一的“编排线程”。
- **可选**：`perf_wait_sigstop()` 在 init 后挂起进程，便于 `perf record -p <pid>` 只采编排阶段；`orch_timing_begin/end` 包住 `build_graph` 用于打印编排耗时。

### 2. 绑核配置（SCHED_CPU / ORCH_CPU）

- **来源**：`SCHED_CPU0`～`SCHED_CPU7`、`ORCH_CPU` 均在 `tests/aicpu_ut/CMakeLists.txt` 中通过 `set(..., CACHE STRING ...)` 定义，并经由 `add_compile_definitions(-DSCHED_CPU0=...)` 等传入编译；运行时可通过 `run_tests.sh` 的环境变量覆盖（如 `ORCH_CPU=4 SCHED_CPU0=8 ./run_tests.sh`），脚本再传给 CMake 的 `-D` 参数。
- **使用位置**：
  - **调度线程**：在 `sim_run_pto2.cpp` 的 lambda 里用 `s_sched_cpus[i]`（即 `SCHED_CPU0`…）调用 `bind_to_cpu(s_sched_cpus[i])`。
  - **编排线程**：在 perf 用例的 `orch_fn` 中（如 `test_orchestrator_scheduler.cpp` 的 `build_graph` 前）调用 `bind_to_cpu(ORCH_CPU)`，与仿真入口解耦，由各用例自行决定是否绑核。
- **实现**：`bind_to_cpu(cpu_core)` 在 `common/cpu_affinity.cpp` 中通过 `pthread_setaffinity_np(pthread_self(), ..., &cpuset)` 将**当前线程**绑到指定核；`current_cpu()` 使用 `sched_getcpu()` 查询当前运行核。

### 3. 功能测试中的线程（`pthread` + `run_threads`）

用于 **CPU 绑核功能测试**（如 `functional/test_cpu_affinity.cpp`），不跑 PTO2 调度，只验证“绑核/不绑核”下线程实际运行核是否符合预期。

- **接口**：`common/cpu_affinity.h/cpp` 中的 `run_threads(void* (*fn)(void*), ThreadReport* reports, int n)`。
- **创建方式**：使用 **pthread**：`pthread_create(&tids[i], nullptr, fn, &reports[i])`，共 `n` 个线程（通常 `n = LAUNCH_AICPU_NUM == 4`，对应 3 个 scheduler + 1 个 orchestrator 的模型）；每个线程入口为 `fn`，传入 `&reports[i]`，且 `reports[i].thread_idx = i` 在创建前已写好。
- **线程入口**：
  - **不绑核**：`thread_fn_no_bind` 先 `unbind_from_cpu()`，再做短时忙等，最后把 `get_bound_cpu()`、`current_cpu()` 写入 `ThreadReport`。
  - **绑核**：`thread_fn_with_bind` 根据 `reports[i].target_cpu`（由测试里按 `SCHED_CPU0/1/2`、`ORCH_CPU` 填好）调用 `bind_to_cpu(r->target_cpu)`，忙等后同样写回 `bound_cpu`、`actual_cpu`、`bind_ok`。
- **主线程**：对每个 `tids[i]` 依次 `pthread_join(tids[i], nullptr)`，再根据 `ThreadReport` 打印或做 CHECK。
- **与仿真入口的区别**：这里不调用任何 `aicpu_sim_run_pto2*`，也不创建 `Runtime`；仅验证“线程创建 + 绑核/不绑核 + 当前核查询”这一小闭环，便于单独回归绑核逻辑。

### 4. 小结对照

| 场景 | 线程 API | 线程角色 | 绑核时机 | 入口代码位置 |
|------|----------|----------|----------|----------------|
| `aicpu_sim_run_pto2` | `std::thread` | 仅 N 个调度线程 | 每个线程入口内 `bind_to_cpu(s_sched_cpus[i])` | `sim_run_pto2.cpp` |
| `aicpu_sim_run_pto2_concurrent` | `std::thread` | N 个调度 + 1 个编排 | 调度同上；编排在 `orch_fn` 内（如 `bind_to_cpu(ORCH_CPU)`） | `sim_run_pto2.cpp` + 用例传入的 lambda |
| **仅编排**（`test_orchestrator.cpp`） | **无**（主线程） | 主线程只做编排，不启动调度线程 | 主线程内 `bind_to_cpu(ORCH_CPU)` | `perf/drivers/test_orchestrator.cpp` 的 `main()` |
| CPU 绑核功能测试 | `pthread` | 4 个“假”AICPU 线程（3 sched + 1 orch） | 在 `thread_fn_with_bind` 内按 `target_cpu` 绑核 | `cpu_affinity.cpp` 的 `run_threads` + `thread_fn_*` |

---
