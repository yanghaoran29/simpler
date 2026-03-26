/**
 * throughput.cpp
 *
 * 分层 DAG 吞吐 backend (PERF_BACKEND=7).
 * 第1层：n 个任务无依赖，可同时进入 ReadyQueue；
 * 第2层：依赖第1层，每层内任务有重叠（依赖数 D、重叠数 O 由参数决定）；
 * 第 N 层：依赖第 N-1 层。层间关系：上一层每个任务在本层有 D 个依赖者，相邻组重叠 O 个。
 *
 * 层大小：size[0]=layer0_size, size[k]=D+(size[k-1]-1)*(D-O), k>=1。
 * 层 k 任务 j 依赖层 k-1 的任务 i 满足 i*(D-O) <= j < i*(D-O)+D。
 *
 * 三个样例（worker 分配策略不同）：
 *   idx=0: 所有任务均为 AIV
 *   idx=1: 奇数编号任务（全局提交序，1-indexed）为 AIC，偶数编号任务为 AIV
 *   idx=2: 每层前半任务为 AIC，后半任务为 AIV
 *
 * Included by test_scheduler.cpp, test_orchestrator.cpp, test_orchestrator_scheduler.cpp.
 * Do not compile as separate TU — it is #included by the test driver.
 */

#include "pto_runtime2.h"
#include "test_common.h"
#if defined(PTO2_SIM_AICORE_UT)
#include "sim_aicore.h"
#endif
#include <unistd.h>

#include <algorithm>
#include <csignal>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include "common/platform_config.h"
#include "cpu_affinity.h"

#ifndef PERF_CASE_IDX
#define PERF_CASE_IDX 0
#endif

// ─── Compile-time parameters (overridable via CMake / run_tests.sh) ──────────

#ifndef AICPU_UT_THROUGHPUT_LAYERS
#define AICPU_UT_THROUGHPUT_LAYERS 10
#endif

#ifndef AICPU_UT_THROUGHPUT_LAYER0_SIZE
#define AICPU_UT_THROUGHPUT_LAYER0_SIZE 320
#endif

#ifndef AICPU_UT_THROUGHPUT_DEPS
#define AICPU_UT_THROUGHPUT_DEPS 6
#endif

#ifndef AICPU_UT_THROUGHPUT_OVERLAP
#define AICPU_UT_THROUGHPUT_OVERLAP 5
#endif

#ifndef AICPU_UT_THROUGHPUT_FIX_TAIL
#define AICPU_UT_THROUGHPUT_FIX_TAIL 1
#endif

// fix_tail: 仅当 deps_per_task - overlap == 1 时生效；开启时每层任务数均为 layer0_size，
// 不添加原公式多出的“溢出”任务（每层末尾依赖数可不足）。

// ─── Stringify helper for embedding macro values in string literals ───────────
#define THROUGHPUT_STRINGIFY_IMPL(x) #x
#define THROUGHPUT_STRINGIFY(x) THROUGHPUT_STRINGIFY_IMPL(x)

// ─── Worker assignment modes ──────────────────────────────────────────────────
// 0: all AIV
// 1: odd global submission index (1-indexed) → AIC, even → AIV
// 2: per-layer first half → AIC, second half → AIV
#define THROUGHPUT_WORKER_ALL_AIV 0
#define THROUGHPUT_WORKER_ODD_AIC 1
#define THROUGHPUT_WORKER_HALF_PER_LAYER 2

// ─── Types ────────────────────────────────────────────────────────────────────

struct ThroughputTestCase {
    const char* name;
    int num_layers;     // 层数 n
    int layer0_size;    // 第一层任务数
    int deps_per_task;  // 依赖数 D：上一层每个任务在本层有 D 个依赖者
    int overlap;        // 重叠数 O：相邻组共享 O 个依赖者
    int worker_mode;    // worker 分配策略（见 THROUGHPUT_WORKER_* 宏）
    int fix_tail;       // 仅当 D-O==1 时生效：每层任务数固定为 layer0_size，不添加溢出任务
};

static constexpr int THROUGHPUT_CASE_COUNT = 3;
static_assert(PERF_CASE_IDX >= 0 && PERF_CASE_IDX < THROUGHPUT_CASE_COUNT, "PERF_CASE_IDX out of range");

extern const ThroughputTestCase PERF_CASES[THROUGHPUT_CASE_COUNT];

extern float g_throughput_input_buf;

#define FUNC_ELEMENT_WISE 0

struct GraphCtx {
    int64_t config[6];  // num_layers, layer0_size, deps_per_task, overlap, worker_mode, fix_tail
    uint64_t args[10];
};

int get_num_sched_threads();
void perf_wait_sigstop();
void build_graph(PTO2Runtime* rt, uint64_t* args, int arg_count);
PTO2Runtime* setup_run(const ThroughputTestCase& tc, GraphCtx& ctx);

#if PTO2_PROFILING
void section_header_100(char pad_char, const char* title);
void print_config(const ThroughputTestCase& tc);
void print_cpu_affinity(int num_sched, int orch_cpu);
#endif

// ─── Definitions ─────────────────────────────────────────────────────────────

#define THROUGHPUT_COMMON_PARAMS \
    AICPU_UT_THROUGHPUT_LAYERS, AICPU_UT_THROUGHPUT_LAYER0_SIZE, AICPU_UT_THROUGHPUT_DEPS, \
        AICPU_UT_THROUGHPUT_OVERLAP

#define THROUGHPUT_PARAM_STR \
    "(n=" THROUGHPUT_STRINGIFY(AICPU_UT_THROUGHPUT_LAYERS)         \
    ", layer0=" THROUGHPUT_STRINGIFY(AICPU_UT_THROUGHPUT_LAYER0_SIZE) \
    ", D=" THROUGHPUT_STRINGIFY(AICPU_UT_THROUGHPUT_DEPS)          \
    ", O=" THROUGHPUT_STRINGIFY(AICPU_UT_THROUGHPUT_OVERLAP) ")"

#define THROUGHPUT_FIX_TAIL AICPU_UT_THROUGHPUT_FIX_TAIL

const ThroughputTestCase PERF_CASES[THROUGHPUT_CASE_COUNT] = {
    // idx=0: 所有任务均为 AIV
    {
        "ThroughputLayers all-AIV " THROUGHPUT_PARAM_STR,
        THROUGHPUT_COMMON_PARAMS,
        THROUGHPUT_WORKER_ALL_AIV,
        THROUGHPUT_FIX_TAIL,
    },
    // idx=1: 奇数编号任务（全局提交序，1-indexed）为 AIC，偶数编号任务为 AIV
    {
        "ThroughputLayers odd-AIC/even-AIV " THROUGHPUT_PARAM_STR,
        THROUGHPUT_COMMON_PARAMS,
        THROUGHPUT_WORKER_ODD_AIC,
        THROUGHPUT_FIX_TAIL,
    },
    // idx=2: 每层前半任务为 AIC，后半任务为 AIV
    {
        "ThroughputLayers half-AIC/half-AIV per layer " THROUGHPUT_PARAM_STR,
        THROUGHPUT_COMMON_PARAMS,
        THROUGHPUT_WORKER_HALF_PER_LAYER,
        THROUGHPUT_FIX_TAIL,
    },
};

float g_throughput_input_buf;

int get_num_sched_threads() {
    int n = 3;
    const char* env = std::getenv("AICPU_UT_NUM_SCHED_THREADS");
    if (env && *env) n = std::atoi(env);
    if (n < 1) n = 1;
    if (n > PLATFORM_MAX_AICPU_THREADS) n = PLATFORM_MAX_AICPU_THREADS;
    return n;
}

void perf_wait_sigstop() {
    if (std::getenv("PERF_WAIT_AFTER_INIT")) {
        pid_t pid = getpid();
        printf("  [perf] Init done. PID=%d — attach perf, then send SIGCONT:\n", (int)pid);
        printf("  [perf]   perf record -g -p %d -o perf.data\n", (int)pid);
        printf("  [perf]   kill -CONT %d\n", (int)pid);
        fflush(stdout);
        raise(SIGSTOP);
    }
}

// 计算第 k 层任务数：size[0]=W0, size[k]=D+(size[k-1]-1)*(D-O)
static int layer_size(int k, int layer0_size, int D, int O) {
    if (k == 0) return layer0_size;
    int prev = layer_size(k - 1, layer0_size, D, O);
    int step = D - O;
    if (step <= 0) return prev;
    return D + (prev - 1) * step;
}

// 层 k 任务 j 依赖层 k-1 的任务 i 满足 i*(D-O) <= j < i*(D-O)+D，收集这些 i 到 preds
static void get_predecessors(int j, int prev_size, int D, int O, std::vector<int>& preds) {
    preds.clear();
    int step = D - O;
    if (step <= 0) return;
    // i*(D-O) <= j < i*(D-O)+D  =>  (j-D)/(step) < i <= j/step
    int i_lo = (j - D + 1 + step - 1) / step;  // ceil((j-D+1)/step)
    if (i_lo < 0) i_lo = 0;
    int i_hi = j / step;
    if (i_hi >= prev_size) i_hi = prev_size - 1;
    for (int i = i_lo; i <= i_hi; i++) preds.push_back(i);
}

// 根据 worker_mode、全局提交序（1-indexed）和层内位置决定 worker 类型
static int task_worker_type(int worker_mode, int task_global_1indexed, int j, int cur_sz) {
    switch (worker_mode) {
        case THROUGHPUT_WORKER_ODD_AIC:
            return (task_global_1indexed % 2 == 1) ? PTO2_WORKER_CUBE : PTO2_WORKER_VECTOR;
        case THROUGHPUT_WORKER_HALF_PER_LAYER:
            return (j < cur_sz / 2) ? PTO2_WORKER_CUBE : PTO2_WORKER_VECTOR;
        default:  // THROUGHPUT_WORKER_ALL_AIV
            return PTO2_WORKER_VECTOR;
    }
}

void build_graph(PTO2Runtime* rt, uint64_t* args, int arg_count) {
    (void)arg_count;

    void* input_buf = reinterpret_cast<void*>(args[0]);
    int64_t* config = reinterpret_cast<int64_t*>(args[1]);

    int num_layers    = static_cast<int>(config[0]);
    int layer0_size   = static_cast<int>(config[1]);
    int deps_per_task = static_cast<int>(config[2]);
    int overlap       = static_cast<int>(config[3]);
    int worker_mode   = static_cast<int>(config[4]);
    int fix_tail      = static_cast<int>(config[5]);

    printf("  num_layers = %d, layer0_size = %d, deps_per_task = %d, overlap = %d%s\n",
        num_layers, layer0_size, deps_per_task, overlap,
        fix_tail ? ", fix_tail=1" : "");

    if (deps_per_task <= overlap) {
        printf("  ERROR: deps_per_task (%d) must be > overlap (%d)\n", deps_per_task, overlap);
        return;
    }

    int step = deps_per_task - overlap;
    int use_fix_tail = fix_tail && (step == 1);
    if (fix_tail && step != 1) {
        printf("  fix_tail ignored (only when deps_per_task - overlap == 1, current step=%d)\n", step);
    }

    DataType dtype    = DataType::FLOAT32;
    uint32_t shape[1] = {1};

    std::vector<int> layer_sizes;
    layer_sizes.resize(static_cast<size_t>(num_layers));
    int total_tasks = 0;
    for (int k = 0; k < num_layers; k++) {
        int sz = use_fix_tail ? layer0_size : layer_size(k, layer0_size, deps_per_task, overlap);
        layer_sizes[static_cast<size_t>(k)] = sz;
        total_tasks += sz;
    }

    std::vector<std::vector<Tensor>> layers;
    layers.resize(static_cast<size_t>(num_layers));
    for (int k = 0; k < num_layers; k++)
        layers[static_cast<size_t>(k)].reserve(static_cast<size_t>(layer_sizes[static_cast<size_t>(k)]));

    Tensor shared_in = make_tensor_external(input_buf, shape, 1, dtype);

    int task_global = 0;  // 全局提交计数，用于 worker_mode==THROUGHPUT_WORKER_ODD_AIC

    PTO2_SCOPE(rt) {
        // 第 1 层：layer_sizes[0] 个任务，无依赖，读共享输入
        int cur_sz0 = layer_sizes[0];
        for (int j = 0; j < cur_sz0; j++) {
            ++task_global;
            int wtype = task_worker_type(worker_mode, task_global, j, cur_sz0);
            Tensor t = make_tensor(shape, 1, dtype);
            PTOParam params;
            params.add_input(shared_in);
            params.add_output(t);
            pto2_submit_task(rt->orchestrators, FUNC_ELEMENT_WISE, wtype, params);
            layers[0].push_back(t);
        }

        std::vector<int> preds;
        preds.reserve(static_cast<size_t>(deps_per_task + 2));

        for (int k = 1; k < num_layers; k++) {
            int prev_sz = static_cast<int>(layers[static_cast<size_t>(k - 1)].size());
            int cur_sz = layer_sizes[static_cast<size_t>(k)];
            for (int j = 0; j < cur_sz; j++) {
                get_predecessors(j, prev_sz, deps_per_task, overlap, preds);
                if (preds.empty()) continue;
                ++task_global;
                int wtype = task_worker_type(worker_mode, task_global, j, cur_sz);
                Tensor out = make_tensor(shape, 1, dtype);
                const int max_inputs = 16;
                PTOParam params;
                for (size_t p = 0; p < preds.size() && (int)p < max_inputs; p++)
                    params.add_input(layers[static_cast<size_t>(k - 1)][static_cast<size_t>(preds[p])]);
                params.add_output(out);
                pto2_submit_task(rt->orchestrators, FUNC_ELEMENT_WISE, wtype, params);
                layers[static_cast<size_t>(k)].push_back(out);
            }
        }
    }

    pto2_orchestrator_done(rt->orchestrators);
    total_tasks = 0;
    for (size_t k = 0; k < layers.size(); k++) total_tasks += static_cast<int>(layers[k].size());
    printf("  Total tasks submitted: %d\n", total_tasks);
}

PTO2Runtime* setup_run(const ThroughputTestCase& tc, GraphCtx& ctx) {
    ctx.config[0] = static_cast<int64_t>(tc.num_layers);
    ctx.config[1] = static_cast<int64_t>(tc.layer0_size);
    ctx.config[2] = static_cast<int64_t>(tc.deps_per_task);
    ctx.config[3] = static_cast<int64_t>(tc.overlap);
    ctx.config[4] = static_cast<int64_t>(tc.worker_mode);
    ctx.config[5] = static_cast<int64_t>(tc.fix_tail);

    ctx.args[0] = reinterpret_cast<uint64_t>(static_cast<void*>(&g_throughput_input_buf));
    ctx.args[1] = reinterpret_cast<uint64_t>(ctx.config);
    for (int i = 2; i < 10; i++) ctx.args[i] = 0;

    return make_runtime();
}

#if PTO2_PROFILING

void section_header_100(char pad_char, const char* title) {
    int len = static_cast<int>(strlen(title));
    int left = (100 - len) / 2;
    int right = 100 - len - left;
    for (int i = 0; i < left; i++) putchar(pad_char);
    printf("%s", title);
    for (int i = 0; i < right; i++) putchar(pad_char);
    putchar('\n');
}

void print_config(const ThroughputTestCase& tc) {
    static const char* worker_mode_names[] = {
        "all-AIV",
        "odd-AIC/even-AIV",
        "half-AIC/half-AIV per layer",
    };
    int mode_idx = tc.worker_mode;
    const char* mode_str = (mode_idx >= 0 && mode_idx < 3) ? worker_mode_names[mode_idx] : "unknown";
    section_header_100('-', "--- Config ---");
    printf("  num_layers = %d, layer0_size = %d, deps_per_task = %d, overlap = %d (layered DAG)\n",
        tc.num_layers,
        tc.layer0_size,
        tc.deps_per_task,
        tc.overlap);
    printf("  worker_mode = %s\n", mode_str);
    if (tc.fix_tail)
        printf("  fix_tail = 1 (each layer size = layer0_size when deps_per_task - overlap == 1)\n");
}

void print_cpu_affinity(int num_sched, int orch_cpu) {
    static const int sched_cpus[] = {
        SCHED_CPU0,
        SCHED_CPU1,
        SCHED_CPU2,
        SCHED_CPU3,
        SCHED_CPU4,
        SCHED_CPU5,
        SCHED_CPU6,
        SCHED_CPU7,
    };
    section_header_100('-', "--- CPU affinity ---");
    printf("  orchestrator → core %d\n", orch_cpu >= 0 ? orch_cpu : ORCH_CPU);
    int max_sched = static_cast<int>(sizeof(sched_cpus) / sizeof(sched_cpus[0]));
    for (int i = 0; i < num_sched && i < max_sched; i++)
        printf("  scheduler[%d]  → core %d (configured)\n", i, sched_cpus[i]);
    printf("\n");
}

#endif  // PTO2_PROFILING
