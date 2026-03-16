/**
 * test_deg4.cpp
 *
 * Degree benchmark backend (PERF_BACKEND=3): layered DAG test case
 * with avg_degree ≈ 4.
 *
 * Graph topology: W tasks per layer, L layers.
 * Each task in layer i+1 takes D inputs from layer i using circular offsets:
 *   task[i+1][j] depends on task[i][(j+d) % W]  for d = 0 .. D-1
 *
 * Case: D=4, W=32,  L=128, total=4096, avg_degree = 4*127/128  = 3.969 → "4.0"
 *
 * Included by test_sched_prof_only.cpp, test_orch_only.cpp, test_concurrent.cpp.
 * Do not compile this file as a separate TU — it is #included by the test driver.
 */

// ─── Declarations ─────────────────────────────────────────────────────────────

#include "pto_runtime2.h"
#include "test_common.h"
#if defined(PTO2_SIM_AICORE_UT)
#include "sim_aicore.h"
#endif
#include "common/platform_config.h"
#include "cpu_affinity.h"
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <csignal>
#include <unistd.h>
#include <vector>

struct DegreeTestCase {
    const char* name;
    int         layer_width;   // W: tasks per layer
    int         num_layers;    // L: number of layers
    int         fanout;        // D: inputs each task takes from the previous layer
    int         tensor_nelems; // elements per intermediate tensor
};

static constexpr int DEG_CASE_COUNT = 1;

extern const DegreeTestCase PERF_CASES[DEG_CASE_COUNT];

static constexpr int DEG_MAX_TENSOR_NELEMS = 64;

extern float g_deg_ext_buf[DEG_MAX_TENSOR_NELEMS];

#define FUNC_ELEMENT_WISE 0

struct DegRunCtx {
    int64_t  config[4];
    uint64_t args[10];
};

int          get_num_sched_threads();
void         perf_wait_sigstop();
void         build_degree_graph(PTO2Runtime* rt, uint64_t* args, int arg_count);
PTO2Runtime* setup_run(const DegreeTestCase& tc, DegRunCtx& ctx);

#if PTO2_PROFILING
void section_header_100(char pad_char, const char* title);
void print_config(const DegreeTestCase& tc);
void print_cpu_affinity(int num_sched);
#endif

// ─── Definitions ──────────────────────────────────────────────────────────────

const DegreeTestCase PERF_CASES[DEG_CASE_COUNT] = {
    // avg_degree = 4*(128-1)/128 = 3.969 → prints "4.0"
    { "Degree-4 (W=32,  L=128, fanout=4, tasks=4096)",  32, 128, 4, DEG_MAX_TENSOR_NELEMS },
};

float g_deg_ext_buf[DEG_MAX_TENSOR_NELEMS];

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

/**
 * build_degree_graph — constructs a layered DAG with circular fan-out.
 *
 * Layer 0  : EW tasks, each reads the shared external input.
 *            No dependency between tasks in the same layer.
 * Layer i>0: EW tasks, task[i][j] depends on task[i-1][(j+d) % EW] for d=0..D-1.
 *            Each predecessor therefore notifies D successors (fanout = D).
 *
 * EW = W * num_sched_threads (effective layer width).
 * Scaling W by num_sched ensures num_sched * W tasks are ready at startup,
 * so every scheduler thread gets initial work without monopolisation.
 * avg_degree = D * (L-1) / L depends only on D and L, not on layer width,
 * so it is preserved exactly.
 */
void build_degree_graph(PTO2Runtime* rt, uint64_t* args, int arg_count) {
    (void)arg_count;

    void*    ext_input  = reinterpret_cast<void*>(args[0]);
    int64_t* config     = reinterpret_cast<int64_t*>(args[1]);

    int W             = static_cast<int>(config[0]);
    int L             = static_cast<int>(config[1]);
    int fanout        = static_cast<int>(config[2]);
    int tensor_nelems = static_cast<int>(config[3]);

    // Scale layer width so every scheduler thread starts with W ready tasks.
    // avg_degree = fanout * (L-1) / L is independent of layer width: unchanged.
    int num_sched = get_num_sched_threads();
    int EW        = W * num_sched;

    printf("  layer_width=%d, num_layers=%d, fanout=%d, tensor_nelems=%d\n",
           EW, L, fanout, tensor_nelems);

    DataType dtype    = DataType::FLOAT32;
    uint32_t shape[1] = {static_cast<uint32_t>(tensor_nelems)};

    Tensor ext_in = make_tensor_external(ext_input, shape, 1, dtype);

    // Two alternating row buffers; std::vector avoids requiring a Tensor
    // default constructor while keeping the ping-pong pattern allocation-free
    // after the initial reserve.
    std::vector<Tensor> prev_layer, curr_layer;
    prev_layer.reserve(static_cast<size_t>(EW));
    curr_layer.reserve(static_cast<size_t>(EW));

    // pto2_submit_task requires an open scope (scope_stack_top >= 0).
    // Wrap all submissions in a single scope that covers the full graph.
    PTO2_SCOPE(rt) {
        // ── Layer 0 ──────────────────────────────────────────────────────────
        // EW independent tasks read the shared external input.
        // All EW are immediately ready, giving every scheduler thread W tasks.
        for (int j = 0; j < EW; j++) {
            Tensor t = make_tensor(shape, 1, dtype);
            PTOParam params[2] = {
                make_input_param(ext_in),
                make_output_param(t),
            };
            pto2_submit_task(rt->orchestrators, FUNC_ELEMENT_WISE, PTO2_WORKER_VECTOR, params, 2);
            prev_layer.push_back(t);
        }

        // ── Layers 1 .. L-1 ──────────────────────────────────────────────────
        // task[i][j] depends on prev_layer[(j+d) % EW] for d = 0 .. fanout-1.
        // Each element of prev_layer is therefore consumed by exactly 'fanout'
        // tasks in curr_layer, giving fanout edges per non-sink task.
        for (int i = 1; i < L; i++) {
            curr_layer.clear();
            for (int j = 0; j < EW; j++) {
                Tensor t = make_tensor(shape, 1, dtype);
                // fanout ≤ 8, so 9 params (8 inputs + 1 output) is sufficient.
                PTOParam params[9];
                int np = 0;
                for (int d = 0; d < fanout; d++)
                    params[np++] = make_input_param(prev_layer[static_cast<size_t>((j + d) % EW)]);
                params[np++] = make_output_param(t);
                pto2_submit_task(rt->orchestrators, FUNC_ELEMENT_WISE, PTO2_WORKER_VECTOR, params, np);
                curr_layer.push_back(t);
            }
            std::swap(prev_layer, curr_layer);
        }
    }

    pto2_orchestrator_done(rt->orchestrators);
    printf("  Total tasks submitted: %d\n", EW * L);
}

PTO2Runtime* setup_run(const DegreeTestCase& tc, DegRunCtx& ctx) {
    ctx.config[0] = static_cast<int64_t>(tc.layer_width);
    ctx.config[1] = static_cast<int64_t>(tc.num_layers);
    ctx.config[2] = static_cast<int64_t>(tc.fanout);
    ctx.config[3] = static_cast<int64_t>(tc.tensor_nelems);

    ctx.args[0] = reinterpret_cast<uint64_t>(static_cast<void*>(g_deg_ext_buf));
    ctx.args[1] = reinterpret_cast<uint64_t>(ctx.config);
    for (int i = 2; i < 10; i++) ctx.args[i] = 0;

    return make_runtime();
}

#if PTO2_PROFILING

void section_header_100(char pad_char, const char* title) {
    int len   = static_cast<int>(strlen(title));
    int left  = (100 - len) / 2;
    int right = 100 - len - left;
    for (int i = 0; i < left;  i++) putchar(pad_char);
    printf("%s", title);
    for (int i = 0; i < right; i++) putchar(pad_char);
    putchar('\n');
}

void print_config(const DegreeTestCase& tc) {
    section_header_100('-', "--- Config ---");
    int    num_sched   = get_num_sched_threads();
    int    EW          = tc.layer_width * num_sched;
    int    total_tasks = EW * tc.num_layers;
    double avg_deg     = static_cast<double>(tc.fanout)
                       * (tc.num_layers - 1) / tc.num_layers;
    printf("  layer_width=%d (W=%d * %d threads), num_layers=%d, fanout=%d, "
           "total_tasks=%d, expected_avg_degree=%.3f\n",
           EW, tc.layer_width, num_sched, tc.num_layers, tc.fanout, total_tasks, avg_deg);
}

void print_cpu_affinity(int num_sched) {
    static const int sched_cpus[] = {
        SCHED_CPU0, SCHED_CPU1, SCHED_CPU2, SCHED_CPU3,
        SCHED_CPU4, SCHED_CPU5, SCHED_CPU6, SCHED_CPU7,
    };
    section_header_100('-', "--- CPU affinity ---");
    int orch_core = current_cpu();
    printf("  orchestrator → core %d\n", orch_core >= 0 ? orch_core : ORCH_CPU);
    int max_sched = static_cast<int>(sizeof(sched_cpus) / sizeof(sched_cpus[0]));
    for (int i = 0; i < num_sched && i < max_sched; i++)
        printf("  scheduler[%d]  → core %d (configured)\n", i, sched_cpus[i]);
    printf("\n");
}

#endif  // PTO2_PROFILING
