/**
 * test_alternating_matmul_add.cpp
 *
 * Alternating Matmul-Add backend (PERF_BACKEND=4): declarations + definitions.
 * Included by test_* when PERF_BACKEND=4. Do not compile as separate TU.
 *
 * Graph topology: interleaved groups of CUBE (matmul) and VECTOR (add) tasks.
 * All tasks are independent — no data dependencies between them.
 *
 * Case 0: batch=500, M=4, N=4, matmul_batch=4, add_batch=4
 *         → 500 matmul groups + 500 add groups = 1000 tasks
 * Case 1: batch=512, M=2, N=5, matmul_batch=4, add_batch=5
 *         → 256 matmul groups + 512 add groups = 768 tasks
 */

// ─── Declarations ─────────────────────────────────────────────────────────────

#include "pto_runtime2.h"
#include "test_common.h"
#if defined(PTO2_SIM_AICORE_UT)
#include "sim_aicore.h"
#endif
#include "common/platform_config.h"
#include "cpu_affinity.h"
#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <csignal>
#include <unistd.h>

#ifndef PERF_CASE_IDX
#define PERF_CASE_IDX 0
#endif

struct AlternatingTestCase {
    const char* name;
    int         batch;
    int         M;             // matmul tasks per batch
    int         N;             // add tasks per batch
    int         matmul_batch;  // matmul tasks per submitted group
    int         add_batch;     // add tasks per submitted group
};

static constexpr int ALT_CASE_COUNT = 2;
static_assert(PERF_CASE_IDX >= 0 && PERF_CASE_IDX < ALT_CASE_COUNT,
              "PERF_CASE_IDX out of range");

extern const AlternatingTestCase PERF_CASES[ALT_CASE_COUNT];

// Fixed tile dimensions: each matmul task is 128×128, each add task is 128×128.
static constexpr uint64_t MATMUL_ELEMS = 128 * 128;
static constexpr uint64_t ADD_ELEMS    = 128 * 128;

// Buffer sizes covering the largest case across all cases.
// Case 0: 500*4 = 2000 matmul tasks, 500*4 = 2000 add tasks
// Case 1: 512*2 = 1024 matmul tasks, 512*5 = 2560 add tasks
static constexpr size_t ALT_MAX_MATMUL_NELEMS = 500 * 4 * MATMUL_ELEMS;  // 32,768,000
static constexpr size_t ALT_MAX_ADD_NELEMS    = 512 * 5 * ADD_ELEMS;     // 41,943,040

extern float g_alt_A[ALT_MAX_MATMUL_NELEMS];
extern float g_alt_B[ALT_MAX_MATMUL_NELEMS];
extern float g_alt_C[ALT_MAX_MATMUL_NELEMS];
extern float g_alt_X[ALT_MAX_ADD_NELEMS];
extern float g_alt_Y[ALT_MAX_ADD_NELEMS];
extern float g_alt_Z[ALT_MAX_ADD_NELEMS];

#define FUNC_MATMUL 0
#define FUNC_ADD    1

struct GraphCtx {
    int64_t  config[5];   // batch, M, N, matmul_batch, add_batch
    uint64_t args[10];
};

int          get_num_sched_threads();
void         perf_wait_sigstop();
void         build_graph(PTO2Runtime* rt, uint64_t* args, int arg_count);
PTO2Runtime* setup_run(const AlternatingTestCase& tc, GraphCtx& ctx);

#if PTO2_PROFILING
void section_header_100(char pad_char, const char* title);
void print_config(const AlternatingTestCase& tc);
void print_cpu_affinity(int num_sched, int orch_cpu);
#endif

// ─── Definitions ──────────────────────────────────────────────────────────────

const AlternatingTestCase PERF_CASES[ALT_CASE_COUNT] = {
    { "Alternating (batch=500, M=4, N=4, mb=4, ab=4)", 500, 4, 4, 4, 4 },
    { "Alternating (batch=512, M=2, N=5, mb=4, ab=5)", 512, 2, 5, 4, 5 },
};

float g_alt_A[ALT_MAX_MATMUL_NELEMS];
float g_alt_B[ALT_MAX_MATMUL_NELEMS];
float g_alt_C[ALT_MAX_MATMUL_NELEMS];
float g_alt_X[ALT_MAX_ADD_NELEMS];
float g_alt_Y[ALT_MAX_ADD_NELEMS];
float g_alt_Z[ALT_MAX_ADD_NELEMS];

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
 * build_graph — submits interleaved CUBE and VECTOR task groups.
 *
 * All tasks are completely independent.  Matmul and add groups alternate in
 * submission order, each group writing to a non-overlapping slice of its
 * respective external buffer.  TensorMap detects the non-overlap and issues
 * no dependencies between groups.
 *
 * args[0]: ptr_A  args[1]: ptr_B  args[2]: ptr_C  (matmul buffers)
 * args[3]: ptr_X  args[4]: ptr_Y  args[5]: ptr_Z  (add buffers)
 * args[6]: size_matmul_bytes  args[7]: size_add_bytes
 * args[8]: ptr_config
 */
void build_graph(PTO2Runtime* rt, uint64_t* args, int arg_count) {
    (void)arg_count;

    void*    host_A      = reinterpret_cast<void*>(args[0]);
    void*    host_B      = reinterpret_cast<void*>(args[1]);
    void*    host_C      = reinterpret_cast<void*>(args[2]);
    void*    host_X      = reinterpret_cast<void*>(args[3]);
    void*    host_Y      = reinterpret_cast<void*>(args[4]);
    void*    host_Z      = reinterpret_cast<void*>(args[5]);
    size_t   size_matmul = static_cast<size_t>(args[6]);
    size_t   size_add    = static_cast<size_t>(args[7]);
    int64_t* config      = reinterpret_cast<int64_t*>(args[8]);

    int batch        = static_cast<int>(config[0]);
    int M            = static_cast<int>(config[1]);
    int N            = static_cast<int>(config[2]);
    int matmul_batch = static_cast<int>(config[3]);
    int add_batch    = static_cast<int>(config[4]);

    int total_matmul      = batch * M;
    int total_add         = batch * N;
    int num_matmul_groups = total_matmul / matmul_batch;
    int num_add_groups    = total_add / add_batch;
    int max_groups        = std::max(num_matmul_groups, num_add_groups);

    printf("  batch=%d, M=%d, N=%d, matmul_batch=%d, add_batch=%d\n",
           batch, M, N, matmul_batch, add_batch);

    uint32_t A_shapes[1] = {static_cast<uint32_t>(size_matmul / sizeof(float))};
    uint32_t X_shapes[1] = {static_cast<uint32_t>(size_add    / sizeof(float))};
    Tensor ext_A = make_tensor_external(host_A, A_shapes, 1, DataType::FLOAT32);
    Tensor ext_B = make_tensor_external(host_B, A_shapes, 1, DataType::FLOAT32);
    Tensor ext_C = make_tensor_external(host_C, A_shapes, 1, DataType::FLOAT32);
    Tensor ext_X = make_tensor_external(host_X, X_shapes, 1, DataType::FLOAT32);
    Tensor ext_Y = make_tensor_external(host_Y, X_shapes, 1, DataType::FLOAT32);
    Tensor ext_Z = make_tensor_external(host_Z, X_shapes, 1, DataType::FLOAT32);

    int total_tasks = 0;

    // All tasks are independent: wrap in a single scope so pto2_submit_task
    // has an open scope (scope_stack_top >= 0) throughout.
    PTO2_SCOPE(rt) {
        for (int group_idx = 0; group_idx < max_groups; group_idx++) {
            if (group_idx < num_matmul_groups) {
                uint64_t offset = static_cast<uint64_t>(group_idx) * matmul_batch * MATMUL_ELEMS;
                uint64_t size   = static_cast<uint64_t>(matmul_batch) * MATMUL_ELEMS;
                uint32_t shapes[1]  = {static_cast<uint32_t>(size)};
                uint32_t offsets[1] = {static_cast<uint32_t>(offset)};
                Tensor A_view = ext_A.view(shapes, offsets);
                Tensor B_view = ext_B.view(shapes, offsets);
                Tensor C_view = ext_C.view(shapes, offsets);
                PTOParam params;
                params.add_input(A_view);
                params.add_input(B_view);
                params.add_output(C_view);
                pto2_submit_task(rt->orchestrators, FUNC_MATMUL, PTO2_WORKER_CUBE, params);
                total_tasks++;
            }
            if (group_idx < num_add_groups) {
                uint64_t offset = static_cast<uint64_t>(group_idx) * add_batch * ADD_ELEMS;
                uint64_t size   = static_cast<uint64_t>(add_batch) * ADD_ELEMS;
                uint32_t shapes[1]  = {static_cast<uint32_t>(size)};
                uint32_t offsets[1] = {static_cast<uint32_t>(offset)};
                Tensor X_view = ext_X.view(shapes, offsets);
                Tensor Y_view = ext_Y.view(shapes, offsets);
                Tensor Z_view = ext_Z.view(shapes, offsets);
                PTOParam params;
                params.add_input(X_view);
                params.add_input(Y_view);
                params.add_output(Z_view);
                pto2_submit_task(rt->orchestrators, FUNC_ADD, PTO2_WORKER_VECTOR, params);
                total_tasks++;
            }
        }
    }

    pto2_orchestrator_done(rt->orchestrators);
    printf("  Total tasks submitted: %d\n", total_tasks);
}

PTO2Runtime* setup_run(const AlternatingTestCase& tc, GraphCtx& ctx) {
    size_t total_matmul = static_cast<size_t>(tc.batch) * tc.M;
    size_t total_add    = static_cast<size_t>(tc.batch) * tc.N;

    ctx.config[0] = static_cast<int64_t>(tc.batch);
    ctx.config[1] = static_cast<int64_t>(tc.M);
    ctx.config[2] = static_cast<int64_t>(tc.N);
    ctx.config[3] = static_cast<int64_t>(tc.matmul_batch);
    ctx.config[4] = static_cast<int64_t>(tc.add_batch);

    size_t size_matmul = total_matmul * MATMUL_ELEMS * sizeof(float);
    size_t size_add    = total_add    * ADD_ELEMS    * sizeof(float);

    ctx.args[0] = reinterpret_cast<uint64_t>(static_cast<void*>(g_alt_A));
    ctx.args[1] = reinterpret_cast<uint64_t>(static_cast<void*>(g_alt_B));
    ctx.args[2] = reinterpret_cast<uint64_t>(static_cast<void*>(g_alt_C));
    ctx.args[3] = reinterpret_cast<uint64_t>(static_cast<void*>(g_alt_X));
    ctx.args[4] = reinterpret_cast<uint64_t>(static_cast<void*>(g_alt_Y));
    ctx.args[5] = reinterpret_cast<uint64_t>(static_cast<void*>(g_alt_Z));
    ctx.args[6] = static_cast<uint64_t>(size_matmul);
    ctx.args[7] = static_cast<uint64_t>(size_add);
    ctx.args[8] = reinterpret_cast<uint64_t>(ctx.config);
    ctx.args[9] = 0;

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

void print_config(const AlternatingTestCase& tc) {
    section_header_100('-', "--- Config ---");
    int matmul_groups = (tc.batch * tc.M) / tc.matmul_batch;
    int add_groups    = (tc.batch * tc.N) / tc.add_batch;
    printf("  batch=%d, M=%d, N=%d, matmul_batch=%d, add_batch=%d\n",
           tc.batch, tc.M, tc.N, tc.matmul_batch, tc.add_batch);
    printf("  matmul_groups=%d, add_groups=%d, total_tasks=%d\n",
           matmul_groups, add_groups, matmul_groups + add_groups);
}

void print_cpu_affinity(int num_sched, int orch_cpu) {
    static const int sched_cpus[] = {
        SCHED_CPU0, SCHED_CPU1, SCHED_CPU2, SCHED_CPU3,
        SCHED_CPU4, SCHED_CPU5, SCHED_CPU6, SCHED_CPU7,
    };
    section_header_100('-', "--- CPU affinity ---");
    printf("  orchestrator → core %d\n", orch_cpu >= 0 ? orch_cpu : ORCH_CPU);
    int max_sched = static_cast<int>(sizeof(sched_cpus) / sizeof(sched_cpus[0]));
    for (int i = 0; i < num_sched && i < max_sched; i++)
        printf("  scheduler[%d]  → core %d (configured)\n", i, sched_cpus[i]);
    printf("\n");
}

#endif  // PTO2_PROFILING
