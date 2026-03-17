/**
 * linear.cpp
 *
 * Linear chain backend (PERF_BACKEND=0): declarations + definitions.
 * Included by test_scheduler.cpp, test_orchestrator.cpp, test_orchestrator_scheduler.cpp.
 * Do not compile this file as a separate TU — it is #included by the test driver.
 */

// ─── Declarations ────────────────────────────────────────────────────────────

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

#ifndef PERF_CASE_IDX
#define PERF_CASE_IDX 0
#endif

struct LinearTestCase {
    const char* name;
    int chain_length;
    int tensor_nelems;
};

static constexpr int LINEAR_CASE_COUNT = 3;
static_assert(PERF_CASE_IDX >= 0 && PERF_CASE_IDX < LINEAR_CASE_COUNT,
              "PERF_CASE_IDX out of range");

extern const LinearTestCase PERF_CASES[LINEAR_CASE_COUNT];

static constexpr size_t LINEAR_MAX_NELEMS = 16384;

extern float g_input_buf[LINEAR_MAX_NELEMS];
extern float g_output_buf[LINEAR_MAX_NELEMS];

#define FUNC_ELEMENT_WISE 0

struct GraphCtx {
    int64_t  config[4];
    uint64_t args[10];
};

int          get_num_sched_threads();
void         perf_wait_sigstop();
void         build_graph(PTO2Runtime* rt, uint64_t* args, int arg_count);
PTO2Runtime* setup_run(const LinearTestCase& tc, GraphCtx& ctx);

#if PTO2_PROFILING
void section_header_100(char pad_char, const char* title);
void print_config(const LinearTestCase& tc);
void print_cpu_affinity(int num_sched);
#endif

// ─── Definitions ─────────────────────────────────────────────────────────────

const LinearTestCase PERF_CASES[LINEAR_CASE_COUNT] = {
    { "Linear-64   (chain=64,   tensor=1024 floats)",    64, 1024 },
    { "Linear-1024 (chain=1024, tensor=1024 floats)",  1024, 1024 },
    { "Linear-256  (chain=256,  tensor=16384 floats)",  256, 16384 },
};

float g_input_buf[LINEAR_MAX_NELEMS];
float g_output_buf[LINEAR_MAX_NELEMS];

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

void build_graph(PTO2Runtime* rt, uint64_t* args, int arg_count) {
    (void)arg_count;

    void*    input_buf  = reinterpret_cast<void*>(args[0]);
    void*    output_buf = reinterpret_cast<void*>(args[1]);
    int64_t* config     = reinterpret_cast<int64_t*>(args[2]);

    int      chain_length  = static_cast<int>(config[0]);
    uint64_t tensor_nelems = static_cast<uint64_t>(config[1]);

    printf("  chain_length = %d, tensor_nelems = %llu\n",
           chain_length, (unsigned long long)tensor_nelems);

    DataType dtype    = DataType::FLOAT32;
    uint32_t shape[1] = {static_cast<uint32_t>(tensor_nelems)};

    Tensor prev = make_tensor_external(input_buf, shape, 1, dtype);

    // pto2_submit_task requires an open scope (scope_stack_top >= 0).
    PTO2_SCOPE(rt) {
        for (int i = 0; i < chain_length; i++) {
            bool is_last = (i == chain_length - 1);
            Tensor next = is_last
                ? make_tensor_external(output_buf, shape, 1, dtype)
                : make_tensor(shape, 1, dtype);

            PTOParam params[] = {
                make_input_param(prev),
                make_output_param(next),
            };
            pto2_submit_task(rt->orchestrators, FUNC_ELEMENT_WISE,
                             PTO2_WORKER_VECTOR, params, 2);
            prev = next;
        }
    }

    pto2_orchestrator_done(rt->orchestrators);
    printf("  Total tasks submitted: %d\n", chain_length);
}

PTO2Runtime* setup_run(const LinearTestCase& tc, GraphCtx& ctx) {
    ctx.config[0] = static_cast<int64_t>(tc.chain_length);
    ctx.config[1] = static_cast<int64_t>(tc.tensor_nelems);
    ctx.config[2] = static_cast<int64_t>(tc.tensor_nelems) * sizeof(float);
    ctx.config[3] = 0;

    ctx.args[0] = reinterpret_cast<uint64_t>(static_cast<void*>(g_input_buf));
    ctx.args[1] = reinterpret_cast<uint64_t>(static_cast<void*>(g_output_buf));
    ctx.args[2] = reinterpret_cast<uint64_t>(ctx.config);
    ctx.args[3] = ctx.config[2];
    for (int i = 4; i < 10; i++) ctx.args[i] = 0;

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

void print_config(const LinearTestCase& tc) {
    section_header_100('-', "--- Config ---");
    printf("  chain_length = %d, tensor_nelems = %d, buf_bytes = %zu\n",
           tc.chain_length, tc.tensor_nelems,
           static_cast<size_t>(tc.tensor_nelems) * sizeof(float));
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
