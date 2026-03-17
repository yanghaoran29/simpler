/**
 * latency.cpp
 *
 * 极限延迟 (Min Latency) backend (PERF_BACKEND=8).
 * Multiple independent linear chains: chain_c = task1 → task2 → … → taskL.
 * Parameters: num_chains (链的数量), chain_length (每条链长度).
 * Used to measure latency / throughput under configurable chain count and length.
 *
 * Case 0: 所有任务均为 aiv (PTO2_WORKER_VECTOR).
 * Case 1: 链上任务 aic / aiv 交替 (PTO2_WORKER_CUBE / PTO2_WORKER_VECTOR).
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

#include <csignal>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>

#include "common/platform_config.h"
#include "cpu_affinity.h"

#ifndef PERF_CASE_IDX
#define PERF_CASE_IDX 0
#endif
#ifndef AICPU_UT_LATENCY_NUM_CHAINS
#define AICPU_UT_LATENCY_NUM_CHAINS 64
#endif
#ifndef AICPU_UT_LATENCY_CHAIN_LENGTH
#define AICPU_UT_LATENCY_CHAIN_LENGTH 64
#endif

// ─── Stringify helper ─────────────────────────────────────────────────────────
#define LATENCY_STRINGIFY_IMPL(x) #x
#define LATENCY_STRINGIFY(x) LATENCY_STRINGIFY_IMPL(x)

#define LATENCY_PARAM_STR                                                                 \
    "(chains=" LATENCY_STRINGIFY(AICPU_UT_LATENCY_NUM_CHAINS) ", len=" LATENCY_STRINGIFY( \
        AICPU_UT_LATENCY_CHAIN_LENGTH) ")"
struct LatencyTestCase {
    const char* name;
    int num_chains;    // number of independent chains
    int chain_length;  // tasks per chain: task1 → task2 → … → taskL
};

static constexpr int LATENCY_CASE_COUNT = 2;
static_assert(PERF_CASE_IDX >= 0 && PERF_CASE_IDX < LATENCY_CASE_COUNT, "PERF_CASE_IDX out of range");

extern const LatencyTestCase PERF_CASES[LATENCY_CASE_COUNT];
extern float g_latency_input_buf[AICPU_UT_LATENCY_NUM_CHAINS];
extern float g_latency_output_buf[AICPU_UT_LATENCY_NUM_CHAINS];

#define FUNC_ELEMENT_WISE 0

struct GraphCtx {
    int64_t config[3];  // num_chains, chain_length, case_index (0=all aiv, 1=aic/aiv alternate)
    uint64_t args[10];
};

int get_num_sched_threads();
void perf_wait_sigstop();
void build_graph(PTO2Runtime* rt, uint64_t* args, int arg_count);
PTO2Runtime* setup_run(const LatencyTestCase& tc, GraphCtx& ctx);

#if PTO2_PROFILING
void section_header_100(char pad_char, const char* title);
void print_config(const LatencyTestCase& tc);
void print_cpu_affinity(int num_sched);
#endif

// ─── Definitions ─────────────────────────────────────────────────────────────

const LatencyTestCase PERF_CASES[LATENCY_CASE_COUNT] = {
    {"Latency all-AIV " LATENCY_PARAM_STR, AICPU_UT_LATENCY_NUM_CHAINS, AICPU_UT_LATENCY_CHAIN_LENGTH},
    {"Latency aic/aiv alternate " LATENCY_PARAM_STR, AICPU_UT_LATENCY_NUM_CHAINS, AICPU_UT_LATENCY_CHAIN_LENGTH},
};

float g_latency_input_buf[AICPU_UT_LATENCY_NUM_CHAINS];
float g_latency_output_buf[AICPU_UT_LATENCY_NUM_CHAINS];

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

    void* input_base = reinterpret_cast<void*>(args[0]);
    void* output_base = reinterpret_cast<void*>(args[1]);
    int64_t* config = reinterpret_cast<int64_t*>(args[2]);

    int num_chains = static_cast<int>(config[0]);
    int chain_length = static_cast<int>(config[1]);
    int case_index = static_cast<int>(config[2]);  // 0 = all aiv, 1 = aic/aiv alternate

    printf("  num_chains = %d, chain_length = %d, case = %d (%s)\n",
        num_chains,
        chain_length,
        case_index,
        case_index == 0 ? "all aiv" : "aic/aiv alternate");

    DataType dtype = DataType::FLOAT32;
    uint32_t shape[1] = {1};
    size_t row_bytes = sizeof(float);

    PTO2_SCOPE(rt) {
        for (int c = 0; c < num_chains; c++) {
            void* in_c = static_cast<char*>(input_base) + c * row_bytes;
            void* out_c = static_cast<char*>(output_base) + c * row_bytes;

            Tensor prev = make_tensor_external(in_c, shape, 1, dtype);

            for (int i = 0; i < chain_length; i++) {
                bool is_last = (i == chain_length - 1);
                Tensor next = is_last ? make_tensor_external(out_c, shape, 1, dtype) : make_tensor(shape, 1, dtype);

                int worker_type =
                    (case_index == 0) ? PTO2_WORKER_VECTOR : ((i % 2 == 0) ? PTO2_WORKER_CUBE : PTO2_WORKER_VECTOR);
                PTOParam params[] = {
                    make_input_param(prev),
                    make_output_param(next),
                };
                pto2_submit_task(rt->orchestrators, FUNC_ELEMENT_WISE, worker_type, params, 2);
                prev = next;
            }
        }
    }

    pto2_orchestrator_done(rt->orchestrators);
    printf("  Total tasks submitted: %d\n", num_chains * chain_length);
}

PTO2Runtime* setup_run(const LatencyTestCase& tc, GraphCtx& ctx) {
    (void)tc;
    ctx.config[0] = static_cast<int64_t>(AICPU_UT_LATENCY_NUM_CHAINS);
    ctx.config[1] = static_cast<int64_t>(AICPU_UT_LATENCY_CHAIN_LENGTH);
    ctx.config[2] = static_cast<int64_t>(PERF_CASE_IDX);

    ctx.args[0] = reinterpret_cast<uint64_t>(static_cast<void*>(g_latency_input_buf));
    ctx.args[1] = reinterpret_cast<uint64_t>(static_cast<void*>(g_latency_output_buf));
    ctx.args[2] = reinterpret_cast<uint64_t>(ctx.config);
    for (int i = 3; i < 10; i++) ctx.args[i] = 0;

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

void print_config(const LatencyTestCase& tc) {
    section_header_100('-', "--- Config ---");
    printf("  num_chains = %d, chain_length = %d (latency: task1→…→taskN per chain)\n", tc.num_chains, tc.chain_length);
}

void print_cpu_affinity(int num_sched) {
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
    int orch_core = current_cpu();
    printf("  orchestrator → core %d\n", orch_core >= 0 ? orch_core : ORCH_CPU);
    int max_sched = static_cast<int>(sizeof(sched_cpus) / sizeof(sched_cpus[0]));
    for (int i = 0; i < num_sched && i < max_sched; i++)
        printf("  scheduler[%d]  → core %d (configured)\n", i, sched_cpus[i]);
    printf("\n");
}

#endif  // PTO2_PROFILING
