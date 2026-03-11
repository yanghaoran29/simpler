/**
 * throughput_case.cpp
 *
 * 极限吞吐 (Max Throughput) backend (PERF_BACKEND=7).
 * Single scope with a large number of independent tasks: all read from one
 * shared external input, each writes to its own output. No dependencies
 * between tasks → maximizes tasks/sec (scheduler + orchestrator throughput).
 *
 * Included by test_sched_prof_only.cpp, test_orch_only.cpp, test_concurrent.cpp.
 * Do not compile as separate TU — it is #included by the test driver.
 */

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

struct ThroughputTestCase {
    const char* name;
    int         num_tasks;     // number of independent tasks in one scope
    int         tensor_nelems; // elements per tensor (small to stress scheduler)
};

static constexpr int THROUGHPUT_CASE_COUNT = 1;
static_assert(PERF_CASE_IDX >= 0 && PERF_CASE_IDX < THROUGHPUT_CASE_COUNT,
              "PERF_CASE_IDX out of range");

extern const ThroughputTestCase PERF_CASES[THROUGHPUT_CASE_COUNT];

static constexpr size_t THROUGHPUT_MAX_NELEMS = 64;
static constexpr size_t THROUGHPUT_MAX_TASKS  = 16384;

extern float g_throughput_input_buf[THROUGHPUT_MAX_NELEMS];
extern float g_throughput_output_buf[THROUGHPUT_MAX_TASKS][THROUGHPUT_MAX_NELEMS];

#define FUNC_ELEMENT_WISE 0

struct ThroughputRunCtx {
    int64_t  config[4];
    uint64_t args[10];
};

int          get_num_sched_threads();
void         perf_wait_sigstop();
void         build_throughput_graph(PTO2Runtime* rt, uint64_t* args, int arg_count);
PTO2Runtime* setup_run(const ThroughputTestCase& tc, ThroughputRunCtx& ctx);

#if PTO2_PROFILING
void section_header_100(char pad_char, const char* title);
void print_config(const ThroughputTestCase& tc);
void print_cpu_affinity(int num_sched);
#endif

// ─── Definitions ─────────────────────────────────────────────────────────────

const ThroughputTestCase PERF_CASES[THROUGHPUT_CASE_COUNT] = {
    { "MaxThroughput (num_tasks=8192, tensor=64 floats)", 8192, 64 },
};

float g_throughput_input_buf[THROUGHPUT_MAX_NELEMS];
float g_throughput_output_buf[THROUGHPUT_MAX_TASKS][THROUGHPUT_MAX_NELEMS];

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

void build_throughput_graph(PTO2Runtime* rt, uint64_t* args, int arg_count) {
    (void)arg_count;

    void*    input_buf  = reinterpret_cast<void*>(args[0]);
    int64_t* config     = reinterpret_cast<int64_t*>(args[1]);

    int num_tasks      = static_cast<int>(config[0]);
    int tensor_nelems  = static_cast<int>(config[1]);

    printf("  num_tasks = %d, tensor_nelems = %d\n", num_tasks, tensor_nelems);

    DataType dtype   = DataType::FLOAT32;
    uint64_t shape[1] = {static_cast<uint64_t>(tensor_nelems)};

    Tensor shared_in = make_tensor_external(input_buf, shape, 1, dtype);

    PTO2_SCOPE(rt) {
        for (int i = 0; i < num_tasks; i++) {
            Tensor out = make_tensor_external(&g_throughput_output_buf[i][0], shape, 1, dtype);
            PTOParam params[] = {
                make_input_param(shared_in),
                make_output_param(out),
            };
            pto2_submit_task(rt->orchestrators, FUNC_ELEMENT_WISE,
                             PTO2_WORKER_VECTOR, params, 2);
        }
    }

    pto2_orchestrator_done(rt->orchestrators);
    printf("  Total tasks submitted: %d\n", num_tasks);
}

PTO2Runtime* setup_run(const ThroughputTestCase& tc, ThroughputRunCtx& ctx) {
    ctx.config[0] = static_cast<int64_t>(tc.num_tasks);
    ctx.config[1] = static_cast<int64_t>(tc.tensor_nelems);
    ctx.config[2] = 0;
    ctx.config[3] = 0;

    ctx.args[0] = reinterpret_cast<uint64_t>(static_cast<void*>(g_throughput_input_buf));
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

void print_config(const ThroughputTestCase& tc) {
    section_header_100('-', "--- Config ---");
    printf("  num_tasks = %d, tensor_nelems = %d (limit throughput: many independent tasks)\n",
           tc.num_tasks, tc.tensor_nelems);
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
