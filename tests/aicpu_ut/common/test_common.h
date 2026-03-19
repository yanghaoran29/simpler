/**
 * test_common.h
 *
 * Common definitions for orchestration unit tests
 */

#ifndef TEST_COMMON_H
#define TEST_COMMON_H

#include <cstdio>
#include <cstdint>

// Forward declaration
struct PTO2Runtime;

// Global test counters
extern int g_pass;
extern int g_fail;

// Test framework macros. When PTO2_PROFILING=0 (--no-profiling), only pass/fail count is output.
#define CHECK(cond)                                                          \
    do {                                                                     \
        if (!(cond)) {                                                       \
            g_fail++;                                                        \
            if (PTO2_PROFILING) fprintf(stderr, "  FAIL [%s:%d]  %s\n",      \
                    __FILE__, __LINE__, #cond);                              \
        } else {                                                             \
            g_pass++;                                                        \
        }                                                                    \
    } while (0)

#if PTO2_PROFILING
#define TEST_BEGIN(name) printf("\n  --- %s ---\n", (name))
#else
#define TEST_BEGIN(name) ((void)0)
#endif
#define TEST_END() printf("  PASS: %d, FAIL: %d\n", g_pass, g_fail)

// Common helper functions
uint64_t float_to_u64(float f);
uint64_t perf_now_us();          // Monotonic clock, returns microseconds, for performance measurement
PTO2Runtime* make_runtime();
int sim_drain_one_pass(PTO2Runtime* rt);
int sim_run_all(PTO2Runtime* rt, int max_rounds = 1000);
// Scheduler profiling
#if PTO2_PROFILING
#include "common/platform_config.h"
#include "pto_runtime2_types.h"
struct SchedProfilingData {
    int64_t tasks_dispatched[PTO2_NUM_WORKER_TYPES];
    int64_t fanout_edges_total;
    int32_t fanout_max_degree;
    int64_t tasks_enqueued_by_completion;
    int64_t fanin_edges_total;
    int32_t fanin_max_degree;
    int64_t rounds_total;
    int64_t rounds_with_progress;
    uint64_t dispatch_cycle;
    uint64_t complete_cycle;
};
#endif

void print_sched_profiling(PTO2Runtime* rt);

#if PTO2_PROFILING
extern SchedProfilingData g_sched_prof_data;
#endif

#if PTO2_SCHED_PROFILING
void pto2_print_sim_sched_summary(SchedProfilingData* data, int64_t tasks_completed, int64_t tasks_consumed);
#endif

#if PTO2_PROFILING
void run_sched_checks(PTO2Runtime* rt, int num_sched);
#endif

// Compatibility shim: old single-kernel pto2_submit_task → pto2_submit_mixed_task.
// Only compiled when pto_orchestrator.h (via pto_runtime2.h) has been included first.
#ifdef PTO_ORCHESTRATOR_H
static inline void pto2_submit_task(PTO2OrchestratorState* orch,
    int32_t kernel_id, int worker_type, PTOParam& params) {
    MixedKernels mk;
    if (worker_type == PTO2_WORKER_CUBE)
        mk.aic_kernel_id = kernel_id;
    else
        mk.aiv0_kernel_id = kernel_id;
    pto2_submit_mixed_task(orch, mk, params);
}
#endif

#endif  // TEST_COMMON_H
