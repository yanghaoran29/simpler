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
// Forward declarations for scope API (declared in pto_runtime2.h)
void pto2_rt_scope_begin(PTO2Runtime* rt);
void pto2_rt_scope_end(PTO2Runtime* rt);

// Runtime helpers used by perf test cases.
#include "tensor.h"
#include "data_type.h"

static inline Tensor make_tensor_external(void* base, const uint32_t* shapes, uint32_t ndims, DataType dtype) {
    uint32_t raw[RUNTIME_MAX_TENSOR_DIMS] = {};
    uint32_t off[RUNTIME_MAX_TENSOR_DIMS] = {};
    uint64_t elems = 1;
    for (uint32_t i = 0; i < ndims; i++) {
        raw[i] = shapes[i];
        off[i] = 0;
        elems *= static_cast<uint64_t>(shapes[i]);
    }
    uint64_t bytes = elems * get_element_size(dtype);
    Tensor t;
    t.init(base, bytes, raw, shapes, off, ndims, dtype, /*version*/0,
           /*is_all_offset_zero*/true, /*is_raw_eq_shapes*/true);
    return t;
}

static inline Tensor make_tensor(const uint32_t* shapes, uint32_t ndims, DataType dtype) {
    return make_tensor_external(nullptr, shapes, ndims, dtype);
}

struct PTO2ScopeGuard {
    explicit PTO2ScopeGuard(PTO2Runtime* rt_) : rt(rt_), active(true) { pto2_rt_scope_begin(rt); }
    ~PTO2ScopeGuard() { if (active) pto2_rt_scope_end(rt); }
    PTO2ScopeGuard(const PTO2ScopeGuard&) = delete;
    PTO2ScopeGuard& operator=(const PTO2ScopeGuard&) = delete;
    PTO2Runtime* rt;
    bool active;
};
#define PTO2_SCOPE(rt) for (PTO2ScopeGuard _pto2_scope_guard((rt)); _pto2_scope_guard.active; _pto2_scope_guard.active = false)

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

static inline void pto2_rt_submit_aic_task(PTO2Runtime* rt, int32_t kernel_id, PTOParam& params) {
    pto2_submit_task(rt->orchestrators, kernel_id, PTO2_WORKER_CUBE, params);
}

static inline void pto2_rt_submit_aiv_task(PTO2Runtime* rt, int32_t kernel_id, PTOParam& params) {
    pto2_submit_task(rt->orchestrators, kernel_id, PTO2_WORKER_VECTOR, params);
}
#endif

#endif  // TEST_COMMON_H
