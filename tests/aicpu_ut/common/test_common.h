/**
 * test_common.h
 *
 * Common definitions for orchestration unit tests
 */

#ifndef TEST_COMMON_H
#define TEST_COMMON_H

#include <cstdio>
#include <cstdint>
#include <functional>

// Forward declaration
struct PTO2Runtime;
// Forward declarations for scope API (declared in pto_runtime2.h)
void pto2_rt_scope_begin(PTO2Runtime* rt);
void pto2_rt_scope_end(PTO2Runtime* rt);

// Runtime helpers used by perf test cases.
#include "pto_types.h"
#include "tensor_factory.h"

#if defined(AICPU_UT_DISABLE_PRINTF) && AICPU_UT_DISABLE_PRINTF
#define printf(...) ((int)0)
#endif

// Backward-compat wrapper: PTOParam → Arg with output-address write-back.
struct PTOParam {
    Arg arg_;
    Tensor* output_ptrs_[PTO2_MAX_OUTPUTS];
    TensorCreateInfo create_infos_[PTO2_MAX_OUTPUTS];
    uint32_t output_count_{0};

    void add_input(const Tensor& t)  { arg_.add_input(t); }
    void add_inout(const Tensor& t)  { arg_.add_inout(t); }
    void add_scalar(uint64_t v)      { arg_.add_scalar(v); }
    void add_output(Tensor& t) {
        // Store TensorCreateInfo by value so the pointer passed to arg_.add_output()
        // remains valid until submit. Use effective raw-shapes to avoid reading
        // uninitialized raw_shapes when is_raw_eq_shapes=true.
        create_infos_[output_count_] = TensorCreateInfo(t.get_raw_shapes(), t.ndims, t.dtype, t.manual_dep);
        arg_.add_output(create_infos_[output_count_]);
        output_ptrs_[output_count_++] = &t;
    }
    void _apply_outputs(const TaskOutputTensors& result) {
        if (output_count_ == 0) {
            return;
        }
        const uint32_t n = (result.size() < output_count_) ? result.size() : output_count_;
        for (uint32_t i = 0; i < n; i++) {
            output_ptrs_[i]->buffer = result.get_ref(i).buffer;
        }
    }
};

struct PTO2ScopeGuard {
    explicit PTO2ScopeGuard(PTO2Runtime* rt_) : rt(rt_), active(true), ended(false) { pto2_rt_scope_begin(rt); }
    ~PTO2ScopeGuard() { close(); }
    PTO2ScopeGuard(const PTO2ScopeGuard&) = delete;
    PTO2ScopeGuard& operator=(const PTO2ScopeGuard&) = delete;
    void close() {
        if (!ended) {
            pto2_rt_scope_end(rt);
            ended = true;
        }
    }
    PTO2Runtime* rt;
    bool active;
    bool ended;
};
#define PTO2_SCOPE(rt) \
    for (PTO2ScopeGuard _pto2_scope_guard((rt)); _pto2_scope_guard.active; \
         _pto2_scope_guard.active = false, _pto2_scope_guard.close())

// Global test counters
extern int g_pass;
extern int g_fail;

// Test framework macros. When PTO2_PROFILING=0 (--profiling 0), only pass/fail count is output.
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
#ifdef __cplusplus
extern "C" {
#endif
int aicpu_sim_run_pto2(PTO2Runtime* pto2_rt, int num_sched_threads);
int aicpu_sim_get_actual_sched_cpu(int thread_idx);
#ifdef __cplusplus
}
#endif
int aicpu_sim_run_pto2_concurrent(
    PTO2Runtime* pto2_rt, int num_sched_threads, std::function<void(PTO2Runtime*)> orch_fn);

// Perf backends: each case file defines GraphCtx + one setup_run overload (included by drivers).
struct GraphCtx;
struct LatencyTestCase;
struct PerfTestCase;
struct AlternatingTestCase;
struct BgemmTestCase;
struct ThroughputTestCase;

int get_num_sched_threads();
void perf_wait_sigstop();
void build_graph(PTO2Runtime* rt, uint64_t* args, int arg_count);
PTO2Runtime* setup_run(const LatencyTestCase& tc, GraphCtx& ctx);
PTO2Runtime* setup_run(const PerfTestCase& tc, GraphCtx& ctx);
PTO2Runtime* setup_run(const AlternatingTestCase& tc, GraphCtx& ctx);
PTO2Runtime* setup_run(const BgemmTestCase& tc, GraphCtx& ctx);
PTO2Runtime* setup_run(const ThroughputTestCase& tc, GraphCtx& ctx);

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
// Implemented in sim_run_pto2.cpp (no separate sim header).
void pto2_sim_get_dispatch_counts(int64_t* out, int n);
#endif

void print_sched_profiling(PTO2Runtime* rt);

#if PTO2_PROFILING
extern SchedProfilingData g_sched_prof_data;
#endif

#if PTO2_SCHED_PROFILING
struct PTO2SchedProfilingData;
void pto2_sim_get_accumulated_cycles(uint64_t* out_complete, uint64_t* out_dispatch);
void aicpu_sim_get_saved_sched_prof(int thread_idx, PTO2SchedProfilingData* out);
void pto2_print_sim_sched_summary(SchedProfilingData* data, int64_t tasks_completed, int64_t tasks_consumed);
#endif

#if PTO2_PROFILING
void section_header_100(char pad_char, const char* title);
void print_cpu_affinity(int num_sched, int orch_cpu);
void print_config(const LatencyTestCase& tc);
void print_config(const PerfTestCase& tc);
void print_config(const AlternatingTestCase& tc);
void print_config(const BgemmTestCase& tc);
void print_config(const ThroughputTestCase& tc);
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
    TaskOutputTensors out = pto2_submit_mixed_task(orch, mk, params.arg_);
    params._apply_outputs(out);
}

static inline void pto2_rt_submit_aic_task(PTO2Runtime* rt, int32_t kernel_id, PTOParam& params) {
    pto2_submit_task(rt->orchestrators, kernel_id, PTO2_WORKER_CUBE, params);
}

static inline void pto2_rt_submit_aiv_task(PTO2Runtime* rt, int32_t kernel_id, PTOParam& params) {
    pto2_submit_task(rt->orchestrators, kernel_id, PTO2_WORKER_VECTOR, params);
}
#endif

#endif  // TEST_COMMON_H
