/**
 * test_orchestrator_scheduler.cpp
 *
 * Orchestration + Scheduler concurrent — one binary per (backend, case). Backend
 * selected at compile time via PERF_BACKEND (0=linear, 1=pa, 2=batch_pa, 4=alt,
 * 5=bgemm, 6=pa_unroll, 7=throughput, 8=latency).
 *
 * Compile: -DPERF_BACKEND=N -DPERF_CASE_IDX=M
 */

#if PERF_BACKEND == 1
#include "test_paged_attention.cpp"
#elif PERF_BACKEND == 2
#include "test_batch_paged_attention.cpp"
#elif PERF_BACKEND == 4
#include "test_alternating_matmul_add.cpp"
#elif PERF_BACKEND == 5
#include "test_benchmark_bgemm.cpp"
#elif PERF_BACKEND == 6
#include "test_paged_attention_unroll.cpp"
#elif PERF_BACKEND == 7
#include "test_throughput.cpp"
#elif PERF_BACKEND == 8
#include "test_latency.cpp"
#endif
#include "sim_swimlane.h"
#include "sim_aicore.h"
#include <cstring>
#include <functional>

int g_pass = 0;
int g_fail = 0;

int main() {
#if PTO2_PROFILING
    setvbuf(stdout, nullptr, _IOLBF, 0);  // line-buffered so profiling prints appear when piped
#endif

    const auto& tc = PERF_CASES[PERF_CASE_IDX];
    int num_sched = get_num_sched_threads();

#if PTO2_PROFILING
    {
        char title_buf[96];
        snprintf(title_buf, sizeof(title_buf), " Performance Tests ");
        section_header_100('=', title_buf);
    }
    print_config(tc);
    section_header_100('-', "--- Orchestrator Profiling ---");
#endif

    GraphCtx ctx;
    PTO2Runtime* rt = setup_run(tc, ctx);
    if (!rt) return 1;

    int orch_actual_cpu = -1;
    if (aicpu_sim_run_pto2_concurrent(rt, num_sched, [&ctx, &orch_actual_cpu](PTO2Runtime* r) {
            bind_to_cpu(ORCH_CPU);
            orch_actual_cpu = current_cpu();
#if PTO2_PROFILING
            orch_timing_begin();
#endif
            build_graph(r, ctx.args, 10);
#if PTO2_PROFILING
            orch_timing_end();
#endif
        }) != 0) {
#if PTO2_PROFILING
        printf("  [ERROR] aicpu_sim_run_pto2_concurrent failed\n");
#endif
    }

#if PTO2_PROFILING
    print_orch_profiling();
    export_sim_swimlane(rt);
    print_cpu_affinity(num_sched, orch_actual_cpu);
    section_header_100('-', "--- Scheduler Profiling ---");
    print_sched_profiling(rt);
    run_sched_checks(rt, num_sched);
#endif

    pto2_runtime_destroy(rt);
    return (g_fail == 0) ? 0 : 1;
}
