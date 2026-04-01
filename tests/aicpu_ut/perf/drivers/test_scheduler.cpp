/**
 * test_scheduler.cpp
 *
 * Scheduler profiling only — one binary per (backend, case). Backend selected at
 * compile time via PERF_BACKEND (0=linear, 1=pa, 2=batch_pa, 4=alt, 5=bgemm,
 * 6=pa_unroll, 7=throughput, 8=latency). Orchestration runs first, then scheduler
 * threads; perf window covers scheduling only.
 *
 * Compile: -DPERF_BACKEND=N -DPERF_CASE_IDX=M
 */

#include "select_backend_case.inc"

int g_pass = 0;
int g_fail = 0;

int main() {
#if PTO2_PROFILING
    setvbuf(stdout, nullptr, _IOLBF, 0);  // line-buffered so profiling prints appear when piped
    {
        char title_buf[96];
        snprintf(title_buf, sizeof(title_buf), " Performance Tests ");
        section_header_100('=', title_buf);
    }
#endif

    const auto& tc = PERF_CASES[PERF_CASE_IDX];
    bind_to_cpu(ORCH_CPU);

#if PTO2_PROFILING
    print_config(tc);
#endif

    GraphCtx ctx;
    PTO2Runtime* rt = setup_run(tc, ctx);
    if (!rt) return 1;
    build_graph(rt, ctx.args, 10);

    int num_sched = get_num_sched_threads();
    perf_wait_sigstop();

    const int rc_sched = aicpu_sim_run_pto2(rt, num_sched);
#if PTO2_PROFILING
    if (rc_sched != 0) {
        printf("  [ERROR] aicpu_sim_run_pto2 failed\n");
    }
#else
    (void)rc_sched;
#endif

#if PTO2_PROFILING
    print_cpu_affinity(num_sched, current_cpu());
    section_header_100('-', "--- Scheduler Profiling ---");
    printf("\n");
    print_sched_profiling(rt);
    run_sched_checks(rt, num_sched);
#endif

    pto2_runtime_destroy(rt);
    return (g_fail == 0) ? 0 : 1;
}
