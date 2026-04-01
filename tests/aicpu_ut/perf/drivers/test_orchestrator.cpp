/**
 * test_orchestrator.cpp
 *
 * Orchestration only (no scheduler) — one binary per (backend, case). Backend
 * selected at compile time via PERF_BACKEND (0=linear, 1=pa, 2=batch_pa, 4=alt,
 * 5=bgemm, 6=pa_unroll, 7=throughput, 8=latency). Orchestration runs in a
 * dedicated thread bound to ORCH_CPU, matching the concurrent execution model.
 *
 * Compile: -DPERF_BACKEND=N -DPERF_CASE_IDX=M
 */

#include "select_backend_case.inc"
#include <thread>

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

#if PTO2_PROFILING
    print_config(tc);
    section_header_100('-', "--- Orchestrator Profiling ---");
#endif

    GraphCtx ctx;
    PTO2Runtime* rt = setup_run(tc, ctx);
    if (!rt) return 1;

    int orch_actual_cpu = -1;
#if PTO2_PROFILING
    uint64_t orch_t0 = perf_now_us();
#endif
    std::thread orch_thread([&ctx, rt, &orch_actual_cpu]() {
        bind_to_cpu(ORCH_CPU);
        orch_actual_cpu = current_cpu();
        perf_wait_sigstop();
        build_graph(rt, ctx.args, 10);
    });
    orch_thread.join();
#if PTO2_PROFILING
    uint64_t orch_elapsed_us = perf_now_us() - orch_t0;
    /* Format matched by format_profiling_output.py (ORCH_TIME_RE) and parse_one in all_in_one.sh */
    printf("  Thread 0: aicpu_orchestration_entry returned, cost %.3fus (orch_idx=0)\n",
           static_cast<double>(orch_elapsed_us));
#endif

#if PTO2_PROFILING
    section_header_100('-', "--- CPU affinity ---");
    printf("  orchestrator → core %d\n", orch_actual_cpu >= 0 ? orch_actual_cpu : ORCH_CPU);
    printf("\n");
    if (!getenv("AICPU_UT_NO_CHECK")) {
        int32_t submitted = 0;
        if (rt->sm_handle && rt->sm_handle->header)
            for (int ri = 0; ri < PTO2_MAX_RING_DEPTH; ri++)
                submitted += rt->sm_handle->header->rings[ri].fc.current_task_index.load(std::memory_order_acquire);
        if (submitted == 0) {
            printf("  FAIL (P1): orchestration submitted 0 tasks\n");
            g_fail++;
        } else {
            g_pass++;
        }
    }
    printf("  Orchestration-only run finished (no scheduler threads).\n");
#endif

    pto2_runtime_destroy(rt);
    return (g_fail == 0) ? 0 : 1;
}
