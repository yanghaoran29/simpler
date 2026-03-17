/**
 * test_orchestrator.cpp
 *
 * Orchestration only (no scheduler) — one binary per (backend, case). Backend
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
    section_header_100('-', "--- CPU affinity ---");
    int orch_core = current_cpu();
    printf("  orchestrator → core %d\n", orch_core >= 0 ? orch_core : ORCH_CPU);
    printf("\n");
    section_header_100('-', "--- Orchestrator Profiling ---");
#endif

    GraphCtx ctx;
    PTO2Runtime* rt = setup_run(tc, ctx);
    if (!rt) return 1;
    perf_wait_sigstop();

#if PTO2_PROFILING
    orch_timing_begin();
#endif
    build_graph(rt, ctx.args, 10);
#if PTO2_PROFILING
    orch_timing_end();
#endif

#if PTO2_PROFILING
    print_orch_profiling();
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
