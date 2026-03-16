/**
 * test_sched_prof_only.cpp
 *
 * Scheduler profiling only — one binary per (backend, case). Backend selected at
 * compile time via PERF_BACKEND (0=linear, 1=pa, 2=batch_pa, 3=degree).
 * Orchestration runs first, then scheduler threads; perf window covers scheduling only.
 *
 * Compile: -DPERF_BACKEND=N -DPERF_CASE_IDX=M
 */

#if PERF_BACKEND == 0
#include "test_linear.cpp"
#elif PERF_BACKEND == 1
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
#elif PERF_CASE_IDX == 1
#include "test_deg4.cpp"
#elif PERF_CASE_IDX == 2
#include "test_deg8.cpp"
#else
#include "test_deg2.cpp"
#endif

int g_pass = 0;
int g_fail = 0;

int main() {
#if PTO2_PROFILING
    setvbuf(stdout, nullptr, _IOLBF, 0);  // line-buffered so profiling prints appear when piped
#if PERF_BACKEND == 0
    printf("\n============================================================\n");
    printf("  Linear Chain — Scheduler Profiling Only\n");
    printf("  (Orchestration first, then Scheduler threads)\n");
    printf("============================================================\n\n");
#elif PERF_BACKEND == 1
    printf("\n============================================================\n");
    printf("  Paged Attention — Scheduler Profiling Only\n");
    printf("  (Orchestration first, then Scheduler threads)\n");
    printf("============================================================\n\n");
#elif PERF_BACKEND == 2
    printf("\n============================================================\n");
    printf("  Batch Paged Attention — Scheduler Profiling Only\n");
    printf("  (Orchestration first, then Scheduler threads)\n");
    printf("============================================================\n\n");
#elif PERF_BACKEND == 4
    printf("\n============================================================\n");
    printf("  Alternating Matmul+Add — Scheduler Profiling Only\n");
    printf("  (Orchestration first, then Scheduler threads)\n");
    printf("============================================================\n\n");
#elif PERF_BACKEND == 5
    printf("\n============================================================\n");
    printf("  Batched GEMM — Scheduler Profiling Only\n");
    printf("  (Orchestration first, then Scheduler threads)\n");
    printf("============================================================\n\n");
#elif PERF_BACKEND == 6
    printf("\n============================================================\n");
    printf("  Paged Attention Unroll — Scheduler Profiling Only\n");
    printf("  (Orchestration first, then Scheduler threads)\n");
    printf("============================================================\n\n");
#elif PERF_BACKEND == 7
    printf("\n============================================================\n");
    printf("  Max Throughput — Scheduler Profiling Only\n");
    printf("  (Orchestration first, then Scheduler threads)\n");
    printf("============================================================\n\n");
#elif PERF_BACKEND == 8
    printf("\n============================================================\n");
    printf("  Min Latency — Scheduler Profiling Only\n");
    printf("  (Orchestration first, then Scheduler threads)\n");
    printf("============================================================\n\n");
#else
    printf("\n============================================================\n");
    printf("  Degree DAG — Scheduler Profiling Only\n");
    printf("  (Orchestration first, then Scheduler threads)\n");
    printf("============================================================\n\n");
#endif
#endif

#if PERF_BACKEND == 0
    const LinearTestCase& tc = PERF_CASES[PERF_CASE_IDX];
    bind_to_cpu(ORCH_CPU);
#if PTO2_PROFILING
    print_config(tc);
#endif
    LinearRunCtx ctx;
    PTO2Runtime* rt = setup_run(tc, ctx);
    if (!rt) return 1;
    build_linear_graph(rt, ctx.args, 10);
#elif PERF_BACKEND == 1
    const PerfTestCase& tc = PERF_CASES[PERF_CASE_IDX];
    bind_to_cpu(ORCH_CPU);
#if PTO2_PROFILING
    print_config(tc);
#endif
    PARunCtx ctx;
    PTO2Runtime* rt = setup_run(tc, ctx);
    if (!rt) return 1;
    build_paged_attention_graph(rt, ctx.args, 10);
#elif PERF_BACKEND == 2
    const PerfTestCase& tc = PERF_CASES[PERF_CASE_IDX];
    bind_to_cpu(ORCH_CPU);
#if PTO2_PROFILING
    print_config(tc);
#endif
    BatchPARunCtx ctx;
    PTO2Runtime* rt = setup_run(tc, ctx);
    if (!rt) return 1;
    build_batch_paged_attention_graph(rt, ctx.args, 10);
#elif PERF_BACKEND == 4
    const AlternatingTestCase& tc = PERF_CASES[PERF_CASE_IDX];
    bind_to_cpu(ORCH_CPU);
#if PTO2_PROFILING
    print_config(tc);
#endif
    AlternatingRunCtx ctx;
    PTO2Runtime* rt = setup_run(tc, ctx);
    if (!rt) return 1;
    build_alternating_graph(rt, ctx.args, 10);
#elif PERF_BACKEND == 5
    const BgemmTestCase& tc = PERF_CASES[PERF_CASE_IDX];
    bind_to_cpu(ORCH_CPU);
#if PTO2_PROFILING
    print_config(tc);
#endif
    BgemmRunCtx ctx;
    PTO2Runtime* rt = setup_run(tc, ctx);
    if (!rt) return 1;
    build_bgemm_graph(rt, ctx.args, 10);
#elif PERF_BACKEND == 6
    const PerfTestCase& tc = PERF_CASES[PERF_CASE_IDX];
    bind_to_cpu(ORCH_CPU);
#if PTO2_PROFILING
    print_config(tc);
#endif
    PAUnrollRunCtx ctx;
    PTO2Runtime* rt = setup_run(tc, ctx);
    if (!rt) return 1;
    build_paged_attention_unroll_graph(rt, ctx.args, 10);
#elif PERF_BACKEND == 7
    const ThroughputTestCase& tc = PERF_CASES[PERF_CASE_IDX];
    bind_to_cpu(ORCH_CPU);
#if PTO2_PROFILING
    print_config(tc);
#endif
    ThroughputRunCtx ctx;
    PTO2Runtime* rt = setup_run(tc, ctx);
    if (!rt) return 1;
    build_throughput_graph(rt, ctx.args, 10);
#elif PERF_BACKEND == 8
    const LatencyTestCase& tc = PERF_CASES[PERF_CASE_IDX];
    bind_to_cpu(ORCH_CPU);
#if PTO2_PROFILING
    print_config(tc);
#endif
    LatencyRunCtx ctx;
    PTO2Runtime* rt = setup_run(tc, ctx);
    if (!rt) return 1;
    build_latency_graph(rt, ctx.args, 10);
#else
    const DegreeTestCase& tc = PERF_CASES[0];
    bind_to_cpu(ORCH_CPU);
#if PTO2_PROFILING
    print_config(tc);
#endif
    DegRunCtx ctx;
    PTO2Runtime* rt = setup_run(tc, ctx);
    if (!rt) return 1;
    build_degree_graph(rt, ctx.args, 10);
#endif

    int num_sched = get_num_sched_threads();
    perf_wait_sigstop();

#if defined(PTO2_SIM_AICORE_UT)
    if (aicpu_sim_run_pto2(rt, num_sched) != 0) {
#if PTO2_PROFILING
        printf("  [ERROR] aicpu_sim_run_pto2 failed\n");
#endif
    }
#else
    sim_run_with_resolve_and_dispatch(rt, num_sched);
#endif

#if PTO2_PROFILING
    print_cpu_affinity(num_sched);
    section_header_100('-', "--- Scheduler Profiling ---");
    printf("\n");
    print_sched_profiling(rt);
    run_sched_checks(rt, num_sched);
#endif

    pto2_runtime_destroy(rt);
    return (g_fail == 0) ? 0 : 1;
}
