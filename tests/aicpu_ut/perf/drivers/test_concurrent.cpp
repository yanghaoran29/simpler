/**
 * test_concurrent.cpp
 *
 * Orchestration + Scheduler concurrent — one binary per (backend, case). Backend
 * selected at compile time via PERF_BACKEND (0=linear, 1=pa, 2=batch_pa, 3=degree).
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
#include "sim_swimlane.h"
#include <cstring>

int g_pass = 0;
int g_fail = 0;

int main() {
#if PTO2_PROFILING
    setvbuf(stdout, nullptr, _IOLBF, 0);  // line-buffered so profiling prints appear when piped
#endif
#if PERF_BACKEND == 0
    const LinearTestCase& tc = PERF_CASES[PERF_CASE_IDX];
    int num_sched = get_num_sched_threads();
#if PTO2_PROFILING
    {
        char title_buf[96];
        snprintf(title_buf, sizeof(title_buf), " Linear Chain — %s ", tc.name);
        section_header_100('=', title_buf);
    }
    print_config(tc);
    section_header_100('-', "--- Orchestrator Profiling ---");
#endif
    bind_to_cpu(ORCH_CPU);
    LinearRunCtx ctx;
    PTO2Runtime* rt = setup_run(tc, ctx);
    if (!rt) return 1;
#if PTO2_PROFILING
    orch_timing_begin();
#endif
    build_linear_graph(rt, ctx.args, 10);
#if PTO2_PROFILING
    orch_timing_end();
#endif
#elif PERF_BACKEND == 1
    const PerfTestCase& tc = PERF_CASES[PERF_CASE_IDX];
    int num_sched = get_num_sched_threads();
#if PTO2_PROFILING
    {
        const char* paren = strchr(tc.name, '(');
        int case_len = paren ? static_cast<int>(paren - tc.name) : static_cast<int>(strlen(tc.name));
        while (case_len > 0 && tc.name[case_len - 1] == ' ') case_len--;
        char title_buf[96];
        snprintf(title_buf, sizeof(title_buf), " Paged Attention — %.*s Performance Tests ", case_len, tc.name);
        section_header_100('=', title_buf);
    }
    print_config(tc);
    section_header_100('-', "--- Orchestrator Profiling ---");
#endif
    bind_to_cpu(ORCH_CPU);
    PARunCtx ctx;
    PTO2Runtime* rt = setup_run(tc, ctx);
    if (!rt) return 1;
#if PTO2_PROFILING
    orch_timing_begin();
#endif
    build_paged_attention_graph(rt, ctx.args, 10);
#if PTO2_PROFILING
    orch_timing_end();
#endif
#elif PERF_BACKEND == 2
    const PerfTestCase& tc = PERF_CASES[PERF_CASE_IDX];
    int num_sched = get_num_sched_threads();
#if PTO2_PROFILING
    {
        const char* paren = strchr(tc.name, '(');
        int case_len = paren ? static_cast<int>(paren - tc.name) : static_cast<int>(strlen(tc.name));
        while (case_len > 0 && tc.name[case_len - 1] == ' ') case_len--;
        char title_buf[96];
        snprintf(title_buf, sizeof(title_buf), " Batch Paged Attention — %.*s Performance Tests ", case_len, tc.name);
        section_header_100('=', title_buf);
    }
    print_config(tc);
    section_header_100('-', "--- Orchestrator Profiling ---");
#endif
    bind_to_cpu(ORCH_CPU);
    BatchPARunCtx ctx;
    PTO2Runtime* rt = setup_run(tc, ctx);
    if (!rt) return 1;
#if PTO2_PROFILING
    orch_timing_begin();
#endif
    build_batch_paged_attention_graph(rt, ctx.args, 10);
#if PTO2_PROFILING
    orch_timing_end();
#endif
#elif PERF_BACKEND == 4
    const AlternatingTestCase& tc = PERF_CASES[PERF_CASE_IDX];
    int num_sched = get_num_sched_threads();
#if PTO2_PROFILING
    {
        char title_buf[96];
        snprintf(title_buf, sizeof(title_buf), " Alternating Matmul+Add — %s ", tc.name);
        section_header_100('=', title_buf);
    }
    print_config(tc);
    section_header_100('-', "--- Orchestrator Profiling ---");
#endif
    bind_to_cpu(ORCH_CPU);
    AlternatingRunCtx ctx;
    PTO2Runtime* rt = setup_run(tc, ctx);
    if (!rt) return 1;
#if PTO2_PROFILING
    orch_timing_begin();
#endif
    build_alternating_graph(rt, ctx.args, 10);
#if PTO2_PROFILING
    orch_timing_end();
#endif
#elif PERF_BACKEND == 5
    const BgemmTestCase& tc = PERF_CASES[PERF_CASE_IDX];
    int num_sched = get_num_sched_threads();
#if PTO2_PROFILING
    {
        char title_buf[96];
        snprintf(title_buf, sizeof(title_buf), " Batched GEMM — %s ", tc.name);
        section_header_100('=', title_buf);
    }
    print_config(tc);
    section_header_100('-', "--- Orchestrator Profiling ---");
#endif
    bind_to_cpu(ORCH_CPU);
    BgemmRunCtx ctx;
    PTO2Runtime* rt = setup_run(tc, ctx);
    if (!rt) return 1;
#if PTO2_PROFILING
    orch_timing_begin();
#endif
    build_bgemm_graph(rt, ctx.args, 10);
#if PTO2_PROFILING
    orch_timing_end();
#endif
#elif PERF_BACKEND == 6
    const PerfTestCase& tc = PERF_CASES[PERF_CASE_IDX];
    int num_sched = get_num_sched_threads();
#if PTO2_PROFILING
    {
        const char* paren = strchr(tc.name, '(');
        int case_len = paren ? static_cast<int>(paren - tc.name) : static_cast<int>(strlen(tc.name));
        while (case_len > 0 && tc.name[case_len - 1] == ' ') case_len--;
        char title_buf[96];
        snprintf(title_buf, sizeof(title_buf), " Paged Attention Unroll — %.*s Performance Tests ", case_len, tc.name);
        section_header_100('=', title_buf);
    }
    print_config(tc);
    section_header_100('-', "--- Orchestrator Profiling ---");
#endif
    bind_to_cpu(ORCH_CPU);
    PAUnrollRunCtx ctx;
    PTO2Runtime* rt = setup_run(tc, ctx);
    if (!rt) return 1;
#if PTO2_PROFILING
    orch_timing_begin();
#endif
    build_paged_attention_unroll_graph(rt, ctx.args, 10);
#if PTO2_PROFILING
    orch_timing_end();
#endif
#elif PERF_BACKEND == 7
    const ThroughputTestCase& tc = PERF_CASES[PERF_CASE_IDX];
    int num_sched = get_num_sched_threads();
#if PTO2_PROFILING
    {
        char title_buf[96];
        snprintf(title_buf, sizeof(title_buf), " Max Throughput — %s ", tc.name);
        section_header_100('=', title_buf);
    }
    print_config(tc);
    section_header_100('-', "--- Orchestrator Profiling ---");
#endif
    bind_to_cpu(ORCH_CPU);
    ThroughputRunCtx ctx;
    PTO2Runtime* rt = setup_run(tc, ctx);
    if (!rt) return 1;
#if PTO2_PROFILING
    orch_timing_begin();
#endif
    build_throughput_graph(rt, ctx.args, 10);
#if PTO2_PROFILING
    orch_timing_end();
#endif
#elif PERF_BACKEND == 8
    const LatencyTestCase& tc = PERF_CASES[PERF_CASE_IDX];
    int num_sched = get_num_sched_threads();
#if PTO2_PROFILING
    {
        char title_buf[96];
        snprintf(title_buf, sizeof(title_buf), " Min Latency — %s ", tc.name);
        section_header_100('=', title_buf);
    }
    print_config(tc);
    section_header_100('-', "--- Orchestrator Profiling ---");
#endif
    bind_to_cpu(ORCH_CPU);
    LatencyRunCtx ctx;
    PTO2Runtime* rt = setup_run(tc, ctx);
    if (!rt) return 1;
#if PTO2_PROFILING
    orch_timing_begin();
#endif
    build_latency_graph(rt, ctx.args, 10);
#if PTO2_PROFILING
    orch_timing_end();
#endif
#else
    const DegreeTestCase& tc = PERF_CASES[0];
    int num_sched = get_num_sched_threads();
#if PTO2_PROFILING
    {
        char title_buf[96];
        snprintf(title_buf, sizeof(title_buf), " Degree DAG — %s ", tc.name);
        section_header_100('=', title_buf);
    }
    print_config(tc);
    section_header_100('-', "--- Orchestrator Profiling ---");
#endif
    bind_to_cpu(ORCH_CPU);
    DegRunCtx ctx;
    PTO2Runtime* rt = setup_run(tc, ctx);
    if (!rt) return 1;
#if PTO2_PROFILING
    orch_timing_begin();
#endif
    build_degree_graph(rt, ctx.args, 10);
#if PTO2_PROFILING
    orch_timing_end();
#endif
#endif

#if PTO2_PROFILING
    print_orch_profiling();
#endif

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
    export_sim_swimlane(rt);
    print_cpu_affinity(num_sched);
    section_header_100('-', "--- Scheduler Profiling ---");
    print_sched_profiling(rt);
#endif

    pto2_runtime_destroy(rt);
    return (g_fail == 0) ? 0 : 1;
}
