/**
 * test_orchestrator_scheduler.cpp
 *
 * Orchestration + Scheduler concurrent — one binary per (backend, case). Backend
 * selected at compile time via PERF_BACKEND (0=linear, 1=pa, 2=batch_pa, 4=alt,
 * 5=bgemm, 6=pa_unroll, 7=throughput, 8=latency).
 *
 * Compile: -DPERF_BACKEND=N -DPERF_CASE_IDX=M
 */

#include "select_backend_case.inc"
#include "sim_swimlane.h"
#include "sim_aicore.h"
#include <cstring>
#include <cstdlib>
#include <functional>

// Instruction-count marker macros (mirrors aicpu_executor.cpp).
#define PTO2_SPECIAL_INS_PLAIN   0
#define PTO2_SPECIAL_INS_MEMORY  1
#ifndef PTO2_INSTR_COUNT_BUILD_GRAPH_ENABLE
#define PTO2_INSTR_COUNT_BUILD_GRAPH_ENABLE 0
#endif
#if defined(__aarch64__)
#define _PTO2_MARKER_ORR_SELF_ASM(n) "orr x" #n ", x" #n ", x" #n
#define _PTO2_SI_CAT(a, b)           a##b
#define _PTO2_SI_DISPATCH(reg, m)    _PTO2_SI_CAT(_PTO2_SI_DO_, m)(reg)
#define _PTO2_SI_DO_0(reg)           __asm__ __volatile__(_PTO2_MARKER_ORR_SELF_ASM(reg))
#define _PTO2_SI_DO_1(reg)           __asm__ __volatile__(_PTO2_MARKER_ORR_SELF_ASM(reg) ::: "memory")
#define _PTO2_SI_2(reg, m)           _PTO2_SI_DISPATCH(reg, m)
#define _PTO2_SI_3(reg, m, flag)     do { if (flag) { _PTO2_SI_DISPATCH(reg, m); } } while(0)
#define _PTO2_SI_SEL(_1, _2, _3, NAME, ...) NAME
#define PTO2_SPECIAL_INSTRUCTION(...) \
    _PTO2_SI_SEL(__VA_ARGS__, _PTO2_SI_3, _PTO2_SI_2, ~)(__VA_ARGS__)
#else
#define PTO2_SPECIAL_INSTRUCTION(...) ((void)0)
#endif

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
    const int rc_concurrent = aicpu_sim_run_pto2_concurrent(rt, num_sched, [&ctx, &orch_actual_cpu](PTO2Runtime* r) {
        bind_to_cpu(ORCH_CPU);
        orch_actual_cpu = current_cpu();
        PTO2_SPECIAL_INSTRUCTION(17, PTO2_SPECIAL_INS_MEMORY, PTO2_INSTR_COUNT_BUILD_GRAPH_ENABLE); /* QEMU insn_count: start (orr x17) enc 0xaa110231 — distinct from submit_mixed_task markers */
        build_graph(r, ctx.args, 10);
        PTO2_SPECIAL_INSTRUCTION(18, PTO2_SPECIAL_INS_MEMORY, PTO2_INSTR_COUNT_BUILD_GRAPH_ENABLE);
    });
#if PTO2_PROFILING
    if (rc_concurrent != 0) {
        printf("  [ERROR] aicpu_sim_run_pto2_concurrent failed\n");
    }
#else
    (void)rc_concurrent;
#endif

#if PTO2_PROFILING
    export_sim_swimlane(rt);
    print_cpu_affinity(num_sched, orch_actual_cpu);
    section_header_100('-', "--- Scheduler Profiling ---");
    print_sched_profiling(rt);
    run_sched_checks(rt, num_sched);
#endif

    pto2_runtime_destroy(rt);
    // Avoid unsafe global-simulator teardown order at process exit.
    fflush(stdout);
    fflush(stderr);
    _Exit((g_fail == 0) ? 0 : 1);
}
