/**
 * test_spmd_multiblock_aiv.cpp
 *
 * SPMD Multi-Block AIV backend (PERF_BACKEND=10): five AIV tasks with
 * increasing block_num to exercise multi-block dispatch paths.
 * Included by test_* when PERF_BACKEND=10. Do not compile as separate TU.
 *
 * Replicates spmd_multiblock_aiv orchestration in perf-test format.
 *
 * Graph topology: 5 independent AIV tasks, each writing to a disjoint
 * region of the output tensor via block_idx × base_cl offset.
 *
 *   T0: block_num=4   — basic multi-block
 *   T1: block_num=16  — saturate one sched thread (8 clusters × 2 AIV)
 *   T2: block_num=24  — cross-thread dispatch via ready_queue re-push
 *   T3: block_num=48  — occupy all AIV cores across all 3 sched threads
 *   T4: block_num=96  — two full rounds of all AIV cores
 *
 * Output tensor: 188 cache lines = 3008 float32.
 */

#include "pto_runtime2.h"
#include "test_common.h"
#include "sim_aicore.h"
#include "common/platform_config.h"
#include "cpu_affinity.h"
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>

#ifndef PERF_CASE_IDX
#define PERF_CASE_IDX 0
#endif

#define FUNC_SPMD_WRITE_AIV 0

struct SpmdMultiblockAivTestCase {
    const char* name;
};

static constexpr int SPMD_AIV_CASE_COUNT = 1;
static_assert(PERF_CASE_IDX >= 0 && PERF_CASE_IDX < SPMD_AIV_CASE_COUNT,
              "PERF_CASE_IDX out of range");

static constexpr int    SPMD_A_FLOATS_PER_CL   = 16;
static constexpr size_t SPMD_AIV_TOTAL_CL      = 188;  // 4+16+24+48+96
static constexpr size_t SPMD_AIV_TOTAL_FLOATS  = SPMD_AIV_TOTAL_CL * SPMD_A_FLOATS_PER_CL;  // 3008

struct GraphCtx {
    uint64_t args[10];
};

const SpmdMultiblockAivTestCase PERF_CASES[SPMD_AIV_CASE_COUNT] = {
    { "SPMD Multi-Block AIV (5 tasks: block_num=4,16,24,48,96)" },
};

float g_spmd_aiv_output[SPMD_AIV_TOTAL_FLOATS];

/**
 * submit_spmd_aiv — submits one AIV task with the given block_num.
 *
 * Instrumented with entry/exit asm markers following the same convention
 * as pto2_submit_mixed_task: orr x3 marks the call entry, orr x4 marks
 * the call exit (via RAII destructor).
 */
static void submit_spmd_aiv(PTO2Runtime* rt,
    int32_t kernel_id, Tensor& out, int16_t block_num, int64_t base_cl) {
#if defined(__aarch64__)
    __asm__ __volatile__("orr x3, x3, x3");
    struct SpmdSubmitMarkerScope {
        ~SpmdSubmitMarkerScope() { __asm__ __volatile__("orr x4, x4, x4"); }
    } _marker;
#endif
    MixedKernels mk;
    mk.aiv0_kernel_id = kernel_id;
    Arg args;
    args.add_inout(out);
    args.add_scalar(static_cast<uint64_t>(base_cl));
    args.launch_spec.set_block_num(block_num);
    pto2_submit_mixed_task(rt->orchestrators, mk, args);
}

void build_graph(PTO2Runtime* rt, uint64_t* args, int arg_count) {
    (void)arg_count;

    void* host_out = reinterpret_cast<void*>(args[0]);

    uint32_t shapes[1] = { static_cast<uint32_t>(SPMD_AIV_TOTAL_FLOATS) };
    Tensor ext_output = make_tensor_external(host_out, shapes, 1, DataType::FLOAT32);

    int total_tasks = 0;
    PTO2_SCOPE(rt) {
        // T0: 4 blocks — basic multi-block
        submit_spmd_aiv(rt, FUNC_SPMD_WRITE_AIV, ext_output, 4, 0);
        total_tasks++;

        // T1: 16 blocks — saturate one sched thread's AIV cores (8 clusters × 2 AIV)
        submit_spmd_aiv(rt, FUNC_SPMD_WRITE_AIV, ext_output, 16, 4);
        total_tasks++;

        // T2: 24 blocks — cross-thread dispatch via ready_queue re-push
        submit_spmd_aiv(rt, FUNC_SPMD_WRITE_AIV, ext_output, 24, 20);
        total_tasks++;

        // T3: 48 blocks — occupy all AIV cores across all 3 sched threads
        submit_spmd_aiv(rt, FUNC_SPMD_WRITE_AIV, ext_output, 48, 44);
        total_tasks++;

        // T4: 96 blocks — two full rounds of all AIV cores
        submit_spmd_aiv(rt, FUNC_SPMD_WRITE_AIV, ext_output, 96, 92);
        total_tasks++;
    }

    pto2_orchestrator_done(rt->orchestrators);
    printf("  Total tasks submitted: %d\n", total_tasks);
}

PTO2Runtime* setup_run(const SpmdMultiblockAivTestCase& tc, GraphCtx& ctx) {
    (void)tc;
    ctx.args[0] = reinterpret_cast<uint64_t>(static_cast<void*>(g_spmd_aiv_output));
    for (int i = 1; i < 10; i++) ctx.args[i] = 0;
    return make_runtime();
}

#if PTO2_PROFILING

void print_config(const SpmdMultiblockAivTestCase& tc) {
    section_header_100('-', "--- Config ---");
    printf("  %s\n", tc.name);
    printf("  Output: %zu floats (%zu cache lines, block_num=4,16,24,48,96)\n",
           SPMD_AIV_TOTAL_FLOATS, SPMD_AIV_TOTAL_CL);
}

#endif  // PTO2_PROFILING
