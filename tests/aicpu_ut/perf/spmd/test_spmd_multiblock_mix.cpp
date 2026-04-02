/**
 * test_spmd_multiblock_mix.cpp
 *
 * SPMD Multi-Block MIX backend (PERF_BACKEND=11): five MIX tasks
 * (AIC + AIV0 + AIV1) with increasing block_num.
 * Included by test_* when PERF_BACKEND=11. Do not compile as separate TU.
 *
 * Replicates spmd_multiblock_mix orchestration in perf-test format.
 *
 * Graph topology: 5 independent MIX tasks, each block occupying 3 cache
 * lines (AIC, AIV0, AIV1) at base_cl + block_idx * 3 in the output tensor.
 *
 *   T0: block_num=2  (6 CL)   — basic multi-block MIX
 *   T1: block_num=8  (24 CL)  — saturate one sched thread (8 clusters)
 *   T2: block_num=12 (36 CL)  — cross-thread dispatch via ready_queue re-push
 *   T3: block_num=24 (72 CL)  — occupy all clusters across all 3 sched threads
 *   T4: block_num=48 (144 CL) — two full rounds of all clusters
 *
 * Output tensor: 282 cache lines = 4512 float32.
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

#define FUNC_SPMD_MIX_AIC  0
#define FUNC_SPMD_MIX_AIV0 1
#define FUNC_SPMD_MIX_AIV1 2

struct SpmdMultiblockMixTestCase {
    const char* name;
};

static constexpr int SPMD_MIX_CASE_COUNT = 1;
static_assert(PERF_CASE_IDX >= 0 && PERF_CASE_IDX < SPMD_MIX_CASE_COUNT,
              "PERF_CASE_IDX out of range");

static constexpr int    SPMD_M_FLOATS_PER_CL  = 16;
static constexpr int    SPMD_SLOTS_PER_BLOCK  = 3;    // AIC + AIV0 + AIV1
// (2+8+12+24+48) * 3 = 282 CL
static constexpr size_t SPMD_MIX_TOTAL_CL     = 282;
static constexpr size_t SPMD_MIX_TOTAL_FLOATS = SPMD_MIX_TOTAL_CL * SPMD_M_FLOATS_PER_CL;  // 4512

struct GraphCtx {
    uint64_t args[10];
};

const SpmdMultiblockMixTestCase PERF_CASES[SPMD_MIX_CASE_COUNT] = {
    { "SPMD Multi-Block MIX (5 tasks: block_num=2,8,12,24,48)" },
};

float g_spmd_mix_output[SPMD_MIX_TOTAL_FLOATS];

/**
 * submit_spmd_mix — submits one MIX task (AIC + AIV0 + AIV1) with the
 * given block_num and base_cl offset.
 *
 * Instrumented with entry/exit asm markers following the same convention
 * as pto2_submit_mixed_task: orr x3 marks the call entry, orr x4 marks
 * the call exit (via RAII destructor).
 */
static void submit_spmd_mix(PTO2Runtime* rt,
    int32_t aic_id, int32_t aiv0_id, int32_t aiv1_id,
    Tensor& out, int16_t block_num, int64_t base_cl) {
#if defined(__aarch64__)
    __asm__ __volatile__("orr x3, x3, x3");
    struct SpmdSubmitMarkerScope {
        ~SpmdSubmitMarkerScope() { __asm__ __volatile__("orr x4, x4, x4"); }
    } _marker;
#endif
    MixedKernels mk;
    mk.aic_kernel_id  = aic_id;
    mk.aiv0_kernel_id = aiv0_id;
    mk.aiv1_kernel_id = aiv1_id;
    Arg args;
    args.add_inout(out);
    args.add_scalar(static_cast<uint64_t>(base_cl));
    args.launch_spec.set_block_num(block_num);
    pto2_submit_mixed_task(rt->orchestrators, mk, args);
}

void build_graph(PTO2Runtime* rt, uint64_t* args, int arg_count) {
    (void)arg_count;

    void* host_out = reinterpret_cast<void*>(args[0]);

    uint32_t shapes[1] = { static_cast<uint32_t>(SPMD_MIX_TOTAL_FLOATS) };
    Tensor ext_output = make_tensor_external(host_out, shapes, 1, DataType::FLOAT32);

    int total_tasks = 0;
    PTO2_SCOPE(rt) {
        // T0: 2 blocks (6 CL) — basic multi-block MIX
        submit_spmd_mix(rt, FUNC_SPMD_MIX_AIC, FUNC_SPMD_MIX_AIV0, FUNC_SPMD_MIX_AIV1,
                        ext_output, 2, 0);
        total_tasks++;

        // T1: 8 blocks (24 CL) — saturate one sched thread's clusters
        submit_spmd_mix(rt, FUNC_SPMD_MIX_AIC, FUNC_SPMD_MIX_AIV0, FUNC_SPMD_MIX_AIV1,
                        ext_output, 8, 6);
        total_tasks++;

        // T2: 12 blocks (36 CL) — cross-thread dispatch via ready_queue re-push
        submit_spmd_mix(rt, FUNC_SPMD_MIX_AIC, FUNC_SPMD_MIX_AIV0, FUNC_SPMD_MIX_AIV1,
                        ext_output, 12, 30);
        total_tasks++;

        // T3: 24 blocks (72 CL) — occupy all clusters across all 3 sched threads
        submit_spmd_mix(rt, FUNC_SPMD_MIX_AIC, FUNC_SPMD_MIX_AIV0, FUNC_SPMD_MIX_AIV1,
                        ext_output, 24, 66);
        total_tasks++;

        // T4: 48 blocks (144 CL) — two full rounds of all clusters
        submit_spmd_mix(rt, FUNC_SPMD_MIX_AIC, FUNC_SPMD_MIX_AIV0, FUNC_SPMD_MIX_AIV1,
                        ext_output, 48, 138);
        total_tasks++;
    }

    pto2_orchestrator_done(rt->orchestrators);
    printf("  Total tasks submitted: %d\n", total_tasks);
}

PTO2Runtime* setup_run(const SpmdMultiblockMixTestCase& tc, GraphCtx& ctx) {
    (void)tc;
    ctx.args[0] = reinterpret_cast<uint64_t>(static_cast<void*>(g_spmd_mix_output));
    for (int i = 1; i < 10; i++) ctx.args[i] = 0;
    return make_runtime();
}

#if PTO2_PROFILING

void print_config(const SpmdMultiblockMixTestCase& tc) {
    section_header_100('-', "--- Config ---");
    printf("  %s\n", tc.name);
    printf("  Output: %zu floats (%zu cache lines, %d slots/block, block_num=2,8,12,24,48)\n",
           SPMD_MIX_TOTAL_FLOATS, SPMD_MIX_TOTAL_CL, SPMD_SLOTS_PER_BLOCK);
}

#endif  // PTO2_PROFILING
