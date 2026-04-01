/**
 * test_spmd_multiblock_mix.cpp
 *
 * SPMD Multi-Block MIX backend (PERF_BACKEND=11): MIX tasks (AIC + AIV0 + AIV1)
 * with selectable workload via PERF_CASE_IDX.
 * Included by test_* when PERF_BACKEND=11. Do not compile as separate TU.
 *
 * Case 0 — five MIX tasks with increasing block_num (original graph):
 *   T0: block_num=2  (6 CL)   — basic multi-block MIX
 *   T1: block_num=8  (24 CL)  — saturate one sched thread (8 clusters)
 *   T2: block_num=12 (36 CL)  — cross-thread dispatch via ready_queue re-push
 *   T3: block_num=24 (72 CL)  — occupy all clusters across all 3 sched threads
 *   T4: block_num=48 (144 CL) — two full rounds of all clusters
 *   Output: 282 cache lines = 4512 float32.
 *
 * Case 1 — block_num=1 × 8192 tasks:
 *   8192 × (1 block × 3 CL) = 24576 CL = 393216 float32.
 *
 * Case 2 — block_num=2 × 4096 tasks (same total work as case 1):
 *   4096 × (2 blocks × 3 CL) = 24576 CL.
 *
 * Case 3 — block_num=4 × 2048 tasks:
 *   2048 × (4 blocks × 3 CL) = 24576 CL.
 *
 * Case 4 — block_num=8 × 1024 tasks:
 *   1024 × (8 blocks × 3 CL) = 24576 CL.
 *
 * Cases 1–4: do not pass the full output tensor per task. A parent tensor covers the
 * allocation; each submit uses a constant-sized 1D view (disjoint slices). Scalar
 * base_cl=0; kernel addressing uses Tensor::start_offset within the slice (kernel_spmd_mix).
 *
 * TensorMap hashes by buffer base address only: every view shares the same bucket chain.
 * Without manual_dep, each insert lengthens that chain and lookup does O(k) overlap checks
 * on submit k — orchestrator "lookup" phase instruction count grows ~linearly with task index
 * (and total ~quadratic in N). manual_dep=true skips OverlapMap lookup/insert for these
 * user-partitioned disjoint slices (manual_dep on views; avoids TensorMap bucket chain growth).
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

static constexpr int SPMD_MIX_CASE_COUNT = 5;
static_assert(PERF_CASE_IDX >= 0 && PERF_CASE_IDX < SPMD_MIX_CASE_COUNT,
              "PERF_CASE_IDX out of range");

static constexpr int    SPMD_M_FLOATS_PER_CL  = 16;
static constexpr int    SPMD_SLOTS_PER_BLOCK  = 3;    // AIC + AIV0 + AIV1

#if PERF_CASE_IDX == 0
// (2+8+12+24+48) * 3 = 282 CL
static constexpr size_t SPMD_MIX_TOTAL_CL     = 282;
#elif PERF_CASE_IDX == 1
// 8192 tasks × 1 block × 3 CL = 24576 CL
static constexpr size_t SPMD_MIX_TOTAL_CL     = 24576;
#elif PERF_CASE_IDX == 2
// 4096 tasks × 2 blocks × 3 CL = 24576 CL
static constexpr size_t SPMD_MIX_TOTAL_CL     = 24576;
#elif PERF_CASE_IDX == 3
// 2048 tasks × 4 blocks × 3 CL = 24576 CL
static constexpr size_t SPMD_MIX_TOTAL_CL     = 24576;
#elif PERF_CASE_IDX == 4
// 1024 tasks × 8 blocks × 3 CL = 24576 CL
static constexpr size_t SPMD_MIX_TOTAL_CL     = 24576;
#endif

static constexpr size_t SPMD_MIX_TOTAL_FLOATS = SPMD_MIX_TOTAL_CL * SPMD_M_FLOATS_PER_CL;

struct GraphCtx {
    uint64_t args[10];
};

const SpmdMultiblockMixTestCase PERF_CASES[SPMD_MIX_CASE_COUNT] = {
    { "SPMD Multi-Block MIX (5 tasks: block_num=2,8,12,24,48)" },
    { "SPMD Multi-Block MIX case1: block_num=1 × 8192 tasks (24576 CL)" },
    { "SPMD Multi-Block MIX case2: block_num=2 × 4096 tasks (24576 CL)" },
    { "SPMD Multi-Block MIX case3: block_num=4 × 2048 tasks (24576 CL)" },
    { "SPMD Multi-Block MIX case4: block_num=8 × 1024 tasks (24576 CL)" },
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
    // Parent buffer descriptor; cases 1–4 submit only per-task views (constant slice size).
    Tensor ext_parent = make_tensor_external(host_out, shapes, 1, DataType::FLOAT32);

    int total_tasks = 0;
    PTO2_SCOPE(rt) {
#if PERF_CASE_IDX == 0
        // T0: 2 blocks (6 CL) — basic multi-block MIX
        submit_spmd_mix(rt, FUNC_SPMD_MIX_AIC, FUNC_SPMD_MIX_AIV0, FUNC_SPMD_MIX_AIV1,
                        ext_parent, 2, 0);
        total_tasks++;

        // T1: 8 blocks (24 CL) — saturate one sched thread's clusters
        submit_spmd_mix(rt, FUNC_SPMD_MIX_AIC, FUNC_SPMD_MIX_AIV0, FUNC_SPMD_MIX_AIV1,
                        ext_parent, 8, 6);
        total_tasks++;

        // T2: 12 blocks (36 CL) — cross-thread dispatch via ready_queue re-push
        submit_spmd_mix(rt, FUNC_SPMD_MIX_AIC, FUNC_SPMD_MIX_AIV0, FUNC_SPMD_MIX_AIV1,
                        ext_parent, 12, 30);
        total_tasks++;

        // T3: 24 blocks (72 CL) — occupy all clusters across all 3 sched threads
        submit_spmd_mix(rt, FUNC_SPMD_MIX_AIC, FUNC_SPMD_MIX_AIV0, FUNC_SPMD_MIX_AIV1,
                        ext_parent, 24, 66);
        total_tasks++;

        // T4: 48 blocks (144 CL) — two full rounds of all clusters
        submit_spmd_mix(rt, FUNC_SPMD_MIX_AIC, FUNC_SPMD_MIX_AIV0, FUNC_SPMD_MIX_AIV1,
                        ext_parent, 48, 138);
        total_tasks++;
#elif PERF_CASE_IDX == 1
        constexpr int k_blocks = 1;
        constexpr int k_num_tasks = 8192;
#elif PERF_CASE_IDX == 2
        constexpr int k_blocks = 2;
        constexpr int k_num_tasks = 4096;
#elif PERF_CASE_IDX == 3
        constexpr int k_blocks = 4;
        constexpr int k_num_tasks = 2048;
#elif PERF_CASE_IDX == 4
        constexpr int k_blocks = 8;
        constexpr int k_num_tasks = 1024;
#endif
#if PERF_CASE_IDX >= 1 && PERF_CASE_IDX <= 4
        constexpr uint32_t k_floats_per_task = static_cast<uint32_t>(
            k_blocks * SPMD_SLOTS_PER_BLOCK * SPMD_M_FLOATS_PER_CL);
        uint32_t slice_shape[1] = { k_floats_per_task };
        for (int t = 0; t < k_num_tasks; t++) {
            uint32_t off[1] = { static_cast<uint32_t>(t) * k_floats_per_task };
            Tensor out_view = ext_parent.view(slice_shape, off, true /* manual_dep */);
            submit_spmd_mix(rt, FUNC_SPMD_MIX_AIC, FUNC_SPMD_MIX_AIV0, FUNC_SPMD_MIX_AIV1,
                            out_view, static_cast<int16_t>(k_blocks), 0);
            total_tasks++;
        }
#endif
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
#if PERF_CASE_IDX == 0
    printf("  Output: %zu floats (%zu cache lines, %d slots/block, block_num=2,8,12,24,48)\n",
           SPMD_MIX_TOTAL_FLOATS, SPMD_MIX_TOTAL_CL, SPMD_SLOTS_PER_BLOCK);
#elif PERF_CASE_IDX == 1
    printf("  Output: %zu floats (%zu cache lines): 8192×MIX block_num=1; "
           "constant 3-CL views + manual_dep (flat TensorMap lookup cost)\n",
           SPMD_MIX_TOTAL_FLOATS, SPMD_MIX_TOTAL_CL);
#elif PERF_CASE_IDX == 2
    printf("  Output: %zu floats (%zu cache lines): 4096×MIX block_num=2; "
           "constant 6-CL views + manual_dep (flat TensorMap lookup cost)\n",
           SPMD_MIX_TOTAL_FLOATS, SPMD_MIX_TOTAL_CL);
#elif PERF_CASE_IDX == 3
    printf("  Output: %zu floats (%zu cache lines): 2048×MIX block_num=4; "
           "constant 12-CL views + manual_dep (flat TensorMap lookup cost)\n",
           SPMD_MIX_TOTAL_FLOATS, SPMD_MIX_TOTAL_CL);
#elif PERF_CASE_IDX == 4
    printf("  Output: %zu floats (%zu cache lines): 1024×MIX block_num=8; "
           "constant 24-CL views + manual_dep (flat TensorMap lookup cost)\n",
           SPMD_MIX_TOTAL_FLOATS, SPMD_MIX_TOTAL_CL);
#endif
}

#endif  // PTO2_PROFILING
