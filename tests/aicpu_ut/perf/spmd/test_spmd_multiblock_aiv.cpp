/**
 * test_spmd_multiblock_aiv.cpp
 *
 * SPMD Multi-Block AIV backend (PERF_BACKEND=10): AIV-only tasks with selectable
 * workload via PERF_CASE_IDX.
 * Included by test_* when PERF_BACKEND=10. Do not compile as separate TU.
 *
 * Case 0 — five AIV tasks with increasing block_num (original graph):
 *   T0: block_num=4   — basic multi-block
 *   T1: block_num=16  — saturate one sched thread (8 clusters × 2 AIV)
 *   T2: block_num=24  — cross-thread dispatch via ready_queue re-push
 *   T3: block_num=48  — occupy all AIV cores across all 3 sched threads
 *   T4: block_num=96  — two full rounds of all AIV cores
 *   Output: 188 cache lines = 3008 float32.
 *
 * Case 1 — block_num=1 × 8192 tasks:
 *   8192 × (1 block × 1 CL) = 8192 CL = 131072 float32.
 *
 * Case 2 — block_num=2 × 4096 tasks (same total block work as case 1):
 *   4096 × (2 blocks × 1 CL) = 8192 CL.
 *
 * Case 3 — block_num=4 × 2048 tasks:
 *   2048 × (4 blocks × 1 CL) = 8192 CL.
 *
 * Case 4 — block_num=8 × 1024 tasks:
 *   1024 × (8 blocks × 1 CL) = 8192 CL.
 *
 * Cases 1–4: parent tensor + per-task 1D views (disjoint slices), base_cl=0,
 * manual_dep on views (same TensorMap rationale as test_spmd_multiblock_mix.cpp).
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

static constexpr int SPMD_AIV_CASE_COUNT = 5;
static_assert(PERF_CASE_IDX >= 0 && PERF_CASE_IDX < SPMD_AIV_CASE_COUNT,
              "PERF_CASE_IDX out of range");

static constexpr int SPMD_A_FLOATS_PER_CL   = 16;
static constexpr int SPMD_AIV_SLOTS_PER_BLOCK = 1;    // AIV-only: one subtask per block

#if PERF_CASE_IDX == 0
static constexpr size_t SPMD_AIV_TOTAL_CL     = 188;  // 4+16+24+48+96
#elif PERF_CASE_IDX == 1
// 8192 tasks × 1 block × 1 CL = 8192 CL
static constexpr size_t SPMD_AIV_TOTAL_CL     = 8192;
#elif PERF_CASE_IDX == 2
// 4096 tasks × 2 blocks × 1 CL = 8192 CL
static constexpr size_t SPMD_AIV_TOTAL_CL     = 8192;
#elif PERF_CASE_IDX == 3
// 2048 tasks × 4 blocks × 1 CL = 8192 CL
static constexpr size_t SPMD_AIV_TOTAL_CL     = 8192;
#elif PERF_CASE_IDX == 4
// 1024 tasks × 8 blocks × 1 CL = 8192 CL
static constexpr size_t SPMD_AIV_TOTAL_CL     = 8192;
#endif

static constexpr size_t SPMD_AIV_TOTAL_FLOATS = SPMD_AIV_TOTAL_CL * SPMD_A_FLOATS_PER_CL;

struct GraphCtx {
    uint64_t args[10];
};

const SpmdMultiblockAivTestCase PERF_CASES[SPMD_AIV_CASE_COUNT] = {
    { "SPMD Multi-Block AIV (5 tasks: block_num=4,16,24,48,96)" },
    { "SPMD Multi-Block AIV case1: block_num=1 × 8192 tasks (8192 CL)" },
    { "SPMD Multi-Block AIV case2: block_num=2 × 4096 tasks (8192 CL)" },
    { "SPMD Multi-Block AIV case3: block_num=4 × 2048 tasks (8192 CL)" },
    { "SPMD Multi-Block AIV case4: block_num=8 × 1024 tasks (8192 CL)" },
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
    // Parent buffer descriptor; cases 1–4 submit only per-task views (constant slice size).
    Tensor ext_parent = make_tensor_external(host_out, shapes, 1, DataType::FLOAT32);

    int total_tasks = 0;
    PTO2_SCOPE(rt) {
#if PERF_CASE_IDX == 0
        // T0: 4 blocks — basic multi-block
        submit_spmd_aiv(rt, FUNC_SPMD_WRITE_AIV, ext_parent, 4, 0);
        total_tasks++;

        // T1: 16 blocks — saturate one sched thread's AIV cores (8 clusters × 2 AIV)
        submit_spmd_aiv(rt, FUNC_SPMD_WRITE_AIV, ext_parent, 16, 4);
        total_tasks++;

        // T2: 24 blocks — cross-thread dispatch via ready_queue re-push
        submit_spmd_aiv(rt, FUNC_SPMD_WRITE_AIV, ext_parent, 24, 20);
        total_tasks++;

        // T3: 48 blocks — occupy all AIV cores across all 3 sched threads
        submit_spmd_aiv(rt, FUNC_SPMD_WRITE_AIV, ext_parent, 48, 44);
        total_tasks++;

        // T4: 96 blocks — two full rounds of all AIV cores
        submit_spmd_aiv(rt, FUNC_SPMD_WRITE_AIV, ext_parent, 96, 92);
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
            k_blocks * SPMD_AIV_SLOTS_PER_BLOCK * SPMD_A_FLOATS_PER_CL);
        uint32_t slice_shape[1] = { k_floats_per_task };
        for (int t = 0; t < k_num_tasks; t++) {
            uint32_t off[1] = { static_cast<uint32_t>(t) * k_floats_per_task };
            Tensor out_view = ext_parent.view(slice_shape, off, true /* manual_dep */);
            submit_spmd_aiv(rt, FUNC_SPMD_WRITE_AIV, out_view, static_cast<int16_t>(k_blocks), 0);
            total_tasks++;
        }
#endif
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
#if PERF_CASE_IDX == 0
    printf("  Output: %zu floats (%zu cache lines, block_num=4,16,24,48,96)\n",
           SPMD_AIV_TOTAL_FLOATS, SPMD_AIV_TOTAL_CL);
#elif PERF_CASE_IDX == 1
    printf("  Output: %zu floats (%zu cache lines): 8192×AIV block_num=1; "
           "constant 1-CL views + manual_dep (flat TensorMap lookup cost)\n",
           SPMD_AIV_TOTAL_FLOATS, SPMD_AIV_TOTAL_CL);
#elif PERF_CASE_IDX == 2
    printf("  Output: %zu floats (%zu cache lines): 4096×AIV block_num=2; "
           "constant 2-CL views + manual_dep (flat TensorMap lookup cost)\n",
           SPMD_AIV_TOTAL_FLOATS, SPMD_AIV_TOTAL_CL);
#elif PERF_CASE_IDX == 3
    printf("  Output: %zu floats (%zu cache lines): 2048×AIV block_num=4; "
           "constant 4-CL views + manual_dep (flat TensorMap lookup cost)\n",
           SPMD_AIV_TOTAL_FLOATS, SPMD_AIV_TOTAL_CL);
#elif PERF_CASE_IDX == 4
    printf("  Output: %zu floats (%zu cache lines): 1024×AIV block_num=8; "
           "constant 8-CL views + manual_dep (flat TensorMap lookup cost)\n",
           SPMD_AIV_TOTAL_FLOATS, SPMD_AIV_TOTAL_CL);
#endif
}

#endif  // PTO2_PROFILING
