/**
 * test_spmd_basic.cpp
 *
 * SPMD Basic backend (PERF_BACKEND=9): single MIX task (AIC + AIV0 + AIV1).
 * Included by test_* when PERF_BACKEND=9. Do not compile as separate TU.
 *
 * Replicates spmd_basic orchestration in perf-test format.
 *
 * Graph topology: one MIX task writing to a shared external output tensor.
 * Each subtask (AIC, AIV0, AIV1) writes to its own cache-line slot.
 *
 * Output tensor: 3 cache lines = 48 float32.
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

#define FUNC_SPMD_READ_AIC  0
#define FUNC_SPMD_READ_AIV0 1
#define FUNC_SPMD_READ_AIV1 2

struct SpmdBasicTestCase {
    const char* name;
};

static constexpr int SPMD_BASIC_CASE_COUNT = 1;
static_assert(PERF_CASE_IDX >= 0 && PERF_CASE_IDX < SPMD_BASIC_CASE_COUNT,
              "PERF_CASE_IDX out of range");

static constexpr int    SPMD_B_FLOATS_PER_CL   = 16;
static constexpr size_t SPMD_BASIC_TOTAL_FLOATS = 3 * SPMD_B_FLOATS_PER_CL;  // 48

struct GraphCtx {
    uint64_t args[10];
};

const SpmdBasicTestCase PERF_CASES[SPMD_BASIC_CASE_COUNT] = {
    { "SPMD Basic (1 MIX task: AIC + AIV0 + AIV1)" },
};

float g_spmd_basic_output[SPMD_BASIC_TOTAL_FLOATS];

/**
 * submit_spmd_mix_basic — submits one MIX task (AIC + AIV0 + AIV1).
 *
 * Instrumented with entry/exit asm markers following the same convention
 * as pto2_submit_mixed_task: orr x3 marks the call entry, orr x4 marks
 * the call exit (via RAII destructor).
 */
static void submit_spmd_mix_basic(PTO2Runtime* rt,
    int32_t aic_id, int32_t aiv0_id, int32_t aiv1_id, Tensor& out) {
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
    pto2_submit_mixed_task(rt->orchestrators, mk, args);
}

void build_graph(PTO2Runtime* rt, uint64_t* args, int arg_count) {
    (void)arg_count;

    void* host_out = reinterpret_cast<void*>(args[0]);

    uint32_t shapes[1] = { static_cast<uint32_t>(SPMD_BASIC_TOTAL_FLOATS) };
    Tensor ext_output = make_tensor_external(host_out, shapes, 1, DataType::FLOAT32);

    int total_tasks = 0;
    PTO2_SCOPE(rt) {
        submit_spmd_mix_basic(rt,
            FUNC_SPMD_READ_AIC, FUNC_SPMD_READ_AIV0, FUNC_SPMD_READ_AIV1,
            ext_output);
        total_tasks++;
    }

    pto2_orchestrator_done(rt->orchestrators);
    printf("  Total tasks submitted: %d\n", total_tasks);
}

PTO2Runtime* setup_run(const SpmdBasicTestCase& tc, GraphCtx& ctx) {
    (void)tc;
    ctx.args[0] = reinterpret_cast<uint64_t>(static_cast<void*>(g_spmd_basic_output));
    for (int i = 1; i < 10; i++) ctx.args[i] = 0;
    return make_runtime();
}

#if PTO2_PROFILING

void print_config(const SpmdBasicTestCase& tc) {
    section_header_100('-', "--- Config ---");
    printf("  %s\n", tc.name);
    printf("  Output: %zu floats (%d cache lines, 1 per subtask slot)\n",
           SPMD_BASIC_TOTAL_FLOATS, 3);
}

#endif  // PTO2_PROFILING
