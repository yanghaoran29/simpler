/**
 * benchmark_bgemm.cpp
 *
 * Batched GEMM benchmark backend (PERF_BACKEND=5): declarations + definitions.
 * Included by test_* when PERF_BACKEND=5. Do not compile as separate TU.
 *
 * Graph topology: per group, GEMM_TILE (CUBE) → TILE_ADD (VECTOR) pairs.
 * Dependencies are created by the intermediate partial-sum tensor P.
 *
 *   A/B layout: [num_groups, grid_k, incore_loop, tile_size, tile_size]
 *   C layout:   [incore_loop * num_groups, tile_size, tile_size]
 *
 * Case 0: task_num=500, tile=128, incore_loop=4,  grid_k=2 → 250 groups, 1000 tasks
 * Case 1: task_num=64,  tile=128, incore_loop=4,  grid_k=2 →  32 groups,  128 tasks
 * Case 2: task_num=256, tile=128, incore_loop=4,  grid_k=2 → 128 groups,  512 tasks
 * Case 3: task_num=64,  tile=128, incore_loop=16, grid_k=2 →  32 groups,  128 tasks
 * Case 4: task_num=64,  tile=128, incore_loop=4,  grid_k=4 →  16 groups,  128 tasks
 */

// ─── Declarations ─────────────────────────────────────────────────────────────

#include "pto_runtime2.h"
#include "test_common.h"
#if defined(PTO2_SIM_AICORE_UT)
#include "sim_aicore.h"
#endif
#include "common/platform_config.h"
#include "cpu_affinity.h"
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <csignal>
#include <unistd.h>

#ifndef PERF_CASE_IDX
#define PERF_CASE_IDX 0
#endif

struct BgemmTestCase {
    const char* name;
    int         tile_size;
    int         grid_k;
    int         num_groups;   // = matmul_add_task_num / grid_k
    int         incore_loop;
};

static constexpr int BGEMM_CASE_COUNT = 5;
static_assert(PERF_CASE_IDX >= 0 && PERF_CASE_IDX < BGEMM_CASE_COUNT,
              "PERF_CASE_IDX out of range");

extern const BgemmTestCase PERF_CASES[BGEMM_CASE_COUNT];

// Buffer sizes covering the largest case.
// Case 0: num_groups=250, grid_k=2, incore_loop=4, tile=128
//   A/B: [250, 2, 4, 128, 128] → 32,768,000 floats
//   C:   [4*250, 128, 128]     → 16,384,000 floats
static constexpr size_t BGEMM_MAX_AB_NELEMS = 250UL * 2 * 4 * 128 * 128;  // 32,768,000
static constexpr size_t BGEMM_MAX_C_NELEMS  =  4UL * 250 * 128 * 128;     // 16,384,000
static constexpr size_t BGEMM_CONFIG_NELEMS = 4;  // [tile_size, grid_k, num_groups, incore_loop]

extern float   g_bgemm_A[BGEMM_MAX_AB_NELEMS];
extern float   g_bgemm_B[BGEMM_MAX_AB_NELEMS];
extern float   g_bgemm_C[BGEMM_MAX_C_NELEMS];
extern int64_t g_bgemm_config[BGEMM_CONFIG_NELEMS];

#define FUNC_GEMM_TILE 0
#define FUNC_TILE_ADD  1

struct BgemmRunCtx {
    int64_t  config[4];   // tile_size, grid_k, num_groups, incore_loop
    uint64_t args[10];
};

int          get_num_sched_threads();
void         perf_wait_sigstop();
void         build_bgemm_graph(PTO2Runtime* rt, uint64_t* args, int arg_count);
PTO2Runtime* setup_run(const BgemmTestCase& tc, BgemmRunCtx& ctx);

#if PTO2_PROFILING
void section_header_100(char pad_char, const char* title);
void print_config(const BgemmTestCase& tc);
void print_cpu_affinity(int num_sched);
#endif

// ─── Definitions ──────────────────────────────────────────────────────────────

const BgemmTestCase PERF_CASES[BGEMM_CASE_COUNT] = {
    { "BGEMM Case0 (task_num=500, tile=128, incore_loop=4,  grid_k=2)",  128, 2, 250, 4  },
    { "BGEMM Case1 (task_num=64,  tile=128, incore_loop=4,  grid_k=2)",  128, 2,  32, 4  },
    { "BGEMM Case2 (task_num=256, tile=128, incore_loop=4,  grid_k=2)",  128, 2, 128, 4  },
    { "BGEMM Case3 (task_num=64,  tile=128, incore_loop=16, grid_k=2)",  128, 2,  32, 16 },
    { "BGEMM Case4 (task_num=64,  tile=128, incore_loop=4,  grid_k=4)",  128, 4,  16, 4  },
};

float   g_bgemm_A[BGEMM_MAX_AB_NELEMS];
float   g_bgemm_B[BGEMM_MAX_AB_NELEMS];
float   g_bgemm_C[BGEMM_MAX_C_NELEMS];
int64_t g_bgemm_config[BGEMM_CONFIG_NELEMS];

int get_num_sched_threads() {
    int n = 3;
    const char* env = std::getenv("AICPU_UT_NUM_SCHED_THREADS");
    if (env && *env) n = std::atoi(env);
    if (n < 1) n = 1;
    if (n > PLATFORM_MAX_AICPU_THREADS) n = PLATFORM_MAX_AICPU_THREADS;
    return n;
}

void perf_wait_sigstop() {
    if (std::getenv("PERF_WAIT_AFTER_INIT")) {
        pid_t pid = getpid();
        printf("  [perf] Init done. PID=%d — attach perf, then send SIGCONT:\n", (int)pid);
        printf("  [perf]   perf record -g -p %d -o perf.data\n", (int)pid);
        printf("  [perf]   kill -CONT %d\n", (int)pid);
        fflush(stdout);
        raise(SIGSTOP);
    }
}

/**
 * build_bgemm_graph — builds a batched tiled-GEMM task graph.
 *
 * Per group (group_idx = 0 .. num_groups-1):
 *   PTO2_SCOPE:
 *     for k_idx in 0 .. grid_k-1:
 *       P ← GEMM_TILE(A_view, B_view)       [CUBE]
 *       C_view += TILE_ADD(C_view, P)        [VECTOR, depends on P]
 *
 * args[0]: ptr_A  args[1]: ptr_B  args[2]: ptr_C
 * args[3]: ptr_config  args[4]: size_A  args[5]: size_B  args[6]: size_C
 */
void build_bgemm_graph(PTO2Runtime* rt, uint64_t* args, int arg_count) {
    (void)arg_count;

    void*    dev_A      = reinterpret_cast<void*>(args[0]);
    void*    dev_B      = reinterpret_cast<void*>(args[1]);
    void*    dev_C      = reinterpret_cast<void*>(args[2]);
    void*    dev_config = reinterpret_cast<void*>(args[3]);
    size_t   size_A     = static_cast<size_t>(args[4]);
    size_t   size_B     = static_cast<size_t>(args[5]);
    size_t   size_C     = static_cast<size_t>(args[6]);

    int64_t* host_config = reinterpret_cast<int64_t*>(args[3]);
    int      tile_size   = static_cast<int>(host_config[0]);
    int      grid_k      = static_cast<int>(host_config[1]);
    int      num_groups  = static_cast<int>(host_config[2]);
    int      incore_loop = static_cast<int>(host_config[3]);
    uint64_t tile_elems  = static_cast<uint64_t>(tile_size) * tile_size;

    printf("  tile_size=%d, grid_k=%d, num_groups=%d, incore_loop=%d\n",
           tile_size, grid_k, num_groups, incore_loop);

    uint64_t A_shapes[1] = {size_A / sizeof(float)};
    uint64_t B_shapes[1] = {size_B / sizeof(float)};
    uint64_t C_shapes[1] = {size_C / sizeof(float)};
    Tensor ext_A = make_tensor_external(dev_A, A_shapes, 1, DataType::FLOAT32);
    Tensor ext_B = make_tensor_external(dev_B, B_shapes, 1, DataType::FLOAT32);
    Tensor ext_C = make_tensor_external(dev_C, C_shapes, 1, DataType::FLOAT32);

    uint64_t config_shapes[1] = {BGEMM_CONFIG_NELEMS};
    Tensor ext_config = make_tensor_external(dev_config, config_shapes, 1, DataType::INT64);

    uint64_t group_tile_elems = static_cast<uint64_t>(incore_loop) * tile_elems;
    uint64_t group_shapes[1]  = {group_tile_elems};

    int total_gemm = 0;
    int total_add  = 0;

    for (int group_idx = 0; group_idx < num_groups; group_idx++) {
        PTO2_SCOPE(rt) {
            uint64_t c_offset      = static_cast<uint64_t>(group_idx) * group_tile_elems;
            uint64_t c_offsets[1]  = {c_offset};
            Tensor C_view = ext_C.view(group_shapes, c_offsets);

            for (int k_idx = 0; k_idx < grid_k; k_idx++) {
                // A/B layout: [num_groups, grid_k, incore_loop, tile_size, tile_size]
                uint64_t ab_offset = (static_cast<uint64_t>(group_idx) * grid_k
                                    + static_cast<uint64_t>(k_idx)) * group_tile_elems;
                uint64_t ab_offsets[1] = {ab_offset};
                Tensor A_view = ext_A.view(group_shapes, ab_offsets);
                Tensor B_view = ext_B.view(group_shapes, ab_offsets);
                Tensor P      = make_tensor(group_shapes, 1, DataType::FLOAT32);

                PTOParam params_gemm[] = {
                    make_input_param(A_view),
                    make_input_param(B_view),
                    make_output_param(P),
                    make_input_param(ext_config),
                };
                pto2_submit_task(rt->orchestrators, FUNC_GEMM_TILE, PTO2_WORKER_CUBE,
                                 params_gemm, 4);
                total_gemm++;

                PTOParam params_add[] = {
                    make_inout_param(C_view),
                    make_input_param(P),
                    make_input_param(ext_config),
                };
                pto2_submit_task(rt->orchestrators, FUNC_TILE_ADD, PTO2_WORKER_VECTOR,
                                 params_add, 3);
                total_add++;
            }
        }
    }

    pto2_orchestrator_done(rt->orchestrators);
    printf("  Total tasks submitted: %d (gemm=%d, add=%d)\n",
           total_gemm + total_add, total_gemm, total_add);
}

PTO2Runtime* setup_run(const BgemmTestCase& tc, BgemmRunCtx& ctx) {
    uint64_t tile_elems       = static_cast<uint64_t>(tc.tile_size) * tc.tile_size;
    uint64_t group_tile_elems = static_cast<uint64_t>(tc.incore_loop) * tile_elems;

    ctx.config[0] = static_cast<int64_t>(tc.tile_size);
    ctx.config[1] = static_cast<int64_t>(tc.grid_k);
    ctx.config[2] = static_cast<int64_t>(tc.num_groups);
    ctx.config[3] = static_cast<int64_t>(tc.incore_loop);

    // Copy config into the device-visible g_bgemm_config buffer.
    for (int i = 0; i < 4; i++) g_bgemm_config[i] = ctx.config[i];

    size_t size_A = static_cast<size_t>(tc.num_groups) * tc.grid_k * group_tile_elems * sizeof(float);
    size_t size_C = static_cast<size_t>(tc.num_groups) * group_tile_elems * sizeof(float);

    ctx.args[0] = reinterpret_cast<uint64_t>(static_cast<void*>(g_bgemm_A));
    ctx.args[1] = reinterpret_cast<uint64_t>(static_cast<void*>(g_bgemm_B));
    ctx.args[2] = reinterpret_cast<uint64_t>(static_cast<void*>(g_bgemm_C));
    ctx.args[3] = reinterpret_cast<uint64_t>(static_cast<void*>(g_bgemm_config));
    ctx.args[4] = static_cast<uint64_t>(size_A);
    ctx.args[5] = static_cast<uint64_t>(size_A);
    ctx.args[6] = static_cast<uint64_t>(size_C);
    for (int i = 7; i < 10; i++) ctx.args[i] = 0;

    return make_runtime();
}

#if PTO2_PROFILING

void section_header_100(char pad_char, const char* title) {
    int len   = static_cast<int>(strlen(title));
    int left  = (100 - len) / 2;
    int right = 100 - len - left;
    for (int i = 0; i < left;  i++) putchar(pad_char);
    printf("%s", title);
    for (int i = 0; i < right; i++) putchar(pad_char);
    putchar('\n');
}

void print_config(const BgemmTestCase& tc) {
    section_header_100('-', "--- Config ---");
    int total_tasks = tc.num_groups * tc.grid_k * 2;  // gemm + add per (group, k)
    printf("  tile_size=%d, grid_k=%d, num_groups=%d, incore_loop=%d, total_tasks=%d\n",
           tc.tile_size, tc.grid_k, tc.num_groups, tc.incore_loop, total_tasks);
}

void print_cpu_affinity(int num_sched) {
    static const int sched_cpus[] = {
        SCHED_CPU0, SCHED_CPU1, SCHED_CPU2, SCHED_CPU3,
        SCHED_CPU4, SCHED_CPU5, SCHED_CPU6, SCHED_CPU7,
    };
    section_header_100('-', "--- CPU affinity ---");
    int orch_core = current_cpu();
    printf("  orchestrator → core %d\n", orch_core >= 0 ? orch_core : ORCH_CPU);
    int max_sched = static_cast<int>(sizeof(sched_cpus) / sizeof(sched_cpus[0]));
    for (int i = 0; i < num_sched && i < max_sched; i++)
        printf("  scheduler[%d]  → core %d (configured)\n", i, sched_cpus[i]);
    printf("\n");
}

#endif  // PTO2_PROFILING
