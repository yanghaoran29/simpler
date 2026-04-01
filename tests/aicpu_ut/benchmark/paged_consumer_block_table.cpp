/**
 * paged_consumer_block_table.cpp
 *
 * Paged-consumer + block-table backend (PERF_BACKEND=12): declarations + definitions.
 * Included by the perf drivers when PERF_BACKEND=12. Do not compile as a separate TU.
 *
 * Ported from the generated device-orchestration entry to the perf/cases UT
 * convention: GraphCtx + PERF_CASES + build_graph(rt,...) + setup_run + print_config,
 * using the explicit-rt submit shims and make_tensor_external (sim AICore: kernels
 * are not executed, only the dependency graph and scheduling are exercised).
 */

#include "pto_runtime2.h"
#include "test_common.h"
#include "sim_aicore.h"
#include "common/platform_config.h"
#include "cpu_affinity.h"
#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstring>

#ifndef PERF_CASE_IDX
#define PERF_CASE_IDX 0
#endif

static constexpr int PCBT_CASE_COUNT = 1;
static_assert(PERF_CASE_IDX >= 0 && PERF_CASE_IDX < PCBT_CASE_COUNT,
              "PERF_CASE_IDX out of range");

// Sim kernel ids (no real kernel runs under the AICore simulator).
#define FUNC_PAGED_PROJ     0  // AIC: x_page @ w1 -> paged_y window
#define FUNC_PAGED_RMSNORM  1  // AIV: rmsnorm(paged_y window, gamma) -> out window

// Fixed problem shape (matches the original generated orchestration).
static constexpr uint32_t NUM_BLOCKS = 16;
static constexpr uint32_t BLOCK_ROWS = 16;
static constexpr uint32_t X_ROWS     = NUM_BLOCKS * BLOCK_ROWS;  // 256
static constexpr uint32_t X_COLS     = 2048;
static constexpr uint32_t Y_COLS     = 256;
static constexpr uint32_t W1_ROWS    = X_COLS;  // 2048
static constexpr uint32_t W1_COLS    = Y_COLS;  // 256

struct PagedConsumerCase {
    char name[128];
    int  num_blocks;
};

struct GraphCtx {
    uint64_t args[10];
};

const PagedConsumerCase PERF_CASES[PCBT_CASE_COUNT] = {
    { "paged_consumer_block_table (16 blocks, x[256,2048] -> out[256,256])", static_cast<int>(NUM_BLOCKS) },
};

static float g_x_buf[X_ROWS * X_COLS];
static float g_w1_buf[W1_ROWS * W1_COLS];
static float g_gamma_buf[Y_COLS];
static int   g_block_table_buf[NUM_BLOCKS];
static float g_out_buf[X_ROWS * Y_COLS];
static float g_paged_y_buf[X_ROWS * Y_COLS];

void build_graph(PTO2Runtime* rt, uint64_t* args, int arg_count) {
    (void)arg_count;
    void* host_x          = reinterpret_cast<void*>(args[0]);
    void* host_w1         = reinterpret_cast<void*>(args[1]);
    void* host_gamma      = reinterpret_cast<void*>(args[2]);
    int*  host_block_tab  = reinterpret_cast<int*>(args[3]);
    void* host_out        = reinterpret_cast<void*>(args[4]);

    uint32_t x_shapes[2]     = {X_ROWS, X_COLS};
    uint32_t w1_shapes[2]    = {W1_ROWS, W1_COLS};
    uint32_t gamma_shapes[1] = {Y_COLS};
    uint32_t out_shapes[2]   = {X_ROWS, Y_COLS};
    uint32_t y_shapes[2]     = {X_ROWS, Y_COLS};

    Tensor ext_x     = make_tensor_external(host_x,            x_shapes,     2, DataType::FLOAT32);
    Tensor ext_w1    = make_tensor_external(host_w1,           w1_shapes,    2, DataType::FLOAT32);
    Tensor ext_gamma = make_tensor_external(host_gamma,        gamma_shapes, 1, DataType::FLOAT32);
    Tensor ext_out   = make_tensor_external(host_out,          out_shapes,   2, DataType::FLOAT32);
    Tensor paged_y   = make_tensor_external(g_paged_y_buf,     y_shapes,     2, DataType::FLOAT32);

    int total_tasks = 0;
    PTO2_SCOPE(rt) {
        // Producer pass: each block projects x_page -> a [BLOCK_ROWS, Y_COLS]
        // window of paged_y. add_inout registers the task as producer of that
        // region so the consumer pass picks up the dependency.
        for (int ob = 0; ob < static_cast<int>(NUM_BLOCKS); ob++) {
            PTO2_SCOPE(rt) {
                uint32_t m0       = static_cast<uint32_t>(ob * static_cast<int>(BLOCK_ROWS));
                uint32_t x_off[2] = {m0, 0};
                uint32_t x_win[2] = {BLOCK_ROWS, X_COLS};
                Tensor x_page = ext_x.view(x_win, x_off);
                uint32_t y_off[2] = {m0, 0};
                uint32_t y_win[2] = {BLOCK_ROWS, Y_COLS};
                Tensor y_window = paged_y.view(y_win, y_off);

                PTOParam params_proj;
                params_proj.add_input(x_page);
                params_proj.add_input(ext_w1);
                params_proj.add_inout(y_window);
                params_proj.add_scalar(m0);
                pto2_rt_submit_aic_task(rt, FUNC_PAGED_PROJ, params_proj);
                total_tasks++;
            }
        }

        // Consumer pass: block_table[ob] selects which paged_y window to rmsnorm
        // into the matching out window. This is the cross-pass dependency edge.
        for (int ob = 0; ob < static_cast<int>(NUM_BLOCKS); ob++) {
            PTO2_SCOPE(rt) {
                int      page_id   = host_block_tab[ob];
                uint32_t src_row   = static_cast<uint32_t>(page_id * static_cast<int>(BLOCK_ROWS));
                uint32_t out_m0    = static_cast<uint32_t>(ob * static_cast<int>(BLOCK_ROWS));
                uint32_t y_off[2]  = {src_row, 0};
                uint32_t y_win[2]  = {BLOCK_ROWS, Y_COLS};
                Tensor y_src = paged_y.view(y_win, y_off);
                uint32_t out_off[2] = {out_m0, 0};
                uint32_t out_win[2] = {BLOCK_ROWS, Y_COLS};
                Tensor out_window = ext_out.view(out_win, out_off);

                PTOParam params_norm;
                params_norm.add_input(ext_gamma);
                params_norm.add_input(y_src);
                params_norm.add_inout(out_window);
                params_norm.add_scalar(out_m0);
                pto2_rt_submit_aiv_task(rt, FUNC_PAGED_RMSNORM, params_norm);
                total_tasks++;
            }
        }
    }

    pto2_orchestrator_done(rt);
    printf("  Total tasks submitted: %d\n", total_tasks);
}

PTO2Runtime* setup_run(const PagedConsumerCase& tc, GraphCtx& ctx) {
    (void)tc;
    // Identity block table (page_id == ob): consumer[ob] depends on producer[ob].
    for (uint32_t i = 0; i < NUM_BLOCKS; i++) {
        g_block_table_buf[i] = static_cast<int>(i);
    }
    ctx.args[0] = reinterpret_cast<uint64_t>(static_cast<void*>(g_x_buf));
    ctx.args[1] = reinterpret_cast<uint64_t>(static_cast<void*>(g_w1_buf));
    ctx.args[2] = reinterpret_cast<uint64_t>(static_cast<void*>(g_gamma_buf));
    ctx.args[3] = reinterpret_cast<uint64_t>(g_block_table_buf);
    ctx.args[4] = reinterpret_cast<uint64_t>(static_cast<void*>(g_out_buf));
    for (int i = 5; i < 10; i++) ctx.args[i] = 0;
    return make_runtime();
}

#if PTO2_PROFILING

void print_config(const PagedConsumerCase& tc) {
    section_header_100('-', "--- Config ---");
    printf("  Config: %s\n", tc.name);
}

#endif  // PTO2_PROFILING
