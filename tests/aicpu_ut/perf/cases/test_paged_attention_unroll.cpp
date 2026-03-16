/**
 * test_paged_attention_unroll.cpp
 *
 * Paged Attention Unroll backend (PERF_BACKEND=6): declarations + definitions.
 * Included by test_* when PERF_BACKEND=6. Do not compile as separate TU.
 *
 * Unrolled variant of paged attention: N_UNROLL=8 KV blocks are fused into
 * one group of 4 tasks instead of 5 tasks per block (as in the basic variant).
 * Per (b_idx, q_idx) scope:
 *   1. AIV_HUB:         initialise accumulators oi, li_update, mi_update
 *   2. QK_MATMUL (×G):  qi @ K^T for N_UNROLL blocks → sij_buf         [CUBE]
 *   3. SOFTMAX  (×G):   two-pass over sij_buf → pij_buf, mi, li         [VECTOR]
 *   4. PV_MATMUL (×G):  pij_buf @ V for N_UNROLL blocks → oi_new        [CUBE]
 *   5. ONLINE_UPDATE(×G): accumulate mi/li/oi_new into mi_update/li_update/oi [VECTOR]
 *
 *   G = ceil(bn_this_batch / N_UNROLL)
 *
 * Case 0: batch=2, num_heads=16, head_dim=128, block_size=128, block_num=256
 * Case 1: batch=2, num_heads=64, head_dim=128, block_size=128, block_num=256
 * Case 2: batch=2, num_heads=64, head_dim=256, block_size=64,  block_num=512
 */

// ─── Declarations ─────────────────────────────────────────────────────────────

#include "pto_runtime2.h"
#include "test_common.h"
#include "json_cases.h"
#if defined(PTO2_SIM_AICORE_UT)
#include "sim_aicore.h"
#endif
#include "common/platform_config.h"
#include "cpu_affinity.h"
#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <csignal>
#include <unistd.h>

#ifndef PERF_CASE_IDX
#define PERF_CASE_IDX 0
#endif

static constexpr int PAU_CASE_COUNT = 3;
static_assert(PERF_CASE_IDX >= 0 && PERF_CASE_IDX < PAU_CASE_COUNT,
              "PERF_CASE_IDX out of range");

extern const PerfTestCase PERF_CASES[PAU_CASE_COUNT];

// N_UNROLL: KV blocks fused per task group (must match the device kernels).
static constexpr int N_UNROLL = 8;

// Buffer sizing: covers the largest case across all three cases.
// Case 2 (heads=64, dim=256, blocks=512) is largest.
static constexpr size_t PAU_MAX_BATCH      = 2;
static constexpr size_t PAU_MAX_NUM_HEADS  = 64;
static constexpr size_t PAU_MAX_HEAD_DIM   = 256;
static constexpr size_t PAU_MAX_BLOCK_NUM  = 512;
static constexpr size_t PAU_MAX_BLOCK_SIZE = 128;

static constexpr size_t PAU_QUERY_NELEMS    = PAU_MAX_BATCH * PAU_MAX_NUM_HEADS * PAU_MAX_HEAD_DIM;
static constexpr size_t PAU_KV_NELEMS       = PAU_MAX_BATCH * PAU_MAX_BLOCK_NUM
                                            * PAU_MAX_BLOCK_SIZE * PAU_MAX_HEAD_DIM;
static constexpr size_t PAU_BLOCK_TABLE_CNT = PAU_MAX_BATCH * PAU_MAX_BLOCK_NUM;

extern float g_pau_query[PAU_QUERY_NELEMS];
extern float g_pau_key_cache[PAU_KV_NELEMS];
extern float g_pau_value_cache[PAU_KV_NELEMS];
extern float g_pau_out[PAU_QUERY_NELEMS];
extern int   g_pau_block_table[PAU_BLOCK_TABLE_CNT];
extern int   g_pau_context_lens[PAU_MAX_BATCH];

#define FUNC_QK_MATMUL       0
#define FUNC_SOFTMAX_PREPARE 1
#define FUNC_PV_MATMUL       2
#define FUNC_ONLINE_UPDATE   3
#define FUNC_AIC_HUB         4
#define FUNC_AIV_HUB         5

struct PAUnrollRunCtx {
    int64_t  config[7];
    uint64_t args[10];
};

int          get_num_sched_threads();
void         perf_wait_sigstop();
void         build_paged_attention_unroll_graph(PTO2Runtime* rt, uint64_t* args, int arg_count);
PTO2Runtime* setup_run(const PerfTestCase& tc, PAUnrollRunCtx& ctx);

#if PTO2_PROFILING
void section_header_100(char pad_char, const char* title);
void print_config(const PerfTestCase& tc);
void print_cpu_affinity(int num_sched);
#endif

// ─── Definitions ──────────────────────────────────────────────────────────────

// context_lens[0] is used for all batches; see setup_run.
const PerfTestCase PERF_CASES[PAU_CASE_COUNT] = {
    { "PAUnroll-0 (batch=2, heads=16, dim=128, bs=128, bn=256, ctx=8192)",
      2, 16, 1, 128, 128, 256, 1.0f, {8192}, 1 },
    { "PAUnroll-1 (batch=2, heads=64, dim=128, bs=128, bn=256, ctx=8192)",
      2, 64, 1, 128, 128, 256, 1.0f, {8192}, 1 },
    { "PAUnroll-2 (batch=2, heads=64, dim=256, bs=64,  bn=512, ctx=8192)",
      2, 64, 1, 256, 64,  512, 1.0f, {8192}, 1 },
};

float g_pau_query[PAU_QUERY_NELEMS];
float g_pau_key_cache[PAU_KV_NELEMS];
float g_pau_value_cache[PAU_KV_NELEMS];
float g_pau_out[PAU_QUERY_NELEMS];
int   g_pau_block_table[PAU_BLOCK_TABLE_CNT];
int   g_pau_context_lens[PAU_MAX_BATCH];

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
 * build_paged_attention_unroll_graph — builds the unrolled paged attention graph.
 *
 * Compared to the basic paged attention, N_UNROLL=8 blocks are batched into a
 * single group of 4 tasks, reducing task-submission overhead by ~8×.
 *
 * Per (b_idx, q_idx) scope:
 *   AIV_HUB (init)
 *   for bn in 0 .. bn_this_batch step N_UNROLL:
 *     n_blocks = min(N_UNROLL, bn_this_batch - bn)
 *     QK_MATMUL  (qi @ K^T for n_blocks blocks)
 *     SOFTMAX_PREPARE (sij_buf → pij_buf, mi, li)
 *     PV_MATMUL  (pij_buf @ V for n_blocks blocks)
 *     ONLINE_UPDATE (accumulate into oi, li_update, mi_update)
 */
void build_paged_attention_unroll_graph(PTO2Runtime* rt, uint64_t* args, int arg_count) {
    (void)arg_count;

    void*    host_query        = reinterpret_cast<void*>(args[0]);
    void*    host_key_cache    = reinterpret_cast<void*>(args[1]);
    void*    host_value_cache  = reinterpret_cast<void*>(args[2]);
    int*     host_block_table  = reinterpret_cast<int*>(args[3]);
    int*     host_context_lens = reinterpret_cast<int*>(args[4]);
    void*    host_out          = reinterpret_cast<void*>(args[5]);
    int64_t* host_config       = reinterpret_cast<int64_t*>(args[6]);

    uint64_t batch      = static_cast<uint64_t>(static_cast<int>(host_config[0]));
    uint64_t num_heads  = static_cast<uint64_t>(static_cast<int>(host_config[1]));
    uint64_t head_dim   = static_cast<uint64_t>(static_cast<int>(host_config[3]));
    uint64_t block_size = static_cast<uint64_t>(static_cast<int>(host_config[4]));
    uint64_t block_num  = static_cast<uint64_t>(static_cast<int>(host_config[5]));
    union { uint32_t u; float f; } scale_conv;
    scale_conv.u = static_cast<uint32_t>(host_config[6]);
    float scale_value = scale_conv.f;

    uint64_t q_head_num = num_heads;
    uint64_t q_tile     = std::min(num_heads, static_cast<uint64_t>(128));
    uint64_t q_loop     = (q_head_num + q_tile - 1) / q_tile;
    DataType data_type  = DataType::FLOAT32;

    printf("  batch=%lu, num_heads=%lu, head_dim=%lu, block_size=%lu, block_num=%lu\n",
           (unsigned long)batch, (unsigned long)num_heads, (unsigned long)head_dim,
           (unsigned long)block_size, (unsigned long)block_num);

    uint32_t query_shapes[2]       = {static_cast<uint32_t>(batch * num_heads), static_cast<uint32_t>(head_dim)};
    uint32_t key_cache_shapes[2]   = {static_cast<uint32_t>(batch * block_num * block_size), static_cast<uint32_t>(head_dim)};
    uint32_t value_cache_shapes[2] = {static_cast<uint32_t>(batch * block_num * block_size), static_cast<uint32_t>(head_dim)};
    uint32_t out_shapes[2]         = {static_cast<uint32_t>(batch * num_heads), static_cast<uint32_t>(head_dim)};
    Tensor query       = make_tensor_external(host_query,       query_shapes,       2, data_type);
    Tensor key_cache   = make_tensor_external(host_key_cache,   key_cache_shapes,   2, data_type);
    Tensor value_cache = make_tensor_external(host_value_cache, value_cache_shapes, 2, data_type);
    Tensor out         = make_tensor_external(host_out,         out_shapes,         2, DataType::FLOAT32);

    int total_tasks = 0;

    for (uint64_t b_idx = 0; b_idx < batch; b_idx++) {
        uint64_t cur_seq       = static_cast<uint64_t>(host_context_lens[b_idx]);
        uint64_t bn_this_batch = (cur_seq + block_size - 1) / block_size;

        for (uint64_t q_idx = 0; q_idx < q_loop; q_idx++) {
            PTO2_SCOPE(rt) {
                uint64_t cur_offset = b_idx * q_head_num + q_idx * q_tile;

                uint32_t oi_shapes[2]  = {static_cast<uint32_t>(q_tile), static_cast<uint32_t>(head_dim)};
                uint32_t li_shapes[1]  = {static_cast<uint32_t>(q_tile)};
                uint32_t mi_shapes[1]  = {static_cast<uint32_t>(q_tile)};
                Tensor oi        = make_tensor(oi_shapes, 2, DataType::FLOAT32);
                Tensor li_update = make_tensor(li_shapes, 1, DataType::FLOAT32);
                Tensor mi_update = make_tensor(mi_shapes, 1, DataType::FLOAT32);

                uint32_t qi_shapes[2]     = {static_cast<uint32_t>(q_tile), static_cast<uint32_t>(head_dim)};
                uint32_t qi_offsets[2]    = {static_cast<uint32_t>(cur_offset), 0};
                uint32_t out_v_shapes[2]  = {static_cast<uint32_t>(q_tile), static_cast<uint32_t>(head_dim)};
                uint32_t out_v_offsets[2] = {static_cast<uint32_t>(cur_offset), 0};
                Tensor qi       = query.view(qi_shapes, qi_offsets);
                Tensor out_view = out.view(out_v_shapes, out_v_offsets);

                PTOParam params_hub[] = {
                    make_output_param(oi),
                    make_output_param(li_update),
                    make_output_param(mi_update),
                };
                pto2_submit_task(rt->orchestrators, FUNC_AIV_HUB, PTO2_WORKER_VECTOR,
                                 params_hub, 3);
                total_tasks++;

                for (uint64_t bn = 0; bn < bn_this_batch; bn += N_UNROLL) {
                    uint64_t n_blocks = std::min(static_cast<uint64_t>(N_UNROLL),
                                                 bn_this_batch - bn);

                    // Collect physical block indices for this group.
                    uint64_t block_indices[N_UNROLL] = {};
                    for (uint64_t i = 0; i < n_blocks; i++)
                        block_indices[i] = static_cast<uint64_t>(
                            host_block_table[b_idx * block_num + bn + i]);

                    uint64_t last_block_start = (bn + n_blocks - 1) * block_size;
                    uint64_t valid_len_last   = std::min(block_size, cur_seq - last_block_start);

                    // Task 1: Batched QK matmul for n_blocks blocks.
                    uint32_t sij_shapes[2] = {static_cast<uint32_t>(q_tile), static_cast<uint32_t>(n_blocks * block_size)};
                    Tensor sij_buf = make_tensor(sij_shapes, 2, DataType::FLOAT32);
                    PTOParam params_qk[] = {
                        make_input_param(qi),
                        make_input_param(key_cache),
                        make_output_param(sij_buf),
                        make_scalar_param(n_blocks),
                        make_scalar_param(block_indices[0]),
                        make_scalar_param(block_indices[1]),
                        make_scalar_param(block_indices[2]),
                        make_scalar_param(block_indices[3]),
                        make_scalar_param(block_indices[4]),
                        make_scalar_param(block_indices[5]),
                        make_scalar_param(block_indices[6]),
                        make_scalar_param(block_indices[7]),
                    };
                    pto2_submit_task(rt->orchestrators, FUNC_QK_MATMUL, PTO2_WORKER_CUBE,
                                     params_qk, 12);
                    total_tasks++;

                    // Task 2: Two-pass softmax over sij_buf.
                    uint32_t pij_shapes[2] = {static_cast<uint32_t>(q_tile), static_cast<uint32_t>(n_blocks * block_size)};
                    Tensor pij_buf = make_tensor(pij_shapes, 2, data_type);
                    Tensor mi      = make_tensor(mi_shapes, 1, DataType::FLOAT32);
                    Tensor li      = make_tensor(li_shapes, 1, DataType::FLOAT32);
                    PTOParam params_sf[] = {
                        make_input_param(sij_buf),
                        make_scalar_param(float_to_u64(scale_value)),
                        make_output_param(pij_buf),
                        make_output_param(mi),
                        make_output_param(li),
                        make_scalar_param(n_blocks),
                        make_scalar_param(valid_len_last),
                    };
                    pto2_submit_task(rt->orchestrators, FUNC_SOFTMAX_PREPARE, PTO2_WORKER_VECTOR,
                                     params_sf, 7);
                    total_tasks++;

                    // Task 3: SplitK PV matmul — pij @ V for n_blocks blocks.
                    uint32_t oi_new_shapes[2] = {static_cast<uint32_t>(q_tile), static_cast<uint32_t>(head_dim)};
                    Tensor oi_new = make_tensor(oi_new_shapes, 2, DataType::FLOAT32);
                    PTOParam params_pv[] = {
                        make_input_param(pij_buf),
                        make_input_param(value_cache),
                        make_output_param(oi_new),
                        make_scalar_param(n_blocks),
                        make_scalar_param(block_indices[0]),
                        make_scalar_param(block_indices[1]),
                        make_scalar_param(block_indices[2]),
                        make_scalar_param(block_indices[3]),
                        make_scalar_param(block_indices[4]),
                        make_scalar_param(block_indices[5]),
                        make_scalar_param(block_indices[6]),
                        make_scalar_param(block_indices[7]),
                    };
                    pto2_submit_task(rt->orchestrators, FUNC_PV_MATMUL, PTO2_WORKER_CUBE,
                                     params_pv, 12);
                    total_tasks++;

                    // Task 4: Online softmax accumulation.
                    uint64_t is_first = (bn == 0) ? 1 : 0;
                    uint64_t is_last  = (bn + n_blocks >= bn_this_batch) ? 1 : 0;
                    PTOParam params_up[] = {
                        make_input_param(mi),
                        make_input_param(li),
                        make_input_param(oi_new),
                        make_inout_param(mi_update),
                        make_inout_param(li_update),
                        make_inout_param(oi),
                        make_output_param(out_view),
                        make_scalar_param(is_first),
                        make_scalar_param(is_last),
                    };
                    pto2_submit_task(rt->orchestrators, FUNC_ONLINE_UPDATE, PTO2_WORKER_VECTOR,
                                     params_up, 9);
                    total_tasks++;
                }
            }
        }
    }

    pto2_orchestrator_done(rt->orchestrators);
    printf("  Total tasks submitted: %d\n", total_tasks);
}

PTO2Runtime* setup_run(const PerfTestCase& tc, PAUnrollRunCtx& ctx) {
    uint64_t batch      = static_cast<uint64_t>(tc.batch);
    uint64_t num_heads  = static_cast<uint64_t>(tc.num_heads);
    int      kv_head_num = tc.kv_head_num;
    uint64_t head_dim   = static_cast<uint64_t>(tc.head_dim);
    uint64_t block_size = static_cast<uint64_t>(tc.block_size);
    uint64_t block_num  = static_cast<uint64_t>(tc.block_num);
    float    scale_value = tc.scale_value;

    for (uint64_t i = 0; i < batch; i++) {
        g_pau_context_lens[i] = (i < static_cast<uint64_t>(tc.context_lens_count))
                              ? tc.context_lens[i]
                              : static_cast<int>(block_size * block_num);
        for (uint64_t j = 0; j < block_num; j++)
            g_pau_block_table[i * block_num + j] = static_cast<int>(i * block_num + j);
    }

    ctx.config[0] = static_cast<int64_t>(batch);
    ctx.config[1] = static_cast<int64_t>(num_heads);
    ctx.config[2] = static_cast<int64_t>(kv_head_num);
    ctx.config[3] = static_cast<int64_t>(head_dim);
    ctx.config[4] = static_cast<int64_t>(block_size);
    ctx.config[5] = static_cast<int64_t>(block_num);
    union { uint32_t u; float f; } scale_conv;
    scale_conv.f = scale_value;
    ctx.config[6] = static_cast<int64_t>(scale_conv.u);

    const size_t query_size    = batch * num_heads * head_dim * sizeof(float);
    const size_t kv_cache_size = batch * block_num * block_size * head_dim * sizeof(float);

    ctx.args[0] = reinterpret_cast<uint64_t>(static_cast<void*>(g_pau_query));
    ctx.args[1] = reinterpret_cast<uint64_t>(static_cast<void*>(g_pau_key_cache));
    ctx.args[2] = reinterpret_cast<uint64_t>(static_cast<void*>(g_pau_value_cache));
    ctx.args[3] = reinterpret_cast<uint64_t>(g_pau_block_table);
    ctx.args[4] = reinterpret_cast<uint64_t>(g_pau_context_lens);
    ctx.args[5] = reinterpret_cast<uint64_t>(static_cast<void*>(g_pau_out));
    ctx.args[6] = reinterpret_cast<uint64_t>(ctx.config);
    ctx.args[7] = static_cast<uint64_t>(query_size);
    ctx.args[8] = static_cast<uint64_t>(kv_cache_size);
    ctx.args[9] = static_cast<uint64_t>(kv_cache_size);

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

void print_config(const PerfTestCase& tc) {
    section_header_100('-', "--- Config ---");
    uint64_t bn_per_req   = (tc.context_lens_count > 0)
                          ? static_cast<uint64_t>((tc.context_lens[0] + tc.block_size - 1)
                                                  / tc.block_size)
                          : static_cast<uint64_t>(tc.block_num);
    uint64_t groups       = (bn_per_req + N_UNROLL - 1) / N_UNROLL;
    uint64_t q_tile       = std::min(static_cast<uint64_t>(tc.num_heads),
                                     static_cast<uint64_t>(128));
    uint64_t q_loop       = (static_cast<uint64_t>(tc.num_heads) + q_tile - 1) / q_tile;
    int total_tasks = static_cast<int>(tc.batch) * static_cast<int>(q_loop)
                    * (1 + static_cast<int>(groups) * 4);
    printf("  batch=%d, num_heads=%d, head_dim=%d, block_size=%d, block_num=%d\n",
           tc.batch, tc.num_heads, tc.head_dim, tc.block_size, tc.block_num);
    printf("  N_UNROLL=%d, groups_per_scope=%lu, expected_total_tasks=%d\n",
           N_UNROLL, (unsigned long)groups, total_tasks);
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
