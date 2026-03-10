/**
 * test_paged_attention.cpp
 *
 * Paged Attention Orchestration Unit Tests
 *
 * Purpose: Verify without hardware (AICore) dependency:
 *   1. Orchestration function correctly builds task graph
 *   2. Task dependencies are correctly established
 *   3. Scope management is correct
 *   4. Tensor creation and view operations are correct
 *
 * Simulation strategy:
 *   - Use make_runtime() to create simulated runtime
 *   - Use sim_run_all() to simulate task execution (skip AICore kernel execution)
 *   - Only verify task graph construction, not computation results
 *
 * Build:
 *   cd simpler/tests/orchestration_ut && make build && make run
 */

// ─────────────────────────────────────────────────────────────────────────────
// [1] Runtime main header
// ─────────────────────────────────────────────────────────────────────────────
#include "pto_runtime2.h"
#include "test_common.h"
#include "json_cases.h"
#if defined(PTO2_SIM_AICORE_UT)
#include "aicpu_sim_api.h"
#endif
#include "common/platform_config.h"
#include "cpu_affinity.h"
#include <cstring>
#include <algorithm>
#include <cstdint>

// ─── Embedded test cases ──────────────────────────────────────────────────────
// Set PERF_CASE_IDX at compile time to select the case, e.g. -DPERF_CASE_IDX=1
#ifndef PERF_CASE_IDX
#define PERF_CASE_IDX 0
#endif

static const PerfTestCase PERF_CASES[] = {
    {
        "Paged Attention Basic",         // name
        2, 4, 1, 8, 4, 2, 0.125f,       // batch, num_heads, kv_head_num, head_dim, block_size, block_num, scale_value
        {8, 8}, 2,                       // context_lens, context_lens_count
    },
};

static_assert(PERF_CASE_IDX >= 0 && PERF_CASE_IDX < (int)(sizeof(PERF_CASES) / sizeof(PERF_CASES[0])),
              "PERF_CASE_IDX out of range");

// ─── Global data segment buffers (avoid stack/heap for large arrays) ───────────
// Sized for PERF_CASES[]: batch=2, num_heads=4, head_dim=8, block_size=4, block_num=2
static constexpr size_t GLOBAL_QUERY_SIZE       = 2 * 4 * 8 * sizeof(float);
static constexpr size_t GLOBAL_KV_SIZE          = 2 * 2 * 4 * 8 * sizeof(float);
static constexpr size_t GLOBAL_OUT_SIZE         = 2 * 4 * 8 * sizeof(float);
static constexpr size_t GLOBAL_BLOCK_TABLE_CNT  = 2 * 2;
static constexpr size_t GLOBAL_CONTEXT_LENS_CNT = 2;

static float    g_query_buf[GLOBAL_QUERY_SIZE / sizeof(float)];
static float    g_key_cache_buf[GLOBAL_KV_SIZE / sizeof(float)];
static float    g_value_cache_buf[GLOBAL_KV_SIZE / sizeof(float)];
static float    g_out_buf[GLOBAL_OUT_SIZE / sizeof(float)];
static int      g_block_table[GLOBAL_BLOCK_TABLE_CNT];
static int      g_context_lens[GLOBAL_CONTEXT_LENS_CNT];

// ─────────────────────────────────────────────────────────────────────────────
// [2] Paged Attention Orchestration Logic (extracted from paged_attention_orch.cpp)
// ─────────────────────────────────────────────────────────────────────────────

#define FUNC_QK_MATMUL 0
#define FUNC_SOFTMAX_PREPARE 1
#define FUNC_PV_MATMUL 2
#define FUNC_ONLINE_UPDATE 3
#define FUNC_AIC_HUB 4
#define FUNC_AIV_HUB 5

/**
 * Paged Attention orchestration function
 *
 * This is adapted from paged_attention_orch.cpp, but:
 * - Uses PTO2Runtime directly instead of PTO2Runtime* from orchestration API
 * - All Tensor objects must be in a local scope to ensure proper destruction
 */
static void build_paged_attention_graph(PTO2Runtime* rt, uint64_t* args, int arg_count) {
    (void)arg_count;  // Suppress unused warning


    // Extract device pointers
    void* host_query = reinterpret_cast<void*>(args[0]);
    void* host_key_cache = reinterpret_cast<void*>(args[1]);
    void* host_value_cache = reinterpret_cast<void*>(args[2]);
    int* host_block_table = reinterpret_cast<int*>(args[3]);
    int* host_context_lens = reinterpret_cast<int*>(args[4]);
    void* host_out = reinterpret_cast<void*>(args[5]);
    int64_t* host_config = reinterpret_cast<int64_t*>(args[6]);

    // Extract sizes (not used in this test, but kept for compatibility)
    (void)args[7];  // query_size
    (void)args[8];  // key_cache_size
    (void)args[9];  // value_cache_size

    // Extract config parameters
    uint64_t batch = static_cast<uint64_t>(static_cast<int>(host_config[0]));
    uint64_t num_heads = static_cast<uint64_t>(static_cast<int>(host_config[1]));
    (void)host_config[2];  // kv_head_num - not used in this test
    uint64_t head_dim = static_cast<uint64_t>(static_cast<int>(host_config[3]));
    uint64_t block_size = static_cast<uint64_t>(static_cast<int>(host_config[4]));
    uint64_t block_num = static_cast<uint64_t>(static_cast<int>(host_config[5]));
    union {
        uint32_t u;
        float f;
    } scale_conv;
    scale_conv.u = static_cast<uint32_t>(host_config[6]);
    float scale_value = scale_conv.f;
    uint64_t q_head_num = num_heads;
    uint64_t q_tile = std::min(num_heads, 128UL);
    uint64_t q_loop = (q_head_num + q_tile - 1) / q_tile;
    DataType data_type = DataType::FLOAT32;  // Use FLOAT32 for simulation instead of BFLOAT16

    printf("  batch = %lu, num_heads = %lu, head_dim = %lu\n",
           (unsigned long)batch, (unsigned long)num_heads, (unsigned long)head_dim);

    // Create external tensors
    uint64_t query_shapes[2] = {batch * num_heads, head_dim};
    uint64_t key_cache_shapes[2] = {batch * block_num * block_size, head_dim};
    uint64_t value_cache_shapes[2] = {batch * block_num * block_size, head_dim};
    uint64_t out_shapes[2] = {batch * num_heads, head_dim};
    Tensor query = make_tensor_external(host_query, query_shapes, 2, data_type);
    Tensor key_cache = make_tensor_external(host_key_cache, key_cache_shapes, 2, data_type);
    Tensor value_cache = make_tensor_external(host_value_cache, value_cache_shapes, 2, data_type);
    Tensor out = make_tensor_external(host_out, out_shapes, 2, DataType::FLOAT32);

    int total_tasks = 0;

    for (uint64_t b_idx = 0; b_idx < batch; b_idx++) {
        uint64_t cur_seq = host_context_lens[b_idx];
        uint64_t bn_this_batch = (cur_seq + block_size - 1) / block_size;
        for (uint64_t q_idx = 0; q_idx < q_loop; q_idx++) {
            PTO2_SCOPE(rt) {
                uint64_t cur_offset = b_idx * q_head_num + q_idx * q_tile;

                uint64_t oi_shapes[2] = {q_tile, head_dim};
                uint64_t li_shapes[1] = {q_tile};
                uint64_t mi_shapes[1] = {q_tile};
                Tensor oi = make_tensor(oi_shapes, 2, DataType::FLOAT32);
                Tensor li_update = make_tensor(li_shapes, 1, DataType::FLOAT32);
                Tensor mi_update = make_tensor(mi_shapes, 1, DataType::FLOAT32);

                uint64_t qi_shapes[2] = {q_tile, head_dim};
                uint64_t qi_offsets[2] = {cur_offset, 0};
                Tensor qi = query.view(qi_shapes, qi_offsets);
                uint64_t out_view_shapes[2] = {q_tile, head_dim};
                uint64_t out_view_offsets[2] = {cur_offset, 0};
                Tensor out_view = out.view(out_view_shapes, out_view_offsets);

                PTOParam params_inplace[] = {
                    make_output_param(oi),
                    make_output_param(li_update),
                    make_output_param(mi_update),
                };
                pto2_submit_task(&rt->orchestrator, FUNC_AIV_HUB, PTO2_WORKER_VECTOR, params_inplace, 3);
                total_tasks++;

                for (uint64_t bn = 0; bn < bn_this_batch; bn++) {
                    uint64_t cur_block_idx = host_block_table[b_idx * block_num + bn];
                    uint64_t valid_len = std::min(block_size, cur_seq - bn * block_size);

                    uint64_t kv_shapes[2] = {block_size, head_dim};
                    uint64_t kv_offsets[2] = {cur_block_idx * block_size, 0};
                    Tensor kj = key_cache.view(kv_shapes, kv_offsets);
                    Tensor vj = value_cache.view(kv_shapes, kv_offsets);

                    uint64_t sij_shapes[2] = {q_tile, block_size};
                    Tensor sij = make_tensor(sij_shapes, 2, DataType::FLOAT32);
                    Tensor pij_f16 = make_tensor(sij_shapes, 2, data_type);

                    PTOParam params_qk[] = {
                        make_input_param(qi),
                        make_input_param(kj),
                        make_output_param(sij),
                    };
                    pto2_submit_task(&rt->orchestrator, FUNC_QK_MATMUL, PTO2_WORKER_CUBE, params_qk, 3);
                    total_tasks++;

                    uint64_t sij_valid_shapes[2] = {q_tile, valid_len};
                    uint64_t sij_valid_offsets[2] = {0, 0};
                    Tensor sij_valid = sij.view(sij_valid_shapes, sij_valid_offsets);

                    Tensor li = make_tensor(li_shapes, 1, DataType::FLOAT32);
                    Tensor mi = make_tensor(mi_shapes, 1, DataType::FLOAT32);

                    PTOParam params_sf[] = {
                        make_input_param(sij_valid),
                        make_scalar_param(float_to_u64(scale_value)),
                        make_output_param(pij_f16),
                        make_output_param(mi),
                        make_output_param(li),
                    };
                    pto2_submit_task(&rt->orchestrator, FUNC_SOFTMAX_PREPARE, PTO2_WORKER_VECTOR, params_sf, 5);
                    total_tasks++;

                    uint64_t oi_tmp_shapes[2] = {q_tile, head_dim};
                    Tensor oi_tmp = make_tensor(oi_tmp_shapes, 2, DataType::FLOAT32);

                    PTOParam params_pv[] = {
                        make_input_param(pij_f16),
                        make_input_param(vj),
                        make_output_param(oi_tmp),
                    };
                    pto2_submit_task(&rt->orchestrator, FUNC_PV_MATMUL, PTO2_WORKER_CUBE, params_pv, 3);
                    total_tasks++;

                    uint64_t is_first = (bn == 0) ? 1 : 0;
                    uint64_t is_last = (bn == bn_this_batch - 1) ? 1 : 0;

                    PTOParam params_up[] = {
                        make_input_param(mi),
                        make_input_param(li),
                        make_input_param(oi_tmp),
                        make_inout_param(mi_update),
                        make_inout_param(li_update),
                        make_inout_param(oi),
                        make_output_param(out_view),
                        make_scalar_param(is_first),
                        make_scalar_param(is_last),
                    };
                    pto2_submit_task(&rt->orchestrator, FUNC_ONLINE_UPDATE, PTO2_WORKER_VECTOR, params_up, 9);
                    total_tasks++;
                }
            }  // PTO2_SCOPE end
        }
    }

    pto2_orchestrator_done(&rt->orchestrator);

    printf("  Total tasks submitted: %d\n", total_tasks);
}

// ─────────────────────────────────────────────────────────────────────────────
// [3] Performance test function
// ─────────────────────────────────────────────────────────────────────────────

static void run_paged_attention_case(const PerfTestCase& tc) {
    TEST_BEGIN(tc.name);

    bind_to_cpu(ORCH_CPU);
    printf("  CPU affinity: orchestrator → core %d\n", ORCH_CPU);

    uint64_t batch      = static_cast<uint64_t>(tc.batch);
    uint64_t num_heads  = static_cast<uint64_t>(tc.num_heads);
    int kv_head_num     = tc.kv_head_num;
    uint64_t head_dim   = static_cast<uint64_t>(tc.head_dim);
    uint64_t block_size = static_cast<uint64_t>(tc.block_size);
    uint64_t block_num  = static_cast<uint64_t>(tc.block_num);
    float scale_value   = tc.scale_value;

    printf("  Config: batch=%lu, num_heads=%lu, head_dim=%lu, block_size=%lu, block_num=%lu\n",
           (unsigned long)batch, (unsigned long)num_heads, (unsigned long)head_dim,
           (unsigned long)block_size, (unsigned long)block_num);
    printf("  context_lens: [");
    for (int i = 0; i < tc.context_lens_count; i++)
        printf("%d%s", tc.context_lens[i], i + 1 < tc.context_lens_count ? ", " : "");
    printf("]\n");

    const size_t query_size       = batch * num_heads * head_dim * sizeof(float);
    const size_t key_cache_size   = batch * block_num * block_size * head_dim * sizeof(float);
    const size_t value_cache_size = batch * block_num * block_size * head_dim * sizeof(float);
    (void)(batch * num_heads * head_dim * sizeof(float));  // out_size unused

    void* query_buf       = static_cast<void*>(g_query_buf);
    void* key_cache_buf   = static_cast<void*>(g_key_cache_buf);
    void* value_cache_buf = static_cast<void*>(g_value_cache_buf);
    void* out_buf         = static_cast<void*>(g_out_buf);
    int*  block_table     = g_block_table;
    int*  context_lens    = g_context_lens;

    for (uint64_t i = 0; i < batch; i++) {
        context_lens[i] = (i < (uint64_t)tc.context_lens_count) ? tc.context_lens[i]
                                                                  : (int)(block_size * block_num);
        for (uint64_t j = 0; j < block_num; j++)
            block_table[i * block_num + j] = static_cast<int>(i * block_num + j);
    }

    int64_t config[7] = {
        static_cast<int64_t>(batch), static_cast<int64_t>(num_heads),
        static_cast<int64_t>(kv_head_num), static_cast<int64_t>(head_dim),
        static_cast<int64_t>(block_size), static_cast<int64_t>(block_num), 0
    };
    union { uint32_t u; float f; } scale_conv;
    scale_conv.f = scale_value;
    config[6] = static_cast<int64_t>(scale_conv.u);

    uint64_t args[10] = {
        reinterpret_cast<uint64_t>(query_buf),
        reinterpret_cast<uint64_t>(key_cache_buf),
        reinterpret_cast<uint64_t>(value_cache_buf),
        reinterpret_cast<uint64_t>(block_table),
        reinterpret_cast<uint64_t>(context_lens),
        reinterpret_cast<uint64_t>(out_buf),
        reinterpret_cast<uint64_t>(config),
        query_size, key_cache_size, value_cache_size
    };

    PTO2Runtime* rt = make_runtime();
    if (!rt) { printf("  [ERROR] make_runtime() failed\n"); return; }

    {  // Tensor scope
        build_paged_attention_graph(rt, args, 10);

#if PTO2_PROFILING
        print_orch_profiling();
#endif

#if defined(PTO2_SIM_AICORE_UT)
        if (aicpu_sim_run_pto2(rt, 3) != 0)
            printf("  [ERROR] aicpu_sim_run_pto2 failed\n");
#else
        sim_run_with_resolve_and_dispatch(rt, 3);
#endif

#if PTO2_PROFILING
        print_sched_profiling(rt);
#endif
    }

    pto2_runtime_destroy(rt);

    // Performance data summary
    // Output removed
}

// ─────────────────────────────────────────────────────────────────────────────
// Main — case selected at compile time via -DPERF_CASE_IDX=N (default 0)
// ─────────────────────────────────────────────────────────────────────────────

int g_pass = 0;
int g_fail = 0;

int main() {
    printf("============================================================\n");
    printf("Paged Attention Performance Tests\n");
    printf("============================================================\n");

    run_paged_attention_case(PERF_CASES[PERF_CASE_IDX]);

    printf("\n============================================================\n");
    printf("Performance Tests Complete\n");
    printf("============================================================\n");

    return (g_fail == 0) ? 0 : 1;
}
