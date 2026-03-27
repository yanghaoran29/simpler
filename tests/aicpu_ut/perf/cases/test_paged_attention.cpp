/**
 * paged_attention.cpp
 *
 * Paged Attention backend (PERF_BACKEND=1): declarations + definitions.
 * Included by test_* when PERF_BACKEND=1. Do not compile as separate TU.
 */

// ─── Declarations (formerly paged_attention.h) ─────────────────────────────────

#include "pto_runtime2.h"
#include "test_common.h"
#include "json_cases.h"
#include "sim_aicore.h"
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

static constexpr int PA_CASE_COUNT = 1;
static_assert(PERF_CASE_IDX >= 0 && PERF_CASE_IDX < PA_CASE_COUNT,
              "PERF_CASE_IDX out of range");

static constexpr size_t PA_QUERY_NELEMS     = 2 * 4 * 8;
static constexpr size_t PA_KV_NELEMS        = 2 * 2 * 4 * 8;
static constexpr size_t PA_BLOCK_TABLE_CNT  = 2 * 2;
static constexpr size_t PA_CONTEXT_LENS_CNT = 2;

#define FUNC_QK_MATMUL       0
#define FUNC_SOFTMAX_PREPARE 1
#define FUNC_PV_MATMUL       2
#define FUNC_ONLINE_UPDATE   3
#define FUNC_AIC_HUB         4
#define FUNC_AIV_HUB         5

struct GraphCtx {
    int64_t  config[7];
    uint64_t args[10];
};

// ─── Definitions ─────────────────────────────────────────────────────────────

const PerfTestCase PERF_CASES[PA_CASE_COUNT] = {
    { "Paged Attention Basic", 2, 4, 1, 8, 4, 2, 0.125f, {8, 8}, 2 },
};

float g_query_buf[PA_QUERY_NELEMS];
float g_key_cache_buf[PA_KV_NELEMS];
float g_value_cache_buf[PA_KV_NELEMS];
float g_out_buf[PA_QUERY_NELEMS];
int   g_block_table[PA_BLOCK_TABLE_CNT];
int   g_context_lens[PA_CONTEXT_LENS_CNT];

void build_graph(PTO2Runtime* rt, uint64_t* args, int arg_count) {
    (void)arg_count;

    void*    host_query        = reinterpret_cast<void*>(args[0]);
    void*    host_key_cache    = reinterpret_cast<void*>(args[1]);
    void*    host_value_cache  = reinterpret_cast<void*>(args[2]);
    int*     host_block_table  = reinterpret_cast<int*>(args[3]);
    int*     host_context_lens = reinterpret_cast<int*>(args[4]);
    void*    host_out          = reinterpret_cast<void*>(args[5]);
    int64_t* host_config       = reinterpret_cast<int64_t*>(args[6]);

    (void)args[7];
    (void)args[8];
    (void)args[9];

    uint64_t batch     = static_cast<uint64_t>(static_cast<int>(host_config[0]));
    uint64_t num_heads = static_cast<uint64_t>(static_cast<int>(host_config[1]));
    (void)host_config[2];
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

    printf("  batch = %lu, num_heads = %lu, head_dim = %lu\n",
           (unsigned long)batch, (unsigned long)num_heads, (unsigned long)head_dim);

    uint32_t query_shapes[2]       = {static_cast<uint32_t>(batch * num_heads), static_cast<uint32_t>(head_dim)};
    uint32_t key_cache_shapes[2]   = {static_cast<uint32_t>(batch * block_num * block_size), static_cast<uint32_t>(head_dim)};
    uint32_t value_cache_shapes[2] = {static_cast<uint32_t>(batch * block_num * block_size), static_cast<uint32_t>(head_dim)};
    uint32_t out_shapes[2]         = {static_cast<uint32_t>(batch * num_heads), static_cast<uint32_t>(head_dim)};
    Tensor query       = make_tensor_external(host_query,       query_shapes,       2, data_type);
    Tensor key_cache   = make_tensor_external(host_key_cache,   key_cache_shapes,   2, data_type);
    Tensor value_cache = make_tensor_external(host_value_cache, value_cache_shapes, 2, data_type);
    Tensor out         = make_tensor_external(host_out,           out_shapes,         2, DataType::FLOAT32);

    int total_tasks = 0;

    for (uint64_t b_idx = 0; b_idx < batch; b_idx++) {
        uint64_t cur_seq       = static_cast<uint64_t>(host_context_lens[b_idx]);
        uint64_t bn_this_batch = (cur_seq + block_size - 1) / block_size;

        for (uint64_t q_idx = 0; q_idx < q_loop; q_idx++) {
            PTO2_SCOPE(rt) {
                uint64_t cur_offset = b_idx * q_head_num + q_idx * q_tile;

                uint32_t oi_shapes[2] = {static_cast<uint32_t>(q_tile), static_cast<uint32_t>(head_dim)};
                uint32_t li_shapes[1] = {static_cast<uint32_t>(q_tile)};
                uint32_t mi_shapes[1] = {static_cast<uint32_t>(q_tile)};
                Tensor oi        = make_tensor(oi_shapes, 2, DataType::FLOAT32);
                Tensor li_update = make_tensor(li_shapes, 1, DataType::FLOAT32);
                Tensor mi_update = make_tensor(mi_shapes, 1, DataType::FLOAT32);

                uint32_t qi_shapes[2]      = {static_cast<uint32_t>(q_tile), static_cast<uint32_t>(head_dim)};
                uint32_t qi_offsets[2]     = {static_cast<uint32_t>(cur_offset), 0};
                uint32_t out_v_shapes[2]   = {static_cast<uint32_t>(q_tile), static_cast<uint32_t>(head_dim)};
                uint32_t out_v_offsets[2]  = {static_cast<uint32_t>(cur_offset), 0};
                Tensor qi       = query.view(qi_shapes,    qi_offsets);
                Tensor out_view = out.view(out_v_shapes,   out_v_offsets);

                PTOParam params_inplace;
                params_inplace.add_output(oi);
                params_inplace.add_output(li_update);
                params_inplace.add_output(mi_update);
                pto2_rt_submit_aiv_task(rt, FUNC_AIV_HUB, params_inplace);
                total_tasks++;

                for (uint64_t bn = 0; bn < bn_this_batch; bn++) {
                    uint64_t cur_block_idx = static_cast<uint64_t>(host_block_table[b_idx * block_num + bn]);
                    uint64_t valid_len     = std::min(block_size, cur_seq - bn * block_size);

                    uint32_t kv_shapes[2]  = {static_cast<uint32_t>(block_size), static_cast<uint32_t>(head_dim)};
                    uint32_t kv_offsets[2] = {static_cast<uint32_t>(cur_block_idx * block_size), 0};
                    Tensor kj = key_cache.view(kv_shapes,   kv_offsets);
                    Tensor vj = value_cache.view(kv_shapes,  kv_offsets);

                    uint32_t sij_shapes[2] = {static_cast<uint32_t>(q_tile), static_cast<uint32_t>(block_size)};
                    Tensor sij     = make_tensor(sij_shapes, 2, DataType::FLOAT32);
                    Tensor pij_f16 = make_tensor(sij_shapes, 2, data_type);

                    PTOParam params_qk;
                    params_qk.add_input(qi);
                    params_qk.add_input(kj);
                    params_qk.add_output(sij);
                    pto2_rt_submit_aic_task(rt, FUNC_QK_MATMUL, params_qk);
                    total_tasks++;

                    uint32_t sij_valid_shapes[2]   = {static_cast<uint32_t>(q_tile), static_cast<uint32_t>(valid_len)};
                    uint32_t sij_valid_offsets[2]  = {0, 0};
                    Tensor sij_valid = sij.view(sij_valid_shapes, sij_valid_offsets);

                    Tensor li = make_tensor(li_shapes, 1, DataType::FLOAT32);
                    Tensor mi = make_tensor(mi_shapes, 1, DataType::FLOAT32);

                    PTOParam params_sf;
                    params_sf.add_input(sij_valid);
                    params_sf.add_output(pij_f16);
                    params_sf.add_output(mi);
                    params_sf.add_output(li);
                    params_sf.add_scalar(float_to_u64(scale_value));
                    pto2_rt_submit_aiv_task(rt, FUNC_SOFTMAX_PREPARE, params_sf);
                    total_tasks++;

                    uint32_t oi_tmp_shapes[2] = {static_cast<uint32_t>(q_tile), static_cast<uint32_t>(head_dim)};
                    Tensor oi_tmp = make_tensor(oi_tmp_shapes, 2, DataType::FLOAT32);

                    PTOParam params_pv;
                    params_pv.add_input(pij_f16);
                    params_pv.add_input(vj);
                    params_pv.add_output(oi_tmp);
                    pto2_rt_submit_aic_task(rt, FUNC_PV_MATMUL, params_pv);
                    total_tasks++;

                    uint64_t is_first = (bn == 0) ? 1 : 0;
                    uint64_t is_last  = (bn == bn_this_batch - 1) ? 1 : 0;

                    PTOParam params_up;
                    params_up.add_input(mi);
                    params_up.add_input(li);
                    params_up.add_input(oi_tmp);
                    params_up.add_inout(mi_update);
                    params_up.add_inout(li_update);
                    params_up.add_inout(oi);
                    params_up.add_output(out_view);
                    params_up.add_scalar(is_first);
                    params_up.add_scalar(is_last);
                    pto2_rt_submit_aiv_task(rt, FUNC_ONLINE_UPDATE, params_up);
                    total_tasks++;
                }
            }
        }
    }

    pto2_orchestrator_done(rt->orchestrators);
    printf("  Total tasks submitted: %d\n", total_tasks);
}

PTO2Runtime* setup_run(const PerfTestCase& tc, GraphCtx& ctx) {
    uint64_t batch      = static_cast<uint64_t>(tc.batch);
    uint64_t num_heads  = static_cast<uint64_t>(tc.num_heads);
    int      kv_head_num = tc.kv_head_num;
    uint64_t head_dim   = static_cast<uint64_t>(tc.head_dim);
    uint64_t block_size = static_cast<uint64_t>(tc.block_size);
    uint64_t block_num  = static_cast<uint64_t>(tc.block_num);
    float    scale_value = tc.scale_value;

    const size_t query_size     = batch * num_heads * head_dim * sizeof(float);
    const size_t kv_cache_size  = batch * block_num * block_size * head_dim * sizeof(float);

    for (uint64_t i = 0; i < batch; i++) {
        g_context_lens[i] = (i < static_cast<uint64_t>(tc.context_lens_count))
                          ? tc.context_lens[i]
                          : static_cast<int>(block_size * block_num);
        for (uint64_t j = 0; j < block_num; j++)
            g_block_table[i * block_num + j] = static_cast<int>(i * block_num + j);
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

    ctx.args[0] = reinterpret_cast<uint64_t>(static_cast<void*>(g_query_buf));
    ctx.args[1] = reinterpret_cast<uint64_t>(static_cast<void*>(g_key_cache_buf));
    ctx.args[2] = reinterpret_cast<uint64_t>(static_cast<void*>(g_value_cache_buf));
    ctx.args[3] = reinterpret_cast<uint64_t>(g_block_table);
    ctx.args[4] = reinterpret_cast<uint64_t>(g_context_lens);
    ctx.args[5] = reinterpret_cast<uint64_t>(static_cast<void*>(g_out_buf));
    ctx.args[6] = reinterpret_cast<uint64_t>(ctx.config);
    ctx.args[7] = static_cast<uint64_t>(query_size);
    ctx.args[8] = static_cast<uint64_t>(kv_cache_size);
    ctx.args[9] = static_cast<uint64_t>(kv_cache_size);

    return make_runtime();
}

#if PTO2_PROFILING

void print_config(const PerfTestCase& tc) {
    section_header_100('-', "--- Config ---");
    printf("  Config: batch=%d, num_heads=%d, head_dim=%d, block_size=%d, block_num=%d\n",
           tc.batch, tc.num_heads, tc.head_dim, tc.block_size, tc.block_num);
    printf("  context_lens: [");
    for (int i = 0; i < tc.context_lens_count; i++)
        printf("%d%s", tc.context_lens[i], i + 1 < tc.context_lens_count ? ", " : "");
    printf("]\n");
}

#endif  // PTO2_PROFILING
