/**
 * batch_paged_attention.cpp
 *
 * Batch Paged Attention backend (PERF_BACKEND=2): declarations + definitions.
 * Included by test_* when PERF_BACKEND=2. Do not compile as separate TU.
 */

// ─── Declarations ─────────────────────────────────────────────────────────────

#include "pto_runtime2.h"
#include "test_common.h"
#include "json_cases.h"
#include "sim_aicore.h"
#include "common/platform_config.h"
#include "cpu_affinity.h"
#include <algorithm>
#include <atomic>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <csignal>
#include <thread>
#include <unistd.h>
#include <vector>

#ifndef PERF_CASE_IDX
#define PERF_CASE_IDX 0
#endif

static constexpr int PA_CASE_COUNT = 3;
static_assert(PERF_CASE_IDX >= 0 && PERF_CASE_IDX < PA_CASE_COUNT,
              "PERF_CASE_IDX out of range");

static constexpr size_t GLOBAL_MAX_BATCH       = 64;
static constexpr size_t GLOBAL_MAX_NUM_HEADS   = 16;
static constexpr size_t GLOBAL_HEAD_DIM        = 128;
static constexpr size_t GLOBAL_MAX_BLOCK_NUM   = 256;
static constexpr size_t GLOBAL_BLOCK_SIZE      = 128;

static constexpr size_t GLOBAL_QUERY_NELEMS    = GLOBAL_MAX_BATCH * GLOBAL_MAX_NUM_HEADS * GLOBAL_HEAD_DIM;
static constexpr size_t GLOBAL_KV_NELEMS       = GLOBAL_MAX_BATCH * GLOBAL_MAX_BLOCK_NUM * GLOBAL_BLOCK_SIZE * GLOBAL_HEAD_DIM;
static constexpr size_t GLOBAL_BLOCK_TABLE_CNT = GLOBAL_MAX_BATCH * GLOBAL_MAX_BLOCK_NUM;

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
    { "Case1 (batch=64, num_heads=16, head_dim=128, block_size=128, context_len=8193)",
      64, 16, 1, 128, 128, 256, 1.0f, {8193}, 1 },
    { "CaseVarSeq2 (batch=2, context_lens=[8192,4096])",
      2, 16, 1, 128, 128, 256, 1.0f, {8192, 4096}, 2 },
    { "CaseVarSeq4 (batch=4, context_lens=[8192,4096,1024,256])",
      4, 16, 1, 128, 128, 256, 1.0f, {8192, 4096, 1024, 256}, 4 },
};

float g_query_buf[GLOBAL_QUERY_NELEMS];
float g_kv_cache_buf[GLOBAL_KV_NELEMS];
float g_out_buf[GLOBAL_QUERY_NELEMS];
int   g_block_table[GLOBAL_BLOCK_TABLE_CNT];
int   g_context_lens[GLOBAL_MAX_BATCH];

void build_graph(PTO2Runtime* rt, uint64_t* args, int arg_count) {
    (void)arg_count;

    void*    host_query       = reinterpret_cast<void*>(args[0]);
    void*    host_key_cache   = reinterpret_cast<void*>(args[1]);
    void*    host_value_cache = reinterpret_cast<void*>(args[2]);
    int*     host_block_table = reinterpret_cast<int*>(args[3]);
    int*     host_context_lens = reinterpret_cast<int*>(args[4]);
    void*    host_out         = reinterpret_cast<void*>(args[5]);
    int64_t* host_config      = reinterpret_cast<int64_t*>(args[6]);

    (void)args[7];
    size_t key_cache_size = static_cast<size_t>(args[8]);
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

    uint64_t q_tile    = std::min(num_heads, static_cast<uint64_t>(128));
    uint64_t q_loop    = (num_heads + q_tile - 1) / q_tile;
    DataType data_type = DataType::FLOAT32;
    uint64_t elem_size = get_element_size(data_type);

    printf("  batch = %lu, num_heads = %lu, head_dim = %lu, q_tile = %lu\n",
           (unsigned long)batch, (unsigned long)num_heads,
           (unsigned long)head_dim, (unsigned long)q_tile);

    uint64_t max_bn = 0;
    for (uint64_t b = 0; b < batch; b++) {
        uint64_t cur_seq = static_cast<uint64_t>(host_context_lens[b]);
        uint64_t bn_b    = (cur_seq + block_size - 1) / block_size;
        if (bn_b > max_bn) max_bn = bn_b;
    }

    uint32_t query_shapes[2]       = {static_cast<uint32_t>(batch * num_heads), static_cast<uint32_t>(head_dim)};
    uint64_t kv_total_rows         = key_cache_size / (head_dim * elem_size);
    uint32_t key_cache_shapes[2]   = {static_cast<uint32_t>(kv_total_rows), static_cast<uint32_t>(head_dim)};
    uint32_t value_cache_shapes[2] = {static_cast<uint32_t>(kv_total_rows), static_cast<uint32_t>(head_dim)};
    uint32_t out_shapes[2]         = {static_cast<uint32_t>(batch * num_heads), static_cast<uint32_t>(head_dim)};

    Tensor query       = make_tensor_external(host_query,       query_shapes,       2, data_type);
    Tensor key_cache   = make_tensor_external(host_key_cache,   key_cache_shapes,   2, data_type);
    Tensor value_cache = make_tensor_external(host_value_cache, value_cache_shapes, 2, data_type);
    Tensor out         = make_tensor_external(host_out,         out_shapes,         2, DataType::FLOAT32);

    uint64_t bt_addr = reinterpret_cast<uint64_t>(host_block_table);
    uint64_t cl_addr = reinterpret_cast<uint64_t>(host_context_lens);

    uint64_t IN_CORE_BATCH = batch;
    uint64_t num_chunks    = 1;

    int total_tasks = 0;

    for (uint64_t q_idx = 0; q_idx < q_loop; q_idx++) {
        uint64_t q_offset = q_idx * q_tile;

        for (uint64_t chunk_idx = 0; chunk_idx < num_chunks; chunk_idx++) {
            uint64_t batch_start = chunk_idx * IN_CORE_BATCH;
            uint64_t chunk_bc = batch - batch_start;
            if (chunk_bc > IN_CORE_BATCH) chunk_bc = IN_CORE_BATCH;

            PTO2_SCOPE(rt) {
                uint32_t oi_acc_shapes[2]    = {static_cast<uint32_t>(chunk_bc * q_tile), static_cast<uint32_t>(head_dim)};
                uint32_t scalar_acc_shapes[1] = {static_cast<uint32_t>(chunk_bc * q_tile)};
                Tensor oi_batch = make_tensor(oi_acc_shapes,    2, DataType::FLOAT32);
                Tensor li_batch = make_tensor(scalar_acc_shapes, 1, DataType::FLOAT32);
                Tensor mi_batch = make_tensor(scalar_acc_shapes, 1, DataType::FLOAT32);

                PTOParam params_hub;
                params_hub.add_output(oi_batch);
                params_hub.add_output(li_batch);
                params_hub.add_output(mi_batch);
                pto2_rt_submit_aiv_task(rt, FUNC_AIV_HUB, params_hub);
                total_tasks++;

                for (uint64_t bn = 0; bn < max_bn; bn++) {
                    PTO2_SCOPE(rt) {
                        uint32_t sij_shapes[2]    = {static_cast<uint32_t>(chunk_bc * q_tile), static_cast<uint32_t>(block_size)};
                        uint32_t vec_shapes[1]    = {static_cast<uint32_t>(chunk_bc * q_tile)};
                        uint32_t oi_new_shapes[2] = {static_cast<uint32_t>(chunk_bc * q_tile), static_cast<uint32_t>(head_dim)};

                        Tensor sij_b    = make_tensor(sij_shapes,    2, DataType::FLOAT32);
                        Tensor pij_b    = make_tensor(sij_shapes,    2, data_type);
                        Tensor mij_b    = make_tensor(vec_shapes,    1, DataType::FLOAT32);
                        Tensor lij_b    = make_tensor(vec_shapes,    1, DataType::FLOAT32);
                        Tensor oi_new_b = make_tensor(oi_new_shapes, 2, DataType::FLOAT32);

                        PTOParam params_qk;
                        params_qk.add_input(query);
                        params_qk.add_input(key_cache);
                        params_qk.add_output(sij_b);
                        params_qk.add_scalar(bt_addr);
                        params_qk.add_scalar(chunk_bc);
                        params_qk.add_scalar(bn);
                        params_qk.add_scalar(q_offset);
                        params_qk.add_scalar(block_num);
                        params_qk.add_scalar(num_heads);
                        params_qk.add_scalar(batch_start);
                        pto2_rt_submit_aic_task(rt, FUNC_QK_MATMUL, params_qk);
                        total_tasks++;

                        PTOParam params_sf;
                        params_sf.add_input(sij_b);
                        params_sf.add_output(pij_b);
                        params_sf.add_output(mij_b);
                        params_sf.add_output(lij_b);
                        params_sf.add_scalar(float_to_u64(scale_value));
                        params_sf.add_scalar(cl_addr);
                        params_sf.add_scalar(chunk_bc);
                        params_sf.add_scalar(bn);
                        params_sf.add_scalar(batch_start);
                        pto2_rt_submit_aiv_task(rt, FUNC_SOFTMAX_PREPARE, params_sf);
                        total_tasks++;

                        PTOParam params_pv;
                        params_pv.add_input(pij_b);
                        params_pv.add_input(value_cache);
                        params_pv.add_output(oi_new_b);
                        params_pv.add_scalar(bt_addr);
                        params_pv.add_scalar(chunk_bc);
                        params_pv.add_scalar(bn);
                        params_pv.add_scalar(block_num);
                        params_pv.add_scalar(batch_start);
                        pto2_rt_submit_aic_task(rt, FUNC_PV_MATMUL, params_pv);
                        total_tasks++;

                        uint64_t is_first = (bn == 0) ? 1 : 0;
                        uint64_t is_last  = (bn == max_bn - 1) ? 1 : 0;
                        PTOParam params_up;
                        params_up.add_input(mij_b);
                        params_up.add_input(lij_b);
                        params_up.add_input(oi_new_b);
                        params_up.add_inout(mi_batch);
                        params_up.add_inout(li_batch);
                        params_up.add_inout(oi_batch);
                        params_up.add_inout(out);
                        params_up.add_scalar(is_first);
                        params_up.add_scalar(is_last);
                        params_up.add_scalar(chunk_bc);
                        params_up.add_scalar(q_offset);
                        params_up.add_scalar(num_heads);
                        params_up.add_scalar(batch_start);
                        pto2_rt_submit_aiv_task(rt, FUNC_ONLINE_UPDATE, params_up);
                        total_tasks++;
                    }
                }
            }
        }
    }

    pto2_orchestrator_done(rt->orchestrators);
    printf("  Total tasks submitted: %d\n", total_tasks);
    printf("  Expected tasks: %lu (num_chunks=%lu, max_bn=%lu, IN_CORE_BATCH=%lu)\n",
           (unsigned long)(num_chunks * q_loop * (1 + max_bn * 4)),
           (unsigned long)num_chunks, (unsigned long)max_bn, (unsigned long)IN_CORE_BATCH);
}

PTO2Runtime* setup_run(const PerfTestCase& tc, GraphCtx& ctx) {
    uint64_t batch      = static_cast<uint64_t>(tc.batch);
    uint64_t num_heads  = static_cast<uint64_t>(tc.num_heads);
    int      kv_head_num = tc.kv_head_num;
    uint64_t head_dim   = static_cast<uint64_t>(tc.head_dim);
    uint64_t block_size = static_cast<uint64_t>(tc.block_size);
    uint64_t block_num  = static_cast<uint64_t>(tc.block_num);
    float    scale_value = tc.scale_value;

    const size_t query_size    = batch * num_heads * head_dim * sizeof(float);
    const size_t kv_cache_size = batch * block_num * block_size * head_dim * sizeof(float);

    for (uint64_t i = 0; i < batch; i++) {
        g_context_lens[i] = (tc.context_lens_count > 0)
                          ? tc.context_lens[i % tc.context_lens_count]
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
    ctx.args[1] = reinterpret_cast<uint64_t>(static_cast<void*>(g_kv_cache_buf));
    ctx.args[2] = reinterpret_cast<uint64_t>(static_cast<void*>(g_kv_cache_buf));
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
