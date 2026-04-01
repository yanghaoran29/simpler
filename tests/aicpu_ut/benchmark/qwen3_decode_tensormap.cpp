/**
 * qwen3_decode_tensormap.cpp
 *
 * Qwen3 decode (tensormap) backend (PERF_BACKEND=13): declarations + definitions.
 * Included by the perf drivers when PERF_BACKEND=13. Do not compile as a separate TU.
 *
 * Ported from the generated device-orchestration entry to the perf/cases UT
 * convention. Under the AICore simulator no kernel runs, so:
 *   - every tensor is backed by a distinct region of a scratch pool (so the
 *     tensormap's byte-range overlap detection yields the same dependency graph);
 *   - oversized weight matrices use small placeholder shapes (their contents are
 *     never read); only data that is sliced/sized keeps its real shape;
 *   - alloc_tensors/from_tensor_arg -> make_tensor_external, producers use
 *     add_inout, consumers add_input, and submits go through the explicit-rt shims.
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

static constexpr int QWEN3_TM_CASE_COUNT = 1;
static_assert(PERF_CASE_IDX >= 0 && PERF_CASE_IDX < QWEN3_TM_CASE_COUNT,
              "PERF_CASE_IDX out of range");

// Model dims (kept small for the simulator; weights are placeholders).
static constexpr int64_t USER_BATCH   = 16;
static constexpr int64_t BATCH_PADDED = 16;   // round_up(USER_BATCH, 16)
static constexpr int64_t CTX_LEN      = 128;  // per-request context length

struct Qwen3TmCase {
    char name[128];
    int  user_batch;
};

struct GraphCtx {
    uint64_t args[10];
};

const Qwen3TmCase PERF_CASES[QWEN3_TM_CASE_COUNT] = {
    { "qwen3_decode_tensormap (batch=16, ctx_len=128)", static_cast<int>(USER_BATCH) },
};

// Per-func_id simulated AICore execution time (ns), indexed by the func_id passed
// to pto2_rt_submit_ai{c,v}_task below. Values are the qwen3 14B Decode per-kernel
// mean durations from V200-benchmark/Readme.md (µs → ns). Installed into the
// AICore simulator at the top of build_graph. Other samples define their own table.
static const int kQwen3TmFuncDurationNs[] = {
    22780,  // 0  rmsnorm               (aiv)
    26980,  // 1  q_proj                (aic)
    17770,  // 2  k_proj                (aic)
    19140,  // 3  v_proj                (aic)
    13380,  // 4  qk_norm               (aiv)
    9560,   // 5  rope_kv_cache         (aiv)
    29500,  // 6  qk_matmul             (aic)
    20010,  // 7  softmax               (aiv)
    31480,  // 8  sv_matmul             (aic)
    20440,  // 9  online_softmax        (aiv)
    43590,  // 10 out_proj_residual_aic (aic)
    91230,  // 11 out_proj_residual_aiv (aiv)
    27790,  // 12 post_rmsnorm          (aiv)
    97020,  // 13 gate_proj             (aic)
    98440,  // 14 up_proj               (aic)
    2940,   // 15 silu                  (aiv)
    74320,  // 16 down_proj             (aic)
    3130,   // 17 down_proj_residual    (aiv)
};
// Subtracted from each per-func duration before completion (ns): models the
// dispatch/handshake overhead already on the critical path.
static constexpr int kQwen3TmDurationCorrectionNs = 1000;

// ─── Scratch pool: distinct non-overlapping region per tensor ─────────────────
static uint8_t g_pool[96u * 1024u * 1024u];
static size_t  g_pool_off = 0;

static void* pool_alloc(uint64_t bytes) {
    size_t a = (g_pool_off + 63u) & ~static_cast<size_t>(63u);
    g_pool_off = a + static_cast<size_t>(bytes);
    return g_pool + a;
}

static Tensor mk(const uint32_t shapes[], uint32_t ndims, DataType dtype) {
    uint64_t numel = 1;
    for (uint32_t i = 0; i < ndims; i++) numel *= shapes[i];
    return make_tensor_external(pool_alloc(numel * get_element_size(dtype)), shapes, ndims, dtype);
}

void build_graph(PTO2Runtime* rt, uint64_t* args, int arg_count) {
    (void)arg_count;
    (void)args;
    g_pool_off = 0;

    // Install this sample's per-func_id AICore duration table before any task is
    // dispatched (the scheduler resolves each task's simulated duration from it).
    pto2_sim_aicore_set_func_duration_table(
        kQwen3TmFuncDurationNs,
        static_cast<int32_t>(sizeof(kQwen3TmFuncDurationNs) / sizeof(kQwen3TmFuncDurationNs[0])),
        kQwen3TmDurationCorrectionNs
    );

    // External inputs/outputs. Weight matrices are never read by the sim, so they
    // use small placeholder shapes; sliced/sized data keeps its real shape.
    uint32_t hs_shapes[2]      = {static_cast<uint32_t>(USER_BATCH), 5120};
    uint32_t wsmall[2]         = {16, 16};
    uint32_t vec5120[1]        = {5120};
    uint32_t vec128[1]         = {128};
    uint32_t rope_shapes[2]    = {static_cast<uint32_t>(CTX_LEN), 128};
    uint32_t kv_cache_shapes[2]= {128, 128};
    uint32_t out_shapes[2]     = {static_cast<uint32_t>(USER_BATCH), 5120};

    Tensor ext_hidden_states   = mk(hs_shapes, 2, DataType::FLOAT32);
    Tensor ext_input_rms_weight= mk(vec5120, 1, DataType::FLOAT32);
    Tensor ext_wq              = mk(wsmall, 2, DataType::FLOAT32);
    Tensor ext_wk              = mk(wsmall, 2, DataType::FLOAT32);
    Tensor ext_wv              = mk(wsmall, 2, DataType::FLOAT32);
    Tensor ext_q_norm_weight   = mk(vec128, 1, DataType::FLOAT32);
    Tensor ext_k_norm_weight   = mk(vec128, 1, DataType::FLOAT32);
    Tensor ext_rope_cos        = mk(rope_shapes, 2, DataType::FLOAT32);
    Tensor ext_rope_sin        = mk(rope_shapes, 2, DataType::FLOAT32);
    Tensor ext_k_cache         = mk(kv_cache_shapes, 2, DataType::FLOAT32);
    Tensor ext_v_cache         = mk(kv_cache_shapes, 2, DataType::FLOAT32);
    Tensor ext_wo              = mk(wsmall, 2, DataType::FLOAT32);
    Tensor ext_post_rms_weight = mk(vec5120, 1, DataType::FLOAT32);
    Tensor ext_w_gate          = mk(wsmall, 2, DataType::FLOAT32);
    Tensor ext_w_up            = mk(wsmall, 2, DataType::FLOAT32);
    Tensor ext_w_down          = mk(wsmall, 2, DataType::FLOAT32);
    Tensor ext_out             = mk(out_shapes, 2, DataType::FLOAT32);

    // Integer side tables (read directly instead of get_tensor_data).
    static int seq_lens[USER_BATCH];
    static int slot_mapping[USER_BATCH];
    for (int i = 0; i < USER_BATCH; i++) {
        seq_lens[i]     = static_cast<int>(CTX_LEN);
        slot_mapping[i] = i;
    }
    // ext_block_table is only ever read whole (no get_tensor_data on it here).
    uint32_t block_table_shapes[1] = {static_cast<uint32_t>(USER_BATCH * 32)};
    Tensor ext_block_table = mk(block_table_shapes, 1, DataType::INT32);

    int total_tasks = 0;

    PTO2_SCOPE(rt) {
        uint32_t all_q_padded_shapes[2] = {11520, 128};
        Tensor all_q_padded = mk(all_q_padded_shapes, 2, DataType::BFLOAT16);
        uint32_t q_proj_shapes[2]      = {static_cast<uint32_t>(BATCH_PADDED), 5120};
        Tensor q_proj = mk(q_proj_shapes, 2, DataType::FLOAT32);
        uint32_t k_proj_shapes[2]      = {static_cast<uint32_t>(BATCH_PADDED), 1024};
        Tensor k_proj = mk(k_proj_shapes, 2, DataType::FLOAT32);
        uint32_t v_proj_shapes[2]      = {static_cast<uint32_t>(BATCH_PADDED), 1024};
        Tensor v_proj = mk(v_proj_shapes, 2, DataType::FLOAT32);
        uint32_t q_proj_norm_shapes[2] = {static_cast<uint32_t>(BATCH_PADDED), 5120};
        Tensor q_proj_norm = mk(q_proj_norm_shapes, 2, DataType::FLOAT32);
        uint32_t k_proj_norm_shapes[2] = {static_cast<uint32_t>(BATCH_PADDED), 1024};
        Tensor k_proj_norm = mk(k_proj_norm_shapes, 2, DataType::FLOAT32);

        for (int64_t b0 = 0; b0 < BATCH_PADDED; b0 += 16) {
            PTO2_SCOPE(rt) {
                uint32_t normed_tile_shapes[2] = {16, 5120};
                Tensor normed_tile = mk(normed_tile_shapes, 2, DataType::BFLOAT16);
                int64_t cur_valid = std::min<int64_t>((USER_BATCH - b0), 16);

                PTOParam p_t0;  // rmsnorm
                p_t0.add_input(ext_hidden_states);
                p_t0.add_inout(normed_tile);
                p_t0.add_input(ext_input_rms_weight);
                p_t0.add_scalar(static_cast<uint64_t>(b0));
                p_t0.add_scalar(static_cast<uint64_t>(cur_valid));
                pto2_rt_submit_aiv_task(rt, 0, p_t0);
                total_tasks++;

                for (int64_t q0 = 0; q0 < 5120; q0 += 256) {
                    PTO2_SCOPE(rt) {
                        // Declare the [.., q0:q0+256] column tile actually written, not
                        // the whole q_proj — otherwise the tensormap sees every tile
                        // writing the same region (WAW) and serializes the 20 tiles.
                        uint32_t q_tile[2] = {static_cast<uint32_t>(BATCH_PADDED), 256};
                        uint32_t q_off[2]  = {0, static_cast<uint32_t>(q0)};
                        Tensor q_proj_tile = q_proj.view(q_tile, q_off);

                        PTOParam p_t1;  // q_proj
                        p_t1.add_input(normed_tile);
                        p_t1.add_input(ext_wq);
                        p_t1.add_inout(q_proj_tile);
                        p_t1.add_scalar(static_cast<uint64_t>(q0));
                        p_t1.add_scalar(static_cast<uint64_t>(b0));
                        pto2_rt_submit_aic_task(rt, 1, p_t1);
                        total_tasks++;
                    }
                }
                for (int64_t kv0 = 0; kv0 < 1024; kv0 += 128) {
                    PTO2_SCOPE(rt) {
                        uint32_t k_tile[2] = {static_cast<uint32_t>(BATCH_PADDED), 128};
                        uint32_t k_off[2]  = {0, static_cast<uint32_t>(kv0)};
                        Tensor k_proj_tile = k_proj.view(k_tile, k_off);

                        PTOParam p_t2;  // k_proj
                        p_t2.add_input(normed_tile);
                        p_t2.add_input(ext_wk);
                        p_t2.add_inout(k_proj_tile);
                        p_t2.add_scalar(static_cast<uint64_t>(kv0));
                        p_t2.add_scalar(static_cast<uint64_t>(b0));
                        pto2_rt_submit_aic_task(rt, 2, p_t2);
                        total_tasks++;
                    }
                }
                for (int64_t kv0 = 0; kv0 < 1024; kv0 += 128) {
                    PTO2_SCOPE(rt) {
                        uint32_t v_tile[2] = {static_cast<uint32_t>(BATCH_PADDED), 128};
                        uint32_t v_off[2]  = {0, static_cast<uint32_t>(kv0)};
                        Tensor v_proj_tile = v_proj.view(v_tile, v_off);

                        PTOParam p_t3;  // v_proj
                        p_t3.add_input(normed_tile);
                        p_t3.add_input(ext_wv);
                        p_t3.add_inout(v_proj_tile);
                        p_t3.add_scalar(static_cast<uint64_t>(kv0));
                        p_t3.add_scalar(static_cast<uint64_t>(b0));
                        pto2_rt_submit_aic_task(rt, 3, p_t3);
                        total_tasks++;
                    }
                }

                PTOParam p_t4;  // qk_norm
                p_t4.add_inout(k_proj_norm);
                p_t4.add_inout(q_proj_norm);
                p_t4.add_input(q_proj);
                p_t4.add_input(ext_q_norm_weight);
                p_t4.add_input(k_proj);
                p_t4.add_input(ext_k_norm_weight);
                p_t4.add_scalar(static_cast<uint64_t>(b0));
                pto2_rt_submit_aiv_task(rt, 4, p_t4);
                total_tasks++;
            }
        }

        uint32_t attn_out_shapes[2] = {static_cast<uint32_t>(BATCH_PADDED), 5120};
        Tensor attn_out = mk(attn_out_shapes, 2, DataType::BFLOAT16);

        for (int64_t b = 0; b < USER_BATCH; b += 1) {
            PTO2_SCOPE(rt) {
                uint32_t s4096x128[2] = {4096, 128};
                uint32_t s4096x1[2]   = {4096, 1};
                Tensor all_raw_scores = mk(s4096x128, 2, DataType::FLOAT32);
                Tensor all_exp_padded = mk(s4096x128, 2, DataType::BFLOAT16);
                Tensor all_cur_mi     = mk(s4096x1, 2, DataType::FLOAT32);
                Tensor all_cur_li     = mk(s4096x1, 2, DataType::FLOAT32);
                Tensor all_oi_tmp     = mk(s4096x128, 2, DataType::FLOAT32);

                int32_t ctx_len  = seq_lens[b];
                int64_t pos      = static_cast<int64_t>(ctx_len) - 1;
                int64_t ctx_blocks = (static_cast<int64_t>(ctx_len) + 127) / 128;
                int64_t block_table_base = b * 32;
                int32_t slot     = slot_mapping[b];
                int64_t slot_block  = static_cast<int64_t>(slot) / 128;
                int64_t slot_offset = static_cast<int64_t>(slot) - slot_block * 128;

                uint32_t row_off[2] = {static_cast<uint32_t>(pos), 0};
                uint32_t row_win[2] = {1, 128};
                Tensor cos_row = ext_rope_cos.view(row_win, row_off);
                Tensor sin_row = ext_rope_sin.view(row_win, row_off);
                uint32_t lo_off[2] = {0, 0};
                uint32_t hi_off[2] = {0, 64};
                uint32_t half_win[2] = {1, 64};
                Tensor cos_lo = cos_row.view(half_win, lo_off);
                Tensor cos_hi = cos_row.view(half_win, hi_off);
                Tensor sin_lo = sin_row.view(half_win, lo_off);
                Tensor sin_hi = sin_row.view(half_win, hi_off);

                // Per-batch sub-views of the shared query scratch and the paged KV
                // cache. Each request b owns a disjoint q_padded slice and a distinct
                // KV-cache slot row; declaring the whole tensors (write in func_5, read
                // in func_6/func_8) would make every batch's chain depend on every
                // other batch's func_5 and serialize the 16 independent chains.
                uint32_t qpad_rows   = static_cast<uint32_t>(11520 / USER_BATCH);
                uint32_t qpad_win[2] = {qpad_rows, 128};
                uint32_t qpad_off[2] = {static_cast<uint32_t>(b) * qpad_rows, 0};
                Tensor q_padded_b = all_q_padded.view(qpad_win, qpad_off);
                uint32_t kv_row_win[2] = {1, 128};
                uint32_t kv_row_off[2] = {static_cast<uint32_t>(slot_offset), 0};
                Tensor k_cache_slot = ext_k_cache.view(kv_row_win, kv_row_off);
                Tensor v_cache_slot = ext_v_cache.view(kv_row_win, kv_row_off);

                PTOParam p_t5;  // rope_kv_cache
                p_t5.add_inout(q_padded_b);
                p_t5.add_inout(k_cache_slot);
                p_t5.add_inout(v_cache_slot);
                p_t5.add_input(k_proj_norm);
                p_t5.add_input(cos_lo);
                p_t5.add_input(sin_lo);
                p_t5.add_input(cos_hi);
                p_t5.add_input(sin_hi);
                p_t5.add_input(v_proj);
                p_t5.add_input(q_proj_norm);
                p_t5.add_scalar(static_cast<uint64_t>(slot_block));
                p_t5.add_scalar(static_cast<uint64_t>(slot_offset));
                p_t5.add_scalar(static_cast<uint64_t>(b));
                pto2_rt_submit_aiv_task(rt, 5, p_t5);
                total_tasks++;

                PTOParam p_t6;  // qk_matmul (spmd)
                p_t6.add_input(q_padded_b);
                p_t6.add_inout(all_raw_scores);
                p_t6.add_input(ext_block_table);
                p_t6.add_input(k_cache_slot);
                p_t6.add_scalar(static_cast<uint64_t>(b));
                p_t6.add_scalar(static_cast<uint64_t>(ctx_blocks));
                p_t6.add_scalar(static_cast<uint64_t>(block_table_base));
                p_t6.arg_.launch_spec.set_block_num(4);
                pto2_rt_submit_aic_task(rt, 6, p_t6);
                total_tasks++;

                PTOParam p_t7;  // softmax (spmd)
                p_t7.add_inout(all_cur_li);
                p_t7.add_inout(all_cur_mi);
                p_t7.add_inout(all_exp_padded);
                p_t7.add_input(all_raw_scores);
                p_t7.add_scalar(static_cast<uint64_t>(ctx_blocks));
                p_t7.add_scalar(static_cast<uint64_t>(ctx_len));
                p_t7.arg_.launch_spec.set_block_num(4);
                pto2_rt_submit_aiv_task(rt, 7, p_t7);
                total_tasks++;

                PTOParam p_t8;  // sv_matmul (spmd)
                p_t8.add_inout(all_oi_tmp);
                p_t8.add_input(ext_block_table);
                p_t8.add_input(all_exp_padded);
                p_t8.add_input(ext_v_cache);
                p_t8.add_scalar(static_cast<uint64_t>(ctx_blocks));
                p_t8.add_scalar(static_cast<uint64_t>(block_table_base));
                p_t8.arg_.launch_spec.set_block_num(4);
                pto2_rt_submit_aic_task(rt, 8, p_t8);
                total_tasks++;

                for (int64_t gi0 = 0; gi0 < 8; gi0 += 2) {
                    PTO2_SCOPE(rt) {
                        // Each gi0 handles 2 of the 8 attention groups (640 cols
                        // each), i.e. the [.., gi0*640 : gi0*640+1280] column tile
                        // of attn_row. Declaring that tile instead of the whole row
                        // keeps the 4 iterations independent instead of a WAW chain.
                        uint32_t os_tile[2] = {1, 1280};
                        uint32_t os_off[2]  = {static_cast<uint32_t>(b),
                                               static_cast<uint32_t>(gi0 * 640)};
                        Tensor attn_row_tile = attn_out.view(os_tile, os_off);

                        PTOParam p_t9;  // online_softmax
                        p_t9.add_input(all_oi_tmp);
                        p_t9.add_input(all_cur_mi);
                        p_t9.add_input(all_cur_li);
                        p_t9.add_inout(attn_row_tile);
                        p_t9.add_scalar(static_cast<uint64_t>(gi0));
                        p_t9.add_scalar(static_cast<uint64_t>(ctx_blocks));
                        pto2_rt_submit_aiv_task(rt, 9, p_t9);
                        total_tasks++;
                    }
                }
            }
        }

        for (int64_t b0 = 0; b0 < BATCH_PADDED; b0 += 16) {
            PTO2_SCOPE(rt) {
                uint32_t resid_shapes[2]   = {16, 5120};
                uint32_t gm_pipe_shapes[1] = {16384u * 40u};
                uint32_t post_shapes[2]    = {16, 5120};
                uint32_t mlp_shapes[2]     = {16, 17408};
                Tensor resid1_tile     = mk(resid_shapes, 2, DataType::FLOAT32);
                Tensor gm_pipe_buffer_0= mk(gm_pipe_shapes, 1, DataType::FLOAT32);
                Tensor post_norm_tile  = mk(post_shapes, 2, DataType::BFLOAT16);
                Tensor mlp_tile        = mk(mlp_shapes, 2, DataType::BFLOAT16);
                int64_t cur_valid = std::min<int64_t>((USER_BATCH - b0), 16);

                PTOParam p_t10;  // out_proj_residual (MixedKernels: AIC + AIV lanes)
                p_t10.add_input(ext_hidden_states);
                p_t10.add_input(attn_out);
                p_t10.add_input(ext_wo);
                p_t10.add_inout(resid1_tile);
                p_t10.add_inout(gm_pipe_buffer_0);
                p_t10.add_scalar(static_cast<uint64_t>(b0));
                p_t10.add_scalar(static_cast<uint64_t>(cur_valid));
                p_t10.arg_.launch_spec.set_block_num(40);
                MixedKernels mixed_10 = {10, 11, 11};
                pto2_submit_mixed_task(rt, mixed_10, p_t10.arg_);
                total_tasks++;

                PTOParam p_t11;  // post_rmsnorm
                p_t11.add_input(resid1_tile);
                p_t11.add_inout(post_norm_tile);
                p_t11.add_input(ext_post_rms_weight);
                pto2_rt_submit_aiv_task(rt, 12, p_t11);
                total_tasks++;

                for (int64_t ob = 0; ob < 34; ob += 1) {
                    PTO2_SCOPE(rt) {
                        uint32_t mlp_out_shapes[2] = {16, 512};
                        Tensor ret0_out   = mk(mlp_out_shapes, 2, DataType::FLOAT32);
                        Tensor ret0_out_1 = mk(mlp_out_shapes, 2, DataType::FLOAT32);
                        int64_t mlp_o0 = ob * 512;

                        PTOParam p_t12;  // gate_proj
                        p_t12.add_input(post_norm_tile);
                        p_t12.add_input(ext_w_gate);
                        p_t12.add_inout(ret0_out);
                        p_t12.add_scalar(static_cast<uint64_t>(mlp_o0));
                        pto2_rt_submit_aic_task(rt, 13, p_t12);
                        total_tasks++;

                        PTOParam p_t13;  // up_proj
                        p_t13.add_input(post_norm_tile);
                        p_t13.add_input(ext_w_up);
                        p_t13.add_inout(ret0_out_1);
                        p_t13.add_scalar(static_cast<uint64_t>(mlp_o0));
                        pto2_rt_submit_aic_task(rt, 14, p_t13);
                        total_tasks++;

                        uint32_t mlp_off[2] = {0, static_cast<uint32_t>(mlp_o0)};
                        uint32_t mlp_win[2] = {16, 512};
                        Tensor mlp_slice = mlp_tile.view(mlp_win, mlp_off);

                        PTOParam p_t14;  // silu
                        p_t14.add_input(ret0_out);
                        p_t14.add_input(ret0_out_1);
                        p_t14.add_inout(mlp_slice);
                        pto2_rt_submit_aiv_task(rt, 15, p_t14);
                        total_tasks++;
                    }
                }

                for (int64_t dob = 0; dob < 40; dob += 1) {
                    PTO2_SCOPE(rt) {
                        uint32_t chunk_shapes[2] = {16, 128};
                        Tensor fp32_chunk_gm = mk(chunk_shapes, 2, DataType::FLOAT32);
                        int64_t d0 = dob * 128;

                        PTOParam p_t15;  // down_proj
                        p_t15.add_input(mlp_tile);
                        p_t15.add_input(ext_w_down);
                        p_t15.add_inout(fp32_chunk_gm);
                        p_t15.add_scalar(static_cast<uint64_t>(d0));
                        pto2_rt_submit_aic_task(rt, 16, p_t15);
                        total_tasks++;

                        // Each dob writes the [.., d0:d0+128] column tile of ext_out;
                        // declare that tile (not the whole ext_out) so the 40 dob
                        // iterations are independent instead of a WAW chain.
                        uint32_t out_tile[2] = {static_cast<uint32_t>(BATCH_PADDED), 128};
                        uint32_t out_off[2]  = {static_cast<uint32_t>(b0), static_cast<uint32_t>(d0)};
                        Tensor ext_out_tile = ext_out.view(out_tile, out_off);

                        PTOParam p_t16;  // down_proj_residual
                        p_t16.add_input(fp32_chunk_gm);
                        p_t16.add_input(resid1_tile);
                        p_t16.add_inout(ext_out_tile);
                        p_t16.add_scalar(static_cast<uint64_t>(d0));
                        p_t16.add_scalar(static_cast<uint64_t>(cur_valid));
                        p_t16.add_scalar(static_cast<uint64_t>(b0));
                        pto2_rt_submit_aiv_task(rt, 17, p_t16);
                        total_tasks++;
                    }
                }
            }
        }
    }

    pto2_orchestrator_done(rt);
    printf("  Total tasks submitted: %d\n", total_tasks);
}

PTO2Runtime* setup_run(const Qwen3TmCase& tc, GraphCtx& ctx) {
    (void)tc;
    for (int i = 0; i < 10; i++) ctx.args[i] = 0;
    return make_runtime();
}

#if PTO2_PROFILING

void print_config(const Qwen3TmCase& tc) {
    section_header_100('-', "--- Config ---");
    printf("  Config: %s\n", tc.name);
}

#endif  // PTO2_PROFILING
