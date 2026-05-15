// Orchestration Function: qwen3_decode (Qwen3-14B V200 manual-scope explicit-deps variant).
// Derived from ``14b/Qwen3Decode_manual_scope`` + V200 kernel split (k_proj_0/v_proj_0, out AIC/AIV).

#include "runtime.h"
#include <iostream>
#include <vector>

#include <stddef.h>
#include <stdint.h>
#include <stdio.h>

#include "pto_orchestration_api.h"
#include "../../../14b/qwen3_14b_decode_macros.h"

// Manual explicit-dependency variant (PTO2_SCOPE(PTO2ScopeMode::MANUAL)).
// Inner PTO2_SCOPE() wrappers removed (AUTO nested in MANUAL is unsupported).
// Per tile: Func0=rmsnorm; all Func1 (q_proj), all Func2 (k_proj), all Func3 (v_proj),
// Func4 (k_proj_0), Func5 (v_proj_0) depend only on Func0 and may run in parallel
// within each projection type where not data-dependent.
// Func6=qk_norm fans in every Func1–5 task id for that tile (requires
// PTO2_LARGE_EXPLICIT_DEPS at link time — see test_qwen3_decode.py).
// Func7=rope_kv_cache: instances are not chained to each other; each waits that
// tile's Func6 and the all_q_padded buffer alloc (no __chain_dep on Func7).
// out_proj_residual: AIC Func12 + AIV Func13; Func12 waits every online_softmax (Func11)
// for batch rows in the tile. Func12 instances for different ``ob`` have no deps on each other;
// Func13 depends only on Func12 in the same ``ob``. post_rmsnorm Func14 fans in every Func13 for the tile.
// MLP Func15–19.
//
// Shape/tiling constants come from ../../../14b/qwen3_14b_decode_macros.h (override via
// PTO2_EXTRA_DEFS, e.g. QWEN3_USER_BATCH=240 QWEN3_BATCH_PADDED=240). Host tensors
// must match QWEN3_USER_BATCH / QWEN3_BATCH_PADDED / QWEN3_HIDDEN / … .

extern "C" {

__attribute__((visibility("default")))
PTO2OrchestrationConfig aicpu_orchestration_config(const ChipStorageTaskArgs& orch_args) {
    (void)orch_args;
    return PTO2OrchestrationConfig{
        .expected_arg_count = 20,
    };
}

__attribute__((visibility("default")))
void aicpu_orchestration_entry(const ChipStorageTaskArgs& orch_args) {
    PTO2TaskId __prev = PTO2TaskId::invalid();
    auto __chain_dep = [&](Arg& __a) {
        if (__prev.is_valid()) {
            __a.add_dep(__prev);
        }
    };

    // External tensors
    Tensor ext_hidden_states = from_tensor_arg(orch_args.tensor(0));
    Tensor ext_input_rms_weight = from_tensor_arg(orch_args.tensor(1));
    Tensor ext_wq = from_tensor_arg(orch_args.tensor(2));
    Tensor ext_wk = from_tensor_arg(orch_args.tensor(3));
    Tensor ext_wv = from_tensor_arg(orch_args.tensor(4));
    Tensor ext_q_norm_weight = from_tensor_arg(orch_args.tensor(5));
    Tensor ext_k_norm_weight = from_tensor_arg(orch_args.tensor(6));
    Tensor ext_seq_lens = from_tensor_arg(orch_args.tensor(7));
    Tensor ext_block_table = from_tensor_arg(orch_args.tensor(8));
    Tensor ext_slot_mapping = from_tensor_arg(orch_args.tensor(9));
    Tensor ext_rope_cos = from_tensor_arg(orch_args.tensor(10));
    Tensor ext_rope_sin = from_tensor_arg(orch_args.tensor(11));
    Tensor ext_k_cache = from_tensor_arg(orch_args.tensor(12));
    Tensor ext_v_cache = from_tensor_arg(orch_args.tensor(13));
    Tensor ext_wo = from_tensor_arg(orch_args.tensor(14));
    Tensor ext_post_rms_weight = from_tensor_arg(orch_args.tensor(15));
    Tensor ext_w_gate = from_tensor_arg(orch_args.tensor(16));
    Tensor ext_w_up = from_tensor_arg(orch_args.tensor(17));
    Tensor ext_w_down = from_tensor_arg(orch_args.tensor(18));
    Tensor ext_out = from_tensor_arg(orch_args.tensor(19));

    PTO2_SCOPE(PTO2ScopeMode::MANUAL) {
        const int64_t user_batch = static_cast<int64_t>(QWEN3_USER_BATCH);
        const int64_t batch_padded = static_cast<int64_t>(QWEN3_BATCH_PADDED);
        const int64_t num_tiles = static_cast<int64_t>(QWEN3_NUM_TILES);
        std::vector<std::vector<PTO2TaskId>> t1_tasks_by_tile(static_cast<size_t>(num_tiles));
        std::vector<std::vector<PTO2TaskId>> t2_tasks_by_tile(static_cast<size_t>(num_tiles));
        std::vector<std::vector<PTO2TaskId>> t3_tasks_by_tile(static_cast<size_t>(num_tiles));
        std::vector<PTO2TaskId> last_t4(static_cast<size_t>(num_tiles), PTO2TaskId::invalid());
        std::vector<PTO2TaskId> rope_kv_task_per_b(static_cast<size_t>(user_batch), PTO2TaskId::invalid());
        std::vector<std::vector<PTO2TaskId>> online_softmax_tasks_by_b(static_cast<size_t>(user_batch));
        uint32_t q_proj_ci_shapes[2] = {static_cast<uint32_t>(batch_padded), QWEN3_HIDDEN}; /* 5120 */
        TensorCreateInfo q_proj_ci(q_proj_ci_shapes, 2, DataType::FLOAT32);
        TaskOutputTensors alloc_0 = alloc_tensors(q_proj_ci);
        __prev = alloc_0.task_id();
        const Tensor& q_proj = alloc_0.get_ref(0);
        uint32_t k_proj_ci_shapes[2] = {static_cast<uint32_t>(batch_padded), QWEN3_KV_HIDDEN}; /* 1024 */
        TensorCreateInfo k_proj_ci(k_proj_ci_shapes, 2, DataType::FLOAT32);
        TaskOutputTensors alloc_1 = alloc_tensors(k_proj_ci);
        __prev = alloc_1.task_id();
        const Tensor& k_proj = alloc_1.get_ref(0);
        uint32_t v_proj_ci_shapes[2] = {static_cast<uint32_t>(batch_padded), QWEN3_KV_HIDDEN}; /* 1024 */
        TensorCreateInfo v_proj_ci(v_proj_ci_shapes, 2, DataType::FLOAT32);
        TaskOutputTensors alloc_2 = alloc_tensors(v_proj_ci);
        __prev = alloc_2.task_id();
        const Tensor& v_proj = alloc_2.get_ref(0);
        uint32_t q_proj_norm_ci_shapes[2] = {static_cast<uint32_t>(batch_padded), QWEN3_HIDDEN}; /* 5120 */
        TensorCreateInfo q_proj_norm_ci(q_proj_norm_ci_shapes, 2, DataType::FLOAT32);
        TaskOutputTensors alloc_3 = alloc_tensors(q_proj_norm_ci);
        __prev = alloc_3.task_id();
        const Tensor& q_proj_norm = alloc_3.get_ref(0);
        uint32_t k_proj_norm_ci_shapes[2] = {static_cast<uint32_t>(batch_padded), QWEN3_KV_HIDDEN}; /* 1024 */
        TensorCreateInfo k_proj_norm_ci(k_proj_norm_ci_shapes, 2, DataType::FLOAT32);
        TaskOutputTensors alloc_4 = alloc_tensors(k_proj_norm_ci);
        __prev = alloc_4.task_id();
        const Tensor& k_proj_norm = alloc_4.get_ref(0);
        for (int64_t b0 = 0; b0 < batch_padded; b0 += QWEN3_BATCH_TILE) { /* tile 16 */
            
                const int ti = static_cast<int>(b0 / QWEN3_BATCH_TILE); /* /16 */
                const size_t tix = static_cast<size_t>(ti);
                t1_tasks_by_tile[tix].clear();
                t2_tasks_by_tile[tix].clear();
                t3_tasks_by_tile[tix].clear();
                uint32_t normed_tile_ci_shapes[2] = {QWEN3_BATCH_TILE, QWEN3_HIDDEN}; /* 16,5120 */
                TensorCreateInfo normed_tile_ci(normed_tile_ci_shapes, 2, DataType::BFLOAT16);
                TaskOutputTensors alloc_5 = alloc_tensors(normed_tile_ci);
                __prev = alloc_5.task_id();
                const Tensor& normed_tile = alloc_5.get_ref(0);
                int64_t cur_valid = std::min<int64_t>((user_batch - b0), QWEN3_BATCH_TILE); /* 16 */

                // Task 0: rmsnorm
                Arg params_t0;
                params_t0.add_input(ext_hidden_states);
                params_t0.add_output(normed_tile);
                params_t0.add_input(ext_input_rms_weight);
                params_t0.add_scalar(b0);
                params_t0.add_scalar(cur_valid);
                __chain_dep(params_t0);
                TaskOutputTensors __rt_97 = rt_submit_aiv_task(0, params_t0);
                __prev = __rt_97.task_id();
                const PTO2TaskId t0_id = __rt_97.task_id();
                const Tensor& normed_tile__rv_v2 = normed_tile;
                for (int64_t q0 = 0; q0 < QWEN3_HIDDEN; q0 += QWEN3_INPUT_PROJ_K_CHUNK) { /* 5120,256 */
                    

                        // Task 1: q_proj
                        Arg params_t1;
                        params_t1.add_input(normed_tile__rv_v2);
                        params_t1.add_input(ext_wq);
                        params_t1.add_output(q_proj);
                        params_t1.add_scalar(q0);
                        params_t1.add_scalar(b0);
                        params_t1.add_dep(t0_id);
                        TaskOutputTensors __rt_109 = rt_submit_aic_task(1, params_t1);
                        __prev = __rt_109.task_id();
                        t1_tasks_by_tile[tix].push_back(__rt_109.task_id());
                        const Tensor& q_proj__ssa_v5 = q_proj;
                    
                }
                for (int64_t kv0 = 0; kv0 < QWEN3_KV_HIDDEN; kv0 += QWEN3_KV_PROJ_K_CHUNK) { /* 1024,256 */
                    

                        // Task 2: k_proj
                        Arg params_t2;
                        params_t2.add_input(normed_tile__rv_v2);
                        params_t2.add_input(ext_wk);
                        params_t2.add_output(k_proj);
                        params_t2.add_scalar(kv0);
                        params_t2.add_scalar(b0);
                        params_t2.add_dep(t0_id);
                        TaskOutputTensors __rt_123 = rt_submit_aic_task(2, params_t2);
                        __prev = __rt_123.task_id();
                        t2_tasks_by_tile[tix].push_back(__rt_123.task_id());
                        const Tensor& k_proj__ssa_v5 = k_proj;

                        // Task 3: v_proj
                        Arg params_t3;
                        params_t3.add_input(normed_tile__rv_v2);
                        params_t3.add_input(ext_wv);
                        params_t3.add_output(v_proj);
                        params_t3.add_scalar(kv0);
                        params_t3.add_scalar(b0);
                        params_t3.add_dep(t0_id);
                        TaskOutputTensors __rt_133 = rt_submit_aic_task(3, params_t3);
                        __prev = __rt_133.task_id();
                        t3_tasks_by_tile[tix].push_back(__rt_133.task_id());
                        const Tensor& v_proj__ssa_v5 = v_proj;

                        int64_t kv_col_b = (kv0 + 128);

                        // Task 4: k_proj_0
                        Arg params_k0;
                        params_k0.add_input(normed_tile__rv_v2);
                        params_k0.add_input(ext_wk);
                        params_k0.add_inout(k_proj);
                        params_k0.add_scalar(kv_col_b);
                        params_k0.add_scalar(b0);
                        params_k0.add_dep(t0_id);
                        TaskOutputTensors __rt_k0 = rt_submit_aic_task(4, params_k0);
                        __prev = __rt_k0.task_id();
                        t2_tasks_by_tile[tix].push_back(__rt_k0.task_id());

                        // Task 5: v_proj_0
                        Arg params_v0;
                        params_v0.add_input(normed_tile__rv_v2);
                        params_v0.add_input(ext_wv);
                        params_v0.add_inout(v_proj);
                        params_v0.add_scalar(kv_col_b);
                        params_v0.add_scalar(b0);
                        params_v0.add_dep(t0_id);
                        TaskOutputTensors __rt_v0 = rt_submit_aic_task(5, params_v0);
                        __prev = __rt_v0.task_id();
                        t3_tasks_by_tile[tix].push_back(__rt_v0.task_id());
                    
                }

                // Task 6: qk_norm (depends on Func1/2/3 and k_proj_0/v_proj_0 for this tile)
                Arg params_t4;
                params_t4.add_output(k_proj_norm);
                params_t4.add_output(q_proj_norm);
                params_t4.add_input(q_proj);
                params_t4.add_input(ext_q_norm_weight);
                params_t4.add_input(k_proj);
                params_t4.add_input(ext_k_norm_weight);
                params_t4.add_scalar(b0);
                for (const PTO2TaskId& __tid : t1_tasks_by_tile[tix]) {
                    params_t4.add_dep(__tid);
                }
                for (const PTO2TaskId& __tid : t2_tasks_by_tile[tix]) {
                    params_t4.add_dep(__tid);
                }
                for (const PTO2TaskId& __tid : t3_tasks_by_tile[tix]) {
                    params_t4.add_dep(__tid);
                }
                __chain_dep(params_t4);
                TaskOutputTensors __rt_151 = rt_submit_aiv_task(6, params_t4);
                __prev = __rt_151.task_id();
                last_t4[static_cast<size_t>(ti)] = __rt_151.task_id();
                const Tensor& k_proj_norm__rv_v4 = k_proj_norm;
                const Tensor& q_proj_norm__rv_v4 = q_proj_norm;
            
        }
        uint32_t attn_out_ci_shapes[2] = {static_cast<uint32_t>(batch_padded), QWEN3_HIDDEN}; /* 5120 */
        TensorCreateInfo attn_out_ci(attn_out_ci_shapes, 2, DataType::BFLOAT16);
        TaskOutputTensors alloc_6 = alloc_tensors(attn_out_ci);
        __prev = alloc_6.task_id();
        const Tensor& attn_out = alloc_6.get_ref(0);
        uint32_t all_q_padded_ci_shapes[2] = {static_cast<uint32_t>((batch_padded * QWEN3_ALL_Q_PAD_STRIDE)), QWEN3_HEAD_DIM}; /* *128,128 */
        TensorCreateInfo all_q_padded_ci(all_q_padded_ci_shapes, 2, DataType::BFLOAT16);
        TaskOutputTensors alloc_7 = alloc_tensors(all_q_padded_ci);
        __prev = alloc_7.task_id();
        const PTO2TaskId all_q_padded_alloc_task = alloc_7.task_id();
        const Tensor& all_q_padded = alloc_7.get_ref(0);
        for (int64_t b = 0; b < user_batch; b += 1) {
            
                size_t idx_ctx_len = b;
                int32_t ctx_len = static_cast<int32_t*>(orch_args.tensor(7).data_as<void>())[idx_ctx_len];
                int64_t pos = (static_cast<int64_t>(ctx_len) - 1);
                int64_t ctx_blocks = ((static_cast<int64_t>(ctx_len) + (QWEN3_SEQ_TILE - 1)) / QWEN3_SEQ_TILE); /* +255/256 */
                int64_t block_table_base = (b * QWEN3_MAX_BLOCKS_PER_SEQ); /* *16 */
                size_t idx_slot = b;
                int32_t slot = static_cast<int32_t*>(orch_args.tensor(9).data_as<void>())[idx_slot];
                int64_t slot_block = (static_cast<int64_t>(slot) / QWEN3_BLOCK_SIZE); /* /256 */
                int64_t slot_offset = (static_cast<int64_t>(slot) - (slot_block * QWEN3_BLOCK_SIZE)); /* %256 */
                uint32_t cos_row_shapes[2] = {1, QWEN3_HEAD_DIM}; /* 128 */
                uint32_t cos_row_offsets[2] = {static_cast<uint32_t>(pos), 0};
                Tensor cos_row = ext_rope_cos.view(cos_row_shapes, cos_row_offsets);
                uint32_t sin_row_shapes[2] = {1, QWEN3_HEAD_DIM}; /* 128 */
                uint32_t sin_row_offsets[2] = {static_cast<uint32_t>(pos), 0};
                Tensor sin_row = ext_rope_sin.view(sin_row_shapes, sin_row_offsets);
                uint32_t cos_lo_shapes[2] = {1, QWEN3_HALF_DIM}; /* 64 */
                uint32_t cos_lo_offsets[2] = {0, 0};
                Tensor cos_lo = cos_row.view(cos_lo_shapes, cos_lo_offsets);
                uint32_t cos_hi_shapes[2] = {1, QWEN3_HALF_DIM}; /* 64 */
                uint32_t cos_hi_offsets[2] = {0, QWEN3_HALF_DIM}; /* 64 */
                Tensor cos_hi = cos_row.view(cos_hi_shapes, cos_hi_offsets);
                uint32_t sin_lo_shapes[2] = {1, QWEN3_HALF_DIM}; /* 64 */
                uint32_t sin_lo_offsets[2] = {0, 0};
                Tensor sin_lo = sin_row.view(sin_lo_shapes, sin_lo_offsets);
                uint32_t sin_hi_shapes[2] = {1, QWEN3_HALF_DIM}; /* 64 */
                uint32_t sin_hi_offsets[2] = {0, QWEN3_HALF_DIM}; /* 64 */
                Tensor sin_hi = sin_row.view(sin_hi_shapes, sin_hi_offsets);

                // Task 7: rope_kv_cache
                Arg params_t5;
                params_t5.add_output(all_q_padded);
                params_t5.add_output(ext_k_cache);
                params_t5.add_output(ext_v_cache);
                params_t5.add_input(k_proj_norm);
                params_t5.add_input(cos_lo);
                params_t5.add_input(sin_lo);
                params_t5.add_input(cos_hi);
                params_t5.add_input(sin_hi);
                params_t5.add_input(v_proj);
                params_t5.add_input(q_proj_norm);
                params_t5.add_scalar(slot_block);
                params_t5.add_scalar(slot_offset);
                params_t5.add_scalar(b);
                params_t5.add_dep(last_t4[static_cast<size_t>(b / QWEN3_BATCH_TILE)]); /* /16 */
                params_t5.add_dep(all_q_padded_alloc_task);
                TaskOutputTensors __rt_209 = rt_submit_aiv_task(7, params_t5);
                __prev = __rt_209.task_id();
                rope_kv_task_per_b[static_cast<size_t>(b)] = __rt_209.task_id();
                const Tensor& all_q_padded__rv_v4 = all_q_padded;
                const Tensor& k_cache__rv_v4 = ext_k_cache;
                const Tensor& v_cache__rv_v4 = ext_v_cache;
                uint32_t attn_row_shapes[2] = {1, QWEN3_HIDDEN}; /* 5120 */
                uint32_t attn_row_offsets[2] = {static_cast<uint32_t>(b), 0};
                Tensor attn_row = attn_out.view(attn_row_shapes, attn_row_offsets);
                for (int64_t gi = 0; gi < QWEN3_NUM_KV_HEADS; gi += 2) { /* 8 */
                    
                        uint32_t all_raw_scores0_ci_shapes[2] = {QWEN3_SEQ_TILE, QWEN3_SEQ_TILE}; /* 256,256 */
                        TensorCreateInfo all_raw_scores0_ci(all_raw_scores0_ci_shapes, 2, DataType::FLOAT32);
                        uint32_t all_raw_scores1_ci_shapes[2] = {QWEN3_SEQ_TILE, QWEN3_SEQ_TILE}; /* 256,256 */
                        TensorCreateInfo all_raw_scores1_ci(all_raw_scores1_ci_shapes, 2, DataType::FLOAT32);
                        uint32_t all_exp_padded0_ci_shapes[2] = {QWEN3_SEQ_TILE, QWEN3_SEQ_TILE}; /* 256,256 */
                        TensorCreateInfo all_exp_padded0_ci(all_exp_padded0_ci_shapes, 2, DataType::BFLOAT16);
                        uint32_t all_exp_padded1_ci_shapes[2] = {QWEN3_SEQ_TILE, QWEN3_SEQ_TILE}; /* 256,256 */
                        TensorCreateInfo all_exp_padded1_ci(all_exp_padded1_ci_shapes, 2, DataType::BFLOAT16);
                        uint32_t all_cur_mi0_ci_shapes[2] = {QWEN3_SEQ_TILE, 1}; /* 256,1 */
                        TensorCreateInfo all_cur_mi0_ci(all_cur_mi0_ci_shapes, 2, DataType::FLOAT32);
                        uint32_t all_cur_mi1_ci_shapes[2] = {QWEN3_SEQ_TILE, 1}; /* 256,1 */
                        TensorCreateInfo all_cur_mi1_ci(all_cur_mi1_ci_shapes, 2, DataType::FLOAT32);
                        uint32_t all_cur_li0_ci_shapes[2] = {QWEN3_SEQ_TILE, 1}; /* 256,1 */
                        TensorCreateInfo all_cur_li0_ci(all_cur_li0_ci_shapes, 2, DataType::FLOAT32);
                        uint32_t all_cur_li1_ci_shapes[2] = {QWEN3_SEQ_TILE, 1}; /* 256,1 */
                        TensorCreateInfo all_cur_li1_ci(all_cur_li1_ci_shapes, 2, DataType::FLOAT32);
                        uint32_t all_oi_tmp0_ci_shapes[2] = {QWEN3_SEQ_TILE, QWEN3_HEAD_DIM}; /* 256,128 */
                        TensorCreateInfo all_oi_tmp0_ci(all_oi_tmp0_ci_shapes, 2, DataType::FLOAT32);
                        uint32_t all_oi_tmp1_ci_shapes[2] = {QWEN3_SEQ_TILE, QWEN3_HEAD_DIM}; /* 256,128 */
                        TensorCreateInfo all_oi_tmp1_ci(all_oi_tmp1_ci_shapes, 2, DataType::FLOAT32);
                        TaskOutputTensors alloc_8 = alloc_tensors(all_raw_scores0_ci, all_raw_scores1_ci, all_exp_padded0_ci, all_exp_padded1_ci, all_cur_mi0_ci, all_cur_mi1_ci, all_cur_li0_ci, all_cur_li1_ci, all_oi_tmp0_ci, all_oi_tmp1_ci);
                        __prev = alloc_8.task_id();
                        const Tensor& all_raw_scores0 = alloc_8.get_ref(0);
                        const Tensor& all_raw_scores1 = alloc_8.get_ref(1);
                        const Tensor& all_exp_padded0 = alloc_8.get_ref(2);
                        const Tensor& all_exp_padded1 = alloc_8.get_ref(3);
                        const Tensor& all_cur_mi0 = alloc_8.get_ref(4);
                        const Tensor& all_cur_mi1 = alloc_8.get_ref(5);
                        const Tensor& all_cur_li0 = alloc_8.get_ref(6);
                        const Tensor& all_cur_li1 = alloc_8.get_ref(7);
                        const Tensor& all_oi_tmp0 = alloc_8.get_ref(8);
                        const Tensor& all_oi_tmp1 = alloc_8.get_ref(9);
                        int64_t gi0 = gi;
                        int64_t gi1 = (gi + 1);
                        int64_t kvh0 = gi0;
                        int64_t qg0 = (gi0 - kvh0);
                        int64_t q_base0 = ((kvh0 + qg0) * QWEN3_Q_PER_KV); /* *5 */
                        int64_t q_padded_row0 = ((b * QWEN3_ALL_Q_PAD_STRIDE) + (gi0 * QWEN3_Q_HEAD_PAD)); /* *128, *16 */
                        uint32_t q_padded0_shapes[2] = {QWEN3_BATCH_TILE, QWEN3_HEAD_DIM}; /* 16,128 */
                        uint32_t q_padded0_offsets[2] = {static_cast<uint32_t>(q_padded_row0), 0};
                        Tensor q_padded0 = all_q_padded__rv_v4.view(q_padded0_shapes, q_padded0_offsets);
                        int64_t kvh1 = gi1;
                        int64_t qg1 = (gi1 - kvh1);
                        int64_t q_base1 = ((kvh1 + qg1) * QWEN3_Q_PER_KV); /* *5 */
                        int64_t q_padded_row1 = ((b * QWEN3_ALL_Q_PAD_STRIDE) + (gi1 * QWEN3_Q_HEAD_PAD)); /* *128, *16 */
                        uint32_t q_padded1_shapes[2] = {QWEN3_BATCH_TILE, QWEN3_HEAD_DIM}; /* 16,128 */
                        uint32_t q_padded1_offsets[2] = {static_cast<uint32_t>(q_padded_row1), 0};
                        Tensor q_padded1 = all_q_padded__rv_v4.view(q_padded1_shapes, q_padded1_offsets);

                        // Task 8: qk_matmul
                        Arg params_t6;
                        params_t6.add_output(all_raw_scores0);
                        params_t6.add_output(all_raw_scores1);
                        params_t6.add_input(ext_block_table);
                        params_t6.add_input(k_cache__rv_v4);
                        params_t6.add_input(q_padded0);
                        params_t6.add_input(q_padded1);
                        params_t6.add_scalar(ctx_blocks);
                        params_t6.add_scalar(block_table_base);
                        params_t6.add_scalar(kvh0);
                        params_t6.add_scalar(kvh1);
                        params_t6.add_dep(rope_kv_task_per_b[static_cast<size_t>(b)]);
                        __chain_dep(params_t6);
                        TaskOutputTensors __rt_278 = rt_submit_aic_task(8, params_t6);
                        __prev = __rt_278.task_id();
                        const Tensor& all_raw_scores0__rv_v2 = all_raw_scores0;
                        const Tensor& all_raw_scores1__rv_v2 = all_raw_scores1;

                        // Task 9: softmax
                        Arg params_t7;
                        params_t7.add_output(all_cur_li0);
                        params_t7.add_output(all_cur_li1);
                        params_t7.add_output(all_cur_mi0);
                        params_t7.add_output(all_cur_mi1);
                        params_t7.add_output(all_exp_padded0);
                        params_t7.add_output(all_exp_padded1);
                        params_t7.add_input(all_raw_scores0__rv_v2);
                        params_t7.add_input(all_raw_scores1__rv_v2);
                        params_t7.add_scalar(ctx_blocks);
                        params_t7.add_scalar(ctx_len);
                        __chain_dep(params_t7);
                        TaskOutputTensors __rt_294 = rt_submit_aiv_task(9, params_t7);
                        __prev = __rt_294.task_id();
                        const Tensor& all_cur_li0__rv_v2 = all_cur_li0;
                        const Tensor& all_cur_li1__rv_v2 = all_cur_li1;
                        const Tensor& all_cur_mi0__rv_v2 = all_cur_mi0;
                        const Tensor& all_cur_mi1__rv_v2 = all_cur_mi1;
                        const Tensor& all_exp_padded0__rv_v2 = all_exp_padded0;
                        const Tensor& all_exp_padded1__rv_v2 = all_exp_padded1;

                        // Task 10: sv_matmul
                        Arg params_t8;
                        params_t8.add_output(all_oi_tmp0);
                        params_t8.add_output(all_oi_tmp1);
                        params_t8.add_input(ext_block_table);
                        params_t8.add_input(all_exp_padded0__rv_v2);
                        params_t8.add_input(v_cache__rv_v4);
                        params_t8.add_input(all_exp_padded1__rv_v2);
                        params_t8.add_scalar(ctx_blocks);
                        params_t8.add_scalar(block_table_base);
                        params_t8.add_scalar(kvh0);
                        params_t8.add_scalar(kvh1);
                        params_t8.add_dep(rope_kv_task_per_b[static_cast<size_t>(b)]);
                        __chain_dep(params_t8);
                        TaskOutputTensors __rt_314 = rt_submit_aic_task(10, params_t8);
                        __prev = __rt_314.task_id();
                        const Tensor& all_oi_tmp0__rv_v2 = all_oi_tmp0;
                        const Tensor& all_oi_tmp1__rv_v2 = all_oi_tmp1;

                        // Task 11: online_softmax
                        Arg params_t9;
                        params_t9.add_input(all_oi_tmp0__rv_v2);
                        params_t9.add_input(all_cur_mi0__rv_v2);
                        params_t9.add_input(all_cur_li0__rv_v2);
                        params_t9.add_input(all_oi_tmp1__rv_v2);
                        params_t9.add_input(all_cur_mi1__rv_v2);
                        params_t9.add_input(all_cur_li1__rv_v2);
                        params_t9.add_output(attn_row);
                        params_t9.add_scalar(ctx_blocks);
                        params_t9.add_scalar(q_base0);
                        params_t9.add_scalar(q_base1);
                        __chain_dep(params_t9);
                        TaskOutputTensors __rt_330 = rt_submit_aiv_task(11, params_t9);
                        __prev = __rt_330.task_id();
                        online_softmax_tasks_by_b[static_cast<size_t>(b)].push_back(__rt_330.task_id());
                        const Tensor& attn_row__ssa_v4 = attn_row;
                    
                }
            
        }
        for (int64_t b0 = 0; b0 < batch_padded; b0 += QWEN3_BATCH_TILE) { /* tile 16 */
            
                uint32_t resid1_tile_ci_shapes[2] = {QWEN3_BATCH_TILE, QWEN3_HIDDEN}; /* 16,5120 */
                TensorCreateInfo resid1_tile_ci(resid1_tile_ci_shapes, 2, DataType::FLOAT32);
                uint32_t post_norm_tile_ci_shapes[2] = {QWEN3_BATCH_TILE, QWEN3_HIDDEN}; /* 16,5120 */
                TensorCreateInfo post_norm_tile_ci(post_norm_tile_ci_shapes, 2, DataType::BFLOAT16);
                uint32_t mlp_tile_ci_shapes[2] = {QWEN3_BATCH_TILE, QWEN3_INTERMEDIATE}; /* 16,17408 */
                TensorCreateInfo mlp_tile_ci(mlp_tile_ci_shapes, 2, DataType::BFLOAT16);
                TaskOutputTensors alloc_9 = alloc_tensors(resid1_tile_ci, post_norm_tile_ci, mlp_tile_ci);
                __prev = alloc_9.task_id();
                const Tensor& resid1_tile = alloc_9.get_ref(0);
                const Tensor& post_norm_tile = alloc_9.get_ref(1);
                const Tensor& mlp_tile = alloc_9.get_ref(2);
                int64_t cur_valid__ssa_v1 = std::min<int64_t>((user_batch - b0), QWEN3_BATCH_TILE); /* 16 */
                std::vector<PTO2TaskId> out_proj_aiv_tasks;
                out_proj_aiv_tasks.reserve(static_cast<size_t>(QWEN3_OUT_PROJ_OB_COUNT));
                for (int64_t ob = 0; ob < QWEN3_OUT_PROJ_OB_COUNT; ob += 1) { /* 20 */
                        uint32_t o_proj_chunk_fp32_ci_shapes[2] = {QWEN3_BATCH_TILE, QWEN3_OUT_PROJ_K_CHUNK}; /* 16,256 */
                        TensorCreateInfo o_proj_chunk_fp32_ci(o_proj_chunk_fp32_ci_shapes, 2, DataType::FLOAT32);
                        TaskOutputTensors alloc_10 = alloc_tensors(o_proj_chunk_fp32_ci);
                        __prev = alloc_10.task_id();
                        const Tensor& o_proj_chunk_fp32 = alloc_10.get_ref(0);
                        int64_t o0 = (ob * QWEN3_OUT_PROJ_K_CHUNK); /* *256 */

                        // Task 12: out_proj_residual_aic — wait every online_softmax for rows in this tile.
                        // No cross-ob dependency between Task 12 instances; Task 13 depends only on Task 12
                        // in the same ob iteration.
                        Arg params_o12;
                        params_o12.add_input(attn_out);
                        params_o12.add_input(ext_wo);
                        params_o12.add_inout(o_proj_chunk_fp32);
                        params_o12.add_scalar(b0);
                        params_o12.add_scalar(o0);
                        for (int64_t __row = 0; __row < cur_valid__ssa_v1; ++__row) {
                            const int64_t bb = b0 + __row;
                            for (const PTO2TaskId& __os_tid : online_softmax_tasks_by_b[static_cast<size_t>(bb)]) {
                                params_o12.add_dep(__os_tid);
                            }
                        }
                        __chain_dep(params_o12);
                        TaskOutputTensors __rt_o12 = rt_submit_aic_task(12, params_o12);
                        __prev = __rt_o12.task_id();

                        // Task 13: out_proj_residual_aiv
                        Arg params_o13;
                        params_o13.add_input(ext_hidden_states);
                        params_o13.add_input(o_proj_chunk_fp32);
                        params_o13.add_output(resid1_tile);
                        params_o13.add_scalar(b0);
                        params_o13.add_scalar(o0);
                        params_o13.add_scalar(cur_valid__ssa_v1);
                        params_o13.add_dep(__rt_o12.task_id());
                        __chain_dep(params_o13);
                        TaskOutputTensors __rt_o13 = rt_submit_aiv_task(13, params_o13);
                        __prev = __rt_o13.task_id();
                        out_proj_aiv_tasks.push_back(__rt_o13.task_id());
                }

                // post_rmsnorm: fan in every out_proj AIV (Task 13) for this tile — resid1_tile is chunked by ob.
                Arg params_t14pr;
                params_t14pr.add_input(resid1_tile);
                params_t14pr.add_output(post_norm_tile);
                params_t14pr.add_input(ext_post_rms_weight);
                for (const PTO2TaskId& __op_tid : out_proj_aiv_tasks) {
                    params_t14pr.add_dep(__op_tid);
                }
                __chain_dep(params_t14pr);
                TaskOutputTensors __rt_377 = rt_submit_aiv_task(14, params_t14pr);
                __prev = __rt_377.task_id();
                const PTO2TaskId post_rmsnorm_id = __rt_377.task_id();
                const Tensor& post_norm_tile__rv_v2 = post_norm_tile;
                constexpr int64_t kMlpSiluSteps = QWEN3_MLP_SILU_STEPS; /* 68 */
                std::vector<PTO2TaskId> silu_task_by_ob(static_cast<size_t>(kMlpSiluSteps), PTO2TaskId::invalid());
                for (int64_t ob = 0; ob < QWEN3_MLP_SILU_STEPS; ob += 1) { /* 68 */
                    
                        uint32_t ret0__out_ci_shapes[2] = {QWEN3_BATCH_TILE, QWEN3_MLP_OUT_CHUNK}; /* 16,256 */
                        TensorCreateInfo ret0__out_ci(ret0__out_ci_shapes, 2, DataType::FLOAT32);
                        uint32_t ret0__out_1_ci_shapes[2] = {QWEN3_BATCH_TILE, QWEN3_MLP_OUT_CHUNK}; /* 16,256 */
                        TensorCreateInfo ret0__out_1_ci(ret0__out_1_ci_shapes, 2, DataType::FLOAT32);
                        TaskOutputTensors alloc_11 = alloc_tensors(ret0__out_ci, ret0__out_1_ci);
                        __prev = alloc_11.task_id();
                        const Tensor& ret0__out = alloc_11.get_ref(0);
                        const Tensor& ret0__out_1 = alloc_11.get_ref(1);
                        int64_t o0 = (ob * QWEN3_MLP_OUT_CHUNK); /* *256 */

                        // Task 15: gate_proj
                        Arg params_t12;
                        params_t12.add_input(post_norm_tile__rv_v2);
                        params_t12.add_input(ext_w_gate);
                        params_t12.add_inout(ret0__out);
                        params_t12.add_scalar(o0);
                        params_t12.add_dep(post_rmsnorm_id);
                        __chain_dep(params_t12);
                        TaskOutputTensors __rt_396 = rt_submit_aic_task(15, params_t12);
                        __prev = __rt_396.task_id();
                        const Tensor& gate_acc = ret0__out;

                        // Task 16: up_proj
                        Arg params_t13;
                        params_t13.add_input(post_norm_tile__rv_v2);
                        params_t13.add_input(ext_w_up);
                        params_t13.add_inout(ret0__out_1);
                        params_t13.add_scalar(o0);
                        params_t13.add_dep(post_rmsnorm_id);
                        __chain_dep(params_t13);
                        TaskOutputTensors __rt_405 = rt_submit_aic_task(16, params_t13);
                        __prev = __rt_405.task_id();
                        const Tensor& up_acc = ret0__out_1;
                        uint32_t ret0__out_2_shapes[2] = {QWEN3_BATCH_TILE, QWEN3_MLP_OUT_CHUNK}; /* 16,256 */
                        uint32_t ret0__out_2_offsets[2] = {0, static_cast<uint32_t>(o0)};
                        Tensor ret0__out_2 = mlp_tile.view(ret0__out_2_shapes, ret0__out_2_offsets);

                        // Task 17: silu
                        Arg params_t14;
                        params_t14.add_input(gate_acc);
                        params_t14.add_input(up_acc);
                        params_t14.add_inout(ret0__out_2);
                        __chain_dep(params_t14);
                        TaskOutputTensors __rt_416 = rt_submit_aiv_task(17, params_t14);
                        __prev = __rt_416.task_id();
                        silu_task_by_ob[static_cast<size_t>(ob)] = __rt_416.task_id();
                    
                }
                for (int64_t dob = 0; dob < QWEN3_DOWN_OUTPUT_CHUNKS; dob += 1) { /* 20 */
                    
                        uint32_t fp32_chunk_gm_ci_shapes[2] = {QWEN3_BATCH_TILE, QWEN3_OUT_PROJ_K_CHUNK}; /* 16,256 */
                        TensorCreateInfo fp32_chunk_gm_ci(fp32_chunk_gm_ci_shapes, 2, DataType::FLOAT32);
                        TaskOutputTensors alloc_12 = alloc_tensors(fp32_chunk_gm_ci);
                        __prev = alloc_12.task_id();
                        const Tensor& fp32_chunk_gm = alloc_12.get_ref(0);
                        int64_t d0 = (dob * QWEN3_OUT_PROJ_K_CHUNK); /* *256 */

                        // Task 18: down_proj — explicit fan-in from every silu (full connection).
                        Arg params_t15;
                        params_t15.add_input(mlp_tile);
                        params_t15.add_input(ext_w_down);
                        params_t15.add_inout(fp32_chunk_gm);
                        params_t15.add_scalar(d0);
                        for (const PTO2TaskId& __silu_tid : silu_task_by_ob) {
                            params_t15.add_dep(__silu_tid);
                        }
                        __chain_dep(params_t15);
                        TaskOutputTensors __rt_433 = rt_submit_aic_task(18, params_t15);
                        __prev = __rt_433.task_id();
                        const PTO2TaskId down_proj_id = __rt_433.task_id();
                        const Tensor& fp32_chunk_gm__ssa_v1 = fp32_chunk_gm;

                        // Task 19: down_proj_residual
                        Arg params_t16;
                        params_t16.add_input(fp32_chunk_gm__ssa_v1);
                        params_t16.add_input(resid1_tile);
                        params_t16.add_output(ext_out);
                        params_t16.add_scalar(d0);
                        params_t16.add_scalar(cur_valid__ssa_v1);
                        params_t16.add_scalar(b0);
                        params_t16.add_dep(down_proj_id);
                        __chain_dep(params_t16);
                        TaskOutputTensors __rt_444 = rt_submit_aiv_task(19, params_t16);
                        __prev = __rt_444.task_id();
                        const Tensor& out = ext_out;
                    
                }
            
        }
    }
}

}  // extern "C"
