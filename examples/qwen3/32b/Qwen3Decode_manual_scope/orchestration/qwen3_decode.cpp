// Orchestration Function: qwen3_decode (fully manual dependency build,
// cap=128 vector-fanin variant)
//
// Every task declares its dependencies explicitly via add_dep(...).
// The TensorMap auto-tracking path is bypassed entirely.
//
// Cross-iteration fan-in is added DIRECTLY: tasks like t3, t8, t13 may
// declare 30-100+ explicit deps. This requires PTO2_MAX_EXPLICIT_DEPS
// to be raised from its default of 16. Build the runtime with
//
//     -DPTO2_LARGE_EXPLICIT_DEPS
//
// to raise the cap to 128. Both the runtime .so and this .cpp must be
// compiled with the same flag (the macro affects the Arg struct layout).
//
// Per-task fanin counts under this variant:
//   t3   = 36  (32 t1 + 4 t2)
//   t8   = 65  (1 alloc_2 + 64 t7)
//   t9   = 16  (16 t8)
//   t13  = 101 (1 alloc_4 + 100 t12)
//
// See Qwen3Decode_manual_scope_barrier for an alternative that keeps the
// default cap=16 by reducing fan-in through barrier tasks.

#include "runtime.h"
#include <iostream>
#include <vector>

#include <stddef.h>
#include <stdint.h>
#include <stdio.h>

#include "pto_orchestration_api.h"
#include "../../qwen3_32b_decode_macros.h"

#if PTO2_MAX_EXPLICIT_DEPS < 101
#error "This orchestration requires PTO2_MAX_EXPLICIT_DEPS >= 101. Build the runtime with -DPTO2_LARGE_EXPLICIT_DEPS."
#endif

// See ../../qwen3_32b_decode_macros.h for compile-time shapes (override via PTO2_EXTRA_DEFS).

extern "C" {

__attribute__((visibility("default")))
PTO2OrchestrationConfig aicpu_orchestration_config(const ChipStorageTaskArgs& orch_args) {
    (void)orch_args;
    return PTO2OrchestrationConfig{
        .expected_arg_count = 16,
    };
}

__attribute__((visibility("default")))
void aicpu_orchestration_entry(const ChipStorageTaskArgs& orch_args) {
    // External tensors
    Tensor ext_hidden_states = from_tensor_arg(orch_args.tensor(0));
    Tensor ext_input_rms_weight = from_tensor_arg(orch_args.tensor(1));
    Tensor ext_wq = from_tensor_arg(orch_args.tensor(2));
    Tensor ext_wk = from_tensor_arg(orch_args.tensor(3));
    Tensor ext_wv = from_tensor_arg(orch_args.tensor(4));
    Tensor ext_seq_lens = from_tensor_arg(orch_args.tensor(5));
    Tensor ext_rope_cos = from_tensor_arg(orch_args.tensor(6));
    Tensor ext_rope_sin = from_tensor_arg(orch_args.tensor(7));
    Tensor ext_k_cache = from_tensor_arg(orch_args.tensor(8));
    Tensor ext_v_cache = from_tensor_arg(orch_args.tensor(9));
    Tensor ext_wo = from_tensor_arg(orch_args.tensor(10));
    Tensor ext_post_rms_weight = from_tensor_arg(orch_args.tensor(11));
    Tensor ext_w_gate = from_tensor_arg(orch_args.tensor(12));
    Tensor ext_w_up = from_tensor_arg(orch_args.tensor(13));
    Tensor ext_w_down = from_tensor_arg(orch_args.tensor(14));
    Tensor ext_out = from_tensor_arg(orch_args.tensor(15));

    PTO2_SCOPE(PTO2ScopeMode::MANUAL) {
        uint32_t q_proj_ci_shapes[2] = {QWEN3_32B_BATCH_PADDED, QWEN3_32B_HIDDEN}; /* 16,8192 */
        TensorCreateInfo q_proj_ci(q_proj_ci_shapes, 2, DataType::FLOAT32);
        uint32_t k_proj_ci_shapes[2] = {QWEN3_32B_BATCH_PADDED, QWEN3_32B_KV_HIDDEN}; /* 16,1024 */
        TensorCreateInfo k_proj_ci(k_proj_ci_shapes, 2, DataType::FLOAT32);
        uint32_t v_proj_ci_shapes[2] = {QWEN3_32B_BATCH_PADDED, QWEN3_32B_KV_HIDDEN}; /* 16,1024 */
        TensorCreateInfo v_proj_ci(v_proj_ci_shapes, 2, DataType::FLOAT32);
        uint32_t normed_states_ci_shapes[2] = {QWEN3_32B_BATCH_PADDED, QWEN3_32B_HIDDEN}; /* 16,8192 */
        TensorCreateInfo normed_states_ci(normed_states_ci_shapes, 2, DataType::BFLOAT16);
        uint32_t all_q_padded_ci_shapes[2] = {QWEN3_32B_ALL_Q_PADDED_ROWS, QWEN3_32B_HEAD_DIM}; /* 2048,128 */
        TensorCreateInfo all_q_padded_ci(all_q_padded_ci_shapes, 2, DataType::BFLOAT16);
        uint32_t attn_out_ci_shapes[2] = {QWEN3_32B_BATCH_PADDED, QWEN3_32B_HIDDEN}; /* 16,8192 */
        TensorCreateInfo attn_out_ci(attn_out_ci_shapes, 2, DataType::BFLOAT16);
        uint32_t resid1_tile_ci_shapes[2] = {QWEN3_32B_BATCH_PADDED, QWEN3_32B_HIDDEN}; /* 16,8192 */
        TensorCreateInfo resid1_tile_ci(resid1_tile_ci_shapes, 2, DataType::FLOAT32);
        uint32_t post_norm_tile_ci_shapes[2] = {QWEN3_32B_BATCH_PADDED, QWEN3_32B_HIDDEN}; /* 16,8192 */
        TensorCreateInfo post_norm_tile_ci(post_norm_tile_ci_shapes, 2, DataType::BFLOAT16);
        uint32_t mlp_tile_ci_shapes[2] = {QWEN3_32B_BATCH_PADDED, QWEN3_32B_INTERMEDIATE}; /* 16,25600 */
        TensorCreateInfo mlp_tile_ci(mlp_tile_ci_shapes, 2, DataType::BFLOAT16);
        TaskOutputTensors alloc_0 = alloc_tensors(q_proj_ci, k_proj_ci, v_proj_ci, normed_states_ci, all_q_padded_ci, attn_out_ci, resid1_tile_ci, post_norm_tile_ci, mlp_tile_ci);
        const Tensor& q_proj = alloc_0.get_ref(0);
        const Tensor& k_proj = alloc_0.get_ref(1);
        const Tensor& v_proj = alloc_0.get_ref(2);
        const Tensor& normed_states = alloc_0.get_ref(3);
        const Tensor& all_q_padded = alloc_0.get_ref(4);
        const Tensor& attn_out = alloc_0.get_ref(5);
        const Tensor& resid1_tile = alloc_0.get_ref(6);
        const Tensor& post_norm_tile = alloc_0.get_ref(7);
        const Tensor& mlp_tile = alloc_0.get_ref(8);

        // Task 0: rmsnorm (dep: alloc_0)
        Arg params_t0;
        params_t0.add_input(ext_hidden_states);
        params_t0.add_output(normed_states);
        params_t0.add_input(ext_input_rms_weight);
        params_t0.add_dep(alloc_0.task_id());
        TaskOutputTensors t0_outs = rt_submit_aiv_task(0, params_t0);
        const Tensor& normed_states__rv_v2 = normed_states;

        // Task 1: q_proj loop (dep: alloc_0, t0)
        std::vector<PTO2TaskId> t1_task_ids;
        for (int64_t q0 = 0; q0 < QWEN3_32B_HIDDEN; q0 += QWEN3_32B_Q_OUT_CHUNK) { /* 8192,256 */
            Arg params_t1;
            params_t1.add_input(normed_states__rv_v2);
            params_t1.add_input(ext_wq);
            params_t1.add_output(q_proj);
            params_t1.add_scalar(q0);
            params_t1.add_dep(alloc_0.task_id());
            params_t1.add_dep(t0_outs.task_id());
            TaskOutputTensors t1_outs = rt_submit_aic_task(1, params_t1);
            t1_task_ids.push_back(t1_outs.task_id());
        }

        // Task 2: kv_proj loop (dep: alloc_0, t0)
        std::vector<PTO2TaskId> t2_task_ids;
        for (int64_t kv0 = 0; kv0 < QWEN3_32B_KV_HIDDEN; kv0 += QWEN3_32B_KV_OUT_CHUNK) { /* 1024,256 */
            Arg params_t2;
            params_t2.add_input(normed_states__rv_v2);
            params_t2.add_input(ext_wk);
            params_t2.add_input(ext_wv);
            params_t2.add_output(k_proj);
            params_t2.add_output(v_proj);
            params_t2.add_scalar(kv0);
            params_t2.add_dep(alloc_0.task_id());
            params_t2.add_dep(t0_outs.task_id());
            TaskOutputTensors t2_outs = rt_submit_aic_task(2, params_t2);
            t2_task_ids.push_back(t2_outs.task_id());
        }

        // Per-batch attention loop
        std::vector<PTO2TaskId> t7_task_ids;
        for (int64_t b = 0; b < QWEN3_32B_USER_BATCH; b += 1) { /* 16 */
            size_t idx_ctx_len = b;
            int32_t ctx_len = static_cast<int32_t*>(orch_args.tensor(5).data_as<void>())[idx_ctx_len];
            int64_t pos = (static_cast<int64_t>(ctx_len) - 1);
            int64_t ctx_blocks = ((static_cast<int64_t>(ctx_len) + (QWEN3_32B_SEQ_TILE - 1)) / QWEN3_32B_SEQ_TILE); /* +255/256 */
            uint32_t cos_lo_shapes[2] = {1, QWEN3_32B_HALF_DIM}; /* 64 */
            uint32_t cos_lo_offsets[2] = {static_cast<uint32_t>(pos), 0};
            Tensor cos_lo = ext_rope_cos.view(cos_lo_shapes, cos_lo_offsets);
            uint32_t cos_hi_shapes[2] = {1, QWEN3_32B_HALF_DIM}; /* 64 */
            uint32_t cos_hi_offsets[2] = {static_cast<uint32_t>(pos), QWEN3_32B_HALF_DIM}; /* 64 */
            Tensor cos_hi = ext_rope_cos.view(cos_hi_shapes, cos_hi_offsets);
            uint32_t sin_lo_shapes[2] = {1, QWEN3_32B_HALF_DIM}; /* 64 */
            uint32_t sin_lo_offsets[2] = {static_cast<uint32_t>(pos), 0};
            Tensor sin_lo = ext_rope_sin.view(sin_lo_shapes, sin_lo_offsets);
            uint32_t sin_hi_shapes[2] = {1, QWEN3_32B_HALF_DIM}; /* 64 */
            uint32_t sin_hi_offsets[2] = {static_cast<uint32_t>(pos), QWEN3_32B_HALF_DIM}; /* 64 */
            Tensor sin_hi = ext_rope_sin.view(sin_hi_shapes, sin_hi_offsets);

            // Task 3: rope_kv_cache (dep: all t1, all t2 — 36 explicit deps)
            Arg params_t3;
            params_t3.add_output(all_q_padded);
            params_t3.add_output(ext_k_cache);
            params_t3.add_output(ext_v_cache);
            params_t3.add_input(k_proj);
            params_t3.add_input(cos_lo);
            params_t3.add_input(sin_lo);
            params_t3.add_input(cos_hi);
            params_t3.add_input(sin_hi);
            params_t3.add_input(v_proj);
            params_t3.add_input(q_proj);
            params_t3.add_scalar(b);
            params_t3.add_scalar(pos);
            for (PTO2TaskId tid : t1_task_ids) params_t3.add_dep(tid);
            for (PTO2TaskId tid : t2_task_ids) params_t3.add_dep(tid);
            TaskOutputTensors t3_outs = rt_submit_aiv_task(3, params_t3);

            const Tensor& all_q_padded__rv_v4 = all_q_padded;
            const Tensor& k_cache__rv_v4 = ext_k_cache;
            const Tensor& v_cache__rv_v4 = ext_v_cache;
            uint32_t attn_row_shapes[2] = {1, QWEN3_32B_HIDDEN}; /* 8192 */
            uint32_t attn_row_offsets[2] = {static_cast<uint32_t>(b), 0};
            Tensor attn_row = attn_out.view(attn_row_shapes, attn_row_offsets);

            for (int64_t gi = 0; gi < QWEN3_32B_NUM_KV_HEADS; gi += 2) { /* 8 */
                uint32_t all_raw_scores0_ci_shapes[2] = {QWEN3_32B_SEQ_TILE, QWEN3_32B_SEQ_TILE}; /* 256,256 */
                TensorCreateInfo all_raw_scores0_ci(all_raw_scores0_ci_shapes, 2, DataType::FLOAT32);
                uint32_t all_raw_scores1_ci_shapes[2] = {QWEN3_32B_SEQ_TILE, QWEN3_32B_SEQ_TILE}; /* 256,256 */
                TensorCreateInfo all_raw_scores1_ci(all_raw_scores1_ci_shapes, 2, DataType::FLOAT32);
                uint32_t all_exp_padded0_ci_shapes[2] = {QWEN3_32B_SEQ_TILE, QWEN3_32B_SEQ_TILE}; /* 256,256 */
                TensorCreateInfo all_exp_padded0_ci(all_exp_padded0_ci_shapes, 2, DataType::BFLOAT16);
                uint32_t all_cur_li0_ci_shapes[2] = {QWEN3_32B_SOFTMAX_ACC_DIM, 1}; /* 128,1 */
                TensorCreateInfo all_cur_li0_ci(all_cur_li0_ci_shapes, 2, DataType::FLOAT32);
                uint32_t all_cur_mi0_ci_shapes[2] = {QWEN3_32B_SOFTMAX_ACC_DIM, 1}; /* 128,1 */
                TensorCreateInfo all_cur_mi0_ci(all_cur_mi0_ci_shapes, 2, DataType::FLOAT32);
                uint32_t all_exp_padded1_ci_shapes[2] = {QWEN3_32B_SEQ_TILE, QWEN3_32B_SEQ_TILE}; /* 256,256 */
                TensorCreateInfo all_exp_padded1_ci(all_exp_padded1_ci_shapes, 2, DataType::BFLOAT16);
                uint32_t all_cur_li1_ci_shapes[2] = {QWEN3_32B_SOFTMAX_ACC_DIM, 1}; /* 128,1 */
                TensorCreateInfo all_cur_li1_ci(all_cur_li1_ci_shapes, 2, DataType::FLOAT32);
                uint32_t all_cur_mi1_ci_shapes[2] = {QWEN3_32B_SOFTMAX_ACC_DIM, 1}; /* 128,1 */
                TensorCreateInfo all_cur_mi1_ci(all_cur_mi1_ci_shapes, 2, DataType::FLOAT32);
                uint32_t all_oi_tmp0_ci_shapes[2] = {QWEN3_32B_SEQ_TILE, QWEN3_32B_HEAD_DIM}; /* 256,128 */
                TensorCreateInfo all_oi_tmp0_ci(all_oi_tmp0_ci_shapes, 2, DataType::FLOAT32);
                uint32_t all_oi_tmp1_ci_shapes[2] = {QWEN3_32B_SEQ_TILE, QWEN3_32B_HEAD_DIM}; /* 256,128 */
                TensorCreateInfo all_oi_tmp1_ci(all_oi_tmp1_ci_shapes, 2, DataType::FLOAT32);
                TaskOutputTensors alloc_1 = alloc_tensors(all_raw_scores0_ci, all_raw_scores1_ci, all_exp_padded0_ci, all_cur_li0_ci, all_cur_mi0_ci, all_exp_padded1_ci, all_cur_li1_ci, all_cur_mi1_ci, all_oi_tmp0_ci, all_oi_tmp1_ci);
                const Tensor& all_raw_scores0 = alloc_1.get_ref(0);
                const Tensor& all_raw_scores1 = alloc_1.get_ref(1);
                const Tensor& all_exp_padded0 = alloc_1.get_ref(2);
                const Tensor& all_cur_li0 = alloc_1.get_ref(3);
                const Tensor& all_cur_mi0 = alloc_1.get_ref(4);
                const Tensor& all_exp_padded1 = alloc_1.get_ref(5);
                const Tensor& all_cur_li1 = alloc_1.get_ref(6);
                const Tensor& all_cur_mi1 = alloc_1.get_ref(7);
                const Tensor& all_oi_tmp0 = alloc_1.get_ref(8);
                const Tensor& all_oi_tmp1 = alloc_1.get_ref(9);
                int64_t gi0 = gi;
                int64_t gi1 = (gi + 1);
                int64_t kvh0 = gi0;
                int64_t qg0 = (gi0 - kvh0);
                int64_t q_base0 = ((kvh0 + qg0) * QWEN3_32B_Q_PER_KV); /* *8 */
                int64_t q_pad_row0_0 = ((b * QWEN3_32B_ALL_Q_PAD_STRIDE) + (gi0 * QWEN3_32B_Q_HEAD_PAD)); /* *128,*16 */
                uint32_t q_padded0_shapes[2] = {QWEN3_32B_BATCH_TILE, QWEN3_32B_HEAD_DIM}; /* 16,128 */
                uint32_t q_padded0_offsets[2] = {static_cast<uint32_t>(q_pad_row0_0), 0};
                Tensor q_padded0 = all_q_padded__rv_v4.view(q_padded0_shapes, q_padded0_offsets);
                int64_t kvh1 = gi1;
                int64_t qg1 = (gi1 - kvh1);
                int64_t q_base1 = ((kvh1 + qg1) * QWEN3_32B_Q_PER_KV); /* *8 */
                int64_t q_pad_row0_1 = ((b * QWEN3_32B_ALL_Q_PAD_STRIDE) + (gi1 * QWEN3_32B_Q_HEAD_PAD)); /* *128,*16 */
                uint32_t q_padded1_shapes[2] = {QWEN3_32B_BATCH_TILE, QWEN3_32B_HEAD_DIM}; /* 16,128 */
                uint32_t q_padded1_offsets[2] = {static_cast<uint32_t>(q_pad_row0_1), 0};
                Tensor q_padded1 = all_q_padded__rv_v4.view(q_padded1_shapes, q_padded1_offsets);

                // Task 4: qk_matmul (dep: alloc_1, t3)
                Arg params_t4;
                params_t4.add_output(all_raw_scores0);
                params_t4.add_output(all_raw_scores1);
                params_t4.add_input(k_cache__rv_v4);
                params_t4.add_input(q_padded0);
                params_t4.add_input(q_padded1);
                params_t4.add_scalar(ctx_blocks);
                params_t4.add_scalar(b);
                params_t4.add_scalar(kvh0);
                params_t4.add_scalar(kvh1);
                params_t4.add_dep(alloc_1.task_id());
                params_t4.add_dep(t3_outs.task_id());
                TaskOutputTensors t4_outs = rt_submit_aic_task(4, params_t4);

                // Task 5: softmax (dep: t4)
                Arg params_t5;
                params_t5.add_output(all_cur_li0);
                params_t5.add_output(all_cur_li1);
                params_t5.add_output(all_cur_mi0);
                params_t5.add_output(all_cur_mi1);
                params_t5.add_output(all_exp_padded0);
                params_t5.add_output(all_exp_padded1);
                params_t5.add_input(all_raw_scores0);
                params_t5.add_input(all_raw_scores1);
                params_t5.add_scalar(ctx_blocks);
                params_t5.add_scalar(ctx_len);
                params_t5.add_dep(t4_outs.task_id());
                TaskOutputTensors t5_outs = rt_submit_aiv_task(5, params_t5);

                // Task 6: sv_matmul (dep: t5)
                Arg params_t6;
                params_t6.add_output(all_oi_tmp0);
                params_t6.add_output(all_oi_tmp1);
                params_t6.add_input(all_exp_padded0);
                params_t6.add_input(v_cache__rv_v4);
                params_t6.add_input(all_exp_padded1);
                params_t6.add_scalar(ctx_blocks);
                params_t6.add_scalar(b);
                params_t6.add_scalar(kvh0);
                params_t6.add_scalar(kvh1);
                params_t6.add_dep(t5_outs.task_id());
                TaskOutputTensors t6_outs = rt_submit_aic_task(6, params_t6);

                // Task 7: online_softmax (dep: t6) — writes attn_row slice
                Arg params_t7;
                params_t7.add_input(all_oi_tmp0);
                params_t7.add_input(all_cur_mi0);
                params_t7.add_input(all_cur_li0);
                params_t7.add_input(all_oi_tmp1);
                params_t7.add_input(all_cur_mi1);
                params_t7.add_input(all_cur_li1);
                params_t7.add_output(attn_row);
                params_t7.add_scalar(ctx_blocks);
                params_t7.add_scalar(q_base0);
                params_t7.add_scalar(q_base1);
                params_t7.add_dep(t6_outs.task_id());
                TaskOutputTensors t7_outs = rt_submit_aiv_task(7, params_t7);
                t7_task_ids.push_back(t7_outs.task_id());
            }
        }

        // out_proj_residual loop (dep: alloc_2 + all 64 t7 = 65 explicit deps)
        std::vector<PTO2TaskId> t8_task_ids;
        for (int64_t ob = 0; ob < QWEN3_32B_OUT_PROJ_OB_MAX; ob += QWEN3_32B_OUT_PROJ_OB_STRIDE) { /* 32, step 2 */
            uint32_t gm_pipe_buffer_0_ci_shapes[1] = {QWEN3_32B_GM_PIPE_NUMEL}; /* 32768 */
            TensorCreateInfo gm_pipe_buffer_0_ci(gm_pipe_buffer_0_ci_shapes, 1, DataType::FLOAT32, /*manual_dep=*/true);
            TaskOutputTensors alloc_2 = alloc_tensors(gm_pipe_buffer_0_ci);
            const Tensor& gm_pipe_buffer_0 = alloc_2.get_ref(0);

            Arg params_t8;
            params_t8.add_output(resid1_tile);
            params_t8.add_input(ext_hidden_states);
            params_t8.add_input(attn_out);
            params_t8.add_input(ext_wo);
            params_t8.add_output(gm_pipe_buffer_0);
            params_t8.add_scalar(ob);
            params_t8.add_dep(alloc_2.task_id());
            for (PTO2TaskId tid : t7_task_ids) params_t8.add_dep(tid);
            MixedKernels mixed_8 = {8, 9, 9};
            TaskOutputTensors t8_outs = rt_submit_task(mixed_8, params_t8);
            t8_task_ids.push_back(t8_outs.task_id());
        }

        // Task 9: post_rmsnorm (dep: all 16 t8 = 16 explicit deps)
        Arg params_t9;
        params_t9.add_input(resid1_tile);
        params_t9.add_output(post_norm_tile);
        params_t9.add_input(ext_post_rms_weight);
        for (PTO2TaskId tid : t8_task_ids) params_t9.add_dep(tid);
        TaskOutputTensors t9_outs = rt_submit_aiv_task(10, params_t9);
        const Tensor& post_norm_tile__rv_v2 = post_norm_tile;

        // MLP gate/up/silu loop
        std::vector<PTO2TaskId> t12_task_ids;
        for (int64_t o0 = 0; o0 < QWEN3_32B_INTERMEDIATE; o0 += QWEN3_32B_MLP_OUT_CHUNK) { /* 25600,256 */
            uint32_t ret0__out_ci_shapes[2] = {QWEN3_32B_BATCH_TILE, QWEN3_32B_MLP_OUT_CHUNK}; /* 16,256 */
            TensorCreateInfo ret0__out_ci(ret0__out_ci_shapes, 2, DataType::FLOAT32);
            uint32_t ret0__out_1_ci_shapes[2] = {QWEN3_32B_BATCH_TILE, QWEN3_32B_MLP_OUT_CHUNK}; /* 16,256 */
            TensorCreateInfo ret0__out_1_ci(ret0__out_1_ci_shapes, 2, DataType::FLOAT32);
            TaskOutputTensors alloc_3 = alloc_tensors(ret0__out_ci, ret0__out_1_ci);
            const Tensor& ret0__out = alloc_3.get_ref(0);
            const Tensor& ret0__out_1 = alloc_3.get_ref(1);

            // Task 10: gate_proj (dep: alloc_3, t9)
            Arg params_t10;
            params_t10.add_input(post_norm_tile__rv_v2);
            params_t10.add_input(ext_w_gate);
            params_t10.add_output(ret0__out);
            params_t10.add_scalar(o0);
            params_t10.add_dep(alloc_3.task_id());
            params_t10.add_dep(t9_outs.task_id());
            TaskOutputTensors t10_outs = rt_submit_aic_task(11, params_t10);
            const Tensor& gate_acc = ret0__out;

            // Task 11: up_proj (dep: alloc_3, t9)
            Arg params_t11;
            params_t11.add_input(post_norm_tile__rv_v2);
            params_t11.add_input(ext_w_up);
            params_t11.add_output(ret0__out_1);
            params_t11.add_scalar(o0);
            params_t11.add_dep(alloc_3.task_id());
            params_t11.add_dep(t9_outs.task_id());
            TaskOutputTensors t11_outs = rt_submit_aic_task(12, params_t11);
            const Tensor& up_acc = ret0__out_1;

            // Task 12: silu (dep: t10, t11) — writes mlp_tile slice
            Arg params_t12;
            params_t12.add_input(gate_acc);
            params_t12.add_input(up_acc);
            params_t12.add_output(mlp_tile);
            params_t12.add_scalar(o0);
            params_t12.add_dep(t10_outs.task_id());
            params_t12.add_dep(t11_outs.task_id());
            TaskOutputTensors t12_outs = rt_submit_aiv_task(13, params_t12);
            t12_task_ids.push_back(t12_outs.task_id());
        }

        // down_proj_residual loop (dep: alloc_4 + all 100 t12 = 101 explicit deps)
        for (int64_t db = 0; db < QWEN3_32B_DOWN_MIXED_OB_MAX; db += QWEN3_32B_DOWN_MIXED_OB_STRIDE) { /* 32, step 2 */
            uint32_t gm_pipe_buffer_1_ci_shapes[1] = {QWEN3_32B_GM_PIPE_NUMEL}; /* 32768 */
            TensorCreateInfo gm_pipe_buffer_1_ci(gm_pipe_buffer_1_ci_shapes, 1, DataType::FLOAT32, /*manual_dep=*/true);
            TaskOutputTensors alloc_4 = alloc_tensors(gm_pipe_buffer_1_ci);
            const Tensor& gm_pipe_buffer_1 = alloc_4.get_ref(0);

            Arg params_t13;
            params_t13.add_output(ext_out);
            params_t13.add_input(resid1_tile);
            params_t13.add_input(mlp_tile);
            params_t13.add_input(ext_w_down);
            params_t13.add_output(gm_pipe_buffer_1);
            params_t13.add_scalar(db);
            params_t13.add_dep(alloc_4.task_id());
            for (PTO2TaskId tid : t12_task_ids) params_t13.add_dep(tid);
            MixedKernels mixed_13 = {14, 15, 15};
            rt_submit_task(mixed_13, params_t13);
        }
    }
}

}  // extern "C"
