// Orchestration Function: qwen3_decode (manual-scope explicit-deps variant).
// Derived from spmd/Qwen3Decode/orchestration/qwen3_decode.cpp with explicit deps.

#include "runtime.h"
#include <iostream>

#include <stddef.h>
#include <stdint.h>
#include <stdio.h>

#include "pto_orchestration_api.h"

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
    auto __chain_dep = [&](auto& __a) {
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
        uint32_t all_q_padded_ci_shapes[2] = {5760, 128};
        TensorCreateInfo all_q_padded_ci(all_q_padded_ci_shapes, 2, DataType::BFLOAT16);
        TaskOutputTensors alloc_0 = alloc_tensors(all_q_padded_ci);
        const Tensor& all_q_padded = alloc_0.get_ref(0);
        int64_t user_batch = 45;
        int64_t batch_padded = (((user_batch + 15) / 16) * 16);
        uint32_t q_proj_ci_shapes[2] = {static_cast<uint32_t>(batch_padded), 5120};
        TensorCreateInfo q_proj_ci(q_proj_ci_shapes, 2, DataType::FLOAT32);
        TaskOutputTensors alloc_1 = alloc_tensors(q_proj_ci);
        const Tensor& q_proj = alloc_1.get_ref(0);
        uint32_t k_proj_ci_shapes[2] = {static_cast<uint32_t>(batch_padded), 1024};
        TensorCreateInfo k_proj_ci(k_proj_ci_shapes, 2, DataType::FLOAT32);
        TaskOutputTensors alloc_2 = alloc_tensors(k_proj_ci);
        const Tensor& k_proj = alloc_2.get_ref(0);
        uint32_t v_proj_ci_shapes[2] = {static_cast<uint32_t>(batch_padded), 1024};
        TensorCreateInfo v_proj_ci(v_proj_ci_shapes, 2, DataType::FLOAT32);
        TaskOutputTensors alloc_3 = alloc_tensors(v_proj_ci);
        const Tensor& v_proj = alloc_3.get_ref(0);
        uint32_t q_proj_norm_ci_shapes[2] = {static_cast<uint32_t>(batch_padded), 5120};
        TensorCreateInfo q_proj_norm_ci(q_proj_norm_ci_shapes, 2, DataType::FLOAT32);
        TaskOutputTensors alloc_4 = alloc_tensors(q_proj_norm_ci);
        const Tensor& q_proj_norm = alloc_4.get_ref(0);
        uint32_t k_proj_norm_ci_shapes[2] = {static_cast<uint32_t>(batch_padded), 1024};
        TensorCreateInfo k_proj_norm_ci(k_proj_norm_ci_shapes, 2, DataType::FLOAT32);
        TaskOutputTensors alloc_5 = alloc_tensors(k_proj_norm_ci);
        const Tensor& k_proj_norm = alloc_5.get_ref(0);
        for (int64_t b0 = 0; b0 < batch_padded; b0 += 16) {
                uint32_t normed_tile_ci_shapes[2] = {16, 5120};
                TensorCreateInfo normed_tile_ci(normed_tile_ci_shapes, 2, DataType::BFLOAT16);
                TaskOutputTensors alloc_6 = alloc_tensors(normed_tile_ci);
                const Tensor& normed_tile = alloc_6.get_ref(0);
                int64_t cur_valid = std::min<int64_t>((user_batch - b0), 16);

                // Task 0: rmsnorm
                ArgWithDeps<256> params_t0;
                params_t0.add_input(ext_hidden_states);
                params_t0.add_output(normed_tile);
                params_t0.add_input(ext_input_rms_weight);
                params_t0.add_scalar(b0);
                params_t0.add_scalar(cur_valid);
                __chain_dep(params_t0);
                TaskOutputTensors __rt_manual_0 = rt_submit_aiv_task(0, params_t0);
                __prev = __rt_manual_0.task_id();
                const Tensor& normed_tile__rv_v2 = normed_tile;

                // Spmd q_proj_spmd: qwen3_decode_incore_1
                ArgWithDeps<256> params_t1;
                params_t1.add_input(normed_tile__rv_v2);
                params_t1.add_input(ext_wq);
                params_t1.add_output(q_proj);
                params_t1.add_scalar(b0);
                params_t1.launch_spec.set_block_num(20);
                __chain_dep(params_t1);
                TaskOutputTensors __rt_manual_1 = rt_submit_aic_task(1, params_t1);
                __prev = __rt_manual_1.task_id();
                const Tensor& q_proj__ssa_v3 = q_proj;

                // Spmd k_proj_spmd: qwen3_decode_incore_2
                ArgWithDeps<256> params_t2;
                params_t2.add_input(normed_tile__rv_v2);
                params_t2.add_input(ext_wk);
                params_t2.add_output(k_proj);
                params_t2.add_scalar(b0);
                params_t2.launch_spec.set_block_num(8);
                __chain_dep(params_t2);
                TaskOutputTensors __rt_manual_2 = rt_submit_aic_task(2, params_t2);
                __prev = __rt_manual_2.task_id();
                const Tensor& k_proj__ssa_v3 = k_proj;

                // Spmd v_proj_spmd: qwen3_decode_incore_3
                ArgWithDeps<256> params_t3;
                params_t3.add_input(normed_tile__rv_v2);
                params_t3.add_input(ext_wv);
                params_t3.add_output(v_proj);
                params_t3.add_scalar(b0);
                params_t3.launch_spec.set_block_num(8);
                __chain_dep(params_t3);
                TaskOutputTensors __rt_manual_3 = rt_submit_aic_task(3, params_t3);
                __prev = __rt_manual_3.task_id();
                const Tensor& v_proj__ssa_v3 = v_proj;
        }
        for (int64_t b0 = 0; b0 < batch_padded; b0 += 16) {

                // Task 4: qk_norm
                ArgWithDeps<256> params_t4;
                params_t4.add_output(k_proj_norm);
                params_t4.add_output(q_proj_norm);
                params_t4.add_input(q_proj);
                params_t4.add_input(ext_q_norm_weight);
                params_t4.add_input(k_proj);
                params_t4.add_input(ext_k_norm_weight);
                params_t4.add_scalar(b0);
                __chain_dep(params_t4);
                TaskOutputTensors __rt_manual_4 = rt_submit_aiv_task(4, params_t4);
                __prev = __rt_manual_4.task_id();
                const Tensor& k_proj_norm__rv_v4 = k_proj_norm;
                const Tensor& q_proj_norm__rv_v4 = q_proj_norm;
        }
        uint32_t attn_out_ci_shapes[2] = {static_cast<uint32_t>(batch_padded), 5120};
        TensorCreateInfo attn_out_ci(attn_out_ci_shapes, 2, DataType::BFLOAT16);
        TaskOutputTensors alloc_7 = alloc_tensors(attn_out_ci);
        const Tensor& attn_out = alloc_7.get_ref(0);
        for (int64_t b = 0; b < user_batch; b += 1) {
                uint32_t all_raw_scores_ci_shapes[2] = {4096, 128};
                TensorCreateInfo all_raw_scores_ci(all_raw_scores_ci_shapes, 2, DataType::FLOAT32);
                uint32_t all_exp_padded_ci_shapes[2] = {4096, 128};
                TensorCreateInfo all_exp_padded_ci(all_exp_padded_ci_shapes, 2, DataType::BFLOAT16);
                uint32_t all_cur_mi_ci_shapes[2] = {4096, 1};
                TensorCreateInfo all_cur_mi_ci(all_cur_mi_ci_shapes, 2, DataType::FLOAT32);
                uint32_t all_cur_li_ci_shapes[2] = {4096, 1};
                TensorCreateInfo all_cur_li_ci(all_cur_li_ci_shapes, 2, DataType::FLOAT32);
                uint32_t all_oi_tmp_ci_shapes[2] = {4096, 128};
                TensorCreateInfo all_oi_tmp_ci(all_oi_tmp_ci_shapes, 2, DataType::FLOAT32);
                TaskOutputTensors alloc_8 = alloc_tensors(all_raw_scores_ci, all_exp_padded_ci, all_cur_mi_ci, all_cur_li_ci, all_oi_tmp_ci);
                const Tensor& all_raw_scores = alloc_8.get_ref(0);
                const Tensor& all_exp_padded = alloc_8.get_ref(1);
                const Tensor& all_cur_mi = alloc_8.get_ref(2);
                const Tensor& all_cur_li = alloc_8.get_ref(3);
                const Tensor& all_oi_tmp = alloc_8.get_ref(4);
                size_t idx_ctx_len = b;
                int32_t ctx_len = static_cast<int32_t*>(orch_args.tensor(7).data_as<void>())[idx_ctx_len];
                int64_t pos = (static_cast<int64_t>(ctx_len) - 1);
                int64_t ctx_blocks = ((static_cast<int64_t>(ctx_len) + 127) / 128);
                int64_t block_table_base = (b * 32);
                size_t idx_slot = b;
                int32_t slot = static_cast<int32_t*>(orch_args.tensor(9).data_as<void>())[idx_slot];
                int64_t slot_block = (static_cast<int64_t>(slot) / 128);
                int64_t slot_offset = (static_cast<int64_t>(slot) - (slot_block * 128));
                uint32_t cos_row_shapes[2] = {1, 128};
                uint32_t cos_row_offsets[2] = {static_cast<uint32_t>(pos), 0};
                Tensor cos_row = ext_rope_cos.view(cos_row_shapes, cos_row_offsets);
                uint32_t sin_row_shapes[2] = {1, 128};
                uint32_t sin_row_offsets[2] = {static_cast<uint32_t>(pos), 0};
                Tensor sin_row = ext_rope_sin.view(sin_row_shapes, sin_row_offsets);
                uint32_t cos_lo_shapes[2] = {1, 64};
                uint32_t cos_lo_offsets[2] = {0, 0};
                Tensor cos_lo = cos_row.view(cos_lo_shapes, cos_lo_offsets);
                uint32_t cos_hi_shapes[2] = {1, 64};
                uint32_t cos_hi_offsets[2] = {0, 64};
                Tensor cos_hi = cos_row.view(cos_hi_shapes, cos_hi_offsets);
                uint32_t sin_lo_shapes[2] = {1, 64};
                uint32_t sin_lo_offsets[2] = {0, 0};
                Tensor sin_lo = sin_row.view(sin_lo_shapes, sin_lo_offsets);
                uint32_t sin_hi_shapes[2] = {1, 64};
                uint32_t sin_hi_offsets[2] = {0, 64};
                Tensor sin_hi = sin_row.view(sin_hi_shapes, sin_hi_offsets);

                // Task 5: rope_kv_cache
                ArgWithDeps<256> params_t5;
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
                __chain_dep(params_t5);
                TaskOutputTensors __rt_manual_5 = rt_submit_aiv_task(5, params_t5);
                __prev = __rt_manual_5.task_id();
                const Tensor& all_q_padded__rv_v4 = all_q_padded;
                const Tensor& k_cache__rv_v4 = ext_k_cache;
                const Tensor& v_cache__rv_v4 = ext_v_cache;
                uint32_t attn_row_shapes[2] = {1, 5120};
                uint32_t attn_row_offsets[2] = {static_cast<uint32_t>(b), 0};
                Tensor attn_row = attn_out.view(attn_row_shapes, attn_row_offsets);

                // Spmd qk_matmul: qwen3_decode_incore_6
                ArgWithDeps<256> params_t6;
                params_t6.add_input(all_q_padded__rv_v4);
                params_t6.add_output(all_raw_scores);
                params_t6.add_input(ext_block_table);
                params_t6.add_input(k_cache__rv_v4);
                params_t6.add_scalar(b);
                params_t6.add_scalar(ctx_blocks);
                params_t6.add_scalar(block_table_base);
                params_t6.launch_spec.set_block_num(4);
                __chain_dep(params_t6);
                TaskOutputTensors __rt_manual_6 = rt_submit_aic_task(6, params_t6);
                __prev = __rt_manual_6.task_id();
                const Tensor& all_raw_scores__rv_v2 = all_raw_scores;

                // Spmd softmax: qwen3_decode_incore_7
                ArgWithDeps<256> params_t7;
                params_t7.add_output(all_cur_li);
                params_t7.add_output(all_cur_mi);
                params_t7.add_output(all_exp_padded);
                params_t7.add_input(all_raw_scores__rv_v2);
                params_t7.add_scalar(ctx_blocks);
                params_t7.add_scalar(ctx_len);
                params_t7.launch_spec.set_block_num(4);
                __chain_dep(params_t7);
                TaskOutputTensors __rt_manual_7 = rt_submit_aiv_task(7, params_t7);
                __prev = __rt_manual_7.task_id();
                const Tensor& all_cur_li__rv_v2 = all_cur_li;
                const Tensor& all_cur_mi__rv_v2 = all_cur_mi;
                const Tensor& all_exp_padded__rv_v2 = all_exp_padded;

                // Spmd sv_matmul: qwen3_decode_incore_8
                ArgWithDeps<256> params_t8;
                params_t8.add_output(all_oi_tmp);
                params_t8.add_input(ext_block_table);
                params_t8.add_input(all_exp_padded__rv_v2);
                params_t8.add_input(v_cache__rv_v4);
                params_t8.add_scalar(ctx_blocks);
                params_t8.add_scalar(block_table_base);
                params_t8.launch_spec.set_block_num(4);
                __chain_dep(params_t8);
                TaskOutputTensors __rt_manual_8 = rt_submit_aic_task(8, params_t8);
                __prev = __rt_manual_8.task_id();
                const Tensor& all_oi_tmp__rv_v2 = all_oi_tmp;

                // Spmd online_softmax: qwen3_decode_incore_9
                ArgWithDeps<256> params_t9;
                params_t9.add_input(all_oi_tmp__rv_v2);
                params_t9.add_input(all_cur_mi__rv_v2);
                params_t9.add_input(all_cur_li__rv_v2);
                params_t9.add_inout(attn_row);
                params_t9.add_scalar(ctx_blocks);
                params_t9.launch_spec.set_block_num(4);
                __chain_dep(params_t9);
                TaskOutputTensors __rt_manual_9 = rt_submit_aiv_task(9, params_t9);
                __prev = __rt_manual_9.task_id();
        }
        for (int64_t b0 = 0; b0 < batch_padded; b0 += 16) {
                uint32_t resid1_tile_ci_shapes[2] = {16, 5120};
                TensorCreateInfo resid1_tile_ci(resid1_tile_ci_shapes, 2, DataType::FLOAT32);
                uint32_t post_norm_tile_ci_shapes[2] = {16, 5120};
                TensorCreateInfo post_norm_tile_ci(post_norm_tile_ci_shapes, 2, DataType::BFLOAT16);
                uint32_t mlp_tile_ci_shapes[2] = {16, 17408};
                TensorCreateInfo mlp_tile_ci(mlp_tile_ci_shapes, 2, DataType::BFLOAT16);
                uint32_t down_fp32_tile_ci_shapes[2] = {16, 5120};
                TensorCreateInfo down_fp32_tile_ci(down_fp32_tile_ci_shapes, 2, DataType::FLOAT32);
                TaskOutputTensors alloc_9 = alloc_tensors(resid1_tile_ci, post_norm_tile_ci, mlp_tile_ci, down_fp32_tile_ci);
                const Tensor& resid1_tile = alloc_9.get_ref(0);
                const Tensor& post_norm_tile = alloc_9.get_ref(1);
                const Tensor& mlp_tile = alloc_9.get_ref(2);
                const Tensor& down_fp32_tile = alloc_9.get_ref(3);
                int64_t cur_valid__ssa_v1 = std::min<int64_t>((user_batch - b0), 16);
                for (int64_t ob_pair = 0; ob_pair < 40; ob_pair += 2) {
                        uint32_t gm_pipe_buffer_0_ci_shapes[1] = {16384};
                        TensorCreateInfo gm_pipe_buffer_0_ci(gm_pipe_buffer_0_ci_shapes, 1, DataType::FLOAT32, /*manual_dep=*/true);
                        TaskOutputTensors alloc_10 = alloc_tensors(gm_pipe_buffer_0_ci);
                        const Tensor& gm_pipe_buffer_0 = alloc_10.get_ref(0);

                        // Group out_proj_residual: MixedKernels (AIC + AIV lanes)
                        ArgWithDeps<256> params_t10;
                        params_t10.add_output(resid1_tile);
                        params_t10.add_input(ext_hidden_states);
                        params_t10.add_input(attn_out);
                        params_t10.add_input(ext_wo);
                        params_t10.add_output(gm_pipe_buffer_0);
                        params_t10.add_scalar(ob_pair);
                        params_t10.add_scalar(b0);
                        params_t10.add_scalar(cur_valid__ssa_v1);
                        MixedKernels mixed_10 = {10, 11, 11};
                        __chain_dep(params_t10);
                        TaskOutputTensors __rt_manual_mixed_0 = rt_submit_task(mixed_10, params_t10);
                        __prev = __rt_manual_mixed_0.task_id();
                        const Tensor& resid1_tile__rv_v4 = resid1_tile;
                }

                // Task 11: post_rmsnorm
                ArgWithDeps<256> params_t11;
                params_t11.add_input(resid1_tile);
                params_t11.add_output(post_norm_tile);
                params_t11.add_input(ext_post_rms_weight);
                __chain_dep(params_t11);
                TaskOutputTensors __rt_manual_10 = rt_submit_aiv_task(12, params_t11);
                __prev = __rt_manual_10.task_id();
                const Tensor& post_norm_tile__rv_v2 = post_norm_tile;
                for (int64_t ob_base = 0; ob_base < 34; ob_base += 2) {
                        uint32_t gate_group_ci_shapes[2] = {16, 1024};
                        TensorCreateInfo gate_group_ci(gate_group_ci_shapes, 2, DataType::FLOAT32);
                        uint32_t up_group_ci_shapes[2] = {16, 1024};
                        TensorCreateInfo up_group_ci(up_group_ci_shapes, 2, DataType::FLOAT32);
                        TaskOutputTensors alloc_11 = alloc_tensors(gate_group_ci, up_group_ci);
                        const Tensor& gate_group = alloc_11.get_ref(0);
                        const Tensor& up_group = alloc_11.get_ref(1);

                        // Spmd gate_proj_spmd: qwen3_decode_incore_12
                        ArgWithDeps<256> params_t12;
                        params_t12.add_input(post_norm_tile__rv_v2);
                        params_t12.add_input(ext_w_gate);
                        params_t12.add_inout(gate_group);
                        params_t12.add_scalar(ob_base);
                        params_t12.launch_spec.set_block_num(2);
                        __chain_dep(params_t12);
                        TaskOutputTensors __rt_manual_11 = rt_submit_aic_task(13, params_t12);
                        __prev = __rt_manual_11.task_id();
                        const Tensor& gate_group__ssa_v1 = gate_group;

                        // Spmd up_proj_spmd: qwen3_decode_incore_13
                        ArgWithDeps<256> params_t13;
                        params_t13.add_input(post_norm_tile__rv_v2);
                        params_t13.add_input(ext_w_up);
                        params_t13.add_inout(up_group);
                        params_t13.add_scalar(ob_base);
                        params_t13.launch_spec.set_block_num(2);
                        __chain_dep(params_t13);
                        TaskOutputTensors __rt_manual_12 = rt_submit_aic_task(14, params_t13);
                        __prev = __rt_manual_12.task_id();
                        const Tensor& up_group__ssa_v1 = up_group;

                        // Spmd silu_spmd: qwen3_decode_incore_14
                        ArgWithDeps<256> params_t14;
                        params_t14.add_input(gate_group__ssa_v1);
                        params_t14.add_input(up_group__ssa_v1);
                        params_t14.add_output(mlp_tile);
                        params_t14.add_scalar(ob_base);
                        params_t14.launch_spec.set_block_num(2);
                        __chain_dep(params_t14);
                        TaskOutputTensors __rt_manual_13 = rt_submit_aiv_task(15, params_t14);
                        __prev = __rt_manual_13.task_id();
                        const Tensor& mlp_tile__ssa_v3 = mlp_tile;
                }

                // Spmd down_proj_spmd: qwen3_decode_incore_15
                ArgWithDeps<256> params_t15;
                params_t15.add_input(mlp_tile);
                params_t15.add_input(ext_w_down);
                params_t15.add_inout(down_fp32_tile);
                params_t15.launch_spec.set_block_num(40);
                __chain_dep(params_t15);
                TaskOutputTensors __rt_manual_14 = rt_submit_aic_task(16, params_t15);
                __prev = __rt_manual_14.task_id();
                const Tensor& down_fp32_tile__ssa_v1 = down_fp32_tile;

                // Spmd down_proj_residual_spmd: qwen3_decode_incore_16
                ArgWithDeps<256> params_t16;
                params_t16.add_input(down_fp32_tile__ssa_v1);
                params_t16.add_input(resid1_tile);
                params_t16.add_output(ext_out);
                params_t16.add_scalar(cur_valid__ssa_v1);
                params_t16.add_scalar(b0);
                params_t16.launch_spec.set_block_num(40);
                __chain_dep(params_t16);
                TaskOutputTensors __rt_manual_15 = rt_submit_aiv_task(17, params_t16);
                __prev = __rt_manual_15.task_id();
                const Tensor& out = ext_out;
        }
    }
}

}  // extern "C"
