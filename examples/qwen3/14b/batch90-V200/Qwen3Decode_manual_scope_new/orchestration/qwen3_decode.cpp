// Orchestration Function: qwen3_decode (Qwen3-14B batch90-V200 manual-scope explicit-deps variant).
// Derived from batch90-V200 auto emit; explicit deps follow batch45-new / batch90-V200 manual_scope patterns.

#include "runtime.h"
#include <iostream>
#include <vector>

#include <stddef.h>
#include <stdint.h>
#include <stdio.h>

#include "pto_orchestration_api.h"

// Manual explicit-dependency variant (PTO2_SCOPE(PTO2ScopeMode::MANUAL)).
// Inner PTO2_SCOPE() wrappers removed (AUTO nested in MANUAL is unsupported).
// SPMD rmsnorm (Func0) then per-tile Func1–3 depend only on Func0; Func4 qk_norm
// fans in Func1–3 for that tile only (no cross-tile dep between Func4 instances).
// Per batch b: Func5 rope_kv depends only on Func4 for tile(b/16) and all_q_padded alloc;
// Func6–9 chain per batch; Func6/8 also wait Func5. No cross-batch rope chaining.
// MLP: Func10/11 mixed out_proj waits online_softmax per row; Func12 depends only on Func10/11;
// Func13/14 depend only on Func12; Func15 silu per ob; Func16 fans in all Func15;
// Func17/18 depend only on Func16 with the same db.

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
        const int64_t user_batch = 90;
        const int64_t batch_padded = (((user_batch + 15) / 16) * 16);
        const int64_t num_tiles = batch_padded / 16;

        uint32_t all_q_padded_ci_shapes[2] = {11520, 128};
        TensorCreateInfo all_q_padded_ci(all_q_padded_ci_shapes, 2, DataType::BFLOAT16);
        TaskOutputTensors alloc_0 = alloc_tensors(all_q_padded_ci);
        __prev = alloc_0.task_id();
        const Tensor& all_q_padded = alloc_0.get_ref(0);
        const PTO2TaskId all_q_padded_alloc_task = alloc_0.task_id();

        uint32_t q_proj_ci_shapes[2] = {static_cast<uint32_t>(batch_padded), 5120};
        TensorCreateInfo q_proj_ci(q_proj_ci_shapes, 2, DataType::FLOAT32);
        TaskOutputTensors alloc_1 = alloc_tensors(q_proj_ci);
        __prev = alloc_1.task_id();
        const Tensor& q_proj = alloc_1.get_ref(0);

        uint32_t k_proj_ci_shapes[2] = {static_cast<uint32_t>(batch_padded), 1024};
        TensorCreateInfo k_proj_ci(k_proj_ci_shapes, 2, DataType::FLOAT32);
        TaskOutputTensors alloc_2 = alloc_tensors(k_proj_ci);
        __prev = alloc_2.task_id();
        const Tensor& k_proj = alloc_2.get_ref(0);

        uint32_t v_proj_ci_shapes[2] = {static_cast<uint32_t>(batch_padded), 1024};
        TensorCreateInfo v_proj_ci(v_proj_ci_shapes, 2, DataType::FLOAT32);
        TaskOutputTensors alloc_3 = alloc_tensors(v_proj_ci);
        __prev = alloc_3.task_id();
        const Tensor& v_proj = alloc_3.get_ref(0);

        uint32_t q_proj_norm_ci_shapes[2] = {static_cast<uint32_t>(batch_padded), 5120};
        TensorCreateInfo q_proj_norm_ci(q_proj_norm_ci_shapes, 2, DataType::FLOAT32);
        TaskOutputTensors alloc_4 = alloc_tensors(q_proj_norm_ci);
        __prev = alloc_4.task_id();
        const Tensor& q_proj_norm = alloc_4.get_ref(0);

        uint32_t k_proj_norm_ci_shapes[2] = {static_cast<uint32_t>(batch_padded), 1024};
        TensorCreateInfo k_proj_norm_ci(k_proj_norm_ci_shapes, 2, DataType::FLOAT32);
        TaskOutputTensors alloc_5 = alloc_tensors(k_proj_norm_ci);
        __prev = alloc_5.task_id();
        const Tensor& k_proj_norm = alloc_5.get_ref(0);

        uint32_t normed_full_ci_shapes[2] = {static_cast<uint32_t>(batch_padded), 5120};
        TensorCreateInfo normed_full_ci(normed_full_ci_shapes, 2, DataType::BFLOAT16);
        TaskOutputTensors alloc_6 = alloc_tensors(normed_full_ci);
        __prev = alloc_6.task_id();
        const Tensor& normed_full = alloc_6.get_ref(0);

        std::vector<PTO2TaskId> q_proj_task_per_tile(static_cast<size_t>(num_tiles), PTO2TaskId::invalid());
        std::vector<PTO2TaskId> k_proj_task_per_tile(static_cast<size_t>(num_tiles), PTO2TaskId::invalid());
        std::vector<PTO2TaskId> v_proj_task_per_tile(static_cast<size_t>(num_tiles), PTO2TaskId::invalid());
        std::vector<PTO2TaskId> qk_norm_task_per_tile(static_cast<size_t>(num_tiles), PTO2TaskId::invalid());
        std::vector<std::vector<PTO2TaskId>> online_softmax_tasks_by_b(static_cast<size_t>(user_batch));

        // Spmd rmsnorm: qwen3_decode_incore_0
        ArgWithDeps<256> params_t0;
        params_t0.add_input(ext_hidden_states);
        params_t0.add_output(normed_full);
        params_t0.add_input(ext_input_rms_weight);
        params_t0.add_scalar(user_batch);
        params_t0.launch_spec.set_block_num(6);
        __chain_dep(params_t0);
        TaskOutputTensors __rt_rms = rt_submit_aiv_task(0, params_t0);
        __prev = __rt_rms.task_id();
        const PTO2TaskId rmsnorm_id = __rt_rms.task_id();
        const Tensor& normed_full__rv_v2 = normed_full;
        Tensor q_proj__rv_v2 = q_proj;

        for (int64_t b0 = 0; b0 < batch_padded; b0 += 16) {
            const size_t tix = static_cast<size_t>(b0 / 16);
            uint32_t normed_tile_shapes[2] = {16, 5120};
            uint32_t normed_tile_offsets[2] = {static_cast<uint32_t>(b0), 0};
            Tensor normed_tile = normed_full__rv_v2.view(normed_tile_shapes, normed_tile_offsets);

            // Spmd q_proj: qwen3_decode_incore_1
            ArgWithDeps<256> params_t1;
            params_t1.add_input(normed_tile);
            params_t1.add_input(ext_wq);
            params_t1.add_output(q_proj__rv_v2);
            params_t1.add_scalar(b0);
            params_t1.launch_spec.set_block_num(20);
            params_t1.add_dep(rmsnorm_id);
            TaskOutputTensors __rt_q = rt_submit_aic_task(1, params_t1);
            q_proj_task_per_tile[tix] = __rt_q.task_id();
            const Tensor& q_proj__ssa_v3 = q_proj__rv_v2;

            // Spmd k_proj: qwen3_decode_incore_2
            ArgWithDeps<256> params_t2;
            params_t2.add_input(normed_tile);
            params_t2.add_input(ext_wk);
            params_t2.add_output(k_proj);
            params_t2.add_scalar(b0);
            params_t2.launch_spec.set_block_num(8);
            params_t2.add_dep(rmsnorm_id);
            TaskOutputTensors __rt_k = rt_submit_aic_task(2, params_t2);
            k_proj_task_per_tile[tix] = __rt_k.task_id();
            const Tensor& k_proj__ssa_v3 = k_proj;

            // Spmd v_proj: qwen3_decode_incore_3
            ArgWithDeps<256> params_t3;
            params_t3.add_input(normed_tile);
            params_t3.add_input(ext_wv);
            params_t3.add_output(v_proj);
            params_t3.add_scalar(b0);
            params_t3.launch_spec.set_block_num(8);
            params_t3.add_dep(rmsnorm_id);
            TaskOutputTensors __rt_v = rt_submit_aic_task(3, params_t3);
            v_proj_task_per_tile[tix] = __rt_v.task_id();
            const Tensor& v_proj__ssa_v3 = v_proj;

            // Task 4: qk_norm
            ArgWithDeps<256> params_t4;
            params_t4.add_output(k_proj_norm);
            params_t4.add_output(q_proj_norm);
            params_t4.add_input(q_proj__ssa_v3);
            params_t4.add_input(ext_q_norm_weight);
            params_t4.add_input(k_proj__ssa_v3);
            params_t4.add_input(ext_k_norm_weight);
            params_t4.add_scalar(0);
            params_t4.add_scalar(b0);
            params_t4.add_dep(q_proj_task_per_tile[tix]);
            params_t4.add_dep(k_proj_task_per_tile[tix]);
            params_t4.add_dep(v_proj_task_per_tile[tix]);
            TaskOutputTensors __rt_qk = rt_submit_aiv_task(4, params_t4);
            qk_norm_task_per_tile[tix] = __rt_qk.task_id();
            q_proj__rv_v2 = q_proj__ssa_v3;
        }

        uint32_t attn_out_ci_shapes[2] = {static_cast<uint32_t>(batch_padded), 5120};
        TensorCreateInfo attn_out_ci(attn_out_ci_shapes, 2, DataType::BFLOAT16);
        TaskOutputTensors alloc_7 = alloc_tensors(attn_out_ci);
        __prev = alloc_7.task_id();
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
            TaskOutputTensors alloc_8 = alloc_tensors(
                all_raw_scores_ci, all_exp_padded_ci, all_cur_mi_ci, all_cur_li_ci, all_oi_tmp_ci);
            __prev = alloc_8.task_id();
            const Tensor& all_raw_scores = alloc_8.get_ref(0);
            const Tensor& all_exp_padded = alloc_8.get_ref(1);
            const Tensor& all_cur_mi = alloc_8.get_ref(2);
            const Tensor& all_cur_li = alloc_8.get_ref(3);
            const Tensor& all_oi_tmp = alloc_8.get_ref(4);

            size_t idx_ctx_len = static_cast<size_t>(b);
            int32_t ctx_len = static_cast<int32_t*>(orch_args.tensor(7).data_as<void>())[idx_ctx_len];
            int64_t pos = (static_cast<int64_t>(ctx_len) - 1);
            int64_t ctx_blocks = ((static_cast<int64_t>(ctx_len) + 127) / 128);
            int64_t block_table_base = (b * 32);
            size_t idx_slot = static_cast<size_t>(b);
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
            params_t5.add_dep(all_q_padded_alloc_task);
            params_t5.add_dep(qk_norm_task_per_tile[static_cast<size_t>(b / 16)]);
            TaskOutputTensors __rt_rope = rt_submit_aiv_task(5, params_t5);
            const PTO2TaskId rope_kv_id = __rt_rope.task_id();
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
            params_t6.add_dep(rope_kv_id);
            __chain_dep(params_t6);
            TaskOutputTensors __rt_qkmm = rt_submit_aic_task(6, params_t6);
            __prev = __rt_qkmm.task_id();
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
            TaskOutputTensors __rt_sm = rt_submit_aiv_task(7, params_t7);
            __prev = __rt_sm.task_id();
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
            params_t8.add_dep(rope_kv_id);
            __chain_dep(params_t8);
            TaskOutputTensors __rt_sv = rt_submit_aic_task(8, params_t8);
            __prev = __rt_sv.task_id();
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
            TaskOutputTensors __rt_os = rt_submit_aiv_task(9, params_t9);
            __prev = __rt_os.task_id();
            online_softmax_tasks_by_b[static_cast<size_t>(b)].push_back(__rt_os.task_id());
        }

        for (int64_t b0 = 0; b0 < batch_padded; b0 += 16) {
            uint32_t resid1_tile_ci_shapes[2] = {16, 5120};
            TensorCreateInfo resid1_tile_ci(resid1_tile_ci_shapes, 2, DataType::FLOAT32);
            uint32_t gm_pipe_buffer_0_ci_shapes[1] = {static_cast<uint32_t>((16384) * (40))};
            TensorCreateInfo gm_pipe_buffer_0_ci(gm_pipe_buffer_0_ci_shapes, 1, DataType::FLOAT32, /*manual_dep=*/true);
            uint32_t post_norm_tile_ci_shapes[2] = {16, 5120};
            TensorCreateInfo post_norm_tile_ci(post_norm_tile_ci_shapes, 2, DataType::BFLOAT16);
            uint32_t mlp_tile_ci_shapes[2] = {16, 17408};
            TensorCreateInfo mlp_tile_ci(mlp_tile_ci_shapes, 2, DataType::BFLOAT16);
            uint32_t down_fp32_tile_ci_shapes[2] = {16, 5120};
            TensorCreateInfo down_fp32_tile_ci(down_fp32_tile_ci_shapes, 2, DataType::FLOAT32);
            TaskOutputTensors alloc_9 = alloc_tensors(
                resid1_tile_ci, gm_pipe_buffer_0_ci, post_norm_tile_ci, mlp_tile_ci, down_fp32_tile_ci);
            __prev = alloc_9.task_id();
            const Tensor& resid1_tile = alloc_9.get_ref(0);
            const Tensor& gm_pipe_buffer_0 = alloc_9.get_ref(1);
            const Tensor& post_norm_tile = alloc_9.get_ref(2);
            const Tensor& mlp_tile = alloc_9.get_ref(3);
            const Tensor& down_fp32_tile = alloc_9.get_ref(4);
            const int64_t cur_valid__ssa_v3 = std::min<int64_t>((user_batch - b0), 16);

            // Group qwen3_decode_incore_10: MixedKernels (AIC + AIV lanes)
            ArgWithDeps<256> params_t10;
            params_t10.add_input(ext_hidden_states);
            params_t10.add_input(attn_out);
            params_t10.add_input(ext_wo);
            params_t10.add_inout(resid1_tile);
            params_t10.add_output(gm_pipe_buffer_0);
            params_t10.add_scalar(b0);
            params_t10.add_scalar(cur_valid__ssa_v3);
            MixedKernels mixed_10 = {10, 11, 11};
            params_t10.launch_spec.set_block_num(40);
            for (int64_t __row = 0; __row < cur_valid__ssa_v3; ++__row) {
                const int64_t bb = b0 + __row;
                for (const PTO2TaskId& __os_tid : online_softmax_tasks_by_b[static_cast<size_t>(bb)]) {
                    params_t10.add_dep(__os_tid);
                }
            }
            TaskOutputTensors __rt_op = rt_submit_task(mixed_10, params_t10);
            const PTO2TaskId out_proj_mixed_id = __rt_op.task_id();
            const Tensor& resid1_tile__ssa_v1 = resid1_tile;

            // Task 12: post_rmsnorm (depends only on Func10/11)
            ArgWithDeps<256> params_t11;
            params_t11.add_input(resid1_tile__ssa_v1);
            params_t11.add_output(post_norm_tile);
            params_t11.add_input(ext_post_rms_weight);
            params_t11.add_dep(out_proj_mixed_id);
            TaskOutputTensors __rt_pr = rt_submit_aiv_task(12, params_t11);
            const PTO2TaskId post_rmsnorm_id = __rt_pr.task_id();
            const Tensor& post_norm_tile__rv_v2 = post_norm_tile;

            constexpr int64_t kMlpSiluSteps = 34;
            std::vector<PTO2TaskId> silu_task_by_ob(static_cast<size_t>(kMlpSiluSteps), PTO2TaskId::invalid());
            for (int64_t ob = 0; ob < kMlpSiluSteps; ob += 1) {
                uint32_t ret0__out_ci_shapes[2] = {16, 512};
                TensorCreateInfo ret0__out_ci(ret0__out_ci_shapes, 2, DataType::FLOAT32);
                uint32_t ret0__out_1_ci_shapes[2] = {16, 512};
                TensorCreateInfo ret0__out_1_ci(ret0__out_1_ci_shapes, 2, DataType::FLOAT32);
                TaskOutputTensors alloc_10 = alloc_tensors(ret0__out_ci, ret0__out_1_ci);
                __prev = alloc_10.task_id();
                const Tensor& ret0__out = alloc_10.get_ref(0);
                const Tensor& ret0__out_1 = alloc_10.get_ref(1);
                const int64_t mlp_o0 = (ob * 512);

                // Task 13: gate_proj
                ArgWithDeps<256> params_t12;
                params_t12.add_input(post_norm_tile__rv_v2);
                params_t12.add_input(ext_w_gate);
                params_t12.add_inout(ret0__out);
                params_t12.add_scalar(mlp_o0);
                params_t12.add_dep(post_rmsnorm_id);
                TaskOutputTensors __rt_gate = rt_submit_aic_task(13, params_t12);
                const PTO2TaskId gate_id = __rt_gate.task_id();
                const Tensor& gate_acc = ret0__out;

                // Task 14: up_proj
                ArgWithDeps<256> params_t13;
                params_t13.add_input(post_norm_tile__rv_v2);
                params_t13.add_input(ext_w_up);
                params_t13.add_inout(ret0__out_1);
                params_t13.add_scalar(mlp_o0);
                params_t13.add_dep(post_rmsnorm_id);
                TaskOutputTensors __rt_up = rt_submit_aic_task(14, params_t13);
                const PTO2TaskId up_id = __rt_up.task_id();
                const Tensor& up_acc = ret0__out_1;
                uint32_t ret0__out_2_shapes[2] = {16, 512};
                uint32_t ret0__out_2_offsets[2] = {0, static_cast<uint32_t>(mlp_o0)};
                Tensor ret0__out_2 = mlp_tile.view(ret0__out_2_shapes, ret0__out_2_offsets);

                // Task 15: silu
                ArgWithDeps<256> params_t14;
                params_t14.add_input(gate_acc);
                params_t14.add_input(up_acc);
                params_t14.add_inout(ret0__out_2);
                params_t14.add_dep(gate_id);
                params_t14.add_dep(up_id);
                TaskOutputTensors __rt_silu = rt_submit_aiv_task(15, params_t14);
                silu_task_by_ob[static_cast<size_t>(ob)] = __rt_silu.task_id();
            }

            std::vector<PTO2TaskId> down_proj_task_by_db;
            down_proj_task_by_db.reserve(40);
            for (int64_t di = 0; di < 40; di += 1) {
                // Task 16: down_proj
                ArgWithDeps<256> params_t15;
                params_t15.add_input(mlp_tile);
                params_t15.add_input(ext_w_down);
                params_t15.add_output(down_fp32_tile);
                params_t15.add_scalar(di);
                for (const PTO2TaskId& __silu_tid : silu_task_by_ob) {
                    if (__silu_tid.is_valid()) {
                        params_t15.add_dep(__silu_tid);
                    }
                }
                TaskOutputTensors __rt_down = rt_submit_aic_task(16, params_t15);
                down_proj_task_by_db.push_back(__rt_down.task_id());
            }

            for (int64_t db = 0; db < 40; db += 1) {
                uint32_t gm_pipe_buffer_1_ci_shapes[1] = {16384};
                TensorCreateInfo gm_pipe_buffer_1_ci(gm_pipe_buffer_1_ci_shapes, 1, DataType::FLOAT32, /*manual_dep=*/true);
                TaskOutputTensors alloc_11 = alloc_tensors(gm_pipe_buffer_1_ci);
                __prev = alloc_11.task_id();
                const Tensor& gm_pipe_buffer_1 = alloc_11.get_ref(0);

                // Group down_proj_residual: MixedKernels (AIC + AIV lanes)
                ArgWithDeps<256> params_t16;
                params_t16.add_input(resid1_tile__ssa_v1);
                params_t16.add_input(mlp_tile);
                params_t16.add_input(ext_w_down);
                params_t16.add_output(ext_out);
                params_t16.add_output(gm_pipe_buffer_1);
                params_t16.add_scalar(db);
                params_t16.add_scalar(cur_valid__ssa_v3);
                params_t16.add_scalar(b0);
                MixedKernels mixed_16 = {17, 18, 18};
                params_t16.add_dep(down_proj_task_by_db[static_cast<size_t>(db)]);
                TaskOutputTensors __rt_res = rt_submit_task(mixed_16, params_t16);
                (void)__rt_res;
            }
        }
    }
}

}  // extern "C"
