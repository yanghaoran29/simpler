// Orchestration: qwen3_decode (Qwen3-14B batch90 SPMD manual-scope, DV200 ring-scheduler variant).
// Graph semantics match Qwen3Decode_manual_scope_simplify; submission uses ring buffer +
// add_predecessor + memory pool (see qwen3_decode_dv200.cpp).

#include "runtime.h"
#include <atomic>
#include <iostream>
#include <vector>

#include <stddef.h>
#include <stdint.h>
#include <stdio.h>

#include "pto_orchestration_api.h"

// Manual explicit-deps graph (unchanged intent):
// Func0 SPMD rmsnorm; per-tile Func1–3 → Func4 qk_norm (fan-in, no cross-tile Func4 deps).
// Per batch: Func5 rope (tile qk_norm only); Func6–9 chain; Func6/8 also wait Func5.
// MLP: Func10/11 mixed out_proj waits online_softmax rows; Func12 ← Func10/11 only;
// Func13/14 ← Func12; Func15 silu; Func16 fans in all Func15; Func17/18 ← Func16[db].

extern "C" {

// ---------------------------------------------------------------------------
// DV200 task ring + memory pool (runtime provides symbols below)
// ---------------------------------------------------------------------------

#define BIT_MASK 0b100000
#define RING_MASK 4095
#define RING_SIZE 4096
#define CELL_SIZE 1024

enum TASK_TYPE : u_int8_t {
    TASK_AIC = 0,
    TASK_AIV = 1,
    TASK_MIX = 2,
    TASK_SPMD = 3,
};

struct taskInfo {
    u_int64_t kernel;
    u_int64_t input[16];
    u_int64_t scalar[32];
    u_int32_t block_num;
    u_int8_t mixed_aic;
    u_int8_t mixed_aiv0;
    u_int8_t mixed_aiv1;
};

struct taskStatus {
    u_int16_t taskID;
    u_int8_t status;
    bool empty;
    std::atomic<bool> lock;
};

struct taskInfo baseRing[RING_SIZE];

extern u_int16_t g_taskIndex;
extern struct taskStatus taskStatusRing[RING_SIZE];
extern struct SuccessorRing successorRing;
extern struct MemoryPool memoryPool;
extern struct ReadyQueue globalReadyQueue;
extern u_int16_t taskHead;
extern u_int16_t head;
extern u_int16_t minTaskID;

void add_predecessor(u_int16_t pre, u_int16_t taskID);
u_int16_t getRingIndex(u_int16_t taskID);
const Tensor& pool_alloc_bytes(int size);
void when_2_free(const Tensor& addr, int cnt, u_int16_t minTaskID);
void free_tensor();

static inline size_t qwen3_dtype_bytes(DataType dt) {
    return (dt == DataType::BFLOAT16) ? 2u : 4u;
}

static inline size_t qwen3_tensor_bytes(const uint32_t* shapes, int ndim, DataType dt) {
    size_t n = 1;
    for (int i = 0; i < ndim; ++i) {
        n *= shapes[i];
    }
    return n * qwen3_dtype_bytes(dt);
}

static const Tensor& pool_alloc(const uint32_t* shapes, int ndim, DataType dt) {
    return pool_alloc_bytes(static_cast<int>(qwen3_tensor_bytes(shapes, ndim, dt)));
}

static inline taskInfo& ring_slot(u_int16_t taskID) {
    return baseRing[taskID & RING_MASK];
}

static u_int16_t ring_emit(u_int8_t kernel_id, TASK_TYPE kind, u_int32_t block_num = 0) {
    const u_int16_t taskID = g_taskIndex++;
    (void)getRingIndex(taskID);
    taskInfo& slot = ring_slot(taskID);
    slot.kernel = kernel_id;
    slot.block_num = block_num;
    slot.mixed_aic = 0;
    slot.mixed_aiv0 = 0;
    slot.mixed_aiv1 = 0;
    globalReadyQueue.push(kind, taskID);
    return taskID;
}

static u_int16_t ring_emit_mixed(u_int8_t aic, u_int8_t aiv0, u_int8_t aiv1, u_int32_t block_num) {
    const u_int16_t taskID = g_taskIndex++;
    (void)getRingIndex(taskID);
    taskInfo& slot = ring_slot(taskID);
    slot.kernel = aic;
    slot.mixed_aic = aic;
    slot.mixed_aiv0 = aiv0;
    slot.mixed_aiv1 = aiv1;
    slot.block_num = block_num;
    globalReadyQueue.push(TASK_MIX, taskID);
    return taskID;
}

// ---------------------------------------------------------------------------

__attribute__((visibility("default")))
PTO2OrchestrationConfig aicpu_orchestration_config(const ChipStorageTaskArgs& orch_args) {
    (void)orch_args;
    return PTO2OrchestrationConfig{
        .expected_arg_count = 20,
    };
}

__attribute__((visibility("default")))
void aicpu_orchestration_entry(const ChipStorageTaskArgs& orch_args) {
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
    (void)ext_seq_lens;

    PTO2_SCOPE(PTO2ScopeMode::MANUAL) {
        const int64_t user_batch = 90;
        const int64_t batch_padded = (((user_batch + 15) / 16) * 16);
        const int64_t num_tiles = batch_padded / 16;
        constexpr int64_t kHidden = 5120;
        constexpr int64_t kKvHidden = 1024;
        constexpr int64_t kHeadDim = 128;
        constexpr int64_t kBatchTile = 16;
        constexpr int64_t kMlpSiluSteps = 34;
        constexpr int64_t kDownBlocks = 40;

        std::vector<u_int16_t> q_proj_task_per_tile(static_cast<size_t>(num_tiles), 0);
        std::vector<u_int16_t> k_proj_task_per_tile(static_cast<size_t>(num_tiles), 0);
        std::vector<u_int16_t> v_proj_task_per_tile(static_cast<size_t>(num_tiles), 0);
        std::vector<u_int16_t> qk_norm_task_per_tile(static_cast<size_t>(num_tiles), 0);
        std::vector<std::vector<u_int16_t>> online_softmax_tasks_by_b(static_cast<size_t>(user_batch));

        // Global buffers (pool alloc; no alloc-task dep chain)
        uint32_t all_q_shapes[2] = {11520, static_cast<uint32_t>(kHeadDim)};
        const Tensor& all_q_padded = pool_alloc(all_q_shapes, 2, DataType::BFLOAT16);

        uint32_t q_proj_shapes[2] = {static_cast<uint32_t>(batch_padded), static_cast<uint32_t>(kHidden)};
        const Tensor& q_proj = pool_alloc(q_proj_shapes, 2, DataType::FLOAT32);

        uint32_t k_proj_shapes[2] = {static_cast<uint32_t>(batch_padded), static_cast<uint32_t>(kKvHidden)};
        const Tensor& k_proj = pool_alloc(k_proj_shapes, 2, DataType::FLOAT32);

        uint32_t v_proj_shapes[2] = {static_cast<uint32_t>(batch_padded), static_cast<uint32_t>(kKvHidden)};
        const Tensor& v_proj = pool_alloc(v_proj_shapes, 2, DataType::FLOAT32);

        uint32_t q_norm_shapes[2] = {static_cast<uint32_t>(batch_padded), static_cast<uint32_t>(kHidden)};
        const Tensor& q_proj_norm = pool_alloc(q_norm_shapes, 2, DataType::FLOAT32);

        uint32_t k_norm_shapes[2] = {static_cast<uint32_t>(batch_padded), static_cast<uint32_t>(kKvHidden)};
        const Tensor& k_proj_norm = pool_alloc(k_norm_shapes, 2, DataType::FLOAT32);

        uint32_t normed_full_shapes[2] = {static_cast<uint32_t>(batch_padded), static_cast<uint32_t>(kHidden)};
        const Tensor& normed_full = pool_alloc(normed_full_shapes, 2, DataType::BFLOAT16);

        // Func0: SPMD rmsnorm → normed_full
        const u_int16_t rmsnorm_id = ring_emit(0, TASK_SPMD, 6);
        {
            taskInfo& s = ring_slot(rmsnorm_id);
            s.input[0] = reinterpret_cast<u_int64_t>(&ext_hidden_states);
            s.input[1] = reinterpret_cast<u_int64_t>(&normed_full);
            s.input[2] = reinterpret_cast<u_int64_t>(&ext_input_rms_weight);
            s.scalar[0] = static_cast<u_int64_t>(user_batch);
        }

        for (int64_t b0 = 0; b0 < batch_padded; b0 += kBatchTile) {
                const size_t tix = static_cast<size_t>(b0 / kBatchTile);
                uint32_t tile_shapes[2] = {static_cast<uint32_t>(kBatchTile), static_cast<uint32_t>(kHidden)};
                uint32_t tile_offsets[2] = {static_cast<uint32_t>(b0), 0};
                Tensor normed_tile = normed_full.view(tile_shapes, tile_offsets);

                // Func1: q_proj SPMD
                {
                    const u_int16_t tid_q = ring_emit(1, TASK_SPMD, 20);
                    taskInfo& sq = ring_slot(tid_q);
                    sq.input[0] = reinterpret_cast<u_int64_t>(&normed_tile);
                    sq.input[1] = reinterpret_cast<u_int64_t>(&ext_wq);
                    sq.input[2] = reinterpret_cast<u_int64_t>(&q_proj);
                    sq.scalar[0] = static_cast<u_int64_t>(b0);
                    add_predecessor(rmsnorm_id, tid_q);
                    q_proj_task_per_tile[tix] = tid_q;
                }

                // Func2: k_proj SPMD
                {
                    const u_int16_t tid_k = ring_emit(2, TASK_SPMD, 8);
                    taskInfo& sk = ring_slot(tid_k);
                    sk.input[0] = reinterpret_cast<u_int64_t>(&normed_tile);
                    sk.input[1] = reinterpret_cast<u_int64_t>(&ext_wk);
                    sk.input[2] = reinterpret_cast<u_int64_t>(&k_proj);
                    sk.scalar[0] = static_cast<u_int64_t>(b0);
                    add_predecessor(rmsnorm_id, tid_k);
                    k_proj_task_per_tile[tix] = tid_k;
                }

                // Func3: v_proj SPMD
                {
                    const u_int16_t tid_v = ring_emit(3, TASK_SPMD, 8);
                    taskInfo& sv = ring_slot(tid_v);
                    sv.input[0] = reinterpret_cast<u_int64_t>(&normed_tile);
                    sv.input[1] = reinterpret_cast<u_int64_t>(&ext_wv);
                    sv.input[2] = reinterpret_cast<u_int64_t>(&v_proj);
                    sv.scalar[0] = static_cast<u_int64_t>(b0);
                    add_predecessor(rmsnorm_id, tid_v);
                    v_proj_task_per_tile[tix] = tid_v;
                }

                // Func4: qk_norm (fan-in Func1–3 for this tile)
                {
                    const u_int16_t tid_qk = ring_emit(4, TASK_AIV, 0);
                    taskInfo& sn = ring_slot(tid_qk);
                    sn.input[0] = reinterpret_cast<u_int64_t>(&k_proj_norm);
                    sn.input[1] = reinterpret_cast<u_int64_t>(&q_proj_norm);
                    sn.input[2] = reinterpret_cast<u_int64_t>(&q_proj);
                    sn.input[3] = reinterpret_cast<u_int64_t>(&ext_q_norm_weight);
                    sn.input[4] = reinterpret_cast<u_int64_t>(&k_proj);
                    sn.input[5] = reinterpret_cast<u_int64_t>(&ext_k_norm_weight);
                    sn.scalar[0] = 0;
                    sn.scalar[1] = static_cast<u_int64_t>(b0);
                    add_predecessor(q_proj_task_per_tile[tix], tid_qk);
                    add_predecessor(k_proj_task_per_tile[tix], tid_qk);
                    add_predecessor(v_proj_task_per_tile[tix], tid_qk);
                    qk_norm_task_per_tile[tix] = tid_qk;
                }
        }

        uint32_t attn_out_shapes[2] = {static_cast<uint32_t>(batch_padded), static_cast<uint32_t>(kHidden)};
        const Tensor& attn_out = pool_alloc(attn_out_shapes, 2, DataType::BFLOAT16);

        // Attention: per batch
        for (int64_t b = 0; b < user_batch; b += 1) {
            uint32_t raw_shapes[2] = {4096, static_cast<uint32_t>(kHeadDim)};
            const Tensor& all_raw_scores = pool_alloc(raw_shapes, 2, DataType::FLOAT32);
            const Tensor& all_exp_padded = pool_alloc(raw_shapes, 2, DataType::BFLOAT16);
            uint32_t mi_shapes[2] = {4096, 1};
            const Tensor& all_cur_mi = pool_alloc(mi_shapes, 2, DataType::FLOAT32);
            const Tensor& all_cur_li = pool_alloc(mi_shapes, 2, DataType::FLOAT32);
            const Tensor& all_oi_tmp = pool_alloc(raw_shapes, 2, DataType::FLOAT32);
            const int attn_pool_bytes = static_cast<int>(
                qwen3_tensor_bytes(raw_shapes, 2, DataType::FLOAT32) +
                qwen3_tensor_bytes(raw_shapes, 2, DataType::BFLOAT16) +
                2 * qwen3_tensor_bytes(mi_shapes, 2, DataType::FLOAT32) +
                qwen3_tensor_bytes(raw_shapes, 2, DataType::FLOAT32));

            const size_t idx_ctx_len = static_cast<size_t>(b);
            const int32_t ctx_len =
                static_cast<int32_t*>(orch_args.tensor(7).data_as<void>())[idx_ctx_len];
            const int64_t pos = static_cast<int64_t>(ctx_len) - 1;
            const int64_t ctx_blocks = (static_cast<int64_t>(ctx_len) + 127) / 128;
            const int64_t block_table_base = b * 32;
            const size_t idx_slot = static_cast<size_t>(b);
            const int32_t slot =
                static_cast<int32_t*>(orch_args.tensor(9).data_as<void>())[idx_slot];
            const int64_t slot_block = static_cast<int64_t>(slot) / 128;
            const int64_t slot_offset = static_cast<int64_t>(slot) - (slot_block * 128);

            uint32_t cos_row_shapes[2] = {1, static_cast<uint32_t>(kHeadDim)};
            uint32_t cos_row_offsets[2] = {static_cast<uint32_t>(pos), 0};
            Tensor cos_row = ext_rope_cos.view(cos_row_shapes, cos_row_offsets);
            uint32_t sin_row_shapes[2] = {1, static_cast<uint32_t>(kHeadDim)};
            uint32_t sin_row_offsets[2] = {static_cast<uint32_t>(pos), 0};
            Tensor sin_row = ext_rope_sin.view(sin_row_shapes, sin_row_offsets);
            uint32_t half_shapes[2] = {1, 64};
            uint32_t hi_offsets[2] = {0, 64};
            Tensor cos_hi = cos_row.view(half_shapes, hi_offsets);
            uint32_t lo_off[2] = {0, 0};
            Tensor cos_lo = cos_row.view(half_shapes, lo_off);
            Tensor sin_lo = sin_row.view(half_shapes, lo_off);
            Tensor sin_hi = sin_row.view(half_shapes, hi_offsets);

            // Func5: rope_kv_cache
            const u_int16_t rope_id = ring_emit(5, TASK_AIV, 0);
            {
                taskInfo& sr = ring_slot(rope_id);
                sr.input[0] = reinterpret_cast<u_int64_t>(&all_q_padded);
                sr.input[1] = reinterpret_cast<u_int64_t>(&ext_k_cache);
                sr.input[2] = reinterpret_cast<u_int64_t>(&ext_v_cache);
                sr.input[3] = reinterpret_cast<u_int64_t>(&k_proj_norm);
                sr.input[4] = reinterpret_cast<u_int64_t>(&cos_lo);
                sr.input[5] = reinterpret_cast<u_int64_t>(&sin_lo);
                sr.input[6] = reinterpret_cast<u_int64_t>(&cos_hi);
                sr.input[7] = reinterpret_cast<u_int64_t>(&sin_hi);
                sr.input[8] = reinterpret_cast<u_int64_t>(&v_proj);
                sr.input[9] = reinterpret_cast<u_int64_t>(&q_proj_norm);
                sr.scalar[0] = static_cast<u_int64_t>(slot_block);
                sr.scalar[1] = static_cast<u_int64_t>(slot_offset);
                sr.scalar[2] = static_cast<u_int64_t>(b);
                add_predecessor(qk_norm_task_per_tile[static_cast<size_t>(b / kBatchTile)], rope_id);
            }

            uint32_t attn_row_shapes[2] = {1, static_cast<uint32_t>(kHidden)};
            uint32_t attn_row_offsets[2] = {static_cast<uint32_t>(b), 0};
            Tensor attn_row = attn_out.view(attn_row_shapes, attn_row_offsets);

            u_int16_t chain = rope_id;

            // Func6: qk_matmul SPMD
            chain = ring_emit(6, TASK_SPMD, 4);
            {
                taskInfo& s6 = ring_slot(chain);
                s6.input[0] = reinterpret_cast<u_int64_t>(&all_q_padded);
                s6.input[1] = reinterpret_cast<u_int64_t>(&all_raw_scores);
                s6.input[2] = reinterpret_cast<u_int64_t>(&ext_block_table);
                s6.input[3] = reinterpret_cast<u_int64_t>(&ext_k_cache);
                s6.scalar[0] = static_cast<u_int64_t>(b);
                s6.scalar[1] = static_cast<u_int64_t>(ctx_blocks);
                s6.scalar[2] = static_cast<u_int64_t>(block_table_base);
                add_predecessor(rope_id, chain);
            }

            // Func7: softmax SPMD
            {
                const u_int16_t tid7 = ring_emit(7, TASK_SPMD, 4);
                taskInfo& s7 = ring_slot(tid7);
                s7.input[0] = reinterpret_cast<u_int64_t>(&all_cur_li);
                s7.input[1] = reinterpret_cast<u_int64_t>(&all_cur_mi);
                s7.input[2] = reinterpret_cast<u_int64_t>(&all_exp_padded);
                s7.input[3] = reinterpret_cast<u_int64_t>(&all_raw_scores);
                s7.scalar[0] = static_cast<u_int64_t>(ctx_blocks);
                s7.scalar[1] = static_cast<u_int64_t>(ctx_len);
                add_predecessor(chain, tid7);
                chain = tid7;
            }

            // Func8: sv_matmul SPMD
            {
                const u_int16_t tid8 = ring_emit(8, TASK_SPMD, 4);
                taskInfo& s8 = ring_slot(tid8);
                s8.input[0] = reinterpret_cast<u_int64_t>(&all_oi_tmp);
                s8.input[1] = reinterpret_cast<u_int64_t>(&ext_block_table);
                s8.input[2] = reinterpret_cast<u_int64_t>(&all_exp_padded);
                s8.input[3] = reinterpret_cast<u_int64_t>(&ext_v_cache);
                s8.scalar[0] = static_cast<u_int64_t>(ctx_blocks);
                s8.scalar[1] = static_cast<u_int64_t>(block_table_base);
                add_predecessor(rope_id, tid8);
                add_predecessor(chain, tid8);
                chain = tid8;
            }

            // Func9: online_softmax SPMD
            {
                const u_int16_t tid9 = ring_emit(9, TASK_SPMD, 4);
                taskInfo& s9 = ring_slot(tid9);
                s9.input[0] = reinterpret_cast<u_int64_t>(&all_oi_tmp);
                s9.input[1] = reinterpret_cast<u_int64_t>(&all_cur_mi);
                s9.input[2] = reinterpret_cast<u_int64_t>(&all_cur_li);
                s9.input[3] = reinterpret_cast<u_int64_t>(&attn_row);
                s9.scalar[0] = static_cast<u_int64_t>(ctx_blocks);
                add_predecessor(chain, tid9);
                online_softmax_tasks_by_b[static_cast<size_t>(b)].push_back(tid9);
                when_2_free(all_raw_scores, attn_pool_bytes, tid9);
            }
        }

        // MLP: per tile
        for (int64_t b0 = 0; b0 < batch_padded; b0 += kBatchTile) {
            uint32_t tile2_shapes[2] = {static_cast<uint32_t>(kBatchTile), static_cast<uint32_t>(kHidden)};
            const Tensor& resid1_tile = pool_alloc(tile2_shapes, 2, DataType::FLOAT32);
            const int gm0_bytes = static_cast<int>(16384 * 40 * 4);
            const Tensor& gm_pipe_buffer_0 = pool_alloc_bytes(gm0_bytes);
            const Tensor& post_norm_tile = pool_alloc(tile2_shapes, 2, DataType::BFLOAT16);
            uint32_t mlp_shapes[2] = {static_cast<uint32_t>(kBatchTile), 17408};
            const Tensor& mlp_tile = pool_alloc(mlp_shapes, 2, DataType::BFLOAT16);
            const Tensor& down_fp32_tile = pool_alloc(tile2_shapes, 2, DataType::FLOAT32);

            const int64_t cur_valid = std::min<int64_t>(user_batch - b0, kBatchTile);

            // Func10/11: mixed out_proj
            const u_int16_t out_proj_id = ring_emit_mixed(10, 11, 11, 40);
            {
                taskInfo& sop = ring_slot(out_proj_id);
                sop.input[0] = reinterpret_cast<u_int64_t>(&ext_hidden_states);
                sop.input[1] = reinterpret_cast<u_int64_t>(&attn_out);
                sop.input[2] = reinterpret_cast<u_int64_t>(&ext_wo);
                sop.input[3] = reinterpret_cast<u_int64_t>(&resid1_tile);
                sop.input[4] = reinterpret_cast<u_int64_t>(&gm_pipe_buffer_0);
                sop.scalar[0] = static_cast<u_int64_t>(b0);
                sop.scalar[1] = static_cast<u_int64_t>(cur_valid);
                for (int64_t row = 0; row < cur_valid; ++row) {
                    const int64_t bb = b0 + row;
                    for (const u_int16_t os_tid : online_softmax_tasks_by_b[static_cast<size_t>(bb)]) {
                        add_predecessor(os_tid, out_proj_id);
                    }
                }
            }

            // Func12: post_rmsnorm
            const u_int16_t post_rms_id = ring_emit(12, TASK_AIV, 0);
            {
                taskInfo& spr = ring_slot(post_rms_id);
                spr.input[0] = reinterpret_cast<u_int64_t>(&resid1_tile);
                spr.input[1] = reinterpret_cast<u_int64_t>(&post_norm_tile);
                spr.input[2] = reinterpret_cast<u_int64_t>(&ext_post_rms_weight);
                add_predecessor(out_proj_id, post_rms_id);
            }

            std::vector<u_int16_t> silu_task_by_ob(static_cast<size_t>(kMlpSiluSteps), 0);
            for (int64_t ob = 0; ob < kMlpSiluSteps; ob += 1) {
                uint32_t chunk_shapes[2] = {static_cast<uint32_t>(kBatchTile), 512};
                const Tensor& gate_acc = pool_alloc(chunk_shapes, 2, DataType::FLOAT32);
                const Tensor& up_acc = pool_alloc(chunk_shapes, 2, DataType::FLOAT32);
                const int64_t mlp_o0 = ob * 512;

                const u_int16_t gate_id = ring_emit(13, TASK_AIC, 0);
                {
                    taskInfo& sg = ring_slot(gate_id);
                    sg.input[0] = reinterpret_cast<u_int64_t>(&post_norm_tile);
                    sg.input[1] = reinterpret_cast<u_int64_t>(&ext_w_gate);
                    sg.input[2] = reinterpret_cast<u_int64_t>(&gate_acc);
                    sg.scalar[0] = static_cast<u_int64_t>(mlp_o0);
                    add_predecessor(post_rms_id, gate_id);
                }

                const u_int16_t up_id = ring_emit(14, TASK_AIC, 0);
                {
                    taskInfo& su = ring_slot(up_id);
                    su.input[0] = reinterpret_cast<u_int64_t>(&post_norm_tile);
                    su.input[1] = reinterpret_cast<u_int64_t>(&ext_w_up);
                    su.input[2] = reinterpret_cast<u_int64_t>(&up_acc);
                    su.scalar[0] = static_cast<u_int64_t>(mlp_o0);
                    add_predecessor(post_rms_id, up_id);
                }

                uint32_t silu_view_shapes[2] = {static_cast<uint32_t>(kBatchTile), 512};
                uint32_t silu_view_offsets[2] = {0, static_cast<uint32_t>(mlp_o0)};
                Tensor silu_out = mlp_tile.view(silu_view_shapes, silu_view_offsets);

                const u_int16_t silu_id = ring_emit(15, TASK_AIV, 0);
                {
                    taskInfo& ss = ring_slot(silu_id);
                    ss.input[0] = reinterpret_cast<u_int64_t>(&gate_acc);
                    ss.input[1] = reinterpret_cast<u_int64_t>(&up_acc);
                    ss.input[2] = reinterpret_cast<u_int64_t>(&silu_out);
                    add_predecessor(gate_id, silu_id);
                    add_predecessor(up_id, silu_id);
                    silu_task_by_ob[static_cast<size_t>(ob)] = silu_id;
                }
            }

            std::vector<u_int16_t> down_proj_task_by_db;
            down_proj_task_by_db.reserve(static_cast<size_t>(kDownBlocks));
            for (int64_t di = 0; di < kDownBlocks; di += 1) {
                const u_int16_t down_id = ring_emit(16, TASK_AIC, 0);
                taskInfo& sd = ring_slot(down_id);
                sd.input[0] = reinterpret_cast<u_int64_t>(&mlp_tile);
                sd.input[1] = reinterpret_cast<u_int64_t>(&ext_w_down);
                sd.input[2] = reinterpret_cast<u_int64_t>(&down_fp32_tile);
                sd.scalar[0] = static_cast<u_int64_t>(di);
                for (const u_int16_t silu_tid : silu_task_by_ob) {
                    if (silu_tid != 0) {
                        add_predecessor(silu_tid, down_id);
                    }
                }
                down_proj_task_by_db.push_back(down_id);
            }

            for (int64_t db = 0; db < kDownBlocks; db += 1) {
                const int gm1_bytes = 16384 * 4;
                const Tensor& gm_pipe_buffer_1 = pool_alloc_bytes(gm1_bytes);

                const u_int16_t resid_id = ring_emit_mixed(17, 18, 18, 0);
                taskInfo& sres = ring_slot(resid_id);
                sres.input[0] = reinterpret_cast<u_int64_t>(&resid1_tile);
                sres.input[1] = reinterpret_cast<u_int64_t>(&mlp_tile);
                sres.input[2] = reinterpret_cast<u_int64_t>(&ext_w_down);
                sres.input[3] = reinterpret_cast<u_int64_t>(&ext_out);
                sres.input[4] = reinterpret_cast<u_int64_t>(&gm_pipe_buffer_1);
                sres.scalar[0] = static_cast<u_int64_t>(db);
                sres.scalar[1] = static_cast<u_int64_t>(cur_valid);
                sres.scalar[2] = static_cast<u_int64_t>(b0);
                add_predecessor(down_proj_task_by_db[static_cast<size_t>(db)], resid_id);
            }
        }
    }
}

}  // extern "C"
