// Orchestration: qwen3_decode (Qwen3-14B batch90 SPMD manual-scope, DV200 ring-scheduler variant).
//
// Same compute graph as Qwen3Decode_manual_scope_new; submission uses the DV200 ring buffer:
//   ring_emit / ring_emit_mixed → fill baseRing[taskID] → globalReadyQueue
//   add_predecessor for explicit edges; pool_alloc_bytes / when_2_free for intermediate lifetimes.
// Intermediate layout matches qwen3_V200_test: group pool allocs by lifetime; pass views to kernels.
// Prototype: qwen3_decode_dv200.cpp at repo root.

#include "runtime.h"
#include <atomic>
#include <iostream>

#include <stddef.h>
#include <stdint.h>
#include <stdio.h>

#include "pto_orchestration_api.h"

// ---------------------------------------------------------------------------
// Qwen3-14B batch90 model constants (compile-time; match Python golden / incores)
// ---------------------------------------------------------------------------
#define QWEN3_USER_BATCH 90    // real batch size; KV/attention loops over this
#define QWEN3_BATCH_TILE 16    // SPMD tile width (AIC tiling)
#define QWEN3_BATCH_PADDED 96  // user_batch rounded up to multiple of 16
#define QWEN3_NUM_TILES 6      // batch_padded / BATCH_TILE

#define QWEN3_HIDDEN 5120
#define QWEN3_KV_HIDDEN 1024  // K/V projection dim (GQA)
#define QWEN3_HEAD_DIM 128
#define QWEN3_HALF_DIM 64         // RoPE cos_lo/sin_hi half-dim split
#define QWEN3_INTERMEDIATE 17408  // MLP intermediate width (gate/up total)
#define QWEN3_MLP_OUT_CHUNK 512   // MLP chunk width; 34 * 512 = 17408
#define QWEN3_MLP_SILU_STEPS 34
#define QWEN3_DOWN_BLOCKS 40         // down_proj block count
#define QWEN3_ATTN_CTX_TILE 4096     // per-batch online-softmax context tile
#define QWEN3_MAX_BLOCKS_PER_SEQ 32  // block_table slots per sequence row
#define QWEN3_BLOCK_SIZE 128
#define QWEN3_BLOCK_SIZE_MASK 127
#define QWEN3_GM_PIPE_ELEMS 16384

// Explicit dependency overview (func_id → incore names in test_qwen3_decode.py CALLABLE):
//   0  SPMD rmsnorm (full hidden → normed_full)
//   1–3  per tile: Q/K/V linear (normed_tile → q/k/v_proj)
//   4  per tile: qk_norm, fan-in from same-tile 1–3 (no cross-tile Func4 deps)
//   5  per real batch b: rope_kv_cache, depends only on tile(b/16) Func4
//   6–9  per b: QK matmul → softmax → SV matmul → online softmax writes attn_out[b]
//   10/11  per tile: mixed out_proj (waits Func9 rows in that tile)
//   12  post_rmsnorm; 13/14 gate/up; 15 silu; 16 down accum; 17/18 residual → out

extern "C" {

// ---------------------------------------------------------------------------
// DV200 task ring + memory pool (symbols from DV200 runtime; orch fills slots + deps)
// ---------------------------------------------------------------------------

#define BIT_MASK 0b100000
#define RING_MASK 4095
#define RING_SIZE 4096
#define CELL_SIZE 1024

// Task kinds enqueued on globalReadyQueue (first arg to push).
enum TASK_TYPE : u_int8_t {
    TASK_AIC = 0,
    TASK_AIV = 1,
    TASK_MIX = 2,
    TASK_SPMD = 3,
};

// One ring slot: tensor pointers, scalars, and SPMD/mixed launch metadata.
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

void add_predecessor(u_int16_t pre, u_int16_t taskID);  // edge: taskID runs after pre completes
u_int16_t getRingIndex(u_int16_t taskID);
const Tensor &pool_alloc_bytes(int size);                            // allocate intermediate tensor from pool
void when_2_free(const Tensor &addr, int cnt, u_int16_t minTaskID);  // free addr after minTaskID
void free_tensor();

// Map taskID to ring slot (low 12 bits index; g_taskIndex monotonic in high bits).
static inline taskInfo &ring_slot(u_int16_t taskID) { return baseRing[taskID & RING_MASK]; }

// New taskID: set kernel_id / block_num, push to ready queue (SPMD/AIC/AIV).
static u_int16_t ring_emit(u_int8_t kernel_id, TASK_TYPE kind, u_int32_t block_num = 0) {
    const u_int16_t taskID = g_taskIndex++;
    (void)getRingIndex(taskID);
    taskInfo &slot = ring_slot(taskID);
    slot.kernel = kernel_id;
    slot.block_num = block_num;
    slot.mixed_aic = 0;
    slot.mixed_aiv0 = 0;
    slot.mixed_aiv1 = 0;
    globalReadyQueue.push(kind, taskID);
    return taskID;
}

// Mixed kernel: slot.kernel = AIC func_id; mixed_aiv0/1 = paired AIV func_ids.
static u_int16_t ring_emit_mixed(u_int8_t aic, u_int8_t aiv0, u_int8_t aiv1, u_int32_t block_num) {
    const u_int16_t taskID = g_taskIndex++;
    (void)getRingIndex(taskID);
    taskInfo &slot = ring_slot(taskID);
    slot.kernel = aic;
    slot.mixed_aic = aic;
    slot.mixed_aiv0 = aiv0;
    slot.mixed_aiv1 = aiv1;
    slot.block_num = block_num;
    globalReadyQueue.push(TASK_MIX, taskID);
    return taskID;
}

// ---------------------------------------------------------------------------
// AICPU orchestration entry: 20 external tensors, hand-written deps in MANUAL scope
// ---------------------------------------------------------------------------

__attribute__((visibility("default"))) PTO2OrchestrationConfig
aicpu_orchestration_config(const ChipStorageTaskArgs &orch_args) {
    (void)orch_args;
    return PTO2OrchestrationConfig{
        .expected_arg_count = 20,
    };
}

__attribute__((visibility("default"))) void aicpu_orchestration_entry(const ChipStorageTaskArgs &orch_args) {
    // --- Bind 20 host tensors (order matches test generate_args) ---
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
    //  0 hidden  1 input_rms  2 wq  3 wk  4 wv  5 q_norm  6 k_norm
    //  7 seq_lens  8 block_table  9 slot_mapping  10 cos  11 sin
    //  12 k_cache  13 v_cache  14 wo  15 post_rms  16 w_gate  17 w_up  18 w_down  19 out
    (void)ext_seq_lens;  // per-batch ctx len read by index b below

    // Compile-time checks: match golden / pool byte counts
    static_assert(QWEN3_BATCH_PADDED == 96, "batch90 padded batch");
    static_assert(QWEN3_NUM_TILES == 6, "batch90 tile count");
    static_assert(QWEN3_USER_BATCH * QWEN3_HEAD_DIM * QWEN3_HEAD_DIM * 2 == 2949120, "all_q_padded bytes");
    static_assert(QWEN3_GM_PIPE_ELEMS * QWEN3_DOWN_BLOCKS * 4 == 2621440, "gm_pipe_buffer bytes");

    // Track taskIDs per tile/batch for add_predecessor fan-in (fixed sizes at compile time)
    u_int16_t q_proj_task_per_tile[QWEN3_NUM_TILES] = {};
    u_int16_t k_proj_task_per_tile[QWEN3_NUM_TILES] = {};
    u_int16_t v_proj_task_per_tile[QWEN3_NUM_TILES] = {};
    u_int16_t qk_norm_task_per_tile[QWEN3_NUM_TILES] = {};
    u_int16_t online_softmax_task_by_b[QWEN3_USER_BATCH] = {};

    // --- Attention lifetime: pool buffers through online-softmax + attn_out ---
    const Tensor &all_q_padded = pool_alloc_bytes(QWEN3_USER_BATCH * QWEN3_HEAD_DIM * QWEN3_HEAD_DIM * 2);
    const Tensor &q_proj = pool_alloc_bytes(QWEN3_BATCH_PADDED * QWEN3_HIDDEN * 4);
    const Tensor &k_proj = pool_alloc_bytes(QWEN3_BATCH_PADDED * QWEN3_KV_HIDDEN * 4);
    const Tensor &v_proj = pool_alloc_bytes(QWEN3_BATCH_PADDED * QWEN3_KV_HIDDEN * 4);
    const Tensor &q_proj_norm = pool_alloc_bytes(QWEN3_BATCH_PADDED * QWEN3_HIDDEN * 4);
    const Tensor &k_proj_norm = pool_alloc_bytes(QWEN3_BATCH_PADDED * QWEN3_KV_HIDDEN * 4);
    const Tensor &normed_full = pool_alloc_bytes(QWEN3_BATCH_PADDED * QWEN3_HIDDEN * 2);
    const Tensor &attn_out = pool_alloc_bytes(QWEN3_BATCH_PADDED * QWEN3_HIDDEN * 2);

    // [Func0] SPMD pre-attention RMSNorm: hidden_states → normed_full (6 SPMD blocks)
    const u_int16_t rmsnorm_id = ring_emit(0, TASK_SPMD, 6);
    {
        taskInfo &s = ring_slot(rmsnorm_id);
        s.input[0] = reinterpret_cast<u_int64_t>(&ext_hidden_states);
        s.input[1] = reinterpret_cast<u_int64_t>(&normed_full);
        s.input[2] = reinterpret_cast<u_int64_t>(&ext_input_rms_weight);
        s.scalar[0] = static_cast<u_int64_t>(QWEN3_USER_BATCH);
    }

    // --- Per 16-wide tile: Q/K/V proj + qk_norm (depend on Func0 and same tile only) ---
    for (int64_t b0 = 0; b0 < QWEN3_BATCH_PADDED; b0 += QWEN3_BATCH_TILE) {
        const size_t tix = static_cast<size_t>(b0 / QWEN3_BATCH_TILE);
        const uint32_t normed_tile_shapes[2] = {
            static_cast<uint32_t>(QWEN3_BATCH_TILE),
            static_cast<uint32_t>(QWEN3_HIDDEN),
        };
        const uint32_t normed_tile_offsets[2] = {static_cast<uint32_t>(b0), 0};
        Tensor normed_tile = normed_full.view(normed_tile_shapes, normed_tile_offsets);

        // [Func1] SPMD Q proj: normed_tile × Wq → q_proj (block_dim=20)
        {
            const u_int16_t tid_q = ring_emit(1, TASK_SPMD, 20);
            taskInfo &sq = ring_slot(tid_q);
            sq.input[0] = reinterpret_cast<u_int64_t>(&normed_tile);
            sq.input[1] = reinterpret_cast<u_int64_t>(&ext_wq);
            sq.input[2] = reinterpret_cast<u_int64_t>(&q_proj);
            sq.scalar[0] = static_cast<u_int64_t>(b0);
            add_predecessor(rmsnorm_id, tid_q);
            q_proj_task_per_tile[tix] = tid_q;
        }

        // [Func2] SPMD K proj (block_dim=8)
        {
            const u_int16_t tid_k = ring_emit(2, TASK_SPMD, 8);
            taskInfo &sk = ring_slot(tid_k);
            sk.input[0] = reinterpret_cast<u_int64_t>(&normed_tile);
            sk.input[1] = reinterpret_cast<u_int64_t>(&ext_wk);
            sk.input[2] = reinterpret_cast<u_int64_t>(&k_proj);
            sk.scalar[0] = static_cast<u_int64_t>(b0);
            add_predecessor(rmsnorm_id, tid_k);
            k_proj_task_per_tile[tix] = tid_k;
        }

        // [Func3] SPMD V proj (block_dim=8)
        {
            const u_int16_t tid_v = ring_emit(3, TASK_SPMD, 8);
            taskInfo &sv = ring_slot(tid_v);
            sv.input[0] = reinterpret_cast<u_int64_t>(&normed_tile);
            sv.input[1] = reinterpret_cast<u_int64_t>(&ext_wv);
            sv.input[2] = reinterpret_cast<u_int64_t>(&v_proj);
            sv.scalar[0] = static_cast<u_int64_t>(b0);
            add_predecessor(rmsnorm_id, tid_v);
            v_proj_task_per_tile[tix] = tid_v;
        }

        // [Func4] Q/K RMSNorm: fan-in from same-tile Func1–3 → q_proj_norm / k_proj_norm
        {
            const u_int16_t tid_qk = ring_emit(4, TASK_AIV, 0);
            taskInfo &sn = ring_slot(tid_qk);
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

    u_int16_t last_rope_id = 0;
    u_int16_t last_online_softmax_id = 0;

    // --- Per real batch row: RoPE+KV cache, then 4-step online-softmax chain ---
    for (int64_t b = 0; b < QWEN3_USER_BATCH; b += 1) {
        // Per-batch attention scratch lifetime (freed after Func9 via when_2_free)
        const Tensor &all_raw_scores = pool_alloc_bytes(QWEN3_ATTN_CTX_TILE * QWEN3_HEAD_DIM * 4);
        const Tensor &all_exp_padded = pool_alloc_bytes(QWEN3_ATTN_CTX_TILE * QWEN3_HEAD_DIM * 2);
        const Tensor &all_cur_mi = pool_alloc_bytes(QWEN3_ATTN_CTX_TILE * 4);
        const Tensor &all_cur_li = pool_alloc_bytes(QWEN3_ATTN_CTX_TILE * 4);
        const Tensor &all_oi_tmp = pool_alloc_bytes(QWEN3_ATTN_CTX_TILE * QWEN3_HEAD_DIM * 4);

        // Derive KV slot, block_table base, and RoPE row from seq_lens / slot_mapping
        const size_t idx_ctx_len = static_cast<size_t>(b);
        const int32_t ctx_len = static_cast<int32_t *>(orch_args.tensor(7).data_as<void>())[idx_ctx_len];
        const int64_t pos = static_cast<int64_t>(ctx_len) - 1;
        const int64_t ctx_blocks = (static_cast<int64_t>(ctx_len) + QWEN3_BLOCK_SIZE_MASK) / QWEN3_BLOCK_SIZE;
        const int64_t block_table_base = b * QWEN3_MAX_BLOCKS_PER_SEQ;
        const size_t idx_slot = static_cast<size_t>(b);
        const int32_t slot = static_cast<int32_t *>(orch_args.tensor(9).data_as<void>())[idx_slot];
        const int64_t slot_block = static_cast<int64_t>(slot) / QWEN3_BLOCK_SIZE;
        const int64_t slot_offset = static_cast<int64_t>(slot) - slot_block * QWEN3_BLOCK_SIZE;

        const uint32_t cos_row_shapes[2] = {1, static_cast<uint32_t>(QWEN3_HEAD_DIM)};
        const uint32_t cos_row_offsets[2] = {static_cast<uint32_t>(pos), 0};
        Tensor cos_row = ext_rope_cos.view(cos_row_shapes, cos_row_offsets);
        const uint32_t sin_row_shapes[2] = {1, static_cast<uint32_t>(QWEN3_HEAD_DIM)};
        const uint32_t sin_row_offsets[2] = {static_cast<uint32_t>(pos), 0};
        Tensor sin_row = ext_rope_sin.view(sin_row_shapes, sin_row_offsets);
        const uint32_t half_shapes[2] = {1, static_cast<uint32_t>(QWEN3_HALF_DIM)};
        const uint32_t hi_offsets[2] = {0, static_cast<uint32_t>(QWEN3_HALF_DIM)};
        const uint32_t lo_off[2] = {0, 0};
        Tensor cos_hi = cos_row.view(half_shapes, hi_offsets);
        Tensor cos_lo = cos_row.view(half_shapes, lo_off);
        Tensor sin_lo = sin_row.view(half_shapes, lo_off);
        Tensor sin_hi = sin_row.view(half_shapes, hi_offsets);

        // [Func5] RoPE + KV cache write: depends on tile(b/16) qk_norm; updates all_q_padded
        const u_int16_t rope_id = ring_emit(5, TASK_AIV, 0);
        {
            taskInfo &sr = ring_slot(rope_id);
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
            add_predecessor(qk_norm_task_per_tile[static_cast<size_t>(b / QWEN3_BATCH_TILE)], rope_id);
        }
        last_rope_id = rope_id;

        const uint32_t attn_row_shapes[2] = {1, static_cast<uint32_t>(QWEN3_HIDDEN)};
        const uint32_t attn_row_offsets[2] = {static_cast<uint32_t>(b), 0};
        Tensor attn_row = attn_out.view(attn_row_shapes, attn_row_offsets);

        u_int16_t chain = rope_id;

        // [Func6] SPMD Q·K^T: all_q_padded × K_cache → all_raw_scores
        chain = ring_emit(6, TASK_SPMD, 4);
        {
            taskInfo &s6 = ring_slot(chain);
            s6.input[0] = reinterpret_cast<u_int64_t>(&all_q_padded);
            s6.input[1] = reinterpret_cast<u_int64_t>(&all_raw_scores);
            s6.input[2] = reinterpret_cast<u_int64_t>(&ext_block_table);
            s6.input[3] = reinterpret_cast<u_int64_t>(&ext_k_cache);
            s6.scalar[0] = static_cast<u_int64_t>(b);
            s6.scalar[1] = static_cast<u_int64_t>(ctx_blocks);
            s6.scalar[2] = static_cast<u_int64_t>(block_table_base);
            add_predecessor(rope_id, chain);
        }

        // [Func7] SPMD online softmax: raw_scores → exp_padded / cur_mi / cur_li
        {
            const u_int16_t tid7 = ring_emit(7, TASK_SPMD, 4);
            taskInfo &s7 = ring_slot(tid7);
            s7.input[0] = reinterpret_cast<u_int64_t>(&all_cur_li);
            s7.input[1] = reinterpret_cast<u_int64_t>(&all_cur_mi);
            s7.input[2] = reinterpret_cast<u_int64_t>(&all_exp_padded);
            s7.input[3] = reinterpret_cast<u_int64_t>(&all_raw_scores);
            s7.scalar[0] = static_cast<u_int64_t>(ctx_blocks);
            s7.scalar[1] = static_cast<u_int64_t>(ctx_len);
            add_predecessor(chain, tid7);
            chain = tid7;
        }

        // [Func8] SPMD attn·V: exp × V_cache → all_oi_tmp (also waits rope; partial overlap with Func6)
        {
            const u_int16_t tid8 = ring_emit(8, TASK_SPMD, 4);
            taskInfo &s8 = ring_slot(tid8);
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

        // [Func9] SPMD online-softmax reduce: write attn_out row b; register for out_proj wait
        {
            const u_int16_t tid9 = ring_emit(9, TASK_SPMD, 4);
            taskInfo &s9 = ring_slot(tid9);
            s9.input[0] = reinterpret_cast<u_int64_t>(&all_oi_tmp);
            s9.input[1] = reinterpret_cast<u_int64_t>(&all_cur_mi);
            s9.input[2] = reinterpret_cast<u_int64_t>(&all_cur_li);
            s9.input[3] = reinterpret_cast<u_int64_t>(&attn_row);
            s9.scalar[0] = static_cast<u_int64_t>(ctx_blocks);
            add_predecessor(chain, tid9);
            online_softmax_task_by_b[static_cast<size_t>(b)] = tid9;
            last_online_softmax_id = tid9;
            when_2_free(all_raw_scores, QWEN3_ATTN_CTX_TILE * QWEN3_HEAD_DIM * 4, tid9);
            when_2_free(all_exp_padded, QWEN3_ATTN_CTX_TILE * QWEN3_HEAD_DIM * 2, tid9);
            when_2_free(all_cur_mi, QWEN3_ATTN_CTX_TILE * 4, tid9);
            when_2_free(all_cur_li, QWEN3_ATTN_CTX_TILE * 4, tid9);
            when_2_free(all_oi_tmp, QWEN3_ATTN_CTX_TILE * QWEN3_HEAD_DIM * 4, tid9);
        }
    }

    // --- Attention intermediate lifetimes: return to pool after last consumer ---
    const u_int16_t last_qk_norm_id = qk_norm_task_per_tile[static_cast<size_t>(QWEN3_NUM_TILES - 1)];
    when_2_free(normed_full, QWEN3_BATCH_PADDED * QWEN3_HIDDEN * 2, last_qk_norm_id);
    when_2_free(q_proj, QWEN3_BATCH_PADDED * QWEN3_HIDDEN * 4, last_qk_norm_id);
    when_2_free(k_proj, QWEN3_BATCH_PADDED * QWEN3_KV_HIDDEN * 4, last_qk_norm_id);
    when_2_free(all_q_padded, QWEN3_USER_BATCH * QWEN3_HEAD_DIM * QWEN3_HEAD_DIM * 2, last_online_softmax_id);
    when_2_free(v_proj, QWEN3_BATCH_PADDED * QWEN3_KV_HIDDEN * 4, last_rope_id);
    when_2_free(q_proj_norm, QWEN3_BATCH_PADDED * QWEN3_HIDDEN * 4, last_rope_id);
    when_2_free(k_proj_norm, QWEN3_BATCH_PADDED * QWEN3_KV_HIDDEN * 4, last_rope_id);

    u_int16_t last_attn_out_user = 0;

    // --- Per tile MLP: out_proj → post_rms → chunked gate/up/silu → down → residual → out ---
    for (int64_t b0 = 0; b0 < QWEN3_BATCH_PADDED; b0 += QWEN3_BATCH_TILE) {
        // Per-tile MLP lifetime (gate/up staging + residual pipe pool; views for silu / gm_pipe slices)
        const Tensor &resid1_tile = pool_alloc_bytes(QWEN3_BATCH_TILE * QWEN3_HIDDEN * 4);
        const Tensor &gm_pipe_buffer_0 = pool_alloc_bytes(QWEN3_GM_PIPE_ELEMS * QWEN3_DOWN_BLOCKS * 4);
        const Tensor &post_norm_tile = pool_alloc_bytes(QWEN3_BATCH_TILE * QWEN3_HIDDEN * 2);
        const Tensor &mlp_tile = pool_alloc_bytes(QWEN3_BATCH_TILE * QWEN3_INTERMEDIATE * 2);
        const Tensor &down_fp32_tile = pool_alloc_bytes(QWEN3_BATCH_TILE * QWEN3_HIDDEN * 4);
        const Tensor &gate_acc = pool_alloc_bytes(QWEN3_BATCH_TILE * QWEN3_MLP_OUT_CHUNK * 4);
        const Tensor &up_acc = pool_alloc_bytes(QWEN3_BATCH_TILE * QWEN3_MLP_OUT_CHUNK * 4);
        const Tensor &gm_pipe_resid_pool = pool_alloc_bytes(QWEN3_GM_PIPE_ELEMS * QWEN3_DOWN_BLOCKS * 4);

        const int64_t cur_valid = std::min<int64_t>(QWEN3_USER_BATCH - b0, QWEN3_BATCH_TILE);

        // [Func10/11] mixed out_proj: attn rows + hidden residual → resid1_tile (waits Func9 per row in tile)
        const u_int16_t out_proj_id = ring_emit_mixed(10, 11, 11, 40);
        {
            taskInfo &sop = ring_slot(out_proj_id);
            sop.input[0] = reinterpret_cast<u_int64_t>(&ext_hidden_states);
            sop.input[1] = reinterpret_cast<u_int64_t>(&attn_out);
            sop.input[2] = reinterpret_cast<u_int64_t>(&ext_wo);
            sop.input[3] = reinterpret_cast<u_int64_t>(&resid1_tile);
            sop.input[4] = reinterpret_cast<u_int64_t>(&gm_pipe_buffer_0);
            sop.scalar[0] = static_cast<u_int64_t>(b0);
            sop.scalar[1] = static_cast<u_int64_t>(cur_valid);
            for (int64_t row = 0; row < cur_valid; ++row) {
                const int64_t bb = b0 + row;
                const u_int16_t os_tid = online_softmax_task_by_b[static_cast<size_t>(bb)];
                if (os_tid != 0) {
                    add_predecessor(os_tid, out_proj_id);
                }
            }
        }

        // [Func12] post-attention RMSNorm: resid1 → post_norm_tile
        const u_int16_t post_rms_id = ring_emit(12, TASK_AIV, 0);
        {
            taskInfo &spr = ring_slot(post_rms_id);
            spr.input[0] = reinterpret_cast<u_int64_t>(&resid1_tile);
            spr.input[1] = reinterpret_cast<u_int64_t>(&post_norm_tile);
            spr.input[2] = reinterpret_cast<u_int64_t>(&ext_post_rms_weight);
            add_predecessor(out_proj_id, post_rms_id);
        }

        when_2_free(gm_pipe_buffer_0, QWEN3_GM_PIPE_ELEMS * QWEN3_DOWN_BLOCKS * 4, out_proj_id);

        // [Func13/14/15] MLP in 34 steps of 512 cols: gate/up AIC → silu AIV into mlp_tile slice
        u_int16_t silu_task_by_ob[QWEN3_MLP_SILU_STEPS] = {};
        for (int64_t ob = 0; ob < QWEN3_MLP_SILU_STEPS; ob += 1) {
            const int64_t mlp_o0 = ob * QWEN3_MLP_OUT_CHUNK;

            // [Func13] gate projection chunk ob (reuses gate_acc staging buffer)
            const u_int16_t gate_id = ring_emit(13, TASK_AIC, 0);
            {
                taskInfo &sg = ring_slot(gate_id);
                sg.input[0] = reinterpret_cast<u_int64_t>(&post_norm_tile);
                sg.input[1] = reinterpret_cast<u_int64_t>(&ext_w_gate);
                sg.input[2] = reinterpret_cast<u_int64_t>(&gate_acc);
                sg.scalar[0] = static_cast<u_int64_t>(mlp_o0);
                add_predecessor(post_rms_id, gate_id);
            }

            // [Func14] up projection chunk ob (reuses up_acc staging buffer)
            const u_int16_t up_id = ring_emit(14, TASK_AIC, 0);
            {
                taskInfo &su = ring_slot(up_id);
                su.input[0] = reinterpret_cast<u_int64_t>(&post_norm_tile);
                su.input[1] = reinterpret_cast<u_int64_t>(&ext_w_up);
                su.input[2] = reinterpret_cast<u_int64_t>(&up_acc);
                su.scalar[0] = static_cast<u_int64_t>(mlp_o0);
                add_predecessor(post_rms_id, up_id);
            }

            const uint32_t silu_view_shapes[2] = {
                static_cast<uint32_t>(QWEN3_BATCH_TILE),
                static_cast<uint32_t>(QWEN3_MLP_OUT_CHUNK),
            };
            const uint32_t silu_view_offsets[2] = {0, static_cast<uint32_t>(ob * QWEN3_MLP_OUT_CHUNK)};
            Tensor silu_out = mlp_tile.view(silu_view_shapes, silu_view_offsets);

            // [Func15] SiLU(gate) * up → mlp_tile[:, ob*512:(ob+1)*512]
            const u_int16_t silu_id = ring_emit(15, TASK_AIV, 0);
            {
                taskInfo &ss = ring_slot(silu_id);
                ss.input[0] = reinterpret_cast<u_int64_t>(&gate_acc);
                ss.input[1] = reinterpret_cast<u_int64_t>(&up_acc);
                ss.input[2] = reinterpret_cast<u_int64_t>(&silu_out);
                add_predecessor(gate_id, silu_id);
                add_predecessor(up_id, silu_id);
                silu_task_by_ob[static_cast<size_t>(ob)] = silu_id;
            }
        }

        const u_int16_t last_silu_id = silu_task_by_ob[static_cast<size_t>(QWEN3_MLP_SILU_STEPS - 1)];
        when_2_free(gate_acc, QWEN3_BATCH_TILE * QWEN3_MLP_OUT_CHUNK * 4, last_silu_id);
        when_2_free(up_acc, QWEN3_BATCH_TILE * QWEN3_MLP_OUT_CHUNK * 4, last_silu_id);
        when_2_free(post_norm_tile, QWEN3_BATCH_TILE * QWEN3_HIDDEN * 2, last_silu_id);

        // [Func16] down_proj in 40 blocks: each block fans in all Func15, accumulates to down_fp32_tile
        u_int16_t down_proj_task_by_db[QWEN3_DOWN_BLOCKS] = {};
        for (int64_t di = 0; di < QWEN3_DOWN_BLOCKS; di += 1) {
            const u_int16_t down_id = ring_emit(16, TASK_AIC, 0);
            taskInfo &sd = ring_slot(down_id);
            sd.input[0] = reinterpret_cast<u_int64_t>(&mlp_tile);
            sd.input[1] = reinterpret_cast<u_int64_t>(&ext_w_down);
            sd.input[2] = reinterpret_cast<u_int64_t>(&down_fp32_tile);
            sd.scalar[0] = static_cast<u_int64_t>(di);
            for (int64_t si = 0; si < QWEN3_MLP_SILU_STEPS; ++si) {
                const u_int16_t silu_tid = silu_task_by_ob[static_cast<size_t>(si)];
                if (silu_tid != 0) {
                    add_predecessor(silu_tid, down_id);
                }
            }
            down_proj_task_by_db[static_cast<size_t>(di)] = down_id;
        }

        // [Func17/18] mixed residual: down block db + resid1 → ext_out (one chain per db)
        u_int16_t last_resid_id = 0;
        for (int64_t db = 0; db < QWEN3_DOWN_BLOCKS; db += 1) {
            const uint32_t gm_pipe_1_shapes[1] = {static_cast<uint32_t>(QWEN3_GM_PIPE_ELEMS)};
            const uint32_t gm_pipe_1_offsets[1] = {static_cast<uint32_t>(db * QWEN3_GM_PIPE_ELEMS)};
            Tensor gm_pipe_buffer_1 = gm_pipe_resid_pool.view(gm_pipe_1_shapes, gm_pipe_1_offsets, true);

            const u_int16_t resid_id = ring_emit_mixed(17, 18, 18, 0);
            taskInfo &sres = ring_slot(resid_id);
            sres.input[0] = reinterpret_cast<u_int64_t>(&resid1_tile);
            sres.input[1] = reinterpret_cast<u_int64_t>(&mlp_tile);
            sres.input[2] = reinterpret_cast<u_int64_t>(&ext_w_down);
            sres.input[3] = reinterpret_cast<u_int64_t>(&ext_out);
            sres.input[4] = reinterpret_cast<u_int64_t>(&gm_pipe_buffer_1);
            sres.scalar[0] = static_cast<u_int64_t>(db);
            sres.scalar[1] = static_cast<u_int64_t>(cur_valid);
            sres.scalar[2] = static_cast<u_int64_t>(b0);
            add_predecessor(down_proj_task_by_db[static_cast<size_t>(db)], resid_id);
            last_resid_id = resid_id;
        }

        when_2_free(gm_pipe_resid_pool, QWEN3_GM_PIPE_ELEMS * QWEN3_DOWN_BLOCKS * 4, last_resid_id);
        when_2_free(resid1_tile, QWEN3_BATCH_TILE * QWEN3_HIDDEN * 4, last_resid_id);
        when_2_free(mlp_tile, QWEN3_BATCH_TILE * QWEN3_INTERMEDIATE * 2, last_resid_id);
        when_2_free(down_fp32_tile, QWEN3_BATCH_TILE * QWEN3_HIDDEN * 4, last_resid_id);
        last_attn_out_user = last_resid_id;

        // Free attn_out after last tile's residual task completes
        when_2_free(attn_out, QWEN3_BATCH_PADDED * QWEN3_HIDDEN * 2, last_attn_out_user);
    }
}

}  // extern "C"
