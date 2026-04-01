/**
 * qwen3_decode_manual_scope.cpp
 *
 * Qwen3 decode (manual-scope, explicit-deps) backend (PERF_BACKEND=14):
 * declarations + definitions. Included by the perf drivers when PERF_BACKEND=14.
 * Do not compile as a separate TU.
 *
 * Ported from the generated manual-scope orchestration to the perf/cases UT
 * convention while PRESERVING the manual dependency model: an outer
 * PTO2ScopeMode::MANUAL scope plus explicit add_dep(task_id) edges (the tensormap
 * is not consulted for ordering in manual scope). The UT scaffolding for this —
 * a manual-scope guard and deps-carrying submit helpers — is defined locally
 * below. Under the AICore simulator no kernel runs, so tensors are backed by a
 * scratch pool and oversized weights use small placeholder shapes; only the
 * explicit task-id edges drive scheduling.
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
#include <type_traits>
#include <vector>

// Minimal local ArgWithDeps: the orchestration header's ArgWithDeps drags in
// pto_orchestration_api.h, whose PTO2Runtime/ops/make_tensor_external/PTO2ScopeGuard
// definitions conflict with the runtime view (pto_runtime2.h) used by the UT. It
// only needs Arg + PTO2TaskId, both already available here.
template <size_t MAX_DEP_COUNT = 256>
class ArgWithDeps : public Arg {
public:
    template <typename... Ids>
    void add_dep(Ids... ids) {
        static_assert((std::is_same_v<std::decay_t<Ids>, PTO2TaskId> && ...), "add_dep: PTO2TaskId only");
        ((deps_[count_++] = ids), ...);
    }
    Arg& finalize_for_submit() {
        Arg::set_dependencies(nullptr, 0);
        Arg::set_dependencies(deps_, count_);
        return *this;
    }

private:
    PTO2TaskId deps_[MAX_DEP_COUNT];
    uint32_t count_ = 0;
};

#ifndef PERF_CASE_IDX
#define PERF_CASE_IDX 0
#endif

static constexpr int QWEN3_MS_CASE_COUNT = 1;
static_assert(PERF_CASE_IDX >= 0 && PERF_CASE_IDX < QWEN3_MS_CASE_COUNT,
              "PERF_CASE_IDX out of range");

static constexpr int64_t USER_BATCH   = 16;
static constexpr int64_t BATCH_PADDED = 16;
static constexpr int64_t CTX_LEN      = 128;

struct Qwen3MsCase {
    char name[128];
    int  user_batch;
};

struct GraphCtx {
    uint64_t args[10];
};

const Qwen3MsCase PERF_CASES[QWEN3_MS_CASE_COUNT] = {
    { "qwen3_decode_manual_scope (batch=16, ctx_len=128, explicit deps)", static_cast<int>(USER_BATCH) },
};

// Per-func_id simulated AICore execution time (ns), indexed by the func_id passed
// to submit_ai{c,v}_deps / submit_mixed_deps below. Values are the qwen3 14B Decode
// per-kernel mean durations from V200-benchmark/Readme.md (µs → ns). Installed into
// the AICore simulator at the top of build_graph. Other samples define their own.
static const int kQwen3MsFuncDurationNs[] = {
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
static constexpr int kQwen3MsDurationCorrectionNs = 1000;

// ─── UT scaffolding for manual-scope + explicit-dep submission ────────────────
// A manual scope is begun by stamping the pending mode and calling the runtime's
// scope_begin op (mirrors orchestration rt_scope_begin(PTO2ScopeMode::MANUAL)).
struct ManualScopeGuard {
    PTO2Runtime* rt;
    explicit ManualScopeGuard(PTO2Runtime* r) : rt(r) {
        rt->pending_scope_mode = PTO2ScopeMode::MANUAL;
        rt->ops->scope_begin(rt);
    }
    ~ManualScopeGuard() { rt->ops->scope_end(rt); }
    ManualScopeGuard(const ManualScopeGuard&) = delete;
    ManualScopeGuard& operator=(const ManualScopeGuard&) = delete;
};

static inline PTO2TaskId submit_aic_deps(PTO2Runtime* rt, int32_t kid, ArgWithDeps<256>& a) {
    MixedKernels mk;
    mk.aic_kernel_id = kid;
    return rt->ops->submit_task(rt, mk, a.finalize_for_submit()).task_id();
}
static inline PTO2TaskId submit_aiv_deps(PTO2Runtime* rt, int32_t kid, ArgWithDeps<256>& a) {
    MixedKernels mk;
    mk.aiv0_kernel_id = kid;
    return rt->ops->submit_task(rt, mk, a.finalize_for_submit()).task_id();
}
static inline PTO2TaskId submit_mixed_deps(PTO2Runtime* rt, MixedKernels mk, ArgWithDeps<256>& a) {
    return rt->ops->submit_task(rt, mk, a.finalize_for_submit()).task_id();
}

// ─── Scratch pool: distinct region per tensor ─────────────────────────────────
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
        kQwen3MsFuncDurationNs,
        static_cast<int32_t>(sizeof(kQwen3MsFuncDurationNs) / sizeof(kQwen3MsFuncDurationNs[0])),
        kQwen3MsDurationCorrectionNs
    );

    uint32_t hs_shapes[2]       = {static_cast<uint32_t>(USER_BATCH), 5120};
    uint32_t wsmall[2]          = {16, 16};
    uint32_t vec5120[1]         = {5120};
    uint32_t vec128[1]          = {128};
    uint32_t rope_shapes[2]     = {static_cast<uint32_t>(CTX_LEN), 128};
    uint32_t kv_cache_shapes[2] = {128, 128};
    uint32_t out_shapes[2]      = {static_cast<uint32_t>(USER_BATCH), 5120};
    uint32_t block_table_shapes[1] = {static_cast<uint32_t>(USER_BATCH * 32)};

    Tensor ext_hidden_states   = mk(hs_shapes, 2, DataType::FLOAT32);
    Tensor ext_input_rms_weight= mk(vec5120, 1, DataType::FLOAT32);
    Tensor ext_wq              = mk(wsmall, 2, DataType::FLOAT32);
    Tensor ext_wk              = mk(wsmall, 2, DataType::FLOAT32);
    Tensor ext_wv              = mk(wsmall, 2, DataType::FLOAT32);
    Tensor ext_q_norm_weight   = mk(vec128, 1, DataType::FLOAT32);
    Tensor ext_k_norm_weight   = mk(vec128, 1, DataType::FLOAT32);
    Tensor ext_block_table     = mk(block_table_shapes, 1, DataType::INT32);
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

    static int seq_lens[USER_BATCH];
    static int slot_mapping[USER_BATCH];
    for (int i = 0; i < USER_BATCH; i++) {
        seq_lens[i]     = static_cast<int>(CTX_LEN);
        slot_mapping[i] = i;
    }

    int total_tasks = 0;
    {
        ManualScopeGuard _scope(rt);
        const int64_t user_batch   = USER_BATCH;
        const int64_t batch_padded = BATCH_PADDED;
        const int64_t num_tiles    = batch_padded / 16;

        uint32_t all_q_padded_shapes[2] = {11520, 128};
        Tensor all_q_padded = mk(all_q_padded_shapes, 2, DataType::BFLOAT16);
        uint32_t q_proj_shapes[2]      = {static_cast<uint32_t>(batch_padded), 5120};
        Tensor q_proj = mk(q_proj_shapes, 2, DataType::FLOAT32);
        uint32_t k_proj_shapes[2]      = {static_cast<uint32_t>(batch_padded), 1024};
        Tensor k_proj = mk(k_proj_shapes, 2, DataType::FLOAT32);
        uint32_t v_proj_shapes[2]      = {static_cast<uint32_t>(batch_padded), 1024};
        Tensor v_proj = mk(v_proj_shapes, 2, DataType::FLOAT32);
        uint32_t q_proj_norm_shapes[2] = {static_cast<uint32_t>(batch_padded), 5120};
        Tensor q_proj_norm = mk(q_proj_norm_shapes, 2, DataType::FLOAT32);
        uint32_t k_proj_norm_shapes[2] = {static_cast<uint32_t>(batch_padded), 1024};
        Tensor k_proj_norm = mk(k_proj_norm_shapes, 2, DataType::FLOAT32);

        std::vector<PTO2TaskId> q_proj_task_per_tile(static_cast<size_t>(num_tiles), PTO2TaskId::invalid());
        std::vector<PTO2TaskId> k_proj_task_per_tile(static_cast<size_t>(num_tiles), PTO2TaskId::invalid());
        std::vector<PTO2TaskId> v_proj_task_per_tile(static_cast<size_t>(num_tiles), PTO2TaskId::invalid());
        std::vector<PTO2TaskId> qk_norm_task_per_tile(static_cast<size_t>(num_tiles), PTO2TaskId::invalid());
        std::vector<std::vector<PTO2TaskId>> online_softmax_tasks_by_b(static_cast<size_t>(user_batch));

        for (int64_t b0 = 0; b0 < batch_padded; b0 += 16) {
            const size_t tix = static_cast<size_t>(b0 / 16);
            uint32_t normed_tile_shapes[2] = {16, 5120};
            Tensor normed_tile = mk(normed_tile_shapes, 2, DataType::BFLOAT16);
            const int64_t cur_valid = std::min<int64_t>(user_batch - b0, 16);

            ArgWithDeps<256> p_t0;  // rmsnorm (no deps)
            p_t0.add_input(ext_hidden_states);
            p_t0.add_output(normed_tile);
            p_t0.add_input(ext_input_rms_weight);
            p_t0.add_scalar(b0);
            p_t0.add_scalar(cur_valid);
            const PTO2TaskId rmsnorm_id = submit_aiv_deps(rt, 0, p_t0);
            total_tasks++;

            for (int64_t q0 = 0; q0 < 5120; q0 += 256) {
                ArgWithDeps<256> p_t1;  // q_proj
                p_t1.add_input(normed_tile);
                p_t1.add_input(ext_wq);
                p_t1.add_output(q_proj);
                p_t1.add_scalar(q0);
                p_t1.add_scalar(b0);
                p_t1.add_dep(rmsnorm_id);
                q_proj_task_per_tile[tix] = submit_aic_deps(rt, 1, p_t1);
                total_tasks++;
            }
            for (int64_t kv0 = 0; kv0 < 1024; kv0 += 128) {
                ArgWithDeps<256> p_t2;  // k_proj
                p_t2.add_input(normed_tile);
                p_t2.add_input(ext_wk);
                p_t2.add_output(k_proj);
                p_t2.add_scalar(kv0);
                p_t2.add_scalar(b0);
                p_t2.add_dep(rmsnorm_id);
                k_proj_task_per_tile[tix] = submit_aic_deps(rt, 2, p_t2);
                total_tasks++;
            }
            for (int64_t kv0 = 0; kv0 < 1024; kv0 += 128) {
                ArgWithDeps<256> p_t3;  // v_proj
                p_t3.add_input(normed_tile);
                p_t3.add_input(ext_wv);
                p_t3.add_output(v_proj);
                p_t3.add_scalar(kv0);
                p_t3.add_scalar(b0);
                p_t3.add_dep(rmsnorm_id);
                v_proj_task_per_tile[tix] = submit_aic_deps(rt, 3, p_t3);
                total_tasks++;
            }

            ArgWithDeps<256> p_t4;  // qk_norm
            p_t4.add_output(k_proj_norm);
            p_t4.add_output(q_proj_norm);
            p_t4.add_input(q_proj);
            p_t4.add_input(ext_q_norm_weight);
            p_t4.add_input(k_proj);
            p_t4.add_input(ext_k_norm_weight);
            p_t4.add_scalar(b0);
            p_t4.add_dep(q_proj_task_per_tile[tix]);
            p_t4.add_dep(k_proj_task_per_tile[tix]);
            p_t4.add_dep(v_proj_task_per_tile[tix]);
            qk_norm_task_per_tile[tix] = submit_aiv_deps(rt, 4, p_t4);
            total_tasks++;
        }

        uint32_t attn_out_shapes[2] = {static_cast<uint32_t>(batch_padded), 5120};
        Tensor attn_out = mk(attn_out_shapes, 2, DataType::BFLOAT16);

        for (int64_t b = 0; b < user_batch; b += 1) {
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

            ArgWithDeps<256> p_t5;  // rope_kv_cache
            p_t5.add_output(all_q_padded);
            p_t5.add_output(ext_k_cache);
            p_t5.add_output(ext_v_cache);
            p_t5.add_input(k_proj_norm);
            p_t5.add_input(cos_lo);
            p_t5.add_input(sin_lo);
            p_t5.add_input(cos_hi);
            p_t5.add_input(sin_hi);
            p_t5.add_input(v_proj);
            p_t5.add_input(q_proj_norm);
            p_t5.add_scalar(slot_block);
            p_t5.add_scalar(slot_offset);
            p_t5.add_scalar(b);
            p_t5.add_dep(qk_norm_task_per_tile[static_cast<size_t>(b / 16)]);
            const PTO2TaskId rope_kv_id = submit_aiv_deps(rt, 5, p_t5);
            total_tasks++;

            uint32_t attn_off[2] = {static_cast<uint32_t>(b), 0};
            uint32_t attn_win[2] = {1, 5120};
            Tensor attn_row = attn_out.view(attn_win, attn_off);

            ArgWithDeps<256> p_t6;  // qk_matmul (spmd)
            p_t6.add_input(all_q_padded);
            p_t6.add_output(all_raw_scores);
            p_t6.add_input(ext_block_table);
            p_t6.add_input(ext_k_cache);
            p_t6.add_scalar(b);
            p_t6.add_scalar(ctx_blocks);
            p_t6.add_scalar(block_table_base);
            p_t6.launch_spec.set_block_num(4);
            p_t6.add_dep(rope_kv_id);
            const PTO2TaskId qk_matmul_id = submit_aic_deps(rt, 6, p_t6);
            total_tasks++;

            ArgWithDeps<256> p_t7;  // softmax (spmd)
            p_t7.add_output(all_cur_li);
            p_t7.add_output(all_cur_mi);
            p_t7.add_output(all_exp_padded);
            p_t7.add_input(all_raw_scores);
            p_t7.add_scalar(ctx_blocks);
            p_t7.add_scalar(ctx_len);
            p_t7.launch_spec.set_block_num(4);
            p_t7.add_dep(qk_matmul_id);
            const PTO2TaskId softmax_id = submit_aiv_deps(rt, 7, p_t7);
            total_tasks++;

            ArgWithDeps<256> p_t8;  // sv_matmul (spmd)
            p_t8.add_output(all_oi_tmp);
            p_t8.add_input(ext_block_table);
            p_t8.add_input(all_exp_padded);
            p_t8.add_input(ext_v_cache);
            p_t8.add_scalar(ctx_blocks);
            p_t8.add_scalar(block_table_base);
            p_t8.launch_spec.set_block_num(4);
            p_t8.add_dep(rope_kv_id);
            p_t8.add_dep(softmax_id);
            const PTO2TaskId sv_matmul_id = submit_aic_deps(rt, 8, p_t8);
            total_tasks++;

            for (int64_t gi0 = 0; gi0 < 8; gi0 += 2) {
                ArgWithDeps<256> p_t9;  // online_softmax
                p_t9.add_input(all_oi_tmp);
                p_t9.add_input(all_cur_mi);
                p_t9.add_input(all_cur_li);
                p_t9.add_output(attn_row);
                p_t9.add_scalar(gi0);
                p_t9.add_scalar(ctx_blocks);
                p_t9.add_dep(sv_matmul_id);
                online_softmax_tasks_by_b[static_cast<size_t>(b)].push_back(submit_aiv_deps(rt, 9, p_t9));
                total_tasks++;
            }
        }

        for (int64_t b0 = 0; b0 < batch_padded; b0 += 16) {
            uint32_t resid_shapes[2]   = {16, 5120};
            uint32_t gm_pipe_shapes[1] = {16384u * 40u};
            uint32_t post_shapes[2]    = {16, 5120};
            uint32_t mlp_shapes[2]     = {16, 17408};
            Tensor resid1_tile     = mk(resid_shapes, 2, DataType::FLOAT32);
            Tensor gm_pipe_buffer_0= mk(gm_pipe_shapes, 1, DataType::FLOAT32);
            Tensor post_norm_tile  = mk(post_shapes, 2, DataType::BFLOAT16);
            Tensor mlp_tile        = mk(mlp_shapes, 2, DataType::BFLOAT16);
            const int64_t cur_valid = std::min<int64_t>(user_batch - b0, 16);

            ArgWithDeps<256> p_t10;  // out_proj_residual (MixedKernels)
            p_t10.add_input(ext_hidden_states);
            p_t10.add_input(attn_out);
            p_t10.add_input(ext_wo);
            p_t10.add_inout(resid1_tile);
            p_t10.add_output(gm_pipe_buffer_0);
            p_t10.add_scalar(b0);
            p_t10.add_scalar(cur_valid);
            p_t10.launch_spec.set_block_num(40);
            for (int64_t __row = 0; __row < cur_valid; ++__row) {
                const int64_t bb = b0 + __row;
                for (const PTO2TaskId& os_tid : online_softmax_tasks_by_b[static_cast<size_t>(bb)]) {
                    p_t10.add_dep(os_tid);
                }
            }
            MixedKernels mixed_10 = {10, 11, 11};
            const PTO2TaskId out_proj_mixed_id = submit_mixed_deps(rt, mixed_10, p_t10);
            total_tasks++;

            ArgWithDeps<256> p_t11;  // post_rmsnorm
            p_t11.add_input(resid1_tile);
            p_t11.add_output(post_norm_tile);
            p_t11.add_input(ext_post_rms_weight);
            p_t11.add_dep(out_proj_mixed_id);
            const PTO2TaskId post_rmsnorm_id = submit_aiv_deps(rt, 12, p_t11);
            total_tasks++;

            std::vector<PTO2TaskId> silu_task_by_ob(34, PTO2TaskId::invalid());
            for (int64_t ob = 0; ob < 34; ob += 1) {
                uint32_t mlp_out_shapes[2] = {16, 512};
                Tensor ret0_out   = mk(mlp_out_shapes, 2, DataType::FLOAT32);
                Tensor ret0_out_1 = mk(mlp_out_shapes, 2, DataType::FLOAT32);
                int64_t mlp_o0 = ob * 512;

                ArgWithDeps<256> p_t12;  // gate_proj
                p_t12.add_input(post_norm_tile);
                p_t12.add_input(ext_w_gate);
                p_t12.add_output(ret0_out);
                p_t12.add_scalar(mlp_o0);
                p_t12.add_dep(post_rmsnorm_id);
                const PTO2TaskId gate_id = submit_aic_deps(rt, 13, p_t12);
                total_tasks++;

                ArgWithDeps<256> p_t13;  // up_proj
                p_t13.add_input(post_norm_tile);
                p_t13.add_input(ext_w_up);
                p_t13.add_output(ret0_out_1);
                p_t13.add_scalar(mlp_o0);
                p_t13.add_dep(post_rmsnorm_id);
                const PTO2TaskId up_id = submit_aic_deps(rt, 14, p_t13);
                total_tasks++;

                uint32_t mlp_off[2] = {0, static_cast<uint32_t>(mlp_o0)};
                uint32_t mlp_win[2] = {16, 512};
                Tensor mlp_slice = mlp_tile.view(mlp_win, mlp_off);

                ArgWithDeps<256> p_t14;  // silu
                p_t14.add_input(ret0_out);
                p_t14.add_input(ret0_out_1);
                p_t14.add_output(mlp_slice);
                p_t14.add_dep(gate_id);
                p_t14.add_dep(up_id);
                silu_task_by_ob[static_cast<size_t>(ob)] = submit_aiv_deps(rt, 15, p_t14);
                total_tasks++;
            }

            for (int64_t dob = 0; dob < 40; dob += 1) {
                uint32_t chunk_shapes[2] = {16, 128};
                Tensor fp32_chunk_gm = mk(chunk_shapes, 2, DataType::FLOAT32);
                int64_t d0 = dob * 128;

                ArgWithDeps<256> p_t15;  // down_proj — reads full mlp_tile
                p_t15.add_input(mlp_tile);
                p_t15.add_input(ext_w_down);
                p_t15.add_inout(fp32_chunk_gm);
                p_t15.add_scalar(d0);
                for (const PTO2TaskId& silu_tid : silu_task_by_ob) {
                    if (silu_tid.is_valid()) p_t15.add_dep(silu_tid);
                }
                const PTO2TaskId down_proj_id = submit_aic_deps(rt, 16, p_t15);
                total_tasks++;

                ArgWithDeps<256> p_t16;  // down_proj_residual
                p_t16.add_input(fp32_chunk_gm);
                p_t16.add_input(resid1_tile);
                p_t16.add_output(ext_out);
                p_t16.add_scalar(d0);
                p_t16.add_scalar(cur_valid);
                p_t16.add_scalar(b0);
                p_t16.add_dep(down_proj_id);
                p_t16.add_dep(out_proj_mixed_id);
                (void)submit_aiv_deps(rt, 17, p_t16);
                total_tasks++;
            }
        }
    }

    pto2_orchestrator_done(rt);
    printf("  Total tasks submitted: %d\n", total_tasks);
}

PTO2Runtime* setup_run(const Qwen3MsCase& tc, GraphCtx& ctx) {
    (void)tc;
    for (int i = 0; i < 10; i++) ctx.args[i] = 0;
    return make_runtime();
}

#if PTO2_PROFILING

void print_config(const Qwen3MsCase& tc) {
    section_header_100('-', "--- Config ---");
    printf("  Config: %s\n", tc.name);
}

#endif  // PTO2_PROFILING
