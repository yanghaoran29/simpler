/**
 * Qwen3 prefill layer perf case (PERF_BACKEND=12): host harness + PyPTO-generated orchestration.
 * Include pto_runtime2.h before pto_orchestration_api.h so the TU uses one PTO2Runtime definition.
 */
// test_common before pto_orchestration_api: tensor_factory + make_runtime; orch header
// then overrides PTO2_SCOPE for generated TLS/orchestration path when PTO_RUNTIME2_H is set.
#include "pto_runtime2.h"
#include "test_common.h"
#include "pto_orchestration_api.h"
#include "sim_aicore.h"
#include "common/platform_config.h"
#include "cpu_affinity.h"
#include "task_args.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#ifndef PERF_CASE_IDX
#define PERF_CASE_IDX 0
#endif

struct Qwen3TestCase {
    const char* name;
    /** Effective sequence length per batch (<= kMaxSeq). Drives tok_blocks in generated orchestration. */
    int32_t seq_len_per_batch;
    /** If true, only batch 0 uses seq_len_per_batch; other batches get 0 (skip inner token work). */
    bool single_batch_only;
};

static constexpr int QWEN3_CASE_COUNT = 2;
static_assert(PERF_CASE_IDX >= 0 && PERF_CASE_IDX < QWEN3_CASE_COUNT, "PERF_CASE_IDX out of range");

struct GraphCtx {
    int64_t  config[4];
    uint64_t args[10];
};

const Qwen3TestCase PERF_CASES[QWEN3_CASE_COUNT] = {
    {"Qwen3 prefill layer (full seq, all batches)", 128, false},
    {"Qwen3 prefill layer (light: small tensors, batch0 seq=16)", 16, true},
};

namespace {

// PERF_CASE_IDX selects tensor geometry and derived loop counts (must match generated orchestration below).
#if PERF_CASE_IDX == 0
constexpr uint32_t kB = 16;
constexpr uint32_t kMaxSeq = 128;
constexpr uint32_t kH = 5120;
constexpr uint32_t kKv = 1024;
constexpr uint32_t kRopeD = 128;
constexpr uint32_t kRopeRows = 4096;
constexpr uint32_t kFfnChunk = 64;
constexpr uint32_t kFfnChunks = 400;
constexpr int kTokTile = 16;
constexpr int kQtileLoops = 10;
constexpr int kKVtileLoops = 2;
constexpr int kAttnCtxBlock = 64;
constexpr int kNumKvHeads = 8;
constexpr int kHeadDim = 128;
constexpr int kCacheHeadStride = 4096;
constexpr int kGiLoops = 8;
constexpr int kDownInnerLoops = 10;
#elif PERF_CASE_IDX == 1
constexpr uint32_t kB = 2;
constexpr uint32_t kMaxSeq = 32;
constexpr uint32_t kH = 1024;
constexpr uint32_t kKv = 1024;
constexpr uint32_t kRopeD = 128;
constexpr uint32_t kRopeRows = 512;
constexpr uint32_t kFfnChunk = 64;
constexpr uint32_t kFfnChunks = 8;
constexpr int kTokTile = 16;
constexpr int kQtileLoops = 2;
constexpr int kKVtileLoops = 2;
constexpr int kAttnCtxBlock = 64;
constexpr int kNumKvHeads = 8;
constexpr int kHeadDim = 128;
constexpr int kCacheHeadStride = 512;
constexpr int kGiLoops = 8;
constexpr int kDownInnerLoops = 2;
#else
#error "Qwen3: PERF_CASE_IDX must be 0 (full) or 1 (light)"
#endif
constexpr uint32_t kFfnInterm = kFfnChunk * kFfnChunks;
constexpr uint32_t kTokTileU = static_cast<uint32_t>(kTokTile);
constexpr uint32_t kRopeHalfU = kRopeD / 2;
constexpr uint32_t kAttnCtxU = static_cast<uint32_t>(kAttnCtxBlock);
constexpr uint32_t kNumKvU = static_cast<uint32_t>(kNumKvHeads);
constexpr uint32_t kHeadDimU = static_cast<uint32_t>(kHeadDim);

struct BytePool {
    std::vector<uint8_t> buf;
    size_t used = 0;
    void reset() { used = 0; }
    static size_t align_up(size_t x, size_t a) { return (x + a - 1) & ~(a - 1); }
    uint8_t* alloc(size_t n, size_t align = 64) {
        size_t o = align_up(used, align);
        if (o + n > buf.size()) buf.resize(o + n);
        used = o + n;
        return buf.data() + o;
    }
};

BytePool g_pool;
ChipStorageTaskArgs g_orch_args;

ContinuousTensor make_ct(void* p, std::initializer_list<uint32_t> sh, DataType dt) {
    ContinuousTensor t{};
    t.data = static_cast<uint64_t>(reinterpret_cast<uintptr_t>(p));
    uint32_t i = 0;
    for (uint32_t v : sh) t.shapes[i++] = v;
    t.ndims = static_cast<uint32_t>(sh.size());
    t.dtype = dt;
    return t;
}

}  // namespace

extern "C" {
PTO2OrchestrationConfig aicpu_orchestration_config(const ChipStorageTaskArgs& orch_args);
void aicpu_orchestration_entry(const ChipStorageTaskArgs& orch_args, int orch_thread_num, int orch_thread_index);
}

void build_graph(PTO2Runtime* rt, uint64_t* args, int arg_count) {
    (void)arg_count;
    auto* orch_args = reinterpret_cast<ChipStorageTaskArgs*>(static_cast<uintptr_t>(args[0]));
    pto2_framework_bind_runtime(rt);
    aicpu_orchestration_entry(*orch_args, 1, 0);
    pto2_framework_bind_runtime(nullptr);
    pto2_orchestrator_done(rt->orchestrators);
}

PTO2Runtime* setup_run(const Qwen3TestCase& tc, GraphCtx& ctx) {
    std::memset(ctx.config, 0, sizeof(ctx.config));
    std::memset(ctx.args, 0, sizeof(ctx.args));
    g_pool.reset();
#if PERF_CASE_IDX == 1
    if (g_pool.buf.capacity() < 64u * 1024u * 1024u) {
        g_pool.buf.reserve(64u * 1024u * 1024u);
    }
#else
    // Full case 实际用量约 970MiB；预留过小会触发 vector 重分配，此前 alloc 返回的指针及 g_orch_args 内
    // ContinuousTensor::data 全部悬空 → 编排线程首访 seq_lens 即 SIGSEGV。
    if (g_pool.buf.capacity() < 1024u * 1024u * 1024u) {
        g_pool.buf.reserve(1024u * 1024u * 1024u);
    }
#endif

    const int32_t seq_cap = static_cast<int32_t>(kMaxSeq);
    if (tc.seq_len_per_batch < 0 || tc.seq_len_per_batch > seq_cap) {
        std::fprintf(stderr, "setup_run: invalid seq_len_per_batch=%d (max %d)\n", tc.seq_len_per_batch, seq_cap);
        std::abort();
    }

    uint8_t* hidden = g_pool.alloc(static_cast<size_t>(kB) * kMaxSeq * kH * sizeof(float));
    uint8_t* out = g_pool.alloc(static_cast<size_t>(kB) * kMaxSeq * kH * sizeof(float));
    uint8_t* seq_lens = g_pool.alloc(static_cast<size_t>(kB) * sizeof(int32_t), sizeof(int32_t));
    for (uint32_t i = 0; i < kB; i++) {
        int32_t sl = 0;
        if (tc.single_batch_only) {
            sl = (i == 0) ? tc.seq_len_per_batch : 0;
        } else {
            sl = tc.seq_len_per_batch;
        }
        reinterpret_cast<int32_t*>(seq_lens)[i] = sl;
    }

    uint8_t* rope_cos = g_pool.alloc(static_cast<size_t>(kRopeRows) * kRopeD * sizeof(float));
    uint8_t* rope_sin = g_pool.alloc(static_cast<size_t>(kRopeRows) * kRopeD * sizeof(float));
    uint8_t* k_cache = g_pool.alloc(static_cast<size_t>(kB) * kMaxSeq * kKv * sizeof(float));
    uint8_t* v_cache = g_pool.alloc(static_cast<size_t>(kB) * kMaxSeq * kKv * sizeof(float));
    uint8_t* rms_pre = g_pool.alloc(static_cast<size_t>(kH) * sizeof(float), sizeof(float));
    uint8_t* rms_post = g_pool.alloc(static_cast<size_t>(kH) * sizeof(float), sizeof(float));

    uint8_t* wq = g_pool.alloc(static_cast<size_t>(kH) * kH * sizeof(uint16_t), sizeof(uint16_t));
    uint8_t* wk = g_pool.alloc(static_cast<size_t>(kH) * kKv * sizeof(uint16_t), sizeof(uint16_t));
    uint8_t* wv = g_pool.alloc(static_cast<size_t>(kH) * kKv * sizeof(uint16_t), sizeof(uint16_t));
    uint8_t* wo = g_pool.alloc(static_cast<size_t>(kH) * kH * sizeof(uint16_t), sizeof(uint16_t));
    uint8_t* w_gate = g_pool.alloc(static_cast<size_t>(kH) * kFfnInterm * sizeof(uint16_t), sizeof(uint16_t));
    uint8_t* w_up = g_pool.alloc(static_cast<size_t>(kH) * kFfnInterm * sizeof(uint16_t), sizeof(uint16_t));
    uint8_t* w_down = g_pool.alloc(static_cast<size_t>(kFfnInterm) * kH * sizeof(uint16_t), sizeof(uint16_t));

    g_orch_args.clear();
    g_orch_args.add_tensor(make_ct(hidden, {kB, kMaxSeq, kH}, DataType::FLOAT32));
    g_orch_args.add_tensor(make_ct(seq_lens, {kB}, DataType::INT32));
    g_orch_args.add_tensor(make_ct(rope_cos, {kRopeRows, kRopeD}, DataType::FLOAT32));
    g_orch_args.add_tensor(make_ct(rope_sin, {kRopeRows, kRopeD}, DataType::FLOAT32));
    g_orch_args.add_tensor(make_ct(k_cache, {kB, kMaxSeq, kKv}, DataType::FLOAT32));
    g_orch_args.add_tensor(make_ct(v_cache, {kB, kMaxSeq, kKv}, DataType::FLOAT32));
    g_orch_args.add_tensor(make_ct(rms_pre, {kH}, DataType::FLOAT32));
    g_orch_args.add_tensor(make_ct(wq, {kH, kH}, DataType::BFLOAT16));
    g_orch_args.add_tensor(make_ct(wk, {kH, kKv}, DataType::BFLOAT16));
    g_orch_args.add_tensor(make_ct(wv, {kH, kKv}, DataType::BFLOAT16));
    g_orch_args.add_tensor(make_ct(wo, {kH, kH}, DataType::BFLOAT16));
    g_orch_args.add_tensor(make_ct(rms_post, {kH}, DataType::FLOAT32));
    g_orch_args.add_tensor(make_ct(w_gate, {kH, kFfnInterm}, DataType::BFLOAT16));
    g_orch_args.add_tensor(make_ct(w_up, {kH, kFfnInterm}, DataType::BFLOAT16));
    g_orch_args.add_tensor(make_ct(w_down, {kFfnInterm, kH}, DataType::BFLOAT16));
    g_orch_args.add_tensor(make_ct(out, {kB, kMaxSeq, kH}, DataType::FLOAT32));

    ctx.args[0] = static_cast<uint64_t>(reinterpret_cast<uintptr_t>(&g_orch_args));
    PTO2Runtime* rt = make_runtime();
    return rt;
}

#if PTO2_PROFILING
void print_config(const Qwen3TestCase& tc) {
    section_header_100('-', "--- Config ---");
    printf("  %s\n", tc.name);
    printf("  batch=%u max_seq=%u hidden=%u kv=%u ffn_intermediate=%u\n",
        kB, kMaxSeq, kH, kKv, kFfnInterm);
    printf("  seq_len_per_batch=%d single_batch_only=%d\n",
        tc.seq_len_per_batch,
        tc.single_batch_only ? 1 : 0);
}
#endif

// --- PyPTO-generated orchestration ---
// Orchestration Function: qwen3_prefill_layer
// Generated by PyPTO IR Compiler

extern "C" {

__attribute__((visibility("default")))
PTO2OrchestrationConfig aicpu_orchestration_config(const ChipStorageTaskArgs& orch_args) {
    (void)orch_args;
    return PTO2OrchestrationConfig{
        .expected_arg_count = 16,
    };
}

__attribute__((visibility("default")))
void aicpu_orchestration_entry(const ChipStorageTaskArgs& orch_args, int orch_thread_num, int orch_thread_index) {
    (void)orch_thread_num;
    (void)orch_thread_index;

    // External tensors
    Tensor ext_hidden_states = from_tensor_arg(orch_args.tensor(0));
    Tensor ext_seq_lens = from_tensor_arg(orch_args.tensor(1));
    Tensor ext_rope_cos = from_tensor_arg(orch_args.tensor(2));
    Tensor ext_rope_sin = from_tensor_arg(orch_args.tensor(3));
    Tensor ext_k_cache = from_tensor_arg(orch_args.tensor(4));
    Tensor ext_v_cache = from_tensor_arg(orch_args.tensor(5));
    Tensor ext_input_rms_weight = from_tensor_arg(orch_args.tensor(6));
    Tensor ext_wq = from_tensor_arg(orch_args.tensor(7));
    Tensor ext_wk = from_tensor_arg(orch_args.tensor(8));
    Tensor ext_wv = from_tensor_arg(orch_args.tensor(9));
    Tensor ext_wo = from_tensor_arg(orch_args.tensor(10));
    Tensor ext_post_rms_weight = from_tensor_arg(orch_args.tensor(11));
    Tensor ext_w_gate = from_tensor_arg(orch_args.tensor(12));
    Tensor ext_w_up = from_tensor_arg(orch_args.tensor(13));
    Tensor ext_w_down = from_tensor_arg(orch_args.tensor(14));
    Tensor ext_out = from_tensor_arg(orch_args.tensor(15));

    PTO2_SCOPE() {
        for (int64_t b = 0; b < static_cast<int64_t>(kB); b += 1) {
            PTO2_SCOPE() {
                size_t idx_seq_len_b = static_cast<size_t>(b);
                int32_t seq_len_b = static_cast<int32_t*>(orch_args.tensor(1).data_as<void>())[idx_seq_len_b];
                const int64_t tt = static_cast<int64_t>(kTokTile);
                int64_t tok_blocks = (((static_cast<int64_t>(seq_len_b) + tt) - 1) / tt);
                for (int64_t p0_idx = 0; p0_idx < tok_blocks; p0_idx += 1) {
                    PTO2_SCOPE() {
                        int64_t p0 = (p0_idx * tt);
                        int64_t valid_tok = std::min<int64_t>(tt, (static_cast<int64_t>(seq_len_b) - p0));
                        uint32_t sq_sum_ci_shapes[2] = {kTokTileU, 1};
                        TensorCreateInfo sq_sum_ci(sq_sum_ci_shapes, 2, DataType::FLOAT32);
                        uint32_t ret0__out_ci_shapes[2] = {kTokTileU, 1};
                        TensorCreateInfo ret0__out_ci(ret0__out_ci_shapes, 2, DataType::FLOAT32);

                        // Task 0: qwen3_prefill_layer_incore_0
                        Arg params_t0;
                        params_t0.add_input(ext_hidden_states);
                        params_t0.add_output(sq_sum_ci);
                        params_t0.add_output(ret0__out_ci);
                        params_t0.add_scalar(b);
                        params_t0.add_scalar(p0);
                        params_t0.add_scalar(valid_tok);
                        TaskOutputTensors outs_t0 = pto2_rt_submit_aiv_task(0, params_t0);
                        const Tensor& sq_sum = outs_t0.get_ref(0);
                        const Tensor& ret0__out = outs_t0.get_ref(1);
                        const Tensor& inv_rms = ret0__out;
                        uint32_t normed_tile_ci_shapes[2] = {kTokTileU, kH};
                        TensorCreateInfo normed_tile_ci(normed_tile_ci_shapes, 2, DataType::BFLOAT16);

                        // Task 1: qwen3_prefill_layer_incore_1
                        Arg params_t1;
                        params_t1.add_input(ext_hidden_states);
                        params_t1.add_input(ext_input_rms_weight);
                        params_t1.add_input(inv_rms);
                        params_t1.add_output(normed_tile_ci);
                        params_t1.add_scalar(b);
                        params_t1.add_scalar(p0);
                        params_t1.add_scalar(valid_tok);
                        TaskOutputTensors outs_t1 = pto2_rt_submit_aiv_task(1, params_t1);
                        const Tensor& normed_tile = outs_t1.get_ref(0);
                        const Tensor& normed_tile__rv_v2 = normed_tile;
                        uint32_t q_proj_tile_ci_shapes[2] = {kTokTileU, kH};
                        TensorCreateInfo q_proj_tile_ci(q_proj_tile_ci_shapes, 2, DataType::BFLOAT16);
                        uint32_t k_proj_tile_ci_shapes[2] = {kTokTileU, kKv};
                        TensorCreateInfo k_proj_tile_ci(k_proj_tile_ci_shapes, 2, DataType::BFLOAT16);
                        uint32_t v_proj_tile_ci_shapes[2] = {kTokTileU, kKv};
                        TensorCreateInfo v_proj_tile_ci(v_proj_tile_ci_shapes, 2, DataType::BFLOAT16);
                        Tensor q_proj_tile__loop_state = make_tensor_external(nullptr, q_proj_tile_ci_shapes, 2, DataType::BFLOAT16);
                        for (int64_t ob = 0; ob < static_cast<int64_t>(kQtileLoops); ob += 1) {
                            PTO2_SCOPE() {

                                // Group qwen3_prefill_layer_incore_2: MixedKernels (AIC + AIV)
                                Arg params_t2;
                                params_t2.add_input(normed_tile__rv_v2);
                                params_t2.add_output(q_proj_tile_ci);
                                params_t2.add_input(ext_wq);
                                params_t2.add_scalar(ob);
                                MixedKernels mixed_2 = {2, 3, 3};
                                TaskOutputTensors outs_t2 = pto2_rt_submit_task(mixed_2, params_t2);
                                const Tensor& q_proj_tile = outs_t2.get_ref(0);
                                q_proj_tile__loop_state = q_proj_tile;
                                const Tensor& q_proj_tile__co_l1_rv_v1 = q_proj_tile;
                            }
                        }
                        Tensor k_proj_tile__loop_state = make_tensor_external(nullptr, k_proj_tile_ci_shapes, 2, DataType::BFLOAT16);
                        Tensor v_proj_tile__loop_state = make_tensor_external(nullptr, v_proj_tile_ci_shapes, 2, DataType::BFLOAT16);
                        for (int64_t ob = 0; ob < static_cast<int64_t>(kKVtileLoops); ob += 1) {
                            PTO2_SCOPE() {

                                // Group qwen3_prefill_layer_incore_3: MixedKernels (AIC + AIV)
                                Arg params_t3;
                                params_t3.add_output(k_proj_tile_ci);
                                params_t3.add_input(normed_tile__rv_v2);
                                params_t3.add_output(v_proj_tile_ci);
                                params_t3.add_input(ext_wk);
                                params_t3.add_input(ext_wv);
                                params_t3.add_scalar(ob);
                                MixedKernels mixed_3 = {4, 5, 5};
                                TaskOutputTensors outs_t3 = pto2_rt_submit_task(mixed_3, params_t3);
                                const Tensor& k_proj_tile = outs_t3.get_ref(0);
                                k_proj_tile__loop_state = k_proj_tile;
                                const Tensor& v_proj_tile = outs_t3.get_ref(1);
                                v_proj_tile__loop_state = v_proj_tile;
                                const Tensor& k_proj_tile__co_l1_rv_v1 = k_proj_tile;
                                const Tensor& v_proj_tile__co_l1_rv_v1 = v_proj_tile;
                            }
                        }
                        uint32_t attn_tile_ci_shapes[2] = {kTokTileU, kH};
                        TensorCreateInfo attn_tile_ci(attn_tile_ci_shapes, 2, DataType::FLOAT32);

                        // Task 4: qwen3_prefill_layer_incore_4
                        Arg params_t4;
                        params_t4.add_output(attn_tile_ci);
                        TaskOutputTensors outs_t4 = pto2_rt_submit_aiv_task(6, params_t4);
                        const Tensor& attn_tile = outs_t4.get_ref(0);
                        const Tensor& attn_tile__rv_v2 = attn_tile;
                        for (int64_t ti = 0; ti < valid_tok; ti += 1) {
                            PTO2_SCOPE() {
                                int64_t pos = (p0 + ti);
                                int64_t ctx_len = (pos + 1);
                                const int64_t acb = static_cast<int64_t>(kAttnCtxBlock);
                                int64_t ctx_blocks = (((ctx_len + acb) - 1) / acb);
                                uint32_t cos_row_shapes[2] = {1, kRopeD};
                                uint32_t cos_row_offsets[2] = {static_cast<uint32_t>(pos), 0};
                                Tensor cos_row = ext_rope_cos.view(cos_row_shapes, cos_row_offsets);
                                uint32_t sin_row_shapes[2] = {1, kRopeD};
                                uint32_t sin_row_offsets[2] = {static_cast<uint32_t>(pos), 0};
                                Tensor sin_row = ext_rope_sin.view(sin_row_shapes, sin_row_offsets);
                                uint32_t cos_lo_shapes[2] = {1, kRopeHalfU};
                                uint32_t cos_lo_offsets[2] = {0, 0};
                                Tensor cos_lo = cos_row.view(cos_lo_shapes, cos_lo_offsets);
                                uint32_t cos_hi_shapes[2] = {1, kRopeHalfU};
                                uint32_t cos_hi_offsets[2] = {0, kRopeHalfU};
                                Tensor cos_hi = cos_row.view(cos_hi_shapes, cos_hi_offsets);
                                uint32_t sin_lo_shapes[2] = {1, kRopeHalfU};
                                uint32_t sin_lo_offsets[2] = {0, 0};
                                Tensor sin_lo = sin_row.view(sin_lo_shapes, sin_lo_offsets);
                                uint32_t sin_hi_shapes[2] = {1, kRopeHalfU};
                                uint32_t sin_hi_offsets[2] = {0, kRopeHalfU};
                                Tensor sin_hi = sin_row.view(sin_hi_shapes, sin_hi_offsets);
                                uint32_t k_group_ci_shapes[2] = {kNumKvU, kHeadDimU};
                                TensorCreateInfo k_group_ci(k_group_ci_shapes, 2, DataType::FLOAT32);

                                // Task 5: qwen3_prefill_layer_incore_5
                                Arg params_t5;
                                params_t5.add_output(k_group_ci);
                                params_t5.add_input(k_proj_tile__loop_state);
                                params_t5.add_scalar(ti);
                                TaskOutputTensors outs_t5 = pto2_rt_submit_aiv_task(7, params_t5);
                                const Tensor& k_group = outs_t5.get_ref(0);
                                const Tensor& k_group__rv_v2 = k_group;

                                // Task 6: qwen3_prefill_layer_incore_6
                                Arg params_t6;
                                params_t6.add_input(cos_hi);
                                params_t6.add_input(cos_lo);
                                params_t6.add_inout(ext_k_cache);
                                params_t6.add_input(k_group__rv_v2);
                                params_t6.add_input(sin_hi);
                                params_t6.add_input(sin_lo);
                                params_t6.add_inout(ext_v_cache);
                                params_t6.add_input(v_proj_tile__loop_state);
                                params_t6.add_scalar(b);
                                params_t6.add_scalar(pos);
                                params_t6.add_scalar(ti);
                                pto2_rt_submit_aiv_task(8, params_t6);
                                const Tensor& k_cache__rv_v8 = ext_k_cache;
                                const Tensor& v_cache__rv_v8 = ext_v_cache;
                                uint32_t attn_row_ci_shapes[2] = {1, kH};
                                TensorCreateInfo attn_row_ci(attn_row_ci_shapes, 2, DataType::FLOAT32);
                                uint32_t ret0__out_1_shapes[2] = {1, kH};
                                uint32_t ret0__out_1_offsets[2] = {static_cast<uint32_t>(ti), 0};
                                Tensor ret0__out_1 = attn_tile__rv_v2.view(ret0__out_1_shapes, ret0__out_1_offsets);

                                // Task 7: qwen3_prefill_layer_incore_7
                                Arg params_t7;
                                params_t7.add_output(attn_row_ci);
                                params_t7.add_inout(ret0__out_1);
                                TaskOutputTensors outs_t7 = pto2_rt_submit_aiv_task(9, params_t7);
                                const Tensor& attn_row = outs_t7.get_ref(0);
                                const Tensor& attn_row__ssa_v1 = ret0__out_1;
                                for (int64_t gi = 0; gi < static_cast<int64_t>(kGiLoops); gi += 1) {
                                    PTO2_SCOPE() {
                                        int64_t kvh = (gi / 1);
                                        int64_t qg = (gi - (kvh * 1));
                                        int64_t q_base = ((kvh * static_cast<int64_t>(kNumKvHeads)) + (qg * static_cast<int64_t>(kNumKvHeads)));
                                        uint32_t q_group_ci_shapes[2] = {kNumKvU, kHeadDimU};
                                        TensorCreateInfo q_group_ci(q_group_ci_shapes, 2, DataType::FLOAT32);

                                        // Task 8: qwen3_prefill_layer_incore_8
                                        Arg params_t8;
                                        params_t8.add_output(q_group_ci);
                                        params_t8.add_input(q_proj_tile__loop_state);
                                        params_t8.add_scalar(q_base);
                                        params_t8.add_scalar(ti);
                                        TaskOutputTensors outs_t8 = pto2_rt_submit_aiv_task(10, params_t8);
                                        const Tensor& q_group = outs_t8.get_ref(0);
                                        const Tensor& q_group__rv_v2 = q_group;
                                        uint32_t ret0__out_2_ci_shapes[2] = {kNumKvU, kHeadDimU};
                                        TensorCreateInfo ret0__out_2_ci(ret0__out_2_ci_shapes, 2, DataType::BFLOAT16);

                                        // Task 9: qwen3_prefill_layer_incore_9
                                        Arg params_t9;
                                        params_t9.add_input(cos_hi);
                                        params_t9.add_input(cos_lo);
                                        params_t9.add_input(q_group__rv_v2);
                                        params_t9.add_input(sin_hi);
                                        params_t9.add_input(sin_lo);
                                        params_t9.add_output(ret0__out_2_ci);
                                        TaskOutputTensors outs_t9 = pto2_rt_submit_aiv_task(11, params_t9);
                                        const Tensor& ret0__out_2 = outs_t9.get_ref(0);
                                        const Tensor& q_rot_bf16 = ret0__out_2;
                                        uint32_t oi_ci_shapes[2] = {kNumKvU, kHeadDimU};
                                        TensorCreateInfo oi_ci(oi_ci_shapes, 2, DataType::FLOAT32);
                                        uint32_t li_ci_shapes[2] = {kNumKvU, 1};
                                        TensorCreateInfo li_ci(li_ci_shapes, 2, DataType::FLOAT32);
                                        uint32_t mi_ci_shapes[2] = {kNumKvU, 1};
                                        TensorCreateInfo mi_ci(mi_ci_shapes, 2, DataType::FLOAT32);
                                        uint32_t ret0__out_3_ci_shapes[2] = {kNumKvU, 1};
                                        TensorCreateInfo ret0__out_3_ci(ret0__out_3_ci_shapes, 2, DataType::FLOAT32);
                                        uint32_t ret1__out_ci_shapes[2] = {kNumKvU, 1};
                                        TensorCreateInfo ret1__out_ci(ret1__out_ci_shapes, 2, DataType::FLOAT32);
                                        uint32_t ret2__out_ci_shapes[2] = {kNumKvU, kHeadDimU};
                                        TensorCreateInfo ret2__out_ci(ret2__out_ci_shapes, 2, DataType::FLOAT32);

                                        // Task 10: qwen3_prefill_layer_incore_10
                                        Arg params_t10;
                                        params_t10.add_output(li_ci);
                                        params_t10.add_output(mi_ci);
                                        params_t10.add_output(oi_ci);
                                        params_t10.add_output(ret0__out_3_ci);
                                        params_t10.add_output(ret1__out_ci);
                                        params_t10.add_output(ret2__out_ci);
                                        TaskOutputTensors outs_t10 = pto2_rt_submit_aiv_task(12, params_t10);
                                        const Tensor& li = outs_t10.get_ref(0);
                                        const Tensor& mi = outs_t10.get_ref(1);
                                        const Tensor& oi = outs_t10.get_ref(2);
                                        const Tensor& ret0__out_3 = outs_t10.get_ref(3);
                                        const Tensor& ret1__out = outs_t10.get_ref(4);
                                        const Tensor& ret2__out = outs_t10.get_ref(5);
                                        const Tensor& li__ssa_v1 = ret0__out_3;
                                        const Tensor& mi__ssa_v1 = ret1__out;
                                        const Tensor& oi__ssa_v1 = ret2__out;
                                        for (int64_t sb = 0; sb < ctx_blocks; sb += 1) {
                                            PTO2_SCOPE() {
                                                int64_t s0 = (sb * acb);
                                                int64_t valid_len = std::min<int64_t>(acb, (ctx_len - s0));
                                                const int64_t chs = static_cast<int64_t>(kCacheHeadStride);
                                                int64_t cache_row0 =
                                                    (((b * static_cast<int64_t>(kNumKvHeads)) * chs) + (kvh * chs)) + s0;
                                                uint32_t ret0__out_4_ci_shapes[2] = {kNumKvU, kAttnCtxU};
                                                TensorCreateInfo ret0__out_4_ci(ret0__out_4_ci_shapes, 2, DataType::FLOAT32);

                                                // Task 11: qwen3_prefill_layer_incore_11
                                                Arg params_t11;
                                                params_t11.add_input(k_cache__rv_v8);
                                                params_t11.add_input(q_rot_bf16);
                                                params_t11.add_output(ret0__out_4_ci);
                                                params_t11.add_scalar(cache_row0);
                                                params_t11.add_scalar(valid_len);
                                                TaskOutputTensors outs_t11 = pto2_rt_submit_aic_task(13, params_t11);
                                                const Tensor& ret0__out_4 = outs_t11.get_ref(0);
                                                const Tensor& raw_scores = ret0__out_4;
                                                uint32_t ret0__out_5_ci_shapes[2] = {kNumKvU, 1};
                                                TensorCreateInfo ret0__out_5_ci(ret0__out_5_ci_shapes, 2, DataType::FLOAT32);
                                                uint32_t ret1__out_1_ci_shapes[2] = {kNumKvU, 1};
                                                TensorCreateInfo ret1__out_1_ci(ret1__out_1_ci_shapes, 2, DataType::FLOAT32);
                                                uint32_t ret2__out_1_ci_shapes[2] = {kNumKvU, kAttnCtxU};
                                                TensorCreateInfo ret2__out_1_ci(ret2__out_1_ci_shapes, 2, DataType::BFLOAT16);

                                                // Task 12: qwen3_prefill_layer_incore_12
                                                Arg params_t12;
                                                params_t12.add_input(raw_scores);
                                                params_t12.add_output(ret0__out_5_ci);
                                                params_t12.add_output(ret1__out_1_ci);
                                                params_t12.add_output(ret2__out_1_ci);
                                                params_t12.add_scalar(valid_len);
                                                TaskOutputTensors outs_t12 = pto2_rt_submit_aiv_task(14, params_t12);
                                                const Tensor& ret0__out_5 = outs_t12.get_ref(0);
                                                const Tensor& ret1__out_1 = outs_t12.get_ref(1);
                                                const Tensor& ret2__out_1 = outs_t12.get_ref(2);
                                                const Tensor& cur_li = ret0__out_5;
                                                const Tensor& cur_mi = ret1__out_1;
                                                const Tensor& exp_pad_bf16 = ret2__out_1;
                                                uint32_t ret0__out_6_ci_shapes[2] = {kNumKvU, kHeadDimU};
                                                TensorCreateInfo ret0__out_6_ci(ret0__out_6_ci_shapes, 2, DataType::FLOAT32);

                                                // Task 13: qwen3_prefill_layer_incore_13
                                                Arg params_t13;
                                                params_t13.add_input(exp_pad_bf16);
                                                params_t13.add_input(v_cache__rv_v8);
                                                params_t13.add_output(ret0__out_6_ci);
                                                params_t13.add_scalar(cache_row0);
                                                params_t13.add_scalar(valid_len);
                                                TaskOutputTensors outs_t13 = pto2_rt_submit_aic_task(15, params_t13);
                                                const Tensor& ret0__out_6 = outs_t13.get_ref(0);
                                                const Tensor& oi_tmp = ret0__out_6;

                                                // Task 14: qwen3_prefill_layer_incore_14
                                                Arg params_t14;
                                                params_t14.add_input(cur_li);
                                                params_t14.add_input(cur_mi);
                                                params_t14.add_inout(li__ssa_v1);
                                                params_t14.add_inout(mi__ssa_v1);
                                                params_t14.add_inout(oi__ssa_v1);
                                                params_t14.add_input(oi_tmp);
                                                params_t14.add_scalar(sb);
                                                pto2_rt_submit_aiv_task(16, params_t14);
                                            }
                                        }

                                        // Task 15: qwen3_prefill_layer_incore_15
                                        Arg params_t15;
                                        params_t15.add_inout(attn_row__ssa_v1);
                                        params_t15.add_input(li__ssa_v1);
                                        params_t15.add_input(oi__ssa_v1);
                                        params_t15.add_scalar(q_base);
                                        pto2_rt_submit_aiv_task(17, params_t15);
                                        const Tensor& attn_row__rv_v5 = attn_row__ssa_v1;
                                    }
                                }
                            }
                        }
                        uint32_t hidden_tok_fp32_ci_shapes[2] = {kTokTileU, kH};
                        TensorCreateInfo hidden_tok_fp32_ci(hidden_tok_fp32_ci_shapes, 2, DataType::FLOAT32);

                        // Task 16: qwen3_prefill_layer_incore_16
                        Arg params_t16;
                        params_t16.add_input(ext_hidden_states);
                        params_t16.add_output(hidden_tok_fp32_ci);
                        params_t16.add_scalar(b);
                        params_t16.add_scalar(p0);
                        params_t16.add_scalar(valid_tok);
                        TaskOutputTensors outs_t16 = pto2_rt_submit_aiv_task(18, params_t16);
                        const Tensor& hidden_tok_fp32 = outs_t16.get_ref(0);
                        const Tensor& hidden_tok_fp32__rv_v2 = hidden_tok_fp32;
                        uint32_t resid1_tile_ci_shapes[2] = {kTokTileU, kH};
                        TensorCreateInfo resid1_tile_ci(resid1_tile_ci_shapes, 2, DataType::FLOAT32);
                        Tensor resid1_tile__loop_state = make_tensor_external(nullptr, resid1_tile_ci_shapes, 2, DataType::FLOAT32);
                        for (int64_t ob = 0; ob < static_cast<int64_t>(kQtileLoops); ob += 1) {
                            PTO2_SCOPE() {

                                // Group qwen3_prefill_layer_incore_17: MixedKernels (AIC + AIV)
                                Arg params_t17;
                                params_t17.add_input(attn_tile__rv_v2);
                                params_t17.add_input(hidden_tok_fp32__rv_v2);
                                params_t17.add_output(resid1_tile_ci);
                                params_t17.add_input(ext_wo);
                                params_t17.add_scalar(ob);
                                MixedKernels mixed_17 = {19, 20, 20};
                                TaskOutputTensors outs_t17 = pto2_rt_submit_task(mixed_17, params_t17);
                                const Tensor& resid1_tile = outs_t17.get_ref(0);
                                resid1_tile__loop_state = resid1_tile;
                                const Tensor& resid1_tile__co_l1_rv_v1 = resid1_tile;
                            }
                        }
                        uint32_t sq_sum__ssa_v5_ci_shapes[2] = {kTokTileU, 1};
                        TensorCreateInfo sq_sum__ssa_v5_ci(sq_sum__ssa_v5_ci_shapes, 2, DataType::FLOAT32);
                        uint32_t ret0__out_7_ci_shapes[2] = {kTokTileU, 1};
                        TensorCreateInfo ret0__out_7_ci(ret0__out_7_ci_shapes, 2, DataType::FLOAT32);

                        // Task 18: qwen3_prefill_layer_incore_18
                        Arg params_t18;
                        params_t18.add_input(resid1_tile__loop_state);
                        params_t18.add_output(sq_sum__ssa_v5_ci);
                        params_t18.add_output(ret0__out_7_ci);
                        TaskOutputTensors outs_t18 = pto2_rt_submit_aiv_task(21, params_t18);
                        const Tensor& sq_sum__ssa_v5 = outs_t18.get_ref(0);
                        const Tensor& ret0__out_7 = outs_t18.get_ref(1);
                        const Tensor& inv_rms__ssa_v1 = ret0__out_7;
                        uint32_t post_norm_tile_ci_shapes[2] = {kTokTileU, kH};
                        TensorCreateInfo post_norm_tile_ci(post_norm_tile_ci_shapes, 2, DataType::BFLOAT16);
                        uint32_t down_proj_tile_ci_shapes[2] = {kTokTileU, kH};
                        TensorCreateInfo down_proj_tile_ci(down_proj_tile_ci_shapes, 2, DataType::FLOAT32);

                        // Task 19: qwen3_prefill_layer_incore_19
                        Arg params_t19;
                        params_t19.add_output(down_proj_tile_ci);
                        params_t19.add_input(inv_rms__ssa_v1);
                        params_t19.add_output(post_norm_tile_ci);
                        params_t19.add_input(ext_post_rms_weight);
                        params_t19.add_input(resid1_tile__loop_state);
                        TaskOutputTensors outs_t19 = pto2_rt_submit_aiv_task(22, params_t19);
                        const Tensor& down_proj_tile = outs_t19.get_ref(0);
                        const Tensor& post_norm_tile = outs_t19.get_ref(1);
                        const Tensor& down_proj_tile__rv_v2 = down_proj_tile;
                        const Tensor& post_norm_tile__rv_v2 = post_norm_tile;
                        for (int64_t ob = 0; ob < static_cast<int64_t>(kFfnChunks); ob += 1) {
                            PTO2_SCOPE() {
                                int64_t o0 = (ob * static_cast<int64_t>(kFfnChunk));
                                uint32_t gate_acc_ci_shapes[2] = {kTokTileU, kFfnChunk};
                                TensorCreateInfo gate_acc_ci(gate_acc_ci_shapes, 2, DataType::FLOAT32);
                                uint32_t up_acc_ci_shapes[2] = {kTokTileU, kFfnChunk};
                                TensorCreateInfo up_acc_ci(up_acc_ci_shapes, 2, DataType::FLOAT32);
                                uint32_t ret0__out_8_ci_shapes[2] = {kTokTileU, kFfnChunk};
                                TensorCreateInfo ret0__out_8_ci(ret0__out_8_ci_shapes, 2, DataType::BFLOAT16);

                                // Group qwen3_prefill_layer_incore_20: MixedKernels (AIC + AIV)
                                Arg params_t20;
                                params_t20.add_output(gate_acc_ci);
                                params_t20.add_input(post_norm_tile__rv_v2);
                                params_t20.add_output(up_acc_ci);
                                params_t20.add_input(ext_w_gate);
                                params_t20.add_input(ext_w_up);
                                params_t20.add_output(ret0__out_8_ci);
                                params_t20.add_scalar(o0);
                                MixedKernels mixed_20 = {23, 24, 24};
                                TaskOutputTensors outs_t20 = pto2_rt_submit_task(mixed_20, params_t20);
                                const Tensor& gate_acc = outs_t20.get_ref(0);
                                const Tensor& up_acc = outs_t20.get_ref(1);
                                const Tensor& ret0__out_8 = outs_t20.get_ref(2);
                                const Tensor& mlp_chunk_bf16 = ret0__out_8;
                                for (int64_t dob = 0; dob < static_cast<int64_t>(kDownInnerLoops); dob += 1) {
                                    PTO2_SCOPE() {

                                        // Group qwen3_prefill_layer_incore_21: MixedKernels (AIC + AIV)
                                        Arg params_t21;
                                        params_t21.add_inout(down_proj_tile__rv_v2);
                                        params_t21.add_input(mlp_chunk_bf16);
                                        params_t21.add_input(ext_w_down);
                                        params_t21.add_scalar(dob);
                                        params_t21.add_scalar(o0);
                                        MixedKernels mixed_21 = {25, 26, 26};
                                        pto2_rt_submit_task(mixed_21, params_t21);
                                        const Tensor& down_proj_tile__co_l1_rv_v6 = down_proj_tile__rv_v2;
                                    }
                                }
                            }
                        }
                        for (int64_t ob = 0; ob < static_cast<int64_t>(kQtileLoops); ob += 1) {
                            PTO2_SCOPE() {

                                // Task 22: qwen3_prefill_layer_incore_22
                                Arg params_t22;
                                params_t22.add_input(down_proj_tile__rv_v2);
                                params_t22.add_inout(ext_out);
                                params_t22.add_input(resid1_tile__loop_state);
                                params_t22.add_scalar(b);
                                params_t22.add_scalar(ob);
                                params_t22.add_scalar(p0);
                                pto2_rt_submit_aiv_task(27, params_t22);
                                const Tensor& out__co_l1_rv_v5 = ext_out;
                            }
                        }
                    }
                }
            }
        }
    }
}

}  // extern "C"
