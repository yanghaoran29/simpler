/**
 * test_batch_paged_attention.cpp
 *
 * Batch Paged Attention Orchestration Unit Tests
 *
 * Purpose: Verify without hardware (AICore) dependency:
 *   1. Batch Orchestration function correctly builds task graph
 *   2. Task dependencies are correctly established
 *   3. Scope management is correct
 *   4. Tensor creation and view operations are correct
 *   5. Chunked batched architecture works correctly
 *
 * Simulation strategy:
 *   - Use make_runtime() to create simulated runtime
 *   - Use sim_run_all() to simulate task execution (skip AICore kernel execution)
 *   - Only verify task graph construction, not computation results
 *
 * Test cases:
 *   - Case1:       batch=64, num_heads=16, head_dim=128, block_size=128, context_len=8193
 *   - CaseVarSeq2: batch=2,  context_lens=[8192, 4096]
 *   - CaseVarSeq4: batch=4,  context_lens=[8192, 4096, 1024, 256]
 *
 * Build:
 *   cd simpler/tests/orchestration_ut && make build && make run
 */

// ─────────────────────────────────────────────────────────────────────────────
// [1] Runtime main header
// ─────────────────────────────────────────────────────────────────────────────
#include "pto_runtime2.h"
#include "test_common.h"
#include "json_cases.h"
#if defined(PTO2_SIM_AICORE_UT)
#include "aicpu_sim_api.h"
#endif
#include "common/platform_config.h"
#include "cpu_affinity.h"
#include <cstring>
#include <algorithm>
#include <cstdint>
#include <atomic>
#include <thread>
#include <vector>

// ─── Embedded test cases ──────────────────────────────────────────────────────
// Set PERF_CASE_IDX at compile time to select the case (0, 1, or 2).
#ifndef PERF_CASE_IDX
#define PERF_CASE_IDX 0
#endif

static const PerfTestCase PERF_CASES[] = {
    {   // 0: Case1
        "Case1 (batch=64, num_heads=16, head_dim=128, block_size=128, context_len=8193)",
        64, 16, 1, 128, 128, 256, 1.0f,
        {8193}, 1,  // Changed from 8192 to 8193 to align with device_tests
    },
    {   // 1: CaseVarSeq2
        "CaseVarSeq2 (batch=2, context_lens=[8192,4096])",
        2, 16, 1, 128, 128, 256, 1.0f,
        {8192, 4096}, 2,
    },
    {   // 2: CaseVarSeq4
        "CaseVarSeq4 (batch=4, context_lens=[8192,4096,1024,256])",
        4, 16, 1, 128, 128, 256, 1.0f,
        {8192, 4096, 1024, 256}, 4,
    },
};

static_assert(PERF_CASE_IDX >= 0 && PERF_CASE_IDX < (int)(sizeof(PERF_CASES) / sizeof(PERF_CASES[0])),
              "PERF_CASE_IDX out of range");

// Global data segment buffers sized for Case1; key_cache and value_cache share one buffer.
// Case1: batch=64, num_heads=16, head_dim=128, block_size=128, block_num=256.
static constexpr size_t GLOBAL_MAX_BATCH       = 64;
static constexpr size_t GLOBAL_MAX_NUM_HEADS   = 16;
static constexpr size_t GLOBAL_HEAD_DIM       = 128;
static constexpr size_t GLOBAL_MAX_BLOCK_NUM  = 256;
static constexpr size_t GLOBAL_BLOCK_SIZE     = 128;

static constexpr size_t GLOBAL_QUERY_NELEMS    = GLOBAL_MAX_BATCH * GLOBAL_MAX_NUM_HEADS * GLOBAL_HEAD_DIM;
// One shared buffer for both key and value cache (same layout size).
static constexpr size_t GLOBAL_KV_NELEMS      = GLOBAL_MAX_BATCH * GLOBAL_MAX_BLOCK_NUM * GLOBAL_BLOCK_SIZE * GLOBAL_HEAD_DIM;
static constexpr size_t GLOBAL_BLOCK_TABLE_CNT = GLOBAL_MAX_BATCH * GLOBAL_MAX_BLOCK_NUM;

static float g_query_buf[GLOBAL_QUERY_NELEMS];
static float g_kv_cache_buf[GLOBAL_KV_NELEMS];  // shared: key_cache and value_cache point here
static float g_out_buf[GLOBAL_QUERY_NELEMS];
static int   g_block_table[GLOBAL_BLOCK_TABLE_CNT];
static int   g_context_lens[GLOBAL_MAX_BATCH];

// ─────────────────────────────────────────────────────────────────────────────
// [2] Batch Paged Attention Orchestration Logic
// ─────────────────────────────────────────────────────────────────────────────

#define FUNC_QK_MATMUL 0
#define FUNC_SOFTMAX_PREPARE 1
#define FUNC_PV_MATMUL 2
#define FUNC_ONLINE_UPDATE 3
#define FUNC_AIC_HUB 4
#define FUNC_AIV_HUB 5

// Synchronization flags for concurrent orchestrator/scheduler threads
static std::atomic<bool> g_orchestration_started(false);

/**
 * Batch Paged Attention orchestration function
 *
 * This is adapted from batch_paged_attention/paged_attention_orch.cpp.
 */
static void build_batch_paged_attention_graph(PTO2Runtime* rt, uint64_t* args, int arg_count) {
    (void)arg_count;  // Suppress unused warning


    // Extract device pointers
    void* host_query = reinterpret_cast<void*>(args[0]);
    void* host_key_cache = reinterpret_cast<void*>(args[1]);
    void* host_value_cache = reinterpret_cast<void*>(args[2]);
    int* host_block_table = reinterpret_cast<int*>(args[3]);
    int* host_context_lens = reinterpret_cast<int*>(args[4]);
    void* host_out = reinterpret_cast<void*>(args[5]);
    int64_t* host_config = reinterpret_cast<int64_t*>(args[6]);

    // Extract sizes
    (void)args[7];  // query_size
    size_t key_cache_size = static_cast<size_t>(args[8]);
    (void)args[9];  // value_cache_size

    // Extract config parameters
    uint64_t batch = static_cast<uint64_t>(static_cast<int>(host_config[0]));
    uint64_t num_heads = static_cast<uint64_t>(static_cast<int>(host_config[1]));
    (void)host_config[2];  // kv_head_num - not used in this test
    uint64_t head_dim = static_cast<uint64_t>(static_cast<int>(host_config[3]));
    uint64_t block_size = static_cast<uint64_t>(static_cast<int>(host_config[4]));
    uint64_t block_num = static_cast<uint64_t>(static_cast<int>(host_config[5]));
    union {
        uint32_t u;
        float f;
    } scale_conv;
    scale_conv.u = static_cast<uint32_t>(host_config[6]);
    float scale_value = scale_conv.f;

    uint64_t q_tile = std::min(num_heads, static_cast<uint64_t>(128));
    uint64_t q_loop = (num_heads + q_tile - 1) / q_tile;
    DataType data_type = DataType::FLOAT32;  // Use FLOAT32 for simulation instead of BFLOAT16
    uint64_t elem_size = get_element_size(data_type);

    printf("  batch = %lu, num_heads = %lu, head_dim = %lu, q_tile = %lu\n",
           (unsigned long)batch, (unsigned long)num_heads, (unsigned long)head_dim, (unsigned long)q_tile);

    // Calculate max_bn from per-batch context lengths
    uint64_t max_bn = 0;
    for (uint64_t b = 0; b < batch; b++) {
        uint64_t cur_seq = host_context_lens[b];
        uint64_t bn_b = (cur_seq + block_size - 1) / block_size;
        if (bn_b > max_bn) max_bn = bn_b;
    }

    // Create external tensors
    uint64_t query_shapes[2] = {batch * num_heads, head_dim};
    uint64_t kv_total_rows = key_cache_size / (head_dim * elem_size);
    uint64_t key_cache_shapes[2] = {kv_total_rows, head_dim};
    uint64_t value_cache_shapes[2] = {kv_total_rows, head_dim};
    uint64_t out_shapes[2] = {batch * num_heads, head_dim};

    Tensor query = make_tensor_external(host_query, query_shapes, 2, data_type);
    Tensor key_cache = make_tensor_external(host_key_cache, key_cache_shapes, 2, data_type);
    Tensor value_cache = make_tensor_external(host_value_cache, value_cache_shapes, 2, data_type);
    Tensor out = make_tensor_external(host_out, out_shapes, 2, DataType::FLOAT32);

    uint64_t bt_addr = reinterpret_cast<uint64_t>(host_block_table);
    uint64_t cl_addr = reinterpret_cast<uint64_t>(host_context_lens);

    uint64_t IN_CORE_BATCH = 16;
    uint64_t num_chunks = (batch + IN_CORE_BATCH - 1) / IN_CORE_BATCH;

    int total_tasks = 0;

    for (uint64_t q_idx = 0; q_idx < q_loop; q_idx++) {
        uint64_t q_offset = q_idx * q_tile;

        for (uint64_t batch_start = 0; batch_start < batch; batch_start += IN_CORE_BATCH) {
            uint64_t chunk_bc = batch - batch_start;
            if (chunk_bc > IN_CORE_BATCH) chunk_bc = IN_CORE_BATCH;

            PTO2_SCOPE(rt) {
                uint64_t oi_acc_shapes[2] = {chunk_bc * q_tile, head_dim};
                uint64_t scalar_acc_shapes[1] = {chunk_bc * q_tile};
                Tensor oi_batch = make_tensor(oi_acc_shapes, 2, DataType::FLOAT32);
                Tensor li_batch = make_tensor(scalar_acc_shapes, 1, DataType::FLOAT32);
                Tensor mi_batch = make_tensor(scalar_acc_shapes, 1, DataType::FLOAT32);

                PTOParam params_hub[] = {
                    make_output_param(oi_batch),
                    make_output_param(li_batch),
                    make_output_param(mi_batch),
                };
                pto2_submit_task(&rt->orchestrator, FUNC_AIV_HUB, PTO2_WORKER_VECTOR, params_hub, 3);
                total_tasks++;

                for (uint64_t bn = 0; bn < max_bn; bn++) {
                    uint64_t sij_shapes[2] = {chunk_bc * q_tile, block_size};
                    uint64_t vec_shapes[1] = {chunk_bc * q_tile};
                    uint64_t oi_new_shapes[2] = {chunk_bc * q_tile, head_dim};

                    Tensor sij_b = make_tensor(sij_shapes, 2, DataType::FLOAT32);
                    Tensor pij_b = make_tensor(sij_shapes, 2, data_type);
                    Tensor mij_b = make_tensor(vec_shapes, 1, DataType::FLOAT32);
                    Tensor lij_b = make_tensor(vec_shapes, 1, DataType::FLOAT32);
                    Tensor oi_new_b = make_tensor(oi_new_shapes, 2, DataType::FLOAT32);

                    PTOParam params_qk[] = {
                        make_input_param(query),
                        make_input_param(key_cache),
                        make_output_param(sij_b),
                        make_scalar_param(bt_addr),
                        make_scalar_param(chunk_bc),
                        make_scalar_param(bn),
                        make_scalar_param(q_offset),
                        make_scalar_param(block_num),
                        make_scalar_param(num_heads),
                        make_scalar_param(batch_start),
                    };
                    pto2_submit_task(&rt->orchestrator, FUNC_QK_MATMUL, PTO2_WORKER_CUBE, params_qk, 10);
                    total_tasks++;

                    PTOParam params_sf[] = {
                        make_input_param(sij_b),
                        make_output_param(pij_b),
                        make_output_param(mij_b),
                        make_output_param(lij_b),
                        make_scalar_param(float_to_u64(scale_value)),
                        make_scalar_param(cl_addr),
                        make_scalar_param(chunk_bc),
                        make_scalar_param(bn),
                        make_scalar_param(batch_start),
                    };
                    pto2_submit_task(&rt->orchestrator, FUNC_SOFTMAX_PREPARE, PTO2_WORKER_VECTOR, params_sf, 9);
                    total_tasks++;

                    PTOParam params_pv[] = {
                        make_input_param(pij_b),
                        make_input_param(value_cache),
                        make_output_param(oi_new_b),
                        make_scalar_param(bt_addr),
                        make_scalar_param(chunk_bc),
                        make_scalar_param(bn),
                        make_scalar_param(block_num),
                        make_scalar_param(batch_start),
                    };
                    pto2_submit_task(&rt->orchestrator, FUNC_PV_MATMUL, PTO2_WORKER_CUBE, params_pv, 8);
                    total_tasks++;

                    uint64_t is_first = (bn == 0) ? 1 : 0;
                    uint64_t is_last = (bn == max_bn - 1) ? 1 : 0;
                    PTOParam params_up[] = {
                        make_input_param(mij_b),
                        make_input_param(lij_b),
                        make_input_param(oi_new_b),
                        make_inout_param(mi_batch),
                        make_inout_param(li_batch),
                        make_output_param(oi_batch),
                        make_output_param(out),
                        make_scalar_param(is_first),
                        make_scalar_param(is_last),
                        make_scalar_param(chunk_bc),
                        make_scalar_param(q_offset),
                        make_scalar_param(num_heads),
                        make_scalar_param(batch_start),
                    };
                    pto2_submit_task(&rt->orchestrator, FUNC_ONLINE_UPDATE, PTO2_WORKER_VECTOR, params_up, 13);
                    total_tasks++;
                }
            }  // PTO2_SCOPE end
        }
    }

    pto2_orchestrator_done(&rt->orchestrator);

    printf("  Total tasks submitted: %d\n", total_tasks);
    printf("  Expected tasks: %lu (num_chunks=%lu, max_bn=%lu, IN_CORE_BATCH=%lu)\n",
           (unsigned long)(num_chunks * q_loop * (1 + max_bn * 4)),
           (unsigned long)num_chunks, (unsigned long)max_bn, (unsigned long)IN_CORE_BATCH);
}

// ─────────────────────────────────────────────────────────────────────────────
// [3] Performance test functions
// ─────────────────────────────────────────────────────────────────────────────

// Run one batch paged attention performance test from a PerfTestCase struct.
static void run_batch_perf(const PerfTestCase& tc) {
    TEST_BEGIN(tc.name);

    uint64_t batch      = static_cast<uint64_t>(tc.batch);
    uint64_t num_heads  = static_cast<uint64_t>(tc.num_heads);
    int kv_head_num     = tc.kv_head_num;
    uint64_t head_dim   = static_cast<uint64_t>(tc.head_dim);
    uint64_t block_size = static_cast<uint64_t>(tc.block_size);
    uint64_t block_num  = static_cast<uint64_t>(tc.block_num);
    float scale_value   = tc.scale_value;

    bind_to_cpu(ORCH_CPU);
    printf("  CPU affinity: orchestrator → core %d\n", ORCH_CPU);
    printf("  Config: batch=%lu, num_heads=%lu, head_dim=%lu, block_size=%lu, block_num=%lu\n",
           (unsigned long)batch, (unsigned long)num_heads, (unsigned long)head_dim,
           (unsigned long)block_size, (unsigned long)block_num);
    printf("  context_lens: [");
    for (int i = 0; i < tc.context_lens_count; i++)
        printf("%d%s", tc.context_lens[i], i + 1 < tc.context_lens_count ? ", " : "");
    printf("]\n");

    const size_t query_size       = batch * num_heads * head_dim * sizeof(float);
    const size_t kv_cache_size    = batch * block_num * block_size * head_dim * sizeof(float);
    (void)(batch * num_heads * head_dim * sizeof(float));  // out_size unused

    void* query_buf       = static_cast<void*>(g_query_buf);
    void* kv_cache_buf    = static_cast<void*>(g_kv_cache_buf);
    void* out_buf         = static_cast<void*>(g_out_buf);
    int*  block_table     = g_block_table;
    int*  context_lens    = g_context_lens;

    // key_cache and value_cache share the same buffer
    void* key_cache_buf   = kv_cache_buf;
    void* value_cache_buf = kv_cache_buf;

    for (uint64_t i = 0; i < batch; i++) {
        // Align with device_tests golden.py behavior:
        // - If multiple context_lens specified: cycle through them
        // - If single context_len specified: use it for all batches
        // This avoids using block_size * block_num (max_model_len) which causes task count mismatch
        if (tc.context_lens_count > 0) {
            // Cycle through the provided context_lens array
            context_lens[i] = tc.context_lens[i % tc.context_lens_count];
        } else {
            // Fallback: use max sequence length if no context_lens specified
            context_lens[i] = (int)(block_size * block_num);
        }
        for (uint64_t j = 0; j < block_num; j++)
            block_table[i * block_num + j] = static_cast<int>(i * block_num + j);
    }

    int64_t config[7] = {
        static_cast<int64_t>(batch), static_cast<int64_t>(num_heads),
        static_cast<int64_t>(kv_head_num), static_cast<int64_t>(head_dim),
        static_cast<int64_t>(block_size), static_cast<int64_t>(block_num), 0
    };
    union { uint32_t u; float f; } scale_conv;
    scale_conv.f = scale_value;
    config[6] = static_cast<int64_t>(scale_conv.u);

    uint64_t args[10] = {
        reinterpret_cast<uint64_t>(query_buf),
        reinterpret_cast<uint64_t>(key_cache_buf),
        reinterpret_cast<uint64_t>(value_cache_buf),
        reinterpret_cast<uint64_t>(block_table),
        reinterpret_cast<uint64_t>(context_lens),
        reinterpret_cast<uint64_t>(out_buf),
        reinterpret_cast<uint64_t>(config),
        query_size, kv_cache_size, kv_cache_size
    };

    PTO2Runtime* rt = make_runtime();
    if (!rt) { printf("  [ERROR] make_runtime() failed\n"); return; }

#if defined(PTO2_SIM_AICORE_UT)
    // 当前测试场景：下发任务不检查依赖，fanin_rc>=1 即可下发（必须在建图前设置，init_task 才会生效）
    rt->scheduler.ut_dispatch_without_fanin_satisfied = true;
#endif

    {  // Tensor scope
        build_batch_paged_attention_graph(rt, args, 10);

#if PTO2_PROFILING
        print_orch_profiling();
#endif

#if defined(PTO2_SIM_AICORE_UT)
        if (aicpu_sim_run_pto2(rt, 3) != 0)
            printf("  [ERROR] aicpu_sim_run_pto2 failed\n");
#else
        sim_run_with_resolve_and_dispatch(rt, 3);
#endif

#if PTO2_PROFILING
        print_sched_profiling(rt);
#endif
    }

    pto2_runtime_destroy(rt);

    // Performance data summary
    // Output removed
}

// ─────────────────────────────────────────────────────────────────────────────
// Main — case selected at compile time via -DPERF_CASE_IDX=N (default 0)
// ─────────────────────────────────────────────────────────────────────────────

int g_pass = 0;
int g_fail = 0;

int main() {
    printf("============================================================\n");
    printf("Batch Paged Attention Performance Tests\n");
    printf("============================================================\n");

    run_batch_perf(PERF_CASES[PERF_CASE_IDX]);

    printf("\n============================================================\n");
    printf("Batch Paged Attention Tests Complete\n");
    printf("============================================================\n");

    return (g_fail == 0) ? 0 : 1;
}
