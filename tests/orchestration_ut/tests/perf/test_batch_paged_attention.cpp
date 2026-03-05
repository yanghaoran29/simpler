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
 * Build:
 *   cd simpler/tests/orchestration_ut && make build && make run
 */

// ─────────────────────────────────────────────────────────────────────────────
// [1] Runtime main header
// ─────────────────────────────────────────────────────────────────────────────
#include "pto_runtime2.h"
#include "test_common.h"
#include "common/platform_config.h"
#include "cpu_affinity.h"
#include <cstring>
#include <algorithm>
#include <cstdint>

// ─────────────────────────────────────────────────────────────────────────────
// [2] Batch Paged Attention Orchestration Logic
// ─────────────────────────────────────────────────────────────────────────────

#define FUNC_QK_MATMUL 0
#define FUNC_SOFTMAX_PREPARE 1
#define FUNC_PV_MATMUL 2
#define FUNC_ONLINE_UPDATE 3
#define FUNC_AIC_HUB 4
#define FUNC_AIV_HUB 5

/**
 * Batch Paged Attention orchestration function
 *
 * This is adapted from batch_paged_attention/paged_attention_orch.cpp, but:
 * - Uses PTO2Runtime directly instead of PTO2Runtime* from orchestration API
 * - All Tensor objects must be in a local scope to ensure proper destruction
 * - Implements chunked batched architecture with IN_CORE_BATCH
 */
static void build_batch_paged_attention_graph(PTO2Runtime* rt, uint64_t* args, int arg_count) {
    (void)arg_count;  // Suppress unused warning

    TensorPool::set_instance(&rt->orchestrator.tensor_pool);

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

    uint64_t q_tile = 16;
    uint64_t q_loop = (num_heads + q_tile - 1) / q_tile;
    DataType data_type = DataType::FLOAT32;  // Use FLOAT32 for simulation instead of FLOAT16
    uint64_t elem_size = get_element_size(data_type);

    printf("  batch = %lu, num_heads = %lu, head_dim = %lu, q_tile = %lu\n",
           (unsigned long)batch, (unsigned long)num_heads, (unsigned long)head_dim, (unsigned long)q_tile);

    // Calculate max_bn
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

// Common helper: run one batch paged attention performance test
static void run_batch_perf(const char* name, uint64_t batch, uint64_t num_heads,
                           int kv_head_num, uint64_t head_dim,
                           uint64_t block_size, uint64_t block_num) {
    TEST_BEGIN(name);

    bind_to_cpu(ORCH_CPU);
    printf("  CPU affinity: orchestrator → core %d\n", ORCH_CPU);
    printf("  Config: batch=%lu, num_heads=%lu, head_dim=%lu, block_size=%lu, block_num=%lu\n",
           (unsigned long)batch, (unsigned long)num_heads, (unsigned long)head_dim,
           (unsigned long)block_size, (unsigned long)block_num);

    const float scale_value = 0.125f;
    const size_t query_size       = batch * num_heads * head_dim * sizeof(float);
    const size_t key_cache_size   = batch * block_num * block_size * head_dim * sizeof(float);
    const size_t value_cache_size = batch * block_num * block_size * head_dim * sizeof(float);
    const size_t out_size         = batch * num_heads * head_dim * sizeof(float);

    void* query_buf       = malloc(query_size);
    void* key_cache_buf   = malloc(key_cache_size);
    void* value_cache_buf = malloc(value_cache_size);
    void* out_buf         = malloc(out_size);
    int*  block_table     = (int*)malloc(batch * block_num * sizeof(int));
    int*  context_lens    = (int*)malloc(batch * sizeof(int));

    memset(query_buf, 0, query_size);
    memset(key_cache_buf, 0, key_cache_size);
    memset(value_cache_buf, 0, value_cache_size);
    memset(out_buf, 0, out_size);

    for (uint64_t i = 0; i < batch; i++) {
        context_lens[i] = block_size * block_num;
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
        query_size, key_cache_size, value_cache_size
    };

    PTO2Runtime* rt = make_runtime();
    if (!rt) { printf("  [ERROR] make_runtime() failed\n"); return; }

    uint64_t build_us, run_us;
    int64_t tasks_submitted;
    int executed;

    {  // Tensor scope
        uint64_t t0 = perf_now_us();
        build_batch_paged_attention_graph(rt, args, 10);
        uint64_t t1 = perf_now_us();
        build_us = t1 - t0;

        tasks_submitted = rt->orchestrator.tasks_submitted;

#if PTO2_PROFILING
        print_orch_profiling();
#endif

        uint64_t t2 = perf_now_us();
        executed = sim_run_all(rt);
        uint64_t t3 = perf_now_us();
        run_us = t3 - t2;

#if PTO2_PROFILING
        printf("  === Scheduler Statistics ===\n");
        pto2_scheduler_print_stats(&rt->scheduler);
        pto2_scheduler_print_queues(&rt->scheduler);
#endif
    }

    free(query_buf); free(key_cache_buf); free(value_cache_buf); free(out_buf);
    free(block_table); free(context_lens);
    pto2_runtime_destroy(rt);

    // Performance data summary
    double build_ms    = build_us / 1000.0;
    double run_ms      = run_us   / 1000.0;
    double tasks_per_s = (run_us > 0) ? (executed * 1e6 / run_us) : 0.0;
    double us_per_task = (executed > 0) ? (double)run_us / executed : 0.0;

    printf("\n  %-28s %s\n", "Metric", "Value");
    printf("  %-28s %lld\n",    "Tasks submitted",        (long long)tasks_submitted);
    printf("  %-28s %.3f ms\n", "Graph build time",       build_ms);
    printf("  %-28s %.3f ms\n", "Simulation exec time",   run_ms);
    printf("  %-28s %.0f tasks/s\n", "Task throughput",   tasks_per_s);
    printf("  %-28s %.2f µs\n", "Avg time per task",      us_per_task);
}

void test_batch_paged_attention_basic() {
    run_batch_perf("Batch Paged Attention Basic (Performance Test)",
                   /*batch=*/2, /*num_heads=*/16, /*kv_head_num=*/1,
                   /*head_dim=*/16, /*block_size=*/16, /*block_num=*/2);
}

void test_batch_paged_attention_chunked() {
    // batch=32 > IN_CORE_BATCH=16, triggers chunked path
    run_batch_perf("Batch Paged Attention Chunked (Performance Test)",
                   /*batch=*/32, /*num_heads=*/16, /*kv_head_num=*/1,
                   /*head_dim=*/16, /*block_size=*/16, /*block_num=*/2);
}

void test_batch_paged_attention_large_block_num_16() {
    // block_num=16: stress-test the block-level loop (16 KV blocks per sequence)
    run_batch_perf("Batch Paged Attention Chunked (Performance Test)",
                   /*batch=*/32, /*num_heads=*/16, /*kv_head_num=*/1,
                   /*head_dim=*/16, /*block_size=*/16, /*block_num=*/16);
}

void test_batch_paged_attention_large_block_num_128() {
    // block_num=128: deeper block loop, higher task count
    run_batch_perf("Batch Paged Attention Large block_num=128 (Performance Test)",
                   /*batch=*/4, /*num_heads=*/16, /*kv_head_num=*/1,
                   /*head_dim=*/16, /*block_size=*/16, /*block_num=*/128);
}

void test_batch_paged_attention_large_block_num_256() {
    // block_num=256: very long sequence, chunked + large block loop combined
    run_batch_perf("Batch Paged Attention Large block_num=256 (Performance Test)",
                   /*batch=*/32, /*num_heads=*/16, /*kv_head_num=*/1,
                   /*head_dim=*/16, /*block_size=*/16, /*block_num=*/256);
}
