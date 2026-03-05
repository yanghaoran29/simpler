/**
 * test_batch_paged_attention.cpp
 *
 * Batch Paged Attention Orchestration 单元测试
 *
 * 目的：在不依赖硬件（AICore）的情况下，验证
 *   1. Batch Orchestration 函数能正确构建任务图
 *   2. 任务依赖关系正确建立
 *   3. Scope 管理正确
 *   4. Tensor 创建和视图操作正确
 *   5. Chunked batched architecture 正确工作
 *
 * 模拟策略：
 *   - 使用 make_runtime() 创建模拟运行时
 *   - 使用 sim_run_all() 模拟任务执行（跳过 AICore kernel 执行）
 *   - 只验证任务图的构建，不验证计算结果
 *
 * 编译：
 *   cd simpler/tests/orchestration_ut && make build && make run
 */

// ─────────────────────────────────────────────────────────────────────────────
// [1] Runtime 主头文件
// ─────────────────────────────────────────────────────────────────────────────
#include "pto_runtime2.h"
#include "test_common.h"
#include "common/platform_config.h"
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
// [3] Test functions
// ─────────────────────────────────────────────────────────────────────────────

void test_batch_paged_attention_basic() {
    TEST_BEGIN("test_batch_paged_attention_basic");

    PTO2Runtime* rt = make_runtime();
    CHECK(rt != nullptr);
    if (!rt) return;

#if PTO2_PROFILING
    printf("  Profiling enabled\n");
#endif

    // Prepare test data
    const uint64_t batch = 2;
    const uint64_t num_heads = 16;
    const int kv_head_num = 1;
    const uint64_t head_dim = 16;
    const uint64_t block_size = 16;
    const uint64_t block_num = 2;
    const float scale_value = 0.125f;

    // Allocate test buffers
    const size_t query_size = batch * num_heads * head_dim * sizeof(float);
    const size_t key_cache_size = batch * block_num * block_size * head_dim * sizeof(float);
    const size_t value_cache_size = batch * block_num * block_size * head_dim * sizeof(float);
    const size_t out_size = batch * num_heads * head_dim * sizeof(float);

    void* query_buf = malloc(query_size);
    void* key_cache_buf = malloc(key_cache_size);
    void* value_cache_buf = malloc(value_cache_size);
    void* out_buf = malloc(out_size);
    int* block_table = (int*)malloc(batch * block_num * sizeof(int));
    int* context_lens = (int*)malloc(batch * sizeof(int));

    // Initialize test data
    memset(query_buf, 0, query_size);
    memset(key_cache_buf, 0, key_cache_size);
    memset(value_cache_buf, 0, value_cache_size);
    memset(out_buf, 0, out_size);

    // Set up block table and context lengths
    for (uint64_t i = 0; i < batch; i++) {
        context_lens[i] = block_size * block_num;  // Full blocks
        for (uint64_t j = 0; j < block_num; j++) {
            block_table[i * block_num + j] = static_cast<int>(i * block_num + j);
        }
    }

    // Prepare config
    int64_t config[7] = {
        static_cast<int64_t>(batch),
        static_cast<int64_t>(num_heads),
        static_cast<int64_t>(kv_head_num),
        static_cast<int64_t>(head_dim),
        static_cast<int64_t>(block_size),
        static_cast<int64_t>(block_num),
        0  // scale_value will be set via union
    };
    union {
        uint32_t u;
        float f;
    } scale_conv;
    scale_conv.f = scale_value;
    config[6] = static_cast<int64_t>(scale_conv.u);

    // Prepare args array
    uint64_t args[10] = {
        reinterpret_cast<uint64_t>(query_buf),
        reinterpret_cast<uint64_t>(key_cache_buf),
        reinterpret_cast<uint64_t>(value_cache_buf),
        reinterpret_cast<uint64_t>(block_table),
        reinterpret_cast<uint64_t>(context_lens),
        reinterpret_cast<uint64_t>(out_buf),
        reinterpret_cast<uint64_t>(config),
        query_size,
        key_cache_size,
        value_cache_size
    };

    {  // Scope for Tensors to ensure proper destruction order
        // Build the task graph
        build_batch_paged_attention_graph(rt, args, 10);

#if PTO2_PROFILING
        print_orch_profiling();
#endif

        // Verify tasks were submitted
        CHECK(rt->orchestrator.tasks_submitted > 0);
        printf("  Tasks submitted: %lld\n", (long long)rt->orchestrator.tasks_submitted);

        // Run simulation (skip AICore execution)
        int executed = sim_run_all(rt);
        printf("  Tasks executed: %d\n", executed);

        // Verify all tasks completed
        CHECK(executed == rt->orchestrator.tasks_submitted);
        CHECK(rt->scheduler.tasks_completed.load() == executed);
        CHECK(rt->scheduler.tasks_consumed.load() == executed);

#if PTO2_PROFILING
        printf("  === Scheduler Statistics ===\n");
        pto2_scheduler_print_stats(&rt->scheduler);
        pto2_scheduler_print_queues(&rt->scheduler);
#endif
    }  // Tensors destructed here

    // Cleanup
    free(query_buf);
    free(key_cache_buf);
    free(value_cache_buf);
    free(out_buf);
    free(block_table);
    free(context_lens);

    pto2_runtime_destroy(rt);

    TEST_END();
}

void test_batch_paged_attention_chunked() {
    TEST_BEGIN("test_batch_paged_attention_chunked");

    PTO2Runtime* rt = make_runtime();
    CHECK(rt != nullptr);
    if (!rt) return;

#if PTO2_PROFILING
    printf("  Profiling enabled\n");
#endif

    // Prepare test data with batch > IN_CORE_BATCH to test chunking
    const uint64_t batch = 32;  // Will be split into 2 chunks (IN_CORE_BATCH=16)
    const uint64_t num_heads = 16;
    const int kv_head_num = 1;
    const uint64_t head_dim = 16;
    const uint64_t block_size = 16;
    const uint64_t block_num = 2;
    const float scale_value = 0.125f;

    // Allocate test buffers
    const size_t query_size = batch * num_heads * head_dim * sizeof(float);
    const size_t key_cache_size = batch * block_num * block_size * head_dim * sizeof(float);
    const size_t value_cache_size = batch * block_num * block_size * head_dim * sizeof(float);
    const size_t out_size = batch * num_heads * head_dim * sizeof(float);

    void* query_buf = malloc(query_size);
    void* key_cache_buf = malloc(key_cache_size);
    void* value_cache_buf = malloc(value_cache_size);
    void* out_buf = malloc(out_size);
    int* block_table = (int*)malloc(batch * block_num * sizeof(int));
    int* context_lens = (int*)malloc(batch * sizeof(int));

    // Initialize test data
    memset(query_buf, 0, query_size);
    memset(key_cache_buf, 0, key_cache_size);
    memset(value_cache_buf, 0, value_cache_size);
    memset(out_buf, 0, out_size);

    // Set up block table and context lengths
    for (uint64_t i = 0; i < batch; i++) {
        context_lens[i] = block_size * block_num;  // Full blocks
        for (uint64_t j = 0; j < block_num; j++) {
            block_table[i * block_num + j] = static_cast<int>(i * block_num + j);
        }
    }

    // Prepare config
    int64_t config[7] = {
        static_cast<int64_t>(batch),
        static_cast<int64_t>(num_heads),
        static_cast<int64_t>(kv_head_num),
        static_cast<int64_t>(head_dim),
        static_cast<int64_t>(block_size),
        static_cast<int64_t>(block_num),
        0  // scale_value will be set via union
    };
    union {
        uint32_t u;
        float f;
    } scale_conv;
    scale_conv.f = scale_value;
    config[6] = static_cast<int64_t>(scale_conv.u);

    // Prepare args array
    uint64_t args[10] = {
        reinterpret_cast<uint64_t>(query_buf),
        reinterpret_cast<uint64_t>(key_cache_buf),
        reinterpret_cast<uint64_t>(value_cache_buf),
        reinterpret_cast<uint64_t>(block_table),
        reinterpret_cast<uint64_t>(context_lens),
        reinterpret_cast<uint64_t>(out_buf),
        reinterpret_cast<uint64_t>(config),
        query_size,
        key_cache_size,
        value_cache_size
    };

    {  // Scope for Tensors to ensure proper destruction order
        // Build the task graph
        build_batch_paged_attention_graph(rt, args, 10);

#if PTO2_PROFILING
        print_orch_profiling();
#endif

        // Verify tasks were submitted
        CHECK(rt->orchestrator.tasks_submitted > 0);
        printf("  Tasks submitted: %lld\n", (long long)rt->orchestrator.tasks_submitted);

        // Run simulation (skip AICore execution)
        int executed = sim_run_all(rt);
        printf("  Tasks executed: %d\n", executed);

        // Verify all tasks completed
        CHECK(executed == rt->orchestrator.tasks_submitted);
        CHECK(rt->scheduler.tasks_completed.load() == executed);
        CHECK(rt->scheduler.tasks_consumed.load() == executed);

#if PTO2_PROFILING
        printf("  === Scheduler Statistics ===\n");
        pto2_scheduler_print_stats(&rt->scheduler);
        pto2_scheduler_print_queues(&rt->scheduler);
#endif
    }  // Tensors destructed here

    // Cleanup
    free(query_buf);
    free(key_cache_buf);
    free(value_cache_buf);
    free(out_buf);
    free(block_table);
    free(context_lens);

    pto2_runtime_destroy(rt);

    TEST_END();
}
