/**
 * test_paged_attention.cpp
 *
 * Paged Attention Orchestration 单元测试
 *
 * 目的：在不依赖硬件（AICore）的情况下，验证
 *   1. Orchestration 函数能正确构建任务图
 *   2. 任务依赖关系正确建立
 *   3. Scope 管理正确
 *   4. Tensor 创建和视图操作正确
 *
 * 模拟策略：
 *   - 使用 make_small_runtime() 创建模拟运行时
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
#include <cstring>
#include <algorithm>
#include <cstdint>
#include <chrono>

// ─────────────────────────────────────────────────────────────────────────────
// [2] Paged Attention Orchestration Logic (extracted from paged_attention_orch.cpp)
// ─────────────────────────────────────────────────────────────────────────────

#define FUNC_QK_MATMUL 0
#define FUNC_SOFTMAX_PREPARE 1
#define FUNC_PV_MATMUL 2
#define FUNC_ONLINE_UPDATE 3
#define FUNC_AIC_HUB 4
#define FUNC_AIV_HUB 5

/**
 * Paged Attention orchestration function
 * 
 * This is adapted from paged_attention_orch.cpp, but:
 * - Removed hardware-specific cycle counting (get_sys_cnt_aicpu)
 * - Uses PTO2Runtime directly instead of PTO2Runtime* from orchestration API
 * - All Tensor objects must be in a local scope to ensure proper destruction
 * - Uses chrono for timing instead of hardware cycle counting
 */
static void build_paged_attention_graph(PTO2Runtime* rt, uint64_t* args, int arg_count) {
    (void)arg_count;  // Suppress unused warning
    
    auto t_start = std::chrono::high_resolution_clock::now();
    
    TensorPool::set_instance(&rt->orchestrator.tensor_pool);
    
    auto t_tensor_pool = std::chrono::high_resolution_clock::now();
    
    // Extract device pointers
    void* host_query = reinterpret_cast<void*>(args[0]);
    void* host_key_cache = reinterpret_cast<void*>(args[1]);
    void* host_value_cache = reinterpret_cast<void*>(args[2]);
    int* host_block_table = reinterpret_cast<int*>(args[3]);
    int* host_context_lens = reinterpret_cast<int*>(args[4]);
    void* host_out = reinterpret_cast<void*>(args[5]);
    int64_t* host_config = reinterpret_cast<int64_t*>(args[6]);
    
    // Extract sizes (not used in this test, but kept for compatibility)
    (void)args[7];  // query_size
    (void)args[8];  // key_cache_size
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
    uint64_t q_head_num = num_heads;
    uint64_t q_tile = std::min(num_heads, 128UL);
    uint64_t q_loop = (q_head_num + q_tile - 1) / q_tile;
    DataType data_type = DataType::FLOAT32;  // Use FLOAT32 for simulation instead of BFLOAT16
    
    printf("  batch = %lu, num_heads = %lu, head_dim = %lu\n", 
           (unsigned long)batch, (unsigned long)num_heads, (unsigned long)head_dim);
    
    // Create external tensors
    uint64_t query_shapes[2] = {batch * num_heads, head_dim};
    uint64_t key_cache_shapes[2] = {batch * block_num * block_size, head_dim};
    uint64_t value_cache_shapes[2] = {batch * block_num * block_size, head_dim};
    uint64_t out_shapes[2] = {batch * num_heads, head_dim};
    Tensor query = make_tensor_external(host_query, query_shapes, 2, data_type);
    Tensor key_cache = make_tensor_external(host_key_cache, key_cache_shapes, 2, data_type);
    Tensor value_cache = make_tensor_external(host_value_cache, value_cache_shapes, 2, data_type);
    Tensor out = make_tensor_external(host_out, out_shapes, 2, DataType::FLOAT32);
    
    int total_tasks = 0;
    
    for (uint64_t b_idx = 0; b_idx < batch; b_idx++) {
        uint64_t cur_seq = host_context_lens[b_idx];
        uint64_t bn_this_batch = (cur_seq + block_size - 1) / block_size;
        for (uint64_t q_idx = 0; q_idx < q_loop; q_idx++) {
            PTO2_SCOPE(rt) {
                uint64_t cur_offset = b_idx * q_head_num + q_idx * q_tile;
                
                uint64_t oi_shapes[2] = {q_tile, head_dim};
                uint64_t li_shapes[1] = {q_tile};
                uint64_t mi_shapes[1] = {q_tile};
                Tensor oi = make_tensor(oi_shapes, 2, DataType::FLOAT32);
                Tensor li_update = make_tensor(li_shapes, 1, DataType::FLOAT32);
                Tensor mi_update = make_tensor(mi_shapes, 1, DataType::FLOAT32);
                
                uint64_t qi_shapes[2] = {q_tile, head_dim};
                uint64_t qi_offsets[2] = {cur_offset, 0};
                Tensor qi = query.view(qi_shapes, qi_offsets);
                uint64_t out_view_shapes[2] = {q_tile, head_dim};
                uint64_t out_view_offsets[2] = {cur_offset, 0};
                Tensor out_view = out.view(out_view_shapes, out_view_offsets);
                
                PTOParam params_inplace[] = {
                    make_output_param(oi),
                    make_output_param(li_update),
                    make_output_param(mi_update),
                };
                pto2_submit_task(&rt->orchestrator, FUNC_AIV_HUB, PTO2_WORKER_VECTOR, params_inplace, 3);
                total_tasks++;
                
                for (uint64_t bn = 0; bn < bn_this_batch; bn++) {
                    uint64_t cur_block_idx = host_block_table[b_idx * block_num + bn];
                    uint64_t valid_len = std::min(block_size, cur_seq - bn * block_size);
                    
                    uint64_t kv_shapes[2] = {block_size, head_dim};
                    uint64_t kv_offsets[2] = {cur_block_idx * block_size, 0};
                    Tensor kj = key_cache.view(kv_shapes, kv_offsets);
                    Tensor vj = value_cache.view(kv_shapes, kv_offsets);
                    
                    uint64_t sij_shapes[2] = {q_tile, block_size};
                    Tensor sij = make_tensor(sij_shapes, 2, DataType::FLOAT32);
                    Tensor pij_f16 = make_tensor(sij_shapes, 2, data_type);
                    
                    PTOParam params_qk[] = {
                        make_input_param(qi),
                        make_input_param(kj),
                        make_output_param(sij),
                    };
                    pto2_submit_task(&rt->orchestrator, FUNC_QK_MATMUL, PTO2_WORKER_CUBE, params_qk, 3);
                    total_tasks++;
                    
                    uint64_t sij_valid_shapes[2] = {q_tile, valid_len};
                    uint64_t sij_valid_offsets[2] = {0, 0};
                    Tensor sij_valid = sij.view(sij_valid_shapes, sij_valid_offsets);
                    
                    Tensor li = make_tensor(li_shapes, 1, DataType::FLOAT32);
                    Tensor mi = make_tensor(mi_shapes, 1, DataType::FLOAT32);
                    
                    PTOParam params_sf[] = {
                        make_input_param(sij_valid),
                        make_scalar_param(float_to_u64(scale_value)),
                        make_output_param(pij_f16),
                        make_output_param(mi),
                        make_output_param(li),
                    };
                    pto2_submit_task(&rt->orchestrator, FUNC_SOFTMAX_PREPARE, PTO2_WORKER_VECTOR, params_sf, 5);
                    total_tasks++;
                    
                    uint64_t oi_tmp_shapes[2] = {q_tile, head_dim};
                    Tensor oi_tmp = make_tensor(oi_tmp_shapes, 2, DataType::FLOAT32);
                    
                    PTOParam params_pv[] = {
                        make_input_param(pij_f16),
                        make_input_param(vj),
                        make_output_param(oi_tmp),
                    };
                    pto2_submit_task(&rt->orchestrator, FUNC_PV_MATMUL, PTO2_WORKER_CUBE, params_pv, 3);
                    total_tasks++;
                    
                    uint64_t is_first = (bn == 0) ? 1 : 0;
                    uint64_t is_last = (bn == bn_this_batch - 1) ? 1 : 0;
                    
                    PTOParam params_up[] = {
                        make_input_param(mi),
                        make_input_param(li),
                        make_input_param(oi_tmp),
                        make_inout_param(mi_update),
                        make_inout_param(li_update),
                        make_inout_param(oi),
                        make_output_param(out_view),
                        make_scalar_param(is_first),
                        make_scalar_param(is_last),
                    };
                    pto2_submit_task(&rt->orchestrator, FUNC_ONLINE_UPDATE, PTO2_WORKER_VECTOR, params_up, 9);
                    total_tasks++;
                }
            }  // PTO2_SCOPE end
        }
    }
    
    pto2_orchestrator_done(&rt->orchestrator);
    
    auto t_end = std::chrono::high_resolution_clock::now();
    
    auto duration_total = std::chrono::duration_cast<std::chrono::microseconds>(t_end - t_start);
    auto duration_tensor_pool = std::chrono::duration_cast<std::chrono::microseconds>(t_tensor_pool - t_start);
    auto duration_orchestration = std::chrono::duration_cast<std::chrono::microseconds>(t_end - t_tensor_pool);
    
    printf("  Total tasks submitted: %d\n", total_tasks);
    printf("  Timing breakdown:\n");
    printf("    TensorPool init:     %lld us\n", (long long)duration_tensor_pool.count());
    printf("    Orchestration:        %lld us\n", (long long)duration_orchestration.count());
    printf("    Total:                %lld us (%.3f ms)\n", 
           (long long)duration_total.count(), duration_total.count() / 1000.0);
}

// ─────────────────────────────────────────────────────────────────────────────
// [3] Test function
// ─────────────────────────────────────────────────────────────────────────────

void test_paged_attention_basic() {
    TEST_BEGIN("test_paged_attention_basic");
    
    auto t_test_start = std::chrono::high_resolution_clock::now();
    
    PTO2Runtime* rt = make_small_runtime();
    CHECK(rt != nullptr);
    if (!rt) return;
    
    auto t_runtime_created = std::chrono::high_resolution_clock::now();
    
    // Prepare test data
    const uint64_t batch = 2;
    const uint64_t num_heads = 4;
    const int kv_head_num = 1;
    const uint64_t head_dim = 8;
    const uint64_t block_size = 4;
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
        auto t_graph_start = std::chrono::high_resolution_clock::now();
        
        // Build the task graph
        build_paged_attention_graph(rt, args, 10);
        
        auto t_graph_end = std::chrono::high_resolution_clock::now();
        auto t_sim_start = std::chrono::high_resolution_clock::now();
        
        // Verify tasks were submitted
        CHECK(rt->orchestrator.tasks_submitted > 0);
        printf("  Tasks submitted: %lld\n", (long long)rt->orchestrator.tasks_submitted);
        
        // Run simulation (skip AICore execution)
        int executed = sim_run_all(rt);
        printf("  Tasks executed: %d\n", executed);
        
        auto t_sim_end = std::chrono::high_resolution_clock::now();
        
        // Verify all tasks completed
        CHECK(executed == rt->orchestrator.tasks_submitted);
        CHECK(rt->scheduler.tasks_completed.load() == executed);
        CHECK(rt->scheduler.tasks_consumed.load() == executed);
        
        // Timing summary
        auto duration_graph = std::chrono::duration_cast<std::chrono::microseconds>(t_graph_end - t_graph_start);
        auto duration_sim = std::chrono::duration_cast<std::chrono::microseconds>(t_sim_end - t_sim_start);
        printf("  Test timing summary:\n");
        printf("    Graph building:     %lld us (%.3f ms)\n", 
               (long long)duration_graph.count(), duration_graph.count() / 1000.0);
        printf("    Task simulation:    %lld us (%.3f ms)\n", 
               (long long)duration_sim.count(), duration_sim.count() / 1000.0);
    }  // Tensors destructed here
    
    // Cleanup
    free(query_buf);
    free(key_cache_buf);
    free(value_cache_buf);
    free(out_buf);
    free(block_table);
    free(context_lens);
    
    pto2_runtime_destroy(rt);
    
    auto t_test_end = std::chrono::high_resolution_clock::now();
    auto duration_runtime = std::chrono::duration_cast<std::chrono::microseconds>(t_runtime_created - t_test_start);
    auto duration_total = std::chrono::duration_cast<std::chrono::microseconds>(t_test_end - t_test_start);
    
    printf("  Overall timing:\n");
    printf("    Runtime creation:     %lld us (%.3f ms)\n", 
           (long long)duration_runtime.count(), duration_runtime.count() / 1000.0);
    printf("    Total test time:      %lld us (%.3f ms)\n", 
           (long long)duration_total.count(), duration_total.count() / 1000.0);
    
    TEST_END();
}
