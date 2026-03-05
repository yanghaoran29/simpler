/**
 * test_common.cpp
 *
 * Common helper functions for orchestration unit tests
 */

#include "test_common.h"
#include "pto_runtime2.h"
#include "common/platform_config.h"
#include <chrono>

/**
 * Helper to encode float as uint64_t for scalar params
 */
uint64_t float_to_u64(float f) {
    union {
        float f32;
        uint64_t u64;
    } conv;
    conv.u64 = 0;  // Clear upper bits
    conv.f32 = f;
    return conv.u64;
}

/**
 * Create a small runtime for testing
 */
PTO2Runtime* make_runtime() {
    return pto2_runtime_create_custom(PTO2_MODE_SIMULATE,
        /*task_window_size=*/512,        // Larger window for paged attention
        /*heap_size=*/32 * 1024 * 1024,  // 32 MB for intermediate buffers
        /*dep_list_size=*/1024);
}

/**
 * Simulate task execution by draining ready queues
 */
int sim_drain_one_pass(PTO2Runtime* rt) {
    int executed = 0;
    for (int wt = 0; wt < PTO2_NUM_WORKER_TYPES; ++wt) {
        int32_t task_id;
        while ((task_id = rt->scheduler.get_ready_task((PTO2WorkerType)wt)) >= 0) {
            rt->scheduler.mark_running(task_id);
            // Skip actual kernel execution, just mark as complete
            rt->scheduler.on_task_complete(task_id);
            ++executed;
        }
    }
    return executed;
}

/**
 * Run simulation until all tasks complete
 */
int sim_run_all(PTO2Runtime* rt, int max_rounds) {
    auto t_start = std::chrono::high_resolution_clock::now();

    int total = 0;
    for (int r = 0; r < max_rounds; ++r) {
        int n = sim_drain_one_pass(rt);
        total += n;
        if (n == 0) break;
    }

    auto t_end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(t_end - t_start);

    printf("  Simulation execution:  %lld us (%.3f ms)\n",
           (long long)duration.count(), duration.count() / 1000.0);

    return total;
}

#if PTO2_PROFILING
void print_orch_profiling() {
    PTO2OrchProfilingData pd = pto2_orchestrator_get_profiling();
    uint64_t orch_total = pd.sync_cycle + pd.alloc_cycle + pd.params_cycle
                        + pd.lookup_cycle + pd.heap_cycle + pd.insert_cycle
                        + pd.fanin_cycle + pd.finalize_cycle + pd.scope_end_cycle;
    if (orch_total == 0) orch_total = 1;
    uint64_t elapsed = pd.end_time - pd.start_time;
    printf("  === Orchestrator Profiling (%lld submits) ===\n", (long long)pd.submit_count);
    printf("    start_time: %8.3f us\n", cycles_to_us(pd.start_time));
    printf("    end_time:   %8.3f us\n", cycles_to_us(pd.end_time));
    printf("    sync:       %8.3f us  (%4.1f%%)\n", cycles_to_us(pd.sync_cycle),     pd.sync_cycle     * 100.0 / orch_total);
    printf("    alloc:      %8.3f us  (%4.1f%%)\n", cycles_to_us(pd.alloc_cycle),    pd.alloc_cycle    * 100.0 / orch_total);
    printf("    params:     %8.3f us  (%4.1f%%)\n", cycles_to_us(pd.params_cycle),   pd.params_cycle   * 100.0 / orch_total);
    printf("    lookup:     %8.3f us  (%4.1f%%)\n", cycles_to_us(pd.lookup_cycle),   pd.lookup_cycle   * 100.0 / orch_total);
    printf("    heap:       %8.3f us  (%4.1f%%)\n", cycles_to_us(pd.heap_cycle),     pd.heap_cycle     * 100.0 / orch_total);
    printf("    insert:     %8.3f us  (%4.1f%%)\n", cycles_to_us(pd.insert_cycle),   pd.insert_cycle   * 100.0 / orch_total);
    printf("    fanin:      %8.3f us  (%4.1f%%)\n", cycles_to_us(pd.fanin_cycle),    pd.fanin_cycle    * 100.0 / orch_total);
    printf("    finalize:   %8.3f us  (%4.1f%%)\n", cycles_to_us(pd.finalize_cycle), pd.finalize_cycle * 100.0 / orch_total);
    printf("    scope:      %8.3f us  (%4.1f%%)\n", cycles_to_us(pd.scope_end_cycle),pd.scope_end_cycle* 100.0 / orch_total);
    printf("    others:     %8.3f us\n", cycles_to_us(elapsed) - cycles_to_us(orch_total));
    printf("    total:      %8.3f us\n", cycles_to_us(orch_total));
}
#else
void print_orch_profiling() {}
#endif
