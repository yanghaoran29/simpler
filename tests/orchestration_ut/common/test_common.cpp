/**
 * test_common.cpp
 *
 * Common helper functions for orchestration unit tests
 */

#include "test_common.h"
#include "pto_runtime2.h"
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
PTO2Runtime* make_small_runtime() {
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
