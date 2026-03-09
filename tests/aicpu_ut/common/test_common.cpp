/**
 * test_common.cpp
 *
 * Common helper functions for orchestration unit tests
 */

#include "test_common.h"
#include "pto_runtime2.h"
#include "pto_orchestrator.h"
#include "pto_scheduler.h"
#include "common/platform_config.h"
#include "aicpu/device_time.h"
#include <time.h>
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
 * Monotonic clock in microseconds, for performance measurement
 */
uint64_t perf_now_us() {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (uint64_t)ts.tv_sec * 1000000ULL + (uint64_t)ts.tv_nsec / 1000ULL;
}

/**
 * Create a runtime for testing.
 *
 * task_window_size must exceed the total live tasks across all scopes simultaneously.
 * For production-scale batch paged attention (batch=64, block_num=512, IN_CORE_BATCH=16):
 *   - Unfilled batch slots use block_size * block_num as context_len → max_bn = block_num
 *   - num_chunks = ceil(64/16) = 4
 *   - Tasks per chunk = 1 (AIV_HUB) + block_num * 4 (QK/SF/PV/UPDATE)
 *   - Case2 worst case: 4 * (1 + 512 * 4) = 8196 tasks
 * Use 16384 to cover all test cases with margin.
 */
PTO2Runtime* make_runtime() {
    return pto2_runtime_create_custom(PTO2_MODE_SIMULATE,
        /*task_window_size=*/65536,
        /*heap_size=*/4ull * 1024 * 1024 * 1024, 
        /*dep_list_size=*/1024);
}

#if PTO2_PROFILING
static SchedProfilingData g_sched_prof_data;
static uint64_t g_orch_start_time = 0;
static uint64_t g_orch_end_time = 0;

void orch_timing_begin() {
    g_orch_start_time = get_sys_cnt_aicpu();
}

void orch_timing_end() {
    g_orch_end_time = get_sys_cnt_aicpu();
}
#endif

/**
 * Simulate task execution by draining ready queues
 */
int sim_drain_one_pass(PTO2Runtime* rt) {
    int executed = 0;
    for (int wt = 0; wt < PTO2_NUM_WORKER_TYPES; ++wt) {
        int32_t task_id;
        while (true) {
#if PTO2_PROFILING
            uint64_t t0 = get_sys_cnt_aicpu();
#endif
            task_id = rt->scheduler.get_ready_task((PTO2WorkerType)wt);
            if (task_id < 0) break;
            rt->scheduler.mark_running(task_id);
#if PTO2_PROFILING
            uint64_t t1 = get_sys_cnt_aicpu();
            g_sched_prof_data.dispatch_cycle += t1 - t0;
#endif
#if PTO2_SCHED_PROFILING
            // In simulation mode, use thread_idx=0 since we're single-threaded
            PTO2CompletionStats cs = rt->scheduler.on_task_complete(task_id, 0);
#else
            PTO2CompletionStats cs = rt->scheduler.on_task_complete(task_id);
#endif
#if PTO2_PROFILING
            uint64_t t2 = get_sys_cnt_aicpu();
            g_sched_prof_data.complete_cycle += t2 - t1;
            g_sched_prof_data.tasks_dispatched[wt]++;
            g_sched_prof_data.fanout_edges_total += cs.fanout_edges;
            if (cs.fanout_edges > g_sched_prof_data.fanout_max_degree)
                g_sched_prof_data.fanout_max_degree = cs.fanout_edges;
            g_sched_prof_data.tasks_enqueued_by_completion += cs.tasks_enqueued;
            g_sched_prof_data.fanin_edges_total += cs.fanin_edges;
            if (cs.fanin_edges > g_sched_prof_data.fanin_max_degree)
                g_sched_prof_data.fanin_max_degree = cs.fanin_edges;
#endif
            ++executed;
        }
    }
    return executed;
}

/**
 * Run simulation until all tasks complete
 */
int sim_run_all(PTO2Runtime* rt, int max_rounds) {
#if PTO2_PROFILING
    g_sched_prof_data = {};
#endif
    auto t_start = std::chrono::high_resolution_clock::now();

    int total = 0;
    for (int r = 0; r < max_rounds; ++r) {
        int n = sim_drain_one_pass(rt);
        total += n;
#if PTO2_PROFILING
        g_sched_prof_data.rounds_total++;
        if (n > 0) g_sched_prof_data.rounds_with_progress++;
#endif
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
    g_orch_start_time = g_orch_end_time = 0;
    printf("  === Orchestrator Profiling (%lld submits) ===\n", (long long)pd.submit_count);
    printf("    sync:       %8.3f us  (%4.1f%%)\n", cycles_to_us(pd.sync_cycle),     pd.sync_cycle     * 100.0 / orch_total);
    printf("    alloc:      %8.3f us  (%4.1f%%)\n", cycles_to_us(pd.alloc_cycle),    pd.alloc_cycle    * 100.0 / orch_total);
    printf("    params:     %8.3f us  (%4.1f%%)\n", cycles_to_us(pd.params_cycle),   pd.params_cycle   * 100.0 / orch_total);
    printf("    lookup:     %8.3f us  (%4.1f%%)\n", cycles_to_us(pd.lookup_cycle),   pd.lookup_cycle   * 100.0 / orch_total);
    printf("    heap:       %8.3f us  (%4.1f%%)\n", cycles_to_us(pd.heap_cycle),     pd.heap_cycle     * 100.0 / orch_total);
    printf("    insert:     %8.3f us  (%4.1f%%)\n", cycles_to_us(pd.insert_cycle),   pd.insert_cycle   * 100.0 / orch_total);
    printf("    fanin:      %8.3f us  (%4.1f%%)\n", cycles_to_us(pd.fanin_cycle),    pd.fanin_cycle    * 100.0 / orch_total);
    printf("    finalize:   %8.3f us  (%4.1f%%)\n", cycles_to_us(pd.finalize_cycle), pd.finalize_cycle * 100.0 / orch_total);
    printf("    scope:      %8.3f us  (%4.1f%%)\n", cycles_to_us(pd.scope_end_cycle),pd.scope_end_cycle* 100.0 / orch_total);
    printf("\n");
    printf("    total:      %8.3f us\n", cycles_to_us(orch_total));
}

void print_sched_profiling(PTO2Runtime* rt) {
    const char* wt_names[] = {"CUBE", "VECTOR", "AI_CPU", "ACCELERATOR"};
    int64_t total = 0;
    for (int i = 0; i < PTO2_NUM_WORKER_TYPES; i++)
        total += g_sched_prof_data.tasks_dispatched[i];

    printf("  === Scheduler Profiling (%lld tasks) ===\n", (long long)total);
    for (int i = 0; i < PTO2_NUM_WORKER_TYPES; i++) {
        int64_t n = g_sched_prof_data.tasks_dispatched[i];
        if (n == 0) continue;
        printf("    %-12s %6lld tasks  (%4.1f%%)\n",
               wt_names[i], (long long)n, total > 0 ? n * 100.0 / total : 0.0);
    }
    printf("    fanout:          %lld edges, max_degree=%d, enqueued=%lld\n",
           (long long)g_sched_prof_data.fanout_edges_total,
           g_sched_prof_data.fanout_max_degree,
           (long long)g_sched_prof_data.tasks_enqueued_by_completion);
    printf("    fanin:           %lld edges, max_degree=%d\n",
           (long long)g_sched_prof_data.fanin_edges_total,
           g_sched_prof_data.fanin_max_degree);
    printf("    sim_rounds:      %lld total, %lld with_progress\n",
           (long long)g_sched_prof_data.rounds_total,
           (long long)g_sched_prof_data.rounds_with_progress);
    printf("    tasks_completed: %lld\n",
           (long long)rt->scheduler.tasks_completed.load(std::memory_order_relaxed));
    printf("    tasks_consumed:  %lld\n",
           (long long)rt->scheduler.tasks_consumed.load(std::memory_order_relaxed));

#if PTO2_SCHED_PROFILING
    // Two-level scheduler profiling output (matching main branch format)
    // In simulation mode, we use thread_idx=0 since we're single-threaded
    int thread_idx = 0;
    PTO2SchedProfilingData sp = pto2_scheduler_get_profiling(thread_idx);
    
    uint64_t sched_total = g_sched_prof_data.complete_cycle + g_sched_prof_data.dispatch_cycle;
    if (sched_total == 0) sched_total = 1;
    
    uint64_t otc_total = sp.lock_cycle + sp.fanout_cycle + sp.fanin_cycle + sp.self_consumed_cycle;
    uint64_t complete_poll = (g_sched_prof_data.complete_cycle > otc_total)
        ? (g_sched_prof_data.complete_cycle - otc_total) : 0;
    uint64_t dispatch_poll = (g_sched_prof_data.dispatch_cycle > 0)
        ? g_sched_prof_data.dispatch_cycle : 0;  // In simulation, dispatch is just get_ready_task
    
    printf("\n  === Scheduler Phase Breakdown: total=%.3f us ===\n",
           cycles_to_us(sched_total));
    
    // Level 1: complete
    if (g_sched_prof_data.complete_cycle > 0) {
        printf("    complete       : %.3f us (%.1f%%)\n",
               cycles_to_us(g_sched_prof_data.complete_cycle),
               g_sched_prof_data.complete_cycle * 100.0 / sched_total);
        
        // Level 2: complete sub-phases
        uint64_t c_parent = g_sched_prof_data.complete_cycle > 0 ? g_sched_prof_data.complete_cycle : 1;
        if (complete_poll > 0) {
            printf("      poll         : %.3f us (%.1f%%)\n",
                   cycles_to_us(complete_poll), complete_poll * 100.0 / c_parent);
        }
        if (sp.lock_cycle > 0) {
            printf("      otc_lock     : %.3f us (%.1f%%)  work=%.3f us wait=%.3f us  atomics=%llu\n",
                   cycles_to_us(sp.lock_cycle), sp.lock_cycle * 100.0 / c_parent,
                   cycles_to_us(sp.lock_cycle - sp.lock_wait_cycle),
                   cycles_to_us(sp.lock_wait_cycle),
                   (unsigned long long)sp.lock_atomic_count);
        }
        if (sp.fanout_cycle > 0) {
            printf("      otc_fanout   : %.3f us (%.1f%%)  work=%.3f us wait=%.3f us  atomics=%llu\n",
                   cycles_to_us(sp.fanout_cycle), sp.fanout_cycle * 100.0 / c_parent,
                   cycles_to_us(sp.fanout_cycle - sp.push_wait_cycle),
                   cycles_to_us(sp.push_wait_cycle),
                   (unsigned long long)sp.fanout_atomic_count);
        }
        if (sp.fanin_cycle > 0) {
            printf("      otc_fanin    : %.3f us (%.1f%%)  atomics=%llu\n",
                   cycles_to_us(sp.fanin_cycle), sp.fanin_cycle * 100.0 / c_parent,
                   (unsigned long long)sp.fanin_atomic_count);
        }
        if (sp.self_consumed_cycle > 0) {
            printf("      otc_self     : %.3f us (%.1f%%)  atomics=%llu\n",
                   cycles_to_us(sp.self_consumed_cycle), sp.self_consumed_cycle * 100.0 / c_parent,
                   (unsigned long long)sp.self_atomic_count);
        }
    }
    
    // Level 1: dispatch
    if (g_sched_prof_data.dispatch_cycle > 0) {
        printf("    dispatch       : %.3f us (%.1f%%)\n",
               cycles_to_us(g_sched_prof_data.dispatch_cycle),
               g_sched_prof_data.dispatch_cycle * 100.0 / sched_total);
        
        // Level 2: dispatch sub-phases (simplified for simulation)
        uint64_t d_parent = g_sched_prof_data.dispatch_cycle > 0 ? g_sched_prof_data.dispatch_cycle : 1;
        if (dispatch_poll > 0) {
            printf("      poll         : %.3f us (%.1f%%)\n",
                   cycles_to_us(dispatch_poll), dispatch_poll * 100.0 / d_parent);
        }
        // In simulation, we don't have separate pop/setup phases, so we show combined dispatch
    }
    
    // Average per completion
    if (sp.complete_count > 0 && g_sched_prof_data.complete_cycle > 0) {
        printf("    avg/complete   : %.3f us\n",
               cycles_to_us(g_sched_prof_data.complete_cycle) / sp.complete_count);
    }
#endif
}
#else
void print_orch_profiling() {}
void print_sched_profiling(PTO2Runtime* rt) { (void)rt; }
#endif
