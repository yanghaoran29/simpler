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
#if defined(PTO2_SIM_AICORE_UT)
#include "aicpu_sim_api.h"
#endif
#include <time.h>
#include <thread>
#include <vector>

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
        /*task_window_size=*/16384,
        /*heap_size=*/4ull * 1024 * 1024 * 1024,
        /*dep_list_size=*/1024);
}

#if PTO2_PROFILING
SchedProfilingData g_sched_prof_data;
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
    printf("  Simulation execution: [1 thread, %d tasks]\n", total);

    return total;
}

// Per-thread profiling for multi-threaded scheduler (merged into g_sched_prof_data after join)
#if PTO2_PROFILING
SchedProfilingData g_sched_prof_per_thread[PLATFORM_MAX_AICPU_THREADS];
#endif

/**
 * Single scheduler thread loop: drain all ready tasks (like sim_drain_one_pass) per round,
 * then yield only when no task was found. This matches single-threaded behavior so that
 * "Total tasks submitted" and "Scheduler Profiling (N tasks)" agree (all submitted tasks get completed).
 * my_prof may be nullptr when PTO2_PROFILING is off.
 */
void sim_drain_scheduler_thread(PTO2Runtime* rt, int thread_idx,
                                       SchedProfilingData* my_prof,
                                       int* out_executed, int max_iterations) {
    int executed = 0;
    int idle_rounds = 0;
    while (idle_rounds < max_iterations) {
        if (rt->scheduler.is_done())
            break;
        int got_any = 0;
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
                if (my_prof) my_prof->dispatch_cycle += t1 - t0;
#endif
#if PTO2_SCHED_PROFILING
                PTO2CompletionStats cs = rt->scheduler.on_task_complete(task_id, thread_idx);
#else
                PTO2CompletionStats cs = rt->scheduler.on_task_complete(task_id);
#endif
#if PTO2_PROFILING
                uint64_t t2 = get_sys_cnt_aicpu();
                if (my_prof) {
                    my_prof->complete_cycle += t2 - t1;
                    my_prof->tasks_dispatched[wt]++;
                    my_prof->fanout_edges_total += cs.fanout_edges;
                    if (cs.fanout_edges > my_prof->fanout_max_degree)
                        my_prof->fanout_max_degree = cs.fanout_edges;
                    my_prof->tasks_enqueued_by_completion += cs.tasks_enqueued;
                    my_prof->fanin_edges_total += cs.fanin_edges;
                    if (cs.fanin_edges > my_prof->fanin_max_degree)
                        my_prof->fanin_max_degree = cs.fanin_edges;
                }
#endif
                ++executed;
                got_any = 1;
            }
        }
        if (!got_any) {
            std::this_thread::yield();
            ++idle_rounds;
        }
    }
    *out_executed = executed;
}

/**
 * Run simulation with multiple scheduler threads (mirrors device: 3 scheduler threads).
 * Orchestrator work was done by the main thread in build_*_graph; this runs 3 threads
 * that drain the ready queues and call on_task_complete, matching resolve_and_dispatch_pto2.
 */
int sim_run_all_multi_thread(PTO2Runtime* rt, int num_sched_threads, int max_iterations_per_thread) {
#if PTO2_PROFILING
    for (int i = 0; i < PLATFORM_MAX_AICPU_THREADS; i++)
        g_sched_prof_per_thread[i] = {};
    g_sched_prof_data = {};
#endif

    std::vector<std::thread> threads;
    std::vector<int> executed_per_thread(static_cast<size_t>(num_sched_threads), 0);
    for (int i = 0; i < num_sched_threads; i++) {
        threads.emplace_back([rt, i, max_iterations_per_thread, &executed_per_thread]() {
#if PTO2_PROFILING
            sim_drain_scheduler_thread(rt, i, &g_sched_prof_per_thread[i],
                                       &executed_per_thread[static_cast<size_t>(i)],
                                       max_iterations_per_thread);
#else
            sim_drain_scheduler_thread(rt, i, nullptr,
                                       &executed_per_thread[static_cast<size_t>(i)],
                                       max_iterations_per_thread);
#endif
        });
    }
    for (int i = 0; i < num_sched_threads; i++) {
        threads[static_cast<size_t>(i)].join();
    }

    int total = 0;
    for (int i = 0; i < num_sched_threads; i++)
        total += executed_per_thread[static_cast<size_t>(i)];

#if PTO2_PROFILING
    for (int t = 0; t < num_sched_threads && t < PLATFORM_MAX_AICPU_THREADS; t++) {
        SchedProfilingData* p = &g_sched_prof_per_thread[t];
        for (int w = 0; w < PTO2_NUM_WORKER_TYPES; w++)
            g_sched_prof_data.tasks_dispatched[w] += p->tasks_dispatched[w];
        g_sched_prof_data.fanout_edges_total += p->fanout_edges_total;
        if (p->fanout_max_degree > g_sched_prof_data.fanout_max_degree)
            g_sched_prof_data.fanout_max_degree = p->fanout_max_degree;
        g_sched_prof_data.tasks_enqueued_by_completion += p->tasks_enqueued_by_completion;
        g_sched_prof_data.fanin_edges_total += p->fanin_edges_total;
        if (p->fanin_max_degree > g_sched_prof_data.fanin_max_degree)
            g_sched_prof_data.fanin_max_degree = p->fanin_max_degree;
        g_sched_prof_data.dispatch_cycle += p->dispatch_cycle;
        g_sched_prof_data.complete_cycle += p->complete_cycle;
    }
    g_sched_prof_data.rounds_total = 1;
    g_sched_prof_data.rounds_with_progress = (total > 0) ? 1 : 0;
#endif

    int32_t expected_tasks = 0;
    if (rt->sm_handle && rt->sm_handle->header)
        expected_tasks = rt->sm_handle->header->current_task_index.load(std::memory_order_acquire);

#if PTO2_PROFILING
    // Check actual completed tasks instead of just scheduler thread executions
    // This handles cases where tasks are completed during graph building (e.g., submit_and_complete_task)
    int64_t tasks_completed_count = rt->scheduler.tasks_completed.load(std::memory_order_relaxed);
    printf("  Simulation execution: [%d scheduler thread(s), %d tasks]\n", num_sched_threads, total);

    if (expected_tasks > 0 && tasks_completed_count != expected_tasks) {
        printf("  WARNING: scheduler completed %lld tasks but orchestrator submitted %d\n",
               (long long)tasks_completed_count, expected_tasks);
    } else if (total == 0 && tasks_completed_count == expected_tasks && expected_tasks > 0) {
        // All tasks were completed during graph building (not by scheduler threads)
        printf("  Note: All %d tasks were completed during graph building (immediate completion)\n", expected_tasks);
    }
#else
    printf("  Simulation execution: [%d scheduler thread(s), %d tasks]\n", num_sched_threads, total);
    if (expected_tasks > 0 && total != expected_tasks)
        printf("  WARNING: scheduler completed %d tasks but orchestrator submitted %d\n", total, expected_tasks);
#endif

    return total;
}

/**
 * Simulate resolve_and_dispatch_pto2 execution model (mirrors device_tests).
 *
 * This function mimics AicpuExecutor::resolve_and_dispatch_pto2 behavior:
 * - Multiple scheduler threads poll ready queues and complete tasks
 * - Uses a three-phase loop: completion detection, task dispatch, idle handling
 * - Tracks total_tasks dynamically (set after orchestration completes)
 * - Exits when all tasks complete or max iterations reached
 */
int sim_run_with_resolve_and_dispatch(PTO2Runtime* rt, int num_sched_threads, int max_iterations_per_thread) {
#if PTO2_PROFILING
    for (int i = 0; i < PLATFORM_MAX_AICPU_THREADS; i++) {
        g_sched_prof_per_thread[i] = {};
    }
    g_sched_prof_data = {};
#endif

    auto scheduler_thread_func = [&](int thread_idx) {
        int cur_thread_completed = 0;
        int idle_iterations = 0;
        const int MAX_IDLE_ITERATIONS = max_iterations_per_thread;
        bool made_progress = false;

#if PTO2_PROFILING
        SchedProfilingData* my_prof = &g_sched_prof_per_thread[thread_idx];
        uint64_t sched_scan_cycle = 0;
        uint64_t sched_complete_cycle = 0;
        uint64_t sched_dispatch_cycle = 0;
        uint64_t sched_loop_count = 0;
#endif

        while (true) {
#if PTO2_PROFILING
            sched_loop_count++;
#endif
            // Use scheduler's is_done() method to check termination
            if (rt->scheduler.is_done()) {
                break;
            }

            made_progress = false;

            // Phase 1: Completion detection - check all worker types for completed tasks
#if PTO2_PROFILING
            uint64_t t0_complete = get_sys_cnt_aicpu();
#endif
            // In simulation, we don't have actual cores to poll, so this phase is minimal
#if PTO2_PROFILING
            uint64_t t1_complete = get_sys_cnt_aicpu();
            sched_scan_cycle += t1_complete - t0_complete;
#endif

            // Phase 2: Task dispatch - get ready tasks and execute them
            for (int wt = 0; wt < PTO2_NUM_WORKER_TYPES; ++wt) {
                int32_t task_id;
                // Drain all ready tasks of this worker type (nested loop like sim_drain_scheduler_thread)
                while (true) {
#if PTO2_PROFILING
                    uint64_t t0_dispatch = get_sys_cnt_aicpu();
#endif
                    task_id = rt->scheduler.get_ready_task((PTO2WorkerType)wt);
                    if (task_id < 0) break;
                    rt->scheduler.mark_running(task_id);
#if PTO2_PROFILING
                    uint64_t t1_dispatch = get_sys_cnt_aicpu();
                    sched_dispatch_cycle += t1_dispatch - t0_dispatch;
#endif
#if PTO2_PROFILING
                    uint64_t t0_comp = get_sys_cnt_aicpu();
#endif
#if PTO2_SCHED_PROFILING
                    PTO2CompletionStats cs = rt->scheduler.on_task_complete(task_id, thread_idx);
#else
                    PTO2CompletionStats cs = rt->scheduler.on_task_complete(task_id);
#endif
#if PTO2_PROFILING
                    uint64_t t1_comp = get_sys_cnt_aicpu();
                    sched_complete_cycle += t1_comp - t0_comp;

                    my_prof->tasks_dispatched[wt]++;
                    my_prof->fanout_edges_total += cs.fanout_edges;
                    if (cs.fanout_edges > my_prof->fanout_max_degree)
                        my_prof->fanout_max_degree = cs.fanout_edges;
                    my_prof->tasks_enqueued_by_completion += cs.tasks_enqueued;
                    my_prof->fanin_edges_total += cs.fanin_edges;
                    if (cs.fanin_edges > my_prof->fanin_max_degree)
                        my_prof->fanin_max_degree = cs.fanin_edges;
#endif
                    cur_thread_completed++;
                    made_progress = true;
                    idle_iterations = 0;
                }
            }

            // Phase 3: Idle handling
            if (!made_progress) {
                idle_iterations++;
                if (idle_iterations > MAX_IDLE_ITERATIONS) {
                    break;
                }
                std::this_thread::yield();
            }
        }

#if PTO2_PROFILING
        my_prof->rounds_total = sched_loop_count;
        my_prof->rounds_with_progress = cur_thread_completed;
        my_prof->dispatch_cycle = sched_dispatch_cycle;
        my_prof->complete_cycle = sched_complete_cycle;
#endif

        return cur_thread_completed;
    };

    std::vector<std::thread> threads;
    std::vector<int> thread_results(num_sched_threads, 0);

    for (int i = 0; i < num_sched_threads; i++) {
        threads.emplace_back([&, i]() {
            thread_results[i] = scheduler_thread_func(i);
        });
    }

    for (auto& t : threads) {
        t.join();
    }

    int total_executed = 0;
    for (int result : thread_results) {
        total_executed += result;
    }

#if PTO2_PROFILING
    for (int i = 0; i < num_sched_threads; i++) {
        for (int wt = 0; wt < PTO2_NUM_WORKER_TYPES; ++wt) {
            g_sched_prof_data.tasks_dispatched[wt] += g_sched_prof_per_thread[i].tasks_dispatched[wt];
        }
        g_sched_prof_data.fanout_edges_total += g_sched_prof_per_thread[i].fanout_edges_total;
        if (g_sched_prof_per_thread[i].fanout_max_degree > g_sched_prof_data.fanout_max_degree)
            g_sched_prof_data.fanout_max_degree = g_sched_prof_per_thread[i].fanout_max_degree;
        g_sched_prof_data.tasks_enqueued_by_completion += g_sched_prof_per_thread[i].tasks_enqueued_by_completion;
        g_sched_prof_data.fanin_edges_total += g_sched_prof_per_thread[i].fanin_edges_total;
        if (g_sched_prof_per_thread[i].fanin_max_degree > g_sched_prof_data.fanin_max_degree)
            g_sched_prof_data.fanin_max_degree = g_sched_prof_per_thread[i].fanin_max_degree;
        g_sched_prof_data.rounds_total += g_sched_prof_per_thread[i].rounds_total;
        g_sched_prof_data.rounds_with_progress += g_sched_prof_per_thread[i].rounds_with_progress;
        g_sched_prof_data.dispatch_cycle += g_sched_prof_per_thread[i].dispatch_cycle;
        g_sched_prof_data.complete_cycle += g_sched_prof_per_thread[i].complete_cycle;
    }
#endif

    int32_t expected_tasks = 0;
    if (rt->sm_handle && rt->sm_handle->header)
        expected_tasks = rt->sm_handle->header->current_task_index.load(std::memory_order_acquire);

#if PTO2_PROFILING
    int64_t tasks_completed_count = rt->scheduler.tasks_completed.load(std::memory_order_relaxed);
    printf("  Simulation execution: [%d scheduler thread(s) (resolve_and_dispatch style), %d tasks]\n",
           num_sched_threads, total_executed);

    if (expected_tasks > 0 && tasks_completed_count != expected_tasks) {
        printf("  WARNING: scheduler completed %lld tasks but orchestrator submitted %d\n",
               (long long)tasks_completed_count, expected_tasks);
    } else if (total_executed == 0 && tasks_completed_count == expected_tasks && expected_tasks > 0) {
        printf("  Note: All %d tasks were completed during graph building (immediate completion)\n", expected_tasks);
    }
#else
    printf("  Simulation execution: [%d scheduler thread(s) (resolve_and_dispatch style), %d tasks]\n",
           num_sched_threads, total_executed);
    if (expected_tasks > 0 && total_executed != expected_tasks)
        printf("  WARNING: scheduler completed %d tasks but orchestrator submitted %d\n", total_executed, expected_tasks);
#endif

    return total_executed;
}

#if PTO2_PROFILING
#if PTO2_ORCH_PROFILING
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
#else
void print_orch_profiling() {}
#endif

void print_sched_profiling(PTO2Runtime* rt) {
#if defined(PTO2_SIM_AICORE_UT)
    // aicpu_sim_run_pto2 路径不更新 g_sched_prof_data，从上次 sim 运行结果回填以便 fanout/fanin 正确显示
    AicpuSimRunProf run_prof;
    aicpu_sim_get_run_prof(&run_prof);
    int64_t sim_sum = 0;
    for (int i = 0; i < AICPU_SIM_PROF_WORKER_TYPES && i < PTO2_NUM_WORKER_TYPES; i++) {
        g_sched_prof_data.tasks_dispatched[i] = run_prof.tasks_dispatched[i];
        sim_sum += run_prof.tasks_dispatched[i];
    }
    if (sim_sum > 0) {
        g_sched_prof_data.fanout_edges_total = run_prof.fanout_edges_total;
        g_sched_prof_data.fanout_max_degree = run_prof.fanout_max_degree;
        g_sched_prof_data.tasks_enqueued_by_completion = run_prof.tasks_enqueued_by_completion;
        g_sched_prof_data.fanin_edges_total = run_prof.fanin_edges_total;
        g_sched_prof_data.fanin_max_degree = run_prof.fanin_max_degree;
        g_sched_prof_data.complete_cycle = run_prof.complete_cycle;
        g_sched_prof_data.dispatch_cycle = run_prof.dispatch_cycle;
    }
#endif

    const char* wt_names[] = {"CUBE", "VECTOR", "AI_CPU", "ACCELERATOR"};
    int64_t total = 0;
    for (int i = 0; i < PTO2_NUM_WORKER_TYPES; i++)
        total += g_sched_prof_data.tasks_dispatched[i];

    printf("\n  === Scheduler Profiling (%lld tasks) ===\n", (long long)total);
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
    // Use the snapshot saved at the end of resolve_and_dispatch_pto2 to avoid
    // double-consuming the g_sched_* counters (which were already reset by
    // pto2_scheduler_get_profiling inside resolve_and_dispatch_pto2).
    int thread_idx = 0;
    PTO2SchedProfilingData sp = {};
    aicpu_sim_get_saved_sched_prof(thread_idx, &sp);
    
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
