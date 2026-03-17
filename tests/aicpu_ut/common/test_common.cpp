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
#include "sim_aicore.h"
#endif
#include <time.h>
#include <thread>
#include <vector>
#include <cstdlib>

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
        /*heap_size=*/4ull * 1024 * 1024 * 1024);
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
    for (int rs = 0; rs < PTO2_NUM_RESOURCE_SHAPES; ++rs) {
        while (true) {
#if PTO2_PROFILING
            uint64_t t0 = get_sys_cnt_aicpu();
#endif
            PTO2TaskSlotState* slot_state = rt->scheduler.get_ready_task(static_cast<PTO2ResourceShape>(rs));
            if (!slot_state) break;
#if PTO2_PROFILING
            uint64_t t1 = get_sys_cnt_aicpu();
            g_sched_prof_data.dispatch_cycle += t1 - t0;
#endif
            uint8_t amask = slot_state->active_mask;
            if (amask & PTO2_SUBTASK_MASK_AIC)  rt->scheduler.on_subtask_complete(*slot_state, PTO2SubtaskSlot::AIC);
            if (amask & PTO2_SUBTASK_MASK_AIV0) rt->scheduler.on_subtask_complete(*slot_state, PTO2SubtaskSlot::AIV0);
            if (amask & PTO2_SUBTASK_MASK_AIV1) rt->scheduler.on_subtask_complete(*slot_state, PTO2SubtaskSlot::AIV1);
#if PTO2_SCHED_PROFILING
            // In simulation mode, use thread_idx=0 since we're single-threaded
            PTO2CompletionStats cs = rt->scheduler.on_mixed_task_complete(*slot_state, 0, nullptr);
#elif PTO2_PROFILING
            rt->scheduler.on_mixed_task_complete(*slot_state, nullptr);
            PTO2CompletionStats cs = {0, 0, 0, false};
#else
            rt->scheduler.on_mixed_task_complete(*slot_state, nullptr);
            PTO2CompletionStats cs = {0, 0, 0, false};
            (void)cs;
#endif
            // Mirror device: after completion, release fanin producers so they can become CONSUMED.
#if PTO2_SCHED_PROFILING
            int32_t fe = rt->scheduler.on_task_release(*slot_state, 0);
#else
            int32_t fe = rt->scheduler.on_task_release(*slot_state);
#endif
#if PTO2_PROFILING
            uint64_t t2 = get_sys_cnt_aicpu();
            g_sched_prof_data.complete_cycle += t2 - t1;
            if (amask & PTO2_SUBTASK_MASK_AIC)  g_sched_prof_data.tasks_dispatched[0]++;
            if (amask & PTO2_SUBTASK_MASK_AIV0) g_sched_prof_data.tasks_dispatched[1]++;
            if (amask & PTO2_SUBTASK_MASK_AIV1) g_sched_prof_data.tasks_dispatched[1]++;
            g_sched_prof_data.fanout_edges_total += cs.fanout_edges;
            if (cs.fanout_edges > g_sched_prof_data.fanout_max_degree)
                g_sched_prof_data.fanout_max_degree = cs.fanout_edges;
            g_sched_prof_data.tasks_enqueued_by_completion += cs.tasks_enqueued;
            g_sched_prof_data.fanin_edges_total += cs.fanin_edges;
            g_sched_prof_data.fanin_edges_total += fe;
            if (fe > g_sched_prof_data.fanin_max_degree)
                g_sched_prof_data.fanin_max_degree = fe;
#else
            (void)fe;
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
#if PTO2_PROFILING
    printf("  Simulation execution: [1 thread, %d tasks]\n", total);
#endif
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
 * my_prof may be nullptr when PTO2_PROFILING is off. When PTO2_PROFILING is on, it is SchedProfilingData*.
 */
void sim_drain_scheduler_thread(PTO2Runtime* rt, int thread_idx,
                                       void* my_prof,
                                       int* out_executed, int max_iterations) {
#if PTO2_PROFILING
    SchedProfilingData* prof = static_cast<SchedProfilingData*>(my_prof);
#endif
    int executed = 0;
    int idle_rounds = 0;
    while (idle_rounds < max_iterations) {
        int got_any = 0;
        for (int rs = 0; rs < PTO2_NUM_RESOURCE_SHAPES; ++rs) {
            while (true) {
#if PTO2_PROFILING
                uint64_t t0 = get_sys_cnt_aicpu();
#endif
                PTO2TaskSlotState* slot_state = rt->scheduler.get_ready_task(static_cast<PTO2ResourceShape>(rs));
                if (!slot_state) break;
#if PTO2_PROFILING
                uint64_t t1 = get_sys_cnt_aicpu();
                if (prof) prof->dispatch_cycle += t1 - t0;
#endif
                uint8_t amask = slot_state->active_mask;
                if (amask & PTO2_SUBTASK_MASK_AIC)  rt->scheduler.on_subtask_complete(*slot_state, PTO2SubtaskSlot::AIC);
                if (amask & PTO2_SUBTASK_MASK_AIV0) rt->scheduler.on_subtask_complete(*slot_state, PTO2SubtaskSlot::AIV0);
                if (amask & PTO2_SUBTASK_MASK_AIV1) rt->scheduler.on_subtask_complete(*slot_state, PTO2SubtaskSlot::AIV1);
#if PTO2_SCHED_PROFILING
                PTO2CompletionStats cs = rt->scheduler.on_mixed_task_complete(*slot_state, thread_idx, nullptr);
#elif PTO2_PROFILING
                rt->scheduler.on_mixed_task_complete(*slot_state, nullptr);
                PTO2CompletionStats cs = {0, 0, 0, false};
#else
                rt->scheduler.on_mixed_task_complete(*slot_state, nullptr);
                PTO2CompletionStats cs = {0, 0, 0, false};
                (void)cs;
#endif
                // Mirror device: after completion, release fanin producers so they can become CONSUMED.
#if PTO2_SCHED_PROFILING
                int32_t fe = rt->scheduler.on_task_release(*slot_state, thread_idx);
#else
                int32_t fe = rt->scheduler.on_task_release(*slot_state);
#endif
#if PTO2_PROFILING
                uint64_t t2 = get_sys_cnt_aicpu();
                if (prof) {
                    prof->complete_cycle += t2 - t1;
                    if (amask & PTO2_SUBTASK_MASK_AIC)  prof->tasks_dispatched[0]++;
                    if (amask & PTO2_SUBTASK_MASK_AIV0) prof->tasks_dispatched[1]++;
                    if (amask & PTO2_SUBTASK_MASK_AIV1) prof->tasks_dispatched[1]++;
                    prof->fanout_edges_total += cs.fanout_edges;
                    if (cs.fanout_edges > prof->fanout_max_degree)
                        prof->fanout_max_degree = cs.fanout_edges;
                    prof->tasks_enqueued_by_completion += cs.tasks_enqueued;
                    prof->fanin_edges_total += cs.fanin_edges + fe;
                    if (fe > prof->fanin_max_degree)
                        prof->fanin_max_degree = fe;
                }
#else
                (void)fe;
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

#if PTO2_PROFILING && PTO2_SCHED_PROFILING
    // Check actual completed tasks instead of just scheduler thread executions
    int64_t tasks_completed_count = rt->scheduler.tasks_completed.load(std::memory_order_relaxed);
    printf("  Simulation execution: [%d scheduler thread(s), %d tasks]\n", num_sched_threads, total);
    if (expected_tasks > 0 && tasks_completed_count != expected_tasks) {
        printf("  WARNING: scheduler completed %lld tasks but orchestrator submitted %d\n",
               (long long)tasks_completed_count, expected_tasks);
    } else if (total == 0 && tasks_completed_count == expected_tasks && expected_tasks > 0) {
        printf("  Note: All %d tasks were completed during graph building (immediate completion)\n", expected_tasks);
    }
#else
    (void)total;
    (void)expected_tasks;
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
            for (int rs = 0; rs < PTO2_NUM_RESOURCE_SHAPES; ++rs) {
                // Drain all ready tasks of this resource shape (nested loop like sim_drain_scheduler_thread)
                while (true) {
#if PTO2_PROFILING
                    uint64_t t0_dispatch = get_sys_cnt_aicpu();
#endif
                    PTO2TaskSlotState* slot_state = rt->scheduler.get_ready_task(static_cast<PTO2ResourceShape>(rs));
                    if (!slot_state) break;
#if PTO2_PROFILING
                    uint64_t t1_dispatch = get_sys_cnt_aicpu();
                    sched_dispatch_cycle += t1_dispatch - t0_dispatch;
#endif
#if PTO2_PROFILING
                    uint64_t t0_comp = get_sys_cnt_aicpu();
#endif
                    uint8_t amask = slot_state->active_mask;
                    if (amask & PTO2_SUBTASK_MASK_AIC)  rt->scheduler.on_subtask_complete(*slot_state, PTO2SubtaskSlot::AIC);
                    if (amask & PTO2_SUBTASK_MASK_AIV0) rt->scheduler.on_subtask_complete(*slot_state, PTO2SubtaskSlot::AIV0);
                    if (amask & PTO2_SUBTASK_MASK_AIV1) rt->scheduler.on_subtask_complete(*slot_state, PTO2SubtaskSlot::AIV1);
#if PTO2_SCHED_PROFILING
                    PTO2CompletionStats cs = rt->scheduler.on_mixed_task_complete(*slot_state, thread_idx, nullptr);
#elif PTO2_PROFILING
                    rt->scheduler.on_mixed_task_complete(*slot_state, nullptr);
                    PTO2CompletionStats cs = {0, 0, 0, false};
#else
                    rt->scheduler.on_mixed_task_complete(*slot_state, nullptr);
                    PTO2CompletionStats cs = {0, 0, 0, false};
                    (void)cs;
#endif
                    // Mirror device: after completion, release fanin producers so they can become CONSUMED.
#if PTO2_SCHED_PROFILING
                    int32_t fe = rt->scheduler.on_task_release(*slot_state, thread_idx);
#else
                    int32_t fe = rt->scheduler.on_task_release(*slot_state);
#endif
#if PTO2_PROFILING
                    uint64_t t1_comp = get_sys_cnt_aicpu();
                    sched_complete_cycle += t1_comp - t0_comp;

                    if (amask & PTO2_SUBTASK_MASK_AIC)  my_prof->tasks_dispatched[0]++;
                    if (amask & PTO2_SUBTASK_MASK_AIV0) my_prof->tasks_dispatched[1]++;
                    if (amask & PTO2_SUBTASK_MASK_AIV1) my_prof->tasks_dispatched[1]++;
                    my_prof->fanout_edges_total += cs.fanout_edges;
                    if (cs.fanout_edges > my_prof->fanout_max_degree)
                        my_prof->fanout_max_degree = cs.fanout_edges;
                    my_prof->tasks_enqueued_by_completion += cs.tasks_enqueued;
                    my_prof->fanin_edges_total += cs.fanin_edges + fe;
                    if (fe > my_prof->fanin_max_degree)
                        my_prof->fanin_max_degree = fe;
#else
                    (void)fe;
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

#if PTO2_PROFILING && PTO2_SCHED_PROFILING
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
    (void)total_executed;
    (void)expected_tasks;
#endif

    return total_executed;
}

#if PTO2_PROFILING
#if PTO2_ORCH_PROFILING
// Used by perf tests (run_tests.sh): prints orchestration profiling table
// (sync_tensormap, task_ring_alloc, param_copy, lookup+dep, heap_alloc, tensormap_ins, fanin+ready, finalize+SM, scope_end, avg/task).
void print_orch_profiling() {
    pto2_print_orch_profiling();
}
#else
// When ORCH_PROFILING=OFF (e.g. --profiling 1), still print orchestrator run time from orch_timing_begin/end.
void print_orch_profiling() {
    if (g_orch_end_time > g_orch_start_time) {
        uint64_t cycles = g_orch_end_time - g_orch_start_time;
        printf("  Orchestrator run time: %.3fus\n", cycles_to_us(cycles));
    }
}
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
        g_sched_prof_data.rounds_total = run_prof.rounds_total;
        g_sched_prof_data.rounds_with_progress = run_prof.rounds_with_progress;
        g_sched_prof_data.complete_cycle = run_prof.complete_cycle;
        g_sched_prof_data.dispatch_cycle = run_prof.dispatch_cycle;
    }
#endif

#if PTO2_SCHED_PROFILING
    pto2_print_sim_sched_summary(
        &g_sched_prof_data,
        (int64_t)rt->scheduler.tasks_completed.load(std::memory_order_relaxed),
        (int64_t)rt->scheduler.tasks_consumed.load(std::memory_order_relaxed));
#endif

#if PTO2_SCHED_PROFILING && !defined(PTO2_SIM_AICORE_UT)
    pto2_print_sched_profiling(0);
#endif
}

/**
 * Scheduler invariant checks (P1=FAIL, P2=WARN).
 * Must be called after print_sched_profiling() so g_sched_prof_data is populated.
 * Skipped when AICPU_UT_NO_CHECK=1.
 */
void run_sched_checks(PTO2Runtime* rt, int num_sched) {
    if (getenv("AICPU_UT_NO_CHECK")) return;

    int32_t submitted = 0;
    if (rt->sm_handle && rt->sm_handle->header)
        submitted = rt->sm_handle->header->current_task_index.load(std::memory_order_acquire);

    // P1: total dispatched == submitted
    int64_t total_dispatched = 0;
    for (int wt = 0; wt < PTO2_NUM_WORKER_TYPES; wt++)
        total_dispatched += g_sched_prof_data.tasks_dispatched[wt];
    if (total_dispatched != (int64_t)submitted) {
        printf("  FAIL (P1): total_dispatched (%lld) != submitted (%d)\n",
               (long long)total_dispatched, submitted);
        g_fail++;
    } else {
        g_pass++;
    }


#if !defined(PTO2_SIM_AICORE_UT)
    // P2: each scheduler thread dispatched > 0 tasks (warning only, no g_fail increment)
    if (submitted > 0) {
        for (int i = 0; i < num_sched && i < PLATFORM_MAX_AICPU_THREADS; i++) {
            int64_t thread_dispatched = 0;
            for (int wt = 0; wt < PTO2_NUM_WORKER_TYPES; wt++)
                thread_dispatched += g_sched_prof_per_thread[i].tasks_dispatched[wt];
            if (thread_dispatched == 0)
                printf("  WARN (P2): scheduler thread %d dispatched 0 tasks (possible starvation)\n", i);
        }
    }
#else
    (void)num_sched;
    // P2 unavailable: PTO2_SIM_AICORE_UT path has no per-thread profiling
#endif
}
#else
void print_orch_profiling() {}
void print_sched_profiling(PTO2Runtime* rt) { (void)rt; }
#endif
