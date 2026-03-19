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
#include "sim_aicore.h"
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
 * Use 262144 to cover all test cases with margin.
 */
PTO2Runtime* make_runtime() {
    return pto2_runtime_create_custom(PTO2_MODE_SIMULATE,
        /*task_window_size=*/262144,
        /*heap_size=*/4ull * 1024 * 1024 * 1024);
}

#if PTO2_PROFILING
SchedProfilingData g_sched_prof_data;
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

#if PTO2_PROFILING
void print_sched_profiling(PTO2Runtime* rt) {
#if PTO2_SCHED_PROFILING
#if defined(PTO2_SIM_AICORE_UT)
    pto2_sim_get_accumulated_cycles(&g_sched_prof_data.complete_cycle, &g_sched_prof_data.dispatch_cycle);
#endif
    pto2_print_sim_sched_summary(
        &g_sched_prof_data,
        (int64_t)rt->scheduler.tasks_completed.load(std::memory_order_relaxed),
        (int64_t)rt->scheduler.tasks_consumed.load(std::memory_order_relaxed));
#else
    (void)rt;
#endif
}

#if PTO2_SCHED_PROFILING
void pto2_print_sim_sched_summary(SchedProfilingData* data, int64_t tasks_completed, int64_t tasks_consumed) {
    const char* wt_names[] = {"CUBE", "VECTOR", "AI_CPU", "ACCELERATOR"};
    int64_t total = 0;
    for (int i = 0; i < PTO2_NUM_WORKER_TYPES; i++)
        total += data->tasks_dispatched[i];

    printf("\n  === Scheduler Profiling (%lld tasks) ===\n", (long long)total);
    for (int i = 0; i < PTO2_NUM_WORKER_TYPES; i++) {
        int64_t n = data->tasks_dispatched[i];
        if (n == 0) continue;
        printf("    %-12s %6lld tasks  (%4.1f%%)\n",
               wt_names[i], (long long)n, total > 0 ? n * 100.0 / total : 0.0);
    }
    printf("    fanout:          %lld edges, max_degree=%d, enqueued=%lld\n",
           (long long)data->fanout_edges_total, data->fanout_max_degree,
           (long long)data->tasks_enqueued_by_completion);
    printf("    fanin:           %lld edges, max_degree=%d\n",
           (long long)data->fanin_edges_total, data->fanin_max_degree);
    printf("    sim_rounds:      %lld total, %lld with_progress\n",
           (long long)data->rounds_total, (long long)data->rounds_with_progress);
    printf("    tasks_completed: %lld\n", (long long)tasks_completed);
    printf("    tasks_consumed:  %lld\n", (long long)tasks_consumed);

#if defined(PTO2_SIM_AICORE_UT)
    PTO2SchedProfilingData sp = {};
    aicpu_sim_get_saved_sched_prof(0, &sp);

    uint64_t sched_total = data->complete_cycle + data->dispatch_cycle;
    if (sched_total == 0) sched_total = 1;
    uint64_t otc_total = sp.lock_cycle + sp.fanout_cycle + sp.fanin_cycle + sp.self_consumed_cycle;
    uint64_t complete_poll = (data->complete_cycle > otc_total) ? (data->complete_cycle - otc_total) : 0;

    printf("\n  === Scheduler Phase Breakdown: total=%.3fus ===\n", cycles_to_us(sched_total));
    if (data->complete_cycle > 0) {
        uint64_t c_parent = data->complete_cycle;
        printf("    complete       : %.3fus (%.1f%%)\n",
               cycles_to_us(data->complete_cycle), data->complete_cycle * 100.0 / sched_total);
        if (complete_poll > 0)
            printf("      poll         : %.3fus (%.1f%%)\n",
                   cycles_to_us(complete_poll), complete_poll * 100.0 / c_parent);
        if (sp.lock_cycle > 0)
            printf("      otc_lock     : %.3fus (%.1f%%)  work=%.3fus wait=%.3fus  atomics=%llu\n",
                   cycles_to_us(sp.lock_cycle), sp.lock_cycle * 100.0 / c_parent,
                   cycles_to_us(sp.lock_cycle - sp.lock_wait_cycle),
                   cycles_to_us(sp.lock_wait_cycle),
                   (unsigned long long)sp.lock_atomic_count);
        if (sp.fanout_cycle > 0)
            printf("      otc_fanout   : %.3fus (%.1f%%)  work=%.3fus wait=%.3fus  atomics=%llu\n",
                   cycles_to_us(sp.fanout_cycle), sp.fanout_cycle * 100.0 / c_parent,
                   cycles_to_us(sp.fanout_cycle - sp.push_wait_cycle),
                   cycles_to_us(sp.push_wait_cycle),
                   (unsigned long long)sp.fanout_atomic_count);
        if (sp.fanin_cycle > 0)
            printf("      otc_fanin    : %.3fus (%.1f%%)  atomics=%llu\n",
                   cycles_to_us(sp.fanin_cycle), sp.fanin_cycle * 100.0 / c_parent,
                   (unsigned long long)sp.fanin_atomic_count);
        if (sp.self_consumed_cycle > 0)
            printf("      otc_self     : %.3fus (%.1f%%)  atomics=%llu\n",
                   cycles_to_us(sp.self_consumed_cycle), sp.self_consumed_cycle * 100.0 / c_parent,
                   (unsigned long long)sp.self_atomic_count);
    }
    if (data->dispatch_cycle > 0)
        printf("    dispatch       : %.3fus (%.1f%%)\n",
               cycles_to_us(data->dispatch_cycle), data->dispatch_cycle * 100.0 / sched_total);
    if (sp.complete_count > 0 && data->complete_cycle > 0)
        printf("    avg/complete   : %.3fus\n",
               cycles_to_us(data->complete_cycle) / sp.complete_count);
#endif  // PTO2_SIM_AICORE_UT
}
#endif  // PTO2_SCHED_PROFILING

/**
 * Scheduler invariant checks (P1=FAIL, P2=WARN).
 * Must be called after print_sched_profiling() so g_sched_prof_data is populated.
 * Skipped when AICPU_UT_NO_CHECK=1.
 */
void run_sched_checks(PTO2Runtime* rt, int num_sched) {
    if (getenv("AICPU_UT_NO_CHECK")) return;

    int32_t submitted = 0;
    if (rt->sm_handle && rt->sm_handle->header)
        for (int ri = 0; ri < PTO2_MAX_RING_DEPTH; ri++)
            submitted += rt->sm_handle->header->rings[ri].fc.current_task_index.load(std::memory_order_acquire);

    // P1: total dispatched == submitted
#if defined(PTO2_SIM_AICORE_UT)
    pto2_sim_get_dispatch_counts(g_sched_prof_data.tasks_dispatched, PTO2_NUM_WORKER_TYPES);
#endif
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


    (void)num_sched;
    // P2 unavailable: PTO2_SIM_AICORE_UT path has no per-thread profiling
}
#else
void print_sched_profiling(PTO2Runtime* rt) { (void)rt; }
#endif
