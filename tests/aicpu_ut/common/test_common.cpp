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
 * Use 65536 to cover all test cases with margin.
 */
PTO2Runtime* make_runtime() {
    return pto2_runtime_create_custom(PTO2_MODE_SIMULATE,
        /*task_window_size=*/65536,
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

#if PTO2_PROFILING
void print_orch_profiling() {
    if (g_orch_end_time > g_orch_start_time) {
        uint64_t cycles = g_orch_end_time - g_orch_start_time;
        printf("  Orchestrator run time: %.3fus\n", cycles_to_us(cycles));
    }
}

void print_sched_profiling(PTO2Runtime* rt) {
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

#if PTO2_SCHED_PROFILING
    pto2_print_sim_sched_summary(
        &g_sched_prof_data,
        (int64_t)rt->scheduler.tasks_completed.load(std::memory_order_relaxed),
        (int64_t)rt->scheduler.tasks_consumed.load(std::memory_order_relaxed));
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


    (void)num_sched;
    // P2 unavailable: PTO2_SIM_AICORE_UT path has no per-thread profiling
}
#else
void print_orch_profiling() {}
void print_sched_profiling(PTO2Runtime* rt) { (void)rt; }
#endif
