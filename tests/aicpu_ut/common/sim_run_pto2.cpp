/**
 * sim_run_pto2.cpp
 *
 * Implementation of aicpu_sim_run_pto2 and related getters/setters.
 * Moved from aicpu_executor.cpp so simulation entry points live in aicpu_ut.
 * Calls executor via aicpu_sim_set_rt, aicpu_executor_sim_* wrappers.
 */

#include <atomic>
#include <cstring>
#include <functional>
#include <thread>
#include <vector>

#include "common/platform_config.h"
#include "cpu_affinity.h"
#include "test_common.h"
#include "pto_runtime2.h"
#include "pto_runtime2_types.h"
#include "pto_shared_memory.h"
#include "runtime.h"
#include "sim_aicore.h"

// Scheduler CPU list (same as formerly in aicpu_executor.cpp; SCHED_CPU* from CMake)
static const int s_sched_cpus[] = {
    SCHED_CPU0,
    SCHED_CPU1,
    SCHED_CPU2,
    SCHED_CPU3,
    SCHED_CPU4,
    SCHED_CPU5,
    SCHED_CPU6,
    SCHED_CPU7,
};
static int s_actual_sched_cpu[PLATFORM_MAX_AICPU_THREADS];
#if PTO2_SCHED_PROFILING
#include <atomic>

#include "pto_scheduler.h"
static PTO2SchedProfilingData s_sched_prof_snapshot[PLATFORM_MAX_AICPU_THREADS] = {};
static std::atomic<uint64_t> s_sim_complete_cycles{0};
static std::atomic<uint64_t> s_sim_dispatch_cycles{0};
#endif

#if PTO2_PROFILING
#include <atomic>

#include "aicpu/device_log.h"
#include "aicpu/device_time.h"
#include "pto_runtime2_types.h"
static std::atomic<int64_t> s_sim_tasks_dispatched[PTO2_NUM_WORKER_TYPES] = {};
#endif

#if PTO2_ORCH_PROFILING
#include "pto_orchestrator.h"
#endif

extern "C" void pto2_sim_reset_run_prof(void) {
#if PTO2_SCHED_PROFILING
    s_sim_complete_cycles.store(0, std::memory_order_relaxed);
    s_sim_dispatch_cycles.store(0, std::memory_order_relaxed);
#endif
#if PTO2_PROFILING
    for (int i = 0; i < PTO2_NUM_WORKER_TYPES; i++) s_sim_tasks_dispatched[i].store(0, std::memory_order_relaxed);
#endif
}

extern "C" {

int aicpu_sim_run_pto2(PTO2Runtime* pto2_rt, int num_sched_threads) {
    (void)num_sched_threads;
    if (!pto2_rt || !pto2_rt->sm_handle) return -1;

    pto2_sim_reset_run_prof();

    for (int i = 0; i < PLATFORM_MAX_AICPU_THREADS; i++) s_actual_sched_cpu[i] = -1;
    int total = sim_run_all(pto2_rt, 1000000);
    return (total >= 0) ? 0 : -1;
}

}  // extern "C"

int aicpu_sim_run_pto2_concurrent(
    PTO2Runtime* pto2_rt, int num_sched_threads, std::function<void(PTO2Runtime*)> orch_fn) {
    if (!pto2_rt || !pto2_rt->sm_handle) return -1;
    if (!orch_fn) return -1;

    std::thread orch_thread([pto2_rt, &orch_fn, num_sched_threads]() {
#if PTO2_PROFILING
        uint64_t orch_t0 = get_sys_cnt_aicpu();
#endif
        orch_fn(pto2_rt);
#if PTO2_PROFILING
        DEV_ALWAYS("Thread %d: aicpu_orchestration_entry returned, cost %.3fus (orch_idx=0)",
                   num_sched_threads, cycles_to_us(get_sys_cnt_aicpu() - orch_t0));
#endif
#if PTO2_ORCH_PROFILING
        {
            PTO2OrchProfilingData p = pto2_orchestrator_get_profiling();
            uint64_t total = p.sync_cycle + p.alloc_cycle + p.params_cycle +
                             p.lookup_cycle + p.insert_cycle + p.fanin_cycle;
            if (total == 0) total = 1;
            DEV_ALWAYS("Thread %d: === Orchestrator Profiling: %lld tasks, total=%.3fus ===", num_sched_threads,
                     (long long)p.submit_count, cycles_to_us(total));
            DEV_ALWAYS("Thread %d:   sync_tensormap : %.3fus (%.1f%%)", num_sched_threads, cycles_to_us(p.sync_cycle), p.sync_cycle * 100.0 / total);
            DEV_ALWAYS("Thread %d:   task_ring_alloc: %.3fus (%.1f%%)  work=%.3fus wait=%.3fus  atomics=%llu", num_sched_threads,
                cycles_to_us(p.alloc_cycle), p.alloc_cycle * 100.0 / total,
                cycles_to_us(p.alloc_cycle - p.alloc_wait_cycle), cycles_to_us(p.alloc_wait_cycle),
                (unsigned long long)p.alloc_atomic_count);
            DEV_ALWAYS("Thread %d:   param_copy     : %.3fus (%.1f%%)  atomics=%llu", num_sched_threads,
                cycles_to_us(p.params_cycle), p.params_cycle * 100.0 / total,
                (unsigned long long)p.params_atomic_count);
            DEV_ALWAYS("Thread %d:   lookup+dep     : %.3fus (%.1f%%)", num_sched_threads, cycles_to_us(p.lookup_cycle), p.lookup_cycle * 100.0 / total);
            DEV_ALWAYS("Thread %d:   tensormap_ins  : %.3fus (%.1f%%)", num_sched_threads, cycles_to_us(p.insert_cycle), p.insert_cycle * 100.0 / total);
            DEV_ALWAYS("Thread %d:   fanin+ready    : %.3fus (%.1f%%)  work=%.3fus wait=%.3fus  atomics=%llu", num_sched_threads,
                cycles_to_us(p.fanin_cycle), p.fanin_cycle * 100.0 / total,
                cycles_to_us(p.fanin_cycle - p.fanin_wait_cycle), cycles_to_us(p.fanin_wait_cycle),
                (unsigned long long)p.fanin_atomic_count);
            DEV_ALWAYS("Thread %d:   scope_end      : %.3fus  atomics=%llu", num_sched_threads,
                cycles_to_us(p.scope_end_cycle),
                (unsigned long long)p.scope_end_atomic_count);
            DEV_ALWAYS("Thread %d:   avg/task       : %.3fus", num_sched_threads,
                p.submit_count > 0 ? cycles_to_us(total) / p.submit_count : 0.0);
        }
#endif
    });
    orch_thread.join();
    return aicpu_sim_run_pto2(pto2_rt, num_sched_threads);
}

extern "C" {

int aicpu_sim_get_actual_sched_cpu(int thread_idx) {
    if (thread_idx < 0 || thread_idx >= PLATFORM_MAX_AICPU_THREADS) return -1;
    return s_actual_sched_cpu[thread_idx];
}

}  // extern "C"

#if PTO2_SCHED_PROFILING
void aicpu_sim_get_saved_sched_prof(int thread_idx, PTO2SchedProfilingData* out) {
    if (!out || thread_idx < 0 || thread_idx >= PLATFORM_MAX_AICPU_THREADS) return;
    *out = s_sched_prof_snapshot[thread_idx];
}

void aicpu_sim_set_saved_sched_prof(int thread_idx, const PTO2SchedProfilingData* data) {
    if (!data || thread_idx < 0 || thread_idx >= PLATFORM_MAX_AICPU_THREADS) return;
    s_sched_prof_snapshot[thread_idx] = *data;
}

void pto2_sim_accumulate_cycles(uint64_t complete_cycle, uint64_t dispatch_cycle) {
    s_sim_complete_cycles.fetch_add(complete_cycle, std::memory_order_relaxed);
    s_sim_dispatch_cycles.fetch_add(dispatch_cycle, std::memory_order_relaxed);
}

void pto2_sim_get_accumulated_cycles(uint64_t* out_complete, uint64_t* out_dispatch) {
    if (out_complete) *out_complete = s_sim_complete_cycles.load(std::memory_order_relaxed);
    if (out_dispatch) *out_dispatch = s_sim_dispatch_cycles.load(std::memory_order_relaxed);
}
#endif

#if PTO2_PROFILING
void pto2_sim_record_dispatch(int wt_idx) {
    if (wt_idx >= 0 && wt_idx < PTO2_NUM_WORKER_TYPES)
        s_sim_tasks_dispatched[wt_idx].fetch_add(1, std::memory_order_relaxed);
}

void pto2_sim_get_dispatch_counts(int64_t* out, int n) {
    for (int i = 0; i < n && i < PTO2_NUM_WORKER_TYPES; i++)
        out[i] = s_sim_tasks_dispatched[i].load(std::memory_order_relaxed);
}
#endif
