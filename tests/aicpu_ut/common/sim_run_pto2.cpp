/**
 * sim_run_pto2.cpp
 *
 * PTO2_SIM_AICORE_UT: implementation of aicpu_sim_run_pto2 and related getters/setters.
 * Moved from aicpu_executor.cpp so simulation entry points live in aicpu_ut.
 * Calls executor via aicpu_sim_set_rt, aicpu_executor_sim_* wrappers.
 */

#if defined(PTO2_SIM_AICORE_UT)

#include "sim_aicore.h"
#include "cpu_affinity.h"
#include "runtime.h"
#include "pto_runtime2.h"
#include "pto_runtime2_types.h"
#include "pto_shared_memory.h"
#include "common/platform_config.h"
#include <atomic>
#include <cstring>
#include <functional>
#include <thread>
#include <vector>

// Scheduler CPU list (same as formerly in aicpu_executor.cpp; SCHED_CPU* from CMake)
static const int s_sched_cpus[] = {
    SCHED_CPU0, SCHED_CPU1, SCHED_CPU2, SCHED_CPU3,
    SCHED_CPU4, SCHED_CPU5, SCHED_CPU6, SCHED_CPU7,
};
static int s_actual_sched_cpu[PLATFORM_MAX_AICPU_THREADS];

#if PTO2_SCHED_PROFILING
#include "pto_scheduler.h"
#include <atomic>
static PTO2SchedProfilingData s_sched_prof_snapshot[PLATFORM_MAX_AICPU_THREADS] = {};
static std::atomic<uint64_t> s_sim_complete_cycles{0};
static std::atomic<uint64_t> s_sim_dispatch_cycles{0};
#endif

#if PTO2_PROFILING
#include <atomic>
#include "pto_runtime2_types.h"
static std::atomic<int64_t> s_sim_tasks_dispatched[PTO2_NUM_WORKER_TYPES] = {};
#endif

extern "C" void pto2_sim_reset_run_prof(void) {
#if PTO2_SCHED_PROFILING
    s_sim_complete_cycles.store(0, std::memory_order_relaxed);
    s_sim_dispatch_cycles.store(0, std::memory_order_relaxed);
#endif
#if PTO2_PROFILING
    for (int i = 0; i < PTO2_NUM_WORKER_TYPES; i++)
        s_sim_tasks_dispatched[i].store(0, std::memory_order_relaxed);
#endif
}

extern "C" {

int aicpu_sim_run_pto2(PTO2Runtime* pto2_rt, int num_sched_threads) {
    if (!pto2_rt || !pto2_rt->sm_handle) return -1;
    void* sm_base = pto2_rt->sm_handle->sm_base;
    if (!sm_base) return -1;

    pto2_sim_reset_run_prof();

    const int SIM_CORE_COUNT = PLATFORM_MAX_CORES;
    Runtime runtime;
    runtime.set_pto2_gm_sm_ptr(sm_base);
    runtime.worker_count = SIM_CORE_COUNT;
    memset(runtime.workers, 0, sizeof(runtime.workers));
    runtime.sche_cpu_num = num_sched_threads;
    runtime.orch_thread_num = 0;  // host already did orchestration, all threads are schedulers
    runtime.set_orch_built_on_host(true);

    int rc = aicpu_executor_sim_init(&runtime);
    if (rc != 0) return rc;

    aicpu_sim_set_rt(pto2_rt);
    PTO2SharedMemoryHeader* header = static_cast<PTO2SharedMemoryHeader*>(sm_base);
    int32_t total = 0;
    for (int r = 0; r < PTO2_MAX_RING_DEPTH; r++) {
        total += header->rings[r].fc.current_task_index.load(std::memory_order_acquire);
    }
    aicpu_executor_sim_setup_after_host_orch(total);

    for (int i = 0; i < PLATFORM_MAX_AICPU_THREADS; i++)
        s_actual_sched_cpu[i] = -1;

    std::vector<std::thread> threads;
    for (int i = 0; i < num_sched_threads; i++) {
        threads.emplace_back([&runtime, i]() {
            if (i < (int)(sizeof(s_sched_cpus) / sizeof(s_sched_cpus[0])))
                bind_to_cpu(s_sched_cpus[i]);
            if (i >= 0 && i < PLATFORM_MAX_AICPU_THREADS) {
                int cur = current_cpu();
                s_actual_sched_cpu[i] = (cur >= 0) ? cur : -1;
            }
            aicpu_executor_sim_run_resolve_and_dispatch_pto2(&runtime, i);
        });
    }
    for (auto& t : threads) t.join();

    aicpu_executor_sim_shutdown_aicore(&runtime);
    return 0;
}

}  // extern "C"

int aicpu_sim_run_pto2_concurrent(PTO2Runtime* pto2_rt, int num_sched_threads,
                                  std::function<void(PTO2Runtime*)> orch_fn) {
    if (!pto2_rt || !pto2_rt->sm_handle) return -1;
    void* sm_base = pto2_rt->sm_handle->sm_base;
    if (!sm_base) return -1;

    pto2_sim_reset_run_prof();

    const int SIM_CORE_COUNT = PLATFORM_MAX_CORES;
    Runtime runtime;
    runtime.set_pto2_gm_sm_ptr(sm_base);
    runtime.worker_count = SIM_CORE_COUNT;
    memset(runtime.workers, 0, sizeof(runtime.workers));
    runtime.sche_cpu_num = num_sched_threads;
    runtime.orch_thread_num = 0;
    runtime.set_orch_built_on_host(true);
    runtime.set_orch_deferred_on_host(true);  // orch runs in separate thread; init must not set orchestrator_done_

    int rc = aicpu_executor_sim_init(&runtime);
    if (rc != 0) return rc;

    aicpu_sim_set_rt(pto2_rt);
    // Do NOT call setup_after_host_orch here: orch thread will call it after build_graph + pto2_orchestrator_done

    for (int i = 0; i < PLATFORM_MAX_AICPU_THREADS; i++)
        s_actual_sched_cpu[i] = -1;

    std::vector<std::thread> sched_threads;
    for (int i = 0; i < num_sched_threads; i++) {
        sched_threads.emplace_back([&runtime, i]() {
            if (i < (int)(sizeof(s_sched_cpus) / sizeof(s_sched_cpus[0])))
                bind_to_cpu(s_sched_cpus[i]);
            if (i >= 0 && i < PLATFORM_MAX_AICPU_THREADS) {
                int cur = current_cpu();
                s_actual_sched_cpu[i] = (cur >= 0) ? cur : -1;
            }
            aicpu_executor_sim_run_resolve_and_dispatch_pto2(&runtime, i);
        });
    }

    std::thread orch_thread([pto2_rt, sm_base, &orch_fn]() {
        orch_fn(pto2_rt);
        PTO2SharedMemoryHeader* hdr = static_cast<PTO2SharedMemoryHeader*>(sm_base);
        int32_t total = 0;
        for (int r = 0; r < PTO2_MAX_RING_DEPTH; r++) {
            total += hdr->rings[r].fc.current_task_index.load(std::memory_order_acquire);
        }
        aicpu_executor_sim_setup_after_host_orch(total);
    });

    orch_thread.join();
    for (auto& t : sched_threads) t.join();

    aicpu_executor_sim_shutdown_aicore(&runtime);
    return 0;
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

#endif  // PTO2_SIM_AICORE_UT
