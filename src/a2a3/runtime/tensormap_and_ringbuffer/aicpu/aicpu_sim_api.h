/**
 * @file aicpu_sim_api.h
 * @brief API for aicpu_ut perf tests to run via AicpuExecutor::resolve_and_dispatch_pto2.
 *
 * When PTO2_SIM_AICORE_UT is defined, aicpu_sim_run_pto2 runs the scheduler using
 * the real resolve_and_dispatch_pto2 (core_num==0 path) instead of test_common's
 * sim_run_with_resolve_and_dispatch.
 */

#ifndef AICPU_SIM_API_H_
#define AICPU_SIM_API_H_

#ifdef __cplusplus
extern "C" {
#endif

struct PTO2Runtime;

/**
 * Run scheduler via AicpuExecutor::resolve_and_dispatch_pto2 (PTO2_SIM_AICORE_UT only).
 * Graph must already be built and pto2_orchestrator_done called.
 *
 * @param pto2_rt         PTO2Runtime from make_runtime() after build_*_graph + orchestrator_done
 * @param num_sched_threads  Number of scheduler threads (e.g. 3)
 * @return 0 on success, -1 on error
 */
int aicpu_sim_run_pto2(struct PTO2Runtime* pto2_rt, int num_sched_threads);

/** Scheduler run profiling (fanout/fanin edges, tasks_dispatched) for aicpu_sim_run_pto2. */
#define AICPU_SIM_PROF_WORKER_TYPES 4
typedef struct AicpuSimRunProf {
    int64_t tasks_dispatched[AICPU_SIM_PROF_WORKER_TYPES];
    int64_t fanout_edges_total;
    int32_t fanout_max_degree;
    int64_t tasks_enqueued_by_completion;
    int64_t fanin_edges_total;
    int32_t fanin_max_degree;
    uint64_t complete_cycle;   /* sched_complete_cycle sum across all threads */
    uint64_t dispatch_cycle;   /* sched_dispatch_cycle sum across all threads */
} AicpuSimRunProf;

/**
 * Fill run profiling from last aicpu_sim_run_pto2 (PTO2_SIM_AICORE_UT only).
 * Call after aicpu_sim_run_pto2 so print_sched_profiling can show fanout/fanin.
 */
void aicpu_sim_get_run_prof(AicpuSimRunProf* out);

#ifdef __cplusplus
}

#if PTO2_SCHED_PROFILING
#include "pto_scheduler.h"
extern "C" {
/**
 * Get the per-thread sched profiling snapshot saved at the end of the last
 * aicpu_sim_run_pto2 call.  Safe to call multiple times (non-destructive).
 */
void aicpu_sim_get_saved_sched_prof(int thread_idx, PTO2SchedProfilingData* out);
} // extern "C"
#endif

#else  // __cplusplus not defined
#endif /* __cplusplus */

#endif /* AICPU_SIM_API_H_ */
