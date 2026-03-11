/**
 * @file sim_aicore.h
 * @brief Simulated AICore state and API for PTO2_SIM_AICORE_UT (aicpu_ut perf tests).
 *
 * When PTO2_SIM_AICORE_UT is defined, AICore is not executed; the scheduler "dispatches"
 * to simulated cores whose COND register state is held in a global array. This header
 * declares the global state, the read API used by read_reg(), and the sim run API
 * (aicpu_sim_run_pto2, aicpu_sim_get_run_prof) for tests.
 */

#ifndef AICPU_SIM_AICORE_H_
#define AICPU_SIM_AICORE_H_

#include <cstdint>

#if defined(PTO2_SIM_AICORE_UT)

/** Global COND register state per simulated core (MAKE_FIN_VALUE(task_id) or AICORE_IDLE_VALUE). */
extern uint32_t s_sim_core_cond_value[];

/** Read simulated COND for core_id; used by platform read_reg() when reg_base_addr==0. */
extern "C" uint64_t pto2_sim_read_cond_reg(int32_t core_id);

/**
 * Simulated AICore: when a task is "received" (dispatched to this core), immediately mark it
 * completed by writing COND = FIN(task_id). Scheduler will see this on the next poll and
 * call on_task_complete. No kernel execution; sim receives task -> instant FIN.
 */
extern "C" void pto2_sim_aicore_on_task_received(int32_t core_id, int32_t task_id);

/**
 * Simulated AICore: set core to idle state (e.g. before first dispatch or after shutdown).
 */
extern "C" void pto2_sim_aicore_set_idle(int32_t core_id);

/**
 * Simulated AICore: initialize all simulated cores to idle. Call once at sim setup.
 */
extern "C" void pto2_sim_aicore_init_all_idle(void);

// -----------------------------------------------------------------------------
// Sim run API (for aicpu_ut perf tests)
// -----------------------------------------------------------------------------

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

/** Return actual CPU core that scheduler thread thread_idx ran on (from sched_getcpu after bind). -1 if invalid or not run. */
int aicpu_sim_get_actual_sched_cpu(int thread_idx);

/** Scheduler run profiling (fanout/fanin edges, tasks_dispatched) for aicpu_sim_run_pto2. */
#define AICPU_SIM_PROF_WORKER_TYPES 4
typedef struct AicpuSimRunProf {
    int64_t tasks_dispatched[AICPU_SIM_PROF_WORKER_TYPES];
    int64_t fanout_edges_total;
    int32_t fanout_max_degree;
    int64_t tasks_enqueued_by_completion;
    int64_t fanin_edges_total;
    int32_t fanin_max_degree;
    int64_t rounds_total;       /* scheduler loop iterations (all threads sum) */
    int64_t rounds_with_progress; /* iterations where made_progress was true */
    uint64_t complete_cycle;   /* sched_complete_cycle sum across all threads */
    uint64_t dispatch_cycle;   /* sched_dispatch_cycle sum across all threads */
} AicpuSimRunProf;

/**
 * Reset sim run profiling counters. Called at start of aicpu_sim_run_pto2.
 */
void pto2_sim_reset_run_prof(void);

/** Accumulate fanin (from on_task_release). */
void pto2_sim_accumulate_fanin(int32_t fe);

/** Accumulate fanout (from on_task_complete). */
void pto2_sim_accumulate_fanout(int64_t edges, int64_t enqueued, int32_t max_degree);

/** Accumulate one dispatched task for worker_type. */
void pto2_sim_accumulate_dispatch(int32_t worker_type);

/** Accumulate cycle counts for complete/dispatch phases. */
void pto2_sim_accumulate_cycles(uint64_t complete_cycle, uint64_t dispatch_cycle);

/** Accumulate one scheduler loop round; with_progress 1 if made_progress this round. */
void pto2_sim_accumulate_rounds(int64_t total_inc, int64_t with_progress_inc);

/**
 * Fill run profiling from last aicpu_sim_run_pto2 (PTO2_SIM_AICORE_UT only).
 * Call after aicpu_sim_run_pto2 so print_sched_profiling can show fanout/fanin.
 */
void aicpu_sim_get_run_prof(AicpuSimRunProf* out);

#ifdef __cplusplus
}
#endif

#if PTO2_SCHED_PROFILING
#include "pto_scheduler.h"
#ifdef __cplusplus
extern "C" {
#endif
/**
 * Get the per-thread sched profiling snapshot saved at the end of the last
 * aicpu_sim_run_pto2 call.  Safe to call multiple times (non-destructive).
 */
void aicpu_sim_get_saved_sched_prof(int thread_idx, PTO2SchedProfilingData* out);
#ifdef __cplusplus
}
#endif
#endif  // PTO2_SCHED_PROFILING

#endif  // PTO2_SIM_AICORE_UT

#endif  // AICPU_SIM_AICORE_H_
