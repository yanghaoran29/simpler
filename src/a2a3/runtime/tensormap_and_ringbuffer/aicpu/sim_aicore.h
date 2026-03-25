/**
 * @file sim_aicore.h
 * @brief Simulated AICore state and API for PTO2_SIM_AICORE_UT (aicpu_ut perf tests).
 *
 * When PTO2_SIM_AICORE_UT is defined, AICore is not executed; the scheduler "dispatches"
 * to simulated cores whose COND register state is held in a global array. This header
 * declares the global state, the read API used by read_reg(), and the sim run API
 * (aicpu_sim_run_pto2) for tests.
 *
 * Register Access Context (PTO2_SIM_AICORE_UT only):
 * - When register operations occur, the sim core context tracks which core_id and mode (sim or hw)
 * - pto2_sim_set_current_core() is called at entry to a sim core region
 * - pto2_sim_clear_current_core() is called at exit
 * - SimCoreGuard provides RAII wrapper for this context
 *
 * MsgQ + HSCB simulation (PTO2_SIM_AICORE_UT):
 * - On task completion, pto2_sim_aicore_on_task_received updates COND and routes the same completion
 *   through HscbCpuSimulator::aicore_post_task_done_over_hscb → MsgqCpuSimulator (models AICore→HSCB→MsgQ).
 * - MsgQ hw_push_pair notifies waiters (condition_variable) as a stand-in for Event/interrupt wakeup.
 * - AICPU-side helpers: pto2_sim_msgq_wait_for_event, pto2_sim_msgq_pop_task_done (optional; scheduler
 *   still uses read_reg(COND) for completion unless you migrate that path).
 */

#ifndef AICPU_SIM_AICORE_H_
#define AICPU_SIM_AICORE_H_

#include <cstdint>

#if defined(PTO2_SIM_AICORE_UT)

/** When reg_base_addr is in [0, PTO2_SIM_REG_ADDR_MAX), it denotes sim core index (not MMIO address). */
#define PTO2_SIM_REG_ADDR_MAX  128

/**
 * Simulated MSGQ payload when AICore completes a task (pto2_sim_aicore_on_task_received):
 * - Low 32b of the 64b DATA word: task_id (same id as passed to MAKE_FIN_VALUE).
 * - High 32b: (PTO2_SIM_MSGQ_TASK_DONE_TAG << 16) | (core_id & 0xFFFF).
 */
#define PTO2_SIM_MSGQ_TASK_DONE_TAG 0x5444u /* "TD" — task done */

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

/** Simulated AICore: set core to idle state. */
extern "C" void pto2_sim_aicore_set_idle(int32_t core_id);

/** Simulated AICore: initialize all simulated cores to idle. Call once at sim setup. */
extern "C" void pto2_sim_aicore_init_all_idle(void);

/** Set current sim core context for register access. */
extern "C" void pto2_sim_set_current_core(int32_t core_id, bool is_sim);

/** Clear current sim core context for register access. */
extern "C" void pto2_sim_clear_current_core();

/** Get current sim core context (internal use by platform_regs.cpp). */
extern "C" int32_t pto2_sim_get_current_core_id();

/** Check if current context is a sim core (internal use by platform_regs.cpp). */
extern "C" bool pto2_sim_is_current_sim();

/** Reset run profiling state before aicpu_sim_run_pto2 (no-op if profiling disabled). */
extern "C" void pto2_sim_reset_run_prof(void);

// =============================================================================
// RAII Guard for Sim Core Register Context
// =============================================================================

#ifdef __cplusplus
struct SimCoreGuard {
    SimCoreGuard(int32_t core_id, bool is_sim) {
        pto2_sim_set_current_core(core_id, is_sim);
    }
    ~SimCoreGuard() {
        pto2_sim_clear_current_core();
    }
};

namespace cpu_sim {
class MsgqCpuSimulator;
class HscbCpuSimulator;
}

/**
 * Global simulated MSGQ (MSQ_* model): each sim AICore task completion is posted via HSCB
 * (aicore_post_task_done_over_hscb) into this object.
 */
cpu_sim::MsgqCpuSimulator* pto2_sim_msgq_for_cpu();

/** Global HSCB simulator (AICore→MsgQ completion path increments RECEIVER_SIO). */
cpu_sim::HscbCpuSimulator* pto2_sim_hscb_for_cpu();

/** Decode lo/hi 32-bit halves from one 64b MSQ_DATA read. Returns false if tag is not task_done. */
inline bool pto2_sim_msgq_decode_task_done(
    uint32_t msg_lo, uint32_t msg_hi, int32_t* out_core_id, int32_t* out_task_id) {
    if (out_core_id == nullptr || out_task_id == nullptr) {
        return false;
    }
    uint32_t tag = (msg_hi >> 16) & 0xFFFFu;
    if (tag != static_cast<uint32_t>(PTO2_SIM_MSGQ_TASK_DONE_TAG)) {
        return false;
    }
    *out_core_id = static_cast<int32_t>(msg_hi & 0xFFFFu);
    *out_task_id = static_cast<int32_t>(msg_lo);
    return true;
}

extern "C" {
/**
 * Wait until MSQ_VLDCLR_EL0 has any pending bit (simulates Event / WFE after MsgQ write).
 * timeout_ms == UINT32_MAX waits indefinitely; 0 returns immediately.
 * Returns 0 if pending, -1 if timed out.
 */
int pto2_sim_msgq_wait_for_event(uint32_t timeout_ms);

/** Pop one task_done message from MsgQ (W1C clear). Returns 0 if popped, -1 if none. */
int pto2_sim_msgq_pop_task_done(int32_t* core_id, int32_t* task_id);
}
#endif  // __cplusplus

// =============================================================================
// Sim run API (for aicpu_ut perf tests)
// =============================================================================

struct Runtime;  // forward decl for executor wrappers

#ifdef __cplusplus
extern "C" {
#endif

struct PTO2Runtime;

/** Set PTO2Runtime* used by executor during aicpu_sim_run_pto2 (called from aicpu_ut before starting scheduler threads). */
void aicpu_sim_set_rt(struct PTO2Runtime* r);

/** Executor sim init/setup/run/shutdown — called from aicpu_sim_run_pto2 in aicpu_ut. */
int aicpu_executor_sim_init(struct Runtime* r);
void aicpu_executor_sim_setup_after_host_orch(int32_t total_task_count);
int aicpu_executor_sim_run_resolve_and_dispatch_pto2(struct Runtime* r, int thread_idx);
int aicpu_executor_sim_shutdown_aicore(struct Runtime* r);

/**
 * Run scheduler via executor resolve_and_dispatch_pto2 (implementation in aicpu_ut).
 * Graph must already be built and pto2_orchestrator_done called.
 *
 * @param pto2_rt            PTO2Runtime from make_runtime() after build_*_graph + orchestrator_done
 * @param num_sched_threads  Number of scheduler threads (e.g. 3)
 * @return 0 on success, -1 on error
 */
int aicpu_sim_run_pto2(struct PTO2Runtime* pto2_rt, int num_sched_threads);

/** Return actual CPU core that scheduler thread thread_idx ran on (-1 if invalid or not run). */
int aicpu_sim_get_actual_sched_cpu(int thread_idx);

#ifdef __cplusplus
}
#endif

#ifdef __cplusplus
#include <functional>

/**
 * Concurrent variant of aicpu_sim_run_pto2: runs orchestration and scheduling
 * concurrently, mirroring real device behavior where all AICPU threads enter
 * simultaneously.
 *
 * @param pto2_rt           PTO2Runtime created by make_runtime() (graph not yet built)
 * @param num_sched_threads  Number of scheduler threads
 * @param orch_fn           Orchestration work: should call build_graph() +
 *                          pto2_orchestrator_done() internally
 * @return 0 on success, -1 on error
 */
int aicpu_sim_run_pto2_concurrent(struct PTO2Runtime* pto2_rt,
                                  int num_sched_threads,
                                  std::function<void(PTO2Runtime*)> orch_fn);
#endif

#if PTO2_SCHED_PROFILING
#include "pto_scheduler.h"
void aicpu_sim_get_saved_sched_prof(int thread_idx, PTO2SchedProfilingData* out);
/** Called by executor to store per-thread sched profiling at end of resolve_and_dispatch_pto2 (implementation in aicpu_ut). */
void aicpu_sim_set_saved_sched_prof(int thread_idx, const PTO2SchedProfilingData* data);
/** Called by executor (one call per scheduler thread) to accumulate complete/dispatch cycles into global sim summary. */
void pto2_sim_accumulate_cycles(uint64_t complete_cycle, uint64_t dispatch_cycle);
/** Retrieve accumulated complete/dispatch cycles from all scheduler threads (for printing after sim run). */
void pto2_sim_get_accumulated_cycles(uint64_t* out_complete, uint64_t* out_dispatch);
#endif  // PTO2_SCHED_PROFILING

#if PTO2_PROFILING
/** Called by executor on each subtask dispatch to record worker-type dispatch count (for P1 check). */
void pto2_sim_record_dispatch(int wt_idx);
/** Retrieve per-worker-type dispatch counts accumulated across all scheduler threads. */
void pto2_sim_get_dispatch_counts(int64_t* out, int n);
#endif  // PTO2_PROFILING

#else  // !defined(PTO2_SIM_AICORE_UT)
#ifdef __cplusplus
/**
 * Empty RAII guard when PTO2_SIM_AICORE_UT is not defined.
 * Compiles to no-op, allowing same code to work with/without sim mode.
 */
struct SimCoreGuard {
    SimCoreGuard(int32_t, bool) {}
    ~SimCoreGuard() {}
};
#endif  // __cplusplus

#endif  // PTO2_SIM_AICORE_UT

#endif  // AICPU_SIM_AICORE_H_
