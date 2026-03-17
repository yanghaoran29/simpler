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
#endif  // __cplusplus

// =============================================================================
// Sim run API (for aicpu_ut perf tests)
// =============================================================================

#ifdef __cplusplus
extern "C" {
#endif

struct PTO2Runtime;

/**
 * Run scheduler via AicpuExecutor::resolve_and_dispatch_pto2 (PTO2_SIM_AICORE_UT only).
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
