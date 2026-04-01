#pragma once

#include <cstdint>

// Simulated register-address space upper bound.
// In sim mode, reg_base_addr is encoded as core_id (small integer).
inline constexpr uint64_t PTO2_SIM_REG_ADDR_MAX = 0x10000ULL;

// Task-done message tag used by cpu-side MsgQ simulator.
inline constexpr uint32_t PTO2_SIM_MSGQ_TASK_DONE_TAG = 0x5444u;

extern "C" uint64_t pto2_sim_read_cond_reg(int32_t core_id);
/**
 * Install the per-func_id simulated AICore duration table (nanoseconds), owned by
 * the caller (test/benchmark case) and indexed by func_id (kernel_id). The sim
 * borrows the pointer, so it must outlive the run. A task's effective duration is
 * `durations_ns[func_id] - correction_ns` (clamped to 0); func_ids >= `count` and
 * a nullptr table complete instantly. Call once during case setup (build_graph),
 * before the scheduler dispatches any task.
 */
extern "C" void pto2_sim_aicore_set_func_duration_table(
    const int* durations_ns, int32_t count, int32_t correction_ns
);
/**
 * Record the func_id (kernel_id) of the next task dispatched to `core_id`. Must
 * be called by the owning scheduler thread immediately before the DATA_MAIN_BASE
 * write that drives pto2_sim_aicore_on_task_received, which resolves the task's
 * simulated duration from the installed per-func_id table. 0 = instant FIN.
 */
extern "C" void pto2_sim_aicore_set_task_func_id(int32_t core_id, int32_t func_id);
extern "C" void pto2_sim_aicore_start_poller(void);
extern "C" void pto2_sim_aicore_on_task_received(int32_t core_id, int32_t task_id);
extern "C" void pto2_sim_aicore_set_idle(int32_t core_id);
/**
 * Publish the host-allocated per-core L2Perf staging-ring table so each
 * dispatched task writes its slot (the scheduler's
 * l2_perf_aicpu_complete_record reads it). `ring_table[core]` is an
 * L2PerfAicoreRing*. Pass nullptr to disable. AICore execution timing is not
 * modeled — the slot carries only the receive timestamp.
 */
extern "C" void pto2_sim_aicore_set_l2_perf_ring_table(uint64_t* ring_table, int32_t num_cores);
extern "C" void pto2_sim_set_current_core(int32_t core_id, bool is_sim);
extern "C" void pto2_sim_clear_current_core();

#if defined(PTO2_SIM_AICORE_UT)
#ifdef __cplusplus
extern "C" {
struct Runtime;
struct PTO2Runtime;
void aicpu_sim_set_rt(struct PTO2Runtime* r);
int aicpu_executor_sim_init(struct Runtime* r);
void aicpu_executor_sim_setup_after_host_orch(int32_t total_task_count);
int aicpu_executor_sim_run_resolve_and_dispatch_pto2(struct Runtime* r, int thread_idx);
int aicpu_executor_sim_shutdown_aicore(struct Runtime* r);
}
#endif
#endif

