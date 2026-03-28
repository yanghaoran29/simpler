#pragma once

#include <cstdint>

// Simulated register-address space upper bound.
// In sim mode, reg_base_addr is encoded as core_id (small integer).
inline constexpr uint64_t PTO2_SIM_REG_ADDR_MAX = 0x10000ULL;

// Task-done message tag used by cpu-side MsgQ simulator.
inline constexpr uint32_t PTO2_SIM_MSGQ_TASK_DONE_TAG = 0x5444u;

extern "C" uint64_t pto2_sim_read_cond_reg(int32_t core_id);
extern "C" void pto2_sim_aicore_on_task_received(int32_t core_id, int32_t task_id);
extern "C" void pto2_sim_aicore_set_idle(int32_t core_id);
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

