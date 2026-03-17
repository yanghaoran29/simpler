/**
 * @file sim_aicore.cpp
 * @brief Simulated AICore COND state and API (PTO2_SIM_AICORE_UT).
 *
 * Simulates the AICore side of the register handshake: scheduler writes
 * DATA_MAIN_BASE = task_id+1 -> sim "receives" task -> we set COND = FIN(task_id)
 * so the scheduler sees completion on the next read_reg(COND) without running a kernel.
 *
 * Also manages thread-local sim core context for register access interception
 * (pto2_sim_set_current_core / pto2_sim_clear_current_core).
 */

#if defined(PTO2_SIM_AICORE_UT)

#include "sim_aicore.h"
#include "runtime.h"                   // RUNTIME_MAX_WORKER
#include "common/platform_config.h"   // AICORE_IDLE_VALUE, MAKE_FIN_VALUE

uint32_t s_sim_core_cond_value[RUNTIME_MAX_WORKER] = {};

// =============================================================================
// Thread-Local Sim Core Context (for register access interception)
// =============================================================================

/**
 * Thread-local context tracking which sim core is being accessed.
 * Set by pto2_sim_set_current_core() at entry to sim core region,
 * cleared by pto2_sim_clear_current_core() at exit.
 * Used by platform_regs.cpp's read_reg/write_reg to intercept sim core access.
 */
struct SimCoreContext {
    int32_t core_id;
    bool is_sim;
};

static thread_local SimCoreContext g_sim_core_ctx = { -1, false };

extern "C" void pto2_sim_set_current_core(int32_t core_id, bool is_sim) {
    g_sim_core_ctx.core_id = core_id;
    g_sim_core_ctx.is_sim = is_sim;
}

extern "C" void pto2_sim_clear_current_core() {
    g_sim_core_ctx.core_id = -1;
    g_sim_core_ctx.is_sim = false;
}

extern "C" int32_t pto2_sim_get_current_core_id() {
    return g_sim_core_ctx.core_id;
}

extern "C" bool pto2_sim_is_current_sim() {
    return g_sim_core_ctx.is_sim;
}

extern "C" uint64_t pto2_sim_read_cond_reg(int32_t core_id) {
    if (core_id < 0 || core_id >= RUNTIME_MAX_WORKER)
        return static_cast<uint64_t>(AICORE_IDLE_VALUE);
    return static_cast<uint64_t>(s_sim_core_cond_value[core_id]);
}

extern "C" void pto2_sim_aicore_on_task_received(int32_t core_id, int32_t task_id) {
    if (core_id < 0 || core_id >= RUNTIME_MAX_WORKER)
        return;
    s_sim_core_cond_value[core_id] = static_cast<uint32_t>(MAKE_FIN_VALUE(task_id));
}

extern "C" void pto2_sim_aicore_set_idle(int32_t core_id) {
    if (core_id < 0 || core_id >= RUNTIME_MAX_WORKER)
        return;
    s_sim_core_cond_value[core_id] = static_cast<uint32_t>(AICORE_IDLE_VALUE);
}

extern "C" void pto2_sim_aicore_init_all_idle(void) {
    for (int32_t i = 0; i < RUNTIME_MAX_WORKER; i++)
        s_sim_core_cond_value[i] = static_cast<uint32_t>(AICORE_IDLE_VALUE);
}

#endif  // PTO2_SIM_AICORE_UT
