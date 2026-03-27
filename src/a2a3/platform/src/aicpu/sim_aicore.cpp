#include "sim_aicore.h"

#include <array>
#include <atomic>

namespace {
constexpr int kMaxSimCores = 4096;
std::array<std::atomic<uint64_t>, kMaxSimCores> g_cond_regs{};

thread_local int32_t g_current_core_id = -1;
thread_local bool g_current_is_sim = false;

inline bool core_valid(int32_t core_id) {
    return core_id >= 0 && core_id < kMaxSimCores;
}
}  // namespace

extern "C" uint64_t pto2_sim_read_cond_reg(int32_t core_id) {
    if (!core_valid(core_id)) {
        return 0;
    }
    return g_cond_regs[core_id].load(std::memory_order_acquire);
}

extern "C" void pto2_sim_aicore_on_task_received(int32_t core_id, int32_t task_id) {
    if (!core_valid(core_id)) {
        return;
    }
    // Non-zero means "has recent task activity"; keep semantics lightweight.
    g_cond_regs[core_id].store(static_cast<uint64_t>(task_id + 1), std::memory_order_release);
}

extern "C" void pto2_sim_aicore_set_idle(int32_t core_id) {
    if (!core_valid(core_id)) {
        return;
    }
    g_cond_regs[core_id].store(0, std::memory_order_release);
}

extern "C" void pto2_sim_set_current_core(int32_t core_id, bool is_sim) {
    g_current_core_id = core_id;
    g_current_is_sim = is_sim;
}

extern "C" void pto2_sim_clear_current_core() {
    g_current_core_id = -1;
    g_current_is_sim = false;
}

