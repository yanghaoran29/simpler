/**
 * @file platform_regs.cpp
 * @brief Platform-level register access implementation for AICPU
 *
 * Provides unified interface for:
 * 1. Platform register base address management (set/get_platform_regs)
 * 2. Register read/write operations with optimized memory barriers
 * 3. Platform-agnostic AICore register initialization/deinitialization
 *
 * Memory Barrier Strategy:
 * - read_reg: Full barriers (__sync_synchronize) to ensure store-load ordering
 * - write_reg: Full barriers (__sync_synchronize) to guarantee global visibility
 *
 * Platform Support:
 * - a2a3: MMIO volatile pointer access to real hardware registers
 * - a2a3sim: Volatile pointer access to host-allocated simulated registers
 */

#include <cstdint>
#include "aicpu/platform_regs.h"
#include "common/platform_config.h"
#if defined(PTO2_SIM_AICORE_UT)
#include "sim_aicore.h"
#endif

static uint64_t g_platform_regs = 0;

void set_platform_regs(uint64_t regs) {
    g_platform_regs = regs;
}

uint64_t get_platform_regs() {
    return g_platform_regs;
}

uint64_t read_reg(uint64_t reg_base_addr, RegId reg) {
#if defined(PTO2_SIM_AICORE_UT)
    if (pto2_sim_is_current_sim() && pto2_sim_get_current_core_id() >= 0 &&
        reg_base_addr == 0 && reg == RegId::COND) {
        return pto2_sim_read_cond_reg(pto2_sim_get_current_core_id());
    }
#endif
    volatile uint32_t* ptr = reinterpret_cast<volatile uint32_t*>(
        reg_base_addr + reg_offset(reg));

    __sync_synchronize();

    uint64_t value = static_cast<uint64_t>(*ptr);

    __sync_synchronize();

    return value;
}

void write_reg(uint64_t reg_base_addr, RegId reg, uint64_t value) {
#if defined(PTO2_SIM_AICORE_UT)
    if (pto2_sim_is_current_sim() && pto2_sim_get_current_core_id() >= 0 &&
        reg_base_addr == 0 && reg == RegId::DATA_MAIN_BASE) {
        int32_t core_id = pto2_sim_get_current_core_id();
        if (value == 0 || value == AICORE_EXIT_SIGNAL)
            pto2_sim_aicore_set_idle(core_id);
        else
            pto2_sim_aicore_on_task_received(core_id, static_cast<int32_t>(value - 1));
        return;
    }
#endif
    volatile uint32_t* ptr = reinterpret_cast<volatile uint32_t*>(
        reg_base_addr + reg_offset(reg));

    __sync_synchronize();

    *ptr = static_cast<uint32_t>(value);

    __sync_synchronize();
}

void platform_init_aicore_regs(uint64_t reg_addr) {
#if defined(PTO2_SIM_AICORE_UT)
    if (reg_addr == 0)
        return;  // sim core: no hardware init
#endif
    // Both a2a3 and a2a3sim require fast path control to be enabled before use
    write_reg(reg_addr, RegId::FAST_PATH_ENABLE, REG_SPR_FAST_PATH_OPEN);
    write_reg(reg_addr, RegId::DATA_MAIN_BASE, 0);
}

void platform_deinit_aicore_regs(uint64_t reg_addr) {
#if defined(PTO2_SIM_AICORE_UT)
    if (reg_addr == 0)
        return;  // sim core: no hardware deinit
#endif
    write_reg(reg_addr, RegId::DATA_MAIN_BASE, AICORE_EXIT_SIGNAL);
    write_reg(reg_addr, RegId::FAST_PATH_ENABLE, REG_SPR_FAST_PATH_CLOSE);
}

uint32_t platform_get_physical_cores_count() {
    return DAV_2201::PLATFORM_MAX_PHYSICAL_CORES * PLATFORM_CORES_PER_BLOCKDIM;
}
