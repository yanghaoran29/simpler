/**
 * @file platform_regs.cpp
 * @brief AICPU register interface — set/get, init/deinit, core count
 *
 * read_reg/write_reg are provided by:
 *   sim/aicpu/inner_platform_regs.cpp    (aicpu_ut / a5sim)
 *   onboard/aicpu/inner_platform_regs.cpp (device)
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

void platform_init_aicore_regs(uint64_t reg_addr) {
#if defined(PTO2_SIM_AICORE_UT)
    if (reg_addr < PTO2_SIM_REG_ADDR_MAX)
        return;
#endif
    write_reg(reg_addr, RegId::DATA_MAIN_BASE, AICPU_IDLE_TASK_ID);
}

void platform_deinit_aicore_regs(uint64_t reg_addr) {
#if defined(PTO2_SIM_AICORE_UT)
    if (reg_addr < PTO2_SIM_REG_ADDR_MAX)
        return;
#endif
    write_reg(reg_addr, RegId::DATA_MAIN_BASE, AICORE_EXIT_SIGNAL);

    while (read_reg(reg_addr, RegId::COND) != AICORE_EXITED_VALUE) {
    }

    write_reg(reg_addr, RegId::DATA_MAIN_BASE, AICPU_IDLE_TASK_ID);
}

uint32_t platform_get_physical_cores_count() {
    return DAV_3510::PLATFORM_MAX_PHYSICAL_CORES * PLATFORM_CORES_PER_BLOCKDIM;
}
