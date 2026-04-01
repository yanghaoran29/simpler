/**
 * @file inner_platform_regs.cpp
 * @brief AICPU register read/write for simulation (a5sim)
 *
 * Simulated registers are two compact 4KB pages per core (8KB total).
 * sparse_reg_ptr() remaps hardware offsets to this layout:
 *   offset < 0x5000  -> page 0: reg_base + offset
 *   offset >= 0x5000 -> page 1: reg_base + 0x1000 + (offset - 0x5000)
 */

#include <cstdint>
#include "aicpu/platform_regs.h"
#include "common/platform_config.h"
#if defined(PTO2_SIM_AICORE_UT)
#include "sim_aicore.h"
#endif

uint64_t read_reg(uint64_t reg_base_addr, RegId reg) {
#if defined(PTO2_SIM_AICORE_UT)
    if (reg_base_addr < PTO2_SIM_REG_ADDR_MAX && reg == RegId::COND) {
        return pto2_sim_read_cond_reg(static_cast<int32_t>(reg_base_addr));
    }
#endif
    uint32_t offset = reg_offset(reg);
    volatile uint8_t* reg_base = reinterpret_cast<volatile uint8_t*>(reg_base_addr);
    volatile uint32_t* ptr = reinterpret_cast<volatile uint32_t*>(
        sparse_reg_ptr(reg_base, offset));

    __sync_synchronize();
    uint64_t value = static_cast<uint64_t>(*ptr);
    __sync_synchronize();

    return value;
}

void write_reg(uint64_t reg_base_addr, RegId reg, uint64_t value) {
#if defined(PTO2_SIM_AICORE_UT)
    if (reg_base_addr < PTO2_SIM_REG_ADDR_MAX && reg == RegId::DATA_MAIN_BASE) {
        int32_t core_id = static_cast<int32_t>(reg_base_addr);
        if (value == 0 || value == AICORE_EXIT_SIGNAL)
            pto2_sim_aicore_set_idle(core_id);
        else {
            // Must match the register token written by dispatch (executing_reg_task_id); do not use value-1
            // or COND's EXTRACT_TASK_ID will never equal the expected id in check_running_cores_for_completion.
            pto2_sim_aicore_on_task_received(core_id, static_cast<int32_t>(value));
        }
        return;
    }
#endif
    uint32_t offset = reg_offset(reg);
    volatile uint8_t* reg_base = reinterpret_cast<volatile uint8_t*>(reg_base_addr);
    volatile uint32_t* ptr = reinterpret_cast<volatile uint32_t*>(
        sparse_reg_ptr(reg_base, offset));

    __sync_synchronize();
    *ptr = static_cast<uint32_t>(value);
    __sync_synchronize();
}
