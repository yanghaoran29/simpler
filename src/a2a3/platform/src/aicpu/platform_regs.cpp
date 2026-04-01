/*
 * Copyright (c) PyPTO Contributors.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 * -----------------------------------------------------------------------------------------------------------
 */
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
#include "aicpu/device_time.h"
#include "common/platform_config.h"
#if defined(PTO2_SIM_AICORE_UT)
#include "sim_aicore.h"
#endif

static uint64_t g_platform_regs = 0;
static uint64_t g_platform_pmu_reg_addrs = 0;

void set_platform_regs(uint64_t regs) { g_platform_regs = regs; }

uint64_t get_platform_regs() { return g_platform_regs; }

void set_platform_pmu_reg_addrs(uint64_t pmu_regs) { g_platform_pmu_reg_addrs = pmu_regs; }

uint64_t get_platform_pmu_reg_addrs() { return g_platform_pmu_reg_addrs; }

uint64_t read_reg(uint64_t reg_base_addr, RegId reg) {
#if defined(PTO2_SIM_AICORE_UT)
    if (reg_base_addr < PTO2_SIM_REG_ADDR_MAX && reg == RegId::COND) {
        // Sim cores use a 1-based reg base (see handshake synthesis); recover core_id.
        return pto2_sim_read_cond_reg(static_cast<int32_t>(reg_base_addr) - 1);
    }
#endif
    volatile uint32_t *ptr = reinterpret_cast<volatile uint32_t *>(reg_base_addr + reg_offset(reg));

    __sync_synchronize();

    // Read the register value
    uint64_t value = static_cast<uint64_t>(*ptr);

    __sync_synchronize();

    return value;
}

void write_reg(uint64_t reg_base_addr, RegId reg, uint64_t value) {
#if defined(PTO2_SIM_AICORE_UT)
    if (reg_base_addr < PTO2_SIM_REG_ADDR_MAX && reg == RegId::DATA_MAIN_BASE) {
        int32_t core_id = static_cast<int32_t>(reg_base_addr) - 1;  // 1-based sim reg base
        // Scheduler writes the per-core monotonic reg_task_id (range [0, AICORE_EXIT_SIGNAL))
        // for a dispatch, or a sentinel (AICPU_IDLE_TASK_ID / AICORE_EXIT_SIGNAL) for idle/exit.
        // The sim core echoes the task id back in COND, so pass it through unchanged.
        if (value == AICPU_IDLE_TASK_ID || value == AICORE_EXIT_SIGNAL)
            pto2_sim_aicore_set_idle(core_id);
        else {
            pto2_sim_aicore_on_task_received(core_id, static_cast<int32_t>(value));
        }
        return;
    }
#endif
    volatile uint32_t *ptr = reinterpret_cast<volatile uint32_t *>(reg_base_addr + reg_offset(reg));

    __sync_synchronize();

    // Write the register value
    *ptr = static_cast<uint32_t>(value);

    __sync_synchronize();
}

void platform_init_aicore_regs(uint64_t reg_addr) {
#if defined(PTO2_SIM_AICORE_UT)
    if (reg_addr < PTO2_SIM_REG_ADDR_MAX)
        return;  // sim core: no hardware init
#endif
    // Both a2a3 and a2a3sim require fast path control to be enabled before use
    write_reg(reg_addr, RegId::FAST_PATH_ENABLE, REG_SPR_FAST_PATH_OPEN);

    // Initialize task dispatch register to idle state
    write_reg(reg_addr, RegId::DATA_MAIN_BASE, AICPU_IDLE_TASK_ID);
}

int32_t platform_deinit_aicore_regs(uint64_t reg_addr) {
#if defined(PTO2_SIM_AICORE_UT)
    if (reg_addr < PTO2_SIM_REG_ADDR_MAX)
        return 0;  // sim core: no hardware deinit
#endif
    // Send exit signal to AICore
    write_reg(reg_addr, RegId::DATA_MAIN_BASE, AICORE_EXIT_SIGNAL);

    // Wait for AICore to acknowledge exit, with timeout.
    // On timeout, skip register cleanup (AICore is unresponsive; host will
    // aclrtResetDevice to clear all hardware state).
    uint64_t t0 = get_sys_cnt_aicpu();
    while (read_reg(reg_addr, RegId::COND) != AICORE_EXITED_VALUE) {
        if (get_sys_cnt_aicpu() - t0 > PLATFORM_DEINIT_TIMEOUT_TICKS) {
            return -1;
        }
    }

    // Initialize task dispatch register to idle state
    write_reg(reg_addr, RegId::DATA_MAIN_BASE, AICPU_IDLE_TASK_ID);
    // Close fast path control
    write_reg(reg_addr, RegId::FAST_PATH_ENABLE, REG_SPR_FAST_PATH_CLOSE);
    return 0;
}

uint32_t platform_get_physical_cores_count() {
    return DAV_2201::PLATFORM_MAX_PHYSICAL_CORES * PLATFORM_CORES_PER_BLOCKDIM;
}
