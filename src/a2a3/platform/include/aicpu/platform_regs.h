/**
 * @file platform_regs.h
 * @brief Platform-level register access interface for AICPU
 *
 * Provides unified interface for:
 * 1. Platform register base address management (set/get_platform_regs)
 * 2. Register read/write operations (read_reg/write_reg)
 *
 * The platform layer calls set_platform_regs() before aicpu_execute(),
 * and runtime code calls get_platform_regs() and read_reg/write_reg()
 * for register communication with AICore.
 *
 * Implementation: src/platform/src/aicpu/platform_regs.cpp (shared across all platforms)
 */

#ifndef PLATFORM_AICPU_PLATFORM_REGS_H_
#define PLATFORM_AICPU_PLATFORM_REGS_H_

#include <cstdint>
#include <cstddef>
#include "common/platform_config.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * Set the platform register base address array.
 * Called by the platform layer before aicpu_execute().
 *
 * @param regs  Pointer (as uint64_t) to per-core register base address array
 */
void set_platform_regs(uint64_t regs);

/**
 * Get the platform register base address array.
 * Called by runtime AICPU executor code that needs register access.
 *
 * @return Pointer (as uint64_t) to per-core register base address array
 */
uint64_t get_platform_regs();

#ifdef __cplusplus
}
#endif

#if defined(PTO2_SIM_AICORE_UT)
/** When PTO2_SIM_AICORE_UT: read simulated COND from global s_sim_core_cond_value[core_id] */
extern "C" uint64_t pto2_sim_read_cond_reg(int32_t core_id);
#endif

/**
 * Read a register value from an AICore's register block
 *
 * @param reg_base_addr  Base address of the AICore's register block (0 in sim => use pto2_sim_read_cond_reg when sim_core_id >= 0)
 * @param reg            Register identifier (C++ enum class)
 * @return Register value (zero-extended to uint64_t)
 */
#if defined(PTO2_SIM_AICORE_UT)
/** When PTO2_SIM_AICORE_UT: sim_core_id is used for sim COND read when reg_base_addr==0 and reg==COND. */
uint64_t read_reg(uint64_t reg_base_addr, RegId reg, int32_t sim_core_id);
#else
uint64_t read_reg(uint64_t reg_base_addr, RegId reg);
#endif

/**
 * Write a value to an AICore's register
 *
 * @param reg_base_addr  Base address of the AICore's register block (0 in sim => use sim_aicore when sim_core_id >= 0)
 * @param reg            Register identifier (C++ enum class)
 * @param value          Value to write (truncated to register width)
 */
#if defined(PTO2_SIM_AICORE_UT)
/** When PTO2_SIM_AICORE_UT: sim_core_id is used for sim dispatch when reg_base_addr==0 and reg==DATA_MAIN_BASE. */
void write_reg(uint64_t reg_base_addr, RegId reg, uint64_t value, int32_t sim_core_id);
#else
void write_reg(uint64_t reg_base_addr, RegId reg, uint64_t value);
#endif

/**
 * Initialize AICore registers after core discovery
 *
 * This function performs platform-agnostic register initialization that works
 * for both a2a3 and a2a3sim, including enabling fast path control and clearing
 * dispatch registers.
 *
 * @param reg_addr  Register base address of the AICore
 */
void platform_init_aicore_regs(uint64_t reg_addr);

/**
 * Deinitialize AICore registers before termination
 *
 * This function sends exit signal and closes fast path control.
 *
 * @param reg_addr  Register base address of the AICore
 */
void platform_deinit_aicore_regs(uint64_t reg_addr);

/**
 * Get physical core count for current platform
 *
 * This function returns the maximum valid physical_core_id value (exclusive upper bound).
 * Used for validating physical_core_id from AICore handshake before using as array index.
 *
 * @return Physical core count (exclusive upper bound)
 */
uint32_t platform_get_physical_cores_count();

/**
 * Invalidate data cache for a memory range.
 *
 * On ARM64 AICPU, DMA writes from the host (rtMemcpy) go directly to HBM
 * without invalidating the AICPU's data cache.  When rtMalloc returns the
 * same device address across rounds, the AICPU may read stale cached data
 * instead of the fresh values written by the host.
 *
 * On real hardware (onboard): performs DC CIVAC per cache line + DSB/ISB.
 * On simulation (sim): no-op.
 *
 * @param addr  Start address of the memory range
 * @param size  Size of the memory range in bytes
 */
void cache_invalidate_range(const void* addr, size_t size);

#endif  // PLATFORM_AICPU_PLATFORM_REGS_H_
