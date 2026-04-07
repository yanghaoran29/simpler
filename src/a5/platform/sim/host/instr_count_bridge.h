/*
 * A5Sim instruction counting integration
 *
 * This module provides instruction counting support for the A5 simulator
 * using PTO2 markers and the InstrCounter class.
 *
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef SIMPLER_SRC_A5_PLATFORM_SIM_HOST_INSTR_COUNT_BRIDGE_H_
#define SIMPLER_SRC_A5_PLATFORM_SIM_HOST_INSTR_COUNT_BRIDGE_H_

#include <cstdint>
#include <string>
#include <memory>

namespace simpler {

class InstrCounter;

/**
 * @brief Bridge for A5Sim to integrate instruction counting
 *
 * Manages instruction counting lifecycle during simulation execution
 */
class A5InstrCountBridge {
public:
    A5InstrCountBridge();
    ~A5InstrCountBridge();

    /**
     * @brief Initialize instruction counter for A5Sim
     *
     * @param num_aicpu Number of AICPU threads
     * @param num_aicore Number of AICore threads (block_dim * 3)
     * @param enable True to enable counting, false to disable
     */
    void init(uint32_t num_aicpu, uint32_t num_aicore, bool enable);

    /**
     * @brief Check if instruction counting is enabled
     */
    bool is_enabled() const { return enabled_; }

    /**
     * @brief Handle a detected marker instruction
     *
     * @param cpu_id CPU/thread ID
     * @param insn_enc Instruction encoding
     * @return True if marker was recognized, false otherwise
     */
    bool handle_marker(uint32_t cpu_id, uint32_t insn_enc);

    /**
     * @brief Record a regular instruction execution
     *
     * @param cpu_id CPU/thread ID
     */
    void record_insn(uint32_t cpu_id);

    /**
     * @brief Export counting results
     *
     * @param output_path Output file path
     * @param format Output format: "json" or "chrome_trace"
     * @return 0 on success, -1 on error
     */
    int export_results(const std::string& output_path, const std::string& format = "json");

    /**
     * @brief Get the underlying counter instance
     */
    InstrCounter* get_counter() { return counter_.get(); }

private:
    std::unique_ptr<InstrCounter> counter_;
    bool enabled_;
    uint32_t num_aicpu_;
    uint32_t num_aicore_;
};

}  // namespace simpler

#endif  // SIMPLER_SRC_A5_PLATFORM_SIM_HOST_INSTR_COUNT_BRIDGE_H_
