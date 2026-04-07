/*
 * Instruction counting support for A2A3Sim and A5Sim
 *
 * This module provides instruction counting capabilities using PTO2 markers
 * and integrates with QEMU TCG plugins for validation.
 *
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef SIMPLER_SRC_COMMON_INSTR_COUNT_H_
#define SIMPLER_SRC_COMMON_INSTR_COUNT_H_

#include <cstdint>
#include <string>
#include <vector>
#include <map>

namespace simpler {

/**
 * @brief Marker-based instruction counting session
 *
 * Tracks instruction count within a specific phase between two markers
 */
struct MarkerSession {
    uint64_t session_id;          ///< Unique session ID
    uint32_t phase_id;            ///< Phase identifier
    uint32_t cpu_id;              ///< CPU/thread ID
    uint64_t insn_count;          ///< Instruction count for this session
    uint64_t start_time;          ///< Start timestamp
    uint64_t end_time;            ///< End timestamp
};

/**
 * @brief Per-thread/vCPU marker state
 */
struct VcpuMarkerState {
    std::vector<int64_t> session_stack;        ///< Stack of active session indices
    std::vector<uint32_t> phase_stack;         ///< Stack of active phase IDs
    uint32_t depth;                            ///< Current stack depth
    uint64_t between_markers_insns;            ///< Total instructions between markers
    std::vector<MarkerSession> sessions;       ///< All completed sessions
};

/**
 * @brief Marker pair definition
 *
 * Defines a start and end instruction encoding for a phase
 */
struct MarkerPair {
    uint32_t phase_id;
    uint32_t start_enc;           ///< Start instruction encoding
    uint32_t end_enc;             ///< End instruction encoding
    std::string phase_name;       ///< Phase name for display
};

/**
 * @brief Instruction counter manager
 *
 * Manages marker-based instruction counting for simulation
 */
class InstrCounter {
public:
    static constexpr int MAX_VCPU = 64;
    static constexpr int MAX_MARKER_STACK = 32;

    InstrCounter() : next_session_id_(0) {}

    /**
     * @brief Initialize counter for given number of vCPUs
     */
    void init(uint32_t num_vcpu);

    /**
     * @brief Start a marker phase
     *
     * @param cpu_id CPU/thread ID
     * @param phase_id Phase identifier
     * @return Session ID or -1 on error
     */
    int64_t marker_start(uint32_t cpu_id, uint32_t phase_id);

    /**
     * @brief End a marker phase
     *
     * @param cpu_id CPU/thread ID
     * @param phase_id Phase identifier (for validation)
     */
    void marker_end(uint32_t cpu_id, uint32_t phase_id);

    /**
     * @brief Record an instruction execution
     *
     * Increments counter for all active sessions on the given vCPU
     *
     * @param cpu_id CPU/thread ID
     */
    void record_insn(uint32_t cpu_id);

    /**
     * @brief Check if a vCPU is currently inside any marker
     */
    bool is_active(uint32_t cpu_id) const;

    /**
     * @brief Get total instruction count between markers
     */
    uint64_t get_total_between_markers_insns(uint32_t cpu_id) const;

    /**
     * @brief Export results to Chrome Trace Format
     *
     * @param output_path Path to output file
     * @return 0 on success, -1 on error
     */
    int export_chrome_trace(const std::string& output_path);

    /**
     * @brief Export results to JSON format
     *
     * @param output_path Path to output file
     * @return 0 on success, -1 on error
     */
    int export_json(const std::string& output_path);

    /**
     * @brief Register marker pair definition
     */
    void register_marker_pair(const MarkerPair& pair);

    /**
     * @brief Get marker pair by start encoding
     */
    const MarkerPair* find_marker_pair_by_start(uint32_t enc) const;

    /**
     * @brief Get marker pair by phase ID
     */
    const MarkerPair* find_marker_pair_by_phase(uint32_t phase_id) const;

private:
    struct VcpuMarkerState vcpu_states_[MAX_VCPU];
    std::vector<MarkerPair> marker_pairs_;
    uint64_t next_session_id_;

    int append_session(uint32_t cpu_id, uint32_t phase_id);
};

}  // namespace simpler

#endif  // SIMPLER_SRC_COMMON_INSTR_COUNT_H_
