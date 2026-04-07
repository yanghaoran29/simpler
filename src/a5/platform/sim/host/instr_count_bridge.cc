/*
 * A5Sim instruction counting bridge implementation
 *
 * SPDX-License-Identifier: Apache-2.0
 */

#include "src/a5/platform/sim/host/instr_count_bridge.h"
#include "src/common/instr_count.h"
#include "src/common/pto2_markers.h"
#include <glog/logging.h>

namespace simpler {

A5InstrCountBridge::A5InstrCountBridge()
    : counter_(nullptr), enabled_(false), num_aicpu_(0), num_aicore_(0) {}

A5InstrCountBridge::~A5InstrCountBridge() = default;

void A5InstrCountBridge::init(uint32_t num_aicpu, uint32_t num_aicore, bool enable) {
    enabled_ = enable;
    num_aicpu_ = num_aicpu;
    num_aicore_ = num_aicore;

    if (!enable) {
        return;
    }

    // Create and initialize counter
    counter_ = std::make_unique<InstrCounter>();
    uint32_t total_threads = num_aicpu + num_aicore;
    counter_->init(total_threads);

    // Register default marker pairs
    // Format: marker ID -> (start_enc, end_enc, phase_name)
    MarkerPair pairs[] = {
        {0, 0xaa030063u, 0xaa040084u, "submit_total"},           // orr x3, x3, x3 / orr x4, x4, x4
        {10, 0xaa0f01efu, 0xaa100210u, "sched_loop"},            // orr x15, x15, x15 / orr x16, x16, x16
        {11, 0xaa1102afu, 0xaa120309u, "sched_complete"},        // orr x17, x17, x17 / orr x18, x18, x18
        {12, 0xaa130329u, 0xaa140349u, "sched_dispatch"},        // orr x19, x19, x19 / orr x20, x20, x20
        {13, 0xaa150369u, 0xaa160389u, "sched_idle"},            // orr x21, x21, x21 / orr x22, x22, x22
    };

    for (const auto& pair : pairs) {
        counter_->register_marker_pair(pair);
    }

    LOG(INFO) << "A5Sim instruction counting initialized: " << total_threads << " threads";
}

bool A5InstrCountBridge::handle_marker(uint32_t cpu_id, uint32_t insn_enc) {
    if (!enabled_ || !counter_) {
        return false;
    }

    // Check for start marker
    const MarkerPair* pair = counter_->find_marker_pair_by_start(insn_enc);
    if (pair) {
        counter_->marker_start(cpu_id, pair->phase_id);
        return true;
    }

    // Check for end marker
    for (uint32_t phase_id = 0; phase_id < 100; ++phase_id) {
        const MarkerPair* p = counter_->find_marker_pair_by_phase(phase_id);
        if (p && p->end_enc == insn_enc) {
            counter_->marker_end(cpu_id, phase_id);
            return true;
        }
    }

    return false;
}

void A5InstrCountBridge::record_insn(uint32_t cpu_id) {
    if (enabled_ && counter_) {
        counter_->record_insn(cpu_id);
    }
}

int A5InstrCountBridge::export_results(const std::string& output_path, const std::string& format) {
    if (!enabled_ || !counter_) {
        return -1;
    }

    if (format == "json") {
        return counter_->export_json(output_path);
    } else if (format == "chrome_trace") {
        return counter_->export_chrome_trace(output_path);
    }

    LOG(ERROR) << "Unknown export format: " << format;
    return -1;
}

}  // namespace simpler
