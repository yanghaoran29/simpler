/*
 * Instruction counting implementation
 *
 * SPDX-License-Identifier: Apache-2.0
 */

#include "src/common/instr_count.h"
#include <cstring>
#include <fstream>
#include <sstream>
#include <algorithm>
#include <json.h>

namespace simpler {

void InstrCounter::init(uint32_t num_vcpu) {
    if (num_vcpu > MAX_VCPU) {
        num_vcpu = MAX_VCPU;
    }

    for (uint32_t i = 0; i < num_vcpu; i++) {
        vcpu_states_[i].depth = 0;
        vcpu_states_[i].between_markers_insns = 0;
        vcpu_states_[i].session_stack.resize(MAX_MARKER_STACK);
        vcpu_states_[i].phase_stack.resize(MAX_MARKER_STACK);
    }
}

int64_t InstrCounter::marker_start(uint32_t cpu_id, uint32_t phase_id) {
    if (cpu_id >= MAX_VCPU) {
        return -1;
    }

    VcpuMarkerState* st = &vcpu_states_[cpu_id];
    if (st->depth >= MAX_MARKER_STACK) {
        return -1;  // Stack overflow
    }

    // Create new session
    int idx = append_session(cpu_id, phase_id);
    if (idx < 0) {
        return -1;
    }

    // Push to stack
    st->session_stack[st->depth] = idx;
    st->phase_stack[st->depth] = phase_id;
    st->depth++;

    return idx;
}

void InstrCounter::marker_end(uint32_t cpu_id, uint32_t phase_id) {
    if (cpu_id >= MAX_VCPU) {
        return;
    }

    VcpuMarkerState* st = &vcpu_states_[cpu_id];

    // Backward match: find matching phase_id on stack
    uint32_t d = st->depth;
    while (d > 0) {
        uint32_t top = d - 1;
        if (st->phase_stack[top] == phase_id) {
            st->depth = top;  // Pop stack
            break;
        }
        d--;
    }
}

void InstrCounter::record_insn(uint32_t cpu_id) {
    if (cpu_id >= MAX_VCPU) {
        return;
    }

    VcpuMarkerState* st = &vcpu_states_[cpu_id];
    uint32_t d = st->depth;

    if (d > 0) {
        // Increment counter for all active sessions
        for (uint32_t k = 0; k < d; k++) {
            int64_t active_idx = st->session_stack[k];
            if (active_idx >= 0 && active_idx < static_cast<int64_t>(st->sessions.size())) {
                st->sessions[active_idx].insn_count++;
            }
        }

        // Global counter
        st->between_markers_insns++;
    }
}

bool InstrCounter::is_active(uint32_t cpu_id) const {
    if (cpu_id >= MAX_VCPU) {
        return false;
    }
    return vcpu_states_[cpu_id].depth > 0;
}

uint64_t InstrCounter::get_total_between_markers_insns(uint32_t cpu_id) const {
    if (cpu_id >= MAX_VCPU) {
        return 0;
    }
    return vcpu_states_[cpu_id].between_markers_insns;
}

int InstrCounter::append_session(uint32_t cpu_id, uint32_t phase_id) {
    if (cpu_id >= MAX_VCPU) {
        return -1;
    }

    VcpuMarkerState* st = &vcpu_states_[cpu_id];

    // Create new session
    MarkerSession session = {
        .session_id = next_session_id_++,
        .phase_id = phase_id,
        .cpu_id = cpu_id,
        .insn_count = 0,
        .start_time = 0,
        .end_time = 0,
    };

    st->sessions.push_back(session);
    return static_cast<int>(st->sessions.size() - 1);
}

void InstrCounter::register_marker_pair(const MarkerPair& pair) {
    // Check for duplicates
    for (const auto& existing : marker_pairs_) {
        if (existing.phase_id == pair.phase_id) {
            // Replace existing
            for (auto& p : marker_pairs_) {
                if (p.phase_id == pair.phase_id) {
                    p = pair;
                    return;
                }
            }
        }
    }
    marker_pairs_.push_back(pair);
}

const MarkerPair* InstrCounter::find_marker_pair_by_start(uint32_t enc) const {
    for (const auto& pair : marker_pairs_) {
        if (pair.start_enc == enc) {
            return &pair;
        }
    }
    return nullptr;
}

const MarkerPair* InstrCounter::find_marker_pair_by_phase(uint32_t phase_id) const {
    for (const auto& pair : marker_pairs_) {
        if (pair.phase_id == phase_id) {
            return &pair;
        }
    }
    return nullptr;
}

int InstrCounter::export_chrome_trace(const std::string& output_path) {
    std::ofstream ofs(output_path);
    if (!ofs) {
        return -1;
    }

    ofs << "{\n  \"traceEvents\": [\n";

    bool first = true;
    for (uint32_t cpu_id = 0; cpu_id < MAX_VCPU; cpu_id++) {
        const auto& st = vcpu_states_[cpu_id];
        for (const auto& session : st.sessions) {
            if (!first) {
                ofs << ",\n";
            }
            first = false;

            const MarkerPair* pair = find_marker_pair_by_phase(session.phase_id);
            std::string phase_name = pair ? pair->phase_name : "unknown";

            ofs << "    {\n";
            ofs << "      \"name\": \"" << phase_name << "\",\n";
            ofs << "      \"ph\": \"X\",\n";
            ofs << "      \"ts\": " << session.start_time << ",\n";
            ofs << "      \"dur\": " << (session.end_time - session.start_time) << ",\n";
            ofs << "      \"pid\": 1,\n";
            ofs << "      \"tid\": " << session.cpu_id << ",\n";
            ofs << "      \"args\": {\n";
            ofs << "        \"insn_count\": " << session.insn_count << ",\n";
            ofs << "        \"phase_id\": " << session.phase_id << ",\n";
            ofs << "        \"session_id\": " << session.session_id << "\n";
            ofs << "      }\n";
            ofs << "    }";
        }
    }

    ofs << "\n  ]\n}\n";
    ofs.close();
    return 0;
}

int InstrCounter::export_json(const std::string& output_path) {
    std::ofstream ofs(output_path);
    if (!ofs) {
        return -1;
    }

    ofs << "{\n";
    ofs << "  \"vcpu_stats\": [\n";

    bool first_vcpu = true;
    for (uint32_t cpu_id = 0; cpu_id < MAX_VCPU; cpu_id++) {
        const auto& st = vcpu_states_[cpu_id];

        if (st.sessions.empty() && st.between_markers_insns == 0) {
            continue;
        }

        if (!first_vcpu) {
            ofs << ",\n";
        }
        first_vcpu = false;

        ofs << "    {\n";
        ofs << "      \"cpu_id\": " << cpu_id << ",\n";
        ofs << "      \"total_between_markers_insns\": " << st.between_markers_insns << ",\n";
        ofs << "      \"session_count\": " << st.sessions.size() << ",\n";
        ofs << "      \"sessions\": [\n";

        bool first_session = true;
        for (const auto& session : st.sessions) {
            if (!first_session) {
                ofs << ",\n";
            }
            first_session = false;

            const MarkerPair* pair = find_marker_pair_by_phase(session.phase_id);
            std::string phase_name = pair ? pair->phase_name : "unknown";

            ofs << "        {\n";
            ofs << "          \"session_id\": " << session.session_id << ",\n";
            ofs << "          \"phase_id\": " << session.phase_id << ",\n";
            ofs << "          \"phase_name\": \"" << phase_name << "\",\n";
            ofs << "          \"insn_count\": " << session.insn_count << "\n";
            ofs << "        }";
        }

        ofs << "\n      ]\n";
        ofs << "    }";
    }

    ofs << "\n  ]\n}\n";
    ofs.close();
    return 0;
}

}  // namespace simpler
