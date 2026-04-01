#include "msgq_cpu_sim.h"

#include <chrono>
#include <limits>

namespace cpu_sim {

void MsgqCpuSimulator::reset() {
    std::lock_guard<std::mutex> lock(sync_mtx_);
    el0_ = {};
    short_ = {};
    for (int i = 0; i < kMaxPairs; ++i) {
        entry_data_[i] = 0;
    }
    last_error_addr_ = 0;
    pending_cv_.notify_all();
}

uint64_t MsgqCpuSimulator::read_vldclr_el0() const {
    std::lock_guard<std::mutex> lock(sync_mtx_);
    return el0_.vldclr;
}

uint64_t MsgqCpuSimulator::read_short_vldclr(MsgqPath p) const {
    std::lock_guard<std::mutex> lock(sync_mtx_);
    return short_.vldclr[static_cast<int>(p)];
}

void MsgqCpuSimulator::write_sel_el0(uint64_t idx) {
    std::lock_guard<std::mutex> lock(sync_mtx_);
    el0_.sel = idx;
}

uint64_t MsgqCpuSimulator::peek_entry_data(int pair_index) const {
    std::lock_guard<std::mutex> lock(sync_mtx_);
    if (pair_index < 0 || pair_index >= kMaxPairs) {
        return 0;
    }
    return entry_data_[pair_index];
}

uint64_t MsgqCpuSimulator::pair_valid_mask(int pair_index) const {
    std::lock_guard<std::mutex> lock(sync_mtx_);
    if (pair_index < 0 || pair_index >= kMaxPairs) return 0;
    uint64_t b0 = 1ull << (2 * pair_index);
    uint64_t b1 = 1ull << (2 * pair_index + 1);
    return el0_.vldclr & (b0 | b1);
}

void MsgqCpuSimulator::touch_data_view() {
    int idx = static_cast<int>(el0_.sel);
    if (idx < 0 || idx >= kMaxPairs) {
        el0_.data = 0;
        return;
    }
    el0_.data = entry_data_[idx];
}

uint64_t MsgqCpuSimulator::read_data_el0() {
    std::lock_guard<std::mutex> lock(sync_mtx_);
    touch_data_view();
    return el0_.data;
}

void MsgqCpuSimulator::write_vldclr_el0_w1c(uint64_t clear_mask) {
    std::lock_guard<std::mutex> lock(sync_mtx_);
    uint64_t cleared = el0_.vldclr & clear_mask;
    el0_.vldclr &= ~clear_mask;
    if (on_comp_ && cleared != 0) {
        on_comp_(cleared);
    }
}

void MsgqCpuSimulator::write_short_vldclr_w1c(MsgqPath p, uint64_t clear_mask) {
    std::lock_guard<std::mutex> lock(sync_mtx_);
    int i = static_cast<int>(p);
    short_.vldclr[i] &= ~clear_mask;
}

void MsgqCpuSimulator::hw_raise_short_valid(MsgqPath p, uint64_t mask) {
    std::lock_guard<std::mutex> lock(sync_mtx_);
    int i = static_cast<int>(p);
    short_.vldclr[i] |= mask;
}

void MsgqCpuSimulator::hw_push_pair(int pair_index, uint32_t msg_lo, uint32_t msg_hi, bool valid_lo, bool valid_hi) {
    std::lock_guard<std::mutex> lock(sync_mtx_);
    if (pair_index < 0 || pair_index >= kMaxPairs) {
        return;
    }
    uint64_t new_valid = 0;
    if (valid_lo) new_valid |= 1ull << (2 * pair_index);
    if (valid_hi) new_valid |= 1ull << (2 * pair_index + 1);

    uint64_t overlap = el0_.vldclr & new_valid;
    if (overlap != 0) {
        last_error_addr_ = static_cast<uint64_t>(pair_index) * 8u;
    }

    uint64_t word = (static_cast<uint64_t>(msg_hi) << 32) | static_cast<uint64_t>(msg_lo);
    entry_data_[pair_index] = word;
    el0_.vldclr |= new_valid;
    pending_cv_.notify_all();
}

void MsgqCpuSimulator::sync_short_into_ctrl(MsgqCtrlBlock& ctrl) {
    std::lock_guard<std::mutex> lock(sync_mtx_);
    for (int p = 0; p < static_cast<int>(MsgqPath::PATH_COUNT); ++p) {
        uint64_t local = short_.vldclr[p] | ctrl.msgb_flag[p];
        ctrl.msgb_flag[p] = 0;
        ctrl.free_flag[p] |= local;
    }
}

bool MsgqCpuSimulator::wait_for_pending_ms(uint32_t timeout_ms) {
    std::unique_lock<std::mutex> lock(sync_mtx_);
    auto pred = [this] { return el0_.vldclr != 0; };
    if (pred()) {
        return true;
    }
    if (timeout_ms == 0) {
        return false;
    }
    if (timeout_ms == std::numeric_limits<uint32_t>::max()) {
        pending_cv_.wait(lock, pred);
        return pred();
    }
    return pending_cv_.wait_for(lock, std::chrono::milliseconds(timeout_ms), pred);
}

bool MsgqCpuSimulator::try_pop_task_done(int32_t* out_core_id, int32_t* out_task_id) {
    if (out_core_id == nullptr || out_task_id == nullptr) {
        return false;
    }
    std::lock_guard<std::mutex> lock(sync_mtx_);
    for (int i = 0; i < kMaxPairs; ++i) {
        uint64_t mask_pair = (3ull << (2 * i));
        if ((el0_.vldclr & mask_pair) == 0) {
            continue;
        }
        uint64_t word = entry_data_[i];
        uint32_t lo = static_cast<uint32_t>(word & 0xFFFFFFFFu);
        uint32_t hi = static_cast<uint32_t>((word >> 32) & 0xFFFFFFFFu);
        uint32_t tag = (hi >> 16) & 0xFFFFu;
        if (tag != kMsgqSimTaskDoneTag) {
            continue;
        }
        uint64_t clr = 0;
        if (el0_.vldclr & (1ull << (2 * i))) clr |= 1ull << (2 * i);
        if (el0_.vldclr & (1ull << (2 * i + 1))) clr |= 1ull << (2 * i + 1);
        el0_.vldclr &= ~clr;
        if (on_comp_ && clr != 0) {
            on_comp_(clr);
        }
        *out_core_id = static_cast<int32_t>(hi & 0xFFFFu);
        *out_task_id = static_cast<int32_t>(lo);
        return true;
    }
    return false;
}

void msgq_for_each_pending(const MsgqCpuSimulator& sim,
    const std::function<void(int pair_index, uint64_t data64, bool valid_lo, bool valid_hi)>& fn) {
    uint64_t v = sim.read_vldclr_el0();
    for (int i = 0; i < MsgqCpuSimulator::kMaxPairs; ++i) {
        uint64_t mask = (1ull << (2 * i)) | (1ull << (2 * i + 1));
        if ((v & mask) == 0) {
            continue;
        }
        bool lo = (v & (1ull << (2 * i))) != 0;
        bool hi = (v & (1ull << (2 * i + 1))) != 0;
        uint64_t data64 = sim.peek_entry_data(i);
        fn(i, data64, lo, hi);
    }
}

void msgq_drain_pending_32(MsgqCpuSimulator& sim, std::vector<uint32_t>& out) {
    uint64_t v = sim.read_vldclr_el0();
    for (int i = 0; i < MsgqCpuSimulator::kMaxPairs; ++i) {
        if ((v & (3ull << (2 * i))) == 0) {
            continue;
        }
        sim.write_sel_el0(static_cast<uint64_t>(i));
        uint64_t r5 = sim.read_data_el0();
        if (v & (1ull << (2 * i))) {
            out.push_back(static_cast<uint32_t>(r5 & 0xFFFFFFFFu));
        }
        if (v & (1ull << (2 * i + 1))) {
            out.push_back(static_cast<uint32_t>((r5 >> 32) & 0xFFFFFFFFu));
        }
    }
}

}  // namespace cpu_sim
