/**
 * CPU-side simulation model for MSGQ (message buffer / short valid-clear path).
 * Register-level model: MSQ_VLDCLR_EL0, MSQ_SEL_EL0, MSQ_DATA_EL0,
 * MSQ_SHORT_VLDCLR0_EL0, CUBE/VECTOR paths (see design notes).
 */
#pragma once

#include <cstdint>
#include <condition_variable>
#include <functional>
#include <mutex>
#include <vector>

namespace cpu_sim {

/** Tag in high 16 bits of task_done message (must match PTO2_SIM_MSGQ_TASK_DONE_TAG in sim_aicore.h). */
inline constexpr uint32_t kMsgqSimTaskDoneTag = 0x5444u;

/** Logical indices used in the notes (CUBE vs VECTOR message paths). */
enum class MsgqPath : int { CUBE = 0, VECTOR = 1, PATH_COUNT = 2 };

/**
 * EL0-visible MSGQ registers (MRS/MSR style), one bank.
 * - vldclr: valid bits on read; write-1-to-clear per bit (0 = no change).
 * - sel:    entry index (pair index: one 64b DATA holds two 32b messages).
 * - data:   64-bit readback for selected entry (low/high 32-bit messages).
 */
struct MsgqEl0Regs {
    uint64_t vldclr{};
    uint64_t sel{};
    uint64_t data{};
};

/**
 * "Short" valid/clear register(s) for fast paths (e.g. MSQ_SHORT_VLDCLR0_EL0).
 * Index with MsgqPath (CUBE / VECTOR).
 */
struct MsgqShortRegs {
    uint64_t vldclr[static_cast<int>(MsgqPath::PATH_COUNT)]{};
};

/**
 * Control block mirrored from reference code: hardware flags + merged free state.
 */
struct MsgqCtrlBlock {
    uint64_t msgb_flag[static_cast<int>(MsgqPath::PATH_COUNT)]{};
    uint64_t free_flag[static_cast<int>(MsgqPath::PATH_COUNT)]{};
};

/**
 * Backing store: up to \a kMaxPairs slots; each slot is 64-bit = two 32-bit messages.
 * Valid bits in vldclr map as: bit (2*i) -> low 32, bit (2*i+1) -> high 32 of pair i.
 */
class MsgqCpuSimulator {
public:
    static constexpr int kMaxPairs = 32;

    using CompSignalFn = std::function<void(uint64_t cleared_mask)>;

    MsgqCpuSimulator() = default;

    void set_completion_callback(CompSignalFn fn) { on_comp_ = std::move(fn); }

    /** Clear all valid bits and entry RAM; keeps completion callback. Sim init / between runs. */
    void reset();

    /** MRS MSQ_VLDCLR_EL0 */
    uint64_t read_vldclr_el0() const;

    /** MSR MSQ_SEL_EL0 */
    void write_sel_el0(uint64_t idx);

    /** MRS MSQ_DATA_EL0 — uses current sel as pair index. */
    uint64_t read_data_el0();

    /**
     * MSR MSQ_VLDCLR_EL0: bits set to 1 clear the corresponding valid bits (W1C).
     * Triggers optional completion callback with the mask that was cleared.
     */
    void write_vldclr_el0_w1c(uint64_t clear_mask);

    uint64_t read_short_vldclr(MsgqPath p) const;

    void write_short_vldclr_w1c(MsgqPath p, uint64_t clear_mask);

    /** Model hardware raising bits on the short-path valid register (OR). */
    void hw_raise_short_valid(MsgqPath p, uint64_t mask);

    /**
     * Simulate DAZCore / upstream write into pair \a pair_index.
     * Sets valid bits for low and/or high 32-bit half.
     */
    void hw_push_pair(int pair_index, uint32_t msg_lo, uint32_t msg_hi, bool valid_lo, bool valid_hi);

    void set_error_address(uint64_t addr) { last_error_addr_ = addr; }
    uint64_t last_error_address() const { return last_error_addr_; }

    /** Reference-code path: pull short regs into ctrl, clear ctrl msgb, OR into free_flag. */
    void sync_short_into_ctrl(MsgqCtrlBlock& ctrl);

    const MsgqEl0Regs& el0_regs() const { return el0_; }
    const MsgqShortRegs& short_regs() const { return short_; }

    /** Const view of backing 64-bit word for pair \a pair_index (two 32-bit messages). */
    uint64_t peek_entry_data(int pair_index) const;

    /**
     * Simulates Event / interrupt wakeup after an HSCB write posts to MsgQ: block until MSQ_VLDCLR_EL0
     * has any bit set, or timeout. Use timeout_ms == UINT32_MAX for infinite wait; 0 returns immediately.
     */
    bool wait_for_pending_ms(uint32_t timeout_ms);

    /**
     * AICPU path: pop one task_done entry (tag kMsgqSimTaskDoneTag), W1C clear its valid bits.
     * Returns false if no matching pending entry.
     */
    bool try_pop_task_done(int32_t* out_core_id, int32_t* out_task_id);

private:
    void touch_data_view();
    uint64_t pair_valid_mask(int pair_index) const;

    mutable std::mutex sync_mtx_{};
    std::condition_variable pending_cv_{};

    MsgqEl0Regs el0_{};
    MsgqShortRegs short_{};
    uint64_t entry_data_[kMaxPairs]{};
    uint64_t last_error_addr_{};
    CompSignalFn on_comp_{};
};

/**
 * High-level helper: scan vldclr like the handwritten loop (pairs of bits).
 * For each pair i with any valid bit, invokes callback with (i, data64, valid_lo, valid_hi).
 */
void msgq_for_each_pending(const MsgqCpuSimulator& sim,
    const std::function<void(int pair_index, uint64_t data64, bool valid_lo, bool valid_hi)>& fn);

/**
 * Read pending messages into \a out (32-bit stream), applying the note's rule:
 * always take low 32 if valid bit (2*i); take high 32 if valid bit (2*i+1).
 * Does not clear valid bits — caller should MSR vldclr after processing.
 */
void msgq_drain_pending_32(MsgqCpuSimulator& sim, std::vector<uint32_t>& out);

/** Names aligned with handwritten READ_REG/WRITE_REG / MRS MSR patterns (simulation only). */
inline uint64_t READ_REG_MSQ_VLDCLR_EL0(const MsgqCpuSimulator& sim) {
    return sim.read_vldclr_el0();
}
inline void WRITE_REG_MSQ_VLDCLR_EL0(MsgqCpuSimulator& sim, uint64_t w1c_clear_mask) {
    sim.write_vldclr_el0_w1c(w1c_clear_mask);
}
inline void WRITE_REG_MSQ_SEL_EL0(MsgqCpuSimulator& sim, uint64_t idx) {
    sim.write_sel_el0(idx);
}
inline uint64_t READ_REG_MSQ_DATA_EL0(MsgqCpuSimulator& sim) {
    return sim.read_data_el0();
}
inline uint64_t READ_REG_MSQ_SHORT_VLDCLR0_EL0(const MsgqCpuSimulator& sim) {
    return sim.read_short_vldclr(MsgqPath::CUBE);
}
inline void WRITE_REG_MSQ_SHORT_VLDCLR0_EL0(MsgqCpuSimulator& sim, uint64_t w1c_clear_mask) {
    sim.write_short_vldclr_w1c(MsgqPath::CUBE, w1c_clear_mask);
}
/** VECTOR path short register (name placeholder MSQ_SHORT_VLDCLR1_EL0 in RTL). */
inline uint64_t READ_REG_MSQ_SHORT_VLDCLR1_EL0(const MsgqCpuSimulator& sim) {
    return sim.read_short_vldclr(MsgqPath::VECTOR);
}
inline void WRITE_REG_MSQ_SHORT_VLDCLR1_EL0(MsgqCpuSimulator& sim, uint64_t w1c_clear_mask) {
    sim.write_short_vldclr_w1c(MsgqPath::VECTOR, w1c_clear_mask);
}

}  // namespace cpu_sim
