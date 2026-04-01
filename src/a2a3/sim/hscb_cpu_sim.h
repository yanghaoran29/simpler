/**
 * CPU-side simulation model for HSCB (High-Speed Control Bus).
 * Five logical blocks, SPR_REG_0[aic_id], task types (inc_cnt, send_wait, check_flag),
 * RegWrite and MsgQ push.
 */
#pragma once

#include <cstdint>
#include <functional>
#include <string>

namespace cpu_sim {

class MsgqCpuSimulator;

enum class HscbModule : uint8_t {
    SENDER = 0,
    DISTRIBUTOR,
    RECEIVER,
    RECEIVER_SIO,
    SUBSENDER,
    MODULE_COUNT
};

enum class HscbSendType : uint8_t { INC_CNT = 0, WAIT = 1, CUSTOM = 2 };

struct HscbSendCommand {
    HscbSendType type{HscbSendType::CUSTOM};
    int stars_c_id{-1};
    uint32_t flag_id{0};
    uint64_t cnt_val{0};
};

struct HscbTaskHandle {
    uint64_t opaque{0};
    int src_aic{-1};
};

/**
 * Simulated per-AIC SPR slot (task pointer / command word as in notes).
 */
struct HscbCpuSimulator {
    static constexpr int kMaxAic = 64;

    using StarsSink = std::function<void(const HscbSendCommand&)>;
    using LogFn = std::function<void(const std::string&)>;

    void set_stars_sink(StarsSink s) { stars_sink_ = std::move(s); }
    void set_log(LogFn l) { log_ = std::move(l); }

    /** HSCB_RegWrite(SPR_REG_0[aic_id], task_ptr) */
    void reg_write(int aic_id, uint64_t task_ptr);

    uint64_t reg_read(int aic_id) const;

    /** inc_cnt: HSCB_Send(STARS[c_id], type=inc) */
    void send_inc_cnt(int stars_c_id);

    /** send_wait: HSCB_Send(src_core=c_id, flag=f_id, cnt=it->CntVal) */
    void send_wait(int src_core, uint32_t flag_id, uint64_t cnt_val);

    /**
     * check_flag: poll until (thread-local value) meets threshold.
     * Returns true if condition satisfied within max_spins.
     */
    bool check_flag_poll(std::function<uint64_t()> read_thread_val, uint64_t threshold, int max_spins = 1 << 20);

    /** HSCB_MsgQ[task_ptr.src].push — deliver completion to MSGQ (optional hook). */
    void push_to_msgq(MsgqCpuSimulator& mq, int src_aic, uint32_t msg_lo, uint32_t msg_hi, bool v_lo, bool v_hi,
        int pair_index = 0);

    /**
     * AICore → HSCB → AICPU MsgQ (simulated): models the bus write that posts task_done and (in hardware)
     * would fire Event/interrupt. Pair index = core_id % MsgqCpuSimulator::kMaxPairs.
     */
    void aicore_post_task_done_over_hscb(MsgqCpuSimulator& mq, int core_id, int32_t task_id);

    int module_activity_count(HscbModule m) const { return module_counts_[static_cast<int>(m)]; }

private:
    void bump(HscbModule m) { ++module_counts_[static_cast<int>(m)]; }
    void trace(const std::string& s) const;

    uint64_t spr_reg0_[kMaxAic]{};
    int module_counts_[static_cast<int>(HscbModule::MODULE_COUNT)]{};
    StarsSink stars_sink_{};
    LogFn log_{};
};

}  // namespace cpu_sim
