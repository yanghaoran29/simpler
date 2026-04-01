#include "hscb_cpu_sim.h"

#include "msgq_cpu_sim.h"

#include <string>

namespace cpu_sim {

void HscbCpuSimulator::trace(const std::string& s) const {
    if (log_) {
        log_(s);
    }
}

void HscbCpuSimulator::reg_write(int aic_id, uint64_t task_ptr) {
    if (aic_id < 0 || aic_id >= kMaxAic) {
        trace("HSCB reg_write: aic_id out of range");
        return;
    }
    bump(HscbModule::DISTRIBUTOR);
    spr_reg0_[aic_id] = task_ptr;
    trace("HSCB RegWrite SPR_REG_0[" + std::to_string(aic_id) + "] = " + std::to_string(task_ptr));
}

uint64_t HscbCpuSimulator::reg_read(int aic_id) const {
    if (aic_id < 0 || aic_id >= kMaxAic) {
        return 0;
    }
    return spr_reg0_[aic_id];
}

void HscbCpuSimulator::send_inc_cnt(int stars_c_id) {
    bump(HscbModule::SENDER);
    HscbSendCommand cmd{};
    cmd.type = HscbSendType::INC_CNT;
    cmd.stars_c_id = stars_c_id;
    trace("HSCB send_inc_cnt -> STARS[" + std::to_string(stars_c_id) + "]");
    if (stars_sink_) {
        stars_sink_(cmd);
    }
}

void HscbCpuSimulator::send_wait(int src_core, uint32_t flag_id, uint64_t cnt_val) {
    bump(HscbModule::SENDER);
    HscbSendCommand cmd{};
    cmd.type = HscbSendType::WAIT;
    cmd.stars_c_id = src_core;
    cmd.flag_id = flag_id;
    cmd.cnt_val = cnt_val;
    trace("HSCB send_wait core=" + std::to_string(src_core) + " flag=" + std::to_string(flag_id));
    if (stars_sink_) {
        stars_sink_(cmd);
    }
}

bool HscbCpuSimulator::check_flag_poll(std::function<uint64_t()> read_thread_val, uint64_t threshold, int max_spins) {
    bump(HscbModule::RECEIVER);
    for (int spin = 0; spin < max_spins; ++spin) {
        if (read_thread_val() >= threshold) {
            return true;
        }
    }
    return false;
}

void HscbCpuSimulator::push_to_msgq(MsgqCpuSimulator& mq, int src_aic, uint32_t msg_lo, uint32_t msg_hi, bool v_lo,
    bool v_hi, int pair_index) {
    (void)src_aic;
    bump(HscbModule::SUBSENDER);
    trace("HSCB push_to_msgq pair=" + std::to_string(pair_index));
    mq.hw_push_pair(pair_index, msg_lo, msg_hi, v_lo, v_hi);
}

void HscbCpuSimulator::aicore_post_task_done_over_hscb(MsgqCpuSimulator& mq, int core_id, int32_t task_id) {
    bump(HscbModule::RECEIVER_SIO);
    const int pair = core_id % MsgqCpuSimulator::kMaxPairs;
    const uint32_t msg_lo = static_cast<uint32_t>(task_id);
    const uint32_t msg_hi = (kMsgqSimTaskDoneTag << 16) | (static_cast<uint32_t>(core_id) & 0xFFFFu);
    trace("HSCB AICore->MsgQ task_done core=" + std::to_string(core_id) + " task=" + std::to_string(task_id));
    mq.hw_push_pair(pair, msg_lo, msg_hi, true, true);
}

}  // namespace cpu_sim
