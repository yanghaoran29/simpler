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

#pragma once

#include <cstdint>

struct PTO2TaskSlotState;

// Per-scheduler-thread FIFO for task slots whose completion has resolved their
// consumers but whose producer references have not been released yet. Keeping
// the oldest slot at the head prevents a sustained completion stream from
// starving ring-head reclamation.
struct DeferredReleaseQueue {
    static constexpr int32_t kCapacity = 256;

    bool empty() const { return count_ == 0; }
    bool full() const { return count_ == kCapacity; }
    int32_t size() const { return count_; }
    int32_t free_slots() const { return kCapacity - count_; }

    bool push(PTO2TaskSlotState *slot_state) {
        if (full()) return false;
        int32_t tail = head_ + count_;
        if (tail >= kCapacity) tail -= kCapacity;
        entries_[tail] = slot_state;
        count_++;
        return true;
    }

    PTO2TaskSlotState *pop() {
        if (empty()) return nullptr;
        PTO2TaskSlotState *slot_state = entries_[head_];
        head_++;
        if (head_ == kCapacity) head_ = 0;
        count_--;
        return slot_state;
    }

private:
    PTO2TaskSlotState *entries_[kCapacity]{};
    int32_t head_{0};
    int32_t count_{0};
};
