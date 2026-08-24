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

#include <gtest/gtest.h>

#include <cstdint>

#include "scheduler/deferred_release_queue.h"

namespace {

PTO2TaskSlotState *fake_slot(uintptr_t value) { return reinterpret_cast<PTO2TaskSlotState *>(value); }

int drain_up_to(DeferredReleaseQueue &queue, int budget) {
    int released = 0;
    while (released < budget && queue.pop() != nullptr)
        released++;
    return released;
}

}  // namespace

TEST(DeferredReleaseQueue, PreservesFifoOrder) {
    DeferredReleaseQueue queue;
    ASSERT_TRUE(queue.push(fake_slot(1)));
    ASSERT_TRUE(queue.push(fake_slot(2)));
    ASSERT_TRUE(queue.push(fake_slot(3)));

    EXPECT_EQ(queue.pop(), fake_slot(1));
    EXPECT_EQ(queue.pop(), fake_slot(2));
    EXPECT_EQ(queue.pop(), fake_slot(3));
    EXPECT_TRUE(queue.empty());
    EXPECT_EQ(queue.pop(), nullptr);
}

TEST(DeferredReleaseQueue, WrapsWithoutReordering) {
    DeferredReleaseQueue queue;
    for (int32_t i = 0; i < DeferredReleaseQueue::kCapacity; i++) {
        ASSERT_TRUE(queue.push(fake_slot(static_cast<uintptr_t>(i + 1))));
    }
    EXPECT_TRUE(queue.full());
    EXPECT_FALSE(queue.push(fake_slot(999)));

    constexpr int32_t kPopped = 17;
    for (int32_t i = 0; i < kPopped; i++) {
        EXPECT_EQ(queue.pop(), fake_slot(static_cast<uintptr_t>(i + 1)));
    }
    for (int32_t i = 0; i < kPopped; i++) {
        ASSERT_TRUE(queue.push(fake_slot(static_cast<uintptr_t>(DeferredReleaseQueue::kCapacity + i + 1))));
    }

    for (int32_t i = kPopped; i < DeferredReleaseQueue::kCapacity + kPopped; i++) {
        EXPECT_EQ(queue.pop(), fake_slot(static_cast<uintptr_t>(i + 1)));
    }
    EXPECT_TRUE(queue.empty());
}

TEST(DeferredReleaseQueue, DrainsInBoundedSlices) {
    DeferredReleaseQueue queue;
    for (uintptr_t i = 1; i <= 21; i++)
        ASSERT_TRUE(queue.push(fake_slot(i)));

    EXPECT_EQ(drain_up_to(queue, 8), 8);
    EXPECT_EQ(queue.size(), 13);
    EXPECT_EQ(drain_up_to(queue, 8), 8);
    EXPECT_EQ(queue.size(), 5);
    EXPECT_EQ(drain_up_to(queue, 8), 5);
    EXPECT_TRUE(queue.empty());
}
