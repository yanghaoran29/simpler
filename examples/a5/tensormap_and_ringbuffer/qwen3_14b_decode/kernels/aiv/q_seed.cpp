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
// Kernel Function: q_seed

#include <cstdint>

#ifndef __gm__
#define __gm__
#endif

#ifndef __aicore__
#if defined(__CPU_SIM)
#define __aicore__
#else
#define __aicore__ [aicore]
#endif
#endif

#include <pto/pto-inst.hpp>
#include "tensor.h"

using namespace pto;

// --- ptoas-generated code ---

enum class PTOAutoSyncTailMode : int {
    kBarrierAll = 0,
    kSetWaitMte3ToSEvent0 = 1,
};

static __aicore__ inline void ptoas_auto_sync_tail(PTOAutoSyncTailMode mode = PTOAutoSyncTailMode::kBarrierAll) {
    switch (mode) {
    case PTOAutoSyncTailMode::kSetWaitMte3ToSEvent0:
        set_flag(PIPE_MTE3, PIPE_S, EVENT_ID0);
        wait_flag(PIPE_MTE3, PIPE_S, EVENT_ID0);
        break;
    case PTOAutoSyncTailMode::kBarrierAll:
    default:
        pipe_barrier(PIPE_ALL);
        break;
    }
}

static __aicore__ void q_seed(__gm__ float *v1, int32_t half_idx) {
    const float zero = 0.0f;
    const int64_t tile_width = 512;
    const int64_t row_stride = 5120;
    const int64_t rows = 16;
    const int64_t first_tile = static_cast<int64_t>(half_idx) * 5;
    const int64_t end_tile = first_tile + 5;
    using T = float;

#if defined(__DAV_VEC__)
    set_mask_norm();
    set_vector_mask(-1, -1);
    set_flag(PIPE_MTE3, PIPE_V, EVENT_ID0);
    for (int64_t tile_idx = first_tile; tile_idx < end_tile; ++tile_idx) {
        Tile<
            TileType::Vec, float, 16, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
            CompactMode::Null>
            zero_tile = Tile<
                TileType::Vec, float, 16, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null,
                CompactMode::Null>(rows, tile_width);
        uint64_t ub_offset = 0;
        TASSIGN(zero_tile, ub_offset);
        wait_flag(PIPE_MTE3, PIPE_V, EVENT_ID0);
        TEXPANDS(zero_tile, zero);
        set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
        int64_t column_offset = tile_idx * tile_width;
        pto::Shape<1, 1, 1, 16, 512> shape = pto::Shape<1, 1, 1, 16, 512>();
        pto::Stride<81920, 81920, 81920, 5120, 1> stride =
            pto::Stride<81920, 81920, 81920, 5120, 1>();
        GlobalTensor<float, pto::Shape<1, 1, 1, 16, 512>, pto::Stride<81920, 81920, 81920, 5120, 1>, pto::Layout::ND>
            destination = GlobalTensor<
                float, pto::Shape<1, 1, 1, 16, 512>, pto::Stride<81920, 81920, 81920, 5120, 1>, pto::Layout::ND>(
                v1 + column_offset, shape, stride
            );
        wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
        pipe_barrier(PIPE_MTE3);
        TSTORE(destination, zero_tile);
        set_flag(PIPE_MTE3, PIPE_V, EVENT_ID0);
    }
    wait_flag(PIPE_MTE3, PIPE_V, EVENT_ID0);
#endif  // __DAV_VEC__

    ptoas_auto_sync_tail(PTOAutoSyncTailMode::kBarrierAll);
    return;
}
// --- Kernel entry point ---
extern "C" __aicore__ __attribute__((always_inline)) void kernel_entry(__gm__ int64_t *args) {
    // Unpack tensor: q_proj_inline139__ssa_v0
    __gm__ TaskTensor *q_proj_inline139__ssa_v0_tensor = reinterpret_cast<__gm__ TaskTensor *>(args[0]);
    __gm__ float *q_proj_inline139__ssa_v0 =
        reinterpret_cast<__gm__ float *>(q_proj_inline139__ssa_v0_tensor->buffer.addr) +
        q_proj_inline139__ssa_v0_tensor->start_offset;

    union {
        int32_t value;
        uint64_t raw;
    } half_idx_conv;
    half_idx_conv.raw = args[1];

    // Forward to ptoas-generated function
    q_seed(q_proj_inline139__ssa_v0, half_idx_conv.value);
}
