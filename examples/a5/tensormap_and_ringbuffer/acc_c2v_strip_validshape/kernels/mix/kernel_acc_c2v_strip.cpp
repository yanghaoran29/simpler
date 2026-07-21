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
/**
 * A5 Acc ValidShape strip C2V repro — matches ISSUE_A5_ACC_VALIDSHAPE_C2V_SRCSTRIDE §样例.
 *
 * AIC: Acc[M,N] = A@B; then M/H × TPUSH Acc[M,N]+Valid(H,N) @ addr=row*64 (TILE_NO_SPLIT).
 * AIV0: TPOP Vec[H,N] ND strip → TSTORE GM (AIV1 idle for C2V).
 *
 * USE_STRIP_C2V=0: one full Acc TPUSH (Valid≡Rows) control path.
 */

#include <cstdint>
#include <pto/pto-inst.hpp>
#include <pto/common/fifo.hpp>

#include "tensor.h"

using pto::BLayout;
using pto::Direction;
using pto::GlobalTensor;
using pto::Shape;
using pto::SLayout;
using pto::Tile;
using pto::TileAcc;
using pto::TileLeft;
using pto::TileRight;
using pto::TileSplitAxis;
using pto::TileType;
using pto::TPipe;

#ifndef __gm__
#define __gm__
#endif

#ifndef __aicore__
#define __aicore__ [aicore]
#endif

#include "intrinsic.h"

#ifdef __DAV_CUBE__
constexpr bool DAV_CUBE = true;
#else
constexpr bool DAV_CUBE = false;
#endif

#ifdef __DAV_VEC__
constexpr bool DAV_VEC = true;
#else
constexpr bool DAV_VEC = false;
#endif

#ifndef USE_STRIP_C2V
#define USE_STRIP_C2V 1
#endif

constexpr int M = 128;
constexpr int N = 256;
constexpr int H = 16;
constexpr int K = 32;

constexpr uint64_t kAccRowByteStride = 64;

constexpr uint16_t PP_FLAG_ID = 0;
constexpr uint8_t PP_FIFO_DEPTH = 2;

#if USE_STRIP_C2V
constexpr int PUSH_ROWS = H;
constexpr int kNumPush = M / H;
#else
constexpr int PUSH_ROWS = M;
constexpr int kNumPush = 1;
#endif

constexpr uint32_t kSlotBytes = static_cast<uint32_t>(PUSH_ROWS * N * sizeof(float));

using AccFullT = TileAcc<float, M, N, M, N>;
using AccPushT = Tile<TileType::Acc, float, M, N, BLayout::ColMajor, PUSH_ROWS, N, SLayout::RowMajor, 1024,
                      pto::PadValue::Null, pto::CompactMode::Null>;
using VecPushT = Tile<TileType::Vec, float, PUSH_ROWS, N, BLayout::RowMajor, PUSH_ROWS, N, SLayout::NoneBox, 512,
                      pto::PadValue::Null, pto::CompactMode::Null>;
// IsNoSplit=true matches TILE_NO_SPLIT (AIV0 only), same as qr_proj full-Acc hand patch.
using PipeT = TPipe<PP_FLAG_ID, Direction::DIR_C2V, kSlotBytes, PP_FIFO_DEPTH, 2, true>;

extern "C" __aicore__ void kernel_entry(__gm__ int64_t *args) {
    __gm__ Tensor *a_tensor = reinterpret_cast<__gm__ Tensor *>(args[0]);
    __gm__ Tensor *b_tensor = reinterpret_cast<__gm__ Tensor *>(args[1]);
    __gm__ Tensor *c_tensor = reinterpret_cast<__gm__ Tensor *>(args[2]);

    PipeT pipe(nullptr, 0U, 0U);

    if constexpr (DAV_CUBE) {
        __gm__ float *a_ptr =
            reinterpret_cast<__gm__ float *>(a_tensor->buffer.addr) + a_tensor->start_offset;
        __gm__ float *b_ptr =
            reinterpret_cast<__gm__ float *>(b_tensor->buffer.addr) + b_tensor->start_offset;

        using GlobalA = GlobalTensor<float, Shape<1, 1, 1, M, K>, pto::Stride<M * K, M * K, M * K, K, 1>>;
        using GlobalB = GlobalTensor<float, Shape<1, 1, 1, K, N>, pto::Stride<K * N, K * N, K * N, N, 1>>;
        GlobalA aGlobal(a_ptr);
        GlobalB bGlobal(b_ptr);

        using TileMatA = Tile<TileType::Mat, float, M, K, BLayout::ColMajor, M, K, SLayout::RowMajor, 512>;
        using TileMatB = Tile<TileType::Mat, float, K, N, BLayout::ColMajor, K, N, SLayout::RowMajor, 512>;
        using LeftT = TileLeft<float, M, K, M, K>;
        using RightT = TileRight<float, K, N, K, N>;

        TileMatA aMat;
        TileMatB bMat;
        TASSIGN(aMat, 0x0);
        TASSIGN(bMat, 0x20000);

        LeftT aL0;
        RightT bL0;
        AccFullT acc;
        TASSIGN(aL0, 0x0);
        TASSIGN(bL0, 0x0);
        TASSIGN(acc, 0x0);

        TLOAD(aMat, aGlobal);
        TLOAD(bMat, bGlobal);

        set_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);
        wait_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);

        TMOV(aL0, aMat);
        TMOV(bL0, bMat);

        set_flag(PIPE_MTE1, PIPE_M, EVENT_ID0);
        wait_flag(PIPE_MTE1, PIPE_M, EVENT_ID0);

        TMATMUL(acc, aL0, bL0);

        set_flag(PIPE_M, PIPE_FIX, EVENT_ID0);
        wait_flag(PIPE_M, PIPE_FIX, EVENT_ID0);

#if USE_STRIP_C2V
        for (int row = 0; row < M; row += H) {
            AccPushT strip;
            TASSIGN(strip, static_cast<uint64_t>(row) * kAccRowByteStride);
            TPUSH<PipeT, AccPushT, TileSplitAxis::TILE_NO_SPLIT>(pipe, strip);
        }
#else
        {
            AccPushT full;
            TASSIGN(full, 0x0);
            TPUSH<PipeT, AccPushT, TileSplitAxis::TILE_NO_SPLIT>(pipe, full);
        }
#endif

        set_flag(PIPE_FIX, PIPE_S, EVENT_ID7);
        wait_flag(PIPE_FIX, PIPE_S, EVENT_ID7);
    }

    if constexpr (DAV_VEC) {
        if (get_sub_block_id(args) != 0) {
            return;
        }

        __gm__ float *c_ptr =
            reinterpret_cast<__gm__ float *>(c_tensor->buffer.addr) + c_tensor->start_offset;

        constexpr uint64_t kScratch = static_cast<uint64_t>(PP_FIFO_DEPTH) * kSlotBytes;
        VecPushT vec;
        TASSIGN(vec, kScratch);

        for (int s = 0; s < kNumPush; ++s) {
            TPOP<PipeT, VecPushT, TileSplitAxis::TILE_NO_SPLIT>(pipe, vec);

            int row0 = s * PUSH_ROWS;
            using GlobalC = GlobalTensor<float, Shape<1, 1, 1, PUSH_ROWS, N>,
                                         pto::Stride<PUSH_ROWS * N, PUSH_ROWS * N, PUSH_ROWS * N, N, 1>>;
            GlobalC cGlobal(c_ptr + static_cast<size_t>(row0) * N);

            set_flag(PIPE_S, PIPE_MTE3, EVENT_ID0);
            wait_flag(PIPE_S, PIPE_MTE3, EVENT_ID0);
            TSTORE(cGlobal, vec);
            TFREE<PipeT, TileSplitAxis::TILE_NO_SPLIT>(pipe);

            set_flag(PIPE_MTE3, PIPE_S, EVENT_ID1);
            wait_flag(PIPE_MTE3, PIPE_S, EVENT_ID1);
        }
    }
}
