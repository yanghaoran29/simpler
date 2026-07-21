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
 * A5 Acc ValidShape strip C2V repro (self-compare).
 *
 * AIC: Acc = A@B once, then
 *   1) full Acc TPUSH (Valid≡Rows) → AIV stores C_full
 *   2) 8× Acc+Valid(H) TPUSH @ addr=row*64 → AIV stores C_strip
 *
 * Host golden: C_strip == C_full.
 *   - unfixed TMovCcToUb (srcStride=validRow): FAIL
 *   - fixed   (srcStride=Rows):                 PASS
 *
 * args[0]=A, args[1]=B, args[2]=C_full, args[3]=C_strip
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

constexpr int M = 32;
constexpr int N = 128;
constexpr int H = 16;
constexpr int K = 32;
constexpr uint64_t kAccRowByteStride = 64;
constexpr uint16_t PP_FLAG_ID = 0;
constexpr uint8_t PP_FIFO_DEPTH = 1;

constexpr uint32_t kFullSlot = static_cast<uint32_t>(M * N * sizeof(float));
constexpr uint32_t kStripSlot = static_cast<uint32_t>(H * N * sizeof(float));

using AccFullT = TileAcc<float, M, N, M, N>;
using AccWinFullT = Tile<TileType::Acc, float, M, N, BLayout::ColMajor, M, N, SLayout::RowMajor, 1024,
                         pto::PadValue::Null, pto::CompactMode::Null>;
using AccWinStripT = Tile<TileType::Acc, float, M, N, BLayout::ColMajor, H, N, SLayout::RowMajor, 1024,
                          pto::PadValue::Null, pto::CompactMode::Null>;
using VecFullT = Tile<TileType::Vec, float, M, N, BLayout::RowMajor, M, N, SLayout::NoneBox, 512,
                      pto::PadValue::Null, pto::CompactMode::Null>;
using VecStripT = Tile<TileType::Vec, float, H, N, BLayout::RowMajor, H, N, SLayout::NoneBox, 512,
                       pto::PadValue::Null, pto::CompactMode::Null>;

using PipeFullT = TPipe<PP_FLAG_ID, Direction::DIR_C2V, kFullSlot, PP_FIFO_DEPTH, 2, true>;
using PipeStripT = TPipe<PP_FLAG_ID + 1, Direction::DIR_C2V, kStripSlot, PP_FIFO_DEPTH, 2, true>;

extern "C" __aicore__ void kernel_entry(__gm__ int64_t *args) {
    __gm__ Tensor *a_tensor = reinterpret_cast<__gm__ Tensor *>(args[0]);
    __gm__ Tensor *b_tensor = reinterpret_cast<__gm__ Tensor *>(args[1]);
    __gm__ Tensor *c_full_tensor = reinterpret_cast<__gm__ Tensor *>(args[2]);
    __gm__ Tensor *c_strip_tensor = reinterpret_cast<__gm__ Tensor *>(args[3]);

    PipeFullT pipeFull(nullptr, 0U, 0U);
    PipeStripT pipeStrip(nullptr, 0U, 0U);

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

        // (1) full Acc C2V — Valid≡Rows, always correct for srcStride either formula
        {
            AccWinFullT full;
            TASSIGN(full, 0x0);
            TPUSH<PipeFullT, AccWinFullT, TileSplitAxis::TILE_NO_SPLIT>(pipeFull, full);
        }

        // (2) strip Acc C2V — Valid(H) windows (the buggy pattern)
        for (int row = 0; row < M; row += H) {
            AccWinStripT strip;
            TASSIGN(strip, static_cast<uint64_t>(row) * kAccRowByteStride);
            TPUSH<PipeStripT, AccWinStripT, TileSplitAxis::TILE_NO_SPLIT>(pipeStrip, strip);
        }

        set_flag(PIPE_FIX, PIPE_S, EVENT_ID7);
        wait_flag(PIPE_FIX, PIPE_S, EVENT_ID7);
    }

    if constexpr (DAV_VEC) {
        if (get_sub_block_id(args) != 0) {
            return;
        }
        __gm__ float *c_full =
            reinterpret_cast<__gm__ float *>(c_full_tensor->buffer.addr) + c_full_tensor->start_offset;
        __gm__ float *c_strip =
            reinterpret_cast<__gm__ float *>(c_strip_tensor->buffer.addr) + c_strip_tensor->start_offset;

        // UB layout: full fifo + strip fifo + scratch
        constexpr uint64_t kFullFifo = static_cast<uint64_t>(PP_FIFO_DEPTH) * kFullSlot;
        constexpr uint64_t kStripFifo = static_cast<uint64_t>(PP_FIFO_DEPTH) * kStripSlot;
        VecFullT vFull;
        VecStripT vStrip;
        TASSIGN(vFull, kFullFifo + kStripFifo);
        TASSIGN(vStrip, kFullFifo + kStripFifo + kFullSlot);

        // Pop full
        TPOP<PipeFullT, VecFullT, TileSplitAxis::TILE_NO_SPLIT>(pipeFull, vFull);
        set_flag(PIPE_S, PIPE_MTE3, EVENT_ID0);
        wait_flag(PIPE_S, PIPE_MTE3, EVENT_ID0);
        using GlobalFull = GlobalTensor<float, Shape<1, 1, 1, M, N>, pto::Stride<M * N, M * N, M * N, N, 1>>;
        GlobalFull gFull(c_full);
        TSTORE(gFull, vFull);
        TFREE<PipeFullT, TileSplitAxis::TILE_NO_SPLIT>(pipeFull);
        set_flag(PIPE_MTE3, PIPE_S, EVENT_ID1);
        wait_flag(PIPE_MTE3, PIPE_S, EVENT_ID1);

        // Pop strips
        for (int row = 0; row < M; row += H) {
            TPOP<PipeStripT, VecStripT, TileSplitAxis::TILE_NO_SPLIT>(pipeStrip, vStrip);
            set_flag(PIPE_S, PIPE_MTE3, EVENT_ID2);
            wait_flag(PIPE_S, PIPE_MTE3, EVENT_ID2);
            using GlobalStrip =
                GlobalTensor<float, Shape<1, 1, 1, H, N>, pto::Stride<H * N, H * N, H * N, N, 1>>;
            GlobalStrip gStrip(c_strip + static_cast<size_t>(row) * N);
            TSTORE(gStrip, vStrip);
            TFREE<PipeStripT, TileSplitAxis::TILE_NO_SPLIT>(pipeStrip);
            set_flag(PIPE_MTE3, PIPE_S, EVENT_ID3);
            wait_flag(PIPE_MTE3, PIPE_S, EVENT_ID3);
        }
    }
}
