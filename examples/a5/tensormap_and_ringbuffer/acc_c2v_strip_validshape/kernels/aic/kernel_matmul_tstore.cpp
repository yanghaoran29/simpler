/*
 * CONTROL: Acc float matmul → TSTORE Acc to GM (no C2V). Isolates matmul golden match.
 */
#include <cstdint>
#include <pto/pto-inst.hpp>
#include "tensor.h"

using namespace pto;

#ifndef __gm__
#define __gm__
#endif
#ifndef __aicore__
#define __aicore__ [aicore]
#endif
#include "intrinsic.h"

constexpr int M = 128;
constexpr int N = 256;
constexpr int K = 32;

extern "C" __aicore__ void kernel_entry(__gm__ int64_t *args) {
    __gm__ Tensor *a_tensor = reinterpret_cast<__gm__ Tensor *>(args[0]);
    __gm__ Tensor *b_tensor = reinterpret_cast<__gm__ Tensor *>(args[1]);
    __gm__ Tensor *c_tensor = reinterpret_cast<__gm__ Tensor *>(args[2]);

    __gm__ float *a_ptr = reinterpret_cast<__gm__ float *>(a_tensor->buffer.addr) + a_tensor->start_offset;
    __gm__ float *b_ptr = reinterpret_cast<__gm__ float *>(b_tensor->buffer.addr) + b_tensor->start_offset;
    __gm__ float *c_ptr = reinterpret_cast<__gm__ float *>(c_tensor->buffer.addr) + c_tensor->start_offset;

    using GlobalA = GlobalTensor<float, Shape<1, 1, 1, M, K>, pto::Stride<M * K, M * K, M * K, K, 1>>;
    using GlobalB = GlobalTensor<float, Shape<1, 1, 1, K, N>, pto::Stride<K * N, K * N, K * N, N, 1>>;
    using GlobalC = GlobalTensor<float, Shape<1, 1, 1, M, N>, pto::Stride<M * N, M * N, M * N, N, 1>>;
    GlobalA aGlobal(a_ptr);
    GlobalB bGlobal(b_ptr);
    GlobalC cGlobal(c_ptr);

    using TileMatA = Tile<TileType::Mat, float, M, K, BLayout::ColMajor, M, K, SLayout::RowMajor, 512>;
    using TileMatB = Tile<TileType::Mat, float, K, N, BLayout::ColMajor, K, N, SLayout::RowMajor, 512>;
    using LeftT = TileLeft<float, M, K, M, K>;
    using RightT = TileRight<float, K, N, K, N>;
    using AccT = TileAcc<float, M, N, M, N>;

    TileMatA aMat;
    TileMatB bMat;
    TASSIGN(aMat, 0x0);
    TASSIGN(bMat, 0x20000);
    LeftT aL0;
    RightT bL0;
    AccT acc;
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
    TSTORE(cGlobal, acc);
    set_flag(PIPE_FIX, PIPE_S, EVENT_ID7);
    wait_flag(PIPE_FIX, PIPE_S, EVENT_ID7);
}
