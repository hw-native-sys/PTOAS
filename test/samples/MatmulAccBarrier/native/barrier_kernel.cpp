// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

#include <cstdint>
#include <pto/pto-inst.hpp>
#include <pto/common/constants.hpp>

using namespace pto;

#if defined(__CCE_AICORE__)
namespace {
constexpr int kRowsPerTile = 16;
constexpr int kOutputCols = 24;
constexpr int kOutputStride = 32;
constexpr int kFullK = 20480;
constexpr int kKTile = 256;
constexpr int kKSplits = 4;
constexpr int kKPerSplit = kFullK / kKSplits;
constexpr int kKSteps = kKPerSplit / kKTile;

using AMat = Tile<TileType::Mat, float, kRowsPerTile, kKTile,
    BLayout::ColMajor, -1, -1, SLayout::RowMajor, 512>;
using WMat = Tile<TileType::Mat, float, kOutputStride, kKTile,
    BLayout::ColMajor, -1, -1, SLayout::RowMajor, 512>;
using WTranspose = Tile<TileType::Mat, float, kKTile, kOutputStride,
    BLayout::RowMajor, -1, -1, SLayout::ColMajor, 512>;
using Left = Tile<TileType::Left, float, kRowsPerTile, kKTile,
    BLayout::ColMajor, -1, -1, SLayout::RowMajor, 512>;
using Right = Tile<TileType::Right, float, kKTile, kOutputStride,
    BLayout::RowMajor, -1, -1, SLayout::ColMajor, 512>;
using Acc = Tile<TileType::Acc, float, kRowsPerTile, kOutputStride,
    BLayout::ColMajor, -1, -1, SLayout::RowMajor, 1024>;
using AShape = pto::Shape<1, 1, 1, -1, kKTile>;
using AStride = pto::Stride<-1, -1, -1, kFullK, 1>;
using AGlobal = GlobalTensor<float, AShape, AStride>;
using WGlobal = GlobalTensor<float, pto::Shape<1, 1, 1, kOutputCols, kKTile>,
    pto::Stride<kOutputCols * kFullK, kOutputCols * kFullK, kOutputCols * kFullK, kFullK, 1>>;
using CGlobal = GlobalTensor<float, pto::Shape<1, 1, 1, kRowsPerTile, kOutputStride>,
    pto::Stride<kRowsPerTile * kOutputStride, kRowsPerTile * kOutputStride,
        kRowsPerTile * kOutputStride, kOutputStride, 1>>;

template <bool Serialize>
AICORE void MatmulAccBarrier(
    __gm__ float *a, __gm__ float *w, __gm__ float *out, int rows, int paddedRows)
{
#if defined(__DAV_CUBE__)
    const int task = get_block_idx();
    const int rowStart = (task / kKSplits) * kRowsPerTile;
    const int split = task % kKSplits;
    if (rowStart >= rows) return;
    const int rowsRemaining = rows - rowStart;
    const int validRows = rowsRemaining < kRowsPerTile ? rowsRemaining : kRowsPerTile;
    const int kBase = split * kKPerSplit;
    a += rowStart * kFullK;
    w += kBase;

    AMat aEven(validRows, kKTile), aOdd(validRows, kKTile);
    WMat wEven(kOutputCols, kKTile), wOdd(kOutputCols, kKTile);
    WTranspose wtEven(kKTile, kOutputCols), wtOdd(kKTile, kOutputCols);
    Left leftEven(validRows, kKTile), leftOdd(validRows, kKTile);
    Right rightEven(kKTile, kOutputCols), rightOdd(kKTile, kOutputCols);
    Acc acc(kRowsPerTile, kOutputStride);
    TASSIGN(aEven, 0);
    TASSIGN(wEven, 16384);
    TASSIGN(wtEven, 16384);
    TASSIGN(aOdd, 49152);
    TASSIGN(wOdd, 65536);
    TASSIGN(wtOdd, 65536);
    TASSIGN(leftEven, 16384);
    TASSIGN(leftOdd, 0);
    TASSIGN(rightEven, 32768);
    TASSIGN(rightOdd, 0);
    TASSIGN(acc, 0);

    const int aSpan = validRows * kFullK;
    AShape aShape(1, 1, 1, validRows, kKTile);
    AStride aStride(aSpan, aSpan, aSpan, kFullK, 1);

    set_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID0);
    set_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID1);
    set_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID2);
    set_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID3);
    set_flag(PIPE_M, PIPE_MTE1, EVENT_ID0);
    set_flag(PIPE_M, PIPE_MTE1, EVENT_ID1);

    for (int kb = 0; kb < kKSteps; kb += 2) {
        const int evenK = kb * kKTile;
        const int oddK = evenK + kKTile;
        AGlobal aEvenGlobal(a + kBase + evenK, aShape, aStride);
        WGlobal wEvenGlobal(w + evenK);
        AGlobal aOddGlobal(a + kBase + oddK, aShape, aStride);
        WGlobal wOddGlobal(w + oddK);

        wait_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID0);
        TLOAD(aEven, aEvenGlobal);
        set_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);
        wait_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID1);
        TLOAD(wEven, wEvenGlobal);
        set_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID1);
        wait_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID2);
        TLOAD(aOdd, aOddGlobal);
        set_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID2);
        wait_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID3);
        TLOAD(wOdd, wOddGlobal);
        set_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID3);

        wait_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);
        wait_flag(PIPE_M, PIPE_MTE1, EVENT_ID0);
        TMOV(leftEven, aEven);
        set_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID0);
        wait_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID1);
        TMOV(rightEven, wtEven);
        set_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID1);
        set_flag(PIPE_MTE1, PIPE_M, EVENT_ID0);
        wait_flag(PIPE_MTE1, PIPE_M, EVENT_ID0);
        if constexpr (Serialize) pipe_barrier(PIPE_M);
        if (kb == 0) TMATMUL(acc, leftEven, rightEven);
        else TMATMUL_ACC(acc, acc, leftEven, rightEven);
        set_flag(PIPE_M, PIPE_MTE1, EVENT_ID0);

        wait_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID2);
        wait_flag(PIPE_M, PIPE_MTE1, EVENT_ID1);
        TMOV(leftOdd, aOdd);
        set_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID2);
        wait_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID3);
        TMOV(rightOdd, wtOdd);
        set_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID3);
        set_flag(PIPE_MTE1, PIPE_M, EVENT_ID1);
        wait_flag(PIPE_MTE1, PIPE_M, EVENT_ID1);
        if constexpr (Serialize) pipe_barrier(PIPE_M);
        TMATMUL_ACC(acc, acc, leftOdd, rightOdd);
        set_flag(PIPE_M, PIPE_MTE1, EVENT_ID1);
    }

    CGlobal result(out + (split * paddedRows + rowStart) * kOutputStride);
    set_flag(PIPE_M, PIPE_FIX, EVENT_ID0);
    wait_flag(PIPE_M, PIPE_FIX, EVENT_ID0);
    TSTORE(result, acc);
    wait_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID0);
    wait_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID1);
    wait_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID2);
    wait_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID3);
    wait_flag(PIPE_M, PIPE_MTE1, EVENT_ID0);
    wait_flag(PIPE_M, PIPE_MTE1, EVENT_ID1);
    pipe_barrier(PIPE_ALL);
#endif
}
} // namespace
#endif

extern "C" __global__ AICORE void BarrierBenchmarkKernel(
    __gm__ float *a, __gm__ float *w, __gm__ float *out, int rows, int paddedRows, int serialize)
{
#if defined(__CCE_AICORE__)
    if (serialize) MatmulAccBarrier<true>(a, w, out, rows, paddedRows);
    else MatmulAccBarrier<false>(a, w, out, rows, paddedRows);
#endif
}

// padded_rows is ceil(rows / 16) * 16; out has shape [4 * padded_rows, 32].
extern "C" void LaunchBarrierBenchmark(
    float *a, float *w, float *out, int rows, int padded_rows, int serialize, uint32_t blocks, void *stream)
{
    BarrierBenchmarkKernel<<<blocks, nullptr, stream>>>(a, w, out, rows, padded_rows, serialize);
}
