// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

#include "pto/pto-inst.hpp"
using namespace pto;

enum class PTOAutoSyncTailMode : int {
  kBarrierAll = 0,
  kSetWaitMte3ToSEvent0 = 1,
};

static AICORE inline void ptoas_auto_sync_tail(
    PTOAutoSyncTailMode mode = PTOAutoSyncTailMode::kBarrierAll) {
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

AICORE void scope3_incore_0_incore_0_aic(__gm__ float* v1, __gm__ bfloat16_t* v2, __gm__ float* v3, __gm__ bfloat16_t* v4) {
  unsigned v5 = 0;
  __gm__ void * v6 = nullptr;
  const int32_t v7 = 1;
  const int32_t v8 = 64;
  const int64_t v9 = 0;
  const int64_t v10 = 16384;
  const int32_t v11 = 0;
  using T = float;

  #if defined(__DAV_CUBE__)
  auto v12 = TPipe<0, Direction::DIR_BOTH, 4096, 4>(v6, v11, v11);
  Tile<TileType::Mat, bfloat16_t, 128, 64, BLayout::ColMajor, 128, 64, SLayout::RowMajor, 512, PadValue::Null> v13;
  TASSIGN(v13, v10);
  pto::Shape<1, 1, 1, 128, 64> v14 = pto::Shape<1, 1, 1, 128, 64>();
  pto::Stride<8192, 8192, 8192, 64, 1> v15 = pto::Stride<8192, 8192, 8192, 64, 1>();
  GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, 128, 64>, pto::Stride<8192, 8192, 8192, 64, 1>, pto::Layout::ND> v16 = GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, 128, 64>, pto::Stride<8192, 8192, 8192, 64, 1>, pto::Layout::ND>(v4 + (v5 + v5 * (unsigned) v8 + v5 * (unsigned) v7), v14, v15);
  TLOAD(v13, v16);
  set_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);
  Tile<TileType::Mat, bfloat16_t, 16, 128, BLayout::ColMajor, 16, 128, SLayout::RowMajor, 512, PadValue::Null> v17;
  TPOP<TPipe<0, Direction::DIR_BOTH, 4096, 4>, Tile<TileType::Mat, bfloat16_t, 16, 128, BLayout::ColMajor, 16, 128, SLayout::RowMajor, 512, PadValue::Null>, TileSplitAxis::TILE_UP_DOWN>(v12, v17);
  set_flag(PIPE_S, PIPE_MTE1, EVENT_ID0);
  Tile<TileType::Left, bfloat16_t, 16, 128, BLayout::ColMajor, 16, 128, SLayout::RowMajor, 512, PadValue::Null> v18;
  TASSIGN(v18, v9);
  wait_flag(PIPE_S, PIPE_MTE1, EVENT_ID0);
  TMOV(v18, v17);
  TFREE<TPipe<0, Direction::DIR_BOTH, 4096, 4>, TileSplitAxis::TILE_UP_DOWN>(v12);
  Tile<TileType::Right, bfloat16_t, 128, 64, BLayout::RowMajor, 128, 64, SLayout::ColMajor, 512, PadValue::Null> v19;
  TASSIGN(v19, v9);
  wait_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);
  TMOV(v19, v13);
  set_flag(PIPE_MTE1, PIPE_M, EVENT_ID0);
  Tile<TileType::Acc, float, 16, 64, BLayout::ColMajor, 16, 64, SLayout::RowMajor, 1024, PadValue::Null> v20;
  TASSIGN(v20, v9);
  wait_flag(PIPE_MTE1, PIPE_M, EVENT_ID0);
  TMATMUL(v20, v18, v19);
  set_flag(PIPE_M, PIPE_FIX, EVENT_ID0);
  wait_flag(PIPE_M, PIPE_FIX, EVENT_ID0);
  TPUSH<TPipe<0, Direction::DIR_BOTH, 4096, 4>, Tile<TileType::Acc, float, 16, 64, BLayout::ColMajor, 16, 64, SLayout::RowMajor, 1024, PadValue::Null>, TileSplitAxis::TILE_UP_DOWN>(v12, v20);
  #endif // __DAV_CUBE__

  ptoas_auto_sync_tail(PTOAutoSyncTailMode::kBarrierAll);
  return;
}

AICORE void scope3_incore_0_incore_0_aiv(__gm__ float* v1, __gm__ bfloat16_t* v2, __gm__ float* v3, __gm__ bfloat16_t* v4) {
  RoundMode v5 = RoundMode::CAST_ROUND;
  unsigned v6 = 0;
  __gm__ void * v7 = nullptr;
  const int64_t v8 = 8;
  const int32_t v9 = 1;
  const int32_t v10 = 64;
  const int32_t v11 = 128;
  const int64_t v12 = 30720;
  const int64_t v13 = 28672;
  const int64_t v14 = 24576;
  const int64_t v15 = 22528;
  const int64_t v16 = 18432;
  const int64_t v17 = 16384;
  const int32_t v18 = 0;
  using T = float;

  #if defined(__DAV_VEC__)
  set_mask_norm();
  set_vector_mask(-1, -1);
  int64_t v19 = get_subblockid();
  auto v20 = TPipe<0, Direction::DIR_BOTH, 4096, 4>(v7, v18, v18);
  int32_t v21 = (int32_t) ((int64_t) (uint64_t) ((int64_t) v19) * (uint64_t) v8);
  Tile<TileType::Vec, float, 8, 128, BLayout::RowMajor, 8, 128, SLayout::NoneBox, 512, PadValue::Null> v22;
  TASSIGN(v22, v16);
  pto::Shape<1, 1, 1, 8, 128> v23 = pto::Shape<1, 1, 1, 8, 128>();
  pto::Stride<1024, 1024, 1024, 128, 1> v24 = pto::Stride<1024, 1024, 1024, 128, 1>();
  GlobalTensor<float, pto::Shape<1, 1, 1, 8, 128>, pto::Stride<1024, 1024, 1024, 128, 1>, pto::Layout::ND> v25 = GlobalTensor<float, pto::Shape<1, 1, 1, 8, 128>, pto::Stride<1024, 1024, 1024, 128, 1>, pto::Layout::ND>(v1 + (v6 + (unsigned) v21 * (unsigned) v11 + v6 * (unsigned) v9), v23, v24);
  TLOAD(v22, v25);
  set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
  Tile<TileType::Vec, bfloat16_t, 8, 128, BLayout::RowMajor, 8, 128, SLayout::NoneBox, 512, PadValue::Null> v26;
  TASSIGN(v26, v15);
  wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
  TCVT(v26, v22, v5);
  Tile<TileType::Vec, bfloat16_t, 8, 128, BLayout::ColMajor, 8, 128, SLayout::RowMajor, 512, PadValue::Null> v27;
  TASSIGN(v27, v14);
  TMOV(v27, v26);
  set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
  wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
  TPUSH<TPipe<0, Direction::DIR_BOTH, 4096, 4>, Tile<TileType::Vec, bfloat16_t, 8, 128, BLayout::ColMajor, 8, 128, SLayout::RowMajor, 512, PadValue::Null>, TileSplitAxis::TILE_UP_DOWN>(v20, v27);
  Tile<TileType::Vec, float, 8, 64, BLayout::RowMajor, 8, 64, SLayout::NoneBox, 512, PadValue::Null> v28;
  TPOP<TPipe<0, Direction::DIR_BOTH, 4096, 4>, Tile<TileType::Vec, float, 8, 64, BLayout::RowMajor, 8, 64, SLayout::NoneBox, 512, PadValue::Null>, TileSplitAxis::TILE_UP_DOWN>(v20, v28);
  set_flag(PIPE_S, PIPE_V, EVENT_ID0);
  Tile<TileType::Vec, bfloat16_t, 8, 64, BLayout::RowMajor, 8, 64, SLayout::NoneBox, 512, PadValue::Null> v29;
  TASSIGN(v29, v12);
  pto::Shape<1, 1, 1, 8, 64> v30 = pto::Shape<1, 1, 1, 8, 64>();
  pto::Stride<512, 512, 512, 64, 1> v31 = pto::Stride<512, 512, 512, 64, 1>();
  GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, 8, 64>, pto::Stride<512, 512, 512, 64, 1>, pto::Layout::ND> v32 = GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, 8, 64>, pto::Stride<512, 512, 512, 64, 1>, pto::Layout::ND>(v2 + (v6 + (unsigned) v21 * (unsigned) v10 + v6 * (unsigned) v9), v30, v31);
  TLOAD(v29, v32);
  set_flag(PIPE_MTE2, PIPE_V, EVENT_ID1);
  Tile<TileType::Vec, float, 8, 64, BLayout::RowMajor, 8, 64, SLayout::NoneBox, 512, PadValue::Null> v33;
  TASSIGN(v33, v13);
  wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID1);
  TCVT(v33, v29, v5);
  Tile<TileType::Vec, float, 8, 64, BLayout::RowMajor, 8, 64, SLayout::NoneBox, 512, PadValue::Null> v34;
  TASSIGN(v34, v17);
  wait_flag(PIPE_S, PIPE_V, EVENT_ID0);
  TADD(v34, v28, v33);
  set_flag(PIPE_V, PIPE_MTE3, EVENT_ID1);
  TFREE<TPipe<0, Direction::DIR_BOTH, 4096, 4>, TileSplitAxis::TILE_UP_DOWN>(v20);
  pto::Shape<1, 1, 1, 8, 64> v35 = pto::Shape<1, 1, 1, 8, 64>();
  pto::Stride<512, 512, 512, 64, 1> v36 = pto::Stride<512, 512, 512, 64, 1>();
  GlobalTensor<float, pto::Shape<1, 1, 1, 8, 64>, pto::Stride<512, 512, 512, 64, 1>, pto::Layout::ND> v37 = GlobalTensor<float, pto::Shape<1, 1, 1, 8, 64>, pto::Stride<512, 512, 512, 64, 1>, pto::Layout::ND>(v3 + (v6 + (unsigned) v21 * (unsigned) v10 + v6 * (unsigned) v9), v35, v36);
  wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID1);
  TSTORE(v37, v34);
  #endif // __DAV_VEC__

  ptoas_auto_sync_tail(PTOAutoSyncTailMode::kBarrierAll);
  return;
}
