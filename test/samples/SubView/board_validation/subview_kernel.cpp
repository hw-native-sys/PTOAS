// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

#ifndef __VEC_SCOPE__
#define __VEC_SCOPE__
#endif

#if defined(__CCE_AICORE__) && defined(__NPU_ARCH__) && (__NPU_ARCH__ == 2201)
typedef struct { unsigned char v; } hifloat8_t;
typedef struct { unsigned char v; } float8_e4m3_t;
typedef struct { unsigned char v; } float8_e5m2_t;
typedef struct { unsigned char v; } float8_e8m0_t;
typedef struct { unsigned char v; } float4_e1m2x2_t;
typedef struct { unsigned char v; } float4_e2m1x2_t;
#endif
#include <stdint.h>

#if defined(__CCE_AICORE__) && defined(PTOAS_ENABLE_CCE_PRINT)
#include <ccelib/print/print.h>
#endif
#include <pto/pto-inst.hpp>
#include <pto/common/constants.hpp>

#if !defined(__CCE_AICORE__) && !defined(TMRGSORT_HPP)
namespace pto {
struct MrgSortExecutedNumList {
    uint16_t mrgSortList0;
    uint16_t mrgSortList1;
    uint16_t mrgSortList2;
    uint16_t mrgSortList3;
};
} // namespace pto
#endif
#ifndef __CPU_SIM
#include "acl/acl.h"
#endif

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

__global__ AICORE void subview_split4(__gm__ float* v1, __gm__ float* v2, __gm__ float* v3, __gm__ float* v4, __gm__ float* v5) {
  unsigned v6 = 8;
  unsigned v7 = 16;
  unsigned v8 = 1;
  unsigned v9 = 0;
  int64_t v10 = 0;
  int32_t v11 = 16;
  int32_t v12 = 8;
  using T = float;
  Tile<TileType::Vec, float, 16, 16, BLayout::RowMajor, 16, 16, SLayout::NoneBox, 512, PadValue::Null> v13;
  TASSIGN(v13, v10);
  using GTShape_5736857712 = pto::Shape<1, 1, 1, 16, 16>;
  using GTStride_5736857712 = pto::Stride<256, 256, 256, 16, 1>;
  constexpr pto::Layout GT_5736857712_layout = pto::Layout::ND;
  GTShape_5736857712 v14 = GTShape_5736857712();
  GTStride_5736857712 v15 = GTStride_5736857712();
  using GT_5736857712 = GlobalTensor<float, GTShape_5736857712, GTStride_5736857712, GT_5736857712_layout>;
  GT_5736857712 v16 = GT_5736857712(v1, v14, v15);
  TLOAD(v13, v16);
  set_flag(PIPE_MTE2, PIPE_MTE3, EVENT_ID0);
  __ubuf__ float* v17 = v13.data();
  Tile<TileType::Vec, float, 16, 16, BLayout::RowMajor, 8, 8, SLayout::NoneBox, 512, PadValue::Null> v18;
  uint64_t v19 = reinterpret_cast<uint64_t>((__ubuf__ float*) (v17 + (v9 + v9 * v7) + v9 * v8));
  TASSIGN(v18, v19);
  __ubuf__ float* v20 = v13.data();
  Tile<TileType::Vec, float, 16, 16, BLayout::RowMajor, 8, 8, SLayout::NoneBox, 512, PadValue::Null> v21;
  uint64_t v22 = reinterpret_cast<uint64_t>((__ubuf__ float*) (v20 + (v9 + v9 * v7) + v6 * v8));
  TASSIGN(v21, v22);
  __ubuf__ float* v23 = v13.data();
  Tile<TileType::Vec, float, 16, 16, BLayout::RowMajor, 8, 8, SLayout::NoneBox, 512, PadValue::Null> v24;
  uint64_t v25 = reinterpret_cast<uint64_t>((__ubuf__ float*) (v23 + (v9 + v6 * v7) + v9 * v8));
  TASSIGN(v24, v25);
  __ubuf__ float* v26 = v13.data();
  Tile<TileType::Vec, float, 16, 16, BLayout::RowMajor, 8, 8, SLayout::NoneBox, 512, PadValue::Null> v27;
  uint64_t v28 = reinterpret_cast<uint64_t>((__ubuf__ float*) (v26 + (v9 + v6 * v7) + v6 * v8));
  TASSIGN(v27, v28);
  wait_flag(PIPE_MTE2, PIPE_MTE3, EVENT_ID0);
  using GTShape_5736826112 = pto::Shape<1, 1, 1, 8, 8>;
  using GTStride_5736826112 = pto::Stride<64, 64, 64, 8, 1>;
  constexpr pto::Layout GT_5736826112_layout = pto::Layout::ND;
  GTShape_5736826112 v29 = GTShape_5736826112();
  GTStride_5736826112 v30 = GTStride_5736826112();
  using GT_5736826112 = GlobalTensor<float, GTShape_5736826112, GTStride_5736826112, GT_5736826112_layout>;
  GT_5736826112 v31 = GT_5736826112(v2, v29, v30);
  TSTORE(v31, v18);
  using GTShape_5736826480 = pto::Shape<1, 1, 1, 8, 8>;
  using GTStride_5736826480 = pto::Stride<64, 64, 64, 8, 1>;
  constexpr pto::Layout GT_5736826480_layout = pto::Layout::ND;
  GTShape_5736826480 v32 = GTShape_5736826480();
  GTStride_5736826480 v33 = GTStride_5736826480();
  using GT_5736826480 = GlobalTensor<float, GTShape_5736826480, GTStride_5736826480, GT_5736826480_layout>;
  GT_5736826480 v34 = GT_5736826480(v3, v32, v33);
  TSTORE(v34, v21);
  using GTShape_5736826624 = pto::Shape<1, 1, 1, 8, 8>;
  using GTStride_5736826624 = pto::Stride<64, 64, 64, 8, 1>;
  constexpr pto::Layout GT_5736826624_layout = pto::Layout::ND;
  GTShape_5736826624 v35 = GTShape_5736826624();
  GTStride_5736826624 v36 = GTStride_5736826624();
  using GT_5736826624 = GlobalTensor<float, GTShape_5736826624, GTStride_5736826624, GT_5736826624_layout>;
  GT_5736826624 v37 = GT_5736826624(v4, v35, v36);
  TSTORE(v37, v24);
  using GTShape_5736826768 = pto::Shape<1, 1, 1, 8, 8>;
  using GTStride_5736826768 = pto::Stride<64, 64, 64, 8, 1>;
  constexpr pto::Layout GT_5736826768_layout = pto::Layout::ND;
  GTShape_5736826768 v38 = GTShape_5736826768();
  GTStride_5736826768 v39 = GTStride_5736826768();
  using GT_5736826768 = GlobalTensor<float, GTShape_5736826768, GTStride_5736826768, GT_5736826768_layout>;
  GT_5736826768 v40 = GT_5736826768(v5, v38, v39);
  TSTORE(v40, v27);
  ptoas_auto_sync_tail(PTOAutoSyncTailMode::kBarrierAll);
  return;
}
