// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

#include "pto/pto-inst.hpp"
using namespace pto;
// v4: isAtranspose, v5: isBtranspose
__global__ AICORE void RunTEXTRACT(__gm__ float* v1, __gm__ float* v2, __gm__ float* v3, bool v4, bool v5) {
  unsigned v6 = 1024;
  unsigned v7 = 8192;
  unsigned v8 = 256;
  unsigned v9 = 32;
  unsigned v10 = 1;
  unsigned v11 = 0;
  int32_t v12 = 256;
  int32_t v13 = 32;
  int32_t v14 = 1;
  int32_t v15 = 0;
  int64_t v16 = 32768;
  int64_t v17 = 0;
  int64_t v18 = 65536;
  int64_t v19 = 98304;
  using T = float;
  pto::Shape<1, 1, 1, 32, 256> v20 = pto::Shape<1, 1, 1, 32, 256>();
  pto::Stride<8192, 8192, 8192, 256, 1> v21 = pto::Stride<8192, 8192, 8192, 256, 1>();
  GlobalTensor<float, pto::Shape<1, 1, 1, 32, 256>, pto::Stride<8192, 8192, 8192, 256, 1>, pto::Layout::ND> v22 = GlobalTensor<float, pto::Shape<1, 1, 1, 32, 256>, pto::Stride<8192, 8192, 8192, 256, 1>, pto::Layout::ND>(v2 + (v11 + v11 * (unsigned) v12 + v11 * (unsigned) v14), v20, v21);
  pto::Shape<1, 1, 1, 32, 256> v23 = pto::Shape<1, 1, 1, 32, 256>();
  pto::Stride<32, 32, 32, 1, 32> v24 = pto::Stride<32, 32, 32, 1, 32>();
  GlobalTensor<float, pto::Shape<1, 1, 1, 32, 256>, pto::Stride<32, 32, 32, 1, 32>, pto::Layout::DN> v25 = GlobalTensor<float, pto::Shape<1, 1, 1, 32, 256>, pto::Stride<32, 32, 32, 1, 32>, pto::Layout::DN>(v2 + (v11 + v11 * (unsigned) v14 + v11 * (unsigned) v13), v23, v24);
  pto::Shape<1, 1, 1, 256, 32> v26 = pto::Shape<1, 1, 1, 256, 32>();
  pto::Stride<8192, 8192, 8192, 32, 1> v27 = pto::Stride<8192, 8192, 8192, 32, 1>();
  GlobalTensor<float, pto::Shape<1, 1, 1, 256, 32>, pto::Stride<8192, 8192, 8192, 32, 1>, pto::Layout::ND> v28 = GlobalTensor<float, pto::Shape<1, 1, 1, 256, 32>, pto::Stride<8192, 8192, 8192, 32, 1>, pto::Layout::ND>(v3 + (v11 + v11 * (unsigned) v13 + v11 * (unsigned) v14), v26, v27);
  pto::Shape<1, 1, 1, 256, 32> v29 = pto::Shape<1, 1, 1, 256, 32>();
  pto::Stride<256, 256, 256, 1, 256> v30 = pto::Stride<256, 256, 256, 1, 256>();
  GlobalTensor<float, pto::Shape<1, 1, 1, 256, 32>, pto::Stride<256, 256, 256, 1, 256>, pto::Layout::DN> v31 = GlobalTensor<float, pto::Shape<1, 1, 1, 256, 32>, pto::Stride<256, 256, 256, 1, 256>, pto::Layout::DN>(v3 + (v11 + v11 * (unsigned) v14 + v11 * (unsigned) v12), v29, v30);
  Tile<TileType::Mat, float, 32, 256, BLayout::ColMajor, 32, 256, SLayout::RowMajor, 512, PadValue::Null> v32;
  TASSIGN(v32, v16);
  Tile<TileType::Mat, float, 32, 256, BLayout::ColMajor, 32, 256, SLayout::RowMajor, 512, PadValue::Null> v33;
  __cbuf__ float* v34 = v32.data();
  uint64_t v35 = reinterpret_cast<uint64_t>(v34);
  TASSIGN(v33, v35);
  Tile<TileType::Mat, float, 32, 256, BLayout::RowMajor, 32, 256, SLayout::ColMajor, 512, PadValue::Null> v36;
  TASSIGN(v36, v17);
  Tile<TileType::Mat, float, 32, 256, BLayout::RowMajor, 32, 256, SLayout::ColMajor, 512, PadValue::Null> v37;
  __cbuf__ float* v38 = v36.data();
  uint64_t v39 = reinterpret_cast<uint64_t>(v38);
  TASSIGN(v37, v39);
  Tile<TileType::Mat, float, 256, 32, BLayout::ColMajor, 256, 32, SLayout::RowMajor, 512, PadValue::Null> v40;
  TASSIGN(v40, v18);
  Tile<TileType::Mat, float, 256, 32, BLayout::ColMajor, 256, 32, SLayout::RowMajor, 512, PadValue::Null> v41;
  __cbuf__ float* v42 = v40.data();
  uint64_t v43 = reinterpret_cast<uint64_t>(v42);
  TASSIGN(v41, v43);
  Tile<TileType::Mat, float, 256, 32, BLayout::RowMajor, 256, 32, SLayout::ColMajor, 512, PadValue::Null> v44;
  TASSIGN(v44, v19);
  Tile<TileType::Mat, float, 256, 32, BLayout::RowMajor, 256, 32, SLayout::ColMajor, 512, PadValue::Null> v45;
  __cbuf__ float* v46 = v44.data();
  uint64_t v47 = reinterpret_cast<uint64_t>(v46);
  TASSIGN(v45, v47);
  Tile<TileType::Left, float, 32, 256, BLayout::RowMajor, 32, 256, SLayout::RowMajor, 512, PadValue::Null> v48;
  TASSIGN(v48, v17);
  Tile<TileType::Left, float, 32, 256, BLayout::RowMajor, 32, 256, SLayout::RowMajor, 512, PadValue::Null> v49;
  __ca__ float* v50 = v48.data();
  uint64_t v51 = reinterpret_cast<uint64_t>(v50);
  TASSIGN(v49, v51);
  Tile<TileType::Right, float, 256, 32, BLayout::RowMajor, 256, 32, SLayout::ColMajor, 512, PadValue::Null> v52;
  TASSIGN(v52, v17);
  Tile<TileType::Right, float, 256, 32, BLayout::RowMajor, 256, 32, SLayout::ColMajor, 512, PadValue::Null> v53;
  __cb__ float* v54 = v52.data();
  uint64_t v55 = reinterpret_cast<uint64_t>(v54);
  TASSIGN(v53, v55);
  Tile<TileType::Acc, float, 32, 32, BLayout::ColMajor, 32, 32, SLayout::RowMajor, 1024, PadValue::Null> v56;
  TASSIGN(v56, v17);
  Tile<TileType::Acc, float, 32, 32, BLayout::ColMajor, 32, 32, SLayout::RowMajor, 1024, PadValue::Null> v57;
  __cc__ float* v58 = v56.data();
  uint64_t v59 = reinterpret_cast<uint64_t>(v58);
  TASSIGN(v57, v59);
  if (v4) {
    TLOAD(v37, v25);
  } else {
    TLOAD(v33, v22);
  }
  if (v5) {
    TLOAD(v41, v28);
  } else {
    TLOAD(v45, v31);
  }
  set_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);
  wait_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);
  if (v4) {
    TEXTRACT(v49, v37, v15, v15);
  } else {
    TEXTRACT(v49, v33, v15, v15);
  }
  if (v5) {
    TEXTRACT(v53, v41, v15, v15);
  } else {
    TEXTRACT(v53, v45, v15, v15);
  }
  set_flag(PIPE_MTE1, PIPE_M, EVENT_ID0);
  wait_flag(PIPE_MTE1, PIPE_M, EVENT_ID0);
  TMATMUL(v57, v49, v53);
  set_flag(PIPE_M, PIPE_FIX, EVENT_ID0);
  wait_flag(PIPE_M, PIPE_FIX, EVENT_ID0);
  pto::Shape<1, 1, 1, 32, 32> v60 = pto::Shape<1, 1, 1, 32, 32>();
  pto::Stride<1024, 1024, 1024, 32, 1> v61 = pto::Stride<1024, 1024, 1024, 32, 1>();
  GlobalTensor<float, pto::Shape<1, 1, 1, 32, 32>, pto::Stride<1024, 1024, 1024, 32, 1>, pto::Layout::ND> v62 = GlobalTensor<float, pto::Shape<1, 1, 1, 32, 32>, pto::Stride<1024, 1024, 1024, 32, 1>, pto::Layout::ND>(v1 + (v11 + v11 * (unsigned) v13 + v11 * (unsigned) v14), v60, v61);
  TSTORE(v62, v57);
  return;
}

