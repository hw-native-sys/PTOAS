// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

#include "common/pto_instr.hpp"
using namespace pto;
template <typename T, int M, int N, int validM, int validN>
__global__ AICORE void Run_vec_add_scalar_kernel_2d(__gm__ T* v1, __gm__ T* v2) {
  int64_t v3 = 4096;
  int64_t v4 = 0;
  unsigned v5 = 0;
  unsigned v6 = 1;
  unsigned v7 = 32;
  int32_t v8 = 88;
  unsigned v9 = v5 * v7;
  unsigned v10 = v5 + v9;
  unsigned v11 = v5 * v6;
  unsigned v12 = v10 + v11;
  __gm__ T* v13 = v1 + v12;
  using GTShape_187651838770208 = pto::Shape<1, 1, 1, 32, 32>;
  using GTStride_187651838770208 = pto::Stride<1, 1, 1, 32, 1>;
  using GT_187651838770208 = GlobalTensor<T, GTShape_187651838770208, GTStride_187651838770208>;
  GT_187651838770208 v14 = GT_187651838770208(v13);
  Tile<TileType::Vec, T, M, N, BLayout::RowMajor, validM, validN> v15;
  TASSIGN(v15, v4);
  Tile<TileType::Vec, T, M, N, BLayout::RowMajor, validM, validN> v16;
  TASSIGN(v16, v3);
  TLOAD(v15, v14);
  TANDS(v16, v15, v8);
  unsigned v17 = v5 * v7;
  unsigned v18 = v5 + v17;
  unsigned v19 = v5 * v6;
  unsigned v20 = v18 + v19;
  __gm__ T* v21 = v2 + v20;
  using GTShape_187651839006800 = pto::Shape<1, 1, 1, 32, 32>;
  using GTStride_187651839006800 = pto::Stride<1, 1, 1, 32, 1>;
  using GT_187651839006800 = GlobalTensor<T, GTShape_187651839006800, GTStride_187651839006800>;
  GT_187651839006800 v22 = GT_187651839006800(v21);
  TSTORE(v22, v16);
  return;
}

extern "C" [aicore] void vec_add_scalar_kernel_2d(__gm__ float* v1, __gm__ float* v2) {
  Run_vec_add_scalar_kernel_2d<float, 32, 32, 32, 32>(v1, v2);
  return;
}
