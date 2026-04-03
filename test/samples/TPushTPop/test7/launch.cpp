// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

// ---------------------------------------------------------------------------
// PTOAS compatibility layer
// ---------------------------------------------------------------------------
#ifndef __VEC_SCOPE__
#define __VEC_SCOPE__
#endif

#if defined(__CCE_AICORE__) && defined(__NPU_ARCH__) && (__NPU_ARCH__ == 2201)
typedef struct {
  unsigned char v;
} hifloat8_t;
typedef struct {
  unsigned char v;
} float8_e4m3_t;
typedef struct {
  unsigned char v;
} float8_e5m2_t;
typedef struct {
  unsigned char v;
} float8_e8m0_t;
typedef struct {
  unsigned char v;
} float4_e1m2x2_t;
typedef struct {
  unsigned char v;
} float4_e2m1x2_t;
#endif

#include <cstdint>

#include "kernel.cpp"

__global__ AICORE void scope3_incore_0_incore_0_kernel(
    __gm__ float *attn_out, __gm__ bfloat16_t *hidden_states,
    __gm__ float *resid_out, __gm__ bfloat16_t *wo) {
  scope3_incore_0_incore_0_aic(attn_out, hidden_states, resid_out, wo);
  scope3_incore_0_incore_0_aiv(attn_out, hidden_states, resid_out, wo);
}

void LaunchScope3Incore0Incore0(float *attn_out, uint16_t *hidden_states,
                                float *resid_out, uint16_t *wo,
                                void *stream) {
  scope3_incore_0_incore_0_kernel<<<1, nullptr, stream>>>(
      (__gm__ float *)attn_out, (__gm__ bfloat16_t *)hidden_states,
      (__gm__ float *)resid_out, (__gm__ bfloat16_t *)wo);
}
