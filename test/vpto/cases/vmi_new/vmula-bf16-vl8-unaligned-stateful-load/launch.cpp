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
#include <cstdint>
#if !defined(__CCE_AICORE__) && !defined(TMRGSORT_HPP)
struct MrgSortExecutedNumList {
  uint16_t mrgSortList0;
  uint16_t mrgSortList1;
  uint16_t mrgSortList2;
  uint16_t mrgSortList3;
};
#endif
#ifndef __CPU_SIM
#include "acl/acl.h"
#endif

// bf16 inputs/outputs are carried as raw int16_t bit patterns.
extern "C" __global__ [aicore] void
vmi_vmula_bf16_vl8_unaligned_kernel(__gm__ int16_t *acc,
                          __gm__ int16_t *lhs,
                          __gm__ int16_t *rhs,
                          __gm__ int16_t *dst);

void LaunchVmi_vmula_bf16_vl8_unaligned_kernel(int16_t *acc, int16_t *lhs, int16_t *rhs,
                                     int16_t *dst, void *stream) {
  vmi_vmula_bf16_vl8_unaligned_kernel<<<1, nullptr, stream>>>(
      (__gm__ int16_t *)acc, (__gm__ int16_t *)lhs,
      (__gm__ int16_t *)rhs, (__gm__ int16_t *)dst);
}
