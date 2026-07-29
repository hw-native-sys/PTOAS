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

extern "C" __global__ [aicore] void
mte_ub_gm_atomic_add_s32_kernel(__gm__ int32_t *src,
                                __gm__ int32_t *dst);

void LaunchMteUbGmAtomicAddS32Kernel(int32_t *src, int32_t *dst,
                                     void *stream) {
  mte_ub_gm_atomic_add_s32_kernel<<<1, nullptr, stream>>>(
      (__gm__ int32_t *)src, (__gm__ int32_t *)dst);
}
