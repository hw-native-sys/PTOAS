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
fa_dn_softmax_128x64(__gm__ float *scores, __gm__ __bf16 *x_exp,
                     __gm__ float *global_max, __gm__ float *global_sum,
                     __gm__ __bf16 *nz_out);

void LaunchFa_dn_softmax_128x64(float *scores, __bf16 *x_exp,
                                 float *global_max, float *global_sum,
                                 __bf16 *nz_out, void *stream) {
  fa_dn_softmax_128x64<<<1, nullptr, stream>>>(
      (__gm__ float *)scores, (__gm__ __bf16 *)x_exp,
      (__gm__ float *)global_max, (__gm__ float *)global_sum,
      (__gm__ __bf16 *)nz_out);
}
