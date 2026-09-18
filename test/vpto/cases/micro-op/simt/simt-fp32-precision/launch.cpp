// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software; you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OR
// CONDITIONS OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the license.
#ifndef __VEC_SCOPE__
#define __VEC_SCOPE__
#endif
#ifndef __CPU_SIM
#include "acl/acl.h"
#endif

extern "C" __global__ [aicore] void simt_fp32_precision_kernel(
    __gm__ float *a, __gm__ float *b, __gm__ float *c, __gm__ float *lhs,
    __gm__ float *rhs, __gm__ float *out);

void LaunchSimt_fp32_precision_kernel(float *a, float *b, float *c,
                                      float *lhs, float *rhs, float *out,
                                      void *stream) {
  simt_fp32_precision_kernel<<<1, nullptr, stream>>>(a, b, c, lhs, rhs, out);
}
