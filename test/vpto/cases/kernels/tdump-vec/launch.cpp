// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

// -----------------------------------------------------------------------------
// case: kernels/tdump-vec
// family: kernels
// target_ops: pto.mte_gm_ub, pto.tdump
// scenarios: tdump-vec-f32-4x16, header-plus-row-major-payload
// -----------------------------------------------------------------------------
#ifndef __VEC_SCOPE__
#define __VEC_SCOPE__
#endif

#include <stdint.h>

#if defined(__CCE_AICORE__) && defined(PTOAS_ENABLE_CCE_PRINT)
#include <ccelib/print/print.h>
#endif
#ifndef __CPU_SIM
#include "acl/acl.h"
#endif

extern "C" __global__ [aicore] void tdump_vec_kernel(__gm__ float *v1,
                                                     __gm__ float *v2);

void LaunchTdump_vec_kernel(float *v1, float *v2, void *stream) {
  tdump_vec_kernel<<<1, nullptr, stream>>>((__gm__ float *)v1,
                                           (__gm__ float *)v2);
}
