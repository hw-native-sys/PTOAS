// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

// -----------------------------------------------------------------------------
// case: async-comm/sdma-gm-gm-local
// target_ops: pto.sdma_gm_gm
// scenarios: single-device, local GM->GM, one channel, one block
// -----------------------------------------------------------------------------
#ifndef __VEC_SCOPE__
#define __VEC_SCOPE__
#endif

#include <stdint.h>

#ifndef __CPU_SIM
#include "acl/acl.h"
#endif

extern "C" __global__ [aicore] void sdma_gm_gm_local(__gm__ int8_t *dst,
                                                    __gm__ int8_t *src,
                                                    __gm__ int8_t *sessionGm,
                                                    int64_t nbytes);

// A single block: the session owns one channel group, and a second core would
// race it on the same queue ring.
void LaunchSdmaGmGmLocal(int8_t *dst, int8_t *src, int8_t *sessionGm,
                         int64_t nbytes, void *stream) {
  sdma_gm_gm_local<<<1, nullptr, stream>>>((__gm__ int8_t *)dst,
                                           (__gm__ int8_t *)src,
                                           (__gm__ int8_t *)sessionGm, nbytes);
}
