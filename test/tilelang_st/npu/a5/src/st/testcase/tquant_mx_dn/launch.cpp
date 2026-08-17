// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

#include <cstdint>

#ifndef AICORE
#define AICORE [aicore]
#endif

#define DEFINE_LAUNCH(name)                                                     \
    extern "C" __global__ AICORE void name(                                   \
        __gm__ uint8_t *, __gm__ uint8_t *, __gm__ uint8_t *,                  \
        __gm__ uint8_t *, __gm__ uint8_t *, __gm__ uint8_t *);                 \
    void Launch##name(void *src, void *dst, void *exp, void *max,               \
                      void *scaling, void *expZz, void *stream) {               \
        name<<<1, nullptr, stream>>>(                                            \
            (__gm__ uint8_t *)src, (__gm__ uint8_t *)dst,                       \
            (__gm__ uint8_t *)exp, (__gm__ uint8_t *)max,                       \
            (__gm__ uint8_t *)scaling, (__gm__ uint8_t *)expZz);                \
    }

DEFINE_LAUNCH(TQUANT_MX_DN_fp8_ocp_64x64)
DEFINE_LAUNCH(TQUANT_MX_DN_fp8_ocp_128x32)
DEFINE_LAUNCH(TQUANT_MX_DN_fp8_nv_64x64)
DEFINE_LAUNCH(TQUANT_MX_DN_fp4_bf16_ocp_64x64)
DEFINE_LAUNCH(TQUANT_MX_DN_fp8_ocp_64x96)
DEFINE_LAUNCH(TQUANT_MX_ND_fp8_ocp_16x128)
DEFINE_LAUNCH(TQUANT_MX_ND_fp8_ocp_20x128)

#undef DEFINE_LAUNCH
