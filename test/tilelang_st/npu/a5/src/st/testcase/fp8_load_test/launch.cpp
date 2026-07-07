// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

#include <stdint.h>

#ifndef AICORE
#define AICORE [aicore]
#endif

extern "C" __global__ AICORE void FP8_LOAD_16x64(__gm__ float8_e4m3_t *src, __gm__ float8_e4m3_t *dst);
extern "C" __global__ AICORE void FP8_LOAD_32x64(__gm__ float8_e4m3_t *src, __gm__ float8_e4m3_t *dst);
extern "C" __global__ AICORE void FP8_LOAD_128x64(__gm__ float8_e4m3_t *src, __gm__ float8_e4m3_t *dst);

void LaunchFP8_LOAD_16x64(uint8_t *src, uint8_t *dst, void *stream) {
    FP8_LOAD_16x64<<<1, nullptr, stream>>>((__gm__ float8_e4m3_t *)src, (__gm__ float8_e4m3_t *)dst);
}

void LaunchFP8_LOAD_32x64(uint8_t *src, uint8_t *dst, void *stream) {
    FP8_LOAD_32x64<<<1, nullptr, stream>>>((__gm__ float8_e4m3_t *)src, (__gm__ float8_e4m3_t *)dst);
}

void LaunchFP8_LOAD_128x64(uint8_t *src, uint8_t *dst, void *stream) {
    FP8_LOAD_128x64<<<1, nullptr, stream>>>((__gm__ float8_e4m3_t *)src, (__gm__ float8_e4m3_t *)dst);
}
