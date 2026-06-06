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

extern "C" __global__ AICORE void TINSERT_vec2vec_nd_f16_16x16_into_32x32_idx00(
    __gm__ uint16_t *src, __gm__ uint16_t *dst, __gm__ uint16_t *out);
extern "C" __global__ AICORE void TINSERT_vec2vec_nd_f16_16x16_into_32x32_idx816(
    __gm__ uint16_t *src, __gm__ uint16_t *dst, __gm__ uint16_t *out);
extern "C" __global__ AICORE void TINSERT_vec2vec_nd_f32_16x16_into_32x32_idx00(
    __gm__ float *src, __gm__ float *dst, __gm__ float *out);

void LaunchVec2VecND_f16_16x16_into_32x32_idx00(uint16_t *src, uint16_t *dst, uint16_t *out, void *stream) {
    TINSERT_vec2vec_nd_f16_16x16_into_32x32_idx00<<<1, nullptr, stream>>>((__gm__ uint16_t *)src, (__gm__ uint16_t *)dst, (__gm__ uint16_t *)out);
}

void LaunchVec2VecND_f16_16x16_into_32x32_idx816(uint16_t *src, uint16_t *dst, uint16_t *out, void *stream) {
    TINSERT_vec2vec_nd_f16_16x16_into_32x32_idx816<<<1, nullptr, stream>>>((__gm__ uint16_t *)src, (__gm__ uint16_t *)dst, (__gm__ uint16_t *)out);
}

void LaunchVec2VecND_f32_16x16_into_32x32_idx00(float *src, float *dst, float *out, void *stream) {
    TINSERT_vec2vec_nd_f32_16x16_into_32x32_idx00<<<1, nullptr, stream>>>((__gm__ float *)src, (__gm__ float *)dst, (__gm__ float *)out);
}