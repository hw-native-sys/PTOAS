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

extern "C" __global__ AICORE void TINSERT_acc2mat_f16_16x16(
    __gm__ uint16_t *a, __gm__ uint16_t *b, __gm__ uint16_t *c);
extern "C" __global__ AICORE void TINSERT_acc2mat_f16_32x32(
    __gm__ uint16_t *a, __gm__ uint16_t *b, __gm__ uint16_t *c);
extern "C" __global__ AICORE void TINSERT_acc2mat_bf16_16x16(
    __gm__ uint16_t *a, __gm__ uint16_t *b, __gm__ uint16_t *c);
extern "C" __global__ AICORE void TINSERT_acc2mat_f32_16x16(
    __gm__ uint16_t *a, __gm__ uint16_t *b, __gm__ uint32_t *c);
extern "C" __global__ AICORE void TINSERT_acc2vec_nd_f16_16x16(
    __gm__ uint16_t *a, __gm__ uint16_t *b, __gm__ uint32_t *c);
extern "C" __global__ AICORE void TINSERT_acc2vec_nd_f32_16x16(
    __gm__ uint16_t *a, __gm__ uint16_t *b, __gm__ uint32_t *c);
extern "C" __global__ AICORE void TINSERT_acc2vec_nz_f32_16x16(
    __gm__ uint16_t *a, __gm__ uint16_t *b, __gm__ uint32_t *c);

void LaunchAcc2Mat_f16_16x16(uint16_t *a, uint16_t *b, uint16_t *c, void *stream) {
    TINSERT_acc2mat_f16_16x16<<<1, nullptr, stream>>>((__gm__ uint16_t *)a, (__gm__ uint16_t *)b, (__gm__ uint16_t *)c);
}

void LaunchAcc2Mat_f16_32x32(uint16_t *a, uint16_t *b, uint16_t *c, void *stream) {
    TINSERT_acc2mat_f16_32x32<<<1, nullptr, stream>>>((__gm__ uint16_t *)a, (__gm__ uint16_t *)b, (__gm__ uint16_t *)c);
}

void LaunchAcc2Mat_bf16_16x16(uint16_t *a, uint16_t *b, uint16_t *c, void *stream) {
    TINSERT_acc2mat_bf16_16x16<<<1, nullptr, stream>>>((__gm__ uint16_t *)a, (__gm__ uint16_t *)b, (__gm__ uint16_t *)c);
}

void LaunchAcc2Mat_f32_16x16(uint16_t *a, uint16_t *b, uint32_t *c, void *stream) {
    TINSERT_acc2mat_f32_16x16<<<1, nullptr, stream>>>((__gm__ uint16_t *)a, (__gm__ uint16_t *)b, (__gm__ uint32_t *)c);
}

void LaunchAcc2VecND_f16_16x16(uint16_t *a, uint16_t *b, uint16_t *c, void *stream) {
    TINSERT_acc2vec_nd_f16_16x16<<<1, nullptr, stream>>>((__gm__ uint16_t *)a, (__gm__ uint16_t *)b, (__gm__ uint32_t *)c);
}

void LaunchAcc2VecND_f32_16x16(uint16_t *a, uint16_t *b, uint16_t *c, void *stream) {
    TINSERT_acc2vec_nd_f32_16x16<<<1, nullptr, stream>>>((__gm__ uint16_t *)a, (__gm__ uint16_t *)b, (__gm__ uint32_t *)c);
}

void LaunchAcc2VecNZ_f32_16x16(uint16_t *a, uint16_t *b, uint16_t *c, void *stream) {
    TINSERT_acc2vec_nz_f32_16x16<<<1, nullptr, stream>>>((__gm__ uint16_t *)a, (__gm__ uint16_t *)b, (__gm__ uint32_t *)c);
}
