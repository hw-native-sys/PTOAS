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

// No-quant TSTORE.ACC cases: dst + x1 + x2 (3 GM pointers)

// Case 0: f16 src, f32 dst (3 float* args)
extern "C" __global__ AICORE void TSTORE_ACC2GM_f16_f32_f32_nz2nd(
    __gm__ float *dst, __gm__ float *x1, __gm__ float *x2);

void LaunchTSTORE_ACC2GM_f16_f32_f32_nz2nd(float *dst, float *x1, float *x2, void *stream) {
    TSTORE_ACC2GM_f16_f32_f32_nz2nd<<<1, nullptr, stream>>>(
        (__gm__ float *)dst, (__gm__ float *)x1, (__gm__ float *)x2);
}

// Case 1: f16 src, f16 dst (3 void* args)
extern "C" __global__ AICORE void TSTORE_ACC2GM_f16_f32_f16_nz2nd(
    __gm__ half *dst, __gm__ half *x1, __gm__ half *x2);

void LaunchTSTORE_ACC2GM_f16_f32_f16_nz2nd(void *dst, void *x1, void *x2, void *stream) {
    TSTORE_ACC2GM_f16_f32_f16_nz2nd<<<1, nullptr, stream>>>(
        (__gm__ half *)dst, (__gm__ half *)x1, (__gm__ half *)x2);
}

// Case 2: bf16 src, f32 dst (float* dst, void* x1/x2 for bf16 as uint16)
extern "C" __global__ AICORE void TSTORE_ACC2GM_bf16_f32_f32_nz2nd(
    __gm__ float *dst, __gm__ uint16_t *x1, __gm__ uint16_t *x2);

void LaunchTSTORE_ACC2GM_bf16_f32_f32_nz2nd(float *dst, void *x1, void *x2, void *stream) {
    TSTORE_ACC2GM_bf16_f32_f32_nz2nd<<<1, nullptr, stream>>>(
        (__gm__ float *)dst, (__gm__ uint16_t *)x1, (__gm__ uint16_t *)x2);
}

// Case 3: bf16 src, bf16 dst (4 void* args)
extern "C" __global__ AICORE void TSTORE_ACC2GM_bf16_f32_bf16_nz2nd(
    __gm__ uint16_t *dst, __gm__ uint16_t *x1, __gm__ uint16_t *x2);

void LaunchTSTORE_ACC2GM_bf16_f32_bf16_nz2nd(void *dst, void *x1, void *x2, void *stream) {
    TSTORE_ACC2GM_bf16_f32_bf16_nz2nd<<<1, nullptr, stream>>>(
        (__gm__ uint16_t *)dst, (__gm__ uint16_t *)x1, (__gm__ uint16_t *)x2);
}

// Case 4: i8 src, i32 dst (3 void* args)
extern "C" __global__ AICORE void TSTORE_ACC2GM_i8_i32_i32_nz2nd(
    __gm__ int32_t *dst, __gm__ int8_t *x1, __gm__ int8_t *x2);

void LaunchTSTORE_ACC2GM_i8_i32_i32_nz2nd(void *dst, void *x1, void *x2, void *stream) {
    TSTORE_ACC2GM_i8_i32_i32_nz2nd<<<1, nullptr, stream>>>(
        (__gm__ int32_t *)dst, (__gm__ int8_t *)x1, (__gm__ int8_t *)x2);
}

// No-quant TSTORE.ACC NZ2DN cases: dst + x1 + x2 (3 GM pointers)

// NZ2DN case: f16 src, f32 dst (3 float* args)
extern "C" __global__ AICORE void TSTORE_ACC2GM_f16_f32_f32_nz2dn(
    __gm__ float *dst, __gm__ half *x1, __gm__ half *x2);

void LaunchTSTORE_ACC2GM_f16_f32_f32_nz2dn(float *dst, float *x1, float *x2, void *stream) {
    TSTORE_ACC2GM_f16_f32_f32_nz2dn<<<1, nullptr, stream>>>(
        (__gm__ float *)dst, (__gm__ half *)x1, (__gm__ half *)x2);
}

// NZ2DN case: bf16 src, bf16 dst (3 void* args)
extern "C" __global__ AICORE void TSTORE_ACC2GM_bf16_f32_bf16_nz2dn(
    __gm__ uint16_t *dst, __gm__ uint16_t *x1, __gm__ uint16_t *x2);

void LaunchTSTORE_ACC2GM_bf16_f32_bf16_nz2dn(void *dst, void *x1, void *x2, void *stream) {
    TSTORE_ACC2GM_bf16_f32_bf16_nz2dn<<<1, nullptr, stream>>>(
        (__gm__ uint16_t *)dst, (__gm__ uint16_t *)x1, (__gm__ uint16_t *)x2);
}

// No-quant TSTORE.ACC NZ2NZ cases: dst + x1 + x2 (3 GM pointers)

// NZ2NZ case: f16 src, f32 dst (3 float* args)
extern "C" __global__ AICORE void TSTORE_ACC2GM_f16_f32_f32_nz2nz(
    __gm__ float *dst, __gm__ half *x1, __gm__ half *x2);

void LaunchTSTORE_ACC2GM_f16_f32_f32_nz2nz(float *dst, float *x1, float *x2, void *stream) {
    TSTORE_ACC2GM_f16_f32_f32_nz2nz<<<1, nullptr, stream>>>(
        (__gm__ float *)dst, (__gm__ half *)x1, (__gm__ half *)x2);
}

// NZ2NZ case: bf16 src, bf16 dst (3 void* args)
extern "C" __global__ AICORE void TSTORE_ACC2GM_bf16_f32_bf16_nz2nz(
    __gm__ uint16_t *dst, __gm__ uint16_t *x1, __gm__ uint16_t *x2);

void LaunchTSTORE_ACC2GM_bf16_f32_bf16_nz2nz(void *dst, void *x1, void *x2, void *stream) {
    TSTORE_ACC2GM_bf16_f32_bf16_nz2nz<<<1, nullptr, stream>>>(
        (__gm__ uint16_t *)dst, (__gm__ uint16_t *)x1, (__gm__ uint16_t *)x2);
}

// Vector quant TSTORE_FP cases: dst + x1 + x2 + quant (4 GM pointers)

// Case 5: encoded vector scaling, f16 dst (4 void* args)
extern "C" __global__ AICORE void TSTORE_ACC2GM_f16_f32_f16_vec(
    __gm__ half *dst, __gm__ half *x1, __gm__ half *x2, __gm__ uint32_t *quant);

void LaunchTSTORE_ACC2GM_f16_f32_f16_vec(void *dst, void *x1, void *x2, void *quant, void *stream) {
    TSTORE_ACC2GM_f16_f32_f16_vec<<<1, nullptr, stream>>>(
        (__gm__ half *)dst, (__gm__ half *)x1, (__gm__ half *)x2, (__gm__ uint32_t *)quant);
}

// Case 6: encoded vector scaling, bf16 dst (4 void* args)
extern "C" __global__ AICORE void TSTORE_ACC2GM_bf16_f32_bf16_vec(
    __gm__ uint16_t *dst, __gm__ uint16_t *x1, __gm__ uint16_t *x2, __gm__ uint32_t *quant);

void LaunchTSTORE_ACC2GM_bf16_f32_bf16_vec(void *dst, void *x1, void *x2, void *quant, void *stream) {
    TSTORE_ACC2GM_bf16_f32_bf16_vec<<<1, nullptr, stream>>>(
        (__gm__ uint16_t *)dst, (__gm__ uint16_t *)x1, (__gm__ uint16_t *)x2, (__gm__ uint32_t *)quant);
}
