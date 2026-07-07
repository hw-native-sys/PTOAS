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

extern "C" __global__ AICORE void TCVT_f32_to_i32_rint_16x64(__gm__ float *src, __gm__ int32_t *dst);
extern "C" __global__ AICORE void TCVT_f32_to_i32_round_16x64(__gm__ float *src, __gm__ int32_t *dst);
extern "C" __global__ AICORE void TCVT_i32_to_f32_rint_16x64(__gm__ int32_t *src, __gm__ float *dst);
extern "C" __global__ AICORE void TCVT_f32_to_f16_rint_16x64(__gm__ float *src, __gm__ uint16_t *dst);
extern "C" __global__ AICORE void TCVT_f16_to_f32_rint_16x64(__gm__ uint16_t *src, __gm__ float *dst);
extern "C" __global__ AICORE void TCVT_f32_to_f8e4m3_16x64(__gm__ float *src, __gm__ float8_e4m3_t *dst);
extern "C" __global__ AICORE void TCVT_f32_to_f8e5m2_16x64(__gm__ float *src, __gm__ float8_e5m2_t *dst);
extern "C" __global__ AICORE void TCVT_f32_to_hif8_16x64(__gm__ float *src, __gm__ hifloat8_t *dst);
extern "C" __global__ AICORE void TCVT_f16_to_hif8_16x64(__gm__ uint16_t *src, __gm__ hifloat8_t *dst);
extern "C" __global__ AICORE void TCVT_bf16_to_f4e1m2x2_16x64_to_16x32(__gm__ uint16_t *src, __gm__ float4_e1m2x2_t *dst);
extern "C" __global__ AICORE void TCVT_bf16_to_f4e2m1x2_16x64_to_16x32(__gm__ uint16_t *src, __gm__ float4_e2m1x2_t *dst);
extern "C" __global__ AICORE void TCVT_f32_to_f8e4m3_4x96(__gm__ float *src, __gm__ float8_e4m3_t *dst);
extern "C" __global__ AICORE void TCVT_f32_to_hif8_4x96(__gm__ float *src, __gm__ hifloat8_t *dst);
extern "C" __global__ AICORE void TCVT_f32_to_f16_1x128(__gm__ float *src, __gm__ uint16_t *dst);
extern "C" __global__ AICORE void TCVT_f32_to_f16_2x64(__gm__ float *src, __gm__ uint16_t *dst);
extern "C" __global__ AICORE void TCVT_f32_to_f16_4x32(__gm__ float *src, __gm__ uint16_t *dst);
extern "C" __global__ AICORE void TCVT_f32_to_f16_2x128(__gm__ float *src, __gm__ uint16_t *dst);
extern "C" __global__ AICORE void TCVT_f32_to_f16_4x65(__gm__ float *src, __gm__ uint16_t *dst);
extern "C" __global__ AICORE void TCVT_f32_to_f16_4x200(__gm__ float *src, __gm__ uint16_t *dst);
extern "C" __global__ AICORE void TCVT_f32_to_f16_1x129(__gm__ float *src, __gm__ uint16_t *dst);
extern "C" __global__ AICORE void TCVT_f32_to_bf16_1x128(__gm__ float *src, __gm__ uint16_t *dst);
extern "C" __global__ AICORE void TCVT_f32_to_bf16_2x64(__gm__ float *src, __gm__ uint16_t *dst);
extern "C" __global__ AICORE void TCVT_f32_to_bf16_4x32(__gm__ float *src, __gm__ uint16_t *dst);
extern "C" __global__ AICORE void TCVT_f32_to_bf16_2x128(__gm__ float *src, __gm__ uint16_t *dst);
extern "C" __global__ AICORE void TCVT_f32_to_bf16_4x65(__gm__ float *src, __gm__ uint16_t *dst);
extern "C" __global__ AICORE void TCVT_f32_to_bf16_4x200(__gm__ float *src, __gm__ uint16_t *dst);
extern "C" __global__ AICORE void TCVT_f32_to_bf16_1x129(__gm__ float *src, __gm__ uint16_t *dst);
extern "C" __global__ AICORE void TCVT_f32_to_i16_1x128(__gm__ float *src, __gm__ int16_t *dst);
extern "C" __global__ AICORE void TCVT_f32_to_i16_2x64(__gm__ float *src, __gm__ int16_t *dst);
extern "C" __global__ AICORE void TCVT_f32_to_i16_4x32(__gm__ float *src, __gm__ int16_t *dst);
extern "C" __global__ AICORE void TCVT_f32_to_i16_2x128(__gm__ float *src, __gm__ int16_t *dst);
extern "C" __global__ AICORE void TCVT_f32_to_i16_4x65(__gm__ float *src, __gm__ int16_t *dst);
extern "C" __global__ AICORE void TCVT_f32_to_i16_4x200(__gm__ float *src, __gm__ int16_t *dst);
extern "C" __global__ AICORE void TCVT_f32_to_i16_1x129(__gm__ float *src, __gm__ int16_t *dst);
extern "C" __global__ AICORE void TCVT_f32_to_i32_1x128(__gm__ float *src, __gm__ int32_t *dst);
extern "C" __global__ AICORE void TCVT_f32_to_i32_2x64(__gm__ float *src, __gm__ int32_t *dst);
extern "C" __global__ AICORE void TCVT_f32_to_i32_4x32(__gm__ float *src, __gm__ int32_t *dst);
extern "C" __global__ AICORE void TCVT_f32_to_i32_2x128(__gm__ float *src, __gm__ int32_t *dst);
extern "C" __global__ AICORE void TCVT_f32_to_i32_4x65(__gm__ float *src, __gm__ int32_t *dst);
extern "C" __global__ AICORE void TCVT_f32_to_i32_4x200(__gm__ float *src, __gm__ int32_t *dst);
extern "C" __global__ AICORE void TCVT_f32_to_i32_1x129(__gm__ float *src, __gm__ int32_t *dst);
extern "C" __global__ AICORE void TCVT_f32_to_i64_1x128(__gm__ float *src, __gm__ int64_t *dst);
extern "C" __global__ AICORE void TCVT_f32_to_i64_2x64(__gm__ float *src, __gm__ int64_t *dst);
extern "C" __global__ AICORE void TCVT_f32_to_i64_4x32(__gm__ float *src, __gm__ int64_t *dst);
extern "C" __global__ AICORE void TCVT_f32_to_i64_2x128(__gm__ float *src, __gm__ int64_t *dst);
extern "C" __global__ AICORE void TCVT_f32_to_i64_4x65(__gm__ float *src, __gm__ int64_t *dst);
extern "C" __global__ AICORE void TCVT_f32_to_i64_4x200(__gm__ float *src, __gm__ int64_t *dst);
extern "C" __global__ AICORE void TCVT_f32_to_i64_1x129(__gm__ float *src, __gm__ int64_t *dst);
extern "C" __global__ AICORE void TCVT_f32_to_f32_1x128(__gm__ float *src, __gm__ float *dst);
extern "C" __global__ AICORE void TCVT_f32_to_f32_2x64(__gm__ float *src, __gm__ float *dst);
extern "C" __global__ AICORE void TCVT_f32_to_f32_4x32(__gm__ float *src, __gm__ float *dst);
extern "C" __global__ AICORE void TCVT_f32_to_f32_2x128(__gm__ float *src, __gm__ float *dst);
extern "C" __global__ AICORE void TCVT_f32_to_f32_4x65(__gm__ float *src, __gm__ float *dst);
extern "C" __global__ AICORE void TCVT_f32_to_f32_4x200(__gm__ float *src, __gm__ float *dst);
extern "C" __global__ AICORE void TCVT_f32_to_f32_1x129(__gm__ float *src, __gm__ float *dst);
extern "C" __global__ AICORE void TCVT_f16_to_f32_1x128(__gm__ uint16_t *src, __gm__ float *dst);
extern "C" __global__ AICORE void TCVT_f16_to_f32_2x64(__gm__ uint16_t *src, __gm__ float *dst);
extern "C" __global__ AICORE void TCVT_f16_to_f32_4x32(__gm__ uint16_t *src, __gm__ float *dst);
extern "C" __global__ AICORE void TCVT_f16_to_f32_2x128(__gm__ uint16_t *src, __gm__ float *dst);
extern "C" __global__ AICORE void TCVT_f16_to_f32_4x65(__gm__ uint16_t *src, __gm__ float *dst);
extern "C" __global__ AICORE void TCVT_f16_to_f32_4x200(__gm__ uint16_t *src, __gm__ float *dst);
extern "C" __global__ AICORE void TCVT_f16_to_f32_1x129(__gm__ uint16_t *src, __gm__ float *dst);
extern "C" __global__ AICORE void TCVT_f16_to_i32_1x128(__gm__ uint16_t *src, __gm__ int32_t *dst);
extern "C" __global__ AICORE void TCVT_f16_to_i32_2x64(__gm__ uint16_t *src, __gm__ int32_t *dst);
extern "C" __global__ AICORE void TCVT_f16_to_i32_4x32(__gm__ uint16_t *src, __gm__ int32_t *dst);
extern "C" __global__ AICORE void TCVT_f16_to_i32_2x128(__gm__ uint16_t *src, __gm__ int32_t *dst);
extern "C" __global__ AICORE void TCVT_f16_to_i32_4x65(__gm__ uint16_t *src, __gm__ int32_t *dst);
extern "C" __global__ AICORE void TCVT_f16_to_i32_4x200(__gm__ uint16_t *src, __gm__ int32_t *dst);
extern "C" __global__ AICORE void TCVT_f16_to_i32_1x129(__gm__ uint16_t *src, __gm__ int32_t *dst);
extern "C" __global__ AICORE void TCVT_f16_to_i16_1x128(__gm__ uint16_t *src, __gm__ int16_t *dst);
extern "C" __global__ AICORE void TCVT_f16_to_i16_2x64(__gm__ uint16_t *src, __gm__ int16_t *dst);
extern "C" __global__ AICORE void TCVT_f16_to_i16_4x32(__gm__ uint16_t *src, __gm__ int16_t *dst);
extern "C" __global__ AICORE void TCVT_f16_to_i16_2x128(__gm__ uint16_t *src, __gm__ int16_t *dst);
extern "C" __global__ AICORE void TCVT_f16_to_i16_4x65(__gm__ uint16_t *src, __gm__ int16_t *dst);
extern "C" __global__ AICORE void TCVT_f16_to_i16_4x200(__gm__ uint16_t *src, __gm__ int16_t *dst);
extern "C" __global__ AICORE void TCVT_f16_to_i16_1x129(__gm__ uint16_t *src, __gm__ int16_t *dst);
extern "C" __global__ AICORE void TCVT_f16_to_si8_1x128(__gm__ uint16_t *src, __gm__ int8_t *dst);
extern "C" __global__ AICORE void TCVT_f16_to_si8_2x64(__gm__ uint16_t *src, __gm__ int8_t *dst);
extern "C" __global__ AICORE void TCVT_f16_to_si8_4x32(__gm__ uint16_t *src, __gm__ int8_t *dst);
extern "C" __global__ AICORE void TCVT_f16_to_si8_2x128(__gm__ uint16_t *src, __gm__ int8_t *dst);
extern "C" __global__ AICORE void TCVT_f16_to_si8_4x65(__gm__ uint16_t *src, __gm__ int8_t *dst);
extern "C" __global__ AICORE void TCVT_f16_to_si8_4x200(__gm__ uint16_t *src, __gm__ int8_t *dst);
extern "C" __global__ AICORE void TCVT_f16_to_si8_1x129(__gm__ uint16_t *src, __gm__ int8_t *dst);
extern "C" __global__ AICORE void TCVT_f16_to_ui8_1x128(__gm__ uint16_t *src, __gm__ uint8_t *dst);
extern "C" __global__ AICORE void TCVT_f16_to_ui8_2x64(__gm__ uint16_t *src, __gm__ uint8_t *dst);
extern "C" __global__ AICORE void TCVT_f16_to_ui8_4x32(__gm__ uint16_t *src, __gm__ uint8_t *dst);
extern "C" __global__ AICORE void TCVT_f16_to_ui8_2x128(__gm__ uint16_t *src, __gm__ uint8_t *dst);
extern "C" __global__ AICORE void TCVT_f16_to_ui8_4x65(__gm__ uint16_t *src, __gm__ uint8_t *dst);
extern "C" __global__ AICORE void TCVT_f16_to_ui8_4x200(__gm__ uint16_t *src, __gm__ uint8_t *dst);
extern "C" __global__ AICORE void TCVT_f16_to_ui8_1x129(__gm__ uint16_t *src, __gm__ uint8_t *dst);
extern "C" __global__ AICORE void TCVT_bf16_to_f32_1x128(__gm__ uint16_t *src, __gm__ float *dst);
extern "C" __global__ AICORE void TCVT_bf16_to_f32_2x64(__gm__ uint16_t *src, __gm__ float *dst);
extern "C" __global__ AICORE void TCVT_bf16_to_f32_4x32(__gm__ uint16_t *src, __gm__ float *dst);
extern "C" __global__ AICORE void TCVT_bf16_to_f32_2x128(__gm__ uint16_t *src, __gm__ float *dst);
extern "C" __global__ AICORE void TCVT_bf16_to_f32_4x65(__gm__ uint16_t *src, __gm__ float *dst);
extern "C" __global__ AICORE void TCVT_bf16_to_f32_4x200(__gm__ uint16_t *src, __gm__ float *dst);
extern "C" __global__ AICORE void TCVT_bf16_to_f32_1x129(__gm__ uint16_t *src, __gm__ float *dst);
extern "C" __global__ AICORE void TCVT_bf16_to_f16_1x128(__gm__ uint16_t *src, __gm__ uint16_t *dst);
extern "C" __global__ AICORE void TCVT_bf16_to_f16_2x64(__gm__ uint16_t *src, __gm__ uint16_t *dst);
extern "C" __global__ AICORE void TCVT_bf16_to_f16_4x32(__gm__ uint16_t *src, __gm__ uint16_t *dst);
extern "C" __global__ AICORE void TCVT_bf16_to_f16_2x128(__gm__ uint16_t *src, __gm__ uint16_t *dst);
extern "C" __global__ AICORE void TCVT_bf16_to_f16_4x65(__gm__ uint16_t *src, __gm__ uint16_t *dst);
extern "C" __global__ AICORE void TCVT_bf16_to_f16_4x200(__gm__ uint16_t *src, __gm__ uint16_t *dst);
extern "C" __global__ AICORE void TCVT_bf16_to_f16_1x129(__gm__ uint16_t *src, __gm__ uint16_t *dst);
extern "C" __global__ AICORE void TCVT_bf16_to_i32_1x128(__gm__ uint16_t *src, __gm__ int32_t *dst);
extern "C" __global__ AICORE void TCVT_bf16_to_i32_2x64(__gm__ uint16_t *src, __gm__ int32_t *dst);
extern "C" __global__ AICORE void TCVT_bf16_to_i32_4x32(__gm__ uint16_t *src, __gm__ int32_t *dst);
extern "C" __global__ AICORE void TCVT_bf16_to_i32_2x128(__gm__ uint16_t *src, __gm__ int32_t *dst);
extern "C" __global__ AICORE void TCVT_bf16_to_i32_4x65(__gm__ uint16_t *src, __gm__ int32_t *dst);
extern "C" __global__ AICORE void TCVT_bf16_to_i32_4x200(__gm__ uint16_t *src, __gm__ int32_t *dst);
extern "C" __global__ AICORE void TCVT_bf16_to_i32_1x129(__gm__ uint16_t *src, __gm__ int32_t *dst);
extern "C" __global__ AICORE void TCVT_ui8_to_f16_1x128(__gm__ uint8_t *src, __gm__ uint16_t *dst);
extern "C" __global__ AICORE void TCVT_ui8_to_f16_2x64(__gm__ uint8_t *src, __gm__ uint16_t *dst);
extern "C" __global__ AICORE void TCVT_ui8_to_f16_4x32(__gm__ uint8_t *src, __gm__ uint16_t *dst);
extern "C" __global__ AICORE void TCVT_ui8_to_f16_2x128(__gm__ uint8_t *src, __gm__ uint16_t *dst);
extern "C" __global__ AICORE void TCVT_ui8_to_f16_4x65(__gm__ uint8_t *src, __gm__ uint16_t *dst);
extern "C" __global__ AICORE void TCVT_ui8_to_f16_4x200(__gm__ uint8_t *src, __gm__ uint16_t *dst);
extern "C" __global__ AICORE void TCVT_ui8_to_f16_1x129(__gm__ uint8_t *src, __gm__ uint16_t *dst);
extern "C" __global__ AICORE void TCVT_ui8_to_ui16_1x128(__gm__ uint8_t *src, __gm__ uint16_t *dst);
extern "C" __global__ AICORE void TCVT_ui8_to_ui16_2x64(__gm__ uint8_t *src, __gm__ uint16_t *dst);
extern "C" __global__ AICORE void TCVT_ui8_to_ui16_4x32(__gm__ uint8_t *src, __gm__ uint16_t *dst);
extern "C" __global__ AICORE void TCVT_ui8_to_ui16_2x128(__gm__ uint8_t *src, __gm__ uint16_t *dst);
extern "C" __global__ AICORE void TCVT_ui8_to_ui16_4x65(__gm__ uint8_t *src, __gm__ uint16_t *dst);
extern "C" __global__ AICORE void TCVT_ui8_to_ui16_4x200(__gm__ uint8_t *src, __gm__ uint16_t *dst);
extern "C" __global__ AICORE void TCVT_ui8_to_ui16_1x129(__gm__ uint8_t *src, __gm__ uint16_t *dst);
extern "C" __global__ AICORE void TCVT_si8_to_f16_1x128(__gm__ int8_t *src, __gm__ uint16_t *dst);
extern "C" __global__ AICORE void TCVT_si8_to_f16_2x64(__gm__ int8_t *src, __gm__ uint16_t *dst);
extern "C" __global__ AICORE void TCVT_si8_to_f16_4x32(__gm__ int8_t *src, __gm__ uint16_t *dst);
extern "C" __global__ AICORE void TCVT_si8_to_f16_2x128(__gm__ int8_t *src, __gm__ uint16_t *dst);
extern "C" __global__ AICORE void TCVT_si8_to_f16_4x65(__gm__ int8_t *src, __gm__ uint16_t *dst);
extern "C" __global__ AICORE void TCVT_si8_to_f16_4x200(__gm__ int8_t *src, __gm__ uint16_t *dst);
extern "C" __global__ AICORE void TCVT_si8_to_f16_1x129(__gm__ int8_t *src, __gm__ uint16_t *dst);
extern "C" __global__ AICORE void TCVT_si8_to_si16_1x128(__gm__ int8_t *src, __gm__ int16_t *dst);
extern "C" __global__ AICORE void TCVT_si8_to_si16_2x64(__gm__ int8_t *src, __gm__ int16_t *dst);
extern "C" __global__ AICORE void TCVT_si8_to_si16_4x32(__gm__ int8_t *src, __gm__ int16_t *dst);
extern "C" __global__ AICORE void TCVT_si8_to_si16_2x128(__gm__ int8_t *src, __gm__ int16_t *dst);
extern "C" __global__ AICORE void TCVT_si8_to_si16_4x65(__gm__ int8_t *src, __gm__ int16_t *dst);
extern "C" __global__ AICORE void TCVT_si8_to_si16_4x200(__gm__ int8_t *src, __gm__ int16_t *dst);
extern "C" __global__ AICORE void TCVT_si8_to_si16_1x129(__gm__ int8_t *src, __gm__ int16_t *dst);
extern "C" __global__ AICORE void TCVT_si8_to_i32_1x128(__gm__ int8_t *src, __gm__ int32_t *dst);
extern "C" __global__ AICORE void TCVT_si8_to_i32_2x64(__gm__ int8_t *src, __gm__ int32_t *dst);
extern "C" __global__ AICORE void TCVT_si8_to_i32_4x32(__gm__ int8_t *src, __gm__ int32_t *dst);
extern "C" __global__ AICORE void TCVT_si8_to_i32_2x128(__gm__ int8_t *src, __gm__ int32_t *dst);
extern "C" __global__ AICORE void TCVT_si8_to_i32_4x65(__gm__ int8_t *src, __gm__ int32_t *dst);
extern "C" __global__ AICORE void TCVT_si8_to_i32_4x200(__gm__ int8_t *src, __gm__ int32_t *dst);
extern "C" __global__ AICORE void TCVT_si8_to_i32_1x129(__gm__ int8_t *src, __gm__ int32_t *dst);
extern "C" __global__ AICORE void TCVT_i16_to_ui8_1x128(__gm__ int16_t *src, __gm__ uint8_t *dst);
extern "C" __global__ AICORE void TCVT_i16_to_ui8_2x64(__gm__ int16_t *src, __gm__ uint8_t *dst);
extern "C" __global__ AICORE void TCVT_i16_to_ui8_4x32(__gm__ int16_t *src, __gm__ uint8_t *dst);
extern "C" __global__ AICORE void TCVT_i16_to_ui8_2x128(__gm__ int16_t *src, __gm__ uint8_t *dst);
extern "C" __global__ AICORE void TCVT_i16_to_ui8_4x65(__gm__ int16_t *src, __gm__ uint8_t *dst);
extern "C" __global__ AICORE void TCVT_i16_to_ui8_4x200(__gm__ int16_t *src, __gm__ uint8_t *dst);
extern "C" __global__ AICORE void TCVT_i16_to_ui8_1x129(__gm__ int16_t *src, __gm__ uint8_t *dst);
extern "C" __global__ AICORE void TCVT_i16_to_f16_1x128(__gm__ int16_t *src, __gm__ uint16_t *dst);
extern "C" __global__ AICORE void TCVT_i16_to_f16_2x64(__gm__ int16_t *src, __gm__ uint16_t *dst);
extern "C" __global__ AICORE void TCVT_i16_to_f16_4x32(__gm__ int16_t *src, __gm__ uint16_t *dst);
extern "C" __global__ AICORE void TCVT_i16_to_f16_2x128(__gm__ int16_t *src, __gm__ uint16_t *dst);
extern "C" __global__ AICORE void TCVT_i16_to_f16_4x65(__gm__ int16_t *src, __gm__ uint16_t *dst);
extern "C" __global__ AICORE void TCVT_i16_to_f16_4x200(__gm__ int16_t *src, __gm__ uint16_t *dst);
extern "C" __global__ AICORE void TCVT_i16_to_f16_1x129(__gm__ int16_t *src, __gm__ uint16_t *dst);
extern "C" __global__ AICORE void TCVT_i16_to_f32_1x128(__gm__ int16_t *src, __gm__ float *dst);
extern "C" __global__ AICORE void TCVT_i16_to_f32_2x64(__gm__ int16_t *src, __gm__ float *dst);
extern "C" __global__ AICORE void TCVT_i16_to_f32_4x32(__gm__ int16_t *src, __gm__ float *dst);
extern "C" __global__ AICORE void TCVT_i16_to_f32_2x128(__gm__ int16_t *src, __gm__ float *dst);
extern "C" __global__ AICORE void TCVT_i16_to_f32_4x65(__gm__ int16_t *src, __gm__ float *dst);
extern "C" __global__ AICORE void TCVT_i16_to_f32_4x200(__gm__ int16_t *src, __gm__ float *dst);
extern "C" __global__ AICORE void TCVT_i16_to_f32_1x129(__gm__ int16_t *src, __gm__ float *dst);
extern "C" __global__ AICORE void TCVT_i16_to_ui32_1x128(__gm__ int16_t *src, __gm__ uint32_t *dst);
extern "C" __global__ AICORE void TCVT_i16_to_ui32_2x64(__gm__ int16_t *src, __gm__ uint32_t *dst);
extern "C" __global__ AICORE void TCVT_i16_to_ui32_4x32(__gm__ int16_t *src, __gm__ uint32_t *dst);
extern "C" __global__ AICORE void TCVT_i16_to_ui32_2x128(__gm__ int16_t *src, __gm__ uint32_t *dst);
extern "C" __global__ AICORE void TCVT_i16_to_ui32_4x65(__gm__ int16_t *src, __gm__ uint32_t *dst);
extern "C" __global__ AICORE void TCVT_i16_to_ui32_4x200(__gm__ int16_t *src, __gm__ uint32_t *dst);
extern "C" __global__ AICORE void TCVT_i16_to_ui32_1x129(__gm__ int16_t *src, __gm__ uint32_t *dst);
extern "C" __global__ AICORE void TCVT_i16_to_i32_1x128(__gm__ int16_t *src, __gm__ int32_t *dst);
extern "C" __global__ AICORE void TCVT_i16_to_i32_2x64(__gm__ int16_t *src, __gm__ int32_t *dst);
extern "C" __global__ AICORE void TCVT_i16_to_i32_4x32(__gm__ int16_t *src, __gm__ int32_t *dst);
extern "C" __global__ AICORE void TCVT_i16_to_i32_2x128(__gm__ int16_t *src, __gm__ int32_t *dst);
extern "C" __global__ AICORE void TCVT_i16_to_i32_4x65(__gm__ int16_t *src, __gm__ int32_t *dst);
extern "C" __global__ AICORE void TCVT_i16_to_i32_4x200(__gm__ int16_t *src, __gm__ int32_t *dst);
extern "C" __global__ AICORE void TCVT_i16_to_i32_1x129(__gm__ int16_t *src, __gm__ int32_t *dst);
extern "C" __global__ AICORE void TCVT_i32_to_f32_1x128(__gm__ int32_t *src, __gm__ float *dst);
extern "C" __global__ AICORE void TCVT_i32_to_f32_2x64(__gm__ int32_t *src, __gm__ float *dst);
extern "C" __global__ AICORE void TCVT_i32_to_f32_4x32(__gm__ int32_t *src, __gm__ float *dst);
extern "C" __global__ AICORE void TCVT_i32_to_f32_2x128(__gm__ int32_t *src, __gm__ float *dst);
extern "C" __global__ AICORE void TCVT_i32_to_f32_4x65(__gm__ int32_t *src, __gm__ float *dst);
extern "C" __global__ AICORE void TCVT_i32_to_f32_4x200(__gm__ int32_t *src, __gm__ float *dst);
extern "C" __global__ AICORE void TCVT_i32_to_f32_1x129(__gm__ int32_t *src, __gm__ float *dst);
extern "C" __global__ AICORE void TCVT_i32_to_i16_1x128(__gm__ int32_t *src, __gm__ int16_t *dst);
extern "C" __global__ AICORE void TCVT_i32_to_i16_2x64(__gm__ int32_t *src, __gm__ int16_t *dst);
extern "C" __global__ AICORE void TCVT_i32_to_i16_4x32(__gm__ int32_t *src, __gm__ int16_t *dst);
extern "C" __global__ AICORE void TCVT_i32_to_i16_2x128(__gm__ int32_t *src, __gm__ int16_t *dst);
extern "C" __global__ AICORE void TCVT_i32_to_i16_4x65(__gm__ int32_t *src, __gm__ int16_t *dst);
extern "C" __global__ AICORE void TCVT_i32_to_i16_4x200(__gm__ int32_t *src, __gm__ int16_t *dst);
extern "C" __global__ AICORE void TCVT_i32_to_i16_1x129(__gm__ int32_t *src, __gm__ int16_t *dst);
extern "C" __global__ AICORE void TCVT_i32_to_i64_1x128(__gm__ int32_t *src, __gm__ int64_t *dst);
extern "C" __global__ AICORE void TCVT_i32_to_i64_2x64(__gm__ int32_t *src, __gm__ int64_t *dst);
extern "C" __global__ AICORE void TCVT_i32_to_i64_4x32(__gm__ int32_t *src, __gm__ int64_t *dst);
extern "C" __global__ AICORE void TCVT_i32_to_i64_2x128(__gm__ int32_t *src, __gm__ int64_t *dst);
extern "C" __global__ AICORE void TCVT_i32_to_i64_4x65(__gm__ int32_t *src, __gm__ int64_t *dst);
extern "C" __global__ AICORE void TCVT_i32_to_i64_4x200(__gm__ int32_t *src, __gm__ int64_t *dst);
extern "C" __global__ AICORE void TCVT_i32_to_i64_1x129(__gm__ int32_t *src, __gm__ int64_t *dst);
extern "C" __global__ AICORE void TCVT_i32_to_ui8_1x128(__gm__ int32_t *src, __gm__ uint8_t *dst);
extern "C" __global__ AICORE void TCVT_i32_to_ui8_2x64(__gm__ int32_t *src, __gm__ uint8_t *dst);
extern "C" __global__ AICORE void TCVT_i32_to_ui8_4x32(__gm__ int32_t *src, __gm__ uint8_t *dst);
extern "C" __global__ AICORE void TCVT_i32_to_ui8_2x128(__gm__ int32_t *src, __gm__ uint8_t *dst);
extern "C" __global__ AICORE void TCVT_i32_to_ui8_4x65(__gm__ int32_t *src, __gm__ uint8_t *dst);
extern "C" __global__ AICORE void TCVT_i32_to_ui8_4x200(__gm__ int32_t *src, __gm__ uint8_t *dst);
extern "C" __global__ AICORE void TCVT_i32_to_ui8_1x129(__gm__ int32_t *src, __gm__ uint8_t *dst);
extern "C" __global__ AICORE void TCVT_i32_to_ui16_1x128(__gm__ int32_t *src, __gm__ uint16_t *dst);
extern "C" __global__ AICORE void TCVT_i32_to_ui16_2x64(__gm__ int32_t *src, __gm__ uint16_t *dst);
extern "C" __global__ AICORE void TCVT_i32_to_ui16_4x32(__gm__ int32_t *src, __gm__ uint16_t *dst);
extern "C" __global__ AICORE void TCVT_i32_to_ui16_2x128(__gm__ int32_t *src, __gm__ uint16_t *dst);
extern "C" __global__ AICORE void TCVT_i32_to_ui16_4x65(__gm__ int32_t *src, __gm__ uint16_t *dst);
extern "C" __global__ AICORE void TCVT_i32_to_ui16_4x200(__gm__ int32_t *src, __gm__ uint16_t *dst);
extern "C" __global__ AICORE void TCVT_i32_to_ui16_1x129(__gm__ int32_t *src, __gm__ uint16_t *dst);
extern "C" __global__ AICORE void TCVT_ui32_to_i16_1x128(__gm__ uint32_t *src, __gm__ int16_t *dst);
extern "C" __global__ AICORE void TCVT_ui32_to_i16_2x64(__gm__ uint32_t *src, __gm__ int16_t *dst);
extern "C" __global__ AICORE void TCVT_ui32_to_i16_4x32(__gm__ uint32_t *src, __gm__ int16_t *dst);
extern "C" __global__ AICORE void TCVT_ui32_to_i16_2x128(__gm__ uint32_t *src, __gm__ int16_t *dst);
extern "C" __global__ AICORE void TCVT_ui32_to_i16_4x65(__gm__ uint32_t *src, __gm__ int16_t *dst);
extern "C" __global__ AICORE void TCVT_ui32_to_i16_4x200(__gm__ uint32_t *src, __gm__ int16_t *dst);
extern "C" __global__ AICORE void TCVT_ui32_to_i16_1x129(__gm__ uint32_t *src, __gm__ int16_t *dst);
extern "C" __global__ AICORE void TCVT_ui32_to_ui16_1x128(__gm__ uint32_t *src, __gm__ uint16_t *dst);
extern "C" __global__ AICORE void TCVT_ui32_to_ui16_2x64(__gm__ uint32_t *src, __gm__ uint16_t *dst);
extern "C" __global__ AICORE void TCVT_ui32_to_ui16_4x32(__gm__ uint32_t *src, __gm__ uint16_t *dst);
extern "C" __global__ AICORE void TCVT_ui32_to_ui16_2x128(__gm__ uint32_t *src, __gm__ uint16_t *dst);
extern "C" __global__ AICORE void TCVT_ui32_to_ui16_4x65(__gm__ uint32_t *src, __gm__ uint16_t *dst);
extern "C" __global__ AICORE void TCVT_ui32_to_ui16_4x200(__gm__ uint32_t *src, __gm__ uint16_t *dst);
extern "C" __global__ AICORE void TCVT_ui32_to_ui16_1x129(__gm__ uint32_t *src, __gm__ uint16_t *dst);
extern "C" __global__ AICORE void TCVT_ui32_to_ui8_1x128(__gm__ uint32_t *src, __gm__ uint8_t *dst);
extern "C" __global__ AICORE void TCVT_ui32_to_ui8_2x64(__gm__ uint32_t *src, __gm__ uint8_t *dst);
extern "C" __global__ AICORE void TCVT_ui32_to_ui8_4x32(__gm__ uint32_t *src, __gm__ uint8_t *dst);
extern "C" __global__ AICORE void TCVT_ui32_to_ui8_2x128(__gm__ uint32_t *src, __gm__ uint8_t *dst);
extern "C" __global__ AICORE void TCVT_ui32_to_ui8_4x65(__gm__ uint32_t *src, __gm__ uint8_t *dst);
extern "C" __global__ AICORE void TCVT_ui32_to_ui8_4x200(__gm__ uint32_t *src, __gm__ uint8_t *dst);
extern "C" __global__ AICORE void TCVT_ui32_to_ui8_1x129(__gm__ uint32_t *src, __gm__ uint8_t *dst);
extern "C" __global__ AICORE void TCVT_i64_to_f32_1x128(__gm__ int64_t *src, __gm__ float *dst);
extern "C" __global__ AICORE void TCVT_i64_to_f32_2x64(__gm__ int64_t *src, __gm__ float *dst);
extern "C" __global__ AICORE void TCVT_i64_to_f32_4x32(__gm__ int64_t *src, __gm__ float *dst);
extern "C" __global__ AICORE void TCVT_i64_to_f32_2x128(__gm__ int64_t *src, __gm__ float *dst);
extern "C" __global__ AICORE void TCVT_i64_to_f32_4x65(__gm__ int64_t *src, __gm__ float *dst);
extern "C" __global__ AICORE void TCVT_i64_to_f32_4x200(__gm__ int64_t *src, __gm__ float *dst);
extern "C" __global__ AICORE void TCVT_i64_to_f32_1x129(__gm__ int64_t *src, __gm__ float *dst);
extern "C" __global__ AICORE void TCVT_i64_to_i32_1x128(__gm__ int64_t *src, __gm__ int32_t *dst);
extern "C" __global__ AICORE void TCVT_i64_to_i32_2x64(__gm__ int64_t *src, __gm__ int32_t *dst);
extern "C" __global__ AICORE void TCVT_i64_to_i32_4x32(__gm__ int64_t *src, __gm__ int32_t *dst);
extern "C" __global__ AICORE void TCVT_i64_to_i32_2x128(__gm__ int64_t *src, __gm__ int32_t *dst);
extern "C" __global__ AICORE void TCVT_i64_to_i32_4x65(__gm__ int64_t *src, __gm__ int32_t *dst);
extern "C" __global__ AICORE void TCVT_i64_to_i32_4x200(__gm__ int64_t *src, __gm__ int32_t *dst);
extern "C" __global__ AICORE void TCVT_i64_to_i32_1x129(__gm__ int64_t *src, __gm__ int32_t *dst);

void LaunchTCVT_f32_to_i32_rint_16x64(void *src, void *dst, void *stream) {
    TCVT_f32_to_i32_rint_16x64<<<1, nullptr, stream>>>((__gm__ float *)src, (__gm__ int32_t *)dst);
}

void LaunchTCVT_f32_to_i32_round_16x64(void *src, void *dst, void *stream) {
    TCVT_f32_to_i32_round_16x64<<<1, nullptr, stream>>>((__gm__ float *)src, (__gm__ int32_t *)dst);
}

void LaunchTCVT_i32_to_f32_rint_16x64(void *src, void *dst, void *stream) {
    TCVT_i32_to_f32_rint_16x64<<<1, nullptr, stream>>>((__gm__ int32_t *)src, (__gm__ float *)dst);
}

void LaunchTCVT_f32_to_f16_rint_16x64(void *src, void *dst, void *stream) {
    TCVT_f32_to_f16_rint_16x64<<<1, nullptr, stream>>>((__gm__ float *)src, (__gm__ uint16_t *)dst);
}

void LaunchTCVT_f16_to_f32_rint_16x64(void *src, void *dst, void *stream) {
    TCVT_f16_to_f32_rint_16x64<<<1, nullptr, stream>>>((__gm__ uint16_t *)src, (__gm__ float *)dst);
}

void LaunchTCVT_f32_to_f8e4m3_16x64(void *src, void *dst, void *stream) {
    TCVT_f32_to_f8e4m3_16x64<<<1, nullptr, stream>>>((__gm__ float *)src, (__gm__ float8_e4m3_t *)dst);
}

void LaunchTCVT_f32_to_f8e5m2_16x64(void *src, void *dst, void *stream) {
    TCVT_f32_to_f8e5m2_16x64<<<1, nullptr, stream>>>((__gm__ float *)src, (__gm__ float8_e5m2_t *)dst);
}

void LaunchTCVT_f32_to_hif8_16x64(void *src, void *dst, void *stream) {
    TCVT_f32_to_hif8_16x64<<<1, nullptr, stream>>>((__gm__ float *)src, (__gm__ hifloat8_t *)dst);
}

void LaunchTCVT_f16_to_hif8_16x64(void *src, void *dst, void *stream) {
    TCVT_f16_to_hif8_16x64<<<1, nullptr, stream>>>((__gm__ uint16_t *)src, (__gm__ hifloat8_t *)dst);
}

void LaunchTCVT_bf16_to_f4e1m2x2_16x64_to_16x32(void *src, void *dst, void *stream) {
    TCVT_bf16_to_f4e1m2x2_16x64_to_16x32<<<1, nullptr, stream>>>((__gm__ uint16_t *)src, (__gm__ float4_e1m2x2_t *)dst);
}

void LaunchTCVT_bf16_to_f4e2m1x2_16x64_to_16x32(void *src, void *dst, void *stream) {
    TCVT_bf16_to_f4e2m1x2_16x64_to_16x32<<<1, nullptr, stream>>>((__gm__ uint16_t *)src, (__gm__ float4_e2m1x2_t *)dst);
}

void LaunchTCVT_f32_to_f8e4m3_4x96(void *src, void *dst, void *stream) {
    TCVT_f32_to_f8e4m3_4x96<<<1, nullptr, stream>>>((__gm__ float *)src, (__gm__ float8_e4m3_t *)dst);
}

void LaunchTCVT_f32_to_hif8_4x96(void *src, void *dst, void *stream) {
    TCVT_f32_to_hif8_4x96<<<1, nullptr, stream>>>((__gm__ float *)src, (__gm__ hifloat8_t *)dst);
}

void LaunchTCVT_f32_to_f16_1x128(void *src, void *dst, void *stream) {
    TCVT_f32_to_f16_1x128<<<1, nullptr, stream>>>((__gm__ float *)src, (__gm__ uint16_t *)dst);
}

void LaunchTCVT_f32_to_f16_2x64(void *src, void *dst, void *stream) {
    TCVT_f32_to_f16_2x64<<<1, nullptr, stream>>>((__gm__ float *)src, (__gm__ uint16_t *)dst);
}

void LaunchTCVT_f32_to_f16_4x32(void *src, void *dst, void *stream) {
    TCVT_f32_to_f16_4x32<<<1, nullptr, stream>>>((__gm__ float *)src, (__gm__ uint16_t *)dst);
}

void LaunchTCVT_f32_to_f16_2x128(void *src, void *dst, void *stream) {
    TCVT_f32_to_f16_2x128<<<1, nullptr, stream>>>((__gm__ float *)src, (__gm__ uint16_t *)dst);
}

void LaunchTCVT_f32_to_f16_4x65(void *src, void *dst, void *stream) {
    TCVT_f32_to_f16_4x65<<<1, nullptr, stream>>>((__gm__ float *)src, (__gm__ uint16_t *)dst);
}

void LaunchTCVT_f32_to_f16_4x200(void *src, void *dst, void *stream) {
    TCVT_f32_to_f16_4x200<<<1, nullptr, stream>>>((__gm__ float *)src, (__gm__ uint16_t *)dst);
}

void LaunchTCVT_f32_to_f16_1x129(void *src, void *dst, void *stream) {
    TCVT_f32_to_f16_1x129<<<1, nullptr, stream>>>((__gm__ float *)src, (__gm__ uint16_t *)dst);
}

void LaunchTCVT_f32_to_bf16_1x128(void *src, void *dst, void *stream) {
    TCVT_f32_to_bf16_1x128<<<1, nullptr, stream>>>((__gm__ float *)src, (__gm__ uint16_t *)dst);
}

void LaunchTCVT_f32_to_bf16_2x64(void *src, void *dst, void *stream) {
    TCVT_f32_to_bf16_2x64<<<1, nullptr, stream>>>((__gm__ float *)src, (__gm__ uint16_t *)dst);
}

void LaunchTCVT_f32_to_bf16_4x32(void *src, void *dst, void *stream) {
    TCVT_f32_to_bf16_4x32<<<1, nullptr, stream>>>((__gm__ float *)src, (__gm__ uint16_t *)dst);
}

void LaunchTCVT_f32_to_bf16_2x128(void *src, void *dst, void *stream) {
    TCVT_f32_to_bf16_2x128<<<1, nullptr, stream>>>((__gm__ float *)src, (__gm__ uint16_t *)dst);
}

void LaunchTCVT_f32_to_bf16_4x65(void *src, void *dst, void *stream) {
    TCVT_f32_to_bf16_4x65<<<1, nullptr, stream>>>((__gm__ float *)src, (__gm__ uint16_t *)dst);
}

void LaunchTCVT_f32_to_bf16_4x200(void *src, void *dst, void *stream) {
    TCVT_f32_to_bf16_4x200<<<1, nullptr, stream>>>((__gm__ float *)src, (__gm__ uint16_t *)dst);
}

void LaunchTCVT_f32_to_bf16_1x129(void *src, void *dst, void *stream) {
    TCVT_f32_to_bf16_1x129<<<1, nullptr, stream>>>((__gm__ float *)src, (__gm__ uint16_t *)dst);
}

void LaunchTCVT_f32_to_i16_1x128(void *src, void *dst, void *stream) {
    TCVT_f32_to_i16_1x128<<<1, nullptr, stream>>>((__gm__ float *)src, (__gm__ int16_t *)dst);
}

void LaunchTCVT_f32_to_i16_2x64(void *src, void *dst, void *stream) {
    TCVT_f32_to_i16_2x64<<<1, nullptr, stream>>>((__gm__ float *)src, (__gm__ int16_t *)dst);
}

void LaunchTCVT_f32_to_i16_4x32(void *src, void *dst, void *stream) {
    TCVT_f32_to_i16_4x32<<<1, nullptr, stream>>>((__gm__ float *)src, (__gm__ int16_t *)dst);
}

void LaunchTCVT_f32_to_i16_2x128(void *src, void *dst, void *stream) {
    TCVT_f32_to_i16_2x128<<<1, nullptr, stream>>>((__gm__ float *)src, (__gm__ int16_t *)dst);
}

void LaunchTCVT_f32_to_i16_4x65(void *src, void *dst, void *stream) {
    TCVT_f32_to_i16_4x65<<<1, nullptr, stream>>>((__gm__ float *)src, (__gm__ int16_t *)dst);
}

void LaunchTCVT_f32_to_i16_4x200(void *src, void *dst, void *stream) {
    TCVT_f32_to_i16_4x200<<<1, nullptr, stream>>>((__gm__ float *)src, (__gm__ int16_t *)dst);
}

void LaunchTCVT_f32_to_i16_1x129(void *src, void *dst, void *stream) {
    TCVT_f32_to_i16_1x129<<<1, nullptr, stream>>>((__gm__ float *)src, (__gm__ int16_t *)dst);
}

void LaunchTCVT_f32_to_i32_1x128(void *src, void *dst, void *stream) {
    TCVT_f32_to_i32_1x128<<<1, nullptr, stream>>>((__gm__ float *)src, (__gm__ int32_t *)dst);
}

void LaunchTCVT_f32_to_i32_2x64(void *src, void *dst, void *stream) {
    TCVT_f32_to_i32_2x64<<<1, nullptr, stream>>>((__gm__ float *)src, (__gm__ int32_t *)dst);
}

void LaunchTCVT_f32_to_i32_4x32(void *src, void *dst, void *stream) {
    TCVT_f32_to_i32_4x32<<<1, nullptr, stream>>>((__gm__ float *)src, (__gm__ int32_t *)dst);
}

void LaunchTCVT_f32_to_i32_2x128(void *src, void *dst, void *stream) {
    TCVT_f32_to_i32_2x128<<<1, nullptr, stream>>>((__gm__ float *)src, (__gm__ int32_t *)dst);
}

void LaunchTCVT_f32_to_i32_4x65(void *src, void *dst, void *stream) {
    TCVT_f32_to_i32_4x65<<<1, nullptr, stream>>>((__gm__ float *)src, (__gm__ int32_t *)dst);
}

void LaunchTCVT_f32_to_i32_4x200(void *src, void *dst, void *stream) {
    TCVT_f32_to_i32_4x200<<<1, nullptr, stream>>>((__gm__ float *)src, (__gm__ int32_t *)dst);
}

void LaunchTCVT_f32_to_i32_1x129(void *src, void *dst, void *stream) {
    TCVT_f32_to_i32_1x129<<<1, nullptr, stream>>>((__gm__ float *)src, (__gm__ int32_t *)dst);
}

void LaunchTCVT_f32_to_i64_1x128(void *src, void *dst, void *stream) {
    TCVT_f32_to_i64_1x128<<<1, nullptr, stream>>>((__gm__ float *)src, (__gm__ int64_t *)dst);
}

void LaunchTCVT_f32_to_i64_2x64(void *src, void *dst, void *stream) {
    TCVT_f32_to_i64_2x64<<<1, nullptr, stream>>>((__gm__ float *)src, (__gm__ int64_t *)dst);
}

void LaunchTCVT_f32_to_i64_4x32(void *src, void *dst, void *stream) {
    TCVT_f32_to_i64_4x32<<<1, nullptr, stream>>>((__gm__ float *)src, (__gm__ int64_t *)dst);
}

void LaunchTCVT_f32_to_i64_2x128(void *src, void *dst, void *stream) {
    TCVT_f32_to_i64_2x128<<<1, nullptr, stream>>>((__gm__ float *)src, (__gm__ int64_t *)dst);
}

void LaunchTCVT_f32_to_i64_4x65(void *src, void *dst, void *stream) {
    TCVT_f32_to_i64_4x65<<<1, nullptr, stream>>>((__gm__ float *)src, (__gm__ int64_t *)dst);
}

void LaunchTCVT_f32_to_i64_4x200(void *src, void *dst, void *stream) {
    TCVT_f32_to_i64_4x200<<<1, nullptr, stream>>>((__gm__ float *)src, (__gm__ int64_t *)dst);
}

void LaunchTCVT_f32_to_i64_1x129(void *src, void *dst, void *stream) {
    TCVT_f32_to_i64_1x129<<<1, nullptr, stream>>>((__gm__ float *)src, (__gm__ int64_t *)dst);
}

void LaunchTCVT_f32_to_f32_1x128(void *src, void *dst, void *stream) {
    TCVT_f32_to_f32_1x128<<<1, nullptr, stream>>>((__gm__ float *)src, (__gm__ float *)dst);
}

void LaunchTCVT_f32_to_f32_2x64(void *src, void *dst, void *stream) {
    TCVT_f32_to_f32_2x64<<<1, nullptr, stream>>>((__gm__ float *)src, (__gm__ float *)dst);
}

void LaunchTCVT_f32_to_f32_4x32(void *src, void *dst, void *stream) {
    TCVT_f32_to_f32_4x32<<<1, nullptr, stream>>>((__gm__ float *)src, (__gm__ float *)dst);
}

void LaunchTCVT_f32_to_f32_2x128(void *src, void *dst, void *stream) {
    TCVT_f32_to_f32_2x128<<<1, nullptr, stream>>>((__gm__ float *)src, (__gm__ float *)dst);
}

void LaunchTCVT_f32_to_f32_4x65(void *src, void *dst, void *stream) {
    TCVT_f32_to_f32_4x65<<<1, nullptr, stream>>>((__gm__ float *)src, (__gm__ float *)dst);
}

void LaunchTCVT_f32_to_f32_4x200(void *src, void *dst, void *stream) {
    TCVT_f32_to_f32_4x200<<<1, nullptr, stream>>>((__gm__ float *)src, (__gm__ float *)dst);
}

void LaunchTCVT_f32_to_f32_1x129(void *src, void *dst, void *stream) {
    TCVT_f32_to_f32_1x129<<<1, nullptr, stream>>>((__gm__ float *)src, (__gm__ float *)dst);
}

void LaunchTCVT_f16_to_f32_1x128(void *src, void *dst, void *stream) {
    TCVT_f16_to_f32_1x128<<<1, nullptr, stream>>>((__gm__ uint16_t *)src, (__gm__ float *)dst);
}

void LaunchTCVT_f16_to_f32_2x64(void *src, void *dst, void *stream) {
    TCVT_f16_to_f32_2x64<<<1, nullptr, stream>>>((__gm__ uint16_t *)src, (__gm__ float *)dst);
}

void LaunchTCVT_f16_to_f32_4x32(void *src, void *dst, void *stream) {
    TCVT_f16_to_f32_4x32<<<1, nullptr, stream>>>((__gm__ uint16_t *)src, (__gm__ float *)dst);
}

void LaunchTCVT_f16_to_f32_2x128(void *src, void *dst, void *stream) {
    TCVT_f16_to_f32_2x128<<<1, nullptr, stream>>>((__gm__ uint16_t *)src, (__gm__ float *)dst);
}

void LaunchTCVT_f16_to_f32_4x65(void *src, void *dst, void *stream) {
    TCVT_f16_to_f32_4x65<<<1, nullptr, stream>>>((__gm__ uint16_t *)src, (__gm__ float *)dst);
}

void LaunchTCVT_f16_to_f32_4x200(void *src, void *dst, void *stream) {
    TCVT_f16_to_f32_4x200<<<1, nullptr, stream>>>((__gm__ uint16_t *)src, (__gm__ float *)dst);
}

void LaunchTCVT_f16_to_f32_1x129(void *src, void *dst, void *stream) {
    TCVT_f16_to_f32_1x129<<<1, nullptr, stream>>>((__gm__ uint16_t *)src, (__gm__ float *)dst);
}

void LaunchTCVT_f16_to_i32_1x128(void *src, void *dst, void *stream) {
    TCVT_f16_to_i32_1x128<<<1, nullptr, stream>>>((__gm__ uint16_t *)src, (__gm__ int32_t *)dst);
}

void LaunchTCVT_f16_to_i32_2x64(void *src, void *dst, void *stream) {
    TCVT_f16_to_i32_2x64<<<1, nullptr, stream>>>((__gm__ uint16_t *)src, (__gm__ int32_t *)dst);
}

void LaunchTCVT_f16_to_i32_4x32(void *src, void *dst, void *stream) {
    TCVT_f16_to_i32_4x32<<<1, nullptr, stream>>>((__gm__ uint16_t *)src, (__gm__ int32_t *)dst);
}

void LaunchTCVT_f16_to_i32_2x128(void *src, void *dst, void *stream) {
    TCVT_f16_to_i32_2x128<<<1, nullptr, stream>>>((__gm__ uint16_t *)src, (__gm__ int32_t *)dst);
}

void LaunchTCVT_f16_to_i32_4x65(void *src, void *dst, void *stream) {
    TCVT_f16_to_i32_4x65<<<1, nullptr, stream>>>((__gm__ uint16_t *)src, (__gm__ int32_t *)dst);
}

void LaunchTCVT_f16_to_i32_4x200(void *src, void *dst, void *stream) {
    TCVT_f16_to_i32_4x200<<<1, nullptr, stream>>>((__gm__ uint16_t *)src, (__gm__ int32_t *)dst);
}

void LaunchTCVT_f16_to_i32_1x129(void *src, void *dst, void *stream) {
    TCVT_f16_to_i32_1x129<<<1, nullptr, stream>>>((__gm__ uint16_t *)src, (__gm__ int32_t *)dst);
}

void LaunchTCVT_f16_to_i16_1x128(void *src, void *dst, void *stream) {
    TCVT_f16_to_i16_1x128<<<1, nullptr, stream>>>((__gm__ uint16_t *)src, (__gm__ int16_t *)dst);
}

void LaunchTCVT_f16_to_i16_2x64(void *src, void *dst, void *stream) {
    TCVT_f16_to_i16_2x64<<<1, nullptr, stream>>>((__gm__ uint16_t *)src, (__gm__ int16_t *)dst);
}

void LaunchTCVT_f16_to_i16_4x32(void *src, void *dst, void *stream) {
    TCVT_f16_to_i16_4x32<<<1, nullptr, stream>>>((__gm__ uint16_t *)src, (__gm__ int16_t *)dst);
}

void LaunchTCVT_f16_to_i16_2x128(void *src, void *dst, void *stream) {
    TCVT_f16_to_i16_2x128<<<1, nullptr, stream>>>((__gm__ uint16_t *)src, (__gm__ int16_t *)dst);
}

void LaunchTCVT_f16_to_i16_4x65(void *src, void *dst, void *stream) {
    TCVT_f16_to_i16_4x65<<<1, nullptr, stream>>>((__gm__ uint16_t *)src, (__gm__ int16_t *)dst);
}

void LaunchTCVT_f16_to_i16_4x200(void *src, void *dst, void *stream) {
    TCVT_f16_to_i16_4x200<<<1, nullptr, stream>>>((__gm__ uint16_t *)src, (__gm__ int16_t *)dst);
}

void LaunchTCVT_f16_to_i16_1x129(void *src, void *dst, void *stream) {
    TCVT_f16_to_i16_1x129<<<1, nullptr, stream>>>((__gm__ uint16_t *)src, (__gm__ int16_t *)dst);
}

void LaunchTCVT_f16_to_si8_1x128(void *src, void *dst, void *stream) {
    TCVT_f16_to_si8_1x128<<<1, nullptr, stream>>>((__gm__ uint16_t *)src, (__gm__ int8_t *)dst);
}

void LaunchTCVT_f16_to_si8_2x64(void *src, void *dst, void *stream) {
    TCVT_f16_to_si8_2x64<<<1, nullptr, stream>>>((__gm__ uint16_t *)src, (__gm__ int8_t *)dst);
}

void LaunchTCVT_f16_to_si8_4x32(void *src, void *dst, void *stream) {
    TCVT_f16_to_si8_4x32<<<1, nullptr, stream>>>((__gm__ uint16_t *)src, (__gm__ int8_t *)dst);
}

void LaunchTCVT_f16_to_si8_2x128(void *src, void *dst, void *stream) {
    TCVT_f16_to_si8_2x128<<<1, nullptr, stream>>>((__gm__ uint16_t *)src, (__gm__ int8_t *)dst);
}

void LaunchTCVT_f16_to_si8_4x65(void *src, void *dst, void *stream) {
    TCVT_f16_to_si8_4x65<<<1, nullptr, stream>>>((__gm__ uint16_t *)src, (__gm__ int8_t *)dst);
}

void LaunchTCVT_f16_to_si8_4x200(void *src, void *dst, void *stream) {
    TCVT_f16_to_si8_4x200<<<1, nullptr, stream>>>((__gm__ uint16_t *)src, (__gm__ int8_t *)dst);
}

void LaunchTCVT_f16_to_si8_1x129(void *src, void *dst, void *stream) {
    TCVT_f16_to_si8_1x129<<<1, nullptr, stream>>>((__gm__ uint16_t *)src, (__gm__ int8_t *)dst);
}

void LaunchTCVT_f16_to_ui8_1x128(void *src, void *dst, void *stream) {
    TCVT_f16_to_ui8_1x128<<<1, nullptr, stream>>>((__gm__ uint16_t *)src, (__gm__ uint8_t *)dst);
}

void LaunchTCVT_f16_to_ui8_2x64(void *src, void *dst, void *stream) {
    TCVT_f16_to_ui8_2x64<<<1, nullptr, stream>>>((__gm__ uint16_t *)src, (__gm__ uint8_t *)dst);
}

void LaunchTCVT_f16_to_ui8_4x32(void *src, void *dst, void *stream) {
    TCVT_f16_to_ui8_4x32<<<1, nullptr, stream>>>((__gm__ uint16_t *)src, (__gm__ uint8_t *)dst);
}

void LaunchTCVT_f16_to_ui8_2x128(void *src, void *dst, void *stream) {
    TCVT_f16_to_ui8_2x128<<<1, nullptr, stream>>>((__gm__ uint16_t *)src, (__gm__ uint8_t *)dst);
}

void LaunchTCVT_f16_to_ui8_4x65(void *src, void *dst, void *stream) {
    TCVT_f16_to_ui8_4x65<<<1, nullptr, stream>>>((__gm__ uint16_t *)src, (__gm__ uint8_t *)dst);
}

void LaunchTCVT_f16_to_ui8_4x200(void *src, void *dst, void *stream) {
    TCVT_f16_to_ui8_4x200<<<1, nullptr, stream>>>((__gm__ uint16_t *)src, (__gm__ uint8_t *)dst);
}

void LaunchTCVT_f16_to_ui8_1x129(void *src, void *dst, void *stream) {
    TCVT_f16_to_ui8_1x129<<<1, nullptr, stream>>>((__gm__ uint16_t *)src, (__gm__ uint8_t *)dst);
}

void LaunchTCVT_bf16_to_f32_1x128(void *src, void *dst, void *stream) {
    TCVT_bf16_to_f32_1x128<<<1, nullptr, stream>>>((__gm__ uint16_t *)src, (__gm__ float *)dst);
}

void LaunchTCVT_bf16_to_f32_2x64(void *src, void *dst, void *stream) {
    TCVT_bf16_to_f32_2x64<<<1, nullptr, stream>>>((__gm__ uint16_t *)src, (__gm__ float *)dst);
}

void LaunchTCVT_bf16_to_f32_4x32(void *src, void *dst, void *stream) {
    TCVT_bf16_to_f32_4x32<<<1, nullptr, stream>>>((__gm__ uint16_t *)src, (__gm__ float *)dst);
}

void LaunchTCVT_bf16_to_f32_2x128(void *src, void *dst, void *stream) {
    TCVT_bf16_to_f32_2x128<<<1, nullptr, stream>>>((__gm__ uint16_t *)src, (__gm__ float *)dst);
}

void LaunchTCVT_bf16_to_f32_4x65(void *src, void *dst, void *stream) {
    TCVT_bf16_to_f32_4x65<<<1, nullptr, stream>>>((__gm__ uint16_t *)src, (__gm__ float *)dst);
}

void LaunchTCVT_bf16_to_f32_4x200(void *src, void *dst, void *stream) {
    TCVT_bf16_to_f32_4x200<<<1, nullptr, stream>>>((__gm__ uint16_t *)src, (__gm__ float *)dst);
}

void LaunchTCVT_bf16_to_f32_1x129(void *src, void *dst, void *stream) {
    TCVT_bf16_to_f32_1x129<<<1, nullptr, stream>>>((__gm__ uint16_t *)src, (__gm__ float *)dst);
}

void LaunchTCVT_bf16_to_f16_1x128(void *src, void *dst, void *stream) {
    TCVT_bf16_to_f16_1x128<<<1, nullptr, stream>>>((__gm__ uint16_t *)src, (__gm__ uint16_t *)dst);
}

void LaunchTCVT_bf16_to_f16_2x64(void *src, void *dst, void *stream) {
    TCVT_bf16_to_f16_2x64<<<1, nullptr, stream>>>((__gm__ uint16_t *)src, (__gm__ uint16_t *)dst);
}

void LaunchTCVT_bf16_to_f16_4x32(void *src, void *dst, void *stream) {
    TCVT_bf16_to_f16_4x32<<<1, nullptr, stream>>>((__gm__ uint16_t *)src, (__gm__ uint16_t *)dst);
}

void LaunchTCVT_bf16_to_f16_2x128(void *src, void *dst, void *stream) {
    TCVT_bf16_to_f16_2x128<<<1, nullptr, stream>>>((__gm__ uint16_t *)src, (__gm__ uint16_t *)dst);
}

void LaunchTCVT_bf16_to_f16_4x65(void *src, void *dst, void *stream) {
    TCVT_bf16_to_f16_4x65<<<1, nullptr, stream>>>((__gm__ uint16_t *)src, (__gm__ uint16_t *)dst);
}

void LaunchTCVT_bf16_to_f16_4x200(void *src, void *dst, void *stream) {
    TCVT_bf16_to_f16_4x200<<<1, nullptr, stream>>>((__gm__ uint16_t *)src, (__gm__ uint16_t *)dst);
}

void LaunchTCVT_bf16_to_f16_1x129(void *src, void *dst, void *stream) {
    TCVT_bf16_to_f16_1x129<<<1, nullptr, stream>>>((__gm__ uint16_t *)src, (__gm__ uint16_t *)dst);
}

void LaunchTCVT_bf16_to_i32_1x128(void *src, void *dst, void *stream) {
    TCVT_bf16_to_i32_1x128<<<1, nullptr, stream>>>((__gm__ uint16_t *)src, (__gm__ int32_t *)dst);
}

void LaunchTCVT_bf16_to_i32_2x64(void *src, void *dst, void *stream) {
    TCVT_bf16_to_i32_2x64<<<1, nullptr, stream>>>((__gm__ uint16_t *)src, (__gm__ int32_t *)dst);
}

void LaunchTCVT_bf16_to_i32_4x32(void *src, void *dst, void *stream) {
    TCVT_bf16_to_i32_4x32<<<1, nullptr, stream>>>((__gm__ uint16_t *)src, (__gm__ int32_t *)dst);
}

void LaunchTCVT_bf16_to_i32_2x128(void *src, void *dst, void *stream) {
    TCVT_bf16_to_i32_2x128<<<1, nullptr, stream>>>((__gm__ uint16_t *)src, (__gm__ int32_t *)dst);
}

void LaunchTCVT_bf16_to_i32_4x65(void *src, void *dst, void *stream) {
    TCVT_bf16_to_i32_4x65<<<1, nullptr, stream>>>((__gm__ uint16_t *)src, (__gm__ int32_t *)dst);
}

void LaunchTCVT_bf16_to_i32_4x200(void *src, void *dst, void *stream) {
    TCVT_bf16_to_i32_4x200<<<1, nullptr, stream>>>((__gm__ uint16_t *)src, (__gm__ int32_t *)dst);
}

void LaunchTCVT_bf16_to_i32_1x129(void *src, void *dst, void *stream) {
    TCVT_bf16_to_i32_1x129<<<1, nullptr, stream>>>((__gm__ uint16_t *)src, (__gm__ int32_t *)dst);
}

void LaunchTCVT_ui8_to_f16_1x128(void *src, void *dst, void *stream) {
    TCVT_ui8_to_f16_1x128<<<1, nullptr, stream>>>((__gm__ uint8_t *)src, (__gm__ uint16_t *)dst);
}

void LaunchTCVT_ui8_to_f16_2x64(void *src, void *dst, void *stream) {
    TCVT_ui8_to_f16_2x64<<<1, nullptr, stream>>>((__gm__ uint8_t *)src, (__gm__ uint16_t *)dst);
}

void LaunchTCVT_ui8_to_f16_4x32(void *src, void *dst, void *stream) {
    TCVT_ui8_to_f16_4x32<<<1, nullptr, stream>>>((__gm__ uint8_t *)src, (__gm__ uint16_t *)dst);
}

void LaunchTCVT_ui8_to_f16_2x128(void *src, void *dst, void *stream) {
    TCVT_ui8_to_f16_2x128<<<1, nullptr, stream>>>((__gm__ uint8_t *)src, (__gm__ uint16_t *)dst);
}

void LaunchTCVT_ui8_to_f16_4x65(void *src, void *dst, void *stream) {
    TCVT_ui8_to_f16_4x65<<<1, nullptr, stream>>>((__gm__ uint8_t *)src, (__gm__ uint16_t *)dst);
}

void LaunchTCVT_ui8_to_f16_4x200(void *src, void *dst, void *stream) {
    TCVT_ui8_to_f16_4x200<<<1, nullptr, stream>>>((__gm__ uint8_t *)src, (__gm__ uint16_t *)dst);
}

void LaunchTCVT_ui8_to_f16_1x129(void *src, void *dst, void *stream) {
    TCVT_ui8_to_f16_1x129<<<1, nullptr, stream>>>((__gm__ uint8_t *)src, (__gm__ uint16_t *)dst);
}

void LaunchTCVT_ui8_to_ui16_1x128(void *src, void *dst, void *stream) {
    TCVT_ui8_to_ui16_1x128<<<1, nullptr, stream>>>((__gm__ uint8_t *)src, (__gm__ uint16_t *)dst);
}

void LaunchTCVT_ui8_to_ui16_2x64(void *src, void *dst, void *stream) {
    TCVT_ui8_to_ui16_2x64<<<1, nullptr, stream>>>((__gm__ uint8_t *)src, (__gm__ uint16_t *)dst);
}

void LaunchTCVT_ui8_to_ui16_4x32(void *src, void *dst, void *stream) {
    TCVT_ui8_to_ui16_4x32<<<1, nullptr, stream>>>((__gm__ uint8_t *)src, (__gm__ uint16_t *)dst);
}

void LaunchTCVT_ui8_to_ui16_2x128(void *src, void *dst, void *stream) {
    TCVT_ui8_to_ui16_2x128<<<1, nullptr, stream>>>((__gm__ uint8_t *)src, (__gm__ uint16_t *)dst);
}

void LaunchTCVT_ui8_to_ui16_4x65(void *src, void *dst, void *stream) {
    TCVT_ui8_to_ui16_4x65<<<1, nullptr, stream>>>((__gm__ uint8_t *)src, (__gm__ uint16_t *)dst);
}

void LaunchTCVT_ui8_to_ui16_4x200(void *src, void *dst, void *stream) {
    TCVT_ui8_to_ui16_4x200<<<1, nullptr, stream>>>((__gm__ uint8_t *)src, (__gm__ uint16_t *)dst);
}

void LaunchTCVT_ui8_to_ui16_1x129(void *src, void *dst, void *stream) {
    TCVT_ui8_to_ui16_1x129<<<1, nullptr, stream>>>((__gm__ uint8_t *)src, (__gm__ uint16_t *)dst);
}

void LaunchTCVT_si8_to_f16_1x128(void *src, void *dst, void *stream) {
    TCVT_si8_to_f16_1x128<<<1, nullptr, stream>>>((__gm__ int8_t *)src, (__gm__ uint16_t *)dst);
}

void LaunchTCVT_si8_to_f16_2x64(void *src, void *dst, void *stream) {
    TCVT_si8_to_f16_2x64<<<1, nullptr, stream>>>((__gm__ int8_t *)src, (__gm__ uint16_t *)dst);
}

void LaunchTCVT_si8_to_f16_4x32(void *src, void *dst, void *stream) {
    TCVT_si8_to_f16_4x32<<<1, nullptr, stream>>>((__gm__ int8_t *)src, (__gm__ uint16_t *)dst);
}

void LaunchTCVT_si8_to_f16_2x128(void *src, void *dst, void *stream) {
    TCVT_si8_to_f16_2x128<<<1, nullptr, stream>>>((__gm__ int8_t *)src, (__gm__ uint16_t *)dst);
}

void LaunchTCVT_si8_to_f16_4x65(void *src, void *dst, void *stream) {
    TCVT_si8_to_f16_4x65<<<1, nullptr, stream>>>((__gm__ int8_t *)src, (__gm__ uint16_t *)dst);
}

void LaunchTCVT_si8_to_f16_4x200(void *src, void *dst, void *stream) {
    TCVT_si8_to_f16_4x200<<<1, nullptr, stream>>>((__gm__ int8_t *)src, (__gm__ uint16_t *)dst);
}

void LaunchTCVT_si8_to_f16_1x129(void *src, void *dst, void *stream) {
    TCVT_si8_to_f16_1x129<<<1, nullptr, stream>>>((__gm__ int8_t *)src, (__gm__ uint16_t *)dst);
}

void LaunchTCVT_si8_to_si16_1x128(void *src, void *dst, void *stream) {
    TCVT_si8_to_si16_1x128<<<1, nullptr, stream>>>((__gm__ int8_t *)src, (__gm__ int16_t *)dst);
}

void LaunchTCVT_si8_to_si16_2x64(void *src, void *dst, void *stream) {
    TCVT_si8_to_si16_2x64<<<1, nullptr, stream>>>((__gm__ int8_t *)src, (__gm__ int16_t *)dst);
}

void LaunchTCVT_si8_to_si16_4x32(void *src, void *dst, void *stream) {
    TCVT_si8_to_si16_4x32<<<1, nullptr, stream>>>((__gm__ int8_t *)src, (__gm__ int16_t *)dst);
}

void LaunchTCVT_si8_to_si16_2x128(void *src, void *dst, void *stream) {
    TCVT_si8_to_si16_2x128<<<1, nullptr, stream>>>((__gm__ int8_t *)src, (__gm__ int16_t *)dst);
}

void LaunchTCVT_si8_to_si16_4x65(void *src, void *dst, void *stream) {
    TCVT_si8_to_si16_4x65<<<1, nullptr, stream>>>((__gm__ int8_t *)src, (__gm__ int16_t *)dst);
}

void LaunchTCVT_si8_to_si16_4x200(void *src, void *dst, void *stream) {
    TCVT_si8_to_si16_4x200<<<1, nullptr, stream>>>((__gm__ int8_t *)src, (__gm__ int16_t *)dst);
}

void LaunchTCVT_si8_to_si16_1x129(void *src, void *dst, void *stream) {
    TCVT_si8_to_si16_1x129<<<1, nullptr, stream>>>((__gm__ int8_t *)src, (__gm__ int16_t *)dst);
}

void LaunchTCVT_si8_to_i32_1x128(void *src, void *dst, void *stream) {
    TCVT_si8_to_i32_1x128<<<1, nullptr, stream>>>((__gm__ int8_t *)src, (__gm__ int32_t *)dst);
}

void LaunchTCVT_si8_to_i32_2x64(void *src, void *dst, void *stream) {
    TCVT_si8_to_i32_2x64<<<1, nullptr, stream>>>((__gm__ int8_t *)src, (__gm__ int32_t *)dst);
}

void LaunchTCVT_si8_to_i32_4x32(void *src, void *dst, void *stream) {
    TCVT_si8_to_i32_4x32<<<1, nullptr, stream>>>((__gm__ int8_t *)src, (__gm__ int32_t *)dst);
}

void LaunchTCVT_si8_to_i32_2x128(void *src, void *dst, void *stream) {
    TCVT_si8_to_i32_2x128<<<1, nullptr, stream>>>((__gm__ int8_t *)src, (__gm__ int32_t *)dst);
}

void LaunchTCVT_si8_to_i32_4x65(void *src, void *dst, void *stream) {
    TCVT_si8_to_i32_4x65<<<1, nullptr, stream>>>((__gm__ int8_t *)src, (__gm__ int32_t *)dst);
}

void LaunchTCVT_si8_to_i32_4x200(void *src, void *dst, void *stream) {
    TCVT_si8_to_i32_4x200<<<1, nullptr, stream>>>((__gm__ int8_t *)src, (__gm__ int32_t *)dst);
}

void LaunchTCVT_si8_to_i32_1x129(void *src, void *dst, void *stream) {
    TCVT_si8_to_i32_1x129<<<1, nullptr, stream>>>((__gm__ int8_t *)src, (__gm__ int32_t *)dst);
}

void LaunchTCVT_i16_to_ui8_1x128(void *src, void *dst, void *stream) {
    TCVT_i16_to_ui8_1x128<<<1, nullptr, stream>>>((__gm__ int16_t *)src, (__gm__ uint8_t *)dst);
}

void LaunchTCVT_i16_to_ui8_2x64(void *src, void *dst, void *stream) {
    TCVT_i16_to_ui8_2x64<<<1, nullptr, stream>>>((__gm__ int16_t *)src, (__gm__ uint8_t *)dst);
}

void LaunchTCVT_i16_to_ui8_4x32(void *src, void *dst, void *stream) {
    TCVT_i16_to_ui8_4x32<<<1, nullptr, stream>>>((__gm__ int16_t *)src, (__gm__ uint8_t *)dst);
}

void LaunchTCVT_i16_to_ui8_2x128(void *src, void *dst, void *stream) {
    TCVT_i16_to_ui8_2x128<<<1, nullptr, stream>>>((__gm__ int16_t *)src, (__gm__ uint8_t *)dst);
}

void LaunchTCVT_i16_to_ui8_4x65(void *src, void *dst, void *stream) {
    TCVT_i16_to_ui8_4x65<<<1, nullptr, stream>>>((__gm__ int16_t *)src, (__gm__ uint8_t *)dst);
}

void LaunchTCVT_i16_to_ui8_4x200(void *src, void *dst, void *stream) {
    TCVT_i16_to_ui8_4x200<<<1, nullptr, stream>>>((__gm__ int16_t *)src, (__gm__ uint8_t *)dst);
}

void LaunchTCVT_i16_to_ui8_1x129(void *src, void *dst, void *stream) {
    TCVT_i16_to_ui8_1x129<<<1, nullptr, stream>>>((__gm__ int16_t *)src, (__gm__ uint8_t *)dst);
}

void LaunchTCVT_i16_to_f16_1x128(void *src, void *dst, void *stream) {
    TCVT_i16_to_f16_1x128<<<1, nullptr, stream>>>((__gm__ int16_t *)src, (__gm__ uint16_t *)dst);
}

void LaunchTCVT_i16_to_f16_2x64(void *src, void *dst, void *stream) {
    TCVT_i16_to_f16_2x64<<<1, nullptr, stream>>>((__gm__ int16_t *)src, (__gm__ uint16_t *)dst);
}

void LaunchTCVT_i16_to_f16_4x32(void *src, void *dst, void *stream) {
    TCVT_i16_to_f16_4x32<<<1, nullptr, stream>>>((__gm__ int16_t *)src, (__gm__ uint16_t *)dst);
}

void LaunchTCVT_i16_to_f16_2x128(void *src, void *dst, void *stream) {
    TCVT_i16_to_f16_2x128<<<1, nullptr, stream>>>((__gm__ int16_t *)src, (__gm__ uint16_t *)dst);
}

void LaunchTCVT_i16_to_f16_4x65(void *src, void *dst, void *stream) {
    TCVT_i16_to_f16_4x65<<<1, nullptr, stream>>>((__gm__ int16_t *)src, (__gm__ uint16_t *)dst);
}

void LaunchTCVT_i16_to_f16_4x200(void *src, void *dst, void *stream) {
    TCVT_i16_to_f16_4x200<<<1, nullptr, stream>>>((__gm__ int16_t *)src, (__gm__ uint16_t *)dst);
}

void LaunchTCVT_i16_to_f16_1x129(void *src, void *dst, void *stream) {
    TCVT_i16_to_f16_1x129<<<1, nullptr, stream>>>((__gm__ int16_t *)src, (__gm__ uint16_t *)dst);
}

void LaunchTCVT_i16_to_f32_1x128(void *src, void *dst, void *stream) {
    TCVT_i16_to_f32_1x128<<<1, nullptr, stream>>>((__gm__ int16_t *)src, (__gm__ float *)dst);
}

void LaunchTCVT_i16_to_f32_2x64(void *src, void *dst, void *stream) {
    TCVT_i16_to_f32_2x64<<<1, nullptr, stream>>>((__gm__ int16_t *)src, (__gm__ float *)dst);
}

void LaunchTCVT_i16_to_f32_4x32(void *src, void *dst, void *stream) {
    TCVT_i16_to_f32_4x32<<<1, nullptr, stream>>>((__gm__ int16_t *)src, (__gm__ float *)dst);
}

void LaunchTCVT_i16_to_f32_2x128(void *src, void *dst, void *stream) {
    TCVT_i16_to_f32_2x128<<<1, nullptr, stream>>>((__gm__ int16_t *)src, (__gm__ float *)dst);
}

void LaunchTCVT_i16_to_f32_4x65(void *src, void *dst, void *stream) {
    TCVT_i16_to_f32_4x65<<<1, nullptr, stream>>>((__gm__ int16_t *)src, (__gm__ float *)dst);
}

void LaunchTCVT_i16_to_f32_4x200(void *src, void *dst, void *stream) {
    TCVT_i16_to_f32_4x200<<<1, nullptr, stream>>>((__gm__ int16_t *)src, (__gm__ float *)dst);
}

void LaunchTCVT_i16_to_f32_1x129(void *src, void *dst, void *stream) {
    TCVT_i16_to_f32_1x129<<<1, nullptr, stream>>>((__gm__ int16_t *)src, (__gm__ float *)dst);
}

void LaunchTCVT_i16_to_ui32_1x128(void *src, void *dst, void *stream) {
    TCVT_i16_to_ui32_1x128<<<1, nullptr, stream>>>((__gm__ int16_t *)src, (__gm__ uint32_t *)dst);
}

void LaunchTCVT_i16_to_ui32_2x64(void *src, void *dst, void *stream) {
    TCVT_i16_to_ui32_2x64<<<1, nullptr, stream>>>((__gm__ int16_t *)src, (__gm__ uint32_t *)dst);
}

void LaunchTCVT_i16_to_ui32_4x32(void *src, void *dst, void *stream) {
    TCVT_i16_to_ui32_4x32<<<1, nullptr, stream>>>((__gm__ int16_t *)src, (__gm__ uint32_t *)dst);
}

void LaunchTCVT_i16_to_ui32_2x128(void *src, void *dst, void *stream) {
    TCVT_i16_to_ui32_2x128<<<1, nullptr, stream>>>((__gm__ int16_t *)src, (__gm__ uint32_t *)dst);
}

void LaunchTCVT_i16_to_ui32_4x65(void *src, void *dst, void *stream) {
    TCVT_i16_to_ui32_4x65<<<1, nullptr, stream>>>((__gm__ int16_t *)src, (__gm__ uint32_t *)dst);
}

void LaunchTCVT_i16_to_ui32_4x200(void *src, void *dst, void *stream) {
    TCVT_i16_to_ui32_4x200<<<1, nullptr, stream>>>((__gm__ int16_t *)src, (__gm__ uint32_t *)dst);
}

void LaunchTCVT_i16_to_ui32_1x129(void *src, void *dst, void *stream) {
    TCVT_i16_to_ui32_1x129<<<1, nullptr, stream>>>((__gm__ int16_t *)src, (__gm__ uint32_t *)dst);
}

void LaunchTCVT_i16_to_i32_1x128(void *src, void *dst, void *stream) {
    TCVT_i16_to_i32_1x128<<<1, nullptr, stream>>>((__gm__ int16_t *)src, (__gm__ int32_t *)dst);
}

void LaunchTCVT_i16_to_i32_2x64(void *src, void *dst, void *stream) {
    TCVT_i16_to_i32_2x64<<<1, nullptr, stream>>>((__gm__ int16_t *)src, (__gm__ int32_t *)dst);
}

void LaunchTCVT_i16_to_i32_4x32(void *src, void *dst, void *stream) {
    TCVT_i16_to_i32_4x32<<<1, nullptr, stream>>>((__gm__ int16_t *)src, (__gm__ int32_t *)dst);
}

void LaunchTCVT_i16_to_i32_2x128(void *src, void *dst, void *stream) {
    TCVT_i16_to_i32_2x128<<<1, nullptr, stream>>>((__gm__ int16_t *)src, (__gm__ int32_t *)dst);
}

void LaunchTCVT_i16_to_i32_4x65(void *src, void *dst, void *stream) {
    TCVT_i16_to_i32_4x65<<<1, nullptr, stream>>>((__gm__ int16_t *)src, (__gm__ int32_t *)dst);
}

void LaunchTCVT_i16_to_i32_4x200(void *src, void *dst, void *stream) {
    TCVT_i16_to_i32_4x200<<<1, nullptr, stream>>>((__gm__ int16_t *)src, (__gm__ int32_t *)dst);
}

void LaunchTCVT_i16_to_i32_1x129(void *src, void *dst, void *stream) {
    TCVT_i16_to_i32_1x129<<<1, nullptr, stream>>>((__gm__ int16_t *)src, (__gm__ int32_t *)dst);
}

void LaunchTCVT_i32_to_f32_1x128(void *src, void *dst, void *stream) {
    TCVT_i32_to_f32_1x128<<<1, nullptr, stream>>>((__gm__ int32_t *)src, (__gm__ float *)dst);
}

void LaunchTCVT_i32_to_f32_2x64(void *src, void *dst, void *stream) {
    TCVT_i32_to_f32_2x64<<<1, nullptr, stream>>>((__gm__ int32_t *)src, (__gm__ float *)dst);
}

void LaunchTCVT_i32_to_f32_4x32(void *src, void *dst, void *stream) {
    TCVT_i32_to_f32_4x32<<<1, nullptr, stream>>>((__gm__ int32_t *)src, (__gm__ float *)dst);
}

void LaunchTCVT_i32_to_f32_2x128(void *src, void *dst, void *stream) {
    TCVT_i32_to_f32_2x128<<<1, nullptr, stream>>>((__gm__ int32_t *)src, (__gm__ float *)dst);
}

void LaunchTCVT_i32_to_f32_4x65(void *src, void *dst, void *stream) {
    TCVT_i32_to_f32_4x65<<<1, nullptr, stream>>>((__gm__ int32_t *)src, (__gm__ float *)dst);
}

void LaunchTCVT_i32_to_f32_4x200(void *src, void *dst, void *stream) {
    TCVT_i32_to_f32_4x200<<<1, nullptr, stream>>>((__gm__ int32_t *)src, (__gm__ float *)dst);
}

void LaunchTCVT_i32_to_f32_1x129(void *src, void *dst, void *stream) {
    TCVT_i32_to_f32_1x129<<<1, nullptr, stream>>>((__gm__ int32_t *)src, (__gm__ float *)dst);
}

void LaunchTCVT_i32_to_i16_1x128(void *src, void *dst, void *stream) {
    TCVT_i32_to_i16_1x128<<<1, nullptr, stream>>>((__gm__ int32_t *)src, (__gm__ int16_t *)dst);
}

void LaunchTCVT_i32_to_i16_2x64(void *src, void *dst, void *stream) {
    TCVT_i32_to_i16_2x64<<<1, nullptr, stream>>>((__gm__ int32_t *)src, (__gm__ int16_t *)dst);
}

void LaunchTCVT_i32_to_i16_4x32(void *src, void *dst, void *stream) {
    TCVT_i32_to_i16_4x32<<<1, nullptr, stream>>>((__gm__ int32_t *)src, (__gm__ int16_t *)dst);
}

void LaunchTCVT_i32_to_i16_2x128(void *src, void *dst, void *stream) {
    TCVT_i32_to_i16_2x128<<<1, nullptr, stream>>>((__gm__ int32_t *)src, (__gm__ int16_t *)dst);
}

void LaunchTCVT_i32_to_i16_4x65(void *src, void *dst, void *stream) {
    TCVT_i32_to_i16_4x65<<<1, nullptr, stream>>>((__gm__ int32_t *)src, (__gm__ int16_t *)dst);
}

void LaunchTCVT_i32_to_i16_4x200(void *src, void *dst, void *stream) {
    TCVT_i32_to_i16_4x200<<<1, nullptr, stream>>>((__gm__ int32_t *)src, (__gm__ int16_t *)dst);
}

void LaunchTCVT_i32_to_i16_1x129(void *src, void *dst, void *stream) {
    TCVT_i32_to_i16_1x129<<<1, nullptr, stream>>>((__gm__ int32_t *)src, (__gm__ int16_t *)dst);
}

void LaunchTCVT_i32_to_i64_1x128(void *src, void *dst, void *stream) {
    TCVT_i32_to_i64_1x128<<<1, nullptr, stream>>>((__gm__ int32_t *)src, (__gm__ int64_t *)dst);
}

void LaunchTCVT_i32_to_i64_2x64(void *src, void *dst, void *stream) {
    TCVT_i32_to_i64_2x64<<<1, nullptr, stream>>>((__gm__ int32_t *)src, (__gm__ int64_t *)dst);
}

void LaunchTCVT_i32_to_i64_4x32(void *src, void *dst, void *stream) {
    TCVT_i32_to_i64_4x32<<<1, nullptr, stream>>>((__gm__ int32_t *)src, (__gm__ int64_t *)dst);
}

void LaunchTCVT_i32_to_i64_2x128(void *src, void *dst, void *stream) {
    TCVT_i32_to_i64_2x128<<<1, nullptr, stream>>>((__gm__ int32_t *)src, (__gm__ int64_t *)dst);
}

void LaunchTCVT_i32_to_i64_4x65(void *src, void *dst, void *stream) {
    TCVT_i32_to_i64_4x65<<<1, nullptr, stream>>>((__gm__ int32_t *)src, (__gm__ int64_t *)dst);
}

void LaunchTCVT_i32_to_i64_4x200(void *src, void *dst, void *stream) {
    TCVT_i32_to_i64_4x200<<<1, nullptr, stream>>>((__gm__ int32_t *)src, (__gm__ int64_t *)dst);
}

void LaunchTCVT_i32_to_i64_1x129(void *src, void *dst, void *stream) {
    TCVT_i32_to_i64_1x129<<<1, nullptr, stream>>>((__gm__ int32_t *)src, (__gm__ int64_t *)dst);
}

void LaunchTCVT_i32_to_ui8_1x128(void *src, void *dst, void *stream) {
    TCVT_i32_to_ui8_1x128<<<1, nullptr, stream>>>((__gm__ int32_t *)src, (__gm__ uint8_t *)dst);
}

void LaunchTCVT_i32_to_ui8_2x64(void *src, void *dst, void *stream) {
    TCVT_i32_to_ui8_2x64<<<1, nullptr, stream>>>((__gm__ int32_t *)src, (__gm__ uint8_t *)dst);
}

void LaunchTCVT_i32_to_ui8_4x32(void *src, void *dst, void *stream) {
    TCVT_i32_to_ui8_4x32<<<1, nullptr, stream>>>((__gm__ int32_t *)src, (__gm__ uint8_t *)dst);
}

void LaunchTCVT_i32_to_ui8_2x128(void *src, void *dst, void *stream) {
    TCVT_i32_to_ui8_2x128<<<1, nullptr, stream>>>((__gm__ int32_t *)src, (__gm__ uint8_t *)dst);
}

void LaunchTCVT_i32_to_ui8_4x65(void *src, void *dst, void *stream) {
    TCVT_i32_to_ui8_4x65<<<1, nullptr, stream>>>((__gm__ int32_t *)src, (__gm__ uint8_t *)dst);
}

void LaunchTCVT_i32_to_ui8_4x200(void *src, void *dst, void *stream) {
    TCVT_i32_to_ui8_4x200<<<1, nullptr, stream>>>((__gm__ int32_t *)src, (__gm__ uint8_t *)dst);
}

void LaunchTCVT_i32_to_ui8_1x129(void *src, void *dst, void *stream) {
    TCVT_i32_to_ui8_1x129<<<1, nullptr, stream>>>((__gm__ int32_t *)src, (__gm__ uint8_t *)dst);
}

void LaunchTCVT_i32_to_ui16_1x128(void *src, void *dst, void *stream) {
    TCVT_i32_to_ui16_1x128<<<1, nullptr, stream>>>((__gm__ int32_t *)src, (__gm__ uint16_t *)dst);
}

void LaunchTCVT_i32_to_ui16_2x64(void *src, void *dst, void *stream) {
    TCVT_i32_to_ui16_2x64<<<1, nullptr, stream>>>((__gm__ int32_t *)src, (__gm__ uint16_t *)dst);
}

void LaunchTCVT_i32_to_ui16_4x32(void *src, void *dst, void *stream) {
    TCVT_i32_to_ui16_4x32<<<1, nullptr, stream>>>((__gm__ int32_t *)src, (__gm__ uint16_t *)dst);
}

void LaunchTCVT_i32_to_ui16_2x128(void *src, void *dst, void *stream) {
    TCVT_i32_to_ui16_2x128<<<1, nullptr, stream>>>((__gm__ int32_t *)src, (__gm__ uint16_t *)dst);
}

void LaunchTCVT_i32_to_ui16_4x65(void *src, void *dst, void *stream) {
    TCVT_i32_to_ui16_4x65<<<1, nullptr, stream>>>((__gm__ int32_t *)src, (__gm__ uint16_t *)dst);
}

void LaunchTCVT_i32_to_ui16_4x200(void *src, void *dst, void *stream) {
    TCVT_i32_to_ui16_4x200<<<1, nullptr, stream>>>((__gm__ int32_t *)src, (__gm__ uint16_t *)dst);
}

void LaunchTCVT_i32_to_ui16_1x129(void *src, void *dst, void *stream) {
    TCVT_i32_to_ui16_1x129<<<1, nullptr, stream>>>((__gm__ int32_t *)src, (__gm__ uint16_t *)dst);
}

void LaunchTCVT_ui32_to_i16_1x128(void *src, void *dst, void *stream) {
    TCVT_ui32_to_i16_1x128<<<1, nullptr, stream>>>((__gm__ uint32_t *)src, (__gm__ int16_t *)dst);
}

void LaunchTCVT_ui32_to_i16_2x64(void *src, void *dst, void *stream) {
    TCVT_ui32_to_i16_2x64<<<1, nullptr, stream>>>((__gm__ uint32_t *)src, (__gm__ int16_t *)dst);
}

void LaunchTCVT_ui32_to_i16_4x32(void *src, void *dst, void *stream) {
    TCVT_ui32_to_i16_4x32<<<1, nullptr, stream>>>((__gm__ uint32_t *)src, (__gm__ int16_t *)dst);
}

void LaunchTCVT_ui32_to_i16_2x128(void *src, void *dst, void *stream) {
    TCVT_ui32_to_i16_2x128<<<1, nullptr, stream>>>((__gm__ uint32_t *)src, (__gm__ int16_t *)dst);
}

void LaunchTCVT_ui32_to_i16_4x65(void *src, void *dst, void *stream) {
    TCVT_ui32_to_i16_4x65<<<1, nullptr, stream>>>((__gm__ uint32_t *)src, (__gm__ int16_t *)dst);
}

void LaunchTCVT_ui32_to_i16_4x200(void *src, void *dst, void *stream) {
    TCVT_ui32_to_i16_4x200<<<1, nullptr, stream>>>((__gm__ uint32_t *)src, (__gm__ int16_t *)dst);
}

void LaunchTCVT_ui32_to_i16_1x129(void *src, void *dst, void *stream) {
    TCVT_ui32_to_i16_1x129<<<1, nullptr, stream>>>((__gm__ uint32_t *)src, (__gm__ int16_t *)dst);
}

void LaunchTCVT_ui32_to_ui16_1x128(void *src, void *dst, void *stream) {
    TCVT_ui32_to_ui16_1x128<<<1, nullptr, stream>>>((__gm__ uint32_t *)src, (__gm__ uint16_t *)dst);
}

void LaunchTCVT_ui32_to_ui16_2x64(void *src, void *dst, void *stream) {
    TCVT_ui32_to_ui16_2x64<<<1, nullptr, stream>>>((__gm__ uint32_t *)src, (__gm__ uint16_t *)dst);
}

void LaunchTCVT_ui32_to_ui16_4x32(void *src, void *dst, void *stream) {
    TCVT_ui32_to_ui16_4x32<<<1, nullptr, stream>>>((__gm__ uint32_t *)src, (__gm__ uint16_t *)dst);
}

void LaunchTCVT_ui32_to_ui16_2x128(void *src, void *dst, void *stream) {
    TCVT_ui32_to_ui16_2x128<<<1, nullptr, stream>>>((__gm__ uint32_t *)src, (__gm__ uint16_t *)dst);
}

void LaunchTCVT_ui32_to_ui16_4x65(void *src, void *dst, void *stream) {
    TCVT_ui32_to_ui16_4x65<<<1, nullptr, stream>>>((__gm__ uint32_t *)src, (__gm__ uint16_t *)dst);
}

void LaunchTCVT_ui32_to_ui16_4x200(void *src, void *dst, void *stream) {
    TCVT_ui32_to_ui16_4x200<<<1, nullptr, stream>>>((__gm__ uint32_t *)src, (__gm__ uint16_t *)dst);
}

void LaunchTCVT_ui32_to_ui16_1x129(void *src, void *dst, void *stream) {
    TCVT_ui32_to_ui16_1x129<<<1, nullptr, stream>>>((__gm__ uint32_t *)src, (__gm__ uint16_t *)dst);
}

void LaunchTCVT_ui32_to_ui8_1x128(void *src, void *dst, void *stream) {
    TCVT_ui32_to_ui8_1x128<<<1, nullptr, stream>>>((__gm__ uint32_t *)src, (__gm__ uint8_t *)dst);
}

void LaunchTCVT_ui32_to_ui8_2x64(void *src, void *dst, void *stream) {
    TCVT_ui32_to_ui8_2x64<<<1, nullptr, stream>>>((__gm__ uint32_t *)src, (__gm__ uint8_t *)dst);
}

void LaunchTCVT_ui32_to_ui8_4x32(void *src, void *dst, void *stream) {
    TCVT_ui32_to_ui8_4x32<<<1, nullptr, stream>>>((__gm__ uint32_t *)src, (__gm__ uint8_t *)dst);
}

void LaunchTCVT_ui32_to_ui8_2x128(void *src, void *dst, void *stream) {
    TCVT_ui32_to_ui8_2x128<<<1, nullptr, stream>>>((__gm__ uint32_t *)src, (__gm__ uint8_t *)dst);
}

void LaunchTCVT_ui32_to_ui8_4x65(void *src, void *dst, void *stream) {
    TCVT_ui32_to_ui8_4x65<<<1, nullptr, stream>>>((__gm__ uint32_t *)src, (__gm__ uint8_t *)dst);
}

void LaunchTCVT_ui32_to_ui8_4x200(void *src, void *dst, void *stream) {
    TCVT_ui32_to_ui8_4x200<<<1, nullptr, stream>>>((__gm__ uint32_t *)src, (__gm__ uint8_t *)dst);
}

void LaunchTCVT_ui32_to_ui8_1x129(void *src, void *dst, void *stream) {
    TCVT_ui32_to_ui8_1x129<<<1, nullptr, stream>>>((__gm__ uint32_t *)src, (__gm__ uint8_t *)dst);
}

void LaunchTCVT_i64_to_f32_1x128(void *src, void *dst, void *stream) {
    TCVT_i64_to_f32_1x128<<<1, nullptr, stream>>>((__gm__ int64_t *)src, (__gm__ float *)dst);
}

void LaunchTCVT_i64_to_f32_2x64(void *src, void *dst, void *stream) {
    TCVT_i64_to_f32_2x64<<<1, nullptr, stream>>>((__gm__ int64_t *)src, (__gm__ float *)dst);
}

void LaunchTCVT_i64_to_f32_4x32(void *src, void *dst, void *stream) {
    TCVT_i64_to_f32_4x32<<<1, nullptr, stream>>>((__gm__ int64_t *)src, (__gm__ float *)dst);
}

void LaunchTCVT_i64_to_f32_2x128(void *src, void *dst, void *stream) {
    TCVT_i64_to_f32_2x128<<<1, nullptr, stream>>>((__gm__ int64_t *)src, (__gm__ float *)dst);
}

void LaunchTCVT_i64_to_f32_4x65(void *src, void *dst, void *stream) {
    TCVT_i64_to_f32_4x65<<<1, nullptr, stream>>>((__gm__ int64_t *)src, (__gm__ float *)dst);
}

void LaunchTCVT_i64_to_f32_4x200(void *src, void *dst, void *stream) {
    TCVT_i64_to_f32_4x200<<<1, nullptr, stream>>>((__gm__ int64_t *)src, (__gm__ float *)dst);
}

void LaunchTCVT_i64_to_f32_1x129(void *src, void *dst, void *stream) {
    TCVT_i64_to_f32_1x129<<<1, nullptr, stream>>>((__gm__ int64_t *)src, (__gm__ float *)dst);
}

void LaunchTCVT_i64_to_i32_1x128(void *src, void *dst, void *stream) {
    TCVT_i64_to_i32_1x128<<<1, nullptr, stream>>>((__gm__ int64_t *)src, (__gm__ int32_t *)dst);
}

void LaunchTCVT_i64_to_i32_2x64(void *src, void *dst, void *stream) {
    TCVT_i64_to_i32_2x64<<<1, nullptr, stream>>>((__gm__ int64_t *)src, (__gm__ int32_t *)dst);
}

void LaunchTCVT_i64_to_i32_4x32(void *src, void *dst, void *stream) {
    TCVT_i64_to_i32_4x32<<<1, nullptr, stream>>>((__gm__ int64_t *)src, (__gm__ int32_t *)dst);
}

void LaunchTCVT_i64_to_i32_2x128(void *src, void *dst, void *stream) {
    TCVT_i64_to_i32_2x128<<<1, nullptr, stream>>>((__gm__ int64_t *)src, (__gm__ int32_t *)dst);
}

void LaunchTCVT_i64_to_i32_4x65(void *src, void *dst, void *stream) {
    TCVT_i64_to_i32_4x65<<<1, nullptr, stream>>>((__gm__ int64_t *)src, (__gm__ int32_t *)dst);
}

void LaunchTCVT_i64_to_i32_4x200(void *src, void *dst, void *stream) {
    TCVT_i64_to_i32_4x200<<<1, nullptr, stream>>>((__gm__ int64_t *)src, (__gm__ int32_t *)dst);
}

void LaunchTCVT_i64_to_i32_1x129(void *src, void *dst, void *stream) {
    TCVT_i64_to_i32_1x129<<<1, nullptr, stream>>>((__gm__ int64_t *)src, (__gm__ int32_t *)dst);
}
