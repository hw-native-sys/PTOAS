// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

#include "acl/acl.h"
#include "test_common.h"
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>

using namespace PtoTestCommon;

void LaunchTCVT_f32_to_i32_rint_16x64(void *src, void *dst, void *stream);
void LaunchTCVT_f32_to_i32_round_16x64(void *src, void *dst, void *stream);
void LaunchTCVT_i32_to_f32_rint_16x64(void *src, void *dst, void *stream);
void LaunchTCVT_f32_to_f16_rint_16x64(void *src, void *dst, void *stream);
void LaunchTCVT_f16_to_f32_rint_16x64(void *src, void *dst, void *stream);
void LaunchTCVT_f32_to_f8e4m3_16x64(void *src, void *dst, void *stream);
void LaunchTCVT_f32_to_f8e5m2_16x64(void *src, void *dst, void *stream);
void LaunchTCVT_f32_to_hif8_16x64(void *src, void *dst, void *stream);
void LaunchTCVT_f16_to_hif8_16x64(void *src, void *dst, void *stream);
void LaunchTCVT_bf16_to_f4e1m2x2_16x64_to_16x32(void *src, void *dst, void *stream);
void LaunchTCVT_bf16_to_f4e2m1x2_16x64_to_16x32(void *src, void *dst, void *stream);
void LaunchTCVT_f32_to_f8e4m3_4x96(void *src, void *dst, void *stream);
void LaunchTCVT_f32_to_hif8_4x96(void *src, void *dst, void *stream);
void LaunchTCVT_f32_to_f16_1x128(void *src, void *dst, void *stream);
void LaunchTCVT_f32_to_f16_2x64(void *src, void *dst, void *stream);
void LaunchTCVT_f32_to_f16_4x32(void *src, void *dst, void *stream);
void LaunchTCVT_f32_to_f16_2x128(void *src, void *dst, void *stream);
void LaunchTCVT_f32_to_f16_4x65(void *src, void *dst, void *stream);
void LaunchTCVT_f32_to_f16_4x200(void *src, void *dst, void *stream);
void LaunchTCVT_f32_to_f16_1x129(void *src, void *dst, void *stream);
void LaunchTCVT_f32_to_bf16_1x128(void *src, void *dst, void *stream);
void LaunchTCVT_f32_to_bf16_2x64(void *src, void *dst, void *stream);
void LaunchTCVT_f32_to_bf16_4x32(void *src, void *dst, void *stream);
void LaunchTCVT_f32_to_bf16_2x128(void *src, void *dst, void *stream);
void LaunchTCVT_f32_to_bf16_4x65(void *src, void *dst, void *stream);
void LaunchTCVT_f32_to_bf16_4x200(void *src, void *dst, void *stream);
void LaunchTCVT_f32_to_bf16_1x129(void *src, void *dst, void *stream);
void LaunchTCVT_f32_to_i16_1x128(void *src, void *dst, void *stream);
void LaunchTCVT_f32_to_i16_2x64(void *src, void *dst, void *stream);
void LaunchTCVT_f32_to_i16_4x32(void *src, void *dst, void *stream);
void LaunchTCVT_f32_to_i16_2x128(void *src, void *dst, void *stream);
void LaunchTCVT_f32_to_i16_4x65(void *src, void *dst, void *stream);
void LaunchTCVT_f32_to_i16_4x200(void *src, void *dst, void *stream);
void LaunchTCVT_f32_to_i16_1x129(void *src, void *dst, void *stream);
void LaunchTCVT_f32_to_i32_1x128(void *src, void *dst, void *stream);
void LaunchTCVT_f32_to_i32_2x64(void *src, void *dst, void *stream);
void LaunchTCVT_f32_to_i32_4x32(void *src, void *dst, void *stream);
void LaunchTCVT_f32_to_i32_2x128(void *src, void *dst, void *stream);
void LaunchTCVT_f32_to_i32_4x65(void *src, void *dst, void *stream);
void LaunchTCVT_f32_to_i32_4x200(void *src, void *dst, void *stream);
void LaunchTCVT_f32_to_i32_1x129(void *src, void *dst, void *stream);
void LaunchTCVT_f32_to_i64_1x128(void *src, void *dst, void *stream);
void LaunchTCVT_f32_to_i64_2x64(void *src, void *dst, void *stream);
void LaunchTCVT_f32_to_i64_4x32(void *src, void *dst, void *stream);
void LaunchTCVT_f32_to_i64_2x128(void *src, void *dst, void *stream);
void LaunchTCVT_f32_to_i64_4x65(void *src, void *dst, void *stream);
void LaunchTCVT_f32_to_i64_4x200(void *src, void *dst, void *stream);
void LaunchTCVT_f32_to_i64_1x129(void *src, void *dst, void *stream);
void LaunchTCVT_f32_to_f32_1x128(void *src, void *dst, void *stream);
void LaunchTCVT_f32_to_f32_2x64(void *src, void *dst, void *stream);
void LaunchTCVT_f32_to_f32_4x32(void *src, void *dst, void *stream);
void LaunchTCVT_f32_to_f32_2x128(void *src, void *dst, void *stream);
void LaunchTCVT_f32_to_f32_4x65(void *src, void *dst, void *stream);
void LaunchTCVT_f32_to_f32_4x200(void *src, void *dst, void *stream);
void LaunchTCVT_f32_to_f32_1x129(void *src, void *dst, void *stream);
void LaunchTCVT_f16_to_f32_1x128(void *src, void *dst, void *stream);
void LaunchTCVT_f16_to_f32_2x64(void *src, void *dst, void *stream);
void LaunchTCVT_f16_to_f32_4x32(void *src, void *dst, void *stream);
void LaunchTCVT_f16_to_f32_2x128(void *src, void *dst, void *stream);
void LaunchTCVT_f16_to_f32_4x65(void *src, void *dst, void *stream);
void LaunchTCVT_f16_to_f32_4x200(void *src, void *dst, void *stream);
void LaunchTCVT_f16_to_f32_1x129(void *src, void *dst, void *stream);
void LaunchTCVT_f16_to_i32_1x128(void *src, void *dst, void *stream);
void LaunchTCVT_f16_to_i32_2x64(void *src, void *dst, void *stream);
void LaunchTCVT_f16_to_i32_4x32(void *src, void *dst, void *stream);
void LaunchTCVT_f16_to_i32_2x128(void *src, void *dst, void *stream);
void LaunchTCVT_f16_to_i32_4x65(void *src, void *dst, void *stream);
void LaunchTCVT_f16_to_i32_4x200(void *src, void *dst, void *stream);
void LaunchTCVT_f16_to_i32_1x129(void *src, void *dst, void *stream);
void LaunchTCVT_f16_to_i16_1x128(void *src, void *dst, void *stream);
void LaunchTCVT_f16_to_i16_2x64(void *src, void *dst, void *stream);
void LaunchTCVT_f16_to_i16_4x32(void *src, void *dst, void *stream);
void LaunchTCVT_f16_to_i16_2x128(void *src, void *dst, void *stream);
void LaunchTCVT_f16_to_i16_4x65(void *src, void *dst, void *stream);
void LaunchTCVT_f16_to_i16_4x200(void *src, void *dst, void *stream);
void LaunchTCVT_f16_to_i16_1x129(void *src, void *dst, void *stream);
void LaunchTCVT_f16_to_si8_1x128(void *src, void *dst, void *stream);
void LaunchTCVT_f16_to_si8_2x64(void *src, void *dst, void *stream);
void LaunchTCVT_f16_to_si8_4x32(void *src, void *dst, void *stream);
void LaunchTCVT_f16_to_si8_2x128(void *src, void *dst, void *stream);
void LaunchTCVT_f16_to_si8_4x65(void *src, void *dst, void *stream);
void LaunchTCVT_f16_to_si8_4x200(void *src, void *dst, void *stream);
void LaunchTCVT_f16_to_si8_1x129(void *src, void *dst, void *stream);
void LaunchTCVT_f16_to_ui8_1x128(void *src, void *dst, void *stream);
void LaunchTCVT_f16_to_ui8_2x64(void *src, void *dst, void *stream);
void LaunchTCVT_f16_to_ui8_4x32(void *src, void *dst, void *stream);
void LaunchTCVT_f16_to_ui8_2x128(void *src, void *dst, void *stream);
void LaunchTCVT_f16_to_ui8_4x65(void *src, void *dst, void *stream);
void LaunchTCVT_f16_to_ui8_4x200(void *src, void *dst, void *stream);
void LaunchTCVT_f16_to_ui8_1x129(void *src, void *dst, void *stream);
void LaunchTCVT_bf16_to_f32_1x128(void *src, void *dst, void *stream);
void LaunchTCVT_bf16_to_f32_2x64(void *src, void *dst, void *stream);
void LaunchTCVT_bf16_to_f32_4x32(void *src, void *dst, void *stream);
void LaunchTCVT_bf16_to_f32_2x128(void *src, void *dst, void *stream);
void LaunchTCVT_bf16_to_f32_4x65(void *src, void *dst, void *stream);
void LaunchTCVT_bf16_to_f32_4x200(void *src, void *dst, void *stream);
void LaunchTCVT_bf16_to_f32_1x129(void *src, void *dst, void *stream);
void LaunchTCVT_bf16_to_f16_1x128(void *src, void *dst, void *stream);
void LaunchTCVT_bf16_to_f16_2x64(void *src, void *dst, void *stream);
void LaunchTCVT_bf16_to_f16_4x32(void *src, void *dst, void *stream);
void LaunchTCVT_bf16_to_f16_2x128(void *src, void *dst, void *stream);
void LaunchTCVT_bf16_to_f16_4x65(void *src, void *dst, void *stream);
void LaunchTCVT_bf16_to_f16_4x200(void *src, void *dst, void *stream);
void LaunchTCVT_bf16_to_f16_1x129(void *src, void *dst, void *stream);
void LaunchTCVT_bf16_to_i32_1x128(void *src, void *dst, void *stream);
void LaunchTCVT_bf16_to_i32_2x64(void *src, void *dst, void *stream);
void LaunchTCVT_bf16_to_i32_4x32(void *src, void *dst, void *stream);
void LaunchTCVT_bf16_to_i32_2x128(void *src, void *dst, void *stream);
void LaunchTCVT_bf16_to_i32_4x65(void *src, void *dst, void *stream);
void LaunchTCVT_bf16_to_i32_4x200(void *src, void *dst, void *stream);
void LaunchTCVT_bf16_to_i32_1x129(void *src, void *dst, void *stream);
void LaunchTCVT_ui8_to_f16_1x128(void *src, void *dst, void *stream);
void LaunchTCVT_ui8_to_f16_2x64(void *src, void *dst, void *stream);
void LaunchTCVT_ui8_to_f16_4x32(void *src, void *dst, void *stream);
void LaunchTCVT_ui8_to_f16_2x128(void *src, void *dst, void *stream);
void LaunchTCVT_ui8_to_f16_4x65(void *src, void *dst, void *stream);
void LaunchTCVT_ui8_to_f16_4x200(void *src, void *dst, void *stream);
void LaunchTCVT_ui8_to_f16_1x129(void *src, void *dst, void *stream);
void LaunchTCVT_ui8_to_ui16_1x128(void *src, void *dst, void *stream);
void LaunchTCVT_ui8_to_ui16_2x64(void *src, void *dst, void *stream);
void LaunchTCVT_ui8_to_ui16_4x32(void *src, void *dst, void *stream);
void LaunchTCVT_ui8_to_ui16_2x128(void *src, void *dst, void *stream);
void LaunchTCVT_ui8_to_ui16_4x65(void *src, void *dst, void *stream);
void LaunchTCVT_ui8_to_ui16_4x200(void *src, void *dst, void *stream);
void LaunchTCVT_ui8_to_ui16_1x129(void *src, void *dst, void *stream);
void LaunchTCVT_si8_to_f16_1x128(void *src, void *dst, void *stream);
void LaunchTCVT_si8_to_f16_2x64(void *src, void *dst, void *stream);
void LaunchTCVT_si8_to_f16_4x32(void *src, void *dst, void *stream);
void LaunchTCVT_si8_to_f16_2x128(void *src, void *dst, void *stream);
void LaunchTCVT_si8_to_f16_4x65(void *src, void *dst, void *stream);
void LaunchTCVT_si8_to_f16_4x200(void *src, void *dst, void *stream);
void LaunchTCVT_si8_to_f16_1x129(void *src, void *dst, void *stream);
void LaunchTCVT_si8_to_si16_1x128(void *src, void *dst, void *stream);
void LaunchTCVT_si8_to_si16_2x64(void *src, void *dst, void *stream);
void LaunchTCVT_si8_to_si16_4x32(void *src, void *dst, void *stream);
void LaunchTCVT_si8_to_si16_2x128(void *src, void *dst, void *stream);
void LaunchTCVT_si8_to_si16_4x65(void *src, void *dst, void *stream);
void LaunchTCVT_si8_to_si16_4x200(void *src, void *dst, void *stream);
void LaunchTCVT_si8_to_si16_1x129(void *src, void *dst, void *stream);
void LaunchTCVT_si8_to_i32_1x128(void *src, void *dst, void *stream);
void LaunchTCVT_si8_to_i32_2x64(void *src, void *dst, void *stream);
void LaunchTCVT_si8_to_i32_4x32(void *src, void *dst, void *stream);
void LaunchTCVT_si8_to_i32_2x128(void *src, void *dst, void *stream);
void LaunchTCVT_si8_to_i32_4x65(void *src, void *dst, void *stream);
void LaunchTCVT_si8_to_i32_4x200(void *src, void *dst, void *stream);
void LaunchTCVT_si8_to_i32_1x129(void *src, void *dst, void *stream);
void LaunchTCVT_i16_to_ui8_1x128(void *src, void *dst, void *stream);
void LaunchTCVT_i16_to_ui8_2x64(void *src, void *dst, void *stream);
void LaunchTCVT_i16_to_ui8_4x32(void *src, void *dst, void *stream);
void LaunchTCVT_i16_to_ui8_2x128(void *src, void *dst, void *stream);
void LaunchTCVT_i16_to_ui8_4x65(void *src, void *dst, void *stream);
void LaunchTCVT_i16_to_ui8_4x200(void *src, void *dst, void *stream);
void LaunchTCVT_i16_to_ui8_1x129(void *src, void *dst, void *stream);
void LaunchTCVT_i16_to_f16_1x128(void *src, void *dst, void *stream);
void LaunchTCVT_i16_to_f16_2x64(void *src, void *dst, void *stream);
void LaunchTCVT_i16_to_f16_4x32(void *src, void *dst, void *stream);
void LaunchTCVT_i16_to_f16_2x128(void *src, void *dst, void *stream);
void LaunchTCVT_i16_to_f16_4x65(void *src, void *dst, void *stream);
void LaunchTCVT_i16_to_f16_4x200(void *src, void *dst, void *stream);
void LaunchTCVT_i16_to_f16_1x129(void *src, void *dst, void *stream);
void LaunchTCVT_i16_to_f32_1x128(void *src, void *dst, void *stream);
void LaunchTCVT_i16_to_f32_2x64(void *src, void *dst, void *stream);
void LaunchTCVT_i16_to_f32_4x32(void *src, void *dst, void *stream);
void LaunchTCVT_i16_to_f32_2x128(void *src, void *dst, void *stream);
void LaunchTCVT_i16_to_f32_4x65(void *src, void *dst, void *stream);
void LaunchTCVT_i16_to_f32_4x200(void *src, void *dst, void *stream);
void LaunchTCVT_i16_to_f32_1x129(void *src, void *dst, void *stream);
void LaunchTCVT_i16_to_ui32_1x128(void *src, void *dst, void *stream);
void LaunchTCVT_i16_to_ui32_2x64(void *src, void *dst, void *stream);
void LaunchTCVT_i16_to_ui32_4x32(void *src, void *dst, void *stream);
void LaunchTCVT_i16_to_ui32_2x128(void *src, void *dst, void *stream);
void LaunchTCVT_i16_to_ui32_4x65(void *src, void *dst, void *stream);
void LaunchTCVT_i16_to_ui32_4x200(void *src, void *dst, void *stream);
void LaunchTCVT_i16_to_ui32_1x129(void *src, void *dst, void *stream);
void LaunchTCVT_i16_to_i32_1x128(void *src, void *dst, void *stream);
void LaunchTCVT_i16_to_i32_2x64(void *src, void *dst, void *stream);
void LaunchTCVT_i16_to_i32_4x32(void *src, void *dst, void *stream);
void LaunchTCVT_i16_to_i32_2x128(void *src, void *dst, void *stream);
void LaunchTCVT_i16_to_i32_4x65(void *src, void *dst, void *stream);
void LaunchTCVT_i16_to_i32_4x200(void *src, void *dst, void *stream);
void LaunchTCVT_i16_to_i32_1x129(void *src, void *dst, void *stream);
void LaunchTCVT_i32_to_f32_1x128(void *src, void *dst, void *stream);
void LaunchTCVT_i32_to_f32_2x64(void *src, void *dst, void *stream);
void LaunchTCVT_i32_to_f32_4x32(void *src, void *dst, void *stream);
void LaunchTCVT_i32_to_f32_2x128(void *src, void *dst, void *stream);
void LaunchTCVT_i32_to_f32_4x65(void *src, void *dst, void *stream);
void LaunchTCVT_i32_to_f32_4x200(void *src, void *dst, void *stream);
void LaunchTCVT_i32_to_f32_1x129(void *src, void *dst, void *stream);
void LaunchTCVT_i32_to_i16_1x128(void *src, void *dst, void *stream);
void LaunchTCVT_i32_to_i16_2x64(void *src, void *dst, void *stream);
void LaunchTCVT_i32_to_i16_4x32(void *src, void *dst, void *stream);
void LaunchTCVT_i32_to_i16_2x128(void *src, void *dst, void *stream);
void LaunchTCVT_i32_to_i16_4x65(void *src, void *dst, void *stream);
void LaunchTCVT_i32_to_i16_4x200(void *src, void *dst, void *stream);
void LaunchTCVT_i32_to_i16_1x129(void *src, void *dst, void *stream);
void LaunchTCVT_i32_to_i64_1x128(void *src, void *dst, void *stream);
void LaunchTCVT_i32_to_i64_2x64(void *src, void *dst, void *stream);
void LaunchTCVT_i32_to_i64_4x32(void *src, void *dst, void *stream);
void LaunchTCVT_i32_to_i64_2x128(void *src, void *dst, void *stream);
void LaunchTCVT_i32_to_i64_4x65(void *src, void *dst, void *stream);
void LaunchTCVT_i32_to_i64_4x200(void *src, void *dst, void *stream);
void LaunchTCVT_i32_to_i64_1x129(void *src, void *dst, void *stream);
void LaunchTCVT_i32_to_ui8_1x128(void *src, void *dst, void *stream);
void LaunchTCVT_i32_to_ui8_2x64(void *src, void *dst, void *stream);
void LaunchTCVT_i32_to_ui8_4x32(void *src, void *dst, void *stream);
void LaunchTCVT_i32_to_ui8_2x128(void *src, void *dst, void *stream);
void LaunchTCVT_i32_to_ui8_4x65(void *src, void *dst, void *stream);
void LaunchTCVT_i32_to_ui8_4x200(void *src, void *dst, void *stream);
void LaunchTCVT_i32_to_ui8_1x129(void *src, void *dst, void *stream);
void LaunchTCVT_i32_to_ui16_1x128(void *src, void *dst, void *stream);
void LaunchTCVT_i32_to_ui16_2x64(void *src, void *dst, void *stream);
void LaunchTCVT_i32_to_ui16_4x32(void *src, void *dst, void *stream);
void LaunchTCVT_i32_to_ui16_2x128(void *src, void *dst, void *stream);
void LaunchTCVT_i32_to_ui16_4x65(void *src, void *dst, void *stream);
void LaunchTCVT_i32_to_ui16_4x200(void *src, void *dst, void *stream);
void LaunchTCVT_i32_to_ui16_1x129(void *src, void *dst, void *stream);
void LaunchTCVT_ui32_to_i16_1x128(void *src, void *dst, void *stream);
void LaunchTCVT_ui32_to_i16_2x64(void *src, void *dst, void *stream);
void LaunchTCVT_ui32_to_i16_4x32(void *src, void *dst, void *stream);
void LaunchTCVT_ui32_to_i16_2x128(void *src, void *dst, void *stream);
void LaunchTCVT_ui32_to_i16_4x65(void *src, void *dst, void *stream);
void LaunchTCVT_ui32_to_i16_4x200(void *src, void *dst, void *stream);
void LaunchTCVT_ui32_to_i16_1x129(void *src, void *dst, void *stream);
void LaunchTCVT_ui32_to_ui16_1x128(void *src, void *dst, void *stream);
void LaunchTCVT_ui32_to_ui16_2x64(void *src, void *dst, void *stream);
void LaunchTCVT_ui32_to_ui16_4x32(void *src, void *dst, void *stream);
void LaunchTCVT_ui32_to_ui16_2x128(void *src, void *dst, void *stream);
void LaunchTCVT_ui32_to_ui16_4x65(void *src, void *dst, void *stream);
void LaunchTCVT_ui32_to_ui16_4x200(void *src, void *dst, void *stream);
void LaunchTCVT_ui32_to_ui16_1x129(void *src, void *dst, void *stream);
void LaunchTCVT_ui32_to_ui8_1x128(void *src, void *dst, void *stream);
void LaunchTCVT_ui32_to_ui8_2x64(void *src, void *dst, void *stream);
void LaunchTCVT_ui32_to_ui8_4x32(void *src, void *dst, void *stream);
void LaunchTCVT_ui32_to_ui8_2x128(void *src, void *dst, void *stream);
void LaunchTCVT_ui32_to_ui8_4x65(void *src, void *dst, void *stream);
void LaunchTCVT_ui32_to_ui8_4x200(void *src, void *dst, void *stream);
void LaunchTCVT_ui32_to_ui8_1x129(void *src, void *dst, void *stream);
void LaunchTCVT_i64_to_f32_1x128(void *src, void *dst, void *stream);
void LaunchTCVT_i64_to_f32_2x64(void *src, void *dst, void *stream);
void LaunchTCVT_i64_to_f32_4x32(void *src, void *dst, void *stream);
void LaunchTCVT_i64_to_f32_2x128(void *src, void *dst, void *stream);
void LaunchTCVT_i64_to_f32_4x65(void *src, void *dst, void *stream);
void LaunchTCVT_i64_to_f32_4x200(void *src, void *dst, void *stream);
void LaunchTCVT_i64_to_f32_1x129(void *src, void *dst, void *stream);
void LaunchTCVT_i64_to_i32_1x128(void *src, void *dst, void *stream);
void LaunchTCVT_i64_to_i32_2x64(void *src, void *dst, void *stream);
void LaunchTCVT_i64_to_i32_4x32(void *src, void *dst, void *stream);
void LaunchTCVT_i64_to_i32_2x128(void *src, void *dst, void *stream);
void LaunchTCVT_i64_to_i32_4x65(void *src, void *dst, void *stream);
void LaunchTCVT_i64_to_i32_4x200(void *src, void *dst, void *stream);
void LaunchTCVT_i64_to_i32_1x129(void *src, void *dst, void *stream);

using LaunchFn = void (*)(void *, void *, void *);

struct TestCase {
    const char *name;
    LaunchFn    launch;
    size_t      srcRows;
    size_t      srcCols;
    size_t      dstRows;
    size_t      dstCols;
    size_t      srcElemSize;
    size_t      dstElemSize;
};

static const TestCase kCases[] = {
    {"f32_to_i32_rint_16x64", LaunchTCVT_f32_to_i32_rint_16x64, 16, 64, 16, 64, sizeof(float), sizeof(int32_t)},
    {"f32_to_i32_round_16x64", LaunchTCVT_f32_to_i32_round_16x64, 16, 64, 16, 64, sizeof(float), sizeof(int32_t)},
    {"i32_to_f32_rint_16x64", LaunchTCVT_i32_to_f32_rint_16x64, 16, 64, 16, 64, sizeof(int32_t), sizeof(float)},
    {"f32_to_f16_rint_16x64", LaunchTCVT_f32_to_f16_rint_16x64, 16, 64, 16, 64, sizeof(float), sizeof(uint16_t)},
    {"f16_to_f32_rint_16x64", LaunchTCVT_f16_to_f32_rint_16x64, 16, 64, 16, 64, sizeof(uint16_t), sizeof(float)},
    {"f32_to_f8e4m3_16x64", LaunchTCVT_f32_to_f8e4m3_16x64, 16, 64, 16, 64, sizeof(float), sizeof(uint8_t)},
    {"f32_to_f8e5m2_16x64", LaunchTCVT_f32_to_f8e5m2_16x64, 16, 64, 16, 64, sizeof(float), sizeof(uint8_t)},
    {"f32_to_hif8_16x64", LaunchTCVT_f32_to_hif8_16x64, 16, 64, 16, 64, sizeof(float), sizeof(uint8_t)},
    {"f16_to_hif8_16x64", LaunchTCVT_f16_to_hif8_16x64, 16, 64, 16, 64, sizeof(uint16_t), sizeof(uint8_t)},
    {"bf16_to_f4e1m2x2_16x64_to_16x32", LaunchTCVT_bf16_to_f4e1m2x2_16x64_to_16x32, 16, 64, 16, 32, sizeof(uint16_t), sizeof(uint8_t)},
    {"bf16_to_f4e2m1x2_16x64_to_16x32", LaunchTCVT_bf16_to_f4e2m1x2_16x64_to_16x32, 16, 64, 16, 32, sizeof(uint16_t), sizeof(uint8_t)},
    {"f32_to_f8e4m3_4x96", LaunchTCVT_f32_to_f8e4m3_4x96, 4, 96, 4, 96, sizeof(float), sizeof(uint8_t)},
    {"f32_to_hif8_4x96", LaunchTCVT_f32_to_hif8_4x96, 4, 96, 4, 96, sizeof(float), sizeof(uint8_t)},
    {"f32_to_f16_1x128", LaunchTCVT_f32_to_f16_1x128, 1, 128, 1, 128, sizeof(float), sizeof(uint16_t)},
    {"f32_to_f16_2x64", LaunchTCVT_f32_to_f16_2x64, 2, 64, 2, 64, sizeof(float), sizeof(uint16_t)},
    {"f32_to_f16_4x32", LaunchTCVT_f32_to_f16_4x32, 4, 32, 4, 32, sizeof(float), sizeof(uint16_t)},
    {"f32_to_f16_2x128", LaunchTCVT_f32_to_f16_2x128, 2, 128, 2, 128, sizeof(float), sizeof(uint16_t)},
    {"f32_to_f16_4x65", LaunchTCVT_f32_to_f16_4x65, 4, 128, 4, 128, sizeof(float), sizeof(uint16_t)},
    {"f32_to_f16_4x200", LaunchTCVT_f32_to_f16_4x200, 4, 256, 4, 256, sizeof(float), sizeof(uint16_t)},
    {"f32_to_f16_1x129", LaunchTCVT_f32_to_f16_1x129, 1, 256, 1, 256, sizeof(float), sizeof(uint16_t)},
    {"f32_to_bf16_1x128", LaunchTCVT_f32_to_bf16_1x128, 1, 128, 1, 128, sizeof(float), sizeof(uint16_t)},
    {"f32_to_bf16_2x64", LaunchTCVT_f32_to_bf16_2x64, 2, 64, 2, 64, sizeof(float), sizeof(uint16_t)},
    {"f32_to_bf16_4x32", LaunchTCVT_f32_to_bf16_4x32, 4, 32, 4, 32, sizeof(float), sizeof(uint16_t)},
    {"f32_to_bf16_2x128", LaunchTCVT_f32_to_bf16_2x128, 2, 128, 2, 128, sizeof(float), sizeof(uint16_t)},
    {"f32_to_bf16_4x65", LaunchTCVT_f32_to_bf16_4x65, 4, 128, 4, 128, sizeof(float), sizeof(uint16_t)},
    {"f32_to_bf16_4x200", LaunchTCVT_f32_to_bf16_4x200, 4, 256, 4, 256, sizeof(float), sizeof(uint16_t)},
    {"f32_to_bf16_1x129", LaunchTCVT_f32_to_bf16_1x129, 1, 256, 1, 256, sizeof(float), sizeof(uint16_t)},
    {"f32_to_i16_1x128", LaunchTCVT_f32_to_i16_1x128, 1, 128, 1, 128, sizeof(float), sizeof(int16_t)},
    {"f32_to_i16_2x64", LaunchTCVT_f32_to_i16_2x64, 2, 64, 2, 64, sizeof(float), sizeof(int16_t)},
    {"f32_to_i16_4x32", LaunchTCVT_f32_to_i16_4x32, 4, 32, 4, 32, sizeof(float), sizeof(int16_t)},
    {"f32_to_i16_2x128", LaunchTCVT_f32_to_i16_2x128, 2, 128, 2, 128, sizeof(float), sizeof(int16_t)},
    {"f32_to_i16_4x65", LaunchTCVT_f32_to_i16_4x65, 4, 128, 4, 128, sizeof(float), sizeof(int16_t)},
    {"f32_to_i16_4x200", LaunchTCVT_f32_to_i16_4x200, 4, 256, 4, 256, sizeof(float), sizeof(int16_t)},
    {"f32_to_i16_1x129", LaunchTCVT_f32_to_i16_1x129, 1, 256, 1, 256, sizeof(float), sizeof(int16_t)},
    {"f32_to_i32_1x128", LaunchTCVT_f32_to_i32_1x128, 1, 128, 1, 128, sizeof(float), sizeof(int32_t)},
    {"f32_to_i32_2x64", LaunchTCVT_f32_to_i32_2x64, 2, 64, 2, 64, sizeof(float), sizeof(int32_t)},
    {"f32_to_i32_4x32", LaunchTCVT_f32_to_i32_4x32, 4, 32, 4, 32, sizeof(float), sizeof(int32_t)},
    {"f32_to_i32_2x128", LaunchTCVT_f32_to_i32_2x128, 2, 128, 2, 128, sizeof(float), sizeof(int32_t)},
    {"f32_to_i32_4x65", LaunchTCVT_f32_to_i32_4x65, 4, 128, 4, 128, sizeof(float), sizeof(int32_t)},
    {"f32_to_i32_4x200", LaunchTCVT_f32_to_i32_4x200, 4, 256, 4, 256, sizeof(float), sizeof(int32_t)},
    {"f32_to_i32_1x129", LaunchTCVT_f32_to_i32_1x129, 1, 256, 1, 256, sizeof(float), sizeof(int32_t)},
    {"f32_to_i64_1x128", LaunchTCVT_f32_to_i64_1x128, 1, 128, 1, 128, sizeof(float), sizeof(int64_t)},
    {"f32_to_i64_2x64", LaunchTCVT_f32_to_i64_2x64, 2, 64, 2, 64, sizeof(float), sizeof(int64_t)},
    {"f32_to_i64_4x32", LaunchTCVT_f32_to_i64_4x32, 4, 32, 4, 32, sizeof(float), sizeof(int64_t)},
    {"f32_to_i64_2x128", LaunchTCVT_f32_to_i64_2x128, 2, 128, 2, 128, sizeof(float), sizeof(int64_t)},
    {"f32_to_i64_4x65", LaunchTCVT_f32_to_i64_4x65, 4, 128, 4, 128, sizeof(float), sizeof(int64_t)},
    {"f32_to_i64_4x200", LaunchTCVT_f32_to_i64_4x200, 4, 256, 4, 256, sizeof(float), sizeof(int64_t)},
    {"f32_to_i64_1x129", LaunchTCVT_f32_to_i64_1x129, 1, 256, 1, 256, sizeof(float), sizeof(int64_t)},
    {"f32_to_f32_1x128", LaunchTCVT_f32_to_f32_1x128, 1, 128, 1, 128, sizeof(float), sizeof(float)},
    {"f32_to_f32_2x64", LaunchTCVT_f32_to_f32_2x64, 2, 64, 2, 64, sizeof(float), sizeof(float)},
    {"f32_to_f32_4x32", LaunchTCVT_f32_to_f32_4x32, 4, 32, 4, 32, sizeof(float), sizeof(float)},
    {"f32_to_f32_2x128", LaunchTCVT_f32_to_f32_2x128, 2, 128, 2, 128, sizeof(float), sizeof(float)},
    {"f32_to_f32_4x65", LaunchTCVT_f32_to_f32_4x65, 4, 128, 4, 128, sizeof(float), sizeof(float)},
    {"f32_to_f32_4x200", LaunchTCVT_f32_to_f32_4x200, 4, 256, 4, 256, sizeof(float), sizeof(float)},
    {"f32_to_f32_1x129", LaunchTCVT_f32_to_f32_1x129, 1, 256, 1, 256, sizeof(float), sizeof(float)},
    {"f16_to_f32_1x128", LaunchTCVT_f16_to_f32_1x128, 1, 128, 1, 128, sizeof(uint16_t), sizeof(float)},
    {"f16_to_f32_2x64", LaunchTCVT_f16_to_f32_2x64, 2, 64, 2, 64, sizeof(uint16_t), sizeof(float)},
    {"f16_to_f32_4x32", LaunchTCVT_f16_to_f32_4x32, 4, 32, 4, 32, sizeof(uint16_t), sizeof(float)},
    {"f16_to_f32_2x128", LaunchTCVT_f16_to_f32_2x128, 2, 128, 2, 128, sizeof(uint16_t), sizeof(float)},
    {"f16_to_f32_4x65", LaunchTCVT_f16_to_f32_4x65, 4, 128, 4, 128, sizeof(uint16_t), sizeof(float)},
    {"f16_to_f32_4x200", LaunchTCVT_f16_to_f32_4x200, 4, 256, 4, 256, sizeof(uint16_t), sizeof(float)},
    {"f16_to_f32_1x129", LaunchTCVT_f16_to_f32_1x129, 1, 256, 1, 256, sizeof(uint16_t), sizeof(float)},
    {"f16_to_i32_1x128", LaunchTCVT_f16_to_i32_1x128, 1, 128, 1, 128, sizeof(uint16_t), sizeof(int32_t)},
    {"f16_to_i32_2x64", LaunchTCVT_f16_to_i32_2x64, 2, 64, 2, 64, sizeof(uint16_t), sizeof(int32_t)},
    {"f16_to_i32_4x32", LaunchTCVT_f16_to_i32_4x32, 4, 32, 4, 32, sizeof(uint16_t), sizeof(int32_t)},
    {"f16_to_i32_2x128", LaunchTCVT_f16_to_i32_2x128, 2, 128, 2, 128, sizeof(uint16_t), sizeof(int32_t)},
    {"f16_to_i32_4x65", LaunchTCVT_f16_to_i32_4x65, 4, 128, 4, 128, sizeof(uint16_t), sizeof(int32_t)},
    {"f16_to_i32_4x200", LaunchTCVT_f16_to_i32_4x200, 4, 256, 4, 256, sizeof(uint16_t), sizeof(int32_t)},
    {"f16_to_i32_1x129", LaunchTCVT_f16_to_i32_1x129, 1, 256, 1, 256, sizeof(uint16_t), sizeof(int32_t)},
    {"f16_to_i16_1x128", LaunchTCVT_f16_to_i16_1x128, 1, 128, 1, 128, sizeof(uint16_t), sizeof(int16_t)},
    {"f16_to_i16_2x64", LaunchTCVT_f16_to_i16_2x64, 2, 64, 2, 64, sizeof(uint16_t), sizeof(int16_t)},
    {"f16_to_i16_4x32", LaunchTCVT_f16_to_i16_4x32, 4, 32, 4, 32, sizeof(uint16_t), sizeof(int16_t)},
    {"f16_to_i16_2x128", LaunchTCVT_f16_to_i16_2x128, 2, 128, 2, 128, sizeof(uint16_t), sizeof(int16_t)},
    {"f16_to_i16_4x65", LaunchTCVT_f16_to_i16_4x65, 4, 128, 4, 128, sizeof(uint16_t), sizeof(int16_t)},
    {"f16_to_i16_4x200", LaunchTCVT_f16_to_i16_4x200, 4, 256, 4, 256, sizeof(uint16_t), sizeof(int16_t)},
    {"f16_to_i16_1x129", LaunchTCVT_f16_to_i16_1x129, 1, 256, 1, 256, sizeof(uint16_t), sizeof(int16_t)},
    {"f16_to_si8_1x128", LaunchTCVT_f16_to_si8_1x128, 1, 128, 1, 128, sizeof(uint16_t), sizeof(int8_t)},
    {"f16_to_si8_2x64", LaunchTCVT_f16_to_si8_2x64, 2, 64, 2, 64, sizeof(uint16_t), sizeof(int8_t)},
    {"f16_to_si8_4x32", LaunchTCVT_f16_to_si8_4x32, 4, 32, 4, 32, sizeof(uint16_t), sizeof(int8_t)},
    {"f16_to_si8_2x128", LaunchTCVT_f16_to_si8_2x128, 2, 128, 2, 128, sizeof(uint16_t), sizeof(int8_t)},
    {"f16_to_si8_4x65", LaunchTCVT_f16_to_si8_4x65, 4, 128, 4, 128, sizeof(uint16_t), sizeof(int8_t)},
    {"f16_to_si8_4x200", LaunchTCVT_f16_to_si8_4x200, 4, 256, 4, 256, sizeof(uint16_t), sizeof(int8_t)},
    {"f16_to_si8_1x129", LaunchTCVT_f16_to_si8_1x129, 1, 256, 1, 256, sizeof(uint16_t), sizeof(int8_t)},
    {"f16_to_ui8_1x128", LaunchTCVT_f16_to_ui8_1x128, 1, 128, 1, 128, sizeof(uint16_t), sizeof(uint8_t)},
    {"f16_to_ui8_2x64", LaunchTCVT_f16_to_ui8_2x64, 2, 64, 2, 64, sizeof(uint16_t), sizeof(uint8_t)},
    {"f16_to_ui8_4x32", LaunchTCVT_f16_to_ui8_4x32, 4, 32, 4, 32, sizeof(uint16_t), sizeof(uint8_t)},
    {"f16_to_ui8_2x128", LaunchTCVT_f16_to_ui8_2x128, 2, 128, 2, 128, sizeof(uint16_t), sizeof(uint8_t)},
    {"f16_to_ui8_4x65", LaunchTCVT_f16_to_ui8_4x65, 4, 128, 4, 128, sizeof(uint16_t), sizeof(uint8_t)},
    {"f16_to_ui8_4x200", LaunchTCVT_f16_to_ui8_4x200, 4, 256, 4, 256, sizeof(uint16_t), sizeof(uint8_t)},
    {"f16_to_ui8_1x129", LaunchTCVT_f16_to_ui8_1x129, 1, 256, 1, 256, sizeof(uint16_t), sizeof(uint8_t)},
    {"bf16_to_f32_1x128", LaunchTCVT_bf16_to_f32_1x128, 1, 128, 1, 128, sizeof(uint16_t), sizeof(float)},
    {"bf16_to_f32_2x64", LaunchTCVT_bf16_to_f32_2x64, 2, 64, 2, 64, sizeof(uint16_t), sizeof(float)},
    {"bf16_to_f32_4x32", LaunchTCVT_bf16_to_f32_4x32, 4, 32, 4, 32, sizeof(uint16_t), sizeof(float)},
    {"bf16_to_f32_2x128", LaunchTCVT_bf16_to_f32_2x128, 2, 128, 2, 128, sizeof(uint16_t), sizeof(float)},
    {"bf16_to_f32_4x65", LaunchTCVT_bf16_to_f32_4x65, 4, 128, 4, 128, sizeof(uint16_t), sizeof(float)},
    {"bf16_to_f32_4x200", LaunchTCVT_bf16_to_f32_4x200, 4, 256, 4, 256, sizeof(uint16_t), sizeof(float)},
    {"bf16_to_f32_1x129", LaunchTCVT_bf16_to_f32_1x129, 1, 256, 1, 256, sizeof(uint16_t), sizeof(float)},
    {"bf16_to_f16_1x128", LaunchTCVT_bf16_to_f16_1x128, 1, 128, 1, 128, sizeof(uint16_t), sizeof(uint16_t)},
    {"bf16_to_f16_2x64", LaunchTCVT_bf16_to_f16_2x64, 2, 64, 2, 64, sizeof(uint16_t), sizeof(uint16_t)},
    {"bf16_to_f16_4x32", LaunchTCVT_bf16_to_f16_4x32, 4, 32, 4, 32, sizeof(uint16_t), sizeof(uint16_t)},
    {"bf16_to_f16_2x128", LaunchTCVT_bf16_to_f16_2x128, 2, 128, 2, 128, sizeof(uint16_t), sizeof(uint16_t)},
    {"bf16_to_f16_4x65", LaunchTCVT_bf16_to_f16_4x65, 4, 128, 4, 128, sizeof(uint16_t), sizeof(uint16_t)},
    {"bf16_to_f16_4x200", LaunchTCVT_bf16_to_f16_4x200, 4, 256, 4, 256, sizeof(uint16_t), sizeof(uint16_t)},
    {"bf16_to_f16_1x129", LaunchTCVT_bf16_to_f16_1x129, 1, 256, 1, 256, sizeof(uint16_t), sizeof(uint16_t)},
    {"bf16_to_i32_1x128", LaunchTCVT_bf16_to_i32_1x128, 1, 128, 1, 128, sizeof(uint16_t), sizeof(int32_t)},
    {"bf16_to_i32_2x64", LaunchTCVT_bf16_to_i32_2x64, 2, 64, 2, 64, sizeof(uint16_t), sizeof(int32_t)},
    {"bf16_to_i32_4x32", LaunchTCVT_bf16_to_i32_4x32, 4, 32, 4, 32, sizeof(uint16_t), sizeof(int32_t)},
    {"bf16_to_i32_2x128", LaunchTCVT_bf16_to_i32_2x128, 2, 128, 2, 128, sizeof(uint16_t), sizeof(int32_t)},
    {"bf16_to_i32_4x65", LaunchTCVT_bf16_to_i32_4x65, 4, 128, 4, 128, sizeof(uint16_t), sizeof(int32_t)},
    {"bf16_to_i32_4x200", LaunchTCVT_bf16_to_i32_4x200, 4, 256, 4, 256, sizeof(uint16_t), sizeof(int32_t)},
    {"bf16_to_i32_1x129", LaunchTCVT_bf16_to_i32_1x129, 1, 256, 1, 256, sizeof(uint16_t), sizeof(int32_t)},
    {"ui8_to_f16_1x128", LaunchTCVT_ui8_to_f16_1x128, 1, 128, 1, 128, sizeof(uint8_t), sizeof(uint16_t)},
    {"ui8_to_f16_2x64", LaunchTCVT_ui8_to_f16_2x64, 2, 64, 2, 64, sizeof(uint8_t), sizeof(uint16_t)},
    {"ui8_to_f16_4x32", LaunchTCVT_ui8_to_f16_4x32, 4, 32, 4, 32, sizeof(uint8_t), sizeof(uint16_t)},
    {"ui8_to_f16_2x128", LaunchTCVT_ui8_to_f16_2x128, 2, 128, 2, 128, sizeof(uint8_t), sizeof(uint16_t)},
    {"ui8_to_f16_4x65", LaunchTCVT_ui8_to_f16_4x65, 4, 128, 4, 128, sizeof(uint8_t), sizeof(uint16_t)},
    {"ui8_to_f16_4x200", LaunchTCVT_ui8_to_f16_4x200, 4, 256, 4, 256, sizeof(uint8_t), sizeof(uint16_t)},
    {"ui8_to_f16_1x129", LaunchTCVT_ui8_to_f16_1x129, 1, 256, 1, 256, sizeof(uint8_t), sizeof(uint16_t)},
    {"ui8_to_ui16_1x128", LaunchTCVT_ui8_to_ui16_1x128, 1, 128, 1, 128, sizeof(uint8_t), sizeof(uint16_t)},
    {"ui8_to_ui16_2x64", LaunchTCVT_ui8_to_ui16_2x64, 2, 64, 2, 64, sizeof(uint8_t), sizeof(uint16_t)},
    {"ui8_to_ui16_4x32", LaunchTCVT_ui8_to_ui16_4x32, 4, 32, 4, 32, sizeof(uint8_t), sizeof(uint16_t)},
    {"ui8_to_ui16_2x128", LaunchTCVT_ui8_to_ui16_2x128, 2, 128, 2, 128, sizeof(uint8_t), sizeof(uint16_t)},
    {"ui8_to_ui16_4x65", LaunchTCVT_ui8_to_ui16_4x65, 4, 128, 4, 128, sizeof(uint8_t), sizeof(uint16_t)},
    {"ui8_to_ui16_4x200", LaunchTCVT_ui8_to_ui16_4x200, 4, 256, 4, 256, sizeof(uint8_t), sizeof(uint16_t)},
    {"ui8_to_ui16_1x129", LaunchTCVT_ui8_to_ui16_1x129, 1, 256, 1, 256, sizeof(uint8_t), sizeof(uint16_t)},
    {"si8_to_f16_1x128", LaunchTCVT_si8_to_f16_1x128, 1, 128, 1, 128, sizeof(int8_t), sizeof(uint16_t)},
    {"si8_to_f16_2x64", LaunchTCVT_si8_to_f16_2x64, 2, 64, 2, 64, sizeof(int8_t), sizeof(uint16_t)},
    {"si8_to_f16_4x32", LaunchTCVT_si8_to_f16_4x32, 4, 32, 4, 32, sizeof(int8_t), sizeof(uint16_t)},
    {"si8_to_f16_2x128", LaunchTCVT_si8_to_f16_2x128, 2, 128, 2, 128, sizeof(int8_t), sizeof(uint16_t)},
    {"si8_to_f16_4x65", LaunchTCVT_si8_to_f16_4x65, 4, 128, 4, 128, sizeof(int8_t), sizeof(uint16_t)},
    {"si8_to_f16_4x200", LaunchTCVT_si8_to_f16_4x200, 4, 256, 4, 256, sizeof(int8_t), sizeof(uint16_t)},
    {"si8_to_f16_1x129", LaunchTCVT_si8_to_f16_1x129, 1, 256, 1, 256, sizeof(int8_t), sizeof(uint16_t)},
    {"si8_to_si16_1x128", LaunchTCVT_si8_to_si16_1x128, 1, 128, 1, 128, sizeof(int8_t), sizeof(int16_t)},
    {"si8_to_si16_2x64", LaunchTCVT_si8_to_si16_2x64, 2, 64, 2, 64, sizeof(int8_t), sizeof(int16_t)},
    {"si8_to_si16_4x32", LaunchTCVT_si8_to_si16_4x32, 4, 32, 4, 32, sizeof(int8_t), sizeof(int16_t)},
    {"si8_to_si16_2x128", LaunchTCVT_si8_to_si16_2x128, 2, 128, 2, 128, sizeof(int8_t), sizeof(int16_t)},
    {"si8_to_si16_4x65", LaunchTCVT_si8_to_si16_4x65, 4, 128, 4, 128, sizeof(int8_t), sizeof(int16_t)},
    {"si8_to_si16_4x200", LaunchTCVT_si8_to_si16_4x200, 4, 256, 4, 256, sizeof(int8_t), sizeof(int16_t)},
    {"si8_to_si16_1x129", LaunchTCVT_si8_to_si16_1x129, 1, 256, 1, 256, sizeof(int8_t), sizeof(int16_t)},
    {"si8_to_i32_1x128", LaunchTCVT_si8_to_i32_1x128, 1, 128, 1, 128, sizeof(int8_t), sizeof(int32_t)},
    {"si8_to_i32_2x64", LaunchTCVT_si8_to_i32_2x64, 2, 64, 2, 64, sizeof(int8_t), sizeof(int32_t)},
    {"si8_to_i32_4x32", LaunchTCVT_si8_to_i32_4x32, 4, 32, 4, 32, sizeof(int8_t), sizeof(int32_t)},
    {"si8_to_i32_2x128", LaunchTCVT_si8_to_i32_2x128, 2, 128, 2, 128, sizeof(int8_t), sizeof(int32_t)},
    {"si8_to_i32_4x65", LaunchTCVT_si8_to_i32_4x65, 4, 128, 4, 128, sizeof(int8_t), sizeof(int32_t)},
    {"si8_to_i32_4x200", LaunchTCVT_si8_to_i32_4x200, 4, 256, 4, 256, sizeof(int8_t), sizeof(int32_t)},
    {"si8_to_i32_1x129", LaunchTCVT_si8_to_i32_1x129, 1, 256, 1, 256, sizeof(int8_t), sizeof(int32_t)},
    {"i16_to_ui8_1x128", LaunchTCVT_i16_to_ui8_1x128, 1, 128, 1, 128, sizeof(int16_t), sizeof(uint8_t)},
    {"i16_to_ui8_2x64", LaunchTCVT_i16_to_ui8_2x64, 2, 64, 2, 64, sizeof(int16_t), sizeof(uint8_t)},
    {"i16_to_ui8_4x32", LaunchTCVT_i16_to_ui8_4x32, 4, 32, 4, 32, sizeof(int16_t), sizeof(uint8_t)},
    {"i16_to_ui8_2x128", LaunchTCVT_i16_to_ui8_2x128, 2, 128, 2, 128, sizeof(int16_t), sizeof(uint8_t)},
    {"i16_to_ui8_4x65", LaunchTCVT_i16_to_ui8_4x65, 4, 128, 4, 128, sizeof(int16_t), sizeof(uint8_t)},
    {"i16_to_ui8_4x200", LaunchTCVT_i16_to_ui8_4x200, 4, 256, 4, 256, sizeof(int16_t), sizeof(uint8_t)},
    {"i16_to_ui8_1x129", LaunchTCVT_i16_to_ui8_1x129, 1, 256, 1, 256, sizeof(int16_t), sizeof(uint8_t)},
    {"i16_to_f16_1x128", LaunchTCVT_i16_to_f16_1x128, 1, 128, 1, 128, sizeof(int16_t), sizeof(uint16_t)},
    {"i16_to_f16_2x64", LaunchTCVT_i16_to_f16_2x64, 2, 64, 2, 64, sizeof(int16_t), sizeof(uint16_t)},
    {"i16_to_f16_4x32", LaunchTCVT_i16_to_f16_4x32, 4, 32, 4, 32, sizeof(int16_t), sizeof(uint16_t)},
    {"i16_to_f16_2x128", LaunchTCVT_i16_to_f16_2x128, 2, 128, 2, 128, sizeof(int16_t), sizeof(uint16_t)},
    {"i16_to_f16_4x65", LaunchTCVT_i16_to_f16_4x65, 4, 128, 4, 128, sizeof(int16_t), sizeof(uint16_t)},
    {"i16_to_f16_4x200", LaunchTCVT_i16_to_f16_4x200, 4, 256, 4, 256, sizeof(int16_t), sizeof(uint16_t)},
    {"i16_to_f16_1x129", LaunchTCVT_i16_to_f16_1x129, 1, 256, 1, 256, sizeof(int16_t), sizeof(uint16_t)},
    {"i16_to_f32_1x128", LaunchTCVT_i16_to_f32_1x128, 1, 128, 1, 128, sizeof(int16_t), sizeof(float)},
    {"i16_to_f32_2x64", LaunchTCVT_i16_to_f32_2x64, 2, 64, 2, 64, sizeof(int16_t), sizeof(float)},
    {"i16_to_f32_4x32", LaunchTCVT_i16_to_f32_4x32, 4, 32, 4, 32, sizeof(int16_t), sizeof(float)},
    {"i16_to_f32_2x128", LaunchTCVT_i16_to_f32_2x128, 2, 128, 2, 128, sizeof(int16_t), sizeof(float)},
    {"i16_to_f32_4x65", LaunchTCVT_i16_to_f32_4x65, 4, 128, 4, 128, sizeof(int16_t), sizeof(float)},
    {"i16_to_f32_4x200", LaunchTCVT_i16_to_f32_4x200, 4, 256, 4, 256, sizeof(int16_t), sizeof(float)},
    {"i16_to_f32_1x129", LaunchTCVT_i16_to_f32_1x129, 1, 256, 1, 256, sizeof(int16_t), sizeof(float)},
    {"i16_to_ui32_1x128", LaunchTCVT_i16_to_ui32_1x128, 1, 128, 1, 128, sizeof(int16_t), sizeof(uint32_t)},
    {"i16_to_ui32_2x64", LaunchTCVT_i16_to_ui32_2x64, 2, 64, 2, 64, sizeof(int16_t), sizeof(uint32_t)},
    {"i16_to_ui32_4x32", LaunchTCVT_i16_to_ui32_4x32, 4, 32, 4, 32, sizeof(int16_t), sizeof(uint32_t)},
    {"i16_to_ui32_2x128", LaunchTCVT_i16_to_ui32_2x128, 2, 128, 2, 128, sizeof(int16_t), sizeof(uint32_t)},
    {"i16_to_ui32_4x65", LaunchTCVT_i16_to_ui32_4x65, 4, 128, 4, 128, sizeof(int16_t), sizeof(uint32_t)},
    {"i16_to_ui32_4x200", LaunchTCVT_i16_to_ui32_4x200, 4, 256, 4, 256, sizeof(int16_t), sizeof(uint32_t)},
    {"i16_to_ui32_1x129", LaunchTCVT_i16_to_ui32_1x129, 1, 256, 1, 256, sizeof(int16_t), sizeof(uint32_t)},
    {"i16_to_i32_1x128", LaunchTCVT_i16_to_i32_1x128, 1, 128, 1, 128, sizeof(int16_t), sizeof(int32_t)},
    {"i16_to_i32_2x64", LaunchTCVT_i16_to_i32_2x64, 2, 64, 2, 64, sizeof(int16_t), sizeof(int32_t)},
    {"i16_to_i32_4x32", LaunchTCVT_i16_to_i32_4x32, 4, 32, 4, 32, sizeof(int16_t), sizeof(int32_t)},
    {"i16_to_i32_2x128", LaunchTCVT_i16_to_i32_2x128, 2, 128, 2, 128, sizeof(int16_t), sizeof(int32_t)},
    {"i16_to_i32_4x65", LaunchTCVT_i16_to_i32_4x65, 4, 128, 4, 128, sizeof(int16_t), sizeof(int32_t)},
    {"i16_to_i32_4x200", LaunchTCVT_i16_to_i32_4x200, 4, 256, 4, 256, sizeof(int16_t), sizeof(int32_t)},
    {"i16_to_i32_1x129", LaunchTCVT_i16_to_i32_1x129, 1, 256, 1, 256, sizeof(int16_t), sizeof(int32_t)},
    {"i32_to_f32_1x128", LaunchTCVT_i32_to_f32_1x128, 1, 128, 1, 128, sizeof(int32_t), sizeof(float)},
    {"i32_to_f32_2x64", LaunchTCVT_i32_to_f32_2x64, 2, 64, 2, 64, sizeof(int32_t), sizeof(float)},
    {"i32_to_f32_4x32", LaunchTCVT_i32_to_f32_4x32, 4, 32, 4, 32, sizeof(int32_t), sizeof(float)},
    {"i32_to_f32_2x128", LaunchTCVT_i32_to_f32_2x128, 2, 128, 2, 128, sizeof(int32_t), sizeof(float)},
    {"i32_to_f32_4x65", LaunchTCVT_i32_to_f32_4x65, 4, 128, 4, 128, sizeof(int32_t), sizeof(float)},
    {"i32_to_f32_4x200", LaunchTCVT_i32_to_f32_4x200, 4, 256, 4, 256, sizeof(int32_t), sizeof(float)},
    {"i32_to_f32_1x129", LaunchTCVT_i32_to_f32_1x129, 1, 256, 1, 256, sizeof(int32_t), sizeof(float)},
    {"i32_to_i16_1x128", LaunchTCVT_i32_to_i16_1x128, 1, 128, 1, 128, sizeof(int32_t), sizeof(int16_t)},
    {"i32_to_i16_2x64", LaunchTCVT_i32_to_i16_2x64, 2, 64, 2, 64, sizeof(int32_t), sizeof(int16_t)},
    {"i32_to_i16_4x32", LaunchTCVT_i32_to_i16_4x32, 4, 32, 4, 32, sizeof(int32_t), sizeof(int16_t)},
    {"i32_to_i16_2x128", LaunchTCVT_i32_to_i16_2x128, 2, 128, 2, 128, sizeof(int32_t), sizeof(int16_t)},
    {"i32_to_i16_4x65", LaunchTCVT_i32_to_i16_4x65, 4, 128, 4, 128, sizeof(int32_t), sizeof(int16_t)},
    {"i32_to_i16_4x200", LaunchTCVT_i32_to_i16_4x200, 4, 256, 4, 256, sizeof(int32_t), sizeof(int16_t)},
    {"i32_to_i16_1x129", LaunchTCVT_i32_to_i16_1x129, 1, 256, 1, 256, sizeof(int32_t), sizeof(int16_t)},
    {"i32_to_i64_1x128", LaunchTCVT_i32_to_i64_1x128, 1, 128, 1, 128, sizeof(int32_t), sizeof(int64_t)},
    {"i32_to_i64_2x64", LaunchTCVT_i32_to_i64_2x64, 2, 64, 2, 64, sizeof(int32_t), sizeof(int64_t)},
    {"i32_to_i64_4x32", LaunchTCVT_i32_to_i64_4x32, 4, 32, 4, 32, sizeof(int32_t), sizeof(int64_t)},
    {"i32_to_i64_2x128", LaunchTCVT_i32_to_i64_2x128, 2, 128, 2, 128, sizeof(int32_t), sizeof(int64_t)},
    {"i32_to_i64_4x65", LaunchTCVT_i32_to_i64_4x65, 4, 128, 4, 128, sizeof(int32_t), sizeof(int64_t)},
    {"i32_to_i64_4x200", LaunchTCVT_i32_to_i64_4x200, 4, 256, 4, 256, sizeof(int32_t), sizeof(int64_t)},
    {"i32_to_i64_1x129", LaunchTCVT_i32_to_i64_1x129, 1, 256, 1, 256, sizeof(int32_t), sizeof(int64_t)},
    {"i32_to_ui8_1x128", LaunchTCVT_i32_to_ui8_1x128, 1, 128, 1, 128, sizeof(int32_t), sizeof(uint8_t)},
    {"i32_to_ui8_2x64", LaunchTCVT_i32_to_ui8_2x64, 2, 64, 2, 64, sizeof(int32_t), sizeof(uint8_t)},
    {"i32_to_ui8_4x32", LaunchTCVT_i32_to_ui8_4x32, 4, 32, 4, 32, sizeof(int32_t), sizeof(uint8_t)},
    {"i32_to_ui8_2x128", LaunchTCVT_i32_to_ui8_2x128, 2, 128, 2, 128, sizeof(int32_t), sizeof(uint8_t)},
    {"i32_to_ui8_4x65", LaunchTCVT_i32_to_ui8_4x65, 4, 128, 4, 128, sizeof(int32_t), sizeof(uint8_t)},
    {"i32_to_ui8_4x200", LaunchTCVT_i32_to_ui8_4x200, 4, 256, 4, 256, sizeof(int32_t), sizeof(uint8_t)},
    {"i32_to_ui8_1x129", LaunchTCVT_i32_to_ui8_1x129, 1, 256, 1, 256, sizeof(int32_t), sizeof(uint8_t)},
    {"i32_to_ui16_1x128", LaunchTCVT_i32_to_ui16_1x128, 1, 128, 1, 128, sizeof(int32_t), sizeof(uint16_t)},
    {"i32_to_ui16_2x64", LaunchTCVT_i32_to_ui16_2x64, 2, 64, 2, 64, sizeof(int32_t), sizeof(uint16_t)},
    {"i32_to_ui16_4x32", LaunchTCVT_i32_to_ui16_4x32, 4, 32, 4, 32, sizeof(int32_t), sizeof(uint16_t)},
    {"i32_to_ui16_2x128", LaunchTCVT_i32_to_ui16_2x128, 2, 128, 2, 128, sizeof(int32_t), sizeof(uint16_t)},
    {"i32_to_ui16_4x65", LaunchTCVT_i32_to_ui16_4x65, 4, 128, 4, 128, sizeof(int32_t), sizeof(uint16_t)},
    {"i32_to_ui16_4x200", LaunchTCVT_i32_to_ui16_4x200, 4, 256, 4, 256, sizeof(int32_t), sizeof(uint16_t)},
    {"i32_to_ui16_1x129", LaunchTCVT_i32_to_ui16_1x129, 1, 256, 1, 256, sizeof(int32_t), sizeof(uint16_t)},
    {"ui32_to_i16_1x128", LaunchTCVT_ui32_to_i16_1x128, 1, 128, 1, 128, sizeof(uint32_t), sizeof(int16_t)},
    {"ui32_to_i16_2x64", LaunchTCVT_ui32_to_i16_2x64, 2, 64, 2, 64, sizeof(uint32_t), sizeof(int16_t)},
    {"ui32_to_i16_4x32", LaunchTCVT_ui32_to_i16_4x32, 4, 32, 4, 32, sizeof(uint32_t), sizeof(int16_t)},
    {"ui32_to_i16_2x128", LaunchTCVT_ui32_to_i16_2x128, 2, 128, 2, 128, sizeof(uint32_t), sizeof(int16_t)},
    {"ui32_to_i16_4x65", LaunchTCVT_ui32_to_i16_4x65, 4, 128, 4, 128, sizeof(uint32_t), sizeof(int16_t)},
    {"ui32_to_i16_4x200", LaunchTCVT_ui32_to_i16_4x200, 4, 256, 4, 256, sizeof(uint32_t), sizeof(int16_t)},
    {"ui32_to_i16_1x129", LaunchTCVT_ui32_to_i16_1x129, 1, 256, 1, 256, sizeof(uint32_t), sizeof(int16_t)},
    {"ui32_to_ui16_1x128", LaunchTCVT_ui32_to_ui16_1x128, 1, 128, 1, 128, sizeof(uint32_t), sizeof(uint16_t)},
    {"ui32_to_ui16_2x64", LaunchTCVT_ui32_to_ui16_2x64, 2, 64, 2, 64, sizeof(uint32_t), sizeof(uint16_t)},
    {"ui32_to_ui16_4x32", LaunchTCVT_ui32_to_ui16_4x32, 4, 32, 4, 32, sizeof(uint32_t), sizeof(uint16_t)},
    {"ui32_to_ui16_2x128", LaunchTCVT_ui32_to_ui16_2x128, 2, 128, 2, 128, sizeof(uint32_t), sizeof(uint16_t)},
    {"ui32_to_ui16_4x65", LaunchTCVT_ui32_to_ui16_4x65, 4, 128, 4, 128, sizeof(uint32_t), sizeof(uint16_t)},
    {"ui32_to_ui16_4x200", LaunchTCVT_ui32_to_ui16_4x200, 4, 256, 4, 256, sizeof(uint32_t), sizeof(uint16_t)},
    {"ui32_to_ui16_1x129", LaunchTCVT_ui32_to_ui16_1x129, 1, 256, 1, 256, sizeof(uint32_t), sizeof(uint16_t)},
    {"ui32_to_ui8_1x128", LaunchTCVT_ui32_to_ui8_1x128, 1, 128, 1, 128, sizeof(uint32_t), sizeof(uint8_t)},
    {"ui32_to_ui8_2x64", LaunchTCVT_ui32_to_ui8_2x64, 2, 64, 2, 64, sizeof(uint32_t), sizeof(uint8_t)},
    {"ui32_to_ui8_4x32", LaunchTCVT_ui32_to_ui8_4x32, 4, 32, 4, 32, sizeof(uint32_t), sizeof(uint8_t)},
    {"ui32_to_ui8_2x128", LaunchTCVT_ui32_to_ui8_2x128, 2, 128, 2, 128, sizeof(uint32_t), sizeof(uint8_t)},
    {"ui32_to_ui8_4x65", LaunchTCVT_ui32_to_ui8_4x65, 4, 128, 4, 128, sizeof(uint32_t), sizeof(uint8_t)},
    {"ui32_to_ui8_4x200", LaunchTCVT_ui32_to_ui8_4x200, 4, 256, 4, 256, sizeof(uint32_t), sizeof(uint8_t)},
    {"ui32_to_ui8_1x129", LaunchTCVT_ui32_to_ui8_1x129, 1, 256, 1, 256, sizeof(uint32_t), sizeof(uint8_t)},
    {"i64_to_f32_1x128", LaunchTCVT_i64_to_f32_1x128, 1, 128, 1, 128, sizeof(int64_t), sizeof(float)},
    {"i64_to_f32_2x64", LaunchTCVT_i64_to_f32_2x64, 2, 64, 2, 64, sizeof(int64_t), sizeof(float)},
    {"i64_to_f32_4x32", LaunchTCVT_i64_to_f32_4x32, 4, 32, 4, 32, sizeof(int64_t), sizeof(float)},
    {"i64_to_f32_2x128", LaunchTCVT_i64_to_f32_2x128, 2, 128, 2, 128, sizeof(int64_t), sizeof(float)},
    {"i64_to_f32_4x65", LaunchTCVT_i64_to_f32_4x65, 4, 128, 4, 128, sizeof(int64_t), sizeof(float)},
    {"i64_to_f32_4x200", LaunchTCVT_i64_to_f32_4x200, 4, 256, 4, 256, sizeof(int64_t), sizeof(float)},
    {"i64_to_f32_1x129", LaunchTCVT_i64_to_f32_1x129, 1, 256, 1, 256, sizeof(int64_t), sizeof(float)},
    {"i64_to_i32_1x128", LaunchTCVT_i64_to_i32_1x128, 1, 128, 1, 128, sizeof(int64_t), sizeof(int32_t)},
    {"i64_to_i32_2x64", LaunchTCVT_i64_to_i32_2x64, 2, 64, 2, 64, sizeof(int64_t), sizeof(int32_t)},
    {"i64_to_i32_4x32", LaunchTCVT_i64_to_i32_4x32, 4, 32, 4, 32, sizeof(int64_t), sizeof(int32_t)},
    {"i64_to_i32_2x128", LaunchTCVT_i64_to_i32_2x128, 2, 128, 2, 128, sizeof(int64_t), sizeof(int32_t)},
    {"i64_to_i32_4x65", LaunchTCVT_i64_to_i32_4x65, 4, 128, 4, 128, sizeof(int64_t), sizeof(int32_t)},
    {"i64_to_i32_4x200", LaunchTCVT_i64_to_i32_4x200, 4, 256, 4, 256, sizeof(int64_t), sizeof(int32_t)},
    {"i64_to_i32_1x129", LaunchTCVT_i64_to_i32_1x129, 1, 256, 1, 256, sizeof(int64_t), sizeof(int32_t)},
};
static constexpr size_t kNumCases = sizeof(kCases) / sizeof(kCases[0]);

static int RunCase(const TestCase &tc, int deviceId, aclrtStream stream) {
    (void)deviceId;
    int rc = 0;
    const size_t srcElemCount = tc.srcRows * tc.srcCols;
    const size_t dstElemCount = tc.dstRows * tc.dstCols;
    size_t srcFileSize = srcElemCount * tc.srcElemSize;
    size_t dstFileSize = dstElemCount * tc.dstElemSize;

    std::printf("[INFO] === case: %s (src=%zux%zu, dst=%zux%zu) ===\n",
                tc.name, tc.srcRows, tc.srcCols, tc.dstRows, tc.dstCols);

    std::string caseDir = std::string("./") + tc.name;

    void *srcHost = nullptr;
    void *dstHost = nullptr;
    void *srcDevice = nullptr;
    void *dstDevice = nullptr;

    aclrtMallocHost(&srcHost, srcFileSize);
    aclrtMallocHost(&dstHost, dstFileSize);

    aclrtMalloc(&srcDevice, srcFileSize, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc(&dstDevice, dstFileSize, ACL_MEM_MALLOC_HUGE_FIRST);

    if (!ReadFile((caseDir + "/input.bin").c_str(), srcFileSize, srcHost, srcFileSize)) {
        std::fprintf(stderr, "[ERROR] failed to read %s/input.bin\n", caseDir.c_str());
        rc = 1;
    }

    if (rc == 0) {
        aclrtMemcpy(srcDevice, srcFileSize, srcHost, srcFileSize, ACL_MEMCPY_HOST_TO_DEVICE);
        tc.launch(srcDevice, dstDevice, stream);
        aclrtSynchronizeStream(stream);
        aclrtMemcpy(dstHost, dstFileSize, dstDevice, dstFileSize, ACL_MEMCPY_DEVICE_TO_HOST);
    }

    if (rc == 0 && !WriteFile((caseDir + "/output.bin").c_str(), dstHost, dstFileSize)) {
        std::fprintf(stderr, "[ERROR] failed to write %s/output.bin\n", caseDir.c_str());
        rc = 1;
    }

    if (srcDevice != nullptr)
        aclrtFree(srcDevice);
    if (dstDevice != nullptr)
        aclrtFree(dstDevice);
    if (srcHost != nullptr)
        aclrtFreeHost(srcHost);
    if (dstHost != nullptr)
        aclrtFreeHost(dstHost);

    if (rc == 0)
        std::printf("[INFO] case %s done\n", tc.name);
    return rc;
}

int main(int argc, char *argv[]) {
    const char *caseFilter = (argc > 1) ? argv[1] : nullptr;

    int rc = 0;
    int deviceId = 0;
    aclrtStream stream = nullptr;

    aclInit(nullptr);
    if (const char *envDevice = std::getenv("ACL_DEVICE_ID")) {
        deviceId = std::atoi(envDevice);
    }
    aclrtSetDevice(deviceId);
    aclrtCreateStream(&stream);

    for (size_t i = 0; i < kNumCases; ++i) {
        if (caseFilter != nullptr && std::strcmp(kCases[i].name, caseFilter) != 0) {
            continue;
        }
        int ret = RunCase(kCases[i], deviceId, stream);
        if (ret != 0) {
            std::fprintf(stderr, "[ERROR] case %s failed\n", kCases[i].name);
            rc = 1;
            break;
        }
    }

    if (stream != nullptr)
        aclrtDestroyStream(stream);
    aclrtResetDevice(deviceId);
    aclFinalize();

    return rc;
}
