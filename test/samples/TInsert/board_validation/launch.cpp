// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

// ---------------------------------------------------------------------------
// PTOAS compatibility layer
//
// The upstream pto-isa headers reference some FP8/FP4 types and the
// __VEC_SCOPE__ marker that are not available on every AICore arch/toolchain
// combination (e.g. __NPU_ARCH__==2201).
//
// For our PTOAS-generated kernels we don't rely on these types today, but the
// headers still mention them in templates/static_asserts. Provide minimal
// fallbacks to keep compilation working on dav-c220.
// ---------------------------------------------------------------------------
#ifndef __VEC_SCOPE__
#define __VEC_SCOPE__
#endif

#if defined(__CCE_AICORE__) && defined(__NPU_ARCH__) && (__NPU_ARCH__ == 2201)
typedef struct { unsigned char v; } hifloat8_t;
typedef struct { unsigned char v; } float8_e4m3_t;
typedef struct { unsigned char v; } float8_e5m2_t;
typedef struct { unsigned char v; } float8_e8m0_t;
typedef struct { unsigned char v; } float4_e1m2x2_t;
typedef struct { unsigned char v; } float4_e2m1x2_t;
#endif
#include <stdint.h>

// AICore printf support is gated behind `--cce-enable-print` on some
// toolchains. When enabled, include the CCE print header so `cce::printf`
// resolves in device compilation.
#if defined(__CCE_AICORE__) && defined(PTOAS_ENABLE_CCE_PRINT)
#include <ccelib/print/print.h>
#endif
#include <pto/pto-inst.hpp>
#include <pto/common/constants.hpp>

// Some PTO-ISA types are only available in the __CCE_AICORE__ compilation
// path, but `bisheng -xcce` still performs a host-side parse pass.
// Provide minimal fallbacks only when the corresponding header wasn't
// pulled in by the selected arch implementation.
#if !defined(__CCE_AICORE__) && !defined(TMRGSORT_HPP)
namespace pto {
struct MrgSortExecutedNumList {
    uint16_t mrgSortList0;
    uint16_t mrgSortList1;
    uint16_t mrgSortList2;
    uint16_t mrgSortList3;
};
} // namespace pto
#endif
#ifndef __CPU_SIM
#include "acl/acl.h"
#endif

#if defined(__CCE_AICORE__)
__global__ AICORE void tinsert_kernel(__gm__ float* v1, __gm__ float* v2, __gm__ half* v3, __gm__ float* v4);
#else
__global__ AICORE void tinsert_kernel(__gm__ float* v1, __gm__ float* v2, __gm__ half* v3, __gm__ float* v4);
#endif

void LaunchTInsert_kernel(float *v1, float *v2, aclFloat16 *v3, float *v4, void *stream) {
#if defined(__CCE_AICORE__)
    tinsert_kernel<<<1, nullptr, stream>>>((__gm__ float*)v1, (__gm__ float*)v2, (__gm__ half*)v3, (__gm__ float*)v4);
#else
    tinsert_kernel<<<1, nullptr, stream>>>((__gm__ float*)v1, (__gm__ float*)v2, (__gm__ half*)v3, (__gm__ float*)v4);
#endif
}
