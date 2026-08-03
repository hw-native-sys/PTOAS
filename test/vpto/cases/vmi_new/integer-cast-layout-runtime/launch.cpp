// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

#ifndef __VEC_SCOPE__
#define __VEC_SCOPE__
#endif
#include <cstdint>
#if !defined(__CCE_AICORE__) && !defined(TMRGSORT_HPP)
struct MrgSortExecutedNumList {
  uint16_t mrgSortList0;
  uint16_t mrgSortList1;
  uint16_t mrgSortList2;
  uint16_t mrgSortList3;
};
#endif
#ifndef __CPU_SIM
#include "acl/acl.h"
#endif

extern "C" __global__[aicore] void
integer_cast_u8_kernel(__gm__ uint8_t *src, __gm__ uint16_t *dst16,
                       __gm__ uint32_t *dst32);
extern "C" __global__[aicore] void
integer_cast_i8_kernel(__gm__ int8_t *src, __gm__ int16_t *dst16,
                       __gm__ int32_t *dst32);
extern "C" __global__[aicore] void
integer_cast_u16_kernel(__gm__ uint16_t *src, __gm__ uint32_t *dst32,
                        __gm__ uint8_t *dst8);
extern "C" __global__[aicore] void
integer_cast_i16_kernel(__gm__ int16_t *src, __gm__ int32_t *dst32,
                        __gm__ int8_t *dst8);
extern "C" __global__[aicore] void
integer_cast_u32_kernel(__gm__ uint32_t *src, __gm__ uint16_t *dst16,
                        __gm__ uint8_t *dst8);
extern "C" __global__[aicore] void
integer_cast_i32_kernel(__gm__ int32_t *src, __gm__ int16_t *dst16,
                        __gm__ int8_t *dst8);
extern "C" __global__[aicore] void integer_cast_zero_gap_kernel(
    __gm__ uint8_t *src8, __gm__ uint16_t *src16, __gm__ uint16_t *dst16,
    __gm__ uint32_t *dst32_from8, __gm__ uint32_t *dst32_from16);

void LaunchIntegerCastU8(uint8_t *src, uint16_t *dst16, uint32_t *dst32,
                         void *stream) {
  integer_cast_u8_kernel<<<1, nullptr, stream>>>(src, dst16, dst32);
}

void LaunchIntegerCastI8(int8_t *src, int16_t *dst16, int32_t *dst32,
                         void *stream) {
  integer_cast_i8_kernel<<<1, nullptr, stream>>>(src, dst16, dst32);
}

void LaunchIntegerCastU16(uint16_t *src, uint32_t *dst32, uint8_t *dst8,
                          void *stream) {
  integer_cast_u16_kernel<<<1, nullptr, stream>>>(src, dst32, dst8);
}

void LaunchIntegerCastI16(int16_t *src, int32_t *dst32, int8_t *dst8,
                          void *stream) {
  integer_cast_i16_kernel<<<1, nullptr, stream>>>(src, dst32, dst8);
}

void LaunchIntegerCastU32(uint32_t *src, uint16_t *dst16, uint8_t *dst8,
                          void *stream) {
  integer_cast_u32_kernel<<<1, nullptr, stream>>>(src, dst16, dst8);
}

void LaunchIntegerCastI32(int32_t *src, int16_t *dst16, int8_t *dst8,
                          void *stream) {
  integer_cast_i32_kernel<<<1, nullptr, stream>>>(src, dst16, dst8);
}

void LaunchIntegerCastZeroGap(uint8_t *src8, uint16_t *src16, uint16_t *dst16,
                              uint32_t *dst32_from8,
                              uint32_t *dst32_from16, void *stream) {
  integer_cast_zero_gap_kernel<<<1, nullptr, stream>>>(
      src8, src16, dst16, dst32_from8, dst32_from16);
}
