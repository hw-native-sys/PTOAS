// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

#include <stdint.h>
extern "C" __global__ [aicore] void indirect_reference(
    __gm__ int64_t *, __gm__ uint16_t *, __gm__ int64_t *, __gm__ int64_t *,
    __gm__ int64_t *, __gm__ int64_t *, __gm__ int64_t *, int32_t);
extern "C" void launch_indirect_reference(void *stream, void *comb, void *initial,
    void *layer_in, void *layer_out, void *post, void *pre, void *residual) {
  indirect_reference<<<72, 196864, stream>>>(
      (__gm__ int64_t *)comb, (__gm__ uint16_t *)initial,
      (__gm__ int64_t *)layer_in, (__gm__ int64_t *)layer_out,
      (__gm__ int64_t *)post, (__gm__ int64_t *)pre,
      (__gm__ int64_t *)residual, 8192);
}
