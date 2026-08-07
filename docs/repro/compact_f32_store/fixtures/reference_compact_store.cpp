// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

#include "reference_compact_store.asc"

#ifndef AICORE
#define AICORE [aicore]
#endif

extern "C" void launch_reference_compact_f32_store(
    void *stream,
    void *input,
    void *output) {
  reference_compact_f32_store<<<1, nullptr, stream>>>(
      (__gm__ float *)input,
      (__gm__ float *)output);
}
