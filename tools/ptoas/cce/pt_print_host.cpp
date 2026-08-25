// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

// Ensure PTOAS-generated VPTO Print host stubs synchronize the asynchronous
// kernel launch before CANN reads and releases the DebugTunnel payload.
#include "acl/acl.h"

namespace cce::internal {
struct DebugTunnelData;
struct DebugTunnel {
  static void Close(DebugTunnelData *data, void *stream);
};
} // namespace cce::internal

extern "C" void __DebugTunnel_Close(
    cce::internal::DebugTunnelData *data, void *stream) {
  (void)aclrtSynchronizeStream(static_cast<aclrtStream>(stream));
  cce::internal::DebugTunnel::Close(data, stream);
}
