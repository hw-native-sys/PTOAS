// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

// The CCE launch wrapper reads and releases the DebugTunnel payload without
// waiting for the asynchronous kernel, so print records can be dropped.
// Object emission compiles this file and merges it into the host object of a
// printing kernel, overriding __DebugTunnel_Close with one that synchronizes
// the stream first.
#include "acl/acl.h"

namespace cce::internal {
struct DebugTunnelData;
struct DebugTunnel {
  static void Close(DebugTunnelData *data, void *stream);
};
} // namespace cce::internal

extern "C" __attribute__((used)) inline void closeDebugTunnel(cce::internal::DebugTunnelData* data, void* stream)
    asm("__DebugTunnel_Close");

extern "C" __attribute__((used)) inline void closeDebugTunnel(cce::internal::DebugTunnelData* data, void* stream)
{
    (void)aclrtSynchronizeStream(static_cast<aclrtStream>(stream));
    cce::internal::DebugTunnel::Close(data, stream);
}
