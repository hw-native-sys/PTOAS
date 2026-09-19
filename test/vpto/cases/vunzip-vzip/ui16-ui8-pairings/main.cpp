// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

// Host driver for the ui16 -> ui8 vunzip/vzip pairing case.
//
// The 384 ui16 input words are split into a 256-lane region (2:1 pairing) and
// a 128-lane region (1:1 half-full pairing).  Each region is a bit-exact round
// trip, so the expected output is the input words themselves.

#include "acl/acl.h"
#include "test_common.h"

#include <cstddef>
#include <cstdint>
#include <cstdio>

using namespace PtoTestCommon;

void LaunchVunzipVzipUi16Ui8Pair256(uint16_t *, uint16_t *, void *);
void LaunchVunzipVzipUi16Ui8Pair128(uint16_t *, uint16_t *, void *);

namespace {

constexpr size_t kTotalElems = 384;
constexpr size_t kBytes = kTotalElems * sizeof(uint16_t);
constexpr size_t kPair128Offset = 256;

struct AclState {
  aclrtStream stream = nullptr;
  bool inited = false;
  bool deviceSet = false;
  int deviceId = 0;
  ~AclState() {
    if (stream != nullptr) {
      aclrtDestroyStream(stream);
    }
    if (deviceSet) {
      aclrtResetDevice(deviceId);
    }
    if (inited) {
      aclFinalize();
    }
  }
};

struct HostBuffer {
  void *data = nullptr;
  ~HostBuffer() {
    if (data != nullptr) {
      aclrtFreeHost(data);
    }
  }
};

struct DeviceBuffer {
  void *data = nullptr;
  ~DeviceBuffer() {
    if (data != nullptr) {
      aclrtFree(data);
    }
  }
};

bool Check(aclError status, const char *step) {
  if (status == ACL_SUCCESS) {
    return true;
  }
  std::fprintf(stderr, "[ERROR] %s failed: %d\n", step, static_cast<int>(status));
  return false;
}

bool Prepare(HostBuffer &srcHost, HostBuffer &dstHost,
             DeviceBuffer &srcDevice, DeviceBuffer &dstDevice) {
  if (!Check(aclrtMallocHost(&srcHost.data, kBytes), "aclrtMallocHost(src)")) {
    return false;
  }
  if (!Check(aclrtMallocHost(&dstHost.data, kBytes), "aclrtMallocHost(dst)")) {
    return false;
  }
  if (!Check(aclrtMalloc(&srcDevice.data, kBytes, ACL_MEM_MALLOC_HUGE_FIRST),
             "aclrtMalloc(src)")) {
    return false;
  }
  if (!Check(aclrtMalloc(&dstDevice.data, kBytes, ACL_MEM_MALLOC_HUGE_FIRST),
             "aclrtMalloc(dst)")) {
    return false;
  }
  size_t fileSize = kBytes;
  if (!ReadFile("./v1.bin", fileSize, srcHost.data, kBytes)) {
    std::fprintf(stderr, "[ERROR] read v1.bin failed\n");
    return false;
  }
  fileSize = kBytes;
  if (!ReadFile("./v2.bin", fileSize, dstHost.data, kBytes)) {
    std::fprintf(stderr, "[ERROR] read v2.bin failed\n");
    return false;
  }
  if (!Check(aclrtMemcpy(srcDevice.data, kBytes, srcHost.data, kBytes,
                         ACL_MEMCPY_HOST_TO_DEVICE),
             "H2D(src)")) {
    return false;
  }
  return Check(aclrtMemcpy(dstDevice.data, kBytes, dstHost.data, kBytes,
                           ACL_MEMCPY_HOST_TO_DEVICE),
               "H2D(dst)");
}

}  // namespace

int main() {
  AclState state;
  if (!Check(aclInit(nullptr), "aclInit")) {
    return 1;
  }
  state.inited = true;
  if (!Check(aclrtSetDevice(state.deviceId), "aclrtSetDevice")) {
    return 1;
  }
  state.deviceSet = true;
  if (!Check(aclrtCreateStream(&state.stream), "aclrtCreateStream")) {
    return 1;
  }

  HostBuffer srcHost;
  HostBuffer dstHost;
  DeviceBuffer srcDevice;
  DeviceBuffer dstDevice;
  if (!Prepare(srcHost, dstHost, srcDevice, dstDevice)) {
    return 1;
  }

  uint16_t *src = static_cast<uint16_t *>(srcDevice.data);
  uint16_t *dst = static_cast<uint16_t *>(dstDevice.data);
  LaunchVunzipVzipUi16Ui8Pair256(src, dst, state.stream);
  LaunchVunzipVzipUi16Ui8Pair128(src + kPair128Offset, dst + kPair128Offset,
                                 state.stream);
  if (!Check(aclrtSynchronizeStream(state.stream), "aclrtSynchronizeStream")) {
    return 1;
  }
  if (!Check(aclrtMemcpy(dstHost.data, kBytes, dstDevice.data, kBytes,
                         ACL_MEMCPY_DEVICE_TO_HOST),
             "D2H(dst)")) {
    return 1;
  }
  if (!WriteFile("./v2.bin", dstHost.data, kBytes)) {
    std::fprintf(stderr, "[ERROR] write v2.bin failed\n");
    return 1;
  }
  std::printf("[INFO] ui16 -> ui8 vunzip/vzip round trip finished\n");
  return 0;
}
