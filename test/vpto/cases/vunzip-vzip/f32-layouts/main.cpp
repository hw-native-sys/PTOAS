// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

// Host driver for the vunzip/vzip f32 layout case.
//
// The 768 f32 input buffer is split into three disjoint regions, one per
// layout, and each region is a bit-exact vunzip -> vzip round trip.  The
// expected output is therefore the input buffer itself, which golden.py writes
// to golden_v2.bin without ever consulting the device result.

#include "acl/acl.h"
#include "test_common.h"

#include <cstddef>
#include <cstdint>
#include <cstdio>

using namespace PtoTestCommon;

void LaunchVunzipVzipF32Contiguous(float *, float *, void *);
void LaunchVunzipVzipF32Deinterleaved2(float *, float *, void *);
void LaunchVunzipVzipF32Deinterleaved4(float *, float *, void *);

namespace {

constexpr size_t kTotalElems = 768;
constexpr size_t kBytes = kTotalElems * sizeof(float);
constexpr size_t kDeinterleaved2Offset = 256;
constexpr size_t kDeinterleaved4Offset = 512;

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

bool ReadInputs(const HostBuffer &srcHost, const HostBuffer &dstHost) {
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
  return true;
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
  if (!ReadInputs(srcHost, dstHost)) {
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

void LaunchAll(const AclState &state, const DeviceBuffer &srcDevice,
               const DeviceBuffer &dstDevice) {
  float *src = static_cast<float *>(srcDevice.data);
  float *dst = static_cast<float *>(dstDevice.data);
  LaunchVunzipVzipF32Contiguous(src, dst, state.stream);
  LaunchVunzipVzipF32Deinterleaved2(src + kDeinterleaved2Offset,
                                    dst + kDeinterleaved2Offset, state.stream);
  LaunchVunzipVzipF32Deinterleaved4(src + kDeinterleaved4Offset,
                                    dst + kDeinterleaved4Offset, state.stream);
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

  LaunchAll(state, srcDevice, dstDevice);
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
  std::printf("[INFO] f32 vunzip/vzip layout round trip finished\n");
  return 0;
}
