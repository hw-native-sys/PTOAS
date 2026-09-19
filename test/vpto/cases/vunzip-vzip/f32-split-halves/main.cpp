// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

// Host driver for the f32 split-halves case.
//
// v1.bin holds the 256 f32 inputs, v2.bin the extracted low halves and v3.bin
// the extracted high halves.

#include "acl/acl.h"
#include "test_common.h"

#include <cstddef>
#include <cstdint>
#include <cstdio>

using namespace PtoTestCommon;

void LaunchVunzipF32SplitHalves(float *, uint16_t *, uint16_t *, void *);

namespace {

constexpr size_t kElems = 256;
constexpr size_t kInBytes = kElems * sizeof(float);
constexpr size_t kOutBytes = kElems * sizeof(uint16_t);
constexpr size_t kBuffers = 3;

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
  size_t bytes = 0;
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

size_t BufferBytes(size_t index) {
  return index == 0 ? kInBytes : kOutBytes;
}

bool Allocate(HostBuffer *host, DeviceBuffer *device) {
  for (size_t i = 0; i < kBuffers; ++i) {
    const size_t bytes = BufferBytes(i);
    host[i].bytes = bytes;
    if (!Check(aclrtMallocHost(&host[i].data, bytes), "aclrtMallocHost")) {
      return false;
    }
    if (!Check(aclrtMalloc(&device[i].data, bytes, ACL_MEM_MALLOC_HUGE_FIRST),
               "aclrtMalloc")) {
      return false;
    }
  }
  return true;
}

bool LoadInputs(HostBuffer *host, DeviceBuffer *device) {
  for (size_t i = 0; i < kBuffers; ++i) {
    char path[16];
    std::snprintf(path, sizeof(path), "./v%zu.bin", i + 1);
    size_t fileSize = host[i].bytes;
    if (!ReadFile(path, fileSize, host[i].data, host[i].bytes)) {
      std::fprintf(stderr, "[ERROR] read %s failed\n", path);
      return false;
    }
    if (!Check(aclrtMemcpy(device[i].data, host[i].bytes, host[i].data,
                           host[i].bytes, ACL_MEMCPY_HOST_TO_DEVICE),
               "H2D")) {
      return false;
    }
  }
  return true;
}

bool StoreOutputs(HostBuffer *host, DeviceBuffer *device) {
  for (size_t i = 1; i < kBuffers; ++i) {
    if (!Check(aclrtMemcpy(host[i].data, host[i].bytes, device[i].data,
                           host[i].bytes, ACL_MEMCPY_DEVICE_TO_HOST),
               "D2H")) {
      return false;
    }
    char path[16];
    std::snprintf(path, sizeof(path), "./v%zu.bin", i + 1);
    if (!WriteFile(path, host[i].data, host[i].bytes)) {
      std::fprintf(stderr, "[ERROR] write %s failed\n", path);
      return false;
    }
  }
  return true;
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

  HostBuffer host[kBuffers];
  DeviceBuffer device[kBuffers];
  if (!Allocate(host, device)) {
    return 1;
  }
  if (!LoadInputs(host, device)) {
    return 1;
  }

  LaunchVunzipF32SplitHalves(static_cast<float *>(device[0].data),
                             static_cast<uint16_t *>(device[1].data),
                             static_cast<uint16_t *>(device[2].data),
                             state.stream);
  if (!Check(aclrtSynchronizeStream(state.stream), "aclrtSynchronizeStream")) {
    return 1;
  }
  if (!StoreOutputs(host, device)) {
    return 1;
  }
  std::printf("[INFO] f32 vunzip split-halves finished\n");
  return 0;
}
