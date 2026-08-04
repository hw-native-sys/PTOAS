// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

#include "acl/acl.h"
#include "test_common.h"

#include <cstdint>
#include <cstdio>
#include <cstdlib>

using namespace PtoTestCommon;

void LaunchSoftPostUpdateAddressUnits(float *input, float *output,
                                      uint32_t *sprInput, uint32_t *sprOutput,
                                      void *stream);

namespace {
constexpr size_t kElementCount = 1024;
constexpr size_t kBufferSize = kElementCount * sizeof(float);
}

#define ACL_CHECK(expr)                                                        \
  do {                                                                         \
    const aclError ret = (expr);                                               \
    if (ret != ACL_SUCCESS) {                                                  \
      std::fprintf(stderr, "[ERROR] %s failed: %d (%s:%d)\n", #expr,          \
                   static_cast<int>(ret), __FILE__, __LINE__);                \
      const char *recent = aclGetRecentErrMsg();                               \
      if (recent != nullptr && recent[0] != '\0')                              \
        std::fprintf(stderr, "[ERROR] RecentErrMsg: %s\n", recent);           \
      rc = 1;                                                                  \
      goto cleanup;                                                            \
    }                                                                          \
  } while (0)

int main() {
  float *inputHost = nullptr;
  float *outputHost = nullptr;
  uint32_t *sprInputHost = nullptr;
  uint32_t *sprOutputHost = nullptr;
  float *inputDevice = nullptr;
  float *outputDevice = nullptr;
  uint32_t *sprInputDevice = nullptr;
  uint32_t *sprOutputDevice = nullptr;
  aclrtStream stream = nullptr;
  int rc = 0;
  bool aclInited = false;
  bool deviceSet = false;
  int deviceId = 0;
  size_t inputSize = kBufferSize;
  size_t outputSize = kBufferSize;
  size_t sprInputSize = kBufferSize;
  size_t sprOutputSize = kBufferSize;

  ACL_CHECK(aclInit(nullptr));
  aclInited = true;
  if (const char *envDevice = std::getenv("ACL_DEVICE_ID"))
    deviceId = std::atoi(envDevice);
  ACL_CHECK(aclrtSetDevice(deviceId));
  deviceSet = true;
  ACL_CHECK(aclrtCreateStream(&stream));

  ACL_CHECK(aclrtMallocHost(reinterpret_cast<void **>(&inputHost), kBufferSize));
  ACL_CHECK(aclrtMallocHost(reinterpret_cast<void **>(&outputHost), kBufferSize));
  ACL_CHECK(
      aclrtMallocHost(reinterpret_cast<void **>(&sprInputHost), kBufferSize));
  ACL_CHECK(
      aclrtMallocHost(reinterpret_cast<void **>(&sprOutputHost), kBufferSize));
  ACL_CHECK(aclrtMalloc(reinterpret_cast<void **>(&inputDevice), kBufferSize,
                        ACL_MEM_MALLOC_HUGE_FIRST));
  ACL_CHECK(aclrtMalloc(reinterpret_cast<void **>(&outputDevice), kBufferSize,
                        ACL_MEM_MALLOC_HUGE_FIRST));
  ACL_CHECK(aclrtMalloc(reinterpret_cast<void **>(&sprInputDevice), kBufferSize,
                        ACL_MEM_MALLOC_HUGE_FIRST));
  ACL_CHECK(aclrtMalloc(reinterpret_cast<void **>(&sprOutputDevice),
                        kBufferSize, ACL_MEM_MALLOC_HUGE_FIRST));

  ReadFile("./input.bin", inputSize, inputHost, kBufferSize);
  ReadFile("./output.bin", outputSize, outputHost, kBufferSize);
  ReadFile("./input_spr.bin", sprInputSize, sprInputHost, kBufferSize);
  ReadFile("./output_spr.bin", sprOutputSize, sprOutputHost, kBufferSize);
  ACL_CHECK(aclrtMemcpy(inputDevice, kBufferSize, inputHost, kBufferSize,
                        ACL_MEMCPY_HOST_TO_DEVICE));
  ACL_CHECK(aclrtMemcpy(outputDevice, kBufferSize, outputHost, kBufferSize,
                        ACL_MEMCPY_HOST_TO_DEVICE));
  ACL_CHECK(aclrtMemcpy(sprInputDevice, kBufferSize, sprInputHost, kBufferSize,
                        ACL_MEMCPY_HOST_TO_DEVICE));
  ACL_CHECK(aclrtMemcpy(sprOutputDevice, kBufferSize, sprOutputHost,
                        kBufferSize, ACL_MEMCPY_HOST_TO_DEVICE));

  LaunchSoftPostUpdateAddressUnits(inputDevice, outputDevice, sprInputDevice,
                                   sprOutputDevice, stream);
  ACL_CHECK(aclrtSynchronizeStream(stream));
  ACL_CHECK(aclrtMemcpy(outputHost, kBufferSize, outputDevice, kBufferSize,
                        ACL_MEMCPY_DEVICE_TO_HOST));
  ACL_CHECK(aclrtMemcpy(sprOutputHost, kBufferSize, sprOutputDevice,
                        kBufferSize, ACL_MEMCPY_DEVICE_TO_HOST));
  WriteFile("./output.bin", outputHost, kBufferSize);
  WriteFile("./output_spr.bin", sprOutputHost, kBufferSize);

cleanup:
  aclrtFree(inputDevice);
  aclrtFree(outputDevice);
  aclrtFree(sprInputDevice);
  aclrtFree(sprOutputDevice);
  aclrtFreeHost(inputHost);
  aclrtFreeHost(outputHost);
  aclrtFreeHost(sprInputHost);
  aclrtFreeHost(sprOutputHost);
  if (stream != nullptr) {
    const aclError ret = aclrtDestroyStream(stream);
    if (ret != ACL_SUCCESS)
      std::fprintf(stderr, "[ERROR] aclrtDestroyStream failed: %d\n",
                   static_cast<int>(ret));
  }
  if (deviceSet) {
    const aclError ret = aclrtResetDevice(deviceId);
    if (ret != ACL_SUCCESS)
      std::fprintf(stderr, "[ERROR] aclrtResetDevice failed: %d\n",
                   static_cast<int>(ret));
  }
  if (aclInited) {
    const aclError ret = aclFinalize();
    if (ret != ACL_SUCCESS)
      std::fprintf(stderr, "[ERROR] aclFinalize failed: %d\n",
                   static_cast<int>(ret));
  }
  return rc;
}
