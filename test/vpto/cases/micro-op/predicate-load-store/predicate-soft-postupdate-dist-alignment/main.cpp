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

void LaunchPredicateSoftPostupdateDistAlignment(uint8_t *input,
                                                uint8_t *output,
                                                void *stream);

namespace {
constexpr size_t kBufferBytes = 2048;
} // namespace

#define ACL_CHECK(expr)                                                        \
  do {                                                                         \
    const aclError ret = (expr);                                               \
    if (ret != ACL_SUCCESS) {                                                  \
      std::fprintf(stderr, "[ERROR] %s failed: %d (%s:%d)\n", #expr,          \
                   static_cast<int>(ret), __FILE__, __LINE__);                 \
      rc = 1;                                                                  \
      goto cleanup;                                                            \
    }                                                                          \
  } while (0)

int main() {
  uint8_t *inputHost = nullptr;
  uint8_t *outputHost = nullptr;
  uint8_t *inputDevice = nullptr;
  uint8_t *outputDevice = nullptr;
  aclrtStream stream = nullptr;
  int rc = 0;
  bool aclInited = false;
  bool deviceSet = false;
  int deviceId = 0;
  size_t inputSize = kBufferBytes;
  size_t outputSize = kBufferBytes;

  ACL_CHECK(aclInit(nullptr));
  aclInited = true;
  if (const char *envDevice = std::getenv("ACL_DEVICE_ID"))
    deviceId = std::atoi(envDevice);
  ACL_CHECK(aclrtSetDevice(deviceId));
  deviceSet = true;
  ACL_CHECK(aclrtCreateStream(&stream));

  ACL_CHECK(
      aclrtMallocHost(reinterpret_cast<void **>(&inputHost), kBufferBytes));
  ACL_CHECK(
      aclrtMallocHost(reinterpret_cast<void **>(&outputHost), kBufferBytes));
  ACL_CHECK(aclrtMalloc(reinterpret_cast<void **>(&inputDevice), kBufferBytes,
                        ACL_MEM_MALLOC_HUGE_FIRST));
  ACL_CHECK(aclrtMalloc(reinterpret_cast<void **>(&outputDevice), kBufferBytes,
                        ACL_MEM_MALLOC_HUGE_FIRST));

  ReadFile("./input.bin", inputSize, inputHost, kBufferBytes);
  ReadFile("./output.bin", outputSize, outputHost, kBufferBytes);
  ACL_CHECK(aclrtMemcpy(inputDevice, kBufferBytes, inputHost, kBufferBytes,
                        ACL_MEMCPY_HOST_TO_DEVICE));
  ACL_CHECK(aclrtMemcpy(outputDevice, kBufferBytes, outputHost, kBufferBytes,
                        ACL_MEMCPY_HOST_TO_DEVICE));

  LaunchPredicateSoftPostupdateDistAlignment(inputDevice, outputDevice, stream);
  ACL_CHECK(aclrtSynchronizeStream(stream));
  ACL_CHECK(aclrtMemcpy(outputHost, kBufferBytes, outputDevice, kBufferBytes,
                        ACL_MEMCPY_DEVICE_TO_HOST));
  WriteFile("./output.bin", outputHost, kBufferBytes);

cleanup:
  aclrtFree(inputDevice);
  aclrtFree(outputDevice);
  aclrtFreeHost(inputHost);
  aclrtFreeHost(outputHost);
  if (stream != nullptr)
    aclrtDestroyStream(stream);
  if (deviceSet)
    aclrtResetDevice(deviceId);
  if (aclInited)
    aclFinalize();
  return rc;
}
