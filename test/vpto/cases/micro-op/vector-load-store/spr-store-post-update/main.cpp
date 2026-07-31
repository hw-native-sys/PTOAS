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

void LaunchSprStorePostUpdate(uint32_t *input, uint32_t *outputI,
                              uint32_t *outputS, void *stream);

namespace {
constexpr size_t kInputElements = 64;
constexpr size_t kOutputElements = 72;
constexpr size_t kInputBytes = kInputElements * sizeof(uint32_t);
constexpr size_t kOutputBytes = kOutputElements * sizeof(uint32_t);
} // namespace

#define ACL_CHECK(expr)                                                        \
  do {                                                                         \
    const aclError ret = (expr);                                               \
    if (ret != ACL_SUCCESS) {                                                  \
      std::fprintf(stderr, "[ERROR] %s failed: %d (%s:%d)\n", #expr,           \
                   static_cast<int>(ret), __FILE__, __LINE__);                 \
      rc = 1;                                                                  \
      goto cleanup;                                                            \
    }                                                                          \
  } while (0)

int main() {
  uint32_t *inputHost = nullptr;
  uint32_t *outputIHost = nullptr;
  uint32_t *outputSHost = nullptr;
  uint32_t *inputDevice = nullptr;
  uint32_t *outputIDevice = nullptr;
  uint32_t *outputSDevice = nullptr;
  aclrtStream stream = nullptr;
  int rc = 0;
  bool aclInited = false;
  bool deviceSet = false;
  int deviceId = 0;
  size_t inputSize = kInputBytes;
  size_t outputISize = kOutputBytes;
  size_t outputSSize = kOutputBytes;

  ACL_CHECK(aclInit(nullptr));
  aclInited = true;
  if (const char *envDevice = std::getenv("ACL_DEVICE_ID"))
    deviceId = std::atoi(envDevice);
  ACL_CHECK(aclrtSetDevice(deviceId));
  deviceSet = true;
  ACL_CHECK(aclrtCreateStream(&stream));

  ACL_CHECK(
      aclrtMallocHost(reinterpret_cast<void **>(&inputHost), kInputBytes));
  ACL_CHECK(
      aclrtMallocHost(reinterpret_cast<void **>(&outputIHost), kOutputBytes));
  ACL_CHECK(
      aclrtMallocHost(reinterpret_cast<void **>(&outputSHost), kOutputBytes));
  ACL_CHECK(aclrtMalloc(reinterpret_cast<void **>(&inputDevice), kInputBytes,
                        ACL_MEM_MALLOC_HUGE_FIRST));
  ACL_CHECK(aclrtMalloc(reinterpret_cast<void **>(&outputIDevice), kOutputBytes,
                        ACL_MEM_MALLOC_HUGE_FIRST));
  ACL_CHECK(aclrtMalloc(reinterpret_cast<void **>(&outputSDevice), kOutputBytes,
                        ACL_MEM_MALLOC_HUGE_FIRST));

  ReadFile("./input.bin", inputSize, inputHost, kInputBytes);
  ReadFile("./output_i.bin", outputISize, outputIHost, kOutputBytes);
  ReadFile("./output_s.bin", outputSSize, outputSHost, kOutputBytes);
  ACL_CHECK(aclrtMemcpy(inputDevice, kInputBytes, inputHost, kInputBytes,
                        ACL_MEMCPY_HOST_TO_DEVICE));
  ACL_CHECK(aclrtMemcpy(outputIDevice, kOutputBytes, outputIHost, kOutputBytes,
                        ACL_MEMCPY_HOST_TO_DEVICE));
  ACL_CHECK(aclrtMemcpy(outputSDevice, kOutputBytes, outputSHost, kOutputBytes,
                        ACL_MEMCPY_HOST_TO_DEVICE));

  LaunchSprStorePostUpdate(inputDevice, outputIDevice, outputSDevice, stream);
  ACL_CHECK(aclrtSynchronizeStream(stream));
  ACL_CHECK(aclrtMemcpy(outputIHost, kOutputBytes, outputIDevice, kOutputBytes,
                        ACL_MEMCPY_DEVICE_TO_HOST));
  ACL_CHECK(aclrtMemcpy(outputSHost, kOutputBytes, outputSDevice, kOutputBytes,
                        ACL_MEMCPY_DEVICE_TO_HOST));
  WriteFile("./output_i.bin", outputIHost, kOutputBytes);
  WriteFile("./output_s.bin", outputSHost, kOutputBytes);

cleanup:
  aclrtFree(inputDevice);
  aclrtFree(outputIDevice);
  aclrtFree(outputSDevice);
  aclrtFreeHost(inputHost);
  aclrtFreeHost(outputIHost);
  aclrtFreeHost(outputSHost);
  if (stream != nullptr)
    aclrtDestroyStream(stream);
  if (deviceSet)
    aclrtResetDevice(deviceId);
  if (aclInited)
    aclFinalize();
  return rc;
}
