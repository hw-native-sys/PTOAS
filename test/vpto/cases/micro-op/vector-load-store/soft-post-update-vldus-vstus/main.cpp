// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

#include "acl/acl.h"
#include "test_common.h"

#include <cstdio>
#include <cstdlib>

using namespace PtoTestCommon;

void LaunchSoftPostUpdateVldusVstus(float *input, float *initial,
                                    float *explicitOutput,
                                    float *rewrittenOutput, void *stream);

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
  float *initialHost = nullptr;
  float *explicitHost = nullptr;
  float *rewrittenHost = nullptr;
  float *inputDevice = nullptr;
  float *initialDevice = nullptr;
  float *explicitDevice = nullptr;
  float *rewrittenDevice = nullptr;
  aclrtStream stream = nullptr;
  int rc = 0;
  bool aclInited = false;
  bool deviceSet = false;
  int deviceId = 0;
  size_t inputSize = kBufferSize;
  size_t initialSize = kBufferSize;
  size_t explicitSize = kBufferSize;
  size_t rewrittenSize = kBufferSize;

  ACL_CHECK(aclInit(nullptr));
  aclInited = true;
  if (const char *envDevice = std::getenv("ACL_DEVICE_ID"))
    deviceId = std::atoi(envDevice);
  ACL_CHECK(aclrtSetDevice(deviceId));
  deviceSet = true;
  ACL_CHECK(aclrtCreateStream(&stream));

  ACL_CHECK(aclrtMallocHost(reinterpret_cast<void **>(&inputHost), kBufferSize));
  ACL_CHECK(
      aclrtMallocHost(reinterpret_cast<void **>(&initialHost), kBufferSize));
  ACL_CHECK(
      aclrtMallocHost(reinterpret_cast<void **>(&explicitHost), kBufferSize));
  ACL_CHECK(
      aclrtMallocHost(reinterpret_cast<void **>(&rewrittenHost), kBufferSize));
  ACL_CHECK(aclrtMalloc(reinterpret_cast<void **>(&inputDevice), kBufferSize,
                        ACL_MEM_MALLOC_HUGE_FIRST));
  ACL_CHECK(aclrtMalloc(reinterpret_cast<void **>(&initialDevice), kBufferSize,
                        ACL_MEM_MALLOC_HUGE_FIRST));
  ACL_CHECK(aclrtMalloc(reinterpret_cast<void **>(&explicitDevice), kBufferSize,
                        ACL_MEM_MALLOC_HUGE_FIRST));
  ACL_CHECK(aclrtMalloc(reinterpret_cast<void **>(&rewrittenDevice), kBufferSize,
                        ACL_MEM_MALLOC_HUGE_FIRST));

  ReadFile("./input.bin", inputSize, inputHost, kBufferSize);
  ReadFile("./initial.bin", initialSize, initialHost, kBufferSize);
  ReadFile("./explicit_output.bin", explicitSize, explicitHost, kBufferSize);
  ReadFile("./rewritten_output.bin", rewrittenSize, rewrittenHost,
           kBufferSize);
  ACL_CHECK(aclrtMemcpy(inputDevice, kBufferSize, inputHost, kBufferSize,
                        ACL_MEMCPY_HOST_TO_DEVICE));
  ACL_CHECK(aclrtMemcpy(initialDevice, kBufferSize, initialHost, kBufferSize,
                        ACL_MEMCPY_HOST_TO_DEVICE));
  ACL_CHECK(aclrtMemcpy(explicitDevice, kBufferSize, explicitHost, kBufferSize,
                        ACL_MEMCPY_HOST_TO_DEVICE));
  ACL_CHECK(aclrtMemcpy(rewrittenDevice, kBufferSize, rewrittenHost,
                        kBufferSize, ACL_MEMCPY_HOST_TO_DEVICE));

  LaunchSoftPostUpdateVldusVstus(inputDevice, initialDevice, explicitDevice,
                                  rewrittenDevice, stream);
  ACL_CHECK(aclrtSynchronizeStream(stream));
  ACL_CHECK(aclrtMemcpy(explicitHost, kBufferSize, explicitDevice, kBufferSize,
                        ACL_MEMCPY_DEVICE_TO_HOST));
  ACL_CHECK(aclrtMemcpy(rewrittenHost, kBufferSize, rewrittenDevice,
                        kBufferSize, ACL_MEMCPY_DEVICE_TO_HOST));
  WriteFile("./explicit_output.bin", explicitHost, kBufferSize);
  WriteFile("./rewritten_output.bin", rewrittenHost, kBufferSize);

cleanup:
  aclrtFree(inputDevice);
  aclrtFree(initialDevice);
  aclrtFree(explicitDevice);
  aclrtFree(rewrittenDevice);
  aclrtFreeHost(inputHost);
  aclrtFreeHost(initialHost);
  aclrtFreeHost(explicitHost);
  aclrtFreeHost(rewrittenHost);
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
