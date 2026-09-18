// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software; you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OR
// CONDITIONS OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the license.
#include "acl/acl.h"
#include "test_common.h"

#include <array>
#include <cstdio>
#include <cstdlib>

using namespace PtoTestCommon;

namespace {
constexpr size_t kElements = 256;
constexpr size_t kInputCount = 5;
constexpr size_t kOutputElements = 3 * kElements;
constexpr size_t kInputBytes = kElements * sizeof(float);
constexpr size_t kOutputBytes = kOutputElements * sizeof(float);
} // namespace

void LaunchSimt_fp32_precision_kernel(float *a, float *b, float *c,
                                      float *lhs, float *rhs, float *out,
                                      void *stream);

int main() {
  const std::array<const char *, kInputCount> names = {
      "a.bin", "b.bin", "c.bin", "lhs.bin", "rhs.bin"};
  std::array<float *, kInputCount> host = {};
  std::array<float *, kInputCount> device = {};
  float *outHost = nullptr;
  float *outDevice = nullptr;
  aclrtStream stream = nullptr;
  int deviceId = 0;
  int rc = 1;
  bool outputAllocFailed = false;
  bool executionFailed = false;

  if (const char *envDevice = std::getenv("ACL_DEVICE_ID")) {
    deviceId = std::atoi(envDevice);
  }
  const bool runtimeInitFailed = aclInit(nullptr) != ACL_SUCCESS ||
                                 aclrtSetDevice(deviceId) != ACL_SUCCESS ||
                                 aclrtCreateStream(&stream) != ACL_SUCCESS;
  if (runtimeInitFailed) {
    std::fprintf(stderr, "[ERROR] failed to initialize ACL runtime\n");
    goto cleanup;
  }
  for (size_t i = 0; i < kInputCount; ++i) {
    const bool inputAllocFailed =
        aclrtMallocHost(reinterpret_cast<void **>(&host[i]), kInputBytes) !=
            ACL_SUCCESS ||
        aclrtMalloc(reinterpret_cast<void **>(&device[i]), kInputBytes,
                    ACL_MEM_MALLOC_HUGE_FIRST) != ACL_SUCCESS;
    if (inputAllocFailed) {
      std::fprintf(stderr, "[ERROR] failed to allocate input %s\n", names[i]);
      goto cleanup;
    }
    size_t inputSize = 0;
    const bool inputRead = ReadFile(names[i], inputSize, host[i], kInputBytes);
    if (!inputRead || inputSize != kInputBytes) {
      std::fprintf(stderr, "[ERROR] failed to read input %s\n", names[i]);
      goto cleanup;
    }
    if (aclrtMemcpy(device[i], kInputBytes, host[i], kInputBytes,
                    ACL_MEMCPY_HOST_TO_DEVICE) != ACL_SUCCESS) {
      goto cleanup;
    }
  }
  outputAllocFailed =
      aclrtMallocHost(reinterpret_cast<void **>(&outHost), kOutputBytes) !=
          ACL_SUCCESS ||
      aclrtMalloc(reinterpret_cast<void **>(&outDevice), kOutputBytes,
                  ACL_MEM_MALLOC_HUGE_FIRST) != ACL_SUCCESS;
  if (outputAllocFailed) {
    goto cleanup;
  }
  LaunchSimt_fp32_precision_kernel(device[0], device[1], device[2], device[3],
                                   device[4], outDevice, stream);
  executionFailed = aclrtSynchronizeStream(stream) != ACL_SUCCESS ||
                               aclrtMemcpy(outHost, kOutputBytes, outDevice,
                                           kOutputBytes,
                                           ACL_MEMCPY_DEVICE_TO_HOST) !=
                                   ACL_SUCCESS;
  if (executionFailed) {
    goto cleanup;
  }
  if (!WriteFile("out.bin", outHost, kOutputBytes)) {
    std::fprintf(stderr, "[ERROR] failed to write out.bin\n");
    goto cleanup;
  }
  rc = 0;

cleanup:
  if (outDevice != nullptr) {
    aclrtFree(outDevice);
  }
  if (outHost != nullptr) {
    aclrtFreeHost(outHost);
  }
  for (size_t i = 0; i < kInputCount; ++i) {
    if (device[i] != nullptr) {
      aclrtFree(device[i]);
    }
    if (host[i] != nullptr) {
      aclrtFreeHost(host[i]);
    }
  }
  if (stream != nullptr) {
    aclrtDestroyStream(stream);
  }
  aclrtResetDevice(deviceId);
  aclFinalize();
  return rc;
}
