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

#define ACL_CHECK(expr)                                                        \
  do {                                                                         \
    const aclError _ret = (expr);                                              \
    if (_ret != ACL_SUCCESS) {                                                 \
      std::fprintf(stderr, "[ERROR] %s failed: %d (%s:%d)\n", #expr,           \
                   (int)_ret, __FILE__, __LINE__);                             \
      const char *_recent = aclGetRecentErrMsg();                              \
      if (_recent != nullptr && _recent[0] != '\0')                            \
        std::fprintf(stderr, "[ERROR] RecentErrMsg: %s\n", _recent);           \
      rc = 1;                                                                  \
      goto cleanup;                                                            \
    }                                                                          \
  } while (0)

void LaunchVectorAddressValdVast(float *src, float *dst, void *stream);

int main() {
  constexpr size_t kElements = 64;
  constexpr size_t kBytes = kElements * sizeof(float);

  float *srcHost = nullptr;
  float *dstHost = nullptr;
  float *srcDevice = nullptr;
  float *dstDevice = nullptr;

  int rc = 0;
  bool aclInited = false;
  bool deviceSet = false;
  int deviceId = 0;
  aclrtStream stream = nullptr;
  size_t srcFileSize = kBytes;
  size_t dstFileSize = kBytes;

  ACL_CHECK(aclInit(nullptr));
  aclInited = true;
  if (const char *envDevice = std::getenv("ACL_DEVICE_ID"))
    deviceId = std::atoi(envDevice);
  ACL_CHECK(aclrtSetDevice(deviceId));
  deviceSet = true;
  ACL_CHECK(aclrtCreateStream(&stream));

  ACL_CHECK(aclrtMallocHost((void **)(&srcHost), kBytes));
  ACL_CHECK(aclrtMallocHost((void **)(&dstHost), kBytes));
  ACL_CHECK(aclrtMalloc((void **)&srcDevice, kBytes, ACL_MEM_MALLOC_HUGE_FIRST));
  ACL_CHECK(aclrtMalloc((void **)&dstDevice, kBytes, ACL_MEM_MALLOC_HUGE_FIRST));

  if (!ReadFile("./v1.bin", srcFileSize, srcHost, kBytes) ||
      srcFileSize != kBytes) {
    std::fprintf(stderr, "[ERROR] failed to read v1.bin\n");
    rc = 1;
    goto cleanup;
  }
  if (!ReadFile("./v2.bin", dstFileSize, dstHost, kBytes) ||
      dstFileSize != kBytes) {
    std::fprintf(stderr, "[ERROR] failed to read v2.bin\n");
    rc = 1;
    goto cleanup;
  }
  ACL_CHECK(aclrtMemcpy(srcDevice, kBytes, srcHost, kBytes,
                        ACL_MEMCPY_HOST_TO_DEVICE));
  ACL_CHECK(aclrtMemcpy(dstDevice, kBytes, dstHost, kBytes,
                        ACL_MEMCPY_HOST_TO_DEVICE));

  LaunchVectorAddressValdVast(srcDevice, dstDevice, stream);

  ACL_CHECK(aclrtSynchronizeStream(stream));
  ACL_CHECK(aclrtMemcpy(dstHost, kBytes, dstDevice, kBytes,
                        ACL_MEMCPY_DEVICE_TO_HOST));
  WriteFile("./v2.bin", dstHost, kBytes);

cleanup:
  aclrtFree(srcDevice);
  aclrtFree(dstDevice);
  aclrtFreeHost(srcHost);
  aclrtFreeHost(dstHost);
  if (stream != nullptr)
    aclrtDestroyStream(stream);
  if (deviceSet)
    aclrtResetDevice(deviceId);
  if (aclInited)
    aclFinalize();
  return rc;
}
