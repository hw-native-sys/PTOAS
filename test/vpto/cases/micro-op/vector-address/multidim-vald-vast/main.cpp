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

void LaunchVectorAddressMultidimValdVast(uint8_t *src, uint8_t *dst,
                                         void *stream);

int main() {
  constexpr size_t kSrcBytes = 4096;
  constexpr size_t kDstBytes = 8192;

  uint8_t *srcHost = nullptr;
  uint8_t *dstHost = nullptr;
  uint8_t *srcDevice = nullptr;
  uint8_t *dstDevice = nullptr;

  int rc = 0;
  bool aclInited = false;
  bool deviceSet = false;
  int deviceId = 0;
  aclrtStream stream = nullptr;
  size_t srcFileSize = kSrcBytes;
  size_t dstFileSize = kDstBytes;

  ACL_CHECK(aclInit(nullptr));
  aclInited = true;
  if (const char *envDevice = std::getenv("ACL_DEVICE_ID"))
    deviceId = std::atoi(envDevice);
  ACL_CHECK(aclrtSetDevice(deviceId));
  deviceSet = true;
  ACL_CHECK(aclrtCreateStream(&stream));

  ACL_CHECK(aclrtMallocHost((void **)(&srcHost), kSrcBytes));
  ACL_CHECK(aclrtMallocHost((void **)(&dstHost), kDstBytes));
  ACL_CHECK(aclrtMalloc((void **)&srcDevice, kSrcBytes, ACL_MEM_MALLOC_HUGE_FIRST));
  ACL_CHECK(aclrtMalloc((void **)&dstDevice, kDstBytes, ACL_MEM_MALLOC_HUGE_FIRST));

  if (!ReadFile("./v1.bin", srcFileSize, srcHost, kSrcBytes) ||
      srcFileSize != kSrcBytes) {
    std::fprintf(stderr, "[ERROR] failed to read v1.bin\n");
    rc = 1;
    goto cleanup;
  }
  if (!ReadFile("./v2.bin", dstFileSize, dstHost, kDstBytes) ||
      dstFileSize != kDstBytes) {
    std::fprintf(stderr, "[ERROR] failed to read v2.bin\n");
    rc = 1;
    goto cleanup;
  }
  ACL_CHECK(aclrtMemcpy(srcDevice, kSrcBytes, srcHost, kSrcBytes,
                        ACL_MEMCPY_HOST_TO_DEVICE));
  ACL_CHECK(aclrtMemcpy(dstDevice, kDstBytes, dstHost, kDstBytes,
                        ACL_MEMCPY_HOST_TO_DEVICE));

  LaunchVectorAddressMultidimValdVast(srcDevice, dstDevice, stream);

  ACL_CHECK(aclrtSynchronizeStream(stream));
  ACL_CHECK(aclrtMemcpy(dstHost, kDstBytes, dstDevice, kDstBytes,
                        ACL_MEMCPY_DEVICE_TO_HOST));
  WriteFile("./v2.bin", dstHost, kDstBytes);

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
