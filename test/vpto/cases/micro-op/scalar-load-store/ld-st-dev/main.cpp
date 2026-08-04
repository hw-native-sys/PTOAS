// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

#include "test_common.h"
#include "acl/acl.h"
#include <cstdint>
#include <cstdio>
#include <cstdlib>

using namespace PtoTestCommon;

#define ACL_CHECK(expr)                                                         \
  do {                                                                          \
    const aclError _ret = (expr);                                               \
    if (_ret != ACL_SUCCESS) {                                                  \
      std::fprintf(stderr, "[ERROR] %s failed: %d (%s:%d)\n", #expr,          \
                   (int)_ret, __FILE__, __LINE__);                              \
      rc = 1;                                                                   \
      goto cleanup;                                                             \
    }                                                                           \
  } while (0)

void LaunchLd_st_dev_kernel(int32_t *src, int32_t *dst, void *stream);

int main() {
  constexpr size_t elemCount = 8;
  size_t fileSize = elemCount * sizeof(int32_t);
  int32_t *srcHost = nullptr;
  int32_t *dstHost = nullptr;
  int32_t *srcDevice = nullptr;
  int32_t *dstDevice = nullptr;
  int rc = 0;
  bool aclInited = false;
  bool deviceSet = false;
  int deviceId = 0;
  aclrtStream stream = nullptr;

  ACL_CHECK(aclInit(nullptr));
  aclInited = true;
  if (const char *envDevice = std::getenv("ACL_DEVICE_ID"))
    deviceId = std::atoi(envDevice);
  ACL_CHECK(aclrtSetDevice(deviceId));
  deviceSet = true;
  ACL_CHECK(aclrtCreateStream(&stream));
  ACL_CHECK(aclrtMallocHost((void **)(&srcHost), fileSize));
  ACL_CHECK(aclrtMallocHost((void **)(&dstHost), fileSize));
  ACL_CHECK(aclrtMalloc((void **)&srcDevice, fileSize,
                        ACL_MEM_MALLOC_HUGE_FIRST));
  ACL_CHECK(aclrtMalloc((void **)&dstDevice, fileSize,
                        ACL_MEM_MALLOC_HUGE_FIRST));

  ReadFile("./src.bin", fileSize, srcHost, fileSize);
  ReadFile("./dst.bin", fileSize, dstHost, fileSize);
  ACL_CHECK(aclrtMemcpy(srcDevice, fileSize, srcHost, fileSize,
                        ACL_MEMCPY_HOST_TO_DEVICE));
  ACL_CHECK(aclrtMemcpy(dstDevice, fileSize, dstHost, fileSize,
                        ACL_MEMCPY_HOST_TO_DEVICE));
  LaunchLd_st_dev_kernel(srcDevice, dstDevice, stream);
  ACL_CHECK(aclrtSynchronizeStream(stream));
  ACL_CHECK(aclrtMemcpy(dstHost, fileSize, dstDevice, fileSize,
                        ACL_MEMCPY_DEVICE_TO_HOST));
  WriteFile("./dst.bin", dstHost, fileSize);

cleanup:
  aclrtFree(dstDevice);
  aclrtFree(srcDevice);
  aclrtFreeHost(dstHost);
  aclrtFreeHost(srcHost);
  if (stream)
    aclrtDestroyStream(stream);
  if (deviceSet)
    aclrtResetDevice(deviceId);
  if (aclInited)
    aclFinalize();
  return rc;
}
