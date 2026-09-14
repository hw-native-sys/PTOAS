// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

// -----------------------------------------------------------------------------
// case: micro-op/vec-scalar/vsubcs
// family: vec-scalar
// target_ops: pto.vsubcs
// scenarios: core-u32-unsigned, full-mask, borrow-boundary
// -----------------------------------------------------------------------------
/**
Copyright (c) 2025 Huawei Technologies Co., Ltd.
*/

#include "test_common.h"
#include "acl/acl.h"
#include <cstdio>
#include <cstdlib>

using namespace PtoTestCommon;

#define ACL_CHECK(expr)                                                          \
  do {                                                                           \
    const aclError _ret = (expr);                                                \
    if (_ret != ACL_SUCCESS) {                                                   \
      std::fprintf(stderr, "[ERROR] %s failed: %d (%s:%d)\n", #expr,             \
                   (int)_ret, __FILE__, __LINE__);                               \
      rc = 1;                                                                    \
      goto cleanup;                                                              \
    }                                                                            \
  } while (0)

void LaunchVsubcs_kernel(uint32_t *v1, uint32_t *v2, uint32_t *v3, uint8_t *v4,
                         void *stream);

int main() {
  size_t elemCount_v1 = 64;
  size_t fileSize_v1 = elemCount_v1 * sizeof(uint32_t);
  size_t elemCount_v2 = 64;
  size_t fileSize_v2 = elemCount_v2 * sizeof(uint32_t);
  size_t elemCount_v3 = 64;
  size_t fileSize_v3 = elemCount_v3 * sizeof(uint32_t);
  size_t elemCount_v4 = 256;
  size_t fileSize_v4 = elemCount_v4 * sizeof(uint8_t);
  uint32_t *v1Host = nullptr;
  uint32_t *v1Device = nullptr;
  uint32_t *v2Host = nullptr;
  uint32_t *v2Device = nullptr;
  uint32_t *v3Host = nullptr;
  uint32_t *v3Device = nullptr;
  uint8_t *v4Host = nullptr;
  uint8_t *v4Device = nullptr;
  int rc = 0;
  bool aclInited = false;
  bool deviceSet = false;
  int deviceId = 0;
  aclrtStream stream = nullptr;

  ACL_CHECK(aclInit(nullptr));
  aclInited = true;
  if (const char *envDevice = std::getenv("ACL_DEVICE_ID")) {
    deviceId = std::atoi(envDevice);
  }
  ACL_CHECK(aclrtSetDevice(deviceId));
  deviceSet = true;
  ACL_CHECK(aclrtCreateStream(&stream));
  ACL_CHECK(aclrtMallocHost((void **)(&v1Host), fileSize_v1));
  ACL_CHECK(aclrtMallocHost((void **)(&v2Host), fileSize_v2));
  ACL_CHECK(aclrtMallocHost((void **)(&v3Host), fileSize_v3));
  ACL_CHECK(aclrtMallocHost((void **)(&v4Host), fileSize_v4));
  ACL_CHECK(aclrtMalloc((void **)&v1Device, fileSize_v1, ACL_MEM_MALLOC_HUGE_FIRST));
  ACL_CHECK(aclrtMalloc((void **)&v2Device, fileSize_v2, ACL_MEM_MALLOC_HUGE_FIRST));
  ACL_CHECK(aclrtMalloc((void **)&v3Device, fileSize_v3, ACL_MEM_MALLOC_HUGE_FIRST));
  ACL_CHECK(aclrtMalloc((void **)&v4Device, fileSize_v4, ACL_MEM_MALLOC_HUGE_FIRST));
  ReadFile("./v1.bin", fileSize_v1, v1Host, fileSize_v1);
  ReadFile("./v2.bin", fileSize_v2, v2Host, fileSize_v2);
  ReadFile("./v3.bin", fileSize_v3, v3Host, fileSize_v3);
  ReadFile("./v4.bin", fileSize_v4, v4Host, fileSize_v4);
  ACL_CHECK(aclrtMemcpy(v1Device, fileSize_v1, v1Host, fileSize_v1,
                        ACL_MEMCPY_HOST_TO_DEVICE));
  ACL_CHECK(aclrtMemcpy(v2Device, fileSize_v2, v2Host, fileSize_v2,
                        ACL_MEMCPY_HOST_TO_DEVICE));
  ACL_CHECK(aclrtMemcpy(v3Device, fileSize_v3, v3Host, fileSize_v3,
                        ACL_MEMCPY_HOST_TO_DEVICE));
  ACL_CHECK(aclrtMemcpy(v4Device, fileSize_v4, v4Host, fileSize_v4,
                        ACL_MEMCPY_HOST_TO_DEVICE));
  LaunchVsubcs_kernel(v1Device, v2Device, v3Device, v4Device, stream);
  ACL_CHECK(aclrtSynchronizeStream(stream));
  ACL_CHECK(aclrtMemcpy(v3Host, fileSize_v3, v3Device, fileSize_v3,
                        ACL_MEMCPY_DEVICE_TO_HOST));
  ACL_CHECK(aclrtMemcpy(v4Host, fileSize_v4, v4Device, fileSize_v4,
                        ACL_MEMCPY_DEVICE_TO_HOST));
  WriteFile("./v3.bin", v3Host, fileSize_v3);
  WriteFile("./v4.bin", v4Host, fileSize_v4);

cleanup:
  aclrtFree(v1Device);
  aclrtFree(v2Device);
  aclrtFree(v3Device);
  aclrtFree(v4Device);
  aclrtFreeHost(v1Host);
  aclrtFreeHost(v2Host);
  aclrtFreeHost(v3Host);
  aclrtFreeHost(v4Host);
  if (stream != nullptr) {
    aclrtDestroyStream(stream);
  }
  if (deviceSet) {
    aclrtResetDevice(deviceId);
  }
  if (aclInited) {
    aclFinalize();
  }
  return rc;
}
