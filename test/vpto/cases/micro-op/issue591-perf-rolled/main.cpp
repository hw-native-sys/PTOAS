// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

// Issue #591 performance acceptance host runner (rolled form).

#include "test_common.h"
#include "acl/acl.h"
#include <cstdio>
#include <cstdlib>
#include <cstdint>

using namespace PtoTestCommon;

#define ACL_CHECK(expr)                                                          \
  do {                                                                           \
    const aclError _ret = (expr);                                                \
    if (_ret != ACL_SUCCESS) {                                                   \
      std::fprintf(stderr, "[ERROR] %s failed: %d (%s:%d)\n", #expr,            \
                   (int)_ret, __FILE__, __LINE__);                               \
      rc = 1;                                                                    \
      goto cleanup;                                                              \
    }                                                                            \
  } while (0)

#define FILE_CHECK(expr, path)                                                   \
  do {                                                                           \
    if (!(expr)) {                                                               \
      std::fprintf(stderr, "[ERROR] file operation failed: %s (%s:%d)\n",       \
                   path, __FILE__, __LINE__);                                    \
      rc = 1;                                                                    \
      goto cleanup;                                                              \
    }                                                                            \
  } while (0)

void LaunchIssue591PerfRolled(uint16_t *in, uint16_t *out, void *stream);

int main() {
  constexpr size_t ELEMS = 16 * 128;
  constexpr size_t SZ = ELEMS * sizeof(uint16_t);

  uint16_t *h_in = nullptr, *h_out = nullptr;
  uint16_t *d_in = nullptr, *d_out = nullptr;

  int rc = 0;
  bool aclInited = false;
  bool deviceSet = false;
  int deviceId = 0;
  aclrtStream stream = nullptr;
  size_t fsize = 0;

  ACL_CHECK(aclInit(nullptr));
  aclInited = true;
  if (const char *envDevice = std::getenv("ACL_DEVICE_ID"))
    deviceId = std::atoi(envDevice);
  ACL_CHECK(aclrtSetDevice(deviceId));
  deviceSet = true;
  ACL_CHECK(aclrtCreateStream(&stream));

  ACL_CHECK(aclrtMallocHost((void **)&h_in, SZ));
  ACL_CHECK(aclrtMallocHost((void **)&h_out, SZ));
  ACL_CHECK(aclrtMalloc((void **)&d_in, SZ, ACL_MEM_MALLOC_HUGE_FIRST));
  ACL_CHECK(aclrtMalloc((void **)&d_out, SZ, ACL_MEM_MALLOC_HUGE_FIRST));

  fsize = SZ;
  FILE_CHECK(ReadFile("./v1.bin", fsize, h_in, SZ) && fsize == SZ, "./v1.bin");
  ACL_CHECK(aclrtMemcpy(d_in, SZ, h_in, SZ, ACL_MEMCPY_HOST_TO_DEVICE));

  LaunchIssue591PerfRolled(d_in, d_out, stream);
  ACL_CHECK(aclrtSynchronizeStream(stream));

  ACL_CHECK(aclrtMemcpy(h_out, SZ, d_out, SZ, ACL_MEMCPY_DEVICE_TO_HOST));
  FILE_CHECK(WriteFile("./v3.bin", h_out, SZ), "./v3.bin");

cleanup:
  aclrtFree(d_in);
  aclrtFree(d_out);
  aclrtFreeHost(h_in);
  aclrtFreeHost(h_out);
  if (stream != nullptr)
    aclrtDestroyStream(stream);
  if (deviceSet)
    aclrtResetDevice(deviceId);
  if (aclInited)
    aclFinalize();
  return rc;
}
