// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

#include "test_common.h"
#include "acl/acl.h"
#include <cstdio>
#include <cstdlib>
#include <cstdint>

using namespace PtoTestCommon;

#ifndef TMRGSORT_HPP
struct MrgSortExecutedNumList {
  uint16_t mrgSortList0;
  uint16_t mrgSortList1;
  uint16_t mrgSortList2;
  uint16_t mrgSortList3;
};
#endif

#define ACL_CHECK(expr)                                                          \
  do {                                                                           \
    const aclError _ret = (expr);                                                \
    if (_ret != ACL_SUCCESS) {                                                   \
      std::fprintf(stderr, "[ERROR] %s failed: %d (%s:%d)\n", #expr,             \
                   (int)_ret, __FILE__, __LINE__);                               \
      const char *_recent = aclGetRecentErrMsg();                                \
      if (_recent != nullptr && _recent[0] != '\0')                             \
        std::fprintf(stderr, "[ERROR] RecentErrMsg: %s\n", _recent);             \
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

void LaunchFifo_tile_data_consume_kernel(__fp16 *a, __fp16 *b, float *c, void *stream);

int main() {
  constexpr size_t kM = 16;
  constexpr size_t kN = 16;
  constexpr size_t kK = 16;
  constexpr size_t aElem = kM * kK;
  constexpr size_t bElem = kK * kN;
  constexpr size_t cElem = 8 * kN;

  constexpr size_t aSize = aElem * sizeof(__fp16);
  constexpr size_t bSize = bElem * sizeof(__fp16);
  constexpr size_t cSize = cElem * sizeof(float);

  __fp16 *aHost = nullptr;
  __fp16 *bHost = nullptr;
  float *cHost = nullptr;
  __fp16 *aDevice = nullptr;
  __fp16 *bDevice = nullptr;
  float *cDevice = nullptr;

  int rc = 0;
  bool aclInited = false;
  bool deviceSet = false;
  int deviceId = 0;
  aclrtStream stream = nullptr;
  size_t inputSize = 0;

  ACL_CHECK(aclInit(nullptr));
  aclInited = true;
  if (const char *envDevice = std::getenv("ACL_DEVICE_ID"))
    deviceId = std::atoi(envDevice);
  ACL_CHECK(aclrtSetDevice(deviceId));
  deviceSet = true;
  ACL_CHECK(aclrtCreateStream(&stream));

  ACL_CHECK(aclrtMallocHost((void **)(&aHost), aSize));
  ACL_CHECK(aclrtMallocHost((void **)(&bHost), bSize));
  ACL_CHECK(aclrtMallocHost((void **)(&cHost), cSize));
  ACL_CHECK(aclrtMalloc((void **)&aDevice, aSize, ACL_MEM_MALLOC_HUGE_FIRST));
  ACL_CHECK(aclrtMalloc((void **)&bDevice, bSize, ACL_MEM_MALLOC_HUGE_FIRST));
  ACL_CHECK(aclrtMalloc((void **)&cDevice, cSize, ACL_MEM_MALLOC_HUGE_FIRST));

  inputSize = aSize;
  FILE_CHECK(ReadFile("./v1.bin", inputSize, aHost, aSize) && inputSize == aSize,
             "./v1.bin");
  inputSize = bSize;
  FILE_CHECK(ReadFile("./v2.bin", inputSize, bHost, bSize) && inputSize == bSize,
             "./v2.bin");
  inputSize = cSize;
  FILE_CHECK(ReadFile("./v3.bin", inputSize, cHost, cSize) && inputSize == cSize,
             "./v3.bin");

  ACL_CHECK(aclrtMemcpy(aDevice, aSize, aHost, aSize, ACL_MEMCPY_HOST_TO_DEVICE));
  ACL_CHECK(aclrtMemcpy(bDevice, bSize, bHost, bSize, ACL_MEMCPY_HOST_TO_DEVICE));
  ACL_CHECK(aclrtMemcpy(cDevice, cSize, cHost, cSize, ACL_MEMCPY_HOST_TO_DEVICE));

  LaunchFifo_tile_data_consume_kernel(aDevice, bDevice, cDevice, stream);
  ACL_CHECK(aclrtSynchronizeStream(stream));

  ACL_CHECK(aclrtMemcpy(cHost, cSize, cDevice, cSize, ACL_MEMCPY_DEVICE_TO_HOST));
  FILE_CHECK(WriteFile("./v3.bin", cHost, cSize), "./v3.bin");

cleanup:
  aclrtFree(aDevice);
  aclrtFree(bDevice);
  aclrtFree(cDevice);
  aclrtFreeHost(aHost);
  aclrtFreeHost(bHost);
  aclrtFreeHost(cHost);
  if (stream != nullptr)
    aclrtDestroyStream(stream);
  if (deviceSet)
    aclrtResetDevice(deviceId);
  if (aclInited)
    aclFinalize();
  return rc;
}
