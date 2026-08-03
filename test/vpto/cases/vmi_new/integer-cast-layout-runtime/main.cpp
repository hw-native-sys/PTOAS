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
      rc = 1;                                                                  \
      goto cleanup;                                                            \
    }                                                                          \
  } while (0)

void LaunchIntegerCastU8(uint8_t *, uint16_t *, uint32_t *, void *);
void LaunchIntegerCastI8(int8_t *, int16_t *, int32_t *, void *);
void LaunchIntegerCastU16(uint16_t *, uint32_t *, uint8_t *, void *);
void LaunchIntegerCastI16(int16_t *, int32_t *, int8_t *, void *);
void LaunchIntegerCastU32(uint32_t *, uint16_t *, uint8_t *, void *);
void LaunchIntegerCastI32(int32_t *, int16_t *, int8_t *, void *);
void LaunchIntegerCastZeroGap(uint8_t *, uint16_t *, uint16_t *, uint32_t *,
                              uint32_t *, void *);

int main() {
  constexpr size_t kBuffers = 23;
  constexpr bool kIsOutput[kBuffers] = {false, true, true, false, true, true,
                                        false, true, true, false, true, true,
                                        false, true, true, false, true, true,
                                        false, false, true, true, true};
  constexpr size_t kSizes[kBuffers] = {256, 512, 1024, 256, 512, 1024,
                                       256, 512, 128,  256, 512, 128,
                                       256, 128, 64,   256, 128, 64,
                                       128, 128, 256,  256, 256};
  void *host[kBuffers] = {};
  void *device[kBuffers] = {};
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

  for (size_t i = 0; i < kBuffers; ++i) {
    ACL_CHECK(aclrtMallocHost(&host[i], kSizes[i]));
    ACL_CHECK(aclrtMalloc(&device[i], kSizes[i], ACL_MEM_MALLOC_HUGE_FIRST));
    char path[32];
    std::snprintf(path, sizeof(path), "./v%zu.bin", i + 1);
    size_t fileSize = kSizes[i];
    ReadFile(path, fileSize, host[i], kSizes[i]);
    ACL_CHECK(aclrtMemcpy(device[i], kSizes[i], host[i], kSizes[i],
                          ACL_MEMCPY_HOST_TO_DEVICE));
  }

  LaunchIntegerCastU8(static_cast<uint8_t *>(device[0]),
                      static_cast<uint16_t *>(device[1]),
                      static_cast<uint32_t *>(device[2]), stream);
  LaunchIntegerCastI8(static_cast<int8_t *>(device[3]),
                      static_cast<int16_t *>(device[4]),
                      static_cast<int32_t *>(device[5]), stream);
  LaunchIntegerCastU16(static_cast<uint16_t *>(device[6]),
                       static_cast<uint32_t *>(device[7]),
                       static_cast<uint8_t *>(device[8]), stream);
  LaunchIntegerCastI16(static_cast<int16_t *>(device[9]),
                       static_cast<int32_t *>(device[10]),
                       static_cast<int8_t *>(device[11]), stream);
  LaunchIntegerCastU32(static_cast<uint32_t *>(device[12]),
                       static_cast<uint16_t *>(device[13]),
                       static_cast<uint8_t *>(device[14]), stream);
  LaunchIntegerCastI32(static_cast<int32_t *>(device[15]),
                       static_cast<int16_t *>(device[16]),
                       static_cast<int8_t *>(device[17]), stream);
  LaunchIntegerCastZeroGap(
      static_cast<uint8_t *>(device[18]),
      static_cast<uint16_t *>(device[19]),
      static_cast<uint16_t *>(device[20]),
      static_cast<uint32_t *>(device[21]),
      static_cast<uint32_t *>(device[22]), stream);
  ACL_CHECK(aclrtSynchronizeStream(stream));

  for (size_t i = 0; i < kBuffers; ++i) {
    if (!kIsOutput[i])
      continue;
    ACL_CHECK(aclrtMemcpy(host[i], kSizes[i], device[i], kSizes[i],
                          ACL_MEMCPY_DEVICE_TO_HOST));
    char path[32];
    std::snprintf(path, sizeof(path), "./v%zu.bin", i + 1);
    WriteFile(path, host[i], kSizes[i]);
  }

cleanup:
  for (void *ptr : device)
    aclrtFree(ptr);
  for (void *ptr : host)
    aclrtFreeHost(ptr);
  if (stream)
    aclrtDestroyStream(stream);
  if (deviceSet)
    aclrtResetDevice(deviceId);
  if (aclInited)
    aclFinalize();
  return rc;
}
