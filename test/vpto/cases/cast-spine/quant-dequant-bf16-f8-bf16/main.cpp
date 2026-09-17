// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

// End-to-end host driver for the cast-spine case "quant-dequant-bf16-f8-bf16".
//
//   CAST_SPINE_ITERS   kernel launches inside the timed window (default 1)
//   CAST_SPINE_WARMUP  untimed launches issued first (default 3)
//
// The window is bracketed by two ACL events on the same stream, so it measures
// device time only; the host-side H2D/D2H copies stay outside it.

#include "acl/acl.h"
#include "test_common.h"
#include <cstdint>
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

void LaunchVmi_cast_spine_quant_dequant_bf16_f8_kernel(uint16_t *src,
                                                       uint8_t *q, uint16_t *y,
                                                       void *stream);

int main() {
  constexpr size_t kElems = 16384;
  size_t srcBytes = kElems * sizeof(uint16_t);
  size_t qBytes = kElems * sizeof(uint8_t);
  size_t yBytes = kElems * sizeof(uint16_t);
  uint16_t *srcHost = nullptr;
  uint8_t *qHost = nullptr;
  uint16_t *yHost = nullptr;
  uint16_t *srcDevice = nullptr;
  uint8_t *qDevice = nullptr;
  uint16_t *yDevice = nullptr;
  int rc = 0;
  bool aclInited = false;
  bool deviceSet = false;
  int deviceId = 0;
  aclrtStream stream = nullptr;
  aclrtEvent evStart = nullptr;
  aclrtEvent evStop = nullptr;
  int iters = 1;
  int warmup = 3;

  if (const char *env = std::getenv("CAST_SPINE_ITERS")) {
    iters = std::atoi(env);
  }
  if (const char *env = std::getenv("CAST_SPINE_WARMUP")) {
    warmup = std::atoi(env);
  }
  if (iters < 1) {
    iters = 1;
  }
  if (warmup < 0) {
    warmup = 0;
  }

  ACL_CHECK(aclInit(nullptr));
  aclInited = true;
  if (const char *envDevice = std::getenv("ACL_DEVICE_ID")) {
    deviceId = std::atoi(envDevice);
  }
  ACL_CHECK(aclrtSetDevice(deviceId));
  deviceSet = true;
  ACL_CHECK(aclrtCreateStream(&stream));
  ACL_CHECK(aclrtMallocHost((void **)(&srcHost), srcBytes));
  ACL_CHECK(aclrtMallocHost((void **)(&qHost), qBytes));
  ACL_CHECK(aclrtMallocHost((void **)(&yHost), yBytes));
  ACL_CHECK(aclrtMalloc((void **)&srcDevice, srcBytes, ACL_MEM_MALLOC_HUGE_FIRST));
  ACL_CHECK(aclrtMalloc((void **)&qDevice, qBytes, ACL_MEM_MALLOC_HUGE_FIRST));
  ACL_CHECK(aclrtMalloc((void **)&yDevice, yBytes, ACL_MEM_MALLOC_HUGE_FIRST));

  ReadFile("./v1.bin", srcBytes, srcHost, srcBytes);
  ReadFile("./v2.bin", qBytes, qHost, qBytes);
  ReadFile("./v3.bin", yBytes, yHost, yBytes);
  ACL_CHECK(aclrtMemcpy(srcDevice, srcBytes, srcHost, srcBytes, ACL_MEMCPY_HOST_TO_DEVICE));
  ACL_CHECK(aclrtMemcpy(qDevice, qBytes, qHost, qBytes, ACL_MEMCPY_HOST_TO_DEVICE));
  ACL_CHECK(aclrtMemcpy(yDevice, yBytes, yHost, yBytes, ACL_MEMCPY_HOST_TO_DEVICE));

  for (int i = 0; i < warmup; ++i) {
    LaunchVmi_cast_spine_quant_dequant_bf16_f8_kernel(srcDevice, qDevice,
                                                      yDevice, stream);
  }
  ACL_CHECK(aclrtCreateEvent(&evStart));
  ACL_CHECK(aclrtCreateEvent(&evStop));
  ACL_CHECK(aclrtRecordEvent(evStart, stream));
  for (int i = 0; i < iters; ++i) {
    LaunchVmi_cast_spine_quant_dequant_bf16_f8_kernel(srcDevice, qDevice,
                                                      yDevice, stream);
  }
  ACL_CHECK(aclrtRecordEvent(evStop, stream));
  ACL_CHECK(aclrtSynchronizeStream(stream));
  {
    float elapsedMs = 0.0f;
    ACL_CHECK(aclrtEventElapsedTime(&elapsedMs, evStart, evStop));
    std::printf("[TIMING] case=quant-dequant-bf16-f8-bf16 iters=%d warmup=%d "
                "total_ms=%.6f per_launch_us=%.6f\n",
                iters, warmup, (double)elapsedMs,
                (double)elapsedMs * 1000.0 / (double)iters);
    std::fflush(stdout);
  }

  ACL_CHECK(aclrtMemcpy(qHost, qBytes, qDevice, qBytes, ACL_MEMCPY_DEVICE_TO_HOST));
  ACL_CHECK(aclrtMemcpy(yHost, yBytes, yDevice, yBytes, ACL_MEMCPY_DEVICE_TO_HOST));
  WriteFile("./v2.bin", qHost, qBytes);
  WriteFile("./v3.bin", yHost, yBytes);

cleanup:
  if (evStart) {
    aclrtDestroyEvent(evStart);
  }
  if (evStop) {
    aclrtDestroyEvent(evStop);
  }
  aclrtFree(srcDevice);
  aclrtFree(qDevice);
  aclrtFree(yDevice);
  aclrtFreeHost(srcHost);
  aclrtFreeHost(qHost);
  aclrtFreeHost(yHost);
  if (stream) {
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
