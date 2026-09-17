// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

// End-to-end host driver for the cast-spine case "roundtrip-bf16-f16-bf16".
//
// The kernel converts 16384 elements per launch.  Two environment
// variables make the same binary usable for both functional validation and
// timing:
//
//   CAST_SPINE_ITERS   number of kernel launches inside the timed window
//                      (default 1)
//   CAST_SPINE_WARMUP  untimed launches issued first (default 3)
//
// The timed window is bracketed by two ACL events recorded on the same stream,
// so it measures device time only; the host-side H2D/D2H copies and the
// allocation are outside the window.  Repeating the run at several ITERS values
// and fitting a line removes the residual fixed launch cost.

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

void LaunchVmi_cast_spine_bf16_f16_bf16_kernel(uint16_t *src, uint16_t *dst, void *stream);

int main() {
  constexpr size_t kElems = 16384;
  // ReadFile() takes a non-const size; keep the byte count in a mutable local.
  size_t kBytes = kElems * sizeof(uint16_t);
  uint16_t *srcHost = nullptr;
  uint16_t *dstHost = nullptr;
  uint16_t *srcDevice = nullptr;
  uint16_t *dstDevice = nullptr;
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
  ACL_CHECK(aclrtMallocHost((void **)(&srcHost), kBytes));
  ACL_CHECK(aclrtMallocHost((void **)(&dstHost), kBytes));
  ACL_CHECK(aclrtMalloc((void **)&srcDevice, kBytes, ACL_MEM_MALLOC_HUGE_FIRST));
  ACL_CHECK(aclrtMalloc((void **)&dstDevice, kBytes, ACL_MEM_MALLOC_HUGE_FIRST));

  ReadFile("./v1.bin", kBytes, srcHost, kBytes);
  ReadFile("./v2.bin", kBytes, dstHost, kBytes);
  ACL_CHECK(aclrtMemcpy(srcDevice, kBytes, srcHost, kBytes, ACL_MEMCPY_HOST_TO_DEVICE));
  ACL_CHECK(aclrtMemcpy(dstDevice, kBytes, dstHost, kBytes, ACL_MEMCPY_HOST_TO_DEVICE));

  for (int i = 0; i < warmup; ++i) {
    LaunchVmi_cast_spine_bf16_f16_bf16_kernel(srcDevice, dstDevice, stream);
  }
  ACL_CHECK(aclrtCreateEvent(&evStart));
  ACL_CHECK(aclrtCreateEvent(&evStop));
  ACL_CHECK(aclrtRecordEvent(evStart, stream));
  for (int i = 0; i < iters; ++i) {
    LaunchVmi_cast_spine_bf16_f16_bf16_kernel(srcDevice, dstDevice, stream);
  }
  ACL_CHECK(aclrtRecordEvent(evStop, stream));
  ACL_CHECK(aclrtSynchronizeStream(stream));
  {
    float elapsedMs = 0.0f;
    ACL_CHECK(aclrtEventElapsedTime(&elapsedMs, evStart, evStop));
    std::printf("[TIMING] case=roundtrip-bf16-f16-bf16 iters=%d warmup=%d total_ms=%.6f "
                "per_launch_us=%.6f\n",
                iters, warmup, (double)elapsedMs,
                (double)elapsedMs * 1000.0 / (double)iters);
    std::fflush(stdout);
  }

  ACL_CHECK(aclrtMemcpy(dstHost, kBytes, dstDevice, kBytes, ACL_MEMCPY_DEVICE_TO_HOST));
  WriteFile("./v2.bin", dstHost, kBytes);

cleanup:
  if (evStart) {
    aclrtDestroyEvent(evStart);
  }
  if (evStop) {
    aclrtDestroyEvent(evStop);
  }
  aclrtFree(srcDevice);
  aclrtFree(dstDevice);
  aclrtFreeHost(srcHost);
  aclrtFreeHost(dstHost);
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
