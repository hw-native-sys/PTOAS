// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

#include "acl/acl.h"
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>

void LaunchMatmulTPushPopLoop4Print(uint8_t *a, uint8_t *bAll,
                                    uint8_t *gmSlotBuffer, void *stream);

#define ACL_CHECK(expr)                                                        \
  do {                                                                         \
    aclError _ret = (expr);                                                    \
    if (_ret != ACL_SUCCESS) {                                                 \
      std::fprintf(stderr, "[ACL ERROR] %s failed: %d (%s:%d)\n", #expr,       \
                   (int)_ret, __FILE__, __LINE__);                             \
      return 1;                                                                \
    }                                                                          \
  } while (0)

int main() {
  constexpr int M = 16;
  constexpr int K = 16;
  constexpr int N = 16;
  constexpr int Iter = 4;
  constexpr size_t aBytes = M * K * sizeof(float);
  constexpr size_t bBytes = Iter * K * N * sizeof(float);
  constexpr size_t slotBytes = 8192;

  std::vector<float> hostA(M * K, 0.0f);
  std::vector<float> hostBAll(Iter * K * N, 0.0f);
  std::vector<uint8_t> hostSlotBuffer(slotBytes, 0);
  for (int i = 0; i < M; ++i)
    hostA[i * K + i] = 1.0f;

  for (int iter = 0; iter < Iter; ++iter) {
    const float value = static_cast<float>(iter + 1);
    const size_t base = static_cast<size_t>(iter) * K * N;
    for (int idx = 0; idx < K * N; ++idx)
      hostBAll[base + idx] = value;
  }

  ACL_CHECK(aclInit(nullptr));
  ACL_CHECK(aclrtSetDevice(0));

  aclrtStream stream = nullptr;
  ACL_CHECK(aclrtCreateStream(&stream));

  uint8_t *devA = nullptr;
  uint8_t *devBAll = nullptr;
  uint8_t *devSlotBuffer = nullptr;
  ACL_CHECK(aclrtMalloc(reinterpret_cast<void **>(&devA), aBytes,
                        ACL_MEM_MALLOC_HUGE_FIRST));
  ACL_CHECK(aclrtMalloc(reinterpret_cast<void **>(&devBAll), bBytes,
                        ACL_MEM_MALLOC_HUGE_FIRST));
  ACL_CHECK(aclrtMalloc(reinterpret_cast<void **>(&devSlotBuffer), slotBytes,
                        ACL_MEM_MALLOC_HUGE_FIRST));

  ACL_CHECK(aclrtMemcpy(devA, aBytes, hostA.data(), aBytes,
                        ACL_MEMCPY_HOST_TO_DEVICE));
  ACL_CHECK(aclrtMemcpy(devBAll, bBytes, hostBAll.data(), bBytes,
                        ACL_MEMCPY_HOST_TO_DEVICE));
  ACL_CHECK(aclrtMemcpy(devSlotBuffer, slotBytes, hostSlotBuffer.data(),
                        slotBytes, ACL_MEMCPY_HOST_TO_DEVICE));

  LaunchMatmulTPushPopLoop4Print(devA, devBAll, devSlotBuffer, stream);
  ACL_CHECK(aclrtSynchronizeStream(stream));

  std::puts("Kernel finished. Expect 4 TPRINT blocks with 16x16 outputs filled "
            "with 1.0, 2.0, 3.0, 4.0 in order.");

  aclrtFree(devSlotBuffer);
  aclrtFree(devA);
  aclrtFree(devBAll);
  aclrtDestroyStream(stream);
  aclrtResetDevice(0);
  aclFinalize();
  return 0;
}
