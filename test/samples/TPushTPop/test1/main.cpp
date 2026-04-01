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

void LaunchMatmulTPushPopPrint(uint8_t *a, uint8_t *b, void *stream);

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
  constexpr size_t aBytes = M * K * sizeof(float);
  constexpr size_t bBytes = K * N * sizeof(float);

  std::vector<float> hostA(M * K, 0.0f);
  std::vector<float> hostB(K * N, 1.0f);
  for (int i = 0; i < M; ++i)
    hostA[i * K + i] = 1.0f;

  ACL_CHECK(aclInit(nullptr));
  ACL_CHECK(aclrtSetDevice(0));

  aclrtStream stream = nullptr;
  ACL_CHECK(aclrtCreateStream(&stream));

  uint8_t *devA = nullptr;
  uint8_t *devB = nullptr;
  ACL_CHECK(aclrtMalloc(reinterpret_cast<void **>(&devA), aBytes,
                        ACL_MEM_MALLOC_HUGE_FIRST));
  ACL_CHECK(aclrtMalloc(reinterpret_cast<void **>(&devB), bBytes,
                        ACL_MEM_MALLOC_HUGE_FIRST));

  ACL_CHECK(aclrtMemcpy(devA, aBytes, hostA.data(), aBytes,
                        ACL_MEMCPY_HOST_TO_DEVICE));
  ACL_CHECK(aclrtMemcpy(devB, bBytes, hostB.data(), bBytes,
                        ACL_MEMCPY_HOST_TO_DEVICE));

  LaunchMatmulTPushPopPrint(devA, devB, stream);
  ACL_CHECK(aclrtSynchronizeStream(stream));

  std::puts("Kernel finished. Expect TPRINT output for the 8x16 Vec tile to be "
            "all 1.0 (A=I, B=all-ones).");

  aclrtFree(devA);
  aclrtFree(devB);
  aclrtDestroyStream(stream);
  aclrtResetDevice(0);
  aclFinalize();
  return 0;
}
