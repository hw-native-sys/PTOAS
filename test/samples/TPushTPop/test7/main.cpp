// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

#include "acl/acl.h"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

void LaunchScope3Incore0Incore0(float *attn_out, uint16_t *hidden_states,
                                float *resid_out, uint16_t *wo, void *stream);

#define ACL_CHECK(expr)                                                        \
  do {                                                                         \
    aclError _ret = (expr);                                                    \
    if (_ret != ACL_SUCCESS) {                                                 \
      std::fprintf(stderr, "[ACL ERROR] %s failed: %d (%s:%d)\n", #expr,       \
                   (int)_ret, __FILE__, __LINE__);                             \
      return 1;                                                                \
    }                                                                          \
  } while (0)

static uint16_t floatToBf16Bits(float value) {
  uint32_t bits = 0;
  std::memcpy(&bits, &value, sizeof(bits));
  const uint32_t lsb = (bits >> 16) & 1u;
  bits += 0x7FFFu + lsb;
  return static_cast<uint16_t>(bits >> 16);
}

static float bf16BitsToFloat(uint16_t bits) {
  uint32_t value = static_cast<uint32_t>(bits) << 16;
  float result = 0.0f;
  std::memcpy(&result, &value, sizeof(result));
  return result;
}

static float attnValue(int row, int col) {
  return static_cast<float>(row * 10 + col);
}

static float hiddenValue(int row, int col) {
  return static_cast<float>((row % 4) * 3 - (col % 5));
}

static void printMatrix(const char *title, const std::vector<float> &data,
                        int rows, int cols) {
  std::printf("%s\n", title);
  for (int row = 0; row < rows; ++row) {
    std::printf("row %02d:", row);
    for (int col = 0; col < cols; ++col)
      std::printf(" %7.1f", data[static_cast<size_t>(row) * cols + col]);
    std::printf("\n");
  }
}

int main() {
  constexpr int Rows = 16;
  constexpr int KCols = 128;
  constexpr int OutCols = 64;
  constexpr int InitValue = -777;
  constexpr float Atol = 1e-3f;
  constexpr float Rtol = 1e-3f;

  constexpr size_t attnElems = static_cast<size_t>(Rows) * KCols;
  constexpr size_t hiddenElems = static_cast<size_t>(Rows) * OutCols;
  constexpr size_t outElems = static_cast<size_t>(Rows) * OutCols;
  constexpr size_t weightElems = static_cast<size_t>(KCols) * OutCols;

  constexpr size_t attnBytes = attnElems * sizeof(float);
  constexpr size_t hiddenBytes = hiddenElems * sizeof(uint16_t);
  constexpr size_t outBytes = outElems * sizeof(float);
  constexpr size_t weightBytes = weightElems * sizeof(uint16_t);

  std::vector<float> hostAttn(attnElems, 0.0f);
  std::vector<float> hostAttnRounded(attnElems, 0.0f);
  std::vector<uint16_t> hostHidden(hiddenElems, 0);
  std::vector<float> hostHiddenF32(hiddenElems, 0.0f);
  std::vector<uint16_t> hostWeight(weightElems, 0);
  std::vector<float> hostOut(outElems, static_cast<float>(InitValue));
  std::vector<float> hostGolden(outElems, 0.0f);

  for (int row = 0; row < Rows; ++row) {
    for (int col = 0; col < KCols; ++col) {
      const size_t idx = static_cast<size_t>(row) * KCols + col;
      hostAttn[idx] = attnValue(row, col);
      hostAttnRounded[idx] = bf16BitsToFloat(floatToBf16Bits(hostAttn[idx]));
    }
  }

  for (int row = 0; row < Rows; ++row) {
    for (int col = 0; col < OutCols; ++col) {
      const size_t idx = static_cast<size_t>(row) * OutCols + col;
      hostHidden[idx] = floatToBf16Bits(hiddenValue(row, col));
      hostHiddenF32[idx] = bf16BitsToFloat(hostHidden[idx]);
      hostGolden[idx] =
          hostAttnRounded[static_cast<size_t>(row) * KCols + col] +
          hostHiddenF32[idx];
    }
  }

  for (int row = 0; row < KCols; ++row) {
    for (int col = 0; col < OutCols; ++col) {
      const size_t idx = static_cast<size_t>(row) * OutCols + col;
      hostWeight[idx] = floatToBf16Bits(row == col ? 1.0f : 0.0f);
    }
  }

  ACL_CHECK(aclInit(nullptr));
  ACL_CHECK(aclrtSetDevice(0));

  aclrtStream stream = nullptr;
  ACL_CHECK(aclrtCreateStream(&stream));

  float *devAttn = nullptr;
  uint16_t *devHidden = nullptr;
  float *devOut = nullptr;
  uint16_t *devWeight = nullptr;

  ACL_CHECK(aclrtMalloc(reinterpret_cast<void **>(&devAttn), attnBytes,
                        ACL_MEM_MALLOC_HUGE_FIRST));
  ACL_CHECK(aclrtMalloc(reinterpret_cast<void **>(&devHidden), hiddenBytes,
                        ACL_MEM_MALLOC_HUGE_FIRST));
  ACL_CHECK(aclrtMalloc(reinterpret_cast<void **>(&devOut), outBytes,
                        ACL_MEM_MALLOC_HUGE_FIRST));
  ACL_CHECK(aclrtMalloc(reinterpret_cast<void **>(&devWeight), weightBytes,
                        ACL_MEM_MALLOC_HUGE_FIRST));

  ACL_CHECK(aclrtMemcpy(devAttn, attnBytes, hostAttn.data(), attnBytes,
                        ACL_MEMCPY_HOST_TO_DEVICE));
  ACL_CHECK(aclrtMemcpy(devHidden, hiddenBytes, hostHidden.data(), hiddenBytes,
                        ACL_MEMCPY_HOST_TO_DEVICE));
  ACL_CHECK(aclrtMemcpy(devOut, outBytes, hostOut.data(), outBytes,
                        ACL_MEMCPY_HOST_TO_DEVICE));
  ACL_CHECK(aclrtMemcpy(devWeight, weightBytes, hostWeight.data(), weightBytes,
                        ACL_MEMCPY_HOST_TO_DEVICE));

  LaunchScope3Incore0Incore0(devAttn, devHidden, devOut, devWeight, stream);

  ACL_CHECK(aclrtSynchronizeStream(stream));
  ACL_CHECK(aclrtMemcpy(hostOut.data(), outBytes, devOut, outBytes,
                        ACL_MEMCPY_DEVICE_TO_HOST));

  printMatrix("device output:", hostOut, Rows, OutCols);
  printMatrix("golden output:", hostGolden, Rows, OutCols);

  int mismatchCount = 0;
  for (int row = 0; row < Rows; ++row) {
    for (int col = 0; col < OutCols; ++col) {
      const size_t idx = static_cast<size_t>(row) * OutCols + col;
      const float got = hostOut[idx];
      const float expect = hostGolden[idx];
      const float diff = std::fabs(got - expect);
      const float limit = Atol + Rtol * std::fabs(expect);
      if (diff > limit) {
        if (mismatchCount < 16) {
          std::fprintf(stderr,
                       "Mismatch at (%d, %d): got %.6f, expect %.6f, diff %.6f, "
                       "limit %.6f\n",
                       row, col, got, expect, diff, limit);
        }
        ++mismatchCount;
      }
    }
  }

  if (mismatchCount == 0) {
    std::puts("scope3_incore_0_incore_0 test7 passed.");
  } else {
    std::fprintf(stderr, "Found %d mismatches.\n", mismatchCount);
  }

  aclrtFree(devWeight);
  aclrtFree(devOut);
  aclrtFree(devHidden);
  aclrtFree(devAttn);
  aclrtDestroyStream(stream);
  aclrtResetDevice(0);
  aclFinalize();
  return mismatchCount == 0 ? 0 : 1;
}
