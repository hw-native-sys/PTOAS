// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

#include "generated.cpp"
#include <cmath>
#include <iostream>
#include <vector>

static bool check(int aStride, int bStride, int cStride)
{
    constexpr int rows = 16;
    constexpr int cols = 128;
    constexpr float sentinel = -65536.0f;
    std::vector<float> a(rows * aStride, sentinel);
    std::vector<float> b(rows * bStride, sentinel);
    std::vector<float> c(rows * cStride + 16, sentinel);
    for (int row = 0; row < rows; ++row) {
        for (int col = 0; col < cols; ++col) {
            a[row * aStride + col] = static_cast<float>(row * 17 - col) / 4.0f;
            b[row * bStride + col] = static_cast<float>(col * 3 - row * 11) / 8.0f;
        }
    }
    auto originalA = a;
    auto originalB = b;
    vadd(a.data(), aStride, b.data(), bStride, c.data(), cStride);
    for (int row = 0; row < rows; ++row) {
        for (int col = 0; col < cStride; ++col) {
            float expected = col < cols ? a[row * aStride + col] + b[row * bStride + col] : sentinel;
            if (!std::isfinite(c[row * cStride + col]) || std::abs(c[row * cStride + col] - expected) > 1e-6f) {
                std::cerr << "mismatch at " << row << "," << col << "\n";
                return false;
            }
        }
    }
    for (size_t i = rows * cStride; i < c.size(); ++i) {
        if (std::abs(c[i] - sentinel) > 1e-6f) {
            return false;
        }
    }
    return a == originalA && b == originalB;
}
int main()
{
    if (!check(128, 128, 128) || !check(137, 145, 151)) {
        return 1;
    }
    std::cout << "PASS: 4096 f32 values; row gaps, tail guard, and inputs preserved\n";
    return 0;
}
