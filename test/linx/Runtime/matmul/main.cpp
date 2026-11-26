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

using Kernel = void (*)(float*, int32_t, float*, int32_t, float*, int32_t);
static bool check(Kernel kernel, int m, int k, int n, int gap, float factor = 1.0f)
{
    constexpr float sentinel = -65536.0f;
    int as = k + gap;
    int bs = n + gap + 1;
    int ds = n + gap + 3;
    std::vector<float> a(m * as, sentinel);
    std::vector<float> b(k * bs, sentinel);
    std::vector<float> d(m * ds + 16, sentinel);
    for (int row = 0; row < m; ++row) {
        for (int col = 0; col < k; ++col) {
            a[row * as + col] = static_cast<float>((row * 3 + col * 7) % 19 - 9) / 4.0f;
        }
    }
    for (int row = 0; row < k; ++row) {
        for (int col = 0; col < n; ++col) {
            b[row * bs + col] = static_cast<float>((row * 11 + col * 5) % 23 - 11) / 8.0f;
        }
    }
    auto originalA = a;
    auto originalB = b;
    kernel(a.data(), as, b.data(), bs, d.data(), ds);
    for (int row = 0; row < m; ++row) {
        for (int col = 0; col < ds; ++col) {
            float expected = sentinel;
            if (col < n) {
                expected = 0.0f;
                for (int inner = 0; inner < k; ++inner) {
                    expected += a[row * as + inner] * b[inner * bs + col];
                }
                expected *= factor;
            }
            float actual = d[row * ds + col];
            if (!std::isfinite(actual) || std::abs(actual - expected) > 1e-6f) {
                std::cerr << "matmul mismatch " << m << "x" << k << "x" << n
                          << " at " << row << "," << col << "\n";
                return false;
            }
        }
    }
    for (size_t i = m * ds; i < d.size(); ++i) {
        if (std::abs(d[i] - sentinel) > 1e-6f) {
            return false;
        }
    }
    return a == originalA && b == originalB;
}
int main()
{
    for (int gap : {0, 7}) {
        if (!check(matmul_m16, 16, 8, 8, gap) || !check(matmul_m32, 32, 12, 16, gap) ||
            !check(matmul_tail, 7, 5, 11, gap) || !check(role_propagation, 16, 8, 8, gap, 2.0f)) {
            return 1;
        }
    }
    std::cout << "PASS: matmul M16/M32/tails, role propagation, strides, guards and inputs preserved\n";
    return 0;
}
