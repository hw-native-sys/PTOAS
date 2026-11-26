// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

#include "generated.cpp"
#include <algorithm>
#include <cmath>
#include <iostream>
#include <vector>

using Kernel = void (*)(float*, int32_t, float*, int32_t);
static bool check(Kernel kernel, int rows, int cols, int gap, int pattern)
{
    constexpr float sentinel = -65536.0f;
    int inputStride = cols + gap;
    int outputStride = cols + gap + 3;
    std::vector<float> input(rows * inputStride, sentinel);
    std::vector<float> output(rows * outputStride + 16, sentinel);
    for (int row = 0; row < rows; ++row) {
        for (int col = 0; col < cols; ++col) {
            float value = static_cast<float>((row * 17 + col * 11) % 53 - 26) / 4.0f;
            if (pattern == 1) {
                value = 10000.0f + value;
            } else if (pattern == 2) {
                value = -10000.0f;
            } else if (pattern == 3) {
                value = col == row % cols ? 1000.0f : -1000.0f;
            }
            input[row * inputStride + col] = value;
        }
    }
    auto original = input;
    kernel(input.data(), inputStride, output.data(), outputStride);
    for (int row = 0; row < rows; ++row) {
        double maximum = *std::max_element(input.begin() + row * inputStride,
                                           input.begin() + row * inputStride + cols);
        double total = 0.0;
        double outputSum = 0.0;
        for (int col = 0; col < cols; ++col) {
            total += std::exp(static_cast<double>(input[row * inputStride + col]) - maximum);
        }
        for (int col = 0; col < outputStride; ++col) {
            double expected = col < cols ?
                std::exp(static_cast<double>(input[row * inputStride + col]) - maximum) / total : sentinel;
            float actual = output[row * outputStride + col];
            if (!std::isfinite(actual) || std::abs(actual - expected) > 1e-6) {
                std::cerr << "softmax mismatch at " << row << "," << col << "\n";
                return false;
            }
            if (col < cols) {
                outputSum += actual;
            }
        }
        if (std::abs(outputSum - 1.0) > 2e-6) {
            return false;
        }
    }
    for (size_t i = rows * outputStride; i < output.size(); ++i) {
        if (std::abs(output[i] - sentinel) > 1e-6f) {
            return false;
        }
    }
    return input == original;
}
int main()
{
    for (int pattern = 0; pattern < 4; ++pattern) {
        for (int gap : {0, 9}) {
            if (!check(softmax, 16, 128, gap, pattern) || !check(softmax_tail, 7, 13, gap, pattern) ||
                !check(softmax_cube, 16, 16, gap, pattern)) {
                return 1;
            }
        }
    }
    std::cout << "PASS: softmax row-major/CUBE/tails, stable exponentials, row sums, strides and guards\n";
    return 0;
}
