// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

#include <array>
#include <cstdint>
#include <initializer_list>

int main() {
    pto::check(capture_time(0) == 32);
    pto::check(capture_time(1) == 32);
    pto::check(capture_time(2) == 16);

    alignas(32) std::array<float, 32 * 256> a{}, b{};
    for (bool first : {false, true}) {
        for (bool second : {false, true}) {
            auto &selected = first ? a : b;
            const int row = (second ? 16 : 8) + 1;
            const auto expected = reinterpret_cast<int64_t>(&selected[row * 256]);
            pto::check(noncompact_subview(first, second, reinterpret_cast<int64_t>(a.data()),
                                         reinterpret_cast<int64_t>(b.data())) == expected);
        }
    }
}
