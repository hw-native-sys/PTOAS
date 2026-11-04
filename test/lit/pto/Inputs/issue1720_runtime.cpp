// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

#include <array>
#include <cstdint>

int main()
{
    alignas(32) std::array<float, 32 * 256> a{}, b{};
    auto addressA = reinterpret_cast<int64_t>(a.data());
    auto addressB = reinterpret_cast<int64_t>(b.data());
    for (bool cond : {false, true}) {
        a.fill(1.0f);
        b.fill(2.0f);
        int64_t expected = (cond ? addressA : addressB) + 16 * 256 * sizeof(float);
        pto::check(subview_address(cond, addressA, addressB) == expected);
        auto &selected = cond ? a : b;
        auto &unselected = cond ? b : a;
        for (unsigned index = 0; index < a.size(); ++index) {
            pto::check(selected[index] == (cond ? 1.0f : 2.0f) * (index < 16 * 256 ? 1 : 2));
            pto::check(unselected[index] == (cond ? 2.0f : 1.0f));
        }
        pto::check(reshape_subview_address(cond, addressA, addressB) == expected);
        pto::check(bitcast_subview_address(cond, addressA, addressB) == expected);
        int64_t branchExpected = cond ? expected : addressB;
        pto::check(branch_subview_address(cond, addressA, addressB, 16, 256) == branchExpected);
    }
}
