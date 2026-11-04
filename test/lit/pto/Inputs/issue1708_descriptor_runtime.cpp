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
  alignas(32) std::array<float, 32 * 32> a, b;
  auto addressA = reinterpret_cast<int64_t>(a.data());
  auto addressB = reinterpret_cast<int64_t>(b.data());
  for (bool cond : {false, true}) {
    a.fill(1.0f);
    b.fill(2.0f);
    const int before = cond ? 32 : 8;
    auto &selected = cond ? a : b;
    auto &unselected = cond ? b : a;
    pto::fillpadCount = 0;
    pto::check(branch_fillpad(cond, addressA, addressB) ==
           before * 1000000 + 16 * 10000 + 32 * 100 + 8);
    pto::check(pto::fillpadCount == 2);
    pto::check(pto::fillpadCalls[0].rows == static_cast<unsigned>(before));
    pto::check(pto::fillpadCalls[1].rows == 16);
    for (unsigned i = 0; i < pto::fillpadCount; ++i) {
      pto::check(pto::fillpadCalls[i].data == selected.data());
      const unsigned expectedCols = i == 0 && !cond ? 16 : 32;
      pto::check(pto::fillpadCalls[i].cols == expectedCols);
    }
    // A consumer receives the selected allocation's storage, not a payload
    // copy. Writing through that address changes only the selected source.
    static_cast<float *>(pto::fillpadCalls[0].data)[0] = 7.0f;
    pto::check(selected[0] == 7.0f);
    pto::check(unselected[0] == (cond ? 2.0f : 1.0f));
    for (int rows : {0, 16}) {
      a.fill(1.0f);
      b.fill(2.0f);
      pto::fillpadCount = 0;
      pto::check(branch_reshape(cond, addressA, addressB, rows) ==
             before * 10000 + rows * 100 + 32);
      pto::check(pto::fillpadCount == 1);
      pto::check(pto::fillpadCalls[0].data == selected.data());
      pto::check(pto::fillpadCalls[0].rows == static_cast<unsigned>(rows));
      pto::check(pto::fillpadCalls[0].cols == 32);
    }
    pto::check(two_reshapes(cond, addressA, addressB) == 16);
    pto::check(two_bitcasts(addressA, addressB) == 16);
    pto::check(two_subviews(addressA, addressB, 16, 32) == 8);
    pto::check(branch_capture(cond, addressA, addressB) == (cond ? 16 : 8));
    pto::check(branch_read_only(cond, addressA, addressB) == (cond ? 16 : 8));
    pto::check(constant_selection(addressA, addressB) == 1600);
    pto::check(arith_selection(cond, addressA, addressB) == (cond ? 3216 : 816));
    pto::check(same_source(cond, addressA, addressB) == 1632);
    pto::check(snapshot_time(cond, addressA, addressB) == before * 100);
    for (bool inner : {false, true}) {
      pto::check(nested_multi_result(cond, inner, addressA, addressB) ==
             (cond ? 716 : 916));
    }
  }
}
