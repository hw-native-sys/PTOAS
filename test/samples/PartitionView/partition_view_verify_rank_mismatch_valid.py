# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

def build() -> str:
    return """module {
  func.func @partition_view_verify_rank_mismatch_valid(%ptr : !pto.ptr<f32>) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %c3 = arith.constant 3 : index
    %c4 = arith.constant 4 : index
    %c8 = arith.constant 8 : index
    %c64 = arith.constant 64 : index
    %c512 = arith.constant 512 : index
    %c4096 = arith.constant 4096 : index

    %tv = pto.make_tensor_view %ptr, shape = [%c8, %c8, %c8, %c64], strides = [%c4096, %c512, %c64, %c1] : !pto.tensor_view<8x8x8x64xf32>
    %pv = pto.partition_view %tv, offsets = [%c2, %c0, %c0, %c0], sizes = [%c4, %c3, %c8, %c64] : !pto.tensor_view<8x8x8x64xf32> -> !pto.partition_tensor_view<48x64xf32>
    return
  }
}
"""


if __name__ == "__main__":
    print(build())
