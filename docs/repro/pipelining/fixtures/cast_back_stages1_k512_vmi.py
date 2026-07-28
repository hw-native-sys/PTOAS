# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Cast-back Persistent: e4m3 -> f32 * scale -> bf16, stages=1, tile_k=512."""

from ptodsl import pto


@pto.jit(target="a5", backend="vpto", mode="explicit")
def persistent_cast_back_k512(*, BLOCK_COUNT: pto.const_expr = 4):
    x_ub = pto.alloc_tile(shape=[1, 512], dtype=pto.f8e4m3)
    y_ub = pto.alloc_tile(shape=[1, 512], dtype=pto.bf16)
    scale_ub = pto.alloc_tile(shape=[1, 8], dtype=pto.f32)
    off0 = pto.const(0, dtype=pto.index)
    scale_slot = pto.vmi.vload(scale_ub.as_ptr(), off0, size=1)
    scale = pto.vmi.vbrc(scale_slot, size=512)
    for i in range(BLOCK_COUNT):
        off = i * 512
        x_f8 = pto.vmi.vload(x_ub.as_ptr(), off, size=512)
        x_f32 = pto.vmi.vcvt(x_f8, pto.f32)
        scaled = pto.vmi.vmul(x_f32, scale)
        y = pto.vmi.vcvt(scaled, pto.bf16)
        pto.vmi.vstore(y, y_ub.as_ptr(), off)


if __name__ == "__main__":
    print(persistent_cast_back_k512.compile().mlir_text())
