# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Python DSL form of ``current_slow_vmi.pto`` — continuous bf16 strip.

Canonical IR: ``current_slow_vmi.pto``.
"""

from ptodsl import pto


@pto.jit(target="a5", backend="vpto", mode="explicit")
def bf16_strip_continuous():
    src = pto.alloc_tile(shape=[1, 64], dtype=pto.bf16)
    dst = pto.alloc_tile(shape=[1, 64], dtype=pto.bf16)
    off = pto.const(0, dtype=pto.index)
    x = pto.vmi.vload(src.as_ptr(), off, size=64)
    xf = pto.vmi.vcvt(x, pto.f32)
    scale = pto.vmi.vbrc(pto.f32(1.0), size=64)
    y = pto.vmi.vmul(xf, scale)
    yo = pto.vmi.vcvt(y, pto.bf16, saturate="SAT")
    pto.vmi.vstore(yo, dst.as_ptr(), off)


if __name__ == "__main__":
    print(bf16_strip_continuous.compile().mlir_text())
