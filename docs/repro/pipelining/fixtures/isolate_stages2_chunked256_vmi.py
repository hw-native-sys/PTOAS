# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Desired VMI schedule cell: stages=2, DMA block_k=512, VF body chunked at 256.

Primary ask for SoftPipeline parity — not a single size=512/1024 vmul.
Matches ASC SoftPipeline (chunked SIMD + dual-buffer MTE overlap).
"""

from ptodsl import pto

_VF = 256
_BLOCK = 512


@pto.jit(target="a5", backend="vpto", mode="explicit")
def persistent_scale_stages2_chunked256(*, BLOCK_COUNT: pto.const_expr = 4):
    x_ub0 = pto.alloc_tile(shape=[1, _BLOCK], dtype=pto.f32)
    x_ub1 = pto.alloc_tile(shape=[1, _BLOCK], dtype=pto.f32)
    y_ub0 = pto.alloc_tile(shape=[1, _BLOCK], dtype=pto.f32)
    y_ub1 = pto.alloc_tile(shape=[1, _BLOCK], dtype=pto.f32)
    # 8×f32 = 32 B row alignment (PTODSL tile contract).
    scale_ub = pto.alloc_tile(shape=[1, 8], dtype=pto.f32)
    off0 = pto.const(0, dtype=pto.index)
    scale_slot = pto.vmi.vload(scale_ub.as_ptr(), off0, size=1)
    scale = pto.vmi.vbrc(scale_slot, size=_VF)
    for i in range(BLOCK_COUNT):
        off = i * _BLOCK
        if (i % 2) == 1:
            x_ptr, y_ptr = x_ub1.as_ptr(), y_ub1.as_ptr()
        else:
            x_ptr, y_ptr = x_ub0.as_ptr(), y_ub0.as_ptr()
        for j in range(_BLOCK // _VF):
            col = j * _VF
            xv = pto.vmi.vload(x_ptr, off + col, size=_VF)
            yv = pto.vmi.vmul(xv, scale)
            pto.vmi.vstore(yv, y_ptr, off + col)


if __name__ == "__main__":
    print(persistent_scale_stages2_chunked256.compile().mlir_text())
