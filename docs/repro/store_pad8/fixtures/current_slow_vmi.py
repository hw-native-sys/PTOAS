# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Python DSL form of ``current_slow_vmi.pto`` — group-reduce into 8-slot pad + store.

Canonical IR: ``current_slow_vmi.pto``.
"""

from ptodsl import pto


@pto.jit(target="a5", backend="vpto", mode="explicit")
def padded_reduce_store():
    tile_ub = pto.alloc_tile(shape=[1, 64], dtype=pto.f32)
    dst = pto.alloc_tile(shape=[1, 8], dtype=pto.f32)
    off = pto.const(0, dtype=pto.index)
    mask = pto.vmi.create_mask(pto.const(64, dtype=pto.index), size=64)
    tile = pto.vmi.vload(tile_ub.as_ptr(), off, size=64)
    sum8 = pto.vmi.vcadd(tile, mask, group=8, reassoc=True)
    pto.vmi.vstore(
        sum8,
        dst.as_ptr(),
        off,
        group=8,
        stride=pto.const(1, dtype=pto.index),
    )


if __name__ == "__main__":
    print(padded_reduce_store.compile().mlir_text())
