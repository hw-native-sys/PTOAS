# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Python DSL form of ``desired_vmi.pto`` — scalar reduce + 1-lane store.

Canonical IR: ``desired_vmi.pto``.
"""

from ptodsl import pto


@pto.jit(target="a5", backend="vpto", mode="explicit")
def direct_scalar_reduce_store():
    tile_ub = pto.alloc_tile(shape=[1, 64], dtype=pto.f32)
    # 32B-aligned tile; store uses a 1-lane mask (ONEPT / 1PT shape).
    dst = pto.alloc_tile(shape=[1, 8], dtype=pto.f32)
    off = pto.const(0, dtype=pto.index)
    mask64 = pto.vmi.create_mask(pto.const(64, dtype=pto.index), size=64)
    mask1 = pto.vmi.create_mask(pto.const(1, dtype=pto.index), size=1)
    tile = pto.vmi.vload(tile_ub.as_ptr(), off, size=64)
    total = pto.vmi.vcadd(tile, mask64, reassoc=True)
    pto.vmi.vstore(total, dst.as_ptr(), off, mask1)


if __name__ == "__main__":
    print(direct_scalar_reduce_store.compile().mlir_text())
