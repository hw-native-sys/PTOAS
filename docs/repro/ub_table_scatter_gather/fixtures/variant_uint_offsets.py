# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Ruled-out variant: reinterpret offsets as uint32 before scatter/gather.

AscendC peers often pass a uint reinterpret of the offset vector. On VMI the
spec types the offset as i32. Both signed and unsigned were tried; both still
return the seed sentinel.
"""

from ptodsl import pto

_VL = 64
_POOL = 256


@pto.jit(
    name="ub_table_scatter_gather_uint_offsets",
    kernel_kind="vector",
    target="a5",
    backend="vpto",
    mode="explicit",
)
def ub_table_scatter_gather_uint_offsets(
    val_gm: pto.ptr(pto.f32, "gm"),
    out_gm: pto.ptr(pto.f32, "gm"),
):
    c0 = pto.const(0)
    c1 = pto.const(1)
    c_vl = pto.const(_VL)
    shape = [c1, c1, c1, c1, c_vl]
    strides = [c_vl, c_vl, c_vl, c_vl, c1]
    off = [c0, c0, c0, c0, c0]

    val_view = pto.make_tensor_view(val_gm, shape=shape, strides=strides)
    out_view = pto.make_tensor_view(out_gm, shape=shape, strides=strides)
    val_part = pto.partition_view(val_view, offsets=off, sizes=shape)
    out_part = pto.partition_view(out_view, offsets=off, sizes=shape)

    pool = pto.alloc_tile(shape=[1, _POOL], dtype=pto.f32)
    val_ub = pto.alloc_tile(shape=[1, _VL], dtype=pto.f32)
    out_ub = pto.alloc_tile(shape=[1, _VL], dtype=pto.f32)
    pto.tile.load(val_part, val_ub)

    mask = pto.vmi.create_mask(_VL, size=_VL)
    idx = pto.vmi.vci(pto.i32(0), size=_VL, order="ASC")
    idx_u = pto.vmi.vinterpret_cast(idx, pto.ui32)
    off0 = pto.const(0, dtype=pto.index)

    neg_inf = pto.vmi.vbrc(pto.f32(float("-inf")), size=_VL)
    pto.vmi.vscatter(neg_inf, pool.as_ptr(), idx_u, mask)
    pto.pipe_barrier("V")
    val = pto.vmi.vload(val_ub.as_ptr(), off0, size=_VL)
    pto.vmi.vscatter(val, pool.as_ptr(), idx_u, mask)
    pto.pipe_barrier("V")
    got = pto.vmi.vgather(pool.as_ptr(), idx_u, mask)
    pto.vmi.vstore(got, out_ub.as_ptr(), off0, mask)
    pto.tile.store(out_ub, out_part)


if __name__ == "__main__":
    print(ub_table_scatter_gather_uint_offsets.compile().mlir_text()[:300])
