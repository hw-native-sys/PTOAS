# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""PTODSL TileLib template for pto.tcolexpanddiv — default precision only."""

from ptodsl import pto
import ptodsl.tilelib as tilelib

from ._expand_binary import _ub_or_vec_row_major, _valid_column_expand_binary, register_column_expand_binary
from SoftOps import div_i32_soft


def _divide_i16(lhs, rhs, mask, f32_mask):
    lhs_even = pto.vcvt(lhs, pto.f32, mask, part=pto.VcvtPartMode.EVEN)
    rhs_even = pto.vcvt(rhs, pto.f32, mask, part=pto.VcvtPartMode.EVEN)
    divided_even = pto.vdiv(lhs_even, rhs_even, f32_mask)
    result_even = pto.vcvt(
        divided_even,
        pto.i16,
        f32_mask,
        rnd=pto.VcvtRoundMode.Z,
        sat=pto.VcvtSatMode.NOSAT,
        part=pto.VcvtPartMode.EVEN,
    )

    lhs_odd = pto.vcvt(lhs, pto.f32, mask, part=pto.VcvtPartMode.ODD)
    rhs_odd = pto.vcvt(rhs, pto.f32, mask, part=pto.VcvtPartMode.ODD)
    divided_odd = pto.vdiv(lhs_odd, rhs_odd, f32_mask)
    result_odd = pto.vcvt(
        divided_odd,
        pto.i16,
        f32_mask,
        rnd=pto.VcvtRoundMode.Z,
        sat=pto.VcvtSatMode.NOSAT,
        part=pto.VcvtPartMode.ODD,
    )
    return pto.vor(result_even, result_odd, mask)


template_tcolexpanddiv = register_column_expand_binary(
    op="pto.tcolexpanddiv",
    name="template_tcolexpanddiv",
    vector_op=pto.vdiv,
    dtypes=[
        ("f16", "f16", "f16"),
        ("f32", "f32", "f32"),
    ],
)


@tilelib.tile_template(
    op="pto.tcolexpanddiv",
    target="a5",
    name="template_tcolexpanddiv_i32",
    dtypes=[("i16", "i16", "i16"), ("i32", "i32", "i32")],
    iteration_axis="column",
    op_engine="vector",
    op_class="broadcast",
    constraints=[
        _ub_or_vec_row_major,
        _valid_column_expand_binary,
    ],
    id=1,
    loop_depth=2,
    is_post_update=False,
    tags=("column_expand", "binary", "integer-div"),
)
def template_tcolexpanddiv_i32(src0: pto.Tile, src1: pto.Tile, dst: pto.Tile):
    valid_rows, valid_cols = dst.valid_shape
    dtype = dst.dtype
    lanes = pto.elements_per_vreg(dtype)

    with pto.for_(0, valid_rows, step=1) as row:
        col_loop = pto.for_(0, valid_cols, step=lanes).carry(remained=valid_cols)
        with col_loop:
            col = col_loop.iv
            mask, remained = pto.make_mask(dtype, col_loop.remained)
            lhs = pto.vlds(src0[row, col:])
            rhs = pto.vlds(src1[0, col:])
            if str(dtype) == "i16":
                f32_remained = (col_loop.remained + 1) // 2
                f32_mask, _ = pto.make_mask(pto.f32, f32_remained)
                result = _divide_i16(lhs, rhs, mask, f32_mask)
            else:
                result = div_i32_soft(lhs, rhs, mask)
            pto.vsts(result, dst[row, col:], mask)
            col_loop.update(remained=remained)
