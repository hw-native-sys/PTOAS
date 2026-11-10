# coding=utf-8
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""PTODSL TileLib template for pto.tcolexpanddiv.

Every ``DivAlgo`` selector is honoured through the shared dispatch in
``TileOps.a5.elementwise.div_dispatch``, exactly like ``pto.tdiv``; the
compensated sequences are pointwise, so no scratch operand is needed.
"""

from ptodsl import pto
import ptodsl.tilelib as tilelib

from ._expand_binary import (
    _emit_column_expand_integer_div,
    _ub_or_vec_row_major,
    _valid_column_expand_binary,
    register_column_expand_binary,
    UNSIGNED_SIGNATURES,
)
from SoftOps import div_i32_soft


def _divide_i16(lhs, rhs, mask, even_mask, odd_mask):
    lhs_even = pto.vcvt(lhs, pto.f32, mask, part=pto.VcvtPartMode.EVEN)
    rhs_even = pto.vcvt(rhs, pto.f32, mask, part=pto.VcvtPartMode.EVEN)
    divided_even = pto.vdiv(lhs_even, rhs_even, even_mask)
    result_even = pto.vcvt(
        divided_even,
        pto.i16,
        even_mask,
        rnd=pto.VcvtRoundMode.Z,
        sat=pto.VcvtSatMode.NOSAT,
        part=pto.VcvtPartMode.EVEN,
    )

    lhs_odd = pto.vcvt(lhs, pto.f32, mask, part=pto.VcvtPartMode.ODD)
    rhs_odd = pto.vcvt(rhs, pto.f32, mask, part=pto.VcvtPartMode.ODD)
    divided_odd = pto.vdiv(lhs_odd, rhs_odd, odd_mask)
    result_odd = pto.vcvt(
        divided_odd,
        pto.i16,
        odd_mask,
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
        ("bf16", "bf16", "bf16"),
        ("f32", "f32", "f32"),
    ] + UNSIGNED_SIGNATURES,
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
    _emit_column_expand_integer_div(src0, src1, dst, _divide_i16, div_i32_soft)
