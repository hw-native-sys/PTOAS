# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""PTODSL TileLib template for pto.tdiv."""

from ptodsl import pto
import ptodsl.tilelib as tilelib

from ._elementwise import emit_binary_1d, emit_binary_2d, traversal_metadata
from .div_hp import _div_ieee754_f32_impl, _div_ieee754_f16_impl
from SoftOps import div_i32_soft


_DTYPES = [
    ("f16", "f16", "f16"),
    ("f32", "f32", "f32"),
    ("i32", "i32", "i32"),
]


def _emit_tdiv(src0, src1, dst, traversal):
    """Emit the operation-specific default or high-precision division."""

    dtype = dst.dtype
    precision_type = pto.get_op_attr("precisionType", "default")

    def divide(lhs, rhs, mask):
        if str(dtype) == "i32":
            return div_i32_soft(lhs, rhs, mask)
        if precision_type == "high_precision":
            if str(dtype) == "f32":
                return _div_ieee754_f32_impl(lhs, rhs, mask)
            return _div_ieee754_f16_impl(lhs, rhs, mask)
        return pto.vdiv(lhs, rhs, mask)

    if traversal == "1d":
        emit_binary_1d(src0, src1, dst, divide)
    else:
        emit_binary_2d(src0, src1, dst, divide)


def _register_tdiv(*, name, traversal):
    constraints = []
    loop_depth, priority, candidate_id = traversal_metadata(traversal)
    if traversal == "1d":
        constraints.append(
            tilelib.require_elementwise_1d("src0", "src1", "dst")
        )

    @tilelib.tile_template(
        op="pto.tdiv",
        target="a5",
        name=name,
        dtypes=_DTYPES,
        iteration_axis="none",
        op_engine="vector",
        op_class="elementwise",
        layouts=["row_major"],
        memory_spaces=["ub"],
        constraints=constraints,
        priority=priority,
        id=candidate_id,
        loop_depth=loop_depth,
        is_post_update=False,
    )
    def template(src0: pto.Tile, src1: pto.Tile, dst: pto.Tile):
        _emit_tdiv(src0, src1, dst, traversal)

    return template


template_tdiv = _register_tdiv(
    name="template_tdiv",
    traversal="2d",
)


template_tdiv_1d = _register_tdiv(
    name="template_tdiv_1d",
    traversal="1d",
)


from ._vmi_common import (  # noqa: E402
    FLOAT_DTYPES,
    _context_attr,
    _div_high_precision,
    _vdiv as _vmi_vdiv,
    canonical_vmi_template,
    emit_elementwise_vmi,
    sinkhorn_compact_elementwise_vmi_constraint,
)


@canonical_vmi_template(
    target="a5",
    op="tdiv",
    name="vmi_tdiv",
    dtypes=(("f16", "f16", "f16"), ("f32", "f32", "f32")),
    context_constraints={"precisionType": ("default", "high_precision")},
    min_row_bytes=128,
)
def vmi_tdiv(src0: pto.Tile, src1: pto.Tile, dst: pto.Tile):
    if _context_attr(src0, "precisionType", "default") == "high_precision":
        emit_elementwise_vmi(
            dst,
            (src0, src1),
            lambda values, mask: _div_high_precision(values[0], values[1], mask),
            allowed_dtypes=FLOAT_DTYPES,
        )
        return
    emit_elementwise_vmi(
        dst,
        (src0, src1),
        lambda values, mask: _vmi_vdiv(values[0], values[1], mask),
        allowed_dtypes=FLOAT_DTYPES,
    )


@canonical_vmi_template(
    target="a5",
    op="tdiv",
    name="vmi_tdiv_sinkhorn_compact",
    dtypes=(("f32", "f32", "f32"),),
    context_constraints={"precisionType": ("default",)},
    constraints=(sinkhorn_compact_elementwise_vmi_constraint,),
    requires_full_physical_row=False,
    tags=("supports_partial_valid_shape",),
)
def vmi_tdiv_sinkhorn_compact(src0: pto.Tile, src1: pto.Tile, dst: pto.Tile):
    emit_elementwise_vmi(
        dst,
        (src0, src1),
        lambda values, mask: _vmi_vdiv(values[0], values[1], mask),
        allowed_dtypes=FLOAT_DTYPES,
    )