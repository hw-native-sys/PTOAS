# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""PTODSL TileLib template for pto.tdivs."""

from ptodsl import pto
import ptodsl.tilelib as tilelib

from ._elementwise import (
    _common_constraints,
    emit_scalar_binary_1d,
    emit_scalar_binary_2d,
    traversal_metadata,
)
from .div_hp import _div_ieee754_f32_impl, _div_ieee754_f16_impl
from SoftOps import div_i32_soft


_DTYPES = [
    ("f16", "f16", "f16"),
    ("f32", "f32", "f32"),
    ("i32", "i32", "i32"),
]


def _tile_scalar_tile(operand_kinds=(), **_):
    return operand_kinds == ("tile", "scalar", "tile")


def _scalar_tile_tile(operand_kinds=(), **_):
    return operand_kinds == ("scalar", "tile", "tile")


def _div(lhs, rhs, dtype, mask, precision_type):
    if str(dtype) == "i32":
        return div_i32_soft(lhs, rhs, mask)
    if precision_type == "high_precision":
        if str(dtype) == "f32":
            return _div_ieee754_f32_impl(lhs, rhs, mask)
        return _div_ieee754_f16_impl(lhs, rhs, mask)
    return pto.vdiv(lhs, rhs, mask)


def _emit_tdivs_body(src, scalar, dst, traversal, *, scalar_lhs=False):
    dtype = dst.dtype
    precision_type = pto.get_op_attr("precisionType", "default")

    def divide(lhs, rhs, mask):
        return _div(lhs, rhs, dtype, mask, precision_type)

    emitter = (
        emit_scalar_binary_1d
        if traversal == "1d"
        else emit_scalar_binary_2d
    )
    emitter(
        src,
        scalar,
        dst,
        divide,
        broadcast_scalar=True,
        scalar_lhs=scalar_lhs,
    )


def _register_tdivs(*, name, traversal, scalar_lhs=False):
    constraints = _common_constraints("src", "dst")
    constraints.append(_scalar_tile_tile if scalar_lhs else _tile_scalar_tile)
    loop_depth, priority, candidate_id = traversal_metadata(
        traversal,
        fallback_candidate_id=1 if scalar_lhs else 0,
        candidate_count=2,
    )
    if traversal == "1d":
        constraints.append(tilelib.require_elementwise_1d("src", "dst"))

    if scalar_lhs:

        @tilelib.tile_template(
            op="pto.tdivs",
            target="a5",
            name=name,
            dtypes=_DTYPES,
            iteration_axis="none",
            op_engine="vector",
            op_class="elementwise",
            constraints=constraints,
            priority=priority,
            id=candidate_id,
            loop_depth=loop_depth,
            is_post_update=False,
            tags=("elementwise", "scalar"),
        )
        def template(scalar, src: pto.Tile, dst: pto.Tile):
            _emit_tdivs_body(
                src,
                scalar,
                dst,
                traversal,
                scalar_lhs=True,
            )

        return template

    @tilelib.tile_template(
        op="pto.tdivs",
        target="a5",
        name=name,
        dtypes=_DTYPES,
        iteration_axis="none",
        op_engine="vector",
        op_class="elementwise",
        constraints=constraints,
        priority=priority,
        id=candidate_id,
        loop_depth=loop_depth,
        is_post_update=False,
        tags=("elementwise", "scalar"),
    )
    def template(src: pto.Tile, scalar, dst: pto.Tile):
        _emit_tdivs_body(src, scalar, dst, traversal)

    return template


template_tdivs_tile_scalar = _register_tdivs(
    name="template_tdivs_tile_scalar",
    traversal="2d",
)

template_tdivs_scalar_tile = _register_tdivs(
    name="template_tdivs_scalar_tile",
    traversal="2d",
    scalar_lhs=True,
)

template_tdivs_tile_scalar_1d = _register_tdivs(
    name="template_tdivs_tile_scalar_1d",
    traversal="1d",
)

template_tdivs_scalar_tile_1d = _register_tdivs(
    name="template_tdivs_scalar_tile_1d",
    traversal="1d",
    scalar_lhs=True,
)


from ._vmi_common import (  # noqa: E402
    FLOAT_DTYPES,
    _context_attr,
    _divide_by_scalar,
    _divide_by_scalar_high_precision,
    _divide_scalar_by_vector,
    _divide_scalar_by_vector_high_precision,
    _operand_kinds_are,
    canonical_vmi_template,
    emit_elementwise_vmi,
    f32,
)


@canonical_vmi_template(
    target="a5",
    op="tdivs",
    name="vmi_tdivs",
    dtypes=(("f32", "f32", "f32"),),
    context_constraints={"precisionType": ("default", "high_precision")},
    constraints=(_operand_kinds_are(("tile", "scalar", "tile")),),
)
def vmi_tdivs(src: pto.Tile, scalar: f32, dst: pto.Tile):
    if _context_attr(src, "precisionType", "default") == "high_precision":
        emit_elementwise_vmi(
            dst,
            (src,),
            lambda values, mask: _divide_by_scalar_high_precision(
                values[0], scalar, mask
            ),
            allowed_dtypes=FLOAT_DTYPES,
        )
        return
    emit_elementwise_vmi(
        dst,
        (src,),
        lambda values, mask: _divide_by_scalar(values[0], scalar, mask),
        allowed_dtypes=FLOAT_DTYPES,
    )


@canonical_vmi_template(
    target="a5",
    op="tdivs",
    name="vmi_tdivs_scalar_tile",
    dtypes=(("f32", "f32", "f32"),),
    context_constraints={"precisionType": ("default", "high_precision")},
    constraints=(_operand_kinds_are(("scalar", "tile", "tile")),),
)
def vmi_tdivs_scalar_tile(scalar: f32, src: pto.Tile, dst: pto.Tile):
    if _context_attr(src, "precisionType", "default") == "high_precision":
        emit_elementwise_vmi(
            dst,
            (src,),
            lambda values, mask: _divide_scalar_by_vector_high_precision(
                scalar, values[0], mask
            ),
            allowed_dtypes=FLOAT_DTYPES,
        )
        return
    emit_elementwise_vmi(
        dst,
        (src,),
        lambda values, mask: _divide_scalar_by_vector(scalar, values[0], mask),
        allowed_dtypes=FLOAT_DTYPES,
    )