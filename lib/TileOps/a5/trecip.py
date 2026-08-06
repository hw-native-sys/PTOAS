# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""PTODSL TileLib template for pto.trecip — default precision only."""

from ptodsl import pto
import ptodsl.tilelib as tilelib

from ._elementwise import emit_unary_1d, emit_unary_2d, traversal_metadata


_DTYPES = [
    ("f16", "f16"),
    ("f32", "f32"),
]


def _base_constraints():
    return [
        tilelib.check_memory_space("ub"),
        tilelib.check_layout("row_major"),
        tilelib.check_s_layout("none_box"),
    ]


def _emit_trecip(src, dst, traversal):
    """Emit the operation-specific reciprocal computation."""

    dtype = dst.dtype
    one_scalar = pto.f16(1.0) if str(dtype) == "f16" else pto.f32(1.0)

    def reciprocal(value, mask):
        one = pto.vbr(one_scalar)
        return pto.vdiv(one, value, mask)

    if traversal == "1d":
        emit_unary_1d(src, dst, reciprocal)
    else:
        emit_unary_2d(src, dst, reciprocal)


def _register_trecip(*, name, traversal):
    constraints = _base_constraints()
    loop_depth, priority, candidate_id = traversal_metadata(traversal)
    if traversal == "1d":
        constraints.append(tilelib.require_elementwise_1d("src", "dst"))

    @tilelib.tile_template(
        op="pto.trecip",
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
        tags=("elementwise", "reciprocal"),
    )
    def template(src: pto.Tile, dst: pto.Tile):
        _emit_trecip(src, dst, traversal)

    return template


template_trecip = _register_trecip(
    name="template_trecip",
    traversal="2d",
)


template_trecip_1d = _register_trecip(
    name="template_trecip_1d",
    traversal="1d",
)
