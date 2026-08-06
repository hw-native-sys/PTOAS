# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""PTODSL TileLib template for pto.tlrelu."""

from ptodsl import pto, scalar
import ptodsl.tilelib as tilelib
from ptodsl._types import _resolve

from ._elementwise import (
    emit_scalar_binary_1d,
    emit_scalar_binary_2d,
    traversal_metadata,
)


def _ub_or_vec_row_major(operand_memory_spaces, operand_b_layouts, operand_s_layouts, **_):
    return (
        all(space in {"ub", "vec"} for space in operand_memory_spaces)
        and all(layout == "row_major" for layout in operand_b_layouts)
        and all(layout == "none_box" for layout in operand_s_layouts)
    )


_DTYPES = [
    ("f16", "f16", "f16"),
    ("f16", "f32", "f16"),
    ("f32", "f32", "f32"),
]


def _register_tlrelu(*, name, traversal):
    constraints = [
        _ub_or_vec_row_major,
        tilelib.require_same_valid_shape("src", "dst"),
    ]
    loop_depth, priority, candidate_id = traversal_metadata(traversal)
    if traversal == "1d":
        constraints.append(tilelib.require_elementwise_1d("src", "dst"))

    @tilelib.tile_template(
        op="pto.tlrelu",
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
    def template(src: pto.Tile, slope, dst: pto.Tile):
        dtype = dst.dtype
        slope_scalar = slope
        if str(dtype) == "f16":
            slope_scalar = scalar.coerce_scalar_to_type(
                slope,
                _resolve(pto.f16),
                context="template_tlrelu(slope)",
            )

        emitter = (
            emit_scalar_binary_1d
            if traversal == "1d"
            else emit_scalar_binary_2d
        )
        emitter(src, slope_scalar, dst, pto.vlrelu)

    return template


template_tlrelu = _register_tlrelu(
    name="template_tlrelu",
    traversal="2d",
)

template_tlrelu_1d = _register_tlrelu(
    name="template_tlrelu_1d",
    traversal="1d",
)
