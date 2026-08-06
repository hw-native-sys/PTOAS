# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""PTODSL TileLib template for pto.tlog."""

from ptodsl import pto
import ptodsl.tilelib as tilelib

from ._elementwise import (
    _common_constraints,
    emit_unary_1d,
    emit_unary_2d,
    register_unary,
    traversal_metadata,
)


_DTYPES = [
    ("f16", "f16"),
    ("f32", "f32"),
]


def _is_default_precision(precisionType="default", **_):
    return precisionType != "high_precision"


def _is_high_precision(precisionType="default", **_):
    return precisionType == "high_precision"


template_tlog = register_unary(
    op="pto.tlog",
    name="template_tlog",
    vector_op=pto.vln,
    dtypes=_DTYPES,
    constraints=[_is_default_precision],
)


template_tlog_1d = register_unary(
    op="pto.tlog",
    name="template_tlog_1d",
    vector_op=pto.vln,
    dtypes=_DTYPES,
    constraints=[_is_default_precision],
    traversal="1d",
    candidate_id=2,
)


def _emit_tlog_high_precision(src, dst, traversal):
    """Emit the operation-specific high-precision logarithm computation."""

    dtype = dst.dtype
    if str(dtype) == "f16":
        subnormal_threshold = pto.f16("0x03FF")
        mul_factor = pto.f16("0x6400")
        compensation = pto.f16(-6.931471805599453094172)
    else:
        subnormal_threshold = pto.f32("0x007FFFFF")
        mul_factor = pto.f32("0x4B000000")
        compensation = pto.f32(-15.9423851528787421)

    def high_precision_log(value, mask):
        cmp_mask = pto.vcmps(
            value,
            subnormal_threshold,
            mask,
            pto.CmpMode.LT,
        )
        scaled = pto.vmuls(value, mul_factor, mask)
        selected_input = pto.vsel(scaled, value, cmp_mask)
        log_result = pto.vln(selected_input, mask)
        compensated = pto.vadds(log_result, compensation, mask)
        return pto.vsel(compensated, log_result, cmp_mask)

    if traversal == "1d":
        emit_unary_1d(src, dst, high_precision_log)
    else:
        emit_unary_2d(src, dst, high_precision_log)


def _register_tlog_high_precision(
    *,
    name,
    traversal,
):
    constraints = _common_constraints("src", "dst") + [_is_high_precision]
    loop_depth, priority, candidate_id = traversal_metadata(
        traversal,
        fallback_candidate_id=1,
        candidate_count=2,
    )
    if traversal == "1d":
        constraints.append(tilelib.require_elementwise_1d("src", "dst"))

    @tilelib.tile_template(
        op="pto.tlog",
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
        tags=("elementwise", "unary"),
    )
    def template(src: pto.Tile, dst: pto.Tile):
        _emit_tlog_high_precision(src, dst, traversal)

    return template


template_tlog_high_precision = _register_tlog_high_precision(
    name="template_tlog_high_precision",
    traversal="2d",
)


template_tlog_high_precision_1d = _register_tlog_high_precision(
    name="template_tlog_high_precision_1d",
    traversal="1d",
)
