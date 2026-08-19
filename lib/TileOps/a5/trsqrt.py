# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""PTODSL TileLib template for pto.trsqrt — default precision only."""

from ptodsl import pto

from ._elementwise import register_unary


_DTYPES = [
    ("f16", "f16"),
    ("f32", "f32"),
]


template_trsqrt = register_unary(
    op="pto.trsqrt",
    name="template_trsqrt",
    vector_op=pto.vrsqrt,
    dtypes=_DTYPES,
)


template_trsqrt_1d = register_unary(
    op="pto.trsqrt",
    name="template_trsqrt_1d",
    vector_op=pto.vrsqrt,
    dtypes=_DTYPES,
    traversal="1d",
)


from ._vmi_common import (  # noqa: E402
    _context_attr,
    canonical_vmi_template,
    emit_rsqrt_vmi,
)


@canonical_vmi_template(
    target="a5",
    op="trsqrt",
    name="vmi_trsqrt",
    dtypes=(("f16", "f16"), ("f32", "f32")),
    context_constraints={"precisionType": ("default",)},
)
def vmi_trsqrt(src: pto.Tile, dst: pto.Tile):
    emit_rsqrt_vmi(src, dst, high_precision=False)


@canonical_vmi_template(
    target="a5",
    op="trsqrt",
    name="vmi_trsqrt_with_tmp",
    dtypes=(
        ("f16", "f16", "f16"),
        ("f32", "f32", "f32"),
    ),
    context_constraints={"precisionType": ("default", "high_precision")},
)
def vmi_trsqrt_with_tmp(src: pto.Tile, dst: pto.Tile, tmp: pto.Tile):
    _ = tmp
    emit_rsqrt_vmi(
        src,
        dst,
        high_precision=_context_attr(src, "precisionType", "default")
        == "high_precision",
    )