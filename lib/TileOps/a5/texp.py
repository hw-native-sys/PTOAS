# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""PTODSL TileLib template for pto.texp — default precision only."""

from ptodsl import pto

from ._elementwise import register_unary


_DTYPES = [
    ("f16", "f16"),
    ("f32", "f32"),
]


template_texp = register_unary(
    op="pto.texp",
    name="template_texp",
    vector_op=pto.vexp,
    dtypes=_DTYPES,
)


template_texp_1d = register_unary(
    op="pto.texp",
    name="template_texp_1d",
    vector_op=pto.vexp,
    dtypes=_DTYPES,
    traversal="1d",
)


from ._vmi_common import (  # noqa: E402
    _exp as _vmi_exp,
    canonical_vmi_template,
    emit_elementwise_vmi,
    f16,
    f32,
    sinkhorn_compact_elementwise_vmi_constraint,
)


@canonical_vmi_template(
    target="a5",
    op="texp",
    name="vmi_texp_block64",
    dtypes=(
        ("f32", "f32"),
        ("f16", "f16"),
    ),
    context_constraints={"precisionType": ("default",)},
    min_row_bytes=128,
)
def vmi_texp_block64(src: pto.Tile, dst: pto.Tile):
    # A5 texp ODS (TExpOp::verify) accepts only f16/f32, not bf16. bf16 texp
    # conservatively falls back to the ordinary PTODSL path per ADR-0003.
    emit_elementwise_vmi(dst, (src,), _vmi_exp, allowed_dtypes=(f32, f16))


@canonical_vmi_template(
    target="a5",
    op="texp",
    name="vmi_texp_sinkhorn_compact",
    dtypes=(
        ("f32", "f32"),
        ("f16", "f16"),
    ),
    context_constraints={"precisionType": ("default",)},
    constraints=(sinkhorn_compact_elementwise_vmi_constraint,),
    requires_full_physical_row=False,
    tags=("supports_partial_valid_shape",),
)
def vmi_texp_sinkhorn_compact(src: pto.Tile, dst: pto.Tile):
    emit_elementwise_vmi(dst, (src,), _vmi_exp, allowed_dtypes=(f32, f16))
