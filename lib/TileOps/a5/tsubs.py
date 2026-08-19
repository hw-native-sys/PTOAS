# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""PTODSL TileLib template for pto.tsubs."""

from ptodsl import pto

from ._common import same_dtype_signatures
from ._elementwise import register_scalar_binary


_DTYPES = same_dtype_signatures(3)


template_tsubs = register_scalar_binary(
    op="pto.tsubs",
    name="template_tsubs",
    vector_op=pto.vsub,
    broadcast_scalar=True,
    dtypes=_DTYPES,
)

template_tsubs_1d = register_scalar_binary(
    op="pto.tsubs",
    name="template_tsubs_1d",
    vector_op=pto.vsub,
    broadcast_scalar=True,
    dtypes=_DTYPES,
    traversal="1d",
)


from ._vmi_common import (  # noqa: E402
    _negate_scalar,
    _vadds as _vmi_vadds,
    bf16,
    canonical_vmi_template,
    emit_elementwise_vmi,
    f16,
    f32,
    i16,
    i32,
    i8,
)


@canonical_vmi_template(
    target="a5",
    op="tsubs",
    name="vmi_tsubs",
    dtypes=(("f32", "f32", "f32"),),
)
def vmi_tsubs(src: pto.Tile, scalar: f32, dst: pto.Tile):
    negated = _negate_scalar(scalar, dst._spec.dtype)
    emit_elementwise_vmi(
        dst,
        (src,),
        lambda values, mask: _vmi_vadds(values[0], negated, mask),
        allowed_dtypes=(f32,),
    )


# Per-dtype vector-scalar candidates (texpand pattern). A5 tsub ODS rejects bf16
# (only i8/i16/i32/ui8/ui16/ui32/f16/f32) — but vsub lowering handles bf16, so
# bf16 tsubs is included. See ADR-0003 PR2.


@canonical_vmi_template(
    target="a5",
    op="tsubs",
    name="vmi_tsubs_f16",
    dtypes=(("f16", "f16", "f16"),),
)
def vmi_tsubs_f16(src: pto.Tile, scalar: f16, dst: pto.Tile):
    negated = _negate_scalar(scalar, f16)
    emit_elementwise_vmi(
        dst,
        (src,),
        lambda values, mask: _vmi_vadds(values[0], negated, mask),
        allowed_dtypes=(f16,),
    )


@canonical_vmi_template(
    target="a5",
    op="tsubs",
    name="vmi_tsubs_bf16",
    dtypes=(("bf16", "bf16", "bf16"),),
)
def vmi_tsubs_bf16(src: pto.Tile, scalar: bf16, dst: pto.Tile):
    negated = _negate_scalar(scalar, bf16)
    emit_elementwise_vmi(
        dst,
        (src,),
        lambda values, mask: _vmi_vadds(values[0], negated, mask),
        allowed_dtypes=(bf16,),
    )


@canonical_vmi_template(
    target="a5",
    op="tsubs",
    name="vmi_tsubs_i8",
    dtypes=(("i8", "i8", "i8"),),
)
def vmi_tsubs_i8(src: pto.Tile, scalar: i8, dst: pto.Tile):
    negated = _negate_scalar(scalar, i8)
    emit_elementwise_vmi(
        dst,
        (src,),
        lambda values, mask: _vmi_vadds(values[0], negated, mask),
        allowed_dtypes=(i8,),
    )


@canonical_vmi_template(
    target="a5",
    op="tsubs",
    name="vmi_tsubs_i16",
    dtypes=(("i16", "i16", "i16"),),
)
def vmi_tsubs_i16(src: pto.Tile, scalar: i16, dst: pto.Tile):
    negated = _negate_scalar(scalar, i16)
    emit_elementwise_vmi(
        dst,
        (src,),
        lambda values, mask: _vmi_vadds(values[0], negated, mask),
        allowed_dtypes=(i16,),
    )


@canonical_vmi_template(
    target="a5",
    op="tsubs",
    name="vmi_tsubs_i32",
    dtypes=(("i32", "i32", "i32"),),
)
def vmi_tsubs_i32(src: pto.Tile, scalar: i32, dst: pto.Tile):
    negated = _negate_scalar(scalar, i32)
    emit_elementwise_vmi(
        dst,
        (src,),
        lambda values, mask: _vmi_vadds(values[0], negated, mask),
        allowed_dtypes=(i32,),
    )