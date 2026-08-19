# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""PTODSL TileLib template for pto.tmuls."""

from ptodsl import pto

from ._common import same_dtype_signatures
from ._elementwise import register_scalar_binary


_DTYPES = same_dtype_signatures(3)


template_tmuls = register_scalar_binary(
    op="pto.tmuls",
    name="template_tmuls",
    vector_op=pto.vmuls,
    dtypes=_DTYPES,
)

template_tmuls_1d = register_scalar_binary(
    op="pto.tmuls",
    name="template_tmuls_1d",
    vector_op=pto.vmuls,
    dtypes=_DTYPES,
    traversal="1d",
)


from ._vmi_common import (  # noqa: E402
    _vmuls as _vmi_vmuls,
    canonical_vmi_template,
    emit_elementwise_vmi,
    f16,
    f32,
    i16,
    i32,
    sinkhorn_compact_elementwise_vmi_constraint,
)


@canonical_vmi_template(
    target="a5",
    op="tmuls",
    name="vmi_tmuls",
    dtypes=(("f32", "f32", "f32"),),
    min_row_bytes=128,
)
def vmi_tmuls(src: pto.Tile, scale: f32, dst: pto.Tile):
    emit_elementwise_vmi(
        dst,
        (src,),
        lambda values, mask: _vmi_vmuls(values[0], scale, mask),
        allowed_dtypes=(f32,),
    )


# Per-dtype vector-scalar candidates (texpand pattern). A5 tmul ODS rejects
# i8/ui8/bf16 (only i16/i32/ui16/ui32/f16/f32), so tmuls covers f16/i16/i32.
# bf16/i8 tmuls conservatively falls back to the ordinary PTODSL path. See
# ADR-0003 PR2.


@canonical_vmi_template(
    target="a5",
    op="tmuls",
    name="vmi_tmuls_f16",
    dtypes=(("f16", "f16", "f16"),),
    min_row_bytes=128,
)
def vmi_tmuls_f16(src: pto.Tile, scale: f16, dst: pto.Tile):
    emit_elementwise_vmi(
        dst,
        (src,),
        lambda values, mask: _vmi_vmuls(values[0], scale, mask),
        allowed_dtypes=(f16,),
    )


@canonical_vmi_template(
    target="a5",
    op="tmuls",
    name="vmi_tmuls_i16",
    dtypes=(("i16", "i16", "i16"),),
    min_row_bytes=128,
)
def vmi_tmuls_i16(src: pto.Tile, scale: i16, dst: pto.Tile):
    emit_elementwise_vmi(
        dst,
        (src,),
        lambda values, mask: _vmi_vmuls(values[0], scale, mask),
        allowed_dtypes=(i16,),
    )


@canonical_vmi_template(
    target="a5",
    op="tmuls",
    name="vmi_tmuls_i32",
    dtypes=(("i32", "i32", "i32"),),
    min_row_bytes=128,
)
def vmi_tmuls_i32(src: pto.Tile, scale: i32, dst: pto.Tile):
    emit_elementwise_vmi(
        dst,
        (src,),
        lambda values, mask: _vmi_vmuls(values[0], scale, mask),
        allowed_dtypes=(i32,),
    )


@canonical_vmi_template(
    target="a5",
    op="tmuls",
    name="vmi_tmuls_sinkhorn_compact",
    dtypes=(("f32", "f32", "f32"),),
    constraints=(sinkhorn_compact_elementwise_vmi_constraint,),
    requires_full_physical_row=False,
    tags=("supports_partial_valid_shape",),
)
def vmi_tmuls_sinkhorn_compact(src: pto.Tile, scale: f32, dst: pto.Tile):
    emit_elementwise_vmi(
        dst,
        (src,),
        lambda values, mask: _vmi_vmuls(values[0], scale, mask),
        allowed_dtypes=(f32,),
    )


@canonical_vmi_template(
    target="a5",
    op="tmuls",
    name="vmi_tmuls_sinkhorn_compact_f16",
    dtypes=(("f16", "f16", "f16"),),
    constraints=(sinkhorn_compact_elementwise_vmi_constraint,),
    requires_full_physical_row=False,
    tags=("supports_partial_valid_shape",),
)
def vmi_tmuls_sinkhorn_compact_f16(src: pto.Tile, scale: f16, dst: pto.Tile):
    emit_elementwise_vmi(
        dst,
        (src,),
        lambda values, mask: _vmi_vmuls(values[0], scale, mask),
        allowed_dtypes=(f16,),
    )
