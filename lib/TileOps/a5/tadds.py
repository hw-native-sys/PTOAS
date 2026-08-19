# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""PTODSL TileLib template for pto.tadds."""

from ptodsl import pto

from ._common import same_dtype_signatures
from ._elementwise import register_scalar_binary


_DTYPES = same_dtype_signatures(3)


template_tadds = register_scalar_binary(
    op="pto.tadds",
    name="template_tadds",
    vector_op=pto.vadds,
    dtypes=_DTYPES,
)

template_tadds_1d = register_scalar_binary(
    op="pto.tadds",
    name="template_tadds_1d",
    vector_op=pto.vadds,
    dtypes=_DTYPES,
    traversal="1d",
)


from ._vmi_common import (  # noqa: E402
    _vadds as _vmi_vadds,
    bf16,
    canonical_vmi_template,
    emit_elementwise_vmi,
    f16,
    f32,
    i16,
    i32,
    i8,
    sinkhorn_compact_elementwise_vmi_constraint,
)


@canonical_vmi_template(
    target="a5",
    op="tadds",
    name="vmi_tadds",
    dtypes=(("f32", "f32", "f32"),),
    min_row_bytes=128,
)
def vmi_tadds(src: pto.Tile, scalar: f32, dst: pto.Tile):
    emit_elementwise_vmi(
        dst,
        (src,),
        lambda values, mask: _vmi_vadds(values[0], scalar, mask),
        allowed_dtypes=(f32,),
    )


# Per-dtype vector-scalar candidates (texpand pattern). The tracing layer
# (_tile_template_tracing.py:525-534) binds a scalar parameter's dtype to its
# annotation, so each non-f32 dtype needs its own candidate function with a
# matching `scalar: <dtype>` annotation. ODS-validated dtypes for A5 tadds:
# i8/i16/i32/f16/bf16/f32 (unsigned rejected). See ADR-0003 PR2.


@canonical_vmi_template(
    target="a5",
    op="tadds",
    name="vmi_tadds_f16",
    dtypes=(("f16", "f16", "f16"),),
    min_row_bytes=128,
)
def vmi_tadds_f16(src: pto.Tile, scalar: f16, dst: pto.Tile):
    emit_elementwise_vmi(
        dst,
        (src,),
        lambda values, mask: _vmi_vadds(values[0], scalar, mask),
        allowed_dtypes=(f16,),
    )


@canonical_vmi_template(
    target="a5",
    op="tadds",
    name="vmi_tadds_bf16",
    dtypes=(("bf16", "bf16", "bf16"),),
    min_row_bytes=128,
)
def vmi_tadds_bf16(src: pto.Tile, scalar: bf16, dst: pto.Tile):
    emit_elementwise_vmi(
        dst,
        (src,),
        lambda values, mask: _vmi_vadds(values[0], scalar, mask),
        allowed_dtypes=(bf16,),
    )


@canonical_vmi_template(
    target="a5",
    op="tadds",
    name="vmi_tadds_i8",
    dtypes=(("i8", "i8", "i8"),),
    min_row_bytes=128,
)
def vmi_tadds_i8(src: pto.Tile, scalar: i8, dst: pto.Tile):
    emit_elementwise_vmi(
        dst,
        (src,),
        lambda values, mask: _vmi_vadds(values[0], scalar, mask),
        allowed_dtypes=(i8,),
    )


@canonical_vmi_template(
    target="a5",
    op="tadds",
    name="vmi_tadds_i16",
    dtypes=(("i16", "i16", "i16"),),
    min_row_bytes=128,
)
def vmi_tadds_i16(src: pto.Tile, scalar: i16, dst: pto.Tile):
    emit_elementwise_vmi(
        dst,
        (src,),
        lambda values, mask: _vmi_vadds(values[0], scalar, mask),
        allowed_dtypes=(i16,),
    )


@canonical_vmi_template(
    target="a5",
    op="tadds",
    name="vmi_tadds_i32",
    dtypes=(("i32", "i32", "i32"),),
    min_row_bytes=128,
)
def vmi_tadds_i32(src: pto.Tile, scalar: i32, dst: pto.Tile):
    emit_elementwise_vmi(
        dst,
        (src,),
        lambda values, mask: _vmi_vadds(values[0], scalar, mask),
        allowed_dtypes=(i32,),
    )


@canonical_vmi_template(
    target="a5",
    op="tadds",
    name="vmi_tadds_sinkhorn_compact",
    dtypes=(("f32", "f32", "f32"),),
    constraints=(sinkhorn_compact_elementwise_vmi_constraint,),
    requires_full_physical_row=False,
    tags=("supports_partial_valid_shape",),
)
def vmi_tadds_sinkhorn_compact(src: pto.Tile, scalar: f32, dst: pto.Tile):
    emit_elementwise_vmi(
        dst,
        (src,),
        lambda values, mask: _vmi_vadds(values[0], scalar, mask),
        allowed_dtypes=(f32,),
    )


# Sinkhorn-compact per-dtype (float-only: the Sinkhorn 8x8 compact form is a
# float-domain shape; int sinkhorn tadds is not a registered form).


@canonical_vmi_template(
    target="a5",
    op="tadds",
    name="vmi_tadds_sinkhorn_compact_f16",
    dtypes=(("f16", "f16", "f16"),),
    constraints=(sinkhorn_compact_elementwise_vmi_constraint,),
    requires_full_physical_row=False,
    tags=("supports_partial_valid_shape",),
)
def vmi_tadds_sinkhorn_compact_f16(src: pto.Tile, scalar: f16, dst: pto.Tile):
    emit_elementwise_vmi(
        dst,
        (src,),
        lambda values, mask: _vmi_vadds(values[0], scalar, mask),
        allowed_dtypes=(f16,),
    )


@canonical_vmi_template(
    target="a5",
    op="tadds",
    name="vmi_tadds_sinkhorn_compact_bf16",
    dtypes=(("bf16", "bf16", "bf16"),),
    constraints=(sinkhorn_compact_elementwise_vmi_constraint,),
    requires_full_physical_row=False,
    tags=("supports_partial_valid_shape",),
)
def vmi_tadds_sinkhorn_compact_bf16(src: pto.Tile, scalar: bf16, dst: pto.Tile):
    emit_elementwise_vmi(
        dst,
        (src,),
        lambda values, mask: _vmi_vadds(values[0], scalar, mask),
        allowed_dtypes=(bf16,),
    )
