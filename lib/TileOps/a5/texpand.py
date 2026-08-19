# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""PTODSL TileLib template for pto.texpands."""

from ptodsl import pto

from ._elementwise import register_scalar_fill


_DTYPES = [
    ("i8", "i8"),
    ("i16", "i16"),
    ("i32", "i32"),
    ("f16", "f16"),
    ("bf16", "bf16"),
    ("f32", "f32"),
]


template_texpands = register_scalar_fill(
    op="pto.texpands",
    name="template_texpands",
    dtypes=_DTYPES,
)

template_texpands_1d = register_scalar_fill(
    op="pto.texpands",
    name="template_texpands_1d",
    dtypes=_DTYPES,
    traversal="1d",
)


from ._vmi_common import (  # noqa: E402
    bf16,
    canonical_vmi_template,
    emit_scalar_fill_vmi,
    f16,
    f32,
    i32,
)


@canonical_vmi_template(
    target="a5",
    op="texpands",
    name="vmi_texpands",
    dtypes=(("f32", "f32"),),
    min_row_bytes=128,
)
def vmi_texpands(scalar: f32, dst: pto.Tile):
    emit_scalar_fill_vmi(scalar, dst)


@canonical_vmi_template(
    target="a5",
    op="texpands",
    name="vmi_texpands_i32",
    dtypes=(("i32", "i32"),),
)
def vmi_texpands_i32(scalar: i32, dst: pto.Tile):
    emit_scalar_fill_vmi(scalar, dst, allowed_dtypes=(i32,))


@canonical_vmi_template(
    target="a5",
    op="texpands",
    name="vmi_texpands_f16",
    dtypes=(("f16", "f16"),),
)
def vmi_texpands_f16(scalar: f16, dst: pto.Tile):
    emit_scalar_fill_vmi(scalar, dst, allowed_dtypes=(f16,))


@canonical_vmi_template(
    target="a5",
    op="texpands",
    name="vmi_texpands_bf16",
    dtypes=(("bf16", "bf16"),),
)
def vmi_texpands_bf16(scalar: bf16, dst: pto.Tile):
    emit_scalar_fill_vmi(scalar, dst, allowed_dtypes=(bf16,))
