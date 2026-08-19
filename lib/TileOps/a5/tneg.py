# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""PTODSL TileLib template for pto.tneg."""

from ptodsl import pto

from ._elementwise import register_unary


_DTYPES = [
    ("i8", "i8"),
    ("i16", "i16"),
    ("i32", "i32"),
    ("f16", "f16"),
    ("bf16", "bf16"),
    ("f32", "f32"),
]


template_tneg = register_unary(
    op="pto.tneg",
    name="template_tneg",
    vector_op=pto.vneg,
    dtypes=_DTYPES,
)


template_tneg_1d = register_unary(
    op="pto.tneg",
    name="template_tneg_1d",
    vector_op=pto.vneg,
    dtypes=_DTYPES,
    traversal="1d",
)


from ._vmi_common import (  # noqa: E402
    NUMERIC_DTYPES,
    _neg as _vmi_neg,
    canonical_vmi_template,
    emit_elementwise_vmi,
)


@canonical_vmi_template(
    target="a5",
    op="tneg",
    name="vmi_tneg",
    dtypes=(
        ("f32", "f32"),
        ("f16", "f16"),
        ("bf16", "bf16"),
        ("i8", "i8"),
        ("i16", "i16"),
        ("i32", "i32"),
    ),
)
def vmi_tneg(src: pto.Tile, dst: pto.Tile):
    # A5 tneg ODS rejects unsigned int (only i8/i16/i32/f16/bf16/f32); unsigned
    # tneg conservatively falls back to the ordinary PTODSL path.
    emit_elementwise_vmi(dst, (src,), _vmi_neg, allowed_dtypes=NUMERIC_DTYPES)
