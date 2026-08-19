# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""PTODSL TileLib template for the basic pto.tcolsum form."""

from ptodsl import pto

from ._reductions import register_column_reduction


template_tcolsum = register_column_reduction(
    op="pto.tcolsum",
    name="template_tcolsum",
    vector_op=pto.vadd,
    dtypes=[
        ("i8", "i8"),
        ("i16", "i16"),
        ("i32", "i32"),
        ("f16", "f16"),
        ("bf16", "bf16"),
        ("f32", "f32"),
    ],
)


from ._vmi_common import (  # noqa: E402
    canonical_vmi_template,
    col_reduce_vmi_constraint,
    emit_col_reduce_vmi,
)


@canonical_vmi_template(
    target="a5",
    op="tcolsum",
    name="vmi_tcolsum",
    # Signed-int + float: the elementwise vadd lowering handles signed int
    # correctly, so tcolsum supports i8/i16/i32 in addition to f16/bf16/f32.
    # Unsigned int fails the VMI vreg-type validation (`unsupported VMI
    # tile-type`); unsigned tcolsum conservatively falls back to the ordinary
    # PTODSL path.
    dtypes=(
        ("f32", "f32"),
        ("f16", "f16"),
        ("bf16", "bf16"),
        ("i8", "i8"),
        ("i16", "i16"),
        ("i32", "i32"),
    ),
    constraints=(col_reduce_vmi_constraint,),
)
def vmi_tcolsum(src: pto.Tile, dst: pto.Tile):
    emit_col_reduce_vmi(src, dst, kind="add")
