# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""PTODSL TileLib template for pto.tcolmin."""

from ptodsl import pto

from ._reductions import register_column_reduction


template_tcolmin = register_column_reduction(
    op="pto.tcolmin",
    name="template_tcolmin",
    vector_op=pto.vmin,
    dtypes=[
        ("i8", "i8"),
        ("i16", "i16"),
        ("i32", "i32"),
        ("ui8", "ui8"),
        ("ui16", "ui16"),
        ("ui32", "ui32"),
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
    op="tcolmin",
    name="vmi_tcolmin",
    # Float-only: see vmi_tcolmax — the elementwise vmin lowering rewrites
    # signed-int vmin to `pto.vmi.minf` (float-only), so int tcolmin fails at
    # VMI lowering. Int tcolmin conservatively falls back to the ordinary
    # PTODSL path until the vmin→mini lowering is fixed (C++ side, out of
    # ADR-0003 scope).
    dtypes=(
        ("f32", "f32"),
        ("f16", "f16"),
        ("bf16", "bf16"),
    ),
    constraints=(col_reduce_vmi_constraint,),
)
def vmi_tcolmin(src: pto.Tile, dst: pto.Tile):
    emit_col_reduce_vmi(src, dst, kind="min", split=4)
