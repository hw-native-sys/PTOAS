# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""PTODSL TileLib template for pto.tcolexpandmul."""

from ptodsl import pto

from ._expand_binary import NUMERIC_SIGNATURES, register_column_expand_binary


template_tcolexpandmul = register_column_expand_binary(
    op="pto.tcolexpandmul",
    name="template_tcolexpandmul",
    vector_op=pto.vmul,
    dtypes=NUMERIC_SIGNATURES,
)


from ._vmi_common import (  # noqa: E402
    canonical_vmi_template,
    col_expand_binary_vmi_constraint,
    emit_col_expand_binary_vmi,
)


@canonical_vmi_template(
    target="a5",
    op="tcolexpandmul",
    name="vmi_tcolexpandmul",
    dtypes=(("f32", "f32", "f32"),),
    min_row_bytes=128,
    constraints=(col_expand_binary_vmi_constraint,),
)
def vmi_tcolexpandmul(src: pto.Tile, col_values: pto.Tile, dst: pto.Tile):
    emit_col_expand_binary_vmi(src, col_values, dst, binop="mul")
