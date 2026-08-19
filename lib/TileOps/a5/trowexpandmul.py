# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""PTODSL TileLib template for pto.trowexpandmul."""

from ptodsl import pto

from ._expand_binary import NUMERIC_SIGNATURES, register_row_expand_binary


template_trowexpandmul = register_row_expand_binary(
    op="pto.trowexpandmul",
    name="template_trowexpandmul",
    vector_op=pto.vmul,
    dtypes=NUMERIC_SIGNATURES,
)


from ._vmi_common import (  # noqa: E402
    canonical_vmi_template,
    emit_row_expand_binary_vmi,
    row_expand_binary_vmi_constraint,
    sinkhorn_row_expand_vmi_constraint,
)


@canonical_vmi_template(
    target="a5",
    op="trowexpandmul",
    name="vmi_trowexpandmul",
    dtypes=(("f32", "f32", "f32"),),
    constraints=(row_expand_binary_vmi_constraint,),
    min_row_bytes=128,
)
def vmi_trowexpandmul(src: pto.Tile, row_values: pto.Tile, dst: pto.Tile):
    emit_row_expand_binary_vmi(src, row_values, dst, "mul")


@canonical_vmi_template(
    target="a5",
    op="trowexpandmul",
    name="vmi_trowexpandmul_sinkhorn_row_loop",
    dtypes=(("f32", "f32", "f32"),),
    constraints=(sinkhorn_row_expand_vmi_constraint,),
    requires_full_physical_row=False,
    tags=("supports_partial_valid_shape",),
)
def vmi_trowexpandmul_sinkhorn_row_loop(
    src: pto.Tile, row_values: pto.Tile, dst: pto.Tile
):
    emit_row_expand_binary_vmi(src, row_values, dst, "mul")
