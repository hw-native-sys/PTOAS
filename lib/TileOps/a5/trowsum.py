# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""PTODSL TileLib template for pto.trowsum."""

from ._row_reductions import register_rowsum


template_trowsum = register_rowsum()


from ._vmi_common import (  # noqa: E402
    Tile,
    canonical_vmi_template,
    emit_row_reduce_vmi,
    emit_row_reduce_streaming_vmi,
    row_reduce_vmi_constraint,
    row_reduce_streaming_vmi_constraint,
    sinkhorn_row_reduce_streaming_vmi_constraint,
)


@canonical_vmi_template(
    target="a5",
    op="trowsum",
    name="vmi_trowsum",
    requires_full_physical_row=False,
    dtypes=(
        ("f32", "f32", "f32"),
        ("i32", "i32", "i32"),
    ),
    constraints=(row_reduce_vmi_constraint,),
    tags=("grouped_rows", "supports_partial_valid_shape"),
    priority=101,
    single_logical_row_loop=False,
    resource_scope="tile",
    resource_vector_values=1,
)
def vmi_trowsum(src: Tile, workspace: Tile, dst: Tile):
    emit_row_reduce_vmi(src, workspace, dst, kind="sum")


@canonical_vmi_template(
    target="a5",
    op="trowsum",
    name="vmi_trowsum_row",
    requires_full_physical_row=False,
    dtypes=(
        ("f32", "f32", "f32"),
        ("i32", "i32", "i32"),
    ),
    constraints=(row_reduce_streaming_vmi_constraint,),
    tags=("row_streaming",),
    candidate_id=1001,
    resource_scope="row",
    resource_vector_values=1,
)
def vmi_trowsum_row(src: Tile, workspace: Tile, dst: Tile):
    emit_row_reduce_streaming_vmi(src, workspace, dst, kind="sum")


@canonical_vmi_template(
    target="a5",
    op="trowsum",
    name="vmi_trowsum_sinkhorn_row",
    requires_full_physical_row=False,
    dtypes=(
        ("f32", "f32", "f32"),
        ("i32", "i32", "i32"),
    ),
    constraints=(sinkhorn_row_reduce_streaming_vmi_constraint,),
    tags=("row_streaming", "supports_partial_valid_shape"),
    priority=102,
    candidate_id=1002,
    resource_scope="row",
    resource_vector_values=1,
)
def vmi_trowsum_sinkhorn_row(src: Tile, workspace: Tile, dst: Tile):
    emit_row_reduce_streaming_vmi(src, workspace, dst, kind="sum")
