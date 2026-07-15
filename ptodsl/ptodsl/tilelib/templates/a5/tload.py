# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""PTODSL TileLib templates for ``pto.tload``."""

from ptodsl import pto
import ptodsl.tilelib as tilelib

from ._load_store import (
    LOAD_STORE_DTYPES,
    MAT_LOAD_DTYPES,
    dma_pad_for,
    tload_dn2dn_constraint,
    tload_mat_dn2nz_constraint,
    tload_mat_nd2nz_constraint,
    tload_nd2nd_constraint,
    tload_nz2nz_constraint,
)


@tilelib.tile_template(
    op="pto.tload",
    target="a5",
    name="template_tload_nd2nd",
    dtypes=LOAD_STORE_DTYPES,
    iteration_axis="none",
    op_engine="other",
    op_class="movement",
    constraints=[tload_nd2nd_constraint],
    id=0,
    loop_depth=3,
    is_post_update=False,
    tags=("load", "gm", "ub", "nd"),
)
def template_tload_nd2nd(src: pto.PartitionTensorView, dst: pto.Tile):
    elem_bytes = pto.bytewidth(dst.dtype)
    if len(src.shape) == 2:
        _, ub_cols = dst.shape
        valid_rows, valid_cols = dst.valid_shape
        row_stride, _ = src.strides
        row_stride = valid_cols if row_stride is None else row_stride
        pto.mte_load(
            src.as_ptr(),
            dst.as_ptr(),
            0,
            valid_cols * elem_bytes,
            nburst=(valid_rows, row_stride * elem_bytes, ub_cols * elem_bytes),
            pad=dma_pad_for(dst),
        )
        return

    g0, g1, g2, g3, g4 = src.shape
    s0, s1, s2, s3, s4 = src.strides
    _, ub_cols = dst.shape
    valid_rows, valid_cols = dst.valid_shape

    n_burst = valid_rows if g0 == 1 and g1 == 1 and g2 == 1 and g3 is None else g3
    len_burst = (valid_cols if g4 is None else g4) * elem_bytes
    gm_stride = 0 if g3 == 1 or s3 is None else s3 * elem_bytes
    ub_stride = ub_cols * elem_bytes

    dst_stride2 = (valid_rows if g3 is None else g3) * ub_cols
    dst_stride1 = g2 * dst_stride2
    dst_stride0 = g1 * dst_stride1

    loops = []
    if g2 not in (1, None):
        loops.append((g2, s2 * elem_bytes, dst_stride2 * elem_bytes))
    if g1 not in (1, None):
        loops.append((g1, s1 * elem_bytes, dst_stride1 * elem_bytes))

    gm_ptr = src.as_ptr()
    ub_ptr = dst.as_ptr()
    if g0 == 1 and s0 is None:
        pto.mte_load(
            gm_ptr,
            ub_ptr,
            0,
            len_burst,
            nburst=(n_burst, gm_stride, ub_stride),
            loops=loops or None,
            pad=dma_pad_for(dst),
        )
    else:
        for i in range(0, g0, 1):
            pto.mte_load(
                pto.addptr(gm_ptr, i * s0),
                pto.addptr(ub_ptr, i * dst_stride0),
                0,
                len_burst,
                nburst=(n_burst, gm_stride, ub_stride),
                loops=loops or None,
                pad=dma_pad_for(dst),
            )


@tilelib.tile_template(
    op="pto.tload",
    target="a5",
    name="template_tload_dn2dn",
    dtypes=LOAD_STORE_DTYPES,
    iteration_axis="none",
    op_engine="other",
    op_class="movement",
    constraints=[tload_dn2dn_constraint],
    id=1,
    loop_depth=3,
    is_post_update=False,
    tags=("load", "gm", "ub", "dn"),
)
def template_tload_dn2dn(src: pto.PartitionTensorView, dst: pto.Tile):
    elem_bytes = pto.bytewidth(dst.dtype)
    if len(src.shape) == 2:
        tile_rows, _ = dst.shape
        valid_rows, valid_cols = dst.valid_shape
        _, col_stride = src.strides
        col_stride = valid_rows if col_stride is None else col_stride
        pto.mte_load(
            src.as_ptr(),
            dst.as_ptr(),
            0,
            valid_rows * elem_bytes,
            nburst=(valid_cols, col_stride * elem_bytes, tile_rows * elem_bytes),
            pad=dma_pad_for(dst),
        )
        return

    g0, g1, g2, g3, g4 = src.shape
    s0, s1, s2, s3, s4 = src.strides
    ub_rows, _ = dst.shape
    valid_rows, _ = dst.valid_shape

    n_burst = dst.valid_shape[1] if g4 is None else g4
    len_burst = valid_rows * elem_bytes
    gm_stride = 0 if g4 == 1 or s4 is None else s4 * elem_bytes
    ub_stride = ub_rows * elem_bytes

    dst_stride2 = ub_rows * n_burst
    dst_stride1 = g2 * dst_stride2
    dst_stride0 = g1 * dst_stride1

    loops = []
    if g2 not in (1, None):
        loops.append((g2, s2 * elem_bytes, dst_stride2 * elem_bytes))
    if g1 not in (1, None):
        loops.append((g1, s1 * elem_bytes, dst_stride1 * elem_bytes))

    gm_ptr = src.as_ptr()
    ub_ptr = dst.as_ptr()
    if g0 == 1 and s0 is None:
        pto.mte_load(
            gm_ptr,
            ub_ptr,
            0,
            len_burst,
            nburst=(n_burst, gm_stride, ub_stride),
            loops=loops or None,
            pad=dma_pad_for(dst),
        )
    else:
        for i in range(0, g0, 1):
            pto.mte_load(
                pto.addptr(gm_ptr, i * s0),
                pto.addptr(ub_ptr, i * dst_stride0),
                0,
                len_burst,
                nburst=(n_burst, gm_stride, ub_stride),
                loops=loops or None,
                pad=dma_pad_for(dst),
            )


@tilelib.tile_template(
    op="pto.tload",
    target="a5",
    name="template_tload_nz2nz",
    dtypes=LOAD_STORE_DTYPES,
    iteration_axis="none",
    op_engine="other",
    op_class="movement",
    constraints=[tload_nz2nz_constraint],
    id=2,
    loop_depth=1,
    is_post_update=False,
    tags=("load", "gm", "ub", "nz"),
)
def template_tload_nz2nz(src: pto.PartitionTensorView, dst: pto.Tile):
    elem_bytes = pto.bytewidth(dst.dtype)
    g0, g1, g2, g3, g4 = src.shape
    s0, s1, s2, s3, s4 = src.strides
    tile_rows, _ = dst.shape
    valid_rows, _ = dst.valid_shape

    c0_size_bytes = 32
    n_burst = g1
    len_burst = valid_rows * c0_size_bytes
    gm_stride = s1 * elem_bytes
    ub_stride = tile_rows * c0_size_bytes
    tile_stride = g1 * tile_rows * g4

    gm_ptr = src.as_ptr()
    ub_ptr = dst.as_ptr()
    for i in range(0, g0, 1):
        pto.mte_load(
            pto.addptr(gm_ptr, i * s0),
            pto.addptr(ub_ptr, i * tile_stride),
            0,
            len_burst,
            nburst=(n_burst, gm_stride, ub_stride),
            pad=dma_pad_for(dst),
        )


@tilelib.tile_template(
    op="pto.tload",
    target="a5",
    name="template_tload_gm_to_mat_nd2nz",
    dtypes=MAT_LOAD_DTYPES,
    iteration_axis="none",
    op_engine="other",
    op_class="movement",
    constraints=[tload_mat_nd2nz_constraint],
    id=3,
    loop_depth=0,
    is_post_update=False,
    tags=("load", "gm", "mat", "nd2nz"),
)
def template_tload_gm_to_mat_nd2nz(src: pto.PartitionTensorView, dst: pto.Tile):
    m, k = dst.valid_shape
    elem_bytes = pto.bytewidth(dst.dtype)
    src_inner_stride = k * elem_bytes
    if len(src.shape) == 5 and src.strides is not None and src.strides[3] is not None:
        src_inner_stride = src.strides[3] * elem_bytes

    pto.mte_gm_l1_frac(
        src.as_ptr(),
        dst.as_ptr(),
        "nd2nz",
        shape=(m, k),
        src_layout=(src_inner_stride,),
        dst_group=(1, 1, m, 0),
        ctrl=(0, False),
    )


@tilelib.tile_template(
    op="pto.tload",
    target="a5",
    name="template_tload_gm_to_mat_dn2nz",
    dtypes=MAT_LOAD_DTYPES,
    iteration_axis="none",
    op_engine="other",
    op_class="movement",
    constraints=[tload_mat_dn2nz_constraint],
    id=4,
    loop_depth=0,
    is_post_update=False,
    tags=("load", "gm", "mat", "dn2nz"),
)
def template_tload_gm_to_mat_dn2nz(src: pto.PartitionTensorView, dst: pto.Tile):
    m, k = dst.valid_shape
    elem_bytes = pto.bytewidth(dst.dtype)
    src_inner_stride = m * elem_bytes
    if len(src.shape) == 5 and src.strides is not None and src.strides[4] is not None:
        src_inner_stride = src.strides[4] * elem_bytes

    pto.mte_gm_l1_frac(
        src.as_ptr(),
        dst.as_ptr(),
        "dn2nz",
        shape=(k, m),
        src_layout=(src_inner_stride,),
        dst_group=(1, 1, k, 0),
        ctrl=(0, False),
    )
