# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""PTODSL TileLib templates for ``pto.tstore`` and ``pto.tstore_fp``."""

from ptodsl import pto
import ptodsl.tilelib as tilelib

from ._load_store import (
    ACC_STORE_DTYPES,
    LOAD_STORE_DTYPES,
    tstore_acc_nz2dn_constraint,
    tstore_acc_nz2nd_constraint,
    tstore_acc_nz2nz_constraint,
    tstore_dn_constraint,
    tstore_fp_constraint,
    tstore_nd_constraint,
    tstore_nz_constraint,
)


@tilelib.tile_template(
    op="pto.tstore",
    target="a5",
    name="template_tstore_nd",
    dtypes=LOAD_STORE_DTYPES,
    iteration_axis="none",
    op_engine="other",
    op_class="movement",
    constraints=[tstore_nd_constraint],
    id=0,
    loop_depth=3,
    is_post_update=False,
    tags=("store", "ub", "gm", "nd"),
)
def template_tstore_nd(src: pto.Tile, dst: pto.PartitionTensorView):
    elem_bytes = pto.bytewidth(src.dtype)
    if len(dst.shape) == 2:
        valid_rows, valid_cols = src.valid_shape
        _, ub_cols = src.shape
        row_stride, _ = dst.strides
        row_stride = valid_cols if row_stride is None else row_stride
        pto.mte_store(
            src.as_ptr(),
            dst.as_ptr(),
            valid_cols * elem_bytes,
            nburst=(valid_rows, ub_cols * elem_bytes, row_stride * elem_bytes),
        )
        return

    g0, g1, g2, g3, g4 = dst.shape
    s0, s1, s2, s3, s4 = dst.strides
    valid_rows, valid_cols = src.valid_shape
    _, ub_cols = src.shape

    n_burst = valid_rows if g0 == 1 and g1 == 1 and g2 == 1 and g3 is None else g3
    len_burst = valid_cols * elem_bytes
    ub_stride = ub_cols * elem_bytes
    gm_stride = 0 if g3 == 1 or s3 is None else s3 * elem_bytes

    src_stride2 = (valid_rows if g3 is None else g3) * ub_cols
    src_stride1 = g2 * src_stride2
    src_stride0 = g1 * src_stride1

    loops = []
    if g2 not in (1, None):
        loops.append((g2, src_stride2 * elem_bytes, s2 * elem_bytes))
    if g1 not in (1, None):
        loops.append((g1, src_stride1 * elem_bytes, s1 * elem_bytes))

    ub_ptr = src.as_ptr()
    gm_ptr = dst.as_ptr()
    if g0 == 1 and s0 is None:
        pto.mte_store(
            ub_ptr,
            gm_ptr,
            len_burst,
            nburst=(n_burst, ub_stride, gm_stride),
            loops=loops or None,
        )
    else:
        for i in range(0, g0, 1):
            pto.mte_store(
                pto.addptr(ub_ptr, i * src_stride0),
                pto.addptr(gm_ptr, i * s0),
                len_burst,
                nburst=(n_burst, ub_stride, gm_stride),
                loops=loops or None,
            )


@tilelib.tile_template(
    op="pto.tstore",
    target="a5",
    name="template_tstore_dn",
    dtypes=LOAD_STORE_DTYPES,
    iteration_axis="none",
    op_engine="other",
    op_class="movement",
    constraints=[tstore_dn_constraint],
    id=1,
    loop_depth=3,
    is_post_update=False,
    tags=("store", "ub", "gm", "dn"),
)
def template_tstore_dn(src: pto.Tile, dst: pto.PartitionTensorView):
    elem_bytes = pto.bytewidth(src.dtype)
    g0, g1, g2, g3, g4 = dst.shape
    s0, s1, s2, s3, s4 = dst.strides
    valid_rows, valid_cols = src.valid_shape
    ub_rows, _ = src.shape

    n_burst = valid_cols if g4 is None else g4
    len_burst = valid_rows * elem_bytes
    gm_stride = 0 if g4 == 1 or s4 is None else s4 * elem_bytes
    ub_stride = ub_rows * elem_bytes

    src_stride2 = ub_rows * n_burst
    src_stride1 = g2 * src_stride2
    src_stride0 = g1 * src_stride1

    loops = []
    if g2 not in (1, None):
        loops.append((g2, src_stride2 * elem_bytes, s2 * elem_bytes))
    if g1 not in (1, None):
        loops.append((g1, src_stride1 * elem_bytes, s1 * elem_bytes))

    ub_ptr = src.as_ptr()
    gm_ptr = dst.as_ptr()
    if g0 == 1 and s0 is None:
        pto.mte_store(
            ub_ptr,
            gm_ptr,
            len_burst,
            nburst=(n_burst, ub_stride, gm_stride),
            loops=loops or None,
        )
    else:
        for i in range(0, g0, 1):
            pto.mte_store(
                pto.addptr(ub_ptr, i * src_stride0),
                pto.addptr(gm_ptr, i * s0),
                len_burst,
                nburst=(n_burst, ub_stride, gm_stride),
                loops=loops or None,
            )


@tilelib.tile_template(
    op="pto.tstore",
    target="a5",
    name="template_tstore_nz",
    dtypes=LOAD_STORE_DTYPES,
    iteration_axis="none",
    op_engine="other",
    op_class="movement",
    constraints=[tstore_nz_constraint],
    id=2,
    loop_depth=1,
    is_post_update=False,
    tags=("store", "ub", "gm", "nz"),
)
def template_tstore_nz(src: pto.Tile, dst: pto.PartitionTensorView):
    elem_bytes = pto.bytewidth(src.dtype)
    g0, g1, g2, g3, g4 = dst.shape
    s0, s1, s2, s3, s4 = dst.strides
    valid_rows, _ = src.valid_shape
    ub_rows, _ = src.shape

    c0_size_bytes = 32
    n_burst = g1
    len_burst = valid_rows * c0_size_bytes
    gm_stride = s1 * elem_bytes
    ub_stride = ub_rows * c0_size_bytes
    tile_stride = g1 * ub_rows * g4

    ub_ptr = src.as_ptr()
    gm_ptr = dst.as_ptr()
    for i in range(0, g0, 1):
        pto.mte_store(
            pto.addptr(ub_ptr, i * tile_stride),
            pto.addptr(gm_ptr, i * s0),
            len_burst,
            nburst=(n_burst, ub_stride, gm_stride),
        )


@tilelib.tile_template(
    op="pto.tstore",
    target="a5",
    name="template_tstore_acc_to_gm_nz2nd",
    dtypes=ACC_STORE_DTYPES,
    iteration_axis="none",
    op_engine="other",
    op_class="movement",
    constraints=[tstore_acc_nz2nd_constraint],
    priority=1,
    id=3,
    loop_depth=0,
    is_post_update=False,
    tags=("store", "acc", "gm", "nz2nd"),
)
def template_tstore_acc_to_gm_nz2nd(src: pto.Tile, dst: pto.PartitionTensorView):
    m, n = src.valid_shape
    strides = dst.strides
    src_stride = src.shape[0]
    dst_stride = n if strides is None or strides[3] is None else strides[3]

    pto.mte_l0c_gm(
        src.as_ptr(),
        dst.as_ptr(),
        m,
        n,
        src_stride,
        dst_stride,
        0,
        0,
        layout="nz2nd",
    )


@tilelib.tile_template(
    op="pto.tstore",
    target="a5",
    name="template_tstore_acc_to_gm_nz2dn",
    dtypes=ACC_STORE_DTYPES,
    iteration_axis="none",
    op_engine="other",
    op_class="movement",
    constraints=[tstore_acc_nz2dn_constraint],
    priority=1,
    id=4,
    loop_depth=0,
    is_post_update=False,
    tags=("store", "acc", "gm", "nz2dn"),
)
def template_tstore_acc_to_gm_nz2dn(src: pto.Tile, dst: pto.PartitionTensorView):
    m, n = src.valid_shape
    strides = dst.strides
    src_stride = src.shape[0]
    dst_stride = m if strides is None or strides[4] is None else strides[4]
    loop0_src_stride = 1

    pto.mte_l0c_gm(
        src.as_ptr(),
        dst.as_ptr(),
        m,
        n,
        src_stride,
        dst_stride,
        0,
        0,
        layout=("nz2dn", loop0_src_stride),
    )


@tilelib.tile_template(
    op="pto.tstore",
    target="a5",
    name="template_tstore_acc_to_gm_nz2nz",
    dtypes=ACC_STORE_DTYPES,
    iteration_axis="none",
    op_engine="other",
    op_class="movement",
    constraints=[tstore_acc_nz2nz_constraint],
    priority=1,
    id=5,
    loop_depth=0,
    is_post_update=False,
    tags=("store", "acc", "gm", "nz2nz"),
)
def template_tstore_acc_to_gm_nz2nz(src: pto.Tile, dst: pto.PartitionTensorView):
    m, n = src.valid_shape
    src_stride = src.shape[0]
    dst_stride = n

    pto.mte_l0c_gm(
        src.as_ptr(),
        dst.as_ptr(),
        m,
        n,
        src_stride,
        dst_stride,
        0,
        0,
        layout=("nz2nz", 1),
    )


@tilelib.tile_template(
    op="pto.tstore_fp",
    target="a5",
    name="template_tstore_fp_acc_to_gm",
    dtypes=(("f32", "f16", "f16"), ("f32", "bf16", "bf16")),
    iteration_axis="none",
    op_engine="other",
    op_class="movement",
    constraints=[tstore_fp_constraint],
    id=0,
    loop_depth=0,
    is_post_update=False,
    tags=("store", "acc", "gm", "fp"),
)
def template_tstore_fp_acc_to_gm(src: pto.Tile, fp: pto.Tile, dst: pto.PartitionTensorView):
    m, n = src.valid_shape
    strides = dst.strides
    quant_mode = "qf322bf16_pre_vec" if str(fp.dtype) == "bf16" else "qf322f16_pre_vec"
    src_stride = src.shape[0]
    dst_stride = n if strides is None or strides[3] is None else strides[3]

    pto.mte_l0c_gm(
        src.as_ptr(),
        dst.as_ptr(),
        m,
        n,
        src_stride,
        dst_stride,
        0,
        0,
        layout="nz2nd",
        pre_quant=(fp.as_ptr(), quant_mode),
    )
