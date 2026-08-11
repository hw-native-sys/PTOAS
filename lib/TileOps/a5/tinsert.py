# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""PTODSL TileLib templates for ``pto.tinsert``."""

from ptodsl import pto
import ptodsl.tilelib as tilelib

from ._common import NUMERIC_DTYPES


BLOCK_BYTE_SIZE = 32


def _acc_to_mat(src_kind, dst_kind, src_memory_space, dst_memory_space, src_config, dst_config, **_):
    return (
        src_kind == "tile"
        and dst_kind == "tile"
        and src_memory_space == "acc"
        and dst_memory_space == "mat"
        and src_config.b_layout == "col_major"
        and src_config.s_layout == "row_major"
        and dst_config.b_layout == "col_major"
        and dst_config.s_layout == "row_major"
    )


def _acc_to_vec_nd(src_kind, dst_kind, src_memory_space, dst_memory_space, dst_config, **_):
    return (
        src_kind == "tile"
        and dst_kind == "tile"
        and src_memory_space == "acc"
        and dst_memory_space == "ub"
        and dst_config.b_layout == "row_major"
        and dst_config.s_layout == "none_box"
    )


def _acc_to_vec_nz(src_kind, dst_kind, src_memory_space, dst_memory_space, dst_config, **_):
    return (
        src_kind == "tile"
        and dst_kind == "tile"
        and src_memory_space == "acc"
        and dst_memory_space == "ub"
        and dst_config.b_layout == "col_major"
        and dst_config.s_layout == "row_major"
    )


def _vec_to_vec_nd(src_memory_space, dst_memory_space, src_config, dst_config, src_valid_shape, src_dtype, dst_dtype, **_):
    return (
        src_memory_space == "ub"
        and dst_memory_space == "ub"
        and src_config.b_layout == "row_major"
        and src_config.s_layout == "none_box"
        and dst_config.b_layout == "row_major"
        and dst_config.s_layout == "none_box"
        and src_dtype == dst_dtype
        and src_valid_shape != (1, 1)
    )


def _vec_to_vec_nd_scalar(src_memory_space, dst_memory_space, src_config, dst_config, src_valid_shape, src_dtype, dst_dtype, **_):
    return (
        src_memory_space == "ub"
        and dst_memory_space == "ub"
        and src_config.b_layout == "row_major"
        and src_config.s_layout == "none_box"
        and dst_config.b_layout == "row_major"
        and dst_config.s_layout == "none_box"
        and src_dtype == dst_dtype
        and src_valid_shape == (1, 1)
    )


_DTYPES = [(dtype, "i32", "i32", dtype) for dtype in NUMERIC_DTYPES]


_FP_DATA_DTYPES = (
    ("f32", "i32", "i32", "i8"),
    ("f32", "i32", "i32", "si8"),
    ("f32", "i32", "i32", "ui8"),
    ("f32", "i32", "i32", "f8e4m3"),
    ("f32", "i32", "i32", "hif8"),
    ("f32", "i32", "i32", "f16"),
    ("f32", "i32", "i32", "bf16"),
    ("f32", "i32", "i32", "f32"),
    ("i32", "i32", "i32", "i8"),
    ("si32", "i32", "i32", "i8"),
    ("i32", "i32", "i32", "si8"),
    ("si32", "i32", "i32", "si8"),
    ("i32", "i32", "i32", "ui8"),
    ("si32", "i32", "i32", "ui8"),
    ("i32", "i32", "i32", "f16"),
    ("si32", "i32", "i32", "f16"),
    ("i32", "i32", "i32", "bf16"),
    ("si32", "i32", "i32", "bf16"),
)
_FP_SCALING_DTYPES = ("f16", "bf16", "f32")
_FP_DTYPES = tuple(
    (*signature, fp_dtype)
    for signature in _FP_DATA_DTYPES
    for fp_dtype in _FP_SCALING_DTYPES
)
_FP_NZ_DTYPES = tuple(
    signature for signature in _FP_DTYPES if signature[3] == "f32"
)


def _canonical_dtype_name(dtype):
    name = str(dtype)
    return {
        "f8E4M3FN": "f8e4m3",
        "!pto.hif8": "hif8",
    }.get(name, name)


def _tinsert_fp_quant_mode(src_dtype, dst_dtype):
    modes = {
        ("f32", "i8"): "qf322b8_pre_vec",
        ("f32", "si8"): "qf322b8_pre_vec",
        ("f32", "ui8"): "qf322b8_pre_vec",
        ("f32", "f8e4m3"): "qf322fp8_pre_vec",
        ("f32", "hif8"): "qf322hif8_pre_vec",
        ("f32", "f16"): "qf322f16_pre_vec",
        ("f32", "bf16"): "qf322bf16_pre_vec",
        ("f32", "f32"): "qf322f32_pre_vec",
        ("i32", "i8"): "req8_vec",
        ("si32", "i8"): "req8_vec",
        ("i32", "si8"): "req8_vec",
        ("si32", "si8"): "req8_vec",
        ("i32", "ui8"): "req8_vec",
        ("si32", "ui8"): "req8_vec",
        ("i32", "f16"): "deqf16_vec",
        ("si32", "f16"): "deqf16_vec",
        ("i32", "bf16"): "qs322bf16_pre_vec",
        ("si32", "bf16"): "qs322bf16_pre_vec",
    }
    return modes[
        (_canonical_dtype_name(src_dtype), _canonical_dtype_name(dst_dtype))
    ]


def _acc_src_ptr(src):
    src_ptr = src.as_ptr()
    if str(src.dtype) == "si32":
        src_ptr = pto.castptr(src_ptr, pto.ptr(pto.i32, "acc"))
    return src_ptr


def _tinsert_pre_relu():
    mode = pto.get_op_attr("relu_pre_mode", "no_relu")
    return None if mode == "no_relu" else (mode, None, None)


def _acc_to_mat_fp(
    src_kind,
    dst_kind,
    src_memory_space,
    dst_memory_space,
    src_config,
    dst_config,
    fp_kind,
    fp_memory_space,
    **_,
):
    return (
        _acc_to_mat(
            src_kind,
            dst_kind,
            src_memory_space,
            dst_memory_space,
            src_config,
            dst_config,
        )
        and fp_kind == "tile"
        and fp_memory_space == "scaling"
    )


def _acc_to_vec_nd_fp(
    src_kind,
    dst_kind,
    src_memory_space,
    dst_memory_space,
    dst_config,
    fp_kind,
    fp_memory_space,
    **_,
):
    return (
        _acc_to_vec_nd(
            src_kind,
            dst_kind,
            src_memory_space,
            dst_memory_space,
            dst_config,
        )
        and fp_kind == "tile"
        and fp_memory_space == "scaling"
    )


def _acc_to_vec_nz_fp(
    src_kind,
    dst_kind,
    src_memory_space,
    dst_memory_space,
    dst_config,
    fp_kind,
    fp_memory_space,
    **_,
):
    return (
        _acc_to_vec_nz(
            src_kind,
            dst_kind,
            src_memory_space,
            dst_memory_space,
            dst_config,
        )
        and fp_kind == "tile"
        and fp_memory_space == "scaling"
    )


def _acc_to_vec_store_kwargs():
    dst_mode = 0
    split_mode = None
    acc_mode_name = pto.get_op_attr("acc_to_vec_mode", "single_mode_vec0")
    if acc_mode_name == "single_mode_vec1":
        dst_mode = 1
    elif acc_mode_name == "dual_mode_split_m":
        split_mode = "M"
    elif acc_mode_name == "dual_mode_split_n":
        split_mode = "N"

    kwargs = {}
    if split_mode is not None:
        kwargs["split"] = split_mode
    return dst_mode, kwargs


@tilelib.tile_template(
    op="pto.tinsert",
    target="a5",
    name="template_tinsert_acc_to_mat_basic",
    dtypes=(
        ("f32", "i32", "i32", "f16"),
        ("f32", "i32", "i32", "bf16"),
        ("f32", "i32", "i32", "f32"),
        ("i32", "i32", "i32", "i32"),
    ),
    iteration_axis="none",
    op_engine="other",
    op_class="movement",
    constraints=[_acc_to_mat],
    priority=1,
    id=2,
    loop_depth=0,
    is_post_update=False,
    tags=("insert", "acc", "mat"),
)
def template_tinsert_acc_to_mat_basic(
    src: pto.Tile,
    index_row: pto.i32,
    index_col: pto.i32,
    dst: pto.Tile,
):
    elem_bytes = pto.bytewidth(dst.dtype)
    c0_size = BLOCK_BYTE_SIZE // elem_bytes
    valid_rows, valid_cols = src.valid_shape
    n_size = (valid_cols + c0_size - 1) // c0_size * c0_size

    col_block = index_col // c0_size
    col_mod = index_col - col_block * c0_size
    dst_offset = dst.shape[0] * c0_size * col_block + index_row * c0_size + col_mod

    pto.mte_l0c_l1(
        src.as_ptr(),
        pto.addptr(dst.as_ptr(), dst_offset),
        valid_rows,
        n_size,
        src.shape[0] * pto.bytewidth(src.dtype),
        dst.shape[0] * c0_size * elem_bytes,
        pre_relu=_tinsert_pre_relu(),
    )


@tilelib.tile_template(
    op="pto.tinsert",
    target="a5",
    name="template_tinsert_fp_acc_to_mat",
    dtypes=_FP_DTYPES,
    iteration_axis="none",
    op_engine="other",
    op_class="movement",
    constraints=[_acc_to_mat_fp],
    priority=1,
    id=5,
    loop_depth=0,
    is_post_update=False,
    tags=("insert", "acc", "mat", "fp"),
)
def template_tinsert_fp_acc_to_mat(
    src: pto.Tile,
    index_row: pto.i32,
    index_col: pto.i32,
    dst: pto.Tile,
    fp: pto.Tile,
):
    elem_bytes = pto.bytewidth(dst.dtype)
    c0_size = BLOCK_BYTE_SIZE // elem_bytes
    valid_rows, valid_cols = src.valid_shape
    n_size = (valid_cols + c0_size - 1) // c0_size * c0_size
    col_block = index_col // c0_size
    col_mod = index_col - col_block * c0_size
    dst_offset = dst.shape[0] * c0_size * col_block + index_row * c0_size + col_mod

    pto.mte_l0c_l1(
        _acc_src_ptr(src),
        pto.addptr(dst.as_ptr(), dst_offset),
        valid_rows,
        n_size,
        src.shape[0] * pto.bytewidth(src.dtype),
        dst.shape[0] * c0_size * elem_bytes,
        pre_quant=(fp.as_ptr(), _tinsert_fp_quant_mode(src.dtype, dst.dtype)),
        pre_relu=_tinsert_pre_relu(),
    )


@tilelib.tile_template(
    op="pto.tinsert",
    target="a5",
    name="template_tinsert_acc_to_vec_nd_basic",
    dtypes=(
        ("f32", "i32", "i32", "f32"),
        ("f32", "i32", "i32", "f16"),
        ("f32", "i32", "i32", "bf16"),
        ("i32", "i32", "i32", "i32"),
    ),
    iteration_axis="none",
    op_engine="other",
    op_class="movement",
    constraints=[_acc_to_vec_nd],
    priority=1,
    id=3,
    loop_depth=0,
    is_post_update=False,
    tags=("insert", "acc", "vec", "nd"),
)
def template_tinsert_acc_to_vec_nd_basic(
    src: pto.Tile,
    index_row: pto.i32,
    index_col: pto.i32,
    dst: pto.Tile,
):
    elem_bytes = pto.bytewidth(dst.dtype)
    c0_size = BLOCK_BYTE_SIZE // elem_bytes
    valid_rows, valid_cols_raw = src.valid_shape
    valid_cols = (valid_cols_raw + c0_size - 1) // c0_size * c0_size
    dst_ptr = pto.addptr(dst.as_ptr(), index_row * dst.shape[1] + index_col)
    dst_mode, kwargs = _acc_to_vec_store_kwargs()
    kwargs["layout"] = "nz2nd"

    if str(src.dtype) == "f32" and str(dst.dtype) == "f16":
        pto.mte_l0c_ub(
            src.as_ptr(),
            dst_ptr,
            valid_rows,
            valid_cols,
            (valid_rows + 15) // 16 * 16,
            dst.shape[1],
            dst_mode,
            pre_quant=(pto.f16(1.0), "f32_f16"),
            pre_relu=_tinsert_pre_relu(),
            **kwargs,
        )
    elif str(src.dtype) == "f32" and str(dst.dtype) == "bf16":
        pto.mte_l0c_ub(
            src.as_ptr(),
            dst_ptr,
            valid_rows,
            valid_cols,
            (valid_rows + 15) // 16 * 16,
            dst.shape[1],
            dst_mode,
            pre_quant=(pto.bf16(1.0), "f32_bf16"),
            pre_relu=_tinsert_pre_relu(),
            **kwargs,
        )
    else:
        pto.mte_l0c_ub(
            src.as_ptr(),
            dst_ptr,
            valid_rows,
            valid_cols,
            (valid_rows + 15) // 16 * 16,
            dst.shape[1],
            dst_mode,
            pre_relu=_tinsert_pre_relu(),
            **kwargs,
        )


@tilelib.tile_template(
    op="pto.tinsert",
    target="a5",
    name="template_tinsert_fp_acc_to_vec_nd",
    dtypes=_FP_DTYPES,
    iteration_axis="none",
    op_engine="other",
    op_class="movement",
    constraints=[_acc_to_vec_nd_fp],
    priority=1,
    id=6,
    loop_depth=0,
    is_post_update=False,
    tags=("insert", "acc", "vec", "nd", "fp"),
)
def template_tinsert_fp_acc_to_vec_nd(
    src: pto.Tile,
    index_row: pto.i32,
    index_col: pto.i32,
    dst: pto.Tile,
    fp: pto.Tile,
):
    elem_bytes = pto.bytewidth(dst.dtype)
    c0_size = BLOCK_BYTE_SIZE // elem_bytes
    valid_rows, valid_cols_raw = src.valid_shape
    valid_cols = (valid_cols_raw + c0_size - 1) // c0_size * c0_size
    dst_ptr = pto.addptr(dst.as_ptr(), index_row * dst.shape[1] + index_col)
    dst_mode, kwargs = _acc_to_vec_store_kwargs()
    kwargs["layout"] = "nz2nd"

    pto.mte_l0c_ub(
        _acc_src_ptr(src),
        dst_ptr,
        valid_rows,
        valid_cols,
        (valid_rows + 15) // 16 * 16,
        dst.shape[1],
        dst_mode,
        pre_quant=(fp.as_ptr(), _tinsert_fp_quant_mode(src.dtype, dst.dtype)),
        pre_relu=_tinsert_pre_relu(),
        **kwargs,
    )


@tilelib.tile_template(
    op="pto.tinsert",
    target="a5",
    name="template_tinsert_acc_to_vec_nz_basic",
    dtypes=(
        ("f32", "i32", "i32", "f32"),
        ("f32", "i32", "i32", "f16"),
        ("f32", "i32", "i32", "bf16"),
        ("i32", "i32", "i32", "i32"),
    ),
    iteration_axis="none",
    op_engine="other",
    op_class="movement",
    constraints=[_acc_to_vec_nz],
    priority=1,
    id=4,
    loop_depth=0,
    is_post_update=False,
    tags=("insert", "acc", "vec", "nz"),
)
def template_tinsert_acc_to_vec_nz_basic(
    src: pto.Tile,
    index_row: pto.i32,
    index_col: pto.i32,
    dst: pto.Tile,
):
    elem_bytes = pto.bytewidth(dst.dtype)
    c0_size = BLOCK_BYTE_SIZE // elem_bytes
    valid_rows, valid_cols_raw = src.valid_shape
    valid_cols_align = 16 if str(dst.dtype) == "f32" else c0_size
    valid_cols = (valid_cols_raw + valid_cols_align - 1) // valid_cols_align * valid_cols_align

    col_block = index_col // c0_size
    col_mod = index_col - col_block * c0_size
    dst_offset = dst.shape[0] * c0_size * col_block + index_row * c0_size + col_mod
    dst_mode, kwargs = _acc_to_vec_store_kwargs()
    kwargs["layout"] = ("nz2nz", 0)

    pto.mte_l0c_ub(
        src.as_ptr(),
        pto.addptr(dst.as_ptr(), dst_offset),
        valid_rows,
        valid_cols,
        (valid_rows + 15) // 16 * 16 * pto.bytewidth(src.dtype),
        dst.shape[0] * c0_size * elem_bytes,
        dst_mode,
        pre_relu=_tinsert_pre_relu(),
        **kwargs,
    )


@tilelib.tile_template(
    op="pto.tinsert",
    target="a5",
    name="template_tinsert_fp_acc_to_vec_nz",
    dtypes=_FP_NZ_DTYPES,
    iteration_axis="none",
    op_engine="other",
    op_class="movement",
    constraints=[_acc_to_vec_nz_fp],
    priority=1,
    id=7,
    loop_depth=0,
    is_post_update=False,
    tags=("insert", "acc", "vec", "nz", "fp"),
)
def template_tinsert_fp_acc_to_vec_nz(
    src: pto.Tile,
    index_row: pto.i32,
    index_col: pto.i32,
    dst: pto.Tile,
    fp: pto.Tile,
):
    elem_bytes = pto.bytewidth(dst.dtype)
    c0_size = BLOCK_BYTE_SIZE // elem_bytes
    valid_rows, valid_cols_raw = src.valid_shape
    valid_cols_align = 16 if str(dst.dtype) == "f32" else c0_size
    valid_cols = (
        (valid_cols_raw + valid_cols_align - 1)
        // valid_cols_align
        * valid_cols_align
    )

    col_block = index_col // c0_size
    col_mod = index_col - col_block * c0_size
    dst_offset = (
        dst.shape[0] * c0_size * col_block + index_row * c0_size + col_mod
    )
    dst_mode, kwargs = _acc_to_vec_store_kwargs()
    kwargs["layout"] = ("nz2nz", 0)

    pto.mte_l0c_ub(
        _acc_src_ptr(src),
        pto.addptr(dst.as_ptr(), dst_offset),
        valid_rows,
        valid_cols,
        (valid_rows + 15) // 16 * 16 * pto.bytewidth(src.dtype),
        dst.shape[0] * c0_size * elem_bytes,
        dst_mode,
        pre_quant=(fp.as_ptr(), _tinsert_fp_quant_mode(src.dtype, dst.dtype)),
        pre_relu=_tinsert_pre_relu(),
        **kwargs,
    )


@tilelib.tile_template(
    op="pto.tinsert",
    target="a5",
    name="template_tinsert_vec_to_vec_nd_basic",
    dtypes=_DTYPES,
    iteration_axis="none",
    op_engine="vector",
    op_class="movement",
    constraints=[_vec_to_vec_nd],
    id=0,
    loop_depth=2,
    is_post_update=False,
    tags=("move", "insert", "ub"),
)
def template_tinsert_vec_to_vec_nd_basic(
    src: pto.Tile,
    index_row: pto.i32,
    index_col: pto.i32,
    dst: pto.Tile,
):
    dtype = dst.dtype
    valid_rows, valid_cols = src.valid_shape
    lanes = pto.elements_per_vreg(dtype)
    for row in range(0, valid_rows, 1):
        remained = valid_cols
        for col in range(0, valid_cols, lanes):
            mask, remained = pto.make_mask(dtype, remained)
            data = pto.vlds(src[row, col:])
            pto.vsts(data, dst[index_row + row, index_col + col:], mask)


@tilelib.tile_template(
    op="pto.tinsert",
    target="a5",
    name="template_tinsert_vec_to_vec_nd_scalar_basic",
    dtypes=_DTYPES,
    iteration_axis="none",
    op_engine="vector",
    op_class="movement",
    constraints=[_vec_to_vec_nd_scalar],
    priority=1,
    id=1,
    loop_depth=0,
    is_post_update=False,
    tags=("move", "insert", "ub", "scalar"),
)
def template_tinsert_vec_to_vec_nd_scalar_basic(
    src: pto.Tile,
    index_row: pto.i32,
    index_col: pto.i32,
    dst: pto.Tile,
):
    value = pto.load_scalar(src.as_ptr(), 0)
    pto.store_scalar(dst.as_ptr(), index_row * dst.shape[1] + index_col, value)
