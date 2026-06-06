# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""TileLang DSL template for pto.tinsert - tile data insertion

Implements six data movement paths using @pto.ckernel:
  - Acc→Mat NZ: pto.mte_l0c_l1 (copy_matrix_cc_to_cbuf)
  - Acc→Vec ND/DN/NZ: pto.mte_l0c_ub (copy_matrix_cc_to_ub)
  - Vec→Vec ND/NZ: pto.copy_ubuf_to_ubuf
  - Vec→Mat NZ/ND: pto.mte_ub_l1 (copy_ubuf_to_cbuf)
"""

import tilelang_dsl as pto

BLOCK_BYTE_SIZE = 32
BLOCK_BYTE_BITS = 256
FRACTAL_NZ_ROW = 16


# ---------------------------------------------------------------------------
# Constraint functions
# ---------------------------------------------------------------------------


def _acc_to_mat_constraint(src, dst) -> bool:
    return (
        src.memory_space == "acc"
        and dst.memory_space == "mat"
        and dst.config.b_layout == pto.BLayout.COL_MAJOR
        and dst.config.s_layout == pto.SLayout.ROW_MAJOR
    )


def _acc_to_vec_nd_constraint(src, dst) -> bool:
    return (
        src.memory_space == "acc"
        and dst.memory_space == "ub"
        and dst.config.b_layout == pto.BLayout.ROW_MAJOR
        and dst.config.s_layout == pto.SLayout.NONE_BOX
    )


def _acc_to_vec_dn_constraint(src, dst) -> bool:
    return (
        src.memory_space == "acc"
        and dst.memory_space == "ub"
        and dst.config.b_layout == pto.BLayout.COL_MAJOR
        and dst.config.s_layout == pto.SLayout.NONE_BOX
    )


def _acc_to_vec_nz_constraint(src, dst) -> bool:
    return (
        src.memory_space == "acc"
        and dst.memory_space == "ub"
        and dst.config.b_layout == pto.BLayout.COL_MAJOR
        and dst.config.s_layout == pto.SLayout.ROW_MAJOR
    )


def _vec_to_vec_nd_constraint(src, dst) -> bool:
    return (
        src.memory_space == "ub"
        and src.config.b_layout == pto.BLayout.ROW_MAJOR
        and src.config.s_layout == pto.SLayout.NONE_BOX
        and dst.memory_space == "ub"
        and dst.config.b_layout == pto.BLayout.ROW_MAJOR
        and dst.config.s_layout == pto.SLayout.NONE_BOX
        and not (src.valid_shape[0] == 1 and src.valid_shape[1] == 1)
    )


def _vec_to_vec_nd_scalar_constraint(src, dst) -> bool:
    return (
        src.memory_space == "ub"
        and src.config.b_layout == pto.BLayout.ROW_MAJOR
        and src.config.s_layout == pto.SLayout.NONE_BOX
        and dst.memory_space == "ub"
        and dst.config.b_layout == pto.BLayout.ROW_MAJOR
        and dst.config.s_layout == pto.SLayout.NONE_BOX
        and src.valid_shape[0] == 1
        and src.valid_shape[1] == 1
    )


def _vec_to_vec_nz_constraint(src, dst) -> bool:
    return (
        src.memory_space == "ub"
        and src.config.b_layout == pto.BLayout.COL_MAJOR
        and src.config.s_layout == pto.SLayout.ROW_MAJOR
        and dst.memory_space == "ub"
        and dst.config.b_layout == pto.BLayout.COL_MAJOR
        and dst.config.s_layout == pto.SLayout.ROW_MAJOR
    )


def _vec_to_mat_nz_constraint(src, dst) -> bool:
    return (
        src.memory_space == "ub"
        and src.config.b_layout == pto.BLayout.COL_MAJOR
        and src.config.s_layout == pto.SLayout.ROW_MAJOR
        and dst.memory_space == "mat"
        and dst.config.b_layout == pto.BLayout.COL_MAJOR
        and dst.config.s_layout == pto.SLayout.ROW_MAJOR
    )


def _vec_to_mat_nd_constraint(src, dst) -> bool:
    return (
        src.memory_space == "ub"
        and src.config.b_layout == pto.BLayout.ROW_MAJOR
        and src.config.s_layout == pto.SLayout.NONE_BOX
        and dst.memory_space == "mat"
        and dst.config.s_layout == pto.SLayout.NONE_BOX
        and src.valid_shape[1] * pto.bytewidth(pto.ScalarType(src.dtype)) >= BLOCK_BYTE_SIZE
    )


def _pre_quant_vec_mode(src_dtype, dst_dtype):
    if pto.constexpr(src_dtype == pto.f32):
        if pto.constexpr(dst_dtype == pto.f16):
            return "qf322f16_pre_vec"
        elif pto.constexpr(dst_dtype == pto.bf16):
            return "qf322bf16_pre_vec"
        elif pto.constexpr(dst_dtype == pto.f32):
            return "qf322f32_pre_vec"
        else:
            return "qf322b8_pre_vec"
    else:
        if pto.constexpr(dst_dtype == pto.f16):
            return "deqf16_vec"
        elif pto.constexpr(dst_dtype == pto.bf16):
            return "qs322bf16_pre_vec"
        elif pto.constexpr(dst_dtype == pto.i32):
            return "deqs32_int_vec"
        else:
            return "req8_vec"


def _pre_quant_scalar_mode(src_dtype, dst_dtype):
    if pto.constexpr(src_dtype == pto.f32):
        if pto.constexpr(dst_dtype == pto.f16):
            return "qf322f16_pre_scalar"
        elif pto.constexpr(dst_dtype == pto.bf16):
            return "qf322bf16_pre_scalar"
        elif pto.constexpr(dst_dtype == pto.f32):
            return "qf322f32_pre_scalar"
        else:
            return "qf322b8_pre_scalar"
    else:
        if pto.constexpr(dst_dtype == pto.f16):
            return "deqf16_scalar"
        elif pto.constexpr(dst_dtype == pto.bf16):
            return "qs322bf16_pre_scalar"
        elif pto.constexpr(dst_dtype == pto.i32):
            return "deqs32_int_scalar"
        else:
            return "req8_scalar"


# ---------------------------------------------------------------------------
# Acc -> Mat
# ---------------------------------------------------------------------------


@pto.ckernel(
    target="a5",
    op="pto.tinsert",
    dtypes=[
        (pto.f32, pto.i64, pto.i64, pto.f16, pto.f32, pto.i64),
        (pto.f32, pto.i64, pto.i64, pto.bf16, pto.f32, pto.i64),
        (pto.f32, pto.i64, pto.i64, pto.i8, pto.f32, pto.i64),
        (pto.i32, pto.i64, pto.i64, pto.f16, pto.i32, pto.i64),
        (pto.i32, pto.i64, pto.i64, pto.bf16, pto.i32, pto.i64),
        (pto.i32, pto.i64, pto.i64, pto.i8, pto.i32, pto.i64),
    ],
    constraints=[_acc_to_mat_constraint],
)
def template_tinsert_acc_to_mat(
    src: pto.Tile,
    index_row: pto.i64, index_col: pto.i64,
    dst: pto.Tile,
    fp: pto.Tile, pre_quant_scalar: pto.i64,
):
    src_ptr = src.as_ptr()
    dst_ptr = dst.as_ptr()

    src_elem = src.element_type
    dst_elem = dst.element_type
    elem_bytes = pto.bytewidth(dst_elem)
    c0_standard = BLOCK_BYTE_SIZE // elem_bytes
    s_frac_bits = dst.config.s_fractal_size
    if pto.constexpr(s_frac_bits == 2 * BLOCK_BYTE_BITS):
        c0_size = 2 * c0_standard
    else:
        c0_size = c0_standard

    valid_rows = src.valid_shape[0]
    valid_cols = src.valid_shape[1]
    n_size = (valid_cols + c0_size - 1) // c0_size * c0_size

    dst_rows = dst.shape[0]
    col_block = index_col // c0_size
    col_mod = index_col - col_block * c0_size
    dst_offset = dst_rows * c0_size * col_block + index_row * c0_size + col_mod
    dst_ptr = pto.addptr(dst_ptr, dst_offset)

    dst_stride = dst_rows * c0_size * elem_bytes
    src_stride = src.shape[0] * elem_bytes

    relu_mode_name = pto.get_op_attr("relu_pre_mode", "no_relu")
    has_fp = fp is not None
    has_scalar = pre_quant_scalar is not None

    if pto.constexpr(has_fp):
        if pto.constexpr(relu_mode_name == "normal_relu"):
            pto.mte_l0c_l1(
                src_ptr, dst_ptr,
                valid_rows, n_size, src_stride, dst_stride,
                pre_relu=("normal_relu", None, None),
                pre_quant=(fp, _pre_quant_vec_mode(src_elem, dst_elem)),
            )
        else:
            pto.mte_l0c_l1(
                src_ptr, dst_ptr,
                valid_rows, n_size, src_stride, dst_stride,
                pre_quant=(fp, _pre_quant_vec_mode(src_elem, dst_elem)),
            )
    elif pto.constexpr(has_scalar):
        if pto.constexpr(relu_mode_name == "normal_relu"):
            pto.mte_l0c_l1(
                src_ptr, dst_ptr,
                valid_rows, n_size, src_stride, dst_stride,
                pre_relu=("normal_relu", None, None),
                pre_quant=(pre_quant_scalar, _pre_quant_scalar_mode(src_elem, dst_elem)),
            )
        else:
            pto.mte_l0c_l1(
                src_ptr, dst_ptr,
                valid_rows, n_size, src_stride, dst_stride,
                pre_quant=(pre_quant_scalar, _pre_quant_scalar_mode(src_elem, dst_elem)),
            )
    elif pto.constexpr(relu_mode_name == "normal_relu"):
        pto.mte_l0c_l1(
            src_ptr, dst_ptr,
            valid_rows, n_size, src_stride, dst_stride,
            pre_relu=("normal_relu", None, None),
        )
    else:
        pto.mte_l0c_l1(
            src_ptr, dst_ptr,
            valid_rows, n_size, src_stride, dst_stride,
        )
    return None


@pto.ckernel(
    target="a5",
    op="pto.tinsert",
    dtypes=[
        (pto.f32, pto.i32, pto.i32, pto.f16),
        (pto.f32, pto.i32, pto.i32, pto.bf16),
        (pto.f32, pto.i32, pto.i32, pto.f32),
        (pto.i32, pto.i32, pto.i32, pto.i32),
    ],
    constraints=[_acc_to_mat_constraint],
)
def template_tinsert_acc_to_mat_basic(
    src: pto.Tile,
    index_row: pto.i32, index_col: pto.i32,
    dst: pto.Tile,
):
    src_ptr = src.as_ptr()
    dst_ptr = dst.as_ptr()

    dst_elem = dst.element_type
    elem_bytes = pto.bytewidth(dst_elem)
    c0_standard = BLOCK_BYTE_SIZE // elem_bytes
    s_frac_bits = dst.config.s_fractal_size
    if pto.constexpr(s_frac_bits == 2 * BLOCK_BYTE_BITS):
        c0_size = 2 * c0_standard
    else:
        c0_size = c0_standard

    valid_rows = src.valid_shape[0]
    valid_cols = src.valid_shape[1]
    n_size = (valid_cols + c0_size - 1) // c0_size * c0_size

    dst_rows = dst.shape[0]
    col_block = index_col // c0_size
    col_mod = index_col - col_block * c0_size
    dst_offset = dst_rows * c0_size * col_block + index_row * c0_size + col_mod
    dst_ptr = pto.addptr(dst_ptr, dst_offset)

    dst_stride = dst_rows * c0_size * elem_bytes
    src_stride = src.shape[0] * elem_bytes

    relu_mode_name = pto.get_op_attr("relu_pre_mode", "no_relu")

    if pto.constexpr(relu_mode_name == "normal_relu"):
        pto.mte_l0c_l1(
            src_ptr, dst_ptr,
            valid_rows, n_size, src_stride, dst_stride,
            pre_relu=("normal_relu", None, None),
        )
    else:
        pto.mte_l0c_l1(
            src_ptr, dst_ptr,
            valid_rows, n_size, src_stride, dst_stride,
        )
    return None


# ---------------------------------------------------------------------------
# Acc -> Vec (ND, NONE_BOX)
# ---------------------------------------------------------------------------


@pto.ckernel(
    target="a5",
    op="pto.tinsert",
    dtypes=[
        (pto.f32, pto.i64, pto.i64, pto.f32, pto.f32, pto.i64),
        (pto.f32, pto.i64, pto.i64, pto.f16, pto.f32, pto.i64),
        (pto.f32, pto.i64, pto.i64, pto.bf16, pto.f32, pto.i64),
        (pto.i32, pto.i64, pto.i64, pto.i32, pto.i32, pto.i64),
        (pto.i32, pto.i64, pto.i64, pto.f16, pto.i32, pto.i64),
        (pto.i32, pto.i64, pto.i64, pto.bf16, pto.i32, pto.i64),
        (pto.i32, pto.i64, pto.i64, pto.i8, pto.i32, pto.i64),
    ],
    priority=10,
    constraints=[_acc_to_vec_nd_constraint],
)
def template_tinsert_acc_to_vec_nd(
    src: pto.Tile,
    index_row: pto.i64, index_col: pto.i64,
    dst: pto.Tile,
    fp: pto.Tile, pre_quant_scalar: pto.i64,
):
    src_ptr = src.as_ptr()
    dst_ptr = dst.as_ptr()

    src_elem = src.element_type
    dst_elem = dst.element_type
    elem_bytes = pto.bytewidth(dst_elem)
    c0_size = 32 // elem_bytes

    valid_rows = src.valid_shape[0]
    valid_cols_raw = src.valid_shape[1]
    valid_cols = (valid_cols_raw + c0_size - 1) // c0_size * c0_size

    dst_cols = dst.shape[1]
    dst_offset = index_row * dst_cols + index_col
    dst_ptr = pto.addptr(dst_ptr, dst_offset)

    dst_stride = dst_cols * elem_bytes
    src_stride = (valid_rows + 15) // 16 * 16 * elem_bytes

    relu_mode_name = pto.get_op_attr("relu_pre_mode", "no_relu")
    acc_mode_name = pto.get_op_attr("acc_to_vec_mode", "single_mode_vec0")
    has_fp = fp is not None
    has_scalar = pre_quant_scalar is not None

    dst_mode = 0
    if pto.constexpr(acc_mode_name == "single_mode_vec1"):
        dst_mode = 1
    elif pto.constexpr(acc_mode_name == "dual_mode_split_m"):
        dst_mode = "split_m"
    elif pto.constexpr(acc_mode_name == "dual_mode_split_n"):
        dst_mode = "split_n"

    supports_pre_quant = acc_mode_name not in ("dual_mode_split_m", "dual_mode_split_n")

    if pto.constexpr(has_fp and supports_pre_quant):
        if pto.constexpr(relu_mode_name == "normal_relu"):
            pto.mte_l0c_ub(
                src_ptr, dst_ptr,
                valid_rows, valid_cols, src_stride, dst_stride, dst_mode,
                layout="nz2nd",
                pre_relu=("normal_relu", None, None),
                pre_quant=(fp, _pre_quant_vec_mode(src_elem, dst_elem)),
            )
        else:
            pto.mte_l0c_ub(
                src_ptr, dst_ptr,
                valid_rows, valid_cols, src_stride, dst_stride, dst_mode,
                layout="nz2nd",
                pre_quant=(fp, _pre_quant_vec_mode(src_elem, dst_elem)),
            )
    elif pto.constexpr(has_scalar and supports_pre_quant):
        if pto.constexpr(relu_mode_name == "normal_relu"):
            pto.mte_l0c_ub(
                src_ptr, dst_ptr,
                valid_rows, valid_cols, src_stride, dst_stride, dst_mode,
                layout="nz2nd",
                pre_relu=("normal_relu", None, None),
                pre_quant=(pre_quant_scalar, _pre_quant_scalar_mode(src_elem, dst_elem)),
            )
        else:
            pto.mte_l0c_ub(
                src_ptr, dst_ptr,
                valid_rows, valid_cols, src_stride, dst_stride, dst_mode,
                layout="nz2nd",
                pre_quant=(pre_quant_scalar, _pre_quant_scalar_mode(src_elem, dst_elem)),
            )
    elif pto.constexpr(relu_mode_name == "normal_relu"):
        pto.mte_l0c_ub(
            src_ptr, dst_ptr,
            valid_rows, valid_cols, src_stride, dst_stride, dst_mode,
            layout="nz2nd",
            pre_relu=("normal_relu", None, None),
        )
    else:
        pto.mte_l0c_ub(
            src_ptr, dst_ptr,
            valid_rows, valid_cols, src_stride, dst_stride, dst_mode,
            layout="nz2nd",
        )
    return None


@pto.ckernel(
    target="a5",
    op="pto.tinsert",
    dtypes=[
        (pto.f32, pto.i32, pto.i32, pto.f32),
        (pto.f32, pto.i32, pto.i32, pto.f16),
        (pto.f32, pto.i32, pto.i32, pto.bf16),
        (pto.i32, pto.i32, pto.i32, pto.i32),
    ],
    priority=10,
    constraints=[_acc_to_vec_nd_constraint],
)
def template_tinsert_acc_to_vec_nd_basic(
    src: pto.Tile,
    index_row: pto.i32, index_col: pto.i32,
    dst: pto.Tile,
):
    src_ptr = src.as_ptr()
    dst_ptr = dst.as_ptr()

    elem_bytes = pto.bytewidth(dst.element_type)
    c0_size = 32 // elem_bytes

    valid_rows = src.valid_shape[0]
    valid_cols_raw = src.valid_shape[1]
    valid_cols = (valid_cols_raw + c0_size - 1) // c0_size * c0_size

    dst_cols = dst.shape[1]
    dst_offset = index_row * dst_cols + index_col
    dst_ptr = pto.addptr(dst_ptr, dst_offset)

    dst_stride = dst_cols * elem_bytes
    src_stride = (valid_rows + 15) // 16 * 16 * elem_bytes

    relu_mode_name = pto.get_op_attr("relu_pre_mode", "no_relu")
    acc_mode_name = pto.get_op_attr("acc_to_vec_mode", "single_mode_vec0")

    dst_mode = 0
    if pto.constexpr(acc_mode_name == "single_mode_vec1"):
        dst_mode = 1
    elif pto.constexpr(acc_mode_name == "dual_mode_split_m"):
        dst_mode = "split_m"
    elif pto.constexpr(acc_mode_name == "dual_mode_split_n"):
        dst_mode = "split_n"

    if pto.constexpr(relu_mode_name == "normal_relu"):
        pto.mte_l0c_ub(
            src_ptr, dst_ptr,
            valid_rows, valid_cols, src_stride, dst_stride, dst_mode,
            layout="nz2nd",
            pre_relu=("normal_relu", None, None),
        )
    else:
        pto.mte_l0c_ub(
            src_ptr, dst_ptr,
            valid_rows, valid_cols, src_stride, dst_stride, dst_mode,
            layout="nz2nd",
        )
    return None


# ---------------------------------------------------------------------------
# Acc -> Vec (DN, NONE_BOX)
# ---------------------------------------------------------------------------


@pto.ckernel(
    target="a5",
    op="pto.tinsert",
    dtypes=[
        (pto.f32, pto.i64, pto.i64, pto.f32, pto.f32, pto.i64),
        (pto.f32, pto.i64, pto.i64, pto.f16, pto.f32, pto.i64),
        (pto.f32, pto.i64, pto.i64, pto.bf16, pto.f32, pto.i64),
        (pto.i32, pto.i64, pto.i64, pto.i32, pto.i32, pto.i64),
        (pto.i32, pto.i64, pto.i64, pto.f16, pto.i32, pto.i64),
        (pto.i32, pto.i64, pto.i64, pto.bf16, pto.i32, pto.i64),
        (pto.i32, pto.i64, pto.i64, pto.i8, pto.i32, pto.i64),
    ],
    priority=10,
    constraints=[_acc_to_vec_dn_constraint],
)
def template_tinsert_acc_to_vec_dn(
    src: pto.Tile,
    index_row: pto.i64, index_col: pto.i64,
    dst: pto.Tile,
    fp: pto.Tile, pre_quant_scalar: pto.i64,
):
    src_ptr = src.as_ptr()
    dst_ptr = dst.as_ptr()

    src_elem = src.element_type
    dst_elem = dst.element_type
    elem_bytes = pto.bytewidth(dst_elem)
    c0_size = 32 // elem_bytes

    valid_rows_raw = src.valid_shape[0]
    valid_rows = (valid_rows_raw + c0_size - 1) // c0_size * c0_size
    valid_cols = src.valid_shape[1]

    dst_rows = dst.shape[0]
    dst_offset = index_col * dst_rows + index_row
    dst_ptr = pto.addptr(dst_ptr, dst_offset)

    dst_stride = dst_rows * elem_bytes
    src_stride = (valid_rows + 15) // 16 * 16 * elem_bytes

    relu_mode_name = pto.get_op_attr("relu_pre_mode", "no_relu")
    acc_mode_name = pto.get_op_attr("acc_to_vec_mode", "single_mode_vec0")
    has_fp = fp is not None
    has_scalar = pre_quant_scalar is not None

    dst_mode = 0
    if pto.constexpr(acc_mode_name == "single_mode_vec1"):
        dst_mode = 1

    if pto.constexpr(has_fp):
        if pto.constexpr(relu_mode_name == "normal_relu"):
            pto.mte_l0c_ub(
                src_ptr, dst_ptr,
                valid_rows, valid_cols, src_stride, dst_stride, dst_mode,
                layout=("nz2dn", pto.i64(1)),
                pre_relu=("normal_relu", None, None),
                pre_quant=(fp, _pre_quant_vec_mode(src_elem, dst_elem)),
            )
        else:
            pto.mte_l0c_ub(
                src_ptr, dst_ptr,
                valid_rows, valid_cols, src_stride, dst_stride, dst_mode,
                layout=("nz2dn", pto.i64(1)),
                pre_quant=(fp, _pre_quant_vec_mode(src_elem, dst_elem)),
            )
    elif pto.constexpr(has_scalar):
        if pto.constexpr(relu_mode_name == "normal_relu"):
            pto.mte_l0c_ub(
                src_ptr, dst_ptr,
                valid_rows, valid_cols, src_stride, dst_stride, dst_mode,
                layout=("nz2dn", pto.i64(1)),
                pre_relu=("normal_relu", None, None),
                pre_quant=(pre_quant_scalar, _pre_quant_scalar_mode(src_elem, dst_elem)),
            )
        else:
            pto.mte_l0c_ub(
                src_ptr, dst_ptr,
                valid_rows, valid_cols, src_stride, dst_stride, dst_mode,
                layout=("nz2dn", pto.i64(1)),
                pre_quant=(pre_quant_scalar, _pre_quant_scalar_mode(src_elem, dst_elem)),
            )
    elif pto.constexpr(relu_mode_name == "normal_relu"):
        pto.mte_l0c_ub(
            src_ptr, dst_ptr,
            valid_rows, valid_cols, src_stride, dst_stride, dst_mode,
            layout=("nz2dn", pto.i64(1)),
            pre_relu=("normal_relu", None, None),
        )
    else:
        pto.mte_l0c_ub(
            src_ptr, dst_ptr,
            valid_rows, valid_cols, src_stride, dst_stride, dst_mode,
            layout=("nz2dn", pto.i64(1)),
        )
    return None


@pto.ckernel(
    target="a5",
    op="pto.tinsert",
    dtypes=[
        (pto.f32, pto.i32, pto.i32, pto.f32),
        (pto.f32, pto.i32, pto.i32, pto.f16),
        (pto.f32, pto.i32, pto.i32, pto.bf16),
        (pto.i32, pto.i32, pto.i32, pto.i32),
    ],
    priority=10,
    constraints=[_acc_to_vec_dn_constraint],
)
def template_tinsert_acc_to_vec_dn_basic(
    src: pto.Tile,
    index_row: pto.i32, index_col: pto.i32,
    dst: pto.Tile,
):
    src_ptr = src.as_ptr()
    dst_ptr = dst.as_ptr()

    elem_bytes = pto.bytewidth(dst.element_type)
    c0_size = 32 // elem_bytes

    valid_rows_raw = src.valid_shape[0]
    valid_rows = (valid_rows_raw + c0_size - 1) // c0_size * c0_size
    valid_cols = src.valid_shape[1]

    dst_rows = dst.shape[0]
    dst_offset = index_col * dst_rows + index_row
    dst_ptr = pto.addptr(dst_ptr, dst_offset)

    dst_stride = dst_rows * elem_bytes
    src_stride = (valid_rows + 15) // 16 * 16 * elem_bytes

    relu_mode_name = pto.get_op_attr("relu_pre_mode", "no_relu")

    dst_mode = 0
    acc_mode_name = pto.get_op_attr("acc_to_vec_mode", "single_mode_vec0")
    if pto.constexpr(acc_mode_name == "single_mode_vec1"):
        dst_mode = 1

    if pto.constexpr(relu_mode_name == "normal_relu"):
        pto.mte_l0c_ub(
            src_ptr, dst_ptr,
            valid_rows, valid_cols, src_stride, dst_stride, dst_mode,
            layout=("nz2dn", pto.i64(1)),
            pre_relu=("normal_relu", None, None),
        )
    else:
        pto.mte_l0c_ub(
            src_ptr, dst_ptr,
            valid_rows, valid_cols, src_stride, dst_stride, dst_mode,
            layout=("nz2dn", pto.i64(1)),
        )
    return None


# ---------------------------------------------------------------------------
# Acc -> Vec (NZ, ROW_MAJOR)
# ---------------------------------------------------------------------------


@pto.ckernel(
    target="a5",
    op="pto.tinsert",
    dtypes=[
        (pto.f32, pto.i64, pto.i64, pto.f32, pto.f32, pto.i64),
        (pto.f32, pto.i64, pto.i64, pto.f16, pto.f32, pto.i64),
        (pto.f32, pto.i64, pto.i64, pto.bf16, pto.f32, pto.i64),
        (pto.i32, pto.i64, pto.i64, pto.i32, pto.i32, pto.i64),
        (pto.i32, pto.i64, pto.i64, pto.f16, pto.i32, pto.i64),
        (pto.i32, pto.i64, pto.i64, pto.bf16, pto.i32, pto.i64),
        (pto.i32, pto.i64, pto.i64, pto.i8, pto.i32, pto.i64),
    ],
    priority=10,
    constraints=[_acc_to_vec_nz_constraint],
)
def template_tinsert_acc_to_vec_nz(
    src: pto.Tile,
    index_row: pto.i64, index_col: pto.i64,
    dst: pto.Tile,
    fp: pto.Tile, pre_quant_scalar: pto.i64,
):
    src_ptr = src.as_ptr()
    dst_ptr = dst.as_ptr()

    src_elem = src.element_type
    dst_elem = dst.element_type
    elem_bytes = pto.bytewidth(dst_elem)
    c0_standard = BLOCK_BYTE_SIZE // elem_bytes
    s_frac_bits = dst.config.s_fractal_size
    if pto.constexpr(s_frac_bits == 2 * BLOCK_BYTE_BITS):
        c0_size = 2 * c0_standard
    else:
        c0_size = c0_standard

    valid_rows = src.valid_shape[0]
    valid_cols_raw = src.valid_shape[1]

    if pto.constexpr(dst_elem == pto.f32):
        s_frac_bits = dst.config.s_fractal_size
        if pto.constexpr(s_frac_bits == BLOCK_BYTE_BITS):
            valid_cols_align = c0_size
        else:
            valid_cols_align = FRACTAL_NZ_ROW
    else:
        valid_cols_align = c0_size
    valid_cols = (valid_cols_raw + valid_cols_align - 1) // valid_cols_align * valid_cols_align

    dst_rows = dst.shape[0]
    col_block = index_col // c0_size
    col_mod = index_col - col_block * c0_size
    dst_offset = dst_rows * c0_size * col_block + index_row * c0_size + col_mod
    dst_ptr = pto.addptr(dst_ptr, dst_offset)

    dst_stride = dst_rows * c0_size * elem_bytes
    src_stride = (valid_rows + 15) // 16 * 16 * elem_bytes

    relu_mode_name = pto.get_op_attr("relu_pre_mode", "no_relu")
    acc_mode_name = pto.get_op_attr("acc_to_vec_mode", "single_mode_vec0")
    has_fp = fp is not None
    has_scalar = pre_quant_scalar is not None

    dst_mode = 0
    if pto.constexpr(acc_mode_name == "single_mode_vec1"):
        dst_mode = 1
    elif pto.constexpr(acc_mode_name == "dual_mode_split_m"):
        dst_mode = "split_m"
    elif pto.constexpr(acc_mode_name == "dual_mode_split_n"):
        dst_mode = "split_n"

    if pto.constexpr(has_fp):
        if pto.constexpr(relu_mode_name == "normal_relu"):
            pto.mte_l0c_ub(
                src_ptr, dst_ptr,
                valid_rows, valid_cols, src_stride, dst_stride, dst_mode,
                layout=("nz2nz", pto.i64(0)),
                pre_relu=("normal_relu", None, None),
                pre_quant=(fp, _pre_quant_vec_mode(src_elem, dst_elem)),
            )
        else:
            pto.mte_l0c_ub(
                src_ptr, dst_ptr,
                valid_rows, valid_cols, src_stride, dst_stride, dst_mode,
                layout=("nz2nz", pto.i64(0)),
                pre_quant=(fp, _pre_quant_vec_mode(src_elem, dst_elem)),
            )
    elif pto.constexpr(has_scalar):
        if pto.constexpr(relu_mode_name == "normal_relu"):
            pto.mte_l0c_ub(
                src_ptr, dst_ptr,
                valid_rows, valid_cols, src_stride, dst_stride, dst_mode,
                layout=("nz2nz", pto.i64(0)),
                pre_relu=("normal_relu", None, None),
                pre_quant=(pre_quant_scalar, _pre_quant_scalar_mode(src_elem, dst_elem)),
            )
        else:
            pto.mte_l0c_ub(
                src_ptr, dst_ptr,
                valid_rows, valid_cols, src_stride, dst_stride, dst_mode,
                layout=("nz2nz", pto.i64(0)),
                pre_quant=(pre_quant_scalar, _pre_quant_scalar_mode(src_elem, dst_elem)),
            )
    elif pto.constexpr(relu_mode_name == "normal_relu"):
        pto.mte_l0c_ub(
            src_ptr, dst_ptr,
            valid_rows, valid_cols, src_stride, dst_stride, dst_mode,
            layout=("nz2nz", pto.i64(0)),
            pre_relu=("normal_relu", None, None),
        )
    else:
        pto.mte_l0c_ub(
            src_ptr, dst_ptr,
            valid_rows, valid_cols, src_stride, dst_stride, dst_mode,
            layout=("nz2nz", pto.i64(0)),
        )
    return None


@pto.ckernel(
    target="a5",
    op="pto.tinsert",
    dtypes=[
        (pto.f32, pto.i32, pto.i32, pto.f32),
        (pto.f32, pto.i32, pto.i32, pto.f16),
        (pto.f32, pto.i32, pto.i32, pto.bf16),
        (pto.i32, pto.i32, pto.i32, pto.i32),
    ],
    priority=10,
    constraints=[_acc_to_vec_nz_constraint],
)
def template_tinsert_acc_to_vec_nz_basic(
    src: pto.Tile,
    index_row: pto.i32, index_col: pto.i32,
    dst: pto.Tile,
):
    src_ptr = src.as_ptr()
    dst_ptr = dst.as_ptr()

    dst_elem = dst.element_type
    elem_bytes = pto.bytewidth(dst_elem)
    c0_standard = BLOCK_BYTE_SIZE // elem_bytes
    s_frac_bits = dst.config.s_fractal_size
    if pto.constexpr(s_frac_bits == 2 * BLOCK_BYTE_BITS):
        c0_size = 2 * c0_standard
    else:
        c0_size = c0_standard

    valid_rows = src.valid_shape[0]
    valid_cols_raw = src.valid_shape[1]

    if pto.constexpr(dst_elem == pto.f32):
        s_frac_bits = dst.config.s_fractal_size
        if pto.constexpr(s_frac_bits == BLOCK_BYTE_BITS):
            valid_cols_align = c0_size
        else:
            valid_cols_align = FRACTAL_NZ_ROW
    else:
        valid_cols_align = c0_size
    valid_cols = (valid_cols_raw + valid_cols_align - 1) // valid_cols_align * valid_cols_align

    dst_rows = dst.shape[0]
    col_block = index_col // c0_size
    col_mod = index_col - col_block * c0_size
    dst_offset = dst_rows * c0_size * col_block + index_row * c0_size + col_mod
    dst_ptr = pto.addptr(dst_ptr, dst_offset)

    dst_stride = dst_rows * c0_size * elem_bytes
    src_stride = (valid_rows + 15) // 16 * 16 * elem_bytes

    relu_mode_name = pto.get_op_attr("relu_pre_mode", "no_relu")
    acc_mode_name = pto.get_op_attr("acc_to_vec_mode", "single_mode_vec0")

    dst_mode = 0
    if pto.constexpr(acc_mode_name == "single_mode_vec1"):
        dst_mode = 1
    elif pto.constexpr(acc_mode_name == "dual_mode_split_m"):
        dst_mode = "split_m"
    elif pto.constexpr(acc_mode_name == "dual_mode_split_n"):
        dst_mode = "split_n"

    if pto.constexpr(relu_mode_name == "normal_relu"):
        pto.mte_l0c_ub(
            src_ptr, dst_ptr,
            valid_rows, valid_cols, src_stride, dst_stride, dst_mode,
            layout=("nz2nz", pto.i64(0)),
            pre_relu=("normal_relu", None, None),
        )
    else:
        pto.mte_l0c_ub(
            src_ptr, dst_ptr,
            valid_rows, valid_cols, src_stride, dst_stride, dst_mode,
            layout=("nz2nz", pto.i64(0)),
        )
    return None


# ---------------------------------------------------------------------------
# Vec -> Vec (ND, ROW_MAJOR + NONE_BOX) - O1
# ---------------------------------------------------------------------------


_VEC_TO_VEC_DTYPES = [
    (pto.f16, pto.i64, pto.i64, pto.f16),
    (pto.bf16, pto.i64, pto.i64, pto.bf16),
    (pto.f32, pto.i64, pto.i64, pto.f32),
    (pto.i32, pto.i64, pto.i64, pto.i32),
    (pto.i8, pto.i64, pto.i64, pto.i8),
]

_VEC_TO_VEC_BASIC_DTYPES = [
    (pto.f16, pto.i32, pto.i32, pto.f16),
    (pto.bf16, pto.i32, pto.i32, pto.bf16),
    (pto.f32, pto.i32, pto.i32, pto.f32),
    (pto.i32, pto.i32, pto.i32, pto.i32),
    (pto.i8, pto.i32, pto.i32, pto.i8),
]

_VEC_TO_MAT_SPLIT_DTYPES = [
    (pto.f16, pto.i32, pto.i32, pto.f16),
    (pto.bf16, pto.i32, pto.i32, pto.bf16),
    (pto.f32, pto.i32, pto.i32, pto.f32),
    (pto.i32, pto.i32, pto.i32, pto.i32),
    (pto.i8, pto.i32, pto.i32, pto.i8),
]


@pto.vkernel(
    target="a5",
    op="pto.tinsert",
    dtypes=_VEC_TO_VEC_DTYPES,
    constraints=[_vec_to_vec_nd_constraint],
    advanced=True,
)
def template_tinsert_vec_to_vec_nd(
    src: pto.Tile,
    index_row: pto.i64, index_col: pto.i64,
    dst: pto.Tile,
):
    dtype = dst.element_type
    elem_bytes = pto.bytewidth(dtype)
    lanes = pto.get_lanes(dtype)

    valid_rows, valid_cols = src.valid_shape
    src_stride = src.shape[1]
    dst_stride = dst.shape[1]

    src_ptr = src.as_ptr()

    dst_offset = index_row * dst_stride + index_col
    dst_ptr = pto.addptr(dst.as_ptr(), dst_offset)

    src_stride_bytes = src_stride * elem_bytes
    dst_stride_bytes = dst_stride * elem_bytes
    strides_aligned = src_stride_bytes % BLOCK_BYTE_SIZE == 0 and dst_stride_bytes % BLOCK_BYTE_SIZE == 0

    if pto.constexpr(strides_aligned):
        if index_col * elem_bytes % BLOCK_BYTE_SIZE == 0:
            if pto.constexpr(valid_cols * elem_bytes % BLOCK_BYTE_SIZE == 0):
                row_bytes = valid_cols * elem_bytes
                total_bytes = valid_rows * row_bytes
                row_burst_len = row_bytes // BLOCK_BYTE_SIZE
                if pto.constexpr(valid_cols == src_stride and valid_cols == dst_stride and total_bytes >= BLOCK_BYTE_SIZE):
                    burst_len = total_bytes // BLOCK_BYTE_SIZE
                    pto.copy_ubuf_to_ubuf(src_ptr, dst_ptr, 0, 1, burst_len, 0, 0)
                elif pto.constexpr(row_bytes >= BLOCK_BYTE_SIZE):
                    src_gap = (src_stride - valid_cols) * elem_bytes // BLOCK_BYTE_SIZE
                    dst_gap = (dst_stride - valid_cols) * elem_bytes // BLOCK_BYTE_SIZE
                    pto.copy_ubuf_to_ubuf(src_ptr, dst_ptr, 0, valid_rows, row_burst_len, src_gap, dst_gap)
                else:
                    burst_len = (total_bytes + BLOCK_BYTE_SIZE - 1) // BLOCK_BYTE_SIZE
                    pto.copy_ubuf_to_ubuf(src_ptr, dst_ptr, 0, 1, burst_len, 0, 0)
            else:
                repeat_times = (valid_cols + lanes - 1) // lanes
                for i in range(0, valid_rows, 1):
                    remained = valid_cols
                    for j in range(0, repeat_times, 1):
                        pred, remained = pto.make_mask(dtype, remained)
                        src_off = i * src_stride + j * lanes
                        dst_off = i * dst_stride + j * lanes
                        data = pto.vlds(pto.addptr(src_ptr, src_off), 0)
                        pto.vsts(data, pto.addptr(dst_ptr, dst_off), 0, pred)
        else:
            full_repeats = valid_cols // lanes
            remainder = valid_cols % lanes
            for i in range(0, valid_rows, 1):
                ureg = pto.init_align()
                src_row_off = i * src_stride
                dst_row_ptr = pto.addptr(dst_ptr, i * dst_stride)
                for j in range(0, full_repeats, 1):
                    data = pto.vlds(pto.addptr(src_ptr, src_row_off + j * lanes), 0)
                    ureg = pto.vstus(ureg, lanes, data, dst_row_ptr)
                    dst_row_ptr = pto.addptr(dst_row_ptr, lanes)
                if pto.constexpr(remainder > 0):
                    data = pto.vlds(pto.addptr(src_ptr, src_row_off + full_repeats * lanes), 0)
                    ureg = pto.vstus(ureg, remainder, data, dst_row_ptr)
                pto.vstas(ureg, dst_row_ptr, 0)
    else:
        full_repeats = valid_cols // lanes
        remainder = valid_cols % lanes
        for i in range(0, valid_rows, 1):
            ureg = pto.init_align()
            src_row_off = i * src_stride
            dst_row_ptr = pto.addptr(dst_ptr, i * dst_stride)
            for j in range(0, full_repeats, 1):
                data = pto.vlds(pto.addptr(src_ptr, src_row_off + j * lanes), 0)
                ureg = pto.vstus(ureg, lanes, data, dst_row_ptr)
                dst_row_ptr = pto.addptr(dst_row_ptr, lanes)
            if pto.constexpr(remainder > 0):
                data = pto.vlds(pto.addptr(src_ptr, src_row_off + full_repeats * lanes), 0)
                ureg = pto.vstus(ureg, remainder, data, dst_row_ptr)
            pto.vstas(ureg, dst_row_ptr, 0)
    return None


@pto.vkernel(
    target="a5",
    op="pto.tinsert",
    dtypes=_VEC_TO_VEC_BASIC_DTYPES,
    constraints=[_vec_to_vec_nd_constraint],
    advanced=True,
)
def template_tinsert_vec_to_vec_nd_basic(
    src: pto.Tile,
    index_row: pto.i32, index_col: pto.i32,
    dst: pto.Tile,
):
    dtype = dst.element_type
    elem_bytes = pto.bytewidth(dtype)
    lanes = pto.get_lanes(dtype)

    valid_rows, valid_cols = src.valid_shape
    src_stride = src.shape[1]
    dst_stride = dst.shape[1]

    src_ptr = src.as_ptr()

    dst_offset = index_row * dst_stride + index_col
    dst_ptr = pto.addptr(dst.as_ptr(), dst_offset)

    src_stride_bytes = src_stride * elem_bytes
    dst_stride_bytes = dst_stride * elem_bytes
    strides_aligned = src_stride_bytes % BLOCK_BYTE_SIZE == 0 and dst_stride_bytes % BLOCK_BYTE_SIZE == 0

    if pto.constexpr(strides_aligned):
        if index_col * elem_bytes % BLOCK_BYTE_SIZE == 0:
            if pto.constexpr(valid_cols * elem_bytes % BLOCK_BYTE_SIZE == 0):
                row_bytes = valid_cols * elem_bytes
                total_bytes = valid_rows * row_bytes
                row_burst_len = row_bytes // BLOCK_BYTE_SIZE
                if pto.constexpr(valid_cols == src_stride and valid_cols == dst_stride and total_bytes >= BLOCK_BYTE_SIZE):
                    burst_len = total_bytes // BLOCK_BYTE_SIZE
                    pto.copy_ubuf_to_ubuf(src_ptr, dst_ptr, 0, 1, burst_len, 0, 0)
                elif pto.constexpr(row_bytes >= BLOCK_BYTE_SIZE):
                    src_gap = (src_stride - valid_cols) * elem_bytes // BLOCK_BYTE_SIZE
                    dst_gap = (dst_stride - valid_cols) * elem_bytes // BLOCK_BYTE_SIZE
                    pto.copy_ubuf_to_ubuf(src_ptr, dst_ptr, 0, valid_rows, row_burst_len, src_gap, dst_gap)
                else:
                    burst_len = (total_bytes + BLOCK_BYTE_SIZE - 1) // BLOCK_BYTE_SIZE
                    pto.copy_ubuf_to_ubuf(src_ptr, dst_ptr, 0, 1, burst_len, 0, 0)
            else:
                repeat_times = (valid_cols + lanes - 1) // lanes
                for i in range(0, valid_rows, 1):
                    remained = valid_cols
                    for j in range(0, repeat_times, 1):
                        pred, remained = pto.make_mask(dtype, remained)
                        src_off = i * src_stride + j * lanes
                        dst_off = i * dst_stride + j * lanes
                        data = pto.vlds(pto.addptr(src_ptr, src_off), 0)
                        pto.vsts(data, pto.addptr(dst_ptr, dst_off), 0, pred)
        else:
            full_repeats = valid_cols // lanes
            remainder = valid_cols % lanes
            for i in range(0, valid_rows, 1):
                ureg = pto.init_align()
                src_row_off = i * src_stride
                dst_row_ptr = pto.addptr(dst_ptr, i * dst_stride)
                for j in range(0, full_repeats, 1):
                    data = pto.vlds(pto.addptr(src_ptr, src_row_off + j * lanes), 0)
                    ureg = pto.vstus(ureg, lanes, data, dst_row_ptr)
                    dst_row_ptr = pto.addptr(dst_row_ptr, lanes)
                if pto.constexpr(remainder > 0):
                    data = pto.vlds(pto.addptr(src_ptr, src_row_off + full_repeats * lanes), 0)
                    ureg = pto.vstus(ureg, remainder, data, dst_row_ptr)
                pto.vstas(ureg, dst_row_ptr, 0)
    else:
        full_repeats = valid_cols // lanes
        remainder = valid_cols % lanes
        for i in range(0, valid_rows, 1):
            ureg = pto.init_align()
            src_row_off = i * src_stride
            dst_row_ptr = pto.addptr(dst_ptr, i * dst_stride)
            for j in range(0, full_repeats, 1):
                data = pto.vlds(pto.addptr(src_ptr, src_row_off + j * lanes), 0)
                ureg = pto.vstus(ureg, lanes, data, dst_row_ptr)
                dst_row_ptr = pto.addptr(dst_row_ptr, lanes)
            if pto.constexpr(remainder > 0):
                data = pto.vlds(pto.addptr(src_ptr, src_row_off + full_repeats * lanes), 0)
                ureg = pto.vstus(ureg, remainder, data, dst_row_ptr)
            pto.vstas(ureg, dst_row_ptr, 0)
    return None


@pto.vkernel(
    target="a5",
    op="pto.tinsert",
    dtypes=_VEC_TO_VEC_DTYPES,
    constraints=[_vec_to_vec_nd_scalar_constraint],
    advanced=True,
)
def template_tinsert_vec_to_vec_nd_scalar(
    src: pto.Tile,
    index_row: pto.i64, index_col: pto.i64,
    dst: pto.Tile,
):
    dst_stride = dst.shape[1]
    src_ptr = src.as_ptr()
    dst_ptr = dst.as_ptr()
    src_val = pto.load_scalar(src_ptr, 0)
    dst_elem_offset = index_row * dst_stride + index_col
    pto.store_scalar(src_val, dst_ptr, dst_elem_offset)
    return None


@pto.vkernel(
    target="a5",
    op="pto.tinsert",
    dtypes=_VEC_TO_VEC_BASIC_DTYPES,
    constraints=[_vec_to_vec_nd_scalar_constraint],
    advanced=True,
)
def template_tinsert_vec_to_vec_nd_scalar_basic(
    src: pto.Tile,
    index_row: pto.i32, index_col: pto.i32,
    dst: pto.Tile,
):
    dst_stride = dst.shape[1]
    src_ptr = src.as_ptr()
    dst_ptr = dst.as_ptr()
    src_val = pto.load_scalar(src_ptr, 0)
    dst_elem_offset = index_row * dst_stride + index_col
    pto.store_scalar(src_val, dst_ptr, dst_elem_offset)
    return None


# ---------------------------------------------------------------------------
# Vec -> Vec (NZ, COL_MAJOR + ROW_MAJOR) - O2
# ---------------------------------------------------------------------------


@pto.vkernel(
    target="a5",
    op="pto.tinsert",
    dtypes=_VEC_TO_VEC_DTYPES,
    constraints=[_vec_to_vec_nz_constraint],
    advanced=True,
)
def template_tinsert_vec_to_vec_nz(
    src: pto.Tile,
    index_row: pto.i64, index_col: pto.i64,
    dst: pto.Tile,
):
    dtype = dst.element_type
    elem_bytes = pto.bytewidth(dtype)
    c0_standard = BLOCK_BYTE_SIZE // elem_bytes
    s_frac_bits = dst.config.s_fractal_size
    if pto.constexpr(s_frac_bits == 2 * BLOCK_BYTE_BITS):
        c0_size = 2 * c0_standard
    else:
        c0_size = c0_standard

    valid_rows = src.valid_shape[0]
    valid_cols = src.valid_shape[1]
    dst_rows = dst.shape[0]

    src_ptr = src.as_ptr()

    col_block = index_col // c0_size
    col_mod = index_col - col_block * c0_size
    dst_offset = dst_rows * c0_size * col_block + index_row * c0_size + col_mod
    dst_ptr = pto.addptr(dst.as_ptr(), dst_offset)

    burst_num = (valid_cols + c0_size - 1) // c0_size
    burst_len = valid_rows * c0_size * elem_bytes // BLOCK_BYTE_SIZE

    compact = src.config.compact_mode
    if pto.constexpr(compact == pto.CompactMode.NULL):
        src_stride_rows = src.shape[0]
    elif pto.constexpr(compact == pto.CompactMode.ROW_PLUS_ONE):
        src_stride_rows = (valid_rows + FRACTAL_NZ_ROW - 1) // FRACTAL_NZ_ROW * FRACTAL_NZ_ROW + 1
    else:
        src_stride_rows = (valid_rows + FRACTAL_NZ_ROW - 1) // FRACTAL_NZ_ROW * FRACTAL_NZ_ROW
    src_gap = src_stride_rows - valid_rows
    dst_gap = dst_rows - valid_rows

    pto.copy_ubuf_to_ubuf(src_ptr, dst_ptr, 0, burst_num, burst_len, src_gap, dst_gap)
    return None


@pto.vkernel(
    target="a5",
    op="pto.tinsert",
    dtypes=_VEC_TO_VEC_BASIC_DTYPES,
    constraints=[_vec_to_vec_nz_constraint],
    advanced=True,
)
def template_tinsert_vec_to_vec_nz_basic(
    src: pto.Tile,
    index_row: pto.i32, index_col: pto.i32,
    dst: pto.Tile,
):
    dtype = dst.element_type
    elem_bytes = pto.bytewidth(dtype)
    c0_standard = BLOCK_BYTE_SIZE // elem_bytes
    s_frac_bits = dst.config.s_fractal_size
    if pto.constexpr(s_frac_bits == 2 * BLOCK_BYTE_BITS):
        c0_size = 2 * c0_standard
    else:
        c0_size = c0_standard

    valid_rows = src.valid_shape[0]
    valid_cols = src.valid_shape[1]
    dst_rows = dst.shape[0]

    src_ptr = src.as_ptr()

    col_block = index_col // c0_size
    col_mod = index_col - col_block * c0_size
    dst_offset = dst_rows * c0_size * col_block + index_row * c0_size + col_mod
    dst_ptr = pto.addptr(dst.as_ptr(), dst_offset)

    burst_num = (valid_cols + c0_size - 1) // c0_size
    burst_len = valid_rows * c0_size * elem_bytes // BLOCK_BYTE_SIZE

    compact = src.config.compact_mode
    if pto.constexpr(compact == pto.CompactMode.NULL):
        src_stride_rows = src.shape[0]
    elif pto.constexpr(compact == pto.CompactMode.ROW_PLUS_ONE):
        src_stride_rows = (valid_rows + FRACTAL_NZ_ROW - 1) // FRACTAL_NZ_ROW * FRACTAL_NZ_ROW + 1
    else:
        src_stride_rows = (valid_rows + FRACTAL_NZ_ROW - 1) // FRACTAL_NZ_ROW * FRACTAL_NZ_ROW
    src_gap = src_stride_rows - valid_rows
    dst_gap = dst_rows - valid_rows

    pto.copy_ubuf_to_ubuf(src_ptr, dst_ptr, 0, burst_num, burst_len, src_gap, dst_gap)
    return None


# ---------------------------------------------------------------------------
# Vec -> Mat (NZ, COL_MAJOR + ROW_MAJOR) - O3
# ---------------------------------------------------------------------------


_VEC_TO_MAT_DTYPES = [
    (pto.f16, pto.i64, pto.i64, pto.f16),
    (pto.bf16, pto.i64, pto.i64, pto.bf16),
    (pto.f32, pto.i64, pto.i64, pto.f32),
    (pto.i32, pto.i64, pto.i64, pto.i32),
    (pto.i8, pto.i64, pto.i64, pto.i8),
]

_VEC_TO_MAT_BASIC_DTYPES = [
    (pto.f16, pto.i32, pto.i32, pto.f16),
    (pto.bf16, pto.i32, pto.i32, pto.bf16),
    (pto.f32, pto.i32, pto.i32, pto.f32),
    (pto.i32, pto.i32, pto.i32, pto.i32),
    (pto.i8, pto.i32, pto.i32, pto.i8),
]


@pto.vkernel(
    target="a5",
    op="pto.tinsert",
    dtypes=_VEC_TO_MAT_DTYPES,
    constraints=[_vec_to_mat_nz_constraint],
    advanced=True,
)
def template_tinsert_vec_to_mat_nz(
    src: pto.Tile,
    index_row: pto.i64, index_col: pto.i64,
    dst: pto.Tile,
):
    dtype = dst.element_type
    elem_bytes = pto.bytewidth(dtype)
    c0_standard = BLOCK_BYTE_SIZE // elem_bytes
    s_frac_bits = dst.config.s_fractal_size
    if pto.constexpr(s_frac_bits == 2 * BLOCK_BYTE_BITS):
        c0_size = 2 * c0_standard
    else:
        c0_size = c0_standard

    valid_rows = src.valid_shape[0]
    valid_cols = src.valid_shape[1]
    dst_rows = dst.shape[0]

    src_ptr = src.as_ptr()

    col_block = index_col // c0_size
    col_mod = index_col - col_block * c0_size
    dst_offset = dst_rows * c0_size * col_block + index_row * c0_size + col_mod
    dst_ptr = pto.addptr(dst.as_ptr(), dst_offset)

    burst_num = (valid_cols + c0_size - 1) // c0_size
    burst_len = valid_rows * c0_size * elem_bytes // BLOCK_BYTE_SIZE

    compact = src.config.compact_mode
    if pto.constexpr(compact == pto.CompactMode.NULL):
        src_stride_rows = src.shape[0]
    elif pto.constexpr(compact == pto.CompactMode.ROW_PLUS_ONE):
        src_stride_rows = (valid_rows + FRACTAL_NZ_ROW - 1) // FRACTAL_NZ_ROW * FRACTAL_NZ_ROW + 1
    else:
        src_stride_rows = (valid_rows + FRACTAL_NZ_ROW - 1) // FRACTAL_NZ_ROW * FRACTAL_NZ_ROW
    src_gap = src_stride_rows - valid_rows
    dst_gap = dst_rows - valid_rows

    pto.mte_ub_l1(src_ptr, dst_ptr, burst_len, nburst=(burst_num, src_gap, dst_gap))
    return None


@pto.vkernel(
    target="a5",
    op="pto.tinsert",
    dtypes=_VEC_TO_MAT_BASIC_DTYPES,
    constraints=[_vec_to_mat_nz_constraint],
    advanced=True,
)
def template_tinsert_vec_to_mat_nz_basic(
    src: pto.Tile,
    index_row: pto.i32, index_col: pto.i32,
    dst: pto.Tile,
):
    dtype = dst.element_type
    elem_bytes = pto.bytewidth(dtype)
    c0_standard = BLOCK_BYTE_SIZE // elem_bytes
    s_frac_bits = dst.config.s_fractal_size
    if pto.constexpr(s_frac_bits == 2 * BLOCK_BYTE_BITS):
        c0_size = 2 * c0_standard
    else:
        c0_size = c0_standard

    valid_rows = src.valid_shape[0]
    valid_cols = src.valid_shape[1]
    dst_rows = dst.shape[0]

    src_ptr = src.as_ptr()

    col_block = index_col // c0_size
    col_mod = index_col - col_block * c0_size
    dst_offset = dst_rows * c0_size * col_block + index_row * c0_size + col_mod
    dst_ptr = pto.addptr(dst.as_ptr(), dst_offset)

    burst_num = (valid_cols + c0_size - 1) // c0_size
    burst_len = valid_rows * c0_size * elem_bytes // BLOCK_BYTE_SIZE

    compact = src.config.compact_mode
    if pto.constexpr(compact == pto.CompactMode.NULL):
        src_stride_rows = src.shape[0]
    elif pto.constexpr(compact == pto.CompactMode.ROW_PLUS_ONE):
        src_stride_rows = (valid_rows + FRACTAL_NZ_ROW - 1) // FRACTAL_NZ_ROW * FRACTAL_NZ_ROW + 1
    else:
        src_stride_rows = (valid_rows + FRACTAL_NZ_ROW - 1) // FRACTAL_NZ_ROW * FRACTAL_NZ_ROW
    src_gap = src_stride_rows - valid_rows
    dst_gap = dst_rows - valid_rows

    pto.mte_ub_l1(src_ptr, dst_ptr, burst_len, nburst=(burst_num, src_gap, dst_gap))
    return None


# ---------------------------------------------------------------------------
# Vec -> Mat (ND, ROW_MAJOR + NONE_BOX) - O4
# ---------------------------------------------------------------------------


@pto.vkernel(
    target="a5",
    op="pto.tinsert",
    dtypes=_VEC_TO_MAT_DTYPES,
    constraints=[_vec_to_mat_nd_constraint],
    advanced=True,
)
def template_tinsert_vec_to_mat_nd(
    src: pto.Tile,
    index_row: pto.i64, index_col: pto.i64,
    dst: pto.Tile,
):
    dtype = dst.element_type
    elem_bytes = pto.bytewidth(dtype)

    valid_rows, valid_cols = src.valid_shape
    src_cols = src.shape[1]
    dst_cols = dst.shape[1]

    src_ptr = src.as_ptr()

    dst_offset = index_row * dst_cols + index_col
    dst_ptr = pto.addptr(dst.as_ptr(), dst_offset)

    row_bytes = valid_cols * elem_bytes
    total_bytes = valid_rows * row_bytes

    if pto.constexpr(valid_cols == src_cols and valid_cols == dst_cols and total_bytes >= BLOCK_BYTE_SIZE):
        burst_len = total_bytes // BLOCK_BYTE_SIZE
        pto.mte_ub_l1(src_ptr, dst_ptr, burst_len, nburst=(1, 0, 0))
    elif pto.constexpr(row_bytes >= BLOCK_BYTE_SIZE):
        row_burst_len = row_bytes // BLOCK_BYTE_SIZE
        src_row_gap = (src_cols - valid_cols) * elem_bytes // BLOCK_BYTE_SIZE
        dst_row_gap = (dst_cols - valid_cols) * elem_bytes // BLOCK_BYTE_SIZE
        pto.mte_ub_l1(src_ptr, dst_ptr, row_burst_len, nburst=(valid_rows, src_row_gap, dst_row_gap))
    else:
        burst_len = (total_bytes + BLOCK_BYTE_SIZE - 1) // BLOCK_BYTE_SIZE
        pto.mte_ub_l1(src_ptr, dst_ptr, burst_len, nburst=(1, 0, 0))
    return None


@pto.vkernel(
    target="a5",
    op="pto.tinsert",
    dtypes=_VEC_TO_MAT_BASIC_DTYPES,
    constraints=[_vec_to_mat_nd_constraint],
    advanced=True,
)
def template_tinsert_vec_to_mat_nd_basic(
    src: pto.Tile,
    index_row: pto.i32, index_col: pto.i32,
    dst: pto.Tile,
):
    dtype = dst.element_type
    elem_bytes = pto.bytewidth(dtype)

    valid_rows, valid_cols = src.valid_shape
    src_cols = src.shape[1]
    dst_cols = dst.shape[1]

    src_ptr = src.as_ptr()

    dst_offset = index_row * dst_cols + index_col
    dst_ptr = pto.addptr(dst.as_ptr(), dst_offset)

    row_bytes = valid_cols * elem_bytes
    total_bytes = valid_rows * row_bytes

    if pto.constexpr(valid_cols == src_cols and valid_cols == dst_cols and total_bytes >= BLOCK_BYTE_SIZE):
        burst_len = total_bytes // BLOCK_BYTE_SIZE
        pto.mte_ub_l1(src_ptr, dst_ptr, burst_len, nburst=(1, 0, 0))
    elif pto.constexpr(row_bytes >= BLOCK_BYTE_SIZE):
        row_burst_len = row_bytes // BLOCK_BYTE_SIZE
        src_row_gap = (src_cols - valid_cols) * elem_bytes // BLOCK_BYTE_SIZE
        dst_row_gap = (dst_cols - valid_cols) * elem_bytes // BLOCK_BYTE_SIZE
        pto.mte_ub_l1(src_ptr, dst_ptr, row_burst_len, nburst=(valid_rows, src_row_gap, dst_row_gap))
    else:
        burst_len = (total_bytes + BLOCK_BYTE_SIZE - 1) // BLOCK_BYTE_SIZE
        pto.mte_ub_l1(src_ptr, dst_ptr, burst_len, nburst=(1, 0, 0))
    return None


# ---------------------------------------------------------------------------
# Vec -> Mat (NZ, Split2/Split4) - O5
# Split large-tile NZ DMA into 2/4 independent segments to reduce L1 bank
# conflicts (mirrors pto-isa TInsertMode::SPLIT2 / SPLIT4).
# ---------------------------------------------------------------------------


def _make_split_template(split_count):
    @pto.vkernel(
        target="a5",
        op="pto.tinsert",
        dtypes=_VEC_TO_MAT_SPLIT_DTYPES,
        name=f"template_tinsert_vec_to_mat_nz_split{split_count}",
        constraints=[_vec_to_mat_nz_constraint],
        advanced=True,
    )
    def _split_fn(src: pto.Tile, index_row: pto.i32, index_col: pto.i32, dst: pto.Tile):
        dtype = dst.element_type
        elem_bytes = pto.bytewidth(dtype)
        c0_standard = BLOCK_BYTE_SIZE // elem_bytes
        s_frac_bits = dst.config.s_fractal_size
        if pto.constexpr(s_frac_bits == 2 * BLOCK_BYTE_BITS):
            c0_size = 2 * c0_standard
        else:
            c0_size = c0_standard

        valid_rows = src.valid_shape[0]
        valid_cols = src.valid_shape[1]
        dst_rows = dst.shape[0]
        aligned_rows = (valid_rows + FRACTAL_NZ_ROW - 1) // FRACTAL_NZ_ROW * FRACTAL_NZ_ROW

        src_ptr = src.as_ptr()

        col_block = index_col // c0_size
        col_mod = index_col - col_block * c0_size
        dst_offset = dst_rows * c0_size * col_block + index_row * c0_size + col_mod
        dst_base = pto.addptr(dst.as_ptr(), dst_offset)

        total_burst_num = (valid_cols + c0_size - 1) // c0_size
        burst_len = aligned_rows * c0_size * elem_bytes // BLOCK_BYTE_SIZE

        compact = src.config.compact_mode
        if pto.constexpr(compact == pto.CompactMode.NULL):
            src_stride_rows = src.shape[0]
        elif pto.constexpr(compact == pto.CompactMode.ROW_PLUS_ONE):
            src_stride_rows = aligned_rows + 1
        else:
            src_stride_rows = aligned_rows
        src_gap = src_stride_rows - aligned_rows
        dst_gap = dst_rows - aligned_rows

        part_num = total_burst_num // split_count
        last_num = total_burst_num - part_num * (split_count - 1)
        src_block_size = (burst_len + src_gap) * BLOCK_BYTE_SIZE // elem_bytes
        dst_block_size = dst_rows * c0_size

        pto.mte_ub_l1(src_ptr, dst_base, burst_len, nburst=(part_num, src_gap, dst_gap))

        src_ptr1 = pto.addptr(src_ptr, part_num * src_block_size)
        dst_ptr1 = pto.addptr(dst_base, part_num * dst_block_size)
        if pto.constexpr(split_count == 2):
            pto.mte_ub_l1(src_ptr1, dst_ptr1, burst_len, nburst=(last_num, src_gap, dst_gap))
        else:
            pto.mte_ub_l1(src_ptr1, dst_ptr1, burst_len, nburst=(part_num, src_gap, dst_gap))

        if pto.constexpr(split_count == 4):
            src_ptr2 = pto.addptr(src_ptr, 2 * part_num * src_block_size)
            dst_ptr2 = pto.addptr(dst_base, 2 * part_num * dst_block_size)
            pto.mte_ub_l1(src_ptr2, dst_ptr2, burst_len, nburst=(part_num, src_gap, dst_gap))

            src_ptr3 = pto.addptr(src_ptr, 3 * part_num * src_block_size)
            dst_ptr3 = pto.addptr(dst_base, 3 * part_num * dst_block_size)
            pto.mte_ub_l1(src_ptr3, dst_ptr3, burst_len, nburst=(last_num, src_gap, dst_gap))
        return None

    return _split_fn


template_tinsert_vec_to_mat_nz_split2 = _make_split_template(2)
template_tinsert_vec_to_mat_nz_split4 = _make_split_template(4)
