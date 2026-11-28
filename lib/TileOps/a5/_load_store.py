# coding=utf-8
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Shared helpers for PTODSL TileLib load/store template ports."""

from ptodsl import pto

from ._common import NUMERIC_DTYPES
from ._cube import CUBE_MOVE_DTYPE_NAMES, CUBE_MOVE_DTYPES


INTEGER_LOAD_STORE_DTYPES = (
    "i64",
    "si8",
    "si16",
    "si32",
    "si64",
    "ui64",
)
LOW_PRECISION_LOAD_STORE_DTYPES = ("f8e4m3", "f8e5m2", "hif8", "f4e1m2x2", "f4e2m1x2")
LOAD_STORE_DTYPES = tuple(
    (dtype, dtype) for dtype in NUMERIC_DTYPES + INTEGER_LOAD_STORE_DTYPES + LOW_PRECISION_LOAD_STORE_DTYPES
)
# GM -> MAT tload packs the source matrix into the NZ fractal form.  The move
# itself is a dtype pass-through (the element type is carried by the pointer
# types), so the accepted dtypes are decided purely by the consumers that
# later read the MAT (L1) tile back out:
#   1. operand path -- pto.tmov m2l/m2r and pto.textract mat2left/right feed
#      the L0A/L0B multiply operands, so they accept the cube matmul operand
#      element types (CUBE_MOVE_DTYPE_NAMES).  i32 is NOT among them: no
#      matmul consumes i32 operands, and the m2l/m2r tables must keep
#      rejecting it.
#   2. bias path -- pto.tmov m2b stages cube bias vectors into the bt slots
#      via pto.mte_l1_bt (which supports i32 -> i32); the bias element type
#      follows the i32 accumulator of i8 x i8 -> i32 gemv_bias (#1742), so
#      i32 extends only this load table.
#   3. scaling path -- pto.tmov m2s feeds the fixpipe scale buffer but only
#      accepts f32, which the operand table already covers.
BIAS_STAGING_DTYPE_NAMES = ("i32",)
MAT_LOAD_DTYPE_NAMES = CUBE_MOVE_DTYPE_NAMES + BIAS_STAGING_DTYPE_NAMES
MAT_LOAD_DTYPES = tuple((name, name) for name in MAT_LOAD_DTYPE_NAMES)
ACC_STORE_DTYPES = (
    ("f32", "f32"),
    ("f32", "f16"),
    ("f32", "bf16"),
    ("i32", "i32"),
)


def _known_eq(lhs, rhs) -> bool:
    return _is_unknown_dim(lhs) or _is_unknown_dim(rhs) or lhs == rhs


def _known_le(lhs, rhs) -> bool:
    return _is_unknown_dim(lhs) or _is_unknown_dim(rhs) or lhs <= rhs


def _is_unknown_dim(value) -> bool:
    return value is None or value in {-1, -(2**63)}


def _shape_size(shape):
    result = 1
    for dim in shape:
        if _is_unknown_dim(dim):
            return None
        result *= dim
    return result


def _view_rank(shape):
    return len(shape) if shape is not None else None


def _stride_at(strides, index):
    if strides is None:
        return None
    return strides[index]


def _is_tile_layout(config, *, row_major: bool, s_layout: str) -> bool:
    if config is None:
        return False
    if row_major:
        return config.b_layout == "row_major" and config.s_layout == s_layout
    return config.b_layout != "row_major" and config.s_layout == s_layout


def _is_row_plus_one_tile(config) -> bool:
    # compact=2 (row_plus_one) pads each C0 column block with one extra row:
    # the physical pitch between blocks is (rows + 1) * 32 bytes, while the
    # NZ load/store templates step bursts by the dense rows * 32 pitch and
    # would silently corrupt the tile (cf. alloc_multi_tile's rejection of
    # row_plus_one slots for the same footprint reason).
    if config is None:
        return False
    mode = getattr(config, "compact_mode", None)
    if mode is None:
        return False
    mode = getattr(mode, "value", mode)
    if isinstance(mode, bool):
        return False
    if isinstance(mode, int):
        return mode == 2
    return str(mode).strip().lower() in {"2", "row_plus_one", "rowplusone"}


# Column-block width, in elements, of one NZ fractal per storage dtype
# (C0 = 32 / storage bytes); packed sub-byte dtypes count one storage element.
# An unknown dtype resolves to None and keeps the ND label rejected.
_NZ_C0_BY_DTYPE = {
    "f32": 8,
    "i32": 8,
    "ui32": 8,
    "si32": 8,
    "f16": 16,
    "bf16": 16,
    "i16": 16,
    "ui16": 16,
    "si16": 16,
    "i8": 32,
    "ui8": 32,
    "si8": 32,
    "f8e4m3": 32,
    "f8e5m2": 32,
    "hif8": 32,
    "f4e1m2x2": 32,
    "f4e2m1x2": 32,
    "i64": 4,
    "ui64": 4,
    "si64": 4,
}


def _nz_c0(dtype):
    # Column-block width, in elements, of one NZ fractal for a storage dtype.
    return _NZ_C0_BY_DTYPE.get(str(dtype).lower())


def _is_nz_dma_tile(config, c0) -> bool:
    if not _is_tile_layout(config, row_major=False, s_layout="row_major") or _is_row_plus_one_tile(config):
        return False
    # Fractal 512 uses a 32-byte inner column block. Fractal 1024 has a fixed
    # width of 16 columns, which matches the DMA block for two-byte elements.
    return config.s_fractal_size == 512 or (config.s_fractal_size == 1024 and c0 == 16)


def _conflicting_nz_view_layout(layout, shape5d, strides5d, c0) -> bool:
    # GM<->UB/VEC MTE moves bytes per the view strides and cannot convert
    # layouts, so an explicitly conflicting view layout must not reach the NZ
    # templates.  Layout inference, however, labels every unannotated 5D view
    # that is not NZ-canonical -- single-fractal shapes included -- as ND, so
    # an ND label is accepted only when the copy degenerates to one contiguous
    # burst over a single C0-wide column block: g0 * g1 == 1 and g4 == C0 with
    # dense row-major inner strides (s4 == 1, s3 == g4, s2 == g3 * g4).  The
    # template's len_burst moves rows * C0 elements in one go, so a wider or
    # gapped view would leave the tile under-filled or misaligned.  DN never
    # coincides with the fractal copy order.
    if layout in {None, "nz", "fractal"}:
        return False
    if layout in {"nd", "row_major"}:
        if _shape_size((shape5d[0], shape5d[1])) != 1:
            return True
        if c0 is None or shape5d[4] != c0 or _is_unknown_dim(shape5d[3]):
            return True
        if strides5d is None:
            return True
        if any(_is_unknown_dim(_stride_at(strides5d, index)) for index in (2, 3, 4)):
            return True
        return not (strides5d[4] == 1 and strides5d[3] == c0 and strides5d[2] == shape5d[3] * c0)
    return True


def _check_load_bounds(
    src_shape,
    src_strides,
    dst_shape,
    dst_valid_shape,
    *,
    logical_rows,
    logical_cols=None,
    stride_axis=None,
    ranks=(5,),
    require_rows_fit_tile=True,
):
    if _view_rank(src_shape) not in ranks:
        return False
    if stride_axis is not None and not _known_eq(_stride_at(src_strides, stride_axis), 1):
        return False
    if not _known_le(dst_valid_shape[0], logical_rows):
        return False
    if require_rows_fit_tile and not _known_le(logical_rows, dst_shape[0]):
        return False
    if not _known_le(dst_valid_shape[0], dst_shape[0]):
        return False
    if logical_cols is not None:
        if not _known_le(dst_valid_shape[1], logical_cols):
            return False
        if not _known_le(logical_cols, dst_shape[1]):
            return False
    if not _known_le(dst_valid_shape[1], dst_shape[1]):
        return False
    return True


def _check_store_bounds(src_shape, src_valid_shape, dst_shape, dst_strides, *, logical_rows, logical_cols, stride_axis=None, ranks=(5,)):
    if _view_rank(dst_shape) not in ranks:
        return False
    if stride_axis is not None and not _known_eq(_stride_at(dst_strides, stride_axis), 1):
        return False
    if not _known_eq(src_valid_shape[0], logical_rows):
        return False
    if not _known_eq(src_valid_shape[1], logical_cols):
        return False
    if not _known_le(src_valid_shape[0], src_shape[0]):
        return False
    if not _known_le(src_valid_shape[1], src_shape[1]):
        return False
    return True


def tload_nd2nd_constraint(src_kind, src_shape, src_strides, src_memory_space, dst_kind, dst_shape, dst_valid_shape, dst_memory_space, dst_config, **_):
    if src_kind != "view" or dst_kind != "tile" or src_memory_space != "gm" or dst_memory_space not in {"ub", "vec"}:
        return False
    if _view_rank(src_shape) == 2:
        logical_rows, logical_cols = src_shape
        stride_axis = 1
    else:
        logical_rows = _shape_size(src_shape[:4])
        logical_cols = src_shape[4]
        stride_axis = 4
    return _is_tile_layout(dst_config, row_major=True, s_layout="none_box") and _check_load_bounds(
        src_shape,
        src_strides,
        dst_shape,
        dst_valid_shape,
        logical_rows=logical_rows,
        logical_cols=logical_cols,
        stride_axis=stride_axis,
        ranks=(2, 5),
    )


def tload_dn2dn_constraint(src_kind, src_shape, src_strides, src_memory_space, dst_kind, dst_shape, dst_valid_shape, dst_memory_space, dst_config, **_):
    if src_kind != "view" or dst_kind != "tile" or src_memory_space != "gm" or dst_memory_space not in {"ub", "vec"}:
        return False
    if _view_rank(src_shape) == 2:
        logical_rows, logical_cols = src_shape
        stride_axis = 0
    else:
        logical_rows = src_shape[3]
        logical_cols = _shape_size((src_shape[0], src_shape[1], src_shape[2], src_shape[4]))
        stride_axis = 3
    return _is_tile_layout(dst_config, row_major=False, s_layout="none_box") and _check_load_bounds(
        src_shape,
        src_strides,
        dst_shape,
        dst_valid_shape,
        logical_rows=logical_rows,
        logical_cols=logical_cols,
        stride_axis=stride_axis,
        ranks=(2, 5),
    )


def tload_nz2nz_constraint(
    src_kind,
    src_shape,
    src_strides,
    src_memory_space,
    dst_kind,
    dst_shape,
    dst_valid_shape,
    dst_memory_space,
    dst_config,
    **_,
):
    # The optional view annotation/dtype context keys arrive through **_.
    src_layout = _.get("src_layout")
    src_dtype = _.get("src_dtype")
    if src_kind != "view" or dst_kind != "tile" or src_memory_space != "gm" or dst_memory_space not in {"ub", "vec"}:
        return False
    c0 = _nz_c0(src_dtype)
    if _view_rank(src_shape) != 5 or _conflicting_nz_view_layout(src_layout, src_shape, src_strides, c0):
        return False
    if not _is_nz_dma_tile(dst_config, c0):
        return False
    # A 5D NZ view [g0, g1, g2, g3, g4] covers g2 * g3 logical rows and
    # g0 * g1 * g4 logical columns.  The template iterates g0 burst groups of
    # g1 C0 column blocks stepping by s0/s1, so those column extents and
    # strides must be statically known.  The DMA length follows the tile's
    # valid rows, so the row extents may stay dynamic and the source may be
    # taller than the tile's physical rows -- only the valid prefix is read.
    logical_rows = _shape_size((src_shape[2], src_shape[3]))
    logical_cols = _shape_size((src_shape[0], src_shape[1], src_shape[4]))
    if logical_cols is None:
        return False
    if _is_unknown_dim(_stride_at(src_strides, 0)) or _is_unknown_dim(_stride_at(src_strides, 1)):
        return False
    return _check_load_bounds(
        src_shape,
        None,
        dst_shape,
        dst_valid_shape,
        logical_rows=logical_rows,
        logical_cols=logical_cols,
        require_rows_fit_tile=False,
    )


def tstore_nd_constraint(src_kind, src_shape, src_valid_shape, src_memory_space, src_config, dst_kind, dst_shape, dst_strides, dst_memory_space, **_):
    if src_kind != "tile" or dst_kind != "view" or src_memory_space not in {"ub", "vec"} or dst_memory_space != "gm":
        return False
    if _view_rank(dst_shape) == 2:
        logical_rows, logical_cols = dst_shape
        stride_axis = 1
    else:
        logical_rows = _shape_size(dst_shape[:4])
        logical_cols = dst_shape[4]
        stride_axis = 4
    return _is_tile_layout(src_config, row_major=True, s_layout="none_box") and _check_store_bounds(
        src_shape,
        src_valid_shape,
        dst_shape,
        dst_strides,
        logical_rows=logical_rows,
        logical_cols=logical_cols,
        stride_axis=stride_axis,
        ranks=(2, 5),
    )


def tstore_dn_constraint(src_kind, src_shape, src_valid_shape, src_memory_space, src_config, dst_kind, dst_shape, dst_strides, dst_memory_space, **_):
    if src_kind != "tile" or dst_kind != "view" or src_memory_space not in {"ub", "vec"} or dst_memory_space != "gm":
        return False
    logical_rows = dst_shape[3]
    logical_cols = _shape_size((dst_shape[0], dst_shape[1], dst_shape[2], dst_shape[4]))
    return _is_tile_layout(src_config, row_major=False, s_layout="none_box") and _check_store_bounds(
        src_shape,
        src_valid_shape,
        dst_shape,
        dst_strides,
        logical_rows=logical_rows,
        logical_cols=logical_cols,
        stride_axis=3,
    )


def tstore_nz_constraint(
    src_kind,
    src_shape,
    src_valid_shape,
    src_memory_space,
    src_config,
    dst_kind,
    dst_shape,
    dst_strides,
    dst_memory_space,
    **_,
):
    # The optional view annotation/dtype context keys arrive through **_.
    dst_layout = _.get("dst_layout")
    dst_dtype = _.get("dst_dtype")
    if src_kind != "tile" or dst_kind != "view" or src_memory_space not in {"ub", "vec"} or dst_memory_space != "gm":
        return False
    c0 = _nz_c0(dst_dtype)
    if _view_rank(dst_shape) != 5 or _conflicting_nz_view_layout(dst_layout, dst_shape, dst_strides, c0):
        return False
    if not _is_nz_dma_tile(src_config, c0):
        return False
    # The store iterates g0 burst groups of g1 C0 column blocks stepping by
    # s0/s1, so the column extents and strides must be statically known.  The
    # store DMA length is fixed by the tile's valid rows at compile time, so
    # the destination must statically prove equal row coverage: an unknown
    # row extent could be shorter at runtime and the fixed-length DMA would
    # write past the partition.
    logical_rows = _shape_size((dst_shape[2], dst_shape[3]))
    logical_cols = _shape_size((dst_shape[0], dst_shape[1], dst_shape[4]))
    if logical_cols is None or logical_rows is None:
        return False
    if _is_unknown_dim(_stride_at(dst_strides, 0)) or _is_unknown_dim(_stride_at(dst_strides, 1)):
        return False
    return _check_store_bounds(
        src_shape,
        src_valid_shape,
        dst_shape,
        None,
        logical_rows=logical_rows,
        logical_cols=logical_cols,
    )


def tload_mat_nd2nz_constraint(src_kind, src_shape, src_strides, src_memory_space, dst_kind, dst_shape, dst_memory_space, dst_config, dst_dtype, **_):
    if src_kind != "view" or dst_kind != "tile" or src_memory_space != "gm" or dst_memory_space != "mat":
        return False
    if dst_config.b_layout != "col_major" or dst_config.s_layout != "row_major":
        return False
    if dst_dtype not in MAT_LOAD_DTYPE_NAMES:
        return False
    if _view_rank(src_shape) != 5:
        return False
    # ND input is row-major: the innermost GM dimension is contiguous.  The
    # stride check also makes square views distinct from the DN2NZ form.
    return _known_eq(src_shape[4], dst_shape[1]) and _known_eq(
        _stride_at(src_strides, 4), 1
    )


def tload_mat_dn2nz_constraint(
    src_kind,
    src_shape,
    src_strides,
    src_memory_space,
    dst_kind,
    dst_shape,
    dst_valid_shape,
    dst_memory_space,
    dst_config,
    dst_dtype,
    **_,
):
    if src_kind != "view" or dst_kind != "tile" or src_memory_space != "gm" or dst_memory_space != "mat":
        return False
    if dst_config.b_layout != "col_major" or dst_config.s_layout != "row_major":
        return False
    if dst_dtype not in MAT_LOAD_DTYPE_NAMES:
        return False
    if _view_rank(src_shape) != 5:
        return False
    # DN keeps the logical [rows, cols] shape; strides make rows contiguous.
    # This template emits one matrix, not a batch of matrices.
    if any(dim != 1 for dim in src_shape[:3]):
        return False
    return _check_load_bounds(
        src_shape,
        src_strides,
        dst_shape,
        dst_valid_shape,
        logical_rows=src_shape[3],
        logical_cols=src_shape[4],
        stride_axis=3,
    )


def tstore_acc_base(src_kind, src_memory_space, src_dtype, dst_kind, dst_memory_space, **_):
    return (
        src_kind == "tile"
        and dst_kind == "view"
        and src_memory_space == "acc"
        and dst_memory_space == "gm"
        and src_dtype in {"f32", "i32", "si32"}
    )


def tstore_acc_nz2nd_constraint(dst_shape, dst_layout, **context):
    if not tstore_acc_base(**context):
        return False
    return dst_layout in {None, "nd", "row_major"} and _view_rank(dst_shape) == 5


def tstore_acc_nz2dn_constraint(dst_shape, dst_layout, **context):
    if not tstore_acc_base(**context):
        return False
    return dst_layout in {"dn", "col_major"}


def tstore_acc_nz2nz_constraint(dst_shape, dst_layout, **context):
    if not tstore_acc_base(**context):
        return False
    return dst_layout in {"nz", "fractal"}


def tstore_fp_constraint(src_kind, src_memory_space, src_dtype, fp_kind, fp_memory_space, dst_kind, dst_memory_space, **_):
    return (
        src_kind == "tile"
        and fp_kind == "tile"
        and dst_kind == "view"
        and src_memory_space == "acc"
        and fp_memory_space in {"scaling", "ub", "vec"}
        and dst_memory_space == "gm"
        and src_dtype == "f32"
    )


def _pad_token(value):
    aliases = {
        "null": "null",
        "0": "null",
        "0x0": "null",
        "0x00": "null",
        "zero": "zero",
        "1": "zero",
        "0x1": "zero",
        "0x01": "zero",
        "max": "max",
        "2": "max",
        "0x2": "max",
        "0x02": "max",
        "min": "min",
        "3": "min",
        "0x3": "min",
        "0x03": "min",
    }
    return aliases.get(str(value).lower(), str(value).lower())


def _pad_zero(dtype):
    name = str(dtype)
    if name == "f32":
        return pto.f32(0.0)
    if name == "f16":
        return pto.f16(0.0)
    if name == "bf16":
        return pto.bf16(0.0)
    if name in {"i64", "si64", "ui64", "i32", "si32", "ui32"}:
        return pto.i32(0)
    if name in {"i16", "si16", "ui16"}:
        return pto.i16(0)
    return pto.i8(0)


def _pad_max(dtype):
    name = str(dtype)
    if name == "f32":
        return pto.f32(3.4028234663852886e38)
    if name == "f16":
        return pto.f16(65504.0)
    if name == "bf16":
        return pto.bf16(3.3895313892515355e38)
    if name in {"i64", "si64"}:
        return pto.i32(2147483647)
    if name == "ui64":
        return pto.i32(-1)
    if name == "ui32":
        return pto.i32(-1)
    if name in {"i32", "si32"}:
        return pto.i32(2147483647)
    if name == "ui16":
        return pto.i16(-1)
    if name in {"i16", "si16"}:
        return pto.i16(32767)
    if name == "ui8":
        return pto.i8(-1)
    return pto.i8(127)


def _pad_min(dtype):
    name = str(dtype)
    if name == "f32":
        return pto.f32(-3.4028234663852886e38)
    if name == "f16":
        return pto.f16(-65504.0)
    if name == "bf16":
        return pto.bf16(-3.3895313892515355e38)
    if name in {"ui64", "ui32", "ui16", "ui8"}:
        return _pad_zero(dtype)
    if name in {"i64", "si64"}:
        return pto.i32(-2147483648)
    if name in {"i32", "si32"}:
        return pto.i32(-2147483648)
    if name in {"i16", "si16"}:
        return pto.i16(-32768)
    return pto.i8(-128)


def dma_pad_for(tile):
    pad_value = _pad_token(getattr(tile, "pad_value", "Null"))
    if pad_value == "null":
        return None
    if pad_value == "max":
        return (_pad_max(tile.dtype), 0, 0)
    if pad_value == "min":
        return (_pad_min(tile.dtype), 0, 0)
    return (_pad_zero(tile.dtype), 0, 0)
