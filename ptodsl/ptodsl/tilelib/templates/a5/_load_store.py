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
MAT_LOAD_DTYPES = (("f16", "f16"), ("bf16", "bf16"), ("f32", "f32"))
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
        if dim is None:
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


def _check_load_bounds(src_shape, src_strides, dst_shape, dst_valid_shape, *, logical_rows, logical_cols=None, stride_axis=None, ranks=(5,)):
    if _view_rank(src_shape) not in ranks:
        return False
    if stride_axis is not None and not _known_eq(_stride_at(src_strides, stride_axis), 1):
        return False
    if not _known_le(dst_valid_shape[0], logical_rows):
        return False
    if not _known_le(logical_rows, dst_shape[0]):
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


def tload_nz2nz_constraint(src_kind, src_shape, src_memory_space, dst_kind, dst_shape, dst_valid_shape, dst_memory_space, dst_config, **_):
    if src_kind != "view" or dst_kind != "tile" or src_memory_space != "gm" or dst_memory_space not in {"ub", "vec"}:
        return False
    logical_rows = src_shape[2]
    return _is_tile_layout(dst_config, row_major=False, s_layout="row_major") and _check_load_bounds(
        src_shape,
        None,
        dst_shape,
        dst_valid_shape,
        logical_rows=logical_rows,
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


def tstore_nz_constraint(src_kind, src_shape, src_valid_shape, src_memory_space, src_config, dst_kind, dst_shape, dst_memory_space, **_):
    if src_kind != "tile" or dst_kind != "view" or src_memory_space not in {"ub", "vec"} or dst_memory_space != "gm":
        return False
    logical_rows = dst_shape[2] * dst_shape[3]
    logical_cols = dst_shape[0] * dst_shape[1] * dst_shape[4]
    return _is_tile_layout(src_config, row_major=False, s_layout="row_major") and _check_store_bounds(
        src_shape,
        src_valid_shape,
        dst_shape,
        None,
        logical_rows=logical_rows,
        logical_cols=logical_cols,
    )


def tload_mat_nd2nz_constraint(src_kind, src_shape, src_memory_space, dst_kind, dst_shape, dst_memory_space, dst_config, dst_dtype, **_):
    if src_kind != "view" or dst_kind != "tile" or src_memory_space != "gm" or dst_memory_space != "mat":
        return False
    if dst_config.b_layout != "col_major" or dst_config.s_layout != "row_major":
        return False
    if dst_dtype not in {"f16", "bf16", "f32"}:
        return False
    if _view_rank(src_shape) != 5:
        return False
    return _known_eq(src_shape[4], dst_shape[1])


def tload_mat_dn2nz_constraint(src_kind, src_shape, src_memory_space, dst_kind, dst_shape, dst_memory_space, dst_config, dst_dtype, **_):
    if src_kind != "view" or dst_kind != "tile" or src_memory_space != "gm" or dst_memory_space != "mat":
        return False
    if dst_config.b_layout != "col_major" or dst_config.s_layout != "row_major":
        return False
    if dst_dtype not in {"f16", "bf16", "f32"}:
        return False
    if _view_rank(src_shape) != 5:
        return False
    return _known_eq(src_shape[4], dst_shape[0])


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
