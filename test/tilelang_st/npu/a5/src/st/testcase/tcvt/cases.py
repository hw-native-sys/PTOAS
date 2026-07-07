#!/usr/bin/python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

# coding=utf-8

"""Single source of truth for tcvt ST test cases.

`dtype` is kept for shared validation compatibility.
Actual data generation and comparison use `src_dtype` / `dst_dtype`.
"""

import numpy as np
from ml_dtypes import bfloat16

F8E4M3 = "f8e4m3"
F8E5M2 = "f8e5m2"
HIF8 = "hif8"
F4E1M2X2 = "f4e1m2x2"
F4E2M1X2 = "f4e2m1x2"

# 7 shapes (aligning with C++ INSTANTIATE_TCVT)
SHAPES = [
    (1, 128, 1, 128),
    (2, 64, 2, 64),
    (4, 32, 4, 32),
    (2, 128, 2, 128),
    (4, 128, 4, 65),   # Partial tiles
    (4, 256, 4, 200),  # Partial tiles
    (1, 256, 1, 129),  # Partial tiles
]

_DTYPE_NAME = {
    np.float32: "f32",
    np.float16: "f16",
    bfloat16: "bf16",
    np.int8: "si8",
    np.uint8: "ui8",
    np.int16: "i16",
    "si16": "si16",
    np.uint16: "ui16",
    np.int32: "i32",
    np.uint32: "ui32",
    np.int64: "i64",
    np.uint64: "ui64",
    F8E4M3: F8E4M3,
    F8E5M2: F8E5M2,
    HIF8: HIF8,
    F4E1M2X2: F4E1M2X2,
    F4E2M1X2: F4E2M1X2,
}

_PTO_DTYPE_NAME = {
    F8E4M3: "f8E4M3FN",
    F8E5M2: "f8E5M2",
    HIF8: "!pto.hif8",
    F4E1M2X2: "!pto.f4E1M2x2",
    F4E2M1X2: "!pto.f4E2M1x2",
}

_LOW_PRECISION_DTYPES = frozenset(_PTO_DTYPE_NAME)


def dtype_name(dtype):
    return _DTYPE_NAME.get(dtype, dtype)


def pto_dtype_name(dtype):
    return _PTO_DTYPE_NAME.get(dtype, dtype_name(dtype))


def is_low_precision_dtype(dtype):
    return dtype in _LOW_PRECISION_DTYPES


def storage_dtype(dtype):
    return np.uint8 if is_low_precision_dtype(dtype) else dtype


def eps_for_dtype(dtype):
    eps_map = {np.float32: 1e-6, np.float16: 1e-3, bfloat16: 1e-3}
    return eps_map.get(dtype, 0.0)


def _make_cases(src_dtype, dst_dtype):
    """Generate cases of 7 test shapes for src_dtype -> dst_dtype"""
    src_name = dtype_name(src_dtype)
    dst_name = dtype_name(dst_dtype)
    eps = eps_for_dtype(dst_dtype)
    
    cases = []
    for rows, cols, v_rows, v_cols in SHAPES:
        shape_name = f"{rows}x{cols}" if v_cols == cols else f"{v_rows}x{v_cols}"
        cases.append({
            "name": f"{src_name}_to_{dst_name}_{shape_name}",
            "dtype": dst_dtype,
            "src_dtype": src_dtype,
            "dst_dtype": dst_dtype,
            "shape": (rows, cols),
            "valid_shape": (v_rows, v_cols),
            "eps": eps,
        })
    return cases


def _make_low_precision_cases():
    """Targeted A5 low-precision tcvt coverage without multiplying all shapes."""
    shape = (16, 64)
    lowp_cases = []
    for src_dtype, dst_dtype in (
        (np.float32, F8E4M3),
        (np.float32, F8E5M2),
        (np.float32, HIF8),
        (np.float16, HIF8),
        (bfloat16, F4E1M2X2),
        (bfloat16, F4E2M1X2),
    ):
        case = {
            "name": f"{dtype_name(src_dtype)}_to_{dtype_name(dst_dtype)}_{shape[0]}x{shape[1]}",
            "dtype": storage_dtype(dst_dtype),
            "src_dtype": src_dtype,
            "dst_dtype": dst_dtype,
            "shape": shape,
            "valid_shape": shape,
            "eps": eps_for_dtype(dst_dtype),
        }
        if dst_dtype in (F4E1M2X2, F4E2M1X2):
            case["name"] = f"{dtype_name(src_dtype)}_to_{dtype_name(dst_dtype)}_16x64_to_16x32"
            case["dst_shape"] = (16, 32)
            case["dst_valid_shape"] = (16, 32)
        lowp_cases.append(case)
    for src_dtype, dst_dtype in (
        (np.float32, F8E4M3),
        (np.float32, HIF8),
    ):
        shape = (4, 96)
        lowp_cases.append({
            "name": f"{dtype_name(src_dtype)}_to_{dtype_name(dst_dtype)}_{shape[0]}x{shape[1]}",
            "dtype": storage_dtype(dst_dtype),
            "src_dtype": src_dtype,
            "dst_dtype": dst_dtype,
            "shape": shape,
            "valid_shape": shape,
            "eps": eps_for_dtype(dst_dtype),
        })
    return lowp_cases


CASES = [
    {
        "name": "f32_to_i32_rint_16x64",
        "dtype": np.int32,
        "src_dtype": np.float32,
        "dst_dtype": np.int32,
        "shape": (16, 64),
        "valid_shape": (16, 64),
        "round_mode": "RINT",
        "eps": 0.0,
    },
    {
        "name": "f32_to_i32_round_16x64",
        "dtype": np.int32,
        "src_dtype": np.float32,
        "dst_dtype": np.int32,
        "shape": (16, 64),
        "valid_shape": (16, 64),
        "round_mode": "ROUND",
        "eps": 0.0,
    },
    {
        "name": "i32_to_f32_rint_16x64",
        "dtype": np.float32,
        "src_dtype": np.int32,
        "dst_dtype": np.float32,
        "shape": (16, 64),
        "valid_shape": (16, 64),
        "round_mode": "RINT",
        "eps": 1e-6,
    },
    {
        "name": "f32_to_f16_rint_16x64",
        "dtype": np.float16,
        "src_dtype": np.float32,
        "dst_dtype": np.float16,
        "shape": (16, 64),
        "valid_shape": (16, 64),
        "round_mode": "RINT",
        "eps": 1e-3,
    },
    {
        "name": "f16_to_f32_rint_16x64",
        "dtype": np.float32,
        "src_dtype": np.float16,
        "dst_dtype": np.float32,
        "shape": (16, 64),
        "valid_shape": (16, 64),
        "round_mode": "RINT",
        "eps": 1e-6,
    },
    *_make_low_precision_cases(),
    # f32 → f16, bf16, i16, i32, i64, f32
    *_make_cases(np.float32, np.float16),
    *_make_cases(np.float32, bfloat16),
    *_make_cases(np.float32, np.int16),
    *_make_cases(np.float32, np.int32),
    *_make_cases(np.float32, np.int64),
    *_make_cases(np.float32, np.float32),
    # f16 → f32, i32, i16, si8, ui8
    *_make_cases(np.float16, np.float32),
    *_make_cases(np.float16, np.int32),
    *_make_cases(np.float16, np.int16),
    *_make_cases(np.float16, np.int8),
    *_make_cases(np.float16, np.uint8),
    # bf16 → f32, f16, i32
    *_make_cases(bfloat16, np.float32),
    *_make_cases(bfloat16, np.float16),
    *_make_cases(bfloat16, np.int32),
    # ui8 → f16, ui16
    *_make_cases(np.uint8, np.float16),
    *_make_cases(np.uint8, np.uint16),
    # si8 → f16, si16, i32
    *_make_cases(np.int8, np.float16),
    *_make_cases(np.int8, "si16"),
    *_make_cases(np.int8, np.int32),
    # i16 → ui8, f16, f32, ui32, i32
    *_make_cases(np.int16, np.uint8),
    *_make_cases(np.int16, np.float16),
    *_make_cases(np.int16, np.float32),
    *_make_cases(np.int16, np.uint32),
    *_make_cases(np.int16, np.int32),
    # i32 → f32, i16, i64, ui8, ui16
    *_make_cases(np.int32, np.float32),
    *_make_cases(np.int32, np.int16),
    *_make_cases(np.int32, np.int64),
    *_make_cases(np.int32, np.uint8),
    *_make_cases(np.int32, np.uint16),
    # ui32 → i16, ui16, ui8
    *_make_cases(np.uint32, np.int16),
    *_make_cases(np.uint32, np.uint16),
    *_make_cases(np.uint32, np.uint8),
    # i64 → f32, i32
    *_make_cases(np.int64, np.float32),
    *_make_cases(np.int64, np.int32),
]
