# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""Shared dtype signatures for cube TileLib templates."""

MATMUL_DTYPES = [
    ("f16", "f16", "f32"),
    ("bf16", "bf16", "f32"),
    ("f32", "f32", "f32"),
    ("i8", "i8", "i32"),
    ("f8e4m3", "f8e4m3", "f32"),
    ("f8e4m3", "f8e5m2", "f32"),
    ("f8e5m2", "f8e4m3", "f32"),
    ("f8e5m2", "f8e5m2", "f32"),
    ("hif8", "hif8", "f32"),
]

MATMUL_ACC_DTYPES = [
    ("f32", "f16", "f16", "f32"),
    ("f32", "bf16", "bf16", "f32"),
    ("f32", "f32", "f32", "f32"),
    ("i32", "i8", "i8", "i32"),
]

MATMUL_BIAS_DTYPES = [
    ("f16", "f16", "f32", "f32"),
    ("bf16", "bf16", "f32", "f32"),
    ("f32", "f32", "f32", "f32"),
    ("i8", "i8", "i32", "i32"),
    ("f8e4m3", "f8e4m3", "f32", "f32"),
    ("f8e4m3", "f8e5m2", "f32", "f32"),
    ("f8e5m2", "f8e4m3", "f32", "f32"),
    ("f8e5m2", "f8e5m2", "f32", "f32"),
    ("hif8", "hif8", "f32", "f32"),
]

LOW_PRECISION_PAIRS = (
    ("f8e4m3", "f8e4m3"),
    ("f8e4m3", "f8e5m2"),
    ("f8e5m2", "f8e4m3"),
    ("f8e5m2", "f8e5m2"),
    ("f4e1m2x2", "f4e1m2x2"),
    ("f4e1m2x2", "f4e2m1x2"),
    ("f4e2m1x2", "f4e1m2x2"),
    ("f4e2m1x2", "f4e2m1x2"),
)

MATMUL_MX_DTYPES = [
    (lhs, "f16", rhs, "f16", "f32")
    for lhs, rhs in LOW_PRECISION_PAIRS
]
MATMUL_MX_ACC_DTYPES = [
    ("f32", lhs, "f16", rhs, "f16", "f32")
    for lhs, rhs in LOW_PRECISION_PAIRS
]
MATMUL_MX_BIAS_DTYPES = [
    (lhs, "f16", rhs, "f16", "f32", "f32")
    for lhs, rhs in LOW_PRECISION_PAIRS
]


__all__ = [
    "MATMUL_DTYPES",
    "MATMUL_ACC_DTYPES",
    "MATMUL_BIAS_DTYPES",
    "MATMUL_MX_DTYPES",
    "MATMUL_MX_ACC_DTYPES",
    "MATMUL_MX_BIAS_DTYPES",
]
