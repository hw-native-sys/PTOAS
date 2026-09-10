# coding=utf-8
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""
e2e tests for A3 VPTO unary elementwise ops (tabs / trelu).

Run:
    pytest ptodsl/tests/e2e/test_unary_elementwise.py -v
"""

from __future__ import annotations

import pytest

from .common import UNARY_OPS, make_unary_kernel, launch_and_check_unary


# ---------------------------------------------------------------------------
# Shape matrix
# ---------------------------------------------------------------------------

F32_SHAPES: list[tuple[int, int, str]] = [
    (1, 64, "modeSmall"),
    (4, 64, "modeSmall-multi-row"),
    (1, 128, "modeNorm1L"),
    (16, 64, "modeNorm1L-16x64"),
    (64, 64, "modeNorm1L-64x64"),
]

F16_SHAPES: list[tuple[int, int, str]] = [
    (1, 64, "modeSmall-f16"),
    (4, 64, "modeSmall-multi-f16"),
    (1, 128, "modeNorm1L-f16"),
    (16, 128, "modeNorm1L-16x128-f16"),
    (64, 128, "modeNorm1L-64x128-f16"),
]


def _params(shapes, dtype_str):
    return [
        pytest.param(
            (op_name, ref_fn, rows, cols, desc),
            id=f"{op_name}-{dtype_str}-{rows}x{cols}-{desc}",
        )
        for op_name, (_, ref_fn) in UNARY_OPS.items()
        for rows, cols, desc in shapes
    ]


F32_PARAMS = _params(F32_SHAPES, "float32")
F16_PARAMS = _params(F16_SHAPES, "float16")


@pytest.mark.require_npu
@pytest.mark.parametrize("case", F32_PARAMS)
def test_unary_f32(case, torch, target_arch, backend):
    op_name, ref_fn, rows, cols, desc = case

    kernel = make_unary_kernel(
        op_name, rows, cols, dtype_str="float32",
        target=target_arch, backend=backend,
    )
    compile_s, launch_s = launch_and_check_unary(
        op_name=op_name,
        kernel_handle=kernel,
        ref_fn=ref_fn,
        shape=(rows, cols),
        dtype_str="float32",
        torch=torch,
    )
    print(f"  PASS {op_name} f32 {rows}x{cols} ({desc}) "
          f"compile={compile_s:.3f}s launch={launch_s:.3f}s")


@pytest.mark.require_npu
@pytest.mark.parametrize("case", F16_PARAMS)
def test_unary_f16(case, torch, target_arch, backend):
    op_name, ref_fn, rows, cols, desc = case

    kernel = make_unary_kernel(
        op_name, rows, cols, dtype_str="float16",
        target=target_arch, backend=backend,
    )
    compile_s, launch_s = launch_and_check_unary(
        op_name=op_name,
        kernel_handle=kernel,
        ref_fn=ref_fn,
        shape=(rows, cols),
        dtype_str="float16",
        torch=torch,
    )
    print(f"  PASS {op_name} f16 {rows}x{cols} ({desc}) "
          f"compile={compile_s:.3f}s launch={launch_s:.3f}s")
