# coding=utf-8
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""
e2e tests for A3 VPTO binary elementwise ops (tadd / tsub / tmul / tdiv).

Exercises every lowering code path in ``LowerPTOToUBufOps::dispatch`` via
a systematic shape × dtype matrix.

Run:
    pytest ptodsl/tests/e2e/test_binary_elementwise.py -v

Filter:
    pytest ptodsl/tests/e2e/test_binary_elementwise.py -v -k "add-float32-1x64"
"""

from __future__ import annotations

import pytest

from .common import BINARY_OPS, INT_OPS, SHIFT_OPS, make_binary_kernel, make_shift_kernel, launch_and_check, launch_and_check_int, launch_and_check_shift


# ---------------------------------------------------------------------------
# Shape matrix — each entry maps to a specific CodePath (see dispatch tree in
#   lib/PTO/Transforms/LowerPTOToUBufOps.cpp:702-743)
# ---------------------------------------------------------------------------

F32_EPR = 64  # elementsPerRepeat = 256 / sizeof(f32)

F32_SHAPES: list[tuple[int, int, str]] = [
    # modeSmall, single-row: rows≤255, cols < epr
    (1, 32, "modeSmall single-row"),
    # modeSmall, multi-row loop: rows≤255, cols < epr, vRows>1
    (4, 32, "modeSmall multi-row"),
    (11, 32, "modeSmall multi-row odd"),
    # modeNorm1L, single repeat: continuous, headRepeats=1, no tail
    (1, F32_EPR, "modeNorm1L single-repeat"),
    # modeNorm1L, multi-repeat no tail: continuous, headRepeats≥1, aligned
    (1, F32_EPR * 2, "modeNorm1L multi-repeat aligned"),
    (F32_EPR, F32_EPR, "modeNorm1L square aligned"),
    (16, F32_EPR, "modeNorm1L multi-row aligned"),
    (16, F32_EPR * 2, "modeNorm1L 16x128"),
    (4, F32_EPR * 4, "modeNorm1L 4x256"),
    (1, F32_EPR * 16, "modeNorm1L 16-repeat"),
    # Large tiles (exercises multi-repeat and UB bank layout)
    (16, 256, "modeNorm1L 16x256"),
    (32, 32, "modeSmall 32x32 square small"),
    # Non-epr-aligned shapes (exercises modeCount1L with repeat>0)
    (1, 200, "modeCount1L 1x200 nonVLAligned"),
    (4, 200, "modeCount1L 4x200 nonVLAligned"),
    (1, 96, "modeCount1L 1x96 nonVLAligned"),
]

F16_EPR = 128  # elementsPerRepeat = 256 / sizeof(f16)

F16_SHAPES: list[tuple[int, int, str]] = [
    (1, 64, "modeSmall 1x64 f16"),
    (4, 64, "modeSmall 4x64 f16"),
    (1, F16_EPR, "modeNorm1L single-repeat f16"),
    (16, F16_EPR, "modeNorm1L 16x128 f16"),
    (F16_EPR // 2, F16_EPR, "modeNorm1L multi-row f16"),
    (1, F16_EPR * 4, "modeNorm1L 4-repeat f16"),
]

OPS = [(name, ref_fn) for name, (_, ref_fn) in BINARY_OPS.items()]


def _shape_id(rows, cols, desc):
    return f"{rows}x{cols}-{desc.replace(' ', '-')}"


# ---------------------------------------------------------------------------
# f32 tests
# ---------------------------------------------------------------------------

F32_PARAMS = [
    pytest.param(
        (op_name, ref_fn, "float32", rows, cols, desc),
        id=f"{op_name}-float32-{_shape_id(rows, cols, desc)}",
    )
    for op_name, ref_fn in OPS
    for rows, cols, desc in F32_SHAPES
]


@pytest.mark.require_npu
@pytest.mark.parametrize("case", F32_PARAMS)
def test_binary_f32(case, torch, target_arch, backend):
    op_name, ref_fn, dtype_str, rows, cols, desc = case

    kernel = make_binary_kernel(
        op_name, rows, cols, dtype_str=dtype_str,
        target=target_arch, backend=backend,
    )
    compile_s, launch_s = launch_and_check(
        op_name=op_name,
        kernel_handle=kernel,
        ref_fn=ref_fn,
        shape=(rows, cols),
        dtype_str=dtype_str,
        torch=torch,
    )
    print(f"  PASS {op_name} {dtype_str} {rows}x{cols} ({desc}) "
          f"compile={compile_s:.3f}s launch={launch_s:.3f}s")


# ---------------------------------------------------------------------------
# f16 tests
# ---------------------------------------------------------------------------

F16_PARAMS = [
    pytest.param(
        (op_name, ref_fn, "float16", rows, cols, desc),
        id=f"{op_name}-float16-{_shape_id(rows, cols, desc)}",
    )
    for op_name, ref_fn in OPS
    for rows, cols, desc in F16_SHAPES
]


@pytest.mark.require_npu
@pytest.mark.parametrize("case", F16_PARAMS)
def test_binary_f16(case, torch, target_arch, backend):
    op_name, ref_fn, dtype_str, rows, cols, desc = case

    kernel = make_binary_kernel(
        op_name, rows, cols, dtype_str=dtype_str,
        target=target_arch, backend=backend,
    )
    compile_s, launch_s = launch_and_check(
        op_name=op_name,
        kernel_handle=kernel,
        ref_fn=ref_fn,
        shape=(rows, cols),
        dtype_str=dtype_str,
        torch=torch,
        rtol=1e-3,
        atol=1e-3,
    )
    print(f"  PASS {op_name} {dtype_str} {rows}x{cols} ({desc}) "
          f"compile={compile_s:.3f}s launch={launch_s:.3f}s")


# ---------------------------------------------------------------------------
# i16 bitwise/shift tests
# ---------------------------------------------------------------------------

INT_SHAPES: list[tuple[int, int, str]] = [
    (1, 64, "modeSmall 1x64"),
    (4, 64, "modeSmall 4x64"),
    (1, 128, "modeNorm1L 1x128"),
    (16, 64, "modeNorm1L 16x64"),
    (64, 64, "modeNorm1L 64x64"),
]

INT_PARAMS = [
    pytest.param(
        (op_name, ref_fn, rows, cols, desc),
        id=f"{op_name}-int16-{rows}x{cols}-{desc.replace(' ', '-')}",
    )
    for op_name, (_, ref_fn) in INT_OPS.items()
    for rows, cols, desc in INT_SHAPES
]


@pytest.mark.require_npu
@pytest.mark.parametrize("case", INT_PARAMS)
def test_binary_int16(case, torch, target_arch, backend):
    op_name, ref_fn, rows, cols, desc = case

    kernel = make_binary_kernel(
        op_name, rows, cols, dtype_str="int16",
        target=target_arch, backend=backend,
    )
    compile_s, launch_s = launch_and_check_int(
        kernel_handle=kernel,
        ref_fn=ref_fn,
        shape=(rows, cols),
        torch=torch,
    )
    print(f"  PASS {op_name} int16 {rows}x{cols} ({desc}) "
          f"compile={compile_s:.3f}s launch={launch_s:.3f}s")


# ---------------------------------------------------------------------------
# i16 scalar shift tests (tshls/tshrs)
# ---------------------------------------------------------------------------

SHIFT_VALS = [1, 3, 7]

SHIFT_PARAMS = [
    pytest.param(
        (op_name, ref_fn, rows, cols, sv, desc),
        id=f"{op_name}-int16-{rows}x{cols}-s{sv}-{desc.replace(' ', '-')}",
    )
    for op_name, (_, ref_fn) in SHIFT_OPS.items()
    for rows, cols, desc in INT_SHAPES
    for sv in SHIFT_VALS
]


@pytest.mark.require_npu
@pytest.mark.parametrize("case", SHIFT_PARAMS)
def test_shift_int16(case, torch, target_arch, backend):
    op_name, ref_fn, rows, cols, shift_val, desc = case

    kernel = make_shift_kernel(
        op_name, rows, cols, shift_val=shift_val,
        target=target_arch, backend=backend,
    )
    compile_s, launch_s = launch_and_check_shift(
        kernel_handle=kernel,
        ref_fn=ref_fn,
        shape=(rows, cols),
        shift_val=shift_val,
        torch=torch,
    )
    print(f"  PASS {op_name} int16 {rows}x{cols} <<{shift_val} ({desc}) "
          f"compile={compile_s:.3f}s launch={launch_s:.3f}s")
