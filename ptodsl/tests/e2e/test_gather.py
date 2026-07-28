# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""
e2e tests for A3 VPTO gather.

Exercises the full lowering chain on NPU hardware:
  pto.tile.gatherb → pto.tgatherb → pto.ub.vgatherb → llvm.hivm.VGATHERB.b32
  pto.tile.gather  → pto.tgather  → pto.ub.vgather  → llvm.hivm.VGATHER.b32

Run:
    pytest ptodsl/tests/e2e/test_gather.py -v

Filter:
    pytest ptodsl/tests/e2e/test_gather.py -v -k "gatherb-float32-1x64"
"""

from __future__ import annotations

import pytest

from .common import (
    make_gather_index_kernel,
    make_gatherb_kernel,
    make_mgather_kernel,
    launch_and_check_gather,
    launch_and_check_gather_index,
    launch_and_check_mgather,
)


# ---------------------------------------------------------------------------
# Shape matrix — f32 (elementsPerRepeat = 64) and f16
# (elementsPerRepeat = 128).  gatherb works in 32B blocks, so the smallest
# useful cases are 8 f32 or 16 f16 values.
# ---------------------------------------------------------------------------

F32_EPR = 64
F16_EPR = 128

GATHERB_F32_SHAPES: list[tuple[int, int, str]] = [
    (1, 8, "single 32B block"),
    (1, F32_EPR, "single-repeat"),
    (1, F32_EPR + 32, "tail repeat padded offsets"),
    (2, F32_EPR, "multi-repeat 2 rows"),
    (1, F32_EPR * 2, "multi-repeat 2 repeats"),
    (4, F32_EPR, "multi-repeat 4 rows"),
]

GATHERB_F32_PARAMS = [
    pytest.param(
        (rows, cols, desc),
        id=f"gatherb-float32-{rows}x{cols}-{desc.replace(' ', '-')}",
    )
    for rows, cols, desc in GATHERB_F32_SHAPES
]

GATHERB_F16_SHAPES: list[tuple[int, int, str]] = [
    (1, 16, "single 32B block"),
    (1, F16_EPR, "single-repeat"),
    (1, F16_EPR + 64, "tail repeat padded offsets"),
]

GATHERB_F16_PARAMS = [
    pytest.param(
        (rows, cols, desc),
        id=f"gatherb-float16-{rows}x{cols}-{desc.replace(' ', '-')}",
    )
    for rows, cols, desc in GATHERB_F16_SHAPES
]

GATHER_INDEX_SHAPES: list[tuple[int, int, str]] = [
    (1, 64, "single-repeat"),
    (2, 64, "multi-row"),
    (2, 128, "multi-row wide"),
]

GATHER_INDEX_PARAMS = [
    pytest.param(
        (rows, cols, desc, dtype_str),
        id=f"gather-index-{dtype_str}-{rows}x{cols}-{desc.replace(' ', '-')}",
    )
    for dtype_str in ("float32", "float16")
    for rows, cols, desc in GATHER_INDEX_SHAPES
]

MGATHER_PARAMS = [
    pytest.param(6, 16, 3, 16, "row", "undefined", "float16", id="mgather-row-undefined-f16"),
    pytest.param(6, 8, 6, 8, "row", "zero", "float32", id="mgather-row-zero-f32"),
    pytest.param(6, 16, 6, 16, "row", "clamp", "float16", id="mgather-row-clamp-f16"),
    pytest.param(6, 8, 6, 8, "row", "wrap", "float32", id="mgather-row-wrap-f32"),
    pytest.param(4, 8, 2, 8, "elem", "undefined", "float32", id="mgather-elem-undefined-f32"),
    pytest.param(4, 16, 2, 16, "elem", "zero", "float16", id="mgather-elem-zero-f16"),
    pytest.param(4, 8, 2, 8, "elem", "clamp", "float32", id="mgather-elem-clamp-f32"),
    pytest.param(4, 16, 2, 16, "elem", "wrap", "float16", id="mgather-elem-wrap-f16"),
]


@pytest.mark.require_npu
@pytest.mark.parametrize("case", GATHERB_F32_PARAMS)
def test_gatherb_f32(case, torch, target_arch, backend):
    rows, cols, desc = case

    kernel = make_gatherb_kernel(
        rows, cols, dtype_str="float32",
        target=target_arch, backend=backend,
    )
    compile_s, launch_s = launch_and_check_gather(
        rows=rows,
        cols=cols,
        dtype_str="float32",
        torch=torch,
        kernel_handle=kernel,
    )
    print(f"  PASS gatherb float32 {rows}x{cols} ({desc}) "
          f"compile={compile_s:.3f}s launch={launch_s:.3f}s")


@pytest.mark.require_npu
@pytest.mark.parametrize("case", GATHERB_F16_PARAMS)
def test_gatherb_f16(case, torch, target_arch, backend):
    rows, cols, desc = case

    kernel = make_gatherb_kernel(
        rows, cols, dtype_str="float16",
        target=target_arch, backend=backend,
    )
    compile_s, launch_s = launch_and_check_gather(
        rows=rows,
        cols=cols,
        dtype_str="float16",
        torch=torch,
        kernel_handle=kernel,
    )
    print(f"  PASS gatherb float16 {rows}x{cols} ({desc}) "
          f"compile={compile_s:.3f}s launch={launch_s:.3f}s")


@pytest.mark.require_npu
@pytest.mark.parametrize("case", GATHER_INDEX_PARAMS)
def test_gather_index(case, torch, target_arch, backend):
    rows, cols, desc, dtype_str = case

    kernel = make_gather_index_kernel(
        rows, cols, dtype_str=dtype_str,
        target=target_arch, backend=backend,
    )
    compile_s, launch_s = launch_and_check_gather_index(
        rows=rows,
        cols=cols,
        dtype_str=dtype_str,
        torch=torch,
        kernel_handle=kernel,
    )
    print(f"  PASS gather index {dtype_str} {rows}x{cols} ({desc}) "
          f"compile={compile_s:.3f}s launch={launch_s:.3f}s")


@pytest.mark.require_npu
@pytest.mark.parametrize(
    "src_rows,src_cols,dst_rows,dst_cols,coalesce,gather_oob,dtype_str",
    MGATHER_PARAMS,
)
def test_mgather(
    src_rows,
    src_cols,
    dst_rows,
    dst_cols,
    coalesce,
    gather_oob,
    dtype_str,
    torch,
    target_arch,
    backend,
):
    kernel = make_mgather_kernel(
        src_rows=src_rows,
        src_cols=src_cols,
        dst_rows=dst_rows,
        dst_cols=dst_cols,
        coalesce=coalesce,
        gather_oob=gather_oob,
        dtype_str=dtype_str,
        target=target_arch,
        backend=backend,
    )
    compile_s, launch_s = launch_and_check_mgather(
        kernel_handle=kernel,
        src_rows=src_rows,
        src_cols=src_cols,
        dst_rows=dst_rows,
        dst_cols=dst_cols,
        coalesce=coalesce,
        gather_oob=gather_oob,
        dtype_str=dtype_str,
        torch=torch,
    )
    print(
        f"  PASS mgather {coalesce}/{gather_oob} {dtype_str} "
        f"{src_rows}x{src_cols}->{dst_rows}x{dst_cols} "
        f"compile={compile_s:.3f}s launch={launch_s:.3f}s"
    )
