# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""
A3 binary-op tile kernels: TSUB, TMUL, TDIV.

End-to-end: @pto.jit -> MLIR -> binary -> launch -> accuracy check.
Exercises the pto.tsub/tmul/tdiv -> pto.ub.vsub/vmul/vdiv lowering paths.
"""

import argparse
import time
from pathlib import Path
import sys

import numpy as np

if __package__ in {None, ""}:
    here = Path(__file__).resolve()
    for candidate in here.parents:
        if (candidate / "ptodsl" / "__init__.py").exists():
            sys.path.insert(0, str(candidate))
            break
    else:
        raise RuntimeError(
            "Unable to locate the PTODSL Python package root from bop_launch.py"
        )

from ptodsl import pto

_DEVICE = "npu:0"


# ---------------------------------------------------------------------------
# Kernel helpers
# ---------------------------------------------------------------------------

_BINOPS = ("add", "sub", "mul", "div")
_REF_FNS = {
    "add": lambda x, y: x + y,
    "sub": lambda x, y: x - y,
    "mul": lambda x, y: x * y,
    "div": lambda x, y: x / (y + 1e-8),
}


def _binop_tile(A, B, C, rows: int, cols: int, op: str) -> None:
    assert op in _BINOPS, f"unknown binop: {op}"
    c0 = pto.const(0)
    c1 = pto.const(1)
    c_rows = pto.const(rows)
    c_cols = c_rows if rows == cols else pto.const(cols)
    c_elems = pto.const(rows * cols)

    shape = [c1, c1, c1, c_rows, c_cols]
    strides = [c_elems, c_elems, c_elems, c_cols, c1]
    off = [c0, c0, c0, c0, c0]

    a_view = pto.make_tensor_view(A, shape=shape, strides=strides)
    b_view = pto.make_tensor_view(B, shape=shape, strides=strides)
    c_view = pto.make_tensor_view(C, shape=shape, strides=strides)

    a_part = pto.partition_view(a_view, offsets=off, sizes=shape)
    b_part = pto.partition_view(b_view, offsets=off, sizes=shape)
    c_part = pto.partition_view(c_view, offsets=off, sizes=shape)

    a_tile = pto.alloc_tile(shape=[rows, cols], dtype=pto.float32)
    b_tile = pto.alloc_tile(shape=[rows, cols], dtype=pto.float32)
    c_tile = pto.alloc_tile(shape=[rows, cols], dtype=pto.float32)

    pto.tile.load(a_part, a_tile)
    pto.tile.load(b_part, b_tile)
    getattr(pto.tile, op)(a_tile, b_tile, c_tile)
    pto.tile.store(c_tile, c_part)


# ---------------------------------------------------------------------------
# TSUB kernels
# ---------------------------------------------------------------------------

@pto.jit(
    name="TSUB_f32_1x64",
    kernel_kind="vector",
    target="a3",
)
def TSUB_f32_1x64(
    A_ptr: pto.ptr(pto.f32, "gm"),
    B_ptr: pto.ptr(pto.f32, "gm"),
    C_ptr: pto.ptr(pto.f32, "gm"),
):
    _binop_tile(A_ptr, B_ptr, C_ptr, 1, 64, "sub")


@pto.jit(
    name="TSUB_f32_16x64",
    kernel_kind="vector",
    target="a3",
)
def TSUB_f32_16x64(
    A_ptr: pto.ptr(pto.f32, "gm"),
    B_ptr: pto.ptr(pto.f32, "gm"),
    C_ptr: pto.ptr(pto.f32, "gm"),
):
    _binop_tile(A_ptr, B_ptr, C_ptr, 16, 64, "sub")


# ---------------------------------------------------------------------------
# TMUL kernels
# ---------------------------------------------------------------------------

@pto.jit(
    name="TMUL_f32_1x64",
    kernel_kind="vector",
    target="a3",
)
def TMUL_f32_1x64(
    A_ptr: pto.ptr(pto.f32, "gm"),
    B_ptr: pto.ptr(pto.f32, "gm"),
    C_ptr: pto.ptr(pto.f32, "gm"),
):
    _binop_tile(A_ptr, B_ptr, C_ptr, 1, 64, "mul")


@pto.jit(
    name="TMUL_f32_16x64",
    kernel_kind="vector",
    target="a3",
)
def TMUL_f32_16x64(
    A_ptr: pto.ptr(pto.f32, "gm"),
    B_ptr: pto.ptr(pto.f32, "gm"),
    C_ptr: pto.ptr(pto.f32, "gm"),
):
    _binop_tile(A_ptr, B_ptr, C_ptr, 16, 64, "mul")


# ---------------------------------------------------------------------------
# TDIV kernels
# ---------------------------------------------------------------------------

@pto.jit(
    name="TDIV_f32_1x64",
    kernel_kind="vector",
    target="a3",
)
def TDIV_f32_1x64(
    A_ptr: pto.ptr(pto.f32, "gm"),
    B_ptr: pto.ptr(pto.f32, "gm"),
    C_ptr: pto.ptr(pto.f32, "gm"),
):
    _binop_tile(A_ptr, B_ptr, C_ptr, 1, 64, "div")


@pto.jit(
    name="TDIV_f32_16x64",
    kernel_kind="vector",
    target="a3",
)
def TDIV_f32_16x64(
    A_ptr: pto.ptr(pto.f32, "gm"),
    B_ptr: pto.ptr(pto.f32, "gm"),
    C_ptr: pto.ptr(pto.f32, "gm"),
):
    _binop_tile(A_ptr, B_ptr, C_ptr, 16, 64, "div")


# ---------------------------------------------------------------------------
# Test cases
# ---------------------------------------------------------------------------

CASES = [
    {"name": "sub_f32_1x64",  "kernel": TSUB_f32_1x64,  "shape": (1, 64),
     "op": "sub",  "eps": 1e-6},
    {"name": "sub_f32_16x64", "kernel": TSUB_f32_16x64, "shape": (16, 64),
     "op": "sub",  "eps": 1e-6},
    {"name": "mul_f32_1x64",  "kernel": TMUL_f32_1x64,  "shape": (1, 64),
     "op": "mul",  "eps": 1e-6},
    {"name": "mul_f32_16x64", "kernel": TMUL_f32_16x64, "shape": (16, 64),
     "op": "mul",  "eps": 1e-6},
    {"name": "div_f32_1x64",  "kernel": TDIV_f32_1x64,  "shape": (1, 64),
     "op": "div",  "eps": 1e-3},
    {"name": "div_f32_16x64", "kernel": TDIV_f32_16x64, "shape": (16, 64),
     "op": "div",  "eps": 1e-3},
]


# ---------------------------------------------------------------------------
# Host
# ---------------------------------------------------------------------------

def init_torch_npu() -> None:
    import torch
    import torch_npu  # noqa: F401

    torch.npu.config.allow_internal_format = False
    torch_npu.npu.set_compile_mode(jit_compile=False)
    torch.npu.set_device(_DEVICE)
    return torch


def npu_stream(torch):
    return torch.npu.current_stream()._as_parameter_  # noqa: SLF001


def run_case(case: dict, torch) -> None:
    shape = case["shape"]
    rng = np.random.RandomState(hash(case["name"]) & 0xFFFFFFFF)
    x = rng.randint(1, 10, size=shape).astype(np.float32)
    y = rng.randint(1, 10, size=shape).astype(np.float32)
    ref = _REF_FNS[case["op"]](x, y)

    a = torch.from_numpy(x).to(_DEVICE)
    b = torch.from_numpy(y).to(_DEVICE)
    c = torch.empty(shape, dtype=torch.float32, device=_DEVICE)
    stream = npu_stream(torch)

    t0 = time.perf_counter()
    compiled = case["kernel"].compile()
    compile_s = time.perf_counter() - t0

    t0 = time.perf_counter()
    compiled[1, stream](a.data_ptr(), b.data_ptr(), c.data_ptr())
    torch.npu.synchronize()
    launch_s = time.perf_counter() - t0

    torch.testing.assert_close(ref, c.cpu().numpy(), rtol=case["eps"], atol=case["eps"])
    print(
        f"PASS {case['name']}  "
        f"compile={compile_s:.3f}s launch={launch_s:.3f}s"
    )


def test_bop() -> None:
    torch = init_torch_npu()
    for case in CASES:
        run_case(case, torch)
    print("All cases passed.")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    args = parser.parse_args(argv)
    test_bop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
