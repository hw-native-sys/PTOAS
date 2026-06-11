# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""
TADD tile kernel — Python DSL equivalent of
  test/tilelang_st/npu/a3/src/st/testcase/tadd/tadd.pto

End-to-end: @pto.jit → MLIR → binary → launch → accuracy check.
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
            "Unable to locate the PTODSL Python package root from tadd_launch.py"
        )

from ptodsl import pto

_DEVICE = "npu:0"


# ---------------------------------------------------------------------------
# Kernel
# ---------------------------------------------------------------------------

def _tadd_tile(A, B, C, rows: int, cols: int) -> None:
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
    pto.tile.add(a_tile, b_tile, c_tile)
    pto.tile.store(c_tile, c_part)


@pto.jit(
    name="TADD_f32_16x64",
    kernel_kind="vector",
    target="a3",
)
def TADD_f32_16x64(
    A_ptr: pto.ptr(pto.f32, "gm"),
    B_ptr: pto.ptr(pto.f32, "gm"),
    C_ptr: pto.ptr(pto.f32, "gm"),
):
    _tadd_tile(A_ptr, B_ptr, C_ptr, 16, 64)


@pto.jit(
    name="TADD_f32_32x32",
    kernel_kind="vector",
    target="a3",
)
def TADD_f32_32x32(
    A_ptr: pto.ptr(pto.f32, "gm"),
    B_ptr: pto.ptr(pto.f32, "gm"),
    C_ptr: pto.ptr(pto.f32, "gm"),
):
    _tadd_tile(A_ptr, B_ptr, C_ptr, 32, 32)


KERNELS = (TADD_f32_16x64, TADD_f32_32x32)


def emit_mlir():
    return pto.merge_jit_modules(*KERNELS)


# ---------------------------------------------------------------------------
# Host
# ---------------------------------------------------------------------------

CASES = [
    {"name": "f32_16x64", "kernel": TADD_f32_16x64, "shape": (16, 64), "eps": 1e-6},
    {"name": "f32_32x32", "kernel": TADD_f32_32x32, "shape": (32, 32), "eps": 1e-6},
]


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
    ref = x + y

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


def test_tadd() -> None:
    torch = init_torch_npu()
    for case in CASES:
        run_case(case, torch)
    print("All cases passed.")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--emit-mlir",
        action="store_true",
        help="print merged MLIR module and exit (compile-only)",
    )
    args = parser.parse_args(argv)

    if args.emit_mlir:
        print(emit_mlir())
        return 0

    test_tadd()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
