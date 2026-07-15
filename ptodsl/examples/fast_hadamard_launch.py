# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""
Fast Hadamard transform launch demo using tile gather.

This keeps the current PTODSL launch surface but follows the old DSL kernel
shape closely: each row is loaded into a tile, every butterfly stage gathers
even and odd lanes with P0101/P1010, computes add/sub halves, stores those
halves back to the corresponding row regions, and reloads the row for the next
stage.

Run under the CPU simulator:

    scripts/sim_dsl.sh ptodsl/examples/fast_hadamard_launch.py
"""

import argparse
import math
import time

import numpy as np

from ptodsl import pto


_DEVICE = "npu:0"
MAX_BATCH = 8
PHYSICAL_N = 256
HALF_PHYSICAL_N = PHYSICAL_N // 2


@pto.jit(
    name="fast_hadamard_f16",
    kernel_kind="vector",
    target="a5",
)
def fast_hadamard_f16(
    x_ptr: pto.ptr(pto.f16, "gm"),
    batch_i32: pto.i32,
    n_i32: pto.i32,
    log2_n_i32: pto.i32,
):
    c0 = pto.const(0, dtype=pto.i32)
    c1 = pto.const(1, dtype=pto.i32)
    c2 = pto.const(2, dtype=pto.i32)

    total_elems = batch_i32 * n_i32
    x_view = pto.make_tensor_view(
        x_ptr,
        shape=[1, 1, 1, batch_i32, n_i32],
        strides=[total_elems, total_elems, total_elems, n_i32, 1],
    )

    active_pairs = n_i32 // c2
    row_tile = pto.alloc_tile(
        shape=[1, PHYSICAL_N],
        dtype=pto.f16,
        valid_shape=[1, n_i32],
        blayout="RowMajor",
    )
    even_tile = pto.alloc_tile(
        shape=[1, HALF_PHYSICAL_N],
        dtype=pto.f16,
        valid_shape=[1, active_pairs],
        blayout="RowMajor",
    )
    odd_tile = pto.alloc_tile(
        shape=[1, HALF_PHYSICAL_N],
        dtype=pto.f16,
        valid_shape=[1, active_pairs],
        blayout="RowMajor",
    )
    plus_tile = pto.alloc_tile(
        shape=[1, HALF_PHYSICAL_N],
        dtype=pto.f16,
        valid_shape=[1, active_pairs],
        blayout="RowMajor",
    )
    minus_tile = pto.alloc_tile(
        shape=[1, HALF_PHYSICAL_N],
        dtype=pto.f16,
        valid_shape=[1, active_pairs],
        blayout="RowMajor",
    )

    for row in range(0, batch_i32, 1):
        row_part = pto.partition_view(
            x_view,
            offsets=[0, 0, 0, row, 0],
            sizes=[1, 1, 1, 1, n_i32],
        )
        plus_part = pto.partition_view(
            x_view,
            offsets=[0, 0, 0, row, 0],
            sizes=[1, 1, 1, 1, active_pairs],
        )
        minus_part = pto.partition_view(
            x_view,
            offsets=[0, 0, 0, row, active_pairs],
            sizes=[1, 1, 1, 1, active_pairs],
        )

        pto.tile.load(row_part, row_tile)

        for _ in range(0, log2_n_i32, 1):
            pto.tile.gather(row_tile, even_tile, mask_pattern="P0101", axis="row")
            pto.tile.gather(row_tile, odd_tile, mask_pattern="P1010", axis="row")
            pto.tile.add(even_tile, odd_tile, plus_tile)
            pto.tile.sub(even_tile, odd_tile, minus_tile)
            pto.tile.store(plus_tile, plus_part)
            pto.tile.store(minus_tile, minus_part)
            pto.tile.load(row_part, row_tile)


CASES = [
    {"name": "batch2_n2", "batch": 2, "n": 2, "eps": 1e-3},
    {"name": "batch3_n3", "batch": 3, "n": 3, "eps": 1e-3},
    {"name": "batch4_n5", "batch": 4, "n": 5, "eps": 1e-3},
]


def emit_mlir():
    return fast_hadamard_f16.mlir_module()


def reference_hadamard(x: np.ndarray, log2_n: int) -> np.ndarray:
    y = x.astype(np.float32).copy()
    for _ in range(log2_n):
        active_pairs = y.shape[1] // 2
        if active_pairs == 0:
            break
        even = y[:, 0 : active_pairs * 2 : 2].copy()
        odd = y[:, 1 : active_pairs * 2 : 2].copy()
        y[:, :active_pairs] = even + odd
        y[:, active_pairs : active_pairs * 2] = even - odd
    return y.astype(np.float16)


def init_runtime():
    import torch
    import torch_npu  # noqa: F401

    torch.npu.config.allow_internal_format = False
    torch_npu.npu.set_compile_mode(jit_compile=False)
    torch.npu.set_device(_DEVICE)
    return torch


def npu_stream(torch):
    return torch.npu.current_stream()._as_parameter_  # noqa: SLF001


def make_case_inputs(case: dict[str, object]) -> np.ndarray:
    batch = int(case["batch"])
    n = int(case["n"])
    rng = np.random.RandomState(hash(case["name"]) & 0xFFFFFFFF)
    return rng.uniform(-1.0, 1.0, size=(batch, n)).astype(np.float16)


def run_case(case: dict[str, object], compiled, torch) -> None:
    x = make_case_inputs(case)
    log2_n = int(math.log2(int(case["n"])))
    ref = reference_hadamard(x, log2_n)
    x_dev = torch.from_numpy(x).to(_DEVICE)
    stream = npu_stream(torch)

    t0 = time.perf_counter()
    compiled[1, stream](
        x_dev.data_ptr(),
        case["batch"],
        case["n"],
        log2_n,
    )
    torch.npu.synchronize()
    launch_s = time.perf_counter() - t0

    np.testing.assert_allclose(
        x_dev.cpu().numpy().astype(np.float32),
        ref.astype(np.float32),
        rtol=case["eps"],
        atol=case["eps"],
    )
    print(f"PASS {case['name']}  launch={launch_s:.3f}s")


def test_fast_hadamard() -> None:
    torch = init_runtime()

    t0 = time.perf_counter()
    compiled = fast_hadamard_f16.compile()
    compile_s = time.perf_counter() - t0
    print(f"compiled fast_hadamard_f16 in {compile_s:.3f}s")

    for case in CASES:
        run_case(case, compiled, torch)
    print("All cases passed.")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--emit-mlir",
        action="store_true",
        help="print MLIR module and exit",
    )
    args = parser.parse_args(argv)

    if args.emit_mlir:
        print(emit_mlir())
        return 0

    test_fast_hadamard()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
