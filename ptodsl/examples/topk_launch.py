# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""
TopK launch demo using tile sort/gather primitives.

This is the current PTODSL form of the old DSL TopK example.  The column count
and K are compile-time parameters, while the row count is a runtime launch
argument.  For each input row, the kernel runs:

  1. ``pto.tile.sort32`` to sort 32-column blocks and produce interleaved
     ``(score_f32, idx_u32)`` records.
  2. ``pto.tile.mrgsort`` to merge those records into descending score order.
  3. ``pto.tile.gather(..., mask_pattern="P0101", axis="row")`` to extract top-K scores.
  4. ``pto.tile.gather(..., mask_pattern="P1010", axis="row")`` to extract top-K indices.

The valid TopK shape here is the smallest old-DSL shape:
``N_COLS=128`` and ``TOPK=64``.  It exercises one merge pass because
``SORT_COLS = N_COLS * 2 = 256`` and ``HW_BLOCK_LEN = 32 * 2 = 64``.

Run under the CPU simulator:

    scripts/sim_dsl.sh ptodsl/examples/topk_launch.py
"""

import argparse
import time

import numpy as np

from ptodsl import pto


_DEVICE = "npu:0"
N_COLS = 128
TOPK = 64
DST_STRIDE = 2
SORT_BLOCK_LEN = 32
SORT_COLS = N_COLS * DST_STRIDE
HW_BLOCK_LEN = SORT_BLOCK_LEN * DST_STRIDE


@pto.jit(
    name=f"topk_c{N_COLS}_k{TOPK}",
    kernel_kind="vector",
    target="a5",
)
def topk_c128_k64(
    src_ptr: pto.ptr(pto.f32, "gm"),
    inidx_ptr: pto.ptr(pto.ui32, "gm"),
    scores_ptr: pto.ptr(pto.f32, "gm"),
    indices_ptr: pto.ptr(pto.ui32, "gm"),
    n_rows: pto.i32,
):
    c1 = pto.const(1)
    c_ncols = pto.const(N_COLS)
    c_topk = pto.const(TOPK)

    src_elems = n_rows * c_ncols
    out_elems = n_rows * c_topk

    src_view = pto.make_tensor_view(
        src_ptr,
        shape=[c1, c1, c1, n_rows, c_ncols],
        strides=[src_elems, src_elems, src_elems, c_ncols, c1],
    )
    inidx_view = pto.make_tensor_view(
        inidx_ptr,
        shape=[c1, c1, c1, c1, c_ncols],
        strides=[c_ncols, c_ncols, c_ncols, c_ncols, c1],
    )
    scores_view = pto.make_tensor_view(
        scores_ptr,
        shape=[c1, c1, c1, n_rows, c_topk],
        strides=[out_elems, out_elems, out_elems, c_topk, c1],
    )
    indices_view = pto.make_tensor_view(
        indices_ptr,
        shape=[c1, c1, c1, n_rows, c_topk],
        strides=[out_elems, out_elems, out_elems, c_topk, c1],
    )

    inidx_part = pto.partition_view(
        inidx_view,
        offsets=[0, 0, 0, 0, 0],
        sizes=[1, 1, 1, 1, N_COLS],
    )

    src_tile = pto.alloc_tile(shape=[1, N_COLS], dtype=pto.f32)
    inidx_tile = pto.alloc_tile(shape=[1, N_COLS], dtype=pto.ui32)
    sort_tile = pto.alloc_tile(shape=[1, SORT_COLS], dtype=pto.f32)
    sort_tmp = pto.alloc_tile(shape=[1, SORT_COLS], dtype=pto.f32)
    gather_win_f32 = pto.alloc_tile(
        shape=[1, SORT_COLS],
        dtype=pto.f32,
        valid_shape=[1, 2 * TOPK],
    )
    gather_win_u32 = pto.alloc_tile(
        shape=[1, SORT_COLS],
        dtype=pto.ui32,
        valid_shape=[1, 2 * TOPK],
    )
    top_scores = pto.alloc_tile(shape=[1, TOPK], dtype=pto.f32)
    top_indices = pto.alloc_tile(shape=[1, TOPK], dtype=pto.ui32)

    pto.tile.load(inidx_part, inidx_tile)

    for row in range(0, n_rows, 1):
        src_part = pto.partition_view(
            src_view,
            offsets=[0, 0, 0, row, 0],
            sizes=[1, 1, 1, 1, N_COLS],
        )
        scores_part = pto.partition_view(
            scores_view,
            offsets=[0, 0, 0, row, 0],
            sizes=[1, 1, 1, 1, TOPK],
        )
        indices_part = pto.partition_view(
            indices_view,
            offsets=[0, 0, 0, row, 0],
            sizes=[1, 1, 1, 1, TOPK],
        )

        pto.tile.load(src_part, src_tile)
        pto.tile.sort32(src_tile, inidx_tile, sort_tile)

        pto.tile.mrgsort(sort_tile, sort_tmp, HW_BLOCK_LEN)
        pto.tile.mov(sort_tmp, sort_tile)

        pto.tile.mov(sort_tile, gather_win_f32)
        pto.tile.gather(gather_win_f32, top_scores, mask_pattern="P0101", axis="row")

        pto.tile.mov(sort_tile, gather_win_u32)
        pto.tile.gather(gather_win_u32, top_indices, mask_pattern="P1010", axis="row")

        pto.tile.store(top_scores, scores_part)
        pto.tile.store(top_indices, indices_part)


CASES = [
    {"name": "rows1_c128_k64", "rows": 1},
    {"name": "rows3_c128_k64", "rows": 3},
]


def emit_mlir():
    return topk_c128_k64.mlir_module()


def make_case_inputs(case: dict[str, object]) -> tuple[np.ndarray, np.ndarray]:
    rows = int(case["rows"])
    rng = np.random.RandomState(hash(case["name"]) & 0xFFFFFFFF)
    scores = rng.uniform(-16.0, 16.0, size=(rows, N_COLS)).astype(np.float32)
    col_indices = np.arange(N_COLS, dtype=np.uint32).reshape(1, N_COLS)
    return scores, col_indices


def reference_topk(scores: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    order = np.argsort(-scores, axis=1, kind="stable")[:, :TOPK]
    values = np.take_along_axis(scores, order, axis=1).astype(np.float32)
    indices = order.astype(np.uint32)
    return values, indices


def init_runtime():
    import torch
    import torch_npu  # noqa: F401

    torch.npu.config.allow_internal_format = False
    torch_npu.npu.set_compile_mode(jit_compile=False)
    torch.npu.set_device(_DEVICE)
    return torch


def npu_stream(torch):
    return torch.npu.current_stream()._as_parameter_  # noqa: SLF001


def run_case(case: dict[str, object], compiled, torch) -> None:
    scores, col_indices = make_case_inputs(case)
    ref_values, ref_indices = reference_topk(scores)
    out_scores = np.zeros((int(case["rows"]), TOPK), dtype=np.float32)
    out_indices = np.zeros((int(case["rows"]), TOPK), dtype=np.uint32)

    scores_dev = torch.from_numpy(scores).to(_DEVICE)
    col_indices_dev = torch.from_numpy(col_indices.astype(np.int32)).to(_DEVICE)
    out_scores_dev = torch.from_numpy(out_scores).to(_DEVICE)
    out_indices_dev = torch.from_numpy(out_indices.astype(np.int32)).to(_DEVICE)
    stream = npu_stream(torch)

    t0 = time.perf_counter()
    compiled[1, stream](
        scores_dev.data_ptr(),
        col_indices_dev.data_ptr(),
        out_scores_dev.data_ptr(),
        out_indices_dev.data_ptr(),
        case["rows"],
    )
    torch.npu.synchronize()
    launch_s = time.perf_counter() - t0

    np.testing.assert_allclose(out_scores_dev.cpu().numpy(), ref_values, rtol=1e-5, atol=1e-5)
    np.testing.assert_array_equal(
        out_indices_dev.cpu().numpy().astype(np.uint32),
        ref_indices,
    )
    print(f"PASS {case['name']}  launch={launch_s:.3f}s")


def test_topk() -> None:
    torch = init_runtime()

    t0 = time.perf_counter()
    compiled = topk_c128_k64.compile()
    compile_s = time.perf_counter() - t0
    print(f"compiled topk_c{N_COLS}_k{TOPK} in {compile_s:.3f}s")

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

    test_topk()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
