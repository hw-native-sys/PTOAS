# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""
Row-wise softmax — end-to-end launch demo.

This example is the launchable counterpart to the compile-only
``softmax_dsl.py`` sample. It uses an online-softmax recurrence internally,
but the public kernel surface is the ordinary softmax contract: load an input
score matrix, compute the per-row softmax, and store the normalized output.

Each kernel instance runs on one NPU, preloads the full score matrix to UB,
where the input is already laid out as ``[seq, rows]`` so each UB row
represents one score column, and then streams 64-row packs through the
online-softmax recurrence:

    running_max = max(running_max, score_col)
    running_sum = running_sum * exp(old_max - new_max) + exp(score_col - new_max)
    out         = exp(score_col - final_max) / final_sum

The demo offers two launchable kernels so the current launch ABI does not need
an extra runtime tile-width parameter:

- ``rows64_seq128``: full-width 64-row packed softmax
- ``rows81_seq96``: same single NPU, but two sequential row-pack updates
"""

import argparse
import time

import numpy as np

from ptodsl import pto

_DEVICE = "npu:0"


def _make_softmax_kernel(name: str, *, rows: int, seq: int):
    if rows <= 0:
        raise ValueError("rows must be positive")
    if seq <= 0:
        raise ValueError("seq must be positive")

    @pto.jit(
        name=name,
        target="a5",
        mode="explicit",
        insert_sync=False
    )
    def kernel(
        scores_ptr: pto.ptr(pto.f32, "gm"),
        out_ptr: pto.ptr(pto.f32, "gm"),
        runtime_seq: pto.i32,
        runtime_rows: pto.i32,
    ):
        lane_num = pto.elements_per_vreg(pto.f32)
        physical_rows = ((rows + lane_num - 1) // lane_num) * lane_num
        scores_tile_bytes = seq * physical_rows * pto.bytewidth(pto.f32)
        total_elems = runtime_rows * runtime_seq

        scores_view = pto.make_tensor_view(
            scores_ptr,
            shape=[1, 1, 1, runtime_seq, runtime_rows],
            strides=[total_elems, total_elems, total_elems, runtime_rows, 1],
        )
        out_view = pto.make_tensor_view(
            out_ptr,
            shape=[1, 1, 1, runtime_seq, runtime_rows],
            strides=[total_elems, total_elems, total_elems, runtime_rows, 1],
        )
        scores_part = pto.partition_view(
            scores_view,
            offsets=[0, 0, 0, 0, 0],
            sizes=[1, 1, 1, runtime_seq, runtime_rows],
        )
        out_part = pto.partition_view(
            out_view,
            offsets=[0, 0, 0, 0, 0],
            sizes=[1, 1, 1, runtime_seq, runtime_rows],
        )

        scores_tile = pto.alloc_tile(
            shape=[seq, physical_rows],
            dtype=pto.float32,
            addr=0,
            valid_shape=[runtime_seq, runtime_rows],
            blayout="RowMajor",
        )
        out_tile = pto.alloc_tile(
            shape=[seq, physical_rows],
            dtype=pto.float32,
            addr=scores_tile_bytes,
            valid_shape=[runtime_seq, runtime_rows],
            blayout="RowMajor",
        )

        pto.tile.load(scores_part, scores_tile)
        out_tile.fill(0.0)

        pto.set_flag("MTE2", "V", event_id=0)
        pto.wait_flag("MTE2", "V", event_id=0)

        with pto.tileop():
            remaining_rows = runtime_rows
            for row_base in range(0, runtime_rows, lane_num):
                active_rows, remaining_after_pack = pto.make_mask(pto.f32, remaining_rows)
                running_max = pto.vlds(scores_tile[0, row_base:])
                running_sum = pto.vbr(1.0)

                for col in range(1, runtime_seq, 1):
                    col_vec = pto.vlds(scores_tile[col, row_base:])
                    merged_max = pto.vmax(running_max, col_vec, active_rows)
                    running_delta = pto.vsub(running_max, merged_max, active_rows)
                    scaled_running = pto.vexp(running_delta, active_rows)
                    running_sum_scaled = pto.vmul(scaled_running, running_sum, active_rows)
                    col_delta = pto.vsub(col_vec, merged_max, active_rows)
                    col_exp = pto.vexp(col_delta, active_rows)
                    running_sum = pto.vadd(running_sum_scaled, col_exp, active_rows)
                    running_max = merged_max

                final_max = running_max
                final_sum = running_sum

                for col in range(0, runtime_seq, 1):
                    col_vec = pto.vlds(scores_tile[col, row_base:])
                    out_delta = pto.vsub(col_vec, final_max, active_rows)
                    exp_vec = pto.vexp(out_delta, active_rows)
                    out_vec = pto.vdiv(exp_vec, final_sum, active_rows)
                    pto.vsts(out_vec, out_tile[col, row_base:], active_rows)

                remaining_rows = remaining_after_pack

        pto.set_flag("V", "MTE3", event_id=0)
        pto.wait_flag("V", "MTE3", event_id=0)

        pto.tile.store(out_tile, out_part)
        pto.pipe_barrier(pto.Pipe.ALL)

    return kernel


SOFTMAX_ROWS64_SEQ128 = _make_softmax_kernel(
    "softmax_rows64_seq128",
    rows=64,
    seq=128,
)
SOFTMAX_ROWS81_SEQ96 = _make_softmax_kernel(
    "softmax_rows81_seq96",
    rows=81,
    seq=96,
)

KERNELS = (
    SOFTMAX_ROWS64_SEQ128,
    SOFTMAX_ROWS81_SEQ96,
)

CASES = [
    {
        "name": "rows64_seq128",
        "kernel": SOFTMAX_ROWS64_SEQ128,
        "rows": 64,
        "seq": 128,
    },
    {
        "name": "rows81_seq96",
        "kernel": SOFTMAX_ROWS81_SEQ96,
        "rows": 81,
        "seq": 96,
    },
]


def emit_mlir():
    return pto.merge_jit_modules(*KERNELS)


def reference_softmax(scores: np.ndarray):
    row_max = np.max(scores, axis=0, keepdims=True)
    shifted = np.exp(scores - row_max, dtype=np.float32)
    row_sum = np.sum(shifted, axis=0, keepdims=True, dtype=np.float32)
    return shifted / row_sum


def init_runtime():
    import torch
    import torch_npu  # noqa: F401

    torch.npu.config.allow_internal_format = False
    torch_npu.npu.set_compile_mode(jit_compile=False)
    torch.npu.set_device(_DEVICE)
    return torch


def npu_stream(torch):
    return torch.npu.current_stream()._as_parameter_  # noqa: SLF001


def make_case_inputs(case: dict[str, object]):
    rows = int(case["rows"])
    seq = int(case["seq"])
    rng = np.random.RandomState(hash(case["name"]) & 0xFFFFFFFF)

    scores = rng.uniform(-4.0, 4.0, size=(seq, rows)).astype(np.float32)
    out = np.zeros((seq, rows), dtype=np.float32)

    return scores, out


def run_case(case: dict[str, object], torch) -> None:
    scores, out = make_case_inputs(case)
    ref_out = reference_softmax(scores)

    scores_t = torch.from_numpy(scores).to(_DEVICE)
    out_t = torch.from_numpy(out).to(_DEVICE)
    stream = npu_stream(torch)

    t0 = time.perf_counter()
    compiled = case["kernel"].compile()
    compile_s = time.perf_counter() - t0

    t0 = time.perf_counter()
    compiled[1, stream](
        scores_t.data_ptr(),
        out_t.data_ptr(),
        case["seq"],
        case["rows"],
    )
    torch.npu.synchronize()
    launch_s = time.perf_counter() - t0

    np.testing.assert_allclose(out_t.cpu().numpy(), ref_out, rtol=1e-5, atol=1e-5)

    print(
        f"PASS {case['name']}  "
        f"compile={compile_s:.3f}s launch={launch_s:.3f}s"
    )


def test_softmax() -> None:
    torch = init_runtime()
    for case in CASES:
        run_case(case, torch)
    print("All cases passed.")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--emit-mlir",
        action="store_true",
        help="print the merged MLIR module and exit",
    )
    args = parser.parse_args(argv)

    if args.emit_mlir:
        print(emit_mlir())
        return 0

    test_softmax()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
