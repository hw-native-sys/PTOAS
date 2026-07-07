# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""
Dynamic-shape row-wise softmax launch demo.

This is a launchable vector-only dynamic kernel based on
``flash_attention_softmax_launch.py``.  One compiled kernel has static physical
capacity, while launch-time ``runtime_seq`` and ``runtime_rows`` define the
valid input/output shape and SIMD loop bounds.

Run under the CPU simulator:

    scripts/sim_dsl.sh ptodsl/examples/dynamic_softmax_launch.py
"""

import argparse
import time

import numpy as np

from ptodsl import pto


_DEVICE = "npu:0"
MAX_SEQ = 16
MAX_ROWS = 64


@pto.jit(
    name="dynamic_softmax",
    kernel_kind="vector",
    target="a5",
    mode="explicit",
    insert_sync=False,
)
def dynamic_softmax(
    scores_ptr: pto.ptr(pto.f32, "gm"),
    out_ptr: pto.ptr(pto.f32, "gm"),
    runtime_seq: pto.i32,
    runtime_rows: pto.i32,
):
    lane_num = pto.elements_per_vreg(pto.f32)
    physical_rows = ((MAX_ROWS + lane_num - 1) // lane_num) * lane_num
    scores_tile_bytes = MAX_SEQ * physical_rows * pto.bytewidth(pto.f32)
    total_elems = runtime_seq * runtime_rows

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
        shape=[MAX_SEQ, physical_rows],
        dtype=pto.float32,
        addr=0,
        valid_shape=[runtime_seq, runtime_rows],
        blayout="RowMajor",
    )
    out_tile = pto.alloc_tile(
        shape=[MAX_SEQ, physical_rows],
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

            for col in range(0, runtime_seq, 1):
                col_vec = pto.vlds(scores_tile[col, row_base:])
                out_delta = pto.vsub(col_vec, running_max, active_rows)
                exp_vec = pto.vexp(out_delta, active_rows)
                out_vec = pto.vdiv(exp_vec, running_sum, active_rows)
                pto.vsts(out_vec, out_tile[col, row_base:], active_rows)

            remaining_rows = remaining_after_pack

    pto.set_flag("V", "MTE3", event_id=0)
    pto.wait_flag("V", "MTE3", event_id=0)

    pto.tile.store(out_tile, out_part)
    pto.pipe_barrier(pto.Pipe.ALL)


CASES = [
    {"name": "seq16_rows64", "seq": 16, "rows": 64},
    {"name": "seq8_rows37", "seq": 8, "rows": 37},
]


def emit_mlir():
    return dynamic_softmax.mlir_module()


def reference_softmax(scores: np.ndarray) -> np.ndarray:
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
    seq = int(case["seq"])
    rows = int(case["rows"])
    rng = np.random.RandomState(hash(case["name"]) & 0xFFFFFFFF)
    scores = rng.uniform(-4.0, 4.0, size=(seq, rows)).astype(np.float32)
    out = np.zeros((seq, rows), dtype=np.float32)
    return scores, out


def run_case(case: dict[str, object], compiled, torch) -> None:
    scores, out = make_case_inputs(case)
    ref_out = reference_softmax(scores)

    scores_t = torch.from_numpy(scores).to(_DEVICE)
    out_t = torch.from_numpy(out).to(_DEVICE)
    stream = npu_stream(torch)

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
    print(f"PASS {case['name']}  launch={launch_s:.3f}s")


def test_dynamic_softmax() -> None:
    torch = init_runtime()

    t0 = time.perf_counter()
    compiled = dynamic_softmax.compile()
    compile_s = time.perf_counter() - t0
    print(f"compiled dynamic_softmax in {compile_s:.3f}s")

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

    test_dynamic_softmax()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
