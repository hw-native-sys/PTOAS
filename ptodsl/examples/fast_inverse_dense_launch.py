# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""
Fast dense inverse launch demo.

This is a simulator-friendly PTODSL port of the old dense fast-inverse example.
The old DSL kernel consumes ``A = M - I`` and validates against
``np.linalg.inv(I + A)``.  This launchable version keeps that reference contract
for dynamic batches of 2x2 dense matrices and uses the closed-form inverse:

    M = I + [[d00, d01], [d10, d11]]
    inv(M) = 1/det(M) * [[m11, -m01], [-m10, m00]]

Input and output are stored as four rows by batch columns, row-major in GM:
``[d00, d01, d10, d11] x batch`` for input and
``[inv00, inv01, inv10, inv11] x batch`` for output.

Run under the CPU simulator:

    scripts/sim_dsl.sh ptodsl/examples/fast_inverse_dense_launch.py
"""

import argparse
import time

import numpy as np

from ptodsl import pto


_DEVICE = "npu:0"
ROWS = 4
MAX_BATCH = 64
IN_ADDR = 0
OUT_ADDR = ROWS * MAX_BATCH * 4


@pto.jit(
    name="fast_inverse_dense_f32",
    kernel_kind="vector",
    target="a5",
    mode="explicit",
    insert_sync=False,
)
def fast_inverse_dense_f32(
    in_ptr: pto.ptr(pto.f32, "gm"),
    out_ptr: pto.ptr(pto.f32, "gm"),
    batch_i32: pto.i32,
):
    total_elems = ROWS * batch_i32
    in_view = pto.make_tensor_view(
        in_ptr,
        shape=[1, 1, 1, ROWS, batch_i32],
        strides=[total_elems, total_elems, total_elems, batch_i32, 1],
    )
    out_view = pto.make_tensor_view(
        out_ptr,
        shape=[1, 1, 1, ROWS, batch_i32],
        strides=[total_elems, total_elems, total_elems, batch_i32, 1],
    )
    in_part = pto.partition_view(
        in_view,
        offsets=[0, 0, 0, 0, 0],
        sizes=[1, 1, 1, ROWS, batch_i32],
    )
    out_part = pto.partition_view(
        out_view,
        offsets=[0, 0, 0, 0, 0],
        sizes=[1, 1, 1, ROWS, batch_i32],
    )

    in_tile = pto.alloc_tile(
        shape=[ROWS, MAX_BATCH],
        dtype=pto.f32,
        addr=IN_ADDR,
        valid_shape=[ROWS, batch_i32],
        blayout="RowMajor",
    )
    out_tile = pto.alloc_tile(
        shape=[ROWS, MAX_BATCH],
        dtype=pto.f32,
        addr=OUT_ADDR,
        valid_shape=[ROWS, batch_i32],
        blayout="RowMajor",
    )

    pto.tile.load(in_part, in_tile)
    pto.set_flag("MTE2", "V", event_id=0)
    pto.wait_flag("MTE2", "V", event_id=0)

    with pto.tileop():
        active, _ = pto.make_mask(pto.f32, batch_i32)
        d00 = pto.vlds(in_tile[0, 0:])
        d01 = pto.vlds(in_tile[1, 0:])
        d10 = pto.vlds(in_tile[2, 0:])
        d11 = pto.vlds(in_tile[3, 0:])
        one = pto.vbr(1.0)
        zero = pto.vsub(d00, d00, active)

        m00 = pto.vadd(one, d00, active)
        m01 = d01
        m10 = d10
        m11 = pto.vadd(one, d11, active)

        diag_prod = pto.vmul(m00, m11, active)
        off_prod = pto.vmul(m01, m10, active)
        det = pto.vsub(diag_prod, off_prod, active)

        inv00 = pto.vdiv(m11, det, active)
        neg_m01 = pto.vsub(zero, m01, active)
        inv01 = pto.vdiv(neg_m01, det, active)
        neg_m10 = pto.vsub(zero, m10, active)
        inv10 = pto.vdiv(neg_m10, det, active)
        inv11 = pto.vdiv(m00, det, active)

        pto.vsts(inv00, out_tile[0, 0:], active)
        pto.vsts(inv01, out_tile[1, 0:], active)
        pto.vsts(inv10, out_tile[2, 0:], active)
        pto.vsts(inv11, out_tile[3, 0:], active)

    pto.set_flag("V", "MTE3", event_id=0)
    pto.wait_flag("V", "MTE3", event_id=0)
    pto.tile.store(out_tile, out_part)
    pto.pipe_barrier(pto.Pipe.ALL)


CASES = [
    {"name": "batch4", "batch": 4, "eps": 1e-5},
    {"name": "batch37", "batch": 37, "eps": 1e-5},
]


def emit_mlir():
    return fast_inverse_dense_f32.mlir_module()


def make_case_inputs(case: dict[str, object]) -> np.ndarray:
    batch = int(case["batch"])
    rng = np.random.RandomState(hash(case["name"]) & 0xFFFFFFFF)
    delta = np.zeros((ROWS, batch), dtype=np.float32)
    delta[0] = rng.uniform(-0.20, 0.20, size=batch).astype(np.float32)
    delta[1] = rng.uniform(-0.20, 0.20, size=batch).astype(np.float32)
    delta[2] = rng.uniform(-0.20, 0.20, size=batch).astype(np.float32)
    delta[3] = rng.uniform(-0.20, 0.20, size=batch).astype(np.float32)
    return delta


def reference_inverse(delta: np.ndarray) -> np.ndarray:
    batch = delta.shape[1]
    mats = np.zeros((batch, 2, 2), dtype=np.float64)
    mats[:, 0, 0] = 1.0 + delta[0]
    mats[:, 0, 1] = delta[1]
    mats[:, 1, 0] = delta[2]
    mats[:, 1, 1] = 1.0 + delta[3]
    inv = np.linalg.inv(mats).astype(np.float32)

    out = np.empty_like(delta, dtype=np.float32)
    out[0] = inv[:, 0, 0]
    out[1] = inv[:, 0, 1]
    out[2] = inv[:, 1, 0]
    out[3] = inv[:, 1, 1]
    return out


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
    delta = make_case_inputs(case)
    ref = reference_inverse(delta)
    out = np.zeros_like(delta)

    delta_dev = torch.from_numpy(delta).to(_DEVICE)
    out_dev = torch.from_numpy(out).to(_DEVICE)
    stream = npu_stream(torch)

    t0 = time.perf_counter()
    compiled[1, stream](
        delta_dev.data_ptr(),
        out_dev.data_ptr(),
        case["batch"],
    )
    torch.npu.synchronize()
    launch_s = time.perf_counter() - t0

    np.testing.assert_allclose(
        out_dev.cpu().numpy(),
        ref,
        rtol=case["eps"],
        atol=case["eps"],
    )
    print(f"PASS {case['name']}  launch={launch_s:.3f}s")


def test_fast_inverse_dense() -> None:
    torch = init_runtime()

    t0 = time.perf_counter()
    compiled = fast_inverse_dense_f32.compile()
    compile_s = time.perf_counter() - t0
    print(f"compiled fast_inverse_dense_f32 in {compile_s:.3f}s")

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

    test_fast_inverse_dense()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
