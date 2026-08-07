#!/usr/bin/env python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Working VMI fallback: 64-lane stores plus a scalar extraction phase."""

import argparse
import os

import numpy as np

from ptodsl import pto, scalar


WIDTH = 24
VL = 64
SCALE = 1.25
BIAS = 1.0
_INPUT_BYTES = WIDTH * 4
_PADDED_BYTES = WIDTH * VL * 4


@pto.simt
def extract_first_lane(
    padded_ub: pto.ptr(pto.f32, "ub"),
    compact_ub: pto.ptr(pto.f32, "ub"),
):
    index = pto.get_tid_x()
    value = scalar.load(padded_ub, scalar.index_cast(index * VL))
    scalar.store(value, compact_ub, scalar.index_cast(index))


@pto.jit(
    name="compact_f32_store_workaround",
    kernel_kind="vector",
    target="a5",
    backend="vpto",
    mode="explicit",
)
def compact_f32_store_workaround(
    input_gm: pto.ptr(pto.f32, "gm"),
    output_gm: pto.ptr(pto.f32, "gm"),
):
    zero = pto.const(0, dtype=pto.i64)
    input_ub = pto.castptr(zero, pto.ptr(pto.f32, "ub"))
    padded_ub = pto.addptr(input_ub, WIDTH)
    compact_ub = pto.addptr(padded_ub, WIDTH * VL)
    offset0 = pto.const(0, dtype=pto.index)
    mask = pto.vmi.create_mask(VL, size=VL)

    pto.mte_gm_ub(
        input_gm,
        input_ub,
        0,
        _INPUT_BYTES,
        nburst=(1, _INPUT_BYTES, _INPUT_BYTES),
    )
    pto.set_flag(pto.Pipe.MTE2, pto.Pipe.V, event_id=0)
    pto.wait_flag(pto.Pipe.MTE2, pto.Pipe.V, event_id=0)

    for index in range(WIDTH):
        value = pto.vmi.vload(
            pto.addptr(input_ub, index),
            offset0,
            size=VL,
            dist_mode="brc",
        )
        value = pto.vmi.vmuls(value, SCALE, mask)
        value = pto.vmi.vadds(value, BIAS, mask)
        pto.vmi.vstore(value, pto.addptr(padded_ub, index * VL), offset0, mask)

    pto.pipe_barrier(pto.Pipe.ALL)
    extract_first_lane[WIDTH, 1, 1](padded_ub, compact_ub)
    pto.pipe_barrier(pto.Pipe.ALL)
    pto.mte_ub_gm(
        compact_ub,
        output_gm,
        _INPUT_BYTES,
        nburst=(1, _INPUT_BYTES, _INPUT_BYTES),
    )
    pto.pipe_barrier(pto.Pipe.ALL)


def compile_kernel():
    return compact_f32_store_workaround.compile()


def _init_runtime(device: str):
    import torch
    import torch_npu  # noqa: F401

    torch.npu.config.allow_internal_format = False
    torch_npu.npu.set_compile_mode(jit_compile=False)
    torch.npu.set_device(device)
    return torch


def run(device: str) -> None:
    torch = _init_runtime(device)
    host_input = np.linspace(-3.0, 4.0, WIDTH, dtype=np.float32)
    expected = host_input * np.float32(SCALE) + np.float32(BIAS)
    input_tensor = torch.from_numpy(host_input).to(device)
    output_tensor = torch.full((WIDTH,), float("nan"), dtype=torch.float32, device=device)
    stream = torch.npu.current_stream()._as_parameter_  # noqa: SLF001
    compiled = compile_kernel()
    compiled[1, stream](input_tensor.data_ptr(), output_tensor.data_ptr())
    torch.npu.synchronize()
    actual = output_tensor.cpu().numpy()
    np.testing.assert_allclose(actual, expected, rtol=1e-6, atol=1e-6)
    print("PASS padded VMI workaround numeric check")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--compile-only", action="store_true")
    parser.add_argument("--emit-mlir", action="store_true")
    parser.add_argument("--device", default=os.environ.get("NPU_TEST_DEVICE", "npu:0"))
    args = parser.parse_args()
    if args.compile_only or args.emit_mlir:
        text = compile_kernel().mlir_text()
        if args.emit_mlir:
            print(text)
        else:
            assert "pto.vmi.vstore" in text and "pto.simt_launch" in text
            print(f"COMPILE PASS padded VMI workaround ({len(text)} bytes MLIR)")
        return 0
    run(args.device)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
