# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""
Minimal PTODSL mixed-backend example with one kernel module.

Shape:
  - entry kernel: EmitC backend with tile load/compute/store
  - kernel module: VPTO backend, explicit mode

This script supports both:

  - compile-only inspection via ``--emit-mlir``
  - end-to-end launch via ``compile -> launch -> accuracy check``

The kernel module ABI intentionally stays within the current C-ABI-compatible
subset: GM pointers plus scalars. Tile values do not cross the module
boundary. The EmitC entry owns the tile-first row load/add/store path, then
passes one row GM pointer into the VPTO child for explicit in-place vector
post-processing.
"""

import argparse
import time

import numpy as np

from ptodsl import pto

_DEVICE = "npu:0"
_ROW_ELEMS = 64
_ROW_BYTES = _ROW_ELEMS * 4
_ENTRY_BIAS = 1.0
_HELPER_SCALE = 2.0


@pto.jit(target="a5", entry=False, backend="vpto", mode="explicit", insert_sync=False)
def scale_row_kernel_module(
    base_gm: pto.ptr(pto.f32, "gm"),
    row: pto.i32,
):
    with pto.tileop():
        c0_i64 = pto.const(0, dtype=pto.i64)
        row_offset = row * _ROW_ELEMS
        row_gm = pto.addptr(base_gm, row_offset)
        ub_row = pto.castptr(c0_i64, pto.ptr(pto.f32, "ub"))
        vec_offset = pto.const(0)

        pto.get_buf(pto.Pipe.MTE2, 0)
        pto.mte_gm_ub(row_gm, ub_row, 0, _ROW_BYTES, nburst=(1, _ROW_BYTES, _ROW_BYTES))
        pto.rls_buf(pto.Pipe.MTE2, 0)

        full_mask = pto.make_mask(pto.f32, pto.const(_ROW_ELEMS, dtype=pto.i32))
        row_vec = pto.vlds(ub_row, vec_offset)
        row_vec = pto.vmuls(row_vec, _HELPER_SCALE, full_mask)
        pto.vsts(row_vec, ub_row, vec_offset, full_mask)

        pto.get_buf(pto.Pipe.MTE3, 0)
        pto.mte_ub_gm(ub_row, row_gm, _ROW_BYTES, nburst=(1, _ROW_BYTES, _ROW_BYTES))
        pto.rls_buf(pto.Pipe.MTE3, 0)
        pto.pipe_barrier(pto.Pipe.ALL)


@pto.jit(target="a5", backend="emitc")
def emitc_entry_calls_vpto_module(
    x_ptr: pto.ptr(pto.f32, "gm"),
    o_ptr: pto.ptr(pto.f32, "gm"),
    rows: pto.i32,
):
    x_view = pto.make_tensor_view(x_ptr, shape=[rows, _ROW_ELEMS], strides=[_ROW_ELEMS, 1])
    o_view = pto.make_tensor_view(o_ptr, shape=[rows, _ROW_ELEMS], strides=[_ROW_ELEMS, 1])
    x_row_tile = pto.alloc_tile(shape=[1, _ROW_ELEMS], dtype=pto.f32)
    o_row_tile = pto.alloc_tile(shape=[1, _ROW_ELEMS], dtype=pto.f32)

    with pto.for_(0, rows, step=1) as row:
        x_part = pto.partition_view(x_view, offsets=[row, 0], sizes=[1, _ROW_ELEMS])
        o_part = pto.partition_view(o_view, offsets=[row, 0], sizes=[1, _ROW_ELEMS])
        pto.tile.load(x_part, x_row_tile)
        pto.tile.adds(x_row_tile, _ENTRY_BIAS, o_row_tile)
        pto.tile.store(o_row_tile, o_part)
        scale_row_kernel_module(o_ptr, row)


def emit_mlir() -> str:
    return emitc_entry_calls_vpto_module.compile().mlir_text()


def init_runtime():
    try:
        import torch
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "mixed_backend_kernel_module.py launch requires a Python environment with torch installed"
        ) from exc
    try:
        import torch_npu
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "mixed_backend_kernel_module.py launch requires a Python environment with torch_npu installed"
        ) from exc

    torch.npu.config.allow_internal_format = False
    torch_npu.npu.set_compile_mode(jit_compile=False)
    torch.npu.set_device(_DEVICE)
    return torch


def current_stream(torch):
    return torch.npu.current_stream()._as_parameter_  # noqa: SLF001


def run_demo(*, rows: int) -> None:
    if rows <= 0:
        raise ValueError("rows must be positive")

    torch = init_runtime()
    rng = np.random.RandomState(20260601)
    host_x = rng.randn(rows, _ROW_ELEMS).astype(np.float32)
    host_ref = (host_x + _ENTRY_BIAS) * _HELPER_SCALE

    x = torch.from_numpy(host_x).to(_DEVICE)
    o = torch.empty((rows, _ROW_ELEMS), dtype=torch.float32, device=_DEVICE)
    stream = current_stream(torch)

    t0 = time.perf_counter()
    compiled = emitc_entry_calls_vpto_module.compile()
    compile_s = time.perf_counter() - t0

    t0 = time.perf_counter()
    compiled[1, stream](x.data_ptr(), o.data_ptr(), rows)
    torch.npu.synchronize()
    launch_s = time.perf_counter() - t0

    host_out = o.cpu().numpy()
    np.testing.assert_allclose(host_out, host_ref, rtol=1e-6, atol=1e-6)
    print(
        f"PASS mixed-backend-kernel-module rows={rows} cols={_ROW_ELEMS} "
        f"compile={compile_s:.3f}s launch={launch_s:.3f}s"
    )


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--emit-mlir",
        action="store_true",
        help="print compiled MLIR and exit",
    )
    parser.add_argument(
        "--rows",
        type=int,
        default=4,
        help="number of rows to launch in the demo run",
    )
    args = parser.parse_args(argv)

    if args.emit_mlir:
        print(emit_mlir())
        return 0

    run_demo(rows=args.rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
