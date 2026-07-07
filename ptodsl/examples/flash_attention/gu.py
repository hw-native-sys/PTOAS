# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""
PTODSL GU helper for Flash Attention vector kernels.

This file mirrors the launchable shape used by ``softmax.py``:

- ``fa_gu_init_vpto_kernel`` / ``fa_gu_update_vpto_kernel`` are ptr-ABI VPTO
  child modules.
- ``fa_gu_init_vpto`` / ``fa_gu_update_vpto`` are Tile-ABI ``@pto.tileop``
  adapters for callers such as ``flash_attention_vf_fusion.py``.
- ``fa_gu_init_vpto_validate`` / ``fa_gu_update_vpto_validate`` are host-visible
  launch wrappers for standalone validation.
"""

import argparse
from pathlib import Path
import time

import numpy as np

from ptodsl import pto


def gu_init_reference(pv: np.ndarray) -> np.ndarray:
    """Return ``O`` for the init GU pass: ``O = PV``."""
    pv_f32 = np.asarray(pv, dtype=np.float32)
    if pv_f32.ndim != 2:
        raise ValueError(f"pv must be 2D, got shape {pv_f32.shape}")
    return pv_f32.copy()


def gu_update_reference(o_prev: np.ndarray, pv: np.ndarray, exp_max: np.ndarray) -> np.ndarray:
    """Return ``O`` for the update GU pass: ``O = O * exp_max + PV``."""
    o_prev_f32 = np.asarray(o_prev, dtype=np.float32)
    pv_f32 = np.asarray(pv, dtype=np.float32)
    if o_prev_f32.ndim != 2:
        raise ValueError(f"o_prev must be 2D, got shape {o_prev_f32.shape}")
    if pv_f32.shape != o_prev_f32.shape:
        raise ValueError(f"pv shape {pv_f32.shape} must match o_prev shape {o_prev_f32.shape}")
    rows = o_prev_f32.shape[0]
    exp_f32 = np.asarray(exp_max, dtype=np.float32).reshape(rows, 1)
    return o_prev_f32 * exp_f32 + pv_f32


@pto.jit(
    name="fa_gu_init_vpto_kernel",
    target="a5",
    entry=False,
    backend="vpto",
    mode="explicit",
    kernel_kind="vector",
    insert_sync=False,
)
def fa_gu_init_vpto_kernel(
    pv_ptr: pto.ptr(pto.f32, "ub"),
    o_ptr: pto.ptr(pto.f32, "ub"),
    rows: pto.i32,
    cols: pto.i32,
):
    lanes = pto.elements_per_vreg(pto.f32)

    for row in range(0, rows, 1):
        row_base = row * cols
        remained = cols
        for col in range(0, cols, lanes):
            mask, remained = pto.make_mask(pto.f32, remained)
            pv_vec = pto.vlds(pv_ptr, row_base + col, dist="NORM")
            pto.vsts(pv_vec, o_ptr, row_base + col, mask, dist="NORM_B32")


@pto.jit(
    name="fa_gu_update_vpto_kernel",
    target="a5",
    entry=False,
    backend="vpto",
    mode="explicit",
    kernel_kind="vector",
    insert_sync=False,
)
def fa_gu_update_vpto_kernel(
    o_ptr: pto.ptr(pto.f32, "ub"),
    pv_ptr: pto.ptr(pto.f32, "ub"),
    exp_max_ptr: pto.ptr(pto.f32, "ub"),
    rows: pto.i32,
    cols: pto.i32,
):
    lanes = pto.elements_per_vreg(pto.f32)

    for row in range(0, rows, 1):
        row_base = row * cols
        exp_row = pto.vlds(exp_max_ptr, row, dist="BRC_B32")
        remained = cols
        for col in range(0, cols, lanes):
            mask, remained = pto.make_mask(pto.f32, remained)
            o_vec = pto.vlds(o_ptr, row_base + col, dist="NORM")
            pv_vec = pto.vlds(pv_ptr, row_base + col, dist="NORM")
            out_vec = pto.vadd(pto.vmul(o_vec, exp_row, mask), pv_vec, mask)
            pto.vsts(out_vec, o_ptr, row_base + col, mask, dist="NORM_B32")


@pto.tileop
def fa_gu_init_vpto(
    pv_tile: pto.Tile,
    o_tile: pto.Tile,
):
    rows, cols = o_tile.valid_shape
    fa_gu_init_vpto_kernel(
        pv_tile.as_ptr(),
        o_tile.as_ptr(),
        rows,
        cols,
    )


@pto.tileop
def fa_gu_update_vpto(
    o_tile: pto.Tile,
    pv_tile: pto.Tile,
    exp_max: pto.Tile,
):
    rows, cols = o_tile.valid_shape
    fa_gu_update_vpto_kernel(
        o_tile.as_ptr(),
        pv_tile.as_ptr(),
        exp_max.as_ptr(),
        rows,
        cols,
    )


@pto.jit(entry=True, target="a5", backend="emitc", mode="auto", insert_sync=True)
def fa_gu_init_vpto_validate(
    pv_gm: pto.ptr(pto.f32, "gm"),
    o_gm: pto.ptr(pto.f32, "gm"),
    *,
    BR: pto.const_expr = 32,
    BC: pto.const_expr = 128,
):
    pv_view = pto.make_tensor_view(
        pv_gm,
        shape=[1, 1, 1, BR, BC],
        strides=[BR * BC, BR * BC, BR * BC, BC, 1],
    )
    o_view = pto.make_tensor_view(
        o_gm,
        shape=[1, 1, 1, BR, BC],
        strides=[BR * BC, BR * BC, BR * BC, BC, 1],
    )

    pv_tile = pto.alloc_tile(shape=[BR, BC], dtype=pto.f32, valid_shape=[BR, BC])
    o_tile = pto.alloc_tile(shape=[BR, BC], dtype=pto.f32, valid_shape=[BR, BC])

    pto.tile.load(
        pto.partition_view(pv_view, offsets=[0, 0, 0, 0, 0], sizes=[1, 1, 1, BR, BC]),
        pv_tile,
    )
    fa_gu_init_vpto(pv_tile, o_tile)
    pto.tile.store(
        o_tile,
        pto.partition_view(o_view, offsets=[0, 0, 0, 0, 0], sizes=[1, 1, 1, BR, BC]),
    )


@pto.jit(entry=True, target="a5", backend="emitc", mode="auto", insert_sync=True)
def fa_gu_update_vpto_validate(
    o_gm: pto.ptr(pto.f32, "gm"),
    pv_gm: pto.ptr(pto.f32, "gm"),
    exp_max_gm: pto.ptr(pto.f32, "gm"),
    *,
    BR: pto.const_expr = 32,
    BC: pto.const_expr = 128,
):
    o_view = pto.make_tensor_view(
        o_gm,
        shape=[1, 1, 1, BR, BC],
        strides=[BR * BC, BR * BC, BR * BC, BC, 1],
    )
    pv_view = pto.make_tensor_view(
        pv_gm,
        shape=[1, 1, 1, BR, BC],
        strides=[BR * BC, BR * BC, BR * BC, BC, 1],
    )
    exp_max_view = pto.make_tensor_view(
        exp_max_gm,
        shape=[1, 1, 1, BR, 1],
        strides=[BR, BR, BR, 1, 1],
    )

    o_tile = pto.alloc_tile(shape=[BR, BC], dtype=pto.f32, valid_shape=[BR, BC])
    pv_tile = pto.alloc_tile(shape=[BR, BC], dtype=pto.f32, valid_shape=[BR, BC])
    exp_max = pto.alloc_tile(shape=[BR, 1], dtype=pto.f32, valid_shape=[BR, 1], blayout="ColMajor")

    pto.tile.load(
        pto.partition_view(o_view, offsets=[0, 0, 0, 0, 0], sizes=[1, 1, 1, BR, BC]),
        o_tile,
    )
    pto.tile.load(
        pto.partition_view(pv_view, offsets=[0, 0, 0, 0, 0], sizes=[1, 1, 1, BR, BC]),
        pv_tile,
    )
    pto.tile.load(
        pto.partition_view(exp_max_view, offsets=[0, 0, 0, 0, 0], sizes=[1, 1, 1, BR, 1]),
        exp_max,
    )
    fa_gu_update_vpto(o_tile, pv_tile, exp_max)
    pto.tile.store(
        o_tile,
        pto.partition_view(o_view, offsets=[0, 0, 0, 0, 0], sizes=[1, 1, 1, BR, BC]),
    )


@pto.jit(target="a5", mode="explicit")
def fa_gu_vpto_probe(
    *,
    BR: pto.const_expr = 8,
    BC: pto.const_expr = 64,
    INIT: pto.const_expr = False,
):
    pv_tile = pto.alloc_tile(shape=[BR, BC], dtype=pto.f32, valid_shape=[BR, BC])
    o_tile = pto.alloc_tile(shape=[BR, BC], dtype=pto.f32, valid_shape=[BR, BC])
    exp_max = pto.alloc_tile(shape=[BR, 1], dtype=pto.f32, valid_shape=[BR, 1], blayout="ColMajor")

    if INIT:
        fa_gu_init_vpto(pv_tile, o_tile)
    else:
        fa_gu_update_vpto(o_tile, pv_tile, exp_max)


__all__ = [
    "fa_gu_init_vpto_kernel",
    "fa_gu_update_vpto_kernel",
    "fa_gu_init_vpto",
    "fa_gu_update_vpto",
    "fa_gu_init_vpto_validate",
    "fa_gu_update_vpto_validate",
    "gu_init_reference",
    "gu_update_reference",
    "fa_gu_vpto_probe",
]


_DEVICE = "npu:0"


def emit_gu_mlir(*, init: bool, br: int, bc: int) -> str:
    compiled = (
        fa_gu_init_vpto_validate.compile(BR=br, BC=bc)
        if init
        else fa_gu_update_vpto_validate.compile(BR=br, BC=bc)
    )
    return compiled.mlir_text()


def compile_gu_kernel(*, init: bool, br: int, bc: int):
    return (
        fa_gu_init_vpto_validate.compile(BR=br, BC=bc)
        if init
        else fa_gu_update_vpto_validate.compile(BR=br, BC=bc)
    )


def init_runtime():
    try:
        import torch
        import torch_npu  # noqa: F401
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "GU launch validation requires a Python environment with both "
            "`torch` and `torch_npu` installed"
        ) from exc

    torch.npu.config.allow_internal_format = False
    torch_npu.npu.set_compile_mode(jit_compile=False)
    torch.npu.set_device(_DEVICE)
    return torch


def npu_stream(torch):
    return torch.npu.current_stream()._as_parameter_  # noqa: SLF001


def _to_device(torch, array: np.ndarray):
    return torch.from_numpy(np.ascontiguousarray(array)).to(_DEVICE)


def _assert_close(name: str, got: np.ndarray, ref: np.ndarray, *, rtol: float, atol: float) -> None:
    try:
        np.testing.assert_allclose(got, ref, rtol=rtol, atol=atol)
    except AssertionError as exc:
        diff = np.max(np.abs(got - ref))
        raise AssertionError(f"{name} mismatch, max_abs_diff={diff}\n{exc}") from exc


def run_demo(
    *,
    init: bool,
    br: int,
    bc: int,
    seed: int = 20260606,
) -> None:
    torch = init_runtime()
    rng = np.random.RandomState(seed)
    stream = npu_stream(torch)

    t0 = time.perf_counter()
    compiled = compile_gu_kernel(init=init, br=br, bc=bc)
    compile_s = time.perf_counter() - t0

    if init:
        pv = rng.uniform(-4.0, 4.0, size=(br, bc)).astype(np.float32)
        ref_o = gu_init_reference(pv)

        pv_t = _to_device(torch, pv)
        o_t = _to_device(torch, np.zeros((br, bc), dtype=np.float32))

        t0 = time.perf_counter()
        compiled[1, stream](pv_t.data_ptr(), o_t.data_ptr())
        torch.npu.synchronize()
        launch_s = time.perf_counter() - t0

        got_o = o_t.cpu().numpy()
        _assert_close("init.o", got_o, ref_o, rtol=1e-6, atol=1e-6)
        print(f"PASS gu-init br={br} bc={bc} compile={compile_s:.3f}s launch={launch_s:.3f}s")
        return

    o_prev = rng.uniform(-4.0, 4.0, size=(br, bc)).astype(np.float32)
    pv = rng.uniform(-4.0, 4.0, size=(br, bc)).astype(np.float32)
    exp_max = rng.uniform(0.25, 1.25, size=(br, 1)).astype(np.float32)
    ref_o = gu_update_reference(o_prev, pv, exp_max)

    o_t = _to_device(torch, o_prev)
    pv_t = _to_device(torch, pv)
    exp_max_t = _to_device(torch, exp_max)

    t0 = time.perf_counter()
    compiled[1, stream](o_t.data_ptr(), pv_t.data_ptr(), exp_max_t.data_ptr())
    torch.npu.synchronize()
    launch_s = time.perf_counter() - t0

    got_o = o_t.cpu().numpy()
    _assert_close("update.o", got_o, ref_o, rtol=2e-5, atol=2e-5)
    print(f"PASS gu-update br={br} bc={bc} compile={compile_s:.3f}s launch={launch_s:.3f}s")


def build_arg_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--emit-mlir",
        action="store_true",
        help="print compiled MLIR and exit",
    )
    parser.add_argument(
        "--init",
        action="store_true",
        help="emit or run init validation; default runs both init and update unless --update is set",
    )
    parser.add_argument(
        "--update",
        action="store_true",
        help="emit or run update validation; default runs both init and update unless --init is set",
    )
    parser.add_argument("--br", type=int, default=32)
    parser.add_argument("--bc", type=int, default=128)
    parser.add_argument("--seed", type=int, default=20260606)
    parser.add_argument("-o", "--output", default="-", help="output MLIR path, or '-' for stdout")
    return parser


def main(argv=None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    run_init = args.init or not args.update
    run_update = args.update or not args.init

    if args.emit_mlir:
        if run_init:
            mlir_text = emit_gu_mlir(init=True, br=args.br, bc=args.bc)
            if args.output == "-":
                print(mlir_text)
            else:
                Path(args.output).write_text(mlir_text, encoding="utf-8")
        if run_update:
            mlir_text = emit_gu_mlir(init=False, br=args.br, bc=args.bc)
            if args.output == "-":
                print(mlir_text)
            else:
                suffix = ".update" if run_init and args.output != "-" else ""
                out_path = Path(args.output + suffix) if suffix else Path(args.output)
                out_path.write_text(mlir_text, encoding="utf-8")
        return 0

    # if run_init:
    #     run_demo(init=True, br=args.br, bc=args.bc, seed=args.seed)
    if run_update:
        run_demo(init=False, br=args.br, bc=args.bc, seed=args.seed + 1)
    print("All requested GU validation cases passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
