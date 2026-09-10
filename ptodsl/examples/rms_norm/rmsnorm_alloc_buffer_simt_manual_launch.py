# coding=utf-8
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""
Launch and validate the RMSNorm alloc_buffer/SIMT example with a hand-written
host wrapper that passes dynamic UB bytes explicitly.

This is intentionally a bypass of PTODSL's ``compiled[grid, stream](...)``
runtime launch path. The PTODSL kernel is still compiled to MLIR, then this
script builds a custom ``launch.cpp`` containing:

    kernel<<<grid, 82496, stream>>>(...)

Use it to validate the kernel while the generated PTODSL runtime wrapper does
not yet expose a dynamic UB launch-size parameter.

The build-and-launch pipeline:

1. Compile the PTODSL kernel to MLIR.
2. Generate a hand-written ``launch.cpp`` that declares the kernel and
   provides a C wrapper with a hardcoded ``dynSharedBytes``, then
   launches via ``kernel<<<grid, dynSharedBytes, stream>>>(...)``.
3. Lower MLIR to object code via ``ptoas``, then compile and link the
   C++ wrapper with the kernel object into a shared library.
4. Load the library with ``ctypes.CDLL``, call the wrapper with grid,
   stream, and pointer arguments.
5. Compare the output against a NumPy reference for validation.
"""

from __future__ import annotations

import argparse
import ctypes
import hashlib
from pathlib import Path
import time

import numpy as np


from ptodsl._runtime.native_build import (  # noqa: E402
    _compile_launch_cpp,
    _effective_insert_sync,
    _link_shared_library,
    _run_ptoas,
)

import rmsnorm_alloc_buffer_simt_launch_common as launch_common  # noqa: E402
from rmsnorm_alloc_buffer_simt_launch_common import (  # noqa: E402
    _DEVICE,
    _EPS,
    _HIDDEN_SIZE,
    _RSTD_GUARD_ELEMS,
    _SENTINEL,
    _Y_GUARD_ELEMS,
    CASES,
    FULL_CASE,
    Case,
    assert_guard_unchanged,
    compile_kernel,
    init_runtime,
    make_inputs,
    npu_stream,
    rmsnorm_reference,
)


_DYN_SHARED_BYTES = 82496


def _manual_launch_cpp(*, ir_function_name: str, launch_symbol: str, dyn_shared_bytes: int) -> str:
    return f"""#include <stdint.h>

#ifndef AICORE
#define AICORE [aicore]
#endif

extern "C" __global__ AICORE void {ir_function_name}(
    __gm__ float *X,
    __gm__ float *Y,
    __gm__ float *W,
    __gm__ float *RSTD,
    float eps);

extern "C" void {launch_symbol}(
    uint32_t grid,
    void *stream,
    float *X,
    float *Y,
    float *W,
    float *RSTD,
    float eps) {{
  constexpr uint32_t dynSharedBytes = {int(dyn_shared_bytes)};
  {ir_function_name}<<<grid, dynSharedBytes, stream>>>(
      (__gm__ float *)X,
      (__gm__ float *)Y,
      (__gm__ float *)W,
      (__gm__ float *)RSTD,
      eps);
}}
"""


def _manual_cache_dir(compiled, launch_cpp_text: str) -> Path:
    payload = "\n".join([
        compiled.mlir_text(),
        launch_cpp_text,
        repr(compiled.specialization_key),
    ]).encode("utf-8")
    digest = hashlib.sha256(payload).hexdigest()[:16]
    return Path.home() / ".cache" / "ptodsl" / f"{compiled._py_name}_manual_dynub_{digest}"


def build_manual_library(compiled, *, dyn_shared_bytes: int = _DYN_SHARED_BYTES) -> tuple[Path, str]:
    module_spec = compiled._module_spec
    ir_function_name = module_spec.function_name
    launch_symbol = f"ptodsl_manual_launch_{ir_function_name}"

    launch_cpp_text = _manual_launch_cpp(
        ir_function_name=ir_function_name,
        launch_symbol=launch_symbol,
        dyn_shared_bytes=dyn_shared_bytes,
    )
    cache_dir = _manual_cache_dir(compiled, launch_cpp_text)
    mlir_path = cache_dir / "kernel.mlir"
    kernel_object = cache_dir / "kernel.o"
    launch_cpp = cache_dir / "manual_launch.cpp"
    launch_object = cache_dir / "manual_launch.o"
    shared_library = cache_dir / f"lib{ir_function_name}_manual_dynub.so"

    if shared_library.is_file():
        return shared_library, launch_symbol

    cache_dir.mkdir(parents=True, exist_ok=True)
    mlir_path.write_text(compiled.mlir_text(), encoding="utf-8")
    launch_cpp.write_text(launch_cpp_text, encoding="utf-8")

    _run_ptoas(
        mlir_path,
        kernel_object,
        target_arch=module_spec.target_arch,
        insert_sync=_effective_insert_sync(
            mode=module_spec.mode,
            insert_sync=module_spec.insert_sync,
        ),
    )
    _compile_launch_cpp(
        launch_cpp,
        launch_object,
        kernel_kind=module_spec.kernel_kind,
        target_arch=module_spec.target_arch,
        export_macro=f"{ir_function_name}_EXPORTS",
    )
    _link_shared_library(
        launch_object,
        kernel_object,
        shared_library,
        kernel_kind=module_spec.kernel_kind,
    )
    return shared_library, launch_symbol


def _manual_launch(compiled, *, grid: int, stream, x_ptr: int, y_ptr: int, w_ptr: int, rstd_ptr: int, eps: float):
    lib_path, launch_symbol = build_manual_library(compiled)
    lib = ctypes.CDLL(str(lib_path))
    launch = getattr(lib, launch_symbol)
    launch.argtypes = [
        ctypes.c_uint32,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_float,
    ]
    launch.restype = None
    launch(
        ctypes.c_uint32(grid),
        ctypes.c_void_p(int(getattr(stream, "value", stream))),
        ctypes.c_void_p(x_ptr),
        ctypes.c_void_p(y_ptr),
        ctypes.c_void_p(w_ptr),
        ctypes.c_void_p(rstd_ptr),
        ctypes.c_float(eps),
    )


def run_case_manual(case: Case, torch) -> None:
    x, w = make_inputs(case)
    y_ref, rstd_ref = rmsnorm_reference(x, w, _EPS)

    x_t = torch.from_numpy(x).to(_DEVICE)
    w_t = torch.from_numpy(w).to(_DEVICE)

    y_storage = torch.full(
        (case.tokens * _HIDDEN_SIZE + _Y_GUARD_ELEMS,),
        float(_SENTINEL),
        dtype=torch.float32,
        device=_DEVICE,
    )
    rstd_storage = torch.full(
        (case.tokens + _RSTD_GUARD_ELEMS,),
        float(_SENTINEL),
        dtype=torch.float32,
        device=_DEVICE,
    )

    stream = npu_stream(torch)

    t0 = time.perf_counter()
    compiled = compile_kernel(case)
    compile_s = time.perf_counter() - t0

    t0 = time.perf_counter()
    _manual_launch(
        compiled,
        grid=case.n_cores,
        stream=stream,
        x_ptr=x_t.data_ptr(),
        y_ptr=y_storage.data_ptr(),
        w_ptr=w_t.data_ptr(),
        rstd_ptr=rstd_storage.data_ptr(),
        eps=float(_EPS),
    )
    torch.npu.synchronize()
    launch_s = time.perf_counter() - t0

    y_out = y_storage[: case.tokens * _HIDDEN_SIZE].cpu().numpy().reshape(case.tokens, _HIDDEN_SIZE)
    rstd_out = rstd_storage[: case.tokens].cpu().numpy()
    y_guard = y_storage[case.tokens * _HIDDEN_SIZE :].cpu().numpy()
    rstd_guard = rstd_storage[case.tokens :].cpu().numpy()

    np.testing.assert_allclose(rstd_out, rstd_ref, rtol=case.rtol, atol=case.rstd_atol)
    np.testing.assert_allclose(y_out, y_ref, rtol=case.rtol, atol=case.y_atol)
    assert_guard_unchanged("Y", y_guard)
    assert_guard_unchanged("RSTD", rstd_guard)

    y_diff = float(np.max(np.abs(y_out - y_ref))) if y_out.size else 0.0
    rstd_diff = float(np.max(np.abs(rstd_out - rstd_ref))) if rstd_out.size else 0.0
    simt_config = getattr(case, "simt_config", "threads=128 rounds=8 lanes=4")
    print(
        f"PASS {case.name} manual-dynub  "
        f"grid={case.n_cores} tokens={case.tokens} {simt_config} "
        f"dynSharedBytes={_DYN_SHARED_BYTES} "
        f"compile={compile_s:.3f}s launch={launch_s:.3f}s "
        f"max|Y|={y_diff:.3e} max|RSTD|={rstd_diff:.3e}"
    )


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default=_DEVICE, help="torch NPU device, default: npu:0")
    parser.add_argument(
        "--case",
        choices=[case.name for case in CASES] + [FULL_CASE.name, "all"],
        default="all",
    )
    parser.add_argument("--include-full", action="store_true", help="include the 64-core x 64-token full case")
    args = parser.parse_args(argv)

    launch_common._DEVICE = args.device
    globals()["_DEVICE"] = args.device

    selected = list(CASES)
    if args.include_full:
        selected.append(FULL_CASE)
    if args.case != "all":
        all_cases = {case.name: case for case in selected + [FULL_CASE]}
        selected = [all_cases[args.case]]

    torch = init_runtime()
    for case in selected:
        run_case_manual(case, torch)
    print("All RMSNorm manual dynamic-UB cases passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
