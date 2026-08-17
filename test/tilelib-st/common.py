#!/usr/bin/env python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path
import sys
import time

import numpy as np

_DEVICE = "npu:0"

def init_runtime():
    import torch
    import torch_npu  # noqa: F401

    torch.npu.config.allow_internal_format = False
    torch_npu.npu.set_compile_mode(jit_compile=False)
    torch.npu.set_device(_DEVICE)
    return torch


def npu_stream(torch):
    return torch.npu.current_stream()._as_parameter_  # noqa: SLF001


def emit_mlir(*kernels):
    from ptodsl import pto

    return pto.merge_jit_modules(*kernels)


def _collect_case_kernels(cases: list[dict]) -> list:
    kernels = []
    seen = set()
    for case in cases:
        kernel = case["kernel"]
        key = id(kernel)
        if key in seen:
            continue
        seen.add(key)
        kernels.append(kernel)
    return kernels


def _to_numpy_array(value):
    return np.array(value, copy=True)


def golden_output_case(
    name: str,
    kernel,
    *,
    inputs,
    expected,
    output_shape=None,
    output_dtype=None,
    output_index: int = -1,
    launch_args=None,
    rtol: float = 1e-5,
    atol: float = 1e-5,
):
    """Build a standard single-output TileLib ST case from host inputs and a golden."""

    def materialize_inputs():
        values = inputs() if callable(inputs) else inputs
        return [_to_numpy_array(value) for value in values]

    def materialize_expected(host_inputs):
        value = expected(*host_inputs) if callable(expected) else expected
        return _to_numpy_array(value)

    def make_case():
        host_inputs = materialize_inputs()
        golden = materialize_expected(host_inputs)
        out = np.zeros(
            output_shape or golden.shape,
            dtype=output_dtype or golden.dtype,
        )
        if launch_args is None:
            return [*host_inputs, out], golden
        extra_launch_args = launch_args(*host_inputs) if callable(launch_args) else list(launch_args)
        return [*host_inputs, out], golden, extra_launch_args

    def check_case(device_inputs, golden):
        actual = device_inputs[output_index].cpu().numpy()
        assert_close(actual, golden, rtol=rtol, atol=atol)

    return {
        "name": name,
        "kernel": kernel,
        "make_case": make_case,
        "check": check_case,
    }


def _list_cases(cases: list[dict]) -> None:
    for case in cases:
        print(case["name"])


def _load_module_from_path(path: Path):
    rel = path.resolve().relative_to(Path(__file__).resolve().parent)
    module_name = "_tilelib_st_" + "_".join(part.replace("-", "_") for part in rel.with_suffix("").parts)
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load TileLib ST module from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def discover_case_modules(root: Path | None = None) -> list:
    case_root = Path(root) if root is not None else Path(__file__).resolve().parent
    modules = []
    paths = []
    seen_paths = set()
    for path in sorted(case_root.glob("*.py")):
        if path.name not in {"common.py", "__main__.py", "run_tilelib_st.py"} and not path.name.startswith("_"):
            resolved = path.resolve()
            paths.append(path)
            seen_paths.add(resolved)
    for path in sorted(case_root.rglob("case.py")):
        resolved = path.resolve()
        if resolved in seen_paths:
            continue
        paths.append(path)
        seen_paths.add(resolved)

    for path in paths:
        module = _load_module_from_path(path)
        if getattr(module, "CASES", None):
            modules.append(module)
    return modules


def discover_cases(root: Path | None = None) -> list[dict]:
    discovered = []
    seen_names = {}
    for module in discover_case_modules(root):
        module_path = Path(getattr(module, "__file__", "<unknown>"))
        for case in module.CASES:
            name = case["name"]
            previous = seen_names.get(name)
            if previous is not None:
                raise RuntimeError(
                    f"Duplicate TileLib ST case name {name!r} discovered in {module_path} and {previous}"
                )
            seen_names[name] = module_path
            discovered.append(case)
    return discovered


def run_cases(cases: list[dict], *, emit_mlir_fn=None, argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--list", action="store_true", help="list discovered case names and exit")
    parser.add_argument("--emit-mlir", action="store_true", help="print merged MLIR module and exit")
    parser.add_argument("--case", type=str, default=None, help="run only the specified case by name")
    args = parser.parse_args(argv)

    if args.list:
        _list_cases(cases)
        return 0

    if args.emit_mlir:
        if emit_mlir_fn is None:
            raise RuntimeError("emit_mlir_fn is required when --emit-mlir is supported")
        print(emit_mlir_fn())
        return 0

    if args.case:
        filtered = [c for c in cases if c["name"] == args.case]
        if not filtered:
            raise RuntimeError(f"no case named {args.case!r}")
        cases = filtered

    torch = init_runtime()
    for case in cases:
        name = case["name"]
        kernel = case["kernel"]
        made_case = case["make_case"]()
        if len(made_case) == 2:
            inputs, expected = made_case
            launch_args = []
        elif len(made_case) == 3:
            inputs, expected, launch_args = made_case
        else:
            raise RuntimeError(
                f"TileLib ST case {name!r} make_case() must return 2 or 3 values, got {len(made_case)}"
            )

        device_inputs = [torch.from_numpy(array).to(_DEVICE) for array in inputs]
        stream = npu_stream(torch)

        t0 = time.perf_counter()
        compiled = kernel.compile()
        compile_s = time.perf_counter() - t0

        t0 = time.perf_counter()
        compiled[1, stream](*device_inputs, *launch_args)
        torch.npu.synchronize()
        launch_s = time.perf_counter() - t0

        case["check"](device_inputs, expected)
        print(f"PASS {name}  compile={compile_s:.3f}s launch={launch_s:.3f}s")
    print("All cases passed.")
    return 0


def run_module_cases(module_globals: dict, argv=None) -> int:
    cases = module_globals.get("CASES")
    if not cases:
        raise RuntimeError("TileLib ST module must define a non-empty CASES list")

    emit_mlir_fn = module_globals.get("EMIT_MLIR_FN")
    if emit_mlir_fn is None:
        kernels = module_globals.get("KERNELS")
        if kernels is None:
            kernels = _collect_case_kernels(cases)
        emit_mlir_fn = lambda: emit_mlir(*kernels)

    return run_cases(cases, emit_mlir_fn=emit_mlir_fn, argv=argv)


def run_discovered_cases(root: Path | None = None, argv=None) -> int:
    cases = discover_cases(root)
    if not cases:
        raise RuntimeError("No TileLib ST cases discovered")
    return run_cases(
        cases,
        emit_mlir_fn=lambda: emit_mlir(*_collect_case_kernels(cases)),
        argv=argv,
    )


def auto_main(module_globals: dict, argv=None) -> None:
    if module_globals.get("__name__") != "__main__":
        return
    raise SystemExit(run_module_cases(module_globals, argv=argv))


def assert_close(actual, expected, *, rtol=1e-5, atol=1e-5):
    np.testing.assert_allclose(actual, expected, rtol=rtol, atol=atol)
