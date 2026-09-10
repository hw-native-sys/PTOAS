#!/usr/bin/env python3
# coding=utf-8
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

from __future__ import annotations

from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
import subprocess
import sys
import tempfile

from ptodsl._runtime.toolchain import resolve_ptoas_binary as resolve_runtime_ptoas_binary


REPO_ROOT = Path(__file__).resolve().parents[2]
SAMPLE = REPO_ROOT / "test" / "samples" / "TPushTPop" / "ptodsl" / "local_c2v" / "kernel.py"


def expect(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def load_sample_module():
    spec = spec_from_file_location("ptodsl_tpush_tpop_local_c2v_sample", SAMPLE)
    expect(spec is not None and spec.loader is not None, f"unable to load sample from {SAMPLE}")
    module = module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def resolve_ptoas_binary() -> Path | None:
    try:
        return resolve_runtime_ptoas_binary()
    except FileNotFoundError:
        return None


def run_ptoas_frontend(ptoas_bin: Path, mlir_text: str) -> str:
    with tempfile.NamedTemporaryFile("w", suffix=".mlir", delete=False, encoding="utf-8") as handle:
        handle.write(mlir_text)
        input_path = Path(handle.name)

    try:
        result = subprocess.run(
            [
                str(ptoas_bin),
                "--pto-arch=a5",
                "--pto-backend=emitc",
                str(input_path),
                "--emit-pto-ir",
                "-o",
                "-",
            ],
            capture_output=True,
            text=True,
            check=False,
        )
    finally:
        input_path.unlink(missing_ok=True)

    expect(
        result.returncode == 0,
        f"sample should pass PTOAS frontend verification\nstdout:\n{result.stdout}\nstderr:\n{result.stderr}",
    )
    return result.stdout


def main() -> None:
    sample = load_sample_module()
    mlir_text = sample.emit_mlir()

    expect("pto.aic_initialize_pipe" in mlir_text, "Cube side should initialize the local pipe")
    expect("pto.aiv_initialize_pipe" in mlir_text, "Vector side should initialize the local pipe")
    expect("pto.tpush_to_aiv" in mlir_text, "Cube side should push to Vector")
    expect("pto.tpop_from_aic" in mlir_text, "Vector side should pop from Cube")
    expect("pto.tfree_from_aic" in mlir_text, "Vector side should free the consumed slot")

    ptoas_bin = resolve_ptoas_binary()
    if ptoas_bin is not None:
        frontend_text = run_ptoas_frontend(ptoas_bin, mlir_text)
        expect("pto.tpush" in frontend_text, "PTOAS output should contain lowered tpush")
        expect("pto.tpop" in frontend_text, "PTOAS output should contain lowered tpop")

    print("ptodsl_pipe_surface_sample_compile: PASS")


if __name__ == "__main__":
    main()
