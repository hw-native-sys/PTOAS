# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""MLIR → ptoas → bisheng native library build."""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

from .cache import (
    _content_digest,
    NativeBuildArtifacts,
    artifact_paths,
    is_native_build_current,
    write_manifest,
)
from .codegen import generate_launch_cpp, launch_symbol_name
from .toolchain import (
    aicore_arch_for_kernel_kind,
    common_include_flags,
    runtime_library_flags,
    resolve_bisheng,
    resolve_ptoas_binary,
)


def _run(cmd: list[str], *, cwd: Path | None = None) -> None:
    result = subprocess.run(cmd, cwd=str(cwd) if cwd else None, capture_output=True, text=True)
    if result.returncode != 0:
        output = (result.stdout or "") + (result.stderr or "")
        raise RuntimeError(
            f"command failed ({result.returncode}): {' '.join(cmd)}\n{output}"
        )


def _run_ptoas(
    mlir_path: Path,
    kernel_object: Path,
    *,
    target_arch: str,
    insert_sync: bool | None = None,
    backend: str | None = None,
    pto_level: str | None = None,
) -> None:
    ptoas = resolve_ptoas_binary()
    cmd = [
        str(ptoas),
        f"--pto-arch={target_arch}",
    ]
    if backend is not None:
        cmd.append(f"--pto-backend={backend}")
    if pto_level is not None:
        cmd.append(f"--pto-level={pto_level}")
    if insert_sync is True:
        cmd.append("--enable-insert-sync")
    cmd.extend([
        "--enable-tile-op-expand",
        str(mlir_path),
        "-o",
        str(kernel_object),
    ])
    _run(
        cmd
    )


def _effective_insert_sync(*, mode: str, insert_sync: bool | None) -> bool:
    if insert_sync is not None:
        return insert_sync
    return mode != "explicit"


def _effective_pto_level(*, mode: str) -> str | None:
    return "level3" if mode == "explicit" else None


def _source_ptoas_overrides(module_spec) -> dict:
    if getattr(module_spec, "jit_source", None) is None:
        return {}
    return {"backend": module_spec.backend}


def _compile_config_text(
    *,
    module_spec,
    effective_insert_sync: bool,
    effective_pto_level: str | None,
    ptoas_overrides: dict,
) -> str:
    return "\n".join(
        [
            f"target_arch={module_spec.target_arch}",
            f"kernel_kind={module_spec.kernel_kind}",
            f"mode={module_spec.mode}",
            f"insert_sync={effective_insert_sync}",
            f"pto_level={effective_pto_level}",
            f"backend={ptoas_overrides.get('backend')}",
            "enable_tile_op_expand=True",
        ]
    )


def _host_compile_flags() -> list[str]:
    return common_include_flags() + [
        "-std=gnu++17",
        "-O2",
        "-Wno-macro-redefined",
        "-Wno-ignored-attributes",
        "-Wno-unknown-attributes",
        "-xc++",
        "-include",
        "stdint.h",
        "-include",
        "stddef.h",
        "-fPIC",
    ]


def _kernel_compile_flags(kernel_kind: str | None, target_arch: str) -> list[str]:
    arch = aicore_arch_for_kernel_kind(kernel_kind, target_arch)
    return common_include_flags() + [
        "-std=gnu++17",
        "-O2",
        "-Wno-macro-redefined",
        "-Wno-ignored-attributes",
        "-Wno-unknown-attributes",
        "-fPIC",
        "-xcce",
        "-Xhost-start",
        "-Xhost-end",
        "-mllvm",
        "-cce-aicore-stack-size=0x8000",
        "-mllvm",
        "-cce-aicore-function-stack-size=0x8000",
        "-mllvm",
        "-cce-aicore-record-overflow=true",
        "-mllvm",
        "-cce-aicore-addr-transform",
        "-mllvm",
        "-cce-aicore-dcci-insert-for-scalar=false",
        f"--cce-aicore-arch={arch}",
    ]


def _compile_launch_cpp(
    launch_cpp: Path,
    launch_object: Path,
    *,
    kernel_kind: str | None,
    target_arch: str,
    export_macro: str,
) -> None:
    bisheng = resolve_bisheng()
    _run(
        [
            bisheng,
            *_kernel_compile_flags(kernel_kind, target_arch),
            f"-D{export_macro}",
            "-c",
            str(launch_cpp),
            "-o",
            str(launch_object),
        ]
    )


def _link_shared_library(
    launch_object: Path,
    kernel_object: Path,
    shared_library: Path,
    *,
    kernel_kind: str | None,
) -> None:
    bisheng = resolve_bisheng()
    soname = shared_library.name
    sim_mode = bool(os.environ.get("MSPROF_SIMULATOR_MODE"))
    _run(
        [
            bisheng,
            "-fPIC",
            "-shared",
            "--cce-fatobj-link",
            "-Wl,--no-undefined",
            f"-Wl,-soname,{soname}",
            "-o",
            str(shared_library),
            str(launch_object),
            str(kernel_object),
            *runtime_library_flags(sim_mode=sim_mode),
        ]
    )


def _native_build_config(module_spec):
    effective_insert_sync = _effective_insert_sync(
        mode=module_spec.mode,
        insert_sync=module_spec.insert_sync,
    )
    effective_pto_level = _effective_pto_level(mode=module_spec.mode)
    ptoas_overrides = _source_ptoas_overrides(module_spec)
    compile_config_text = _compile_config_text(
        module_spec=module_spec,
        effective_insert_sync=effective_insert_sync,
        effective_pto_level=effective_pto_level,
        ptoas_overrides=ptoas_overrides,
    )
    sim_mode = bool(os.environ.get("MSPROF_SIMULATOR_MODE"))
    link_config_text = "\n".join(runtime_library_flags(sim_mode=sim_mode))
    return {
        "insert_sync": effective_insert_sync,
        "pto_level": effective_pto_level,
        "ptoas_overrides": ptoas_overrides,
        "compile_config": compile_config_text,
        "link_config": link_config_text,
    }


def _compile_native_artifacts(artifacts, module_spec, *, effective_insert_sync, effective_pto_level, ptoas_overrides):
    _run_ptoas(
        artifacts.mlir_path,
        artifacts.kernel_object,
        target_arch=module_spec.target_arch,
        insert_sync=effective_insert_sync,
        pto_level=effective_pto_level,
        **ptoas_overrides,
    )
    launch_object = artifacts.cache_dir / "launch.o"
    _compile_launch_cpp(
        artifacts.launch_cpp,
        launch_object,
        kernel_kind=module_spec.kernel_kind,
        target_arch=module_spec.target_arch,
        export_macro=f"{module_spec.function_name}_EXPORTS",
    )
    _link_shared_library(
        launch_object,
        artifacts.kernel_object,
        artifacts.shared_library,
        kernel_kind=module_spec.kernel_kind,
    )


def _write_native_manifest(
    artifacts,
    *,
    ir_function_name,
    launch_symbol,
    mlir_text,
    launch_cpp_text,
    compile_config_text,
    link_config_text,
):
    write_manifest(
        artifacts,
        ir_function_name=ir_function_name,
        launch_symbol=launch_symbol,
        mlir_digest=_content_digest(mlir_text),
        launch_cpp_digest=_content_digest(launch_cpp_text),
        compile_config_digest=_content_digest(compile_config_text),
        link_config_digest=_content_digest(link_config_text),
    )


def build_native_library(
    *,
    py_name: str,
    module_spec,
    kernel_signature,
    mlir_text: str,
    specialization_key,
) -> tuple[Path, str]:
    """Build or reuse the shared library for one compiled specialization."""
    ir_function_name = module_spec.function_name
    artifacts = artifact_paths(py_name, ir_function_name, specialization_key)
    launch_symbol = launch_symbol_name(ir_function_name)
    launch_cpp_text = generate_launch_cpp(
        ir_function_name=ir_function_name,
        kernel_signature=kernel_signature,
    )
    build_config = _native_build_config(module_spec)
    effective_insert_sync = build_config["insert_sync"]
    effective_pto_level = build_config["pto_level"]
    ptoas_overrides = build_config["ptoas_overrides"]
    compile_config_text = build_config["compile_config"]
    link_config_text = build_config["link_config"]

    if is_native_build_current(
        artifacts,
        mlir_text=mlir_text,
        launch_cpp_text=launch_cpp_text,
        compile_config_text=compile_config_text,
        link_config_text=link_config_text,
    ):
        return artifacts.shared_library, launch_symbol

    artifacts.cache_dir.mkdir(parents=True, exist_ok=True)
    artifacts.mlir_path.write_text(mlir_text, encoding="utf-8")
    artifacts.launch_cpp.write_text(launch_cpp_text, encoding="utf-8")

    _compile_native_artifacts(
        artifacts,
        module_spec,
        effective_insert_sync=effective_insert_sync,
        effective_pto_level=effective_pto_level,
        ptoas_overrides=ptoas_overrides,
    )
    _write_native_manifest(
        artifacts,
        ir_function_name=ir_function_name,
        launch_symbol=launch_symbol,
        mlir_text=mlir_text,
        launch_cpp_text=launch_cpp_text,
        compile_config_text=compile_config_text,
        link_config_text=link_config_text,
    )
    return artifacts.shared_library, launch_symbol


__all__ = [
    "NativeBuildArtifacts",
    "artifact_paths",
    "build_native_library",
    "is_native_build_current",
]
