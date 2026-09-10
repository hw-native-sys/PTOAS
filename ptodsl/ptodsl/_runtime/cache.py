# coding=utf-8
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Artifact cache layout for JIT-compiled native libraries."""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path


def _default_cache_root() -> Path:
    override = os.environ.get("PTODSL_CACHE_DIR")
    if override:
        return Path(override)
    return Path.home() / ".cache" / "ptodsl"


def _specialization_digest(specialization_key) -> str:
    payload = repr(specialization_key).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:16]


def _content_digest(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class NativeBuildArtifacts:
    """Paths to one compiled native kernel specialization."""

    cache_dir: Path
    mlir_path: Path
    kernel_object: Path
    launch_cpp: Path
    shared_library: Path
    manifest_path: Path


def artifact_paths(py_name: str, ir_function_name: str, specialization_key) -> NativeBuildArtifacts:
    """Return stable artifact paths for one compiled specialization."""
    digest = _specialization_digest(specialization_key)
    safe_name = py_name.replace("/", "_")
    cache_dir = _default_cache_root() / f"{safe_name}_{digest}"
    lib_name = f"lib{ir_function_name}.so"
    return NativeBuildArtifacts(
        cache_dir=cache_dir,
        mlir_path=cache_dir / "kernel.mlir",
        kernel_object=cache_dir / "kernel.o",
        launch_cpp=cache_dir / "launch.cpp",
        shared_library=cache_dir / lib_name,
        manifest_path=cache_dir / "manifest.json",
    )


def write_manifest(
    artifacts: NativeBuildArtifacts,
    *,
    ir_function_name: str,
    launch_symbol: str,
    mlir_digest: str,
    launch_cpp_digest: str,
    compile_config_digest: str,
    link_config_digest: str,
) -> None:
    artifacts.cache_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "ir_function_name": ir_function_name,
        "launch_symbol": launch_symbol,
        "shared_library": str(artifacts.shared_library),
        "mlir_digest": mlir_digest,
        "launch_cpp_digest": launch_cpp_digest,
        "compile_config_digest": compile_config_digest,
        "link_config_digest": link_config_digest,
    }
    artifacts.manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")


def read_manifest(artifacts: NativeBuildArtifacts) -> dict:
    return json.loads(artifacts.manifest_path.read_text(encoding="utf-8"))


def is_native_build_current(
    artifacts: NativeBuildArtifacts,
    *,
    mlir_text: str,
    launch_cpp_text: str,
    compile_config_text: str,
    link_config_text: str,
) -> bool:
    required = (
        artifacts.mlir_path,
        artifacts.kernel_object,
        artifacts.launch_cpp,
        artifacts.shared_library,
        artifacts.manifest_path,
    )
    if not all(path.is_file() for path in required):
        return False

    try:
        manifest = read_manifest(artifacts)
    except Exception:
        return False

    return (
        manifest.get("mlir_digest") == _content_digest(mlir_text)
        and manifest.get("launch_cpp_digest") == _content_digest(launch_cpp_text)
        and manifest.get("compile_config_digest") == _content_digest(compile_config_text)
        and manifest.get("link_config_digest") == _content_digest(link_config_text)
    )


__all__ = [
    "NativeBuildArtifacts",
    "artifact_paths",
    "_content_digest",
    "is_native_build_current",
    "read_manifest",
    "write_manifest",
]
