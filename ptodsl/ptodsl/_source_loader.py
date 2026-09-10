# coding=utf-8
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Source-backed module loading for ``@pto.jit(source=...)``."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

from ._context import make_context
from ._diagnostics import (
    jit_source_abi_error,
    jit_source_entry_error,
    jit_source_file_error,
)

from ptoas.mlir.ir import Location, Module


@dataclass(frozen=True)
class SourceModuleArtifact:
    """Parsed source-backed module plus identity metadata."""

    module: Module
    mlir_text: str
    resolved_path: Path | None
    source_kind: str
    content_digest: str


class SourceModuleLoader:
    """Resolve, parse, and verify one source-backed JIT module."""

    def __init__(self, module_spec, kernel_signature):
        self._module_spec = module_spec
        self._kernel_signature = kernel_signature
        self._artifact: SourceModuleArtifact | None = None

    @property
    def source(self) -> str:
        source = self._module_spec.jit_source
        if source is None:
            raise RuntimeError("source-backed loader requires KernelModuleSpec.jit_source")
        return source

    def cache_identity(self) -> tuple:
        """Return source identity for the specialization key."""
        artifact = self._load()
        return (
            "source",
            str(artifact.resolved_path),
            artifact.content_digest,
            self._module_spec.function_name,
            self._module_spec.entry,
            self._module_spec.target_arch,
            self._module_spec.kernel_kind,
            self._module_spec.backend,
            self._module_spec.mode,
            self._module_spec.insert_sync,
            self._module_spec.module_style,
        )

    def build_module(self):
        """Return ``(module, metadata)`` for ``ModuleArtifact``."""
        artifact = self._load()
        metadata = {
            "mlir_text": artifact.mlir_text,
            "source_kind": artifact.source_kind,
            "source_digest": artifact.content_digest,
        }
        if artifact.resolved_path is not None:
            metadata["source_path"] = str(artifact.resolved_path)
        return artifact.module, metadata

    def _load(self) -> SourceModuleArtifact:
        if self._artifact is None:
            resolved_path, mlir_text, source_kind = self._resolve_source()
            content_digest = hashlib.sha256(mlir_text.encode("utf-8")).hexdigest()
            ctx = make_context()
            with ctx, Location.unknown():
                module = Module.parse(mlir_text)
                entry = self._select_entry(module, resolved_path)
                self._verify_entry_abi(entry, resolved_path)
                module.operation.verify()
            self._artifact = SourceModuleArtifact(
                module=module,
                mlir_text=mlir_text,
                resolved_path=resolved_path,
                source_kind=source_kind,
                content_digest=content_digest,
            )
        return self._artifact

    def _resolve_source_path(self) -> Path:
        raw_path = Path(self.source)
        if raw_path.is_absolute():
            return raw_path.resolve()
        declaring_file = self._module_spec.source_file
        if declaring_file:
            return (Path(declaring_file).resolve().parent / raw_path).resolve()
        return raw_path.resolve()

    def _looks_like_inline_source(self, source: str) -> bool:
        stripped = source.lstrip()
        if "\n" in source or "\r" in source:
            return True
        return stripped.startswith("module {") or stripped.startswith("builtin.module {") or (
            "module {" in stripped and "func.func @" in stripped
        )

    def _resolve_source(self) -> tuple[Path | None, str, str]:
        if self._looks_like_inline_source(self.source):
            return None, self.source, "inline"
        resolved_path = self._resolve_source_path()
        return resolved_path, self._read_source_text(resolved_path), "path"

    def _read_source_text(self, resolved_path: Path) -> str:
        try:
            return resolved_path.read_text(encoding="utf-8")
        except FileNotFoundError as exc:
            raise jit_source_file_error(self.source, resolved_path, "file does not exist") from exc
        except OSError as exc:
            raise jit_source_file_error(self.source, resolved_path, str(exc)) from exc

    def _select_entry(self, module: Module, resolved_path: Path | None):
        matches = []
        for op in _walk_ops(module.operation):
            if op.operation.name != "func.func":
                continue
            if getattr(op, "is_external", False):
                continue
            if _symbol_name(op) == self._module_spec.function_name:
                matches.append(op)

        if not matches:
            raise jit_source_entry_error(
                _source_location_label(self.source, resolved_path),
                self._module_spec.function_name,
                "missing non-declaration func.func with this symbol name",
            )
        if len(matches) > 1:
            raise jit_source_entry_error(
                _source_location_label(self.source, resolved_path),
                self._module_spec.function_name,
                f"found {len(matches)} matching non-declaration func.func ops",
            )
        return matches[0]

    def _verify_entry_abi(self, entry, resolved_path: Path | None) -> None:
        source_label = _source_location_label(self.source, resolved_path)
        expected = tuple(str(type_obj) for type_obj in self._kernel_signature.compute_entry_arg_types())
        actual = tuple(str(type_obj) for type_obj in entry.type.inputs)
        results = tuple(str(type_obj) for type_obj in entry.type.results)

        if results:
            raise jit_source_abi_error(
                source_label,
                self._module_spec.function_name,
                f"source entry must return no values, got ({', '.join(results)})",
            )
        if len(actual) != len(expected):
            raise jit_source_abi_error(
                source_label,
                self._module_spec.function_name,
                "parameter count differs; "
                f"expected ({', '.join(expected)}), got ({', '.join(actual)})",
            )
        for index, (expected_type, actual_type) in enumerate(zip(expected, actual)):
            if expected_type != actual_type:
                raise jit_source_abi_error(
                    source_label,
                    self._module_spec.function_name,
                    f"parameter {index} differs; expected {expected_type}, got {actual_type}",
                )


def _symbol_name(op) -> str | None:
    attrs = op.attributes
    if "sym_name" not in attrs:
        return None
    return str(attrs["sym_name"]).strip('"')


def _walk_ops(root_op):
    for region in root_op.regions:
        for block in region.blocks:
            for op in block.operations:
                yield op
                yield from _walk_ops(op.operation)


def _source_location_label(source: str, resolved_path: Path | None):
    if resolved_path is not None:
        return resolved_path
    digest = hashlib.sha256(source.encode("utf-8")).hexdigest()[:12]
    return f"<inline pto source {digest}>"


__all__ = ["SourceModuleLoader"]
