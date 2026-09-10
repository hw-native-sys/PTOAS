# coding=utf-8
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Kernel discovery and registry loading for kernel-test."""

from __future__ import annotations

import hashlib
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
import importlib
from importlib.machinery import ModuleSpec
import os
from pathlib import Path
import sys
import types

from .backends import BackendAdapter
from .results import CaseResult


DEFAULT_KERNEL_ROOT = Path(__file__).resolve().parents[1] / "kernels"


class RegistryError(RuntimeError):
    """Raised when registry discovery fails."""


@dataclass(frozen=True)
class OperatorSpec:
    """Framework contract for one registered kernel operator."""

    name: str
    default_backend: str
    backend_names: tuple[str, ...]
    create_backend: Callable[[str], BackendAdapter]
    list_cases: Callable[[str], Mapping[str, object]]
    verify: Callable[[str, object, object], CaseResult]
    cycle_fields: Callable[[str, object, BackendAdapter], Mapping[str, object]]
    summary: str = ""


def _empty_verify(case_id: str, case: object, output: object) -> CaseResult:
    del case_id, case, output
    raise NotImplementedError("operator spec does not provide a verifier")


def _empty_cycle_fields(
    case_id: str,
    case: object,
    backend: BackendAdapter,
) -> Mapping[str, object]:
    del case_id, case, backend
    return {}


def make_operator_spec(
    *,
    name: str,
    default_backend: str = "cce",
    backend_names: Iterable[str] = ("cce",),
    create_backend: Callable[[str], BackendAdapter],
    list_cases: Callable[[str], Mapping[str, object]],
    verify: Callable[[str, object, object], CaseResult] = _empty_verify,
    cycle_fields: Callable[[str, object, BackendAdapter], Mapping[str, object]] = _empty_cycle_fields,
    summary: str = "",
) -> OperatorSpec:
    """Build a normalized operator spec with immutable backend names."""

    return OperatorSpec(
        name=name,
        default_backend=default_backend,
        backend_names=tuple(backend_names),
        create_backend=create_backend,
        list_cases=list_cases,
        verify=verify,
        cycle_fields=cycle_fields,
        summary=summary,
    )


class KernelRegistry:
    """In-memory mapping of kernel name to discovered spec."""

    def __init__(self) -> None:
        self._entries: dict[str, OperatorSpec] = {}

    def register(self, spec: OperatorSpec) -> None:
        existing = self._entries.get(spec.name)
        if existing is not None:
            raise RegistryError(f"duplicate kernel registration: {spec.name}")
        self._entries[spec.name] = spec

    def get(self, name: str) -> OperatorSpec | None:
        return self._entries.get(name)

    def list_names(self) -> list[str]:
        return sorted(self._entries.keys())


def _resolve_kernel_dir(kernel_dir: str | os.PathLike[str] | None) -> Path:
    candidate = DEFAULT_KERNEL_ROOT if kernel_dir is None else Path(kernel_dir).expanduser()
    resolved = candidate.resolve()
    if not resolved.exists():
        raise RegistryError(f"kernel directory does not exist: {resolved}")
    if not resolved.is_dir():
        raise RegistryError(f"kernel directory is not a directory: {resolved}")
    return resolved


def _looks_like_kernel_package_dir(kernel_dir: Path) -> bool:
    if not (kernel_dir / "__init__.py").is_file():
        return False
    return any((kernel_dir / name).exists() for name in ("backends.py", "spec.py", "cycle_metrics.py"))


def _namespace_package_name(kernel_dir: Path) -> str:
    digest = hashlib.sha1(str(kernel_dir).encode("utf-8")).hexdigest()[:12]
    return f"_kernel_test_ext_{digest}"


def _ensure_namespace_package(module_name: str, search_path: Path) -> None:
    existing = sys.modules.get(module_name)
    if existing is not None:
        module_path = getattr(existing, "__path__", None)
        if module_path is None:
            raise RegistryError(f"module namespace conflict while loading kernels: {module_name}")
        if str(search_path) not in module_path:
            module_path.append(str(search_path))
        return

    module = types.ModuleType(module_name)
    spec = ModuleSpec(name=module_name, loader=None, is_package=True)
    spec.submodule_search_locations = [str(search_path)]
    module.__file__ = str(search_path)
    module.__package__ = module_name
    module.__path__ = list(spec.submodule_search_locations)
    module.__spec__ = spec
    sys.modules[module_name] = module


def _module_namespace_for_kernel_dir(kernel_dir: Path) -> tuple[str, str | None]:
    namespace = _namespace_package_name(kernel_dir)
    if _looks_like_kernel_package_dir(kernel_dir):
        _ensure_namespace_package(namespace, kernel_dir.parent)
        return namespace, kernel_dir.name

    _ensure_namespace_package(namespace, kernel_dir)
    return namespace, None


def _iter_kernel_module_names(kernel_dir: Path) -> tuple[str, ...]:
    _, single_kernel_name = _module_namespace_for_kernel_dir(kernel_dir)
    if single_kernel_name is not None:
        return (single_kernel_name,)

    names: list[str] = []
    for child in sorted(kernel_dir.iterdir()):
        if child.name.startswith("_") or not child.is_dir():
            continue
        if _looks_like_kernel_package_dir(child):
            names.append(child.name)
    return tuple(names)


def import_kernel_module(
    kernel_name: str,
    *,
    kernel_dir: str | os.PathLike[str] | None = None,
    submodule: str | None = None,
):
    """Import one kernel package or submodule from the requested kernel directory."""

    resolved_dir = _resolve_kernel_dir(kernel_dir)
    namespace, single_kernel_name = _module_namespace_for_kernel_dir(resolved_dir)
    if single_kernel_name is not None and kernel_name != single_kernel_name:
        raise ModuleNotFoundError(
            f"kernel directory {resolved_dir} only exposes package {single_kernel_name!r}, "
            f"not {kernel_name!r}"
        )

    qualified_name = f"{namespace}.{single_kernel_name or kernel_name}"
    if submodule:
        qualified_name = f"{qualified_name}.{submodule}"
    return importlib.import_module(qualified_name)


def _load_from_module(module_name: str, registry: KernelRegistry) -> None:
    module = importlib.import_module(module_name)

    if hasattr(module, "register"):
        module.register(registry)
        return

    if hasattr(module, "get_operator_spec"):
        registry.register(module.get_operator_spec())
        return

    if hasattr(module, "OPERATOR_SPEC"):
        registry.register(module.OPERATOR_SPEC)
        return

    if hasattr(module, "get_kernel_spec"):
        registry.register(module.get_kernel_spec())
        return

    if hasattr(module, "KERNEL_SPEC"):
        registry.register(module.KERNEL_SPEC)
        return

    raise RegistryError(
        f"kernel module {module_name!r} must expose register(), get_kernel_spec(), or KERNEL_SPEC"
    )

def load_registry(kernel_dir: str | os.PathLike[str] | None = None) -> KernelRegistry:
    """Load all kernel operators from the default or requested kernel directory."""

    resolved_dir = _resolve_kernel_dir(kernel_dir)
    registry = KernelRegistry()
    namespace, _ = _module_namespace_for_kernel_dir(resolved_dir)

    for kernel_name in _iter_kernel_module_names(resolved_dir):
        _load_from_module(f"{namespace}.{kernel_name}", registry)

    return registry
