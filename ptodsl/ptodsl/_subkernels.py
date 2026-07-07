# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""Layered PTODSL subkernel decorators."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from functools import update_wrapper
import inspect

from ._diagnostics import (
    illegal_inline_subkernel_placement_error,
    illegal_subkernel_placement_error,
    legacy_subkernel_surface_error,
    simd_value_escape_error,
    subkernel_argument_type_error,
    subkernel_host_tensor_boundary_error,
    subkernel_illegal_annotation_error,
    subkernel_illegal_parameter_kind_error,
    subkernel_missing_annotation_error,
    subkernel_return_boundary_error,
    subkernel_signature_boundary_error,
)
from ._ast_rewrite import rewrite_jit_function
from ._host_tensors import TensorSpec, looks_like_host_tensor
from ._surface_types import PartitionTensorView, TensorView, Tile
from ._surface_values import unwrap_surface_value
from ._tracing import current_runtime, current_session
from ._types import (
    _DType,
    _MaskDescriptor,
    _PtrDescriptor,
    _VRegDescriptor,
    bf16,
    f4e1m2x2,
    f4e2m1x2,
    f8e4m3,
    f8e5m2,
    float16,
    float32,
    hif8,
    index,
    int1,
    int8,
    int16,
    int32,
    int64,
    ptr,
    si8,
    si16,
    si32,
    si64,
    ui8,
    ui16,
    ui32,
    ui64,
)
from mlir.ir import FloatType, IndexType, IntegerType


class KernelRole(str, Enum):
    TILEOP = "tileop"
    CUBE = "cube"
    SIMD = "simd"
    SIMT = "simt"


@dataclass(frozen=True)
class SubkernelSpec:
    """Declarative metadata for a PTODSL subkernel surface."""

    role: KernelRole
    symbol_name: str
    target: str = "a5"
    simt_max_threads: int | None = None
    simt_max_regs: int | None = None


class SubkernelTemplate:
    """Callable decorated PTODSL subkernel surface."""

    def __init__(self, spec: SubkernelSpec, py_fn, *, ast_rewrite: bool = True):
        self.spec = spec
        self.py_fn = py_fn
        self._ast_rewrite = ast_rewrite
        self.signature = inspect.signature(py_fn)
        self._validate_definition()
        update_wrapper(self, py_fn)

    def emit_body(self, *args, **kwargs):
        """Emit this subkernel body into the currently active trace."""
        py_fn = rewrite_jit_function(self.py_fn) if self._ast_rewrite else self.py_fn
        result = py_fn(*args, **kwargs)
        self._validate_result(result)
        return result

    def trace_body(self, *args, **kwargs):
        """Backward-compatible alias for body emission."""
        return self.emit_body(*args, **kwargs)

    def __call__(self, *args, **kwargs):
        runtime = current_runtime()
        if runtime is None:
            raise RuntimeError(
                f"@pto.{self.spec.role.value} kernels may only be called while tracing "
                "a compatible PTODSL kernel"
            )
        if self.spec.role == KernelRole.SIMT:
            self._validate_invocation(*args, **kwargs)
        else:
            args, kwargs = self._normalize_invocation(*args, **kwargs)
        return runtime.dispatch_subkernel_call(self, *args, **kwargs)

    def __getitem__(self, dims):
        if self.spec.role != KernelRole.SIMT:
            raise TypeError(
                f"@pto.{self.spec.role.value} kernels do not support launch dimensions; "
                "only @pto.simt helpers support helper[dim_x, dim_y, dim_z](...)"
            )
        if not isinstance(dims, tuple):
            dims = (dims,)
        if len(dims) != 3:
            raise TypeError("@pto.simt launch syntax expects helper[dim_x, dim_y, dim_z](...)")
        return _SimtLaunchTemplate(self, dims)

    def _validate_definition(self) -> None:
        if self.spec.role == KernelRole.SIMT:
            for param in self.signature.parameters.values():
                if isinstance(param.annotation, TensorSpec):
                    raise subkernel_signature_boundary_error(self.spec.role.value, param.name)
            return

        for param in self.signature.parameters.values():
            if param.kind == inspect.Parameter.KEYWORD_ONLY:
                if self.spec.role == KernelRole.SIMT:
                    continue
                raise subkernel_illegal_parameter_kind_error(self.spec.role.value, param.name, param.kind)
            if param.kind not in {
                inspect.Parameter.POSITIONAL_ONLY,
                inspect.Parameter.POSITIONAL_OR_KEYWORD,
            }:
                raise subkernel_illegal_parameter_kind_error(self.spec.role.value, param.name, param.kind)
            if param.annotation is inspect.Parameter.empty:
                raise subkernel_missing_annotation_error(self.spec.role.value, param.name)
            annotation = _normalize_subkernel_annotation(param.annotation)
            if isinstance(annotation, TensorSpec):
                raise subkernel_signature_boundary_error(self.spec.role.value, param.name)
            if not _is_supported_subkernel_annotation(self.spec.role, annotation):
                raise subkernel_illegal_annotation_error(
                    self.spec.role.value,
                    param.name,
                    param.annotation,
                    _expected_subkernel_annotation_summary(self.spec.role),
                )

    def _normalize_invocation(self, *args, **kwargs):
        session = current_session()
        outer = session.current_subkernel if session is not None else None
        _validate_subkernel_placement(self.spec.role, outer)

        bound = self.signature.bind(*args, **kwargs)
        bound.apply_defaults()
        for name, value in bound.arguments.items():
            param = self.signature.parameters[name]
            if param.kind == inspect.Parameter.KEYWORD_ONLY:
                continue
            if looks_like_host_tensor(value):
                raise subkernel_host_tensor_boundary_error(self.spec.role.value, name)
            annotation = _normalize_subkernel_annotation(param.annotation)
            bound.arguments[name] = _normalize_subkernel_argument(
                self.spec.role,
                name,
                annotation,
                value,
            )
        return tuple(bound.args), dict(bound.kwargs)

    def _validate_invocation(self, *args, **kwargs) -> None:
        session = current_session()
        outer = session.current_subkernel if session is not None else None
        _validate_subkernel_placement(self.spec.role, outer)

        bound = self.signature.bind_partial(*args, **kwargs)
        for name, value in bound.arguments.items():
            if looks_like_host_tensor(value):
                raise subkernel_host_tensor_boundary_error(self.spec.role.value, name)

    def _validate_result(self, result) -> None:
        if self.spec.role in {KernelRole.TILEOP, KernelRole.SIMD}:
            escaped_type = _find_transient_simd_escape(result)
            if escaped_type is not None:
                raise simd_value_escape_error(escaped_type, surface=f"@pto.{self.spec.role.value}")
            _validate_subkernel_scalar_result(self.spec.role.value, result)
            return
        if self.spec.role == KernelRole.CUBE:
            _validate_subkernel_scalar_result(self.spec.role.value, result)
            return


class _SimtLaunchTemplate:
    """Callable ``helper[x, y, z]`` launch descriptor for a decorated SIMT helper."""

    def __init__(self, body: SubkernelTemplate, dims):
        self._body = body
        self._dims = dims

    def __call__(self, *args, **kwargs):
        from ._ops import simt_launch

        return simt_launch(self._body, *args, dims=self._dims, **kwargs)


def _find_transient_simd_escape(value):
    if value is None:
        return None
    if isinstance(value, (tuple, list)):
        for item in value:
            escaped = _find_transient_simd_escape(item)
            if escaped is not None:
                return escaped
        return None
    if isinstance(value, dict):
        for item in value.values():
            escaped = _find_transient_simd_escape(item)
            if escaped is not None:
                return escaped
        return None
    raw_value = unwrap_surface_value(value)
    type_obj = getattr(raw_value, "type", None)
    if type_obj is None:
        return None
    type_text = str(type_obj)
    if type_text.startswith("!pto.vreg<") or type_text.startswith("!pto.mask<"):
        return type_text
    return None


def _is_scalar_result_type(type_obj) -> bool:
    return (
        IndexType.isinstance(type_obj)
        or IntegerType.isinstance(type_obj)
        or FloatType.isinstance(type_obj)
    )


def _validate_subkernel_scalar_result(role: str, value) -> None:
    if value is None:
        return
    if isinstance(value, tuple):
        for item in value:
            _validate_subkernel_scalar_result(role, item)
        return
    if isinstance(value, list):
        for item in value:
            _validate_subkernel_scalar_result(role, item)
        return
    if isinstance(value, dict):
        for item in value.values():
            _validate_subkernel_scalar_result(role, item)
        return
    raw_value = unwrap_surface_value(value)
    type_obj = getattr(raw_value, "type", None)
    if type_obj is None:
        raise subkernel_return_boundary_error(role, type(value).__name__)
    if _is_scalar_result_type(type_obj):
        return
    raise subkernel_return_boundary_error(role, str(type_obj))


def _is_supported_runtime_scalar_annotation(annotation) -> bool:
    return (
        isinstance(annotation, _DType)
        and not isinstance(annotation, (_PtrDescriptor, _VRegDescriptor, _MaskDescriptor))
    )


_POSTPONED_DTYPE_ANNOTATIONS = {
    "pto.f32": float32,
    "pto.float32": float32,
    "pto.f16": float16,
    "pto.float16": float16,
    "pto.bf16": bf16,
    "pto.f8e4m3": f8e4m3,
    "pto.f8e5m2": f8e5m2,
    "pto.hif8": hif8,
    "pto.f4e1m2x2": f4e1m2x2,
    "pto.f4e2m1x2": f4e2m1x2,
    "pto.i1": int1,
    "pto.int1": int1,
    "pto.i8": int8,
    "pto.int8": int8,
    "pto.i16": int16,
    "pto.int16": int16,
    "pto.i32": int32,
    "pto.int32": int32,
    "pto.i64": int64,
    "pto.int64": int64,
    "pto.si8": si8,
    "pto.si16": si16,
    "pto.si32": si32,
    "pto.si64": si64,
    "pto.ui8": ui8,
    "pto.ui16": ui16,
    "pto.ui32": ui32,
    "pto.ui64": ui64,
    "pto.index": index,
}


_POSTPONED_MEMORY_SPACES = {
    "pto.MemorySpace.GM": "gm",
    "pto.MemorySpace.UB": "ub",
    "pto.MemorySpace.VEC": "vec",
    "pto.MemorySpace.MAT": "mat",
    "pto.MemorySpace.LEFT": "left",
    "pto.MemorySpace.RIGHT": "right",
    "pto.MemorySpace.ACC": "acc",
    "pto.MemorySpace.BIAS": "bias",
    "pto.MemorySpace.SCALING": "scaling",
}


def _normalize_subkernel_annotation(annotation):
    if not isinstance(annotation, str):
        return annotation
    text = annotation.strip()
    if text in {"Tile", "pto.Tile"}:
        return Tile
    if text in {"TensorView", "pto.TensorView"}:
        return TensorView
    if text in {"PartitionTensorView", "pto.PartitionTensorView"}:
        return PartitionTensorView
    if text in _POSTPONED_DTYPE_ANNOTATIONS:
        return _POSTPONED_DTYPE_ANNOTATIONS[text]
    if text.startswith("pto.ptr(") and text.endswith(")"):
        args = [part.strip() for part in text[len("pto.ptr("):-1].split(",")]
        if len(args) == 2 and args[0] in _POSTPONED_DTYPE_ANNOTATIONS and args[1] in _POSTPONED_MEMORY_SPACES:
            return ptr(_POSTPONED_DTYPE_ANNOTATIONS[args[0]], _POSTPONED_MEMORY_SPACES[args[1]])
    return annotation


def _allows_view_annotations(role: KernelRole) -> bool:
    return role in {KernelRole.TILEOP, KernelRole.SIMD}


def _is_supported_subkernel_annotation(role: KernelRole, annotation) -> bool:
    if annotation is Tile:
        return True
    if role == KernelRole.CUBE:
        return False
    if _allows_view_annotations(role) and annotation in {TensorView, PartitionTensorView}:
        return True
    if _is_supported_runtime_scalar_annotation(annotation):
        return True
    if role == KernelRole.SIMT and isinstance(annotation, _PtrDescriptor):
        return True
    return False


def _expected_subkernel_annotation_summary(role: KernelRole) -> str:
    if role == KernelRole.CUBE:
        return "pto.Tile parameters only"
    if role in {KernelRole.TILEOP, KernelRole.SIMD}:
        return (
            "pto.Tile / pto.TensorView / pto.PartitionTensorView parameters plus PTO scalar "
            "annotations such as pto.i32/pto.f32"
        )
    return "pto.Tile parameters, typed pto.ptr(...) values, and PTO scalar annotations"


def _is_pto_ptr_like(value) -> bool:
    raw_value = unwrap_surface_value(value)
    type_obj = getattr(raw_value, "type", None)
    if type_obj is None:
        return False
    type_text = str(type_obj)
    return type_text.startswith("!pto.ptr<") or type_text.startswith("memref<")


def _is_runtime_scalar_value(value) -> bool:
    raw_value = unwrap_surface_value(value)
    type_obj = getattr(raw_value, "type", None)
    if type_obj is None:
        return False
    type_text = str(type_obj)
    return not (
        type_text.startswith("!pto.tile_buf<")
        or type_text.startswith("!pto.ptr<")
        or type_text.startswith("memref<")
        or type_text.startswith("!pto.tensor_view<")
        or type_text.startswith("!pto.partition_tensor_view<")
        or type_text.startswith("!pto.vreg<")
        or type_text.startswith("!pto.mask<")
    )


def _is_tensor_view_value(value) -> bool:
    raw_value = unwrap_surface_value(value)
    type_obj = getattr(raw_value, "type", None)
    if type_obj is None:
        return False
    return str(type_obj).startswith("!pto.tensor_view<")


def _is_partition_tensor_view_value(value) -> bool:
    raw_value = unwrap_surface_value(value)
    type_obj = getattr(raw_value, "type", None)
    if type_obj is None:
        return False
    return str(type_obj).startswith("!pto.partition_tensor_view<")


def _normalize_subkernel_argument(role: KernelRole, name: str, annotation, value):
    if annotation is Tile:
        if isinstance(value, Tile):
            return value
        raise subkernel_argument_type_error(role.value, name, "a pto.Tile value", type(value).__name__)

    if annotation is TensorView:
        if _allows_view_annotations(role) and isinstance(value, TensorView) and _is_tensor_view_value(value):
            return value
        raise subkernel_argument_type_error(role.value, name, "a pto.TensorView value", type(value).__name__)

    if annotation is PartitionTensorView:
        if (
            _allows_view_annotations(role)
            and isinstance(value, PartitionTensorView)
            and _is_partition_tensor_view_value(value)
        ):
            return value
        raise subkernel_argument_type_error(
            role.value,
            name,
            "a pto.PartitionTensorView value",
            type(value).__name__,
        )

    if _is_supported_runtime_scalar_annotation(annotation):
        if isinstance(value, (bool, int, float)):
            from ._ops import const

            return const(value, dtype=annotation)
        if _is_runtime_scalar_value(value):
            return value
        raise subkernel_argument_type_error(
            role.value,
            name,
            f"one PTO scalar value compatible with {annotation!r}",
            type(value).__name__,
        )

    if role == KernelRole.SIMT and isinstance(annotation, _PtrDescriptor):
        if _is_pto_ptr_like(value):
            return value
        raise subkernel_argument_type_error(
            role.value,
            name,
            f"one typed pointer compatible with {annotation!r}",
            type(value).__name__,
        )

    raise subkernel_illegal_annotation_error(
        role.value,
        name,
        annotation,
        _expected_subkernel_annotation_summary(role),
    )


def _validate_subkernel_placement(role: KernelRole, outer_frame, *, inline: bool = False) -> None:
    if outer_frame is None:
        return
    if inline:
        raise illegal_inline_subkernel_placement_error(role.value, outer_frame.role)
    raise illegal_subkernel_placement_error(role.value, outer_frame.role)


class _SubkernelSurface:
    """Dual-use surface that supports both decorators and inline context-manager scopes."""

    def __init__(
        self,
        role: KernelRole,
        *,
        name: str | None = None,
        target: str = "a5",
        ast_rewrite: bool = True,
        simt_max_threads: int | None = None,
        simt_max_regs: int | None = None,
    ):
        self._role = role
        self._name = name
        self._target = target
        self._ast_rewrite = ast_rewrite
        self._simt_max_threads = simt_max_threads
        self._simt_max_regs = simt_max_regs
        self._session_cm = None

    def __call__(self, fn):
        if self._role in {KernelRole.CUBE, KernelRole.SIMD}:
            raise legacy_subkernel_surface_error(f"@pto.{self._role.value}")
        return SubkernelTemplate(
            SubkernelSpec(
                role=self._role,
                symbol_name=self._name or fn.__name__,
                target=self._target,
                simt_max_threads=self._simt_max_threads,
                simt_max_regs=self._simt_max_regs,
            ),
            fn,
            ast_rewrite=self._ast_rewrite,
        )

    def __enter__(self):
        if self._role in {KernelRole.CUBE, KernelRole.SIMD}:
            raise legacy_subkernel_surface_error(f"with pto.{self._role.value}()")
        if self._role == KernelRole.SIMT and (
            self._simt_max_threads is not None or self._simt_max_regs is not None
        ):
            raise TypeError("@pto.simt(max_threads=..., max_regs=...) is only supported as a function decorator")
        runtime = current_runtime()
        if runtime is None:
            raise RuntimeError(
                f"inline pto.{self._role.value}() may only be used while tracing "
                "a compatible PTODSL kernel"
            )
        session = current_session()
        outer = session.current_subkernel if session is not None else None
        _validate_subkernel_placement(self._role, outer, inline=True)
        symbol_name = self._name or f"inline_{self._role.value}"
        self._session_cm = session.enter_inline_subkernel(
            self._role.value,
            symbol_name,
            self._target,
        )
        self._session_cm.__enter__()
        return None

    def __exit__(self, *exc):
        try:
            return self._session_cm.__exit__(*exc)
        finally:
            self._session_cm = None


def _subkernel_decorator(
    role: KernelRole,
    *,
    name: str | None = None,
    target: str = "a5",
    ast_rewrite: bool = True,
    simt_max_threads: int | None = None,
    simt_max_regs: int | None = None,
):
    return _SubkernelSurface(
        role,
        name=name,
        target=target,
        ast_rewrite=ast_rewrite,
        simt_max_threads=simt_max_threads,
        simt_max_regs=simt_max_regs,
    )


def _decorate_subkernel(
    role: KernelRole,
    fn=None,
    *,
    name: str | None = None,
    target: str = "a5",
    ast_rewrite: bool = True,
    simt_max_threads: int | None = None,
    simt_max_regs: int | None = None,
):
    if fn is not None:
        return _subkernel_decorator(
            role,
            name=name,
            target=target,
            ast_rewrite=ast_rewrite,
            simt_max_threads=simt_max_threads,
            simt_max_regs=simt_max_regs,
        )(fn)
    return _subkernel_decorator(
        role,
        name=name,
        target=target,
        ast_rewrite=ast_rewrite,
        simt_max_threads=simt_max_threads,
        simt_max_regs=simt_max_regs,
    )


def tileop(fn=None, *, name: str | None = None, target: str = "a5", ast_rewrite: bool = True):
    return _decorate_subkernel(KernelRole.TILEOP, fn, name=name, target=target, ast_rewrite=ast_rewrite)


def cube(fn=None, *, name: str | None = None, target: str = "a5", ast_rewrite: bool = True):
    return _decorate_subkernel(KernelRole.CUBE, fn, name=name, target=target, ast_rewrite=ast_rewrite)


def simd(fn=None, *, name: str | None = None, target: str = "a5", ast_rewrite: bool = True):
    return _decorate_subkernel(KernelRole.SIMD, fn, name=name, target=target, ast_rewrite=ast_rewrite)


def _validate_simt_resource_attr(name: str, value: int | None) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"@pto.simt(..., {name}=...) expects a positive Python int")
    if value <= 0:
        raise ValueError(f"@pto.simt(..., {name}=...) expects a positive Python int")
    if value > 2**31 - 1:
        raise ValueError(f"@pto.simt(..., {name}=...) must fit in signless i32")
    return value


def simt(
    fn=None,
    *,
    name: str | None = None,
    target: str = "a5",
    ast_rewrite: bool = True,
    max_threads: int | None = None,
    max_regs: int | None = None,
):
    max_threads = _validate_simt_resource_attr("max_threads", max_threads)
    max_regs = _validate_simt_resource_attr("max_regs", max_regs)
    return _decorate_subkernel(
        KernelRole.SIMT,
        fn,
        name=name,
        target=target,
        ast_rewrite=ast_rewrite,
        simt_max_threads=max_threads,
        simt_max_regs=max_regs,
    )


__all__ = [
    "KernelRole",
    "SubkernelSpec",
    "SubkernelTemplate",
    "tileop",
    "cube",
    "simd",
    "simt",
]
