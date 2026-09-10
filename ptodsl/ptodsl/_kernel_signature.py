# coding=utf-8
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Declarative PTODSL kernel-signature parsing and entry-ABI binding."""

from __future__ import annotations

import inspect
from dataclasses import dataclass

from ._diagnostics import (
    jit_constexpr_missing_default_error,
    jit_helper_illegal_formal_annotation_error,
    jit_helper_missing_annotation_error,
    jit_helper_standalone_type_inference_error,
    jit_illegal_formal_annotation_error,
    jit_struct_annotation_error,
    jit_keyword_only_non_constexpr_error,
    jit_legacy_tensor_spec_entry_error,
    jit_legacy_tensor_spec_helper_error,
    jit_missing_annotation_error,
    jit_non_gm_ptr_entry_error,
)
from ._host_tensors import TensorSpec
from ._surface_types import (
    PartitionTensorView,
    TensorView,
    Tile,
    const_expr as _const_expr_marker,
)
from ._surface_values import wrap_surface_value
from ._types import (
    _DType,
    _MaskDescriptor,
    _PtrDescriptor,
    _StructDescriptor,
    _VRegDescriptor,
    _resolve,
)


@dataclass(frozen=True)
class KernelSpecializationKey:
    kernel_identity: int
    abi_signature: tuple
    constexpr_signature: tuple[tuple[str, object], ...]


@dataclass(frozen=True)
class DeviceParameterSpec:
    name: str
    annotation: object

    def entry_arg_types(self):
        return (_resolve(self.annotation),)

    def bind_entry_arguments(self, entry_arguments):
        if not entry_arguments:
            raise RuntimeError(f"entry ABI for device parameter '{self.name}' is incomplete")
        return wrap_surface_value(entry_arguments[0]), entry_arguments[1:]

    def abi_signature(self):
        return ("device", self.name, _hashable_signature_atom(self.annotation))


@dataclass(frozen=True)
class RuntimeScalarParameterSpec:
    name: str
    annotation: object

    def entry_arg_types(self):
        return (_resolve(self.annotation),)

    def bind_entry_arguments(self, entry_arguments):
        if not entry_arguments:
            raise RuntimeError(f"entry ABI for runtime scalar parameter '{self.name}' is incomplete")
        return wrap_surface_value(entry_arguments[0]), entry_arguments[1:]

    def abi_signature(self):
        return ("scalar", self.name, _hashable_signature_atom(self.annotation))


@dataclass(frozen=True)
class HelperMarkerParameterSpec:
    name: str
    annotation: object

    def entry_arg_types(self):
        raise jit_helper_standalone_type_inference_error(self.name, self.annotation)

    def bind_entry_arguments(self, entry_arguments):
        if not entry_arguments:
            raise RuntimeError(f"kernel-module ABI for parameter '{self.name}' is incomplete")
        return wrap_surface_value(entry_arguments[0]), entry_arguments[1:]

    def abi_signature(self):
        return ("helper-marker", self.name, getattr(self.annotation, "__name__", repr(self.annotation)))


@dataclass(frozen=True)
class ConstexprParameterSpec:
    name: str
    default: object

    def bind_specialization(self, provided_bindings):
        value = provided_bindings.get(self.name, self.default)
        try:
            hash(value)
        except TypeError as exc:
            raise TypeError(
                f"@pto.jit constexpr parameter '{self.name}' must be hashable so it can "
                "participate in the specialization cache"
            ) from exc
        return value


def _hashable_signature_atom(value):
    try:
        hash(value)
    except TypeError:
        return repr(value)
    return value


def _is_supported_runtime_scalar_annotation(annotation) -> bool:
    return (
        isinstance(annotation, _DType)
        and not isinstance(annotation, (_PtrDescriptor, _StructDescriptor, _VRegDescriptor, _MaskDescriptor))
    )


def _is_explicit_gm_ptr_annotation(annotation) -> bool:
    return (
        isinstance(annotation, _PtrDescriptor)
        and str(getattr(annotation, "_space", "")).lower() == "gm"
    )


def _is_helper_marker_annotation(annotation) -> bool:
    return annotation in {Tile, TensorView, PartitionTensorView}


@dataclass(frozen=True)
class KernelSignature:
    positional_parameters: tuple
    constexpr_parameters: tuple[ConstexprParameterSpec, ...]

    def compute_entry_arg_types(self):
        arg_types = []
        for param in self.positional_parameters:
            arg_types.extend(param.entry_arg_types())
        return tuple(arg_types)

    def bind_entry_arguments(self, entry_arguments):
        remaining = tuple(entry_arguments)
        bound_args = []
        for param in self.positional_parameters:
            bound_value, remaining = param.bind_entry_arguments(remaining)
            bound_args.append(bound_value)
        if remaining:
            raise RuntimeError(f"unexpected trailing entry arguments in PTODSL kernel ABI: {len(remaining)}")
        return tuple(bound_args)

    def default_constexpr_bindings(self):
        return {param.name: param.default for param in self.constexpr_parameters}

    def bind_constexpr_bindings(self, provided_bindings):
        provided = dict(provided_bindings)
        expected_names = {param.name for param in self.constexpr_parameters}
        unknown = sorted(name for name in provided if name not in expected_names)
        if unknown:
            raise TypeError(
                f"unknown @pto.jit constexpr parameter(s): {', '.join(unknown)}"
            )

        bound = {}
        for param in self.constexpr_parameters:
            bound[param.name] = param.bind_specialization(provided)
        return bound

    def abi_signature(self):
        return tuple(param.abi_signature() for param in self.positional_parameters)

    def specialization_key(self, kernel_identity, constexpr_bindings):
        return KernelSpecializationKey(
            kernel_identity=kernel_identity,
            abi_signature=self.abi_signature(),
            constexpr_signature=tuple(
                (param.name, constexpr_bindings[param.name])
                for param in self.constexpr_parameters
            ),
        )


def _parse_entry_jit_kernel_signature(py_fn) -> KernelSignature:
    """Parse one authored launch-entry ``@pto.jit(entry=True)`` signature."""
    sig = inspect.signature(py_fn)
    positional_parameters = []
    constexpr_parameters = []

    for param in sig.parameters.values():
        if param.kind in {
            inspect.Parameter.POSITIONAL_ONLY,
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
        }:
            if param.annotation is inspect.Parameter.empty:
                raise jit_missing_annotation_error(param.name)
            if isinstance(param.annotation, TensorSpec):
                raise jit_legacy_tensor_spec_entry_error(param.name, param.annotation)
            if isinstance(param.annotation, _StructDescriptor):
                raise jit_struct_annotation_error(param.name)
            if isinstance(param.annotation, _PtrDescriptor):
                if not _is_explicit_gm_ptr_annotation(param.annotation):
                    raise jit_non_gm_ptr_entry_error(param.name, param.annotation)
                positional_parameters.append(
                    DeviceParameterSpec(param.name, param.annotation)
                )
            elif _is_supported_runtime_scalar_annotation(param.annotation):
                positional_parameters.append(
                    RuntimeScalarParameterSpec(param.name, param.annotation)
                )
            else:
                raise jit_illegal_formal_annotation_error(param.name, param.annotation)
            continue

        if param.kind is inspect.Parameter.KEYWORD_ONLY:
            if param.annotation is not _const_expr_marker:
                raise jit_keyword_only_non_constexpr_error(param.name, param.annotation)
            if param.default is inspect.Parameter.empty:
                raise jit_constexpr_missing_default_error(param.name)
            constexpr_parameters.append(ConstexprParameterSpec(param.name, param.default))
            continue

        raise TypeError(
            f"@pto.jit parameter '{param.name}' uses unsupported parameter kind "
            f"{param.kind!r}"
        )

    return KernelSignature(
        positional_parameters=tuple(positional_parameters),
        constexpr_parameters=tuple(constexpr_parameters),
    )


def _parse_helper_jit_kernel_signature(py_fn) -> KernelSignature:
    """Parse one authored kernel-module ``@pto.jit(entry=False)`` signature."""
    sig = inspect.signature(py_fn)
    positional_parameters = []

    for param in sig.parameters.values():
        if param.kind not in {
            inspect.Parameter.POSITIONAL_ONLY,
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
        }:
            if param.kind is inspect.Parameter.KEYWORD_ONLY and param.annotation is _const_expr_marker:
                raise TypeError(
                    f"@pto.jit(entry=False) keyword-only parameter '{param.name}' uses unsupported kernel-module "
                    "compile-time annotation pto.const_expr. Kernel-module ABI does not support "
                    "keyword-only constexpr specialization parameters."
                )
            raise TypeError(
                f"@pto.jit(entry=False) parameter '{param.name}' uses unsupported parameter kind "
                f"{param.kind!r}"
            )

        if param.annotation is inspect.Parameter.empty:
            raise jit_helper_missing_annotation_error(param.name)
        if isinstance(param.annotation, TensorSpec):
            raise jit_legacy_tensor_spec_helper_error(param.name, param.annotation)
        if isinstance(param.annotation, _StructDescriptor):
            raise jit_struct_annotation_error(param.name)
        if isinstance(param.annotation, _PtrDescriptor):
            positional_parameters.append(DeviceParameterSpec(param.name, param.annotation))
            continue
        if _is_supported_runtime_scalar_annotation(param.annotation):
            positional_parameters.append(RuntimeScalarParameterSpec(param.name, param.annotation))
            continue
        if _is_helper_marker_annotation(param.annotation):
            positional_parameters.append(HelperMarkerParameterSpec(param.name, param.annotation))
            continue
        raise jit_helper_illegal_formal_annotation_error(param.name, param.annotation)

    return KernelSignature(
        positional_parameters=tuple(positional_parameters),
        constexpr_parameters=(),
    )


def parse_jit_kernel_signature(py_fn, *, entry: bool = True) -> KernelSignature:
    """Parse one authored ``@pto.jit`` function signature."""
    if entry:
        return _parse_entry_jit_kernel_signature(py_fn)
    return _parse_helper_jit_kernel_signature(py_fn)


__all__ = [
    "ConstexprParameterSpec",
    "DeviceParameterSpec",
    "HelperMarkerParameterSpec",
    "KernelSpecializationKey",
    "KernelSignature",
    "RuntimeScalarParameterSpec",
    "parse_jit_kernel_signature",
]
