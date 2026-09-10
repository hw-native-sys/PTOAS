# coding=utf-8
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Host-tensor boundary helpers for ``@pto.jit``."""

from __future__ import annotations

import inspect
from dataclasses import dataclass

from ._diagnostics import host_tensor_metadata_error
from ._types import _ensure_non_storage_only_authored_dtype, _resolve, index, ptr


def _normalize_tensor_shape(shape):
    try:
        return tuple(int(dim) for dim in shape)
    except TypeError as exc:
        raise host_tensor_metadata_error("missing iterable .shape") from exc
    except ValueError as exc:
        raise host_tensor_metadata_error(".shape must contain integer-like dimensions") from exc


def _normalize_tensor_strides(tensor):
    stride_method = getattr(tensor, "stride", None)
    if callable(stride_method):
        try:
            return tuple(int(dim) for dim in stride_method())
        except TypeError as exc:
            raise host_tensor_metadata_error(".stride() must return an iterable of integer-like dimensions") from exc
        except ValueError as exc:
            raise host_tensor_metadata_error(".stride() must return integer-like dimensions") from exc
    strides = getattr(tensor, "strides", None)
    if strides is None:
        raise host_tensor_metadata_error("missing .strides or .stride()")
    try:
        return tuple(int(dim) for dim in strides)
    except TypeError as exc:
        raise host_tensor_metadata_error(".strides must be iterable") from exc
    except ValueError as exc:
        raise host_tensor_metadata_error(".strides must contain integer-like dimensions") from exc


def _extract_tensor_data_handle(tensor):
    for attr_name in ("data_ptr", "ptr"):
        attr = getattr(tensor, attr_name, None)
        if callable(attr):
            value = attr()
        else:
            value = attr
        if value is not None:
            try:
                return int(value)
            except (TypeError, ValueError) as exc:
                raise host_tensor_metadata_error(
                    f"{attr_name} must return an integer-like data handle"
                ) from exc
    array_interface = getattr(tensor, "__array_interface__", None)
    if array_interface is not None:
        data = array_interface.get("data")
        if isinstance(data, tuple) and data:
            try:
                return int(data[0])
            except (TypeError, ValueError) as exc:
                raise host_tensor_metadata_error(
                    "__array_interface__['data'][0] must be an integer-like data handle"
                ) from exc
    raise host_tensor_metadata_error(
        "missing data handle; expected .data_ptr(), .ptr, or __array_interface__"
    )


@dataclass(frozen=True)
class HostTensorMetadata:
    """Concrete runtime metadata extracted from a Python host tensor."""

    shape: tuple[int, ...]
    strides: tuple[int, ...]
    dtype: object
    data_handle: int


def inspect_host_tensor_metadata(tensor) -> HostTensorMetadata:
    """Extract shape / strides / dtype / data-handle from a Python tensor-like object."""
    shape = _normalize_tensor_shape(getattr(tensor, "shape", None))
    strides = _normalize_tensor_strides(tensor)
    dtype = getattr(tensor, "dtype", None)
    if dtype is None:
        raise host_tensor_metadata_error("missing .dtype")
    return HostTensorMetadata(
        shape=shape,
        strides=strides,
        dtype=dtype,
        data_handle=_extract_tensor_data_handle(tensor),
    )


@dataclass(frozen=True)
class TensorSpec:
    """Static ABI hint for one Python-native ``@pto.jit`` tensor parameter."""

    rank: int
    dtype: object
    address_space: str = "gm"

    def __post_init__(self):
        if self.rank <= 0:
            raise ValueError("tensor_spec(rank=...) expects a positive rank")
        _ensure_non_storage_only_authored_dtype(
            self.dtype,
            context="pto.tensor_spec(...)",
        )

    def entry_arg_types(self):
        data_type = _resolve(ptr(self.dtype, self.address_space))
        index_type = _resolve(index)
        return (
            data_type,
            *([index_type] * self.rank),
            *([index_type] * self.rank),
        )

    def abi_signature(self):
        return (
            "tensor_spec",
            self.rank,
            self.dtype,
            self.address_space,
        )

    def __repr__(self):
        return (
            f"pto.tensor_spec(rank={self.rank}, dtype={self.dtype!r}, "
            f"address_space={self.address_space!r})"
        )


def tensor_spec(*, rank: int, dtype, address_space: str = "gm") -> TensorSpec:
    """Declare the ABI contract of one Python-native ``@pto.jit`` tensor parameter."""
    return TensorSpec(rank=rank, dtype=dtype, address_space=address_space)


class HostTensorValue:
    """Tracing-time proxy for one Python-native tensor at the ``@pto.jit`` boundary."""

    def __init__(self, name: str, spec: TensorSpec, data_handle, shape, strides):
        from ._surface_values import wrap_surface_value
        self.name = name
        self.spec = spec
        self.data_handle = wrap_surface_value(data_handle)
        self.shape = tuple(wrap_surface_value(dim) for dim in shape)
        self.strides = tuple(wrap_surface_value(dim) for dim in strides)
        self.dtype = spec.dtype

    @property
    def rank(self):
        return self.spec.rank

    def __repr__(self):
        return (
            f"<ptodsl.tensor {self.name}: rank={self.rank}, "
            f"dtype={self.dtype!r}, address_space={self.spec.address_space!r}>"
        )


def bind_host_tensor_argument(name: str, spec: TensorSpec, entry_arguments):
    """Bind one flattened entry-ABI slice into a ``HostTensorValue``."""
    expected = 1 + spec.rank + spec.rank
    if len(entry_arguments) < expected:
        raise RuntimeError(
            f"entry ABI for host tensor '{name}' is incomplete: expected {expected} "
            f"arguments, got {len(entry_arguments)}"
        )
    data_handle = entry_arguments[0]
    shape = entry_arguments[1:1 + spec.rank]
    strides = entry_arguments[1 + spec.rank:1 + spec.rank + spec.rank]
    return (
        HostTensorValue(name, spec, data_handle, shape, strides),
        entry_arguments[expected:],
    )


def infer_jit_host_tensor_spec(param: inspect.Parameter):
    """
    Resolve one ``@pto.jit`` positional parameter to a host-tensor contract.

    V1 cannot infer rank/dtype from an unannotated formal parameter while still
    tracing at compile time, so host tensors currently require an explicit
    ``pto.tensor_spec(...)`` ABI hint.
    """
    if isinstance(param.annotation, TensorSpec):
        return param.annotation
    if param.annotation is inspect.Parameter.empty:
        raise TypeError(
            f"@pto.jit positional parameter '{param.name}' uses the host-tensor "
            "boundary but does not declare an ABI hint. Add an annotation such "
            "as `Q: pto.tensor_spec(rank=4, dtype=pto.f32)`."
        )
    return None


def resolve_tensor_data_entry(value):
    """Return the pointer-like data entry behind a host tensor proxy or raw value."""
    if isinstance(value, HostTensorValue):
        return value.data_handle
    return value


def looks_like_host_tensor(value) -> bool:
    """Best-effort predicate for Python-native tensor-like objects at the JIT boundary."""
    if isinstance(value, HostTensorValue):
        return True
    return (
        getattr(value, "shape", None) is not None
        and getattr(value, "dtype", None) is not None
        and (
            callable(getattr(value, "stride", None))
            or getattr(value, "strides", None) is not None
        )
        and (
            callable(getattr(value, "data_ptr", None))
            or getattr(value, "ptr", None) is not None
            or getattr(value, "__array_interface__", None) is not None
        )
    )


__all__ = [
    "HostTensorMetadata",
    "TensorSpec",
    "HostTensorValue",
    "bind_host_tensor_argument",
    "tensor_spec",
    "infer_jit_host_tensor_spec",
    "inspect_host_tensor_metadata",
    "looks_like_host_tensor",
    "resolve_tensor_data_entry",
]
