# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""Launch handles and ctypes dispatch for compiled PTODSL kernels."""

from __future__ import annotations

import ctypes
from typing import TYPE_CHECKING

from .._kernel_signature import DeviceParameterSpec, RuntimeScalarParameterSpec
from .._types import _resolve
from .native_build import build_native_library

from ptoas.mlir.ir import BF16Type, Context, F16Type, F32Type, IndexType, IntegerType

if TYPE_CHECKING:
    from .._kernel_compilation import CompiledKernelHandle

def _normalize_stream_ptr(stream):
    if stream is None:
        try:
            import torch
        except ImportError as exc:
            raise ImportError(
                "stream=None requires torch; install torch and torch_npu for default-stream launch"
            ) from exc
        return torch.npu.current_stream()._as_parameter_  # noqa: SLF001

    if isinstance(stream, ctypes.c_void_p):
        return stream
    if isinstance(stream, int):
        return ctypes.c_void_p(stream)
    if hasattr(stream, "value"):
        return ctypes.c_void_p(int(stream.value))
    return stream


def _as_void_ptr(value):
    if isinstance(value, ctypes.c_void_p):
        return value
    if hasattr(value, "data_ptr"):
        return ctypes.c_void_p(value.data_ptr())
    if isinstance(value, int):
        return ctypes.c_void_p(value)
    raise TypeError(f"expected a pointer-like launch argument, got {type(value)!r}")


def _ctype_for_runtime_scalar(annotation):
    try:
        type_obj = _resolve(annotation)
    except RuntimeError as exc:
        if "requires a Context" not in str(exc):
            raise
        with Context():
            type_obj = _resolve(annotation)
    if IndexType.isinstance(type_obj):
        return ctypes.c_int64
    if IntegerType.isinstance(type_obj):
        width = IntegerType(type_obj).width
        if width == 1:
            return ctypes.c_bool
        if width == 8:
            return ctypes.c_int8
        if width == 16:
            return ctypes.c_int16
        if width == 32:
            return ctypes.c_int32
        if width == 64:
            return ctypes.c_int64
    if F32Type.isinstance(type_obj):
        return ctypes.c_float
    if F16Type.isinstance(type_obj) or BF16Type.isinstance(type_obj):
        raise TypeError(
            f"runtime launch does not yet support host scalar marshaling for {type_obj}; "
            "use pto.f32 / integer scalar parameters or tensorize this value for now"
        )
    raise TypeError(f"unsupported @pto.jit runtime scalar launch type {type_obj}")


def _marshal_runtime_scalar(annotation, value):
    ctype = _ctype_for_runtime_scalar(annotation)
    if ctype is ctypes.c_bool:
        return ctype(bool(value))
    return ctype(value)


def _marshal_launch_args(kernel_signature, args):
    if len(args) != len(kernel_signature.positional_parameters):
        raise TypeError(
            f"expected {len(kernel_signature.positional_parameters)} launch argument(s), "
            f"got {len(args)}"
        )

    marshaled = []
    for param, value in zip(kernel_signature.positional_parameters, args):
        if isinstance(param, DeviceParameterSpec):
            marshaled.append(_as_void_ptr(value))
            continue
        if isinstance(param, RuntimeScalarParameterSpec):
            marshaled.append(_marshal_runtime_scalar(param.annotation, value))
            continue
        raise TypeError(f"unsupported launch parameter spec: {param!r}")
    return marshaled


def build_and_load_native_library(compiled: CompiledKernelHandle):
    """Build this specialization if needed and return ``(library, path, symbol)``.

    The loaded library is cached on the compiled handle. Several launch handles
    over one specialization share it, and so does a caller reaching past the
    launch entry for a symbol that ``native_options['host_sources']`` contributed.
    """
    cached = getattr(compiled, "_native_library_cache", None)
    if cached is not None:
        return cached

    lib_path, launch_symbol = build_native_library(
        py_name=compiled._py_name,
        module_spec=compiled._module_spec,
        kernel_signature=compiled._kernel_signature,
        mlir_text=compiled.mlir_text(),
        specialization_key=compiled.specialization_key,
    )
    loaded = (ctypes.CDLL(str(lib_path)), lib_path, launch_symbol)
    compiled._native_library_cache = loaded
    return loaded


class LaunchHandle:
    """Callable launch binding returned by ``compiled[grid, stream]``."""

    def __init__(self, compiled: CompiledKernelHandle, grid: int, stream):
        if not isinstance(grid, int) or grid <= 0:
            raise ValueError("launch grid must be a positive integer")
        self._compiled = compiled
        self._grid = grid
        self._stream = stream
        self._launch_fn = None
        self._launch_symbol = None

    def _ensure_launch_fn(self):
        if self._launch_fn is not None:
            return

        lib, _lib_path, launch_symbol = build_and_load_native_library(self._compiled)
        fn = getattr(lib, launch_symbol)
        fn.argtypes = _launch_argtypes(self._compiled._kernel_signature)
        fn.restype = None
        self._launch_fn = fn
        self._launch_symbol = launch_symbol

    def __call__(self, *args):
        self._ensure_launch_fn()
        marshaled = _marshal_launch_args(self._compiled._kernel_signature, args)
        self._launch_fn(
            ctypes.c_uint32(self._grid),
            _normalize_stream_ptr(self._stream),
            *marshaled,
        )


def _launch_argtypes(kernel_signature):
    argtypes = [ctypes.c_uint32, ctypes.c_void_p]
    for param in kernel_signature.positional_parameters:
        if isinstance(param, DeviceParameterSpec):
            argtypes.append(ctypes.c_void_p)
            continue
        if isinstance(param, RuntimeScalarParameterSpec):
            argtypes.append(_ctype_for_runtime_scalar(param.annotation))
            continue
        raise TypeError(f"unsupported launch parameter spec: {param!r}")
    return argtypes


def parse_launch_spec(launch_spec) -> tuple[int, object]:
    if not isinstance(launch_spec, tuple) or len(launch_spec) != 2:
        raise TypeError(
            "compiled launch syntax expects compiled[grid, stream]; "
            f"got {type(launch_spec)!r} with length "
            f"{len(launch_spec) if isinstance(launch_spec, tuple) else 'n/a'}"
        )
    grid, stream = launch_spec
    return grid, stream


__all__ = [
    "LaunchHandle",
    "build_and_load_native_library",
    "parse_launch_spec",
]
