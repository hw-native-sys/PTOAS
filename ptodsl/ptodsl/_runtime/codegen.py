# coding=utf-8
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Generate host-side launch wrappers for traced PTODSL kernels."""

from __future__ import annotations

from .._context import make_context
from ptoas.mlir.ir import BF16Type, F16Type, F32Type, IndexType, IntegerType

from .._kernel_signature import DeviceParameterSpec, RuntimeScalarParameterSpec
from .._types import _PtrDescriptor, _resolve


def _elem_cpp_type(elem) -> str:
    name = getattr(elem, "__name__", repr(elem)).lower()
    mapping = {
        "float32": "float",
        "f32": "float",
        "float16": "__fp16",
        "f16": "__fp16",
        "bf16": "__bf16",
        "int8": "int8_t",
        "int16": "int16_t",
        "int32": "int32_t",
        "int64": "int64_t",
        "ui8": "uint8_t",
        "ui16": "uint16_t",
        "ui32": "uint32_t",
        "ui64": "uint64_t",
    }
    for key, cpp in mapping.items():
        if key in name:
            return cpp
    return "float"


def _device_param_cpp_type(annotation) -> str:
    if isinstance(annotation, _PtrDescriptor):
        return _elem_cpp_type(annotation._elem)
    type_repr = repr(annotation).replace(" ", "").lower()
    if "f32" in type_repr or "float32" in type_repr:
        return "float"
    if "i32" in type_repr or "int32" in type_repr:
        return "int32_t"
    if "i64" in type_repr or "int64" in type_repr:
        return "int64_t"
    return "float"


def _runtime_scalar_cpp_type(annotation) -> str:
    try:
        type_obj = _resolve(annotation)
    except RuntimeError as exc:
        if "requires a Context" not in str(exc):
            raise
        with make_context():
            type_obj = _resolve(annotation)
    if IndexType.isinstance(type_obj):
        return "int64_t"
    if IntegerType.isinstance(type_obj):
        width = IntegerType(type_obj).width
        if width == 1:
            return "bool"
        signedness = str(type_obj)
        if signedness.startswith("ui"):
            return {
                8: "uint8_t",
                16: "uint16_t",
                32: "uint32_t",
                64: "uint64_t",
            }[width]
        return {
            8: "int8_t",
            16: "int16_t",
            32: "int32_t",
            64: "int64_t",
        }[width]
    if F32Type.isinstance(type_obj):
        return "float"
    if F16Type.isinstance(type_obj):
        return "__fp16"
    if BF16Type.isinstance(type_obj):
        return "__bf16"
    raise TypeError(f"unsupported @pto.jit runtime scalar codegen type {type_obj}")


def launch_symbol_name(ir_function_name: str) -> str:
    return f"ptodsl_launch_{ir_function_name}"

def generate_launch_cpp(*, ir_function_name: str, kernel_signature) -> str:
    """Return C++ source for one extern-C launch entry point."""
    gm_params = []
    host_params = []
    kernel_args = []

    for param in kernel_signature.positional_parameters:
        if isinstance(param, DeviceParameterSpec):
            cpp_type = _device_param_cpp_type(param.annotation)
            gm_params.append(f"__gm__ {cpp_type} *{param.name}")
            host_params.append(f"{cpp_type} *{param.name}")
            kernel_args.append(f"(__gm__ {cpp_type} *){param.name}")
            continue
        if isinstance(param, RuntimeScalarParameterSpec):
            cpp_type = _runtime_scalar_cpp_type(param.annotation)
            gm_params.append(f"{cpp_type} {param.name}")
            host_params.append(f"{cpp_type} {param.name}")
            kernel_args.append(param.name)
            continue
        raise TypeError(f"unsupported launch parameter spec: {param!r}")

    gm_sig = ", ".join(gm_params)
    host_sig = ", ".join(["uint32_t grid", "void *stream"] + host_params)
    kernel_call = ", ".join(kernel_args)
    launch_symbol = launch_symbol_name(ir_function_name)

    return (
        "#include <stdint.h>\n\n"
        "#ifndef AICORE\n"
        "#define AICORE [aicore]\n"
        "#endif\n\n"
        f'extern "C" __global__ AICORE void {ir_function_name}({gm_sig});\n\n'
        f"extern \"C\" void {launch_symbol}({host_sig}) {{\n"
        f"    {ir_function_name}<<<grid, nullptr, stream>>>({kernel_call});\n"
        "}\n"
    )


__all__ = [
    "generate_launch_cpp",
    "launch_symbol_name",
]
