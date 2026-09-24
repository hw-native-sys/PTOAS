# coding=utf-8
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""In-process materialization entry point for ``lib/SoftOps`` DSL helpers."""

from __future__ import annotations

import json
import importlib

from .. import pto
from .._surface_values import unwrap_surface_value, wrap_surface_value
from .._tracing import KernelModuleSpec, ModuleStyle, CallbackTracingRuntime
from .._types import _resolve, mask_type, vreg_type
from ptoas.mlir.dialects import func
from ptoas.mlir.ir import InsertionPoint, Location, Module, StringAttr, Attribute, UnitAttr


class _SoftLibTraceRuntime(CallbackTracingRuntime):
    def emit_return(self):
        # A soft helper is a real vector function, unlike a launch entry.  The
        # callback returns the helper result and the return value is captured
        # while tracing below.
        from ptoas.mlir.dialects import func
        func.ReturnOp([self._result])

    def trace_entry(self, *args):
        self._result = unwrap_surface_value(self._callback(*args))


_INTEGER_VDIV_DTYPES = {"i16", "si16", "i32", "si32"}


def _integer_vdiv_dtype(dtype):
    if dtype == "si16":
        return pto.si16
    if dtype == "si32":
        return pto.si32
    return pto.i32 if dtype == "i32" else pto.i16


def _resolve_integer_vdiv_request(specs):
    dtype = specs["dtype"]
    lanes = int(specs.get("lanes", 128 if dtype in {"i16", "si16"} else 64))
    mask_bits = specs.get("mask", "b16" if dtype in {"i16", "si16"} else "b32")
    module_name = "div_i16_soft" if dtype in {"i16", "si16"} else "div_i32_soft"
    helper = getattr(importlib.import_module("SoftOps"), module_name)
    vreg = vreg_type(lanes, _integer_vdiv_dtype(dtype))
    arg_types = [vreg, vreg, mask_type(mask_bits)]
    return module_name, helper, arg_types, [vreg]


def _resolve_high_precision_vdiv_request(target, specs):
    if target != "a5" or specs.get("dtype") != "f32" or specs.get("mask") != "b32":
        raise ValueError("high-precision vector division currently requires A5 f32 with b32 mask")
    from SoftOps.div_vector import div_vf32_soft

    vector = vreg_type(int(specs["lanes"]), pto.f32)
    return "div_vf32_soft", div_vf32_soft, [vector, vector, mask_type("b32")], [vector]


def _resolve_high_precision_divf_request(target):
    if target != "a5":
        raise ValueError("high-precision scalar division requires target A5")
    helper = getattr(importlib.import_module("SoftOps"), "div_f32_soft")
    return "div_f32_soft", helper, [pto.f32, pto.f32], [pto.f32]


def _resolve_trig_request(op):
    module_name = "sin_f32_soft" if op == "pto.sin" else "cos_f32_soft"
    helper = getattr(importlib.import_module("SoftOps"), module_name)
    return module_name, helper, [pto.f32], [pto.f32]


def _resolve_softlib_request(target, op, specs):
    if op == "pto.vdiv" and specs.get("dtype") in _INTEGER_VDIV_DTYPES:
        return _resolve_integer_vdiv_request(specs)
    if op == "pto.vdiv" and specs.get("precision") == "high_precision":
        return _resolve_high_precision_vdiv_request(target, specs)
    if op == "pto.divf" and specs == {"dtype": "f32", "precision": "high_precision"}:
        return _resolve_high_precision_divf_request(target)
    if op in {"pto.sin", "pto.cos"}:
        return _resolve_trig_request(op)
    raise ValueError(f"no SoftOps implementation registered for {target}:{op}:{specs}")


def _softlib_callback(helper):
    def callback(*args):
        return helper(*[wrap_surface_value(arg) for arg in args])

    return callback


def _create_softlib_module(runtime, target, context, module_name, arg_types, result_types):
    with context, Location.unknown():
        arg_types = list(runtime.compute_argument_types())
        result_types = [_resolve(result_type) for result_type in result_types]
        module = Module.create()
        module.operation.attributes["pto.target_arch"] = StringAttr.get(target)
        with InsertionPoint(module.body):
            ir_fn = func.FuncOp(
                module_name,
                func.FunctionType.get(arg_types, result_types),
            )
            ir_fn.attributes["pto.kernel_kind"] = Attribute.parse("#pto.kernel_kind<vector>")
        session = runtime.create_session(module, ir_fn)
        entry = ir_fn.add_entry_block()
        from .._tracing.active import activate_runtime, activate_session

        with InsertionPoint(entry), activate_runtime(runtime), activate_session(session):
            runtime.initialize_session(session, entry)
            runtime.trace_entry(*runtime.bind_entry_arguments(entry.arguments))
            runtime.emit_return()
            runtime.finalize_session(session)
            session.validate_final_state()
        ir_fn.attributes["pto.softlib.instance"] = UnitAttr.get()
        module.operation.verify()
    return module


def materialize(target: str, op: str, operand_specs_json: str, context):
    try:
        specs = json.loads(operand_specs_json)
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid SoftLib materialization request: {exc}") from exc
    module_name, helper, arg_types, result_types = _resolve_softlib_request(target, op, specs)
    callback = _softlib_callback(helper)

    runtime = _SoftLibTraceRuntime(
        KernelModuleSpec(
            function_name=module_name,
            target_arch=target,
            kernel_kind="vector",
            module_style=ModuleStyle.FLAT_AICORE,
            entry=False,
            mode="explicit",
        ),
        arg_types,
        callback,
    )
    return _create_softlib_module(runtime, target, context, module_name, arg_types, result_types), module_name


__all__ = ["materialize"]
