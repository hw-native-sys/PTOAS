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


def materialize(target: str, op: str, operand_specs_json: str, context):
    try:
        specs = json.loads(operand_specs_json)
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid SoftLib materialization request: {exc}") from exc
    if op == "pto.vdiv" and specs.get("dtype") in {"i32", "si32"}:
        lanes = int(specs.get("lanes", 64))
        mask_bits = specs.get("mask", "b32")
        module_name = "div_i32_soft"
        helper = getattr(importlib.import_module("SoftOps"), module_name)
        dtype = specs["dtype"]
        integer_dtype = pto.si32 if dtype == "si32" else pto.i32
        arg_types = [
            vreg_type(lanes, integer_dtype),
            vreg_type(lanes, integer_dtype),
            mask_type(mask_bits),
        ]
        result_types = [arg_types[0]]
    elif op in {"pto.sin", "pto.cos"}:
        module_name = "sin_f32_soft" if op == "pto.sin" else "cos_f32_soft"
        helper = getattr(importlib.import_module("SoftOps"), module_name)
        # Keep type construction lazy until the caller-provided context is
        # active.  F32Type.get() is context-bound in the MLIR Python API.
        arg_types = [pto.f32]
        result_types = [pto.f32]
    else:
        raise ValueError(f"no SoftOps implementation registered for {target}:{op}:{specs}")

    callback = lambda *args: helper(*[wrap_surface_value(arg) for arg in args])

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
    return module, module_name


__all__ = ["materialize"]
