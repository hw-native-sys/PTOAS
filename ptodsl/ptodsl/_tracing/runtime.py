# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""Base tracing runtimes shared by PTODSL frontends."""

from __future__ import annotations

from .active import activate_runtime, activate_session, require_active_session
from .module_builder import create_kernel_module
from .session import TraceSession
from .._diagnostics import kernel_module_return_value_error
from .._context import make_context
from .._types import _resolve

from ptoas.mlir.dialects import func
from ptoas.mlir.ir import InsertionPoint, Location


class TracingRuntime:
    """Shared module-building runtime for tracing-based PTODSL frontends."""

    def __init__(self, module_spec):
        self.module_spec = module_spec

    def compute_argument_types(self):
        """Return the MLIR entry argument types for this runtime."""
        raise NotImplementedError

    def bind_entry_arguments(self, entry_arguments):
        """Wrap raw entry-block arguments into surface values."""
        return tuple(entry_arguments)

    def trace_entry(self, *args):
        """Emit the traced function body using wrapped entry arguments."""
        raise NotImplementedError

    def validate_trace_state(self):
        """Validate runtime-local tracing state before the function returns."""

    def emit_return(self):
        """Emit the function return terminator."""
        func.ReturnOp([])

    def verify_module(self, module):
        """Verify the completed module."""
        module.operation.verify()

    def create_session(self, module, entry_function):
        """Create the shared trace session for this build."""
        return TraceSession(self.module_spec, module, entry_function)

    def initialize_session(self, session, entry_block):
        """Populate runtime-specific session state before tracing."""
        session.bind_entry_block(entry_block)

    def finalize_session(self, session):
        """Finalize runtime-specific session state after tracing."""

    def dispatch_subkernel_call(self, subkernel, *args, **kwargs):
        """Dispatch a decorated PTODSL subkernel call in the active trace."""
        session = require_active_session(f"@pto.{subkernel.spec.role.value}")
        if subkernel.spec.role.value in {"cube", "simd", "simt", "tileop"}:
            return session.lower_helper_subkernel(subkernel, *args, **kwargs)
        return subkernel.emit_body(*args, **kwargs)

    def dispatch_ptodsl_func_call(self, func_template, *args, **kwargs):
        """Dispatch one ``@pto.func`` helper call in the active trace."""
        session = require_active_session("@pto.func")
        return session.lower_ptodsl_func_call(func_template, *args, **kwargs)

    def dispatch_kernel_module_call(self, kernel_handle, *args, **kwargs):
        """Dispatch one ``@pto.jit(entry=False)`` kernel-module call in the active trace."""
        session = require_active_session("@pto.jit(entry=False)")
        return session.lower_kernel_module_call(kernel_handle, *args, **kwargs)

    def build_module(self):
        """Materialize the full MLIR module for this runtime."""
        ctx = make_context()
        with ctx, Location.unknown():
            arg_types = list(self.compute_argument_types())
            module, ir_fn = create_kernel_module(self.module_spec, arg_types)
            session = self.create_session(module, ir_fn)
            entry = ir_fn.add_entry_block()
            with InsertionPoint(entry), activate_runtime(self), activate_session(session):
                self.initialize_session(session, entry)
                args = self.bind_entry_arguments(entry.arguments)
                self.trace_entry(*args)
                self.validate_trace_state()
                self.emit_return()
                self.finalize_session(session)
                session.validate_final_state()
            self.verify_module(module)
            return module, {"kernel_module_graph": session.snapshot_kernel_module_graph()}


class CallbackTracingRuntime(TracingRuntime):
    """Small tracing runtime for eager callback-style module materialization."""

    def __init__(self, module_spec, arg_types, callback):
        super().__init__(module_spec)
        self._arg_types = tuple(arg_types)
        self._callback = callback

    def compute_argument_types(self):
        return tuple(_resolve(arg_type) for arg_type in self._arg_types)

    def trace_entry(self, *args):
        self._callback(*args)


class SignatureTracingRuntime(TracingRuntime):
    """Tracing runtime that binds a parsed PTODSL kernel signature."""

    def __init__(self, module_spec, kernel_signature, callback, *, constexpr_bindings=None):
        super().__init__(module_spec)
        self._kernel_signature = kernel_signature
        self._callback = callback
        self._constexpr_bindings = dict(constexpr_bindings or {})

    def compute_argument_types(self):
        return self._kernel_signature.compute_entry_arg_types()

    def bind_entry_arguments(self, entry_arguments):
        return self._kernel_signature.bind_entry_arguments(entry_arguments)

    def trace_entry(self, *args):
        kwargs = self._kernel_signature.default_constexpr_bindings()
        kwargs.update(self._constexpr_bindings)
        result = self._callback(*args, **kwargs)
        if self.module_spec.entry is False and result is not None:
            raise kernel_module_return_value_error(result)


__all__ = [
    "CallbackTracingRuntime",
    "SignatureTracingRuntime",
    "TracingRuntime",
]
