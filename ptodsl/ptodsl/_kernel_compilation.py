# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""Kernel specialization and compilation helpers for ``@pto.jit``."""

from __future__ import annotations

from ._ast_rewrite import rewrite_jit_function
from ._cache_signature import closure_cache_signature
from ._diagnostics import (
    jit_source_compile_constexpr_error,
    kernel_module_compile_error,
    kernel_module_launch_error,
)
from ._runtime.launch import LaunchHandle, parse_launch_spec
from ._source_loader import SourceModuleLoader
from ._tracing import ModuleArtifact, SignatureTracingRuntime


class CompiledKernelHandle(ModuleArtifact):
    """One compiled ``@pto.jit`` specialization."""

    def __init__(
        self,
        py_name: str,
        *,
        specialization_key,
        constexpr_bindings,
        module_factory,
        module_spec,
        kernel_signature,
    ):
        super().__init__(py_name, module_factory=module_factory)
        self._specialization_key = specialization_key
        self._constexpr_bindings = dict(constexpr_bindings)
        self._module_spec = module_spec
        self._kernel_signature = kernel_signature

    @property
    def specialization_key(self):
        return self._specialization_key

    @property
    def constexpr_bindings(self):
        return dict(self._constexpr_bindings)

    @property
    def ir_function_name(self):
        return self._module_spec.function_name

    @property
    def kernel_module_graph(self):
        """Return traced kernel-module import/dependency metadata for this build."""
        return self.build_metadata().get("kernel_module_graph")

    def __getitem__(self, launch_spec):
        if self._module_spec.entry is False:
            raise kernel_module_launch_error(self._py_name)
        grid, stream = parse_launch_spec(launch_spec)
        return LaunchHandle(self, grid, stream)


class KernelCompiler:
    """Per-kernel specialization cache and module builder."""

    def __init__(
        self,
        py_name: str,
        module_spec,
        kernel_signature,
        callback,
        *,
        ast_rewrite=True,
    ):
        self._py_name = py_name
        self._module_spec = module_spec
        self._kernel_signature = kernel_signature
        self._callback = callback
        self._kernel_identity = id(callback)
        self._ast_rewrite = ast_rewrite
        self._trace_callback_cache = {}
        self._compiled_cache = {}

    def tracing_callback(self, constexpr_bindings=None):
        cache_key = tuple(
            (name, constexpr_bindings[name])
            for name in sorted(constexpr_bindings or {})
        )
        cache_key = (self._ast_rewrite, cache_key)
        cached = self._trace_callback_cache.get(cache_key)
        if cached is None:
            cached = rewrite_jit_function(
                self._callback,
                static_bindings=constexpr_bindings,
                rewrite_control_flow=self._ast_rewrite,
            )
            self._trace_callback_cache[cache_key] = cached
        return cached

    def compile(self, **constexpr_bindings):
        if self._module_spec.entry is False:
            raise kernel_module_compile_error(self._py_name)
        if self._module_spec.jit_source is not None:
            return self._compile_source_backed(**constexpr_bindings)
        normalized_bindings = self._kernel_signature.bind_constexpr_bindings(constexpr_bindings)
        kernel_identity = self._kernel_identity
        if self._ast_rewrite:
            kernel_identity = (
                kernel_identity,
                closure_cache_signature(self._callback),
            )
        specialization_key = self._kernel_signature.specialization_key(
            kernel_identity,
            normalized_bindings,
        )

        cached = self._compiled_cache.get(specialization_key)
        if cached is not None:
            return cached

        callback = self.tracing_callback(normalized_bindings)
        runtime = SignatureTracingRuntime(
            self._module_spec,
            self._kernel_signature,
            callback,
            constexpr_bindings=normalized_bindings,
        )
        compiled = CompiledKernelHandle(
            self._py_name,
            specialization_key=specialization_key,
            constexpr_bindings=normalized_bindings,
            module_factory=runtime.build_module,
            module_spec=self._module_spec,
            kernel_signature=self._kernel_signature,
        )
        compiled.build()
        self._compiled_cache[specialization_key] = compiled
        return compiled

    def _compile_source_backed(self, **constexpr_bindings):
        if constexpr_bindings:
            raise jit_source_compile_constexpr_error(
                tuple(sorted(constexpr_bindings)),
                self._module_spec.jit_source,
                function_name=self._module_spec.function_name,
            )

        loader = SourceModuleLoader(self._module_spec, self._kernel_signature)
        specialization_key = self._kernel_signature.specialization_key(
            loader.cache_identity(),
            {},
        )

        cached = self._compiled_cache.get(specialization_key)
        if cached is not None:
            return cached

        compiled = CompiledKernelHandle(
            self._py_name,
            specialization_key=specialization_key,
            constexpr_bindings={},
            module_factory=loader.build_module,
            module_spec=self._module_spec,
            kernel_signature=self._kernel_signature,
        )
        compiled.build()
        self._compiled_cache[specialization_key] = compiled
        return compiled

    def cached_specializations(self):
        return tuple(self._compiled_cache.values())


__all__ = [
    "CompiledKernelHandle",
    "KernelCompiler",
]
