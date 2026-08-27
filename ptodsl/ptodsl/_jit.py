# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""``@pto.jit`` decorator and compiled-kernel handles."""

from __future__ import annotations

import inspect
from collections.abc import Mapping

from ._diagnostics import (
    invalid_jit_backend_error,
    invalid_jit_mode_error,
    jit_source_constexpr_error,
    jit_source_entry_false_error,
    kernel_module_launch_error,
)
from ._kernel_compilation import CompiledKernelHandle, KernelCompiler
from ._kernel_signature import parse_jit_kernel_signature
from ._tracing import (
    KernelModuleSpec,
    ModuleArtifact,
    ModuleStyle,
    current_runtime,
)

from ptoas.mlir.ir import InsertionPoint, Module, WalkResult


_MODULE_ATTRS = ("pto.target_arch", "pto.backend")
_SUPPORTED_FRONTEND_OPTION_KEYS = {"ast_rewrite", "rewrite_part", "dump_rewritten_source"}
_SUPPORTED_REWRITE_PARTS = {"control_flow"}
_DEFAULT_KERNEL_KIND = None


class _DefaultKernelKindSentinel:
    def __repr__(self) -> str:
        return repr(_DEFAULT_KERNEL_KIND)


_DEFAULT_KERNEL_KIND_SENTINEL = _DefaultKernelKindSentinel()


def _normalize_mode(mode: str, *, fn=None) -> str:
    if mode not in {"auto", "explicit"}:
        source_file = None
        source_line = None
        function_name = None
        if fn is not None:
            function_name = fn.__name__
            try:
                source_file = inspect.getsourcefile(fn) or inspect.getfile(fn)
            except (OSError, TypeError):
                source_file = None
            source_line = getattr(getattr(fn, "__code__", None), "co_firstlineno", None)
        raise invalid_jit_mode_error(
            mode,
            function_name=function_name,
            source_file=source_file,
            source_line=source_line,
        )
    return mode


def _normalize_backend(backend: str, *, fn=None) -> str:
    if backend not in {"vpto", "emitc"}:
        source_file = None
        source_line = None
        function_name = None
        if fn is not None:
            function_name = fn.__name__
            try:
                source_file = inspect.getsourcefile(fn) or inspect.getfile(fn)
            except (OSError, TypeError):
                source_file = None
            source_line = getattr(getattr(fn, "__code__", None), "co_firstlineno", None)
        raise invalid_jit_backend_error(
            backend,
            function_name=function_name,
            source_file=source_file,
            source_line=source_line,
        )
    return backend


def _module_attr_map(module):
    attrs = module.operation.attributes
    return {name: str(attrs[name]) for name in _MODULE_ATTRS if name in attrs}


def _normalize_rewrite_parts(frontend_options):
    rewrite_part = frontend_options.get("rewrite_part", {"control_flow"})
    if isinstance(rewrite_part, str):
        rewrite_parts = {rewrite_part}
    else:
        try:
            rewrite_parts = set(rewrite_part)
        except TypeError as exc:
            raise TypeError(
                "@pto.jit frontend_options['rewrite_part'] must be a string or iterable of strings"
            ) from exc
    unsupported_parts = rewrite_parts - _SUPPORTED_REWRITE_PARTS
    if unsupported_parts:
        raise ValueError(
            "@pto.jit frontend_options['rewrite_part'] currently only supports "
            f"{sorted(_SUPPORTED_REWRITE_PARTS)!r}; got unsupported parts: {sorted(unsupported_parts)!r}"
        )
    return rewrite_parts


def _validate_dump_rewritten_source(frontend_options):
    dump_rewritten_source = frontend_options.get("dump_rewritten_source", False)
    if not isinstance(dump_rewritten_source, bool):
        raise TypeError("@pto.jit frontend_options['dump_rewritten_source'] must be a bool")
    if dump_rewritten_source:
        raise ValueError(
            "@pto.jit frontend_options['dump_rewritten_source']=True is reserved but not implemented yet"
        )


def _normalize_frontend_options(*, ast_rewrite, frontend_options):
    if frontend_options is None:
        return True if ast_rewrite is None else bool(ast_rewrite)
    if not isinstance(frontend_options, Mapping):
        raise TypeError("@pto.jit frontend_options must be a mapping")

    unknown = set(frontend_options) - _SUPPORTED_FRONTEND_OPTION_KEYS
    if unknown:
        raise ValueError(f"@pto.jit frontend_options has unsupported keys: {sorted(unknown)!r}")

    option_ast_rewrite = frontend_options.get("ast_rewrite")
    if option_ast_rewrite is not None and not isinstance(option_ast_rewrite, bool):
        raise TypeError("@pto.jit frontend_options['ast_rewrite'] must be a bool")
    if ast_rewrite is not None and option_ast_rewrite is not None and bool(ast_rewrite) != option_ast_rewrite:
        raise ValueError("@pto.jit ast_rewrite conflicts with frontend_options['ast_rewrite']")

    enabled = option_ast_rewrite if option_ast_rewrite is not None else (
        True if ast_rewrite is None else bool(ast_rewrite)
    )
    rewrite_parts = _normalize_rewrite_parts(frontend_options)
    if enabled and "control_flow" not in rewrite_parts:
        raise ValueError("@pto.jit ast_rewrite=True requires rewrite_part to include 'control_flow'")
    _validate_dump_rewritten_source(frontend_options)
    return enabled


def merge_jit_modules(*kernels: KernelHandle):
    """
    Merge multiple ``@pto.jit`` kernels into one MLIR module.

    Each handle must have compatible top-level attributes and use the same
    flat or backend-partitioned module shape. Payload order follows *kernels*.
    Backend child modules are independent symbol tables, so they may contain
    equal function names. Flat modules must use distinct function symbols;
    pass ``name=`` to ``@pto.jit`` when merging otherwise identical kernels.
    """
    if not kernels:
        raise ValueError("merge_jit_modules() requires at least one kernel handle")

    merged = kernels[0].build()
    expected_attrs = _module_attr_map(merged)
    merged_is_partitioned = all(
        op.operation.name == "builtin.module" for op in merged.body.operations
    )

    def direct_function_symbols(module):
        return {
            str(op.operation.attributes["sym_name"])
            for op in module.body.operations
            if op.operation.name == "func.func"
            and "sym_name" in op.operation.attributes
        }

    merged_symbols = (
        direct_function_symbols(merged) if not merged_is_partitioned else set()
    )

    target_context = merged.context
    for kernel in kernels[1:]:
        module = kernel.build()
        actual_attrs = _module_attr_map(module)
        if actual_attrs != expected_attrs:
            raise ValueError(
                "merge_jit_modules() requires compatible module attributes; "
                f"expected {expected_attrs}, got {actual_attrs}"
            )
        module_is_partitioned = all(
            op.operation.name == "builtin.module" for op in module.body.operations
        )
        if module_is_partitioned != merged_is_partitioned:
            raise ValueError(
                "merge_jit_modules() cannot mix flat and backend-partitioned PTODSL modules"
            )
        if not merged_is_partitioned:
            module_symbols = direct_function_symbols(module)
            duplicates = merged_symbols & module_symbols
            if duplicates:
                raise ValueError(
                    "merge_jit_modules() cannot merge duplicate function symbols "
                    "in a flat module; assign distinct @pto.jit(name=...) values: "
                    f"{sorted(duplicates)!r}"
                )
            merged_symbols.update(module_symbols)

        # Kernel modules are built in independent MLIR contexts.  Cloning an
        # operation directly across contexts leaves its types and attributes
        # owned by the source context, which may be destroyed before the
        # merged module.  Reparse the source text in the target context before
        # cloning so every operation in the merged module has one owner.
        module = Module.parse(str(module), target_context)
        with InsertionPoint(merged.body):
            for op in module.body.operations:
                op.operation.clone()

    merged.operation.verify()
    return merged


def jit(
    name=None,
    *,
    target: str = "a5",
    kernel_kind: str | None = _DEFAULT_KERNEL_KIND_SENTINEL,
    backend: str = "vpto",
    entry: bool = True,
    mode: str = "auto",
    insert_sync: bool | None = None,
    ast_rewrite: bool | None = None,
    frontend_options: Mapping | None = None,
    source: str | None = None,
):
    """
    Decorator that wraps a Python function as a PTODSL JIT kernel template.

    Parameters
    ----------
    name:        IR function name (defaults to the Python function name).
    target:      Target architecture string, e.g. ``"a5"``.
    kernel_kind: optional authored physical kind, used for native build selection
                 and explicit single-kind VPTO authoring intent. When omitted,
                 PTODSL leaves the kind unspecified so PTOAS can split mixed
                 cube/vector regions.
    backend:     ``"vpto"`` or ``"emitc"`` – records the intended backend.
    entry:       ``True`` for launchable kernel entries, ``False`` for helpers.
    mode:        ``"auto"`` or ``"explicit"`` – feeds child compile policy.
    insert_sync: ``True``/``False`` to explicitly control PTOAS sync insertion
                 for launch builds. ``None`` keeps the mode-based default
                 behavior.
    ast_rewrite:
                 ``True`` enables AST rewriting of Python ``if`` /
                 ``for range(...)`` into device-side PTODSL control flow.
                 Defaults to ``True``. ``False`` is intended for frontend
                 debugging and trace-time compatibility checks.
    frontend_options:
                 Reserved structured frontend options. Currently supports
                 ``ast_rewrite`` and ``rewrite_part={"control_flow"}``.
    source:
                 Optional filesystem path to a PTO IR source file. When
                 provided, PTODSL keeps the Python signature as the host ABI
                 declaration but loads the kernel implementation from the
                 source file instead of tracing the Python body.

    The decorated function is replaced by a :class:`KernelHandle` that:

    - supports ``my_kernel.compile(**constexprs)`` specialization,
    - prints as the default-specialization MLIR text,
    - exposes ``my_kernel.mlir_module()`` / ``verify()`` / ``emit()`` on the
      default specialization for convenience.
    - emits a backend-partitioned outer container by default.
    """
    normalized_ast_rewrite = _normalize_frontend_options(
        ast_rewrite=ast_rewrite,
        frontend_options=frontend_options,
    )

    def decorator(fn):
        fn_name = name or fn.__name__
        kernel_signature = parse_jit_kernel_signature(fn, entry=entry)
        normalized_mode = _normalize_mode(mode, fn=fn)
        normalized_backend = _normalize_backend(backend, fn=fn)
        if source is not None:
            if not isinstance(source, str):
                raise TypeError("@pto.jit source must be a filesystem path string when provided")
            if entry is False:
                raise jit_source_entry_false_error(source, function_name=fn_name)
            if kernel_signature.constexpr_parameters:
                raise jit_source_constexpr_error(
                    kernel_signature.constexpr_parameters[0].name,
                    source,
                    function_name=fn_name,
                )
        source_file = None
        try:
            source_file = inspect.getsourcefile(fn) or inspect.getfile(fn)
        except (OSError, TypeError):
            source_file = None
        kernel_kind_explicit = kernel_kind is not _DEFAULT_KERNEL_KIND_SENTINEL
        effective_kernel_kind = kernel_kind if kernel_kind_explicit else _DEFAULT_KERNEL_KIND
        compiler = KernelCompiler(
            fn.__name__,
            KernelModuleSpec(
                function_name=fn_name,
                target_arch=target,
                kernel_kind=effective_kernel_kind,
                kernel_kind_explicit=kernel_kind_explicit,
                backend=normalized_backend,
                entry=entry,
                mode=normalized_mode,
                insert_sync=insert_sync,
                module_style=ModuleStyle.BACKEND_PARTITIONED,
                source_file=source_file,
                source_line=getattr(fn.__code__, "co_firstlineno", None),
                jit_source=source,
            ),
            kernel_signature,
            fn,
            ast_rewrite=normalized_ast_rewrite,
        )
        return KernelHandle(fn.__name__, compiler)

    return decorator


class KernelHandle(ModuleArtifact):
    """
    Represents a JIT kernel template plus its compiled specializations.

    ``handle.compile(**constexprs)`` returns one compiled specialization.
    ``print(handle)`` emits the default-specialization MLIR module text.
    """

    def __init__(self, py_name: str, compiler: KernelCompiler):
        self._compiler = compiler
        super().__init__(py_name, module_factory=self._build_default_module)

    def compile(self, **constexpr_bindings) -> CompiledKernelHandle:
        return self._compiler.compile(**constexpr_bindings)

    def __call__(self, *args, **kwargs):
        if self._compiler._module_spec.entry is not False:
            raise TypeError(
                f"@pto.jit entry kernel {self._py_name!r} is not callable from traced PTODSL bodies; "
                "only @pto.jit(entry=False) kernel modules may be called there."
            )
        runtime = current_runtime()
        if runtime is None:
            raise RuntimeError(
                f"@pto.jit(entry=False) kernel module {self._py_name!r} may only be called while tracing "
                "a compatible PTODSL kernel"
            )
        return runtime.dispatch_kernel_module_call(self, *args, **kwargs)

    def __getitem__(self, launch_spec):
        if self._compiler._module_spec.entry is False:
            raise kernel_module_launch_error(self._py_name)
        return self.compile().__getitem__(launch_spec)

    def cached_specializations(self):
        return self._compiler.cached_specializations()

    def __ptodsl_cache_signature__(self):
        module_spec = self._compiler._module_spec
        return (
            type(self).__name__,
            self._compiler._py_name,
            self._compiler._kernel_identity,
            module_spec.function_name,
            module_spec.entry,
            module_spec.backend,
            module_spec.mode,
            module_spec.kernel_kind,
            module_spec.kernel_kind_explicit,
        )

    def _build_default_module(self):
        return self.compile().build()


__all__ = ["CompiledKernelHandle", "jit", "KernelHandle", "merge_jit_modules"]
