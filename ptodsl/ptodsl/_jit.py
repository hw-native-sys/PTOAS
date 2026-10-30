# coding=utf-8
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
import numbers
from collections.abc import Mapping
from dataclasses import dataclass

from ._diagnostics import (
    invalid_jit_backend_error,
    invalid_jit_mode_error,
    jit_sched_mode_entry_false_error,
    jit_sched_mode_range_error,
    jit_sched_mode_source_error,
    jit_sched_mode_type_error,
    jit_source_constexpr_error,
    jit_source_entry_false_error,
    kernel_module_launch_error,
)
from ._kernel_compilation import CompiledKernelHandle, KernelCompiler
from ._kernel_signature import KernelSignature, parse_jit_kernel_signature
from ._native_options import normalize_native_options
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


@dataclass(frozen=True)
class _JitOptions:
    name: str | None
    target: str
    kernel_kind: object
    backend: str
    entry: bool
    sched_mode: int | None
    mode: str
    insert_sync: bool | None
    source: str | None
    native_options: Mapping | None
    ptoas_flags: tuple[str, ...]
    ast_rewrite: bool


def _jit_source_file(fn) -> str | None:
    try:
        return inspect.getsourcefile(fn) or inspect.getfile(fn)
    except (OSError, TypeError):
        return None


def _validate_jit_source(
    source, *, entry: bool, kernel_signature, fn_name: str
) -> None:
    if source is None:
        return
    if not isinstance(source, str):
        raise TypeError(
            "@pto.jit source must be a filesystem path string when provided"
        )
    if entry is False:
        raise jit_source_entry_false_error(source, function_name=fn_name)
    if kernel_signature.constexpr_parameters:
        raise jit_source_constexpr_error(
            kernel_signature.constexpr_parameters[0].name,
            source,
            function_name=fn_name,
        )


def _build_jit_kernel_spec(
    fn, options: _JitOptions
) -> tuple[KernelModuleSpec, KernelSignature]:
    fn_name = options.name or fn.__name__
    kernel_signature = parse_jit_kernel_signature(fn, entry=options.entry)
    normalized_mode = _normalize_mode(options.mode, fn=fn)
    normalized_backend = _normalize_backend(options.backend, fn=fn)
    _validate_jit_source(
        options.source,
        entry=options.entry,
        kernel_signature=kernel_signature,
        fn_name=fn_name,
    )
    source_file = _jit_source_file(fn)
    normalized_native_options = normalize_native_options(
        options.native_options,
        declaring_file=source_file,
        function_name=fn_name,
    )
    kernel_kind_explicit = options.kernel_kind is not _DEFAULT_KERNEL_KIND_SENTINEL
    effective_kernel_kind = (
        options.kernel_kind if kernel_kind_explicit else _DEFAULT_KERNEL_KIND
    )
    module_spec = KernelModuleSpec(
        function_name=fn_name,
        target_arch=options.target,
        kernel_kind=effective_kernel_kind,
        kernel_kind_explicit=kernel_kind_explicit,
        backend=normalized_backend,
        entry=options.entry,
        sched_mode=options.sched_mode,
        mode=normalized_mode,
        insert_sync=options.insert_sync,
        module_style=ModuleStyle.BACKEND_PARTITIONED,
        source_file=source_file,
        source_line=getattr(fn.__code__, "co_firstlineno", None),
        jit_source=options.source,
        native_options=normalized_native_options,
        ptoas_flags=options.ptoas_flags,
    )
    return module_spec, kernel_signature


def _normalize_sched_mode(value, *, entry: bool, source) -> int | None:
    """Validate ``@pto.jit(sched_mode=...)`` before a kernel captures it.

    The accepted value set ({0, 1}) and the kernel-entry placement rule stay
    PTOAS's job: `validatePTOSchedModeAttr` rejects anything else with the
    documented diagnostic, and a `@pto.jit(sched_mode=2)` kernel fails at
    compilation. What is checked here is the Python-side contract the verifier
    cannot express: the argument has to be an integer in the signless-i32
    range, and it has to reach the IR at all. `True` would silently become 1,
    `1.0` and "1" would raise a raw pybind TypeError, and 2**32 + 1 would wrap
    to a valid 1 -- none of those should depend on the caller reading the
    docstring. numpy integer scalars are accepted and narrowed to int, because
    host code routinely produces them for sizes and counts.
    """
    if value is None:
        return None
    # bool is an Integral too, but True == 1 would silently mean batch mode.
    # Anything else integral (numpy scalars included) is narrowed to int so the
    # module spec and the cache signature hold a plain Python int.
    if isinstance(value, bool) or not isinstance(value, numbers.Integral):
        raise jit_sched_mode_type_error(value)
    value = int(value)
    if not -(2**31) <= value <= 2**31 - 1:
        raise jit_sched_mode_range_error(value)
    if entry is False:
        raise jit_sched_mode_entry_false_error()
    if source is not None:
        raise jit_sched_mode_source_error(source)
    return value


def jit(
    name=None,
    *,
    target: str = "a5",
    kernel_kind: str | None = _DEFAULT_KERNEL_KIND_SENTINEL,
    backend: str = "vpto",
    entry: bool = True,
    sched_mode: int | None = None,
    mode: str = "auto",
    insert_sync: bool | None = None,
    ast_rewrite: bool | None = None,
    frontend_options: Mapping | None = None,
    native_options: Mapping | None = None,
    ptoas_flags: tuple[str, ...] = (),
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
    sched_mode:  optional CCE kernel scheduling mode for an entry kernel:
                 ``0`` normal mode (the compiler default) or ``1`` batch
                 mode. Recorded as the signless-i32 ``pto.sched_mode``
                 attribute on the generated entry function. The value must be
                 an integer in the signless-i32 range -- ``int`` or any
                 integral scalar, numpy integers included; ``bool``, floats,
                 strings and anything that would wrap are rejected here so a
                 coerced value can never reach the IR. Which values are legal
                 stays PTOAS's job, so ``2`` is only rejected once the kernel
                 compiles.
                 Omit it to leave the attribute off the function. Combining it
                 with ``entry=False`` or ``source=`` is rejected at decoration
                 time: neither path can carry the attribute to the IR.
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
    native_options:
                 Host-side additions to the native build, for a kernel whose
                 host side already exists in C++. Supports ``host_sources``
                 (C++ files compiled and linked into the kernel's shared
                 library), ``include_dirs`` for those sources, and
                 ``link_libraries`` / ``library_dirs`` for the link. Relative
                 paths resolve against the declaring file. Host sources are part
                 of the build's identity, so editing one rebuilds the library.
    source:
                 Optional filesystem path to a PTO IR source file. When
                 provided, PTODSL keeps the Python signature as the host ABI
                 declaration but loads the kernel implementation from the
                 source file instead of tracing the Python body.
    ptoas_flags:
                 Additional PTOAS command-line options for this kernel's native
                 build, for example layout search budgets. Pass a tuple of
                 individual arguments. The options are part of the build cache
                 identity and do not affect other kernels.

    The decorated function is replaced by a :class:`KernelHandle` that:

    - supports ``my_kernel.compile(**constexprs)`` specialization,
    - prints as the default-specialization MLIR text,
    - exposes ``my_kernel.mlir_module()`` / ``verify()`` / ``emit()`` on the
      default specialization for convenience.
    - emits a backend-partitioned outer container by default.
    """
    if not isinstance(ptoas_flags, (tuple, list)) or any(
        not isinstance(flag, str) or not flag.startswith("--") or "\x00" in flag
        for flag in ptoas_flags
    ):
        raise TypeError("ptoas_flags must be a tuple or list of --option or --option=value strings")
    normalized_ptoas_flags = tuple(ptoas_flags)
    normalized_sched_mode = _normalize_sched_mode(sched_mode, entry=entry, source=source)
    normalized_ast_rewrite = _normalize_frontend_options(
        ast_rewrite=ast_rewrite,
        frontend_options=frontend_options,
    )

    options = _JitOptions(
        name=name,
        target=target,
        kernel_kind=kernel_kind,
        backend=backend,
        entry=entry,
        sched_mode=normalized_sched_mode,
        mode=mode,
        insert_sync=insert_sync,
        source=source,
        native_options=native_options,
        ptoas_flags=normalized_ptoas_flags,
        ast_rewrite=normalized_ast_rewrite,
    )

    def decorator(fn):
        module_spec, kernel_signature = _build_jit_kernel_spec(fn, options)
        compiler = KernelCompiler(
            fn.__name__,
            module_spec,
            kernel_signature,
            fn,
            ast_rewrite=options.ast_rewrite,
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
        # Keep sched_mode ahead of ptoas_flags: tests index this tuple by position.
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
            module_spec.native_options,
            module_spec.sched_mode,
            module_spec.ptoas_flags,
        )

    def _build_default_module(self):
        return self.compile().build()


__all__ = ["CompiledKernelHandle", "jit", "KernelHandle", "merge_jit_modules"]
