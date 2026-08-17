# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""``@pto.func`` decorator and reusable callable handle."""

from __future__ import annotations

from dataclasses import dataclass
from functools import update_wrapper
import inspect
import typing

from ._ast_rewrite import rewrite_jit_function
from ._cache_signature import cache_signature_atom, closure_cache_signature, function_cache_signature
from ._surface_values import unwrap_surface_value
from ._tracing import current_runtime

_RETURNS_UNSET = object()


@dataclass(frozen=True)
class FuncSpec:
    """Declarative metadata for a rewrite-capable reusable PTODSL callable."""

    symbol_name: str


class FuncTemplate:
    """Callable decorated PTODSL helper surface."""

    def __init__(self, spec: FuncSpec, py_fn, *, ast_rewrite: bool = True, returns=_RETURNS_UNSET):
        self.spec = spec
        self.py_fn = py_fn
        self._ast_rewrite = ast_rewrite
        self.signature = inspect.signature(py_fn)
        try:
            self.type_hints = typing.get_type_hints(py_fn)
        except Exception as exc:
            if _has_annotations(self.signature):
                raise TypeError(
                    f"failed to resolve @pto.func annotations for {py_fn.__qualname__!r}"
                ) from exc
            self.type_hints = {}
        if returns is not _RETURNS_UNSET:
            self.declared_returns = returns
        elif "return" in self.type_hints:
            self.declared_returns = self.type_hints["return"]
        elif self.signature.return_annotation is not inspect.Signature.empty:
            self.declared_returns = self.signature.return_annotation
        else:
            raise TypeError(
                "@pto.func helpers must explicitly declare return types with "
                "@pto.func(returns=...) or a Python return annotation; use "
                "returns=None or -> None for helpers that do not return values"
            )
        update_wrapper(self, py_fn)

    def emit_body(self, *args, **kwargs):
        """Emit this helper body into the currently active trace."""
        py_fn = (
            rewrite_jit_function(self.py_fn, reject_bare_returns=True)
            if self._ast_rewrite
            else self.py_fn
        )
        return py_fn(*args, **kwargs)

    def _validate_closure_captures(self):
        for name, value in inspect.getclosurevars(self.py_fn).nonlocals.items():
            captured = _find_runtime_value(value)
            if captured is not None:
                raise TypeError(
                    f"@pto.func {self.spec.symbol_name!r} captures runtime value {name!r} "
                    f"of type {captured.type}; pass it as an explicit parameter"
                )

    def __call__(self, *args, **kwargs):
        runtime = current_runtime()
        if runtime is None:
            raise RuntimeError(
                "@pto.func helpers may only be called while tracing a compatible PTODSL kernel"
            )
        self._validate_closure_captures()
        return runtime.dispatch_ptodsl_func_call(self, *args, **kwargs)

    def __ptodsl_cache_signature__(self):
        return (
            type(self).__name__,
            self.spec.symbol_name,
            function_cache_signature(self.py_fn),
            self._ast_rewrite,
            cache_signature_atom(self.declared_returns),
            closure_cache_signature(self.py_fn),
        )


def func(fn=None, *, name: str | None = None, ast_rewrite: bool = True, returns=_RETURNS_UNSET):
    """Decorate a Python function as a reusable PTODSL callable helper."""

    def decorator(py_fn):
        return FuncTemplate(
            FuncSpec(symbol_name=name or py_fn.__name__),
            py_fn,
            ast_rewrite=ast_rewrite,
            returns=returns,
        )

    if fn is not None:
        return decorator(fn)
    return decorator


def _has_annotations(signature: inspect.Signature) -> bool:
    if signature.return_annotation is not inspect.Signature.empty:
        return True
    return any(
        param.annotation is not inspect.Parameter.empty
        for param in signature.parameters.values()
    )


def _find_runtime_value(value):
    if isinstance(value, dict):
        values = value.items()
    elif isinstance(value, (tuple, list, set, frozenset)):
        values = value
    else:
        raw_value = unwrap_surface_value(value)
        return raw_value if hasattr(raw_value, "type") else None
    for item in values:
        captured = _find_runtime_value(item)
        if captured is not None:
            return captured
    return None


__all__ = [
    "FuncSpec",
    "FuncTemplate",
    "func",
]
