# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""Shared cache-signature helpers for PTODSL tracing frontends."""

from __future__ import annotations


def function_cache_signature(fn):
    """Return one stable cache signature for a Python function body."""
    code = fn.__code__
    return (
        "python-function",
        code.co_filename,
        code.co_firstlineno,
        getattr(code, "co_qualname", fn.__qualname__),
        code.co_argcount,
        code.co_posonlyargcount,
        code.co_kwonlyargcount,
        code.co_names,
        code.co_consts,
        code.co_code,
    )


def closure_cache_signature(fn):
    """Return one stable cache signature for the closure state captured by *fn*."""
    try:
        import inspect

        closure_vars = inspect.getclosurevars(fn)
    except TypeError:
        return ()
    return tuple(
        (name, cache_signature_atom(value))
        for name, value in sorted(closure_vars.nonlocals.items())
    )


def cache_signature_atom(value):
    """Return one hashable cache-signature atom for arbitrary captured values."""
    cache_signature = getattr(value, "__ptodsl_cache_signature__", None)
    if callable(cache_signature):
        return ("ptodsl-cache-signature", cache_signature_atom(cache_signature()))
    try:
        hash(value)
    except TypeError:
        if isinstance(value, dict):
            items = (
                (cache_signature_atom(key), cache_signature_atom(item))
                for key, item in value.items()
            )
            return ("dict", tuple(sorted(items, key=repr)))
        if isinstance(value, (list, tuple)):
            return (
                type(value).__name__,
                tuple(cache_signature_atom(item) for item in value),
            )
        if isinstance(value, set):
            return (
                "set",
                tuple(sorted((cache_signature_atom(item) for item in value), key=repr)),
            )
        return (type(value).__name__, repr(value))
    return value


__all__ = [
    "cache_signature_atom",
    "closure_cache_signature",
    "function_cache_signature",
]
