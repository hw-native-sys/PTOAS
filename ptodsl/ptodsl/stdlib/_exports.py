# coding=utf-8
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Immutable public export catalog for the PTODSL standard library.

Each catalog entry maps one public ``pto.*`` name to its implementation
module and attribute::

    EXPORTS = {
        "init_core": ("ptodsl.stdlib.init_core", "init_core"),
    }

``ptodsl.pto`` consults :data:`EXPORTS` lazily: the implementation module is
imported only when a kernel first resolves the public name, and the loaded
object is cached on the ``pto`` module afterwards.  Importing ``ptodsl``
alone therefore never eagerly loads any implementation module.

:data:`EXPORTS` is exposed as a read-only ``MappingProxyType``; runtime code
and tests must treat it as immutable and never mutate the production catalog.
"""

__all__ = ["EXPORTS", "public_export_names", "resolve_export"]

from importlib import import_module
import inspect
from types import MappingProxyType

_STDLIB_PACKAGE_PREFIX = "ptodsl.stdlib."

_EXPORTS = {
    "init_core": ("ptodsl.stdlib.init_core", "init_core"),
}

EXPORTS = MappingProxyType(_EXPORTS)


def _validate_catalog():
    """Fail deterministically on malformed catalog structure at import time."""
    for name, entry in EXPORTS.items():
        if not name.isidentifier():
            raise ValueError(
                f"PTODSL stdlib export name {name!r} is not a valid identifier"
            )
        if (
            not isinstance(entry, tuple)
            or len(entry) != 2
            or not all(isinstance(part, str) for part in entry)
        ):
            raise TypeError(
                f"PTODSL stdlib export {name!r} must map to a "
                f"(module_path, attribute_name) string pair, got {entry!r}"
            )
        module_path, attribute_name = entry
        if not module_path.startswith(_STDLIB_PACKAGE_PREFIX):
            raise ValueError(
                f"PTODSL stdlib export {name!r} must live under "
                f"{_STDLIB_PACKAGE_PREFIX}*, got module {module_path!r}"
            )
        if not attribute_name.isidentifier():
            raise ValueError(
                f"PTODSL stdlib export {name!r} declares an invalid "
                f"implementation attribute {attribute_name!r}"
            )


_validate_catalog()


def public_export_names():
    """Return the immutable public names offered by the stdlib catalog."""
    return tuple(EXPORTS)


def resolve_export(name):
    """Import, validate, and return the implementation object for *name*.

    The returned object must be a ``@pto.func`` ``FuncTemplate`` whose public
    name, implementation name, zero-argument signature, and ``returns=None``
    declaration all agree.  Every failure names the public export, the
    implementation module, and the original cause.
    """
    from .._func import FuncTemplate

    if name not in EXPORTS:
        raise KeyError(f"{name!r} is not a PTODSL stdlib export")
    module_path, attribute_name = EXPORTS[name]
    try:
        module = import_module(module_path)
    except Exception as exc:
        raise RuntimeError(
            f"failed to load PTODSL stdlib export {name!r} from module "
            f"{module_path!r}: {exc}"
        ) from exc

    impl = getattr(module, attribute_name, None)
    if not isinstance(impl, FuncTemplate):
        raise TypeError(
            f"PTODSL stdlib export {name!r} from module {module_path!r} must "
            f"be a @pto.func FuncTemplate, got {type(impl).__name__}"
        )
    if impl.spec.symbol_name != name or impl.py_fn.__name__ != name:
        raise TypeError(
            f"PTODSL stdlib export {name!r} from module {module_path!r} must "
            f"match its implementation name "
            f"({impl.spec.symbol_name!r}/{impl.py_fn.__name__!r})"
        )
    if len(impl.signature.parameters) != 0:
        raise TypeError(
            f"PTODSL stdlib export {name!r} from module {module_path!r} must "
            f"take no arguments, got signature "
            f"{inspect.signature(impl.py_fn)}"
        )
    if impl.declared_returns is not None:
        raise TypeError(
            f"PTODSL stdlib export {name!r} from module {module_path!r} must "
            f"declare returns=None, got {impl.declared_returns!r}"
        )
    return impl
