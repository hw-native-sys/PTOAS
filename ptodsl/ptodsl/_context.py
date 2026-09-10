# coding=utf-8
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""MLIR context construction for an installed or explicitly configured PTODSL."""

from ptoas.mlir.dialects import pto as _pto_dialect

try:
    from ptoas.mlir.dialects import llvm as _llvm_dialect
except Exception:  # pragma: no cover - depends on the installed MLIR package.
    _llvm_dialect = None

from ptoas.mlir.ir import Context


def make_context() -> Context:
    """Create a fresh MLIR context with the PTO dialect loaded."""
    ctx = Context()
    _pto_dialect.register_dialect(ctx, load=True)
    if _llvm_dialect is not None and hasattr(_llvm_dialect, "register_dialect"):
        _llvm_dialect.register_dialect(ctx, load=True)
    return ctx


__all__ = ["make_context"]
