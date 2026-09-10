# coding=utf-8
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Shared error type and sentinel for the PTODSL AST rewrite."""

from __future__ import annotations


class PTODSLAstRewriteError(SyntaxError):
    """Raised when AST rewrite sees unsupported Python control flow."""


_MISSING_GLOBAL = object()


__all__ = [
    "PTODSLAstRewriteError",
    "_MISSING_GLOBAL",
]
