# coding=utf-8
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Tracing-time helpers for coercing authored runtime values to MLIR index."""

from __future__ import annotations

from ._scalar_adaptation import coerce_runtime_index_value


def coerce_runtime_index(value, *, context: str):
    """Normalize one authored loop/slice bound to an MLIR index SSA value."""
    return coerce_runtime_index_value(value, context=context)


__all__ = [
    "coerce_runtime_index",
]
