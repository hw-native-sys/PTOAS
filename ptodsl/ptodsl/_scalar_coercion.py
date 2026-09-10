# coding=utf-8
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Shared authored scalar type-adaptation helpers for PTODSL surface lowering."""

from __future__ import annotations

from ._scalar_adaptation import (
    coerce_scalar_value_to_type,
    materialize_scalar_literal,
)
from ._surface_values import unwrap_surface_value


def coerce_scalar_to_type(value, target_type, *, context: str):
    """Normalize one authored scalar value/literal to *target_type*."""
    raw_value = unwrap_surface_value(value)
    return coerce_scalar_value_to_type(raw_value, target_type, context=context)


__all__ = [
    "coerce_scalar_to_type",
    "materialize_scalar_literal",
]
