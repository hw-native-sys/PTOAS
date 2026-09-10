# coding=utf-8
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Rope kernel adapter for the kernel-test framework."""

from __future__ import annotations

from kernel_test.registry import OperatorSpec, make_operator_spec

from .backends import create_backend
from .spec import cycle_fields, list_cases, verify_case


def get_operator_spec() -> OperatorSpec:
    """Return the rope operator registration for the shared registry."""

    return make_operator_spec(
        name="rope",
        default_backend="cce",
        backend_names=("cce", "mi", "vmi"),
        create_backend=create_backend,
        list_cases=list_cases,
        verify=verify_case,
        cycle_fields=cycle_fields,
        summary="VF sim rope kernel adapter",
    )
