# coding=utf-8
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""PTODSL TileLib template for pto.tabs."""

from ptodsl import pto

from .._elementwise import register_unary


# The A5 pto.tabs verifier and ISA accept the signed integer widths next to
# the float types. Unsigned element types are not registered: the A5 ISA list
# for tabs has no unsigned entry.
_DTYPES = [
    ("f16", "f16"),
    ("f32", "f32"),
    ("i8", "i8"),
    ("i16", "i16"),
    ("i32", "i32"),
]


template_tabs = register_unary(
    op="pto.tabs",
    name="template_tabs",
    vector_op=pto.vabs,
    dtypes=_DTYPES,
)


template_tabs_1d = register_unary(
    op="pto.tabs",
    name="template_tabs_1d",
    vector_op=pto.vabs,
    dtypes=_DTYPES,
    traversal="1d",
)
