# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""PTODSL TileLib template for pto.tsqrt — default precision only."""

from ptodsl import pto

from ._elementwise import register_unary


_DTYPES = [
    ("f16", "f16"),
    ("f32", "f32"),
]


template_tsqrt = register_unary(
    op="pto.tsqrt",
    name="template_tsqrt",
    vector_op=pto.vsqrt,
    dtypes=_DTYPES,
)


template_tsqrt_1d = register_unary(
    op="pto.tsqrt",
    name="template_tsqrt_1d",
    vector_op=pto.vsqrt,
    dtypes=_DTYPES,
    traversal="1d",
)
