# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""PTODSL TileLib template for pto.texpands."""

from ._elementwise import register_scalar_fill


_DTYPES = [
    ("i8", "i8"),
    ("i16", "i16"),
    ("i32", "i32"),
    ("f16", "f16"),
    ("bf16", "bf16"),
    ("f32", "f32"),
]


template_texpands = register_scalar_fill(
    op="pto.texpands",
    name="template_texpands",
    dtypes=_DTYPES,
)

template_texpands_1d = register_scalar_fill(
    op="pto.texpands",
    name="template_texpands_1d",
    dtypes=_DTYPES,
    traversal="1d",
)
