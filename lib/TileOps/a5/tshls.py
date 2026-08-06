# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""PTODSL TileLib template for pto.tshls."""

from ptodsl import pto

from ._common import INT_DTYPES
from ._elementwise import register_scalar_binary


_DTYPES = [(dtype, "i16", dtype) for dtype in INT_DTYPES]


template_tshls = register_scalar_binary(
    op="pto.tshls",
    name="template_tshls",
    vector_op=pto.vshls,
    dtypes=_DTYPES,
)

template_tshls_1d = register_scalar_binary(
    op="pto.tshls",
    name="template_tshls_1d",
    vector_op=pto.vshls,
    dtypes=_DTYPES,
    traversal="1d",
)
