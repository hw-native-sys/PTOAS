# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""PTODSL TileLib template for pto.tmax."""

from ptodsl import pto

from ._common import same_dtype_signatures
from ._elementwise import register_binary


def _vmax(lhs, rhs, mask):
    return pto.vmax(lhs, rhs, mask)


_DTYPES = same_dtype_signatures(3)


template_tmax = register_binary(
    op="pto.tmax",
    name="template_tmax",
    vector_op=_vmax,
    dtypes=_DTYPES,
)


template_tmax_1d = register_binary(
    op="pto.tmax",
    name="template_tmax_1d",
    vector_op=_vmax,
    dtypes=_DTYPES,
    traversal="1d",
)
