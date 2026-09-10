# coding=utf-8
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""PTODSL TileLib template for pto.tcolexpandmin."""

from ptodsl import pto

from ._expand_binary import NUMERIC_SIGNATURES, register_column_expand_binary


template_tcolexpandmin = register_column_expand_binary(
    op="pto.tcolexpandmin",
    name="template_tcolexpandmin",
    vector_op=pto.vmin,
    dtypes=NUMERIC_SIGNATURES,
)
