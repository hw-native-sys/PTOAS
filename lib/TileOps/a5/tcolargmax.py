# coding=utf-8
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""PTODSL TileLib template for ``pto.tcolargmax``."""

from ptodsl import pto

from ._col_arg import register_col_arg_template


template_tcolargmax_f32_to_i32 = register_col_arg_template(
    op="pto.tcolargmax",
    name="template_tcolargmax_f32_to_i32",
    cmp_mode="gt",
    reduce_op=pto.vmax,
)
