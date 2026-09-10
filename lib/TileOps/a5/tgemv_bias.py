# coding=utf-8
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""PTODSL TileLib template for pto.tgemv.bias."""

from ptodsl import pto
import ptodsl.tilelib as tilelib

from ._cube import MATMUL_BIAS_DTYPES


@tilelib.tile_template(op="pto.tgemv.bias", target="a5", name="template_tgemv_bias",
                       dtypes=MATMUL_BIAS_DTYPES, iteration_axis="none",
                       op_engine="cube", op_class="other", id=0, loop_depth=1,
                       is_post_update=False, tags=("cube", "gemv", "bias"))
def template_tgemv_bias(lhs: pto.Tile, rhs: pto.Tile, bias: pto.Tile, dst: pto.Tile):
    _, k = lhs.valid_shape
    _, n = rhs.valid_shape
    pto.mad_bias(lhs.as_ptr(), rhs.as_ptr(), dst.as_ptr(), bias.as_ptr(), 1, n, k)
