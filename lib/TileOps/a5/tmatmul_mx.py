# coding=utf-8
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""PTODSL TileLib templates for pto.tmatmul.mx variants."""

from ptodsl import pto
import ptodsl.tilelib as tilelib

from ._cube import MATMUL_MX_ACC_DTYPES, MATMUL_MX_BIAS_DTYPES, MATMUL_MX_DTYPES


@tilelib.tile_template(op="pto.tmatmul.mx", target="a5", name="template_tmatmul_mx",
                       dtypes=MATMUL_MX_DTYPES, iteration_axis="none",
                       op_engine="cube", op_class="other", id=0, loop_depth=1,
                       is_post_update=False, tags=("cube", "matmul", "mx"))
def template_tmatmul_mx(lhs: pto.Tile, lhs_scale: pto.Tile, rhs: pto.Tile, rhs_scale: pto.Tile, dst: pto.Tile):
    _ = lhs_scale, rhs_scale
    m, k = lhs.valid_shape
    _, n = rhs.valid_shape
    pto.mad_mx(lhs.as_ptr(), rhs.as_ptr(), dst.as_ptr(), m, n, k, disable_gemv=True, sat="sat")


@tilelib.tile_template(op="pto.tmatmul.mx.acc", target="a5", name="template_tmatmul_mx_acc",
                       dtypes=MATMUL_MX_ACC_DTYPES, iteration_axis="none",
                       op_engine="cube", op_class="other", id=0, loop_depth=1,
                       is_post_update=False, tags=("cube", "matmul", "mx", "acc"))
def template_tmatmul_mx_acc(acc_in: pto.Tile, lhs: pto.Tile, lhs_scale: pto.Tile,
                            rhs: pto.Tile, rhs_scale: pto.Tile, dst: pto.Tile):
    _ = acc_in, lhs_scale, rhs_scale
    m, k = lhs.valid_shape
    _, n = rhs.valid_shape
    pto.mad_mx_acc(lhs.as_ptr(), rhs.as_ptr(), dst.as_ptr(), m, n, k, disable_gemv=True, sat="sat")


@tilelib.tile_template(op="pto.tmatmul.mx.bias", target="a5", name="template_tmatmul_mx_bias",
                       dtypes=MATMUL_MX_BIAS_DTYPES, iteration_axis="none",
                       op_engine="cube", op_class="other", id=0, loop_depth=1,
                       is_post_update=False, tags=("cube", "matmul", "mx", "bias"))
def template_tmatmul_mx_bias(lhs: pto.Tile, lhs_scale: pto.Tile, rhs: pto.Tile,
                             rhs_scale: pto.Tile, bias: pto.Tile, dst: pto.Tile):
    _ = lhs_scale, rhs_scale
    m, k = lhs.valid_shape
    _, n = rhs.valid_shape
    pto.mad_mx_bias(lhs.as_ptr(), rhs.as_ptr(), dst.as_ptr(), bias.as_ptr(), m, n, k,
                    disable_gemv=True, sat="sat")
