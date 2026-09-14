# coding=utf-8
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""PTODSL TileLib templates for pto.tmov ACC-to-MAT with fixpipe parameters."""

from ptodsl import pto
import ptodsl.tilelib as tilelib


@tilelib.tile_template(
    op="pto.tmov",
    target="a5",
    name="template_tmov_fp_f32_f16",
    dtypes=[("f32", "f16", "f32")],
    iteration_axis="none",
    op_engine="other",
    op_class="movement",
    memory_spaces=("acc", "mat", "scaling"),
    id=6,
    loop_depth=1,
    is_post_update=False,
    tags=("move", "acc", "mat", "fixpipe"),
)
def template_tmov_fp_f32_f16(src: pto.Tile, dst: pto.Tile, fp: pto.Tile):
    m, n = dst.valid_shape
    pto.mte_l0c_l1(src.as_ptr(), dst.as_ptr(), m, n, src.shape[0], dst.shape[1],
                   pre_quant=(fp.as_ptr(), "qf322f16_pre_vec"))


@tilelib.tile_template(
    op="pto.tmov",
    target="a5",
    name="template_tmov_fp_f32_bf16",
    dtypes=[("f32", "bf16", "f32")],
    iteration_axis="none",
    op_engine="other",
    op_class="movement",
    memory_spaces=("acc", "mat", "scaling"),
    id=7,
    loop_depth=1,
    is_post_update=False,
    tags=("move", "acc", "mat", "fixpipe"),
)
def template_tmov_fp_f32_bf16(src: pto.Tile, dst: pto.Tile, fp: pto.Tile):
    m, n = dst.valid_shape
    pto.mte_l0c_l1(src.as_ptr(), dst.as_ptr(), m, n, src.shape[0], dst.shape[1],
                   pre_quant=(fp.as_ptr(), "qf322bf16_pre_vec"))
