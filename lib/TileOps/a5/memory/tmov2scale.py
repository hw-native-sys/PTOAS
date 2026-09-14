# coding=utf-8
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""PTODSL TileLib template for pto.tmov MAT-to-SCALING."""

from ptodsl import pto
import ptodsl.tilelib as tilelib


@tilelib.tile_template(
    op="pto.tmov",
    target="a5",
    name="template_tmov_m2s",
    dtypes=[("f32", "f32")],
    iteration_axis="none",
    op_engine="other",
    op_class="movement",
    memory_spaces=("mat", "scaling"),
    id=4,
    loop_depth=1,
    is_post_update=False,
    tags=("move", "mat", "scaling"),
)
def template_tmov_m2s(src: pto.Tile, dst: pto.Tile):
    rows, cols = src.valid_shape
    pto.mte_l1_fb(src.as_ptr(), dst.as_ptr(), cols, nburst=(rows, 0, 0))
