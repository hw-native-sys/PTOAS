# coding=utf-8
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""PTODSL TileLib templates for pto.tgatherb."""

from ptodsl import pto
import ptodsl.tilelib as tilelib
from ._common import NUMERIC_DTYPES


def gatherb_dtype_signatures(dtypes=NUMERIC_DTYPES):
    return [(dtype, "ui32", dtype) for dtype in dtypes]


@tilelib.tile_template(
    op="pto.tgatherb",
    target="a5",
    name="template_tgatherb",
    dtypes=gatherb_dtype_signatures(),
    iteration_axis="none",
    op_engine="vector",
    op_class="other",
    layouts=("row_major",),
    loop_depth=2,
    is_post_update=False
)
def template_tgatherb(src: pto.Tile, offset: pto.Tile, dst: pto.Tile):
    dtype = dst.element_type
    dtype_offset = offset.element_type
    src_ptr = src.as_ptr()
    valid_rows, valid_cols = dst.valid_shape
    _, off_valid_cols = offset.valid_shape
    lanes = pto.elements_per_vreg(dtype)
    blocks_per_vreg = 8
    block_size = lanes // blocks_per_vreg
    for row in range(0, valid_rows, 1):
        remained = valid_cols
        for off_col in range(0, off_valid_cols, blocks_per_vreg):
            offset_reg = pto.vlds(offset[row, off_col:])
            mask_dst, _ = pto.make_mask(dtype, remained)
            mask, remained = pto.make_mask(dtype_offset, remained)
            dst_reg = pto.vgatherb(src_ptr, offset_reg, mask)
            dst_col = off_col * block_size
            pto.vsts(dst_reg, dst[row, dst_col:], mask_dst)
    return
