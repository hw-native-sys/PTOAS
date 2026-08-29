# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""PTODSL TileLib templates for pto.tci."""

from ptodsl import pto
import ptodsl.tilelib as tilelib


@tilelib.tile_template(
    op="pto.tci",
    target="a5",
    name="template_tci",
    dtypes=[("i16", "i16"), ("ui16", "ui16"), ("i32", "i32"), ("ui32", "ui32")],
    iteration_axis="none",
    op_engine="other",
    op_class="other",
    constraints=[
        tilelib.check_memory_space("ub"),
        tilelib.check_layout("row_major"),
        tilelib.check_s_layout("none_box"),
        tilelib.require_valid_rows("dst", 1),
    ],
    id=0,
    loop_depth=1,
    is_post_update=False,
)
def template_tci(start, dst: pto.Tile):
    descending = pto.get_op_attr("descending", "false") == "true"
    dtype = dst.dtype
    cast_dtype = pto.i16 if str(dtype) in ("ui16",) else pto.i32
    valid_rows, valid_cols = dst.valid_shape
    ptr = dst.as_ptr()
    if descending:
        for col in range(0, valid_cols, 1):
            pto.store(pto.index_cast(cast_dtype, start - col), ptr, col)
    else:
        for col in range(0, valid_cols, 1):
            pto.store(pto.index_cast(cast_dtype, start + col), ptr, col)
