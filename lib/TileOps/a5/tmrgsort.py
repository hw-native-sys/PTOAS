# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""PTODSL TileLib templates for ``pto.tmrgsort``."""

from ptodsl import pto
import ptodsl.tilelib as tilelib


STRUCT_SIZE = 8
STRUCT_SIZE_SHIFT = 3
BLOCK_NUM = 4


def _structures(valid_cols, dtype):
    if pto.bytewidth(dtype) == 4:
        return valid_cols // 2
    return valid_cols // 4


def _as_i64(value):
    return pto.add(value, pto.i64(0))


def _pack_count(*structures):
    count = _as_i64(structures[0])
    for shift, structure in zip((16, 32, 48), structures[1:]):
        count = count | pto.mul(_as_i64(structure), 1 << shift)
    return count


def _copy_tmp_to_dst(tmp, dst):
    dtype = dst.dtype
    valid_cols = dst.valid_shape[1]
    lanes = pto.elements_per_vreg(dtype)
    col_loop = pto.for_(0, valid_cols, step=lanes).carry(remained=valid_cols)
    with col_loop:
        col = col_loop.iv
        mask, remained = pto.make_mask(dtype, col_loop.remained)
        data = pto.vlds(tmp[0, col:])
        pto.vsts(data, dst[0, col:], mask)
        col_loop.update(remained=remained)


@tilelib.tile_template(
    op="pto.tmrgsort",
    target="a5",
    name="template_tmrgsort_single_list",
    dtypes=[("f32", "i32", "f32"), ("f16", "i32", "f16")],
    iteration_axis="none",
    op_engine="vector",
    op_class="other",
    layouts=("row_major",),
    id=0,
    loop_depth=0,
    is_post_update=False,
    tags=("sort", "merge", "single-list"),
)
def template_tmrgsort_single_list(src: pto.Tile, block_len: pto.i32, dst: pto.Tile):
    num_structures = block_len * pto.bytewidth(src.dtype) // STRUCT_SIZE
    repeat_times = src.valid_shape[1] // (block_len * BLOCK_NUM)
    repeat_times_i64 = _as_i64(repeat_times)
    count = _pack_count(num_structures, num_structures, num_structures, num_structures)
    offset = num_structures * STRUCT_SIZE // pto.bytewidth(dst.dtype)
    config = repeat_times_i64
    config = config | pto.i64(0b1111 << 8)
    src_ptr = src.as_ptr()
    pto.vmrgsort4(
        dst.as_ptr(),
        src_ptr,
        pto.addptr(src_ptr, offset),
        pto.addptr(src_ptr, offset * 2),
        pto.addptr(src_ptr, offset * 3),
        count,
        config,
    )


@tilelib.tile_template(
    op="pto.tmrgsort",
    target="a5",
    name="template_tmrgsort_multi_list2",
    dtypes=[("f32", "f32", "f32", "f32", "i16"), ("f16", "f16", "f16", "f16", "i16")],
    iteration_axis="none",
    op_engine="vector",
    op_class="other",
    layouts=("row_major",),
    id=1,
    loop_depth=1,
    is_post_update=False,
    tags=("sort", "merge", "multi-list"),
)
def template_tmrgsort_multi_list2(src0: pto.Tile, src1: pto.Tile, tmp: pto.Tile, dst: pto.Tile, ex_vec):
    _ = ex_vec
    src0_structures = _structures(src0.valid_shape[1], dst.dtype)
    src1_structures = _structures(src1.valid_shape[1], dst.dtype)
    count = _pack_count(src0_structures, src1_structures)
    exhausted = int(pto.get_op_attr("exhausted", "0"))
    config = pto.i64(1 | (0b0011 << 8) | (exhausted << 12))
    pto.vmrgsort4(tmp.as_ptr(), src0.as_ptr(), src1.as_ptr(), src0.as_ptr(), src0.as_ptr(), count, config)
    _copy_tmp_to_dst(tmp, dst)


@tilelib.tile_template(
    op="pto.tmrgsort",
    target="a5",
    name="template_tmrgsort_multi_list3",
    dtypes=[("f32", "f32", "f32", "f32", "f32", "i16"), ("f16", "f16", "f16", "f16", "f16", "i16")],
    iteration_axis="none",
    op_engine="vector",
    op_class="other",
    layouts=("row_major",),
    id=2,
    loop_depth=1,
    is_post_update=False,
    tags=("sort", "merge", "multi-list"),
)
def template_tmrgsort_multi_list3(
    src0: pto.Tile,
    src1: pto.Tile,
    src2: pto.Tile,
    tmp: pto.Tile,
    dst: pto.Tile,
    ex_vec,
):
    _ = ex_vec
    src0_structures = _structures(src0.valid_shape[1], dst.dtype)
    src1_structures = _structures(src1.valid_shape[1], dst.dtype)
    src2_structures = _structures(src2.valid_shape[1], dst.dtype)
    count = _pack_count(src0_structures, src1_structures, src2_structures)
    exhausted = int(pto.get_op_attr("exhausted", "0"))
    config = pto.i64(1 | (0b0111 << 8) | (exhausted << 12))
    pto.vmrgsort4(tmp.as_ptr(), src0.as_ptr(), src1.as_ptr(), src2.as_ptr(), src0.as_ptr(), count, config)
    _copy_tmp_to_dst(tmp, dst)


@tilelib.tile_template(
    op="pto.tmrgsort",
    target="a5",
    name="template_tmrgsort_multi_list4",
    dtypes=[
        ("f32", "f32", "f32", "f32", "f32", "f32", "i16"),
        ("f16", "f16", "f16", "f16", "f16", "f16", "i16"),
    ],
    iteration_axis="none",
    op_engine="vector",
    op_class="other",
    layouts=("row_major",),
    id=3,
    loop_depth=1,
    is_post_update=False,
    tags=("sort", "merge", "multi-list"),
)
def template_tmrgsort_multi_list4(
    src0: pto.Tile,
    src1: pto.Tile,
    src2: pto.Tile,
    src3: pto.Tile,
    tmp: pto.Tile,
    dst: pto.Tile,
    ex_vec,
):
    _ = ex_vec
    src0_structures = _structures(src0.valid_shape[1], dst.dtype)
    src1_structures = _structures(src1.valid_shape[1], dst.dtype)
    src2_structures = _structures(src2.valid_shape[1], dst.dtype)
    src3_structures = _structures(src3.valid_shape[1], dst.dtype)
    count = _pack_count(src0_structures, src1_structures, src2_structures, src3_structures)
    exhausted = int(pto.get_op_attr("exhausted", "0"))
    config = pto.i64(1 | (0b1111 << 8) | (exhausted << 12))
    pto.vmrgsort4(tmp.as_ptr(), src0.as_ptr(), src1.as_ptr(), src2.as_ptr(), src3.as_ptr(), count, config)
    _copy_tmp_to_dst(tmp, dst)
