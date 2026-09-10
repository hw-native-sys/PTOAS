# coding=utf-8
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""PTODSL TileLib templates for ``pto.tdequant`` (ports A5 CCE ``TDeQuantImpl``).

``dst[r,c] = (float(src[r,c]) - offset[r,0]) * scale[r,0]``; one template per src
dtype -- i16 (single ``vcvt`` on an ``UNPK_B16`` load) and i8 (``UNPK_B8`` +
sign-extend + ``vcvt`` -> i32, then i32->f32 rnd=Z, like ``tcvt.py``).
"""

from ptodsl import pto
import ptodsl.tilelib as tilelib


def _dequant_layout(
    src_config, scale_config, offset_config, dst_config,
    operand_memory_spaces, scale_shape, offset_shape, **_,
):
    # scale/offset may be row- or col-major [rows,1]; the row broadcast is layout-agnostic.
    if not all(space in {"ub", "vec"} for space in operand_memory_spaces):
        return False
    if src_config.b_layout != "row_major" or src_config.s_layout != "none_box":
        return False
    if dst_config.b_layout != "row_major" or dst_config.s_layout != "none_box":
        return False
    for config, shape in ((scale_config, scale_shape), (offset_config, offset_shape)):
        if config.b_layout == "row_major":
            if config.s_layout != "none_box":
                return False
        elif (
            config.b_layout == "col_major"
            and len(shape) == 2
            and shape[1] == 1
            and config.s_layout in {"none_box", "row_major"}
        ):
            pass
        else:
            return False
    return True


def _dequant_shapes(
    src_valid_shape, scale_valid_shape, offset_valid_shape, dst_valid_shape, **_,
):
    return (
        len(src_valid_shape) == 2
        and len(dst_valid_shape) == 2
        and tuple(src_valid_shape) == tuple(dst_valid_shape)
        and len(scale_valid_shape) == 2
        and scale_valid_shape[0] == dst_valid_shape[0]
        # Per-row coefficients: exactly one column -- _broadcast_row reads only
        # lane 0, so a wider tile would be silently truncated.
        and scale_valid_shape[1] == 1
        and len(offset_valid_shape) == 2
        and offset_valid_shape[0] == dst_valid_shape[0]
        and offset_valid_shape[1] == 1
    )


_DEQUANT_CONSTRAINTS = [_dequant_layout, _dequant_shapes]


def _broadcast_row(tile, row, mask):
    # Broadcast the per-row value across lanes (cf. _expand_binary._emit_row_expand_body).
    return pto.vdup(pto.vlds(tile[row, :]), mask)


@tilelib.tile_template(
    op="pto.tdequant",
    target="a5",
    name="template_tdequant_i16",
    dtypes=[("i16", "f32", "f32", "f32")],
    iteration_axis="none",
    op_engine="vector",
    op_class="elementwise",
    constraints=_DEQUANT_CONSTRAINTS,
    id=0,
    loop_depth=2,
    is_post_update=False,
    tags=("dequant", "i16"),
)
def template_tdequant_i16(
    src: pto.Tile, scale: pto.Tile, offset: pto.Tile, dst: pto.Tile,
):
    valid_rows, valid_cols = dst.valid_shape
    lanes = pto.elements_per_vreg(pto.f32)
    src_full = pto.make_mask(pto.i16, pto.PAT.ALL)
    for row in range(0, valid_rows, 1):
        remained = valid_cols
        for col in range(0, valid_cols, lanes):
            mask, remained = pto.make_mask(pto.f32, remained)
            # i16 -> f32 in one shot (unpack-on-load + even-part convert).
            src_v = pto.vlds(src[row, col:], dist="UNPK_B16")
            value = pto.vcvt(src_v, pto.f32, src_full, part=pto.VcvtPartMode.EVEN)
            offset_b = _broadcast_row(offset, row, mask)
            scaled = pto.vmul(pto.vsub(value, offset_b, mask), _broadcast_row(scale, row, mask), mask)
            pto.vsts(scaled, dst[row, col:], mask)


@tilelib.tile_template(
    op="pto.tdequant",
    target="a5",
    name="template_tdequant_i8",
    dtypes=[("i8", "f32", "f32", "f32")],
    iteration_axis="none",
    op_engine="vector",
    op_class="elementwise",
    constraints=_DEQUANT_CONSTRAINTS,
    id=1,
    loop_depth=2,
    is_post_update=False,
    tags=("dequant", "i8"),
)
def template_tdequant_i8(
    src: pto.Tile, scale: pto.Tile, offset: pto.Tile, dst: pto.Tile,
):
    valid_rows, valid_cols = dst.valid_shape
    b8_mask = pto.make_mask(pto.ui8, pto.PAT.ALL)
    v_zero = pto.vbitcast(pto.vdup(pto.i8(0), b8_mask), pto.ui8)
    lanes_i16 = pto.elements_per_vreg(pto.i16)
    lanes_i32 = pto.elements_per_vreg(pto.i32)
    for row in range(0, valid_rows, 1):
        remained = valid_cols
        next_remained = valid_cols - lanes_i32
        for col in range(0, valid_cols, lanes_i16):
            # i8 -> i32 sign-extending interleave (mirrors template_tcvt_si8_to_i32).
            mask_b16_cur, remained = pto.make_mask(pto.i16, remained)
            mask_b16_next, next_remained = pto.make_mask(pto.i16, next_remained)
            mask_b32_cur = pto.punpack(mask_b16_cur, pto.PredicatePart.LOWER, to_type=pto.mask_b32)
            mask_b32_next = pto.punpack(mask_b16_next, pto.PredicatePart.LOWER, to_type=pto.mask_b32)
            vec_si8_0 = pto.vlds(src[row, col:], dist="UNPK_B8")
            vec_ui8_1, vec_ui8_2 = pto.vintlv(pto.vbitcast(vec_si8_0, pto.ui8), v_zero)
            i32_cur = pto.vcvt(pto.vbitcast(vec_ui8_1, pto.si8), pto.i32, b8_mask, part=pto.VcvtPartMode.P0)
            i32_next = pto.vcvt(pto.vbitcast(vec_ui8_2, pto.si8), pto.i32, b8_mask, part=pto.VcvtPartMode.P0)
            value_cur = pto.vcvt(i32_cur, pto.f32, mask_b32_cur, rnd=pto.VcvtRoundMode.Z)
            value_next = pto.vcvt(i32_next, pto.f32, mask_b32_next, rnd=pto.VcvtRoundMode.Z)
            scaled_cur = pto.vmul(
                pto.vsub(value_cur, _broadcast_row(offset, row, mask_b32_cur), mask_b32_cur),
                _broadcast_row(scale, row, mask_b32_cur),
                mask_b32_cur,
            )
            scaled_next = pto.vmul(
                pto.vsub(value_next, _broadcast_row(offset, row, mask_b32_next), mask_b32_next),
                _broadcast_row(scale, row, mask_b32_next),
                mask_b32_next,
            )
            pto.vsts(scaled_cur, dst[row, col:], mask_b32_cur)
            pto.vsts(scaled_next, dst[row, col + lanes_i32:], mask_b32_next)
