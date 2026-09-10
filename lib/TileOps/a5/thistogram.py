# coding=utf-8
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""PTODSL TileLib template for pto.thistogram.

Uses a factory-function pattern (like ``_col_arg.register_col_arg_template``)
to share the histogram body across byte variants via closure variables,
avoiding code duplication while keeping the framework unchanged.

Factory calls register 6 templates total:
  - u16: byte 0 (LSB) and byte 1 (MSB)
  - u32: byte 0..3 with cascaded radix filtering
"""

from ptodsl import pto
import ptodsl.tilelib as tilelib
from ptodsl._ops import _elements_per_vreg, vreg_type as _vreg_type, _resolve as _resolve_type
from ptodsl._types import ui8 as _ui8_dtype, ui16 as _ui16_dtype


def _u8_vreg_type():
    return _resolve_type(_vreg_type(_elements_per_vreg(_resolve_type(_ui8_dtype)), _resolve_type(_ui8_dtype)))

def _u16_vreg_type():
    return _resolve_type(_vreg_type(_elements_per_vreg(_resolve_type(_ui16_dtype)), _resolve_type(_ui16_dtype)))


_HIST_U16_DTYPES = [("ui16", "ui8", "ui32")]
_HIST_U32_DTYPES = [("ui32", "ui8", "ui32")]
_ELEM_PER_REPEAT_B8 = 256
_MIN_SRC_BYTES = 256


# ── constraint predicates ────────────────────────────────────────────────────

def _is_byte(byte_val):
    def _check(byte=1, **_):
        return int(byte) == byte_val
    return _check

def _hist_layout(src_config, dst_config, **_):
    return src_config.b_layout == "row_major" and src_config.s_layout == "none_box" and \
           dst_config.b_layout == "row_major" and dst_config.s_layout == "none_box"

def _hist_mem_space(operand_memory_spaces, **_):
    return all(ms == "ub" for ms in operand_memory_spaces)

def _idx_col_major(idx_config, **_):
    return idx_config.b_layout == "col_major" and idx_config.s_layout == "none_box"

def _idx_row_major(idx_config, **_):
    return idx_config.b_layout == "row_major" and idx_config.s_layout == "none_box"

def _src_min_cols(src_dtype, src_cols, **_):
    _BYTEWIDTH = {"ui16": 2, "ui32": 4}
    return src_cols * _BYTEWIDTH.get(src_dtype, 1) >= _MIN_SRC_BYTES

# ── shared computation helpers ───────────────────────────────────────────────

def _histogram_core(vb8_src, elem_mask,
                    vb16_bins_low, vb16_bins_high,
                    vb32_bins_low_even, vb32_bins_low_odd,
                    vb32_bins_high_even, vb32_bins_high_odd,
                    preg_all_b16, preg_all_b32):
    vb16_bins_low = pto.chistv2(vb16_bins_low, vb8_src, elem_mask, pto.i32(0))
    vb16_bins_high = pto.chistv2(vb16_bins_high, vb8_src, elem_mask, pto.i32(1))
    vb32_bins_even = pto.vcvt(vb16_bins_low, pto.ui32, preg_all_b16, part=pto.VcvtPartMode.EVEN)
    vb32_bins_odd = pto.vcvt(vb16_bins_low, pto.ui32, preg_all_b16, part=pto.VcvtPartMode.ODD)
    vb32_bins_low_even = pto.vadd(vb32_bins_low_even, vb32_bins_even, preg_all_b32)
    vb32_bins_low_odd = pto.vadd(vb32_bins_low_odd, vb32_bins_odd, preg_all_b32)
    vb32_bins_even = pto.vcvt(vb16_bins_high, pto.ui32, preg_all_b16, part=pto.VcvtPartMode.EVEN)
    vb32_bins_odd = pto.vcvt(vb16_bins_high, pto.ui32, preg_all_b16, part=pto.VcvtPartMode.ODD)
    vb32_bins_high_even = pto.vadd(vb32_bins_high_even, vb32_bins_even, preg_all_b32)
    vb32_bins_high_odd = pto.vadd(vb32_bins_high_odd, vb32_bins_odd, preg_all_b32)
    vb16_bins_low = pto.vbr(pto.ui16(0))
    vb16_bins_high = pto.vbr(pto.ui16(0))
    return (vb16_bins_low, vb16_bins_high,
            vb32_bins_low_even, vb32_bins_low_odd,
            vb32_bins_high_even, vb32_bins_high_odd)

def _store_histogram(vb32_bins_low_even, vb32_bins_low_odd,
                     vb32_bins_high_even, vb32_bins_high_odd,
                     dst_ptr, row, preg_all_b32):
    pto.vstsx2(vb32_bins_low_even, vb32_bins_low_odd, dst_ptr, row * 256,
               pto.InterleaveDist.INTLV_B32, preg_all_b32)
    pto.vstsx2(vb32_bins_high_even, vb32_bins_high_odd, dst_ptr, row * 256 + 128,
               pto.InterleaveDist.INTLV_B32, preg_all_b32)

def _deintlv_u32_bytes(src_ptr, offset):
    """Extract four byte planes from ui32 data at (src_ptr + offset)."""
    src_addr = pto.addptr(src_ptr, offset)
    chunk0_lo, chunk0_hi = pto.vldsx2(src_addr, 0, pto.DeinterleaveDist.DINTLV_B16,
                                      result_vreg_type=_u16_vreg_type())
    chunk1_addr = pto.addptr(src_addr, 128)
    chunk1_lo, chunk1_hi = pto.vldsx2(chunk1_addr, 0, pto.DeinterleaveDist.DINTLV_B16,
                                      result_vreg_type=_u16_vreg_type())
    chunk0_lo_u8 = pto.vbitcast(chunk0_lo, pto.ui8)
    chunk1_lo_u8 = pto.vbitcast(chunk1_lo, pto.ui8)
    chunk0_hi_u8 = pto.vbitcast(chunk0_hi, pto.ui8)
    chunk1_hi_u8 = pto.vbitcast(chunk1_hi, pto.ui8)
    byte0, byte1 = pto.vdintlv(chunk0_lo_u8, chunk1_lo_u8)
    byte2, byte3 = pto.vdintlv(chunk0_hi_u8, chunk1_hi_u8)
    return byte0, byte1, byte2, byte3

def _init_bins():
    """Create zero-initialised accumulator vectors for one histogram row."""
    return (
        pto.vbr(pto.ui32(0)),
        pto.vbr(pto.ui32(0)),
        pto.vbr(pto.ui32(0)),
        pto.vbr(pto.ui32(0)),
        pto.vbr(pto.ui16(0)),
        pto.vbr(pto.ui16(0)),
    )


# ── u16 factory ──────────────────────────────────────────────────────────────

def _make_u16_template(byte_val, template_id):
    """Register a thistogram template for ui16 source.

    byte_val=1 (MSB): histogram the high byte, no filtering.
    byte_val=0 (LSB): histogram the low byte, filtered by MSB == idx.
    """
    is_msb = (byte_val == 1)
    name = f"template_thistogram_u16_{'msb' if is_msb else 'lsb'}"

    @tilelib.tile_template(
        op="pto.thistogram", target="a5", name=name,
        dtypes=_HIST_U16_DTYPES, iteration_axis="none", op_engine="vector",
        op_class="other",
        constraints=[_is_byte(byte_val), _src_min_cols, _hist_layout, _hist_mem_space, _idx_col_major],
        loop_depth=2, is_post_update=False, id=template_id,
    )
    def template(src: pto.Tile, idx: pto.Tile, dst: pto.Tile):
        valid_rows, valid_cols = src.valid_shape
        src_cols = src.shape[1]
        src_ptr = src.as_ptr()
        dst_ptr = dst.as_ptr()
        idx_ptr = idx.as_ptr()
        preg_all_b16 = pto.pset_b16(pto.PAT.ALL)
        preg_all_b32 = pto.pset_b32(pto.PAT.ALL)
        for row in range(valid_rows):
            remained = valid_cols
            (vb32_bins_low_even, vb32_bins_low_odd,
             vb32_bins_high_even, vb32_bins_high_odd,
             vb16_bins_low, vb16_bins_high) = _init_bins()
            vb8_idx = pto.vbr(pto.ui8(0))
            if not is_msb:
                vb8_idx = pto.vlds(idx_ptr, row, dist="BRC_B8")
            for col in range(0, valid_cols, _ELEM_PER_REPEAT_B8):
                elem_mask, remained = pto.make_mask(pto.ui8, remained)
                src_addr = pto.addptr(src_ptr, row * src_cols + col)
                vb8_lsb, vb8_msb = pto.vldsx2(src_addr, 0, pto.DeinterleaveDist.DINTLV_B8,
                                              result_vreg_type=_u8_vreg_type())
                if is_msb:
                    vb16_bins_low, vb16_bins_high, \
                        vb32_bins_low_even, vb32_bins_low_odd, \
                        vb32_bins_high_even, vb32_bins_high_odd = \
                        _histogram_core(vb8_msb, elem_mask, vb16_bins_low, vb16_bins_high,
                                        vb32_bins_low_even, vb32_bins_low_odd,
                                        vb32_bins_high_even, vb32_bins_high_odd,
                                        preg_all_b16, preg_all_b32)
                else:
                    filt_mask = pto.vcmp(vb8_msb, vb8_idx, elem_mask, pto.CmpMode.EQ)
                    vb16_bins_low, vb16_bins_high, \
                        vb32_bins_low_even, vb32_bins_low_odd, \
                        vb32_bins_high_even, vb32_bins_high_odd = \
                        _histogram_core(vb8_lsb, filt_mask, vb16_bins_low, vb16_bins_high,
                                        vb32_bins_low_even, vb32_bins_low_odd,
                                        vb32_bins_high_even, vb32_bins_high_odd,
                                        preg_all_b16, preg_all_b32)
            _store_histogram(vb32_bins_low_even, vb32_bins_low_odd,
                             vb32_bins_high_even, vb32_bins_high_odd,
                             dst_ptr, row, preg_all_b32)
    return template


# ── u32 factory ──────────────────────────────────────────────────────────────

def _make_u32_template(byte_val, template_id):
    """Register a thistogram template for ui32 source.

    Radix-sort pass selection:
      byte=3 (MSB): histogram byte3, no filtering.
      byte=2:       histogram byte2, filtered by byte3 == idx row 0.
      byte=1:       histogram byte1, filtered by byte3==idx0 AND byte2==idx1.
      byte=0 (LSB): histogram byte0, filtered by byte3==idx0, byte2==idx1, byte1==idx2.
    """
    num_filters = 3 - byte_val
    name = f"template_thistogram_u32_byte{byte_val}"

    idx_layout_con = [] if byte_val == 3 else [_idx_row_major]

    @tilelib.tile_template(
        op="pto.thistogram", target="a5", name=name,
        dtypes=_HIST_U32_DTYPES, iteration_axis="none", op_engine="vector",
        op_class="other",
        constraints=[_is_byte(byte_val), _src_min_cols, _hist_layout, _hist_mem_space] + idx_layout_con,
        loop_depth=2, is_post_update=False, id=template_id,
    )
    def template(src: pto.Tile, idx: pto.Tile, dst: pto.Tile):
        valid_rows, valid_cols = src.valid_shape
        src_cols = src.shape[1]
        src_ptr = src.as_ptr()
        dst_ptr = dst.as_ptr()
        idx_ptr = idx.as_ptr()
        idx_cols = idx.shape[1]
        preg_all_b16 = pto.pset_b16(pto.PAT.ALL)
        preg_all_b32 = pto.pset_b32(pto.PAT.ALL)
        for row in range(valid_rows):
            remained = valid_cols
            (vb32_bins_low_even, vb32_bins_low_odd,
             vb32_bins_high_even, vb32_bins_high_odd,
             vb16_bins_low, vb16_bins_high) = _init_bins()
            vb8_idx0 = pto.vbr(pto.ui8(0))
            vb8_idx1 = pto.vbr(pto.ui8(0))
            vb8_idx2 = pto.vbr(pto.ui8(0))
            if num_filters > 0:
                vb8_idx0 = pto.vlds(idx_ptr, 0, dist="BRC_B8")
            if num_filters > 1:
                vb8_idx1 = pto.vlds(idx_ptr, idx_cols, dist="BRC_B8")
            if num_filters > 2:
                vb8_idx2 = pto.vlds(idx_ptr, 2 * idx_cols, dist="BRC_B8")
            for col in range(0, valid_cols, _ELEM_PER_REPEAT_B8):
                elem_mask, remained = pto.make_mask(pto.ui8, remained)
                byte0, byte1, byte2, byte3 = _deintlv_u32_bytes(src_ptr, row * src_cols + col)
                if num_filters == 0:
                    filt_mask = elem_mask
                else:
                    filt_mask = pto.vcmp(byte3, vb8_idx0, elem_mask, pto.CmpMode.EQ)
                    if num_filters > 1:
                        filt_mask = pto.vcmp(byte2, vb8_idx1, filt_mask, pto.CmpMode.EQ)
                    if num_filters > 2:
                        filt_mask = pto.vcmp(byte1, vb8_idx2, filt_mask, pto.CmpMode.EQ)
                hist_bytes = (byte0, byte1, byte2, byte3)
                vb16_bins_low, vb16_bins_high, \
                    vb32_bins_low_even, vb32_bins_low_odd, \
                    vb32_bins_high_even, vb32_bins_high_odd = \
                    _histogram_core(hist_bytes[byte_val], filt_mask, vb16_bins_low, vb16_bins_high,
                                    vb32_bins_low_even, vb32_bins_low_odd,
                                    vb32_bins_high_even, vb32_bins_high_odd,
                                    preg_all_b16, preg_all_b32)
            _store_histogram(vb32_bins_low_even, vb32_bins_low_odd,
                             vb32_bins_high_even, vb32_bins_high_odd,
                             dst_ptr, row, preg_all_b32)
    return template


# ── instantiate templates ────────────────────────────────────────────────────

template_thistogram_u16_msb = _make_u16_template(1, 0)
template_thistogram_u16_lsb = _make_u16_template(0, 1)
template_thistogram_u32_byte3 = _make_u32_template(3, 2)
template_thistogram_u32_byte2 = _make_u32_template(2, 3)
template_thistogram_u32_byte1 = _make_u32_template(1, 4)
template_thistogram_u32_byte0 = _make_u32_template(0, 5)
