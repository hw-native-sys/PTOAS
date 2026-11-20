#!/usr/bin/env python3
# coding=utf-8
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

# Issue #1612 system tests for the vector tcvt TileLib templates (8x32, rmode NONE).
# Covers the 4:1 unsigned integer narrowing (i32/ui32 -> ui8, packed with vselr) and
# the FP8 <-> F32 pairs.  The signed 4:1 pairs (i32/ui32 -> si8) have no A5 intrinsic
# and intentionally stay unsupported.
#
# Issue #1686 adds the f32 -> i16 saturation case: pto.tcvt defaults sat_mode to ON,
# so every out-of-range element must clamp to the int16 range after RINT instead of
# wrapping through the low 16 bits of the rounded i32 value.

from pathlib import Path
import sys

import numpy as np

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from common import auto_main, golden_output_case
from ptodsl import pto

ROWS, COLS = 8, 32
TOTAL = ROWS * COLS

INT_PROBE = np.array(
    [0, 1, 2, 3, 4, 5, 6, 7, 127, 128, 254, 255, 256, 300, 1000, 65535, -1, -2, -128, -1000],
    dtype=np.int32,
)

# Exactly representable in e4m3, so any rounding mode gives the same bits.
E4M3_VALUES = np.array([0.0, 0.25, 0.5, 1.0, 1.5, 2.0, 4.0, 448.0, -1.0, -4.0], dtype=np.float32)
E4M3_BITS = np.array([0x00, 0x28, 0x30, 0x38, 0x3C, 0x40, 0x48, 0x7E, 0xB8, 0xC8], dtype=np.uint8)


def _tile(value, dtype):
    return np.tile(value, (TOTAL + len(value) - 1) // len(value))[:TOTAL].reshape(ROWS, COLS).astype(dtype)


def _convert_kernel(name, src_dtype, dst_dtype):
    @pto.jit(name=name, target="a5")
    def _kernel(src_ptr: pto.ptr(src_dtype, "gm"), out_ptr: pto.ptr(dst_dtype, "gm")):
        src_view = pto.make_tensor_view(src_ptr, shape=[ROWS, COLS], strides=[COLS, 1])
        out_view = pto.make_tensor_view(out_ptr, shape=[ROWS, COLS], strides=[COLS, 1])
        src = pto.alloc_tile(shape=[ROWS, COLS], dtype=src_dtype, valid_shape=[ROWS, COLS])
        dst = pto.alloc_tile(shape=[ROWS, COLS], dtype=dst_dtype, valid_shape=[ROWS, COLS])
        pto.tile.load(src_view, src)
        pto.tile.cvt(src, dst, rmode=pto.RoundMode.NONE)
        pto.tile.store(dst, out_view)

    return _kernel


_i32_ui8 = _convert_kernel("cvt_i32_to_ui8_s8x32", pto.i32, pto.ui8)
_ui32_ui8 = _convert_kernel("cvt_ui32_to_ui8_s8x32", pto.ui32, pto.ui8)
_f32_f8e4m3 = _convert_kernel("cvt_f32_to_f8e4m3_s8x32", pto.f32, pto.f8e4m3)
_f8e4m3_f32 = _convert_kernel("cvt_f8e4m3_to_f32_s8x32", pto.f8e4m3, pto.f32)


def _int_inputs() -> list:
    return [_tile(INT_PROBE, np.int32)]


def _uint_inputs() -> list:
    return [_tile(np.abs(INT_PROBE), np.uint32)]


def _f32_inputs() -> list:
    return [_tile(E4M3_VALUES, np.float32)]


def _f8bits_inputs() -> list:
    return [_tile(E4M3_BITS, np.uint8)]


# f32 -> i16 saturation (issue #1686): a range wide enough that 48 of the 256
# elements fall outside int16, so a missing saturation is visible as a wrap.
I16_LOW, I16_HIGH = -32768, 32767
SAT_VALUES = np.linspace(-40000.0, 40000.0, TOTAL, dtype=np.float32).reshape(ROWS, COLS)


def _sat_kernel(name, sat_mode):
    @pto.jit(name=name, target="a5")
    def _kernel(src_ptr: pto.ptr(pto.f32, "gm"), out_ptr: pto.ptr(pto.i16, "gm")):
        src_view = pto.make_tensor_view(src_ptr, shape=[ROWS, COLS], strides=[COLS, 1])
        out_view = pto.make_tensor_view(out_ptr, shape=[ROWS, COLS], strides=[COLS, 1])
        src = pto.alloc_tile(shape=[ROWS, COLS], dtype=pto.f32, valid_shape=[ROWS, COLS])
        dst = pto.alloc_tile(shape=[ROWS, COLS], dtype=pto.i16, valid_shape=[ROWS, COLS])
        pto.tile.load(src_view, src)
        pto.tile.cvt(src, dst, rmode=pto.RoundMode.RINT, sat_mode=sat_mode)
        pto.tile.store(dst, out_view)

    return _kernel


_f32_i16_sat_on = _sat_kernel("cvt_f32_to_i16_s8x32_sat_on", "ON")
# sat_mode=OFF keeps the plain truncation, so the out-of-range half wraps.  This
# case exists to lock the sat_mode *attribute plumbing*: with the template
# defaulting to ON, an ON-only case would still pass if ExpandTileOp stopped
# forwarding the attribute.
_f32_i16_sat_off = _sat_kernel("cvt_f32_to_i16_s8x32_sat_off", "OFF")


def _sat_inputs() -> list:
    return [SAT_VALUES]


def _i16_sat_expected(src):
    return np.clip(np.rint(src), I16_LOW, I16_HIGH).astype(np.int16)


def _i16_nosat_expected(src):
    return np.rint(src).astype(np.int64).astype(np.int16)


# bf16 -> f4e2m1x2: two source values become one packed destination byte.
FP4_ROWS, FP4_SRC_COLS = 8, 64
FP4_DST_COLS = FP4_SRC_COLS // 2
FP4_E2M1_VALUES = np.array(
    [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0],
    dtype=np.float32,
)
# Keyed by bf16 bit pattern so that +0.0 and -0.0 stay distinct.
FP4_E2M1_CODES = {
    int(np.float32(value).view(np.uint32) >> 16): code
    for code, value in enumerate(FP4_E2M1_VALUES)
}


def _bf16_bits(values):
    return (np.asarray(values, dtype=np.float32).view(np.uint32) >> 16).astype(np.uint16)


def _fp4_expected(src_bits):
    codes = np.vectorize(
        lambda bits: FP4_E2M1_CODES[int(bits)],
        otypes=[np.uint8],
    )(src_bits.reshape(FP4_ROWS, FP4_SRC_COLS))
    return (codes[:, 0::2] | (codes[:, 1::2] << 4)).astype(np.uint8)


def _fp4_kernel(name):
    @pto.jit(name=name, target="a5")
    def _kernel(src_ptr: pto.ptr(pto.bf16, "gm"), out_ptr: pto.ptr(pto.f4e2m1x2, "gm")):
        src_view = pto.make_tensor_view(src_ptr, shape=[FP4_ROWS, FP4_SRC_COLS], strides=[FP4_SRC_COLS, 1])
        out_view = pto.make_tensor_view(out_ptr, shape=[FP4_ROWS, FP4_DST_COLS], strides=[FP4_DST_COLS, 1])
        src = pto.alloc_tile(shape=[FP4_ROWS, FP4_SRC_COLS], dtype=pto.bf16, valid_shape=[FP4_ROWS, FP4_SRC_COLS])
        dst = pto.alloc_tile(shape=[FP4_ROWS, FP4_DST_COLS], dtype=pto.f4e2m1x2, valid_shape=[FP4_ROWS, FP4_DST_COLS])
        pto.tile.load(src_view, src)
        pto.tile.cvt(src, dst, rmode=pto.RoundMode.NONE)
        pto.tile.store(dst, out_view)

    return _kernel


_bf16_f4e2m1x2 = _fp4_kernel("cvt_bf16_to_f4e2m1x2_s8x64")


def _fp4_inputs() -> list:
    count = FP4_ROWS * FP4_SRC_COLS
    values = np.tile(FP4_E2M1_VALUES, (count + len(FP4_E2M1_VALUES) - 1) // len(FP4_E2M1_VALUES))[:count]
    return [_bf16_bits(values).reshape(FP4_ROWS, FP4_SRC_COLS)]


CASES = [
    golden_output_case(
        "cvt_i32_to_ui8_s8x32",
        _i32_ui8,
        inputs=_int_inputs,
        expected=lambda src: np.clip(src, 0, 255).astype(np.uint8),
        rtol=0.0,
        atol=0.0,
    ),
    golden_output_case(
        "cvt_ui32_to_ui8_s8x32",
        _ui32_ui8,
        inputs=_uint_inputs,
        expected=lambda src: np.clip(src, 0, 255).astype(np.uint8),
        rtol=0.0,
        atol=0.0,
    ),
    golden_output_case(
        "cvt_f32_to_f8e4m3_s8x32",
        _f32_f8e4m3,
        inputs=_f32_inputs,
        expected=lambda src: _tile(E4M3_BITS, np.uint8),
        output_dtype=np.uint8,
        rtol=0.0,
        atol=0.0,
    ),
    golden_output_case(
        "cvt_f8e4m3_to_f32_s8x32",
        _f8e4m3_f32,
        inputs=_f8bits_inputs,
        expected=lambda src: _tile(E4M3_VALUES, np.float32),
        rtol=0.0,
        atol=0.0,
    ),
    golden_output_case(
        "cvt_bf16_to_f4e2m1x2_s8x64",
        _bf16_f4e2m1x2,
        inputs=_fp4_inputs,
        expected=lambda src: _fp4_expected(src),
        output_shape=(FP4_ROWS, FP4_DST_COLS),
        output_dtype=np.uint8,
        rtol=0.0,
        atol=0.0,
    ),
    golden_output_case(
        "cvt_f32_to_i16_s8x32_sat_on",
        _f32_i16_sat_on,
        inputs=_sat_inputs,
        expected=_i16_sat_expected,
        rtol=0.0,
        atol=0.0,
    ),
    golden_output_case(
        "cvt_f32_to_i16_s8x32_sat_off",
        _f32_i16_sat_off,
        inputs=_sat_inputs,
        expected=_i16_nosat_expected,
        rtol=0.0,
        atol=0.0,
    ),
]

auto_main(globals())
