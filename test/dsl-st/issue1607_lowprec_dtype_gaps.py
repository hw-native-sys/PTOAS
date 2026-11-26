#!/usr/bin/env python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Real-device numeric checks for the issue #1607 low-precision signatures.

Two cases exercise the paths that the design document marks as silent-failure
risks:

- ``tcmps`` bf16 uses the 2-byte packed predicate path (``psts`` with the
  packed distribution and a b16 mask). The golden checks the resulting mask
  byte image, not just that the op compiled.
- ``tsels`` ui16 must read its predicate through ``PredicateDist.US``. The
  golden checks the selected ui16 values; taking the NORM path still compiles
  but computes wrong values.

Run on an A5 NPU host with:

    python test/dsl-st/issue1607_lowprec_dtype_gaps.py
"""

import numpy as np

from common import auto_main, golden_output_case
from ptodsl import pto


ROWS = 4
COLS = 128
# The predicate destination is deliberately padded to a 128-byte row stride
# while only COLS // 8 = 16 bytes carry predicates. Both tcmps and tsels must
# use this same row stride; the 1D tcmps writer must reject a padded row so it
# cannot lay the rows out contiguously.
MASK_COLS = 128
MASK_BYTES = COLS // 8
TSELS_SCALAR = 1234
MASK_PADDING = 0xA5
FMOD_NAN = float("nan")


@pto.jit(name="issue1607_tcmps_bf16_mask", kernel_kind="vector", target="a5", backend="vpto")
def _tcmps_bf16_kernel(
    src_ptr: pto.ptr(pto.bf16, "gm"),
    dst_ptr: pto.ptr(pto.ui8, "gm"),
):
    src_view = pto.make_tensor_view(src_ptr, shape=[ROWS, COLS], strides=[COLS, 1])
    dst_view = pto.make_tensor_view(dst_ptr, shape=[ROWS, MASK_COLS], strides=[MASK_COLS, 1])
    src = pto.alloc_tile(
        shape=[ROWS, COLS], dtype=pto.bf16, valid_shape=[ROWS, COLS], blayout="RowMajor"
    )
    dst = pto.alloc_tile(
        shape=[ROWS, MASK_COLS], dtype=pto.ui8, valid_shape=[ROWS, MASK_COLS], blayout="RowMajor"
    )
    pto.tile.load(src_view, src)
    pto.tile.expands(MASK_PADDING, dst)
    pto.tile.cmps(src, 0.0, dst, cmp_mode="lt")
    pto.tile.store(dst, dst_view)


@pto.jit(name="issue1607_tsels_ui16_us", kernel_kind="vector", target="a5", backend="vpto")
def _tsels_ui16_kernel(
    mask_ptr: pto.ptr(pto.i8, "gm"),
    src_ptr: pto.ptr(pto.ui16, "gm"),
    dst_ptr: pto.ptr(pto.ui16, "gm"),
):
    mask_view = pto.make_tensor_view(mask_ptr, shape=[ROWS, MASK_COLS], strides=[MASK_COLS, 1])
    src_view = pto.make_tensor_view(src_ptr, shape=[ROWS, COLS], strides=[COLS, 1])
    dst_view = pto.make_tensor_view(dst_ptr, shape=[ROWS, COLS], strides=[COLS, 1])
    mask = pto.alloc_tile(
        shape=[ROWS, MASK_COLS], dtype=pto.i8, valid_shape=[ROWS, MASK_COLS], blayout="RowMajor"
    )
    src = pto.alloc_tile(
        shape=[ROWS, COLS], dtype=pto.ui16, valid_shape=[ROWS, COLS], blayout="RowMajor"
    )
    dst = pto.alloc_tile(
        shape=[ROWS, COLS], dtype=pto.ui16, valid_shape=[ROWS, COLS], blayout="RowMajor"
    )
    pto.tile.load(mask_view, mask)
    pto.tile.load(src_view, src)
    pto.tile.sels(mask, src, TSELS_SCALAR, dst)
    pto.tile.store(dst, dst_view)


@pto.jit(name="issue1607_tcmps_mask_to_tsels", kernel_kind="vector", target="a5", backend="vpto")
def _tcmps_to_tsels_kernel(
    cmp_ptr: pto.ptr(pto.bf16, "gm"),
    src_ptr: pto.ptr(pto.ui16, "gm"),
    dst_ptr: pto.ptr(pto.ui16, "gm"),
):
    cmp_view = pto.make_tensor_view(cmp_ptr, shape=[ROWS, COLS], strides=[COLS, 1])
    src_view = pto.make_tensor_view(src_ptr, shape=[ROWS, COLS], strides=[COLS, 1])
    dst_view = pto.make_tensor_view(dst_ptr, shape=[ROWS, COLS], strides=[COLS, 1])
    cmp = pto.alloc_tile(
        shape=[ROWS, COLS], dtype=pto.bf16, valid_shape=[ROWS, COLS], blayout="RowMajor"
    )
    mask = pto.alloc_tile(
        shape=[ROWS, MASK_COLS], dtype=pto.ui8,
        valid_shape=[ROWS, MASK_COLS], blayout="RowMajor",
    )
    src = pto.alloc_tile(
        shape=[ROWS, COLS], dtype=pto.ui16, valid_shape=[ROWS, COLS], blayout="RowMajor"
    )
    dst = pto.alloc_tile(
        shape=[ROWS, COLS], dtype=pto.ui16, valid_shape=[ROWS, COLS], blayout="RowMajor"
    )
    pto.tile.load(cmp_view, cmp)
    pto.tile.load(src_view, src)
    # tcmps writes a ui8 predicate tile; tsels must accept it without a
    # separate bitcast tile.
    pto.tile.cmps(cmp, 0.0, mask, cmp_mode="lt")
    pto.tile.sels(mask, src, TSELS_SCALAR, dst)
    pto.tile.store(dst, dst_view)


@pto.jit(name="issue1607_tfmods_bf16", kernel_kind="vector", target="a5", backend="vpto")
def _tfmods_bf16_kernel(
    src_ptr: pto.ptr(pto.bf16, "gm"),
    dst_ptr: pto.ptr(pto.bf16, "gm"),
):
    src_view = pto.make_tensor_view(src_ptr, shape=[ROWS, COLS], strides=[COLS, 1])
    dst_view = pto.make_tensor_view(dst_ptr, shape=[ROWS, COLS], strides=[COLS, 1])
    src = pto.alloc_tile(
        shape=[ROWS, COLS], dtype=pto.bf16, valid_shape=[ROWS, COLS], blayout="RowMajor"
    )
    dst = pto.alloc_tile(
        shape=[ROWS, COLS], dtype=pto.bf16, valid_shape=[ROWS, COLS], blayout="RowMajor"
    )
    pto.tile.load(src_view, src)
    pto.tile.fmods(src, 3.0, dst)
    pto.tile.store(dst, dst_view)


@pto.jit(name="issue1607_tfmods_bf16_overflow", kernel_kind="vector", target="a5", backend="vpto")
def _tfmods_bf16_overflow_kernel(
    src_ptr: pto.ptr(pto.bf16, "gm"),
    dst_ptr: pto.ptr(pto.bf16, "gm"),
):
    src_view = pto.make_tensor_view(src_ptr, shape=[ROWS, COLS], strides=[COLS, 1])
    dst_view = pto.make_tensor_view(dst_ptr, shape=[ROWS, COLS], strides=[COLS, 1])
    src = pto.alloc_tile(
        shape=[ROWS, COLS], dtype=pto.bf16, valid_shape=[ROWS, COLS], blayout="RowMajor"
    )
    dst = pto.alloc_tile(
        shape=[ROWS, COLS], dtype=pto.bf16, valid_shape=[ROWS, COLS], blayout="RowMajor"
    )
    pto.tile.load(src_view, src)
    pto.tile.fmods(src, 0.5, dst)
    pto.tile.store(dst, dst_view)


@pto.jit(name="issue1607_tfmods_bf16_nan", kernel_kind="vector", target="a5", backend="vpto")
def _tfmods_bf16_nan_kernel(
    src_ptr: pto.ptr(pto.bf16, "gm"),
    dst_ptr: pto.ptr(pto.bf16, "gm"),
):
    src_view = pto.make_tensor_view(src_ptr, shape=[ROWS, COLS], strides=[COLS, 1])
    dst_view = pto.make_tensor_view(dst_ptr, shape=[ROWS, COLS], strides=[COLS, 1])
    src = pto.alloc_tile(
        shape=[ROWS, COLS], dtype=pto.bf16, valid_shape=[ROWS, COLS], blayout="RowMajor"
    )
    dst = pto.alloc_tile(
        shape=[ROWS, COLS], dtype=pto.bf16, valid_shape=[ROWS, COLS], blayout="RowMajor"
    )
    pto.tile.load(src_view, src)
    pto.tile.fmods(src, FMOD_NAN, dst)
    pto.tile.store(dst, dst_view)


def _bf16_bits(values):
    return (values.view(np.uint32) >> 16).astype(np.uint16)


def _bf16_float(bits):
    return (bits.astype(np.uint32) << 16).view(np.float32)


def _tcmps_inputs():
    rng = np.random.default_rng(0x1607)
    values = rng.uniform(-4.0, 4.0, size=(ROWS, COLS)).astype(np.float32)
    return [_bf16_bits(values)]


def _tcmps_expected(bits):
    values = _bf16_float(bits)
    packed = np.packbits(values < np.float32(0.0), axis=1, bitorder="little")
    result = np.full((ROWS, MASK_COLS), MASK_PADDING, dtype=np.uint8)
    result[:, :MASK_BYTES] = packed
    return result


def _tsels_inputs():
    rng = np.random.default_rng(0x1608)
    predicate = rng.integers(0, 2, size=(ROWS, COLS), dtype=np.uint8).astype(np.bool_)
    packed = np.packbits(predicate, axis=1, bitorder="little").view(np.int8)
    mask = np.zeros((ROWS, MASK_COLS), dtype=np.int8)
    mask[:, :MASK_BYTES] = packed
    src = rng.integers(0, np.iinfo(np.uint16).max + 1, size=(ROWS, COLS), dtype=np.uint16)
    return [mask, src]


def _tsels_expected(mask, src):
    predicate = np.unpackbits(
        mask[:, :MASK_BYTES].view(np.uint8),
        axis=1,
        count=COLS,
        bitorder="little",
    )
    return np.where(predicate.astype(bool), src, np.uint16(TSELS_SCALAR))


def _tcmps_to_tsels_inputs():
    rng = np.random.default_rng(0x1610)
    values = rng.uniform(-4.0, 4.0, size=(ROWS, COLS)).astype(np.float32)
    src = rng.integers(
        0, np.iinfo(np.uint16).max + 1, size=(ROWS, COLS), dtype=np.uint16
    )
    return [_bf16_bits(values), src]


def _tcmps_to_tsels_expected(bits, src):
    values = _bf16_float(bits)
    return np.where(values < np.float32(0.0), src, np.uint16(TSELS_SCALAR))


def _tfmods_bf16_inputs():
    rng = np.random.default_rng(0x1609)
    # Integer magnitudes keep every remainder exactly representable in bf16.
    values = rng.integers(-12, 13, size=(ROWS, COLS), dtype=np.int16)
    return [_bf16_bits(values.astype(np.float32))]


def _tfmods_bf16_expected(bits):
    values = _bf16_float(bits)
    remainder = np.fmod(values, np.float32(3.0))
    return (remainder.view(np.uint32) >> 16).astype(np.uint16)


def _tfmods_bf16_overflow_inputs():
    # Include the largest finite bf16; dividing it by 0.5 exceeds the f32
    # quotient range, which the naive widening path turned into infinity.
    bf16_max = np.array([0x7F7F0000], dtype=np.uint32).view(np.float32)[0]
    rng = np.random.default_rng(0x1611)
    values = rng.uniform(-1.0e38, 1.0e38, size=(ROWS, COLS)).astype(np.float32)
    values[0, 0] = bf16_max
    values[1, 0] = -bf16_max
    return [_bf16_bits(values)]


def _tfmods_bf16_overflow_expected(bits):
    values = _bf16_float(bits)
    remainder = np.fmod(values, np.float32(0.5))
    return (remainder.view(np.uint32) >> 16).astype(np.uint16)


def _tfmods_bf16_nan_inputs():
    # Both zero signs must yield NaN, rather than being restored by the guard.
    values = np.resize(np.array([0.0, -0.0, 1.0, -1.0], dtype=np.float32), (ROWS, COLS))
    return [_bf16_bits(values)]


def _tfmods_bf16_nan_expected(bits):
    return np.full(bits.shape, np.nan, dtype=np.float32)


def _check_tfmods_bf16_nan(device_inputs, expected):
    actual = _bf16_float(device_inputs[-1].cpu().numpy())
    np.testing.assert_allclose(actual, expected, rtol=0, atol=0, equal_nan=True)


CASES = [
    golden_output_case(
        "issue1607_tcmps_bf16_mask_pack",
        _tcmps_bf16_kernel,
        inputs=_tcmps_inputs,
        expected=_tcmps_expected,
        rtol=0,
        atol=0,
    ),
    golden_output_case(
        "issue1607_tsels_ui16_us_predicate",
        _tsels_ui16_kernel,
        inputs=_tsels_inputs,
        expected=_tsels_expected,
        rtol=0,
        atol=0,
    ),
    golden_output_case(
        "issue1607_tcmps_ui8_mask_to_tsels_ui16",
        _tcmps_to_tsels_kernel,
        inputs=_tcmps_to_tsels_inputs,
        expected=_tcmps_to_tsels_expected,
        rtol=0,
        atol=0,
    ),
    golden_output_case(
        "issue1607_tfmods_bf16_widened",
        _tfmods_bf16_kernel,
        inputs=_tfmods_bf16_inputs,
        expected=_tfmods_bf16_expected,
        rtol=0,
        atol=0,
    ),
    golden_output_case(
        "issue1607_tfmods_bf16_overflow",
        _tfmods_bf16_overflow_kernel,
        inputs=_tfmods_bf16_overflow_inputs,
        expected=_tfmods_bf16_overflow_expected,
        rtol=0,
        atol=0,
    ),
]


_nan_case = golden_output_case(
    "issue1607_tfmods_bf16_nan",
    _tfmods_bf16_nan_kernel,
    inputs=_tfmods_bf16_nan_inputs,
    expected=_tfmods_bf16_nan_expected,
    output_dtype=np.uint16,
)
_nan_case["check"] = _check_tfmods_bf16_nan
CASES.append(_nan_case)


auto_main(globals())
