#!/usr/bin/env python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

from pathlib import Path
import sys, numpy as np

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from common import auto_main, golden_output_case
from ptodsl import pto

_BLOCK_BYTES = 32


def _ceil(val, align):
    return ((val + align - 1) // align) * align


def cumulative_histogram_asc(byte_values):
    counts = np.bincount(byte_values, minlength=256).astype(np.uint32)
    return np.cumsum(counts, dtype=np.uint32)


def get_k_index(cumulative_hist_asc, k):
    if cumulative_hist_asc.size == 0:
        return 0
    total = cumulative_hist_asc[-1]
    cumulative_hist_desc = total - np.concatenate(([0], cumulative_hist_asc[:-1])).astype(np.uint32)
    valid_bins = np.flatnonzero(cumulative_hist_desc >= k)
    if valid_bins.size == 0:
        return 0
    return int(valid_bins[-1])


def _gen_u16_golden(rows, cols, mode, k):
    src = np.random.randint(0, 65536, size=(rows, cols), dtype=np.uint16)
    msb_bytes = ((src >> 8) & 0xFF).astype(np.uint8)
    lsb_bytes = (src & 0xFF).astype(np.uint8)
    golden = np.zeros((rows, 256), dtype=np.uint32)
    idx = np.zeros(rows, dtype=np.uint8)
    if mode == "MSB":
        for row in range(rows):
            golden[row] = cumulative_histogram_asc(msb_bytes[row])
    else:
        for row in range(rows):
            row_msb_hist = cumulative_histogram_asc(msb_bytes[row])
            k_index = get_k_index(row_msb_hist, k)
            idx[row] = np.uint8(k_index)
            selected_lsb = lsb_bytes[row][msb_bytes[row] == k_index]
            golden[row] = cumulative_histogram_asc(selected_lsb)
    return src, idx, golden


def _gen_u32_golden(rows, cols, byte, k):
    src = np.random.randint(0, 2**32, size=(rows, cols), dtype=np.uint32)
    byte3 = ((src >> 24) & 0xFF).astype(np.uint8)
    byte2 = ((src >> 16) & 0xFF).astype(np.uint8)
    byte1 = ((src >> 8) & 0xFF).astype(np.uint8)
    byte0 = (src & 0xFF).astype(np.uint8)
    all_bytes = [byte3, byte2, byte1, byte0]
    num_filter_passes = 3 - byte
    golden = np.zeros((rows, 256), dtype=np.uint32)
    k_indices = np.zeros(max(num_filter_passes, 1), dtype=np.uint8)
    if num_filter_passes > 0:
        mask_row0 = np.ones(cols, dtype=bool)
        for b in range(num_filter_passes):
            byte_data = all_bytes[b][0]
            hist = cumulative_histogram_asc(byte_data[mask_row0])
            ki = get_k_index(hist, k)
            k_indices[b] = np.uint8(ki)
            mask_row0 = mask_row0 & (byte_data == ki)
    for row in range(rows):
        mask = np.ones(cols, dtype=bool)
        for b in range(num_filter_passes):
            byte_data = all_bytes[b][row]
            mask = mask & (byte_data == k_indices[b])
        target_byte = all_bytes[3 - byte][row]
        golden[row] = cumulative_histogram_asc(target_byte[mask])
    if num_filter_passes > 0:
        idx = np.zeros((num_filter_passes, cols), dtype=np.uint8)
        for b in range(num_filter_passes):
            idx[b, :] = k_indices[b]
    else:
        idx = np.zeros((1, 1), dtype=np.uint8)
    return src, idx, golden


# Each branch keeps partial-repeat, exact-repeat, and multi-row coverage. U32
# also retains a multi-repeat tail and every byte/filter depth.
_U16_CASES = [
    (1, 128, "MSB", 2), (1, 256, "MSB", 2), (2, 100, "MSB", 2),
    (1, 128, "LSB", 1), (1, 256, "LSB", 1), (2, 100, "LSB", 1),
]

_U32_CASES = [
    (1, 128, 3, 1), (1, 256, 3, 1), (2, 384, 3, 1),
    (1, 128, 2, 1), (1, 256, 2, 1), (2, 384, 2, 1),
    (1, 128, 1, 1), (1, 256, 1, 1), (2, 384, 1, 1),
    (1, 128, 0, 1), (1, 256, 0, 1), (2, 384, 0, 1),
]

CASES = []
_KERNELS = {}

np.random.seed(42)

for _rows, _cols, _mode, _k in _U16_CASES:
    _byte = 1 if _mode == "MSB" else 0
    _is_msb = _mode == "MSB"
    _aligned_src = max(_ceil(_cols, _BLOCK_BYTES // 2), 128)
    _aligned_rows = _ceil(_rows, _BLOCK_BYTES)
    _kname = f"thistogram_u16_{_mode.lower()}_{_rows}x{_cols}_k{_k}"

    def _make_u16(rows=_rows, cols=_cols, byte=_byte, is_msb=_is_msb,
                  aligned_src=_aligned_src, aligned_rows=_aligned_rows, kname=_kname):
        @pto.jit(name=kname, target="a5")
        def _kernel(src_ptr: pto.ptr(pto.ui16, "gm"),
                    idx_ptr: pto.ptr(pto.ui8, "gm"),
                    out_ptr: pto.ptr(pto.ui32, "gm")):
            src_view = pto.make_tensor_view(src_ptr, shape=[rows, cols], strides=[cols, 1])
            out_view = pto.make_tensor_view(out_ptr, shape=[rows, 256], strides=[256, 1])
            src_tile = pto.alloc_tile(shape=[aligned_rows, aligned_src], dtype=pto.ui16,
                                       valid_shape=[rows, cols])
            out_tile = pto.alloc_tile(shape=[aligned_rows, 256], dtype=pto.ui32,
                                      valid_shape=[rows, 256])
            idx_tile = pto.alloc_tile(shape=[aligned_rows, 1], dtype=pto.ui8,
                                      valid_shape=[rows, 1], blayout="ColMajor")
            pto.tile.load(src_view, src_tile)
            if not is_msb:
                idx_view = pto.make_tensor_view(idx_ptr, shape=[rows, 1], strides=[1, 1])
                pto.tile.load(idx_view, idx_tile)
            pto.tile.histogram(src_tile, idx_tile, out_tile, byte=byte)
            pto.tile.store(out_tile, out_view)
        return _kernel

    _KERNELS[_kname] = _make_u16()

    _src, _idx, _golden = _gen_u16_golden(_rows, _cols, _mode, _k)
    _idx_data = np.zeros(_rows, dtype=np.uint8) if _is_msb else _idx

    CASES.append(golden_output_case(
        _kname, _KERNELS[_kname],
        inputs=lambda s=_src, i=_idx_data: [s, i],
        expected=_golden,
        output_shape=[_rows, 256],
        output_dtype=np.uint32,
        rtol=0.0, atol=1e-6,
    ))


for _rows, _cols, _byte, _k in _U32_CASES:
    _num_idx_rows = max(3 - _byte, 1)
    _aligned_col = max(_ceil(_cols, _BLOCK_BYTES), 64)
    _aligned_idx_col = _ceil(_cols, _BLOCK_BYTES) if _byte < 3 else _BLOCK_BYTES
    _kname = f"thistogram_u32_b{_byte}_{_rows}x{_cols}_k{_k}"

    def _make_u32(rows=_rows, cols=_cols, byte=_byte,
                  aligned_col=_aligned_col, num_idx_rows=_num_idx_rows,
                  aligned_idx_col=_aligned_idx_col, kname=_kname):
        @pto.jit(name=kname, target="a5")
        def _kernel(src_ptr: pto.ptr(pto.ui32, "gm"),
                    idx_ptr: pto.ptr(pto.ui8, "gm"),
                    out_ptr: pto.ptr(pto.ui32, "gm")):
            src_view = pto.make_tensor_view(src_ptr, shape=[rows, cols], strides=[cols, 1])
            out_view = pto.make_tensor_view(out_ptr, shape=[rows, 256], strides=[256, 1])
            src_tile = pto.alloc_tile(shape=[rows, aligned_col], dtype=pto.ui32,
                                       valid_shape=[rows, cols])
            out_tile = pto.alloc_tile(shape=[rows, 256], dtype=pto.ui32,
                                      valid_shape=[rows, 256])
            idx_tile = pto.alloc_tile(shape=[num_idx_rows, aligned_idx_col], dtype=pto.ui8,
                                      valid_shape=[num_idx_rows, cols] if byte < 3 else [1, 1])
            pto.tile.load(src_view, src_tile)
            if byte < 3:
                idx_view = pto.make_tensor_view(idx_ptr, shape=[num_idx_rows, cols], strides=[cols, 1])
                pto.tile.load(idx_view, idx_tile)
            pto.tile.histogram(src_tile, idx_tile, out_tile, byte=byte)
            pto.tile.store(out_tile, out_view)
        return _kernel

    _KERNELS[_kname] = _make_u32()

    _src, _idx, _golden = _gen_u32_golden(_rows, _cols, _byte, _k)
    _idx_data = np.zeros(1, dtype=np.uint8) if _byte == 3 else _idx

    CASES.append(golden_output_case(
        _kname, _KERNELS[_kname],
        inputs=lambda s=_src, i=_idx_data: [s, i],
        expected=_golden,
        output_shape=[_rows, 256],
        output_dtype=np.uint32,
        rtol=0.0, atol=1e-6,
    ))

auto_main(globals())
