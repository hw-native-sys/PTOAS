# coding=utf-8
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Reference case metadata for the dequant kernel."""

from __future__ import annotations

import numpy as np
import torch

DTYPES = ("f32", "bf16", "f16")
SRC_FORMATS = ("e4m3", "e5m2")
SCALE_FORMAT = "e8m0"
DEFAULT_ROW_BLOCK_NUM = 4
DEFAULT_COL_BLOCK_NUM = 4
TOLERANCE = {
    "f32": 1e-6,
    "bf16": 4e-3,
    "f16": 1e-3,
}

_SEED = 42
_BLOCK_SIZE = 32
_BLOCKS_PER_LOOP = 16
_ELEMS_PER_LOOP = _BLOCK_SIZE * _BLOCKS_PER_LOOP
_RAW_SCALE_BYTES_PER_LOOP = _BLOCKS_PER_LOOP
_E8M0_BYTES = np.array([0x7E, 0x7F, 0x80, 0x81], dtype=np.uint8)


def _case_id(src_fmt: str, dst_fmt: str) -> str:
    return f"{src_fmt}_{dst_fmt}"


def _e8m0_to_f32(bits: np.ndarray) -> np.ndarray:
    return np.exp2(np.asarray(bits, dtype=np.uint8).astype(np.int32) - 127).astype(np.float32)


def _torch_src_dtype(src_fmt: str) -> torch.dtype:
    if src_fmt == "e4m3":
        return torch.float8_e4m3fn
    if src_fmt == "e5m2":
        return torch.float8_e5m2
    raise ValueError(f"unknown src_fmt: {src_fmt}")


def _quantize_src_to_bits(src_fmt: str, values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    values_t = torch.from_numpy(np.asarray(values, dtype=np.float32))
    quantized_t = values_t.to(_torch_src_dtype(src_fmt))
    x_bits = quantized_t.view(torch.uint8).numpy().copy()
    x_decoded = quantized_t.float().numpy().astype(np.float32, copy=True)
    return x_bits, x_decoded


def _cast_output(dst_fmt: str, values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    if dst_fmt == "f32":
        return values.astype(np.float32, copy=True)
    if dst_fmt == "f16":
        return values.astype(np.float16).astype(np.float32)
    if dst_fmt == "bf16":
        return torch.from_numpy(values).to(torch.bfloat16).float().numpy().astype(np.float32, copy=True)
    raise ValueError(f"unknown dst_fmt: {dst_fmt}")


def _effective_col_block_num(col_block_num: int) -> int:
    return col_block_num + (col_block_num % 2)


def _loop_num2vf(row_block_num: int, col_block_num: int) -> int:
    total_block_num = row_block_num * _effective_col_block_num(col_block_num)
    return (total_block_num + _BLOCKS_PER_LOOP - 1) // _BLOCKS_PER_LOOP


def generate_case(
    src_fmt: str,
    dst_fmt: str,
    *,
    row_block_num: int = DEFAULT_ROW_BLOCK_NUM,
    col_block_num: int = DEFAULT_COL_BLOCK_NUM,
) -> dict[str, object]:
    """Generate one lightweight CPU reference case."""

    if src_fmt not in SRC_FORMATS:
        raise ValueError(f"unknown src_fmt: {src_fmt}")
    if dst_fmt not in DTYPES:
        raise ValueError(f"unknown dst_fmt: {dst_fmt}")

    effective_col_block_num = _effective_col_block_num(col_block_num)
    total_block_num = row_block_num * effective_col_block_num
    loop_num2vf = _loop_num2vf(row_block_num, col_block_num)
    padded_block_num = loop_num2vf * _BLOCKS_PER_LOOP

    rng = np.random.default_rng(_SEED)
    x_valid = rng.normal(0.0, 0.5, size=total_block_num * _BLOCK_SIZE).astype(np.float32)
    x_padded = np.zeros(padded_block_num * _BLOCK_SIZE, dtype=np.float32)
    x_padded[: x_valid.size] = x_valid

    scale_bits = np.tile(_E8M0_BYTES, (total_block_num + _E8M0_BYTES.size - 1) // _E8M0_BYTES.size)[:total_block_num]
    scale_bits = scale_bits.astype(np.uint8, copy=True)
    scale_values = _e8m0_to_f32(scale_bits)
    scale_bits_padded = np.zeros(loop_num2vf * _RAW_SCALE_BYTES_PER_LOOP, dtype=np.uint8)
    for i in range(loop_num2vf):
        src_off = i * _BLOCKS_PER_LOOP
        dst_off = i * _RAW_SCALE_BYTES_PER_LOOP
        scale_bits_padded[dst_off : dst_off + _BLOCKS_PER_LOOP] = scale_bits[src_off : src_off + _BLOCKS_PER_LOOP]

    x_bits, x_quantized = _quantize_src_to_bits(src_fmt, x_padded)
    block_scale = np.zeros(padded_block_num, dtype=np.float32)
    block_scale[:total_block_num] = scale_values
    y_f32 = (
        x_quantized.reshape(padded_block_num, _BLOCK_SIZE)
        * block_scale.reshape(padded_block_num, 1)
    ).reshape(-1)
    y_expected = _cast_output(dst_fmt, y_f32)

    return {
        "case_id": _case_id(src_fmt, dst_fmt),
        "src_fmt": src_fmt,
        "scale_fmt": SCALE_FORMAT,
        "dst_fmt": dst_fmt,
        "row_block_num": row_block_num,
        "col_block_num": col_block_num,
        "effective_col_block_num": effective_col_block_num,
        "total_block_num": total_block_num,
        "total_scale_num": total_block_num,
        "loop_num2vf": loop_num2vf,
        "x_bits": x_bits,
        "x_f32": x_quantized,
        "scale_bits": scale_bits_padded,
        "scale_f32": scale_values,
        "y_expected": y_expected,
        "tolerance": TOLERANCE[dst_fmt],
        "default_alias": src_fmt == "e4m3" and dst_fmt == "f32",
    }


def generate_all(
    *,
    row_block_num: int = DEFAULT_ROW_BLOCK_NUM,
    col_block_num: int = DEFAULT_COL_BLOCK_NUM,
) -> dict[str, dict[str, object]]:
    """Generate the default dequant case matrix."""

    return {
        _case_id(src_fmt, dst_fmt): generate_case(
            src_fmt,
            dst_fmt,
            row_block_num=row_block_num,
            col_block_num=col_block_num,
        )
        for src_fmt in SRC_FORMATS
        for dst_fmt in DTYPES
    }
