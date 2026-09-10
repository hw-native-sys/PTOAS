# coding=utf-8
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""CPU golden references for the rope kernel."""

from __future__ import annotations

import numpy as np
import torch

from .tile_config import DEFAULT_TILE, DTYPES, MODES, TileConfig

SEED = 42


def cpu_rotary_half(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    x1, x2 = torch.chunk(x, 2, -1)
    x_new = torch.cat((-x2, x1), dim=-1)
    cos_b = cos.unsqueeze(1)
    sin_b = sin.unsqueeze(1)
    return cos_b * x + sin_b * x_new


def cpu_rotary_interleave(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    x1 = x[..., ::2]
    x2 = x[..., 1::2]
    x_new = torch.stack((-x2, x1), dim=-1).reshape(x.shape)
    cos_b = cos.unsqueeze(1)
    sin_b = sin.unsqueeze(1)
    return x * cos_b + x_new * sin_b


def _to_numpy(t: torch.Tensor) -> np.ndarray:
    if t.dtype == torch.bfloat16:
        return t.float().numpy()
    return t.numpy()


def _torch_dtype(dtype: str) -> torch.dtype:
    if dtype == "f16":
        return torch.float16
    if dtype == "bf16":
        return torch.bfloat16
    if dtype == "f32":
        return torch.float32
    raise ValueError(f"unknown dtype: {dtype}")


def generate_case(mode: str, dtype: str, tile: TileConfig | None = None) -> dict:
    if mode not in MODES:
        raise ValueError(f"unknown mode: {mode}")
    if dtype not in DTYPES:
        raise ValueError(f"unknown dtype: {dtype}")
    tile = tile or DEFAULT_TILE

    x_dtype = _torch_dtype(dtype)
    cs_dtype = torch.float16 if dtype == "bf16" else x_dtype

    torch.manual_seed(SEED)
    x = torch.randn(tile.x_shape, dtype=x_dtype)
    cos = torch.randn(tile.cs_shape, dtype=cs_dtype).abs().clamp(0.1, 0.9)
    sin = torch.randn(tile.cs_shape, dtype=cs_dtype).abs().clamp(0.1, 0.9)

    if mode == "half":
        y = cpu_rotary_half(x, cos, sin)
    else:
        y = cpu_rotary_interleave(x, cos, sin)

    return {
        "mode": mode,
        "dtype": dtype,
        "tile": tile,
        "x": _to_numpy(x),
        "cos": _to_numpy(cos),
        "sin": _to_numpy(sin),
        "y": _to_numpy(y),
        "params": np.array([tile.s, tile.n], dtype=np.int32),
    }


def generate_all(tile: TileConfig | None = None) -> dict[str, dict]:
    tile = tile or DEFAULT_TILE
    cases: dict[str, dict] = {}
    for dtype in DTYPES:
        for mode in MODES:
            cases[f"{dtype}_{mode}"] = generate_case(mode, dtype, tile=tile)
    return cases
