# coding=utf-8
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Shared tile configuration for the rope kernel."""

from __future__ import annotations

from dataclasses import dataclass

SIM_B = 1
SIM_D = 64
SIM_D_ALIGN = 64

DEFAULT_S = 15
DEFAULT_N = 32

MAX_S = 15
MAX_N = 32

MODES = ("half", "interleave")
DTYPES = ("f16", "bf16", "f32")

TOLERANCE = {
    "f16": 0.01,
    "bf16": 0.07,
    "f32": 1e-5,
}


@dataclass(frozen=True)
class TileConfig:
    name: str
    s: int
    n: int

    @property
    def vf_inner_iters(self) -> int:
        return self.s * self.n

    @property
    def x_shape(self) -> tuple[int, int, int]:
        return (self.s, self.n, SIM_D)

    @property
    def cs_shape(self) -> tuple[int, int]:
        return (self.s, SIM_D)


WALLTIME_CONFIGS: tuple[TileConfig, ...] = (
    TileConfig("tiny", s=1, n=2),
    TileConfig("n4", s=15, n=4),
    TileConfig("n8", s=15, n=8),
    TileConfig("n16", s=15, n=16),
    TileConfig("prod", s=DEFAULT_S, n=DEFAULT_N),
)

DEFAULT_TILE = TileConfig("prod", s=DEFAULT_S, n=DEFAULT_N)


def sim_fn_name(mode: str, dtype: str, cycle: bool = False) -> str:
    prefix = "call_rope_cce_cycle" if cycle else "call_rope_cce_sim"
    return f"{prefix}_{mode}_{dtype}"
