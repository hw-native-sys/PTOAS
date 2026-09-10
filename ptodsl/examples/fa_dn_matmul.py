# coding=utf-8
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""
PTODSL structural port of ``fa_dn_matmul.cpp``.

This file mirrors the helper-only C++ implementation closely:

- ``LayoutT``
- ``AccMode``
- ``calculate_fitting_cube_k(...)``
- ``resolve_acc_mode(...)``
- ``pto_macro_matmul(...)``

The goal here is to preserve the same orchestration and hook ordering rather
than to provide one finalized, production-ready cube helper.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from ptodsl import pto


CUBE_K_256 = 256
CUBE_K_128 = 128
CUBE_K_64 = 64
CUBE_K_SMALLEST = 32

L0A_BUF0 = 0x0
L0A_BUF1 = 0x8000
L0B_BUF0 = 0x0
L0B_BUF1 = 0x8000
L0C_BUF0 = 0x0
L0C_BUF1 = 0x20000

MEM_BUFFER_SIZE_BYTES = 64 * 1024 // 2
HALF_SIZE_BYTES = 2


class LayoutT(Enum):
    NN = "NN"
    NT = "NT"
    TN = "TN"
    TT = "TT"
    NONE = "NONE"


class AccMode(Enum):
    Init = "Init"
    Acc = "Acc"
    InitPartialSum = "InitPartialSum"
    InitFinalSum = "InitFinalSum"
    AccPartialSum = "AccPartialSum"
    AccFinalSum = "AccFinalSum"


class AccPhase(Enum):
    Unknown = "Unknown"
    Partial = "Partial"
    Final = "Final"


@dataclass(frozen=True)
class MatmulCallConfig:
    useAcc: bool
    phase: AccPhase


class NoOpMatmulHook:
    def __call__(self):
        return None


def get_ping_pong(pingpong, flip):
    one = pto.const(1, dtype=pto.i32)
    with pto.if_(flip) as br:
        with br.then_:
            br.assign(pingpong=one - pingpong)
        with br.else_:
            br.assign(pingpong=pingpong)
    return br.pingpong


def calculate_fitting_cube_k(Cube_M: int, Cube_N: int) -> int:
    maxElements = MEM_BUFFER_SIZE_BYTES // HALF_SIZE_BYTES
    bestCubeK = CUBE_K_SMALLEST
    if Cube_M * CUBE_K_256 <= maxElements and CUBE_K_256 * Cube_N <= maxElements:
        bestCubeK = CUBE_K_256
    elif Cube_M * CUBE_K_128 <= maxElements and CUBE_K_128 * Cube_N <= maxElements:
        bestCubeK = CUBE_K_128
    elif Cube_M * CUBE_K_64 <= maxElements and CUBE_K_64 * Cube_N <= maxElements:
        bestCubeK = CUBE_K_64
    return bestCubeK


def deduce_layout(TileDataA, TileDataB):
    _ = (TileDataA, TileDataB)
    return LayoutT.NONE


def resolve_acc_mode(mode: AccMode, isFirstSlice: bool, isLastSlice: bool) -> MatmulCallConfig:
    if mode == AccMode.Init:
        return MatmulCallConfig(not isFirstSlice, AccPhase.Unknown)
    elif mode == AccMode.Acc:
        return MatmulCallConfig(True, AccPhase.Unknown)
    elif mode == AccMode.InitPartialSum:
        return MatmulCallConfig(not isFirstSlice, AccPhase.Partial)
    elif mode == AccMode.InitFinalSum:
        return MatmulCallConfig(not isFirstSlice, AccPhase.Final)
    elif mode == AccMode.AccPartialSum:
        return MatmulCallConfig(True, AccPhase.Partial)
    elif mode == AccMode.AccFinalSum:
        return MatmulCallConfig(True, AccPhase.Final)
    _ = isLastSlice
    return MatmulCallConfig(not isFirstSlice, AccPhase.Partial)


def pto_macro_matmul(
    *,
    Cube_M: int,
    Tile_K: int,
    Cube_N: int,
    L1LoadBFirst: bool = False,
    LAYOUT: LayoutT = LayoutT.NONE,
    TileDataA=None,
    TileDataB=None,
    TileDataC=None,
    OpHook=NoOpMatmulHook,
    OpHook2=NoOpMatmulHook,
    OpHook3=NoOpMatmulHook,
    aMatTile,
    bMatTile,
    cAccTile,
    accMode: AccMode = AccMode.Init,
    preATExtOpHook=NoOpMatmulHook(),
    preBTExtOpHook=NoOpMatmulHook(),
    postTExtOpHook=NoOpMatmulHook(),
):
    pingpong = pto.const(0, dtype=pto.i32)
    fittingCubeK = calculate_fitting_cube_k(Cube_M, Cube_N)
    Cube_K = Tile_K if fittingCubeK > Tile_K else fittingCubeK
    kSegments = Tile_K // Cube_K

    LeftTile = lambda addr: pto.alloc_tile(
        shape=[Cube_M, Cube_K],
        dtype=pto.f16,
        memory_space=pto.MemorySpace.LEFT,
        valid_shape=[Cube_M, Cube_K],
        addr=addr,
    )
    RightTile = lambda addr: pto.alloc_tile(
        shape=[Cube_K, Cube_N],
        dtype=pto.f16,
        memory_space=pto.MemorySpace.RIGHT,
        valid_shape=[Cube_K, Cube_N],
        addr=addr,
    )
    # Structural mirror of:
    #   using LeftTile = TileLeft<...>;
    #   using RightTile = TileRight<...>;
    #   LeftTile al0Tiles[2] = {LeftTile(), LeftTile()};
    #   RightTile bl0Tiles[2] = {RightTile(), RightTile()};
    #   TASSIGN(... L0A/L0B BUF0/BUF1)
    al0Tiles = [LeftTile(L0A_BUF0), LeftTile(L0A_BUF1)]
    bl0Tiles = [RightTile(L0B_BUF0), RightTile(L0B_BUF1)]

    k_loop = pto.for_(0, kSegments, step=1).carry(pingpong=pingpong)
    with k_loop:
        k = k_loop.iv
        pingpong = k_loop.pingpong
        pto.wait_flag(pto.Pipe.M, pto.Pipe.MTE1, event_id=pingpong)

        kOffset = k * Cube_K
        isFirst = k == 0
        isLast = (k + 1) == kSegments

        def emit_pingpong_path(al0Tile, bl0Tile):
            if L1LoadBFirst:
                with pto.if_(isFirst) as is_first_b_br:
                    with is_first_b_br.then_:
                        preBTExtOpHook()
                pto.tile.extract(bMatTile, bl0Tile, kOffset, 0)
                with pto.if_(isFirst) as is_first_a_br:
                    with is_first_a_br.then_:
                        preATExtOpHook()
                pto.tile.extract(aMatTile, al0Tile, 0, kOffset)
            else:
                with pto.if_(isFirst) as is_first_a_br:
                    with is_first_a_br.then_:
                        preATExtOpHook()
                pto.tile.extract(aMatTile, al0Tile, 0, kOffset)
                with pto.if_(isFirst) as is_first_b_br:
                    with is_first_b_br.then_:
                        preBTExtOpHook()
                pto.tile.extract(bMatTile, bl0Tile, kOffset, 0)

            pto.set_flag(pto.Pipe.MTE1, pto.Pipe.M, event_id=pingpong)
            pto.wait_flag(pto.Pipe.MTE1, pto.Pipe.M, event_id=pingpong)

            with pto.if_(isLast) as is_last_br:
                with is_last_br.then_:
                    postTExtOpHook()

            cfg = resolve_acc_mode(accMode, isFirst, isLast)
            # FIXME: tmatmul does not have AccPhase in PTO IR
            with pto.if_(cfg.useAcc) as use_acc_br:
                with use_acc_br.then_:
                    with pto.if_(cfg.phase == AccPhase.Final) as phase_is_final_br:
                        with phase_is_final_br.then_:
                            pto.tile.matmul_acc(cAccTile, al0Tile, bl0Tile, cAccTile)
                        with phase_is_final_br.else_:
                            with pto.if_(cfg.phase == AccPhase.Partial) as phase_is_partial_br:
                                with phase_is_partial_br.then_:
                                    pto.tile.matmul_acc(cAccTile, al0Tile, bl0Tile, cAccTile)
                                with phase_is_partial_br.else_:
                                    pto.tile.matmul_acc(cAccTile, al0Tile, bl0Tile, cAccTile)
                with use_acc_br.else_:
                    with pto.if_(cfg.phase == AccPhase.Final) as phase_is_final_br:
                        with phase_is_final_br.then_:
                            pto.tile.matmul(al0Tile, bl0Tile, cAccTile)
                        with phase_is_final_br.else_:
                            with pto.if_(cfg.phase == AccPhase.Partial) as phase_is_partial_br:
                                with phase_is_partial_br.then_:
                                    pto.tile.matmul(al0Tile, bl0Tile, cAccTile)
                                with phase_is_partial_br.else_:
                                    pto.tile.matmul(al0Tile, bl0Tile, cAccTile)

            pto.set_flag(pto.Pipe.M, pto.Pipe.MTE1, event_id=pingpong)

        with pto.if_(pingpong == 0) as pingpong_is_zero_br:
            with pingpong_is_zero_br.then_:
                emit_pingpong_path(al0Tiles[0], bl0Tiles[0])
            with pingpong_is_zero_br.else_:
                emit_pingpong_path(al0Tiles[1], bl0Tiles[1])

        k_loop.update(pingpong=get_ping_pong(pingpong, pto.const(1, dtype=pto.i1)))


__all__ = [
    "AccMode",
    "AccPhase",
    "LayoutT",
    "MatmulCallConfig",
    "NoOpMatmulHook",
    "calculate_fitting_cube_k",
    "deduce_layout",
    "get_ping_pong",
    "pto_macro_matmul",
    "resolve_acc_mode",
]
