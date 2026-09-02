#!/usr/bin/env python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""PTODSL TileLib templates for pipe push (producer side)."""

from __future__ import annotations

from ptodsl import pto
import ptodsl.tilelib as tilelib
from ._pushpop import (
    _is_c2v, _is_c2v_ub, _is_c2v_gm, _is_v2c_gm, _is_v2c_mat,
    _is_v2c_ctrl, _is_both, _is_both_gm,
    _const_i32, _as_value, _as_i32,
    _should_wait_free,
    _byte_addr, _add_byte_offset,
    _SPLIT_NUM,
    TileSplitAxis, Direction,
    _pipe_from_op,
)
from dataclasses import replace
from ptodsl._surface_values import wrap_surface_value


_VEC_CORE_ID_OFFSET = 16

# (allocate_wait_pipe, record_pipe)
_PUSH_FLAG = {
    Direction.C2V:      ("PIPE_FIX",  "PIPE_FIX"),
    Direction.C2V_UB:   ("PIPE_FIX",  "PIPE_FIX"),
    Direction.C2V_GM:   ("PIPE_FIX",  "PIPE_FIX"),
    Direction.V2C:      ("PIPE_MTE3", "PIPE_MTE3"),
    Direction.V2C_GM:   ("PIPE_MTE3", "PIPE_MTE3"),
    Direction.V2C_MAT:  ("PIPE_MTE3", "PIPE_MTE3"),
    Direction.V2C_CTRL: ("PIPE_S",    "PIPE_S"),
    Direction.BOTH:     ("PIPE_FIX",  "PIPE_FIX"),
    Direction.BOTH_GM:  ("PIPE_FIX",  "PIPE_FIX"),
}


# ODD split modes have no move variant.
_SPLIT_MODE = {
    TileSplitAxis.TILE_NO_SPLIT: "single_mode_vec0",
    TileSplitAxis.TILE_UP_DOWN: "dual_mode_split_m",
    TileSplitAxis.TILE_LEFT_RIGHT: "dual_mode_split_n",
}


def _set_intra_block_by_split(pipe_name, flag_id, is_split):
    pto.set_intra_block(pipe_name, flag_id)
    if is_split:
        pto.set_intra_block(pipe_name, flag_id + _VEC_CORE_ID_OFFSET)


def _wait_intra_block_by_split(pipe_name, flag_id, is_split):
    pto.wait_intra_block(pipe_name, flag_id)
    if is_split:
        pto.wait_intra_block(pipe_name, flag_id + _VEC_CORE_ID_OFFSET)


def _allocate(direction, flag_id, is_split):
    wait_pipe = _PUSH_FLAG[direction][0]

    if _is_c2v(direction):
        _wait_intra_block_by_split(wait_pipe, flag_id + 1, is_split)
    elif _is_v2c_gm(direction) or _is_v2c_mat(direction):
        pto.wait_intra_block(wait_pipe, flag_id + 1)
    elif _is_both(direction):
        _wait_intra_block_by_split(wait_pipe, flag_id + 1, is_split)
        pto.wait_intra_block("PIPE_MTE3", flag_id + 3)
    elif _is_v2c_ctrl(direction):
        pto.wait_intra_block(wait_pipe, flag_id + 1)
    elif _is_both_gm(direction):
        _wait_intra_block_by_split(wait_pipe, flag_id + 1, is_split)
        pto.wait_intra_block("PIPE_MTE3", flag_id + 3)


def _record(direction, flag_id, is_split):
    record_pipe = _PUSH_FLAG[direction][1]

    if _is_c2v(direction):
        _set_intra_block_by_split(record_pipe, flag_id, is_split)
    elif _is_v2c_gm(direction) or _is_v2c_mat(direction):
        pto.set_intra_block(record_pipe, flag_id)
    elif _is_both(direction):
        _set_intra_block_by_split(record_pipe, flag_id, is_split)
        pto.set_intra_block("PIPE_MTE3", flag_id + 2)
    elif _is_v2c_ctrl(direction):
        pto.set_intra_block(record_pipe, flag_id)
    elif _is_both_gm(direction):
        _set_intra_block_by_split(record_pipe, flag_id, is_split)
        pto.set_intra_block("PIPE_MTE3", flag_id + 2)


def _push_acc2_vec_fiFo(pipe, tile):
    tile_index = pipe.get_tile_id()

    # alloc_tile needs the static shape; dynamic sizes go via valid_shape.
    prod_m, prod_n = tile.shape
    valid_rows = tile.valid_shape[0]
    valid_cols = tile.valid_shape[1]

    if pipe.split == TileSplitAxis.TILE_UP_DOWN:
        cons_m = prod_m // _SPLIT_NUM
    else:
        cons_m = prod_m
    if pipe.split == TileSplitAxis.TILE_LEFT_RIGHT:
        cons_n = prod_n // _SPLIT_NUM
    else:
        cons_n = prod_n
    if pipe.split == TileSplitAxis.TILE_UP_DOWN:
        v_rows = (valid_rows // _SPLIT_NUM)
    else:
        v_rows = valid_rows
    if pipe.split == TileSplitAxis.TILE_LEFT_RIGHT:
        v_cols = (valid_cols // _SPLIT_NUM)
    else:
        v_cols = valid_cols

    entry_base = ((tile_index % pipe.slot_num) * pipe.slot_size)
    entry_base = (entry_base + pipe.entry_offset_bytes)

    slot_addr = _byte_addr(pipe.slot_ptr, entry_base)

    vec_tile = pto.alloc_tile(shape=[cons_m, cons_n], dtype=tile.dtype,
                              valid_shape=[v_rows, v_cols],
                              addr=slot_addr)
    if pipe.split not in (TileSplitAxis.TILE_UP_DOWN_ODD, TileSplitAxis.TILE_LEFT_RIGHT_ODD):
        mode = _SPLIT_MODE.get(pipe.split, "single_mode_vec0")
        pto.tmov(tile, vec_tile, mode=mode)


def _push_acc2_gm_fiFo(pipe, tile):
    tile_index = pipe.get_tile_id()

    valid_r = tile.valid_shape[0]
    valid_c = tile.valid_shape[1]
    entry_base = ((tile_index % pipe.slot_num) * pipe.slot_size)
    entry_base = (entry_base + pipe.entry_offset_bytes)

    gm_ptr = _add_byte_offset(pipe.slot_ptr, entry_base)

    if pipe.en_unit_flag:
        pto.mte_l0c_gm(
            tile.as_ptr(),
            gm_ptr,
            valid_r,
            valid_c,
            tile.shape[0],
            valid_c,
            0,
            0,
            layout="nz2nd",
            unit_flag="check_and_clear",
        )
    else:
        pto.mte_l0c_gm(
            tile.as_ptr(),
            gm_ptr,
            valid_r,
            valid_c,
            tile.shape[0],
            valid_c,
            0,
            0,
            layout="nz2nd",
        )


def _push_vec2_mat_fiFo(pipe, tile, sub_block_id=0):
    tile_index = pipe.get_tile_id()

    prod_m, prod_n = tile.shape
    valid_rows = tile.valid_shape[0]
    valid_cols = tile.valid_shape[1]

    if pipe.split == TileSplitAxis.TILE_UP_DOWN:
        cons_m = prod_m * _SPLIT_NUM
    else:
        cons_m = prod_m
    if pipe.split == TileSplitAxis.TILE_LEFT_RIGHT:
        cons_n = prod_n * _SPLIT_NUM
    else:
        cons_n = prod_n
    if pipe.split == TileSplitAxis.TILE_UP_DOWN:
        v_rows = (valid_rows * _SPLIT_NUM)
    else:
        v_rows = valid_rows
    if pipe.split == TileSplitAxis.TILE_LEFT_RIGHT:
        v_cols = (valid_cols * _SPLIT_NUM)
    else:
        v_cols = valid_cols

    entry_base = ((tile_index % pipe.slot_num) * pipe.slot_size)
    entry_base = (entry_base + pipe.entry_offset_bytes)

    mat_addr = _byte_addr(pipe.slot_ptr, entry_base)

    sbid = _as_i32(sub_block_id)
    if pipe.split == TileSplitAxis.TILE_NO_SPLIT:
        row_index = 0
        col_index = 0
    elif pipe.split == TileSplitAxis.TILE_UP_DOWN:
        row_index = (_as_value(valid_rows) * sbid)
        col_index = 0
    elif pipe.split == TileSplitAxis.TILE_LEFT_RIGHT:
        row_index = 0
        col_index = (_as_value(valid_cols) * sbid)

    mat_tile = pto.alloc_tile(shape=[cons_m, cons_n], dtype=tile.dtype,
                              memory_space="MAT",
                              valid_shape=[v_rows, v_cols],
                              blayout="ColMajor", slayout="RowMajor",
                              addr=mat_addr)
    if pipe.split not in (TileSplitAxis.TILE_UP_DOWN_ODD, TileSplitAxis.TILE_LEFT_RIGHT_ODD):
        pto.tinsert(tile, mat_tile, row_index, col_index)


def _push_vec2_gm_fiFo(pipe, tile, sub_block_id=0):
    tile_index = pipe.get_tile_id()

    gm_valid_r = tile.valid_shape[0]
    gm_valid_c = tile.valid_shape[1]
    elem_bytes = pto.bytewidth(tile.dtype)

    if pipe.split == TileSplitAxis.TILE_LEFT_RIGHT:
        gm_stride_r = (gm_valid_c * _SPLIT_NUM)
    else:
        gm_stride_r = gm_valid_c

    entry_base = ((tile_index % pipe.slot_num) * pipe.slot_size)

    sbid = _as_i32(sub_block_id)
    if pipe.split == TileSplitAxis.TILE_NO_SPLIT:
        sub_aiv_offset = _const_i32(0)
    elif pipe.split == TileSplitAxis.TILE_UP_DOWN:
        sub_aiv_offset = (((sbid * _as_value(gm_valid_r)) * _as_value(gm_valid_c)) * _as_value(elem_bytes))
    else:
        sub_aiv_offset = ((sbid * _as_value(gm_valid_c)) * _as_value(elem_bytes))

    total_offset = (entry_base + sub_aiv_offset)
    total_offset = (total_offset + pipe.entry_offset_bytes)

    gm_ptr = _add_byte_offset(pipe.slot_ptr, total_offset)

    prod_n = tile.shape[1]
    len_burst = (gm_valid_c * elem_bytes)
    n_burst = gm_valid_r
    ub_stride = (prod_n * elem_bytes)
    gm_stride = (gm_stride_r * elem_bytes)
    pto.mte_store(
        tile.as_ptr(),
        gm_ptr,
        len_burst,
        nburst=(n_burst, ub_stride, gm_stride),
    )


def _push_vec2_ctrl_fiFo(pipe, tile):
    tile_index = pipe.get_tile_id()

    slot_index = (tile_index % pipe.slot_num)

    entry_base = (slot_index * 4)
    entry_base = (entry_base + pipe.entry_offset_bytes)

    ctrl_ptr = _add_byte_offset(pipe.slot_ptr, entry_base)

    tile_ptr = tile.as_ptr()
    ctrl_signal = pto.load_scalar(tile_ptr, 0)
    pto.store_scalar(ctrl_ptr, 0, ctrl_signal)


def _push_vec2_fiFo_by_dir(pipe, tile, sub_block_id=0):
    direction = pipe.direction
    if _is_v2c_mat(direction) or _is_both(direction):
        _push_vec2_mat_fiFo(pipe, tile, sub_block_id)
    elif _is_v2c_gm(direction) or _is_both_gm(direction):
        _push_vec2_gm_fiFo(pipe, tile, sub_block_id)
    elif _is_v2c_ctrl(direction):
        _push_vec2_ctrl_fiFo(pipe, tile)


def _push(pipe, tile, sub_block_id):
    tile_loc = tile.memory_space

    direction = pipe.direction
    if tile_loc == 'acc':
        if _is_c2v_ub(direction) or _is_both(direction):
            _push_acc2_vec_fiFo(pipe, tile)
        elif _is_c2v_gm(direction):
            _push_acc2_gm_fiFo(pipe, tile)
    elif tile_loc == 'vec':
        _push_vec2_fiFo_by_dir(pipe, tile, sub_block_id)


def _emit_push_body(pipe, tile):
    tile_index = pipe.get_tile_id()
    is_allocate = pipe.get_allocate_status()
    need_wait = _should_wait_free(tile_index, pipe.slot_num)
    cond = (is_allocate & need_wait)
    with pto.if_(wrap_surface_value(cond)) as br:
        with br.then_:
            _allocate(pipe.direction, pipe.flag_id, pipe.is_split)

    if pipe.split == TileSplitAxis.TILE_NO_SPLIT:
        _push(pipe, tile, 0)
    else:
        _push(pipe, tile, pto.get_subblock_idx())
    pipe.bump_tile_index()

    is_record = pipe.get_record_status()
    with pto.if_(wrap_surface_value(is_record)) as br:
        with br.then_:
            _record(pipe.direction, pipe.flag_id, pipe.is_split)


def _emit_push_body_subblock(pipe, tile, sub_block_id):
    tile_index = pipe.get_tile_id()
    is_allocate = pipe.get_allocate_status()
    need_wait = _should_wait_free(tile_index, pipe.slot_num)
    cond = (is_allocate & need_wait)
    with pto.if_(wrap_surface_value(cond)) as br:
        with br.then_:
            _allocate(pipe.direction, pipe.flag_id, pipe.is_split)

    _push(pipe, tile, sub_block_id)
    pipe.bump_tile_index()

    is_record = pipe.get_record_status()
    with pto.if_(wrap_surface_value(is_record)) as br:
        with br.then_:
            _record(pipe.direction, pipe.flag_id, pipe.is_split)


def _emit_push_body_global(pipe):
    assert _is_c2v_gm(pipe.direction) or _is_v2c_gm(pipe.direction) or _is_both_gm(pipe.direction), \
        "TPUSH with GlobalTensor is only supported by GM FIFO directions on A5."

    _record(pipe.direction, pipe.flag_id, pipe.is_split)


def _emit_push_body_config(pipe, tile):
    pipe = replace(pipe, split=0)

    tile_index = pipe.get_tile_id()
    is_allocate = pipe.get_allocate_status()
    need_wait = _should_wait_free(tile_index, pipe.slot_num)
    cond = (is_allocate & need_wait)
    with pto.if_(wrap_surface_value(cond)) as br:
        with br.then_:
            _allocate(pipe.direction, pipe.flag_id, pipe.is_split)

    _push(pipe, tile, 0)
    pipe.bump_tile_index()

    is_record = pipe.get_record_status()
    with pto.if_(wrap_surface_value(is_record)) as br:
        with br.then_:
            _record(pipe.direction, pipe.flag_id, pipe.is_split)


def _emit_push_body_simplified(pipe, tile):
    pipe = replace(pipe, split=0)

    is_allocate = pipe.get_allocate_status()
    with pto.if_(wrap_surface_value(is_allocate)) as br:
        with br.then_:
            _allocate(pipe.direction, pipe.flag_id, pipe.is_split)

    _push(pipe, tile, 0)

    is_record = pipe.get_record_status()
    with pto.if_(wrap_surface_value(is_record)) as br:
        with br.then_:
            _record(pipe.direction, pipe.flag_id, pipe.is_split)


def _entry_is_tile(operand_kinds, **_):
    return bool(operand_kinds) and operand_kinds[0] == "tile"


def _entry_is_view(operand_kinds, **_):
    return bool(operand_kinds) and operand_kinds[0] == "view"


def _slot_ptr_is_gm(slot_ptr_memory_space, **_):
    return slot_ptr_memory_space == "gm"


def _slot_ptr_is_ub(slot_ptr_memory_space, **_):
    return slot_ptr_memory_space == "ub"


def _slot_ptr_is_mat(slot_ptr_memory_space, **_):
    return slot_ptr_memory_space == "mat"

@tilelib.tile_template(
    op="pto.tpush_to_aiv",
    target="a5",
    name="template_tpush_to_aiv",
    dtypes=(),
    iteration_axis="none",
    op_engine="cube",
    op_class="other",
    constraints=(_entry_is_tile, _slot_ptr_is_gm),
    id=0,
    loop_depth=1,
    is_post_update=False,
    tags=("pipe", "push", "c2v"),
)
def template_tpush_to_aiv(tile: pto.Tile, state: pto.Struct, slot_ptr):
    _emit_push_body(_pipe_from_op(Direction.C2V, state, slot_ptr), tile)


@tilelib.tile_template(
    op="pto.tpush_to_aiv",
    target="a5",
    name="template_tpush_to_aiv_gm_view",
    dtypes=(),
    iteration_axis="none",
    op_engine="cube",
    op_class="other",
    constraints=(_entry_is_view, _slot_ptr_is_gm),
    id=3,
    loop_depth=1,
    is_post_update=False,
    tags=("pipe", "push", "c2v", "gm", "view"),
)
def template_tpush_to_aiv_gm_view(gm_tensor, state: pto.Struct, slot_ptr):
    _emit_push_body_global(_pipe_from_op(Direction.C2V, state, slot_ptr))


@tilelib.tile_template(
    op="pto.tpush_to_aic",
    target="a5",
    name="template_tpush_to_aic",
    dtypes=(),
    iteration_axis="none",
    op_engine="vector",
    op_class="other",
    constraints=(_entry_is_tile, _slot_ptr_is_gm),
    id=0,
    loop_depth=1,
    is_post_update=False,
    tags=("pipe", "push", "v2c"),
)
def template_tpush_to_aic(tile: pto.Tile, state: pto.Struct, slot_ptr):
    _emit_push_body(_pipe_from_op(Direction.V2C, state, slot_ptr), tile)


@tilelib.tile_template(
    op="pto.tpush_to_aic",
    target="a5",
    name="template_tpush_to_aic_gm_view",
    dtypes=(),
    iteration_axis="none",
    op_engine="vector",
    op_class="other",
    constraints=(_entry_is_view, _slot_ptr_is_gm),
    id=3,
    loop_depth=1,
    is_post_update=False,
    tags=("pipe", "push", "v2c", "gm", "view"),
)
def template_tpush_to_aic_gm_view(gm_tensor, state: pto.Struct, slot_ptr):
    _emit_push_body_global(_pipe_from_op(Direction.V2C, state, slot_ptr))


@tilelib.tile_template(
    op="pto.tpush_to_aic",
    target="a5",
    name="template_tpush_to_aic_subblock",
    dtypes=(),
    iteration_axis="none",
    op_engine="vector",
    op_class="other",
    constraints=(_entry_is_tile, _slot_ptr_is_gm),
    id=1,
    loop_depth=1,
    is_post_update=False,
    tags=("pipe", "push", "v2c", "subblock"),
)
def template_tpush_to_aic_subblock(tile: pto.Tile, sub_block_id, state: pto.Struct, slot_ptr):
    _emit_push_body_subblock(_pipe_from_op(Direction.V2C, state, slot_ptr), tile,
                              _as_i32(sub_block_id))


@tilelib.tile_template(
    op="pto.tpush_to_aiv",
    target="a5",
    name="template_tpush_to_aiv_local",
    dtypes=(),
    iteration_axis="none",
    op_engine="cube",
    op_class="other",
    constraints=(_slot_ptr_is_ub,),
    id=1,
    loop_depth=1,
    is_post_update=False,
    tags=("pipe", "push", "c2v", "local"),
)
def template_tpush_to_aiv_local(tile: pto.Tile, state: pto.Struct, slot_ptr):
    _emit_push_body(_pipe_from_op(Direction.C2V_UB, state, slot_ptr), tile)


@tilelib.tile_template(
    op="pto.tpush_to_aic",
    target="a5",
    name="template_tpush_to_aic_local",
    dtypes=(),
    iteration_axis="none",
    op_engine="vector",
    op_class="other",
    constraints=(_slot_ptr_is_mat,),
    id=2,
    loop_depth=1,
    is_post_update=False,
    tags=("pipe", "push", "v2c", "local"),
)
def template_tpush_to_aic_local(tile: pto.Tile, state: pto.Struct, slot_ptr):
    _emit_push_body(_pipe_from_op(Direction.V2C_MAT, state, slot_ptr), tile)
