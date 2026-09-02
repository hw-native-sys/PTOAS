#!/usr/bin/env python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""PTODSL TileLib templates for pipe pop (consumer side)."""

from __future__ import annotations

from dataclasses import replace

from ptodsl import pto
import ptodsl.tilelib as tilelib
from ._pushpop import (
    _i32,
    _is_c2v, _is_v2c, _is_c2v_ub, _is_c2v_gm, _is_v2c_gm, _is_v2c_mat,
    _is_v2c_ctrl, _is_both, _is_both_gm,
    _const_i32, _as_value, _as_i32,
    _should_notify_free, _pipe_from_op,
    _byte_addr, _add_byte_offset,
    TileSplitAxis, Direction,
)
from ptodsl._surface_values import unwrap_surface_value, wrap_surface_value
from ptoas.mlir.dialects import arith


_VEC_CORE_ID_OFFSET = 16

# (wait_pipe, free_pipe)
_POP_FLAG = {
    Direction.C2V:      ("PIPE_MTE2", "PIPE_MTE2"),
    Direction.C2V_UB:   ("PIPE_V",    "PIPE_V"),
    Direction.C2V_GM:   ("PIPE_MTE2", "PIPE_MTE2"),
    Direction.V2C:      ("PIPE_MTE2", "PIPE_MTE1"),
    Direction.V2C_GM:   ("PIPE_MTE2", "PIPE_MTE1"),
    Direction.V2C_MAT:  ("PIPE_MTE1", "PIPE_MTE1"),
    Direction.V2C_CTRL: ("PIPE_S",    "PIPE_S"),
    Direction.BOTH:     ("PIPE_V",    "PIPE_V"),
    Direction.BOTH_GM:  ("PIPE_MTE2", "PIPE_MTE2"),
}


def _resolve_pop_direction(direction):
    if direction == Direction.C2V:
        return Direction.C2V_GM
    if direction == Direction.V2C:
        return Direction.V2C_GM
    return direction


def _set_intra_block_by_split(pipe_name, flag_id, is_split):
    pto.set_intra_block(pipe_name, flag_id)
    if is_split:
        pto.set_intra_block(pipe_name, flag_id + _VEC_CORE_ID_OFFSET)


def _wait_intra_block_by_split(pipe_name, flag_id, is_split):
    pto.wait_intra_block(pipe_name, flag_id)
    if is_split:
        pto.wait_intra_block(pipe_name, flag_id + _VEC_CORE_ID_OFFSET)


def _wait(direction, flag_id, is_split):
    wait_pipe = _POP_FLAG[direction][0]

    if _is_c2v(direction):
        # c2v waits are unsplit
        pto.wait_intra_block(wait_pipe, flag_id)
    elif _is_v2c(direction):
        _wait_intra_block_by_split(wait_pipe, flag_id, is_split)
    elif direction == Direction.BOTH:
        pto.wait_intra_block("PIPE_V", flag_id)
        _wait_intra_block_by_split("PIPE_MTE1", flag_id + 2, is_split)
    elif direction == Direction.BOTH_GM:
        pto.wait_intra_block("PIPE_MTE2", flag_id + 1)
        _wait_intra_block_by_split("PIPE_MTE2", flag_id + 2, is_split)


def _free(direction, flag_id, is_split):
    free_pipe = _POP_FLAG[direction][1]

    if _is_c2v(direction):
        pto.set_intra_block(free_pipe, flag_id + 1)
    elif _is_v2c(direction):
        _set_intra_block_by_split(free_pipe, flag_id + 1, is_split)
    elif direction == Direction.BOTH:
        pto.set_intra_block("PIPE_V", flag_id + 1)
        _set_intra_block_by_split("PIPE_MTE1", flag_id + 3, is_split)
    elif direction == Direction.BOTH_GM:
        pto.set_intra_block("PIPE_MTE2", flag_id + 1)
        _set_intra_block_by_split("PIPE_MTE1", flag_id + 3, is_split)


def _pop_tile_from_vec_fiFo(pipe):
    tile_index = pipe.get_tile_id()

    entry_base = ((tile_index % pipe.slot_num) * pipe.slot_size)
    entry_base = (entry_base + pipe.entry_offset_bytes)

    slot_addr = _byte_addr(pipe.slot_ptr, entry_base)

    tile = pto.alloc_tile(shape=[pipe.cons_rows, pipe.cons_cols],
                           dtype=pipe.cons_dtype,
                           memory_space="vec",
                           valid_shape=[pipe.cons_rows, pipe.cons_cols],
                           addr=slot_addr)
    return tile


def _pop_vec_tile_from_gm_fiFo(pipe, valid_row=None, valid_col=None,
                               sub_block_id=None):
    tile_index = pipe.get_tile_id()

    elem_bytes = pipe.elem_bytes
    entry_base = ((tile_index % pipe.slot_num) * pipe.slot_size)

    if sub_block_id is None:
        sbid = _as_value(0)
    else:
        sbid = _as_i32(sub_block_id)
    cons_rows = pipe.cons_rows
    cons_cols = pipe.cons_cols
    off_rows = valid_row if valid_row is not None else cons_rows
    off_cols = valid_col if valid_col is not None else cons_cols
    if pipe.split == TileSplitAxis.TILE_NO_SPLIT:
        sub_aiv_offset = _const_i32(0)
    elif pipe.split == TileSplitAxis.TILE_UP_DOWN:
        sub_aiv_offset = (((sbid * _as_value(off_rows)) * _as_value(off_cols)) * _as_value(elem_bytes))
    elif pipe.split == TileSplitAxis.TILE_UP_DOWN_ODD:
        sub_aiv_offset = (((sbid * (_as_value(off_rows) + sbid)) * _as_value(off_cols)) * _as_value(elem_bytes))
    elif pipe.split == TileSplitAxis.TILE_LEFT_RIGHT:
        sub_aiv_offset = ((sbid * _as_value(off_cols)) * _as_value(elem_bytes))
    elif pipe.split == TileSplitAxis.TILE_LEFT_RIGHT_ODD:
        sub_aiv_offset = ((sbid * (_as_value(off_cols) + sbid)) * _as_value(elem_bytes))
    else:
        sub_aiv_offset = _const_i32(0)

    total_offset = (entry_base + sub_aiv_offset)
    total_offset = (total_offset + pipe.entry_offset_bytes)

    gm_slot_ptr = _add_byte_offset(pipe.slot_ptr, total_offset)

    off_cols_v = _as_value(off_cols)
    if pipe.split == TileSplitAxis.TILE_LEFT_RIGHT:
        gm_stride_r = (2 * off_cols_v)
    elif pipe.split == TileSplitAxis.TILE_LEFT_RIGHT_ODD:
        two_sbid = (2 * sbid)
        gm_stride_r = (((2 * off_cols_v) + two_sbid) + -1)
    else:
        gm_stride_r = _as_value(off_cols)

    v_shape = [valid_row if valid_row is not None else cons_rows,
               valid_col if valid_col is not None else cons_cols]
    ub_tile = pto.alloc_tile(shape=[cons_rows, cons_cols], dtype=pipe.cons_dtype,
                             valid_shape=v_shape)
    len_burst = (off_cols_v * _as_value(elem_bytes))
    ub_stride = (_as_value(cons_cols) * _as_value(elem_bytes))
    gm_stride = (gm_stride_r * _as_value(elem_bytes))
    pto.mte_load(gm_slot_ptr, ub_tile.as_ptr(), 0, len_burst,
                 nburst=(_as_value(off_rows), gm_stride, ub_stride))
    return ub_tile


def _pop_tile_from_mat_fiFo(pipe):
    tile_index = pipe.get_tile_id()

    entry_base = ((tile_index % pipe.slot_num) * pipe.slot_size)
    entry_base = (entry_base + pipe.entry_offset_bytes)

    mat_addr = _byte_addr(pipe.slot_ptr, entry_base)

    tile = pto.alloc_tile(shape=[pipe.cons_rows, pipe.cons_cols],
                           dtype=pipe.cons_dtype,
                           memory_space="mat",
                           valid_shape=[pipe.cons_rows, pipe.cons_cols],
                           addr=mat_addr)
    return tile


def _pop_mat_tile_from_gm_fiFo(pipe, valid_row=None, valid_col=None):
    tile_index = pipe.get_tile_id()

    entry_base = ((tile_index % pipe.slot_num) * pipe.slot_size)
    entry_base = (entry_base + pipe.entry_offset_bytes)
    gm_ptr = _add_byte_offset(pipe.slot_ptr, entry_base)

    cons_rows = pipe.cons_rows
    cons_cols = pipe.cons_cols
    elem_bytes = pipe.elem_bytes
    off_rows = valid_row if valid_row is not None else cons_rows
    off_cols = valid_col if valid_col is not None else cons_cols
    v_shape = [valid_row if valid_row is not None else cons_rows,
               valid_col if valid_col is not None else cons_cols]
    ub_tile = pto.alloc_tile(shape=[cons_rows, cons_cols], dtype=pipe.cons_dtype,
                             valid_shape=v_shape)
    off_cols_v = _as_value(off_cols)
    len_burst = (off_cols_v * _as_value(elem_bytes))
    ub_stride = (off_cols_v * _as_value(elem_bytes))
    gm_stride = (off_cols_v * _as_value(elem_bytes))
    pto.mte_load(gm_ptr, ub_tile.as_ptr(), 0, len_burst,
                 nburst=(_as_value(off_rows), gm_stride, ub_stride))
    return ub_tile


def _pop_ctrl_from_ctrl_fiFo(pipe):
    tile_index = pipe.get_tile_id()

    slot_index = (tile_index % pipe.slot_num)

    entry_base = (slot_index * 4)
    entry_base = (entry_base + pipe.entry_offset_bytes)

    ctrl_ptr = _add_byte_offset(pipe.slot_ptr, entry_base)

    # slot_base_ptr is ptr<f32>: bitcast the loaded scalar to i32 for the
    # comparison against 1.
    ctrl_val = pto.load_scalar(ctrl_ptr, 0)
    ctrl_i32 = arith.BitcastOp(_i32(), unwrap_surface_value(ctrl_val)).result
    ctrl_signal = arith.CmpIOp(
        arith.CmpIPredicate.eq, ctrl_i32, _const_i32(1)
    ).result
    return ctrl_signal


def _pop(pipe, tile_type="vec", valid_row=None, valid_col=None, sub_block_id=None):
    """Returns (tile, needs_free): local aliasing pops need no free signal,
    GM copies must free the slot."""
    direction = pipe.direction
    if tile_type == "ctrl":
        tile = _pop_ctrl_from_ctrl_fiFo(pipe)
        return tile, False
    if tile_type == "vec":
        if _is_c2v(direction) or _is_both(direction):
            if _is_c2v_ub(direction) or _is_both(direction):
                tile = _pop_tile_from_vec_fiFo(pipe)
                return tile, False
            else:
                tile = _pop_vec_tile_from_gm_fiFo(pipe,
                                                    valid_row=valid_row, valid_col=valid_col,
                                                    sub_block_id=sub_block_id)
                return tile, True
        else:
            tile = _pop_vec_tile_from_gm_fiFo(pipe,
                                                valid_row=valid_row, valid_col=valid_col,
                                                sub_block_id=sub_block_id)
            return tile, True
    elif tile_type == "mat":
        if _is_v2c_mat(direction) or _is_v2c_gm(direction) or _is_both(direction):
            if _is_v2c_mat(direction) or _is_both(direction):
                tile = _pop_tile_from_mat_fiFo(pipe)
                return tile, False
            else:
                tile = _pop_mat_tile_from_gm_fiFo(pipe,
                                                    valid_row=valid_row, valid_col=valid_col)
                return tile, True
        else:
            tile = _pop_mat_tile_from_gm_fiFo(pipe,
                                                valid_row=valid_row, valid_col=valid_col)
            return tile, True
    else:
        raise ValueError(f"Unsupported tile type for pop: {tile_type}")


def _emit_pop_body(pipe, valid_row=None, valid_col=None):
    pipe = replace(pipe, direction=_resolve_pop_direction(pipe.direction))

    is_wait = pipe.get_wait_status()
    with pto.if_(wrap_surface_value(is_wait)) as br:
        with br.then_:
            _wait(pipe.direction, pipe.flag_id, pipe.is_split)

    if _is_v2c_ctrl(pipe.direction):
        tile_type = "ctrl"
    elif _is_v2c_mat(pipe.direction) or _is_v2c_gm(pipe.direction):
        tile_type = "mat"
    else:
        tile_type = "vec"
    if pipe.split == TileSplitAxis.TILE_NO_SPLIT:
        sbid_arg = 0
    else:
        sbid_arg = pto.get_subblock_idx()
    tile, req_free = _pop(pipe, tile_type=tile_type,
                           valid_row=valid_row, valid_col=valid_col,
                           sub_block_id=sbid_arg)

    is_free_status = pipe.get_free_status()
    tile_index = pipe.get_tile_id()
    notify = _should_notify_free(tile_index, pipe.slot_num)

    # slot_num == 1 returns a non-zero i32; convert to i1 for the conjunction.
    if pipe.slot_num <= 1:
        notify_i1 = _struct_get_bool_from_val(notify)
    else:
        notify_i1 = notify

    pipe.bump_tile_index()

    if req_free:
        cond = (is_free_status & notify_i1)
        with pto.if_(wrap_surface_value(cond)) as br:
            with br.then_:
                _free(pipe.direction, pipe.flag_id, pipe.is_split)

    return tile


def _emit_pop_body_subblock(pipe, sub_block_id):
    pipe = replace(pipe, direction=_resolve_pop_direction(pipe.direction))

    is_wait = pipe.get_wait_status()
    with pto.if_(wrap_surface_value(is_wait)) as br:
        with br.then_:
            _wait(pipe.direction, pipe.flag_id, pipe.is_split)

    if _is_v2c_ctrl(pipe.direction):
        tile_type = "ctrl"
    elif _is_v2c_mat(pipe.direction) or _is_v2c_gm(pipe.direction):
        tile_type = "mat"
    else:
        tile_type = "vec"
    tile, req_free = _pop(pipe, tile_type=tile_type, sub_block_id=sub_block_id)

    is_free_status = pipe.get_free_status()
    tile_index = pipe.get_tile_id()
    notify = _should_notify_free(tile_index, pipe.slot_num)

    if pipe.slot_num <= 1:
        notify_i1 = _struct_get_bool_from_val(notify)
    else:
        notify_i1 = notify

    pipe.bump_tile_index()

    if req_free:
        cond = (is_free_status & notify_i1)
        with pto.if_(wrap_surface_value(cond)) as br:
            with br.then_:
                _free(pipe.direction, pipe.flag_id, pipe.is_split)

    return tile


def _struct_get_bool_from_val(val):
    zero = _const_i32(0)
    return arith.CmpIOp(arith.CmpIPredicate.ne, val, zero).result


def _emit_pop_body_global(pipe, valid_row=None, valid_col=None, sub_block_id=None):
    pipe = replace(pipe, direction=_resolve_pop_direction(pipe.direction))

    _wait(pipe.direction, pipe.flag_id, pipe.is_split)

    tile_index = pipe.get_tile_id()

    elem_bytes = pipe.elem_bytes
    slot_offset = ((tile_index % pipe.slot_num) * pipe.slot_size)
    slot_offset = (slot_offset + pipe.entry_offset_bytes)

    if sub_block_id is not None:
        sbid = _as_i32(sub_block_id)
        if pipe.split == TileSplitAxis.TILE_UP_DOWN:
            sub_aiv_offset = (((sbid * _as_value(pipe.cons_rows)) * _as_value(pipe.cons_cols)) * _as_value(elem_bytes))
        elif pipe.split == TileSplitAxis.TILE_LEFT_RIGHT:
            sub_aiv_offset = ((sbid * _as_value(pipe.cons_cols)) * _as_value(elem_bytes))
        else:
            sub_aiv_offset = _const_i32(0)
    else:
        sub_aiv_offset = _const_i32(0)

    if _is_c2v_gm(pipe.direction):
        total_offset = (slot_offset + sub_aiv_offset)
    elif _is_v2c_gm(pipe.direction):
        total_offset = slot_offset
    else:
        total_offset = (slot_offset + sub_aiv_offset)

    gm_ptr = _add_byte_offset(pipe.slot_ptr, total_offset)

    pipe.bump_tile_index()

    # 5D ND view: ExpandTileOp runs after PTOCanonicalizeIR, so a 2D view
    # produced here would never be promoted (rank mismatch at the consumer).
    tile = pto.make_tensor_view(gm_ptr,
                                  shape=[1, 1, 1, pipe.cons_rows, pipe.cons_cols],
                                  strides=[pipe.cons_rows * pipe.cons_cols,
                                           pipe.cons_rows * pipe.cons_cols,
                                           pipe.cons_rows * pipe.cons_cols,
                                           pipe.cons_cols, 1])
    return tile


def _emit_pop_body_simplified(pipe):
    pipe = replace(pipe, direction=_resolve_pop_direction(pipe.direction))

    is_wait = pipe.get_wait_status()
    with pto.if_(wrap_surface_value(is_wait)) as br:
        with br.then_:
            _wait(pipe.direction, pipe.flag_id, pipe.is_split)

    if _is_v2c_ctrl(pipe.direction):
        tile_type = "ctrl"
    elif _is_v2c_mat(pipe.direction) or _is_v2c_gm(pipe.direction):
        tile_type = "mat"
    else:
        tile_type = "vec"
    tile, req_free = _pop(pipe, tile_type=tile_type)

    is_free_status = pipe.get_free_status()
    if req_free:
        with pto.if_(wrap_surface_value(is_free_status)) as br:
            with br.then_:
                _free(pipe.direction, pipe.flag_id, pipe.is_split)

    return tile


@tilelib.tile_template(
    op="pto.tpop_from_aic",
    target="a5",
    name="template_tpop_from_aic",
    dtypes=(),
    iteration_axis="none",
    op_engine="vector",
    op_class="other",
    constraints=(),
    memory_spaces=("gm",),
    id=0,
    loop_depth=1,
    is_post_update=False,
    tags=("pipe", "pop", "c2v"),
)
def template_tpop_from_aic(state: pto.Struct, slot_ptr):
    return _emit_pop_body_global(_pipe_from_op(Direction.C2V, state, slot_ptr))


@tilelib.tile_template(
    op="pto.tpop_from_aic",
    target="a5",
    name="template_tpop_from_aic_dyn",
    dtypes=(),
    iteration_axis="none",
    op_engine="vector",
    op_class="other",
    constraints=(),
    memory_spaces=("gm",),
    id=1,
    loop_depth=1,
    is_post_update=False,
    tags=("pipe", "pop", "c2v", "dynamic"),
)
def template_tpop_from_aic_dyn(valid_row, valid_col, state: pto.Struct, slot_ptr):
    return _emit_pop_body_global(_pipe_from_op(Direction.C2V, state, slot_ptr),
                                 valid_row, valid_col)


@tilelib.tile_template(
    op="pto.tpop_from_aiv",
    target="a5",
    name="template_tpop_from_aiv",
    dtypes=(),
    iteration_axis="none",
    op_engine="cube",
    op_class="other",
    constraints=(),
    memory_spaces=("gm",),
    id=0,
    loop_depth=1,
    is_post_update=False,
    tags=("pipe", "pop", "v2c"),
)
def template_tpop_from_aiv(state: pto.Struct, slot_ptr):
    return _emit_pop_body_global(_pipe_from_op(Direction.V2C, state, slot_ptr))


@tilelib.tile_template(
    op="pto.tpop_from_aiv",
    target="a5",
    name="template_tpop_from_aiv_dyn",
    dtypes=(),
    iteration_axis="none",
    op_engine="cube",
    op_class="other",
    constraints=(),
    memory_spaces=("gm",),
    id=1,
    loop_depth=1,
    is_post_update=False,
    tags=("pipe", "pop", "v2c", "dynamic"),
)
def template_tpop_from_aiv_dyn(valid_row, valid_col, state: pto.Struct, slot_ptr):
    return _emit_pop_body_global(_pipe_from_op(Direction.V2C, state, slot_ptr),
                                 valid_row, valid_col)


@tilelib.tile_template(
    op="pto.tpop_from_aic",
    target="a5",
    name="template_tpop_from_aic_subblock",
    dtypes=(),
    iteration_axis="none",
    op_engine="vector",
    op_class="other",
    constraints=(),
    memory_spaces=("gm",),
    id=2,
    loop_depth=1,
    is_post_update=False,
    tags=("pipe", "pop", "c2v", "subblock"),
)
def template_tpop_from_aic_subblock(sub_block_id, state: pto.Struct, slot_ptr):
    return _emit_pop_body_global(_pipe_from_op(Direction.C2V, state, slot_ptr),
                                  sub_block_id=_as_i32(sub_block_id))


@tilelib.tile_template(
    op="pto.tpop_from_aic",
    target="a5",
    name="template_tpop_from_aic_local",
    dtypes=(),
    iteration_axis="none",
    op_engine="vector",
    op_class="other",
    constraints=(),
    memory_spaces=("ub",),
    id=3,
    loop_depth=1,
    is_post_update=False,
    tags=("pipe", "pop", "c2v", "local"),
)
def template_tpop_from_aic_local(state: pto.Struct, slot_ptr):
    return _emit_pop_body(_pipe_from_op(Direction.C2V_UB, state, slot_ptr))


@tilelib.tile_template(
    op="pto.tpop_from_aiv",
    target="a5",
    name="template_tpop_from_aiv_local",
    dtypes=(),
    iteration_axis="none",
    op_engine="cube",
    op_class="other",
    constraints=(),
    memory_spaces=("mat",),
    id=3,
    loop_depth=1,
    is_post_update=False,
    tags=("pipe", "pop", "v2c", "local"),
)
def template_tpop_from_aiv_local(state: pto.Struct, slot_ptr):
    return _emit_pop_body(_pipe_from_op(Direction.V2C_MAT, state, slot_ptr))
