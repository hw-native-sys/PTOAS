#!/usr/bin/env python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""Shared helpers for the pipe push/pop templates (tpush.py / tpop.py)."""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum, StrEnum

from ptodsl import pto
from ptodsl._surface_values import _SurfaceValue, unwrap_surface_value, wrap_surface_value
from ptoas.mlir.dialects import arith
from ptoas.mlir.dialects import pto as _pto
from ptoas.mlir.ir import IntegerType


def _i32():
    return IntegerType.get_signless(32)


def _i8():
    return IntegerType.get_signless(8)


def _i64():
    return IntegerType.get_signless(64)


def _byte_addr(ptr, byte_offset):
    base = _pto.PtrToIntOp(unwrap_surface_value(ptr)).result
    off = arith.ExtSIOp(_i64(), unwrap_surface_value(_as_value(byte_offset))).result
    return arith.AddIOp(base, off).result


def _add_byte_offset(ptr, byte_offset):
    raw = unwrap_surface_value(ptr)
    return wrap_surface_value(_pto.CastPtrOp(raw.type, _byte_addr(ptr, byte_offset)).result)


class Direction(StrEnum):
    C2V = "c2v"
    V2C = "v2c"
    BOTH = "both"
    V2C_CTRL = "v2c_ctrl"
    C2V_GM = "c2v_gm"
    V2C_GM = "v2c_gm"
    BOTH_GM = "both_gm"
    C2V_UB = "c2v_ub"
    V2C_MAT = "v2c_mat"


# (family, kind): family = dataflow direction, kind = storage/sync mode
_DIRECTION_PROP = {
    Direction.C2V:     ("c2v", "gm"),
    Direction.C2V_UB:  ("c2v", "ub"),
    Direction.C2V_GM:  ("c2v", "gm"),
    Direction.V2C:     ("v2c", "gm"),
    Direction.V2C_MAT: ("v2c", "mat"),
    Direction.V2C_GM:  ("v2c", "gm"),
    Direction.V2C_CTRL: ("v2c", "ctrl"),
    Direction.BOTH:    ("both", "local"),
    Direction.BOTH_GM: ("both", "gm"),
}

_DIRECTION_FAMILY = {d: f for d, (f, _) in _DIRECTION_PROP.items()}
_DIRECTION_KIND = {d: k for d, (_, k) in _DIRECTION_PROP.items()}


def _family(direction):
    return _DIRECTION_FAMILY[direction]


def _kind(direction):
    return _DIRECTION_KIND[direction]


def _is_c2v_ub(direction):
    return _kind(direction) == "ub"


def _is_v2c_mat(direction):
    return _kind(direction) == "mat"


def _is_both(direction):
    return _kind(direction) == "local"


def _is_v2c_ctrl(direction):
    return _kind(direction) == "ctrl"


def _is_c2v_gm(direction):
    return _family(direction) == "c2v" and _kind(direction) == "gm"


def _is_v2c_gm(direction):
    return _family(direction) == "v2c" and _kind(direction) == "gm"


def _is_both_gm(direction):
    return _family(direction) == "both" and _kind(direction) == "gm"


def _is_c2v(direction):
    return _is_c2v_gm(direction) or _is_c2v_ub(direction)


def _is_v2c(direction):
    return _is_v2c_gm(direction) or _is_v2c_mat(direction) or _is_v2c_ctrl(direction)


@dataclass
class TPipe:
    direction: Direction | str
    flag_id: int
    split: int  # TileSplitAxis
    slot_num: int
    slot_size: int  # bytes
    state: object
    slot_ptr: object
    en_unit_flag: int = 0
    elem_bytes: int = 4
    cons_rows: int = 16
    cons_cols: int = 32
    cons_dtype: object = None

    @property
    def is_split(self) -> int:
        return 1 if self.split != TileSplitAxis.TILE_NO_SPLIT else 0

    def get_tile_id(self):
        return _struct_get_i32(self.state, _STATE_TILE_INDEX)

    def get_sub_tile_id(self):
        return _struct_get_i32(self.state, _STATE_SUB_TILE_INDEX)

    def set_tile_id(self, t_index, sub_index):
        _struct_set_i32(self.state, _STATE_SUB_TILE_INDEX, _as_value(sub_index))
        _struct_set_i32(self.state, _STATE_TILE_INDEX, _as_value(t_index))

    def bump_tile_index(self):
        _struct_set_i32(self.state, _STATE_TILE_INDEX, self.get_tile_id() + 1)

    @property
    def entry_offset_bytes(self):
        return _struct_get_i32(self.state, _STATE_ENTRY_OFFSET)

    def set_entry_offset(self, offset):
        _struct_set_i32(self.state, _STATE_ENTRY_OFFSET, _as_value(offset))

    def get_record_status(self):
        return _struct_get_bool(self.state, _STATE_IS_RECORD)

    def set_record_status(self, value):
        _struct_set_bool(self.state, _STATE_IS_RECORD, value)

    def get_wait_status(self):
        # Aliases record status: producer "record" and consumer "wait" share a field.
        return _struct_get_bool(self.state, _STATE_IS_RECORD)

    def set_wait_status(self, value):
        _struct_set_bool(self.state, _STATE_IS_RECORD, value)

    def get_allocate_status(self):
        return _struct_get_bool(self.state, _STATE_IS_ALLOCATE)

    def set_allocate_status(self, value):
        _struct_set_bool(self.state, _STATE_IS_ALLOCATE, value)

    def get_free_status(self):
        # Aliases allocate status: producer "allocate" and consumer "free" share a field.
        return _struct_get_bool(self.state, _STATE_IS_ALLOCATE)

    def set_free_status(self, value):
        _struct_set_bool(self.state, _STATE_IS_ALLOCATE, value)


def _pipe_from_op(direction, state, slot_ptr) -> TPipe:
    split = pto.get_op_attr("split", 0)
    cons_dtype = pto.get_op_attr("cons_dtype", "f32")
    return TPipe(
        direction=direction,
        flag_id=pto.get_op_attr("id", 0),
        split=split,
        slot_num=pto.get_op_attr("slot_num", 2),
        slot_size=pto.get_op_attr("slot_size", 0),
        state=state,
        slot_ptr=slot_ptr,
        en_unit_flag=pto.get_op_attr("en_unit_flag", 0),
        elem_bytes=pto.get_op_attr("elem_bytes", 4),
        cons_rows=pto.get_op_attr("cons_rows", 16),
        cons_cols=pto.get_op_attr("cons_cols", 32),
        cons_dtype=getattr(pto, cons_dtype, pto.f32),
    )


# Shared producer/consumer state struct: !pto.struct<i32, i32, i8, i8, i32>
# (subTileIndex, tileIndex, isRecord/isWait, isAllocate/isFree, entryOffset)
_STATE_SUB_TILE_INDEX = 0
_STATE_TILE_INDEX = 1
_STATE_IS_RECORD = 2
_STATE_IS_ALLOCATE = 3
_STATE_ENTRY_OFFSET = 4


def _struct_get_i32(state, path):
    return wrap_surface_value(pto.struct_get(state, path))


def _struct_get_bool(state, path):
    val = unwrap_surface_value(pto.struct_get(state, path))
    zero = arith.ConstantOp(_i8(), 0).result
    return wrap_surface_value(arith.CmpIOp(arith.CmpIPredicate.ne, val, zero).result)


def _struct_set_i32(state, path, value):
    pto.struct_set(state, path, value)


def _struct_set_bool(state, path, value):
    # struct_set requires the exact i8 field type: extend i1, normalize bool.
    raw = unwrap_surface_value(value)
    if isinstance(raw, bool):
        raw = int(raw)
    elif not isinstance(raw, int) and raw.type != _i8():
        raw = arith.ExtUIOp(_i8(), raw).result
    pto.struct_set(state, path, raw)


def _const_i32(val):
    return arith.ConstantOp(_i32(), val).result


def _as_value(val):
    if isinstance(val, int):
        return wrap_surface_value(_const_i32(val))
    if isinstance(val, _SurfaceValue):
        return val
    return wrap_surface_value(val)


def _as_i32(val):
    if isinstance(val, int):
        return wrap_surface_value(_const_i32(val))
    raw = unwrap_surface_value(val)
    if raw.type == _i64():
        raw = arith.TruncIOp(_i32(), raw).result
    return wrap_surface_value(raw)


def _noi(val):
    one = arith.ConstantOp(IntegerType.get_signless(1), 1).result
    return wrap_surface_value(arith.XOrIOp(unwrap_surface_value(_as_value(val)), one).result)


def _cmpi_sgt(val, const_val):
    return arith.CmpIOp(arith.CmpIPredicate.sgt, unwrap_surface_value(val), _const_i32(const_val)).result


def _cmpi_sge(val, const_val):
    return arith.CmpIOp(arith.CmpIPredicate.sge, unwrap_surface_value(val), _const_i32(const_val)).result


def _cmpi_slt(val, const_val):
    return arith.CmpIOp(arith.CmpIPredicate.slt, unwrap_surface_value(val), _const_i32(const_val)).result


def _cmpi_eq(val, const_val):
    return arith.CmpIOp(arith.CmpIPredicate.eq, unwrap_surface_value(val), _const_i32(const_val)).result


def sync_period(slot_num):
    return slot_num if slot_num <= 2 else slot_num // 2


class TileSplitAxis(IntEnum):
    TILE_NO_SPLIT = 0        # 1:1 mode, no split, using AIV0
    TILE_UP_DOWN = 1         # Split along rows: AIV0=upper half, AIV1=lower half
    TILE_LEFT_RIGHT = 2      # Split along cols: AIV0=left half, AIV1=right half
    TILE_UP_DOWN_ODD = 3     # Split along rows: AIV0=rows/2 + 1, AIV1=rows/2
    TILE_LEFT_RIGHT_ODD = 4  # Split along cols: AIV0=cols/2 + 1, AIV1=cols/2


_SPLIT_NUM = 2


def _should_wait_free(tile_index, slot_num):
    sp = sync_period(slot_num)

    if slot_num == 1:
        return _cmpi_sgt(tile_index, 0)
    # Folded conjunction (no early return under SSA).
    lt_slot = _cmpi_slt(tile_index, slot_num)
    mod_zero = _cmpi_eq((tile_index % sp), 0)
    return (_noi(lt_slot) & mod_zero)


def _should_notify_free(tile_index, slot_num):
    sp = sync_period(slot_num)

    if slot_num == 1:
        # i32 result; converted to i1 at the call site (tpop.py).
        return _const_i32(1)
    next_idx = (tile_index + 1)
    return _cmpi_eq((next_idx % sp), 0)
