# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""High-level PTODSL pipe surface."""

from __future__ import annotations

from dataclasses import dataclass, replace

from ._ops import (
    _coerce_i64,
    _coerce_index,
    _element_bytewidth,
    _pto,
    declare_struct,
    struct_set,
    unwrap_surface_value,
    wrap_surface_value,
)
from ._types import struct_type, si32, si8


def _require_pipe_id(id, *, context: str) -> int:
    if id is None:
        raise TypeError(f"{context} requires an explicit stable id")
    return int(id)


def _infer_global_slot_size(gm_slot_tensor) -> int:
    surface_shape = getattr(gm_slot_tensor, "shape", None)
    value = unwrap_surface_value(gm_slot_tensor)
    tensor_type = value.type
    if not _pto.TensorViewType.isinstance(tensor_type):
        raise TypeError(
            "pipe.c2v/v2c expects gm_slot_tensor to be !pto.tensor_view"
        )

    tensor_view_type = _pto.TensorViewType(tensor_type)
    shape = tuple(surface_shape) if surface_shape is not None else tuple(tensor_view_type.shape)
    if any(dim < 0 for dim in shape):
        raise ValueError(
            "pipe.c2v/v2c cannot infer slot_size from a dynamic gm_slot_tensor shape"
        )
    count = 1
    for dim in shape:
        count *= int(dim)
    return count * _element_bytewidth(tensor_view_type.element_type)


def _infer_unambiguous_global_slot_size(gm_slot_tensor, *, nosplit, context: str) -> int:
    if nosplit is True:
        return _infer_global_slot_size(gm_slot_tensor)
    raise TypeError(
        f"{context} requires explicit slot_size for split-capable gm_slot_tensor pipes; "
        "pass slot_size=... or set nosplit=True when one gm_slot_tensor shape equals one full slot"
    )


def _as_int(value, *, context: str):
    if value is None:
        return None
    if isinstance(value, int):
        return value
    return _coerce_index(value, context=context)


_VALID_SPLIT = (0, 1, 2, 3, 4)  # TileSplitAxis


def _validate_split(value, *, context: str):
    if value not in _VALID_SPLIT:
        raise ValueError(
            f"{context} split must be one of {_VALID_SPLIT} "
            "(pto TileSplitAxis: 0=NO_SPLIT, 1=UP_DOWN, 2=LEFT_RIGHT, "
            f"3=UP_DOWN_ODD, 4=LEFT_RIGHT_ODD), got {value}")
    return value


def _normalize_entry_value(value):
    if value is None:
        return None
    return unwrap_surface_value(value)


def _normalize_result_type(value, *, context: str):
    if value is None:
        return None
    unwrapped = unwrap_surface_value(value)
    return getattr(unwrapped, "type", unwrapped)


def _normalize_valid_shape(valid_shape, *, valid_row, valid_col):
    if valid_shape is not None:
        if valid_row is not None or valid_col is not None:
            raise TypeError("pipe.pop(...) accepts either valid_shape or valid_row/valid_col, not both")
        if len(valid_shape) == 1:
            valid_row, valid_col = 1, valid_shape[0]
        elif len(valid_shape) == 2:
            valid_row, valid_col = valid_shape
        else:
            raise TypeError("pipe.pop(valid_shape=...) expects one or two dimensions")
    if (valid_row is None) != (valid_col is None):
        raise TypeError("pipe.pop(...) requires valid_row and valid_col to be provided together")
    if valid_row is None:
        return None, None
    return (
        _coerce_index(valid_row, context="pipe.pop(..., valid_row=...)"),
        _coerce_index(valid_col, context="pipe.pop(..., valid_col=...)"),
    )


@dataclass(frozen=True)
class _PipeDescriptor:
    kind: str
    direction: str
    id: int
    slot_size: int
    entry_type: object | None
    gm_slot_tensor: object | None = None
    gm_slot_buffer: object | None = None
    c2v_consumer_buf: object | None = None
    v2c_consumer_buf: object | None = None
    local_slot_num: int | None = None
    slot_num: int | None = None  # TPipe SlotNum (FIFO_DEPTH)
    nosplit: bool | None = None
    en_unit_flag: bool = False  # unit flag for acc->GM stores

    @property
    def uses_slot_protocol(self) -> bool:
        # slot_num / en_unit_flag are the slot-state protocol opt-ins; pipes
        # built without them keep the legacy operand surface.
        return self.slot_num is not None or bool(self.en_unit_flag)


class _PipeSurface:
    """Direction-aware logical pipe object."""

    def __init__(self, descriptor: _PipeDescriptor):
        self._descriptor = descriptor

    @property
    def entry_type(self):
        return self._descriptor.entry_type

    @property
    def id(self):
        return self._descriptor.id

    @property
    def slot_size(self):
        return self._descriptor.slot_size

    @property
    def c2v(self):
        if self._descriptor.direction != "both":
            raise TypeError("c2v endpoint is only available on bidirectional pipes")
        return _PipeSurface(
            replace(
                self._descriptor,
                direction="c2v",
                v2c_consumer_buf=None,
            )
        )

    @property
    def v2c(self):
        if self._descriptor.direction != "both":
            raise TypeError("v2c endpoint is only available on bidirectional pipes")
        return _PipeSurface(
            replace(
                self._descriptor,
                direction="v2c",
                c2v_consumer_buf=None,
            )
        )

    def init_cube(self):
        if self._descriptor.uses_slot_protocol:
            self._init_state()
        self._emit_init("cube")

    def init_simd(self):
        if self._descriptor.uses_slot_protocol:
            self._init_state()
        self._emit_init("simd")

    def _init_state(self):
        """Declare a stack-local struct for the pipe state.

        Field indices follow the shared producer/consumer layout defined by
        the template layer (lib/TileOps/a5/_pushpop.py `_STATE_*`).
        """
        from ._types import int32, int8
        state_type = struct_type(int32, int32, int8, int8, int32)
        self._state = declare_struct(state_type)
        struct_set(self._state, [0], 0)  # subTileIndex
        struct_set(self._state, [1], 0)  # tileIndex
        struct_set(self._state, [2], 1)  # isRecord / isWait
        struct_set(self._state, [3], 1)  # isAllocate / isFree
        struct_set(self._state, [4], 0)  # entryOffset

    def alloc(self, split=0):
        if self._descriptor.gm_slot_tensor is None:
            raise TypeError("alloc() is only available on global-entry pipes")
        split = _validate_split(_as_int(split, context="pipe.alloc(..., split=...)"),
                                context="pipe.alloc()")
        entry_type = self.entry_type
        if entry_type is None:
            raise RuntimeError("global-entry pipe is missing entry_type metadata")
        if self._descriptor.direction == "c2v":
            op = _pto.TAllocToAivOp(entry_type, split, id=self.id)
        else:
            op = _pto.TAllocToAicOp(entry_type, split, id=self.id)
        return wrap_surface_value(op.result)

    def _get_slot_ptr(self):
        """Get the FIFO slot base pointer from gm_slot_tensor or consumer_buf.

        For GM pipes: gm_slot_tensor.as_ptr() → ptr<f32, gm>.
        For local pipes: consumer_buf (I32) → castptr → ptr<f32, ub> (C2V)
        or ptr<f32, mat> (V2C). This mirrors the three-buffer pipe
        construction where the C2V consumer buffer is a uint32_t UB/L1
        offset used directly as address.
        """
        desc = self._descriptor
        if desc.gm_slot_tensor is not None:
            from ._ops import as_ptr
            return as_ptr(desc.gm_slot_tensor)
        from ._types import float32 as _f32, ptr as _ptr
        from ._ops import castptr as _castptr
        if desc.direction in ("c2v", "both") and desc.c2v_consumer_buf is not None:
            return _castptr(desc.c2v_consumer_buf, _ptr(_f32, "ub"))
        if desc.direction in ("v2c", "both") and desc.v2c_consumer_buf is not None:
            return _castptr(desc.v2c_consumer_buf, _ptr(_f32, "mat"))
        return None

    def push(self, entry, split=0, *, sub_block_id=None):
        if self._descriptor.direction == "both":
            raise TypeError("bidirectional local pipes do not have an unambiguous push direction")
        split = _validate_split(_as_int(split, context="pipe.push(..., split=...)"),
                                context="pipe.push()")
        if entry is None:
            raise TypeError("push() requires an entry value")
        entry_value = _normalize_entry_value(entry)
        if not self._descriptor.uses_slot_protocol:
            if sub_block_id is not None:
                raise TypeError(
                    "pipe.push(..., sub_block_id=...) requires the slot-state protocol; "
                    "pass slot_num=... when constructing the pipe")
            if self._descriptor.direction == "c2v":
                _pto.TPushToAivOp(entry_value, split, id=self.id)
            else:
                _pto.TPushToAicOp(entry_value, split, id=self.id)
            return
        state = getattr(self, "_state", None)
        if state is not None:
            state = unwrap_surface_value(state)
        slot_ptr = self._get_slot_ptr()
        if slot_ptr is not None:
            slot_ptr = unwrap_surface_value(slot_ptr)
        slot_num = self._descriptor.local_slot_num or self._descriptor.slot_num or 2
        slot_size = self._descriptor.slot_size
        if self._descriptor.direction == "c2v":
            _pto.TPushToAivOp(entry_value, split,
                              state=state, slot_base_ptr=slot_ptr,
                              id=self.id,
                              slot_num=slot_num, slot_size=slot_size,
                              en_unit_flag=1 if self._descriptor.en_unit_flag else 0)
        else:
            _pto.TPushToAicOp(entry_value, split,
                              aiv_subblockid=(_coerce_i64(sub_block_id, context="pipe.push(..., sub_block_id=...)")
                                              if sub_block_id is not None else None),
                              state=state, slot_base_ptr=slot_ptr,
                              id=self.id,
                              slot_num=slot_num, slot_size=slot_size,
                              en_unit_flag=1 if self._descriptor.en_unit_flag else 0)

    def pop(self, split=0, result_type=None, *, valid_shape=None, valid_row=None, valid_col=None,
            sub_block_id=None):
        if self._descriptor.direction == "both":
            raise TypeError("bidirectional local pipes do not have an unambiguous pop direction")
        split = _validate_split(_as_int(split, context="pipe.pop(..., split=...)"),
                                context="pipe.pop()")
        if not self._descriptor.uses_slot_protocol and sub_block_id is not None:
            raise TypeError(
                "pipe.pop(..., sub_block_id=...) requires the slot-state protocol; "
                "pass slot_num=... when constructing the pipe")
        if result_type is None:
            result_type = self.entry_type
        result_type = _normalize_result_type(result_type, context="pipe.pop(..., result_type=...)")
        if result_type is None:
            raise TypeError("pop() requires result_type for local/tile-entry pipes")
        valid_row, valid_col = _normalize_valid_shape(
            valid_shape,
            valid_row=valid_row,
            valid_col=valid_col,
        )
        if self._descriptor.direction == "c2v":
            op = self._emit_pop_from_aic(result_type, split, valid_row, valid_col, sub_block_id)
        else:
            op = self._emit_pop_from_aiv(result_type, split, valid_row, valid_col, sub_block_id)
        return wrap_surface_value(op.result)

    def free(self, entry=None, split=0):
        if self._descriptor.direction == "both":
            raise TypeError("bidirectional local pipes do not have an unambiguous free direction")
        split = _validate_split(_as_int(split, context="pipe.free(..., split=...)"),
                                context="pipe.free()")
        if self._descriptor.gm_slot_tensor is not None and entry is None:
            raise TypeError("free() requires an entry value for global-entry pipes")
        entry_value = _normalize_entry_value(entry)
        if self._descriptor.direction == "c2v":
            _pto.TFreeFromAicOp(split, entry=entry_value, id=self.id)
        else:
            _pto.TFreeFromAivOp(split, entry=entry_value, id=self.id)

    def _get_cons_dims(self, result_type=None):
        """Compute consumer tile dimensions from result_type / entry_type."""
        from ptoas.mlir.dialects import pto as _pto_type
        desc = self._descriptor
        cons_rows = cons_cols = None
        shape_source = result_type if result_type is not None else desc.entry_type
        tv_type = None
        if shape_source is not None:
            try:
                tv_type = _pto_type.TensorViewType(shape_source)
            except Exception:
                try:
                    tv_type = _pto_type.TileBufType(shape_source)
                except Exception:
                    pass
        if tv_type is not None:
            try:
                shape = list(tv_type.shape)
                if len(shape) == 2:
                    cons_rows, cons_cols = int(shape[0]), int(shape[1])
            except (AttributeError, TypeError):
                pass
        elem_bytes = 4
        cons_dtype = None
        if tv_type is not None:
            from ._ops import _element_bytewidth
            try:
                elem_type = tv_type.element_type
                elem_bytes = _element_bytewidth(elem_type)
                cons_dtype = str(elem_type)
            except (AttributeError, TypeError):
                pass
        return cons_rows, cons_cols, elem_bytes, cons_dtype

    def _emit_pop_from_aic(self, result_type, split, valid_row, valid_col, sub_block_id=None):
        if not self._descriptor.uses_slot_protocol:
            if valid_row is None:
                return _pto.TPopFromAicOp(result_type, split, id=self.id)
            return _pto.TPopFromAicOp(
                result_type,
                split,
                valid_row=valid_row,
                valid_col=valid_col,
                id=self.id,
            )
        state = getattr(self, "_state", None)
        if state is not None:
            state = unwrap_surface_value(state)
        slot_ptr = self._get_slot_ptr()
        if slot_ptr is not None:
            slot_ptr = unwrap_surface_value(slot_ptr)
        slot_num = self._descriptor.local_slot_num or self._descriptor.slot_num or 2
        slot_size = self._descriptor.slot_size
        cons_rows, cons_cols, elem_bytes, cons_dtype = self._get_cons_dims(result_type)
        vr = unwrap_surface_value(valid_row) if valid_row is not None else None
        vc = unwrap_surface_value(valid_col) if valid_col is not None else None
        return _pto.TPopFromAicOp(
            result_type, split,
            valid_row=vr, valid_col=vc,
            aiv_subblockid=(_coerce_i64(sub_block_id, context="pipe.pop(..., sub_block_id=...)")
                            if sub_block_id is not None else None),
            state=state, slot_base_ptr=slot_ptr,
            id=self.id, slot_num=slot_num, slot_size=slot_size,
            cons_rows=cons_rows, cons_cols=cons_cols, elem_bytes=elem_bytes,
            cons_dtype=cons_dtype,
            en_unit_flag=1 if self._descriptor.en_unit_flag else 0,
        )

    def _emit_pop_from_aiv(self, result_type, split, valid_row, valid_col, sub_block_id=None):
        if not self._descriptor.uses_slot_protocol:
            if valid_row is None:
                return _pto.TPopFromAivOp(result_type, split, id=self.id)
            return _pto.TPopFromAivOp(
                result_type,
                split,
                valid_row=valid_row,
                valid_col=valid_col,
                id=self.id,
            )
        state = getattr(self, "_state", None)
        if state is not None:
            state = unwrap_surface_value(state)
        slot_ptr = self._get_slot_ptr()
        if slot_ptr is not None:
            slot_ptr = unwrap_surface_value(slot_ptr)
        slot_num = self._descriptor.local_slot_num or self._descriptor.slot_num or 2
        slot_size = self._descriptor.slot_size
        cons_rows, cons_cols, elem_bytes, cons_dtype = self._get_cons_dims(result_type)
        vr = unwrap_surface_value(valid_row) if valid_row is not None else None
        vc = unwrap_surface_value(valid_col) if valid_col is not None else None
        return _pto.TPopFromAivOp(
            result_type, split,
            valid_row=vr, valid_col=vc,
            aiv_subblockid=(_coerce_i64(sub_block_id, context="pipe.pop(..., sub_block_id=...)")
                            if sub_block_id is not None else None),
            state=state, slot_base_ptr=slot_ptr,
            id=self.id, slot_num=slot_num, slot_size=slot_size,
            cons_rows=cons_rows, cons_cols=cons_cols, elem_bytes=elem_bytes,
            cons_dtype=cons_dtype,
            en_unit_flag=1 if self._descriptor.en_unit_flag else 0,
        )

    def _emit_init(self, side: str):
        desc = self._descriptor
        init_kwargs = {
            "id": desc.id,
        }

        init_kwargs["local_slot_num"] = desc.local_slot_num
        if desc.uses_slot_protocol:
            init_kwargs["slot_num"] = desc.slot_num
        init_kwargs["nosplit"] = desc.nosplit
        init_kwargs["gm_slot_buffer"] = desc.gm_slot_buffer
        init_kwargs["gm_slot_tensor"] = desc.gm_slot_tensor
        if desc.direction in ("c2v", "both"):
            init_kwargs["c2v_consumer_buf"] = desc.c2v_consumer_buf
        if desc.direction in ("v2c", "both"):
            init_kwargs["v2c_consumer_buf"] = desc.v2c_consumer_buf

        init_kwargs = {key: value for key, value in init_kwargs.items() if value is not None}

        if side == "cube":
            _pto.AicInitializePipeOp(
                1 if desc.direction == "c2v" else 2 if desc.direction == "v2c" else 3,
                desc.slot_size,
                **init_kwargs,
            )
            return

        _pto.AivInitializePipeOp(
            1 if desc.direction == "c2v" else 2 if desc.direction == "v2c" else 3,
            desc.slot_size,
            **init_kwargs,
        )


class _PipeNamespace:
    def _directional_pipe(
        self,
        *,
        direction,
        slot_size=None,
        consumer_buf=None,
        gm_slot_buffer=None,
        gm_slot_tensor=None,
        id=None,
        local_slot_num=None,
        slot_num=None,
        nosplit=None,
        en_unit_flag=False,
    ):
        id = _require_pipe_id(id, context=f"pipe.{direction}(...)")
        entry_type = None
        if slot_size is None:
            if gm_slot_tensor is None:
                raise TypeError(f"pipe.{direction}(...) requires slot_size when gm_slot_tensor is not provided")
            slot_size = _infer_unambiguous_global_slot_size(
                gm_slot_tensor,
                nosplit=nosplit,
                context=f"pipe.{direction}(...)",
            )
        if gm_slot_tensor is not None:
            if consumer_buf is not None:
                raise TypeError(f"pipe.{direction}(...) does not accept consumer_buf when gm_slot_tensor is provided")
            if gm_slot_buffer is not None:
                raise TypeError(f"pipe.{direction}(...) does not accept gm_slot_buffer when gm_slot_tensor is provided")
            if local_slot_num is not None:
                raise TypeError(f"pipe.{direction}(...) does not accept local_slot_num when gm_slot_tensor is provided")
            entry_type = unwrap_surface_value(gm_slot_tensor).type
        elif consumer_buf is None:
            raise TypeError(f"pipe.{direction}(...) requires consumer_buf for local pipes")
        descriptor = _PipeDescriptor(
            kind="global" if gm_slot_tensor is not None else "local",
            direction=direction,
            id=int(id),
            slot_size=int(slot_size),
            entry_type=entry_type,
            gm_slot_tensor=_normalize_entry_value(gm_slot_tensor),
            gm_slot_buffer=_normalize_entry_value(gm_slot_buffer),
            c2v_consumer_buf=_normalize_entry_value(consumer_buf) if direction == "c2v" else None,
            v2c_consumer_buf=_normalize_entry_value(consumer_buf) if direction == "v2c" else None,
            local_slot_num=None if local_slot_num is None else int(local_slot_num),
            slot_num=None if slot_num is None else int(slot_num),
            nosplit=nosplit,
            en_unit_flag=en_unit_flag,
        )
        return _PipeSurface(descriptor)

    def c2v(
        self,
        *,
        slot_size=None,
        consumer_buf=None,
        gm_slot_buffer=None,
        gm_slot_tensor=None,
        id=None,
        local_slot_num=None,
        slot_num=None,
        nosplit=None,
        en_unit_flag=False,
    ):
        return self._directional_pipe(
            direction="c2v",
            slot_size=slot_size,
            consumer_buf=consumer_buf,
            gm_slot_buffer=gm_slot_buffer,
            gm_slot_tensor=gm_slot_tensor,
            id=id,
            local_slot_num=local_slot_num,
            slot_num=slot_num,
            nosplit=nosplit,
            en_unit_flag=en_unit_flag,
        )

    def v2c(
        self,
        *,
        slot_size=None,
        consumer_buf=None,
        gm_slot_buffer=None,
        gm_slot_tensor=None,
        id=None,
        local_slot_num=None,
        slot_num=None,
        nosplit=None,
        en_unit_flag=False,
    ):
        return self._directional_pipe(
            direction="v2c",
            slot_size=slot_size,
            consumer_buf=consumer_buf,
            gm_slot_buffer=gm_slot_buffer,
            gm_slot_tensor=gm_slot_tensor,
            id=id,
            local_slot_num=local_slot_num,
            slot_num=slot_num,
            nosplit=nosplit,
            en_unit_flag=en_unit_flag,
        )

    def bidirectional(
        self,
        *,
        slot_size,
        c2v_consumer_buf,
        v2c_consumer_buf,
        gm_slot_buffer=None,
        id=None,
        local_slot_num=None,
        slot_num=None,
        nosplit=None,
        en_unit_flag=False,
    ):
        id = _require_pipe_id(id, context="pipe.bidirectional(...)")
        descriptor = _PipeDescriptor(
            kind="local",
            direction="both",
            id=int(id),
            slot_size=int(slot_size),
            entry_type=None,
            gm_slot_buffer=_normalize_entry_value(gm_slot_buffer),
            c2v_consumer_buf=_normalize_entry_value(c2v_consumer_buf),
            v2c_consumer_buf=_normalize_entry_value(v2c_consumer_buf),
            local_slot_num=None if local_slot_num is None else int(local_slot_num),
            slot_num=None if slot_num is None else int(slot_num),
            nosplit=nosplit,
            en_unit_flag=en_unit_flag,
        )
        return _PipeSurface(descriptor)


pipe = _PipeNamespace()
