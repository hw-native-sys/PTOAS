# coding=utf-8
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
    _coerce_index,
    _element_bytewidth,
    _pto,
    unwrap_surface_value,
    wrap_surface_value,
)


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
    nosplit: bool | None = None


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
        self._emit_init("cube")

    def init_simd(self):
        self._emit_init("simd")

    def alloc(self, split=0):
        if self._descriptor.gm_slot_tensor is None:
            raise TypeError("alloc() is only available on global-entry pipes")
        split = _as_int(split, context="pipe.alloc(..., split=...)")
        entry_type = self.entry_type
        if entry_type is None:
            raise RuntimeError("global-entry pipe is missing entry_type metadata")
        if self._descriptor.direction == "c2v":
            op = _pto.TAllocToAivOp(entry_type, split, id=self.id)
        else:
            op = _pto.TAllocToAicOp(entry_type, split, id=self.id)
        return wrap_surface_value(op.result)

    def push(self, entry, split=0):
        if self._descriptor.direction == "both":
            raise TypeError("bidirectional local pipes do not have an unambiguous push direction")
        split = _as_int(split, context="pipe.push(..., split=...)")
        if entry is None:
            raise TypeError("push() requires an entry value")
        entry_value = _normalize_entry_value(entry)
        if self._descriptor.direction == "c2v":
            _pto.TPushToAivOp(entry_value, split, id=self.id)
        else:
            _pto.TPushToAicOp(entry_value, split, id=self.id)

    def pop(self, split=0, result_type=None, *, valid_shape=None, valid_row=None, valid_col=None):
        if self._descriptor.direction == "both":
            raise TypeError("bidirectional local pipes do not have an unambiguous pop direction")
        split = _as_int(split, context="pipe.pop(..., split=...)")
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
            op = self._emit_pop_from_aic(result_type, split, valid_row, valid_col)
        else:
            op = self._emit_pop_from_aiv(result_type, split, valid_row, valid_col)
        return wrap_surface_value(op.result)

    def free(self, entry=None, split=0):
        if self._descriptor.direction == "both":
            raise TypeError("bidirectional local pipes do not have an unambiguous free direction")
        split = _as_int(split, context="pipe.free(..., split=...)")
        if self._descriptor.gm_slot_tensor is not None and entry is None:
            raise TypeError("free() requires an entry value for global-entry pipes")
        entry_value = _normalize_entry_value(entry)
        if self._descriptor.direction == "c2v":
            _pto.TFreeFromAicOp(split, entry=entry_value, id=self.id)
        else:
            _pto.TFreeFromAivOp(split, entry=entry_value, id=self.id)

    def _emit_pop_from_aic(self, result_type, split, valid_row, valid_col):
        if valid_row is None:
            return _pto.TPopFromAicOp(result_type, split, id=self.id)
        return _pto.TPopFromAicOp(
            result_type,
            split,
            valid_row=valid_row,
            valid_col=valid_col,
            id=self.id,
        )

    def _emit_pop_from_aiv(self, result_type, split, valid_row, valid_col):
        if valid_row is None:
            return _pto.TPopFromAivOp(result_type, split, id=self.id)
        return _pto.TPopFromAivOp(
            result_type,
            split,
            valid_row=valid_row,
            valid_col=valid_col,
            id=self.id,
        )

    def _emit_init(self, side: str):
        desc = self._descriptor
        init_kwargs = {
            "id": desc.id,
        }

        init_kwargs["local_slot_num"] = desc.local_slot_num
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
        nosplit=None,
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
            nosplit=nosplit,
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
        nosplit=None,
    ):
        return self._directional_pipe(
            direction="c2v",
            slot_size=slot_size,
            consumer_buf=consumer_buf,
            gm_slot_buffer=gm_slot_buffer,
            gm_slot_tensor=gm_slot_tensor,
            id=id,
            local_slot_num=local_slot_num,
            nosplit=nosplit,
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
        nosplit=None,
    ):
        return self._directional_pipe(
            direction="v2c",
            slot_size=slot_size,
            consumer_buf=consumer_buf,
            gm_slot_buffer=gm_slot_buffer,
            gm_slot_tensor=gm_slot_tensor,
            id=id,
            local_slot_num=local_slot_num,
            nosplit=nosplit,
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
        nosplit=None,
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
            nosplit=nosplit,
        )
        return _PipeSurface(descriptor)


pipe = _PipeNamespace()
