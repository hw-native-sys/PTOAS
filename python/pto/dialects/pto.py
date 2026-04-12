# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

from . import _pto_ops_gen as _pto_ops_gen
from ._pto_ops_gen import *
from mlir import ir as _ods_ir

def _load_local_pto_ext():
    import importlib.util
    from pathlib import Path
    lib_dir = Path(__file__).resolve().parent.parent / "_mlir_libs"
    for suffix in ("*.so", "*.pyd", "*.dll", "*.dylib"):
        for so_path in lib_dir.glob(f"_pto{suffix}"):
            spec = importlib.util.spec_from_file_location("mlir._mlir_libs._pto", so_path)
            if spec and spec.loader:
                mod = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(mod)
                return mod
    raise ImportError("cannot locate local _pto extension in _mlir_libs")


try:
    _pto_mod = _load_local_pto_ext()
except Exception:
    from .._mlir_libs import _pto as _pto_mod

register_dialect = _pto_mod.register_dialect
PtrType = _pto_mod.PtrType
AsyncSessionType = _pto_mod.AsyncSessionType
AsyncEventType = _pto_mod.AsyncEventType
TensorViewType = _pto_mod.TensorViewType
PartitionTensorViewType = _pto_mod.PartitionTensorViewType
TileType = _pto_mod.TileType
TileBufType = _pto_mod.TileBufType
AddressSpace = _pto_mod.AddressSpace
AddressSpaceAttr = _pto_mod.AddressSpaceAttr
TileBufConfigAttr = _pto_mod.TileBufConfigAttr
BLayout = _pto_mod.BLayout
BLayoutAttr = _pto_mod.BLayoutAttr
SLayout = _pto_mod.SLayout
SLayoutAttr = _pto_mod.SLayoutAttr
PadValue = _pto_mod.PadValue
PadValueAttr = _pto_mod.PadValueAttr
CompactMode = _pto_mod.CompactMode
CompactModeAttr = _pto_mod.CompactModeAttr
AccToVecMode = _pto_mod.AccToVecMode
AccToVecModeAttr = _pto_mod.AccToVecModeAttr
ReluPreMode = _pto_mod.ReluPreMode
ReluPreModeAttr = _pto_mod.ReluPreModeAttr
RoundMode = _pto_mod.RoundMode
RoundModeAttr = _pto_mod.RoundModeAttr
CmpMode = _pto_mod.CmpMode
CmpModeAttr = _pto_mod.CmpModeAttr
PIPE = _pto_mod.PIPE
PipeAttr = _pto_mod.PipeAttr
Layout = _pto_mod.Layout
LayoutAttr = _pto_mod.LayoutAttr
SyncOpType = _pto_mod.SyncOpType
SyncOpTypeAttr = _pto_mod.SyncOpTypeAttr
EVENT = _pto_mod.EVENT
EventAttr = _pto_mod.EventAttr
MaskPattern = _pto_mod.MaskPattern
MaskPatternAttr = _pto_mod.MaskPatternAttr
QuantType = _pto_mod.QuantType
QuantTypeAttr = _pto_mod.QuantTypeAttr

__all__ = [
    # Dialect utilities
    "register_dialect",

    # Types
    "PtrType",
    "AsyncSessionType",
    "AsyncEventType",
    "TensorViewType",
    "PartitionTensorViewType",
    "TileType",
    "TileBufType",
    "AddressSpace", "AddressSpaceAttr",
    "BLayout","BLayoutAttr",
    "SLayout","SLayoutAttr",
    "PadValue","PadValueAttr",
    "CompactMode", "CompactModeAttr",
    "AccToVecMode", "AccToVecModeAttr",
    "ReluPreMode", "ReluPreModeAttr",
    "RoundMode", "RoundModeAttr",
    "CmpMode", "CmpModeAttr",
    "PIPE", "PipeAttr",
    "Layout", "LayoutAttr",
    "SyncOpType", "SyncOpTypeAttr",
    "EVENT", "EventAttr",
    "MaskPattern", "MaskPatternAttr",
    "QuantType", "QuantTypeAttr",
    "TileBufConfigAttr",
    "TileConfig",
    # High-level sync helpers
    "record_event", "wait_event", "barrier",
    # Low-level sync helpers (static/dynamic event id unified API)
    "set_flag", "wait_flag", "set_flag_dyn", "wait_flag_dyn",
    # Inter-core sync helpers
    "sync_set", "sync_wait", "sync_set_dyn", "sync_wait_dyn", "set_ffts",
    # A5 buffer-id sync helpers
    "get_buf", "rls_buf",
    # Scalar pointer helpers
    "load_scalar", "store_scalar"

    # Aliases for SyncOpType enums (for terse calls)
    ,"TLOAD","TSTORE_ACC","TSTORE_VEC","TMOV_M2L","TMOV_M2S",
    "TMOV_M2B","TMOV_M2V","TMOV_V2M","TMATMUL","TVEC","TVECWAIT_EVENT"
    # Aliases for EVENT enums
    ,"EVENT_ID0","EVENT_ID1","EVENT_ID2","EVENT_ID3",
    "EVENT_ID4","EVENT_ID5","EVENT_ID6","EVENT_ID7"
]

# -----------------------------------------------------------------------------
# Convenience wrappers for high-level sync to allow passing enums directly
# -----------------------------------------------------------------------------

def _ensure_sync_attr(val, ctx):
    # Accept SyncOpType enum, SyncOpTypeAttr, or string name ("TMATMUL"/"tmatmul").
    if isinstance(val, SyncOpType):
        return SyncOpTypeAttr.get(val, ctx)
    if isinstance(val, str):
        name = val.upper()
        try:
            enum_val = getattr(SyncOpType, name)
        except AttributeError:
            raise ValueError(f"Unknown SyncOpType name: {val}")
        return SyncOpTypeAttr.get(enum_val, ctx)
    return val

def _ensure_event_attr(val, ctx):
    if isinstance(val, EVENT):
        return EventAttr.get(val, ctx)
    if isinstance(val, int):
        if val < 0 or val > 7:
            raise ValueError(f"event id out of range [0,7]: {val}")
        enum_name = f"EVENT_ID{val}"
        try:
            enum_val = getattr(EVENT, enum_name)
        except AttributeError:
            raise ValueError(f"Unknown EVENT integer id: {val}")
        return EventAttr.get(enum_val, ctx)
    if isinstance(val, str):
        name = val.upper()
        try:
            enum_val = getattr(EVENT, name)
        except AttributeError:
            raise ValueError(f"Unknown EVENT name: {val}")
        return EventAttr.get(enum_val, ctx)
    return val

def _ensure_pipe_attr(val, ctx):
    if isinstance(val, PipeAttr):
        return val
    if isinstance(val, PIPE):
        return PipeAttr.get(val, ctx)
    if isinstance(val, str):
        name = val.upper()
        try:
            enum_val = getattr(PIPE, name)
        except AttributeError:
            raise ValueError(f"Unknown PIPE name: {val}")
        return PipeAttr.get(enum_val, ctx)
    return val

def _ensure_i32_attr(val, name, ctx):
    if isinstance(val, _ods_ir.IntegerAttr):
        return val
    if isinstance(val, int):
        i32 = _ods_ir.IntegerType.get_signless(32, ctx)
        return _ods_ir.IntegerAttr.get(i32, val)
    raise TypeError(f"{name} must be int or IntegerAttr, got {type(val).__name__}")

def record_event(src_op, dst_op, event_id, *, loc=None, ip=None):
    ctx = loc.context if loc else _ods_ir.Context.current
    return _pto_ops_gen.record_event(
        _ensure_sync_attr(src_op, ctx),
        _ensure_sync_attr(dst_op, ctx),
        _ensure_event_attr(event_id, ctx),
        loc=loc, ip=ip)

def wait_event(src_op, dst_op, event_id, *, loc=None, ip=None):
    ctx = loc.context if loc else _ods_ir.Context.current
    return _pto_ops_gen.wait_event(
        _ensure_sync_attr(src_op, ctx),
        _ensure_sync_attr(dst_op, ctx),
        _ensure_event_attr(event_id, ctx),
        loc=loc, ip=ip)

def barrier(op, *, loc=None, ip=None):
    ctx = loc.context if loc else _ods_ir.Context.current
    # If user passes SyncOpType/Attr, route to barrier_sync (maps to PIPE)
    if isinstance(op, (SyncOpType, SyncOpTypeAttr, str)):
        op_attr = _ensure_sync_attr(op, ctx)
        return _pto_ops_gen.barrier_sync(op_attr, loc=loc, ip=ip)
    # Otherwise fall back to low-level barrier expecting PipeAttr
    return _pto_ops_gen.barrier(op, loc=loc, ip=ip)


def _is_static_event_id(event_id):
    if isinstance(event_id, (EVENT, EventAttr, str, int)):
        return True
    return isinstance(event_id, _ods_ir.Attribute)


def _is_static_i32_event_id(event_id):
    if isinstance(event_id, (int, _ods_ir.IntegerAttr)):
        return True
    return False


def set_flag_dyn(src_pipe, dst_pipe, event_id, *, loc=None, ip=None):
    """Low-level dynamic event-id set_flag helper."""
    ctx = loc.context if loc else _ods_ir.Context.current
    src_attr = _ensure_pipe_attr(src_pipe, ctx)
    dst_attr = _ensure_pipe_attr(dst_pipe, ctx)
    event_val = _pto_ops_gen._get_op_result_or_value(event_id)
    if hasattr(_pto_ops_gen, "set_flag_dyn"):
        return _pto_ops_gen.set_flag_dyn(
            src_attr, dst_attr, event_val, loc=loc, ip=ip
        )
    return _ods_ir.Operation.create(
        "pto.set_flag_dyn",
        attributes={"src_pipe": src_attr, "dst_pipe": dst_attr},
        operands=[event_val],
        loc=loc,
        ip=ip,
    )


def wait_flag_dyn(src_pipe, dst_pipe, event_id, *, loc=None, ip=None):
    """Low-level dynamic event-id wait_flag helper."""
    ctx = loc.context if loc else _ods_ir.Context.current
    src_attr = _ensure_pipe_attr(src_pipe, ctx)
    dst_attr = _ensure_pipe_attr(dst_pipe, ctx)
    event_val = _pto_ops_gen._get_op_result_or_value(event_id)
    if hasattr(_pto_ops_gen, "wait_flag_dyn"):
        return _pto_ops_gen.wait_flag_dyn(
            src_attr, dst_attr, event_val, loc=loc, ip=ip
        )
    return _ods_ir.Operation.create(
        "pto.wait_flag_dyn",
        attributes={"src_pipe": src_attr, "dst_pipe": dst_attr},
        operands=[event_val],
        loc=loc,
        ip=ip,
    )


def set_flag(src_pipe, dst_pipe, event_id, *, loc=None, ip=None):
    """Unified low-level set_flag API.

    - Static path: EVENT/EventAttr/str/int -> pto.set_flag
    - Dynamic path: SSA value -> pto.set_flag_dyn
    """
    ctx = loc.context if loc else _ods_ir.Context.current
    src_attr = _ensure_pipe_attr(src_pipe, ctx)
    dst_attr = _ensure_pipe_attr(dst_pipe, ctx)
    if _is_static_event_id(event_id):
        return _pto_ops_gen.set_flag(
            src_attr, dst_attr, _ensure_event_attr(event_id, ctx), loc=loc, ip=ip
        )
    return set_flag_dyn(src_attr, dst_attr, event_id, loc=loc, ip=ip)


def wait_flag(src_pipe, dst_pipe, event_id, *, loc=None, ip=None):
    """Unified low-level wait_flag API.

    - Static path: EVENT/EventAttr/str/int -> pto.wait_flag
    - Dynamic path: SSA value -> pto.wait_flag_dyn
    """
    ctx = loc.context if loc else _ods_ir.Context.current
    src_attr = _ensure_pipe_attr(src_pipe, ctx)
    dst_attr = _ensure_pipe_attr(dst_pipe, ctx)
    if _is_static_event_id(event_id):
        return _pto_ops_gen.wait_flag(
            src_attr, dst_attr, _ensure_event_attr(event_id, ctx), loc=loc, ip=ip
        )
    return wait_flag_dyn(src_attr, dst_attr, event_id, loc=loc, ip=ip)

# -----------------------------------------------------------------------------
# Inter-core sync helpers (pto.sync.set / pto.sync.wait / pto.set_ffts)
# -----------------------------------------------------------------------------
def sync_set_dyn(pipe, event_id, ffts_mode=2, *, loc=None, ip=None):
    ctx = loc.context if loc else _ods_ir.Context.current
    pipe_attr = _ensure_pipe_attr(pipe, ctx)
    event_val = _pto_ops_gen._get_op_result_or_value(event_id)
    mode_attr = None
    if ffts_mode != 2:
        mode_attr = _ensure_i32_attr(ffts_mode, "ffts_mode", ctx)
    # Preferred unified-op path: pto.sync.set(pipe, event_id_dyn=%v)
    try:
        return _pto_ops_gen.sync_set(
            pipe_attr,
            event_id=None,
            ffts_mode=mode_attr,
            event_id_dyn=event_val,
            loc=loc,
            ip=ip,
        )
    except TypeError:
        attrs = {"pipe": pipe_attr}
        if mode_attr is not None:
            attrs["ffts_mode"] = mode_attr
        return _ods_ir.Operation.create(
            "pto.sync.set", attributes=attrs, operands=[event_val], loc=loc, ip=ip
        )


def sync_set(pipe, event_id, ffts_mode=2, *, loc=None, ip=None):
    ctx = loc.context if loc else _ods_ir.Context.current
    pipe_attr = _ensure_pipe_attr(pipe, ctx)
    mode_attr = None
    if ffts_mode != 2:
        mode_attr = _ensure_i32_attr(ffts_mode, "ffts_mode", ctx)
    if _is_static_i32_event_id(event_id):
        event_attr = _ensure_i32_attr(event_id, "event_id", ctx)
        try:
            return _pto_ops_gen.sync_set(
                pipe_attr,
                event_id=event_attr,
                ffts_mode=mode_attr,
                event_id_dyn=None,
                loc=loc,
                ip=ip,
            )
        except TypeError:
            attrs = {"pipe": pipe_attr, "event_id": event_attr}
            if mode_attr is not None:
                attrs["ffts_mode"] = mode_attr
            return _ods_ir.Operation.create(
                "pto.sync.set",
                attributes=attrs,
                loc=loc,
                ip=ip,
            )
    return sync_set_dyn(pipe_attr, event_id, ffts_mode=ffts_mode, loc=loc, ip=ip)


def sync_wait_dyn(pipe, event_id, *, loc=None, ip=None):
    ctx = loc.context if loc else _ods_ir.Context.current
    pipe_attr = _ensure_pipe_attr(pipe, ctx)
    event_val = _pto_ops_gen._get_op_result_or_value(event_id)
    try:
        return _pto_ops_gen.sync_wait(
            pipe_attr, event_id=None, event_id_dyn=event_val, loc=loc, ip=ip
        )
    except TypeError:
        if hasattr(_pto_ops_gen, "sync_wait_dyn"):
            return _pto_ops_gen.sync_wait_dyn(pipe_attr, event_val, loc=loc, ip=ip)
        raise


def sync_wait(pipe, event_id, *, loc=None, ip=None):
    ctx = loc.context if loc else _ods_ir.Context.current
    pipe_attr = _ensure_pipe_attr(pipe, ctx)
    if _is_static_i32_event_id(event_id):
        event_attr = _ensure_i32_attr(event_id, "event_id", ctx)
        try:
            return _pto_ops_gen.sync_wait(
                pipe_attr, event_id=event_attr, event_id_dyn=None, loc=loc, ip=ip
            )
        except TypeError:
            return _ods_ir.Operation.create(
                "pto.sync.wait",
                attributes={"pipe": pipe_attr, "event_id": event_attr},
                loc=loc,
                ip=ip,
            )
    return sync_wait_dyn(pipe_attr, event_id, loc=loc, ip=ip)

def set_ffts(ffts, *, loc=None, ip=None):
    return _ods_ir.Operation.create(
        "pto.set_ffts",
        operands=[_pto_ops_gen._get_op_result_or_value(ffts)],
        loc=loc,
        ip=ip,
    )

# -----------------------------------------------------------------------------
# A5 buffer-id sync helpers
# -----------------------------------------------------------------------------
def get_buf(op_type, buf_id, mode=0, *, loc=None, ip=None):
    ctx = loc.context if loc else _ods_ir.Context.current
    if isinstance(op_type, (PipeAttr, PIPE)):
        raise TypeError("get_buf expects SyncOpType (or SyncOpTypeAttr), not PIPE/PipeAttr")
    attrs = {
        "op_type": _ensure_sync_attr(op_type, ctx),
        "buf_id": _ensure_i32_attr(buf_id, "buf_id", ctx),
        "mode": _ensure_i32_attr(mode, "mode", ctx),
    }
    return _ods_ir.Operation.create(
        "pto.get_buf",
        attributes=attrs,
        loc=loc,
        ip=ip,
    )


def rls_buf(op_type, buf_id, mode=0, *, loc=None, ip=None):
    ctx = loc.context if loc else _ods_ir.Context.current
    if isinstance(op_type, (PipeAttr, PIPE)):
        raise TypeError("rls_buf expects SyncOpType (or SyncOpTypeAttr), not PIPE/PipeAttr")
    attrs = {
        "op_type": _ensure_sync_attr(op_type, ctx),
        "buf_id": _ensure_i32_attr(buf_id, "buf_id", ctx),
        "mode": _ensure_i32_attr(mode, "mode", ctx),
    }
    return _ods_ir.Operation.create(
        "pto.rls_buf",
        attributes=attrs,
        loc=loc,
        ip=ip,
    )

# -----------------------------------------------------------------------------
# Scalar pointer helpers (manual wrappers until python ops are regenerated)
# -----------------------------------------------------------------------------
def load_scalar(result_type, ptr, offset, *, loc=None, ip=None):
    operands = [
        _pto_ops_gen._get_op_result_or_value(ptr),
        _pto_ops_gen._get_op_result_or_value(offset),
    ]
    op = _ods_ir.Operation.create(
        "pto.load_scalar",
        results=[result_type],
        operands=operands,
        loc=loc,
        ip=ip,
    )
    return op.results[0]


def store_scalar(ptr, offset, value, *, loc=None, ip=None):
    operands = [
        _pto_ops_gen._get_op_result_or_value(ptr),
        _pto_ops_gen._get_op_result_or_value(offset),
        _pto_ops_gen._get_op_result_or_value(value),
    ]
    return _ods_ir.Operation.create(
        "pto.store_scalar",
        operands=operands,
        loc=loc,
        ip=ip,
    )

# -----------------------------------------------------------------------------
# Export enum aliases for terse calls: pto.record_event(TLOAD, TLOAD, EVENT_ID0)
# -----------------------------------------------------------------------------
TLOAD = SyncOpType.TLOAD
TSTORE_ACC = SyncOpType.TSTORE_ACC
TSTORE_VEC = SyncOpType.TSTORE_VEC
TMOV_M2L = SyncOpType.TMOV_M2L
TMOV_M2S = SyncOpType.TMOV_M2S
TMOV_M2B = SyncOpType.TMOV_M2B
TMOV_M2V = SyncOpType.TMOV_M2V
TMOV_V2M = SyncOpType.TMOV_V2M
TMATMUL = SyncOpType.TMATMUL
TVEC = SyncOpType.TVEC
TVECWAIT_EVENT = SyncOpType.TVECWAIT_EVENT

EVENT_ID0 = EVENT.EVENT_ID0
EVENT_ID1 = EVENT.EVENT_ID1
EVENT_ID2 = EVENT.EVENT_ID2
EVENT_ID3 = EVENT.EVENT_ID3
EVENT_ID4 = EVENT.EVENT_ID4
EVENT_ID5 = EVENT.EVENT_ID5
EVENT_ID6 = EVENT.EVENT_ID6
EVENT_ID7 = EVENT.EVENT_ID7

class TileConfig:
    alignedSize = 32
    fixedRowSize = 16
    fixedColSize = 16
    fixedMxRowSize = 16
    fixedMxColSize = 2
    fractalABSize = 512
    fractalCSize = 1024
    fractalMxSize = 32

# -----------------------------------------------------------------------------
# Op aliases without "Op" suffix (user-facing)
# -----------------------------------------------------------------------------
def _install_op_aliases():
    added = []
    for name, obj in _pto_ops_gen.__dict__.items():
        if not isinstance(obj, type):
            continue
        if not issubclass(obj, _ods_ir.OpView):
            continue
        alias = None
        if name.endswith("Op_DPS"):
            alias = f"{name[:-6]}_DPS"
        elif name.endswith("Op"):
            alias = name[:-2]
        if not alias or alias in globals():
            continue
        globals()[alias] = obj
        added.append(alias)
    return added

__all__.extend(_install_op_aliases())

# -----------------------------------------------------------------------------
# Experimental VPTO Python DSL (`@pto.vkernel`)
# -----------------------------------------------------------------------------
import ast as _ast
import inspect as _inspect
import textwrap as _textwrap
from dataclasses import dataclass as _dataclass


class _VKernelType:
    def render(self):
        raise NotImplementedError


@_dataclass(frozen=True)
class _VKernelScalarType(_VKernelType):
    name: str

    def render(self):
        return self.name


@_dataclass(frozen=True)
class _VKernelPtrType(_VKernelType):
    elem: _VKernelType
    space: str

    def render(self):
        return f"!pto.ptr<{self.elem.render()}, {self.space}>"


@_dataclass(frozen=True)
class _VKernelVRegType(_VKernelType):
    lanes: int
    elem: _VKernelType

    def render(self):
        return f"!pto.vreg<{self.lanes}x{self.elem.render()}>"


@_dataclass(frozen=True)
class _VKernelConstBinding:
    value: object


@_dataclass(frozen=True)
class _VKernelStructDef(_VKernelType):
    name: str
    fields: tuple

    def render(self):
        raise _VKernelCompileError(f"{self.name} is a template-only surface type; use .jit(...) to specialize it")

    def __call__(self, **kwargs):
        return _VKernelStructBinding(self, dict(kwargs))


@_dataclass(frozen=True)
class _VKernelStructBinding:
    schema: _VKernelStructDef
    values: dict


@_dataclass(frozen=True)
class _VKStaticSequence:
    values: tuple


@_dataclass(frozen=True)
class _VKStructValue:
    schema: _VKernelStructDef
    fields: dict


i1 = _VKernelScalarType("i1")
i8 = _VKernelScalarType("i8")
i16 = _VKernelScalarType("i16")
i32 = _VKernelScalarType("i32")
i64 = _VKernelScalarType("i64")
f16 = _VKernelScalarType("f16")
bf16 = _VKernelScalarType("bf16")
f32 = _VKernelScalarType("f32")
_vk_index = _VKernelScalarType("index")
mask = _VKernelScalarType("!pto.mask")
align = _VKernelScalarType("!pto.align")


def ptr(elem_type, space):
    return _VKernelPtrType(elem_type, space)


def vreg(lanes, elem_type):
    return _VKernelVRegType(lanes, elem_type)


def const(value):
    return _VKernelConstBinding(value)


def struct(cls):
    annotations = dict(getattr(cls, "__annotations__", {}))
    if not annotations:
        raise _VKernelCompileError("@pto.struct requires annotated fields")
    fields = []
    for name, field_ty in annotations.items():
        if field_ty not in (ptr, const):
            raise _VKernelCompileError(
                f"unsupported field annotation for {cls.__name__}.{name}: {field_ty!r}"
            )
        fields.append((name, field_ty))
    return _VKernelStructDef(cls.__name__, tuple(fields))


@struct
class Tile:
    ub_ptr: ptr
    shape: const


tile = Tile


class _VKernelCompileError(Exception):
    pass


@_dataclass
class _VKValue:
    name: str | None = None
    type: _VKernelType | None = None
    literal: object | None = None

    def render_type(self):
        if self.type is None:
            raise _VKernelCompileError(f"unresolved type for {self.name}")
        return self.type.render()


def _project_result(group, index, ty):
    return _VKValue(f"{group.name}#{index}", ty)


def _load_standard_dialects():
    try:
        from mlir.dialects import arith as _mlir_arith  # noqa: F401
        from mlir.dialects import func as _mlir_func  # noqa: F401
        from mlir.dialects import scf as _mlir_scf  # noqa: F401
    except ImportError as exc:
        raise RuntimeError("mlir standard dialect python bindings are required for vkernel parsing") from exc


class _VKernelContext:
    def __init__(self):
        self.ssa_counter = 0
        self.arg_counter = 0

    def new_ssa(self):
        name = f"%{self.ssa_counter}"
        self.ssa_counter += 1
        return name

    def new_arg(self):
        name = f"%arg{self.arg_counter}"
        self.arg_counter += 1
        return name


def _type_key(ty):
    return ty.render() if ty is not None else None


def _types_equal(lhs, rhs):
    if lhs is None or rhs is None:
        return lhs is rhs
    return lhs.render() == rhs.render()


def _ensure_type(value, expected):
    if value.type is None:
        value.type = expected
        return
    if not _types_equal(value.type, expected):
        raise _VKernelCompileError(
            f"type mismatch for {value.name}: expected {expected.render()}, got {value.type.render()}"
        )


def _literal_text(value):
    if isinstance(value, bool):
        return "true" if value else "false"
    return str(value)


def _coerce_surface_type(value):
    if value is bool:
        return i1
    if value is float:
        return f32
    return value


def _ptr_elem_bytes(ptr_type):
    if not isinstance(ptr_type, _VKernelPtrType):
        raise _VKernelCompileError("elem_bytes requires a ptr type")
    elem_name = ptr_type.elem.render()
    table = {
        "i8": 1,
        "i16": 2,
        "i32": 4,
        "i64": 8,
        "f16": 2,
        "bf16": 2,
        "f32": 4,
    }
    if elem_name not in table:
        raise _VKernelCompileError(f"unsupported elem_bytes for {elem_name}")
    return table[elem_name]


def _ptr_vector_lanes(ptr_type):
    return 256 // _ptr_elem_bytes(ptr_type)


class _VKernelBuilder:
    def __init__(self, py_fn, fn_def, target, kernel_name, specialization=None):
        self.py_fn = py_fn
        self.fn_def = fn_def
        self.target = target
        self.kernel_name = kernel_name
        self.ctx = _VKernelContext()
        self.specialization = specialization or {}

    def _emit(self, lines, indent, text):
        lines.append("  " * indent + text)

    def _eval_type_expr(self, node):
        expr = _ast.Expression(body=node)
        globals_dict = dict(self.py_fn.__globals__)
        globals_dict.update(globals())
        value = eval(compile(expr, self.py_fn.__code__.co_filename, "eval"),
                     globals_dict, {})
        value = _coerce_surface_type(value)
        if not isinstance(value, _VKernelType):
            raise _VKernelCompileError(f"unsupported vkernel type annotation: {value!r}")
        return value

    def _new_value(self, ty=None):
        return _VKValue(self.ctx.new_ssa(), ty)

    def _new_arg_value(self, ty=None):
        return _VKValue(self.ctx.new_arg(), ty)

    def _materialize_value(self, value, lines, indent, expected_type=None):
        if expected_type is not None:
            _ensure_type(value, expected_type)
        if value.name is not None:
            return value
        if value.literal is None:
            raise _VKernelCompileError("value has no SSA name and cannot be materialized")
        if value.type is None:
            raise _VKernelCompileError("literal requires type context")
        value.name = self.ctx.new_ssa()
        lit = _literal_text(value.literal)
        if isinstance(value.literal, bool):
            self._emit(lines, indent, f"{value.name} = arith.constant {lit}")
        else:
            self._emit(lines, indent, f"{value.name} = arith.constant {lit} : {value.type.render()}")
        return value

    def _literal_value(self, node, lines, indent, expected_type):
        value = _VKValue(type=expected_type, literal=node.value)
        if expected_type is None:
            return value
        return self._materialize_value(value, lines, indent)

    def _lower_attribute(self, node, env, lines, indent, expected_type=None):
        if isinstance(node.value, _ast.Name):
            if node.value.id not in env:
                raise _VKernelCompileError(f"unknown name '{node.value.id}'")
            base = env[node.value.id]
        else:
            base = self._lower_expr(node.value, env, lines, indent)
        if isinstance(base, _VKStructValue):
            if node.attr not in base.fields:
                raise _VKernelCompileError(f"unsupported struct attribute '{node.attr}'")
            field = base.fields[node.attr]
            if isinstance(field, _VKValue):
                return self._materialize_value(field, lines, indent, expected_type)
            return field
        if isinstance(base, _VKValue) and isinstance(base.type, _VKernelPtrType):
            if node.attr == "elem_bytes":
                return _VKValue(type=expected_type, literal=_ptr_elem_bytes(base.type))
        raise _VKernelCompileError(f"unsupported attribute access '{node.attr}'")

    def _lower_subscript(self, node, env, lines, indent, expected_type=None):
        base = self._lower_expr(node.value, env, lines, indent)
        if not isinstance(base, _VKStaticSequence):
            raise _VKernelCompileError("subscript base must be a static sequence")
        if not isinstance(node.slice, _ast.Constant) or not isinstance(node.slice.value, int):
            raise _VKernelCompileError("only constant integer subscripts are supported")
        index = node.slice.value
        if index < 0 or index >= len(base.values):
            raise _VKernelCompileError("subscript out of range")
        value = base.values[index]
        if not isinstance(value, _VKValue):
            value = _VKValue(type=expected_type, literal=value)
        return self._materialize_value(value, lines, indent, expected_type) if expected_type is not None else value

    def _lower_binop(self, node, env, lines, indent, expected_type=None):
        lhs = self._lower_expr(node.left, env, lines, indent)
        rhs = self._lower_expr(node.right, env, lines, indent)
        if lhs.literal is not None and rhs.literal is not None:
            if isinstance(node.op, _ast.Mult):
                result = lhs.literal * rhs.literal
            elif isinstance(node.op, _ast.FloorDiv):
                result = lhs.literal // rhs.literal
            else:
                raise _VKernelCompileError(f"unsupported binary operator: {type(node.op).__name__}")
            return _VKValue(type=expected_type, literal=result)
        raise _VKernelCompileError("non-constant binary expressions are not supported yet")

    def _lower_expr(self, node, env, lines, indent, expected_type=None):
        if isinstance(node, _ast.Name):
            if node.id not in env:
                raise _VKernelCompileError(f"unknown name '{node.id}'")
            value = env[node.id]
            if isinstance(value, (_VKStructValue, _VKStaticSequence)):
                raise _VKernelCompileError(f"name '{node.id}' is not a scalar/SSA value")
            if (
                isinstance(value, _VKValue)
                and value.name is None
                and value.literal is not None
                and expected_type is not None
            ):
                return self._materialize_value(
                    _VKValue(type=expected_type, literal=value.literal),
                    lines,
                    indent,
                )
            return self._materialize_value(value, lines, indent, expected_type)
        if isinstance(node, _ast.Constant):
            return self._literal_value(node, lines, indent, expected_type)
        if isinstance(node, _ast.Attribute):
            return self._lower_attribute(node, env, lines, indent, expected_type)
        if isinstance(node, _ast.Subscript):
            return self._lower_subscript(node, env, lines, indent, expected_type)
        if isinstance(node, _ast.BinOp):
            return self._lower_binop(node, env, lines, indent, expected_type)
        if isinstance(node, _ast.Call):
            results = self._lower_call(node, env, lines, indent, expected_types=[expected_type] if expected_type else None)
            if len(results) != 1:
                raise _VKernelCompileError("expression expected single result")
            return results[0]
        raise _VKernelCompileError(f"unsupported expression: {type(node).__name__}")

    def _lower_call_name(self, node):
        if isinstance(node, _ast.Attribute) and isinstance(node.value, _ast.Name) and node.value.id == "pto":
            return node.attr
        raise _VKernelCompileError("only pto.* calls are supported")

    def _infer_expr_type(self, node, env):
        if isinstance(node, _ast.Name):
            if node.id not in env:
                raise _VKernelCompileError(f"unknown name '{node.id}'")
            value = env[node.id]
            return value.type if isinstance(value, _VKValue) else None
        if isinstance(node, _ast.Attribute):
            try:
                value = self._lower_attribute(node, env, [], 0)
            except _VKernelCompileError:
                return None
            return value.type if isinstance(value, _VKValue) else None
        if isinstance(node, _ast.Constant):
            return None
        return None

    def _format_typed_operands(self, values):
        return ", ".join(v.name for v in values), ", ".join(v.render_type() for v in values)

    def _lower_call(self, node, env, lines, indent, expected_types=None):
        opname = self._lower_call_name(node.func)

        if opname in ("set_loop_size_outtoub", "set_loop_size_ubtoout"):
            ops = [self._lower_expr(arg, env, lines, indent, i64) for arg in node.args]
            operands, types = self._format_typed_operands(ops)
            self._emit(lines, indent, f"pto.{opname} {operands} : {types}")
            return []

        if opname == "castptr":
            if len(node.args) != 2:
                raise _VKernelCompileError("pto.castptr expects 2 arguments")
            result_type = self._eval_type_expr(node.args[1])
            addr = self._lower_expr(node.args[0], env, lines, indent, i64)
            result = self._new_value(result_type)
            self._emit(lines, indent, f"{result.name} = pto.castptr {addr.name} : {addr.render_type()} -> {result.render_type()}")
            return [result]

        if opname == "copy_gm_to_ubuf":
            expected = [None, None, i64, i64, i64, i64, i64, i1, i64, i64, i64]
            ops = [self._lower_expr(arg, env, lines, indent, expected[i]) for i, arg in enumerate(node.args)]
            operands, types = self._format_typed_operands(ops)
            self._emit(lines, indent, f"pto.copy_gm_to_ubuf {operands} : {types}")
            return []

        if opname == "copy_ubuf_to_gm":
            expected = [None, None, i64, i64, i64, i64, i64, i64]
            ops = [self._lower_expr(arg, env, lines, indent, expected[i]) for i, arg in enumerate(node.args)]
            operands, types = self._format_typed_operands(ops)
            self._emit(lines, indent, f"pto.copy_ubuf_to_gm {operands} : {types}")
            return []

        if opname in ("set_flag", "wait_flag"):
            attrs = []
            for arg in node.args:
                if not isinstance(arg, _ast.Constant) or not isinstance(arg.value, str):
                    raise _VKernelCompileError(f"pto.{opname} expects string literals")
                attrs.append(arg.value)
            self._emit(lines, indent, f'pto.{opname}["{attrs[0]}", "{attrs[1]}", "{attrs[2]}"]')
            return []

        if opname == "barrier":
            arg = node.args[0]
            if not isinstance(arg, _ast.Constant) or not isinstance(arg.value, str):
                raise _VKernelCompileError("pto.barrier expects a string literal")
            self._emit(lines, indent, f"pto.barrier #pto.pipe<{arg.value}>")
            return []

        if opname == "plt_b32":
            src = self._lower_expr(node.args[0], env, lines, indent, i32)
            res0 = self._new_value(mask)
            res1 = self._new_value(i32)
            self._emit(lines, indent, f"{res0.name}, {res1.name} = pto.plt_b32 {src.name} : i32 -> !pto.mask, i32")
            return [res0, res1]

        if opname == "pset_b32":
            arg = node.args[0]
            if not isinstance(arg, _ast.Constant) or not isinstance(arg.value, str):
                raise _VKernelCompileError("pto.pset_b32 expects a string literal")
            res = self._new_value(mask)
            self._emit(lines, indent, f'{res.name} = pto.pset_b32 "{arg.value}" : !pto.mask')
            return [res]

        if opname == "vlds":
            ptr_value = self._lower_expr(node.args[0], env, lines, indent)
            if not isinstance(ptr_value.type, _VKernelPtrType):
                raise _VKernelCompileError("pto.vlds expects a ptr operand")
            offset = self._lower_expr(node.args[1], env, lines, indent, _vk_index)
            result = self._new_value(vreg(_ptr_vector_lanes(ptr_value.type), ptr_value.type.elem))
            self._emit(lines, indent,
                       f"{result.name} = pto.vlds {ptr_value.name}[{offset.name}] : {ptr_value.render_type()} -> {result.render_type()}")
            return [result]

        if opname == "vabs":
            vec_value = self._lower_expr(node.args[0], env, lines, indent)
            mask_value = self._lower_expr(node.args[1], env, lines, indent, mask)
            result = self._new_value(vec_value.type)
            self._emit(lines, indent,
                       f"{result.name} = pto.vabs {vec_value.name}, {mask_value.name} : {vec_value.render_type()}, {mask_value.render_type()} -> {result.render_type()}")
            return [result]

        if opname == "vsts":
            vec_value = self._lower_expr(node.args[0], env, lines, indent)
            ptr_value = self._lower_expr(node.args[1], env, lines, indent)
            offset = self._lower_expr(node.args[2], env, lines, indent, _vk_index)
            mask_value = self._lower_expr(node.args[3], env, lines, indent, mask)
            self._emit(lines, indent,
                       f"pto.vsts {vec_value.name}, {ptr_value.name}[{offset.name}], {mask_value.name} : {vec_value.render_type()}, {ptr_value.render_type()}, {mask_value.render_type()}")
            return []

        raise _VKernelCompileError(f"unsupported pto op in vkernel: {opname}")

    def _collect_assigned_names(self, statements):
        names = set()

        class Visitor(_ast.NodeVisitor):
            def visit_Assign(self, node):
                for target in node.targets:
                    self._collect_target(target)

            def _collect_target(self, target):
                if isinstance(target, _ast.Name):
                    names.add(target.id)
                elif isinstance(target, _ast.Tuple):
                    for elt in target.elts:
                        self._collect_target(elt)

        visitor = Visitor()
        for stmt in statements:
            if isinstance(stmt, (_ast.With, _ast.For, _ast.If)):
                continue
            visitor.visit(stmt)
        return names

    def _compile_block(self, statements, env, indent):
        lines = []
        current_env = dict(env)

        for stmt in statements:
            if isinstance(stmt, _ast.Assign):
                if len(stmt.targets) != 1:
                    raise _VKernelCompileError("multiple assignment targets are not supported")
                target = stmt.targets[0]
                if isinstance(target, _ast.Name):
                    value = self._lower_expr(stmt.value, current_env, lines, indent)
                    current_env[target.id] = value
                elif isinstance(target, _ast.Tuple):
                    results = self._lower_call(stmt.value, current_env, lines, indent)
                    if len(results) != len(target.elts):
                        raise _VKernelCompileError("tuple assignment arity mismatch")
                    for elt, value in zip(target.elts, results):
                        if not isinstance(elt, _ast.Name):
                            raise _VKernelCompileError("tuple assignment only supports names")
                        current_env[elt.id] = value
                else:
                    raise _VKernelCompileError("unsupported assignment target")
                continue

            if isinstance(stmt, _ast.AnnAssign):
                if stmt.value is None:
                    raise _VKernelCompileError("annotation-only assignment is not supported")
                if not isinstance(stmt.target, _ast.Name):
                    raise _VKernelCompileError("annotated assignment only supports names")
                target_type = self._eval_type_expr(stmt.annotation)
                value = self._lower_expr(stmt.value, current_env, lines, indent, target_type)
                current_env[stmt.target.id] = value
                continue

            if isinstance(stmt, _ast.Expr):
                if isinstance(stmt.value, _ast.Call):
                    self._lower_call(stmt.value, current_env, lines, indent)
                else:
                    self._lower_expr(stmt.value, current_env, lines, indent)
                continue

            if isinstance(stmt, _ast.Return):
                if stmt.value is not None:
                    raise _VKernelCompileError("only empty return is supported")
                self._emit(lines, indent, "return")
                continue

            if isinstance(stmt, _ast.With):
                if len(stmt.items) != 1:
                    raise _VKernelCompileError("only single with item is supported")
                item = stmt.items[0]
                name = self._lower_call_name(item.context_expr.func)
                if name not in ("strict_vecscope", "vecscope"):
                    raise _VKernelCompileError("unsupported with context")
                if name == "strict_vecscope":
                    body_lines, body_result = self._compile_strict_vecscope(item, stmt.body, current_env, indent)
                else:
                    body_lines, body_result = self._compile_vecscope(stmt.body, current_env, indent)
                lines.extend(body_lines)
                current_env.update(body_result)
                continue

            if isinstance(stmt, _ast.For):
                loop_lines, updated_env = self._compile_for(stmt, current_env, indent)
                lines.extend(loop_lines)
                current_env = updated_env
                continue

            if isinstance(stmt, _ast.If):
                if_lines, updated_env = self._compile_if(stmt, current_env, indent)
                lines.extend(if_lines)
                current_env = updated_env
                continue

            raise _VKernelCompileError(f"unsupported statement: {type(stmt).__name__}")

        return lines, current_env

    def _compile_vecscope(self, body, outer_env, indent):
        body_lines, _ = self._compile_block(body, dict(outer_env), indent + 1)
        lines = []
        self._emit(lines, indent, "pto.vecscope {")
        lines.extend(body_lines)
        self._emit(lines, indent, "}")
        return lines, {}

    def _compile_strict_vecscope(self, item, body, outer_env, indent):
        if not isinstance(item.optional_vars, _ast.Tuple):
            raise _VKernelCompileError("pto.strict_vecscope requires tuple binding in 'as'")
        if len(item.context_expr.args) != len(item.optional_vars.elts):
            raise _VKernelCompileError("strict_vecscope capture arity must match bound block arguments")
        arg_names = []
        inner_env = {}
        for elt in item.optional_vars.elts:
            if not isinstance(elt, _ast.Name):
                raise _VKernelCompileError("pto.strict_vecscope bindings must be names")
            arg = self._new_arg_value()
            arg_names.append((elt.id, arg))
            inner_env[elt.id] = arg

        for expr, (_, arg) in zip(item.context_expr.args, arg_names):
            inferred_type = self._infer_expr_type(expr, outer_env)
            if inferred_type is not None:
                arg.type = inferred_type

        lines = []
        body_lines, body_env = self._compile_block(body, inner_env, indent + 1)
        captures = []
        for name, arg in arg_names:
            if arg.type is None and name in body_env and body_env[name].type is not None:
                arg.type = body_env[name].type
        for expr, (_, arg) in zip(item.context_expr.args, arg_names):
            if arg.type is None:
                raise _VKernelCompileError("strict_vecscope block argument type could not be inferred")
            capture = self._lower_expr(expr, outer_env, lines, indent, expected_type=arg.type)
            captures.append(capture)
        capture_operands = ", ".join(value.name for value in captures)
        block_args = ", ".join(f"{arg.name}: {arg.render_type()}" for _, arg in arg_names)
        func_type = ", ".join(arg.render_type() for _, arg in arg_names)

        self._emit(lines, indent, f"pto.strict_vecscope({capture_operands}) {{")
        self._emit(lines, indent, f"^bb0({block_args}):")
        lines.extend(body_lines)
        self._emit(lines, indent, f"}} : ({func_type}) -> ()")
        return lines, {}

    def _compile_for(self, stmt, outer_env, indent):
        if not isinstance(stmt.target, _ast.Name):
            raise _VKernelCompileError("for target must be a single name")
        if not isinstance(stmt.iter, _ast.Call) or not isinstance(stmt.iter.func, _ast.Name) or stmt.iter.func.id != "range":
            raise _VKernelCompileError("only Python range(...) loops are supported")
        if len(stmt.iter.args) != 3:
            raise _VKernelCompileError("range expects exactly 3 arguments in vkernel")

        lines = []
        lb = self._lower_expr(stmt.iter.args[0], outer_env, lines, indent, _vk_index)
        ub = self._lower_expr(stmt.iter.args[1], outer_env, lines, indent, _vk_index)
        step = self._lower_expr(stmt.iter.args[2], outer_env, lines, indent, _vk_index)

        loop_env = dict(outer_env)
        iv = self._new_arg_value(_vk_index)
        loop_env[stmt.target.id] = iv
        candidate_carried = []
        for name in self._collect_assigned_names(stmt.body):
            if name in outer_env and name != stmt.target.id:
                iter_arg = self._new_arg_value(outer_env[name].type)
                loop_env[name] = iter_arg
                candidate_carried.append((name, outer_env[name], iter_arg))

        body_lines, body_env = self._compile_block(stmt.body, loop_env, indent + 1)
        carried = []
        for name, before, iter_arg in candidate_carried:
            after = body_env.get(name)
            if after is not None and after is not iter_arg:
                carried.append((name, before, after))

        result_prefix = ""
        yield_line = None
        if carried:
            results = [after.render_type() for _, _, after in carried]
            result_value = self._new_value()
            result_prefix = f"{result_value.name}:{len(carried)} = "
            iter_arg_map = {name: iter_arg for name, _, iter_arg in candidate_carried}
            carried_with_initials = []
            for name, before, after in carried:
                before = self._materialize_value(before, lines, indent, after.type)
                carried_with_initials.append((name, before, after))
            carried = carried_with_initials
            iter_args = ", ".join(
                f"{iter_arg_map[name].name} = {before.name}" for name, before, _ in carried
            )
            self._emit(
                lines,
                indent,
                f"{result_prefix}scf.for {iv.name} = {lb.name} to {ub.name} step {step.name} iter_args({iter_args}) -> ({', '.join(results)}) {{",
            )
            yield_line = f"scf.yield {', '.join(after.name for _, _, after in carried)} : {', '.join(results)}"
        else:
            self._emit(lines, indent, f"scf.for {iv.name} = {lb.name} to {ub.name} step {step.name} {{")
        lines.extend(body_lines)
        if yield_line:
            self._emit(lines, indent + 1, yield_line)
        self._emit(lines, indent, "}")

        updated_env = dict(outer_env)
        if carried:
            for idx, (name, _, after) in enumerate(carried):
                updated_env[name] = _project_result(result_value, idx, after.type)
        return lines, updated_env

    def _compile_if(self, stmt, outer_env, indent):
        lines = []
        cond = self._lower_expr(stmt.test, outer_env, lines, indent, i1)
        then_lines, then_env = self._compile_block(stmt.body, dict(outer_env), indent + 1)
        else_lines, else_env = self._compile_block(stmt.orelse, dict(outer_env), indent + 1)
        updated = []
        for name, before in outer_env.items():
            then_val = then_env.get(name, before)
            else_val = else_env.get(name, before)
            if then_val is not before or else_val is not before:
                if not _types_equal(then_val.type, else_val.type):
                    raise _VKernelCompileError(f"if merge type mismatch for '{name}'")
                updated.append((name, then_val, else_val))

        if updated:
            result = self._new_value()
            types = ", ".join(val.type.render() for _, val, _ in updated)
            self._emit(lines, indent, f"{result.name}:{len(updated)} = scf.if {cond.name} -> ({types}) {{")
            lines.extend(then_lines)
            self._emit(lines, indent + 1, f"scf.yield {', '.join(val.name for _, val, _ in updated)} : {types}")
            self._emit(lines, indent, "} else {")
            lines.extend(else_lines)
            self._emit(lines, indent + 1, f"scf.yield {', '.join(val.name for _, _, val in updated)} : {types}")
            self._emit(lines, indent, "}")
            updated_env = dict(outer_env)
            for idx, (name, then_val, _) in enumerate(updated):
                updated_env[name] = _project_result(result, idx, then_val.type)
            return lines, updated_env

        self._emit(lines, indent, f"scf.if {cond.name} {{")
        lines.extend(then_lines)
        self._emit(lines, indent, "} else {")
        lines.extend(else_lines)
        self._emit(lines, indent, "}")
        return lines, dict(outer_env)

    def build_text(self):
        lines = [f'module attributes {{pto.target_arch = "{self.target}"}} {{']
        arg_types = []
        env = {}
        for arg in self.fn_def.args.args:
            arg_ty = _coerce_surface_type(self.py_fn.__annotations__.get(arg.arg))
            if arg_ty is None:
                raise _VKernelCompileError(f"missing type annotation for argument '{arg.arg}'")
            if not isinstance(arg_ty, _VKernelType):
                raise _VKernelCompileError(f"unsupported type annotation for argument '{arg.arg}'")
            if isinstance(arg_ty, _VKernelStructDef):
                if arg.arg not in self.specialization:
                    raise _VKernelCompileError(
                        f"template argument '{arg.arg}: {arg_ty.name}' requires .jit(...) specialization"
                    )
                binding = self.specialization[arg.arg]
                if not isinstance(binding, _VKernelStructBinding) or binding.schema != arg_ty:
                    raise _VKernelCompileError(
                        f"specialization for '{arg.arg}' must be a {arg_ty.name}(...) binding"
                    )
                struct_fields = {}
                for field_name, field_kind in arg_ty.fields:
                    if field_name not in binding.values:
                        raise _VKernelCompileError(
                            f"missing field '{field_name}' in specialization for '{arg.arg}'"
                        )
                    field_value = binding.values[field_name]
                    if field_kind is ptr:
                        if not isinstance(field_value, _VKernelPtrType):
                            raise _VKernelCompileError(
                                f"{arg_ty.name}.{field_name} must be a pto.ptr(...) type object"
                            )
                        arg_val = self._new_arg_value(field_value)
                        arg_types.append(f"{arg_val.name}: {field_value.render()}")
                        struct_fields[field_name] = arg_val
                        continue
                    if field_kind is const:
                        if not isinstance(field_value, _VKernelConstBinding):
                            raise _VKernelCompileError(
                                f"{arg_ty.name}.{field_name} must use pto.const(...)"
                            )
                        static_value = field_value.value
                        if not isinstance(static_value, (list, tuple)) or not all(
                            isinstance(v, int) for v in static_value
                        ):
                            raise _VKernelCompileError(
                                f"{arg_ty.name}.{field_name} must be a list/tuple of ints"
                            )
                        struct_fields[field_name] = _VKStaticSequence(
                            tuple(_VKValue(literal=v) for v in static_value)
                        )
                        continue
                    raise _VKernelCompileError(
                        f"unsupported struct field kind for {arg_ty.name}.{field_name}"
                    )
                env[arg.arg] = _VKStructValue(arg_ty, struct_fields)
                continue
            arg_val = self._new_arg_value(arg_ty)
            arg_types.append(f"{arg_val.name}: {arg_ty.render()}")
            env[arg.arg] = arg_val
        self._emit(lines, 1, f"func.func @{self.kernel_name}({', '.join(arg_types)}) {{")
        body_lines, _ = self._compile_block(self.fn_def.body, env, 2)
        lines.extend(body_lines)
        if not any(line.strip() == "return" for line in body_lines):
            self._emit(lines, 2, "return")
        self._emit(lines, 1, "}")
        lines.append("}")
        return "\n".join(lines) + "\n"


class VKernelHandle:
    def __init__(self, py_fn, target="a5", name=None, verify=True, specialization=None):
        self._py_fn = py_fn
        self._target = target
        self._name = name or py_fn.__name__
        self._verify = verify
        self._specialization = specialization or {}
        self._cached_text = None

    def _load_ast(self):
        source = _textwrap.dedent(_inspect.getsource(self._py_fn))
        module = _ast.parse(source)
        for node in module.body:
            if isinstance(node, _ast.FunctionDef) and node.name == self._py_fn.__name__:
                return node
        raise _VKernelCompileError(f"failed to locate function AST for {self._py_fn.__name__}")

    def mlir_text(self):
        if self._cached_text is None:
            builder = _VKernelBuilder(
                self._py_fn,
                self._load_ast(),
                self._target,
                self._name,
                specialization=self._specialization,
            )
            self._cached_text = builder.build_text()
        return self._cached_text

    def mlir_module(self):
        with _ods_ir.Context() as ctx:
            _load_standard_dialects()
            register_dialect(ctx, load=True)
            return _ods_ir.Module.parse(self.mlir_text(), ctx)

    def verify(self):
        mod = self.mlir_module()
        mod.operation.verify()
        return True

    def dump(self):
        print(self.mlir_text(), end="")

    def emit(self, path):
        with open(path, "w", encoding="utf-8") as f:
            f.write(self.mlir_text())

    def jit(self, **kwargs):
        return VKernelHandle(
            self._py_fn,
            target=self._target,
            name=self._name,
            verify=self._verify,
            specialization=kwargs,
        )

    def __str__(self):
        return self.mlir_text()


def vkernel(py_fn=None, *, target="a5", name=None, verify=True):
    def wrap(fn):
        return VKernelHandle(fn, target=target, name=name, verify=verify)

    if py_fn is None:
        return wrap
    return wrap(py_fn)


__all__.extend([
    "vkernel",
    "VKernelHandle",
    "struct",
    "Tile",
    "tile",
    "const",
    "ptr",
    "vreg",
    "i1", "i8", "i16", "i32", "i64",
    "f16", "bf16", "f32",
    "mask", "align",
])
