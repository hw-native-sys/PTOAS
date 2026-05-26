# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

import importlib
import importlib.util
from pathlib import Path

from mlir import ir as _ods_ir

from . import _pto_ops_gen as _pto_ops_gen


def _load_local_pto_ext():
    lib_dir = Path(__file__).resolve().parent.parent / "_mlir_libs"
    for suffix in ("*.so", "*.pyd", "*.dll", "*.dylib"):
        for so_path in lib_dir.glob(f"_pto{suffix}"):
            spec = importlib.util.spec_from_file_location(
                "mlir._mlir_libs._pto", so_path
            )
            if spec and spec.loader:
                mod = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(mod)
                return mod
    raise ImportError("cannot locate local _pto extension in _mlir_libs")


try:
    _pto_mod = _load_local_pto_ext()
except Exception:
    _pto_mod = importlib.import_module(".._mlir_libs._pto", __package__)


def _export_generated_symbols():
    for name, obj in _pto_ops_gen.__dict__.items():
        if name.startswith("_"):
            continue
        globals()[name] = obj


def get_op_result_or_value(value):
    return getattr(_pto_ops_gen, "_get_op_result_or_value")(value)


_export_generated_symbols()

register_dialect = _pto_mod.register_dialect
PtrType = _pto_mod.PtrType
AsyncSessionType = _pto_mod.AsyncSessionType
AsyncEventType = _pto_mod.AsyncEventType
PrefetchAsyncContextType = _pto_mod.PrefetchAsyncContextType
HiF8Type = _pto_mod.HiF8Type
F4E1M2x2Type = _pto_mod.F4E1M2x2Type
F4E2M1x2Type = _pto_mod.F4E2M1x2Type
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
SetFmatrixMode = _pto_mod.SetFmatrixMode
SetFmatrixModeAttr = _pto_mod.SetFmatrixModeAttr
CompactMode = _pto_mod.CompactMode
CompactModeAttr = _pto_mod.CompactModeAttr
AccToVecMode = _pto_mod.AccToVecMode
AccToVecModeAttr = _pto_mod.AccToVecModeAttr
ReluPreMode = _pto_mod.ReluPreMode
ReluPreModeAttr = _pto_mod.ReluPreModeAttr
AtomicType = _pto_mod.AtomicType
AtomicTypeAttr = _pto_mod.AtomicTypeAttr
NotifyOp = _pto_mod.NotifyOp
NotifyOpAttr = _pto_mod.NotifyOpAttr
WaitCmp = _pto_mod.WaitCmp
WaitCmpAttr = _pto_mod.WaitCmpAttr
ReduceOp = _pto_mod.ReduceOp
ReduceOpAttr = _pto_mod.ReduceOpAttr
RoundMode = _pto_mod.RoundMode
RoundModeAttr = _pto_mod.RoundModeAttr
SaturationMode = _pto_mod.SaturationMode
SaturationModeAttr = _pto_mod.SaturationModeAttr
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
    "PrefetchAsyncContextType",
    "HiF8Type",
    "F4E1M2x2Type",
    "F4E2M1x2Type",
    "TensorViewType",
    "PartitionTensorViewType",
    "TileType",
    "TileBufType",
    "AddressSpace",
    "AddressSpaceAttr",
    "BLayout",
    "BLayoutAttr",
    "SLayout",
    "SLayoutAttr",
    "PadValue",
    "PadValueAttr",
    "SetFmatrixMode",
    "SetFmatrixModeAttr",
    "CompactMode",
    "CompactModeAttr",
    "AccToVecMode",
    "AccToVecModeAttr",
    "ReluPreMode",
    "ReluPreModeAttr",
    "AtomicType",
    "AtomicTypeAttr",
    "NotifyOp",
    "NotifyOpAttr",
    "WaitCmp",
    "WaitCmpAttr",
    "ReduceOp",
    "ReduceOpAttr",
    "RoundMode",
    "RoundModeAttr",
    "SaturationMode",
    "SaturationModeAttr",
    "CmpMode",
    "CmpModeAttr",
    "PIPE",
    "PipeAttr",
    "Layout",
    "LayoutAttr",
    "SyncOpType",
    "SyncOpTypeAttr",
    "EVENT",
    "EventAttr",
    "MaskPattern",
    "MaskPatternAttr",
    "QuantType",
    "QuantTypeAttr",
    "TileBufConfigAttr",
    "TileConfig",
    "make_img2col_config",
    # High-level sync helpers
    "record_event",
    "wait_event",
    "barrier",
    # Low-level sync helpers (static/dynamic event id unified API)
    "set_flag",
    "wait_flag",
    "set_flag_dyn",
    "wait_flag_dyn",
    # Inter-core sync helpers
    "sync_set",
    "sync_wait",
    "sync_set_dyn",
    "sync_wait_dyn",
    "set_ffts",
    # A5 buffer-id sync helpers
    "get_buf",
    "rls_buf",
    # Scalar pointer helpers
    "ptrtoint",
    "inttoptr",
    "load_scalar",
    "store_scalar",
    # Aliases for SyncOpType enums (for terse calls)
    "TLOAD",
    "TSTORE_ACC",
    "TSTORE_VEC",
    "TMOV_M2L",
    "TMOV_M2S",
    "TMOV_M2B",
    "TMOV_M2V",
    "TMOV_V2M",
    "TMATMUL",
    "TVEC",
    "TVECWAIT_EVENT",
    # Aliases for EVENT enums
    "EVENT_ID0",
    "EVENT_ID1",
    "EVENT_ID2",
    "EVENT_ID3",
    "EVENT_ID4",
    "EVENT_ID5",
    "EVENT_ID6",
    "EVENT_ID7",
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
        except AttributeError as exc:
            raise ValueError(f"Unknown SyncOpType name: {val}") from exc
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
        except AttributeError as exc:
            raise ValueError(f"Unknown EVENT integer id: {val}") from exc
        return EventAttr.get(enum_val, ctx)
    if isinstance(val, str):
        name = val.upper()
        try:
            enum_val = getattr(EVENT, name)
        except AttributeError as exc:
            raise ValueError(f"Unknown EVENT name: {val}") from exc
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
        except AttributeError as exc:
            raise ValueError(f"Unknown PIPE name: {val}") from exc
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
        loc=loc,
        ip=ip,
    )


def wait_event(src_op, dst_op, event_id, *, loc=None, ip=None):
    ctx = loc.context if loc else _ods_ir.Context.current
    return _pto_ops_gen.wait_event(
        _ensure_sync_attr(src_op, ctx),
        _ensure_sync_attr(dst_op, ctx),
        _ensure_event_attr(event_id, ctx),
        loc=loc,
        ip=ip,
    )


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


def _create_pipe_event_op(op_name, src_attr, dst_attr, event_id, *, loc=None, ip=None):
    return _ods_ir.Operation.create(
        op_name,
        attributes={"src_pipe": src_attr, "dst_pipe": dst_attr},
        operands=[get_op_result_or_value(event_id)],
        loc=loc,
        ip=ip,
    )


def set_flag_dyn(src_pipe, dst_pipe, event_id, *, loc=None, ip=None):
    """Low-level dynamic event-id set_flag helper."""
    ctx = loc.context if loc else _ods_ir.Context.current
    src_attr = _ensure_pipe_attr(src_pipe, ctx)
    dst_attr = _ensure_pipe_attr(dst_pipe, ctx)
    if hasattr(_pto_ops_gen, "set_flag_dyn"):
        return _pto_ops_gen.set_flag_dyn(
            src_attr,
            dst_attr,
            get_op_result_or_value(event_id),
            loc=loc,
            ip=ip,
        )
    return _create_pipe_event_op(
        "pto.set_flag_dyn", src_attr, dst_attr, event_id, loc=loc, ip=ip
    )


def wait_flag_dyn(src_pipe, dst_pipe, event_id, *, loc=None, ip=None):
    """Low-level dynamic event-id wait_flag helper."""
    ctx = loc.context if loc else _ods_ir.Context.current
    src_attr = _ensure_pipe_attr(src_pipe, ctx)
    dst_attr = _ensure_pipe_attr(dst_pipe, ctx)
    if hasattr(_pto_ops_gen, "wait_flag_dyn"):
        return _pto_ops_gen.wait_flag_dyn(
            src_attr,
            dst_attr,
            get_op_result_or_value(event_id),
            loc=loc,
            ip=ip,
        )
    return _create_pipe_event_op(
        "pto.wait_flag_dyn", src_attr, dst_attr, event_id, loc=loc, ip=ip
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
    event_val = get_op_result_or_value(event_id)
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
    event_val = get_op_result_or_value(event_id)
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
        operands=[get_op_result_or_value(ffts)],
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


def ptrtoint(ptr, *, loc=None, ip=None):
    operands = [
        get_op_result_or_value(ptr),
    ]
    op = _ods_ir.Operation.create(
        "pto.ptrtoint",
        results=[_ods_ir.IntegerType.get_signless(64)],
        operands=operands,
        loc=loc,
        ip=ip,
    )
    return op.results[0]


def inttoptr(result_type, addr, *, loc=None, ip=None):
    operands = [
        get_op_result_or_value(addr),
    ]
    op = _ods_ir.Operation.create(
        "pto.inttoptr",
        results=[result_type],
        operands=operands,
        loc=loc,
        ip=ip,
    )
    return op.results[0]


def load_scalar(result_type, ptr, offset, *, loc=None, ip=None):
    operands = [
        get_op_result_or_value(ptr),
        get_op_result_or_value(offset),
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
        get_op_result_or_value(ptr),
        get_op_result_or_value(offset),
        get_op_result_or_value(value),
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


def make_img2col_config(
    *,
    fmap_h,
    fmap_w,
    pad_l=0,
    pad_r=0,
    pad_t=0,
    pad_b=0,
    stride_h=1,
    stride_w=1,
    dilation_h=1,
    dilation_w=1,
    filter_h=1,
    filter_w=1,
    channel_size,
    dst_stride,
    dst_m_position=0,
    repeat_stride,
    repeat_time,
    repeat_mode,
    transpose=0,
    context=None,
):
    ctx = context if context is not None else _ods_ir.Context.current
    i64 = _ods_ir.IntegerType.get_signless(64, ctx)

    def _i64(value):
        return _ods_ir.IntegerAttr.get(i64, int(value))

    return _ods_ir.DictAttr.get(
        {
            "fmap_h": _i64(fmap_h),
            "fmap_w": _i64(fmap_w),
            "pad_l": _i64(pad_l),
            "pad_r": _i64(pad_r),
            "pad_t": _i64(pad_t),
            "pad_b": _i64(pad_b),
            "stride_h": _i64(stride_h),
            "stride_w": _i64(stride_w),
            "dilation_h": _i64(dilation_h),
            "dilation_w": _i64(dilation_w),
            "filter_h": _i64(filter_h),
            "filter_w": _i64(filter_w),
            "channel_size": _i64(channel_size),
            "dst_stride": _i64(dst_stride),
            "dst_m_position": _i64(dst_m_position),
            "repeat_stride": _i64(repeat_stride),
            "repeat_time": _i64(repeat_time),
            "repeat_mode": _i64(repeat_mode),
            "transpose": _i64(transpose),
        },
        context=ctx,
    )


_PARTITION_VIEW_UNSET = object()
_GeneratedPartitionViewOp = PartitionViewOp
_TSCATTER_UNSET = object()


class PartitionViewOp(_GeneratedPartitionViewOp):
    """Compatibility wrapper for inferred-result partition_view builders."""

    def __init__(
        self,
        *args,
        result=_PARTITION_VIEW_UNSET,
        source=_PARTITION_VIEW_UNSET,
        offsets=_PARTITION_VIEW_UNSET,
        sizes=_PARTITION_VIEW_UNSET,
        loc=None,
        ip=None,
    ):
        if result is not _PARTITION_VIEW_UNSET:
            if source is _PARTITION_VIEW_UNSET:
                if not args:
                    raise TypeError("missing required argument: source")
                source, *args = args
            self._init_explicit(result, source, offsets, sizes, args, loc, ip)
            return

        if source is not _PARTITION_VIEW_UNSET:
            self._init_inferred(source, offsets, sizes, args, loc, ip)
            return

        if len(args) == 4 and offsets is _PARTITION_VIEW_UNSET and sizes is _PARTITION_VIEW_UNSET:
            result, source, offsets, sizes = args
            self._init_explicit(result, source, offsets, sizes, (), loc, ip)
            return

        if len(args) == 2 and offsets is not _PARTITION_VIEW_UNSET and sizes is not _PARTITION_VIEW_UNSET:
            result, source = args
            self._init_explicit(result, source, offsets, sizes, (), loc, ip)
            return

        kwargs = {}
        if offsets is not _PARTITION_VIEW_UNSET:
            kwargs["offsets"] = offsets
        if sizes is not _PARTITION_VIEW_UNSET:
            kwargs["sizes"] = sizes
        super().__init__(*args, **kwargs, loc=loc, ip=ip)

    def _init_inferred(self, source, offsets, sizes, args, loc, ip):
        if offsets is _PARTITION_VIEW_UNSET:
            if not args:
                raise TypeError("missing required argument: offsets")
            offsets, *args = args
        if sizes is _PARTITION_VIEW_UNSET:
            if not args:
                raise TypeError("missing required argument: sizes")
            sizes, *args = args
        if args:
            raise TypeError(f"too many positional arguments: {len(args)}")
        source_value = _pto_ops_gen._get_op_result_or_value(source)
        source_type = source_value.type
        result = PartitionTensorViewType.get(source_type.rank, source_type.element_type)
        self._init_explicit(result, source_value, offsets, sizes, (), loc, ip)

    def _init_explicit(self, result, source, offsets, sizes, args, loc, ip):
        if offsets is _PARTITION_VIEW_UNSET:
            if not args:
                raise TypeError("missing required argument: offsets")
            offsets, *args = args
        if sizes is _PARTITION_VIEW_UNSET:
            if not args:
                raise TypeError("missing required argument: sizes")
            sizes, *args = args
        if args:
            raise TypeError(f"too many positional arguments: {len(args)}")
        operands = [
            _pto_ops_gen._get_op_result_or_value(source),
            _pto_ops_gen._get_op_results_or_values(offsets),
            _pto_ops_gen._get_op_results_or_values(sizes),
        ]
        op = self.build_generic(
            attributes={},
            results=[result],
            operands=operands,
            successors=None,
            regions=None,
            loc=loc,
            ip=ip,
        )
        _ods_ir.OpView.__init__(self, op)


def partition_view(*args, **kwargs) -> _ods_ir.Value:
    return PartitionViewOp(*args, **kwargs).result


PartitionView = PartitionViewOp


_GeneratedTScatterOp = TScatterOp


class TScatterOp(_GeneratedTScatterOp):
    """Compatibility wrapper for legacy positional-index tscatter builders."""

    def __init__(
        self,
        src,
        *args,
        dst=_TSCATTER_UNSET,
        indexes=_TSCATTER_UNSET,
        maskPattern=_TSCATTER_UNSET,
        loc=None,
        ip=None,
    ):
        if len(args) > 2:
            raise TypeError(f"too many positional arguments: {len(args) + 1}")

        def _is_mask_pattern(value):
            return isinstance(value, (MaskPattern, MaskPatternAttr))

        def _value_type(value):
            try:
                return _pto_ops_gen._get_op_result_or_value(value).type
            except Exception:
                return None

        def _matches_src_type(value):
            src_value = _pto_ops_gen._get_op_result_or_value(src)
            value_type = _value_type(value)
            return value_type is not None and value_type == src_value.type

        if len(args) == 2:
            if (
                dst is not _TSCATTER_UNSET
                or indexes is not _TSCATTER_UNSET
                or maskPattern is not _TSCATTER_UNSET
            ):
                raise TypeError(
                    "two positional mode arguments cannot be combined with "
                    "'dst', 'indexes', or 'maskPattern' keywords"
                )
            first, second = args
            if _is_mask_pattern(second):
                dst = first
                maskPattern = second
            elif _matches_src_type(first) and not _matches_src_type(second):
                dst = first
                indexes = second
            elif _matches_src_type(second) and not _matches_src_type(first):
                indexes = first
                dst = second
            else:
                indexes = first
                dst = second
        elif len(args) == 1:
            positional = args[0]
            if dst is _TSCATTER_UNSET:
                if indexes is _TSCATTER_UNSET and maskPattern is _TSCATTER_UNSET:
                    raise TypeError(
                        "missing required mode for tscatter; provide 'indexes' "
                        "or 'maskPattern'"
                    )
                dst = positional
            else:
                if indexes is not _TSCATTER_UNSET or maskPattern is not _TSCATTER_UNSET:
                    raise TypeError(
                        "positional mode argument cannot be combined with "
                        "'indexes' or 'maskPattern' keywords"
                    )
                if _is_mask_pattern(positional):
                    maskPattern = positional
                else:
                    indexes = positional

        kwargs = {}
        if dst is _TSCATTER_UNSET:
            raise TypeError("missing required argument: dst")
        kwargs["dst"] = dst
        if indexes is not _TSCATTER_UNSET:
            kwargs["indexes"] = indexes
        if maskPattern is not _TSCATTER_UNSET:
            kwargs["maskPattern"] = maskPattern
        super().__init__(src, **kwargs, loc=loc, ip=ip)


TScatter = TScatterOp
if "TScatter" not in __all__:
    __all__.append("TScatter")


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
