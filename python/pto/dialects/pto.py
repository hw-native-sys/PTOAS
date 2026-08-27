# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

import functools
from typing import Optional

from ptoas.mlir import ir as _ods_ir

from . import _pto_ops_gen as _pto_ops_gen
from ._ods_common import (
    get_default_loc_context as _ods_get_default_loc_context,
    get_op_result_or_value as _get_op_result_or_value,
    get_op_results_or_values as _get_op_results_or_values,
)
from ptoas import _core as _pto_mod


def _export_generated_symbols():
    for name, obj in _pto_ops_gen.__dict__.items():
        if name.startswith("_"):
            continue
        globals()[name] = obj


def get_op_result_or_value(value):
    return _get_op_result_or_value(value)


def _export_optional_cext_symbol(name):
    symbol = getattr(_pto_mod, name, None)
    if symbol is not None:
        globals()[name] = symbol
    return symbol


_export_generated_symbols()

register_dialect = _pto_mod.register_dialect
PtrType = _pto_mod.PtrType
VRegType = _pto_mod.VRegType
MaskType = _pto_mod.MaskType
VMIVRegType = _pto_mod.VMIVRegType
VMIMaskType = _pto_mod.VMIMaskType
AlignType = _pto_mod.AlignType
StructType = _export_optional_cext_symbol("StructType")
AsyncSessionType = _pto_mod.AsyncSessionType
AsyncEventType = _pto_mod.AsyncEventType
PrefetchAsyncContextType = _export_optional_cext_symbol("PrefetchAsyncContextType")
HiF8Type = _pto_mod.HiF8Type
HiF8x2Type = _pto_mod.HiF8x2Type
F8E8M0Type = _pto_mod.F8E8M0Type
BF16x2Type = _pto_mod.BF16x2Type
F4E1M2x2Type = _pto_mod.F4E1M2x2Type
F4E2M1x2Type = _pto_mod.F4E2M1x2Type
TensorViewType = _pto_mod.TensorViewType
PartitionTensorViewType = _pto_mod.PartitionTensorViewType
TileType = _pto_mod.TileType
TileBufType = _pto_mod.TileBufType
AddressSpace = _pto_mod.AddressSpace
AddressSpaceAttr = _pto_mod.AddressSpaceAttr
FenceScope = _pto_mod.FenceScope
FenceScopeAttr = _pto_mod.FenceScopeAttr
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
TInsertMode = _pto_mod.TInsertMode
TInsertModeAttr = _pto_mod.TInsertModeAttr
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
DivPrecision = _pto_mod.DivPrecision
DivPrecisionAttr = _pto_mod.DivPrecisionAttr
ExpPrecision = _pto_mod.ExpPrecision
ExpPrecisionAttr = _pto_mod.ExpPrecisionAttr
LogPrecision = _pto_mod.LogPrecision
LogPrecisionAttr = _pto_mod.LogPrecisionAttr
RecipPrecision = _pto_mod.RecipPrecision
RecipPrecisionAttr = _pto_mod.RecipPrecisionAttr
RemPrecision = _pto_mod.RemPrecision
RemPrecisionAttr = _pto_mod.RemPrecisionAttr
RsqrtPrecision = _pto_mod.RsqrtPrecision
RsqrtPrecisionAttr = _pto_mod.RsqrtPrecisionAttr
SqrtPrecision = _pto_mod.SqrtPrecision
SqrtPrecisionAttr = _pto_mod.SqrtPrecisionAttr
FmodPrecision = _pto_mod.FmodPrecision
FmodPrecisionAttr = _pto_mod.FmodPrecisionAttr
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
Coalesce = _pto_mod.Coalesce
CoalesceAttr = _pto_mod.CoalesceAttr
MaskPattern = _pto_mod.MaskPattern
MaskPatternAttr = _pto_mod.MaskPatternAttr
QuantType = _pto_mod.QuantType
QuantTypeAttr = _pto_mod.QuantTypeAttr
QuantScaleAlg = _pto_mod.QuantScaleAlg
QuantScaleAlgAttr = _pto_mod.QuantScaleAlgAttr
MxGroupAxis = _pto_mod.MxGroupAxis
MxGroupAxisAttr = _pto_mod.MxGroupAxisAttr
VecStoreMode = _pto_mod.VecStoreMode
VecStoreModeAttr = _pto_mod.VecStoreModeAttr


_ptr_type_get_impl = PtrType.get


def _ptr_type_get_compat(cls, element_type, memory_space=None, context=None):
    if isinstance(memory_space, _ods_ir.Context):
        if context is not None:
            raise TypeError("PtrType.get got multiple context arguments")
        context = memory_space
        memory_space = None
    if memory_space is None:
        if context is None:
            return _ptr_type_get_impl(element_type)
        return _ptr_type_get_impl(element_type, context=context)
    try:
        return _ptr_type_get_impl(
            element_type, memory_space=memory_space, context=context
        )
    except TypeError as exc:
        raise TypeError(
            "PtrType.get(element_type, memory_space=...) requires a PTO Python "
            "extension built with non-default address-space pointer support"
        ) from exc


PtrType.get = classmethod(_ptr_type_get_compat)


def _default_precision_type_attr(attr_cls, enum_value, loc=None):
    ctx = _ods_get_default_loc_context(loc)
    return attr_cls.get(enum_value, ctx)


def _install_default_precision_type_compat():
    specs = (
        ("TDivOp", DivPrecisionAttr, DivPrecision.Default),
        ("TDivSOp", DivPrecisionAttr, DivPrecision.Default),
        ("TExpOp", ExpPrecisionAttr, ExpPrecision.Default),
        ("TLogOp", LogPrecisionAttr, LogPrecision.Default),
        ("TRecipOp", RecipPrecisionAttr, RecipPrecision.Default),
        ("TRowExpandDivOp", DivPrecisionAttr, DivPrecision.Default),
        ("TRsqrtOp", RsqrtPrecisionAttr, RsqrtPrecision.Default),
        ("TSqrtOp", SqrtPrecisionAttr, SqrtPrecision.Default),
        ("TColExpandDivOp", DivPrecisionAttr, DivPrecision.Default),
    )
    for op_name, attr_cls, enum_value in specs:
        op_cls = getattr(_pto_ops_gen, op_name, None)
        if op_cls is None or getattr(op_cls, "_pto_default_precision_type_compat", False):
            continue
        original_init = op_cls.__init__

        @functools.wraps(original_init)
        def compat_init(
            self,
            *args,
            __orig_init=original_init,
            __attr_cls=attr_cls,
            __enum_value=enum_value,
            precisionType=None,
            **kwargs,
        ):
            if precisionType is None:
                precisionType = _default_precision_type_attr(
                    __attr_cls, __enum_value, kwargs.get("loc")
                )
            __orig_init(self, *args, precisionType=precisionType, **kwargs)

        op_cls.__init__ = compat_init
        op_cls._pto_default_precision_type_compat = True


_install_default_precision_type_compat()


def _install_enum_attr_builders():
    def address_space_attr_builder(value, context=None):
        return AddressSpaceAttr.get(value, context)

    def fence_scope_attr_builder(value, context=None):
        return FenceScopeAttr.get(value, context)

    _ods_ir.AttrBuilder.insert(
        "PTO_AddressSpaceAttr", address_space_attr_builder, replace=True
    )
    _ods_ir.AttrBuilder.insert(
        "PTO_FenceScopeAttr", fence_scope_attr_builder, replace=True
    )


_install_enum_attr_builders()

__all__ = [
    # Dialect utilities
    "register_dialect",
    # Types
    "PtrType",
    "VRegType",
    "MaskType",
    "VMIVRegType",
    "VMIMaskType",
    "AlignType",
    "AsyncSessionType",
    "AsyncEventType",
    "HiF8Type",
    "HiF8x2Type",
    "F8E8M0Type",
    "BF16x2Type",
    "F4E1M2x2Type",
    "F4E2M1x2Type",
    "TensorViewType",
    "PartitionTensorViewType",
    "TileType",
    "TileBufType",
    "AddressSpace",
    "AddressSpaceAttr",
    "FenceScope",
    "FenceScopeAttr",
    "BLayout",
    "BLayoutAttr",
    "SLayout",
    "SLayoutAttr",
    "PadValue",
    "PadValueAttr",
    "CompactMode",
    "CompactModeAttr",
    "AccToVecMode",
    "AccToVecModeAttr",
    "TInsertMode",
    "TInsertModeAttr",
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
    "DivPrecision",
    "DivPrecisionAttr",
    "ExpPrecision",
    "ExpPrecisionAttr",
    "LogPrecision",
    "LogPrecisionAttr",
    "RecipPrecision",
    "RecipPrecisionAttr",
    "RemPrecision",
    "RemPrecisionAttr",
    "RsqrtPrecision",
    "RsqrtPrecisionAttr",
    "SqrtPrecision",
    "SqrtPrecisionAttr",
    "FmodPrecision",
    "FmodPrecisionAttr",
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
    "Coalesce",
    "CoalesceAttr",
    "MaskPattern",
    "MaskPatternAttr",
    "QuantType",
    "QuantTypeAttr",
    "QuantScaleAlg",
    "QuantScaleAlgAttr",
    "MxGroupAxis",
    "MxGroupAxisAttr",
    "VecStoreMode",
    "VecStoreModeAttr",
    "TileBufConfigAttr",
    "TileConfig",
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

if PrefetchAsyncContextType is not None:
    __all__.insert(__all__.index("HiF8Type"), "PrefetchAsyncContextType")

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


def sync_wait_dyn(pipe, event_id, ffts_mode=2, *, loc=None, ip=None):
    ctx = loc.context if loc else _ods_ir.Context.current
    pipe_attr = _ensure_pipe_attr(pipe, ctx)
    event_val = get_op_result_or_value(event_id)
    try:
        mode_attr = None if ffts_mode == 2 else _ensure_i32_attr(ffts_mode, "ffts_mode", ctx)
        return _pto_ops_gen.sync_wait(
            pipe_attr, event_id=None, ffts_mode=mode_attr, event_id_dyn=event_val, loc=loc, ip=ip
        )
    except TypeError:
        if hasattr(_pto_ops_gen, "sync_wait_dyn"):
            return _pto_ops_gen.sync_wait_dyn(pipe_attr, event_val, loc=loc, ip=ip)
        raise


def sync_wait(pipe, event_id, ffts_mode=2, *, loc=None, ip=None):
    ctx = loc.context if loc else _ods_ir.Context.current
    pipe_attr = _ensure_pipe_attr(pipe, ctx)
    if _is_static_i32_event_id(event_id):
        event_attr = _ensure_i32_attr(event_id, "event_id", ctx)
        try:
            mode_attr = None if ffts_mode == 2 else _ensure_i32_attr(ffts_mode, "ffts_mode", ctx)
            return _pto_ops_gen.sync_wait(
                pipe_attr, event_id=event_attr, ffts_mode=mode_attr, event_id_dyn=None, loc=loc, ip=ip
            )
        except TypeError:
            return _ods_ir.Operation.create(
                "pto.sync.wait",
                attributes={
                    "pipe": pipe_attr,
                    "event_id": event_attr,
                    **(
                        {}
                        if ffts_mode == 2
                        else {"ffts_mode": _ensure_i32_attr(ffts_mode, "ffts_mode", ctx)}
                    ),
                },
                loc=loc,
                ip=ip,
            )
    return sync_wait_dyn(pipe_attr, event_id, ffts_mode=ffts_mode, loc=loc, ip=ip)


# -----------------------------------------------------------------------------
# Named cross-/intra-block synchronization instructions.
# -----------------------------------------------------------------------------

def _named_sync_event_op(name, pipe, event_id, *, loc=None, ip=None):
    ctx = loc.context if loc else _ods_ir.Context.current
    pipe_attr = _ensure_pipe_attr(pipe, ctx)
    attrs = {"pipe": pipe_attr}
    if _is_static_i32_event_id(event_id):
        attrs["event_id"] = _ensure_i32_attr(event_id, "event_id", ctx)
        return _ods_ir.Operation.create(name, attributes=attrs, loc=loc, ip=ip)
    return _ods_ir.Operation.create(
        name, attributes=attrs, operands=[get_op_result_or_value(event_id)], loc=loc, ip=ip
    )


def set_cross_block(pipe, event_id, *, loc=None, ip=None):
    """Emit ``pto.set_cross_block`` (FFTS cross-block signal)."""
    return _named_sync_event_op("pto.set_cross_block", pipe, event_id, loc=loc, ip=ip)


def wait_cross_block(pipe, event_id, *, loc=None, ip=None):
    """Emit ``pto.wait_cross_block`` (FFTS cross-block wait)."""
    return _named_sync_event_op("pto.wait_cross_block", pipe, event_id, loc=loc, ip=ip)


def set_intra_block(pipe, event_id, *, loc=None, ip=None):
    """Emit ``pto.set_intra_block`` (intra-block signal)."""
    return _named_sync_event_op("pto.set_intra_block", pipe, event_id, loc=loc, ip=ip)


def wait_intra_block(pipe, event_id, *, loc=None, ip=None):
    """Emit ``pto.wait_intra_block`` (intra-block wait)."""
    return _named_sync_event_op("pto.wait_intra_block", pipe, event_id, loc=loc, ip=ip)


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
        source_value = _get_op_result_or_value(source)
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
            _get_op_result_or_value(source),
            _get_op_results_or_values(offsets),
            _get_op_results_or_values(sizes),
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


def _is_tscatter_mask_pattern(value):
    return isinstance(value, (MaskPattern, MaskPatternAttr))


def _tscatter_value_type(value):
    try:
        return _get_op_result_or_value(value).type
    except Exception:
        return None


def _normalize_tscatter_positional(src, args, dst, indexes, mask_pattern):
    if len(args) > 2:
        raise TypeError(f"too many positional arguments: {len(args) + 1}")
    if len(args) == 2:
        if dst is not _TSCATTER_UNSET or indexes is not _TSCATTER_UNSET or mask_pattern is not _TSCATTER_UNSET:
            raise TypeError(
                "two positional mode arguments cannot be combined with "
                "'dst', 'indexes', or 'maskPattern' keywords"
            )
        first, second = args
        src_type = _get_op_result_or_value(src).type
        first_type = _tscatter_value_type(first)
        second_type = _tscatter_value_type(second)
        if _is_tscatter_mask_pattern(second):
            return first, _TSCATTER_UNSET, second
        if first_type is not None and first_type == src_type and second_type != src_type:
            return first, second, _TSCATTER_UNSET
        if second_type is not None and second_type == src_type and first_type != src_type:
            return second, first, _TSCATTER_UNSET
        return second, first, _TSCATTER_UNSET
    if len(args) == 1:
        positional = args[0]
        if dst is _TSCATTER_UNSET:
            if indexes is _TSCATTER_UNSET and mask_pattern is _TSCATTER_UNSET:
                raise TypeError(
                    "missing required mode for tscatter; provide 'indexes' "
                    "or 'maskPattern'"
                )
            return positional, indexes, mask_pattern
        if indexes is not _TSCATTER_UNSET or mask_pattern is not _TSCATTER_UNSET:
            raise TypeError(
                "positional mode argument cannot be combined with "
                "'indexes' or 'maskPattern' keywords"
            )
        if _is_tscatter_mask_pattern(positional):
            return dst, _TSCATTER_UNSET, positional
        return dst, positional, _TSCATTER_UNSET
    return dst, indexes, mask_pattern


def _build_tscatter_kwargs(dst, indexes, mask_pattern, axis):
    if dst is _TSCATTER_UNSET:
        raise TypeError("missing required argument: dst")
    kwargs = {"dst": dst}
    if indexes is not _TSCATTER_UNSET:
        kwargs["indexes"] = indexes
    if mask_pattern is not _TSCATTER_UNSET:
        kwargs["maskPattern"] = mask_pattern
    if axis is not None:
        kwargs["axis"] = axis
    return kwargs


class TScatterOp(_GeneratedTScatterOp):
    """Compatibility wrapper for legacy positional-index tscatter builders."""

    def __init__(
        self,
        src,
        *args,
        dst=_TSCATTER_UNSET,
        indexes=_TSCATTER_UNSET,
        maskPattern=_TSCATTER_UNSET,
        axis=None,
        loc=None,
        ip=None,
    ):
        dst, indexes, maskPattern = _normalize_tscatter_positional(
            src,
            args,
            dst,
            indexes,
            maskPattern,
        )
        kwargs = _build_tscatter_kwargs(dst, indexes, maskPattern, axis)
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
    name: Optional[str] = None
    type: Optional[_VKernelType] = None
    literal: Optional[object] = None

    def render_type(self):
        if self.type is None:
            raise _VKernelCompileError(f"unresolved type for {self.name}")
        return self.type.render()


def _project_result(group, index, ty):
    return _VKValue(f"{group.name}#{index}", ty)


def _load_standard_dialects():
    try:
        from ptoas.mlir.dialects import arith as _mlir_arith  # noqa: F401
        from ptoas.mlir.dialects import func as _mlir_func  # noqa: F401
        from ptoas.mlir.dialects import scf as _mlir_scf  # noqa: F401
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

    def _lower_call_memory(self, opname, node, env, lines, indent):
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
            self._emit(
                lines,
                indent,
                f"{result.name} = pto.castptr {addr.name} : {addr.render_type()} -> {result.render_type()}",
            )
            return [result]
        copy_specs = {
            "copy_gm_to_ubuf": [None, None, i64, i64, i64, i64, i64, i1, i64, i64, i64],
            "copy_ubuf_to_gm": [None, None, i64, i64, i64, i64, i64, i64],
        }
        if opname in copy_specs:
            expected = copy_specs[opname]
            ops = [
                self._lower_expr(arg, env, lines, indent, expected[i])
                for i, arg in enumerate(node.args)
            ]
            operands, types = self._format_typed_operands(ops)
            self._emit(lines, indent, f"pto.{opname} {operands} : {types}")
            return []
        return None

    def _lower_call_sync(self, opname, node, lines, indent):
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
        return None

    def _lower_call_vector_mask(self, opname, node, env, lines, indent):
        if opname == "plt_b32":
            src = self._lower_expr(node.args[0], env, lines, indent, i32)
            res0 = self._new_value(mask)
            res1 = self._new_value(i32)
            self._emit(
                lines,
                indent,
                f"{res0.name}, {res1.name} = pto.plt_b32 {src.name} : i32 -> !pto.mask, i32",
            )
            return [res0, res1]
        if opname == "pset_b32":
            arg = node.args[0]
            if not isinstance(arg, _ast.Constant) or not isinstance(arg.value, str):
                raise _VKernelCompileError("pto.pset_b32 expects a string literal")
            result = self._new_value(mask)
            self._emit(lines, indent, f'{result.name} = pto.pset_b32 "{arg.value}" : !pto.mask')
            return [result]
        return None

    def _lower_call_vector_memory(self, opname, node, env, lines, indent):
        if opname == "vlds":
            ptr_value = self._lower_expr(node.args[0], env, lines, indent)
            if not isinstance(ptr_value.type, _VKernelPtrType):
                raise _VKernelCompileError("pto.vlds expects a ptr operand")
            offset = self._lower_expr(node.args[1], env, lines, indent, _vk_index)
            result = self._new_value(vreg(_ptr_vector_lanes(ptr_value.type), ptr_value.type.elem))
            self._emit(
                lines,
                indent,
                f"{result.name} = pto.vlds {ptr_value.name}[{offset.name}] : "
                f"{ptr_value.render_type()} -> {result.render_type()}",
            )
            return [result]
        if opname == "vabs":
            vec_value = self._lower_expr(node.args[0], env, lines, indent)
            mask_value = self._lower_expr(node.args[1], env, lines, indent, mask)
            result = self._new_value(vec_value.type)
            self._emit(
                lines,
                indent,
                f"{result.name} = pto.vabs {vec_value.name}, {mask_value.name} : "
                f"{vec_value.render_type()}, {mask_value.render_type()} -> {result.render_type()}",
            )
            return [result]
        if opname == "vsts":
            vec_value = self._lower_expr(node.args[0], env, lines, indent)
            ptr_value = self._lower_expr(node.args[1], env, lines, indent)
            offset = self._lower_expr(node.args[2], env, lines, indent, _vk_index)
            mask_value = self._lower_expr(node.args[3], env, lines, indent, mask)
            self._emit(
                lines,
                indent,
                f"pto.vsts {vec_value.name}, {ptr_value.name}[{offset.name}], {mask_value.name} : "
                f"{vec_value.render_type()}, {ptr_value.render_type()}, {mask_value.render_type()}",
            )
            return []
        return None

    def _lower_call_vector(self, opname, node, env, lines, indent):
        result = self._lower_call_vector_mask(opname, node, env, lines, indent)
        return result if result is not None else self._lower_call_vector_memory(
            opname, node, env, lines, indent
        )

    def _lower_call(self, node, env, lines, indent, expected_types=None):
        opname = self._lower_call_name(node.func)
        handlers = (
            (self._lower_call_memory, True),
            (self._lower_call_sync, False),
            (self._lower_call_vector, True),
        )
        for handler, needs_env in handlers:
            result = (
                handler(opname, node, env, lines, indent)
                if needs_env
                else handler(opname, node, lines, indent)
            )
            if result is not None:
                return result
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

    def _compile_assignment(self, stmt, current_env, lines, indent):
        if len(stmt.targets) != 1:
            raise _VKernelCompileError("multiple assignment targets are not supported")
        target = stmt.targets[0]
        if isinstance(target, _ast.Name):
            current_env[target.id] = self._lower_expr(stmt.value, current_env, lines, indent)
            return current_env
        if isinstance(target, _ast.Tuple):
            results = self._lower_call(stmt.value, current_env, lines, indent)
            if len(results) != len(target.elts):
                raise _VKernelCompileError("tuple assignment arity mismatch")
            for elt, value in zip(target.elts, results):
                if not isinstance(elt, _ast.Name):
                    raise _VKernelCompileError("tuple assignment only supports names")
                current_env[elt.id] = value
            return current_env
        raise _VKernelCompileError("unsupported assignment target")

    def _compile_annotated_assignment(self, stmt, current_env, lines, indent):
        if stmt.value is None:
            raise _VKernelCompileError("annotation-only assignment is not supported")
        if not isinstance(stmt.target, _ast.Name):
            raise _VKernelCompileError("annotated assignment only supports names")
        target_type = self._eval_type_expr(stmt.annotation)
        current_env[stmt.target.id] = self._lower_expr(
            stmt.value,
            current_env,
            lines,
            indent,
            target_type,
        )
        return current_env

    def _compile_with(self, stmt, current_env, lines, indent):
        if len(stmt.items) != 1:
            raise _VKernelCompileError("only single with item is supported")
        item = stmt.items[0]
        name = self._lower_call_name(item.context_expr.func)
        if name not in ("strict_vecscope", "vecscope"):
            raise _VKernelCompileError("unsupported with context")
        if name == "strict_vecscope":
            body_lines, body_result = self._compile_strict_vecscope(
                item, stmt.body, current_env, indent
            )
        else:
            body_lines, body_result = self._compile_vecscope(stmt.body, current_env, indent)
        lines.extend(body_lines)
        current_env.update(body_result)
        return current_env

    def _compile_statement(self, stmt, current_env, lines, indent):
        if isinstance(stmt, _ast.Assign):
            return self._compile_assignment(stmt, current_env, lines, indent)
        if isinstance(stmt, _ast.AnnAssign):
            return self._compile_annotated_assignment(stmt, current_env, lines, indent)
        if isinstance(stmt, _ast.Expr):
            if isinstance(stmt.value, _ast.Call):
                self._lower_call(stmt.value, current_env, lines, indent)
            else:
                self._lower_expr(stmt.value, current_env, lines, indent)
            return current_env
        if isinstance(stmt, _ast.Return):
            if stmt.value is not None:
                raise _VKernelCompileError("only empty return is supported")
            self._emit(lines, indent, "return")
            return current_env
        if isinstance(stmt, _ast.With):
            return self._compile_with(stmt, current_env, lines, indent)
        if isinstance(stmt, _ast.For):
            loop_lines, updated_env = self._compile_for(stmt, current_env, indent)
            lines.extend(loop_lines)
            return updated_env
        if isinstance(stmt, _ast.If):
            if_lines, updated_env = self._compile_if(stmt, current_env, indent)
            lines.extend(if_lines)
            return updated_env
        raise _VKernelCompileError(f"unsupported statement: {type(stmt).__name__}")

    def _compile_block(self, statements, env, indent):
        lines = []
        current_env = dict(env)
        for stmt in statements:
            current_env = self._compile_statement(stmt, current_env, lines, indent)

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
        bounds = [
            self._lower_expr(arg, outer_env, lines, indent, _vk_index)
            for arg in stmt.iter.args
        ]
        loop_env, iv, candidate_carried = self._prepare_loop_environment(stmt, outer_env)
        body_lines, body_env = self._compile_block(stmt.body, loop_env, indent + 1)
        carried = self._resolve_carried_values(candidate_carried, body_env)
        result_value = self._emit_for_header(lines, indent, bounds, iv, candidate_carried, carried)
        lines.extend(body_lines)
        if carried:
            result_types = ", ".join(after.render_type() for _, _, after in carried)
            carried_names = ", ".join(after.name for _, _, after in carried)
            self._emit(
                lines,
                indent + 1,
                f"scf.yield {carried_names} : {result_types}",
            )
        self._emit(lines, indent, "}")
        return lines, self._loop_updated_env(outer_env, carried, result_value)

    def _prepare_loop_environment(self, stmt, outer_env):
        loop_env = dict(outer_env)
        iv = self._new_arg_value(_vk_index)
        loop_env[stmt.target.id] = iv
        candidate_carried = []
        for name in self._collect_assigned_names(stmt.body):
            if name in outer_env and name != stmt.target.id:
                iter_arg = self._new_arg_value(outer_env[name].type)
                loop_env[name] = iter_arg
                candidate_carried.append((name, outer_env[name], iter_arg))
        return loop_env, iv, candidate_carried

    def _resolve_carried_values(self, candidate_carried, body_env):
        return [
            (name, before, after)
            for name, before, iter_arg in candidate_carried
            if (after := body_env.get(name)) is not None and after is not iter_arg
        ]

    def _emit_for_header(self, lines, indent, bounds, iv, candidates, carried):
        if not carried:
            self._emit(
                lines,
                indent,
                f"scf.for {iv.name} = {bounds[0].name} to {bounds[1].name} step {bounds[2].name} {{",
            )
            return None
        result_value = self._new_value()
        iter_arg_map = {name: iter_arg for name, _, iter_arg in candidates}
        carried_with_initials = [
            (name, self._materialize_value(before, lines, indent, after.type), after)
            for name, before, after in carried
        ]
        carried[:] = carried_with_initials
        iter_args = ", ".join(
            f"{iter_arg_map[name].name} = {before.name}" for name, before, _ in carried
        )
        result_types = ", ".join(after.render_type() for _, _, after in carried)
        self._emit(
            lines,
            indent,
            f"{result_value.name}:{len(carried)} = scf.for {iv.name} = {bounds[0].name} "
            f"to {bounds[1].name} step {bounds[2].name} iter_args({iter_args}) -> ({result_types}) {{",
        )
        return result_value

    def _loop_updated_env(self, outer_env, carried, result_value):
        updated_env = dict(outer_env)
        if result_value is not None:
            for idx, (name, _, after) in enumerate(carried):
                updated_env[name] = _project_result(result_value, idx, after.type)
        return updated_env

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

    def _build_struct_argument(self, arg, arg_ty, arg_types):
        if arg.arg not in self.specialization:
            raise _VKernelCompileError(
                f"template argument '{arg.arg}: {arg_ty.name}' requires .jit(...) specialization"
            )
        binding = self.specialization[arg.arg]
        if not isinstance(binding, _VKernelStructBinding) or binding.schema != arg_ty:
            raise _VKernelCompileError(
                f"specialization for '{arg.arg}' must be a {arg_ty.name}(...) binding"
            )
        fields = {}
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
                fields[field_name] = arg_val
                continue
            if field_kind is const:
                if not isinstance(field_value, _VKernelConstBinding):
                    raise _VKernelCompileError(f"{arg_ty.name}.{field_name} must use pto.const(...)")
                static_value = field_value.value
                if not isinstance(static_value, (list, tuple)) or not all(
                    isinstance(v, int) for v in static_value
                ):
                    raise _VKernelCompileError(
                        f"{arg_ty.name}.{field_name} must be a list/tuple of ints"
                    )
                fields[field_name] = _VKStaticSequence(
                    tuple(_VKValue(literal=v) for v in static_value)
                )
                continue
            raise _VKernelCompileError(
                f"unsupported struct field kind for {arg_ty.name}.{field_name}"
            )
        return _VKStructValue(arg_ty, fields)

    def _build_kernel_arguments(self):
        arg_types = []
        env = {}
        for arg in self.fn_def.args.args:
            arg_ty = _coerce_surface_type(self.py_fn.__annotations__.get(arg.arg))
            if arg_ty is None:
                raise _VKernelCompileError(f"missing type annotation for argument '{arg.arg}'")
            if not isinstance(arg_ty, _VKernelType):
                raise _VKernelCompileError(f"unsupported type annotation for argument '{arg.arg}'")
            if isinstance(arg_ty, _VKernelStructDef):
                env[arg.arg] = self._build_struct_argument(arg, arg_ty, arg_types)
                continue
            arg_val = self._new_arg_value(arg_ty)
            arg_types.append(f"{arg_val.name}: {arg_ty.render()}")
            env[arg.arg] = arg_val
        return arg_types, env

    def build_text(self):
        lines = [f'module attributes {{pto.target_arch = "{self.target}"}} {{']
        arg_types, env = self._build_kernel_arguments()
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
