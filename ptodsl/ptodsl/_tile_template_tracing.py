# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""
Tile-template tracing implementation for PTODSL tile templates.

This module keeps the authored Python body close to TileLang-style templates,
but traces execution directly into MLIR Python bindings instead of going through
an AST-capture frontend.

Current scope:
- bare ``Tile`` parameters with static 2D specializations
- ``dst.element_type`` / ``dst.valid_shape``
- optional `with pto.vecscope():`
- explicit structured `with pto.for_(...) as ...:`
- optional named loop-carried state via ``state={...}``
- ``get_lanes(dtype)``
- ``make_mask(dtype, remained)``
- ``vlds(tile[row, col:])``
- ``vadd(lhs, rhs, mask)``
- ``vsts(vec, tile[row, col:], mask)``

The current goal is to keep a narrow tile-template tracing path that already
builds real MLIR Python objects, while keeping its scope explicit and aligned
with the main PTODSL tracing runtime.
"""

from __future__ import annotations

import inspect
from dataclasses import dataclass
from pathlib import Path
from . import scalar as _scalar
from ._surface_types import Tile
from ._tracing.control_flow import apply_unroll_hint, normalize_unroll_hint
from ._tracing import (
    KernelModuleSpec,
    ModuleArtifact,
    ModuleStyle,
    TracingRuntime,
    require_active_runtime,
)
from ._types import (
    _resolve,
    float16 as _float16,
    float32 as _float32,
    index as _index,
    int8 as _int8,
    int16 as _int16,
    int32 as _int32,
    int64 as _int64,
    mask_type as _mask_type,
    ptr as _ptr,
    tile_buf_type as _tile_buf_type,
    vreg_type as _vreg_type,
)

from ptoas.mlir.dialects import arith, pto as _pto, scf
from ptoas.mlir.ir import InsertionPoint, IntegerType, Type


@dataclass(frozen=True)
class ScalarType:
    name: str
    lanes: int
    mask_bits: int
    bytewidth: int

    def __repr__(self) -> str:
        return self.name


f32 = ScalarType("f32", lanes=64, mask_bits=32, bytewidth=4)
f16 = ScalarType("f16", lanes=128, mask_bits=16, bytewidth=2)
bf16 = ScalarType("bf16", lanes=128, mask_bits=16, bytewidth=2)
i32 = ScalarType("i32", lanes=64, mask_bits=32, bytewidth=4)
i16 = ScalarType("i16", lanes=128, mask_bits=16, bytewidth=2)
i8 = ScalarType("i8", lanes=256, mask_bits=8, bytewidth=1)


@dataclass(frozen=True)
class TileSpec:
    shape: tuple[int, int]
    dtype: ScalarType
    memory_space: str = "ub"

    def __post_init__(self):
        if len(self.shape) != 2:
            raise ValueError("TileSpec currently only supports rank-2 tile shapes")
        if any(not isinstance(dim, int) or dim <= 0 for dim in self.shape):
            raise ValueError("TileSpec.shape must contain positive integers")
        if self.memory_space != "ub":
            raise ValueError("TileSpec currently only supports ub tiles")

    def mlir_type(self):
        rows, cols = self.shape
        return _tile_buf_type(
            [rows, cols],
            _scalar_descriptor(self.dtype),
            [rows, cols],
            blayout="RowMajor",
            address_space=self.memory_space,
            slayout="NoneBox",
            fractal_size=512,
            pad="Null",
        )


@dataclass(frozen=True)
class _Value:
    value: object
    const_value: int | None = None

    def __repr__(self) -> str:
        return str(self.value)

    @property
    def type_text(self) -> str:
        return str(self.value.type)

    @property
    def is_const(self) -> bool:
        return self.const_value is not None


@dataclass(frozen=True)
class _MaskValue:
    value: object
    dtype: ScalarType

    @property
    def type_text(self) -> str:
        return str(self.value.type)


@dataclass(frozen=True)
class _VectorValue:
    value: object
    dtype: ScalarType

    @property
    def type_text(self) -> str:
        return str(self.value.type)


@dataclass(frozen=True)
class _TileSlice:
    tile: "_TileProxy"
    row: int | _Value
    col: int | _Value


class _TileProxy:
    def __init__(self, trace: "_TraceBuilder", arg_value, spec: TileSpec):
        self._trace = trace
        self._arg_value = arg_value
        self._spec = spec

    @property
    def element_type(self) -> ScalarType:
        return self._spec.dtype

    @property
    def valid_shape(self) -> tuple[_Value, _Value]:
        return (
            self._trace.index_const(self._spec.shape[0]),
            self._trace.index_const(self._spec.shape[1]),
        )

    @property
    def type_text(self) -> str:
        return str(self._arg_value.type)

    def __getitem__(self, key):
        if (
            not isinstance(key, tuple)
            or len(key) != 2
            or not _is_index_like(key[0])
            or not isinstance(key[1], slice)
        ):
            raise TypeError("tile-template tracing only supports tile[row, col:] indexing")
        row, col_slice = key
        if col_slice.stop is not None or col_slice.step is not None:
            raise TypeError("tile-template tracing only supports tile[row, col:] slices")
        col = 0 if col_slice.start is None else col_slice.start
        if not _is_index_like(col):
            raise TypeError("tile-template tracing only supports integer/index column offsets")
        _validate_static_bound(row, self._spec.shape[0], "row")
        _validate_static_bound(col, self._spec.shape[1], "column")
        return _TileSlice(self, row=row, col=col)


class _LoopStateView:
    def __init__(self, names: tuple[str, ...], values: tuple[_Value, ...]):
        self._values = dict(zip(names, values))

    def __getattr__(self, name: str) -> _Value:
        try:
            return self._values[name]
        except KeyError as exc:
            raise AttributeError(name) from exc


class _LoopHandle:
    def __init__(
        self,
        trace: "_TraceBuilder",
        for_op,
        iv: _Value,
        iter_args: tuple[_Value, ...],
        state_names: tuple[str, ...] = (),
    ):
        self._trace = trace
        self._for_op = for_op
        self.iv = iv
        self.iter_args = iter_args
        self._state_names = state_names
        self.state = _LoopStateView(state_names, iter_args) if state_names else None
        self.results: tuple[_Value, ...] = ()

    def _finalize(self) -> None:
        self.results = tuple(_Value(result) for result in self._for_op.results)

    def yield_state(self, **kwargs) -> None:
        if not self._state_names:
            raise RuntimeError("loop.yield_state(...) requires for_(..., state={...})")
        missing = [name for name in self._state_names if name not in kwargs]
        extra = [name for name in kwargs if name not in self._state_names]
        if missing or extra:
            pieces = []
            if missing:
                pieces.append(f"missing: {', '.join(missing)}")
            if extra:
                pieces.append(f"unexpected: {', '.join(extra)}")
            raise RuntimeError(
                "loop.yield_state(...) must match loop state names exactly; "
                + "; ".join(pieces)
            )
        ordered = tuple(kwargs[name] for name in self._state_names)
        self._trace._yield_loop_values(ordered, surface="loop.yield_state", from_named_state=True)


class _VecScopeCM:
    def __init__(self, trace: "_TraceBuilder"):
        self._trace = trace

    def __enter__(self):
        self._trace._enter_vecscope()
        return None

    def __exit__(self, exc_type, exc, tb):
        self._trace._exit_vecscope(exc_type, exc, tb)


class _ForCM:
    def __init__(self, trace: "_TraceBuilder", start, stop, step, iter_args, state,
                 unroll=None, unroll_factor=None):
        self._trace = trace
        self._start = start
        self._stop = stop
        self._step = step
        self._unroll = unroll
        self._unroll_factor = unroll_factor
        self._iter_args = list(iter_args) if iter_args is not None else []
        self._state = tuple(state.items()) if state is not None else ()
        self._handle: _LoopHandle | None = None

    def __enter__(self):
        self._handle = self._trace._enter_for(
            self._start,
            self._stop,
            self._step,
            self._iter_args,
            self._state,
            unroll=self._unroll,
            unroll_factor=self._unroll_factor,
        )
        if self._iter_args or self._state:
            return self._handle
        return self._handle.iv

    def __exit__(self, exc_type, exc, tb):
        self._trace._exit_for(self._handle, exc_type, exc, tb)


class _TraceBuilder(TracingRuntime):
    def __init__(self, descriptor: "TileTemplate", tile_specs: dict[str, TileSpec]):
        super().__init__(
            KernelModuleSpec(
                function_name=descriptor.name,
                target_arch=descriptor.target,
                kernel_kind="vector",
                mode="auto",
                module_style=ModuleStyle.NESTED,
                source_file=inspect.getsourcefile(descriptor.py_fn) or inspect.getfile(descriptor.py_fn),
                source_line=getattr(descriptor.py_fn.__code__, "co_firstlineno", None),
            )
        )
        self.descriptor = descriptor
        self.tile_specs = tile_specs
        self._const_cache: dict[tuple[int, str], _Value] = {}
        self._tile_ptr_cache: dict[int, _Value] = {}
        self._row_offset_cache: dict[tuple[str, str], _Value] = {}
        self._loop_stack: list[dict] = []
        self._inside_vecscope = False
        self._ordered_specs: list[tuple[str, TileSpec]] = []
        signature = inspect.signature(self.descriptor.py_fn)
        self._signature_parameters = tuple(signature.parameters.items())

    def compute_argument_types(self):
        arg_types = []
        ordered_specs = []
        for param_name, param in self._signature_parameters:
            if not _is_tile_annotation(param.annotation):
                raise TypeError(
                    "tile-template tracing currently only supports Tile parameters; "
                    f"parameter {param_name!r} uses {param.annotation!r}"
                )
            spec = self.tile_specs.get(param_name)
            if spec is None:
                raise ValueError(f"missing specialization for Tile parameter {param_name!r}")
            ordered_specs.append((param_name, spec))
            arg_types.append(spec.mlir_type())
        self._ordered_specs = ordered_specs
        return arg_types

    def bind_entry_arguments(self, entry_arguments):
        args = []
        for arg_value, (_, spec) in zip(entry_arguments, self._ordered_specs):
            args.append(_TileProxy(self, arg_value, spec))
        return tuple(args)

    def trace_entry(self, *args):
        self.descriptor.py_fn(*args)

    def validate_trace_state(self):
        if self._inside_vecscope:
            raise RuntimeError("tile-template trace exited with an open vecscope block")
        if self._loop_stack:
            raise RuntimeError("tile-template trace exited with an open scf.for block")

    def vecscope(self) -> _VecScopeCM:
        return _VecScopeCM(self)

    def for_(self, start, stop, *, step, iter_args=None, state=None,
             unroll=None, unroll_factor=None) -> _ForCM:
        if iter_args is not None and state is not None:
            raise ValueError("for_() accepts either iter_args= or state=, not both")
        if state is not None:
            if not hasattr(state, "items"):
                raise TypeError("for_(..., state=...) expects a mapping of name -> initial value")
            for name in state:
                if not isinstance(name, str) or not name:
                    raise TypeError("for_ state names must be non-empty strings")
        normalize_unroll_hint(unroll, unroll_factor, context="tile-template for_(...)")
        return _ForCM(self, start, stop, step, iter_args, state,
                      unroll=unroll, unroll_factor=unroll_factor)

    def yield_(self, *vals):
        self._yield_loop_values(vals, surface="yield_", from_named_state=False)

    def _yield_loop_values(self, vals, *, surface: str, from_named_state: bool):
        if not self._loop_stack:
            raise RuntimeError(f"{surface}(...) may only be used inside a tile-template for_ block")
        frame = self._loop_stack[-1]
        if frame["kind"] != "for":
            raise RuntimeError(f"{surface}(...) may only be used inside a tile-template for_ block")
        if frame["state_names"] and not from_named_state:
            raise RuntimeError(
                f"{surface}(...) is ambiguous for tile-template for_ with named state; "
                "use loop.yield_state(...) instead"
            )
        if frame["yielded"]:
            raise RuntimeError(
                f"{surface}(...) may only be emitted once per tile-template for_ block"
            )
        if len(vals) != len(frame["iter_args"]):
            raise RuntimeError(
                f"{surface}(...) expected {len(frame['iter_args'])} value(s), got {len(vals)}"
            )
        coerced = tuple(
            self._coerce_like(arg, expected.type_text)
            for arg, expected in zip(vals, frame["iter_args"])
        )
        scf.YieldOp([val.value for val in coerced])
        frame["yielded"] = True
        frame["yield_vals"] = coerced

    def index_const(self, value: int) -> _Value:
        return self._const(value, _resolve(_index))

    def scalar_const(self, value: int, dtype: ScalarType) -> _Value:
        return self._const(value, _resolve(_scalar_descriptor(dtype)))

    def _const(self, value: int, mlir_type) -> _Value:
        cache_key = (value, str(mlir_type))
        cached = self._const_cache.get(cache_key)
        if cached is not None:
            return cached
        const = _Value(arith.ConstantOp(mlir_type, value).result, const_value=value)
        self._const_cache[cache_key] = const
        return const

    def ensure_tile_ptr(self, tile: _TileProxy) -> _Value:
        cache_key = id(tile._arg_value)
        cached = self._tile_ptr_cache.get(cache_key)
        if cached is not None:
            return cached
        ptr_type = _resolve(_ptr(_scalar_descriptor(tile.element_type), tile._spec.memory_space))
        ptr_value = _Value(_pto.TileBufAddrOp(ptr_type, tile._arg_value).result)
        self._tile_ptr_cache[cache_key] = ptr_value
        return ptr_value

    def materialize_linear_offset(self, tile_slice: _TileSlice) -> _Value:
        cols = tile_slice.tile._spec.shape[1]
        row = self._coerce_index(tile_slice.row)
        col = self._coerce_index(tile_slice.col)
        if row.is_const and col.is_const:
            return self.index_const(row.const_value * cols + col.const_value)
        row_stride = self.index_const(cols)
        row_off = self._materialize_row_offset(row, row_stride)
        return _Value(_scalar.addi(row_off.value, col.value))

    def _enter_vecscope(self):
        if self._inside_vecscope:
            raise RuntimeError(
                "nested tile-template vecscope blocks are not supported in the current implementation"
            )
        vecscope_op = _pto.VecScopeOp()
        vecscope_block = vecscope_op.body.blocks.append()
        vecscope_ip = InsertionPoint(vecscope_block)
        vecscope_ip.__enter__()
        self._loop_stack.append(
            {
                "kind": "vecscope",
                "ip": vecscope_ip,
            }
        )
        self._inside_vecscope = True

    def _exit_vecscope(self, exc_type, exc, tb):
        if not self._inside_vecscope:
            raise RuntimeError("vecscope exit without matching enter")
        frame = self._loop_stack.pop()
        if frame["kind"] != "vecscope":
            raise RuntimeError("tile-template vecscope stack corruption detected")
        frame["ip"].__exit__(exc_type, exc, tb)
        self._inside_vecscope = False

    def _enter_for(self, start, stop, step, iter_args, state_items,
                   unroll=None, unroll_factor=None) -> _LoopHandle:
        start_val = self._coerce_index(start)
        stop_val = self._coerce_index(stop)
        step_val = self._coerce_index(step)
        state_names = tuple(name for name, _ in state_items)
        if state_names:
            iter_arg_vals = tuple(self._coerce_value(arg) for _, arg in state_items)
        else:
            iter_arg_vals = tuple(self._coerce_value(arg) for arg in iter_args)
        for_op = scf.ForOp(
            start_val.value,
            stop_val.value,
            step_val.value,
            [arg.value for arg in iter_arg_vals] if iter_arg_vals else None,
        )
        apply_unroll_hint(for_op, unroll, unroll_factor)
        loop_ip = InsertionPoint(for_op.body)
        loop_ip.__enter__()
        iv = _Value(for_op.induction_variable)
        inner_iter_args = tuple(_Value(arg) for arg in for_op.inner_iter_args)
        handle = _LoopHandle(self, for_op, iv, inner_iter_args, state_names=state_names)
        self._loop_stack.append(
            {
                "kind": "for",
                "handle": handle,
                "ip": loop_ip,
                "iter_args": inner_iter_args,
                "state_names": state_names,
                "yielded": False,
                "yield_vals": (),
            }
        )
        return handle

    def _exit_for(self, handle: _LoopHandle | None, exc_type, exc, tb):
        if handle is None:
            raise RuntimeError("for_ exit without a loop handle")
        frame = self._loop_stack.pop()
        if frame["kind"] != "for" or frame["handle"] is not handle:
            raise RuntimeError("tile-template for_ stack corruption detected")
        if exc_type is None:
            if frame["iter_args"] and not frame["yielded"]:
                if frame["state_names"]:
                    raise RuntimeError(
                        "tile-template for_ with named state requires explicit loop.yield_state(...)"
                    )
                raise RuntimeError("tile-template for_ with iter_args requires explicit yield_(...)")
            if not frame["iter_args"]:
                scf.YieldOp([])
        frame["ip"].__exit__(exc_type, exc, tb)
        if exc_type is not None:
            return
        handle._finalize()

    def _materialize_row_offset(self, row: _Value, row_stride: _Value) -> _Value:
        if row.is_const and row_stride.is_const:
            return self.index_const(row.const_value * row_stride.const_value)
        cache_key = (str(row.value), str(row_stride.value))
        cached = self._row_offset_cache.get(cache_key)
        if cached is not None:
            return cached
        result = _Value(_scalar.muli(row.value, row_stride.value))
        self._row_offset_cache[cache_key] = result
        return result

    def _coerce_index(self, value) -> _Value:
        coerced = self._coerce_value(value)
        if coerced.type_text != str(_resolve(_index)):
            raise TypeError(f"expected index value, got {coerced.type_text}")
        return coerced

    def _coerce_value(self, value) -> _Value:
        if isinstance(value, _Value):
            return value
        if isinstance(value, int):
            return self.index_const(value)
        if hasattr(value, "type"):
            return _Value(value)
        raise TypeError(f"unsupported tile-template scalar value {value!r}")

    def _coerce_like(self, value, ty: str) -> _Value:
        coerced = self._coerce_value(value)
        if coerced.type_text != ty:
            raise TypeError(f"expected value of type {ty}, got {coerced.type_text}")
        return coerced


@dataclass(frozen=True)
class TileTemplate:
    py_fn: object
    target: str
    op: str
    name: str
    source_label: str

    def specialize(self, **tile_specs: TileSpec) -> "SpecializedTileTemplate":
        return SpecializedTileTemplate(self, tile_specs)


class SpecializedTileTemplate(ModuleArtifact):
    def __init__(self, descriptor: TileTemplate, tile_specs: dict[str, TileSpec]):
        super().__init__(
            descriptor.name,
            module_factory=lambda: _TraceBuilder(descriptor, tile_specs).build_module(),
        )
        self.descriptor = descriptor
        self.tile_specs = tile_specs


def tile_template(*, target: str = "a5", op: str, name: str | None = None):
    if target != "a5":
        raise ValueError("tile-template tracing currently only supports target='a5'")

    def decorator(fn):
        source_path = Path(inspect.getsourcefile(fn) or "<unknown>")
        descriptor_name = name or fn.__name__
        return TileTemplate(
            py_fn=fn,
            target=target,
            op=op,
            name=descriptor_name,
            source_label=f"{source_path}:{fn.__name__}",
        )

    return decorator


def vecscope() -> _VecScopeCM:
    return require_active_runtime("vecscope", expected_type=_TraceBuilder).vecscope()


def for_(start, stop, *, step, iter_args=None, state=None,
         unroll=None, unroll_factor=None) -> _ForCM:
    return require_active_runtime("for_", expected_type=_TraceBuilder).for_(
        start, stop, step=step, iter_args=iter_args, state=state,
        unroll=unroll, unroll_factor=unroll_factor,
    )


def yield_(*vals):
    require_active_runtime("yield_", expected_type=_TraceBuilder).yield_(*vals)


def get_lanes(dtype: ScalarType) -> _Value:
    return require_active_runtime("get_lanes", expected_type=_TraceBuilder).index_const(dtype.lanes)


def scalar_const(value: int, dtype: ScalarType) -> _Value:
    return require_active_runtime("scalar_const", expected_type=_TraceBuilder).scalar_const(value, dtype)


def make_mask(dtype: ScalarType, remained) -> tuple[_MaskValue, _Value]:
    trace = require_active_runtime("make_mask", expected_type=_TraceBuilder)
    remained_val = trace._coerce_value(remained)
    expected_scalar_ty = str(_resolve(_scalar_descriptor(_scalar_type_for_mask(dtype))))
    if remained_val.type_text != expected_scalar_ty:
        raise TypeError(
            f"tile-template tracing expects make_mask remained to use {expected_scalar_ty}, got {remained_val.type_text}"
        )
    if dtype.mask_bits not in {8, 16, 32}:
        raise ValueError(f"unsupported mask bit-width {dtype.mask_bits}")
    mask_ty = _resolve(_mask_type(f"b{dtype.mask_bits}"))
    scalar_ty = IntegerType.get_signless(dtype.mask_bits)
    op_cls = getattr(_pto, f"PltB{dtype.mask_bits}Op", None)
    if op_cls is None:
        raise NotImplementedError(
            f"pto.PltB{dtype.mask_bits}Op is not available in the current Python bindings"
        )
    plt_op = op_cls(mask_ty, scalar_ty, remained_val.value)
    lanes = trace.scalar_const(dtype.lanes, _scalar_type_for_mask(dtype))
    next_value = _Value(_scalar.subi(remained_val.value, lanes.value))
    return _MaskValue(plt_op.mask, dtype), next_value


def vlds(tile_slice: _TileSlice) -> _VectorValue:
    trace = require_active_runtime("vlds", expected_type=_TraceBuilder)
    if not isinstance(tile_slice, _TileSlice):
        raise TypeError("tile-template tracing only supports vlds(tile[row, col:])")
    ptr_value = trace.ensure_tile_ptr(tile_slice.tile)
    offset = trace.materialize_linear_offset(tile_slice)
    vector_ty = _resolve(_vreg_type(tile_slice.tile.element_type.lanes, _scalar_descriptor(tile_slice.tile.element_type)))
    result = _pto.VldsOp(vector_ty, None, ptr_value.value, offset.value).result
    return _VectorValue(result, tile_slice.tile.element_type)


def vadd(lhs: _VectorValue, rhs: _VectorValue, mask: _MaskValue) -> _VectorValue:
    if lhs.dtype != rhs.dtype:
        raise TypeError("tile-template tracing expects vadd operands to use the same dtype")
    if lhs.dtype != mask.dtype:
        raise TypeError("tile-template tracing expects vadd mask dtype to match vector dtype")
    result = _pto.VaddOp(lhs.value.type, lhs.value, rhs.value, mask.value).result
    return _VectorValue(result, lhs.dtype)


def vsts(vec: _VectorValue, tile_slice: _TileSlice, mask: _MaskValue) -> None:
    trace = require_active_runtime("vsts", expected_type=_TraceBuilder)
    if vec.dtype != mask.dtype:
        raise TypeError("tile-template tracing expects vsts mask dtype to match vector dtype")
    if vec.dtype != tile_slice.tile.element_type:
        raise TypeError("tile-template tracing expects vsts destination dtype to match vector dtype")
    ptr_value = trace.ensure_tile_ptr(tile_slice.tile)
    offset = trace.materialize_linear_offset(tile_slice)
    _pto.VstsOp(None, vec.value, ptr_value.value, offset.value, mask.value)


def _is_tile_annotation(annotation) -> bool:
    if annotation is Tile:
        return True
    if isinstance(annotation, str):
        return annotation == "Tile" or annotation.endswith(".Tile")
    return getattr(annotation, "__name__", None) == "Tile"


def _is_index_like(value) -> bool:
    return isinstance(value, int) or (isinstance(value, _Value) and value.type_text == str(_resolve(_index)))


def _validate_static_bound(value, upper_bound: int, label: str):
    if isinstance(value, int):
        if value < 0 or value >= upper_bound:
            raise IndexError(f"{label} {value} is outside tile bound {upper_bound}")
        return
    if isinstance(value, _Value) and value.is_const:
        concrete = value.const_value
        if concrete < 0 or concrete >= upper_bound:
            raise IndexError(f"{label} {concrete} is outside tile bound {upper_bound}")


def _scalar_descriptor(dtype: ScalarType):
    descriptors = {
        "f32": _float32,
        "f16": _float16,
        "bf16": Type.parse("bf16"),
        "i8": _int8,
        "i16": _int16,
        "i32": _int32,
        "i64": _int64,
    }
    descriptor = descriptors.get(dtype.name)
    if descriptor is None:
        raise ValueError(f"unsupported scalar dtype {dtype.name}")
    return descriptor


def _scalar_type_for_mask(dtype: ScalarType) -> ScalarType:
    if dtype.mask_bits == 8:
        return i8
    if dtype.mask_bits == 16:
        return i16
    if dtype.mask_bits == 32:
        return i32
    raise ValueError(f"unsupported mask bit-width {dtype.mask_bits}")


__all__ = [
    "Tile",
    "TileSpec",
    "TileTemplate",
    "SpecializedTileTemplate",
    "ScalarType",
    "f32",
    "f16",
    "bf16",
    "i32",
    "i16",
    "i8",
    "tile_template",
    "vecscope",
    "for_",
    "yield_",
    "get_lanes",
    "scalar_const",
    "make_mask",
    "vlds",
    "vadd",
    "vsts",
]
