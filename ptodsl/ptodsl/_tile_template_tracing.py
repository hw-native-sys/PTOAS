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
- explicit ``ir_level="vpto"`` or ``ir_level="vmi"`` template selection
- optional `with pto.vecscope():`
- explicit structured `with pto.for_(...) as ...:`
- optional named loop-carried state via ``state={...}``
- ``get_lanes(dtype)``
- ``make_mask(dtype, remained)``
- ``vlds(tile[row, col:])``
- ``vadd(lhs, rhs, mask)``
- ``vsts(vec, tile[row, col:], mask)``
- fixed-shape VMI logical-block helpers for ``vload/vadd/vexp/vstore``

The current goal is to keep a narrow tile-template tracing path that already
builds real MLIR Python objects, while keeping its scope explicit and aligned
with the main PTODSL tracing runtime.
"""

from __future__ import annotations

import inspect
from dataclasses import dataclass
from pathlib import Path
from ._surface_types import Tile
from ._surface_values import unwrap_surface_value
from ._tracing.control_flow import apply_unroll_hint, normalize_unroll_hint
from ._tracing import (
    KernelModuleSpec,
    ModuleArtifact,
    ModuleStyle,
    TracingRuntime,
    require_active_runtime,
)
from ._vmi_namespace import vmi as _vmi
from .tilelib.metadata import TemplateMetadata as _RegistryTemplateMetadata
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
from ptoas.mlir.ir import InsertionPoint, IntegerType, Type, FloatAttr


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
ui8 = ScalarType("ui8", lanes=256, mask_bits=8, bytewidth=1)

_SCALAR_TYPES_BY_NAME = {
    dtype.name: dtype
    for dtype in (f32, f16, bf16, i32, i16, i8, ui8)
}


@dataclass(frozen=True)
class TileSpec:
    shape: tuple[int, int]
    dtype: ScalarType
    memory_space: str = "ub"
    b_layout: str = "row_major"
    # valid_shape: may be smaller than shape (e.g. RowPlusOne: shape=(129,64),
    # valid_shape=(128,64)). Defaults to shape when None (plain tiles).
    valid_shape: tuple[int, int] | None = None
    # compact_mode: "normal" (default) or "row_plus_one" (UB +1 padding band).
    compact_mode: str = "normal"
    # s_fractal_size: physical fractal granularity; 0/None defaults to 512.
    s_fractal_size: int | None = 512
    # pad_value: pad-value token for the tile_buf config ("Null"/"Zero"/...).
    pad_value: str = "Null"

    def __post_init__(self):
        if len(self.shape) != 2:
            raise ValueError("TileSpec currently only supports rank-2 tile shapes")
        if any(not isinstance(dim, int) or dim <= 0 for dim in self.shape):
            raise ValueError("TileSpec.shape must contain positive integers")
        if self.memory_space != "ub":
            raise ValueError("TileSpec currently only supports ub tiles")
        if self.b_layout not in {"row_major", "col_major"}:
            raise ValueError("TileSpec.b_layout must be 'row_major' or 'col_major'")
        if self.compact_mode not in {"normal", "row_plus_one"}:
            raise ValueError(
                "TileSpec.compact_mode must be 'normal' or 'row_plus_one'"
            )
        # pto-isa invariant: ValidRow <= alignRow (physical rows). valid_shape
        # defaults to shape when None (handled in mlir_type / consumers).
        if self.valid_shape is not None:
            if len(self.valid_shape) != 2:
                raise ValueError("TileSpec.valid_shape must be rank-2")
            if any(not isinstance(d, int) or d < 0 for d in self.valid_shape):
                raise ValueError("TileSpec.valid_shape must contain non-negative ints")
            if self.valid_shape[0] > self.shape[0] or self.valid_shape[1] > self.shape[1]:
                raise ValueError(
                    "TileSpec.valid_shape must not exceed physical shape "
                    f"{self.shape}, got {self.valid_shape}"
                )

    @property
    def effective_valid_shape(self) -> tuple[int, int]:
        """valid_shape, defaulting to shape when None (plain tiles)."""
        return self.valid_shape if self.valid_shape is not None else self.shape

    def mlir_type(self):
        rows, cols = self.shape
        vrow, vcol = self.effective_valid_shape
        fractal = self.s_fractal_size if self.s_fractal_size else 512
        return _tile_buf_type(
            [rows, cols],
            _scalar_descriptor(self.dtype),
            [vrow, vcol],
            blayout="RowMajor" if self.b_layout == "row_major" else "ColMajor",
            address_space=self.memory_space,
            slayout="NoneBox",
            fractal_size=fractal,
            pad=self.pad_value,
            compact_mode=self.compact_mode,
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


@dataclass(frozen=True)
class CanonicalBlockMap:
    """Static mapping contract for one logical block per tile row.

    The canonical VMI Fusion contract maps one logical row to one logical VMI
    block. The logical lane count is therefore the tile's inner width, not the
    dtype's native physical VL. Physical chunking is left to later VMI layout
    and VMI-to-VPTO lowering passes.
    """

    shape: tuple[int, int]
    logical_lanes: int

    def __post_init__(self):
        if len(self.shape) != 2:
            raise ValueError("CanonicalBlockMap requires a rank-2 shape")
        rows, cols = self.shape
        if any(not isinstance(dim, int) or dim <= 0 for dim in self.shape):
            raise ValueError("CanonicalBlockMap shape must contain positive integers")
        if not isinstance(self.logical_lanes, int) or self.logical_lanes <= 0:
            raise ValueError("CanonicalBlockMap logical_lanes must be a positive integer")
        if cols != self.logical_lanes:
            raise ValueError(
                "CanonicalBlockMap requires exactly one logical VL block per row; "
                f"got cols={cols}, logical_lanes={self.logical_lanes}"
            )

    @classmethod
    def from_tile(cls, tile: "_TileProxy", *, logical_lanes: int | None = None):
        if not isinstance(tile, _TileProxy):
            raise TypeError("CanonicalBlockMap.from_tile(...) expects a traced Tile argument")
        lanes = tile._spec.shape[1] if logical_lanes is None else logical_lanes
        return cls(tile._spec.shape, lanes)

    @property
    def rows(self) -> int:
        return self.shape[0]

    @property
    def cols(self) -> int:
        return self.shape[1]

    @property
    def blocks_per_row(self) -> int:
        return self.cols // self.logical_lanes

    @property
    def logical_block_count(self) -> int:
        return self.rows * self.blocks_per_row

    def coordinate(self, logical_block) -> "CanonicalBlockCoordinate":
        if isinstance(logical_block, int):
            if logical_block < 0 or logical_block >= self.logical_block_count:
                raise IndexError(
                    f"logical block {logical_block} is outside [0, {self.logical_block_count})"
                )
            return CanonicalBlockCoordinate(self, logical_block)

        trace = require_active_runtime("CanonicalBlockMap.coordinate", expected_type=_TraceBuilder)
        block = trace._coerce_index(logical_block)
        if block.is_const and (
            block.const_value < 0 or block.const_value >= self.logical_block_count
        ):
            raise IndexError(
                f"logical block {block.const_value} is outside [0, {self.logical_block_count})"
            )
        return CanonicalBlockCoordinate(self, block)


class CanonicalBlockCoordinate:
    """One lazily materialized logical-block coordinate."""

    def __init__(self, block_map: CanonicalBlockMap, logical_block: int | _Value):
        self.block_map = block_map
        self.logical_block = logical_block
        self._cache: dict[str, int | _Value] = {}

    def _cached(self, name: str, build):
        if name not in self._cache:
            self._cache[name] = build()
        return self._cache[name]

    def _binary(self, op_name: str, lhs, rhs):
        if isinstance(lhs, int) and isinstance(rhs, int):
            if op_name == "mul":
                return lhs * rhs
            if op_name == "floordiv":
                return lhs // rhs
            if op_name == "mod":
                return lhs % rhs
            raise ValueError(f"unsupported coordinate operation {op_name!r}")
        trace = require_active_runtime(
            f"CanonicalBlockCoordinate.{op_name}", expected_type=_TraceBuilder
        )
        return trace.index_binary(op_name, lhs, rhs)

    @property
    def row(self):
        return self._cached(
            "row",
            lambda: self._binary("floordiv", self.logical_block, self.block_map.blocks_per_row),
        )

    @property
    def block_in_row(self):
        return self._cached(
            "block_in_row",
            lambda: self._binary("mod", self.logical_block, self.block_map.blocks_per_row),
        )

    @property
    def col_start(self):
        return self._cached(
            "col_start",
            lambda: self._binary("mul", self.block_in_row, self.block_map.logical_lanes),
        )

    @property
    def linear_offset(self):
        return self._cached(
            "linear_offset",
            lambda: self._binary("mul", self.logical_block, self.block_map.logical_lanes),
        )

    @property
    def active_lanes(self) -> int:
        return self.block_map.logical_lanes


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
        # valid_shape (defaults to shape for plain tiles); RowPlusOne carries
        # a smaller valid_shape (e.g. 128 vs physical 129) so the row loop iterates
        # valid_rows times, not the padded physical rows.
        vrow, vcol = self._spec.effective_valid_shape
        return (
            self._trace.index_const(vrow),
            self._trace.index_const(vcol),
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
        iter_arg_templates: tuple = (),
    ):
        self._trace = trace
        self._for_op = for_op
        self.iv = iv
        self.iter_args = iter_args
        self._state_names = state_names
        self._iter_arg_templates = iter_arg_templates
        self.state = _LoopStateView(state_names, iter_args) if state_names else None
        self.results: tuple[_Value, ...] = ()

    def _finalize(self) -> None:
        if self._iter_arg_templates:
            self.results = tuple(
                self._trace._rewrap_state_value(tpl, result)
                for tpl, result in zip(
                    self._iter_arg_templates, self._for_op.results
                )
            )
        else:
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
    def __init__(
        self,
        descriptor: "TileTemplate",
        parameter_specs: dict[str, TileSpec | ScalarType],
        context_attrs: dict[str, object] | None = None,
    ):
        is_vmi = descriptor.ir_level == "vmi"
        super().__init__(
            KernelModuleSpec(
                function_name=descriptor.name,
                target_arch=descriptor.target,
                kernel_kind="vector",
                backend="vpto",
                entry=not is_vmi,
                mode="auto",
                module_style=(
                    ModuleStyle.BACKEND_PARTITIONED if is_vmi else ModuleStyle.NESTED
                ),
                source_file=(
                    inspect.getsourcefile(descriptor.py_fn)
                    or inspect.getfile(descriptor.py_fn)
                ),
                source_line=getattr(descriptor.py_fn.__code__, "co_firstlineno", None),
            )
        )
        self.descriptor = descriptor
        self.context_attrs = dict(context_attrs or {})
        self.parameter_specs = parameter_specs
        self.tile_specs = {
            name: spec
            for name, spec in parameter_specs.items()
            if isinstance(spec, TileSpec)
        }
        self._const_cache: dict[tuple[int, str], _Value] = {}
        self._tile_ptr_cache: dict[int, _Value] = {}
        self._row_offset_cache: dict[tuple[str, str], _Value] = {}
        self._loop_stack: list[dict] = []
        self._inside_vecscope = False
        self._ordered_specs: list[tuple[str, TileSpec | ScalarType]] = []
        signature = inspect.signature(self.descriptor.py_fn)
        self._signature_parameters = tuple(signature.parameters.items())

    def compute_argument_types(self):
        arg_types = []
        ordered_specs = []
        for param_name, param in self._signature_parameters:
            spec = self.parameter_specs.get(param_name)
            if spec is None:
                raise ValueError(f"missing specialization for parameter {param_name!r}")
            if _is_tile_annotation(param.annotation):
                if not isinstance(spec, TileSpec):
                    raise TypeError(
                        f"parameter {param_name!r} is annotated as Tile but uses {spec!r}"
                    )
                arg_type = spec.mlir_type()
            else:
                annotation_dtype = _scalar_type_from_annotation(param.annotation)
                if annotation_dtype is None:
                    raise TypeError(
                        "tile-template tracing supports Tile or scalar dtype parameters; "
                        f"parameter {param_name!r} uses {param.annotation!r}"
                    )
                if not isinstance(spec, ScalarType) or spec != annotation_dtype:
                    raise TypeError(
                        f"parameter {param_name!r} expects scalar {annotation_dtype}, got {spec!r}"
                    )
                arg_type = _resolve(_scalar_descriptor(spec))
            ordered_specs.append((param_name, spec))
            arg_types.append(arg_type)
        self._ordered_specs = ordered_specs
        return arg_types

    def bind_entry_arguments(self, entry_arguments):
        args = []
        for arg_value, (_, spec) in zip(entry_arguments, self._ordered_specs):
            if isinstance(spec, TileSpec):
                args.append(_TileProxy(self, arg_value, spec))
            else:
                args.append(_Value(arg_value))
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

    def index_binary(self, op_name: str, lhs, rhs) -> _Value:
        lhs_val = self._coerce_index(lhs)
        rhs_val = self._coerce_index(rhs)
        if lhs_val.is_const and rhs_val.is_const:
            if op_name == "add":
                result = lhs_val.const_value + rhs_val.const_value
            elif op_name == "mul":
                result = lhs_val.const_value * rhs_val.const_value
            elif op_name == "floordiv":
                result = lhs_val.const_value // rhs_val.const_value
            elif op_name == "mod":
                result = lhs_val.const_value % rhs_val.const_value
            else:
                raise ValueError(f"unsupported index operation {op_name!r}")
            return self.index_const(result)
        op_cls = {
            "add": arith.AddIOp,
            "mul": arith.MulIOp,
            "floordiv": arith.FloorDivSIOp,
            "mod": arith.RemSIOp,
        }.get(op_name)
        if op_cls is None:
            raise ValueError(f"unsupported index operation {op_name!r}")
        return _Value(op_cls(lhs_val.value, rhs_val.value).result)

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
        return _Value(arith.AddIOp(row_off.value, col.value).result)

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
            # Preserve the authored surface templates (carry dtype for vreg/mask
            # state) so inner_iter_args can be re-wrapped for the loop body.
            iter_arg_templates = tuple(arg for _, arg in state_items)
        else:
            iter_arg_vals = tuple(self._coerce_value(arg) for arg in iter_args)
            iter_arg_templates = tuple(iter_args)
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
        inner_iter_args = tuple(
            self._rewrap_state_value(tpl, arg)
            for tpl, arg in zip(iter_arg_templates, for_op.inner_iter_args)
        )
        handle = _LoopHandle(
            self,
            for_op,
            iv,
            inner_iter_args,
            state_names=state_names,
            iter_arg_templates=iter_arg_templates,
        )
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
        result = _Value(arith.MulIOp(row.value, row_stride.value).result)
        self._row_offset_cache[cache_key] = result
        return result

    def _coerce_index(self, value) -> _Value:
        coerced = self._coerce_value(value)
        if coerced.type_text != str(_resolve(_index)):
            raise TypeError(f"expected index value, got {coerced.type_text}")
        return coerced

    def _rewrap_state_value(self, template, mlir_value):
        """Re-wrap an scf.for inner_iter_arg / result using the same authored
        surface contract as the loop-carried *template* value.

        For scalar (index) state the template is a plain _Value and is returned
        unchanged (after wrapping the MLIR value). For VMI vector/mask state the
        template carries a dtype that _vmi_binary/_vmi_vec_scalar etc. consume,
        so the inner iter_arg must be handed back as _VectorValue/_MaskValue."""
        if isinstance(template, _VectorValue):
            return _VectorValue(mlir_value, template.dtype)
        if isinstance(template, _MaskValue):
            return _MaskValue(mlir_value, template.dtype)
        return _Value(mlir_value)

    def _coerce_value(self, value) -> _Value:
        if isinstance(value, _Value):
            return value
        # VMI vector/mask values carry a dtype alongside their MLIR value; wrap
        # the MLIR value in a plain _Value so scf.for iter_args / yields (which
        # only consume .value) accept them. The dtype is re-attached by
        # _rewrap_state_value when handing the inner iter_arg back to the body.
        if isinstance(value, (_VectorValue, _MaskValue)):
            return _Value(value.value)
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


def _dtype_name(dtype) -> str:
    return getattr(dtype, "name", str(dtype))


def _coerce_parameter_spec(spec):
    if isinstance(spec, (TileSpec, ScalarType)):
        return spec

    if hasattr(spec, "shape") and hasattr(spec, "dtype"):
        shape = tuple(spec.shape)
        valid_shape = getattr(spec, "valid_shape", None)
        if valid_shape is not None:
            valid_shape = tuple(valid_shape)
        dtype = _SCALAR_TYPES_BY_NAME.get(_dtype_name(spec.dtype))
        if dtype is None:
            raise ValueError(f"unsupported VMI tile-template dtype {spec.dtype!r}")
        s_layout = getattr(spec, "s_layout", "none_box")
        compact_mode = getattr(spec, "compact_mode", "normal") or "normal"
        is_nd2nz_layout = (
            s_layout == "row_major"
            and getattr(spec, "b_layout", "row_major") == "col_major"
        )
        is_row_plus_one_layout = (
            is_nd2nz_layout
            and valid_shape is not None
            and valid_shape != shape
        )
        if s_layout != "none_box" and not (
            is_nd2nz_layout or compact_mode == "row_plus_one"
        ):
            raise ValueError(
                "VMI tile-template tracing currently supports only none_box "
                "or ND-to-NZ row_major secondary layout, "
                f"got {s_layout!r}"
            )
        return TileSpec(
            shape=shape,
            dtype=dtype,
            memory_space=getattr(spec, "memory_space", "ub"),
            b_layout=getattr(spec, "b_layout", "row_major"),
            valid_shape=valid_shape,
            compact_mode="row_plus_one" if is_row_plus_one_layout else compact_mode,
        )

    if hasattr(spec, "dtype"):
        dtype = _SCALAR_TYPES_BY_NAME.get(_dtype_name(spec.dtype))
        if dtype is None:
            raise ValueError(f"unsupported VMI scalar dtype {spec.dtype!r}")
        return dtype

    return spec


@dataclass(frozen=True)
class TileTemplate:
    py_fn: object
    target: str
    op: str
    name: str
    source_label: str
    ir_level: str
    dtypes: tuple
    context_constraints: tuple[tuple[str, tuple[object, ...]], ...]
    constraints: tuple[object, ...] = ()
    tags: tuple[str, ...] = ()
    priority: int = 100
    candidate_id: int = 1000
    single_logical_row_loop: bool = True
    resource_scope: str | None = None
    resource_vector_values: int | None = None
    resource_chunk_streaming: bool = False

    @property
    def param_names(self) -> tuple[str, ...]:
        return tuple(inspect.signature(self.py_fn).parameters)

    @property
    def metadata(self):
        if self.ir_level == "vmi":
            constraints = [self._vmi_trace_specs_supported]
            if self.context_constraints:
                constraints.append(self._context_constraints_match)
            constraints.extend(self.constraints)
            tags = ["vmi"]
            if self.single_logical_row_loop:
                tags.extend(("fusion_eligible", "single_logical_row_loop"))
            tags.extend(self.tags)
            return _RegistryTemplateMetadata.build(
                op=self.op,
                target=self.target,
                name=self.name,
                dtypes=self.dtypes,
                constraints=tuple(constraints),
                priority=self.priority,
                fusible=self.single_logical_row_loop,
                loop_depth=1 if self.single_logical_row_loop else 0,
                id=self.candidate_id,
                is_post_update=False,
                iteration_axis="row",
                op_engine="vector",
                op_class="other",
                tags=tuple(tags),
                resource_scope=self.resource_scope,
                resource_vector_values=self.resource_vector_values,
                resource_chunk_streaming=self.resource_chunk_streaming,
            )

        return _RegistryTemplateMetadata.build(
            op=self.op,
            target=self.target,
            name=self.name,
        )

    def _context_constraints_match(self, **context) -> bool:
        for key, allowed_values in self.context_constraints:
            if context.get(key) not in allowed_values:
                return False
        return True

    def _vmi_trace_specs_supported(self, **context) -> bool:
        for name in self.param_names:
            if context.get(f"{name}_kind") != "tile":
                continue
            if context.get(f"{name}_memory_space") != "ub":
                return False
            config = context.get(f"{name}_config")
            if config is None:
                return False
            b_layout = getattr(config, "b_layout", "row_major")
            s_layout = getattr(config, "s_layout", "none_box")
            valid_shape = context.get(f"{name}_valid_shape")
            shape = context.get(f"{name}_shape")
            is_nd2nz_layout = s_layout == "row_major" and b_layout == "col_major"
            is_row_plus_one_layout = (
                is_nd2nz_layout
                and valid_shape is not None
                and shape is not None
                and valid_shape != shape
            )
            if s_layout != "none_box" and not is_nd2nz_layout:
                return False
            if valid_shape is None:
                return False
            if any(not isinstance(dim, int) or dim < 0 for dim in valid_shape):
                return False
            if is_row_plus_one_layout:
                continue
        return True

    def validate_context_attrs(self, context_attrs=None) -> None:
        attrs = dict(context_attrs or {})
        if not attrs:
            return

        supported = dict(self.context_constraints)
        if any(
            key not in supported or value not in supported[key]
            for key, value in attrs.items()
        ):
            raise ValueError(
                f"tile template {self.name!r} does not support context attrs {attrs!r}; "
                f"supported constraints are {supported!r}"
            )

    def specialize(
        self, context_attrs=None, **parameter_specs: TileSpec | ScalarType
    ) -> "SpecializedTileTemplate":
        self.validate_context_attrs(context_attrs)
        converted_specs = {
            name: _coerce_parameter_spec(spec)
            for name, spec in parameter_specs.items()
        }
        return SpecializedTileTemplate(self, converted_specs, context_attrs)


class SpecializedTileTemplate(ModuleArtifact):
    def __init__(
        self,
        descriptor: TileTemplate,
        parameter_specs: dict[str, TileSpec | ScalarType],
        context_attrs: dict[str, object] | None = None,
    ):
        super().__init__(
            descriptor.name,
            module_factory=lambda: _TraceBuilder(
                descriptor, parameter_specs, context_attrs
            ).build_module(),
        )
        self.descriptor = descriptor
        self.parameter_specs = parameter_specs
        self.context_attrs = dict(context_attrs or {})
        self.tile_specs = {
            name: spec for name, spec in parameter_specs.items() if isinstance(spec, TileSpec)
        }


def tile_template(
    *,
    target: str = "a5",
    op: str,
    name: str | None = None,
    ir_level: str = "vpto",
    dtypes: tuple | list = (),
    context_constraints: dict[str, tuple[object, ...]] | None = None,
    constraints: tuple[object, ...] | list[object] = (),
    tags: tuple[str, ...] | list[str] = (),
    priority: int = 100,
    candidate_id: int = 1000,
    single_logical_row_loop: bool = True,
    resource_scope: str | None = None,
    resource_vector_values: int | None = None,
    resource_chunk_streaming: bool = False,
):
    if target != "a5":
        raise ValueError("tile-template tracing currently only supports target='a5'")
    if ir_level not in {"vpto", "vmi"}:
        raise ValueError("tile-template tracing ir_level must be 'vpto' or 'vmi'")

    normalized_context_constraints = tuple(
        (key, tuple(values)) for key, values in (context_constraints or {}).items()
    )

    def decorator(fn):
        source_path = Path(inspect.getsourcefile(fn) or "<unknown>")
        descriptor_name = name or fn.__name__
        return TileTemplate(
            py_fn=fn,
            target=target,
            op=op,
            name=descriptor_name,
            source_label=f"{source_path}:{fn.__name__}",
            ir_level=ir_level,
            dtypes=tuple(tuple(signature) for signature in dtypes),
            context_constraints=normalized_context_constraints,
            constraints=tuple(constraints),
            tags=tuple(tags),
            priority=priority,
            candidate_id=candidate_id,
            single_logical_row_loop=single_logical_row_loop,
            resource_scope=resource_scope,
            resource_vector_values=resource_vector_values,
            resource_chunk_streaming=resource_chunk_streaming,
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


def index_add(lhs, rhs) -> _Value:
    return require_active_runtime("index_add", expected_type=_TraceBuilder).index_binary(
        "add", lhs, rhs
    )


def index_mul(lhs, rhs) -> _Value:
    return require_active_runtime("index_mul", expected_type=_TraceBuilder).index_binary(
        "mul", lhs, rhs
    )


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
    next_value = _Value(arith.SubIOp(remained_val.value, lanes.value).result)
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


def _require_vmi_trace(operation: str) -> _TraceBuilder:
    trace = require_active_runtime(operation, expected_type=_TraceBuilder)
    if trace.descriptor.ir_level != "vmi":
        raise RuntimeError(f"{operation} requires tile_template(..., ir_level='vmi')")
    return trace


def _validate_vmi_block_access(
    tile: _TileProxy,
    coordinate: CanonicalBlockCoordinate,
    *,
    operation: str,
) -> None:
    if not isinstance(tile, _TileProxy):
        raise TypeError(f"{operation} expects a traced Tile argument")
    if not isinstance(coordinate, CanonicalBlockCoordinate):
        raise TypeError(f"{operation} expects a CanonicalBlockCoordinate")
    if tile._spec.shape != coordinate.block_map.shape:
        raise ValueError(
            f"{operation} tile shape {tile._spec.shape} does not match "
            f"CanonicalBlockMap shape {coordinate.block_map.shape}"
        )


def vmi_create_mask(block_map: CanonicalBlockMap, dtype: ScalarType) -> _MaskValue:
    if not isinstance(block_map, CanonicalBlockMap):
        raise TypeError("vmi_create_mask expects a CanonicalBlockMap")
    return vmi_create_mask_lanes(
        block_map.logical_lanes, block_map.logical_lanes, dtype
    )


def vmi_create_mask_lanes(
    active_lanes: int, vector_lanes: int, dtype: ScalarType
) -> _MaskValue:
    trace = _require_vmi_trace("vmi_create_mask_lanes")
    if not isinstance(dtype, ScalarType):
        raise TypeError("vmi_create_mask_lanes expects a tile-template ScalarType")
    if not 0 < active_lanes <= vector_lanes:
        raise ValueError("active_lanes must be in the range [1, vector_lanes]")
    active = trace.index_const(active_lanes)
    result = _vmi.create_mask(active.value, size=vector_lanes)
    return _MaskValue(unwrap_surface_value(result), dtype)


def vmi_prepare_tile_access(*tiles: _TileProxy) -> None:
    trace = _require_vmi_trace("vmi_prepare_tile_access")
    if not tiles:
        raise ValueError("vmi_prepare_tile_access requires at least one Tile")
    for tile in tiles:
        if not isinstance(tile, _TileProxy):
            raise TypeError("vmi_prepare_tile_access expects traced Tile arguments")
        trace.ensure_tile_ptr(tile)


def vmi_vload(tile: _TileProxy, coordinate: CanonicalBlockCoordinate) -> _VectorValue:
    trace = _require_vmi_trace("vmi_vload")
    _validate_vmi_block_access(tile, coordinate, operation="vmi_vload")
    ptr_value = trace.ensure_tile_ptr(tile)
    offset = trace._coerce_index(coordinate.linear_offset)
    result = _vmi.vload(
        ptr_value.value,
        offset.value,
        size=coordinate.block_map.logical_lanes,
    )
    return _VectorValue(unwrap_surface_value(result), tile.element_type)


def vmi_vload_linear(tile: _TileProxy, offset, *, lanes: int) -> _VectorValue:
    trace = _require_vmi_trace("vmi_vload_linear")
    if not isinstance(tile, _TileProxy):
        raise TypeError("vmi_vload_linear expects a traced Tile argument")
    if not isinstance(lanes, int) or lanes <= 0:
        raise ValueError("vmi_vload_linear lanes must be a positive integer")
    ptr_value = trace.ensure_tile_ptr(tile)
    offset_value = trace._coerce_index(offset)
    result = _vmi.vload(ptr_value.value, offset_value.value, size=lanes)
    return _VectorValue(unwrap_surface_value(result), tile.element_type)


def _vmi_binary(
    operation: str,
    lhs: _VectorValue,
    rhs: _VectorValue,
    mask: _MaskValue,
) -> _VectorValue:
    _require_vmi_trace(operation)
    if lhs.dtype != rhs.dtype or lhs.dtype != mask.dtype:
        raise TypeError(f"{operation} operands and mask must use the same dtype")
    emitter = getattr(_vmi, operation.removeprefix("vmi_"))
    result = emitter(lhs.value, rhs.value, mask.value)
    return _VectorValue(unwrap_surface_value(result), lhs.dtype)


def vmi_vadd(lhs: _VectorValue, rhs: _VectorValue, mask: _MaskValue) -> _VectorValue:
    return _vmi_binary("vmi_vadd", lhs, rhs, mask)


def vmi_vsub(lhs: _VectorValue, rhs: _VectorValue, mask: _MaskValue) -> _VectorValue:
    return _vmi_binary("vmi_vsub", lhs, rhs, mask)


def vmi_vmul(lhs: _VectorValue, rhs: _VectorValue, mask: _MaskValue) -> _VectorValue:
    return _vmi_binary("vmi_vmul", lhs, rhs, mask)


def vmi_vdiv(lhs: _VectorValue, rhs: _VectorValue, mask: _MaskValue) -> _VectorValue:
    return _vmi_binary("vmi_vdiv", lhs, rhs, mask)


def vmi_vmax(lhs: _VectorValue, rhs: _VectorValue, mask: _MaskValue) -> _VectorValue:
    return _vmi_binary("vmi_vmax", lhs, rhs, mask)


def _vmi_vec_scalar(
    operation: str,
    source: _VectorValue,
    scalar: _Value,
    mask: _MaskValue,
) -> _VectorValue:
    _require_vmi_trace(operation)
    if source.dtype != mask.dtype:
        raise TypeError(f"{operation} source and mask must use the same dtype")
    expected_scalar = str(_resolve(_scalar_descriptor(source.dtype)))
    if scalar.type_text != expected_scalar:
        raise TypeError(
            f"{operation} scalar must use {expected_scalar}, got {scalar.type_text}"
        )
    emitter = getattr(_vmi, operation.removeprefix("vmi_"))
    result = emitter(source.value, scalar.value, mask.value)
    return _VectorValue(unwrap_surface_value(result), source.dtype)


def vmi_vadds(
    source: _VectorValue, scalar: _Value, mask: _MaskValue
) -> _VectorValue:
    return _vmi_vec_scalar("vmi_vadds", source, scalar, mask)


def vmi_vmuls(
    source: _VectorValue, scalar: _Value, mask: _MaskValue
) -> _VectorValue:
    return _vmi_vec_scalar("vmi_vmuls", source, scalar, mask)


def vmi_vmaxs(
    source: _VectorValue, scalar: _Value, mask: _MaskValue
) -> _VectorValue:
    return _vmi_vec_scalar("vmi_vmaxs", source, scalar, mask)


def vmi_vmins(
    source: _VectorValue, scalar: _Value, mask: _MaskValue
) -> _VectorValue:
    return _vmi_vec_scalar("vmi_vmins", source, scalar, mask)


def vmi_vexp(source: _VectorValue, mask: _MaskValue) -> _VectorValue:
    _require_vmi_trace("vmi_vexp")
    if source.dtype != mask.dtype:
        raise TypeError("vmi_vexp source and mask must use the same dtype")
    result = _vmi.vexp(source.value, mask.value)
    return _VectorValue(unwrap_surface_value(result), source.dtype)


def vmi_vbroadcast(source: _VectorValue, *, lanes: int) -> _VectorValue:
    _require_vmi_trace("vmi_vbroadcast")
    if not isinstance(lanes, int) or lanes <= 0:
        raise ValueError("vmi_vbroadcast lanes must be a positive integer")
    result = _vmi.vbrc(source.value, size=lanes)
    return _VectorValue(unwrap_surface_value(result), source.dtype)


def vmi_scalar_constant(value: float, dtype: ScalarType) -> _Value:
    """Materialize a scalar constant of ``value`` with element type ``dtype``.

    General-purpose literal construction (not reduce-specific): callers choose
    the value. For a reduce accumulator init, the caller picks the op's
    identity element (max->-inf, min->+inf, add->0, prod->1, matching pto-isa
    ``InstrOp::InitVal``), then broadcasts it with ``vmi_vbroadcast_scalar``.
    """
    _require_vmi_trace("vmi_scalar_constant")
    elem_type = _resolve(_scalar_descriptor(dtype))
    return _Value(arith.ConstantOp(elem_type, FloatAttr.get(elem_type, value)).result)


def vmi_vbroadcast_scalar(
    scalar: _Value, *, like: _VectorValue | None = None, dtype: ScalarType | None = None
) -> _VectorValue:
    """Broadcast a scalar into a VL vreg.

    The result element type / lanes come either from an existing vreg
    (``like=some_vreg``) or directly from a dtype (``dtype=f32`` etc.) — use
    ``dtype`` when you only need the type and would otherwise have to emit a
    throwaway load just to obtain it (that load has a Read memory effect and
    cannot be DCE'd).
    """
    _require_vmi_trace("vmi_vbroadcast_scalar")
    if like is None and dtype is None:
        raise TypeError("vmi_vbroadcast_scalar requires like= or dtype=")
    ref_dtype = like.dtype if like is not None else dtype
    expected_scalar = str(_resolve(_scalar_descriptor(ref_dtype)))
    if scalar.type_text != expected_scalar:
        raise TypeError(
            "vmi_vbroadcast_scalar scalar must use "
            f"{expected_scalar}, got {scalar.type_text}"
        )
    if like is not None:
        size = _pto.VMIVRegType(like.value.type).element_count
    else:
        size = dtype.lanes
    result = _vmi.vbrc(scalar.value, size=size)
    return _VectorValue(unwrap_surface_value(result), ref_dtype)


def vmi_vreduce_max(source: _VectorValue, mask: _MaskValue) -> _VectorValue:
    _require_vmi_trace("vmi_vreduce_max")
    if source.dtype != mask.dtype:
        raise TypeError("vmi_vreduce_max source and mask must use the same dtype")
    result = _vmi.vcmax(source.value, mask.value)
    return _VectorValue(unwrap_surface_value(result), source.dtype)


def vmi_vreduce_add(source: _VectorValue, mask: _MaskValue) -> _VectorValue:
    _require_vmi_trace("vmi_vreduce_add")
    if source.dtype != mask.dtype:
        raise TypeError("vmi_vreduce_add source and mask must use the same dtype")
    result = _vmi.vcadd(source.value, mask.value, reassoc=True)
    return _VectorValue(unwrap_surface_value(result), source.dtype)


def vmi_vcvt(source: _VectorValue, dst_dtype: ScalarType) -> _VectorValue:
    _require_vmi_trace("vmi_vcvt")
    if not isinstance(dst_dtype, ScalarType):
        raise TypeError("vmi_vcvt expects a tile-template destination ScalarType")
    result = _vmi.vcvt(source.value, to_dtype=_scalar_descriptor(dst_dtype))
    return _VectorValue(unwrap_surface_value(result), dst_dtype)


def vmi_vstore(
    vec: _VectorValue,
    tile: _TileProxy,
    coordinate: CanonicalBlockCoordinate,
    mask: _MaskValue,
) -> None:
    trace = _require_vmi_trace("vmi_vstore")
    _validate_vmi_block_access(tile, coordinate, operation="vmi_vstore")
    if vec.dtype != tile.element_type or vec.dtype != mask.dtype:
        raise TypeError("vmi_vstore value, destination, and mask must use the same dtype")
    ptr_value = trace.ensure_tile_ptr(tile)
    offset = trace._coerce_index(coordinate.linear_offset)
    _vmi.vstore(vec.value, ptr_value.value, offset.value, mask.value)


def vmi_vstore_linear(
    vec: _VectorValue,
    tile: _TileProxy,
    offset,
    mask: _MaskValue,
) -> None:
    trace = _require_vmi_trace("vmi_vstore_linear")
    if not isinstance(tile, _TileProxy):
        raise TypeError("vmi_vstore_linear expects a traced Tile destination")
    if vec.dtype != tile.element_type or vec.dtype != mask.dtype:
        raise TypeError("vmi_vstore_linear value, destination, and mask must use the same dtype")
    ptr_value = trace.ensure_tile_ptr(tile)
    offset_value = trace._coerce_index(offset)
    _vmi.vstore(vec.value, ptr_value.value, offset_value.value, mask.value)


def _is_tile_annotation(annotation) -> bool:
    if annotation is Tile:
        return True
    if isinstance(annotation, str):
        return annotation == "Tile" or annotation.endswith(".Tile")
    return getattr(annotation, "__name__", None) == "Tile"


def _scalar_type_from_annotation(annotation) -> ScalarType | None:
    if isinstance(annotation, ScalarType):
        return annotation
    if isinstance(annotation, str):
        token = annotation.rsplit(".", 1)[-1]
        return {
            "f32": f32,
            "f16": f16,
            "bf16": bf16,
            "i32": i32,
            "i16": i16,
            "i8": i8,
        }.get(token)
    return None


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
        "ui8": _int8,
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
    "CanonicalBlockMap",
    "CanonicalBlockCoordinate",
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
    "index_add",
    "index_mul",
    "make_mask",
    "vlds",
    "vadd",
    "vsts",
    "vmi_create_mask",
    "vmi_create_mask_lanes",
    "vmi_prepare_tile_access",
    "vmi_vload",
    "vmi_vload_linear",
    "vmi_vadd",
    "vmi_vsub",
    "vmi_vmul",
    "vmi_vdiv",
    "vmi_vmax",
    "vmi_vadds",
    "vmi_vmuls",
    "vmi_vmaxs",
    "vmi_vmins",
    "vmi_vexp",
    "vmi_vbroadcast",
    "vmi_vbroadcast_scalar",
    "vmi_vreduce_max",
    "vmi_vreduce_add",
    "vmi_vcvt",
    "vmi_vstore",
    "vmi_vstore_linear",
]
