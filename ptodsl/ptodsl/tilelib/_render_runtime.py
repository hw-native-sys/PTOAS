# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""
TileLib render runtime.

Traces a tilelang-style template body into a standalone ``func.func`` whose MLIR
uses tile-native handles plus pointer-style ``pto.tile_buf_addr`` /
``pto.vlds/vadd/vsts`` and dynamic ``pto.tile_valid_rows/cols``.

Control flow is handled by the engine's AST rewrite (``rewrite_jit_function``): plain
``for x in range(...)`` in the template body is rewritten at trace time to
``pto.for_(...).carry(...)`` on the public PTODSL control-flow surface, with
loop-carried variables detected by liveness. This module therefore owns only
the entry tile_buf typing and the golden-shaped module/func container; loops,
slicing, valid-shape, mask and vector ops all come from the existing PTODSL
engine.
"""

from __future__ import annotations

import inspect
from dataclasses import dataclass

from .metadata import ScalarSpec, TileSpec, VectorSpec, ViewSpec, scalar_descriptor
from .._ast_rewrite import rewrite_jit_function
from .._context import make_context
from .._surface_types import Tile
from .._surface_values import PartitionTensorViewValue, TensorViewValue, TileValue
from .._tracing import KernelModuleSpec, ModuleStyle, TracingRuntime
from .._tracing.active import activate_runtime, activate_session
from .._surface_values import wrap_surface_value
from .._types import _DType, _resolve

from ptoas.mlir.dialects import func
from ptoas.mlir.ir import Attribute, InsertionPoint, Location, Module, StringAttr, UnitAttr


# ── tile handle handed to the template body ────────────────────────────────────────

@dataclass(frozen=True)
class _TemplateTileConfig:
    b_layout: str
    s_layout: str


class _TemplateTile(TileValue):
    """Engine ``TileValue`` with the template-author alias ``element_type`` and forced
    dynamic ``valid_shape`` (emit ``pto.tile_valid_rows/cols`` rather than folding the
    static ``v_row/v_col`` carried in the tile_buf type).

    Metadata (shape/dtype/memory_space) is supplied from the ``TileSpec`` because a raw
    entry-block ``tile_buf`` type is not introspectable by ``parse_tile_type_metadata``.
    """

    def __init__(self, value, spec: TileSpec):
        elem = _resolve(scalar_descriptor(spec.dtype))
        super().__init__(
            value,
            shape=tuple(spec.shape),
            physical_shape=tuple(spec.shape),
            dtype=elem,
            memory_space=spec.memory_space,
            valid_shape=None,
        )
        # Force the dynamic valid-shape ops to match the tilelang render.
        self.static_valid_shape = None
        self._valid_shape_cache.clear()
        self._template_static_valid_shape = tuple(spec.valid_shape or spec.shape)
        self._template_config = _TemplateTileConfig(
            b_layout=spec.b_layout,
            s_layout=spec.s_layout,
        )
        self.pad_value = getattr(spec, "pad_value", "Null")

    @property
    def element_type(self):
        return self.dtype

    @property
    def config(self):
        return self._template_config


# ── tracing runtime ────────────────────────────────────────────────────────────────

class _TemplateTrace(TracingRuntime):
    def __init__(self, descriptor, tile_specs: dict, context_attrs: dict | None = None):
        super().__init__(
            KernelModuleSpec(
                function_name=descriptor.name,
                target_arch=descriptor.target,
                kernel_kind="vector",
                mode="explicit",
                module_style=ModuleStyle.NESTED,
                source_file=inspect.getsourcefile(descriptor.py_fn) or inspect.getfile(descriptor.py_fn),
                source_line=getattr(descriptor.py_fn.__code__, "co_firstlineno", None),
            )
        )
        self.descriptor = descriptor
        self.operand_specs = tile_specs
        self.context_attrs = dict(context_attrs or {})
        self._ordered_specs: list = []
        self._signature_parameters = tuple(inspect.signature(descriptor.py_fn).parameters.items())

    def compute_argument_types(self):
        arg_types = []
        ordered = []
        for param_name, param in self._signature_parameters:
            spec = self.operand_specs.get(param_name)
            if spec is None:
                raise ValueError(f"missing operand spec for parameter {param_name!r}")
            if isinstance(spec, TileSpec):
                if not _is_tile_annotation(param.annotation):
                    raise TypeError(
                        f"tile-template tile parameter {param_name!r} must be annotated Tile; "
                        f"got {param.annotation!r}"
                    )
                arg_types.append(spec.mlir_type())
            elif isinstance(spec, ScalarSpec):
                if _is_tile_annotation(param.annotation):
                    raise TypeError(
                        f"tile-template scalar parameter {param_name!r} cannot be annotated Tile"
                    )
                arg_types.append(_scalar_argument_type(param.annotation, spec))
            elif isinstance(spec, (ViewSpec, VectorSpec)):
                if _is_tile_annotation(param.annotation):
                    raise TypeError(
                        f"tile-template {type(spec).__name__} parameter "
                        f"{param_name!r} cannot be annotated Tile"
                    )
                arg_types.append(spec.mlir_type())
            else:
                raise TypeError(
                    f"unsupported operand spec for parameter {param_name!r}: {type(spec).__name__}"
                )
            ordered.append((param_name, spec))
        self._ordered_specs = ordered
        return arg_types

    def bind_entry_arguments(self, entry_arguments):
        bound = []
        for arg, (_, spec) in zip(entry_arguments, self._ordered_specs):
            if isinstance(spec, TileSpec):
                bound.append(_TemplateTile(arg, spec))
            elif isinstance(spec, ScalarSpec):
                bound.append(wrap_surface_value(arg))
            elif isinstance(spec, ViewSpec):
                root = TensorViewValue(
                    arg,
                    shape=tuple(spec.shape),
                    strides=tuple(spec.strides) if spec.strides is not None else None,
                )
                bound.append(
                    PartitionTensorViewValue(
                        arg,
                        root_tensor_view=root,
                        offsets=tuple(0 for _ in spec.shape),
                        sizes=tuple(spec.shape),
                    )
                )
            elif isinstance(spec, VectorSpec):
                bound.append(wrap_surface_value(arg))
            else:
                raise TypeError(f"unsupported operand spec {type(spec).__name__}")
        return tuple(bound)

    def trace_entry(self, *args):
        # Apply the engine's AST control-flow rewrite so the template body can use plain
        # `for x in range(...)` (rewritten to pto.for_(...).carry(...)) like tilelang.
        rewritten = rewrite_jit_function(self.descriptor.py_fn)
        rewritten(*args)

    # Custom golden-shaped container: single module(target_arch) + func(instance, kernel_kind).
    def build_standalone_module(self):
        """Build a module in a fresh context for standalone PTODSL use."""
        return self.build_module_in_context(make_context())

    def build_module_in_context(self, context):
        """Build a compiler-owned source module in the explicit context."""
        if context is None:
            raise TypeError("compiler materialization requires an explicit context")
        with context, Location.unknown():
            arg_types = list(self.compute_argument_types())
            module, ir_fn = self._create_instance_module(arg_types)
            session = self.create_session(module, ir_fn)
            entry = ir_fn.add_entry_block()
            with InsertionPoint(entry), activate_runtime(self), activate_session(session):
                self.initialize_session(session, entry)
                args = self.bind_entry_arguments(entry.arguments)
                self.trace_entry(*args)
                self.validate_trace_state()
                self.emit_return()
                self.finalize_session(session)
                session.validate_final_state()
            self.verify_module(module)
            if module.context is not context:
                raise RuntimeError(
                    "TileLib materialization returned a module from a different context"
                )
            return module

    def _create_instance_module(self, arg_types):
        module = Module.create()
        module.operation.attributes["pto.target_arch"] = StringAttr.get(self.descriptor.target)
        with InsertionPoint(module.body):
            fn_ty = func.FunctionType.get(arg_types, [])
            ir_fn = func.FuncOp(self.descriptor.name, fn_ty)
            ir_fn.attributes["pto.tilelang.instance"] = UnitAttr.get()
            ir_fn.attributes["pto.kernel_kind"] = Attribute.parse("#pto.kernel_kind<vector>")
        return module, ir_fn


def _is_tile_annotation(annotation) -> bool:
    if annotation is Tile:
        return True
    if isinstance(annotation, str):
        return annotation == "Tile" or annotation.endswith(".Tile")
    return getattr(annotation, "__name__", None) == "Tile"


def _scalar_argument_type(annotation, spec: ScalarSpec):
    if isinstance(annotation, _DType):
        return _resolve(annotation)
    return spec.mlir_type()


__all__ = ["_TemplateTrace", "_TemplateTile"]
