# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""Common MLIR module/container builders for PTODSL tracing frontends."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from mlir.dialects import func
from mlir.ir import Attribute, InsertionPoint, Module, Operation, StringAttr, UnitAttr


class ModuleStyle(str, Enum):
    """Supported top-level PTODSL module layouts."""

    FLAT_AICORE = "flat_aicore"
    NESTED = "nested"
    BACKEND_PARTITIONED = "backend_partitioned"


@dataclass(frozen=True)
class KernelModuleSpec:
    """Declarative description of a traced PTODSL kernel container."""

    function_name: str
    target_arch: str
    kernel_kind: str
    kernel_kind_explicit: bool = False
    backend: str = "vpto"
    entry: bool = True
    mode: str = "auto"
    insert_sync: bool | None = None
    module_style: ModuleStyle = ModuleStyle.NESTED
    source_file: str | None = None
    source_line: int | None = None
    jit_source: str | None = None


def _build_flat_aicore_module(spec: KernelModuleSpec, arg_types):
    module = Module.create()
    module.operation.attributes["pto.target_arch"] = StringAttr.get(spec.target_arch)
    module.operation.attributes["pto.mode"] = StringAttr.get(spec.mode)
    fn_ty = func.FunctionType.get(arg_types, [])
    with InsertionPoint(module.body):
        ir_fn = func.FuncOp(spec.function_name, fn_ty)
        ir_fn.attributes["pto.entry"] = UnitAttr.get()
    return module, ir_fn


def _build_nested_module(spec: KernelModuleSpec, arg_types):
    outer = Module.create()
    outer.operation.attributes["pto.target_arch"] = StringAttr.get(spec.target_arch)
    outer.operation.attributes["pto.mode"] = StringAttr.get(spec.mode)

    with InsertionPoint(outer.body):
        inner_op = Operation.create("builtin.module", regions=1)
        inner_op.attributes["pto.target_arch"] = StringAttr.get(spec.target_arch)
        inner_op.attributes["pto.mode"] = StringAttr.get(spec.mode)
        inner_body = inner_op.regions[0].blocks.append()

        with InsertionPoint(inner_body):
            fn_ty = func.FunctionType.get(arg_types, [])
            ir_fn = func.FuncOp(spec.function_name, fn_ty)

    return outer, ir_fn


def _apply_child_module_attrs(child_op, spec: KernelModuleSpec) -> None:
    """Populate one child module with PTOAS-facing backend metadata."""
    child_op.attributes["pto.target_arch"] = StringAttr.get(spec.target_arch)
    child_op.attributes["pto.backend"] = StringAttr.get(spec.backend)
    if (
        spec.kernel_kind in {"cube", "vector"}
        and (
            (spec.backend == "vpto" and spec.kernel_kind_explicit)
            or (spec.backend == "emitc" and not spec.entry)
        )
    ):
        child_op.attributes["pto.kernel_kind"] = Attribute.parse(
            f"#pto.kernel_kind<{spec.kernel_kind}>"
        )


def create_container_child_module(outer_module, spec: KernelModuleSpec):
    """Create one child module under *outer_module* and return ``(op, body_block)``."""
    with InsertionPoint(outer_module.body):
        child_op = Operation.create("builtin.module", regions=1)
    _apply_child_module_attrs(child_op, spec)
    body_block = child_op.regions[0].blocks.append()
    return child_op, body_block


def _build_backend_partitioned_module(spec: KernelModuleSpec, arg_types):
    outer = Module.create()
    outer.operation.attributes["pto.target_arch"] = StringAttr.get(spec.target_arch)

    _, child_body = create_container_child_module(outer, spec)
    fn_ty = func.FunctionType.get(arg_types, [])
    with InsertionPoint(child_body):
        ir_fn = func.FuncOp(spec.function_name, fn_ty)
        if spec.entry:
            ir_fn.attributes["pto.entry"] = UnitAttr.get()
        if (
            spec.kernel_kind in {"cube", "vector"}
            and (spec.kernel_kind_explicit or (spec.backend == "emitc" and not spec.entry))
        ):
            ir_fn.attributes["pto.kernel_kind"] = Attribute.parse(
                f"#pto.kernel_kind<{spec.kernel_kind}>"
            )

    return outer, ir_fn


def create_kernel_module(spec: KernelModuleSpec, arg_types):
    """Create the top-level module and entry function for *spec*."""
    if spec.module_style == ModuleStyle.FLAT_AICORE:
        return _build_flat_aicore_module(spec, arg_types)
    if spec.module_style == ModuleStyle.NESTED:
        return _build_nested_module(spec, arg_types)
    if spec.module_style == ModuleStyle.BACKEND_PARTITIONED:
        return _build_backend_partitioned_module(spec, arg_types)
    raise ValueError(f"unsupported PTODSL module style {spec.module_style!r}")


__all__ = [
    "KernelModuleSpec",
    "ModuleStyle",
    "create_container_child_module",
    "create_kernel_module",
]
