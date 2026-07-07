# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Pybinding-based VPTO lowering renderer for TileLang DSL.

This module implements the PybindBackend for Issue #237, providing
direct MLIR IR construction using Python bindings instead of text emission.

Phase 2 implementation: core framework with basic stmt/expr rendering.

Dependencies:
    - mlir.ir: MLIR Python bindings from LLVM build
    - mlir.dialects.func, arith, scf: Standard MLIR dialects
    - mlir.dialects.pto: PTO dialect bindings from PTOAS build

To use this module, ensure PYTHONPATH includes:
    1. LLVM MLIR Python package: $LLVM_BUILD_DIR/tools/mlir/python_packages/mlir_core
    2. PTOAS Python bindings: $PTOAS_BUILD_DIR/python

Example:
    export PYTHONPATH="$LLVM_BUILD_DIR/tools/mlir/python_packages/mlir_core:$PTOAS_BUILD_DIR/python"
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    from mlir import ir as _ods_ir

# Lazy imports - only load when actually needed
_mlir_ir: Any = None
_func_dialect: Any = None
_arith_dialect: Any = None
_scf_dialect: Any = None
_pto_dialect: Any = None
_mlir_types_cache: dict[str, Any] = {}


def _ensure_mlir_bindings() -> None:
    """Ensure MLIR Python bindings are loaded.

    Raises:
        ImportError: If MLIR bindings are not available in PYTHONPATH.
    """
    global _mlir_ir, _func_dialect, _arith_dialect, _scf_dialect, _pto_dialect

    if _mlir_ir is not None:
        return  # Already loaded

    try:
        from mlir import ir as mlir_ir
        from mlir.dialects import func as func_dialect
        from mlir.dialects import arith as arith_dialect
        from mlir.dialects import scf as scf_dialect

        _mlir_ir = mlir_ir
        _func_dialect = func_dialect
        _arith_dialect = arith_dialect
        _scf_dialect = scf_dialect
    except ImportError as exc:
        raise ImportError(
            "MLIR Python bindings not found. Please ensure PYTHONPATH includes "
            "the MLIR Python package from your LLVM build. "
            "Example: export PYTHONPATH=$LLVM_BUILD_DIR/tools/mlir/python_packages/mlir_core "
            "See README.md for build instructions."
        ) from exc

    try:
        from mlir.dialects import pto as pto_dialect
        _pto_dialect = pto_dialect
    except ImportError:
        # PTO dialect is optional for some operations
        _pto_dialect = None


def _get_mlir_context():
    """Get mlir.ir.Context class."""
    _ensure_mlir_bindings()
    return _mlir_ir.Context


def _get_mlir_location():
    """Get mlir.ir.Location class."""
    _ensure_mlir_bindings()
    return _mlir_ir.Location


def _get_mlir_module():
    """Get mlir.ir.Module class."""
    _ensure_mlir_bindings()
    return _mlir_ir.Module


def _get_mlir_insertion_point():
    """Get mlir.ir.InsertionPoint class."""
    _ensure_mlir_bindings()
    return _mlir_ir.InsertionPoint


def _get_mlir_operation():
    """Get mlir.ir.Operation class."""
    _ensure_mlir_bindings()
    return _mlir_ir.Operation


def _get_mlir_f32_type(ctx):
    """Get mlir.ir.F32Type."""
    _ensure_mlir_bindings()
    return _mlir_ir.F32Type.get(ctx)


def _get_mlir_f16_type(ctx):
    """Get mlir.ir.F16Type."""
    _ensure_mlir_bindings()
    return _mlir_ir.F16Type.get(ctx)


def _get_mlir_bf16_type(ctx):
    """Get mlir.ir.BF16Type."""
    _ensure_mlir_bindings()
    return _mlir_ir.BF16Type.get(ctx)


def _get_mlir_f8e4m3_type(ctx):
    """Get mlir.ir.Float8E4M3FNType."""
    _ensure_mlir_bindings()
    return _mlir_ir.Float8E4M3FNType.get(ctx)


def _get_mlir_f8e5m2_type(ctx):
    """Get mlir.ir.Float8E5M2Type."""
    _ensure_mlir_bindings()
    return _mlir_ir.Float8E5M2Type.get(ctx)


def _get_mlir_index_type(ctx):
    """Get mlir.ir.IndexType."""
    _ensure_mlir_bindings()
    return _mlir_ir.IndexType.get(ctx)


def _get_mlir_integer_type(ctx, width, signless=True):
    """Get mlir.ir.IntegerType."""
    _ensure_mlir_bindings()
    if signless:
        return _mlir_ir.IntegerType.get_signless(width, ctx)
    return _mlir_ir.IntegerType.get_signed(width, ctx)


def _get_mlir_string_attr(ctx, value):
    """Get mlir.ir.StringAttr."""
    _ensure_mlir_bindings()
    return _mlir_ir.StringAttr.get(value, ctx)


def _get_mlir_unit_attr(ctx):
    """Get mlir.ir.UnitAttr."""
    _ensure_mlir_bindings()
    return _mlir_ir.UnitAttr.get(ctx)


def _get_mlir_opaque_type(ctx, dialect, name):
    """Get mlir.ir.OpaqueType."""
    _ensure_mlir_bindings()
    return _mlir_ir.OpaqueType.get(ctx, dialect, name)


def _create_operation(name: str, **kwargs) -> Any:
    """Create MLIR Operation with given name and arguments."""
    _ensure_mlir_bindings()
    return _mlir_ir.Operation.create(name, **kwargs)


from .semantic import (
    SemanticAlignStoreStmt,
    SemanticAssignStmt,
    SemanticAttributeAccess,
    SemanticBinaryExpr,
    SemanticBinding,
    SemanticBindingRef,
    SemanticCallExpr,
    SemanticDmaConfigStmt,
    SemanticDmaLoadStmt,
    SemanticDmaStoreStmt,
    SemanticDmaUnaryConfigStmt,
    SemanticExpr,
    SemanticExprStmt,
    SemanticForStmt,
    SemanticGetBufStmt,
    SemanticIfStmt,
    SemanticIfResult,
    SemanticIndexType,
    SemanticKernel,
    SemanticLiteralExpr,
    SemanticLowLevelCopyStmt,
    SemanticMaskType,
    SemanticMemBarStmt,
    SemanticParameter,
    SemanticPipeBarrierStmt,
    SemanticPredicateStoreStmt,
    SemanticPtrType,
    SemanticReturnStmt,
    SemanticRlsBufStmt,
    SemanticScalarStoreStmt,
    SemanticScalarType,
    SemanticSetCrossCoreStmt,
    SemanticSetFlagStmt,
    SemanticSetIntraBlockStmt,
    SemanticSetIntraCoreStmt,
    SemanticShapeType,
    SemanticStrictVecscopeStmt,
    SemanticStmt,
    SemanticSubscriptAccess,
    SemanticSymbolExpr,
    SemanticTensorSliceExpr,
    SemanticTensorViewType,
    SemanticPartitionTensorViewType,
    SemanticTileType,
    SemanticTupleExpr,
    SemanticType,
    SemanticVecscopeStmt,
    SemanticVectorPairStoreStmt,
    SemanticVectorStoreStmt,
    SemanticWaitFlagDevStmt,
    SemanticWaitFlagStmt,
    SemanticWaitIntraCoreStmt,
)
from .types import ScalarType, TileConfig


@dataclass(frozen=True)
class _PybindValue:
    """Wrapper for MLIR Value with semantic type info.

    For ops without SSA results (like TLoad, TStore), value can be None.
    """

    value: Any | None  # _mlir_ir.Value | None (lazy loaded)
    semantic_type: SemanticType

    @property
    def name(self) -> str:
        """Debug name for the value."""
        if self.value is None:
            return "%void"
        return getattr(self.value, "name", None) or f"%v{id(self.value)}"

    @property
    def has_result(self) -> bool:
        """Check if this value has an actual SSA result."""
        return self.value is not None


class PybindRenderer:
    """Core implementation for pybinding-based lowering.

    This renderer constructs MLIR IR directly using Python bindings,
    producing the same authoring-form VPTO as the text backend.
    """

    def __init__(self, kernel: SemanticKernel):
        # Ensure MLIR bindings are available before initialization
        _ensure_mlir_bindings()
        self.kernel = kernel
        self._ctx: Any | None = None  # Context
        self._module: Any | None = None  # Module
        self._entry_block: Any | None = None  # Block
        self._env: dict[str, _PybindValue] = {}
        self._constant_cache: dict[tuple[str, Any], _PybindValue] = {}
        self._insertion_point_stack: list[Any] = []  # InsertionPoint

    def render(self) -> Any:  # Module
        """Build MLIR Module using pybinding ops."""
        Context = _get_mlir_context()
        Location = _get_mlir_location()
        Module = _get_mlir_module()

        with Context() as ctx:
            self._ctx = ctx
            self._register_dialects(ctx)
            with Location.unknown(ctx):
                self._module = Module.create()
                self._build_kernel_function()

        return self._module

    def _register_dialects(self, ctx: Any) -> None:
        """Register all required dialects.

        Modern MLIR Python bindings auto-load dialects when their ops are used.
        We call load_all_available_dialects() to ensure dialects are ready.
        """
        # Load all available dialects - this includes func, arith, scf, etc.
        ctx.load_all_available_dialects()

    def _build_kernel_function(self) -> None:
        """Build the kernel function with parameters."""
        InsertionPoint = _get_mlir_insertion_point()

        # Construct parameter types
        param_types = []
        for param in self.kernel.parameters:
            mlir_type = self._convert_type(param.type)
            param_types.append(mlir_type)

        # Create function type
        fn_ty = _func_dialect.FunctionType.get(param_types, [])

        # Create function in module body
        with InsertionPoint(self._module.body):
            fn = _func_dialect.FuncOp(self.kernel.symbol_name, fn_ty)
            fn.attributes["pto.entry"] = _get_mlir_unit_attr(self._ctx)
            fn.attributes["pto.tilelang.instance"] = _get_mlir_unit_attr(self._ctx)
            self._entry_block = fn.add_entry_block()

        # Set up parameter environment
        arg_index = 0
        for param in self.kernel.parameters:
            mlir_type = self._convert_type(param.type)
            self._env[param.name] = _PybindValue(
                value=self._entry_block.arguments[arg_index],
                semantic_type=param.type,
            )
            arg_index += 1

        # Render body statements
        with InsertionPoint(self._entry_block):
            for stmt in self.kernel.body:
                self._render_stmt(stmt)
            _func_dialect.ReturnOp([])

    def _convert_type(self, semantic_type: SemanticType) -> Any:
        """Convert SemanticType to MLIR Type."""
        if isinstance(semantic_type, SemanticScalarType):
            return self._convert_scalar_type(semantic_type)
        if isinstance(semantic_type, SemanticPtrType):
            return self._convert_ptr_type(semantic_type)
        if isinstance(semantic_type, SemanticTensorViewType):
            return self._convert_tensor_view_type(semantic_type)
        if isinstance(semantic_type, SemanticPartitionTensorViewType):
            return self._convert_partition_tensor_view_type(semantic_type)
        if isinstance(semantic_type, SemanticTileType):
            return self._convert_tile_type(semantic_type)
        if isinstance(semantic_type, SemanticIndexType):
            return _get_mlir_index_type(self._ctx)
        if isinstance(semantic_type, SemanticMaskType):
            return self._convert_mask_type(semantic_type)
        raise NotImplementedError(
            f"Type conversion not implemented for {type(semantic_type).__name__}"
        )

    def _convert_scalar_type(self, semantic_type: SemanticScalarType) -> Any:
        """Convert SemanticScalarType to MLIR Type."""
        dtype = semantic_type.dtype
        name = dtype.name

        if name == "f32":
            return _get_mlir_f32_type(self._ctx)
        if name == "f16":
            return _get_mlir_f16_type(self._ctx)
        if name == "bf16":
            return _get_mlir_bf16_type(self._ctx)
        if name == "f8e4m3":
            return _get_mlir_f8e4m3_type(self._ctx)
        if name == "f8e5m2":
            return _get_mlir_f8e5m2_type(self._ctx)
        if name == "hif8" and _pto_dialect is not None:
            return _pto_dialect.HiF8Type.get()
        if name == "f4e1m2x2" and _pto_dialect is not None:
            return _pto_dialect.F4E1M2x2Type.get()
        if name == "f4e2m1x2" and _pto_dialect is not None:
            return _pto_dialect.F4E2M1x2Type.get()
        if name == "index":
            return _get_mlir_index_type(self._ctx)

        # Integer types (signless for MLIR convention)
        if name == "i1":
            return _get_mlir_integer_type(self._ctx, 1)
        if name == "i8":
            return _get_mlir_integer_type(self._ctx, 8)
        if name == "i16":
            return _get_mlir_integer_type(self._ctx, 16)
        if name == "i32":
            return _get_mlir_integer_type(self._ctx, 32)
        if name == "i64":
            return _get_mlir_integer_type(self._ctx, 64)

        raise NotImplementedError(f"Scalar type conversion not implemented for {name}")

    def _convert_ptr_type(self, semantic_type: SemanticPtrType) -> Any:
        """Convert SemanticPtrType to PTO PtrType."""
        if _pto_dialect is None:
            raise NotImplementedError("PTO dialect not available")
        elem_type = self._convert_type(semantic_type.element_type)
        return _pto_dialect.PtrType.get(elem_type, self._ctx)

    def _convert_tensor_view_type(
        self, semantic_type: SemanticTensorViewType
    ) -> Any:
        """Convert SemanticTensorViewType to PTO TensorViewType."""
        if _pto_dialect is None:
            raise NotImplementedError("PTO dialect not available")
        elem_type = self._convert_scalar_type(
            SemanticScalarType(dtype=semantic_type.element_dtype)
        )
        return _pto_dialect.TensorViewType.get(semantic_type.rank, elem_type, self._ctx)

    def _convert_partition_tensor_view_type(
        self, semantic_type: SemanticPartitionTensorViewType
    ) -> Any:
        """Convert SemanticPartitionTensorViewType to PTO PartitionTensorViewType."""
        if _pto_dialect is None:
            raise NotImplementedError("PTO dialect not available")
        elem_type = self._convert_scalar_type(
            SemanticScalarType(dtype=semantic_type.element_dtype)
        )
        return _pto_dialect.PartitionTensorViewType.get(
            list(semantic_type.shape), elem_type, self._ctx
        )

    def _convert_tile_type(self, semantic_type: SemanticTileType) -> Any:
        """Convert SemanticTileType to PTO TileBufType (Phase 3)."""
        # Placeholder - will be implemented in Phase 3
        raise NotImplementedError("TileBufType conversion (Phase 3)")

    def _convert_mask_type(self, semantic_type: SemanticMaskType) -> Any:
        """Convert SemanticMaskType to PTO mask type."""
        if _pto_dialect is None:
            raise NotImplementedError("PTO dialect not available")
        # Use !pto.mask type
        name = semantic_type.dtype.name
        if name == "b8":
            return _get_mlir_integer_type(self._ctx, 8)  # Placeholder
        if name == "b16":
            return _get_mlir_integer_type(self._ctx, 16)  # Placeholder
        if name == "b32":
            return _get_mlir_integer_type(self._ctx, 32)  # Placeholder
        raise NotImplementedError(f"Mask type conversion not implemented for {name}")

    def _render_stmt(self, stmt: SemanticStmt) -> None:
        """Render a statement using pybinding ops."""
        if isinstance(stmt, SemanticAssignStmt):
            self._render_assign(stmt)
        elif isinstance(stmt, SemanticExprStmt):
            self._render_expr_stmt(stmt)
        elif isinstance(stmt, SemanticForStmt):
            self._render_for(stmt)
        elif isinstance(stmt, SemanticIfStmt):
            self._render_if(stmt)
        elif isinstance(stmt, SemanticVecscopeStmt):
            self._render_vecscope(stmt)
        elif isinstance(stmt, SemanticStrictVecscopeStmt):
            self._render_strict_vecscope(stmt)
        elif isinstance(stmt, SemanticReturnStmt):
            self._render_return(stmt)
        elif isinstance(stmt, SemanticSetFlagStmt):
            self._render_set_flag(stmt)
        elif isinstance(stmt, SemanticWaitFlagStmt):
            self._render_wait_flag(stmt)
        elif isinstance(stmt, SemanticPipeBarrierStmt):
            self._render_pipe_barrier(stmt)
        elif isinstance(stmt, SemanticDmaLoadStmt):
            self._render_dma_load(stmt)
        elif isinstance(stmt, SemanticDmaStoreStmt):
            self._render_dma_store(stmt)
        elif isinstance(stmt, SemanticVectorStoreStmt):
            self._render_vector_store(stmt)
        elif isinstance(stmt, SemanticVectorPairStoreStmt):
            self._render_vector_pair_store(stmt)
        elif isinstance(stmt, SemanticPredicateStoreStmt):
            self._render_predicate_store(stmt)
        elif isinstance(stmt, SemanticAlignStoreStmt):
            self._render_align_store(stmt)
        elif isinstance(stmt, SemanticScalarStoreStmt):
            self._render_scalar_store(stmt)
        elif isinstance(stmt, SemanticGetBufStmt):
            self._render_get_buf(stmt)
        elif isinstance(stmt, SemanticRlsBufStmt):
            self._render_rls_buf(stmt)
        elif isinstance(stmt, SemanticMemBarStmt):
            self._render_mem_bar(stmt)
        elif isinstance(stmt, SemanticSetCrossCoreStmt):
            self._render_set_cross_core(stmt)
        elif isinstance(stmt, SemanticSetIntraBlockStmt):
            self._render_set_intra_block(stmt)
        elif isinstance(stmt, SemanticSetIntraCoreStmt):
            self._render_set_intra_core(stmt)
        elif isinstance(stmt, SemanticWaitFlagDevStmt):
            self._render_wait_flag_dev(stmt)
        elif isinstance(stmt, SemanticWaitIntraCoreStmt):
            self._render_wait_intra_core(stmt)
        elif isinstance(stmt, SemanticDmaConfigStmt):
            self._render_dma_config(stmt)
        elif isinstance(stmt, SemanticDmaUnaryConfigStmt):
            self._render_dma_unary_config(stmt)
        elif isinstance(stmt, SemanticLowLevelCopyStmt):
            self._render_low_level_copy(stmt)
        else:
            # Placeholder for remaining stmt types (Phase 3+)
            raise NotImplementedError(
                f"Statement rendering not implemented for {type(stmt).__name__} (Phase 3+)"
            )

    def _render_assign(self, stmt: SemanticAssignStmt) -> None:
        """Render assignment statement."""
        value = self._lower_expr(stmt.value)
        self._env[stmt.target.name] = value

    def _render_expr_stmt(self, stmt: SemanticExprStmt) -> None:
        """Render expression statement (side-effect only)."""
        self._lower_expr(stmt.expr)

    def _render_for(self, stmt: SemanticForStmt) -> None:
        """Render scf.for loop with loop-carried variable support."""
        lb = self._lower_expr(stmt.lower_bound)
        ub = self._lower_expr(stmt.upper_bound)
        step = self._lower_expr(stmt.step)

        # Ensure index type for loop bounds
        lb_index = self._ensure_index(lb)
        ub_index = self._ensure_index(ub)
        step_index = self._ensure_index(step)

        # Save outer environment
        outer_env = dict(self._env)

        # Handle loop-carried variables
        carried_bindings = stmt.loop_carried if hasattr(stmt, "loop_carried") else ()

        if not carried_bindings:
            # Simple loop without iter_args
            for_op = _scf_dialect.ForOp(lb_index.value, ub_index.value, step_index.value)

            with InsertionPoint(for_op.body):
                # Bind induction variable
                iv = for_op.induction_variable
                self._env[stmt.loop_var.name] = _PybindValue(
                    value=iv,
                    semantic_type=SemanticIndexType(),
                )

                # Render body statements
                for body_stmt in stmt.body:
                    self._render_stmt(body_stmt)

                _scf_dialect.YieldOp([])

            # Restore outer environment
            self._env = outer_env
            return

        # Loop with iter_args (loop-carried values)
        # Collect initial values for iter_args
        initial_values = []
        for binding in carried_bindings:
            if binding.name in outer_env:
                init_val = outer_env[binding.name]
                initial_values.append(init_val)
            else:
                raise ValueError(f"Loop-carried variable {binding.name} not in outer scope")

        # Create scf.for with iter_args
        init_operand_values = [v.value for v in initial_values]
        for_op = _scf_dialect.ForOp(
            lb_index.value,
            ub_index.value,
            step_index.value,
            iterArgs=init_operand_values,
        )

        # Enter loop body
        with InsertionPoint(for_op.body):
            # Bind induction variable
            iv = for_op.induction_variable
            self._env[stmt.loop_var.name] = _PybindValue(
                value=iv,
                semantic_type=SemanticIndexType(),
            )

            # Bind iter_args (block arguments for loop-carried values)
            for i, binding in enumerate(carried_bindings):
                iter_arg_val = for_op.inner_iter_args[i]
                self._env[binding.name] = _PybindValue(
                    value=iter_arg_val,
                    semantic_type=binding.type,
                )

            # Render body statements
            for body_stmt in stmt.body:
                self._render_stmt(body_stmt)

            # Yield the final values of loop-carried variables
            yield_values = []
            for binding in carried_bindings:
                if binding.name in self._env:
                    final_val = self._env[binding.name]
                    yield_values.append(final_val.value)
                else:
                    raise ValueError(f"Loop-carried variable {binding.name} not in body scope")

            _scf_dialect.YieldOp(yield_values)

        # Bind loop results to carried variable names in outer scope
        for i, binding in enumerate(carried_bindings):
            result_val = for_op.result[i]
            self._env[binding.name] = _PybindValue(
                value=result_val,
                semantic_type=binding.type,
            )

    def _render_if(self, stmt: SemanticIfStmt) -> None:
        """Render scf.if conditional."""
        cond = self._lower_expr(stmt.condition)
        cond_i1 = self._ensure_i1(cond)

        # Determine if we need results (if-then-else with yield)
        has_else = len(stmt.else_body) > 0

        # Create scf.if
        if_op = _scf_dialect.IfOp(cond_i1.value, hasElse=has_else)

        # Save outer environment
        outer_env = dict(self._env)

        # Render then block
        with InsertionPoint(if_op.then_block):
            for then_stmt in stmt.then_body:
                self._render_stmt(then_stmt)
            _scf_dialect.YieldOp([])

        # Render else block if present
        if has_else:
            with InsertionPoint(if_op.else_block):
                for else_stmt in stmt.else_body:
                    self._render_stmt(else_stmt)
                _scf_dialect.YieldOp([])

        # Restore outer environment
        self._env = outer_env

    def _render_vecscope(self, stmt: SemanticVecscopeStmt) -> None:
        """Render pto.vecscope region."""
        if _pto_dialect is None:
            raise NotImplementedError("PTO dialect not available")

        # Create vecscope region
        # pto.vecscope { ... }
        vecscope_op = _create_operation(
            "pto.vecscope",
            regions=1,
        )

        # Enter the vecscope body
        with InsertionPoint(vecscope_op.regions[0].blocks.append()):
            for body_stmt in stmt.body:
                self._render_stmt(body_stmt)

    def _render_strict_vecscope(self, stmt: SemanticStrictVecscopeStmt) -> None:
        """Render pto.strict_vecscope region with captures."""
        if _pto_dialect is None:
            raise NotImplementedError("PTO dialect not available")

        # Lower capture expressions
        capture_values = []
        for capture_expr in stmt.captures:
            capture_val = self._lower_expr(capture_expr)
            capture_values.append(capture_val)

        # Create strict_vecscope with captures
        # pto.strict_vecscope(%captures) ^bb0(%args): { ... }
        operands = [cv.value for cv in capture_values]

        # Determine block argument types from captures
        block_arg_types = [cv.semantic_type for cv in capture_values]

        # Create the operation
        strict_vecscope_op = _create_operation(
            "pto.strict_vecscope",
            operands=operands,
            regions=1,
        )

        # Enter the strict_vecscope body with block arguments
        block = strict_vecscope_op.regions[0].blocks.append(
            *[self._convert_type(t) for t in block_arg_types]
        )

        # Bind block arguments to the environment
        for i, binding in enumerate(stmt.block_arguments):
            arg_value = block.arguments[i]
            self._env[binding.name] = _PybindValue(
                value=arg_value,
                semantic_type=block_arg_types[i],
            )

        with InsertionPoint(block):
            for body_stmt in stmt.body:
                self._render_stmt(body_stmt)

    def _render_return(self, stmt: SemanticReturnStmt) -> None:
        """Render return statement."""
        if stmt.value is None:
            _func_dialect.ReturnOp([])
        else:
            value = self._lower_expr(stmt.value)
            _func_dialect.ReturnOp([value.value])

    def _render_set_flag(self, stmt: SemanticSetFlagStmt) -> None:
        """Render pto.set_flag operation."""
        if _pto_dialect is None:
            raise NotImplementedError("PTO dialect not available")

        # pto.set_flag["src_pipe", "dst_pipe", "event"]
        # Use the convenience helper from pto dialect
        src_pipe = stmt.src_pipe
        dst_pipe = stmt.dst_pipe
        event = stmt.event

        # Create the operation with pipe attributes
        try:
            _pto_dialect.set_flag(src_pipe, dst_pipe, event)
        except (TypeError, AttributeError):
            # Fall back to raw operation creation
            src_attr = _pto_dialect.PipeAttr.get(
                _pto_dialect.PIPE[src_pipe.upper()], self._ctx
            )
            dst_attr = _pto_dialect.PipeAttr.get(
                _pto_dialect.PIPE[dst_pipe.upper()], self._ctx
            )
            event_attr = _pto_dialect.EventAttr.get(
                _pto_dialect.EVENT[event.upper()], self._ctx
            )
            _create_operation(
                "pto.set_flag",
                attributes={"src_pipe": src_attr, "dst_pipe": dst_attr, "event": event_attr},
            )

    def _render_wait_flag(self, stmt: SemanticWaitFlagStmt) -> None:
        """Render pto.wait_flag operation."""
        if _pto_dialect is None:
            raise NotImplementedError("PTO dialect not available")

        # pto.wait_flag["src_pipe", "dst_pipe", "event"]
        src_pipe = stmt.src_pipe
        dst_pipe = stmt.dst_pipe
        event = stmt.event

        try:
            _pto_dialect.wait_flag(src_pipe, dst_pipe, event)
        except (TypeError, AttributeError):
            src_attr = _pto_dialect.PipeAttr.get(
                _pto_dialect.PIPE[src_pipe.upper()], self._ctx
            )
            dst_attr = _pto_dialect.PipeAttr.get(
                _pto_dialect.PIPE[dst_pipe.upper()], self._ctx
            )
            event_attr = _pto_dialect.EventAttr.get(
                _pto_dialect.EVENT[event.upper()], self._ctx
            )
            _create_operation(
                "pto.wait_flag",
                attributes={"src_pipe": src_attr, "dst_pipe": dst_attr, "event": event_attr},
            )

    def _render_pipe_barrier(self, stmt: SemanticPipeBarrierStmt) -> None:
        """Render pto.barrier operation."""
        if _pto_dialect is None:
            raise NotImplementedError("PTO dialect not available")

        # pto.barrier #pto.pipe<pipe_name>
        pipe_name = stmt.pipe

        try:
            _pto_dialect.barrier(pipe_name)
        except (TypeError, AttributeError):
            pipe_attr = _pto_dialect.PipeAttr.get(
                _pto_dialect.PIPE[pipe_name.upper()], self._ctx
            )
            _create_operation(
                "pto.barrier",
                attributes={"pipe": pipe_attr},
            )

    def _lower_expr(self, expr: SemanticExpr) -> _PybindValue:
        """Lower expression to MLIR Value."""
        if isinstance(expr, SemanticLiteralExpr):
            return self._lower_literal(expr)
        if isinstance(expr, SemanticBindingRef):
            return self._env[expr.binding.name]
        if isinstance(expr, SemanticSymbolExpr):
            return self._lower_symbol(expr)
        if isinstance(expr, SemanticBinaryExpr):
            return self._lower_binary(expr)
        if isinstance(expr, SemanticCallExpr):
            return self._lower_call(expr)
        if isinstance(expr, SemanticAttributeAccess):
            return self._lower_attribute_access(expr)
        if isinstance(expr, SemanticSubscriptAccess):
            return self._lower_subscript_access(expr)
        if isinstance(expr, SemanticTupleExpr):
            return self._lower_tuple(expr)
        raise NotImplementedError(
            f"Expression lowering not implemented for {type(expr).__name__}"
        )

    def _lower_literal(self, expr: SemanticLiteralExpr) -> _PybindValue:
        """Lower literal constant."""
        value = expr.value
        semantic_type = expr.type

        # Check cache for repeated constants
        cache_key = (semantic_type.__class__.__name__, value)
        if cache_key in self._constant_cache:
            return self._constant_cache[cache_key]

        mlir_type = self._convert_type(semantic_type)

        if isinstance(semantic_type, SemanticIndexType):
            const_op = _arith_dialect.ConstantOp(_get_mlir_index_type(self._ctx), value)
        elif isinstance(semantic_type, SemanticScalarType):
            dtype = semantic_type.dtype
            name = dtype.name
            if name in ("i1", "i8", "i16", "i32", "i64"):
                const_op = _arith_dialect.ConstantOp(mlir_type, value)
            elif name in ("f16", "bf16", "f32"):
                const_op = _arith_dialect.ConstantOp(mlir_type, float(value))
            else:
                raise NotImplementedError(f"Literal constant for {name}")
        else:
            raise NotImplementedError(
                f"Literal constant not implemented for {type(semantic_type).__name__}"
            )

        result = _PybindValue(value=const_op.result, semantic_type=semantic_type)
        self._constant_cache[cache_key] = result
        return result

    def _lower_symbol(self, expr: SemanticSymbolExpr) -> _PybindValue:
        """Lower symbol expression (constant symbols like enums)."""
        # Symbol expressions represent compile-time known values (enums, etc.)
        # They are lowered as constants
        value = expr.value
        semantic_type = expr.type

        mlir_type = self._convert_type(semantic_type)

        # For integer-valued symbols
        if isinstance(value, int):
            const_op = _arith_dialect.ConstantOp(mlir_type, value)
            return _PybindValue(value=const_op.result, semantic_type=semantic_type)

        raise NotImplementedError(f"Symbol lowering for value type {type(value)}")

    def _lower_binary(self, expr: SemanticBinaryExpr) -> _PybindValue:
        """Lower binary expression."""
        lhs = self._lower_expr(expr.lhs)
        rhs = self._lower_expr(expr.rhs)

        # Determine the operation based on op type
        op_name = expr.op

        if op_name == "add":
            if isinstance(expr.type, SemanticIndexType):
                result = _arith_dialect.AddIOp(lhs.value, rhs.value).result
            elif isinstance(expr.type, SemanticScalarType):
                dtype = expr.type.dtype
                if dtype.name in ("f16", "bf16", "f32"):
                    result = _arith_dialect.AddFOp(lhs.value, rhs.value).result
                else:
                    result = _arith_dialect.AddIOp(lhs.value, rhs.value).result
            else:
                raise NotImplementedError(f"Add for {type(expr.type).__name__}")
            return _PybindValue(value=result, semantic_type=expr.type)

        if op_name == "sub":
            if isinstance(expr.type, SemanticIndexType):
                result = _arith_dialect.SubIOp(lhs.value, rhs.value).result
            elif isinstance(expr.type, SemanticScalarType):
                dtype = expr.type.dtype
                if dtype.name in ("f16", "bf16", "f32"):
                    result = _arith_dialect.SubFOp(lhs.value, rhs.value).result
                else:
                    result = _arith_dialect.SubIOp(lhs.value, rhs.value).result
            else:
                raise NotImplementedError(f"Sub for {type(expr.type).__name__}")
            return _PybindValue(value=result, semantic_type=expr.type)

        if op_name == "mul":
            if isinstance(expr.type, SemanticIndexType):
                result = _arith_dialect.MulIOp(lhs.value, rhs.value).result
            elif isinstance(expr.type, SemanticScalarType):
                dtype = expr.type.dtype
                if dtype.name in ("f16", "bf16", "f32"):
                    result = _arith_dialect.MulFOp(lhs.value, rhs.value).result
                else:
                    result = _arith_dialect.MulIOp(lhs.value, rhs.value).result
            else:
                raise NotImplementedError(f"Mul for {type(expr.type).__name__}")
            return _PybindValue(value=result, semantic_type=expr.type)

        if op_name == "div":
            if isinstance(expr.type, SemanticScalarType):
                dtype = expr.type.dtype
                if dtype.name in ("f16", "bf16", "f32"):
                    result = _arith_dialect.DivFOp(lhs.value, rhs.value).result
                else:
                    result = _arith_dialect.DivSIOp(lhs.value, rhs.value).result
            else:
                raise NotImplementedError(f"Div for {type(expr.type).__name__}")
            return _PybindValue(value=result, semantic_type=expr.type)

        if op_name == "floordiv":
            result = _arith_dialect.FloorDivSIOp(lhs.value, rhs.value).result
            return _PybindValue(value=result, semantic_type=expr.type)

        raise NotImplementedError(f"Binary op '{op_name}' not implemented")

    def _lower_call(self, expr: SemanticCallExpr) -> _PybindValue:
        """Lower call expression."""
        namespace = expr.namespace
        name = expr.name
        args = [self._lower_expr(arg) for arg in expr.args]

        if namespace == "pto":
            return self._lower_pto_call(name, args, expr)
        elif namespace == "arith":
            return self._lower_arith_call(name, args, expr)
        elif namespace == "func":
            return self._lower_func_call(name, args, expr)

        raise NotImplementedError(f"Call namespace '{namespace}' not implemented")

    def _lower_pto_call(
        self, name: str, args: list[_PybindValue], expr: SemanticCallExpr
    ) -> _PybindValue:
        """Lower pto.* call."""
        if _pto_dialect is None:
            raise NotImplementedError("PTO dialect not available")

        # =========================================================================
        # Tile/Tensor ops (AllocTile, MakeTensorView, PartitionView)
        # =========================================================================

        if name == "alloc_tile":
            # pto.alloc_tile -> returns TileBuf
            result_type = self._convert_type(expr.type)
            alloc_op = _create_operation(
                "pto.alloc_tile",
                results=[result_type],
            )
            return _PybindValue(value=alloc_op.result, semantic_type=expr.type)

        if name == "make_tensor_view":
            # pto.make_tensor_view (ptr, shape, strides) -> TensorView
            result_type = self._convert_type(expr.type)
            ptr_val = args[0].value
            shape_vals = [args[i].value for i in range(1, len(args))]
            make_op = _create_operation(
                "pto.make_tensor_view",
                results=[result_type],
                operands=[ptr_val] + shape_vals,
            )
            return _PybindValue(value=make_op.result, semantic_type=expr.type)

        if name == "partition_view":
            # pto.partition_view (tensor_view, offsets, sizes) -> PartitionTensorView
            result_type = self._convert_type(expr.type)
            tv_val = args[0].value
            offset_vals = [args[i].value for i in range(1, len(args) - 1)]
            size_vals_start = len(args) // 2
            size_vals = [args[i].value for i in range(size_vals_start, len(args))]
            partition_op = _create_operation(
                "pto.partition_view",
                results=[result_type],
                operands=[tv_val] + offset_vals + size_vals,
            )
            return _PybindValue(value=partition_op.result, semantic_type=expr.type)

        # =========================================================================
        # Compute ops (TLoad, TStore, TAdd, TAbs, etc.)
        # =========================================================================

        if name == "tload":
            # pto.tload (partition_view, tile_buf) - no result (side-effect op)
            pv_val = args[0].value
            tb_val = args[1].value
            _create_operation(
                "pto.tload",
                operands=[pv_val, tb_val],
            )
            # Return void value (TLoad has no SSA result)
            return _PybindValue(
                value=None,
                semantic_type=expr.type,
            )

        if name == "tstore":
            # pto.tstore (tile_buf, partition_view) - no result (side-effect op)
            tb_val = args[0].value
            pv_val = args[1].value
            _create_operation(
                "pto.tstore",
                operands=[tb_val, pv_val],
            )
            return _PybindValue(
                value=None,
                semantic_type=expr.type,
            )

        if name in ("tadd", "tsub", "tmul"):
            # pto.tadd/tsub/tmul (src0, src1, dst) - no result (side-effect op)
            src0_val = args[0].value
            src1_val = args[1].value
            dst_val = args[2].value
            _create_operation(
                f"pto.{name}",
                operands=[src0_val, src1_val, dst_val],
            )
            return _PybindValue(
                value=None,
                semantic_type=expr.type,
            )

        if name == "tabs":
            # pto.tabs (src, dst) - no result (side-effect op)
            src_val = args[0].value
            dst_val = args[1].value
            _create_operation(
                "pto.tabs",
                operands=[src_val, dst_val],
            )
            return _PybindValue(
                value=None,
                semantic_type=expr.type,
            )

        if name == "tmad":
            # pto.tmad (src0, src1, dst) - tile multiply-add (no result)
            # Computes: dst = src0 * src1 + dst (or similar MAD pattern)
            src0_val = args[0].value
            src1_val = args[1].value
            dst_val = args[2].value
            _create_operation(
                "pto.tmad",
                operands=[src0_val, src1_val, dst_val],
            )
            return _PybindValue(
                value=None,
                semantic_type=expr.type,
            )

        if name == "tmmad":
            # pto.tmmad (src0, src1, dst) - tile matrix multiply-add (no result)
            # Computes matrix multiplication: dst += src0 @ src1
            # Used for GEMM-style operations on tile buffers
            src0_val = args[0].value
            src1_val = args[1].value
            dst_val = args[2].value
            _create_operation(
                "pto.tmmad",
                operands=[src0_val, src1_val, dst_val],
            )
            return _PybindValue(
                value=None,
                semantic_type=expr.type,
            )

        if name == "trelu":
            # pto.trelu (src, dst) - tile ReLU activation (no result)
            src_val = args[0].value
            dst_val = args[1].value
            _create_operation(
                "pto.trelu",
                operands=[src_val, dst_val],
            )
            return _PybindValue(
                value=None,
                semantic_type=expr.type,
            )

        if name == "tdiv":
            # pto.tdiv (src0, src1, dst) - tile division (no result)
            src0_val = args[0].value
            src1_val = args[1].value
            dst_val = args[2].value
            _create_operation(
                "pto.tdiv",
                operands=[src0_val, src1_val, dst_val],
            )
            return _PybindValue(
                value=None,
                semantic_type=expr.type,
            )

        if name == "tcopy":
            # pto.tcopy (src, dst) - tile copy (no result)
            src_val = args[0].value
            dst_val = args[1].value
            _create_operation(
                "pto.tcopy",
                operands=[src_val, dst_val],
            )
            return _PybindValue(
                value=None,
                semantic_type=expr.type,
            )

        # =========================================================================
        # Vector ops (vlds, vsts, vabs, etc.)
        # =========================================================================

        if name in ("vlds", "vldas", "vldus"):
            # pto.vlds/vldas/vldus (ptr, offset) -> VReg
            result_type = self._convert_type(expr.type)
            ptr_val = args[0].value
            offset_val = args[1].value
            load_op = _create_operation(
                f"pto.{name}",
                results=[result_type],
                operands=[ptr_val, offset_val],
            )
            return _PybindValue(value=load_op.result, semantic_type=expr.type)

        if name == "vabs":
            # pto.vabs (vec, mask) -> VReg
            result_type = self._convert_type(expr.type)
            vec_val = args[0].value
            mask_val = args[1].value
            abs_op = _create_operation(
                "pto.vabs",
                results=[result_type],
                operands=[vec_val, mask_val],
            )
            return _PybindValue(value=abs_op.result, semantic_type=expr.type)

        if name == "vadd":
            # pto.vadd (vec0, vec1, mask) -> VReg
            result_type = self._convert_type(expr.type)
            vec0_val = args[0].value
            vec1_val = args[1].value
            mask_val = args[2].value
            add_op = _create_operation(
                "pto.vadd",
                results=[result_type],
                operands=[vec0_val, vec1_val, mask_val],
            )
            return _PybindValue(value=add_op.result, semantic_type=expr.type)

        if name == "vsub":
            # pto.vsub (vec0, vec1, mask) -> VReg
            result_type = self._convert_type(expr.type)
            vec0_val = args[0].value
            vec1_val = args[1].value
            mask_val = args[2].value
            sub_op = _create_operation(
                "pto.vsub",
                results=[result_type],
                operands=[vec0_val, vec1_val, mask_val],
            )
            return _PybindValue(value=sub_op.result, semantic_type=expr.type)

        if name == "vmul":
            # pto.vmul (vec0, vec1, mask) -> VReg
            result_type = self._convert_type(expr.type)
            vec0_val = args[0].value
            vec1_val = args[1].value
            mask_val = args[2].value
            mul_op = _create_operation(
                "pto.vmul",
                results=[result_type],
                operands=[vec0_val, vec1_val, mask_val],
            )
            return _PybindValue(value=mul_op.result, semantic_type=expr.type)

        if name == "vdiv":
            # pto.vdiv (vec0, vec1, mask) -> VReg
            result_type = self._convert_type(expr.type)
            vec0_val = args[0].value
            vec1_val = args[1].value
            mask_val = args[2].value
            div_op = _create_operation(
                "pto.vdiv",
                results=[result_type],
                operands=[vec0_val, vec1_val, mask_val],
            )
            return _PybindValue(value=div_op.result, semantic_type=expr.type)

        if name == "vrelu":
            # pto.vrelu (vec, mask) -> VReg (ReLU activation)
            result_type = self._convert_type(expr.type)
            vec_val = args[0].value
            mask_val = args[1].value if len(args) > 1 else self._create_all_true_mask()
            relu_op = _create_operation(
                "pto.vrelu",
                results=[result_type],
                operands=[vec_val, mask_val],
            )
            return _PybindValue(value=relu_op.result, semantic_type=expr.type)

        if name == "vexp":
            # pto.vexp (vec, mask) -> VReg (exponential)
            result_type = self._convert_type(expr.type)
            vec_val = args[0].value
            mask_val = args[1].value if len(args) > 1 else self._create_all_true_mask()
            exp_op = _create_operation(
                "pto.vexp",
                results=[result_type],
                operands=[vec_val, mask_val],
            )
            return _PybindValue(value=exp_op.result, semantic_type=expr.type)

        if name == "vsqrt":
            # pto.vsqrt (vec, mask) -> VReg (square root)
            result_type = self._convert_type(expr.type)
            vec_val = args[0].value
            mask_val = args[1].value if len(args) > 1 else self._create_all_true_mask()
            sqrt_op = _create_operation(
                "pto.vsqrt",
                results=[result_type],
                operands=[vec_val, mask_val],
            )
            return _PybindValue(value=sqrt_op.result, semantic_type=expr.type)

        if name == "vmax":
            # pto.vmax (vec0, vec1, mask) -> VReg (element-wise max)
            result_type = self._convert_type(expr.type)
            vec0_val = args[0].value
            vec1_val = args[1].value
            mask_val = args[2].value if len(args) > 2 else self._create_all_true_mask()
            max_op = _create_operation(
                "pto.vmax",
                results=[result_type],
                operands=[vec0_val, vec1_val, mask_val],
            )
            return _PybindValue(value=max_op.result, semantic_type=expr.type)

        if name == "vmin":
            # pto.vmin (vec0, vec1, mask) -> VReg (element-wise min)
            result_type = self._convert_type(expr.type)
            vec0_val = args[0].value
            vec1_val = args[1].value
            mask_val = args[2].value if len(args) > 2 else self._create_all_true_mask()
            min_op = _create_operation(
                "pto.vmin",
                results=[result_type],
                operands=[vec0_val, vec1_val, mask_val],
            )
            return _PybindValue(value=min_op.result, semantic_type=expr.type)

        # =========================================================================
        # Utility ops
        # =========================================================================

        if name == "castptr":
            # pto.castptr (addr, type) -> PtrType
            result_type = self._convert_type(expr.type)
            addr_val = args[0].value
            cast_op = _create_operation(
                "pto.castptr",
                results=[result_type],
                operands=[addr_val],
            )
            return _PybindValue(value=cast_op.result, semantic_type=expr.type)

        if name == "get_tensor_view_dim":
            # pto.get_tensor_view_dim (tensor_view, dim) -> index
            result_type = _get_mlir_index_type(self._ctx)
            tv_val = args[0].value
            dim_val = args[1].value
            dim_op = _create_operation(
                "pto.get_tensor_view_dim",
                results=[result_type],
                operands=[tv_val, dim_val],
            )
            return _PybindValue(
                value=dim_op.result,
                semantic_type=SemanticIndexType(),
            )

        # Placeholder for other PTO ops
        raise NotImplementedError(f"PTO op '{name}' not yet implemented")

    def _lower_arith_call(
        self, name: str, args: list[_PybindValue], expr: SemanticCallExpr
    ) -> _PybindValue:
        """Lower arith.* call."""
        if name == "constant":
            # args[0] should be the value, type from expr
            mlir_type = self._convert_type(expr.type)
            const_op = _arith_dialect.ConstantOp(mlir_type, args[0].value)
            return _PybindValue(value=const_op.result, semantic_type=expr.type)

        raise NotImplementedError(f"Arith op '{name}' not implemented")

    def _lower_func_call(
        self, name: str, args: list[_PybindValue], expr: SemanticCallExpr
    ) -> _PybindValue:
        """Lower func.* call."""
        raise NotImplementedError(f"Func op '{name}' not implemented")

    def _lower_attribute_access(self, expr: SemanticAttributeAccess) -> _PybindValue:
        """Lower attribute access expression.

        Handles attributes like:
        - TensorView.shape, TensorView.strides
        - Tile.shape, Tile.valid_shape
        - VReg.lanes
        """
        attr = expr.attr
        base = self._lower_expr(expr.base)
        semantic_type = expr.type

        # Handle type-level attributes (shape, strides, valid_shape)
        # These return compile-time known values or runtime dimension ops
        if attr == "shape":
            # Shape access on TensorView/Tile
            if isinstance(expr.type, SemanticShapeType):
                # Shape is a tuple of dimensions
                # For now, handle single dimension access via subscript
                # Full shape tuple handling needs tuple support
                raise NotImplementedError(
                    "Full shape tuple access - use subscript for individual dims"
                )

        if attr == "strides":
            # Strides access on TensorView
            if isinstance(expr.type, SemanticShapeType):
                raise NotImplementedError(
                    "Full strides tuple access - use subscript for individual dims"
                )

        if attr == "valid_shape":
            # Valid shape access on Tile
            if isinstance(expr.type, SemanticShapeType):
                raise NotImplementedError(
                    "Full valid_shape tuple access - use subscript for individual dims"
                )

        if attr == "element_type":
            # Element type is a type-level attribute, compile-time constant
            # Return as a symbol/constant
            if isinstance(expr.type, SemanticScalarType):
                # Return a placeholder for dtype (Phase 3: proper dtype handling)
                return self._create_constant_index(0)

        if attr == "lanes":
            # VReg lanes count
            if isinstance(expr.type, SemanticScalarType):
                lanes_val = getattr(expr.base.type, "lanes", None)
                if lanes_val is not None:
                    return self._create_constant_index(lanes_val)

        if attr == "rank":
            # Rank of tensor/tile
            if isinstance(expr.type, SemanticScalarType):
                rank_val = getattr(expr.base.type, "rank", None)
                if rank_val is not None:
                    return self._create_constant_index(rank_val)

        if attr == "dtype":
            # Data type attribute
            if isinstance(expr.type, SemanticScalarType):
                # Return dtype as string or constant
                dtype_obj = getattr(expr.base.type, "element_dtype", None)
                if dtype_obj is not None:
                    # Return as symbol constant
                    return self._create_constant_index(0)

        # Handle dynamic dimension access via pto.get_tensor_view_dim
        if isinstance(base.semantic_type, SemanticTensorViewType):
            if attr.isdigit() or attr.startswith("dim"):
                # Access specific dimension
                dim_idx = int(attr) if attr.isdigit() else int(attr[3:])
                result_type = _get_mlir_index_type(self._ctx)
                dim_idx_val = self._create_constant_index(dim_idx)
                dim_op = _create_operation(
                    "pto.get_tensor_view_dim",
                    results=[result_type],
                    operands=[base.value, dim_idx_val.value],
                )
                return _PybindValue(
                    value=dim_op.result,
                    semantic_type=SemanticIndexType(),
                )

        raise NotImplementedError(f"Attribute access '{attr}' not implemented")

    def _lower_subscript_access(self, expr: SemanticSubscriptAccess) -> _PybindValue:
        """Lower subscript access expression.

        Handles:
        - TensorView[i] -> get_tensor_view_dim
        - Shape[i] -> dimension value
        - Tile[i] -> offset/subview
        """
        base = self._lower_expr(expr.base)
        index = self._lower_expr(expr.index)
        semantic_type = expr.type

        # TensorView subscript -> get_tensor_view_dim
        if isinstance(base.semantic_type, SemanticTensorViewType):
            result_type = _get_mlir_index_type(self._ctx)
            dim_op = _create_operation(
                "pto.get_tensor_view_dim",
                results=[result_type],
                operands=[base.value, index.value],
            )
            return _PybindValue(
                value=dim_op.result,
                semantic_type=SemanticIndexType(),
            )

        # PartitionTensorView subscript -> dimension access
        if isinstance(base.semantic_type, SemanticPartitionTensorViewType):
            result_type = _get_mlir_index_type(self._ctx)
            dim_op = _create_operation(
                "pto.get_tensor_view_dim",
                results=[result_type],
                operands=[base.value, index.value],
            )
            return _PybindValue(
                value=dim_op.result,
                semantic_type=SemanticIndexType(),
            )

        # Tile subscript -> valid_dim access or subview
        if isinstance(base.semantic_type, SemanticTileType):
            # For Tile, subscript accesses valid_shape dimensions
            # This is typically compile-time known
            shape = getattr(base.semantic_type, "shape", None)
            valid_shape = getattr(base.semantic_type, "valid_shape", shape)
            if valid_shape is not None and isinstance(index.semantic_type, SemanticLiteralExpr):
                dim_idx = index.semantic_type.value
                if dim_idx < len(valid_shape):
                    dim_val = valid_shape[dim_idx]
                    if isinstance(dim_val, int):
                        return self._create_constant_index(dim_val)
            # Dynamic tile dimension access
            result_type = _get_mlir_index_type(self._ctx)
            dim_op = _create_operation(
                "pto.get_tile_dim",
                results=[result_type],
                operands=[base.value, index.value],
            )
            return _PybindValue(
                value=dim_op.result,
                semantic_type=SemanticIndexType(),
            )

        # Pointer subscript -> memory access
        if isinstance(base.semantic_type, SemanticPtrType):
            # ptr[i] -> load from offset
            elem_type = self._convert_type(base.semantic_type.element_type)
            # Compute byte offset: index * element_size
            elem_size = self._get_element_size(base.semantic_type.element_type)
            offset = self._emit_mul(index, self._create_constant_index(elem_size))
            load_op = _create_operation(
                "pto.load",
                results=[elem_type],
                operands=[base.value, offset.value],
            )
            return _PybindValue(
                value=load_op.result,
                semantic_type=base.semantic_type.element_type,
            )

        raise NotImplementedError(
            f"Subscript access not implemented for {type(base.semantic_type).__name__}"
        )

    def _lower_tuple(self, expr: SemanticTupleExpr) -> _PybindValue:
        """Lower tuple expression.

        Tuples are used for multi-result operations like:
        - Shape tuples (shape[0], shape[1])
        - Multiple SSA results from ops
        """
        elements = [self._lower_expr(elem) for elem in expr.elements]

        # For tuple expressions, we typically destructure them at use sites
        # rather than creating actual tuple MLIR values

        # If this tuple is used directly, we need to handle it specially
        # Most tuple uses are destructured via assignment

        # Return the first element as a placeholder
        # (Actual tuple handling happens in assignment destructuring)
        if elements:
            return elements[0]

        raise NotImplementedError("Empty tuple expression")

    def _emit_mul(self, lhs: _PybindValue, rhs: _PybindValue) -> _PybindValue:
        """Emit multiplication operation."""
        # Ensure both are index type
        lhs_index = self._ensure_index(lhs)
        rhs_index = self._ensure_index(rhs)

        mul_op = _arith_dialect.MulIOp(lhs_index.value, rhs_index.value)
        return _PybindValue(
            value=mul_op.result,
            semantic_type=SemanticIndexType(),
        )

    def _emit_add(self, lhs: _PybindValue, rhs: _PybindValue) -> _PybindValue:
        """Emit addition operation."""
        lhs_index = self._ensure_index(lhs)
        rhs_index = self._ensure_index(rhs)

        add_op = _arith_dialect.AddIOp(lhs_index.value, rhs_index.value)
        return _PybindValue(
            value=add_op.result,
            semantic_type=SemanticIndexType(),
        )

    def _get_element_size(self, semantic_type: SemanticType) -> int:
        """Get byte size of element type."""
        if isinstance(semantic_type, SemanticScalarType):
            dtype = semantic_type.dtype
            name = dtype.name
            sizes = {
                "i1": 1, "i8": 1, "i16": 2, "i32": 4, "i64": 8,
                "si64": 8, "ui64": 8,
                "f16": 2, "bf16": 2, "f32": 4,
                "hif8": 1, "f8e4m3": 1, "f8e5m2": 1,
                "f4e1m2x2": 1, "f4e2m1x2": 1,
            }
            return sizes.get(name, 4)
        return 4  # Default size

    def _ensure_index(self, value: _PybindValue) -> _PybindValue:
        """Ensure value is of index type."""
        if isinstance(value.semantic_type, SemanticIndexType):
            return value

        # Convert to index type
        mlir_type = IndexType.get(self._ctx)
        converted = _arith_dialect.IndexCastOp(mlir_type, value.value).result
        return _PybindValue(value=converted, semantic_type=SemanticIndexType())

    def _ensure_i1(self, value: _PybindValue) -> _PybindValue:
        """Ensure value is of i1 type."""
        if isinstance(value.semantic_type, SemanticScalarType):
            if value.semantic_type.dtype.name == "i1":
                return value

        # Convert to i1 (comparison result, etc.)
        mlir_type = _mlir_ir.IntegerType.get_signless(1, self._ctx)
        converted = _arith_dialect.TruncIOp(mlir_type, value.value).result
        return _PybindValue(
            value=converted,
            semantic_type=SemanticScalarType(dtype=ScalarType("i1")),
        )

    # =========================================================================
    # DMA and Vector Store operations (Phase 3.4+)
    # =========================================================================

    def _render_dma_load(self, stmt: SemanticDmaLoadStmt) -> None:
        """Render DMA load operation (copy_gm_to_ubuf) with precise stride calculation.

        This operation transfers data from Global Memory (GM) to Unified Buffer (UB).
        It involves:
        1. Materializing source TensorSlice pointer with offset
        2. Computing DMA transfer configuration (strides, burst lengths)
        3. Setting DMA loop parameters
        4. Executing the copy operation
        """
        if _pto_dialect is None:
            raise NotImplementedError("PTO dialect not available")

        # Lower source TensorSlice and destination TileBuf
        src_base = self._lower_expr(stmt.src.base)
        dst = self._lower_expr(stmt.dst)

        # Materialize source pointer with slice offset
        src_ptr, src_ptr_type = self._materialize_tensor_slice_ptr(stmt.src, src_base)

        # Materialize destination pointer (TileBuf)
        dst_ptr, dst_ptr_type = self._materialize_tile_window_ptr(dst)

        # Compute DMA transfer configuration
        transfer_config = self._infer_dma_load_transfer(stmt.src, stmt.dst, src_base)

        # Issue DMA configuration ops
        # pto.set_loop_size_outtoub (n_burst, 1)
        _create_operation(
            "pto.set_loop_size_outtoub",
            operands=[transfer_config.n_burst.value, self._create_constant_i64(1).value],
        )

        # pto.set_loop1_stride_outtoub (loop_src_stride, loop_dst_stride)
        _create_operation(
            "pto.set_loop1_stride_outtoub",
            operands=[
                transfer_config.loop_src_stride.value,
                transfer_config.loop_dst_stride.value,
            ],
        )

        # pto.set_loop2_stride_outtoub (loop_src_stride, loop_dst_stride)
        _create_operation(
            "pto.set_loop2_stride_outtoub",
            operands=[
                transfer_config.loop_src_stride.value,
                transfer_config.loop_dst_stride.value,
            ],
        )

        # pto.copy_gm_to_ubuf (src, dst, offset, n_burst, len_burst, ...)
        c0 = self._create_constant_i64(0)
        _create_operation(
            "pto.copy_gm_to_ubuf",
            operands=[
                src_ptr.value,
                dst_ptr.value,
                c0.value,  # offset
                transfer_config.n_burst.value,
                transfer_config.len_burst.value,
                c0.value,  # loop offset
                transfer_config.copy_dst_stride.value,
                transfer_config.copy_src_stride.value,
            ],
        )

    def _render_dma_store(self, stmt: SemanticDmaStoreStmt) -> None:
        """Render DMA store operation (copy_ubuf_to_gm) with precise stride calculation.

        This operation transfers data from Unified Buffer (UB) to Global Memory (GM).
        """
        if _pto_dialect is None:
            raise NotImplementedError("PTO dialect not available")

        # Lower source TileBuf and destination TensorSlice
        src = self._lower_expr(stmt.src)
        dst_base = self._lower_expr(stmt.dst.base)

        # Materialize source pointer (TileBuf)
        src_ptr, src_ptr_type = self._materialize_tile_window_ptr(src)

        # Materialize destination pointer with slice offset
        dst_ptr, dst_ptr_type = self._materialize_tensor_slice_ptr(stmt.dst, dst_base)

        # Compute DMA transfer configuration
        transfer_config = self._infer_dma_store_transfer(stmt.dst, stmt.src, dst_base)

        # Issue DMA configuration ops
        c1 = self._create_constant_i64(1)

        # pto.set_loop_size_ubtoout
        _create_operation(
            "pto.set_loop_size_ubtoout",
            operands=[c1.value, c1.value],
        )

        # pto.set_loop1_stride_ubtoout
        _create_operation(
            "pto.set_loop1_stride_ubtoout",
            operands=[
                transfer_config.loop_src_stride.value,
                transfer_config.loop_dst_stride.value,
            ],
        )

        # pto.set_loop2_stride_ubtoout
        _create_operation(
            "pto.set_loop2_stride_ubtoout",
            operands=[
                transfer_config.loop_src_stride.value,
                transfer_config.loop_dst_stride.value,
            ],
        )

        # pto.copy_ubuf_to_gm
        c0 = self._create_constant_i64(0)
        _create_operation(
            "pto.copy_ubuf_to_gm",
            operands=[
                src_ptr.value,
                dst_ptr.value,
                c0.value,
                transfer_config.n_burst.value,
                transfer_config.len_burst.value,
                c0.value,
                transfer_config.copy_dst_stride.value,
                transfer_config.copy_src_stride.value,
            ],
        )

    # =========================================================================
    # DMA Helper methods for stride/offset calculation (Phase 3.11)
    # =========================================================================

    def _materialize_tensor_slice_ptr(
        self,
        slice_expr: SemanticTensorSliceExpr,
        tensor_base: _PybindValue,
    ) -> tuple[_PybindValue, _mlir_ir.Type]:
        """Materialize TensorSlice pointer with offset adjustment.

        If slice starts at non-zero offset, compute the adjusted pointer.
        """
        base_ptr, base_ptr_type = self._materialize_copy_buffer_ptr(tensor_base)

        # Check if slice starts at origin (0, 0)
        if self._is_zero_index_expr(slice_expr.slices[0].start) and \
           self._is_zero_index_expr(slice_expr.slices[1].start):
            return base_ptr, base_ptr_type

        # Need offset adjustment
        # Convert to byte pointer: pto.castptr base_ptr -> !pto.ptr<i8, gm>
        byte_ptr_type = self._get_byte_ptr_type()
        byte_ptr = self._emit_castptr(base_ptr, byte_ptr_type)

        # Compute offset in bytes
        offset_bytes = self._materialize_tensor_slice_offset_bytes(slice_expr, tensor_base)

        # Add offset: pto.addptr byte_ptr, offset_bytes
        offset_ptr = self._emit_addptr(byte_ptr, offset_bytes)

        # Cast back to typed pointer
        typed_ptr = self._emit_castptr(offset_ptr, base_ptr_type)

        return typed_ptr, base_ptr_type

    def _materialize_tile_window_ptr(
        self,
        tile_value: _PybindValue,
        col_offset: int = 0,
        row_offset: int = 0,
    ) -> tuple[_PybindValue, _mlir_ir.Type]:
        """Materialize Tile window pointer with optional offset adjustment.

        For TileBuf with rank-2 shape, we need to adjust for column/row offsets.
        This is commonly needed for padded tiles or tile windows within tiles.

        Args:
            tile_value: The TileBuf value
            col_offset: Column offset in elements (horizontal shift)
            row_offset: Row offset in elements (vertical shift)

        Returns:
            Tuple of (adjusted tile pointer, pointer type)
        """
        tile_type = tile_value.semantic_type
        ptr_type = self._convert_type(tile_type)

        # If no offset, return the tile value directly
        if col_offset == 0 and row_offset == 0:
            return tile_value, ptr_type

        # Get tile shape for stride computation
        shape = getattr(tile_type, "shape", None)
        if shape is None or len(shape) != 2:
            # For non-rank-2 or dynamic tiles, use runtime computation
            # pto.tile_subview operation
            row_offset_val = self._create_constant_i64(row_offset)
            col_offset_val = self._create_constant_i64(col_offset)

            subview_op = _create_operation(
                "pto.tile_subview",
                results=[ptr_type],
                operands=[tile_value.value, row_offset_val.value, col_offset_val.value],
            )
            return _PybindValue(
                value=subview_op.result,
                semantic_type=tile_type,
            ), ptr_type

        # For static rank-2 tiles, compute stride-adjusted pointer
        # Row stride = cols * element_bytes
        element_bytes = self._get_element_bytes_from_tile(tile_type)
        row_stride_bytes = shape[1] * element_bytes

        # Compute byte offset: row_offset * row_stride + col_offset * element_bytes
        total_offset_bytes = row_offset * row_stride_bytes + col_offset * element_bytes

        if total_offset_bytes == 0:
            return tile_value, ptr_type

        # Cast to byte pointer for offset arithmetic
        byte_ptr_type = self._get_byte_ptr_type()
        byte_ptr = self._emit_castptr(tile_value, byte_ptr_type)

        # Add offset
        offset_val = self._create_constant_i64(total_offset_bytes)
        offset_ptr = self._emit_addptr(byte_ptr, offset_val)

        # Cast back to tile pointer type
        adjusted_ptr = self._emit_castptr(offset_ptr, ptr_type)

        return _PybindValue(
            value=adjusted_ptr.value,
            semantic_type=tile_type,
        ), ptr_type

    def _get_element_bytes_from_tile(self, tile_type: SemanticTileType) -> int:
        """Get element byte size from tile type."""
        element_dtype = getattr(tile_type, "element_dtype", None)
        if element_dtype is None:
            # Try other attribute names
            element_dtype = getattr(tile_type, "dtype", None)
        if element_dtype is None:
            return 4  # Default to float32 size
        return self._get_element_bytes(element_dtype)

    def _infer_dma_load_transfer(
        self,
        slice_expr: SemanticTensorSliceExpr,
        dst_tile: _PybindValue,
        tensor_base: _PybindValue,
    ) -> "_DmaTransferConfig":
        """Compute DMA transfer configuration for load.

        Returns: n_burst, len_burst, copy_src_stride, copy_dst_stride,
                 loop_src_stride, loop_dst_stride
        """
        element_bytes = self._get_element_bytes(slice_expr.type.element_dtype)

        # Get row and column counts
        row_count = self._materialize_dma_axis_extent(slice_expr, 0)
        col_count = self._materialize_dma_axis_extent(slice_expr, 1)

        # Compute GM row stride in bytes
        gm_row_stride = self._materialize_tensor_row_stride_bytes(
            slice_expr, tensor_base, element_bytes
        )

        # Get row step
        row_step = self._materialize_dma_row_step(slice_expr)

        # Compute copy stride
        copy_src_stride = self._emit_mul_i64(gm_row_stride, row_step)
        copy_dst_stride = self._materialize_tile_row_stride_bytes(
            dst_tile.semantic_type, element_bytes
        )

        # Compute burst length
        len_burst = self._emit_mul_i64(col_count, self._create_constant_i64(element_bytes))

        # Compute loop strides
        loop_src_stride = self._emit_mul_i64(row_count, copy_src_stride)
        loop_dst_stride = self._emit_mul_i64(row_count, copy_dst_stride)

        return _DmaTransferConfig(
            n_burst=row_count,
            len_burst=len_burst,
            copy_src_stride=copy_src_stride,
            copy_dst_stride=copy_dst_stride,
            loop_src_stride=loop_src_stride,
            loop_dst_stride=loop_dst_stride,
        )

    def _infer_dma_store_transfer(
        self,
        slice_expr: SemanticTensorSliceExpr,
        src_tile: _PybindValue,
        tensor_base: _PybindValue,
    ) -> "_DmaTransferConfig":
        """Compute DMA transfer configuration for store."""
        element_bytes = self._get_element_bytes(slice_expr.type.element_dtype)

        row_count = self._materialize_dma_axis_extent(slice_expr, 0)
        col_count = self._materialize_dma_axis_extent(slice_expr, 1)

        # UB (TileBuf) stride
        copy_src_stride = self._materialize_tile_row_stride_bytes(
            src_tile.semantic_type, element_bytes
        )

        # GM stride
        gm_row_stride = self._materialize_tensor_row_stride_bytes(
            slice_expr, tensor_base, element_bytes
        )

        row_step = self._materialize_dma_row_step(slice_expr)

        copy_dst_stride = self._emit_mul_i64(gm_row_stride, row_step)
        len_burst = self._emit_mul_i64(col_count, self._create_constant_i64(element_bytes))

        loop_src_stride = self._emit_mul_i64(row_count, copy_src_stride)
        loop_dst_stride = self._emit_mul_i64(row_count, copy_dst_stride)

        return _DmaTransferConfig(
            n_burst=row_count,
            len_burst=len_burst,
            copy_src_stride=copy_src_stride,
            copy_dst_stride=copy_dst_stride,
            loop_src_stride=loop_src_stride,
            loop_dst_stride=loop_dst_stride,
        )

    def _materialize_dma_axis_extent(
        self,
        slice_expr: SemanticTensorSliceExpr,
        axis: int,
    ) -> _PybindValue:
        """Get the extent (count) of a DMA axis from slice expression."""
        axis_slice = slice_expr.slices[axis]

        # If extent is statically known
        if axis_slice.extent is not None:
            return self._create_constant_i64(axis_slice.extent)

        # Compute extent from slice range: (stop - start) // step (rounded up)
        start = self._lower_expr(axis_slice.start)
        stop = self._lower_expr(axis_slice.stop)

        distance = self._emit_sub_index(
            self._ensure_index(stop),
            self._ensure_index(start),
        )

        # Handle step value
        step_value = self._get_static_step_value(axis_slice.step)

        if step_value == 1:
            return self._ensure_i64(distance)

        # Ceil division: (distance + step - 1) // step
        numerator = self._emit_add_index(distance, self._create_constant_index(step_value - 1))
        divisor = self._create_constant_index(step_value)

        extent = self._emit_floordiv_index(numerator, divisor)
        return self._ensure_i64(extent)

    def _materialize_dma_row_step(
        self,
        slice_expr: SemanticTensorSliceExpr,
    ) -> _PybindValue:
        """Get the row step value from slice expression."""
        step_expr = slice_expr.slices[0].step
        if step_expr is None:
            return self._create_constant_i64(1)

        step = self._lower_expr(step_expr)
        return self._ensure_i64(step)

    def _materialize_tensor_row_stride_bytes(
        self,
        slice_expr: SemanticTensorSliceExpr,
        tensor_base: _PybindValue,
        element_bytes: int,
    ) -> _PybindValue:
        """Compute tensor row stride in bytes.

        Row stride = stride_elems * element_bytes
        stride_elems = product of dimensions after row axis
        """
        # Get physical axis for row
        physical_axes = getattr(slice_expr.type, "physical_axes", None)
        if physical_axes is None:
            physical_axes = (0, 1)

        row_axis = physical_axes[0]

        # Compute stride in elements
        stride_elems = self._materialize_tensor_axis_stride_elems(tensor_base, row_axis)

        # Convert to bytes
        stride_bytes = self._emit_mul_index(
            stride_elems,
            self._create_constant_index(element_bytes),
        )

        return self._ensure_i64(stride_bytes)

    def _materialize_tensor_axis_stride_elems(
        self,
        tensor_base: _PybindValue,
        axis: int,
    ) -> _PybindValue:
        """Compute stride in elements for a tensor axis.

        Stride[axis] = product(dim[axis+1] * dim[axis+2] * ...)
        """
        stride = self._create_constant_index(1)
        tensor_rank = getattr(tensor_base.semantic_type, "rank", 2)

        for dim_axis in range(axis + 1, tensor_rank):
            dim_value = self._materialize_tensor_dim(tensor_base, dim_axis)
            stride = self._emit_mul_index(stride, dim_value)

        return stride

    def _materialize_tensor_dim(
        self,
        tensor_base: _PybindValue,
        axis: int,
    ) -> _PybindValue:
        """Get tensor dimension value at given axis."""
        result_type = _get_mlir_index_type(self._ctx)
        dim_idx = self._create_constant_index(axis)

        dim_op = _create_operation(
            "pto.get_tensor_view_dim",
            results=[result_type],
            operands=[tensor_base.value, dim_idx.value],
        )

        return _PybindValue(
            value=dim_op.result,
            semantic_type=SemanticIndexType(),
        )

    def _materialize_tile_row_stride_bytes(
        self,
        tile_type: SemanticTileType,
        element_bytes: int,
    ) -> _PybindValue:
        """Compute tile row stride in bytes.

        For rank-2 tile, row_stride = cols * element_bytes
        """
        shape = getattr(tile_type, "shape", None)

        if shape is None or len(shape) != 2:
            # Dynamic shape - need runtime computation
            raise NotImplementedError("DMA requires static rank-2 tile shape")

        row_bytes = shape[1] * element_bytes
        return self._create_constant_i64(row_bytes)

    def _materialize_tensor_slice_offset_bytes(
        self,
        slice_expr: SemanticTensorSliceExpr,
        tensor_base: _PybindValue,
    ) -> _PybindValue:
        """Compute total offset in bytes for tensor slice.

        offset = sum(start[axis] * stride[axis] * element_bytes)
        """
        offset_elems = self._create_constant_index(0)
        physical_axes = getattr(slice_expr.type, "physical_axes", (0, 1))

        for axis_index, slice_axis in enumerate(slice_expr.slices):
            axis_start = self._lower_expr(slice_axis.start)
            axis_start_index = self._ensure_index(axis_start)

            physical_axis = physical_axes[axis_index]
            axis_stride = self._materialize_tensor_axis_stride_elems(tensor_base, physical_axis)

            axis_offset = self._emit_mul_index(axis_start_index, axis_stride)
            offset_elems = self._emit_add_index(offset_elems, axis_offset)

        element_bytes = self._get_element_bytes(slice_expr.type.element_dtype)
        offset_bytes = self._emit_mul_index(
            offset_elems,
            self._create_constant_index(element_bytes),
        )

        return self._ensure_i64(offset_bytes)

    def _render_vector_store(self, stmt: SemanticVectorStoreStmt) -> None:
        """Render vector store operation (pto.vsts)."""
        if _pto_dialect is None:
            raise NotImplementedError("PTO dialect not available")

        # Lower value, destination, indices, and mask
        value = self._lower_expr(stmt.value)
        destination = self._lower_expr(stmt.destination)
        indices = [self._lower_expr(idx) for idx in stmt.indices]
        mask = self._lower_expr(stmt.mask)

        # Create index operands string
        index_values = [idx.value for idx in indices]

        # pto.vsts with optional dist attribute
        attrs = {}
        if stmt.dist is not None:
            attrs["dist"] = _mlir_ir.StringAttr.get(stmt.dist, self._ctx)

        _create_operation(
            "pto.vsts",
            operands=[value.value, destination.value] + index_values + [mask.value],
            attributes=attrs,
        )

    def _render_vector_pair_store(self, stmt: SemanticVectorPairStoreStmt) -> None:
        """Render vector pair store operation (pto.vstps)."""
        if _pto_dialect is None:
            raise NotImplementedError("PTO dialect not available")

        # Lower low, high, destination, indices, and mask
        low = self._lower_expr(stmt.low)
        high = self._lower_expr(stmt.high)
        destination = self._lower_expr(stmt.destination)
        indices = [self._lower_expr(idx) for idx in stmt.indices]
        mask = self._lower_expr(stmt.mask)

        index_values = [idx.value for idx in indices]

        _create_operation(
            "pto.vstps",
            operands=[low.value, high.value, destination.value] + index_values + [mask.value],
        )

    def _render_scalar_store(self, stmt: SemanticScalarStoreStmt) -> None:
        """Render scalar store operation."""
        # Lower ptr, offset, and value
        ptr = self._lower_expr(stmt.ptr)
        offset = self._lower_expr(stmt.offset)
        value = self._lower_expr(stmt.value)

        _create_operation(
            "pto.store_scalar",
            operands=[ptr.value, offset.value, value.value],
        )

    def _render_predicate_store(self, stmt: SemanticPredicateStoreStmt) -> None:
        """Render predicate store operation (psti/pstb).

        These ops store vector data with predicate/broadcast patterns.
        Format: pto.psti/pstb value, destination[offset], dist_pattern
        """
        value = self._lower_expr(stmt.value)
        destination = self._lower_expr(stmt.destination)

        # Handle tile destination specially
        if isinstance(destination.semantic_type, SemanticTileType):
            # Tile memory reference handling
            pass

        # Get offset from indices
        if stmt.indices:
            offset = self._lower_expr(stmt.indices[0])
        else:
            offset = self._create_constant_index(0)

        # Get dist pattern string
        dist_str = stmt.dist if stmt.dist else ""

        op_name = stmt.op_name if hasattr(stmt, "op_name") else "psti"

        _create_operation(
            f"pto.{op_name}",
            operands=[value.value, destination.value, offset.value],
            attributes={"dist": _mlir_ir.StringAttr.get(dist_str, self._ctx)},
        )

    def _render_align_store(self, stmt: SemanticAlignStoreStmt) -> None:
        """Render align store operation (vstas/vstar).

        These ops store vector data with alignment patterns.
        Format: pto.vstas value, destination, offset
        Format: pto.vstar value, destination
        """
        value = self._lower_expr(stmt.value)
        destination = self._lower_expr(stmt.destination)

        op_name = stmt.op_name if hasattr(stmt, "op_name") else "vstas"

        if op_name == "vstar":
            # vstar has no offset operand
            _create_operation(
                "pto.vstar",
                operands=[value.value, destination.value],
            )
        else:
            # vstas requires offset
            if stmt.offset is None:
                offset = self._create_constant_i32(0)
            else:
                offset = self._lower_expr(stmt.offset)
                # Ensure i32 type
                offset = self._ensure_i32(offset)

            _create_operation(
                "pto.vstas",
                operands=[value.value, destination.value, offset.value],
            )

    def _render_get_buf(self, stmt: SemanticGetBufStmt) -> None:
        """Render pto.get_buf operation."""
        if _pto_dialect is None:
            raise NotImplementedError("PTO dialect not available")

        pipe = stmt.pipe
        buf_id = self._lower_expr(stmt.buf_id)
        mode = self._lower_expr(stmt.mode)

        # Create sync_op_type and buf_id attributes
        sync_attr = _pto_dialect.SyncOpTypeAttr.get(
            _pto_dialect.SyncOpType[pipe.upper()], self._ctx
        )

        _create_operation(
            "pto.get_buf",
            attributes={"op_type": sync_attr},
            operands=[buf_id.value, mode.value],
        )

    def _render_rls_buf(self, stmt: SemanticRlsBufStmt) -> None:
        """Render pto.rls_buf operation."""
        if _pto_dialect is None:
            raise NotImplementedError("PTO dialect not available")

        pipe = stmt.pipe
        buf_id = self._lower_expr(stmt.buf_id)
        mode = self._lower_expr(stmt.mode)

        sync_attr = _pto_dialect.SyncOpTypeAttr.get(
            _pto_dialect.SyncOpType[pipe.upper()], self._ctx
        )

        _create_operation(
            "pto.rls_buf",
            attributes={"op_type": sync_attr},
            operands=[buf_id.value, mode.value],
        )

    def _render_mem_bar(self, stmt: SemanticMemBarStmt) -> None:
        """Render pto.mem_bar operation."""
        barrier_type = stmt.barrier_type

        _create_operation(
            "pto.mem_bar",
            attributes={"barrier_type": _mlir_ir.StringAttr.get(barrier_type, self._ctx)},
        )

    def _render_set_cross_core(self, stmt: SemanticSetCrossCoreStmt) -> None:
        """Render pto.set_cross_core operation.

        Format: pto.set_cross_core core_id, event_id : i64, i64
        """
        core_id = self._lower_expr(stmt.core_id)
        event_id = self._lower_expr(stmt.event_id)

        # Ensure i64 type
        core_id_i64 = self._ensure_i64(core_id)
        event_id_i64 = self._ensure_i64(event_id)

        _create_operation(
            "pto.set_cross_core",
            operands=[core_id_i64.value, event_id_i64.value],
        )

    def _render_set_intra_block(self, stmt: SemanticSetIntraBlockStmt) -> None:
        """Render pto.set_intra_block operation.

        Format: pto.set_intra_block block_id, event_id : i64, i64
        """
        block_id = self._lower_expr(stmt.block_id)
        event_id = self._lower_expr(stmt.event_id)

        block_id_i64 = self._ensure_i64(block_id)
        event_id_i64 = self._ensure_i64(event_id)

        _create_operation(
            "pto.set_intra_block",
            operands=[block_id_i64.value, event_id_i64.value],
        )

    def _render_set_intra_core(self, stmt: SemanticSetIntraCoreStmt) -> None:
        """Render pto.set_intra_core operation.

        Format: pto.set_intra_core config : i32
        """
        config = self._lower_expr(stmt.config)
        config_i32 = self._ensure_i32(config)

        _create_operation(
            "pto.set_intra_core",
            operands=[config_i32.value],
        )

    def _render_wait_flag_dev(self, stmt: SemanticWaitFlagDevStmt) -> None:
        """Render pto.wait_flag_dev operation.

        Format: pto.wait_flag_dev core_id, event_id : i64, i64
        """
        core_id = self._lower_expr(stmt.core_id)
        event_id = self._lower_expr(stmt.event_id)

        core_id_i64 = self._ensure_i64(core_id)
        event_id_i64 = self._ensure_i64(event_id)

        _create_operation(
            "pto.wait_flag_dev",
            operands=[core_id_i64.value, event_id_i64.value],
        )

    def _render_wait_intra_core(self, stmt: SemanticWaitIntraCoreStmt) -> None:
        """Render pto.wait_intra_core operation.

        Format: pto.wait_intra_core block_id, event_id : i64, i64
        """
        block_id = self._lower_expr(stmt.block_id)
        event_id = self._lower_expr(stmt.event_id)

        block_id_i64 = self._ensure_i64(block_id)
        event_id_i64 = self._ensure_i64(event_id)

        _create_operation(
            "pto.wait_intra_core",
            operands=[block_id_i64.value, event_id_i64.value],
        )

    def _ensure_i64(self, value: _PybindValue) -> _PybindValue:
        """Ensure value is of i64 type."""
        if isinstance(value.semantic_type, SemanticScalarType):
            if value.semantic_type.dtype.name == "i64":
                return value

        mlir_type = _mlir_ir.IntegerType.get_signless(64, self._ctx)
        if isinstance(value.semantic_type, SemanticIndexType):
            converted = _arith_dialect.IndexCastOp(mlir_type, value.value).result
        else:
            converted = _arith_dialect.ExtSIOp(mlir_type, value.value).result
        return _PybindValue(
            value=converted,
            semantic_type=SemanticScalarType(dtype=ScalarType("i64")),
        )

    def _render_dma_config(self, stmt: SemanticDmaConfigStmt) -> None:
        """Render DMA config operation (set_loop_size, etc.)."""
        first = self._lower_expr(stmt.first)
        second = self._lower_expr(stmt.second)

        _create_operation(
            f"pto.{stmt.name}",
            operands=[first.value, second.value],
        )

    def _render_dma_unary_config(self, stmt: SemanticDmaUnaryConfigStmt) -> None:
        """Render DMA unary config operation."""
        value = self._lower_expr(stmt.value)

        _create_operation(
            f"pto.{stmt.name}",
            operands=[value.value],
        )

    def _render_low_level_copy(self, stmt: SemanticLowLevelCopyStmt) -> None:
        """Render low-level copy operation."""
        source = self._lower_expr(stmt.source)
        destination = self._lower_expr(stmt.destination)
        operands = [self._lower_expr(op) for op in stmt.operands]

        _create_operation(
            f"pto.{stmt.name}",
            operands=[source.value, destination.value] + [op.value for op in operands],
        )

    # =========================================================================
    # Helper methods
    # =========================================================================

    def _create_constant_i64(self, value: int) -> _PybindValue:
        """Create an i64 constant."""
        mlir_type = _mlir_ir.IntegerType.get_signless(64, self._ctx)
        const_op = _arith_dialect.ConstantOp(mlir_type, value)
        return _PybindValue(
            value=const_op.result,
            semantic_type=SemanticScalarType(dtype=ScalarType("i64")),
        )

    def _create_constant_i32(self, value: int) -> _PybindValue:
        """Create an i32 constant."""
        mlir_type = _mlir_ir.IntegerType.get_signless(32, self._ctx)
        const_op = _arith_dialect.ConstantOp(mlir_type, value)
        return _PybindValue(
            value=const_op.result,
            semantic_type=SemanticScalarType(dtype=ScalarType("i32")),
        )

    def _create_constant_index(self, value: int) -> _PybindValue:
        """Create an index constant."""
        const_op = _arith_dialect.ConstantOp(_get_mlir_index_type(self._ctx), value)
        return _PybindValue(
            value=const_op.result,
            semantic_type=SemanticIndexType(),
        )

    def _ensure_i32(self, value: _PybindValue) -> _PybindValue:
        """Ensure value is of i32 type."""
        if isinstance(value.semantic_type, SemanticScalarType):
            if value.semantic_type.dtype.name == "i32":
                return value

        # Convert to i32
        mlir_type = _mlir_ir.IntegerType.get_signless(32, self._ctx)
        if isinstance(value.semantic_type, SemanticIndexType):
            converted = _arith_dialect.IndexCastOp(mlir_type, value.value).result
        else:
            converted = _arith_dialect.TruncIOp(mlir_type, value.value).result
        return _PybindValue(
            value=converted,
            semantic_type=SemanticScalarType(dtype=ScalarType("i32")),
        )

    # =========================================================================
    # Additional helper methods for DMA stride/offset calculation (Phase 3.11)
    # =========================================================================

    def _emit_mul_i64(self, lhs: _PybindValue, rhs: _PybindValue) -> _PybindValue:
        """Emit i64 multiplication."""
        lhs_i64 = self._ensure_i64(lhs)
        rhs_i64 = self._ensure_i64(rhs)

        mul_op = _arith_dialect.MulIOp(lhs_i64.value, rhs_i64.value)
        return _PybindValue(
            value=mul_op.result,
            semantic_type=SemanticScalarType(dtype=ScalarType("i64")),
        )

    def _emit_castptr(self, ptr: _PybindValue, target_type: _mlir_ir.Type) -> _PybindValue:
        """Emit pto.castptr operation."""
        cast_op = _create_operation(
            "pto.castptr",
            results=[target_type],
            operands=[ptr.value],
        )
        # The semantic type changes - use a placeholder PtrType
        return _PybindValue(
            value=cast_op.result,
            semantic_type=SemanticPtrType(element_type=SemanticScalarType(dtype=ScalarType("i8"))),
        )

    def _emit_addptr(self, ptr: _PybindValue, offset: _PybindValue) -> _PybindValue:
        """Emit pto.addptr operation to add offset to pointer."""
        # Result type is the same as ptr type
        ptr_type = ptr.value.type
        add_op = _create_operation(
            "pto.addptr",
            results=[ptr_type],
            operands=[ptr.value, offset.value],
        )
        return _PybindValue(
            value=add_op.result,
            semantic_type=ptr.semantic_type,
        )

    def _emit_mul_index(self, lhs: _PybindValue, rhs: _PybindValue) -> _PybindValue:
        """Emit index multiplication."""
        lhs_index = self._ensure_index(lhs)
        rhs_index = self._ensure_index(rhs)

        mul_op = _arith_dialect.MulIOp(lhs_index.value, rhs_index.value)
        return _PybindValue(
            value=mul_op.result,
            semantic_type=SemanticIndexType(),
        )

    def _emit_add_index(self, lhs: _PybindValue, rhs: _PybindValue) -> _PybindValue:
        """Emit index addition."""
        lhs_index = self._ensure_index(lhs)
        rhs_index = self._ensure_index(rhs)

        add_op = _arith_dialect.AddIOp(lhs_index.value, rhs_index.value)
        return _PybindValue(
            value=add_op.result,
            semantic_type=SemanticIndexType(),
        )

    def _emit_sub_index(self, lhs: _PybindValue, rhs: _PybindValue) -> _PybindValue:
        """Emit index subtraction."""
        lhs_index = self._ensure_index(lhs)
        rhs_index = self._ensure_index(rhs)

        sub_op = _arith_dialect.SubIOp(lhs_index.value, rhs_index.value)
        return _PybindValue(
            value=sub_op.result,
            semantic_type=SemanticIndexType(),
        )

    def _emit_floordiv_index(self, lhs: _PybindValue, rhs: _PybindValue) -> _PybindValue:
        """Emit index floor division."""
        lhs_index = self._ensure_index(lhs)
        rhs_index = self._ensure_index(rhs)

        div_op = _arith_dialect.FloorDivSIOp(lhs_index.value, rhs_index.value)
        return _PybindValue(
            value=div_op.result,
            semantic_type=SemanticIndexType(),
        )

    def _get_byte_ptr_type(self) -> Any:
        """Get !pto.ptr<i8, gm> type for byte pointer operations."""
        if _pto_dialect is None:
            # Fallback: use opaque type
            return _mlir_ir.OpaqueType.get(self._ctx, "pto", "ptr")
        i8_type = _mlir_ir.IntegerType.get_signless(8, self._ctx)
        return _pto_dialect.PtrType.get(i8_type, self._ctx)

    def _materialize_copy_buffer_ptr(
        self,
        buffer_value: _PybindValue,
    ) -> tuple[_PybindValue, _mlir_ir.Type]:
        """Materialize buffer pointer for DMA copy operations.

        For TensorView/TensorSlice, get the underlying pointer.
        For TileBuf, return the tile buffer directly.
        """
        semantic_type = buffer_value.semantic_type

        if isinstance(semantic_type, SemanticTensorViewType):
            # TensorView has an underlying pointer attribute
            # pto.get_tensor_view_ptr op
            ptr_type = self._get_byte_ptr_type()
            ptr_op = _create_operation(
                "pto.get_tensor_view_ptr",
                results=[ptr_type],
                operands=[buffer_value.value],
            )
            return _PybindValue(
                value=ptr_op.result,
                semantic_type=SemanticPtrType(element_type=SemanticScalarType(dtype=ScalarType("i8"))),
            ), ptr_type

        if isinstance(semantic_type, SemanticTileType):
            # TileBuf is already a buffer reference
            ptr_type = self._convert_type(semantic_type)
            return buffer_value, ptr_type

        # Generic pointer type
        ptr_type = self._convert_type(semantic_type)
        return buffer_value, ptr_type

    def _is_zero_index_expr(self, expr: SemanticExpr) -> bool:
        """Check if expression is a literal zero index."""
        if isinstance(expr, SemanticLiteralExpr):
            return expr.value == 0
        return False

    def _get_static_step_value(self, step_expr: SemanticExpr | None) -> int:
        """Get static step value from step expression, default 1."""
        if step_expr is None:
            return 1
        if isinstance(step_expr, SemanticLiteralExpr):
            return step_expr.value
        # Dynamic step - assume 1 for now
        return 1

    def _get_element_bytes(self, dtype: ScalarType) -> int:
        """Get byte size for element dtype."""
        name = dtype.name
        sizes = {
            "i1": 1, "i8": 1, "i16": 2, "i32": 4, "i64": 8,
            "f16": 2, "bf16": 2, "f32": 4,
            "hif8": 1, "f8e4m3": 1, "f8e5m2": 1,
            "f4e1m2x2": 1, "f4e2m1x2": 1,
        }
        return sizes.get(name, 4)

    def _create_all_true_mask(self) -> _PybindValue:
        """Create an all-true mask for vector operations.

        When mask is not explicitly provided, use all-true mask
        to process all vector lanes.
        """
        # Create a mask constant with all bits set (all true)
        # The mask type depends on the vector length, but for default cases
        # we use a simple all-ones pattern
        # For 128-lane vectors, we need a b128 mask type
        # For simplicity, create a constant i64 with all bits set
        mlir_type = _mlir_ir.IntegerType.get_signless(64, self._ctx)
        const_op = _arith_dialect.ConstantOp(mlir_type, -1)  # All bits set
        return _PybindValue(
            value=const_op.result,
            semantic_type=SemanticMaskType(dtype=ScalarType("b64")),
        )


@dataclass(frozen=True)
class _DmaTransferConfig:
    """Configuration for DMA transfer operations.

    Contains all parameters needed for copy_gm_to_ubuf/copy_ubuf_to_gm:
    - n_burst: number of burst transfers (rows)
    - len_burst: burst length in bytes (cols * element_bytes)
    - copy_src_stride: source stride per burst (row stride * step)
    - copy_dst_stride: destination stride per burst
    - loop_src_stride: source stride per loop iteration
    - loop_dst_stride: destination stride per loop iteration
    """
    n_burst: _PybindValue
    len_burst: _PybindValue
    copy_src_stride: _PybindValue
    copy_dst_stride: _PybindValue
    loop_src_stride: _PybindValue
    loop_dst_stride: _PybindValue


__all__ = ["PybindRenderer"]
