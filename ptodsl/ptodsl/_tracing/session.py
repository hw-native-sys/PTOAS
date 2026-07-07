# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""Trace-session objects shared by PTODSL tracing runtimes."""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
import hashlib

from .._diagnostics import inline_subkernel_value_escape_error, subkernel_kernel_kind_mismatch_error
from .._kernel_signature import RuntimeScalarParameterSpec
from .._ops import const
from .._surface_values import unwrap_surface_value, wrap_like_surface_value
from .control_flow import (
    build_carry_loop_frame,
    finish_carry_loop_frame,
    yield_carry_loop_state,
)
from .._types import _strip_integer_signedness
from .module_builder import create_container_child_module

from mlir.dialects import arith, func
from mlir.dialects import pto as _pto
from mlir.ir import (
    Attribute,
    FlatSymbolRefAttr,
    IndexType,
    InsertionPoint,
    IntegerAttr,
    IntegerType,
    Operation,
    StringAttr,
    TypeAttr,
    UnitAttr,
)


@dataclass(frozen=True)
class HelperFunctionSpec:
    """Declarative description of a helper function emitted during tracing."""

    symbol_name: str
    arg_types: tuple
    result_types: tuple = ()
    attributes: tuple[tuple[str, object], ...] = ()

    def cache_key(self) -> tuple:
        """Return one stable ABI-sensitive cache key for this helper signature."""
        return (
            self.symbol_name,
            tuple(str(arg_type) for arg_type in self.arg_types),
            tuple(str(result_type) for result_type in self.result_types),
            tuple((attr_name, str(attr_value)) for attr_name, attr_value in self.attributes),
        )

    def specialized_symbol_name(self) -> str:
        """Return one stable symbol name that is unique for this helper ABI."""
        digest = hashlib.sha1(repr(self.cache_key()).encode("utf-8")).hexdigest()[:10]
        return f"{self.symbol_name}__ptodsl_{digest}"


@dataclass(frozen=True)
class KernelModuleImportRecord:
    """One private import declaration emitted for a kernel-module callsite."""

    caller_symbol_name: str
    import_symbol_name: str
    target_symbol_name: str


@dataclass(frozen=True)
class KernelModuleGraphSnapshot:
    """Immutable snapshot of traced kernel-module imports and dependencies."""

    imports: tuple[KernelModuleImportRecord, ...] = ()
    dependencies: tuple[tuple[str, tuple[str, ...]], ...] = ()


@dataclass(frozen=True)
class TracedChildModuleRecord:
    """Metadata for one child module assembled during tracing."""

    symbol_name: str
    primary_symbol_name: str
    role: str
    module_spec: object


@dataclass(frozen=True)
class SimtHelperSpecializationKey:
    """Cache key for one specialized ``@pto.simt`` helper body."""

    owner_symbol_name: str
    symbol_name: str
    arg_types: tuple
    static_kwargs: tuple[tuple[str, object], ...]


@dataclass(frozen=True)
class SubkernelTraceFrame:
    """Active lowering frame for one PTODSL subkernel call or inline scope."""

    role: str
    symbol_name: str
    target: str


@dataclass(frozen=True)
class InlineSubkernelOutlineFrame:
    """Tracing-time placeholder for one outlined inline subkernel scope."""

    trace_frame: SubkernelTraceFrame
    helper_symbol_name: str
    owner_symbol_name: str
    wrapper_op: object
    body_block: object


class TraceSession:
    """Shared per-build state for a traced PTODSL module."""

    def __init__(self, module_spec, module, entry_function):
        self.module_spec = module_spec
        self.module = module
        self.entry_function = entry_function
        self.entry_block = None
        self._function_stack = [entry_function]
        self._function_owner_symbol_stack = [entry_function.name.value]
        self._entry_child_op = entry_function.operation.parent.parent
        self._entry_child_symbol_table = entry_function.operation.parent.regions[0].blocks[0]
        self._helpers: dict[tuple[str, tuple], object] = {}
        self._simt_helper_specializations: dict[SimtHelperSpecializationKey, object] = {}
        self._simt_helper_symbol_counters: dict[str, int] = {}
        self._kernel_module_primary_functions: dict[tuple, object] = {}
        self._kernel_module_private_imports: dict[tuple[str, str], object] = {}
        self._kernel_module_dependencies: dict[str, set[str]] = {}
        self._kernel_module_child_symbol_tables: dict[str, object] = {
            self.current_function_symbol_name: self._entry_child_symbol_table,
        }
        self._kernel_module_child_records: dict[str, TracedChildModuleRecord] = {
            self.current_function_symbol_name: TracedChildModuleRecord(
                symbol_name=f"{self.current_function_symbol_name}$child",
                primary_symbol_name=self.current_function_symbol_name,
                role="entry" if module_spec.entry else "kernel_module",
                module_spec=module_spec,
            )
        }
        self._subkernel_stack: list[SubkernelTraceFrame] = []
        self._carry_loop_stack = []
        self._inline_subkernel_counter = 0
        self._escaped_inline_values: dict[object, tuple[str, str]] = {}
        self._helper_result_templates: dict[tuple[str, tuple], object] = {}

    @property
    def current_function(self):
        return self._function_stack[-1]

    @property
    def current_function_symbol_name(self):
        return self.current_function.name.value

    @property
    def current_function_owner_symbol_name(self):
        return self._function_owner_symbol_stack[-1]

    @property
    def current_function_module_spec(self):
        """Return the module spec that owns the actively lowered function."""
        current_record = self._kernel_module_child_records.get(self.current_function_owner_symbol_name)
        if current_record is not None:
            return current_record.module_spec
        return self.module_spec

    @property
    def _function_symbol_table(self):
        """Compatibility alias for fixtures that inject peer declarations."""
        owner_symbol_name = self.current_function_owner_symbol_name
        return self.get_or_create_kernel_module_child_symbol_table(
            owner_symbol_name,
            self._kernel_module_child_records[owner_symbol_name].module_spec,
        )

    @property
    def current_subkernel(self):
        if not self._subkernel_stack:
            return None
        return self._subkernel_stack[-1]

    @property
    def subkernel_stack_depth(self):
        return len(self._subkernel_stack)

    @property
    def current_carry_loop(self):
        if not self._carry_loop_stack:
            return None
        return self._carry_loop_stack[-1]

    def bind_entry_block(self, entry_block) -> None:
        """Record the root entry block for the active trace."""
        self.entry_block = entry_block

    def validate_surface_value_access(self, value) -> None:
        """Reject inline-subkernel SSA values that escaped their outlined helper body."""
        record = self._escaped_inline_values.get(value)
        if record is None:
            return
        role, type_text = record
        raise inline_subkernel_value_escape_error(role, type_text)

    @contextmanager
    def enter_function(self, ir_fn, *, owner_symbol_name: str | None = None):
        """Push *ir_fn* as the current active function in this session."""
        owner_symbol_name = ir_fn.name.value if owner_symbol_name is None else owner_symbol_name
        self._function_stack.append(ir_fn)
        self._function_owner_symbol_stack.append(owner_symbol_name)
        try:
            yield ir_fn
        finally:
            popped_owner = self._function_owner_symbol_stack.pop()
            popped = self._function_stack.pop()
            if popped is not ir_fn or popped_owner != owner_symbol_name:
                raise RuntimeError("PTODSL trace-session function stack corruption detected")

    def _next_inline_subkernel_symbol(self, base_symbol_name: str) -> str:
        suffix = self._inline_subkernel_counter
        self._inline_subkernel_counter += 1
        return f"{base_symbol_name}_{suffix}"

    def _create_subkernel_section_op(self, role: str):
        return None

    def _canonical_helper_role(self, role: str) -> str:
        if role in {"tileop", "simd", "cube"}:
            return "tileop"
        return role

    def _create_inline_subkernel_wrapper(self, role: str):
        wrapper_op = self._create_subkernel_section_op(role)
        if wrapper_op is None:
            wrapper_op = _pto.VecScopeOp()
        body_block = wrapper_op.body.blocks.append()
        return wrapper_op, body_block

    def _subkernel_role_kernel_kind(self, role: str) -> str | None:
        if role == "simd":
            return "vector"
        if role == "cube":
            return "cube"
        return None

    def _current_explicit_kernel_kind(self) -> str | None:
        module_spec = self.current_function_module_spec
        if not getattr(module_spec, "kernel_kind_explicit", False):
            return None
        kind = getattr(module_spec, "kernel_kind", None)
        return kind if kind in {"cube", "vector"} else None

    def _subkernel_section_policy(self, role: str) -> str:
        role_kind = self._subkernel_role_kernel_kind(role)
        explicit_kind = self._current_explicit_kernel_kind()
        if role_kind is None or explicit_kind is None:
            return "section"
        if explicit_kind != role_kind:
            raise subkernel_kernel_kind_mismatch_error(role, explicit_kind)
        return "function_kind"

    def _subkernel_helper_attributes(self, role: str) -> tuple[tuple[str, object], ...]:
        attrs: list[tuple[str, object]] = []
        helper_role = self._canonical_helper_role(role)
        if helper_role == "tileop":
            attrs.append(("pto.tileop.helper", UnitAttr.get()))
        if role == "simt":
            attrs.append(("pto.simt_entry", UnitAttr.get()))
        return tuple(attrs)

    def _erase_attached_op(self, op_view) -> None:
        parent = op_view.operation.parent
        if parent is not None:
            op_view.operation.erase()

    @contextmanager
    def enter_subkernel_body(self, role: str, symbol_name: str, target: str):
        """Push one named/decorated subkernel body frame onto the tracing stack."""
        frame = SubkernelTraceFrame(
            role=role,
            symbol_name=symbol_name,
            target=target,
        )
        self._subkernel_stack.append(frame)
        try:
            if self._subkernel_section_policy(role) == "function_kind":
                yield frame
                return

            section_op = self._create_subkernel_section_op(role)
            if section_op is None:
                yield frame
                return

            block = section_op.body.blocks.append()
            with InsertionPoint(block):
                yield frame
        finally:
            popped = self._subkernel_stack.pop()
            if popped is not frame:
                raise RuntimeError("PTODSL trace-session subkernel stack corruption detected")

    @contextmanager
    def enter_inline_subkernel(self, role: str, symbol_name: str, target: str):
        """Capture one inline subkernel scope and outline it into a helper on exit."""
        frame = SubkernelTraceFrame(
            role=role,
            symbol_name=symbol_name,
            target=target,
        )
        wrapper_op, body_block = self._create_inline_subkernel_wrapper(role)
        outline_frame = InlineSubkernelOutlineFrame(
            trace_frame=frame,
            helper_symbol_name=self._next_inline_subkernel_symbol(symbol_name),
            owner_symbol_name=self.current_function_owner_symbol_name,
            wrapper_op=wrapper_op,
            body_block=body_block,
        )
        self._subkernel_stack.append(frame)
        try:
            with InsertionPoint(body_block):
                yield frame
        except BaseException:
            self._erase_attached_op(wrapper_op)
            raise
        else:
            self._outline_inline_subkernel(outline_frame)
        finally:
            popped = self._subkernel_stack.pop()
            if popped is not frame:
                raise RuntimeError("PTODSL trace-session subkernel stack corruption detected")

    @contextmanager
    def enter_subkernel(self, subkernel):
        """Push *subkernel* as the current active subkernel body frame."""
        with self.enter_subkernel_body(
            subkernel.spec.role.value,
            subkernel.spec.symbol_name,
            subkernel.spec.target,
        ) as frame:
            yield frame

    @contextmanager
    def suspend_subkernel_scope(self):
        """Temporarily clear caller-owned subkernel scope while lowering a new function body."""
        saved_stack = self._subkernel_stack
        self._subkernel_stack = []
        try:
            yield
        finally:
            self._subkernel_stack = saved_stack

    def _walk_op_tree(self, root_ops):
        def visit_op(op_view):
            yield op_view
            for region in op_view.operation.regions:
                for block in region.blocks:
                    yield from visit_block(block)

        def visit_block(block):
            for op_view in block.operations:
                yield from visit_op(op_view)

        for root_op in root_ops:
            yield from visit_op(root_op)

    def _collect_defined_values(self, root_ops):
        defined = set()
        for op_view in self._walk_op_tree(root_ops):
            for region in op_view.operation.regions:
                for block in region.blocks:
                    for arg in block.arguments:
                        defined.add(arg)
            for result in op_view.operation.results:
                defined.add(result)
        return defined

    def _collect_capture_values(self, root_ops):
        captures = []
        seen = set()
        defined = self._collect_defined_values(root_ops)
        for op_view in self._walk_op_tree(root_ops):
            operands = op_view.operation.operands
            for operand_index in range(len(operands)):
                operand = operands[operand_index]
                if operand in defined or operand in seen:
                    continue
                seen.add(operand)
                captures.append(operand)
        return tuple(captures)

    def _note_escaped_inline_values(self, values, *, role: str) -> None:
        for value in values:
            self._escaped_inline_values[value] = (role, str(value.type))

    def _flatten_helper_result_templates(self, value) -> tuple:
        if value is None:
            return ()
        if isinstance(value, tuple):
            flattened = []
            for item in value:
                flattened.extend(self._flatten_helper_result_templates(item))
            return tuple(flattened)
        if isinstance(value, list):
            flattened = []
            for item in value:
                flattened.extend(self._flatten_helper_result_templates(item))
            return tuple(flattened)
        if isinstance(value, dict):
            flattened = []
            for item in value.values():
                flattened.extend(self._flatten_helper_result_templates(item))
            return tuple(flattened)
        return (value,)

    def _wrap_helper_call_results(self, template, results_iter):
        if template is None:
            return None
        if isinstance(template, tuple):
            return tuple(self._wrap_helper_call_results(item, results_iter) for item in template)
        if isinstance(template, list):
            return [self._wrap_helper_call_results(item, results_iter) for item in template]
        if isinstance(template, dict):
            return {name: self._wrap_helper_call_results(item, results_iter) for name, item in template.items()}
        return wrap_like_surface_value(template, next(results_iter))

    def _remap_captured_operands(self, root_ops, capture_mapping) -> None:
        for op_view in self._walk_op_tree(root_ops):
            operands = op_view.operation.operands
            for operand_index in range(len(operands)):
                replacement = capture_mapping.get(operands[operand_index])
                if replacement is not None:
                    operands[operand_index] = replacement

    def _outline_inline_subkernel(self, outline_frame: InlineSubkernelOutlineFrame) -> None:
        role = outline_frame.trace_frame.role
        root_ops = tuple(outline_frame.body_block.operations)

        defined_values = self._collect_defined_values(root_ops)
        captures = self._collect_capture_values(root_ops)
        helper_spec = HelperFunctionSpec(
            symbol_name=outline_frame.helper_symbol_name,
            arg_types=tuple(value.type for value in captures),
            attributes=self._subkernel_helper_attributes(role),
        )
        helper_fn, created = self.get_or_create_helper_function(
            helper_spec,
            owner_symbol_name=outline_frame.owner_symbol_name,
        )
        if not created:
            raise RuntimeError(
                f"duplicate inline subkernel helper symbol {helper_fn.name.value!r} in one trace session"
            )

        with InsertionPoint(outline_frame.wrapper_op.operation):
            if role == "simt":
                self._emit_simt_launch_call(helper_fn, captures, dims=(1, 1, 1))
            else:
                func.CallOp(helper_fn, list(captures))

        entry_block = helper_fn.add_entry_block()
        with InsertionPoint(entry_block):
            terminator = func.ReturnOp([])
        return_anchor = terminator.operation.opview

        body_ops = tuple(outline_frame.body_block.operations)
        for op_view in body_ops:
            op_view.move_before(return_anchor)
        outline_frame.wrapper_op.operation.erase()
        outlined_roots = body_ops

        capture_mapping = dict(zip(captures, entry_block.arguments))
        self._remap_captured_operands(outlined_roots, capture_mapping)
        self._note_escaped_inline_values(defined_values, role=role)

    def lower_helper_subkernel(self, subkernel, *args, **kwargs):
        """Lower one decorated PTODSL subkernel call through a dedicated helper function."""
        if subkernel.spec.role.value == "simt":
            return self.lower_simt_helper_subkernel(subkernel, *args, **kwargs)

        arg_templates = tuple(args)
        arg_types = tuple(unwrap_surface_value(arg).type for arg in arg_templates)
        owner_symbol_name = self.current_function_owner_symbol_name
        result_template = None
        helper_spec = HelperFunctionSpec(
            symbol_name=subkernel.spec.symbol_name,
            arg_types=arg_types,
            attributes=self._subkernel_helper_attributes(subkernel.spec.role.value),
        )
        helper_cache_key = (owner_symbol_name, helper_spec.cache_key())
        helper_fn, created = self.get_or_create_helper_function(
            helper_spec,
            owner_symbol_name=owner_symbol_name,
        )

        if created:
            entry_block = helper_fn.add_entry_block()
            wrapped_args = tuple(
                wrap_like_surface_value(template, value)
                for template, value in zip(arg_templates, entry_block.arguments)
            )
            with (
                self.enter_function(helper_fn, owner_symbol_name=owner_symbol_name),
                self.suspend_subkernel_scope(),
                InsertionPoint(entry_block),
            ):
                with self.enter_subkernel(subkernel):
                    result_template = subkernel.emit_body(*wrapped_args, **kwargs)
                flat_results = self._flatten_helper_result_templates(result_template)
                result_types = [unwrap_surface_value(value).type for value in flat_results]
                helper_fn.operation.attributes["function_type"] = TypeAttr.get(
                    func.FunctionType.get(list(arg_types), result_types)
                )
                func.ReturnOp([unwrap_surface_value(value) for value in flat_results])
            self._helper_result_templates[helper_cache_key] = result_template
        else:
            result_template = self._helper_result_templates.get(helper_cache_key)

        call_op = func.CallOp(helper_fn, [unwrap_surface_value(arg) for arg in arg_templates])
        if result_template is None:
            return None
        return self._wrap_helper_call_results(result_template, iter(call_op.results))

    def begin_carry_loop(self, start, stop, step, state_items):
        """Materialize one authored ``pto.for_(...).carry(...)`` loop body."""
        frame = build_carry_loop_frame(start, stop, step, state_items)
        self._carry_loop_stack.append(frame)
        return frame

    def update_carry_loop(self, frame, **kwargs):
        """Emit the one legal ``loop.update(...)`` for the active carry loop."""
        active = self.current_carry_loop
        if active is None or active is not frame:
            raise RuntimeError("loop.update(...) may only be called inside the active carry loop body")
        yield_carry_loop_state(frame, **kwargs)

    def finish_carry_loop(self, frame, exc_type, exc, tb):
        """Finalize one active authored carry loop and close its body insertion point."""
        if not self._carry_loop_stack:
            raise RuntimeError("carry-loop exit without a matching active PTODSL trace-session frame")
        popped = self._carry_loop_stack.pop()
        if popped is not frame:
            raise RuntimeError("PTODSL trace-session carry-loop stack corruption detected")
        finish_carry_loop_frame(frame, exc_type, exc, tb)

    def lower_simt_helper_subkernel(self, subkernel, *args, **kwargs):
        """Lower one ``@pto.simt`` call through a dedicated helper function."""
        helper_fn, arg_templates = self._get_or_create_simt_helper_function(subkernel, *args, **kwargs)

        self._emit_simt_launch_call(helper_fn, arg_templates, dims=(1, 1, 1))

    def lower_simt_launch_subkernel(self, subkernel, *args, dims, **kwargs):
        """Lower one explicit ``pto.simt_launch`` call through a SIMT helper."""
        helper_fn, arg_templates = self._get_or_create_simt_helper_function(subkernel, *args, **kwargs)
        self._emit_simt_launch_call(helper_fn, arg_templates, dims=dims)

    def _emit_simt_launch_call(self, helper_fn, arg_templates, *, dims) -> None:
        dim_x, dim_y, dim_z = _coerce_simt_launch_dims(dims)
        Operation.create(
            "pto.simt_launch",
            attributes={"callee": FlatSymbolRefAttr.get(_symbol_name(helper_fn))},
            operands=[dim_x, dim_y, dim_z, *[unwrap_surface_value(arg) for arg in arg_templates]],
        )

    def _get_or_create_simt_helper_function(self, subkernel, *args, **kwargs):
        """Return the reusable ``pto.simt_entry`` helper for *subkernel*."""
        outer_frame = self.current_subkernel
        if outer_frame is not None and outer_frame.role == "simt":
            raise RuntimeError("@pto.simt helper lowering does not support nested SIMT helper calls")

        arg_templates = tuple(args)
        arg_types = tuple(unwrap_surface_value(arg).type for arg in arg_templates)
        static_kwargs = _simt_static_kwargs_signature(kwargs)
        owner_symbol_name = self.current_function_owner_symbol_name
        specialization_key = SimtHelperSpecializationKey(
            owner_symbol_name=owner_symbol_name,
            symbol_name=subkernel.spec.symbol_name,
            arg_types=arg_types,
            static_kwargs=static_kwargs,
        )
        helper_fn = self._simt_helper_specializations.get(specialization_key)
        if helper_fn is not None:
            return helper_fn, arg_templates

        helper_symbol = self._next_simt_helper_symbol(subkernel.spec.symbol_name)
        helper_attributes = [("pto.simt_entry", UnitAttr.get())]
        i32_attr_type = IntegerType.get_signless(32)
        if subkernel.spec.simt_max_threads is not None:
            helper_attributes.append(
                (
                    "pto.simt_max_threads",
                    IntegerAttr.get(i32_attr_type, subkernel.spec.simt_max_threads),
                )
            )
        if subkernel.spec.simt_max_regs is not None:
            helper_attributes.append(
                (
                    "pto.simt_max_regs",
                    IntegerAttr.get(i32_attr_type, subkernel.spec.simt_max_regs),
                )
            )
        helper_fn = self._create_named_helper_function(
            helper_symbol,
            arg_types,
            attributes=tuple(helper_attributes),
            owner_symbol_name=owner_symbol_name,
        )
        self._simt_helper_specializations[specialization_key] = helper_fn

        entry_block = helper_fn.add_entry_block()
        wrapped_args = tuple(
            wrap_like_surface_value(template, value)
            for template, value in zip(arg_templates, entry_block.arguments)
        )
        with (
            self.enter_function(helper_fn, owner_symbol_name=owner_symbol_name),
            self.suspend_subkernel_scope(),
            InsertionPoint(entry_block),
        ):
            with self.enter_subkernel(subkernel):
                subkernel.emit_body(*wrapped_args, **kwargs)
            func.ReturnOp([])

        return helper_fn, arg_templates

    def _create_named_helper_function(
        self,
        symbol_name: str,
        arg_types: tuple,
        *,
        attributes: tuple[tuple[str, object], ...] = (),
        owner_symbol_name: str | None = None,
    ):
        """Create one helper using an already materialized symbol name."""
        owner_symbol_name = (
            self.current_function_owner_symbol_name if owner_symbol_name is None else owner_symbol_name
        )
        fn_ty = func.FunctionType.get(list(arg_types), [])
        symbol_table = self.get_or_create_kernel_module_child_symbol_table(
            owner_symbol_name,
            self._kernel_module_child_records[owner_symbol_name].module_spec,
        )
        with InsertionPoint(symbol_table):
            helper = func.FuncOp(symbol_name, fn_ty)
            for attr_name, attr_value in attributes:
                helper.attributes[attr_name] = attr_value
        self._helpers[(owner_symbol_name, ("named", symbol_name))] = helper
        return helper

    def _next_simt_helper_symbol(self, base_symbol: str) -> str:
        index = self._simt_helper_symbol_counters.get(base_symbol, 0)
        while True:
            symbol = f"{base_symbol}__simt_{index}"
            index += 1
            if self.lookup_helper(symbol) is None:
                self._simt_helper_symbol_counters[base_symbol] = index
                return symbol

    def resolve_simt_peer_symbol(self, subkernel) -> str:
        """Return the unique materialized helper symbol for a ``@pto.simt`` peer."""
        symbol_name = subkernel.spec.symbol_name
        matches = [
            helper_fn
            for key, helper_fn in self._simt_helper_specializations.items()
            if key.owner_symbol_name == self.current_function_owner_symbol_name
            and key.symbol_name == symbol_name
        ]
        if not matches:
            raise RuntimeError(
                f"pto.import_reserved_buffer(..., peer_func={symbol_name}) cannot resolve "
                "the @pto.simt helper symbol before the helper is called or launched"
            )
        if len(matches) > 1:
            raise RuntimeError(
                f"pto.import_reserved_buffer(..., peer_func={symbol_name}) is ambiguous "
                "because the @pto.simt helper has multiple specializations; pass the "
                "materialized peer function symbol explicitly"
            )
        return _symbol_name(matches[0])

    def lower_kernel_module_call(self, kernel_handle, *args, **kwargs):
        """Lower one ``@pto.jit(entry=False)`` kernel-module call in the active trace."""
        if kwargs:
            raise TypeError("@pto.jit(entry=False) kernel module calls do not support keyword arguments yet")

        compiler = kernel_handle._compiler
        kernel_signature = compiler._kernel_signature
        if kernel_signature.constexpr_parameters:
            raise RuntimeError(
                "@pto.jit(entry=False) kernel modules do not support constexpr specialization parameters"
            )
        positional_params = kernel_signature.positional_parameters
        if len(args) != len(positional_params):
            raise TypeError(
                f"@pto.jit(entry=False) kernel module {kernel_handle._py_name!r} expects "
                f"{len(positional_params)} argument(s), got {len(args)}"
            )
        arg_templates = tuple(
            const(arg, dtype=param.annotation)
            if isinstance(param, RuntimeScalarParameterSpec) and not hasattr(unwrap_surface_value(arg), "type")
            else arg
            for param, arg in zip(positional_params, args)
        )

        arg_types = tuple(unwrap_surface_value(arg).type for arg in arg_templates)
        helper_spec = HelperFunctionSpec(
            symbol_name=compiler._module_spec.function_name,
            arg_types=arg_types,
        )
        helper_fn, created = self.get_or_create_kernel_module_primary_function(
            helper_spec,
            compiler._module_spec,
        )

        if created:
            entry_block = helper_fn.add_entry_block()
            wrapped_args = tuple(
                wrap_like_surface_value(template, value)
                for template, value in zip(arg_templates, entry_block.arguments)
            )
            with (
                self.enter_function(helper_fn, owner_symbol_name=helper_fn.name.value),
                self.suspend_subkernel_scope(),
                InsertionPoint(entry_block),
            ):
                compiler.tracing_callback()(*wrapped_args)
                func.ReturnOp([])

        caller_symbol_name = self.current_function_owner_symbol_name
        import_fn, _ = self.get_or_create_kernel_module_import_declaration(
            caller_symbol_name,
            helper_spec,
        )
        self.record_kernel_module_dependency(caller_symbol_name, helper_spec.symbol_name)
        call_args = [unwrap_surface_value(arg) for arg in arg_templates]
        func.CallOp(import_fn, call_args)

    def lookup_helper(self, symbol_name: str):
        """Return a previously declared helper function, or ``None``."""
        for helper in self._helpers.values():
            if helper.name.value == symbol_name:
                return helper
        return None

    def _attach_ptodsl_logical_name_attr(self, func_op, logical_name: str) -> None:
        """Mark one ABI-specialized PTODSL symbol with its authored logical name."""
        func_op.attributes["pto.ptodsl.logical_name"] = StringAttr.get(logical_name)

    def get_or_create_helper_function(self, spec: HelperFunctionSpec, *, owner_symbol_name: str | None = None):
        """
        Look up or create a helper ``func.func`` in the owning child module.

        Returns ``(helper_fn, created)`` where *created* reports whether a new
        symbol was emitted in this trace session.
        """
        owner_symbol_name = (
            self.current_function_owner_symbol_name if owner_symbol_name is None else owner_symbol_name
        )
        cache_key = (owner_symbol_name, spec.cache_key())
        helper = self._helpers.get(cache_key)
        if helper is not None:
            return helper, False

        fn_ty = func.FunctionType.get(list(spec.arg_types), list(spec.result_types))
        specialized_symbol_name = spec.specialized_symbol_name()
        symbol_table = self.get_or_create_kernel_module_child_symbol_table(
            owner_symbol_name,
            self._kernel_module_child_records[owner_symbol_name].module_spec,
        )
        with InsertionPoint(symbol_table):
            helper = func.FuncOp(specialized_symbol_name, fn_ty)
            self._attach_ptodsl_logical_name_attr(helper, spec.symbol_name)
            for attr_name, attr_value in spec.attributes:
                helper.attributes[attr_name] = attr_value
        self._helpers[cache_key] = helper
        return helper, True

    def get_or_create_kernel_module_primary_function(self, spec: HelperFunctionSpec, module_spec):
        """Look up or create the primary definition for one kernel-module callee."""
        cache_key = spec.cache_key()
        helper = self._kernel_module_primary_functions.get(cache_key)
        if helper is not None:
            return helper, False

        fn_ty = func.FunctionType.get(list(spec.arg_types), list(spec.result_types))
        specialized_symbol_name = spec.specialized_symbol_name()
        symbol_table = self.get_or_create_kernel_module_child_symbol_table(specialized_symbol_name, module_spec)
        with InsertionPoint(symbol_table):
            helper = func.FuncOp(specialized_symbol_name, fn_ty)
            self._attach_ptodsl_logical_name_attr(helper, spec.symbol_name)
            helper.attributes["sym_visibility"] = StringAttr.get("public")
            helper.attributes["pto.visibility"] = StringAttr.get("external")
            if (
                module_spec.backend == "emitc"
                and not module_spec.entry
                and module_spec.kernel_kind in {"cube", "vector"}
            ):
                helper.attributes["pto.kernel_kind"] = Attribute.parse(
                    f"#pto.kernel_kind<{module_spec.kernel_kind}>"
                )
            for attr_name, attr_value in spec.attributes:
                helper.attributes[attr_name] = attr_value
        self._kernel_module_primary_functions[cache_key] = helper
        return helper, True

    def kernel_module_import_symbol_name(self, caller_symbol_name: str, callee_symbol_name: str) -> str:
        """Return the import declaration symbol for one caller/callee pair."""
        _ = caller_symbol_name
        return callee_symbol_name

    def get_or_create_kernel_module_import_declaration(
        self,
        caller_symbol_name: str,
        spec: HelperFunctionSpec,
    ):
        """Look up or create the private import declaration for one kernel-module callee."""
        target_symbol_name = spec.specialized_symbol_name()
        key = (caller_symbol_name, target_symbol_name)
        helper = self._kernel_module_private_imports.get(key)
        if helper is not None:
            return helper, False

        fn_ty = func.FunctionType.get(list(spec.arg_types), list(spec.result_types))
        import_symbol_name = self.kernel_module_import_symbol_name(caller_symbol_name, target_symbol_name)
        caller_symbol_table = self.get_or_create_kernel_module_child_symbol_table(
            caller_symbol_name,
            self._kernel_module_child_records[caller_symbol_name].module_spec,
        )
        with InsertionPoint(caller_symbol_table):
            helper = func.FuncOp(import_symbol_name, fn_ty)
            self._attach_ptodsl_logical_name_attr(helper, spec.symbol_name)
            helper.attributes["sym_visibility"] = StringAttr.get("private")
        self._kernel_module_private_imports[key] = helper
        return helper, True

    def record_kernel_module_dependency(self, caller_symbol_name: str, callee_symbol_name: str) -> None:
        """Record one caller->callee dependency edge for kernel-module assembly."""
        deps = self._kernel_module_dependencies.setdefault(caller_symbol_name, set())
        deps.add(callee_symbol_name)

    def snapshot_kernel_module_graph(self) -> KernelModuleGraphSnapshot:
        """Return an immutable snapshot of traced kernel-module imports/dependencies."""
        imports = tuple(
            sorted(
                (
                    KernelModuleImportRecord(
                        caller_symbol_name=caller_symbol_name,
                        import_symbol_name=helper.name.value,
                        target_symbol_name=callee_symbol_name,
                    )
                    for (caller_symbol_name, callee_symbol_name), helper in self._kernel_module_private_imports.items()
                ),
                key=lambda record: (
                    record.caller_symbol_name,
                    record.target_symbol_name,
                    record.import_symbol_name,
                ),
            )
        )
        dependencies = tuple(
            (caller_symbol_name, tuple(sorted(callee_symbol_names)))
            for caller_symbol_name, callee_symbol_names in sorted(self._kernel_module_dependencies.items())
        )
        return KernelModuleGraphSnapshot(imports=imports, dependencies=dependencies)

    def get_or_create_kernel_module_child_symbol_table(self, primary_symbol_name: str, module_spec):
        """Return the child-module symbol table that owns *primary_symbol_name*."""
        symbol_table = self._kernel_module_child_symbol_tables.get(primary_symbol_name)
        if symbol_table is not None:
            return symbol_table

        child_op, symbol_table = create_container_child_module(self.module, module_spec)
        self._kernel_module_child_symbol_tables[primary_symbol_name] = symbol_table
        self._kernel_module_child_records[primary_symbol_name] = TracedChildModuleRecord(
            symbol_name=f"{primary_symbol_name}$child",
            primary_symbol_name=primary_symbol_name,
            role="kernel_module",
            module_spec=module_spec,
        )
        return symbol_table

    def validate_final_state(self) -> None:
        """Check that tracing-time session stacks were fully unwound."""
        if self._subkernel_stack:
            raise RuntimeError("PTODSL trace-session exited with an open subkernel lowering frame")
        if self._carry_loop_stack:
            raise RuntimeError("PTODSL trace-session exited with an open loop-carry lowering frame")


def _coerce_simt_launch_dims(dims):
    if not isinstance(dims, (tuple, list)) or len(dims) != 3:
        raise TypeError("pto.simt_launch(..., dims=...) expects a 3-item (dim_x, dim_y, dim_z) tuple")
    return tuple(
        _coerce_i32_dim(dim, context=f"pto.simt_launch(..., dims[{index}])")
        for index, dim in enumerate(dims)
    )


def _coerce_i32_dim(value, *, context: str):
    raw_value = unwrap_surface_value(value)
    i32 = IntegerType.get_signless(32)
    if isinstance(raw_value, bool):
        raise TypeError(f"{context} does not accept bool values")
    if isinstance(raw_value, int):
        if raw_value < 0:
            raise ValueError(f"{context} expects a non-negative i32 launch dimension, got {raw_value}")
        return arith.ConstantOp(i32, raw_value).result
    if IndexType.isinstance(raw_value.type):
        return arith.IndexCastOp(i32, raw_value).result
    if IntegerType.isinstance(raw_value.type):
        width = IntegerType(raw_value.type).width
        if width != 32:
            raise TypeError(f"{context} expects i32 launch dimension, got {raw_value.type}")
        return _strip_integer_signedness(raw_value)
    raise TypeError(f"{context} expects i32 launch dimension, got {raw_value.type}")


def _symbol_name(ir_fn) -> str:
    try:
        name_attr = ir_fn.attributes["sym_name"]
    except KeyError as exc:
        raise RuntimeError("PTODSL helper function is missing sym_name")
    if name_attr is None:
        raise RuntimeError("PTODSL helper function has empty sym_name")
    return str(name_attr.value)


def _simt_static_kwargs_signature(kwargs):
    return tuple(
        (name, _simt_static_signature_atom(value))
        for name, value in sorted(kwargs.items())
    )


def _simt_static_signature_atom(value):
    raw_value = unwrap_surface_value(value)
    if hasattr(raw_value, "type"):
        raise TypeError(
            "pto.simt_launch keyword arguments must be static hashable values; "
            "pass runtime SSA arguments positionally"
        )
    try:
        hash(value)
    except TypeError:
        if isinstance(value, dict):
            return (
                "dict",
                tuple(
                    sorted(
                        tuple(
                            (
                                _simt_static_signature_atom(key),
                                _simt_static_signature_atom(item),
                            )
                            for key, item in value.items()
                        ),
                        key=repr,
                    )
                ),
            )
        if isinstance(value, (list, tuple)):
            return (
                type(value).__name__,
                tuple(_simt_static_signature_atom(item) for item in value),
            )
        if isinstance(value, set):
            return (
                "set",
                tuple(sorted((_simt_static_signature_atom(item) for item in value), key=repr)),
            )
        return (type(value).__name__, repr(value))
    return value


__all__ = [
    "HelperFunctionSpec",
    "KernelModuleGraphSnapshot",
    "KernelModuleImportRecord",
    "SubkernelTraceFrame",
    "TraceSession",
]
