# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""Shared user-facing diagnostics for PTODSL tracing misuse."""

from __future__ import annotations


class PTODSLTracingMisuseError(TypeError):
    """Raised when authored Python misuses PTODSL runtime values during tracing."""


def _format_source_context(function_name: str | None, source_file: str | None, source_line: int | None) -> str:
    details = []
    if function_name:
        details.append(f"kernel {function_name!r}")
    if source_file is not None:
        location = source_file
        if source_line is not None:
            location = f"{location}:{source_line}"
        details.append(location)
    if not details:
        return ""
    return f" ({', '.join(details)})"


def native_python_control_flow_error(usage: str) -> PTODSLTracingMisuseError:
    """Return one actionable diagnostic for native Python control-flow misuse."""
    return PTODSLTracingMisuseError(
        f"native Python {usage} cannot consume a PTODSL runtime value during tracing. "
        "This value is a device-side SSA/runtime-metadata value, not a Python bool/int. "
        "Use pto.if_(...) or pto.for_(...) for device-side control flow, or keep the "
        "bound/condition in pto.const_expr."
    )


def host_tensor_metadata_error(message: str, *, param_name: str | None = None) -> TypeError:
    """Return one actionable diagnostic for unsupported host-tensor metadata."""
    prefix = "host tensor metadata is incomplete or unsupported"
    if param_name is not None:
        prefix = f"@pto.jit host tensor '{param_name}' metadata is incomplete or unsupported"
    return TypeError(f"{prefix}: {message}")


def jit_missing_annotation_error(name: str) -> TypeError:
    """Return one diagnostic for missing ``@pto.jit`` positional ABI annotations."""
    return TypeError(
        f"@pto.jit positional parameter '{name}' does not declare an entry ABI annotation. "
        'Use an explicit GM pointer such as pto.ptr(pto.f32, "gm") for device buffers, '
        "a PTO scalar type such as pto.i32/pto.f32/pto.i1 for runtime scalars, "
        "or move compile-time values to keyword-only pto.const_expr parameters."
    )


def jit_helper_missing_annotation_error(name: str) -> TypeError:
    """Return one diagnostic for missing ``@pto.jit(entry=False)`` module annotations."""
    return TypeError(
        f"@pto.jit(entry=False) parameter '{name}' does not declare a kernel-module ABI annotation. "
        "Use kernel-module value types such as pto.Tile / pto.TensorView / "
        "pto.PartitionTensorView, a typed pto.ptr(...) in any memory space, or "
        "a PTO scalar annotation such as pto.i32/pto.f32/pto.i1."
    )


def jit_illegal_formal_annotation_error(name: str, annotation: object) -> TypeError:
    """Return one diagnostic for unsupported ``@pto.jit`` positional annotations."""
    return TypeError(
        f"@pto.jit positional parameter '{name}' uses unsupported entry annotation {annotation!r}. "
        'The public @pto.jit entry ABI accepts explicit GM pointers such as pto.ptr(pto.f32, "gm"), '
        "PTO scalar annotations such as pto.i32/pto.f32/pto.i1 for runtime scalars, "
        "and keyword-only pto.const_expr compile-time parameters. "
        "Legacy host tensor annotations such as pto.tensor_spec(...), and low-level PTODSL "
        "types such as Tile, PartitionTensorView, VReg, or non-entry pointer forms do not "
        "belong at the host/kernel entry."
    )


def jit_helper_illegal_formal_annotation_error(name: str, annotation: object) -> TypeError:
    """Return one diagnostic for unsupported ``@pto.jit(entry=False)`` module annotations."""
    return TypeError(
        f"@pto.jit(entry=False) parameter '{name}' uses unsupported kernel-module annotation {annotation!r}. "
        "The kernel-module ABI accepts pto.Tile / pto.TensorView / "
        "pto.PartitionTensorView, typed pto.ptr(...) values in any memory space, "
        "and PTO scalar annotations such as pto.i32/pto.f32/pto.i1. "
        "Legacy host tensor annotations, VReg, and mask values do not belong at "
        "this kernel-module boundary."
    )


def jit_legacy_tensor_spec_entry_error(name: str, annotation: object) -> TypeError:
    """Return one diagnostic for legacy ``pto.tensor_spec(...)`` entry annotations."""
    return TypeError(
        f"@pto.jit positional parameter '{name}' still uses legacy host-tensor entry annotation "
        f"{annotation!r}. The public @pto.jit entry ABI no longer accepts pto.tensor_spec(...) "
        'mainline. Migrate this parameter to an explicit GM pointer such as pto.ptr(pto.f32, "gm"), '
        "pass shape/stride metadata as runtime scalars, and reconstruct the tensor view in-kernel "
        "with pto.make_tensor_view(...)."
    )


def jit_legacy_tensor_spec_helper_error(name: str, annotation: object) -> TypeError:
    """Return one diagnostic for legacy ``pto.tensor_spec(...)`` kernel-module annotations."""
    return TypeError(
        f"@pto.jit(entry=False) parameter '{name}' still uses legacy host-tensor annotation "
        f"{annotation!r}. Kernel-module ABI does not accept pto.tensor_spec(...); "
        "pass Tile / TensorView / PartitionTensorView / typed ptr / PTO scalar values "
        "across the kernel-module boundary instead."
    )


def jit_non_gm_ptr_entry_error(name: str, annotation: object) -> TypeError:
    """Return one diagnostic for non-GM pointer entry annotations."""
    return TypeError(
        f"@pto.jit positional parameter '{name}' uses non-GM pointer entry annotation {annotation!r}. "
        'The host-visible @pto.jit boundary only accepts explicit GM pointers such as pto.ptr(pto.f32, "gm"). '
        "This boundary contract does not change the global pto.ptr(...) defaults; "
        'spell out "gm" explicitly at the @pto.jit entry.'
    )


def jit_helper_standalone_type_inference_error(name: str, annotation: object) -> RuntimeError:
    """Return one diagnostic for kernel-module params that need caller-provided concrete types."""
    return RuntimeError(
        f"@pto.jit(entry=False) parameter '{name}' annotated as {annotation!r} uses an "
        "abstract kernel-module marker type. Standalone kernel-module compilation cannot infer a "
        "concrete MLIR argument type for this boundary yet; compile the module through a "
        "caller that supplies concrete Tile/TensorView values."
    )


def kernel_module_return_value_error(result) -> RuntimeError:
    """Return one diagnostic for illegal ``@pto.jit(entry=False)`` return values."""
    return RuntimeError(
        "@pto.jit(entry=False) kernel modules must return None. "
        f"Got {type(result).__name__!r} instead; pass data across the module "
        "boundary through Tile / TensorView / PartitionTensorView / ptr / scalar "
        "arguments, not Python return values."
    )


def kernel_module_compile_error(function_name: str | None = None) -> RuntimeError:
    """Return one diagnostic for direct Python-side kernel-module compilation."""
    target = "@pto.jit(entry=False) kernel module"
    if function_name:
        target = f"@pto.jit(entry=False) kernel module {function_name!r}"
    return RuntimeError(
        f"{target} is not directly compilable from Python. Compile an entry kernel "
        "that calls this module instead."
    )


def kernel_module_launch_error(function_name: str | None = None) -> RuntimeError:
    """Return one diagnostic for Python launch on ``@pto.jit(entry=False)``."""
    target = "@pto.jit(entry=False) kernel module"
    if function_name:
        target = f"@pto.jit(entry=False) kernel module {function_name!r}"
    return RuntimeError(
        f"{target} is not launchable from Python. Only @pto.jit(entry=True) "
        "kernels support compiled[grid, stream](...)."
    )


def jit_source_entry_false_error(
    source: object,
    *,
    function_name: str | None = None,
) -> TypeError:
    """Return one diagnostic for unsupported ``@pto.jit(entry=False, source=...)``."""
    target = "@pto.jit(source=...) kernel"
    if function_name:
        target = f"@pto.jit(source=...) kernel {function_name!r}"
    return TypeError(
        f"{target} does not support entry=False while source={source!r}. "
        "Source-backed JIT is currently limited to launchable entry kernels."
    )


def jit_source_constexpr_error(
    name: str,
    source: object,
    *,
    function_name: str | None = None,
) -> TypeError:
    """Return one diagnostic for unsupported source-backed ``pto.const_expr`` params."""
    target = "@pto.jit(source=...) kernel"
    if function_name:
        target = f"@pto.jit(source=...) kernel {function_name!r}"
    return TypeError(
        f"{target} does not support keyword-only pto.const_expr parameter '{name}' while source={source!r}. "
        "Source-backed JIT currently loads a fixed PTO IR file and does not template or specialize source text."
    )


def jit_source_compile_constexpr_error(
    names: list[str] | tuple[str, ...],
    source: object,
    *,
    function_name: str | None = None,
) -> TypeError:
    """Return one diagnostic for ``.compile(...)`` constexpr bindings in source mode."""
    target = "@pto.jit(source=...) kernel"
    if function_name:
        target = f"@pto.jit(source=...) kernel {function_name!r}"
    joined = ", ".join(names)
    return TypeError(
        f"{target} does not accept .compile(...) constexpr binding(s) {joined} while source={source!r}. "
        "Source-backed JIT currently loads a fixed PTO IR file and does not template or specialize source text."
    )


def jit_source_file_error(source: object, resolved_path: object, reason: str) -> FileNotFoundError:
    """Return one diagnostic for source path resolution/loading failures."""
    return FileNotFoundError(
        f"@pto.jit(source={source!r}) could not load PTO IR source file {str(resolved_path)!r}: {reason}"
    )


def jit_source_entry_error(source_path: object, entry_name: str, reason: str) -> TypeError:
    """Return one diagnostic for source entry selection failures."""
    return TypeError(
        f"@pto.jit(source=...) could not bind entry {entry_name!r} in {str(source_path)!r}: {reason}"
    )


def jit_source_abi_error(source_path: object, entry_name: str, reason: str) -> TypeError:
    """Return one diagnostic for source ABI verification failures."""
    return TypeError(
        f"@pto.jit(source=...) ABI mismatch for entry {entry_name!r} in {str(source_path)!r}: {reason}"
    )


def jit_keyword_only_non_constexpr_error(name: str, annotation: object) -> TypeError:
    """Return one diagnostic for keyword-only params that are not ``pto.const_expr``."""
    return TypeError(
        f"@pto.jit keyword-only parameter '{name}' uses unsupported compile-time annotation {annotation!r}. "
        "Compile-time @pto.jit parameters must remain keyword-only pto.const_expr values in this change; "
        "move runtime data to positional pointer/scalar parameters instead."
    )


def jit_constexpr_missing_default_error(name: str) -> TypeError:
    """Return one diagnostic for ``pto.const_expr`` params missing a default value."""
    return TypeError(
        f"@pto.jit constexpr parameter '{name}' must declare a default value until explicit "
        "compile-time specialization is implemented. Keep this parameter keyword-only and "
        "override it through .compile(...) when a non-default specialization is needed."
    )


def make_tensor_view_missing_metadata_error(ptr: object) -> TypeError:
    """Return one diagnostic for ``make_tensor_view`` calls missing shape/stride metadata."""
    return TypeError(
        f"make_tensor_view({ptr!r}, ...) requires explicit shape= and strides= in the pointer-first "
        "@pto.jit contract. Do not rely on host tensor proxy metadata; pass runtime shape/stride "
        "scalars through the kernel entry and forward them explicitly here."
    )


def make_tensor_view_invalid_layout_error(layout: object) -> TypeError:
    """Return one diagnostic for unsupported ``make_tensor_view(layout=...)`` spellings."""
    return TypeError(
        "make_tensor_view(..., layout=...) expects one of the public layout spellings "
        f"'ND', 'DN', or 'NZ' (case-insensitive), or a raw PTO Layout enum/attr. Got {layout!r}."
    )


def subkernel_host_tensor_boundary_error(role: str, name: str) -> TypeError:
    """Return one diagnostic for host-tensor usage outside the JIT boundary."""
    return TypeError(
        f"@pto.{role} parameter '{name}' uses a host tensor value, but host tensors only belong "
        "at the @pto.jit boundary. Pass PTODSL device-side values such as Tile, "
        "PartitionTensorView, typed pointers, or PTO scalars instead."
    )


def subkernel_signature_boundary_error(role: str, name: str) -> TypeError:
    """Return one diagnostic for illegal host-tensor formal annotations on a subkernel."""
    return TypeError(
        f"@pto.{role} parameter '{name}' cannot be annotated with pto.tensor_spec(...). "
        "Host tensors are only valid as @pto.jit positional parameters."
    )


def subkernel_missing_annotation_error(role: str, name: str) -> TypeError:
    """Return one diagnostic for missing subkernel ABI annotations."""
    return TypeError(
        f"@pto.{role} parameter '{name}' does not declare a subkernel ABI annotation. "
        "Subkernel interfaces are explicit: use the role-specific PTODSL boundary types "
        "instead of relying on implicit inference."
    )


def subkernel_illegal_parameter_kind_error(role: str, name: str, kind) -> TypeError:
    """Return one diagnostic for unsupported subkernel parameter kinds."""
    return TypeError(
        f"@pto.{role} parameter '{name}' uses unsupported parameter kind {kind!r}. "
        "Subkernel interfaces only accept positional parameters in this PTODSL surface."
    )


def subkernel_illegal_annotation_error(role: str, name: str, annotation: object, expected: str) -> TypeError:
    """Return one diagnostic for unsupported subkernel ABI annotations."""
    return TypeError(
        f"@pto.{role} parameter '{name}' uses unsupported subkernel annotation {annotation!r}. "
        f"Legal @pto.{role} interfaces are restricted to {expected}. If you need a different ABI, "
        "use @pto.jit(entry=False) instead of a subkernel decorator."
    )


def subkernel_argument_type_error(role: str, name: str, expected: str, observed: object) -> TypeError:
    """Return one diagnostic for runtime/callsite subkernel ABI mismatches."""
    return TypeError(
        f"@pto.{role} argument '{name}' violates the declared subkernel interface. "
        f"Expected {expected}, got {observed!r}. Subkernel ABI checks are unconditional; "
        "either pass a legal PTODSL boundary value or remove the subkernel decorator."
    )


def illegal_subkernel_placement_error(role: str, outer_role: str | None) -> RuntimeError:
    """Return one diagnostic for a subkernel call placed outside the supported layer graph."""
    if role == "simt":
        return RuntimeError(
            "@pto.simt helper materialization is only supported from the top-level @pto.jit body; "
            f"it cannot be materialized inside @pto.{outer_role}."
        )
    return RuntimeError(
        f"@pto.{role} may only be called from the top-level @pto.jit body; "
        f"nested invocation inside @pto.{outer_role} is not part of the PTODSL layer contract."
    )


def illegal_inline_subkernel_placement_error(role: str, outer_role: str | None) -> RuntimeError:
    """Return one diagnostic for an inline subkernel scope placed outside the supported layer graph."""
    return RuntimeError(
        f"inline pto.{role}() may only be used from the top-level @pto.jit body; "
        f"nested use inside @pto.{outer_role} is not part of the PTODSL layer contract."
    )


def subkernel_kernel_kind_mismatch_error(role: str, kernel_kind: str) -> RuntimeError:
    """Return one diagnostic for mixing explicit @pto.jit kernel kind with the opposite subkernel kind."""
    return RuntimeError(
        f"@pto.{role} cannot be lowered inside an explicit @pto.jit(kernel_kind={kernel_kind!r}) "
        "module. Remove the explicit kernel_kind so PTOAS can split cube/vector sections, "
        "or keep subkernel scopes in the same physical kind."
    )


def inline_subkernel_value_escape_error(role: str, type_text: str) -> RuntimeError:
    """Return one diagnostic for outlined inline-scope values escaping their helper boundary."""
    return RuntimeError(
        f"inline pto.{role}() cannot let values defined inside the outlined subkernel "
        f"escape the scope boundary (got {type_text}). Write through a Tile/UB buffer "
        "or keep the consumer inside the same inline subkernel."
    )


def simd_value_escape_error(type_text: str, *, surface: str = "@pto.simd") -> RuntimeError:
    """Return one diagnostic for transient SIMD values escaping a simd subkernel boundary."""
    return RuntimeError(
        f"{surface} cannot return transient SIMD values across the subkernel boundary "
        f"(got {type_text}). Write the value back to a Tile/UB buffer instead."
    )


def subkernel_return_boundary_error(role: str, observed: object) -> TypeError:
    """Return one diagnostic for unsupported PTODSL subkernel return values."""
    return TypeError(
        f"@pto.{role} return values must be PTO scalar values or tuples/lists/dicts of PTO scalar values. "
        f"Got {observed!r}. Return Tile/TensorView/PartitionTensorView/ptr data through explicit "
        "subkernel operands instead."
    )




def tile_row_alignment_error(*, shape, dtype, row_bytes: int, required_alignment: int) -> TypeError:
    """Return one diagnostic for authored tile shapes violating row-byte alignment."""
    return TypeError(
        "alloc_tile(shape=...) physical row layout is invalid for the current PTODSL tile contract: "
        f"shape={list(shape)!r} with dtype={dtype!r} gives a row byte size of {row_bytes}, "
        f"but row-major none-box tiles must be {required_alignment}-byte aligned. "
        "For logical column tiles such as [Br, 1], prefer blayout='ColMajor' instead of authoring them "
        "as row-major narrow tiles. If row-major is truly required, keep the physical tile shape explicitly "
        "aligned and express the logical tail with valid_shape=[...]."
    )


def explicit_mode_required_error(surface: str, current_mode: str | None) -> RuntimeError:
    """Return one diagnostic for explicit-only surfaces used outside explicit mode."""
    observed_mode = "unknown" if current_mode is None else current_mode
    return RuntimeError(
        f"{surface} is an auto-mode contract violation: it is only available in "
        f'@pto.jit(mode="explicit"); current kernel mode is {observed_mode!r}. '
        "Move the kernel to explicit mode before authoring this surface."
    )


def explicit_mode_required_with_context_error(surface: str, module_spec) -> RuntimeError:
    """Return one diagnostic for explicit-only surfaces used outside explicit mode with source context."""
    observed_mode = getattr(module_spec, "mode", None)
    context = _format_source_context(
        getattr(module_spec, "function_name", None),
        getattr(module_spec, "source_file", None),
        getattr(module_spec, "source_line", None),
    )
    observed_mode = "unknown" if observed_mode is None else observed_mode
    return RuntimeError(
        f"{surface} is an auto-mode contract violation{context}: it is only available in "
        f'@pto.jit(mode="explicit"); current kernel mode is {observed_mode!r}. '
        "Move the kernel to explicit mode before authoring this surface."
    )


def invalid_jit_mode_error(
    mode: str,
    *,
    function_name: str | None = None,
    source_file: str | None = None,
    source_line: int | None = None,
) -> ValueError:
    """Return one diagnostic for unsupported ``@pto.jit(mode=...)`` values."""
    context = _format_source_context(function_name, source_file, source_line)
    return ValueError(
        f"unsupported PTODSL jit mode {mode!r}{context}; expected 'auto' or 'explicit'"
    )


def invalid_jit_backend_error(
    backend: str,
    *,
    function_name: str | None = None,
    source_file: str | None = None,
    source_line: int | None = None,
) -> ValueError:
    """Return one diagnostic for unsupported ``@pto.jit(backend=...)`` values."""
    context = _format_source_context(function_name, source_file, source_line)
    return ValueError(
        f"unsupported PTODSL jit backend {backend!r}{context}; expected 'vpto' or 'emitc'"
    )


def legacy_subkernel_surface_error(surface: str) -> TypeError:
    """Return one diagnostic for removed legacy subkernel public surfaces."""
    return TypeError(
        f"{surface} is a legacy PTODSL subkernel surface and is no longer supported. "
        "Use @pto.tileop for named custom-op helpers, or inline custom-op code with "
        "`with pto.tileop():`. Keep @pto.simt only for launched SIMT helpers."
    )


def unsupported_public_surface_error(name: str) -> AttributeError:
    """Return one diagnostic for unsupported names on the public ``pto`` surface."""
    hints = {
        "ukernel": (
            'Use @pto.jit(mode="explicit") for explicit DMA orchestration, and call or inline '
            "@pto.tileop helpers directly from that kernel."
        ),
        "tile_buf_type": (
            "Use pto.alloc_tile(shape=..., dtype=..., memory_space=..., valid_shape=..., addr=...) "
            "to author tiles, and keep explicit tile-type construction inside internal implementation code only."
        ),
        "vecscope": (
            "Use @pto.tileop for named custom OP helpers, or inline custom OP code with `with pto.tileop():`."
        ),
        "as_ptr": (
            "Use tile.as_ptr(), view.as_ptr(), or partition.as_ptr() on the authored object itself "
            "instead of the removed pto.as_ptr(...) helper."
        ),
        "vbrc_load": (
            'Use pto.vlds(ptr, offset, dist="BRC_B32") instead of the removed pto.vbrc_load(...) helper.'
        ),
        "vsts_1pt": (
            'Use pto.vsts(vec, ptr, offset, mask, dist="1PT_B32") instead of the removed pto.vsts_1pt(...) helper.'
        ),
        "constexpr": (
            "Use pto.const_expr for compile-time @pto.jit parameters and trace-time control-flow guards."
        ),
        "tensor_spec": (
            "Host tensor ABI hints were removed from the PTODSL public surface. Use explicit "
            'GM pointers such as pto.ptr(pto.f32, "gm"), pass runtime shape/stride scalars, '
            "and reconstruct TensorView descriptors in-kernel with pto.make_tensor_view(...)."
        ),
        "TensorSpec": (
            "TensorSpec was removed from the PTODSL public surface together with pto.tensor_spec(...). "
            "Use explicit GM pointers plus runtime shape/stride scalars instead."
        ),
    }
    suffix = hints.get(name, "Use the documented PTODSL public surface instead.")
    return AttributeError(
        f"pto.{name} is not a supported PTODSL public interface. {suffix}"
    )


__all__ = [
    "PTODSLTracingMisuseError",
    "explicit_mode_required_error",
    "explicit_mode_required_with_context_error",
    "host_tensor_metadata_error",
    "jit_illegal_formal_annotation_error",
    "jit_constexpr_missing_default_error",
    "jit_keyword_only_non_constexpr_error",
    "jit_legacy_tensor_spec_entry_error",
    "jit_missing_annotation_error",
    "jit_non_gm_ptr_entry_error",
    "inline_subkernel_value_escape_error",
    "make_tensor_view_missing_metadata_error",
    "illegal_inline_subkernel_placement_error",
    "illegal_subkernel_placement_error",
    "jit_helper_illegal_formal_annotation_error",
    "jit_helper_missing_annotation_error",
    "jit_helper_standalone_type_inference_error",
    "kernel_module_compile_error",
    "kernel_module_launch_error",
    "kernel_module_return_value_error",
    "jit_source_abi_error",
    "jit_source_compile_constexpr_error",
    "jit_source_constexpr_error",
    "jit_source_entry_false_error",
    "jit_source_entry_error",
    "jit_source_file_error",
    "invalid_jit_mode_error",
    "invalid_jit_backend_error",
    "legacy_subkernel_surface_error",
    "jit_legacy_tensor_spec_helper_error",
    "native_python_control_flow_error",
    "simd_value_escape_error",
    "subkernel_argument_type_error",
    "subkernel_host_tensor_boundary_error",
    "subkernel_illegal_annotation_error",
    "subkernel_illegal_parameter_kind_error",
    "subkernel_kernel_kind_mismatch_error",
    "subkernel_missing_annotation_error",
    "subkernel_signature_boundary_error",
    "tile_row_alignment_error",
    "unsupported_public_surface_error",
]
