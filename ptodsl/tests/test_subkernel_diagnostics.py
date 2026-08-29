#!/usr/bin/env python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

from ptodsl import pto
from ptodsl._host_tensors import TensorSpec


def expect(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def expect_raises(callback, exc_type, *message_fragments: str) -> None:
    try:
        callback()
    except exc_type as exc:
        text = str(exc)
        for fragment in message_fragments:
            expect(fragment in text, f"expected diagnostic fragment {fragment!r} in {text!r}")
    else:
        raise AssertionError(f"expected {exc_type.__name__} to be raised")


def define_bad_subkernel_signature_probe():
    @pto.tileop
    def bad_tensor_formal(A: TensorSpec(rank=2, dtype=pto.f32)):
        pass

    return bad_tensor_formal


def define_illegal_simd_ptr_signature_probe():
    @pto.tileop
    def bad_ptr_formal(meta_ptr: pto.ptr(pto.i32, pto.MemorySpace.UB)):
        pass

    return bad_ptr_formal


def define_legacy_cube_decorator_probe():
    @pto.cube
    def legacy_cube(tile: pto.Tile):
        pass

    return legacy_cube


def define_legacy_simd_decorator_probe():
    @pto.simd
    def legacy_simd(tile: pto.Tile):
        pass

    return legacy_simd


def define_legacy_parenthesized_simd_decorator_probe():
    @pto.simd(name="legacy_simd")
    def legacy_simd(tile: pto.Tile):
        pass

    return legacy_simd


def define_legacy_cube_context_probe():
    with pto.cube():
        pass


def define_legacy_simd_context_probe():
    with pto.simd():
        pass


def define_removed_ukernel_surface_probe():
    return pto.ukernel


def define_removed_tensor_spec_surface_probe():
    return pto.tensor_spec


def define_removed_tensor_spec_type_surface_probe():
    return pto.TensorSpec


def define_invalid_jit_mode_probe():
    @pto.jit(target="a5", mode="hybrid")
    def bad_mode_probe():
        pass

    return bad_mode_probe


@pto.tileop
def host_tensor_operand_probe(tensor: pto.Tile):
    pass


def define_host_tensor_into_subkernel_probe():
    @pto.jit(target="a5")
    def bad_probe(A: TensorSpec(rank=2, dtype=pto.f32)):
        host_tensor_operand_probe(A)

    return bad_probe


@pto.simt
def nested_simt_probe():
    pto.get_tid_x()


@pto.tileop
def illegal_simt_placement_probe():
    nested_simt_probe()


@pto.jit(target="a5")
def nested_simt_from_simd_entry(*, TRACE_TOKEN: pto.const_expr = 0):
    illegal_simt_placement_probe()


@pto.tileop
def illegal_inline_simt_placement_probe():
    with pto.simt():
        pto.get_tid_x()


@pto.jit(target="a5")
def nested_inline_simt_from_simd_entry(*, TRACE_TOKEN: pto.const_expr = 0):
    illegal_inline_simt_placement_probe()


@pto.tileop
def tile_only_probe(inp_tile: pto.Tile):
    pass


@pto.jit(target="a5")
def illegal_subkernel_callsite_entry(A_ptr: pto.ptr(pto.f32, "gm")):
    tile_only_probe(A_ptr)


@pto.jit(target="a5", mode="explicit")
def inline_simt_value_escape_entry():
    meta_tile = pto.alloc_tile(shape=[1, 8], dtype=pto.i32, valid_shape=[1, 1])
    with pto.simt():
        leaked_tid = pto.get_tid_x()
    pto.store(leaked_tid, meta_tile.as_ptr() + 0)


def main() -> None:
    expect_raises(
        define_removed_ukernel_surface_probe,
        AttributeError,
        "pto.ukernel is not a supported PTODSL public interface",
        '@pto.jit(mode="explicit")',
        "@pto.tileop/@pto.simt",
    )
    expect_raises(
        define_removed_tensor_spec_surface_probe,
        AttributeError,
        "pto.tensor_spec is not a supported PTODSL public interface",
        "Host tensor ABI hints were removed",
        "pto.make_tensor_view(...)",
    )
    expect_raises(
        define_removed_tensor_spec_type_surface_probe,
        AttributeError,
        "pto.TensorSpec is not a supported PTODSL public interface",
        "TensorSpec was removed from the PTODSL public surface",
    )
    expect_raises(
        define_invalid_jit_mode_probe,
        ValueError,
        "unsupported PTODSL jit mode 'hybrid'",
        "bad_mode_probe",
        __file__,
        "expected 'auto' or 'explicit'",
    )
    expect_raises(
        define_bad_subkernel_signature_probe,
        TypeError,
        "@pto.tileop parameter 'A' cannot be annotated with pto.tensor_spec(...)",
        "@pto.jit positional parameters",
    )
    expect_raises(
        define_illegal_simd_ptr_signature_probe,
        TypeError,
        "@pto.tileop parameter 'meta_ptr' uses unsupported subkernel annotation",
        "pto.Tile parameters plus PTO scalar annotations",
        "@pto.jit(entry=False)",
    )
    expect_raises(
        define_legacy_cube_decorator_probe,
        TypeError,
        "pto.cube is a legacy single-core subkernel interface",
        "as either @pto.cube or with pto.cube():",
        "Use @pto.tileop",
        "Move MTE operations, pipe synchronization",
    )
    expect_raises(
        define_legacy_simd_decorator_probe,
        TypeError,
        "pto.simd is a legacy single-core subkernel interface",
        "as either @pto.simd or with pto.simd():",
        "PTOAS infers whether the helper is Vector or Cube",
    )
    expect_raises(
        define_legacy_parenthesized_simd_decorator_probe,
        TypeError,
        "pto.simd is a legacy single-core subkernel interface",
        "Use @pto.tileop",
    )
    expect_raises(
        define_legacy_cube_context_probe,
        TypeError,
        "pto.cube is a legacy single-core subkernel interface",
        "with pto.tileop(): for inline Tile/Scalar compute scopes",
    )
    expect_raises(
        define_legacy_simd_context_probe,
        TypeError,
        "pto.simd is a legacy single-core subkernel interface",
        "with pto.tileop(): for inline Tile/Scalar compute scopes",
    )
    expect_raises(
        define_host_tensor_into_subkernel_probe,
        TypeError,
        "@pto.jit positional parameter 'A' still uses legacy host-tensor entry annotation",
        "no longer accepts pto.tensor_spec(...)",
        "pto.make_tensor_view(...)",
    )
    expect_raises(
        nested_simt_from_simd_entry.compile,
        RuntimeError,
        "@pto.tileop may only invoke @pto.simt through an explicit launch",
        "helper[dim_x, dim_y, dim_z](...)",
    )
    expect_raises(
        nested_inline_simt_from_simd_entry.compile,
        RuntimeError,
        "inline pto.simt() may only be used from the top-level @pto.jit body",
        "inside @pto.tileop",
    )
    expect_raises(
        illegal_subkernel_callsite_entry.compile,
        TypeError,
        "@pto.tileop argument 'inp_tile' violates the declared subkernel interface",
        "Expected a pto.Tile value",
        "either pass a legal PTODSL boundary value or remove the subkernel decorator",
    )
    expect_raises(
        inline_simt_value_escape_entry.compile,
        RuntimeError,
        "inline pto.simt() cannot let values defined inside the outlined subkernel escape the scope boundary",
        "Write through a Tile/UB buffer",
    )
    print("ptodsl_subkernel_diagnostics: PASS")


if __name__ == "__main__":
    main()
