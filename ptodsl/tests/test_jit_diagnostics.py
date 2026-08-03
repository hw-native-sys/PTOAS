#!/usr/bin/env python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

from pathlib import Path
from tempfile import TemporaryDirectory

from ptodsl import pto, scalar
from ptodsl._ast_rewrite import PTODSLAstRewriteError
from ptodsl._host_tensors import TensorSpec
from ptodsl._host_tensors import inspect_host_tensor_metadata


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


@pto.jit(target="a5", ast_rewrite=False)
def native_python_if_runtime_const_probe():
    if pto.const(1):
        pto.pipe_barrier(pto.Pipe.ALL)


@pto.jit(target="a5", ast_rewrite=False)
def native_python_range_runtime_metadata_probe(rows: pto.i32):
    for _ in range(rows):
        pto.pipe_barrier(pto.Pipe.ALL)


@pto.jit(target="a5")
def float_loop_bound_probe():
    with pto.for_(0, pto.const(1.5, dtype=pto.f32), step=1):
        pto.pipe_barrier(pto.Pipe.ALL)


@pto.jit(target="a5")
def float_addptr_offset_probe():
    tile = pto.alloc_tile(shape=[1, 8], dtype=pto.i32, valid_shape=[1, 4])
    _ = pto.addptr(tile.as_ptr(), pto.const(1.5, dtype=pto.f32))


@pto.jit(target="a5")
def float_bitwise_probe():
    tile = pto.alloc_tile(shape=[1, 8], dtype=pto.f32, valid_shape=[1, 1])
    value = scalar.load(tile[0, 0])
    _ = value & 1


@pto.jit(target="a5")
def float_literal_index_store_probe(ptr: pto.ptr(pto.index, "gm")):
    scalar.store(1.5, ptr)


@pto.jit(target="a5")
def float_literal_index_binary_probe():
    index_value = pto.const(1, dtype=pto.index)
    _ = index_value + 1.5


@pto.jit(target="a5")
def same_width_float_store_probe():
    f16_tile = pto.alloc_tile(shape=[1, 16], dtype=pto.f16, valid_shape=[1, 1])
    bf16_tile = pto.alloc_tile(shape=[1, 16], dtype=pto.bf16, valid_shape=[1, 1])
    f16_value = scalar.load(f16_tile[0, 0])
    scalar.store(f16_value, bf16_tile[0, 0])


@pto.jit(target="a5")
def vmi_float_vcadd_missing_reassoc_probe():
    src = pto.vmi.vbrc(pto.f32(0.0), size=64)
    mask = pto.vmi.create_mask(64, size=64)
    _ = pto.vmi.vcadd(src, mask)


@pto.jit(target="a5")
def vmi_float_vcadd_none_reassoc_probe():
    src = pto.vmi.vbrc(pto.f32(0.0), size=64)
    mask = pto.vmi.create_mask(64, size=64)
    _ = pto.vmi.vcadd(src, mask, reassoc=None)


@pto.jit(target="a5")
def vmi_vbrc_untyped_scalar_probe():
    _ = pto.vmi.vbrc(0.0, size=64)


@pto.jit(target="a5")
def vmi_vci_untyped_scalar_probe():
    _ = pto.vmi.vci(0, size=64, order="ASC")

@pto.jit(target="a5")
def vmi_vinterpret_cast_missing_dtype_probe():
    src = pto.vmi.vbrc(pto.f32(0.0), size=64)
    _ = pto.vmi.vinterpret_cast(src)


@pto.jit(target="a5")
def vmi_vinterpret_cast_footprint_mismatch_probe():
    src = pto.vmi.vbrc(pto.f16(0.0), size=1)
    _ = pto.vmi.vinterpret_cast(src, to_dtype=pto.f32)


@pto.jit(target="a5")
def vmi_create_mask_group_mismatch_probe():
    _ = pto.vmi.create_mask(
        9,
        size=64,
        group=8,
    )


@pto.jit(target="a5")
def vmi_vload_mixed_mode_probe():
    tile = pto.alloc_tile(shape=[1, 64], dtype=pto.f32)
    src = tile.as_ptr()
    offset = pto.const(0, dtype=pto.index)
    _ = pto.vmi.vload(src, offset, size=64, dist_mode="continuous", group=8, stride=8)


@pto.jit(target="a5")
def vmi_vload_brc_stride_without_group_probe():
    tile = pto.alloc_tile(shape=[1, 64], dtype=pto.f32)
    src = tile.as_ptr()
    offset = pto.const(0, dtype=pto.index)
    _ = pto.vmi.vload(src, offset, size=64, dist_mode="brc", stride=8)


@pto.jit(target="a5")
def vmi_vstore_group_mask_probe():
    tile = pto.alloc_tile(shape=[1, 64], dtype=pto.f32)
    src = tile.as_ptr()
    offset = pto.const(0, dtype=pto.index)
    vec = pto.vmi.vload(src, offset, size=64)
    mask = pto.vmi.create_mask(64, size=64)
    pto.vmi.vstore(vec, src, offset, mask, group=8, stride=8)


@pto.jit(target="a5")
def vmi_vbrc_group_lane_mismatch_probe():
    tile = pto.alloc_tile(shape=[1, 16], dtype=pto.f32)
    src = tile.as_ptr()
    offset = pto.const(0, dtype=pto.index)
    compact = pto.vmi.vload(src, offset, size=16)
    _ = pto.vmi.vbrc(compact, size=64, group=8)

@pto.jit(target="a5")
def vmi_vload_missing_size_probe():
    tile = pto.alloc_tile(shape=[1, 64], dtype=pto.f32)
    src = tile.as_ptr()
    offset = pto.const(0, dtype=pto.index)
    _ = pto.vmi.vload(src, offset)


@pto.jit(target="a5")
def bool_loop_bound_probe():
    with pto.for_(0, True, step=1):
        pto.pipe_barrier(pto.Pipe.ALL)


@pto.jit(target="a5")
def bool_addptr_offset_probe():
    tile = pto.alloc_tile(shape=[1, 8], dtype=pto.i32, valid_shape=[1, 4])
    _ = pto.addptr(tile.as_ptr(), True)


@pto.jit(target="a5")
def bool_event_id_probe():
    pto.wait_flag(pto.Pipe.V, pto.Pipe.MTE2, event_id=True)


@pto.jit(target="a5")
def bool_fixed_integer_probe():
    _ = pto.make_mask(pto.f32, True)


@pto.jit(target="a5")
def bool_tile_element_probe():
    tile = pto.alloc_tile(shape=[2, 8], dtype=pto.i32, valid_shape=[2, 4])
    _ = scalar.load(tile[True, 0])


@pto.jit(target="a5")
def bool_address_sugar_probe():
    tile = pto.alloc_tile(shape=[1, 8], dtype=pto.i32, valid_shape=[1, 4])
    _ = scalar.load(tile.as_ptr() + True)


@pto.jit(target="a5")
def bool_tile_slice_probe():
    tile = pto.alloc_tile(shape=[8], dtype=pto.i32, valid_shape=[4])
    _ = tile[True:]


@pto.jit(target="a5")
def carry_update_mismatch_probe(*, BLOCK: pto.const_expr = 8):
    acc = pto.alloc_tile(shape=[1, BLOCK], dtype=pto.f32)
    loop = pto.for_(0, 1, step=1).carry(acc=acc)
    with loop:
        loop.update(other=acc)


@pto.jit(target="a5")
def carry_final_mismatch_probe(*, BLOCK: pto.const_expr = 8):
    acc = pto.alloc_tile(shape=[1, BLOCK], dtype=pto.f32)
    loop = pto.for_(0, 1, step=1).carry(acc=acc)
    with loop:
        loop.update(acc=acc)
    loop.final("missing")


@pto.jit(target="a5")
def misaligned_row_major_tile_probe():
    pto.alloc_tile(shape=[128, 1], dtype=pto.f32, valid_shape=[128, 1])


class MissingDTypeTensor:
    shape = (4, 8)
    strides = (8, 1)

    def data_ptr(self):
        return 1024


class BadDataHandleTensor:
    shape = (4, 8)
    strides = (8, 1)
    dtype = "float32"

    def data_ptr(self):
        return "not-an-int"


def define_missing_constexpr_default_probe():
    @pto.jit(target="a5")
    def bad_probe(*, BLOCK: pto.const_expr):
        pto.pipe_barrier(pto.Pipe.ALL)

    return bad_probe


def define_illegal_keyword_only_probe():
    @pto.jit(target="a5")
    def bad_probe(*, BLOCK: pto.i32 = 8):
        pto.pipe_barrier(pto.Pipe.ALL)

    return bad_probe


def define_missing_entry_annotation_probe():
    @pto.jit(target="a5")
    def bad_probe(A):
        pto.pipe_barrier(pto.Pipe.ALL)

    return bad_probe


def define_gm_ptr_entry_annotation_probe():
    @pto.jit(target="a5")
    def good_probe(A: pto.ptr(pto.f32, "gm"), rows: pto.i32):
        pto.pipe_barrier(pto.Pipe.ALL)

    return good_probe


def define_default_ptr_entry_annotation_probe():
    @pto.jit(target="a5")
    def bad_probe(A: pto.ptr(pto.f32)):
        pto.pipe_barrier(pto.Pipe.ALL)

    return bad_probe


def define_ub_ptr_entry_annotation_probe():
    @pto.jit(target="a5")
    def bad_probe(A: pto.ptr(pto.f32, "ub")):
        pto.pipe_barrier(pto.Pipe.ALL)

    return bad_probe


def define_emitc_ub_ptr_entry_annotation_probe():
    @pto.jit(target="a5", backend="emitc")
    def bad_probe(A: pto.ptr(pto.f32, "ub")):
        pto.pipe_barrier(pto.Pipe.ALL)

    return bad_probe


def define_frontend_options_conflict_probe():
    @pto.jit(target="a5", ast_rewrite=False, frontend_options={"ast_rewrite": True})
    def bad_probe():
        pto.pipe_barrier(pto.Pipe.ALL)

    return bad_probe


def define_frontend_options_scalar_rewrite_probe():
    @pto.jit(target="a5", frontend_options={"rewrite_part": {"control_flow", "scalar"}})
    def bad_probe():
        pto.pipe_barrier(pto.Pipe.ALL)

    return bad_probe


def define_frontend_options_dump_source_probe():
    @pto.jit(target="a5", frontend_options={"dump_rewritten_source": True})
    def bad_probe():
        pto.pipe_barrier(pto.Pipe.ALL)

    return bad_probe


def define_legacy_tensor_spec_entry_probe():
    @pto.jit(target="a5")
    def bad_probe(A: TensorSpec(rank=2, dtype=pto.f32)):
        pto.pipe_barrier(pto.Pipe.ALL)

    return bad_probe


def define_tile_entry_annotation_probe():
    @pto.jit(target="a5")
    def bad_probe(tile: pto.Tile):
        pto.pipe_barrier(pto.Pipe.ALL)

    return bad_probe


def define_view_entry_annotation_probe():
    @pto.jit(target="a5")
    def bad_probe(view: pto.TensorView):
        pto.pipe_barrier(pto.Pipe.ALL)

    return bad_probe


def define_partition_view_entry_annotation_probe():
    @pto.jit(target="a5")
    def bad_probe(view: pto.PartitionTensorView):
        pto.pipe_barrier(pto.Pipe.ALL)

    return bad_probe


def define_ast_if_undefined_old_value_probe():
    @pto.jit(target="a5")
    def bad_probe():
        cond = pto.const(1, dtype=pto.i1)
        lhs = pto.const(4, dtype=pto.i32)
        if cond:
            value = lhs
        _ = value

    return bad_probe


def define_ast_for_else_probe():
    @pto.jit(target="a5")
    def bad_probe(rows: pto.i32):
        for _ in range(rows):
            pto.pipe_barrier(pto.Pipe.ALL)
        else:
            pto.mem_bar(pto.BarrierType.VST_VLD)

    return bad_probe


def define_ast_for_non_range_probe():
    @pto.jit(target="a5")
    def bad_probe():
        for _ in [0, 1]:
            pto.pipe_barrier(pto.Pipe.ALL)

    return bad_probe


def define_ast_for_tuple_target_probe():
    @pto.jit(target="a5")
    def bad_probe(rows: pto.i32):
        for _, other in range(rows):
            _ = other

    return bad_probe


def define_ast_for_break_probe():
    @pto.jit(target="a5")
    def bad_probe(rows: pto.i32):
        for _ in range(rows):
            break

    return bad_probe


def define_ast_runtime_for_constexpr_break_probe():
    @pto.jit(target="a5")
    def bad_probe(rows: pto.i32):
        for _ in range(rows):
            if pto.const_expr(True):
                break

    return bad_probe


def define_ast_for_last_value_probe():
    @pto.jit(target="a5")
    def bad_probe(rows: pto.i32):
        for i in range(rows):
            last = i
        _ = last

    return bad_probe


def define_ast_for_loop_target_live_after_probe():
    @pto.jit(target="a5")
    def bad_probe(rows: pto.i32):
        for i in range(rows):
            pto.pipe_barrier(pto.Pipe.ALL)
        _ = i

    return bad_probe


def define_helper_missing_annotation_probe():
    @pto.jit(target="a5", entry=False)
    def bad_probe(tile):
        pto.pipe_barrier(pto.Pipe.ALL)

    return bad_probe


def define_helper_constexpr_probe():
    @pto.jit(target="a5", entry=False)
    def bad_probe(*, BLOCK: pto.const_expr = 8):
        pto.pipe_barrier(pto.Pipe.ALL)

    return bad_probe


def define_helper_legacy_tensor_spec_probe():
    @pto.jit(target="a5", entry=False)
    def bad_probe(A: TensorSpec(rank=2, dtype=pto.f32)):
        pto.pipe_barrier(pto.Pipe.ALL)

    return bad_probe


def define_helper_vreg_probe():
    @pto.jit(target="a5", entry=False)
    def bad_probe(vec: pto.vreg_type(64, pto.f32)):
        pto.pipe_barrier(pto.Pipe.ALL)

    return bad_probe


def define_helper_mask_probe():
    @pto.jit(target="a5", entry=False)
    def bad_probe(mask: pto.mask_b32):
        pto.pipe_barrier(pto.Pipe.ALL)

    return bad_probe


def define_kernel_module_return_value_probe():
    @pto.jit(target="a5", entry=False)
    def bad_probe(ptr: pto.ptr(pto.f32, "gm"), rows: pto.i32):
        return rows

    return bad_probe


def define_source_non_string_probe():
    @pto.jit(target="a5", source=123)
    def bad_probe(rows: pto.i32):
        _ = rows

    return bad_probe


def define_source_entry_false_probe():
    @pto.jit(target="a5", entry=False, source="kernel.pto")
    def bad_probe(tile: pto.Tile):
        pto.pipe_barrier(pto.Pipe.ALL)

    return bad_probe


def define_source_constexpr_probe():
    @pto.jit(target="a5", source="kernel.pto")
    def bad_probe(ptr: pto.ptr(pto.f32, "gm"), *, BLOCK: pto.const_expr = 8):
        _ = ptr
        _ = BLOCK

    return bad_probe


@pto.jit(target="a5")
def regular_entry_probe(rows: pto.i32):
    _ = rows


@pto.jit(target="a5")
def entry_calling_entry_probe(rows: pto.i32):
    regular_entry_probe(rows)


@pto.jit(target="a5")
def make_tensor_view_missing_metadata_probe(
    x_ptr: pto.ptr(pto.f32, "gm"),
    rows: pto.i32,
    cols: pto.i32,
):
    _ = rows
    _ = cols
    pto.make_tensor_view(x_ptr)


@pto.jit(target="a5")
def missing_if_branch_probe():
    with pto.if_(pto.const(1, dtype=pto.i1)) as br:
        _ = br


@pto.jit(target="a5")
def stray_if_body_op_probe():
    with pto.if_(pto.const(1, dtype=pto.i1)) as br:
        pto.pipe_barrier(pto.Pipe.ALL)
        with br.then_:
            pto.mem_bar(pto.BarrierType.VST_VLD)


@pto.jit(target="a5")
def assign_outside_branch_probe():
    with pto.if_(pto.const(1, dtype=pto.i1)) as br:
        br.assign(val=pto.const(1, dtype=pto.i32))


@pto.jit(target="a5")
def missing_else_assign_probe():
    lhs = pto.const(4, dtype=pto.i32)
    rhs = pto.const(2, dtype=pto.i32)
    with pto.if_(lhs > rhs) as br:
        with br.then_:
            br.assign(val=lhs)
        with br.else_:
            pto.pipe_barrier(pto.Pipe.ALL)


@pto.jit(target="a5")
def assign_name_mismatch_probe():
    lhs = pto.const(4, dtype=pto.i32)
    rhs = pto.const(2, dtype=pto.i32)
    with pto.if_(lhs > rhs) as br:
        with br.then_:
            br.assign(val=lhs)
        with br.else_:
            br.assign(other=rhs)


@pto.jit(target="a5")
def assign_type_mismatch_probe():
    lhs = pto.const(4, dtype=pto.i32)
    rhs = pto.const(2.0, dtype=pto.f32)
    cond = lhs > pto.const(0, dtype=pto.i32)
    with pto.if_(cond) as br:
        with br.then_:
            br.assign(val=lhs)
        with br.else_:
            br.assign(val=rhs)


@pto.jit(target="a5")
def assign_untyped_literal_without_anchor_probe():
    cond = pto.const(1, dtype=pto.i1)
    with pto.if_(cond) as br:
        with br.then_:
            br.assign(val=1)
        with br.else_:
            br.assign(val=2)
    _ = br.val


@pto.jit(target="a5")
def duplicate_assign_probe():
    lhs = pto.const(4, dtype=pto.i32)
    cond = lhs > pto.const(0, dtype=pto.i32)
    with pto.if_(cond) as br:
        with br.then_:
            br.assign(val=lhs)
            br.assign(val=lhs)
        with br.else_:
            br.assign(val=lhs)


@pto.jit(target="a5")
def unknown_branch_result_probe():
    lhs = pto.const(4, dtype=pto.i32)
    rhs = pto.const(2, dtype=pto.i32)
    with pto.if_(lhs > rhs) as br:
        with br.then_:
            br.assign(val=lhs)
        with br.else_:
            br.assign(val=rhs)
    _ = br.other


@pto.jit(target="a5")
def taddrelu_a5_probe():
    lhs = pto.alloc_tile(shape=[1, 64], dtype=pto.f32)
    rhs = pto.alloc_tile(shape=[1, 64], dtype=pto.f32)
    dst = pto.alloc_tile(shape=[1, 64], dtype=pto.f32)
    pto.tile.addrelu(lhs, rhs, dst)


def main() -> None:
    expect_raises(
        native_python_if_runtime_const_probe.compile,
        TypeError,
        "native Python if/while condition",
        "pto.if_(...)",
        "pto.const_expr",
    )
    expect_raises(
        native_python_range_runtime_metadata_probe.compile,
        TypeError,
        "native Python range()/loop bound",
        "pto.for_(...)",
        "runtime value",
    )
    expect_raises(
        float_loop_bound_probe.compile,
        TypeError,
        "pto.for_(...) loop bound",
        "expects an index or integer runtime scalar",
        "f32",
    )
    expect_raises(
        float_addptr_offset_probe.compile,
        TypeError,
        "addptr(ptr, offset)",
        "expects an index-like scalar",
        "f32",
    )
    expect_raises(
        float_bitwise_probe.compile,
        TypeError,
        "runtime scalar bitwise operator",
        "expects integer-like operands",
    )
    expect_raises(
        float_literal_index_store_probe.compile,
        TypeError,
        "scalar.store(...)",
        "cannot materialize a floating-point literal against non-floating",
        "index",
    )
    expect_raises(
        float_literal_index_binary_probe.compile,
        TypeError,
        "runtime scalar operators cannot materialize a floating-point literal",
        "index",
    )
    expect_raises(
        same_width_float_store_probe.compile,
        TypeError,
        "cannot coerce between different floating-point types of the same width",
        "f16",
        "bf16",
    )
    expect_raises(
        vmi_float_vcadd_missing_reassoc_probe.compile,
        TypeError,
        "pto.vmi.vcadd(...)",
        "floating-point vectors",
        "reassoc",
        "reassoc=True or reassoc=False",
    )
    expect_raises(
        vmi_float_vcadd_none_reassoc_probe.compile,
        TypeError,
        "pto.vmi.vcadd(...)",
        "True or False",
        "received None",
    )
    expect_raises(
        vmi_vbrc_untyped_scalar_probe.compile,
        TypeError,
        "pto.vmi.vbrc(...)",
        "typed scalar",
        "plain Python scalars are ambiguous",
    )
    expect_raises(
        vmi_vci_untyped_scalar_probe.compile,
        TypeError,
        "pto.vmi.vci(...)",
        "typed scalar",
        "plain Python scalars are ambiguous",
    )
    expect_raises(
        vmi_vinterpret_cast_missing_dtype_probe.compile,
        TypeError,
        "pto.vmi.vinterpret_cast(...)",
        "requires to_dtype",
    )
    expect_raises(
        vmi_vinterpret_cast_footprint_mismatch_probe.compile,
        TypeError,
        "pto.vmi.vinterpret_cast(...)",
        "source bit count to be divisible",
    )
    expect_raises(
        vmi_create_mask_group_mismatch_probe.compile,
        ValueError,
        "pto.vmi.create_mask(...)",
        "active_lanes to be <= the inferred group_size",
    )
    expect_raises(
        vmi_vload_mixed_mode_probe.compile,
        TypeError,
        "pto.vmi.vload(...)",
        "dist_mode together with group",
    )
    expect_raises(
        vmi_vload_brc_stride_without_group_probe.compile,
        TypeError,
        "pto.vmi.vload(...)",
        "accepts stride only when group is provided",
    )
    expect_raises(
        vmi_vstore_group_mask_probe.compile,
        TypeError,
        "pto.vmi.vstore(...)",
        "group mode does not take a mask operand",
    )
    expect_raises(
        vmi_vbrc_group_lane_mismatch_probe.compile,
        ValueError,
        "pto.vmi.vbrc(...)",
        "input lane count to match group",
    )
    expect_raises(
        vmi_vload_missing_size_probe.compile,
        TypeError,
        "size",
    )
    expect_raises(
        bool_loop_bound_probe.compile,
        TypeError,
        "pto.for_(...) loop bound",
        "does not accept bool values",
    )
    expect_raises(
        bool_addptr_offset_probe.compile,
        TypeError,
        "addptr(ptr, offset)",
        "does not accept bool values",
    )
    expect_raises(
        bool_event_id_probe.compile,
        TypeError,
        "wait_flag(..., event_id=...)",
        "does not accept bool values",
    )
    expect_raises(
        bool_fixed_integer_probe.compile,
        TypeError,
        "make_mask(..., value)",
        "does not accept bool values",
    )
    expect_raises(
        bool_tile_element_probe.compile,
        TypeError,
        "surface index value",
        "does not accept bool values",
    )
    expect_raises(
        bool_address_sugar_probe.compile,
        TypeError,
        "surface index value",
        "does not accept bool values",
    )
    expect_raises(
        bool_tile_slice_probe.compile,
        TypeError,
        "surface index value",
        "does not accept bool values",
    )
    expect_raises(
        carry_update_mismatch_probe.compile,
        RuntimeError,
        "loop.update(...) must match carry names exactly",
        "missing: acc",
        "unexpected: other",
    )
    expect_raises(
        carry_final_mismatch_probe.compile,
        RuntimeError,
        "loop.final(...) requested unknown carry state 'missing'",
        "expected one of: acc",
    )
    expect_raises(
        misaligned_row_major_tile_probe.compile,
        TypeError,
        "alloc_tile(shape=...) physical row layout is invalid",
        "shape=[128, 1]",
        "row byte size of 4",
        "32-byte aligned",
        "prefer blayout='ColMajor'",
    )
    expect_raises(
        define_missing_constexpr_default_probe,
        TypeError,
        "@pto.jit constexpr parameter 'BLOCK' must declare a default value",
        ".compile(...)",
    )
    expect_raises(
        define_illegal_keyword_only_probe,
        TypeError,
        "@pto.jit keyword-only parameter 'BLOCK' uses unsupported compile-time annotation",
        "pto.const_expr",
        "move runtime data to positional pointer/scalar parameters",
    )
    expect_raises(
        define_missing_entry_annotation_probe,
        TypeError,
        "@pto.jit positional parameter 'A' does not declare an entry ABI annotation",
        'pto.ptr(pto.f32, "gm")',
        "pto.i32/pto.f32/pto.i1",
    )
    gm_ptr_entry_probe = define_gm_ptr_entry_annotation_probe()
    expect(hasattr(gm_ptr_entry_probe, "compile"), "expected explicit GM pointer entry to be accepted")
    expect_raises(
        define_default_ptr_entry_annotation_probe,
        TypeError,
        "@pto.jit positional parameter 'A' uses non-GM pointer entry annotation",
        'pto.ptr(pto.f32, "gm")',
        'spell out "gm" explicitly',
    )
    expect_raises(
        define_ub_ptr_entry_annotation_probe,
        TypeError,
        "@pto.jit positional parameter 'A' uses non-GM pointer entry annotation",
        'pto.ptr(pto.f32, "gm")',
        "only accepts explicit GM pointers",
    )
    expect_raises(
        define_emitc_ub_ptr_entry_annotation_probe,
        TypeError,
        "@pto.jit positional parameter 'A' uses non-GM pointer entry annotation",
        'pto.ptr(pto.f32, "gm")',
        "only accepts explicit GM pointers",
    )
    expect_raises(
        define_legacy_tensor_spec_entry_probe,
        TypeError,
        "@pto.jit positional parameter 'A' still uses legacy host-tensor entry annotation",
        "no longer accepts pto.tensor_spec(...)",
        "pto.make_tensor_view(...)",
    )
    expect_raises(
        define_frontend_options_conflict_probe,
        ValueError,
        "ast_rewrite conflicts with frontend_options",
    )
    expect_raises(
        define_frontend_options_scalar_rewrite_probe,
        ValueError,
        "rewrite_part",
        "currently only supports",
        "scalar",
    )
    expect_raises(
        define_frontend_options_dump_source_probe,
        ValueError,
        "dump_rewritten_source",
        "reserved but not implemented",
    )
    expect_raises(
        lambda: define_ast_if_undefined_old_value_probe().compile(),
        NameError,
        "value",
    )
    expect_raises(
        lambda: define_ast_for_else_probe().compile(),
        PTODSLAstRewriteError,
        "does not support for-else",
    )
    expect_raises(
        lambda: define_ast_for_non_range_probe().compile(),
        PTODSLAstRewriteError,
        "only rewrites for-loops over range(...)",
    )
    expect_raises(
        lambda: define_ast_for_tuple_target_probe().compile(),
        PTODSLAstRewriteError,
        "runtime for-loops require a simple name target",
    )
    expect_raises(
        lambda: define_ast_for_break_probe().compile(),
        PTODSLAstRewriteError,
        "does not support break/continue",
    )
    expect_raises(
        lambda: define_ast_runtime_for_constexpr_break_probe().compile(),
        PTODSLAstRewriteError,
        "does not support break/continue",
    )
    expect_raises(
        lambda: define_ast_for_last_value_probe().compile(),
        PTODSLAstRewriteError,
        "cannot expose last-iteration-only values yet",
        "last",
    )
    expect_raises(
        lambda: define_ast_for_loop_target_live_after_probe().compile(),
        PTODSLAstRewriteError,
        "cannot expose the loop induction variable outside the loop yet",
        "i",
    )
    expect_raises(
        define_tile_entry_annotation_probe,
        TypeError,
        "@pto.jit positional parameter 'tile' uses unsupported entry annotation",
        "Tile",
        "do not belong at the host/kernel entry",
    )
    expect_raises(
        define_view_entry_annotation_probe,
        TypeError,
        "@pto.jit positional parameter 'view' uses unsupported entry annotation",
        "TensorView",
        "do not belong at the host/kernel entry",
    )
    expect_raises(
        define_partition_view_entry_annotation_probe,
        TypeError,
        "@pto.jit positional parameter 'view' uses unsupported entry annotation",
        "PartitionTensorView",
        "do not belong at the host/kernel entry",
    )
    expect_raises(
        define_helper_missing_annotation_probe,
        TypeError,
        "@pto.jit(entry=False) parameter 'tile' does not declare a kernel-module ABI annotation",
        "pto.Tile",
        "typed pto.ptr(...)",
    )
    expect_raises(
        define_helper_constexpr_probe,
        TypeError,
        "@pto.jit(entry=False) keyword-only parameter 'BLOCK' uses unsupported kernel-module compile-time annotation",
        "pto.const_expr",
        "does not support keyword-only constexpr specialization parameters",
    )
    expect_raises(
        define_helper_legacy_tensor_spec_probe,
        TypeError,
        "@pto.jit(entry=False) parameter 'A' still uses legacy host-tensor annotation",
        "does not accept pto.tensor_spec(...)",
        "Tile / TensorView / PartitionTensorView",
    )
    expect_raises(
        define_helper_vreg_probe,
        TypeError,
        "@pto.jit(entry=False) parameter 'vec' uses unsupported kernel-module annotation",
        "VReg",
        "do not belong at this kernel-module boundary",
    )
    expect_raises(
        define_helper_mask_probe,
        TypeError,
        "@pto.jit(entry=False) parameter 'mask' uses unsupported kernel-module annotation",
        "mask",
        "do not belong at this kernel-module boundary",
    )
    expect_raises(
        define_source_non_string_probe,
        TypeError,
        "@pto.jit source must be a filesystem path string when provided",
    )
    expect_raises(
        define_source_entry_false_probe,
        TypeError,
        "@pto.jit(source=...) kernel 'bad_probe' does not support entry=False while source='kernel.pto'",
        "Source-backed JIT is currently limited to launchable entry kernels",
    )
    expect_raises(
        define_source_constexpr_probe,
        TypeError,
        "@pto.jit(source=...) kernel 'bad_probe' does not support keyword-only pto.const_expr parameter 'BLOCK' while source='kernel.pto'",
        "does not template or specialize source text",
    )
    with TemporaryDirectory() as tmpdir:
        source_dir = Path(tmpdir)
        missing_path = source_dir / "missing.pto"

        @pto.jit(target="a5", source=str(missing_path))
        def source_missing_file_probe(ptr: pto.ptr(pto.f32, "gm")):
            raise RuntimeError("source-backed JIT should not trace the Python body")

        expect_raises(
            source_missing_file_probe.compile,
            FileNotFoundError,
            "@pto.jit(source=",
            "missing.pto",
            "file does not exist",
        )

        missing_entry_path = source_dir / "missing_entry.pto"
        missing_entry_path.write_text(
            "module {\n"
            "  func.func @other_entry(%arg0: !pto.ptr<f32, gm>) {\n"
            "    return\n"
            "  }\n"
            "}\n",
            encoding="utf-8",
        )

        @pto.jit(name="wanted_entry", target="a5", source=str(missing_entry_path))
        def source_missing_entry_probe(ptr: pto.ptr(pto.f32, "gm")):
            raise RuntimeError("source-backed JIT should not trace the Python body")

        expect_raises(
            source_missing_entry_probe.compile,
            TypeError,
            "could not bind entry 'wanted_entry'",
            "missing non-declaration func.func",
        )

        ambiguous_entry_path = source_dir / "ambiguous_entry.pto"
        ambiguous_entry_path.write_text(
            "module {\n"
            "  func.func @ambiguous_entry(%arg0: !pto.ptr<f32, gm>) {\n"
            "    return\n"
            "  }\n"
            "  builtin.module {\n"
            "    func.func @ambiguous_entry(%arg0: !pto.ptr<f32, gm>) {\n"
            "      return\n"
            "    }\n"
            "  }\n"
            "}\n",
            encoding="utf-8",
        )

        @pto.jit(target="a5", source=str(ambiguous_entry_path))
        def ambiguous_entry(ptr: pto.ptr(pto.f32, "gm")):
            raise RuntimeError("source-backed JIT should not trace the Python body")

        expect_raises(
            ambiguous_entry.compile,
            TypeError,
            "could not bind entry 'ambiguous_entry'",
            "found 2 matching non-declaration func.func ops",
        )

        count_mismatch_path = source_dir / "count_mismatch.pto"
        count_mismatch_path.write_text(
            "module {\n"
            "  func.func @count_mismatch(%arg0: !pto.ptr<f32, gm>) {\n"
            "    return\n"
            "  }\n"
            "}\n",
            encoding="utf-8",
        )

        @pto.jit(target="a5", source=str(count_mismatch_path))
        def count_mismatch(ptr: pto.ptr(pto.f32, "gm"), rows: pto.i32):
            raise RuntimeError("source-backed JIT should not trace the Python body")

        expect_raises(
            count_mismatch.compile,
            TypeError,
            "ABI mismatch for entry 'count_mismatch'",
            "parameter count differs",
            "expected (!pto.ptr<f32, gm>, i32)",
            "got (!pto.ptr<f32, gm>)",
        )

        type_mismatch_path = source_dir / "type_mismatch.pto"
        type_mismatch_path.write_text(
            "module {\n"
            "  func.func @type_mismatch(%arg0: !pto.ptr<f16, gm>, %arg1: i32) {\n"
            "    return\n"
            "  }\n"
            "}\n",
            encoding="utf-8",
        )

        @pto.jit(target="a5", source=str(type_mismatch_path))
        def type_mismatch(ptr: pto.ptr(pto.f32, "gm"), rows: pto.i32):
            raise RuntimeError("source-backed JIT should not trace the Python body")

        expect_raises(
            type_mismatch.compile,
            TypeError,
            "ABI mismatch for entry 'type_mismatch'",
            "parameter 0 differs",
            "expected !pto.ptr<f32, gm>",
            "got !pto.ptr<f16, gm>",
        )

        non_void_path = source_dir / "non_void.pto"
        non_void_path.write_text(
            "module {\n"
            "  func.func @non_void(%arg0: !pto.ptr<f32, gm>) -> i32 {\n"
            "    %c0 = arith.constant 0 : i32\n"
            "    return %c0 : i32\n"
            "  }\n"
            "}\n",
            encoding="utf-8",
        )

        @pto.jit(target="a5", source=str(non_void_path))
        def non_void(ptr: pto.ptr(pto.f32, "gm")):
            raise RuntimeError("source-backed JIT should not trace the Python body")

        expect_raises(
            non_void.compile,
            TypeError,
            "ABI mismatch for entry 'non_void'",
            "source entry must return no values",
            "i32",
        )

        compile_constexpr_path = source_dir / "compile_constexpr.pto"
        compile_constexpr_path.write_text(
            "module {\n"
            "  func.func @compile_constexpr(%arg0: !pto.ptr<f32, gm>) {\n"
            "    return\n"
            "  }\n"
            "}\n",
            encoding="utf-8",
        )

        @pto.jit(target="a5", source=str(compile_constexpr_path))
        def compile_constexpr(ptr: pto.ptr(pto.f32, "gm")):
            raise RuntimeError("source-backed JIT should not trace the Python body")

        expect_raises(
            lambda: compile_constexpr.compile(BLOCK=8),
            TypeError,
            "@pto.jit(source=...) kernel 'compile_constexpr' does not accept .compile(...) constexpr binding(s) BLOCK",
            "does not template or specialize source text",
        )

        inline_count_mismatch_source = (
            "module {\n"
            "  func.func @inline_count_mismatch(%arg0: !pto.ptr<f32, gm>) {\n"
            "    return\n"
            "  }\n"
            "}\n"
        )

        @pto.jit(target="a5", source=inline_count_mismatch_source)
        def inline_count_mismatch(ptr: pto.ptr(pto.f32, "gm"), rows: pto.i32):
            raise RuntimeError("source-backed JIT should not trace the Python body")

        expect_raises(
            inline_count_mismatch.compile,
            TypeError,
            "ABI mismatch for entry 'inline_count_mismatch'",
            "parameter count differs",
            "<inline pto source",
        )
    kernel_module_return_value_probe = define_kernel_module_return_value_probe()
    expect_raises(
        kernel_module_return_value_probe.compile,
        RuntimeError,
        "@pto.jit(entry=False) kernel module 'bad_probe' is not directly compilable from Python",
        "Compile an entry kernel that calls this module instead",
    )
    expect_raises(
        lambda: kernel_module_return_value_probe[1, None],
        RuntimeError,
        "@pto.jit(entry=False) kernel module 'bad_probe' is not launchable from Python",
        "Only @pto.jit(entry=True) kernels support compiled[grid, stream](...)",
    )
    expect_raises(
        entry_calling_entry_probe.compile,
        TypeError,
        "@pto.jit entry kernel 'regular_entry_probe' is not callable from traced PTODSL bodies",
        "only @pto.jit(entry=False) kernel modules may be called there",
    )
    expect_raises(
        make_tensor_view_missing_metadata_probe.compile,
        TypeError,
        "make_tensor_view(",
        "requires explicit shape= and strides=",
        "Do not rely on host tensor proxy metadata",
    )
    expect_raises(
        missing_if_branch_probe.compile,
        RuntimeError,
        "requires at least one explicit branch block",
        "with br.then_:",
    )
    expect_raises(
        stray_if_body_op_probe.compile,
        RuntimeError,
        "body may only contain explicit 'with br.then_:' / 'with br.else_:' blocks",
        "outer if body",
    )
    expect_raises(
        assign_outside_branch_probe.compile,
        RuntimeError,
        "br.assign(...) may only be used inside br.then_ or br.else_",
    )
    expect_raises(
        missing_else_assign_probe.compile,
        RuntimeError,
        "automatic branch merge requires both br.then_ and br.else_ to call br.assign(...)",
    )
    expect_raises(
        assign_name_mismatch_probe.compile,
        RuntimeError,
        "br.assign(...) names must match across branches",
        "missing in else: val",
        "missing in then: other",
    )
    expect_raises(
        assign_type_mismatch_probe.compile,
        RuntimeError,
        "br.assign(...) type mismatch for 'val'",
    )
    expect_raises(
        assign_untyped_literal_without_anchor_probe.compile,
        TypeError,
        "br.assign(...) cannot infer a PTO type",
        "materialize one side explicitly with pto.const(...)",
    )
    expect_raises(
        duplicate_assign_probe.compile,
        RuntimeError,
        "br.then_ may call br.assign(...) at most once",
    )
    expect_raises(
        unknown_branch_result_probe.compile,
        AttributeError,
        "br.other was not assigned by this conditional",
    )
    expect_raises(
        taddrelu_a5_probe.compile,
        ValueError,
        "pto.tile.addrelu",
        "target='a2'",
        "target='a3'",
        "target='a5'",
    )
    expect_raises(
        lambda: inspect_host_tensor_metadata(MissingDTypeTensor()),
        TypeError,
        "host tensor metadata is incomplete or unsupported",
        "missing .dtype",
    )
    expect_raises(
        lambda: inspect_host_tensor_metadata(BadDataHandleTensor()),
        TypeError,
        "host tensor metadata is incomplete or unsupported",
        "data_ptr must return an integer-like data handle",
    )
    print("ptodsl_jit_diagnostics: PASS")


if __name__ == "__main__":
    main()
