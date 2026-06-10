#!/usr/bin/env python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

from pathlib import Path
import re
import sys


sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "ptodsl"))

from ptodsl import pto, scalar
from ptodsl import _types as pto_types
from ptodsl._bootstrap import make_context
from ptodsl._runtime.cache import artifact_paths
from ptodsl._runtime.codegen import generate_launch_cpp
from ptodsl._runtime.launch import _marshal_launch_args
from ptodsl._tracing import current_session
from mlir.ir import InsertionPoint, Location, Module


def expect(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def expect_raises(exc_type, func, message_substring: str | None = None) -> Exception:
    try:
        func()
    except exc_type as exc:
        if message_substring is not None and message_substring not in str(exc):
            raise AssertionError(
                f"expected {exc_type.__name__} containing {message_substring!r}, got {exc!r}"
            ) from exc
        return exc
    except Exception as exc:
        raise AssertionError(
            f"expected {exc_type.__name__}, got {exc.__class__.__name__}: {exc}"
        ) from exc
    raise AssertionError(f"expected {exc_type.__name__} to be raised")


def expect_parse_roundtrip_and_verify(text: str, label: str) -> None:
    with make_context() as ctx:
        parsed = Module.parse(text, ctx)
        parsed.operation.verify()
        roundtrip_text = str(parsed)
    expect(
        roundtrip_text == text,
        f"{label} should survive Module.parse(...) round-trip without textual drift",
    )


def mlir_op_sequence(text: str) -> list[str]:
    ops = []
    for line in text.splitlines():
        stripped = line.strip()
        match = re.search(r"(?:%[\w#]+(?:\s*:\s*[^=]+)?\s*=\s*)?([a-z][\w]*\.[\w_]+)", stripped)
        if match is not None:
            ops.append(match.group(1))
    return ops


expect_raises(
    TypeError,
    lambda: pto.for_(0, 1, step=1, iter_args=(0,)),
    "iter_args",
)


class _FakeTileWithoutValidShape:
    shape = (1, 16)
    valid_shape = None


class _FakeTileWithPartialValidShape:
    shape = (1, 16)
    valid_shape = [1, None]


expect_raises(
    TypeError,
    lambda: pto.tile.load(object(), _FakeTileWithoutValidShape(), offsets=[0, 0]),
    "requires tile valid_shape metadata",
)
expect_raises(
    ValueError,
    lambda: pto.tile.load(object(), _FakeTileWithPartialValidShape(), offsets=[0, 0]),
    "tile.valid_shape[1] is None",
)


@pto.jit(target="a5")
def host_vec_copy(
    A_ptr: pto.ptr(pto.f32, "gm"),
    O_ptr: pto.ptr(pto.f32, "gm"),
    rows: pto.i32,
    cols: pto.i32,
    *,
    BLOCK: pto.constexpr = 128,
):
    a_view = pto.make_tensor_view(A_ptr, shape=[rows, cols], strides=[cols, 1])
    o_view = pto.make_tensor_view(O_ptr, shape=[rows, cols], strides=[cols, 1])
    a_tile = pto.alloc_tile(shape=[1, BLOCK], dtype=pto.f32)
    o_tile = pto.alloc_tile(shape=[1, BLOCK], dtype=pto.f32)
    part = pto.partition_view(a_view, offsets=[0, 0], sizes=[rows, cols])
    out = pto.partition_view(o_view, offsets=[0, 0], sizes=[rows, cols])
    pto.tile.load(part, a_tile)
    pto.tile.store(o_tile, out)


@pto.jit(target="a5", mode="explicit")
def host_vec_copy_explicit(
    A_ptr: pto.ptr(pto.f32, "gm"),
    O_ptr: pto.ptr(pto.f32, "gm"),
    rows: pto.i32,
    cols: pto.i32,
    *,
    BLOCK: pto.constexpr = 128,
):
    a_view = pto.make_tensor_view(A_ptr, shape=[rows, cols], strides=[cols, 1])
    o_view = pto.make_tensor_view(O_ptr, shape=[rows, cols], strides=[cols, 1])
    a_tile = pto.alloc_tile(shape=[1, BLOCK], dtype=pto.f32)
    o_tile = pto.alloc_tile(shape=[1, BLOCK], dtype=pto.f32)
    part = pto.partition_view(a_view, offsets=[0, 0], sizes=[rows, cols])
    out = pto.partition_view(o_view, offsets=[0, 0], sizes=[rows, cols])
    pto.tile.load(part, a_tile)
    pto.tile.store(o_tile, out)


@pto.jit(target="a5")
def pointer_runtime_shape_specialization_probe(
    x_ptr: pto.ptr(pto.f32, "gm"),
    rows: pto.i32,
    cols: pto.i32,
    row_stride: pto.i32,
    *,
    BLOCK: pto.constexpr = 128,
):
    x_view = pto.make_tensor_view(x_ptr, shape=[rows, cols], strides=[row_stride, 1])
    x_part = pto.partition_view(x_view, offsets=[0, 0], sizes=[rows, cols])
    x_tile = pto.alloc_tile(shape=[1, BLOCK], dtype=pto.f32, valid_shape=[1, cols])
    pto.tile.load(x_part, x_tile)


@pto.jit(target="a5")
def tile_transfer_surface_probe(
    A_ptr: pto.ptr(pto.f32, "gm"),
    O_ptr: pto.ptr(pto.f32, "gm"),
    rows: pto.i32,
    cols: pto.i32,
    *,
    BLOCK: pto.constexpr = 128,
):
    a_view = pto.make_tensor_view(A_ptr, shape=[rows, cols], strides=[cols, 1])
    o_view = pto.make_tensor_view(O_ptr, shape=[rows, cols], strides=[cols, 1])
    a_tile = pto.alloc_tile(shape=[1, BLOCK], dtype=pto.f32, valid_shape=[rows, cols])
    o_tile = pto.alloc_tile(shape=[1, BLOCK], dtype=pto.f32, valid_shape=[rows, cols])
    pto.tile.load(a_view, a_tile)
    pto.tile.store(o_tile, o_view)


@pto.jit(target="a5", insert_sync=False)
def host_vec_copy_no_insert_sync(
    A_ptr: pto.ptr(pto.f32, "gm"),
    O_ptr: pto.ptr(pto.f32, "gm"),
    rows: pto.i32,
    cols: pto.i32,
):
    a_view = pto.make_tensor_view(A_ptr, shape=[rows, cols], strides=[cols, 1])
    o_view = pto.make_tensor_view(O_ptr, shape=[rows, cols], strides=[cols, 1])
    a_tile = pto.alloc_tile(shape=[1, 16], dtype=pto.f32)
    o_tile = pto.alloc_tile(shape=[1, 16], dtype=pto.f32)
    part = pto.partition_view(a_view, offsets=[0, 0], sizes=[rows, cols])
    out = pto.partition_view(o_view, offsets=[0, 0], sizes=[rows, cols])
    pto.tile.load(part, a_tile)
    pto.tile.store(o_tile, out)


@pto.jit(target="a5", mode="explicit", insert_sync=True)
def host_vec_copy_explicit_insert_sync(
    A_ptr: pto.ptr(pto.f32, "gm"),
    O_ptr: pto.ptr(pto.f32, "gm"),
    rows: pto.i32,
    cols: pto.i32,
):
    a_view = pto.make_tensor_view(A_ptr, shape=[rows, cols], strides=[cols, 1])
    o_view = pto.make_tensor_view(O_ptr, shape=[rows, cols], strides=[cols, 1])
    a_tile = pto.alloc_tile(shape=[1, 16], dtype=pto.f32)
    o_tile = pto.alloc_tile(shape=[1, 16], dtype=pto.f32)
    part = pto.partition_view(a_view, offsets=[0, 0], sizes=[rows, cols])
    out = pto.partition_view(o_view, offsets=[0, 0], sizes=[rows, cols])
    pto.tile.load(part, a_tile)
    pto.tile.store(o_tile, out)


@pto.jit(target="a5")
def runtime_metadata_kernel(
    A_ptr: pto.ptr(pto.f32, "gm"),
    O_ptr: pto.ptr(pto.f32, "gm"),
    rows: pto.i32,
    cols: pto.i32,
    row_stride: pto.i32,
    col_stride: pto.i32,
    *,
    BLOCK: pto.constexpr = 128,
):
    a_view = pto.make_tensor_view(A_ptr, shape=[rows, cols], strides=[row_stride, col_stride])
    o_view = pto.make_tensor_view(O_ptr, shape=[rows, cols], strides=[row_stride, col_stride])
    a_tile = pto.alloc_tile(shape=[1, BLOCK], dtype=pto.f32, valid_shape=[rows, cols])
    o_tile = pto.alloc_tile(shape=[1, BLOCK], dtype=pto.f32, valid_shape=[rows, cols])
    part = pto.partition_view(a_view, offsets=[0, 0], sizes=[rows, cols])
    out = pto.partition_view(o_view, offsets=[0, 0], sizes=[rows, cols])
    pto.tile.load(part, a_tile)
    pto.tile.store(o_tile, out)


@pto.jit(target="a5", mode="explicit")
def authored_addr_tile_surface_probe(
    rows: pto.i32,
    cols: pto.i32,
):
    tile = pto.alloc_tile(shape=[1, 128], dtype=pto.f32, addr=0, valid_shape=[rows, cols])
    _ = tile


@pto.jit(target="a5", mode="explicit")
def dynamic_addr_tile_surface_probe(
    rows: pto.i32,
    cols: pto.i32,
):
    tile = pto.alloc_tile(
        shape=[1, 128],
        dtype=pto.f32,
        addr=scalar.index_cast(pto.i64, scalar.index_cast(rows)),
        valid_shape=[rows, cols],
    )
    _ = tile


@pto.jit(target="a5")
def tile_surface_compute_probe():
    lhs = pto.alloc_tile(shape=[2, 16], dtype=pto.f32)
    rhs = pto.alloc_tile(shape=[2, 16], dtype=pto.f32)
    out = pto.alloc_tile(shape=[2, 16], dtype=pto.f32)
    cmp_out = pto.alloc_tile(shape=[2, 32], dtype=pto.i8, valid_shape=[2, 16])

    pto.tile.expands(1.0, lhs)
    pto.tile.expands(2.0, rhs)
    pto.tile.add(lhs, rhs, out)
    pto.tile.adds(out, 3.0, out)
    pto.tile.cmps(out, 0.0, cmp_out, cmp_mode=pto.CmpMode.GT)


SUBKERNEL_OBSERVATIONS = []
INLINE_SUBKERNEL_SCOPE_OBSERVATIONS = []


@pto.simd
def nested_simd_probe():
    session = current_session()
    frame = session.current_subkernel
    SUBKERNEL_OBSERVATIONS.append((frame.role, frame.symbol_name, session.subkernel_stack_depth))


@pto.cube
def top_level_cube_probe():
    session = current_session()
    frame = session.current_subkernel
    SUBKERNEL_OBSERVATIONS.append((frame.role, frame.symbol_name, session.subkernel_stack_depth))


@pto.simd
def top_level_simd_probe():
    session = current_session()
    frame = session.current_subkernel
    SUBKERNEL_OBSERVATIONS.append((frame.role, frame.symbol_name, session.subkernel_stack_depth))


@pto.jit(target="a5")
def shared_subkernel_lowering_probe(*, TRACE_TOKEN: pto.constexpr = 0):
    top_level_cube_probe()
    top_level_simd_probe()
    nested_simd_probe()


@pto.jit(target="a5", mode="explicit")
def inline_subkernel_scope_probe(*, TRACE_TOKEN: pto.constexpr = 0):
    session = current_session()
    meta_tile = pto.alloc_tile(shape=[1, 8], dtype=pto.i32, valid_shape=[1, 1])

    with pto.simt():
        frame = session.current_subkernel
        INLINE_SUBKERNEL_SCOPE_OBSERVATIONS.append((frame.role, frame.symbol_name, session.subkernel_stack_depth))
        scalar.store(0, meta_tile.as_ptr() + 0)
    with pto.simd():
        frame = session.current_subkernel
        INLINE_SUBKERNEL_SCOPE_OBSERVATIONS.append((frame.role, frame.symbol_name, session.subkernel_stack_depth))
        pto.pipe_barrier(pto.Pipe.ALL)
    with pto.cube():
        frame = session.current_subkernel
        INLINE_SUBKERNEL_SCOPE_OBSERVATIONS.append((frame.role, frame.symbol_name, session.subkernel_stack_depth))
        pto.pipe_barrier(pto.Pipe.ALL)


@pto.simt
def simt_tid_probe():
    pto.get_tid_x()
    pto.get_tid_y()
    pto.get_tid_z()


@pto.simd
def ast_subkernel_runtime_for_helper(rows: pto.i32):
    for row in range(0, rows, 1):
        _ = row
        pto.pipe_barrier(pto.Pipe.ALL)


@pto.jit(target="a5")
def simt_helper_lowering_probe(*, TRACE_TOKEN: pto.constexpr = 0):
    simt_tid_probe()
    simt_tid_probe()


@pto.jit(target="a5")
def ast_subkernel_runtime_for_probe(rows: pto.i32):
    ast_subkernel_runtime_for_helper(rows)


@pto.jit(target="a5")
def carry_loop_lowering_probe(*, BLOCK: pto.constexpr = 128):
    m_prev = pto.alloc_tile(shape=[1, BLOCK], dtype=pto.f32)
    l_prev = pto.alloc_tile(shape=[1, BLOCK], dtype=pto.f32)
    o_prev = pto.alloc_tile(shape=[1, BLOCK], dtype=pto.f32)
    m_next = pto.alloc_tile(shape=[1, BLOCK], dtype=pto.f32)
    l_next = pto.alloc_tile(shape=[1, BLOCK], dtype=pto.f32)
    o_next = pto.alloc_tile(shape=[1, BLOCK], dtype=pto.f32)

    m_prev.fill(0.0)
    l_prev.fill(0.0)
    o_prev.fill(0.0)

    m = m_prev
    l = l_prev
    o = o_prev
    for _ in range(0, 4, 1):
        m.fill(1.0)
        l.fill(2.0)
        o.fill(3.0)
        m = m_next
        l = l_next
        o = o_next

    o.fill(4.0)


@pto.jit(target="a5")
def branch_handle_then_only_probe():
    cond = pto.const(1, dtype=pto.i1)
    with pto.if_(cond) as br:
        with br.then_:
            pto.pipe_barrier(pto.Pipe.ALL)


@pto.jit(target="a5")
def branch_handle_side_effect_probe():
    lhs = pto.const(4, dtype=pto.i32)
    rhs = pto.const(2, dtype=pto.i32)
    with pto.if_(lhs > rhs) as br:
        with br.then_:
            pto.pipe_barrier(pto.Pipe.ALL)
        with br.else_:
            pto.mem_bar(pto.BarrierType.VST_VLD)


@pto.jit(target="a5")
def branch_handle_merge_probe():
    lhs = pto.const(4, dtype=pto.i32)
    rhs = pto.const(2, dtype=pto.i32)
    with pto.if_(lhs > rhs) as br:
        with br.then_:
            br.assign(total=lhs + rhs, diff=lhs - rhs)
        with br.else_:
            br.assign(total=rhs + lhs, diff=rhs - lhs)
    total = br.total
    diff = br.diff
    merged = total + diff
    _ = merged


@pto.jit(target="a5")
def ast_if_side_effect_probe():
    cond = pto.const(1, dtype=pto.i1)
    if cond:
        pto.pipe_barrier(pto.Pipe.ALL)


@pto.jit(target="a5")
def ast_if_merge_probe():
    lhs = pto.const(4, dtype=pto.i32)
    rhs = pto.const(2, dtype=pto.i32)
    if lhs > rhs:
        total = lhs + rhs
        diff = lhs - rhs
    else:
        total = rhs + lhs
        diff = rhs - lhs
    merged = total + diff
    _ = merged


@pto.jit(target="a5")
def ast_if_old_value_merge_probe():
    lhs = pto.const(4, dtype=pto.i32)
    rhs = pto.const(2, dtype=pto.i32)
    total = rhs
    if lhs > rhs:
        total = lhs + rhs
    merged = total + rhs
    _ = merged


@pto.jit(target="a5")
def ast_if_branch_local_temp_liveness_probe():
    c0 = pto.const(0, dtype=pto.i1)
    c1 = pto.const(1, dtype=pto.i1)
    one = pto.const(1, dtype=pto.i32)
    zero = pto.const(0, dtype=pto.i32)

    if c0:
        tmp = one
        _ = tmp

    if c1:
        tmp = one
        out = tmp + one
    else:
        out = zero

    _ = out


@pto.jit(target="a5")
def ast_nested_with_if_merge_probe():
    lhs = pto.const(4, dtype=pto.i32)
    rhs = pto.const(2, dtype=pto.i32)
    with pto.simt():
        if lhs > rhs:
            value = lhs + rhs
        else:
            value = rhs + lhs
    merged = value + rhs
    _ = merged


@pto.jit(target="a5")
def ast_runtime_for_probe(rows: pto.i32):
    for row in range(0, rows, 1):
        _ = row
        pto.pipe_barrier(pto.Pipe.ALL)


@pto.jit(target="a5")
def ast_runtime_for_carry_probe(rows: pto.i32):
    one = pto.const(1, dtype=pto.i32)
    acc = pto.const(0, dtype=pto.i32)
    for _ in range(rows):
        acc = acc + one
    _ = acc


@pto.jit(target="a5")
def ast_runtime_for_augassign_carry_probe(rows: pto.i32):
    one = pto.const(1, dtype=pto.i32)
    acc = pto.const(0, dtype=pto.i32)
    for _ in range(rows):
        acc += one
    _ = acc


@pto.jit(target="a5")
def ast_runtime_for_branch_local_temp_probe(rows: pto.i32):
    cond = pto.const(1, dtype=pto.i1)
    one = pto.const(1, dtype=pto.i32)
    zero = pto.const(0, dtype=pto.i32)

    for _ in range(rows):
        if cond:
            tmp = one
            out = tmp + one
        else:
            tmp = zero
            out = tmp + zero
        _ = out


@pto.jit(target="a5", ast_rewrite=False)
def ast_rewrite_disabled_nested_helper_python_control_probe():
    def helper(enabled):
        if enabled:
            for _ in range(2):
                pto.pipe_barrier(pto.Pipe.ALL)

    helper(True)


@pto.jit(target="a5")
def ast_nested_helper_ast_rewrite_probe(rows: pto.i32):
    cond = pto.const(1, dtype=pto.i1)

    def helper(limit, enabled):
        one = pto.const(1, dtype=pto.i32)
        acc = pto.const(0, dtype=pto.i32)
        for _ in range(limit):
            acc += one
        if enabled:
            acc = acc + one
        return acc

    value = helper(rows, cond)
    _ = value


@pto.jit(target="a5")
def ast_nested_helper_freevar_if_merge_probe():
    lhs = pto.const(4, dtype=pto.i32)
    rhs = pto.const(2, dtype=pto.i32)

    if lhs > rhs:
        x = lhs + rhs
    else:
        x = rhs + lhs

    def helper():
        return x + rhs

    y = helper()
    _ = y


@pto.jit(target="a5")
def ast_nested_helper_name_store_liveness_probe():
    lhs = pto.const(4, dtype=pto.i32)
    rhs = pto.const(2, dtype=pto.i32)

    if lhs > rhs:
        helper = lhs
    else:
        helper = rhs

    def helper():
        return lhs + rhs

    y = helper()
    _ = y


@pto.jit(target="a5", name="ast_control_flow_equiv_explicit")
def ast_control_flow_equiv_explicit_probe(rows: pto.i32):
    one = pto.const(1, dtype=pto.i32)
    acc = pto.const(0, dtype=pto.i32)

    def helper(limit, initial):
        total = initial
        loop = pto.for_(0, limit, step=1).carry(total=total)
        with loop:
            i = loop.iv
            total = loop.total
            cond = i > pto.const(0)
            with pto.if_(cond) as br:
                with br.then_:
                    br.assign(total=total + one)
                with br.else_:
                    br.assign(total=total)
            total = br.total
            loop.update(total=total)
        return loop.final("total")

    acc = helper(rows, acc)
    _ = acc


@pto.jit(target="a5", name="ast_control_flow_equiv_native")
def ast_control_flow_equiv_native_probe(rows: pto.i32):
    one = pto.const(1, dtype=pto.i32)
    acc = pto.const(0, dtype=pto.i32)

    def helper(limit, initial):
        total = initial
        for i in range(limit):
            if i > pto.const(0):
                total = total + one
        return total

    acc = helper(rows, acc)
    _ = acc


def make_ast_closure_kernel(limit: int):
    @pto.jit(target="a5")
    def ast_closure_kernel():
        for _ in pto.static_range(limit):
            pto.pipe_barrier(pto.Pipe.ALL)

    return ast_closure_kernel


ast_closure_kernel_probe = make_ast_closure_kernel(3)


def make_ast_rebound_closure_kernel():
    limit = 2

    @pto.jit(target="a5")
    def ast_rebound_closure_kernel():
        for _ in pto.static_range(limit):
            pto.pipe_barrier(pto.Pipe.ALL)

    limit = 4
    return ast_rebound_closure_kernel


ast_rebound_closure_kernel_probe = make_ast_rebound_closure_kernel()


def make_ast_mutable_closure_cache_kernel():
    limit = 2

    @pto.jit(target="a5")
    def ast_mutable_closure_cache_kernel():
        for _ in pto.static_range(limit):
            pto.pipe_barrier(pto.Pipe.ALL)

    def set_limit(value: int):
        nonlocal limit
        limit = value

    return ast_mutable_closure_cache_kernel, set_limit


ast_mutable_closure_cache_kernel_probe, set_ast_mutable_closure_cache_limit = (
    make_ast_mutable_closure_cache_kernel()
)


def make_ast_signature_closure_default_kernel(limit: int):
    @pto.jit(target="a5")
    def ast_signature_closure_default_kernel(*, BLOCK: pto.constexpr = limit):
        for _ in pto.static_range(BLOCK):
            pto.pipe_barrier(pto.Pipe.ALL)

    return ast_signature_closure_default_kernel


ast_signature_closure_default_kernel_probe = make_ast_signature_closure_default_kernel(2)


def make_ast_rebound_subkernel_probe():
    limit = 2

    @pto.simd
    def helper():
        for _ in pto.static_range(limit):
            pto.pipe_barrier(pto.Pipe.ALL)

    limit = 4

    @pto.jit(target="a5")
    def ast_rebound_subkernel_probe(*, TRACE_TOKEN: pto.constexpr = 0):
        helper()

    return ast_rebound_subkernel_probe


ast_rebound_subkernel_probe = make_ast_rebound_subkernel_probe()


def make_sourceless_ast_rewrite_kernel():
    namespace = {"pto": pto}
    exec(
        """
@pto.jit(target="a5")
def sourceless_ast_rewrite_kernel():
    if True:
        pto.pipe_barrier(pto.Pipe.ALL)
""",
        namespace,
    )
    return namespace["sourceless_ast_rewrite_kernel"]


sourceless_ast_rewrite_kernel_probe = make_sourceless_ast_rewrite_kernel()


def make_sourceless_subkernel_entry():
    namespace = {"pto": pto}
    exec(
        """
@pto.simd
def sourceless_subkernel_helper():
    if True:
        pto.pipe_barrier(pto.Pipe.ALL)
""",
        namespace,
    )
    helper = namespace["sourceless_subkernel_helper"]

    @pto.jit(target="a5")
    def sourceless_subkernel_entry_probe(*, TRACE_TOKEN: pto.constexpr = 0):
        helper()

    return sourceless_subkernel_entry_probe


sourceless_subkernel_entry_probe = make_sourceless_subkernel_entry()


@pto.jit(target="a5")
def ast_static_control_flow_probe(*, ENABLE: pto.constexpr = True):
    if pto.const_expr(ENABLE):
        for _ in pto.static_range(2):
            pto.pipe_barrier(pto.Pipe.ALL)


@pto.jit(target="a5")
def ast_python_bool_guard_probe(
    *,
    BLOCK: pto.constexpr = 128,
    ENABLE: pto.constexpr = True,
):
    if BLOCK == 128:
        pto.pipe_barrier(pto.Pipe.ALL)
    if ENABLE:
        pto.pipe_barrier(pto.Pipe.ALL)


@pto.jit(target="a5")
def ast_static_range_loop_target_live_after_probe():
    for stage in pto.static_range(3):
        pto.pipe_barrier(pto.Pipe.ALL)
    _ = stage


@pto.jit(target="a5")
def ast_static_range_break_continue_probe():
    for stage in pto.static_range(4):
        if pto.const_expr(stage == 2):
            break
        pto.pipe_barrier(pto.Pipe.ALL)

    for stage in pto.static_range(4):
        if pto.const_expr(stage % 2 == 0):
            continue
        pto.pipe_barrier(pto.Pipe.ALL)


@pto.jit(target="a5")
def runtime_scalar_operator_probe(
    O_ptr: pto.ptr(pto.f32, "gm"),
    rows: pto.i32,
    cols: pto.i32,
    row_stride: pto.i32,
    *,
    BLOCK: pto.constexpr = 8,
):
    block_idx = pto.get_block_idx()
    o_view = pto.make_tensor_view(O_ptr, shape=[rows, cols], strides=[row_stride, 1])
    o_part = pto.partition_view(o_view, offsets=[0, 0], sizes=[rows, cols])
    o_ptr = o_part.as_ptr()

    batch_idx = block_idx // rows
    head_idx = block_idx % rows
    chunks = (cols + BLOCK - 1) // BLOCK
    tail = cols % BLOCK

    x = pto.const(2.0, dtype=pto.f32)
    y = (x + 1.0) * 2.0
    z = 4.0 - y
    w = 1.0 / z
    m = scalar.max(w, x)
    n = scalar.min(m, x)
    e = scalar.exp(m)
    lg = scalar.log(e)
    rt = scalar.sqrt(e)
    mag = scalar.abs(z)
    gt_zero = m > x
    eq_self = x == x
    in_range = (m >= x) & (m <= e)
    scalar.store(e, o_ptr + 0)

    _ = batch_idx
    _ = head_idx
    _ = chunks
    _ = tail
    _ = w
    _ = m
    _ = n
    _ = e
    _ = lg
    _ = rt
    _ = mag
    _ = gt_zero
    _ = eq_self
    _ = in_range


@pto.jit(target="a5")
def host_runtime_scalar_entry_probe(
    A_ptr: pto.ptr(pto.f32, "gm"),
    O_ptr: pto.ptr(pto.f32, "gm"),
    rows: pto.i32,
    cols: pto.i32,
    limit: pto.i32,
    alpha: pto.f32,
):
    a_view = pto.make_tensor_view(A_ptr, shape=[rows, cols], strides=[cols, 1])
    o_view = pto.make_tensor_view(O_ptr, shape=[rows, cols], strides=[cols, 1])
    a_part = pto.partition_view(a_view, offsets=[0, 0], sizes=[rows, cols])
    o_part = pto.partition_view(o_view, offsets=[0, 0], sizes=[rows, cols])
    a_tile = pto.alloc_tile(shape=[1, 8], dtype=pto.f32, valid_shape=[1, cols])
    o_tile = pto.alloc_tile(shape=[1, 8], dtype=pto.f32, valid_shape=[1, cols])
    pto.tile.load(a_part, a_tile)
    row_limit = limit // pto.const(2, dtype=pto.i32)
    scaled = alpha + 1.0
    _ = row_limit
    _ = scaled
    pto.tile.store(o_tile, o_part)


@pto.simd
def tile_slice_vector_probe(inp_tile: pto.Tile, out_tile: pto.Tile, row: pto.index):
    mask, _ = pto.plt_b32(pto.const(64, dtype=pto.i32))
    vec = pto.vlds(inp_tile[row, 0:])
    pto.vsts(vec, out_tile[row, 0:], mask)


@pto.jit(target="a5")
def tile_slice_surface_probe(*, BLOCK: pto.constexpr = 128):
    inp_tile = pto.alloc_tile(shape=[2, BLOCK], dtype=pto.f32)
    out_tile = pto.alloc_tile(shape=[2, BLOCK], dtype=pto.f32)
    for row in range(0, 1, 1):
        tile_slice_vector_probe(inp_tile, out_tile, row)


@pto.jit(target="a5")
def tile_slice_1d_surface_probe(*, BLOCK: pto.constexpr = 128):
    inp_tile = pto.alloc_tile(shape=[BLOCK], dtype=pto.f32)
    out_tile = pto.alloc_tile(shape=[BLOCK], dtype=pto.f32)
    start = pto.const(0, dtype=pto.i32)
    mask, _ = pto.plt_b32(pto.const(64, dtype=pto.i32))
    align = pto.vldas(inp_tile[start:])
    vec, align = pto.vldus(inp_tile[start:], align)
    pto.vsts(vec, out_tile[start:], mask)
    _ = align


@pto.jit(target="a5")
def tile_valid_shape_update_probe(
    rows: pto.i32,
    cols: pto.i32,
    *,
    BLOCK: pto.constexpr = 128,
):
    tile = pto.alloc_tile(
        shape=[1, BLOCK],
        dtype=pto.f32,
        valid_shape=[pto.const(1), cols],
    )
    tile.valid_shape = [rows, cols]


@pto.jit(target="a5")
def tile_valid_shape_update_1d_probe(
    length: pto.i32,
    *,
    BLOCK: pto.constexpr = 128,
):
    tile = pto.alloc_tile(
        shape=[BLOCK],
        dtype=pto.f32,
        valid_shape=[pto.const(BLOCK)],
    )
    tile.valid_shape = [length]


@pto.jit(target="a5", mode="explicit")
def make_mask_index_roundtrip_probe(
    cols: pto.i32,
):
    remained = cols
    for _ in range(0, cols, 64):
        mask, remained = pto.make_mask(pto.f32, remained)
        _ = mask


@pto.jit(target="a5")
def integer_loop_bound_probe(*, BLOCK: pto.constexpr = 8):
    row_start = pto.const(0, dtype=pto.i32)
    row_stop = pto.const(BLOCK, dtype=pto.i32)
    valid_dim = pto.const(BLOCK // 2, dtype=pto.i32)
    for row in range(row_start, row_stop, 1):
        for col in range(0, valid_dim, 1):
            _ = row
            _ = col


@pto.jit(target="a5")
def scalar_pointer_offset_probe():
    meta_tile = pto.alloc_tile(shape=[1, 8], dtype=pto.i32, valid_shape=[1, 3])
    meta_ptr = meta_tile.as_ptr()
    scalar.store(0, meta_ptr, 0)
    scalar.store(1, meta_ptr, 1)
    scalar.store(2, meta_ptr + 2)
    row_start = scalar.load(meta_ptr, 0)
    row_stop = scalar.load(meta_ptr, 1)
    valid_cols = scalar.load(meta_ptr + 2)
    _ = row_start
    _ = row_stop
    _ = valid_cols


@pto.jit(target="a5")
def addptr_surface_probe():
    meta_tile = pto.alloc_tile(shape=[1, 8], dtype=pto.i32, valid_shape=[1, 4])
    meta_ptr = meta_tile.as_ptr()
    ptr_pyint = pto.addptr(meta_ptr, 2)
    ptr_i32 = pto.addptr(meta_ptr, pto.i32(3))
    scalar.store(11, ptr_pyint)
    scalar.store(13, ptr_i32)
    val_pyint = scalar.load(ptr_pyint)
    val_i32 = scalar.load(ptr_i32)
    _ = val_pyint
    _ = val_i32


@pto.simt
def simt_pointer_offset_helper(meta_ptr: pto.ptr(pto.i32, pto.MemorySpace.UB)):
    scalar.store(7, meta_ptr + 0)
    scalar.store(9, meta_ptr + 1)


@pto.jit(target="a5")
def simt_pointer_offset_probe():
    meta_tile = pto.alloc_tile(shape=[1, 8], dtype=pto.i32, valid_shape=[1, 2])
    simt_pointer_offset_helper(meta_tile.as_ptr())
    first = scalar.load(meta_tile.as_ptr() + 0)
    second = scalar.load(meta_tile.as_ptr() + 1)
    _ = first
    _ = second


@pto.jit(target="a5")
def scalar_store_element_coercion_probe():
    meta_tile = pto.alloc_tile(shape=[1, 8], dtype=pto.i32, valid_shape=[1, 4])
    meta_ptr = meta_tile.as_ptr()
    row_start = pto.const(0)
    row_stop = pto.const(4)
    scalar.store(row_start, meta_ptr + 0)
    scalar.store(row_stop, meta_ptr + 1)
    scalar.store(pto.const(2, dtype=pto.i64), meta_ptr + 2)
    scalar.store(3, meta_ptr + 3)


@pto.simd
def public_vector_surface_probe(inp_tile: pto.Tile, out_tile: pto.Tile, stats_tile: pto.Tile):
    col_mask = pto.make_mask(pto.f32, pto.const(16, dtype=pto.i32))
    row = pto.const(0)
    s_row = pto.vlds(inp_tile[row, 0:])
    row_max = pto.vcgmax(s_row, col_mask)
    s_shifted = pto.vsubs(s_row, row_max, col_mask)
    p_row = pto.vexp(s_shifted, col_mask)
    row_sum = pto.vcgadd(p_row, col_mask)
    pto.vsts(p_row, out_tile[row, 0:], col_mask)
    scalar.store(row_max, stats_tile[row, 0])
    scalar.store(row_sum, stats_tile[row, 1])


@pto.cube
def public_cube_surface_probe(
    lhs_tile: pto.Tile,
    rhs_tile: pto.Tile,
    lhs_tile_mx: pto.Tile,
    rhs_tile_mx: pto.Tile,
    lhs_l0a: pto.Tile,
    rhs_l0b: pto.Tile,
    lhs_l0a_f32: pto.Tile,
    rhs_l0b_f32: pto.Tile,
    lhs_l0a_mx: pto.Tile,
    rhs_l0b_mx: pto.Tile,
    acc_tile: pto.Tile,
    bias_tile: pto.Tile,
    l1_out_tile: pto.Tile,
    out_tile: pto.Tile,
    gm_out_ptr,
):
    m = pto.const(16)
    k = pto.const(16)
    n = pto.const(16)
    start_row = pto.const(2)
    start_col = pto.const(4)
    pto.mte_l1_l0a(
        lhs_tile.as_ptr(),
        lhs_l0a.as_ptr(),
        m,
        k,
        start_row=start_row,
        start_col=start_col,
    )
    pto.mte_l1_l0b(
        rhs_tile.as_ptr(),
        rhs_l0b.as_ptr(),
        k,
        n,
        start_row=start_col,
        start_col=start_row,
        transpose=True,
    )
    pto.mte_l1_l0a_mx(lhs_tile_mx.as_ptr(), lhs_l0a_mx.as_ptr(), m, k, transpose=True)
    pto.mte_l1_l0b_mx(rhs_tile_mx.as_ptr(), rhs_l0b_mx.as_ptr(), k, n, transpose=True)
    pto.mad(
        lhs_l0a.as_ptr(),
        rhs_l0b.as_ptr(),
        acc_tile.as_ptr(),
        m,
        n,
        k,
        unit_flag=pto.MadUnitFlagMode.CHECK_ONLY,
        disable_gemv=True,
        sat=pto.SatMode.OFF,
        n_dir=True,
    )
    pto.mad(lhs_l0a_f32.as_ptr(), rhs_l0b_f32.as_ptr(), acc_tile.as_ptr(), m, n, k, tf32_mode=pto.Tf32Mode.ROUND_EVEN)
    pto.mad_acc(lhs_l0a.as_ptr(), rhs_l0b.as_ptr(), acc_tile.as_ptr(), m, n, k, unit_flag=pto.MadUnitFlagMode.CHECK_AND_SET)
    pto.mad_bias(lhs_l0a.as_ptr(), rhs_l0b.as_ptr(), acc_tile.as_ptr(), bias_tile.as_ptr(), m, n, k, sat=pto.SatMode.ON)
    pto.mad_mx(lhs_l0a_mx.as_ptr(), rhs_l0b_mx.as_ptr(), acc_tile.as_ptr(), m, n, k, unit_flag=pto.MadUnitFlagMode.CHECK_ONLY)
    pto.mad_mx_acc(lhs_l0a_mx.as_ptr(), rhs_l0b_mx.as_ptr(), acc_tile.as_ptr(), m, n, k, disable_gemv=True)
    pto.mad_mx_bias(lhs_l0a_mx.as_ptr(), rhs_l0b_mx.as_ptr(), acc_tile.as_ptr(), bias_tile.as_ptr(), m, n, k, n_dir=True)
    pto.mte_l0c_l1(
        acc_tile.as_ptr(),
        l1_out_tile.as_ptr(),
        m,
        n,
        n,
        n,
        unit_flag=pto.AccStoreUnitFlagCtrl.CHECK_ONLY,
        layout="nz2nd",
        loop3=(1, n, n),
        sat=pto.SatMode.ON,
    )
    pto.mte_l0c_gm(
        acc_tile.as_ptr(),
        gm_out_ptr,
        m,
        n,
        n,
        n,
        0,
        0,
        layout=("nz2dn", n),
        loop3=(1, n, n),
        sat=pto.SatMode.OFF,
        atomic=("f32", "add"),
    )
    pto.mte_l0c_ub(acc_tile.as_ptr(), out_tile.as_ptr(), m, n, n, n, 0)
    pto.mte_l0c_ub(acc_tile.as_ptr(), out_tile.as_ptr(), m, n, n, n, split=pto.SplitMode.M, layout="nz2nd")
    pto.mte_l0c_ub(acc_tile.as_ptr(), out_tile.as_ptr(), m, n, n, n, 1, layout=("nz2nz", 1), sat=pto.SatMode.PRESERVE_NAN)


@pto.jit(target="a5", mode="explicit")
def public_surface_exports_probe(
    A_ptr: pto.ptr(pto.f32, "gm"),
    O_ptr: pto.ptr(pto.f32, "gm"),
    rows: pto.i32,
    cols: pto.i32,
):
    pto.mem_bar(pto.BarrierType.VST_VLD)
    pto.pipe_barrier(pto.Pipe.ALL)

    a_view = pto.make_tensor_view(A_ptr, shape=[rows, cols], strides=[cols, 1])
    o_view = pto.make_tensor_view(O_ptr, shape=[rows, cols], strides=[cols, 1])
    a_part = pto.partition_view(a_view, offsets=[0, 0], sizes=[1, 16])
    o_part = pto.partition_view(o_view, offsets=[0, 0], sizes=[1, 16])
    dma_tile = pto.alloc_tile(shape=[1, 128], dtype=pto.f32, valid_shape=[1, 16])
    pto.mte_load(a_part.as_ptr(), dma_tile.as_ptr(), 0, 64, nburst=(1, 0, 128))
    pto.pipe_barrier(pto.Pipe.ALL)
    pto.mte_store(dma_tile.as_ptr(), o_part.as_ptr(), 64, nburst=(1, 128, 0))

    vec_in = pto.alloc_tile(shape=[1, 128], dtype=pto.f32, valid_shape=[1, 16])
    vec_out = pto.alloc_tile(shape=[1, 128], dtype=pto.f32, valid_shape=[1, 16])
    stats_tile = pto.alloc_tile(shape=[1, 8], dtype=pto.f32, valid_shape=[1, 2])
    public_vector_surface_probe(vec_in, vec_out, stats_tile)

    lhs_tile = pto.alloc_tile(
        shape=[16, 16],
        dtype=pto.f16,
        memory_space=pto.MemorySpace.MAT,
        valid_shape=[16, 16],
    )
    rhs_tile = pto.alloc_tile(
        shape=[16, 16],
        dtype=pto.f16,
        memory_space=pto.MemorySpace.MAT,
        valid_shape=[16, 16],
    )
    lhs_tile_mx = pto.alloc_tile(
        shape=[16, 32],
        dtype=pto.f8e4m3,
        memory_space=pto.MemorySpace.MAT,
        valid_shape=[16, 16],
    )
    rhs_tile_mx = pto.alloc_tile(
        shape=[16, 32],
        dtype=pto.f8e4m3,
        memory_space=pto.MemorySpace.MAT,
        valid_shape=[16, 16],
    )
    lhs_l0a = pto.alloc_tile(
        shape=[16, 16],
        dtype=pto.f16,
        memory_space=pto.MemorySpace.LEFT,
        valid_shape=[16, 16],
    )
    rhs_l0b = pto.alloc_tile(
        shape=[16, 16],
        dtype=pto.f16,
        memory_space=pto.MemorySpace.RIGHT,
        valid_shape=[16, 16],
    )
    lhs_l0a_f32 = pto.alloc_tile(
        shape=[16, 16],
        dtype=pto.f32,
        memory_space=pto.MemorySpace.LEFT,
        valid_shape=[16, 16],
    )
    rhs_l0b_f32 = pto.alloc_tile(
        shape=[16, 16],
        dtype=pto.f32,
        memory_space=pto.MemorySpace.RIGHT,
        valid_shape=[16, 16],
    )
    lhs_l0a_mx = pto.alloc_tile(
        shape=[16, 32],
        dtype=pto.f8e4m3,
        memory_space=pto.MemorySpace.LEFT,
        valid_shape=[16, 16],
    )
    rhs_l0b_mx = pto.alloc_tile(
        shape=[16, 32],
        dtype=pto.f8e4m3,
        memory_space=pto.MemorySpace.RIGHT,
        valid_shape=[16, 16],
    )
    acc_tile = pto.alloc_tile(
        shape=[16, 16],
        dtype=pto.f32,
        memory_space=pto.MemorySpace.ACC,
        valid_shape=[16, 16],
    )
    bias_tile = pto.alloc_tile(
        shape=[16, 16],
        dtype=pto.f32,
        memory_space=pto.MemorySpace.BIAS,
        valid_shape=[16, 16],
    )
    l1_out = pto.alloc_tile(
        shape=[16, 16],
        dtype=pto.f32,
        memory_space=pto.MemorySpace.MAT,
        valid_shape=[16, 16],
    )
    cube_out = pto.alloc_tile(shape=[16, 16], dtype=pto.f32, valid_shape=[16, 16])
    public_cube_surface_probe(
        lhs_tile,
        rhs_tile,
        lhs_tile_mx,
        rhs_tile_mx,
        lhs_l0a,
        rhs_l0b,
        lhs_l0a_f32,
        rhs_l0b_f32,
        lhs_l0a_mx,
        rhs_l0b_mx,
        acc_tile,
        bias_tile,
        l1_out,
        cube_out,
        O_ptr,
    )


@pto.jit(target="a5")
def compile_time_query_probe():
    f32_bw = pto.bytewidth(pto.f32)
    f16_bw = pto.bytewidth(pto.f16)
    i8_bw = pto.bytewidth(pto.i8)
    f32_vec = pto.elements_per_vreg(pto.f32)
    f16_vec = pto.elements_per_vreg(pto.f16)
    i8_vec = pto.elements_per_vreg(pto.i8)

    expect(f32_bw == 4, "pto.bytewidth(pto.f32) should evaluate to 4")
    expect(f16_bw == 2, "pto.bytewidth(pto.f16) should evaluate to 2")
    expect(i8_bw == 1, "pto.bytewidth(pto.i8) should evaluate to 1")
    expect(f32_vec == 64, "pto.elements_per_vreg(pto.f32) should evaluate to 64")
    expect(f16_vec == 128, "pto.elements_per_vreg(pto.f16) should evaluate to 128")
    expect(i8_vec == 256, "pto.elements_per_vreg(pto.i8) should evaluate to 256")


@pto.jit(target="a5")
def eager_scalar_constructor_probe():
    i32_val = pto.i32(1024)
    ui16_val = pto.ui16(7)
    si32_val = pto.si32(-7)
    ui8_val = pto.ui8(255)
    si8_neg1 = pto.si8(-1)
    ui8_bits = pto.ui8("0xFF")
    si16_bits = pto.si16("0xFFFF")
    f16_val = pto.f16(-1.5)
    f32_val = pto.f32("inf")
    f16_bits = pto.f16("0xFC00")
    i32_bits = pto.i32("0x80000000")

    _ = i32_val
    _ = ui16_val
    _ = si32_val
    _ = ui8_val
    _ = si8_neg1
    _ = ui8_bits
    _ = si16_bits
    _ = f16_val
    _ = f32_val
    _ = f16_bits
    _ = i32_bits


@pto.jit(target="a5")
def signed_integer_scalar_probe():
    signed_tile = pto.alloc_tile(shape=[1, 8], dtype=pto.si32, valid_shape=[1, 3])
    unsigned_tile = pto.alloc_tile(shape=[1, 8], dtype=pto.ui32, valid_shape=[1, 3])

    signed_ptr = signed_tile.as_ptr()
    unsigned_ptr = unsigned_tile.as_ptr()

    scalar.store(pto.si32(-7), signed_ptr + 0)
    scalar.store(pto.si32(5), signed_ptr + 1)
    scalar.store(pto.ui32("0xFFFFFFFF"), unsigned_ptr + 0)
    scalar.store(pto.ui32(9), unsigned_ptr + 1)

    s0 = scalar.load(signed_ptr + 0)
    s1 = scalar.load(signed_ptr + 1)
    u0 = scalar.load(unsigned_ptr + 0)
    u1 = scalar.load(unsigned_ptr + 1)

    s_add = s0 + 1
    u_add = u1 + 2
    s_max = scalar.max(s0, s1)
    s_min = scalar.min(s0, s1)
    u_max = scalar.max(u0, u1)
    u_min = scalar.min(u0, u1)
    s_abs = scalar.abs(s0)
    u_abs = scalar.abs(u0)
    s_cmp = s1 > s0
    u_cmp = u0 > u1

    scalar.store(s_add, signed_ptr + 2)
    scalar.store(u_add, unsigned_ptr + 2)

    _ = s_max
    _ = s_min
    _ = u_max
    _ = u_min
    _ = s_abs
    _ = u_abs
    _ = s_cmp
    _ = u_cmp


@pto.jit(target="a5")
def low_precision_storage_probe():
    lp_tile = pto.alloc_tile(shape=[128, 64], dtype=pto.f8e4m3)
    lp_tile_hif8 = pto.alloc_tile(shape=[64, 64], dtype=pto.hif8)
    lp_tile_ty = pto_types.tile_buf_type([16, 16], pto.f4e2m1x2, [16, 16])
    _ = lp_tile
    _ = lp_tile_hif8
    _ = lp_tile_ty


@pto.jit(target="a5")
def pointer_vlds_inference_probe(*, BLOCK: pto.constexpr = 128):
    tile = pto.alloc_tile(shape=[2, BLOCK], dtype=pto.f32)
    vec = pto.vlds(tile.as_ptr(), pto.const(0))
    vec_brc = pto.vlds(tile.as_ptr(), pto.const(0), dist="BRC_B32")
    ivec = pto.vbitcast(vec, pto.i32)
    f16_vec = pto.vbitcast(vec, pto.f16)
    _ = vec
    _ = vec_brc
    _ = ivec
    _ = f16_vec


@pto.jit(target="a5")
def public_mask_bitcast_probe():
    mask_b8, _ = pto.make_mask(pto.i8, pto.const(256, dtype=pto.i32))
    mask_b16 = pto.pbitcast(mask_b8, pto.mask_b16)
    mask_b32 = pto.pbitcast(mask_b16, pto.mask_b32)
    _ = mask_b8
    _ = mask_b16
    _ = mask_b32


@pto.jit(target="a5")
def public_mask_surface_probe():
    remained = pto.const(16, dtype=pto.i32)

    mask8_full = pto.pset_b8(pto.MaskPattern.ALL)
    mask16_full = pto.pset_b16(pto.MaskPattern.ALL)
    mask32_full = pto.pset_b32(pto.MaskPattern.ALL)
    mask8_prefix = pto.pge_b8(pto.MaskPattern.VL32)
    mask16_prefix = pto.pge_b16(pto.MaskPattern.VL16)
    mask32_prefix = pto.pge_b32(pto.MaskPattern.VL8)

    mask8_tail, remained8 = pto.plt_b8(remained)
    mask16_tail, remained16 = pto.plt_b16(remained)
    mask32_tail, remained32 = pto.plt_b32(remained)

    merged = pto.pand(mask32_full, mask32_prefix, mask32_full)
    union = pto.por(mask32_full, mask32_prefix, mask32_full)
    flipped = pto.pxor(mask32_full, mask32_prefix, mask32_full)
    inverted = pto.pnot(mask32_prefix, mask32_full)
    selected = pto.psel(mask32_full, mask32_prefix, mask32_tail)

    packed = pto.ppack(mask32_full, pto.PredicatePart.LOWER)
    unpacked = pto.punpack(packed, pto.PredicatePart.LOWER)
    packed_hi = pto.ppack(mask32_full, pto.PredicatePart.HIGHER)
    unpacked_hi = pto.punpack(packed_hi, pto.PredicatePart.HIGHER)
    packed_hi_b16 = pto.ppack(mask32_full, pto.PredicatePart.HIGHER, to_type=pto.mask_b16)
    unpacked_hi_b32 = pto.punpack(packed_hi_b16, pto.PredicatePart.HIGHER, to_type=pto.mask_b32)
    lo8, hi8 = pto.pintlv_b8(mask8_full, mask8_prefix)
    dlo8, dhi8 = pto.pdintlv_b8(lo8, hi8)
    lo16, hi16 = pto.pintlv_b16(mask16_full, mask16_prefix)
    dlo16, dhi16 = pto.pdintlv_b16(lo16, hi16)
    lo32, hi32 = pto.pintlv_b32(mask32_full, mask32_prefix)
    dlo32, dhi32 = pto.pdintlv_b32(lo32, hi32)

    vec_tile = pto.alloc_tile(shape=[1, 64], dtype=pto.f32, valid_shape=[1, 64])
    scores = pto.vlds(vec_tile.as_ptr(), pto.const(0))
    cmp_eq = pto.vcmp(scores, scores, mask32_full, pto.CmpMode.EQ)
    cmp_gt = pto.vcmps(scores, pto.f32(0.0), mask32_full, pto.CmpMode.GT)

    mask8_buf = pto.alloc_tile(shape=[1, 64], dtype=pto.ui8, valid_shape=[1, 64])
    mask16_buf = pto.alloc_tile(shape=[1, 64], dtype=pto.ui16, valid_shape=[1, 64])
    mask32_buf = pto.alloc_tile(shape=[1, 64], dtype=pto.ui32, valid_shape=[1, 64])
    pto.psts(mask8_full, mask8_buf.as_ptr(), pto.const(0), dist=pto.PredicateDist.NORM)
    pto.psts(mask16_full, mask16_buf.as_ptr(), pto.const(0), dist=pto.PredicateDist.PK)
    loaded8 = pto.plds(mask8_buf.as_ptr(), pto.const(0), dist=pto.PredicateDist.NORM)
    loaded16 = pto.plds(mask16_buf.as_ptr(), pto.const(0), dist=pto.PredicateDist.US)
    loaded32 = pto.plds(mask32_buf.as_ptr(), pto.const(0), dist=pto.PredicateDist.DS)

    align0 = pto.init_align()
    align1, base1 = pto.pstu(align0, mask16_full, mask16_buf.as_ptr())
    align2, base2 = pto.pstu(align1, mask32_full, mask32_buf.as_ptr())

    _ = mask8_tail
    _ = mask16_tail
    _ = mask32_tail
    _ = remained8
    _ = remained16
    _ = remained32
    _ = merged
    _ = union
    _ = flipped
    _ = inverted
    _ = selected
    _ = packed
    _ = unpacked
    _ = packed_hi
    _ = unpacked_hi
    _ = packed_hi_b16
    _ = unpacked_hi_b32
    _ = dlo8
    _ = dhi8
    _ = dlo16
    _ = dhi16
    _ = dlo32
    _ = dhi32
    _ = cmp_eq
    _ = cmp_gt
    _ = loaded8
    _ = loaded16
    _ = loaded32
    _ = align2
    _ = base1
    _ = base2


@pto.jit(target="a5")
def public_sync_surface_probe():
    dynamic_event = pto.const(3)
    pto.get_buf(pto.Pipe.V, 0)
    pto.rls_buf(pto.Pipe.MTE2, 1, 2)
    pto.set_flag(pto.Pipe.MTE2, pto.Pipe.V, event_id=0)
    pto.wait_flag(pto.Pipe.MTE2, pto.Pipe.V, event_id=0)
    pto.set_flag(pto.Pipe.V, pto.Pipe.MTE3, event_id=dynamic_event)
    pto.wait_flag(pto.Pipe.V, pto.Pipe.MTE3, event_id=dynamic_event)
    pto.set_cross_flag(pto.Pipe.FIX, 0)
    pto.set_intra_flag(pto.Pipe.MTE3, dynamic_event)
    pto.wait_cross_flag(pto.Pipe.FIX, 0)
    pto.wait_intra_flag(pto.Pipe.V, dynamic_event)


@pto.jit(target="a5", mode="explicit")
def public_data_movement_surface_probe():
    zero_u64 = pto.const(0, dtype=pto.ui64)
    gm_src = pto.castptr(zero_u64, pto.ptr(pto.f16, "gm"))
    gm_dst = pto.castptr(zero_u64, pto.ptr(pto.f16, "gm"))
    ub_src = pto.castptr(zero_u64, pto.ptr(pto.f16, "ub"))
    ub_dst = pto.castptr(zero_u64, pto.ptr(pto.f16, "ub"))
    l1_dst = pto.castptr(zero_u64, pto.ptr(pto.f16, "mat"))
    bias_dst = pto.castptr(zero_u64, pto.ptr(pto.f32, pto.MemorySpace.BIAS))
    scaling_dst = pto.castptr(zero_u64, pto.ptr(pto.f32, pto.MemorySpace.SCALING))
    ub_src_f32 = pto.castptr(zero_u64, pto.ptr(pto.f32, "ub"))
    ub_dst_f32 = pto.castptr(zero_u64, pto.ptr(pto.f32, "ub"))

    pto.mte_gm_ub(gm_src, ub_dst, 0, 256, nburst=(8, 256, 256), loops=[(4, 2048, 2048)])
    pto.mte_gm_ub(gm_src, ub_dst, 0, 200, nburst=(64, 200, 256), pad=(0.0, 0, 0))
    pto.mte_ub_gm(ub_src, gm_dst, 256, nburst=(64, 256, 1024))
    pto.mte_ub_ub(ub_src, ub_dst, 8, nburst=(16, 0, 4))
    pto.mte_ub_l1(ub_src, l1_dst, 8, nburst=(16, 0, 4))
    pto.mte_gm_l1(gm_src, l1_dst, 256, nburst=(8, 256, 256), loops=[(2, 2048, 2048)])
    pto.mte_l1_ub(l1_dst, ub_dst, 256, nburst=(8, 256, 256), loops=[(2, 2048, 2048)])
    pto.mte_gm_l1_frac(
        gm_src,
        l1_dst,
        pto.FractalMode.ND2NZ,
        shape=(16, 16),
        src_layout=(16, 256),
        dst_group=(1, 0, 0, 0),
        ctrl=(0, False),
    )
    pto.mte_l1_bt(l1_dst, bias_dst, 8, nburst=(1, 0, 0))
    pto.mte_l1_fb(l1_dst, scaling_dst, 8, nburst=(1, 0, 0))

    load_align0 = pto.vldas(ub_src)
    vec0, load_align1 = pto.vldus(ub_src, load_align0)
    low0, high0 = pto.vldsx2(ub_src, pto.const(0), pto.DeinterleaveDist.DINTLV_B16)
    store_align0 = pto.init_align()
    store_align1 = pto.vstur(store_align0, vec0, ub_dst, pto.PostUpdate.OFF)
    pto.vstar(store_align1, ub_dst)

    store_align2 = pto.init_align()
    store_align3 = pto.vstus(store_align2, pto.const(32), vec0, ub_dst)
    pto.vstas(store_align3, ub_dst, pto.const(64))

    vec_f32 = pto.vlds(ub_src_f32, pto.const(0))
    offsets_i16 = pto.vbitcast(vec0, pto.i16)
    offsets_i32 = pto.vbitcast(vec_f32, pto.i32)
    mask16_full = pto.pset_b16(pto.MaskPattern.ALL)
    mask32_full = pto.pset_b32(pto.MaskPattern.ALL)
    gather0 = pto.vgather2(ub_src_f32, offsets_i32, mask32_full)
    gather1 = pto.vgather2_bc(ub_src_f32, offsets_i32, mask32_full)
    gatherb = pto.vgatherb(ub_src_f32, offsets_i32, mask32_full)
    pto.vscatter(gather0, ub_dst_f32, offsets_i32, mask32_full)
    blocked = pto.vsldb(ub_src, pto.i16(32), pto.i16(0), mask32_full)
    pto.vsstb(vec0, ub_dst, pto.i16(32), pto.i16(0), mask32_full)
    pto.vstsx2(low0, high0, ub_dst, pto.const(0), pto.InterleaveDist.INTLV_B16, mask16_full)

    _ = vec0
    _ = vec_f32
    _ = load_align1
    _ = low0
    _ = high0
    _ = gather0
    _ = gather1
    _ = gatherb
    _ = blocked


@pto.jit(target="a5", mode="explicit")
def vector_post_update_surface_probe():
    zero_u64 = pto.const(0, dtype=pto.ui64)
    ub_src = pto.castptr(zero_u64, pto.ptr(pto.f32, "ub"))
    ub_dst = pto.castptr(zero_u64, pto.ptr(pto.f32, "ub"))
    mask32_full = pto.pset_b32(pto.MaskPattern.ALL)

    vec, load_base = pto.vlds(ub_src, pto.const(0), return_updated_base=True)
    store_base = pto.vsts(vec, ub_dst, pto.const(0), mask32_full, return_updated_base=True)
    block_base = pto.vsstb(
        vec,
        ub_dst,
        pto.i16(32),
        pto.i16(0),
        mask32_full,
        return_updated_base=True,
    )

    _ = load_base
    _ = store_base
    _ = block_base


@pto.jit(name="pipe_tile_metadata_probe", target="a5", kernel_kind="vector")
def pipe_tile_metadata_probe():
    c2v_buf = pto.reserve_buffer("c2v_fifo", size=8192, location="vec")
    c2v = pto.pipe.c2v(slot_size=1024, consumer_buf=c2v_buf, id=0)
    dst_type = pto.alloc_tile(shape=[16, 16], dtype=pto.f32)
    dst_type_f4e2 = pto.alloc_tile(shape=[16, 32], dtype=pto.f4e2m1x2)
    dst_type_f4e1 = pto.alloc_tile(shape=[16, 32], dtype=pto.f4e1m2x2)

    @pto.simd
    def _inner():
        c2v.init_simd()
        fifo_tile = c2v.pop(result_type=dst_type, split=0)
        # Slicing requires tile shape metadata parsing
        val = pto.vlds(fifo_tile[0, 0:])
        c2v.free(split=0)

        fifo_tile_f4e2 = c2v.pop(result_type=dst_type_f4e2, split=0)
        _ = fifo_tile_f4e2[0, 0:]
        c2v.free(split=0)

        fifo_tile_f4e1 = c2v.pop(result_type=dst_type_f4e1, split=0)
        _ = fifo_tile_f4e1[0, 0:]
        c2v.free(split=0)

    _inner()


@pto.jit(target="a5")
def auto_mode_explicit_surface_violation_probe():
    zero_u64 = pto.const(0, dtype=pto.ui64)
    gm_src = pto.castptr(zero_u64, pto.ptr(pto.f16, "gm"))
    ub_dst = pto.castptr(zero_u64, pto.ptr(pto.f16, "ub"))
    pto.mte_gm_ub(gm_src, ub_dst, 0, 256, nburst=(8, 256, 256))


class _FakeTensor:
    def __init__(self, shape):
        self.shape = tuple(shape)

    def new_empty(self, shape):
        return _FakeTensor(shape)


def main() -> None:
    expected_public_exports = [
        "f8e4m3",
        "f8e5m2",
        "hif8",
        "f4e1m2x2",
        "f4e2m1x2",
        "si8",
        "si16",
        "si32",
        "si64",
        "ui8",
        "ui16",
        "ui32",
        "ui64",
        "i32",
        "f16",
        "f32",
        "bytewidth",
        "elements_per_vreg",
        "make_mask",
        "mask_b8",
        "mask_b16",
        "mask_b32",
        "MaskPattern",
        "CmpMode",
        "PredicatePart",
        "PredicateDist",
        "VStoreDist",
        "DeinterleaveDist",
        "InterleaveDist",
        "PostUpdate",
        "AlignType",
        "init_align",
        "plt_b8",
        "plt_b16",
        "pset_b8",
        "pset_b16",
        "pge_b8",
        "pge_b16",
        "pge_b32",
        "pand",
        "por",
        "pxor",
        "pnot",
        "psel",
        "pbitcast",
        "ppack",
        "punpack",
        "pintlv_b8",
        "pintlv_b16",
        "pintlv_b32",
        "pdintlv_b8",
        "pdintlv_b16",
        "pdintlv_b32",
        "vcmp",
        "vcmps",
        "plds",
        "psts",
        "pstu",
        "vbitcast",
        "vexp",
        "vcgmax",
        "vcgadd",
        "vsubs",
        "mte_load",
        "mte_store",
        "mem_bar",
        "BarrierType",
        "Pipe",
        "pipe_barrier",
        "get_buf",
        "rls_buf",
        "set_cross_flag",
        "wait_cross_flag",
        "set_intra_flag",
        "wait_intra_flag",
        "mte_gm_ub",
        "mte_ub_gm",
        "mte_ub_ub",
        "mte_ub_l1",
        "mte_gm_l1",
        "mte_l1_ub",
        "mte_gm_l1_frac",
        "mte_l1_bt",
        "mte_l1_fb",
        "vldsx2",
        "vldas",
        "vldus",
        "vstsx2",
        "vgather2",
        "vgather2_bc",
        "vgatherb",
        "vscatter",
        "vsldb",
        "vsstb",
        "vstar",
        "vstas",
        "vstur",
        "vstus",
        "mte_l1_l0a",
        "mte_l1_l0b",
        "mte_l1_l0a_mx",
        "mte_l1_l0b_mx",
        "mte_l0c_l1",
        "mte_l0c_gm",
        "mte_l0c_ub",
        "mad",
        "mad_acc",
        "mad_bias",
        "mad_mx",
        "mad_mx_acc",
        "mad_mx_bias",
        "empty_like",
    ]
    for name in expected_public_exports:
        expect(hasattr(pto, name), f"pto.{name} should be exported from the public namespace")

    fake_tensor = _FakeTensor((2, 3, 4))
    fake_empty = pto.empty_like(fake_tensor)
    expect(isinstance(fake_empty, _FakeTensor), "pto.empty_like(...) should preserve host tensor factory type")
    expect(fake_empty.shape == fake_tensor.shape, "pto.empty_like(...) should preserve the logical tensor shape")
    expect(not hasattr(pto, "scalar"), "pto.scalar should not remain in the public pto namespace")
    expect(hasattr(pto, "tile"), "pto.tile should be exported from the public namespace")
    expect(hasattr(pto.tile, "load"), "pto.tile.load should be exported from the public tile namespace")
    expect(hasattr(pto.tile, "add"), "pto.tile.add should be exported from the public tile namespace")
    expect(hasattr(pto.tile, "cmps"), "pto.tile.cmps should be exported from the public tile namespace")
    expect(not hasattr(pto, "load_tile"), "pto.load_tile should not remain on the public pto namespace")
    expect(not hasattr(pto, "store_tile"), "pto.store_tile should not remain on the public pto namespace")
    expect(not hasattr(pto, "tload"), "legacy pto.tload should not remain on the public pto namespace")
    expect(not hasattr(pto, "tstore"), "legacy pto.tstore should not remain on the public pto namespace")
    expect(not hasattr(pto, "tadd"), "legacy pto.tadd should not remain on the public pto namespace")
    expect(not hasattr(pto, "tile_buf_type"), "pto.tile_buf_type should not remain on the public pto namespace")
    expect(not hasattr(pto, "vecscope"), "pto.vecscope should not remain on the public pto namespace")
    expect(not hasattr(pto, "as_ptr"), "pto.as_ptr should not remain on the public pto namespace")
    expect(not hasattr(pto, "vbrc_load"), "pto.vbrc_load should not remain on the public pto namespace")
    expect(not hasattr(pto, "vsts_1pt"), "pto.vsts_1pt should not remain on the public pto namespace")
    expect(not hasattr(scalar, "sts"), "scalar.sts should not remain in the public scalar namespace")
    expect(not hasattr(scalar, "cmpi"), "scalar.cmpi should not remain in the public scalar namespace")
    expect(not hasattr(scalar, "cmpi_sgt"), "scalar.cmpi_sgt should not remain in the public scalar namespace")
    removed_tile_buf_type = expect_raises(AttributeError, lambda: getattr(pto, "tile_buf_type"))
    expect(
        "pto.tile_buf_type is not a supported PTODSL public interface" in str(removed_tile_buf_type),
        "removed pto.tile_buf_type should diagnose the authored alloc_tile replacement",
    )
    removed_vecscope = expect_raises(AttributeError, lambda: getattr(pto, "vecscope"))
    expect(
        "pto.vecscope is not a supported PTODSL public interface" in str(removed_vecscope),
        "removed pto.vecscope should diagnose the public SIMD replacements",
    )
    removed_as_ptr = expect_raises(AttributeError, lambda: getattr(pto, "as_ptr"))
    expect(
        "pto.as_ptr is not a supported PTODSL public interface" in str(removed_as_ptr),
        "removed pto.as_ptr should diagnose the authored object-method replacements",
    )
    removed_vbrc_load = expect_raises(AttributeError, lambda: getattr(pto, "vbrc_load"))
    expect(
        "pto.vbrc_load is not a supported PTODSL public interface" in str(removed_vbrc_load),
        "removed pto.vbrc_load should diagnose the public vlds(dist=...) replacement",
    )
    removed_vsts_1pt = expect_raises(AttributeError, lambda: getattr(pto, "vsts_1pt"))
    expect(
        "pto.vsts_1pt is not a supported PTODSL public interface" in str(removed_vsts_1pt),
        "removed pto.vsts_1pt should diagnose the public vsts(dist=...) replacement",
    )
    for name in ("max", "min", "exp", "log", "sqrt", "abs"):
        expect(hasattr(scalar, name), f"scalar.{name} should be exported from the public scalar namespace")

    with make_context() as ctx, Location.unknown(ctx):
        tile_buf_ty = pto_types.tile_buf_type(
            [16, 32],
            pto.f32,
            [16, 8],
            address_space="mat",
            blayout="ColMajor",
            slayout="RowMajor",
        )
        expect(hasattr(tile_buf_ty, "memory_space"), "TileBufType should expose a memory_space accessor")
        expect(hasattr(tile_buf_ty, "shape"), "TileBufType should expose a shape accessor")
        expect(hasattr(tile_buf_ty, "valid_shape"), "TileBufType should expose a valid_shape accessor")
        expect(hasattr(tile_buf_ty, "element_type"), "TileBufType should expose an element_type accessor")
        expect(tile_buf_ty.memory_space.value == pto.MemorySpace.MAT.value, "TileBufType.memory_space should preserve the authored address space")
        expect(list(tile_buf_ty.shape) == [16, 32], "TileBufType.shape should preserve the authored physical shape")
        expect(list(tile_buf_ty.valid_shape) == [16, 8], "TileBufType.valid_shape should preserve the authored valid shape")
        expect(str(tile_buf_ty.element_type) == "f32", "TileBufType.element_type should preserve the authored element type")

    host_vec_copy.verify()
    runtime_metadata_kernel.verify()
    authored_addr_tile_surface_probe.verify()
    dynamic_addr_tile_surface_probe.verify()
    tile_surface_compute_probe.verify()
    shared_subkernel_lowering_probe.verify()
    simt_helper_lowering_probe.verify()
    carry_loop_lowering_probe.verify()
    branch_handle_then_only_probe.verify()
    branch_handle_side_effect_probe.verify()
    branch_handle_merge_probe.verify()
    runtime_scalar_operator_probe.verify()
    tile_slice_surface_probe.verify()
    tile_slice_1d_surface_probe.verify()
    tile_valid_shape_update_probe.verify()
    tile_valid_shape_update_1d_probe.verify()
    make_mask_index_roundtrip_probe.verify()
    integer_loop_bound_probe.verify()
    scalar_pointer_offset_probe.verify()
    addptr_surface_probe.verify()
    simt_pointer_offset_probe.verify()
    scalar_store_element_coercion_probe.verify()
    public_surface_exports_probe.verify()
    compile_time_query_probe.verify()
    eager_scalar_constructor_probe.verify()
    signed_integer_scalar_probe.verify()
    low_precision_storage_probe.verify()
    pointer_vlds_inference_probe.verify()
    public_mask_bitcast_probe.verify()
    public_mask_surface_probe.verify()
    public_sync_surface_probe.verify()
    public_data_movement_surface_probe.verify()

    with make_context() as ctx, Location.unknown(ctx):
        expect(
            pto.MaskPattern.ALL == "PAT_ALL",
            "pto.MaskPattern.ALL should expose the documented PAT_ALL token",
        )
        expect(
            pto.MaskPattern.VL16 == "PAT_VL16",
            "pto.MaskPattern.VL16 should expose the documented PAT_VL16 token",
        )
        expect(
            pto.CmpMode.GT == "gt",
            "pto.CmpMode.GT should lower through the documented compare-mode surface",
        )
        expect(
            pto.PredicatePart.LOWER == "LOWER",
            "pto.PredicatePart.LOWER should expose the documented pack/unpack token",
        )
        expect(
            pto.PredicateDist.PK == "PK",
            "pto.PredicateDist.PK should expose the documented predicate-store distribution token",
        )
        expect(
            str(pto.si8.resolve()) == "si8",
            "pto.si8 should resolve to a signed 8-bit integer type",
        )
        expect(
            str(pto.ui8.resolve()) == "ui8",
            "pto.ui8 should resolve to an unsigned 8-bit integer type",
        )
        bit_pattern_module = Module.create()
        with InsertionPoint(bit_pattern_module.body):
            _ = pto.si32("0xFFFFFFFF")
            _ = pto.ui32("0xFFFFFFFF")
            _ = pto.i32("0x80000000")
        bit_pattern_text = str(bit_pattern_module)
        expect(
            "unrealized_conversion_cast" in bit_pattern_text,
            "signed/unsigned integer bit-pattern constructors should bridge through unrealized_conversion_cast",
        )
        expect(
            "arith.constant -1 : i32" in bit_pattern_text,
            "pto.si32/ui32 bit-pattern initialization should materialize the expected signless constant payload",
        )
        expect(
            "arith.constant -2147483648 : i32" in bit_pattern_text,
            "pto.i32 bit-pattern initialization should materialize the documented signless bit pattern",
        )
        expect(
            str(pto.f8e4m3.resolve()) == "f8E4M3FN",
            "pto.f8e4m3 should resolve to the public E4M3 float8 type",
        )
        expect(
            "hif8" in str(pto.hif8.resolve()),
            "pto.hif8 should resolve to the public HiF8 type",
        )
        expect(
            "f4E1M2x2" in str(pto.f4e1m2x2.resolve()),
            "pto.f4e1m2x2 should resolve to the packed 4-bit float type",
        )
        expect(
            str(pto.mask_b8.resolve()) == "!pto.mask<b8>",
            "pto.mask_b8 should resolve to the public 8-bit mask type",
        )
        expect(
            str(pto.mask_b16.resolve()) == "!pto.mask<b16>",
            "pto.mask_b16 should resolve to the public 16-bit mask type",
        )
        expect(
            str(pto.mask_b32.resolve()) == "!pto.mask<b32>",
            "pto.mask_b32 should resolve to the public 32-bit mask type",
        )

        lp_tile_ty = pto_types.tile_buf_type([16, 16], pto.hif8, [16, 16])
        lp_tv_ty = pto_types.tensor_view_type(2, pto.f8e4m3)
        lp_part_ty = pto_types.part_tensor_view_type(2, pto.f4e2m1x2)
        expect(
            "hif8" in str(lp_tile_ty.element_type),
            "low-precision tile buffers should preserve their authored element type",
        )
        expect(
            str(lp_tv_ty.element_type) == "f8E4M3FN",
            "internal tensor-view type helper should preserve low-precision element types",
        )
        expect(
            "f4E2M1x2" in str(lp_part_ty.element_type),
            "internal partition tensor-view type helper should preserve low-precision element types",
        )

        expect_raises(
            TypeError,
            lambda: pto.ptr(pto.hif8).resolve(),
            "Tile / TensorView / PartitionTensorView construction",
        )
        expect_raises(
            TypeError,
            lambda: pto.vreg_type(64, pto.f8e4m3).resolve(),
            "Tile / TensorView / PartitionTensorView construction",
        )
        expect_raises(
            TypeError,
            lambda: pto.hif8(1.0),
            "unsupported eager constructor target type",
        )
        expect_raises(
            TypeError,
            lambda: pto.f8e4m3(1.0),
            "unsupported eager constructor target type",
        )

    expect_raises(
        TypeError,
        lambda: pto.tensor_spec(rank=2, dtype=pto.hif8),
        "Tile / TensorView / PartitionTensorView construction",
    )
    expect_raises(
        TypeError,
        lambda: pto.tensor_spec(rank=2, dtype=pto.f8e4m3),
        "Tile / TensorView / PartitionTensorView construction",
    )
    expect(
        not hasattr(pto, "tensor_view_type"),
        "pto.tensor_view_type should remain an internal helper and not be exported on the public namespace",
    )
    expect(
        not hasattr(pto, "part_tensor_view_type"),
        "pto.part_tensor_view_type should remain an internal helper and not be exported on the public namespace",
    )

    default_compiled = host_vec_copy.compile()
    explicit_default = host_vec_copy.compile(BLOCK=128)
    block64 = host_vec_copy.compile(BLOCK=64)

    expect(default_compiled is explicit_default, "default constexpr compile should hit specialization cache")
    expect(default_compiled is not block64, "different constexpr values should materialize different specializations")
    expect(len(host_vec_copy.cached_specializations()) == 2, "expected exactly two cached specializations")
    expect(default_compiled.constexpr_bindings == {"BLOCK": 128}, "default constexpr binding mismatch")
    expect(block64.constexpr_bindings == {"BLOCK": 64}, "BLOCK=64 constexpr binding mismatch")
    expect_raises(
        TypeError,
        lambda: host_vec_copy.compile(BLOCK=[128]),
        "@pto.jit constexpr parameter 'BLOCK' must be hashable",
    )
    expect(
        default_compiled.specialization_key.abi_signature == block64.specialization_key.abi_signature,
        "ABI signature should stay stable across constexpr-only specializations",
    )
    expect(
        default_compiled.specialization_key.constexpr_signature
        != block64.specialization_key.constexpr_signature,
        "constexpr specialization key should differ when BLOCK changes",
    )
    pointer_default = pointer_runtime_shape_specialization_probe.compile()
    pointer_explicit_default = pointer_runtime_shape_specialization_probe.compile(BLOCK=128)
    pointer_block64 = pointer_runtime_shape_specialization_probe.compile(BLOCK=64)
    expect(
        pointer_default is pointer_explicit_default,
        "pointer-first kernels should reuse the same specialization when only runtime launch values vary",
    )
    expect(
        pointer_default is not pointer_block64,
        "pointer-first kernels should still specialize on constexpr values",
    )
    expect(
        pointer_default.specialization_key.abi_signature == pointer_block64.specialization_key.abi_signature,
        "pointer-first runtime shape scalars should stay in the ABI signature, not the constexpr signature",
    )
    expect(
        pointer_default.specialization_key.constexpr_signature == (("BLOCK", 128),),
        "pointer-first specialization key should only capture constexpr bindings",
    )
    expect(
        pointer_block64.specialization_key.constexpr_signature == (("BLOCK", 64),),
        "pointer-first specialization key should change only with constexpr bindings",
    )
    pointer_artifacts_default = artifact_paths(
        pointer_default._py_name,
        pointer_default.ir_function_name,
        pointer_default.specialization_key,
    )
    pointer_artifacts_explicit_default = artifact_paths(
        pointer_explicit_default._py_name,
        pointer_explicit_default.ir_function_name,
        pointer_explicit_default.specialization_key,
    )
    pointer_artifacts_block64 = artifact_paths(
        pointer_block64._py_name,
        pointer_block64.ir_function_name,
        pointer_block64.specialization_key,
    )
    expect(
        pointer_artifacts_default.cache_dir == pointer_artifacts_explicit_default.cache_dir,
        "native artifact paths should stay stable for the same pointer-first specialization",
    )
    expect(
        pointer_artifacts_default.cache_dir != pointer_artifacts_block64.cache_dir,
        "native artifact paths should only change when constexpr specializations change",
    )
    marshaled_small = _marshal_launch_args(pointer_default._kernel_signature, (0x1000, 16, 32, 32))
    marshaled_padded = _marshal_launch_args(pointer_default._kernel_signature, (0x1000, 7, 29, 64))
    expect(len(marshaled_small) == 4, "pointer-first launch ABI should marshal one pointer plus rows/cols/stride scalars")
    expect(len(marshaled_padded) == 4, "pointer-first launch ABI width should stay constant across dynamic shape launches")
    expect(
        marshaled_small[1].value == 16 and marshaled_small[2].value == 32 and marshaled_small[3].value == 32,
        "contiguous runtime-shape launch should preserve rows/cols/stride values",
    )
    expect(
        marshaled_padded[1].value == 7 and marshaled_padded[2].value == 29 and marshaled_padded[3].value == 64,
        "same compiled kernel should accept padded runtime shape/stride values without changing specialization",
    )
    launch_cpp = generate_launch_cpp(
        ir_function_name=pointer_default.ir_function_name,
        kernel_signature=pointer_default._kernel_signature,
    )
    expect(
        "extern \"C\" void ptodsl_launch_" in launch_cpp and "int32_t rows" in launch_cpp,
        "launch wrapper codegen should resolve pointer-first runtime scalar annotations without requiring an ambient MLIR Context",
    )

    default_text = default_compiled.mlir_text()
    block64_text = block64.mlir_text()
    explicit_text = host_vec_copy_explicit.compile().mlir_text()
    expect_parse_roundtrip_and_verify(default_text, "default host_vec_copy specialization")
    expect_parse_roundtrip_and_verify(block64_text, "BLOCK=64 host_vec_copy specialization")
    expect_parse_roundtrip_and_verify(explicit_text, "explicit host_vec_copy specialization")
    expect("!pto.tile_buf<vec, 1x128xf32>" in default_text, "default specialization MLIR missing BLOCK=128 tile")
    expect("!pto.tile_buf<vec, 1x64xf32>" in block64_text, "BLOCK=64 specialization MLIR missing specialized tile")
    expect("attributes {pto.aicore}" in default_text, "default @pto.jit should emit a flat aicore entry by default")
    expect("attributes {pto.aicore}" in explicit_text, "explicit @pto.jit should emit a flat aicore entry by default")
    expect("builtin.module" not in default_text, "default @pto.jit should no longer emit a nested builtin.module container")
    expect('pto.mode = "auto"' in default_text, "default specialization should carry auto mode module metadata")
    expect('pto.mode = "explicit"' in explicit_text, "explicit specialization should carry explicit mode module metadata")
    expect(
        host_vec_copy.compile()._module_spec.insert_sync is None,
        "default @pto.jit insert_sync should stay unset and follow mode defaults",
    )
    expect(
        host_vec_copy_explicit.compile()._module_spec.insert_sync is None,
        "explicit @pto.jit insert_sync should stay unset and follow mode defaults",
    )
    expect(
        host_vec_copy_no_insert_sync.compile()._module_spec.insert_sync is False,
        "@pto.jit(insert_sync=False) should preserve the explicit override",
    )
    expect(
        host_vec_copy_explicit_insert_sync.compile()._module_spec.insert_sync is True,
        "@pto.jit(insert_sync=True) should preserve the explicit override",
    )
    expect("valid=?" not in default_text, "default alloc_tile() should keep full static valid-shape when valid_shape= is omitted")
    auto_mode_violation = expect_raises(
        RuntimeError,
        auto_mode_explicit_surface_violation_probe.compile,
        '@pto.jit(mode="explicit")',
    )
    expect(
        "auto-mode contract violation" in str(auto_mode_violation),
        "explicit-only surface use in auto mode should be diagnosed as an auto-mode contract violation",
    )
    expect(
        "auto_mode_explicit_surface_violation_probe" in str(auto_mode_violation),
        "auto-mode DMA violation should identify the authored kernel name",
    )
    expect(
        __file__ in str(auto_mode_violation),
        "auto-mode DMA violation should preserve the authored source file",
    )
    expect_raises(
        ValueError,
        lambda: pto.merge_jit_modules(host_vec_copy.compile(), host_vec_copy_explicit.compile()),
        "compatible module attributes",
    )

    runtime_metadata_text = runtime_metadata_kernel.compile().mlir_text()
    expect_parse_roundtrip_and_verify(runtime_metadata_text, "runtime metadata specialization")
    expect(
        re.search(
            r"pto\.make_tensor_view %arg0, shape = \[%[a-zA-Z0-9_]+, %[a-zA-Z0-9_]+\], strides = \[%[a-zA-Z0-9_]+, %[a-zA-Z0-9_]+\]",
            runtime_metadata_text,
        ) is not None,
        "make_tensor_view should preserve explicitly authored runtime shape/stride metadata",
    )

    tile_surface_text = tile_surface_compute_probe.compile().mlir_text()
    expect_parse_roundtrip_and_verify(tile_surface_text, "tile surface compute specialization")
    expect("pto.texpands" in tile_surface_text, "pto.tile.expands should lower to pto.texpands")
    expect("pto.tadd " in tile_surface_text, "pto.tile.add should lower to pto.tadd")
    expect("pto.tadds" in tile_surface_text, "pto.tile.adds should lower to pto.tadds")
    expect("pto.tcmps" in tile_surface_text, "pto.tile.cmps should lower to pto.tcmps")
    expect(
        re.search(
            r"pto\.alloc_tile valid_row = %[a-zA-Z0-9_]+ valid_col = %[a-zA-Z0-9_]+ : !pto\.tile_buf<vec, 1x128xf32, valid=\?x\?>",
            runtime_metadata_text,
        ) is not None,
        "alloc_tile(valid_shape=[rows, cols]) should lower runtime metadata through valid_row/valid_col operands",
    )
    expect(
        re.search(
            r"sizes = \[%[a-zA-Z0-9_]+, %[a-zA-Z0-9_]+\]",
            runtime_metadata_text,
        ) is not None,
        "partition_view sizes derived from tensor metadata should remain runtime MLIR values",
    )

    tile_transfer_text = tile_transfer_surface_probe.compile().mlir_text()
    expect_parse_roundtrip_and_verify(tile_transfer_text, "tile transfer surface specialization")
    expect("pto.tload" in tile_transfer_text, "pto.tile.load(tensor, tile) should lower to pto.tload")
    expect("pto.tstore" in tile_transfer_text, "pto.tile.store(tile, tensor) should lower to pto.tstore")
    expect(
        re.search(
            r"sizes = \[%[a-zA-Z0-9_]+, %[a-zA-Z0-9_]+\]",
            tile_transfer_text,
        ) is not None,
        "pto.tile.load/store overloads should infer partition sizes from tile.valid_shape",
    )

    authored_addr_tile_text = authored_addr_tile_surface_probe.compile().mlir_text()
    expect_parse_roundtrip_and_verify(authored_addr_tile_text, "authored alloc_tile addr specialization")
    expect(
        re.search(
            r"pto\.alloc_tile addr = %c0_i64 valid_row = %[a-zA-Z0-9_]+ valid_col = %[a-zA-Z0-9_]+ : !pto\.tile_buf<vec, 1x128xf32, valid=\?x\?>",
            authored_addr_tile_text,
        ) is not None,
        "alloc_tile(shape=..., dtype=..., addr=int, valid_shape=...) should coerce Python ints to i64 operands",
    )

    dynamic_addr_tile_text = dynamic_addr_tile_surface_probe.compile().mlir_text()
    expect_parse_roundtrip_and_verify(dynamic_addr_tile_text, "dynamic alloc_tile addr specialization")
    expect(
        "arith.index_cast %arg0 : i32 to index" in dynamic_addr_tile_text,
        "alloc_tile(addr=runtime integer metadata) should first bridge the public i32 scalar to index",
    )
    expect(
        re.search(
            r"arith\.index_cast %[a-zA-Z0-9_]+ : index to i64",
            dynamic_addr_tile_text,
        ) is not None,
        "alloc_tile(addr=runtime index) should still cast the bridged index metadata to i64 before lowering",
    )
    expect(
        re.search(
            r"pto\.alloc_tile addr = %[a-zA-Z0-9_]+ valid_row = %[a-zA-Z0-9_]+ valid_col = %[a-zA-Z0-9_]+ : !pto\.tile_buf<vec, 1x128xf32, valid=\?x\?>",
            dynamic_addr_tile_text,
        ) is not None,
        "alloc_tile(shape=..., dtype=..., addr=runtime value, valid_shape=...) should accept dynamic i64-like operands",
    )

    tile_valid_shape_text = tile_valid_shape_update_probe.compile().mlir_text()
    expect_parse_roundtrip_and_verify(tile_valid_shape_text, "tile valid-shape update specialization")
    expect(
        re.search(
            r"pto\.set_validshape %[a-zA-Z0-9_]+, %[a-zA-Z0-9_]+, %[a-zA-Z0-9_]+ : !pto\.tile_buf<vec, 1x128xf32, valid=\?x\?>",
            tile_valid_shape_text,
        ) is not None,
        "tile.valid_shape = [rows, cols] should lower to pto.set_validshape on a dynamic-valid tile",
    )

    tile_valid_shape_1d_text = tile_valid_shape_update_1d_probe.compile().mlir_text()
    expect_parse_roundtrip_and_verify(tile_valid_shape_1d_text, "1D tile valid-shape update specialization")
    expect(
        re.search(
            r"pto\.set_validshape %[a-zA-Z0-9_]+, %[a-zA-Z0-9_]+, %[a-zA-Z0-9_]+ : !pto\.tile_buf<vec, 1x128xf32, valid=\?x\?>",
            tile_valid_shape_1d_text,
        ) is not None,
        "tile.valid_shape = [length] should lower to pto.set_validshape on a rank-1 dynamic-valid tile",
    )

    make_mask_index_roundtrip_text = make_mask_index_roundtrip_probe.compile().mlir_text()
    expect_parse_roundtrip_and_verify(make_mask_index_roundtrip_text, "make_mask index round-trip specialization")
    expect(
        "pto.plt_b32 %arg2 : i32 -> !pto.mask<b32>, i32" in make_mask_index_roundtrip_text,
        "make_mask(...) should accept public i32 runtime counts directly at the hardware tail-mask operand type",
    )
    expect(
        re.search(
            r"iter_args\(%[a-zA-Z0-9_]+ = %arg0\) -> \(i32\)",
            make_mask_index_roundtrip_text,
        ) is not None,
        "make_mask(...) should allow loop-carried public i32 remainders without manual index casts",
    )
    expect(
        re.search(
            r"scf\.yield %[a-zA-Z0-9_]+ : i32",
            make_mask_index_roundtrip_text,
        ) is not None,
        "make_mask(...) should keep the carried remainder in public i32 form after tail-mask generation",
    )

    SUBKERNEL_OBSERVATIONS.clear()
    shared_subkernel_lowering_probe.compile(TRACE_TOKEN=1)
    expect(
        SUBKERNEL_OBSERVATIONS == [
            ("cube", "top_level_cube_probe", 1),
            ("simd", "top_level_simd_probe", 1),
            ("simd", "nested_simd_probe", 1),
        ],
        f"unexpected shared subkernel lowering observations: {SUBKERNEL_OBSERVATIONS!r}",
    )

    INLINE_SUBKERNEL_SCOPE_OBSERVATIONS.clear()
    inline_subkernel_scope_text = inline_subkernel_scope_probe.compile(TRACE_TOKEN=1).mlir_text()
    expect_parse_roundtrip_and_verify(inline_subkernel_scope_text, "inline subkernel scope specialization")
    expect(
        INLINE_SUBKERNEL_SCOPE_OBSERVATIONS == [
            ("simt", "inline_simt", 1),
            ("simd", "inline_simd", 1),
            ("cube", "inline_cube", 1),
        ],
        f"unexpected inline subkernel scope observations: {INLINE_SUBKERNEL_SCOPE_OBSERVATIONS!r}",
    )
    expect(
        "pto.store" in inline_subkernel_scope_text,
        "inline pto.simt() body should lower authored scalar ops inside the surrounding kernel trace",
    )
    expect(
        inline_subkernel_scope_text.count("pto.barrier <PIPE_ALL>") >= 2,
        "inline pto.simd()/pto.cube() bodies should lower their authored operations in place",
    )

    simt_text = simt_helper_lowering_probe.compile(TRACE_TOKEN=1).mlir_text()
    expect_parse_roundtrip_and_verify(simt_text, "simt helper lowering specialization")
    expect(
        simt_text.count("pto.store_vfsimt_info") == 2,
        "each @pto.simt callsite should materialize a caller-side store_vfsimt_info",
    )
    expect(
        simt_text.count("call @simt_tid_probe()") == 2,
        "each @pto.simt callsite should lower to a func.call of the helper symbol",
    )
    expect(
        simt_text.count("func.func @simt_tid_probe() attributes {pto.simt_entry}") == 1,
        "@pto.simt helper should materialize exactly one reusable pto.simt_entry function",
    )
    expect("pto.get_tid_x" in simt_text, "SIMT helper body should contain pto.get_tid_x")
    expect("pto.get_tid_y" in simt_text, "SIMT helper body should contain pto.get_tid_y")
    expect("pto.get_tid_z" in simt_text, "SIMT helper body should contain pto.get_tid_z")

    ast_subkernel_runtime_for_text = ast_subkernel_runtime_for_probe.compile().mlir_text()
    expect_parse_roundtrip_and_verify(
        ast_subkernel_runtime_for_text,
        "AST-rewritten subkernel runtime for specialization",
    )
    expect(
        ast_subkernel_runtime_for_text.count("scf.for") == 1,
        "@pto.simd helper should rewrite Python range(...) loops into runtime scf.for",
    )
    expect(
        "pto.barrier <PIPE_ALL>" in ast_subkernel_runtime_for_text,
        "rewritten @pto.simd helper body should lower inside the caller trace",
    )

    carry_text = carry_loop_lowering_probe.compile(BLOCK=32).mlir_text()
    expect_parse_roundtrip_and_verify(carry_text, "carry loop specialization")
    expect("scf.for" in carry_text, "carry loop should lower to scf.for")
    expect("iter_args(" in carry_text, "carry loop should lower named state through scf.for iter_args")
    expect("scf.yield" in carry_text, "carry loop should lower loop.update(...) to scf.yield")
    expect(
        carry_text.count("!pto.tile_buf<vec, 1x32xf32>") >= 3,
        "carry loop MLIR should materialize the specialized carried tile types",
    )
    expect(
        re.search(r"outs\(%[^\s]+#2 : !pto\.tile_buf<vec, 1x32xf32>\)", carry_text) is not None,
        "loop.final(\"o\") should materialize the third scf.for result as the final carried state",
    )

    branch_then_only_text = branch_handle_then_only_probe.compile().mlir_text()
    expect_parse_roundtrip_and_verify(branch_then_only_text, "branch handle then-only specialization")
    expect(branch_then_only_text.count("scf.if") == 1, "then-only branch handle should lower to one scf.if")
    expect("pto.barrier <PIPE_ALL>" in branch_then_only_text, "br.then_ body should lower into the scf.if then branch")
    expect("}, {" not in branch_then_only_text, "then-only branch handle should not materialize an else region")

    branch_side_effect_text = branch_handle_side_effect_probe.compile().mlir_text()
    expect_parse_roundtrip_and_verify(branch_side_effect_text, "branch handle side-effect specialization")
    expect(branch_side_effect_text.count("scf.if") == 1, "side-effect branch handle should lower to one scf.if")
    expect("pto.barrier <PIPE_ALL>" in branch_side_effect_text, "br.then_ side effects should lower into the then branch")
    expect("pto.mem_bar" in branch_side_effect_text, "br.else_ side effects should lower into the else branch")
    expect("} else {" in branch_side_effect_text, "side-effect branch handle should materialize an explicit else region")

    branch_merge_text = branch_handle_merge_probe.compile().mlir_text()
    expect_parse_roundtrip_and_verify(branch_merge_text, "branch handle merge specialization")
    expect(
        re.search(r"scf\.if %\d+ -> \(i32, i32\)", branch_merge_text) is not None,
        "br.assign(...) should infer scf.if result types from the assigned branch values",
    )
    expect(
        branch_merge_text.count("scf.yield") >= 2,
        "automatic branch merge should materialize one internal scf.yield per branch",
    )
    expect(
        "arith.addi" in branch_merge_text and "arith.subi" in branch_merge_text,
        "merged branch values should remain usable as ordinary runtime scalars after the conditional",
    )

    ast_if_side_effect_text = ast_if_side_effect_probe.compile().mlir_text()
    expect_parse_roundtrip_and_verify(ast_if_side_effect_text, "AST-rewritten side-effect if specialization")
    expect(
        ast_if_side_effect_text.count("scf.if") == 1,
        "ast_rewrite=True Python if should lower to one scf.if for runtime conditions",
    )
    expect(
        "pto.barrier <PIPE_ALL>" in ast_if_side_effect_text,
        "ast_rewrite=True if body should lower into the scf.if then branch",
    )

    ast_if_merge_text = ast_if_merge_probe.compile().mlir_text()
    expect_parse_roundtrip_and_verify(ast_if_merge_text, "AST-rewritten if-merge specialization")
    expect(
        re.search(r"scf\.if %\d+ -> \(i32, i32\)", ast_if_merge_text) is not None,
        "ast_rewrite=True Python if/else live-outs should lower to scf.if results",
    )
    expect(
        ast_if_merge_text.count("scf.yield") >= 2,
        "ast_rewrite=True Python if/else live-outs should materialize branch yields",
    )

    ast_if_old_value_merge_text = ast_if_old_value_merge_probe.compile().mlir_text()
    expect_parse_roundtrip_and_verify(ast_if_old_value_merge_text, "AST-rewritten old-value if-merge specialization")
    expect(
        re.search(r"scf\.if %\d+ -> \(i32\)", ast_if_old_value_merge_text) is not None,
        "ast_rewrite=True one-sided Python if assignment should merge the old value through scf.if",
    )
    expect(
        ast_if_old_value_merge_text.count("scf.yield") >= 2,
        "ast_rewrite=True old-value if merge should yield from both branches",
    )

    ast_if_branch_local_temp_liveness_text = ast_if_branch_local_temp_liveness_probe.compile().mlir_text()
    expect_parse_roundtrip_and_verify(
        ast_if_branch_local_temp_liveness_text,
        "AST-rewritten if branch-local temp liveness specialization",
    )
    expect(
        "old_tmp" not in ast_if_branch_local_temp_liveness_text,
        "branch-local temporaries should not be merged as old live-out values",
    )

    ast_nested_with_if_merge_text = ast_nested_with_if_merge_probe.compile().mlir_text()
    expect_parse_roundtrip_and_verify(ast_nested_with_if_merge_text, "AST-rewritten nested with if-merge specialization")
    expect(
        re.search(r"scf\.if %\d+ -> \(i32\)", ast_nested_with_if_merge_text) is not None,
        "ast_rewrite=True nested with block should preserve outer live-out liveness for if merge",
    )

    ast_runtime_for_text = ast_runtime_for_probe.compile().mlir_text()
    expect_parse_roundtrip_and_verify(ast_runtime_for_text, "AST-rewritten runtime for specialization")
    expect(
        ast_runtime_for_text.count("scf.for") == 1,
        "ast_rewrite=True Python range(...) should lower to one scf.for",
    )

    ast_runtime_for_carry_text = ast_runtime_for_carry_probe.compile().mlir_text()
    expect_parse_roundtrip_and_verify(ast_runtime_for_carry_text, "AST-rewritten runtime carry for specialization")
    expect(
        "iter_args(" in ast_runtime_for_carry_text and "scf.yield" in ast_runtime_for_carry_text,
        "ast_rewrite=True accumulator loops should lower through scf.for iter_args",
    )

    ast_runtime_for_augassign_carry_text = ast_runtime_for_augassign_carry_probe.compile().mlir_text()
    expect_parse_roundtrip_and_verify(
        ast_runtime_for_augassign_carry_text,
        "AST-rewritten augassign runtime carry for specialization",
    )
    expect(
        "iter_args(" in ast_runtime_for_augassign_carry_text and "scf.yield" in ast_runtime_for_augassign_carry_text,
        "ast_rewrite=True accumulator loops using += should lower through scf.for iter_args",
    )

    ast_runtime_for_branch_local_temp_text = ast_runtime_for_branch_local_temp_probe.compile().mlir_text()
    expect_parse_roundtrip_and_verify(
        ast_runtime_for_branch_local_temp_text,
        "AST-rewritten runtime for branch-local temp specialization",
    )
    expect(
        "iter_args(" not in ast_runtime_for_branch_local_temp_text,
        "branch-local temporaries should not be inferred as loop-carried state",
    )

    ast_rewrite_disabled_nested_helper_python_control_text = (
        ast_rewrite_disabled_nested_helper_python_control_probe.compile().mlir_text()
    )
    expect_parse_roundtrip_and_verify(
        ast_rewrite_disabled_nested_helper_python_control_text,
        "AST rewrite disabled nested Python helper control specialization",
    )
    expect(
        "scf.if" not in ast_rewrite_disabled_nested_helper_python_control_text
        and "scf.for" not in ast_rewrite_disabled_nested_helper_python_control_text,
        "ast_rewrite=False should keep nested helper function bodies as trace-time Python",
    )
    expect(
        ast_rewrite_disabled_nested_helper_python_control_text.count("pto.barrier <PIPE_ALL>") == 2,
        "nested Python helper should keep trace-time if/range behavior",
    )

    ast_nested_helper_ast_rewrite_text = ast_nested_helper_ast_rewrite_probe.compile().mlir_text()
    expect_parse_roundtrip_and_verify(
        ast_nested_helper_ast_rewrite_text,
        "AST-rewritten nested helper function specialization",
    )
    expect(
        ast_nested_helper_ast_rewrite_text.count("scf.for") == 1,
        "default AST rewrite should rewrite range(...) loops inside nested helpers",
    )
    expect(
        ast_nested_helper_ast_rewrite_text.count("scf.if") == 1,
        "default AST rewrite should rewrite runtime if statements inside nested helpers",
    )
    expect(
        "iter_args(" in ast_nested_helper_ast_rewrite_text and "scf.yield" in ast_nested_helper_ast_rewrite_text,
        "rewritten nested helpers should preserve loop-carried and branch live-out values",
    )

    ast_nested_helper_freevar_if_merge_text = ast_nested_helper_freevar_if_merge_probe.compile().mlir_text()
    expect_parse_roundtrip_and_verify(
        ast_nested_helper_freevar_if_merge_text,
        "AST-rewritten nested helper freevar if-merge specialization",
    )
    expect(
        re.search(r"scf\.if %\d+ -> \(i32\)", ast_nested_helper_freevar_if_merge_text) is not None,
        "nested helper free variables should keep outer branch assignments live for SSA merge",
    )

    ast_nested_helper_name_store_liveness_text = ast_nested_helper_name_store_liveness_probe.compile().mlir_text()
    expect_parse_roundtrip_and_verify(
        ast_nested_helper_name_store_liveness_text,
        "AST-rewritten nested helper name-store liveness specialization",
    )
    expect(
        re.search(r"scf\.if %\d+ ->", ast_nested_helper_name_store_liveness_text) is None,
        "nested function definitions should store their function name and kill earlier liveness",
    )

    ast_control_flow_equiv_explicit_text = ast_control_flow_equiv_explicit_probe.compile().mlir_text()
    ast_control_flow_equiv_native_text = ast_control_flow_equiv_native_probe.compile().mlir_text()
    expect_parse_roundtrip_and_verify(
        ast_control_flow_equiv_explicit_text,
        "explicit control-flow integration specialization",
    )
    expect_parse_roundtrip_and_verify(
        ast_control_flow_equiv_native_text,
        "native AST-rewritten control-flow integration specialization",
    )
    expect(
        mlir_op_sequence(ast_control_flow_equiv_native_text) == mlir_op_sequence(ast_control_flow_equiv_explicit_text),
        "native Python for/if rewrite should emit the same operation sequence as explicit pto.for_/pto.if_",
    )
    for pattern in ("scf.for", "scf.if", "iter_args(", "scf.yield", "arith.addi"):
        expect(
            ast_control_flow_equiv_native_text.count(pattern) == ast_control_flow_equiv_explicit_text.count(pattern),
            f"native AST rewrite should match explicit control-flow count for {pattern}",
        )

    ast_closure_kernel_text = ast_closure_kernel_probe.compile().mlir_text()
    expect_parse_roundtrip_and_verify(ast_closure_kernel_text, "AST-rewritten closure kernel specialization")
    expect(
        ast_closure_kernel_text.count("pto.barrier <PIPE_ALL>") == 3,
        "ast_rewrite=True factory kernels should preserve captured Python closure values",
    )

    ast_rebound_closure_kernel_text = ast_rebound_closure_kernel_probe.compile().mlir_text()
    expect_parse_roundtrip_and_verify(
        ast_rebound_closure_kernel_text,
        "AST-rewritten rebound closure kernel specialization",
    )
    expect(
        ast_rebound_closure_kernel_text.count("pto.barrier <PIPE_ALL>") == 4,
        "ast_rewrite=True should read nonlocal closure values at compile time",
    )

    ast_mutable_closure_cache_first = ast_mutable_closure_cache_kernel_probe.compile()
    ast_mutable_closure_cache_first_text = ast_mutable_closure_cache_first.mlir_text()
    expect_parse_roundtrip_and_verify(
        ast_mutable_closure_cache_first_text,
        "AST-rewritten mutable closure cache first specialization",
    )
    expect(
        ast_mutable_closure_cache_first_text.count("pto.barrier <PIPE_ALL>") == 2,
        "first mutable closure specialization should use the initial nonlocal value",
    )
    set_ast_mutable_closure_cache_limit(4)
    ast_mutable_closure_cache_second = ast_mutable_closure_cache_kernel_probe.compile()
    ast_mutable_closure_cache_second_text = ast_mutable_closure_cache_second.mlir_text()
    expect_parse_roundtrip_and_verify(
        ast_mutable_closure_cache_second_text,
        "AST-rewritten mutable closure cache second specialization",
    )
    expect(
        ast_mutable_closure_cache_second is not ast_mutable_closure_cache_first,
        "AST rewrite closure changes should participate in the specialization cache key",
    )
    expect(
        ast_mutable_closure_cache_second_text.count("pto.barrier <PIPE_ALL>") == 4,
        "changed mutable closure specialization should recompile with the new nonlocal value",
    )

    ast_signature_closure_default_text = ast_signature_closure_default_kernel_probe.compile().mlir_text()
    expect_parse_roundtrip_and_verify(
        ast_signature_closure_default_text,
        "AST-rewritten signature closure default specialization",
    )
    expect(
        ast_signature_closure_default_text.count("pto.barrier <PIPE_ALL>") == 2,
        "ast_rewrite=True should resolve closure names used by signature defaults",
    )

    ast_rebound_subkernel_text = ast_rebound_subkernel_probe.compile(TRACE_TOKEN=1).mlir_text()
    expect_parse_roundtrip_and_verify(
        ast_rebound_subkernel_text,
        "AST-rewritten rebound subkernel specialization",
    )
    expect(
        ast_rebound_subkernel_text.count("pto.barrier <PIPE_ALL>") == 4,
        "named subkernels should read nonlocal closure values when traced",
    )

    sourceless_ast_rewrite_text = sourceless_ast_rewrite_kernel_probe.compile().mlir_text()
    expect_parse_roundtrip_and_verify(
        sourceless_ast_rewrite_text,
        "source-less AST rewrite fallback specialization",
    )
    expect(
        sourceless_ast_rewrite_text.count("pto.barrier <PIPE_ALL>") == 1,
        "source-less kernels should fall back to original trace-time Python execution",
    )

    sourceless_subkernel_text = sourceless_subkernel_entry_probe.compile(TRACE_TOKEN=1).mlir_text()
    expect_parse_roundtrip_and_verify(
        sourceless_subkernel_text,
        "source-less subkernel AST rewrite fallback specialization",
    )
    expect(
        sourceless_subkernel_text.count("pto.barrier <PIPE_ALL>") == 1,
        "source-less subkernels should fall back to original trace-time Python execution",
    )

    ast_python_bool_guard_enabled_text = ast_python_bool_guard_probe.compile().mlir_text()
    expect_parse_roundtrip_and_verify(
        ast_python_bool_guard_enabled_text,
        "AST Python bool guard enabled specialization",
    )
    expect(
        "scf.if" not in ast_python_bool_guard_enabled_text,
        "Python bool guards should remain trace-time branches under ast_rewrite=True",
    )
    expect(
        ast_python_bool_guard_enabled_text.count("pto.barrier <PIPE_ALL>") == 2,
        "Python bool guards should execute enabled trace-time branches",
    )
    ast_python_bool_guard_disabled_text = ast_python_bool_guard_probe.compile(
        BLOCK=64,
        ENABLE=False,
    ).mlir_text()
    expect_parse_roundtrip_and_verify(
        ast_python_bool_guard_disabled_text,
        "AST Python bool guard disabled specialization",
    )
    expect(
        "scf.if" not in ast_python_bool_guard_disabled_text,
        "disabled Python bool guards should not lower to runtime branches",
    )
    expect(
        "pto.barrier <PIPE_ALL>" not in ast_python_bool_guard_disabled_text,
        "disabled Python bool guards should skip their trace-time branches",
    )

    ast_static_enabled_text = ast_static_control_flow_probe.compile(ENABLE=True).mlir_text()
    expect_parse_roundtrip_and_verify(ast_static_enabled_text, "AST static control-flow enabled specialization")
    expect(
        "scf.if" not in ast_static_enabled_text and "scf.for" not in ast_static_enabled_text,
        "pto.const_expr/pto.static_range should keep compile-time control flow under ast_rewrite=True",
    )
    expect(
        ast_static_enabled_text.count("pto.barrier <PIPE_ALL>") == 2,
        "pto.static_range(2) should unroll at trace time under ast_rewrite=True",
    )
    ast_static_range_loop_target_live_after_text = ast_static_range_loop_target_live_after_probe.compile().mlir_text()
    expect_parse_roundtrip_and_verify(
        ast_static_range_loop_target_live_after_text,
        "AST static_range loop target live-after specialization",
    )
    expect(
        "scf.for" not in ast_static_range_loop_target_live_after_text,
        "pto.static_range(...) should keep Python loop semantics under ast_rewrite=True",
    )
    expect(
        ast_static_range_loop_target_live_after_text.count("pto.barrier <PIPE_ALL>") == 3,
        "pto.static_range(...) should allow the Python loop target to remain live after unrolling",
    )
    ast_static_range_break_continue_text = ast_static_range_break_continue_probe.compile().mlir_text()
    expect_parse_roundtrip_and_verify(
        ast_static_range_break_continue_text,
        "AST static_range break/continue specialization",
    )
    expect(
        "scf.for" not in ast_static_range_break_continue_text,
        "pto.static_range(...) break/continue should stay in trace-time Python control flow",
    )
    expect(
        ast_static_range_break_continue_text.count("pto.barrier <PIPE_ALL>") == 4,
        "pto.static_range(...) should preserve Python break/continue behavior under ast_rewrite=True",
    )
    ast_static_disabled_text = ast_static_control_flow_probe.compile(ENABLE=False).mlir_text()
    expect(
        "pto.barrier <PIPE_ALL>" not in ast_static_disabled_text,
        "pto.const_expr(False) should skip the trace-time branch under ast_rewrite=True",
    )

    runtime_scalar_text = runtime_scalar_operator_probe.compile(BLOCK=8).mlir_text()
    expect_parse_roundtrip_and_verify(runtime_scalar_text, "runtime scalar operator specialization")
    expect("arith.index_cast" in runtime_scalar_text, "mixed i64/index runtime arithmetic should materialize index_cast")
    expect("arith.floordivsi" in runtime_scalar_text, "runtime // should lower to arith.floordivsi")
    expect("arith.remsi" in runtime_scalar_text, "runtime % should lower to arith.remsi")
    expect("arith.addf" in runtime_scalar_text, "runtime float + should lower to arith.addf")
    expect("arith.mulf" in runtime_scalar_text, "runtime float * should lower to arith.mulf")
    expect("arith.subf" in runtime_scalar_text, "runtime float - should lower to arith.subf")
    expect("arith.divf" in runtime_scalar_text, "runtime float / should lower to arith.divf")
    expect("arith.maximumf" in runtime_scalar_text, "scalar.max(float, float) should lower to arith.maximumf")
    expect("arith.minimumf" in runtime_scalar_text, "scalar.min(float, float) should lower to arith.minimumf")
    expect("math.exp" in runtime_scalar_text, "scalar.exp(...) should lower to math.exp")
    expect("math.log" in runtime_scalar_text, "scalar.log(...) should lower to math.log")
    expect("math.sqrt" in runtime_scalar_text, "scalar.sqrt(...) should lower to math.sqrt")
    expect("math.absf" in runtime_scalar_text, "scalar.abs(float) should lower to math.absf")
    expect("arith.cmpf ogt" in runtime_scalar_text, "float runtime '>' should lower to arith.cmpf ogt")
    expect("arith.cmpf oeq" in runtime_scalar_text, "float runtime '==' should lower to arith.cmpf oeq")
    expect("arith.cmpf oge" in runtime_scalar_text, "float runtime '>=' should lower to arith.cmpf oge")
    expect("arith.cmpf ole" in runtime_scalar_text, "float runtime '<=' should lower to arith.cmpf ole")
    expect("arith.andi" in runtime_scalar_text, "i1 conjunction from native '&' should lower to arith.andi")

    host_runtime_scalar_entry_text = host_runtime_scalar_entry_probe.compile().mlir_text()
    expect_parse_roundtrip_and_verify(
        host_runtime_scalar_entry_text,
        "host runtime scalar entry specialization",
    )
    expect(
        "func.func @host_runtime_scalar_entry_probe" in host_runtime_scalar_entry_text,
        "host runtime scalar entry probe should compile into a launchable kernel",
    )
    expect(
        host_runtime_scalar_entry_probe.compile() is host_runtime_scalar_entry_probe.compile(),
        "kernels without constexpr parameters should still reuse one compiled specialization",
    )
    expect_raises(
        TypeError,
        lambda: host_runtime_scalar_entry_probe.compile(limit=4),
        "unknown @pto.jit constexpr parameter(s): limit",
    )
    expect_raises(
        TypeError,
        lambda: host_runtime_scalar_entry_probe.compile(alpha=1.0),
        "unknown @pto.jit constexpr parameter(s): alpha",
    )
    expect(
        "i32" in host_runtime_scalar_entry_text and "f32" in host_runtime_scalar_entry_text,
        "host runtime scalar entry probe should preserve scalar ABI argument types in MLIR",
    )

    signed_integer_scalar_text = signed_integer_scalar_probe.compile().mlir_text()
    expect_parse_roundtrip_and_verify(signed_integer_scalar_text, "signed integer scalar specialization")
    expect(
        "builtin.unrealized_conversion_cast" in signed_integer_scalar_text,
        "signed/unsigned scalar lowering should bridge signless arith values through unrealized_conversion_cast",
    )
    expect("arith.addi" in signed_integer_scalar_text, "signed/unsigned scalar addition should lower through arith.addi")
    expect("arith.maxsi" in signed_integer_scalar_text, "signed scalar max should lower through arith.maxsi")
    expect("arith.minsi" in signed_integer_scalar_text, "signed scalar min should lower through arith.minsi")
    expect("arith.maxui" in signed_integer_scalar_text, "unsigned scalar max should lower through arith.maxui")
    expect("arith.minui" in signed_integer_scalar_text, "unsigned scalar min should lower through arith.minui")
    expect("math.absi" in signed_integer_scalar_text, "signed scalar abs should lower through math.absi")
    expect("arith.cmpi sgt" in signed_integer_scalar_text, "signed scalar cmp should preserve signed predicate")
    expect("arith.cmpi ugt" in signed_integer_scalar_text, "unsigned scalar cmp should preserve unsigned predicate")
    expect("pto.store" in runtime_scalar_text, "scalar.store(...) should lower to pto.store")

    tile_slice_text = tile_slice_surface_probe.compile(BLOCK=128).mlir_text()
    expect_parse_roundtrip_and_verify(tile_slice_text, "tile slice surface specialization")
    expect("memref.subview" in tile_slice_text, "tile[row, col:] should lower through memref.subview")
    expect("memref.collapse_shape" not in tile_slice_text, "2D tile[row, col:] should lower directly to a rank-reduced memref view")
    expect("pto.tile_buf_addr" in tile_slice_text, "tile[row, col:] should materialize a memref tile address view")
    expect(
        "pto.vlds" in tile_slice_text and "memref<128xf32, strided<[1], offset: ?>, #pto.address_space<vec>>" in tile_slice_text,
        "vlds(tile[row, col:]) should lower against the memref slice view",
    )
    expect(
        "pto.vsts" in tile_slice_text and "memref<128xf32, strided<[1], offset: ?>, #pto.address_space<vec>>" in tile_slice_text,
        "vsts(vec, tile[row, col:], mask) should lower against the memref slice view",
    )

    tile_slice_1d_text = tile_slice_1d_surface_probe.compile(BLOCK=128).mlir_text()
    expect_parse_roundtrip_and_verify(tile_slice_1d_text, "1D tile slice surface specialization")
    expect("memref.subview" in tile_slice_1d_text, "tile[start:] should lower through memref.subview")
    expect("pto.vldas" in tile_slice_1d_text, "vldas(tile[start:]) should lower against the 1D slice view")
    expect("pto.vldus" in tile_slice_1d_text, "vldus(tile[start:], align) should lower against the 1D slice view")
    expect("pto.vsts" in tile_slice_1d_text, "vsts(vec, tile[start:], mask) should lower against the 1D slice view")

    integer_loop_text = integer_loop_bound_probe.compile(BLOCK=8).mlir_text()
    expect_parse_roundtrip_and_verify(integer_loop_text, "integer loop bound specialization")
    expect(
        integer_loop_text.count("arith.index_cast") >= 2,
        "integer runtime loop bounds should be normalized to index with arith.index_cast",
    )
    expect(
        integer_loop_text.count("scf.for") == 2,
        "integer loop bound probe should still lower nested authored loops to scf.for",
    )

    scalar_pointer_offset_text = scalar_pointer_offset_probe.compile().mlir_text()
    expect_parse_roundtrip_and_verify(scalar_pointer_offset_text, "scalar pointer offset specialization")
    expect(
        re.search(r"pto\.store %c1_i32, %\d+\[%c1\]", scalar_pointer_offset_text) is not None,
        "scalar.store(ptr, 1) should lower as element offset 1",
    )
    expect(
        re.search(r"pto\.store %c2_i32, %\d+\[%c2\]", scalar_pointer_offset_text) is not None,
        "scalar.store(ptr + 2) should lower as element offset 2",
    )
    expect(
        re.search(r"pto\.load %\d+\[%c1(?:_\d+)?\]", scalar_pointer_offset_text) is not None,
        "scalar.load(ptr, 1) should lower as element offset 1",
    )
    expect(
        re.search(r"pto\.load %\d+\[%c2(?:_\d+)?\]", scalar_pointer_offset_text) is not None,
        "scalar.load(ptr + 2) should lower as element offset 2",
    )

    addptr_surface_text = addptr_surface_probe.compile().mlir_text()
    expect_parse_roundtrip_and_verify(addptr_surface_text, "addptr surface specialization")
    expect(
        addptr_surface_text.count("pto.addptr") == 2,
        "addptr(...) should lower one PTO addptr op per authored pointer-advance call",
    )
    expect(
        "arith.index_cast" in addptr_surface_text,
        "addptr(ptr, i32-value) should coerce integer runtime scalars to index",
    )
    expect(
        re.search(r"pto\.addptr %\d+, %c2(?:_\d+)?", addptr_surface_text) is not None,
        "addptr(ptr, 2) should accept a Python int offset and lower it as an index element offset",
    )
    expect(
        re.search(r"pto\.addptr %\d+, %\d+", addptr_surface_text) is not None,
        "addptr(ptr, pto.i32(...)) should accept integer runtime scalars as element offsets",
    )

    simt_pointer_offset_text = simt_pointer_offset_probe.compile().mlir_text()
    expect_parse_roundtrip_and_verify(simt_pointer_offset_text, "simt pointer offset specialization")
    expect(
        "call @simt_pointer_offset_helper" in simt_pointer_offset_text,
        "@pto.simt pointer helper should lower to a helper func.call",
    )
    expect(
        re.search(r"pto\.store %c9_i32, %(?:arg0|\d+)\[%c1(?:_\d+)?\]", simt_pointer_offset_text) is not None,
        "ptr+offset sugar inside @pto.simt helpers should lower as address offsets, not scalar add",
    )
    expect(
        re.search(r"pto\.load %\d+\[%c1(?:_\d+)?\]", simt_pointer_offset_text) is not None,
        "@pto.simt pointer helper probe should preserve ptr+offset load syntax on the caller side",
    )

    scalar_store_coercion_text = scalar_store_element_coercion_probe.compile().mlir_text()
    expect_parse_roundtrip_and_verify(scalar_store_coercion_text, "scalar store coercion specialization")
    expect(
        scalar_store_coercion_text.count("arith.index_cast") >= 2,
        "scalar.store(...) should coerce index runtime values to the destination integer element type",
    )
    expect(
        "arith.trunci" in scalar_store_coercion_text,
        "scalar.store(...) should coerce wider integer runtime values down to the destination element type",
    )
    expect(
        scalar_store_coercion_text.count("pto.store") == 4,
        "scalar.store(...) coercion probe should still lower to four pto.store operations",
    )

    public_surface_text = public_surface_exports_probe.compile().mlir_text()
    expect_parse_roundtrip_and_verify(public_surface_text, "public surface export specialization")
    compile_time_query_text = compile_time_query_probe.compile().mlir_text()
    expect_parse_roundtrip_and_verify(compile_time_query_text, "compile-time query specialization")
    eager_scalar_text = eager_scalar_constructor_probe.compile().mlir_text()
    expect_parse_roundtrip_and_verify(eager_scalar_text, "eager scalar constructor specialization")
    low_precision_storage_text = low_precision_storage_probe.compile().mlir_text()
    expect_parse_roundtrip_and_verify(low_precision_storage_text, "low-precision storage specialization")
    pointer_vlds_text = pointer_vlds_inference_probe.compile(BLOCK=128).mlir_text()
    expect_parse_roundtrip_and_verify(pointer_vlds_text, "pointer vlds inference specialization")
    mask_bitcast_text = public_mask_bitcast_probe.compile().mlir_text()
    expect_parse_roundtrip_and_verify(mask_bitcast_text, "public mask bitcast specialization")
    mask_surface_text = public_mask_surface_probe.compile().mlir_text()
    expect_parse_roundtrip_and_verify(mask_surface_text, "public mask surface specialization")
    sync_surface_text = public_sync_surface_probe.compile().mlir_text()
    expect_parse_roundtrip_and_verify(sync_surface_text, "public sync surface specialization")
    data_movement_surface_text = public_data_movement_surface_probe.compile().mlir_text()
    expect_parse_roundtrip_and_verify(data_movement_surface_text, "public data movement surface specialization")
    vector_post_update_surface_text = vector_post_update_surface_probe.compile().mlir_text()
    expect_parse_roundtrip_and_verify(vector_post_update_surface_text, "vector post-update surface specialization")
    expect("pto.mte_gm_ub" in public_surface_text, "mte_load(...) should lower to pto.mte_gm_ub")
    expect("pto.mte_ub_gm" in public_surface_text, "mte_store(...) should lower to pto.mte_ub_gm")
    expect(public_surface_text.count("pto.mem_bar") >= 1, "mem_bar(...) should still lower explicit memory barriers")
    expect("pto.barrier <PIPE_ALL>" in public_surface_text, "pipe_barrier(Pipe.ALL) should lower to pto.barrier")
    expect("pto.vexp" in public_surface_text, "vexp(...) should lower to pto.vexp")
    expect("pto.vcgmax" in public_surface_text, "vcgmax(...) should lower to pto.vcgmax")
    expect("pto.vcgadd" in public_surface_text, "vcgadd(...) should lower to pto.vcgadd")
    expect("pto.vadds" in public_surface_text, "vsubs(...) should lower via scalar negation plus pto.vadds")
    expect("pto.mte_l1_l0a" in public_surface_text, "mte_l1_l0a(...) should lower to pto.mte_l1_l0a")
    expect("start(" not in public_surface_text, "mte_l1_l0a/l0b start_row/start_col should lower as operands")
    expect('pto.get_buf "PIPE_V", 0, 0' in sync_surface_text, 'get_buf(Pipe.V, 0) should lower to pto.get_buf with PIPE_V')
    expect('pto.rls_buf "PIPE_MTE2", 1, 2' in sync_surface_text, 'rls_buf(Pipe.MTE2, 1, 2) should lower to pto.rls_buf with PIPE_MTE2')
    expect("pto.set_flag[<PIPE_MTE2>, <PIPE_V>, <EVENT_ID0>]" in sync_surface_text, "set_flag(..., event_id=0) should lower static event ids to pto.set_flag")
    expect("pto.wait_flag[<PIPE_MTE2>, <PIPE_V>, <EVENT_ID0>]" in sync_surface_text, "wait_flag(..., event_id=0) should lower static event ids to pto.wait_flag")
    expect("pto.set_flag_dyn[<PIPE_V>, <PIPE_MTE3>, %c3]" in sync_surface_text, "set_flag(..., event_id=dynamic_event) should lower runtime event ids to pto.set_flag_dyn")
    expect("pto.wait_flag_dyn[<PIPE_V>, <PIPE_MTE3>, %c3]" in sync_surface_text, "wait_flag(..., event_id=dynamic_event) should lower runtime event ids to pto.wait_flag_dyn")
    expect("pto.sync.set <PIPE_FIX>, 0" in sync_surface_text, "set_cross_flag(Pipe.FIX, 0) should lower to pto.sync.set")
    expect("pto.sync.wait <PIPE_FIX>, 0" in sync_surface_text, "wait_cross_flag(Pipe.FIX, 0) should lower to pto.sync.wait")
    expect("pto.sync.set <PIPE_MTE3>, %c3" in sync_surface_text, "set_intra_flag(Pipe.MTE3, dynamic_event) should lower dynamic event ids through pto.sync.set")
    expect("pto.sync.wait <PIPE_V>, %c3" in sync_surface_text, "wait_intra_flag(Pipe.V, dynamic_event) should lower dynamic event ids through pto.sync.wait")
    expect(data_movement_surface_text.count("pto.mte_gm_ub") == 2, "public grouped GM->UB wrappers should lower to pto.mte_gm_ub")
    expect("pto.mte_ub_gm" in data_movement_surface_text, "public grouped UB->GM wrapper should lower to pto.mte_ub_gm")
    expect("pto.mte_ub_ub" in data_movement_surface_text, "public grouped UB->UB wrapper should lower to pto.mte_ub_ub")
    expect("pto.mte_ub_l1" in data_movement_surface_text, "public grouped UB->L1 wrapper should lower to pto.mte_ub_l1")
    expect("pto.mte_gm_l1" in data_movement_surface_text, "public grouped GM->L1 wrapper should lower to pto.mte_gm_l1")
    expect("pto.mte_l1_ub" in data_movement_surface_text, "public grouped L1->UB wrapper should lower to pto.mte_l1_ub")
    expect("pto.mte_gm_l1_frac" in data_movement_surface_text, "public GM->L1 frac wrapper should lower to pto.mte_gm_l1_frac")
    expect("pto.mte_l1_bt" in data_movement_surface_text, "public L1->BT wrapper should lower to pto.mte_l1_bt")
    expect("pto.mte_l1_fb" in data_movement_surface_text, "public L1->FB wrapper should lower to pto.mte_l1_fb")
    expect("pto.vldas" in data_movement_surface_text, "vldas(...) should lower to pto.vldas")
    expect("pto.vldus" in data_movement_surface_text, "vldus(...) should lower to pto.vldus")
    expect("pto.vldsx2" in data_movement_surface_text, "vldsx2(...) should lower to pto.vldsx2")
    expect("pto.vstur" in data_movement_surface_text, "vstur(...) should lower to pto.vstur")
    expect("pto.vstus" in data_movement_surface_text, "vstus(...) should lower to pto.vstus")
    expect("pto.vstsx2" in data_movement_surface_text, "vstsx2(...) should lower to pto.vstsx2")
    expect("pto.vgather2" in data_movement_surface_text, "vgather2(...) should lower to pto.vgather2")
    expect("pto.vgather2_bc" in data_movement_surface_text, "vgather2_bc(...) should lower to pto.vgather2_bc")
    expect("pto.vgatherb" in data_movement_surface_text, "vgatherb(...) should lower to pto.vgatherb")
    expect("pto.vscatter" in data_movement_surface_text, "vscatter(...) should lower to pto.vscatter")
    expect("pto.vsldb" in data_movement_surface_text, "vsldb(...) should lower to pto.vsldb")
    expect("pto.vsstb" in data_movement_surface_text, "vsstb(...) should lower to pto.vsstb")
    expect(
        "-> !pto.vreg<64xf32>, !pto.ptr<f32, ub>" in vector_post_update_surface_text,
        "vlds(..., return_updated_base=True) should request the updated base result",
    )
    expect(
        vector_post_update_surface_text.count("-> !pto.ptr<f32, ub>") >= 2,
        "vsts/vsstb(..., return_updated_base=True) should request updated base results",
    )
    expect("pto.vstar" in data_movement_surface_text, "vstar(...) should lower to pto.vstar")
    expect("pto.vstas" in data_movement_surface_text, "vstas(...) should lower to pto.vstas")
    expect("pto.mte_l1_l0b" in public_surface_text, "mte_l1_l0b(...) should lower to pto.mte_l1_l0b")
    expect("pto.mte_l1_l0a_mx" in public_surface_text, "mte_l1_l0a_mx(...) should lower to pto.mte_l1_l0a_mx")
    expect("pto.mte_l1_l0b_mx" in public_surface_text, "mte_l1_l0b_mx(...) should lower to pto.mte_l1_l0b_mx")
    expect("pto.mte_l0c_l1" in public_surface_text, "mte_l0c_l1(...) should lower to pto.mte_l0c_l1")
    expect("pto.mte_l0c_gm" in public_surface_text, "mte_l0c_gm(...) should lower to pto.mte_l0c_gm")
    expect(public_surface_text.count("pto.mte_l0c_ub") >= 2, "mte_l0c_ub(...) should lower sub-block and split modes")
    expect("pto.mad" in public_surface_text, "mad(...) should lower to pto.mad")
    expect("unit_flag(check_only)" in public_surface_text, "mad unit_flag option should lower to PTO unit_flag clause")
    expect("disable_gemv" in public_surface_text, "mad disable_gemv option should lower to PTO disable_gemv clause")
    expect("tf32_mode(round_even)" in public_surface_text, "mad tf32_mode option should lower to PTO tf32_mode clause")
    expect("n_dir" in public_surface_text, "mad n_dir option should lower to PTO n_dir clause")
    expect("pto.mad_acc" in public_surface_text, "mad_acc(...) should lower to pto.mad_acc")
    expect("pto.mad_bias" in public_surface_text, "mad_bias(...) should lower to pto.mad_bias")
    expect("pto.mad_mx" in public_surface_text, "mad_mx(...) should lower to pto.mad_mx")
    expect("pto.mad_mx_acc" in public_surface_text, "mad_mx_acc(...) should lower to pto.mad_mx_acc")
    expect("pto.mad_mx_bias" in public_surface_text, "mad_mx_bias(...) should lower to pto.mad_mx_bias")
    expect("!pto.tile_buf<vec, 128x64xf8E4M3FN>" in low_precision_storage_text, "low-precision tile allocation should preserve float8 element types in MLIR")
    expect("!pto.tile_buf<vec, 64x64x!pto.hif8>" in low_precision_storage_text, "low-precision tile allocation should preserve HiF8 element types in MLIR")
    expect("pto.vlds" in pointer_vlds_text, "vlds(ptr, offset) should still lower to pto.vlds")
    expect("!pto.vreg<64xf32>" in pointer_vlds_text, "vlds(ptr, offset) should infer the result vreg type from the pointer element type")
    expect('dist = "BRC_B32"' in pointer_vlds_text, 'vlds(ptr, offset, dist="BRC_B32") should lower the authored load distribution')
    expect("pto.vbitcast" in pointer_vlds_text, "vbitcast(...) should lower to pto.vbitcast")
    expect("!pto.vreg<128xf16>" in pointer_vlds_text, "vbitcast(vec, pto.f16) should preserve the 256-byte payload while adjusting the lane count")
    expect(mask_bitcast_text.count("pto.pbitcast") == 2, "pbitcast(...) should lower to pto.pbitcast for each authored mask reinterpretation")
    expect("!pto.mask<b16>" in mask_bitcast_text, "pbitcast(mask, pto.mask_b16) should materialize the requested result mask type")
    expect("!pto.mask<b32>" in mask_bitcast_text, "pbitcast(mask, pto.mask_b32) should materialize the requested result mask type")
    expect("pto.pset_b8" in mask_surface_text, "pset_b8(...) should lower to pto.pset_b8")
    expect("pto.pset_b16" in mask_surface_text, "pset_b16(...) should lower to pto.pset_b16")
    expect("pto.pset_b32" in mask_surface_text, "pset_b32(...) should lower to pto.pset_b32")
    expect("pto.pge_b8" in mask_surface_text, "pge_b8(...) should lower to pto.pge_b8")
    expect("pto.pge_b16" in mask_surface_text, "pge_b16(...) should lower to pto.pge_b16")
    expect("pto.pge_b32" in mask_surface_text, "pge_b32(...) should lower to pto.pge_b32")
    expect("pto.plt_b8" in mask_surface_text, "plt_b8(...) should lower to pto.plt_b8")
    expect("pto.plt_b16" in mask_surface_text, "plt_b16(...) should lower to pto.plt_b16")
    expect(mask_surface_text.count("pto.plt_b32") >= 1, "plt_b32(...) should still lower to pto.plt_b32")
    expect("pto.pand" in mask_surface_text, "pand(...) should lower to pto.pand")
    expect("pto.por" in mask_surface_text, "por(...) should lower to pto.por")
    expect("pto.pxor" in mask_surface_text, "pxor(...) should lower to pto.pxor")
    expect("pto.pnot" in mask_surface_text, "pnot(...) should lower to pto.pnot")
    expect("pto.psel" in mask_surface_text, "psel(...) should lower to pto.psel")
    expect("pto.ppack" in mask_surface_text, "ppack(...) should lower to pto.ppack")
    expect("pto.punpack" in mask_surface_text, "punpack(...) should lower to pto.punpack")
    expect(
        mask_surface_text.count('"HIGHER"') >= 2,
        "ppack/punpack should accept PredicatePart.HIGHER and preserve it in MLIR",
    )
    expect(
        '!pto.mask<b32> -> !pto.mask<b16>' in mask_surface_text,
        "ppack(..., to_type=pto.mask_b16) should materialize the requested narrowed result mask type",
    )
    expect(
        '!pto.mask<b16> -> !pto.mask<b32>' in mask_surface_text,
        "punpack(..., to_type=pto.mask_b32) should materialize the requested widened result mask type",
    )
    expect("pto.pintlv_b8" in mask_surface_text, "pintlv_b8(...) should lower to pto.pintlv_b8")
    expect("pto.pintlv_b16" in mask_surface_text, "pintlv_b16(...) should lower to pto.pintlv_b16")
    expect("pto.pintlv_b32" in mask_surface_text, "pintlv_b32(...) should lower to pto.pintlv_b32")
    expect("pto.pdintlv_b8" in mask_surface_text, "pdintlv_b8(...) should lower to pto.pdintlv_b8")
    expect("pto.pdintlv_b16" in mask_surface_text, "pdintlv_b16(...) should lower to pto.pdintlv_b16")
    expect("pto.pdintlv_b32" in mask_surface_text, "pdintlv_b32(...) should lower to pto.pdintlv_b32")
    expect("pto.vcmp" in mask_surface_text, "vcmp(...) should lower to pto.vcmp")
    expect("pto.vcmps" in mask_surface_text, "vcmps(...) should lower to pto.vcmps")
    expect('pto.vcmp %' in mask_surface_text and ', "eq"' in mask_surface_text, "vcmp(..., pto.CmpMode.EQ) should normalize to the compare attribute spelling")
    expect('pto.vcmps %' in mask_surface_text and ', "gt"' in mask_surface_text, "vcmps(..., pto.CmpMode.GT) should normalize to the compare attribute spelling")
    expect(mask_surface_text.count("pto.plds") == 3, "plds(...) should lower once per authored predicate load")
    expect(mask_surface_text.count("pto.psts") == 2, "psts(...) should lower once per authored predicate store")
    expect(mask_surface_text.count("pto.pstu") == 2, "pstu(...) should lower once per authored predicate unaligned store")
    expect(', "US"' in mask_surface_text, "plds(..., dist=pto.PredicateDist.US) should preserve the documented DIST token")
    expect(', "DS"' in mask_surface_text, "plds(..., dist=pto.PredicateDist.DS) should preserve the documented DIST token")
    expect(', "PK"' in mask_surface_text, "psts(..., dist=pto.PredicateDist.PK) should preserve the documented DIST token")
    expect("pto.init_align" in mask_surface_text, "init_align() should lower to pto.init_align")

    pipe_metadata_text = pipe_tile_metadata_probe.compile().mlir_text()
    expect_parse_roundtrip_and_verify(pipe_metadata_text, "pipe tile metadata probe")
    expect("pto.vlds" in pipe_metadata_text, "vlds from a popped tile should lower successfully without shape metadata errors")

    launch_handle = block64[1, None]
    expect(callable(launch_handle), "compiled[grid, stream] should return a launch callable")
    expect(hasattr(launch_handle, "__call__"), "launch handle should support __call__")

    print("ptodsl_jit_compile: PASS")


if __name__ == "__main__":
    main()
