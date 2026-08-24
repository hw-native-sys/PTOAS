#!/usr/bin/env python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Focused tracing coverage for explicit physical section hints."""

import re

from ptodsl import pto
from ptodsl._ast_rewrite import PTODSLAstRewriteError
from ptodsl._context import make_context
from ptodsl._tracing.active import current_session
from ptoas.mlir.ir import Module


@pto.jit(target="a5", mode="explicit", ast_rewrite=False)
def explicit_sections_probe():
    event_id = pto.const(1, dtype=pto.i32)
    with pto.section("cube"):
        pto.set_flag("MTE2", "S", event_id=0)
        pto.wait_flag("S", "MTE2", event_id=event_id)
    with pto.section("vector"):
        pto.wait_flag("MTE2", "S", event_id=0)
        pto.set_flag("S", "MTE2", event_id=event_id)


@pto.jit(target="a5", mode="explicit", ast_rewrite=False)
def duplicate_section_probe():
    with pto.section("cube"):
        pass
    with pto.section("cube"):
        pass


@pto.jit(target="a5", mode="explicit", ast_rewrite=False)
def nested_section_probe():
    with pto.section("cube"):
        with pto.section("vector"):
            pass


@pto.jit(
    target="a5",
    mode="explicit",
    kernel_kind="cube",
    ast_rewrite=False,
)
def kernel_kind_section_probe():
    with pto.section("cube"):
        pass


@pto.jit(target="a5", ast_rewrite=False)
def auto_mode_section_probe():
    with pto.section("vector"):
        pass


@pto.jit(
    target="a5",
    mode="explicit",
    kernel_kind=None,
    ast_rewrite=False,
)
def unspecified_kernel_kind_section_probe():
    with pto.section("vector"):
        pto.wait_flag("MTE2", "S", event_id=0)


@pto.jit(target="a5", mode="explicit", ast_rewrite=False)
def recovered_section_probe():
    try:
        with pto.section("cube"):
            raise RuntimeError("abort authored section")
    except RuntimeError:
        pass
    with pto.section("cube"):
        pto.set_flag("MTE2", "S", event_id=0)


@pto.jit(target="a5", mode="explicit", ast_rewrite=False)
def loop_subkernel_then_section_probe():
    c0 = pto.const(0, dtype=pto.i32)
    c1 = pto.const(1, dtype=pto.i32)
    with pto.for_(c0, c1, step=c1):
        session = current_session()
        with session.enter_subkernel_body("cube", "loop_cube", "a5"):
            pass
    with pto.section("cube"):
        pass


@pto.jit(target="a5", mode="explicit", ast_rewrite=False)
def section_then_subkernel_probe():
    with pto.section("cube"):
        pass
    session = current_session()
    with session.enter_subkernel_body("cube", "late_cube", "a5"):
        pass


@pto.jit(target="a5", mode="explicit", ast_rewrite=False)
def section_inside_subkernel_probe():
    session = current_session()
    with session.enter_subkernel_body("cube", "outer_cube", "a5"):
        with pto.section("vector"):
            pass


@pto.jit(target="a5", mode="explicit")
def lexical_section_rebinding_probe():
    event_id = pto.const(1, dtype=pto.i32)
    one = pto.const(1, dtype=pto.i32)
    with pto.section("cube"):
        event_id = event_id + one
        pto.wait_flag("S", "MTE2", event_id=event_id)
    with pto.section("vector"):
        pto.wait_flag("MTE2", "S", event_id=event_id)


@pto.jit(target="a5")
def lexical_non_section_conditional_rebinding_probe():
    one = pto.const(1, dtype=pto.i32)
    value = pto.const(0, dtype=pto.i32)
    if pto.get_block_idx() < one:
        value = one
    else:
        value = pto.const(2, dtype=pto.i32)
    pto.wait_flag("S", "MTE2", event_id=value)


@pto.jit(target="a5", mode="explicit")
def lexical_section_conditional_rebinding_probe():
    one = pto.const(1, dtype=pto.i32)
    with pto.section("cube"):
        value = pto.const(0, dtype=pto.i32)
        if pto.get_block_idx() < one:
            value = one
        else:
            value = pto.const(2, dtype=pto.i32)
        pto.wait_flag("S", "MTE2", event_id=value)


@pto.jit(target="a5", mode="explicit")
def lexical_section_sibling_conditional_rebinding_probe():
    one = pto.const(1, dtype=pto.i32)
    two = pto.const(2, dtype=pto.i32)
    m_tile = pto.const(0, dtype=pto.i32)
    n_tile = pto.const(0, dtype=pto.i32)
    with pto.section("cube"):
        if pto.get_block_idx() < one:
            m_tile = one
            n_tile = one
        else:
            m_tile = two
            n_tile = two
        pto.wait_flag("S", "MTE2", event_id=m_tile)
        pto.wait_flag("S", "MTE2", event_id=n_tile)
    with pto.section("vector"):
        if pto.get_block_idx() < one:
            m_tile = one
            n_tile = one
        else:
            m_tile = two
            n_tile = two
        pto.wait_flag("MTE2", "S", event_id=m_tile)
        pto.wait_flag("MTE2", "S", event_id=n_tile)


@pto.jit(target="a5", mode="explicit")
def lexical_section_sibling_single_sided_conditional_rebinding_probe():
    one = pto.const(1, dtype=pto.i32)
    m_tile = pto.const(0, dtype=pto.i32)
    n_tile = pto.const(0, dtype=pto.i32)
    with pto.section("cube"):
        if pto.get_block_idx() < one:
            m_tile = one
            n_tile = one
        m_tile_1 = pto.const(0, dtype=pto.i32)
        n_tile_1 = pto.const(0, dtype=pto.i32)
        if pto.get_block_idx() < one:
            m_tile_1 = m_tile
            n_tile_1 = n_tile
        pto.wait_flag("S", "MTE2", event_id=m_tile_1)
        pto.wait_flag("S", "MTE2", event_id=n_tile_1)
    with pto.section("vector"):
        if pto.get_block_idx() < one:
            m_tile = one
            n_tile = one
        m_tile_2 = pto.const(0, dtype=pto.i32)
        n_tile_2 = pto.const(0, dtype=pto.i32)
        if pto.get_block_idx() < one:
            m_tile_2 = m_tile
            n_tile_2 = n_tile
        pto.wait_flag("MTE2", "S", event_id=m_tile_2)
        pto.wait_flag("MTE2", "S", event_id=n_tile_2)


@pto.jit(target="a5", mode="explicit")
def lexical_section_sequential_single_sided_conditional_probe():
    m_tile = pto.const(0, dtype=pto.i64)
    n_tile = pto.const(0, dtype=pto.i64)
    with pto.section("cube"):
        if pto.get_block_idx() < 16:
            m_tile = pto.get_block_idx() & 3
            n_tile = pto.get_block_idx() // 4
        if 16 <= pto.get_block_idx():
            m_tile = (pto.get_block_idx() & 3) + 4
            n_tile = (pto.get_block_idx() // 4) - 4
        if 15 < pto.get_block_idx():
            n_tile = 3 - n_tile
        pto.wait_flag("S", "MTE2", event_id=m_tile + n_tile)


@pto.jit(target="a5", mode="explicit")
def lexical_section_single_sided_read_before_rebinding_probe():
    value = pto.const(0, dtype=pto.i64)
    with pto.section("cube"):
        if pto.get_block_idx() < 16:
            previous_value = value
            value = pto.get_block_idx()
        else:
            previous_value = value
        pto.wait_flag("S", "MTE2", event_id=previous_value + value)


@pto.jit(target="a5", mode="explicit")
def lexical_section_branch_merge_then_while_probe(limit: pto.i32):
    value = pto.const(0, dtype=pto.i32)
    one = pto.const(1, dtype=pto.i32)
    with pto.section("cube"):
        if pto.get_block_idx() < one:
            value = value + one
        index = pto.const(0, dtype=pto.i32)
        while index < limit:
            value = value + one
            index = index + one
        pto.wait_flag("S", "MTE2", event_id=value)


@pto.jit(target="a5", mode="explicit")
def lexical_section_uninitialized_conditional_probe():
    one = pto.const(1, dtype=pto.i32)
    with pto.section("cube"):
        if pto.get_block_idx() < one:
            value = one
        pto.wait_flag("S", "MTE2", event_id=value)


@pto.jit(target="a5", mode="explicit")
def lexical_section_conditional_loop_carry_probe():
    base = pto.const(0, dtype=pto.ui64)
    one = pto.const(1, dtype=pto.i32)
    with pto.section("vector"):
        if pto.get_block_idx() < one:
            with pto.vecscope():
                dst_ptr = pto.castptr(base, pto.ptr(pto.f32, "ub"))
                mask = pto.pset_b32(pto.MaskPattern.ALL)
                value = pto.vlds(dst_ptr, pto.const(0))
                for _ in range(2):
                    dst_ptr = pto.vsstb(
                        value,
                        dst_ptr,
                        pto.i16(32),
                        pto.i16(0),
                        mask,
                        post_update=pto.PostUpdate.ON,
                    )


@pto.jit(target="a5", mode="explicit")
def lexical_section_nested_conditional_rebinding_probe():
    one = pto.const(1, dtype=pto.i32)
    value = pto.const(0, dtype=pto.i32)
    with pto.section("cube"):
        if pto.get_block_idx() < one:
            if pto.get_block_idx() < one:
                value = one
            else:
                value = pto.const(2, dtype=pto.i32)
        else:
            value = pto.const(3, dtype=pto.i32)
        pto.wait_flag("S", "MTE2", event_id=value)


@pto.jit(target="a5", mode="explicit")
def lexical_section_loop_carry_probe():
    one = pto.const(1, dtype=pto.i32)
    m_tile = pto.const(0, dtype=pto.i64)
    n_tile = pto.const(0, dtype=pto.i64)
    with pto.section("cube"):
        for w in range(0, 4):
            if w < 2:
                m_tile = pto.get_block_idx()
            if 1 < (w & 3):
                n_tile = pto.get_block_idx()
            pto.wait_flag("S", "MTE2", event_id=m_tile)
            pto.wait_flag("S", "MTE2", event_id=n_tile)


@pto.jit(target="a5", mode="explicit", ast_rewrite=False)
def lexical_section_rebinding_no_rewrite_probe():
    event_id = pto.const(1, dtype=pto.i32)
    one = pto.const(1, dtype=pto.i32)
    with pto.section("cube"):
        event_id = event_id + one
        pto.wait_flag("S", "MTE2", event_id=event_id)
    with pto.section("vector"):
        pto.wait_flag("MTE2", "S", event_id=event_id)


@pto.jit(target="a5", mode="explicit", ast_rewrite=False)
def lexical_section_escape_probe():
    escaped = []
    with pto.section("cube"):
        escaped.append(pto.const(2, dtype=pto.i32))
    with pto.section("vector"):
        pto.wait_flag("MTE2", "S", event_id=escaped[0])


def _expect_raises(exc_type, callback, message):
    try:
        callback()
    except exc_type as exc:
        assert message in str(exc), str(exc)
        return
    raise AssertionError(f"expected {exc_type.__name__} containing {message!r}")


def main() -> None:
    text = explicit_sections_probe.compile().mlir_text()
    assert text.count("pto.section.cube {") == 1
    assert text.count("pto.section.vector {") == 1
    assert "pto.set_flag[<PIPE_MTE2>, <PIPE_S>, <EVENT_ID0>]" in text
    assert "pto.wait_flag_dyn[<PIPE_S>, <PIPE_MTE2>" in text
    assert "pto.wait_flag[<PIPE_MTE2>, <PIPE_S>, <EVENT_ID0>]" in text
    assert "pto.set_flag_dyn[<PIPE_S>, <PIPE_MTE2>" in text
    with make_context() as context:
        module = Module.parse(text, context)
        module.operation.verify()

    recovered_text = recovered_section_probe.compile().mlir_text()
    assert recovered_text.count("pto.section.cube {") == 1
    assert "pto.set_flag[<PIPE_MTE2>, <PIPE_S>, <EVENT_ID0>]" in recovered_text

    _expect_raises(TypeError, lambda: pto.section(1), "expects 'cube' or 'vector'")
    _expect_raises(ValueError, lambda: pto.section("scalar"), "expects 'cube' or 'vector'")
    _expect_raises(ValueError, lambda: pto.section("CUBE"), "expects 'cube' or 'vector'")
    _expect_raises(
        RuntimeError,
        lambda: duplicate_section_probe.compile(),
        "each physical section kind may appear at most once in a function",
    )
    _expect_raises(
        RuntimeError,
        lambda: nested_section_probe.compile(),
        "nested pto.section() scopes are not allowed",
    )
    _expect_raises(
        RuntimeError,
        lambda: loop_subkernel_then_section_probe.compile(),
        "each physical section kind may appear at most once in a function",
    )
    _expect_raises(
        RuntimeError,
        lambda: section_then_subkernel_probe.compile(),
        "each physical section kind may appear at most once in a function",
    )
    _expect_raises(
        RuntimeError,
        lambda: section_inside_subkernel_probe.compile(),
        "pto.section() is not allowed inside a cube or simd subkernel body",
    )

    lexical_text = lexical_section_rebinding_probe.compile().mlir_text()
    assert lexical_text.count("pto.section.cube {") == 1
    assert lexical_text.count("pto.section.vector {") == 1

    non_section_lexical_text = lexical_non_section_conditional_rebinding_probe.compile().mlir_text()
    assert non_section_lexical_text.count("scf.if") == 1
    with make_context() as context:
        module = Module.parse(non_section_lexical_text, context)
        module.operation.verify()

    conditional_lexical_text = lexical_section_conditional_rebinding_probe.compile().mlir_text()
    assert conditional_lexical_text.count("pto.section.cube {") == 1
    assert "scf.if" in conditional_lexical_text

    sibling_conditional_text = lexical_section_sibling_conditional_rebinding_probe.compile().mlir_text()
    assert sibling_conditional_text.count("pto.section.cube {") == 1
    assert sibling_conditional_text.count("pto.section.vector {") == 1
    assert sibling_conditional_text.count("scf.if") == 2
    with make_context() as context:
        module = Module.parse(sibling_conditional_text, context)
        module.operation.verify()

    sibling_single_sided_text = lexical_section_sibling_single_sided_conditional_rebinding_probe.compile().mlir_text()
    assert sibling_single_sided_text.count("pto.section.cube {") == 1
    assert sibling_single_sided_text.count("pto.section.vector {") == 1
    assert sibling_single_sided_text.count("scf.if") == 4
    with make_context() as context:
        module = Module.parse(sibling_single_sided_text, context)
        module.operation.verify()

    sequential_single_sided_text = lexical_section_sequential_single_sided_conditional_probe.compile().mlir_text()
    if_results = re.findall(r"^\s*(%\d+)(?::\d+)? = scf\.if", sequential_single_sided_text, re.MULTILINE)
    assert len(if_results) == 3
    second_if_text = sequential_single_sided_text.split(f"{if_results[1]}:2 = scf.if", 1)[1]
    second_if_text = second_if_text.split(f"{if_results[2]} = scf.if", 1)[0]
    assert re.search(
        rf"else \{{\s+scf\.yield {re.escape(if_results[0])}#0, {re.escape(if_results[0])}#1 : i64, i64",
        second_if_text,
    )
    third_if_text = sequential_single_sided_text.split(f"{if_results[2]} = scf.if", 1)[1]
    assert re.search(
        rf"else \{{\s+scf\.yield {re.escape(if_results[1])}#1 : i64",
        third_if_text,
    )
    with make_context() as context:
        module = Module.parse(sequential_single_sided_text, context)
        module.operation.verify()

    read_before_rebinding_text = lexical_section_single_sided_read_before_rebinding_probe.compile().mlir_text()
    assert re.search(
        r"scf\.yield %c0_i64, %[\d]+ : i64, i64",
        read_before_rebinding_text,
    )
    with make_context() as context:
        module = Module.parse(read_before_rebinding_text, context)
        module.operation.verify()

    branch_merge_then_while_text = lexical_section_branch_merge_then_while_probe.compile().mlir_text()
    assert branch_merge_then_while_text.count("pto.section.cube {") == 1
    assert "scf.if" in branch_merge_then_while_text
    assert "scf.while" in branch_merge_then_while_text
    with make_context() as context:
        module = Module.parse(branch_merge_then_while_text, context)
        module.operation.verify()

    nested_conditional_text = lexical_section_nested_conditional_rebinding_probe.compile().mlir_text()
    assert nested_conditional_text.count("pto.section.cube {") == 1
    assert nested_conditional_text.count("scf.if") == 2
    with make_context() as context:
        module = Module.parse(nested_conditional_text, context)
        module.operation.verify()

    _expect_raises(
        PTODSLAstRewriteError,
        lambda: lexical_section_uninitialized_conditional_probe.compile(),
        "reads a section-local value before it is initialized",
    )

    conditional_loop_carry_text = lexical_section_conditional_loop_carry_probe.compile().mlir_text()
    assert conditional_loop_carry_text.count("pto.section.vector {") == 1
    assert "scf.for" in conditional_loop_carry_text

    loop_carry_text = lexical_section_loop_carry_probe.compile().mlir_text()
    assert loop_carry_text.count("pto.section.cube {") == 1
    assert "scf.for" in loop_carry_text

    no_rewrite_text = lexical_section_rebinding_no_rewrite_probe.compile().mlir_text()
    vector_start = no_rewrite_text.index("pto.section.vector {")
    vector_text = no_rewrite_text[vector_start:]
    assert "arith.addi" not in vector_text
    assert "pto.wait_flag_dyn" in vector_text

    _expect_raises(
        RuntimeError,
        lambda: lexical_section_escape_probe.compile(),
        "cannot use a value defined in pto.section.cube outside that physical section",
    )
    _expect_raises(
        RuntimeError,
        lambda: kernel_kind_section_probe.compile(),
        "cannot be combined with explicit @pto.jit(kernel_kind=...)",
    )
    _expect_raises(
        RuntimeError,
        lambda: auto_mode_section_probe.compile(),
        "only available in @pto.jit(mode=\"explicit\")",
    )
    _expect_raises(
        RuntimeError,
        lambda: pto.section("cube").__enter__(),
        "may only be used while tracing",
    )

    unspecified_text = unspecified_kernel_kind_section_probe.compile().mlir_text()
    assert "pto.section.vector {" in unspecified_text


if __name__ == "__main__":
    main()
