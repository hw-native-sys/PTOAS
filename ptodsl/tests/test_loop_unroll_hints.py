#!/usr/bin/env python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Loop-unroll hint tests: pto.for_/pto.range -> pto.unroll / pto.unroll_factor attrs.

Covers the frontend half of the loop-unroll hint design
(docs/designs/ptodsl-loop-unroll-hint-design.md): hint validation, attribute
placement on scf.for for all loop construction paths (explicit pto.for_,
carry loops, tile-template tracing, and AST-rewritten ``for i in
pto.range(...)``), the marker misuse diagnostics, and the "no hint -> no
pto.unroll* attribute" contract.
"""

import re

from ptodsl import pto
from ptodsl._ast_rewrite import PTODSLAstRewriteError


def _assert_loop_attr(mlir_text: str, attr_pattern: str, context: str):
    assert re.search(r"scf\.for", mlir_text), f"{context}: no scf.for in MLIR:\n{mlir_text}"
    assert re.search(attr_pattern, mlir_text), (
        f"{context}: expected attribute matching {attr_pattern!r} in MLIR:\n{mlir_text}"
    )


# ── AST-rewritten pto.range kernels ──────────────────────────────────────────


@pto.jit(target="a5")
def range_hint_full():
    acc = pto.const(0, dtype=pto.i32)
    for i in pto.range(4, unroll="full"):
        acc = acc + pto.const(1, dtype=pto.i32)
    _ = acc


@pto.jit(target="a5")
def range_hint_factor():
    acc = pto.const(0, dtype=pto.i32)
    for i in pto.range(0, 8, 2, unroll_factor=4):
        acc = acc + pto.const(1, dtype=pto.i32)
    _ = acc


@pto.jit(target="a5")
def range_hint_carry(limit: pto.i32):
    acc = pto.const(0, dtype=pto.i32)
    for i in pto.range(limit, unroll_factor=2):
        acc = acc + pto.const(1, dtype=pto.i32)
    _ = acc


# ── Explicit pto.for_ kernels ────────────────────────────────────────────────


@pto.jit(target="a5")
def explicit_for_hint_full():
    acc = pto.const(0, dtype=pto.i32)
    with pto.for_(0, 4, step=1, unroll="full") as i:
        acc = acc + pto.const(1, dtype=pto.i32)
    _ = acc


@pto.jit(target="a5")
def explicit_for_carry_hint():
    acc = pto.alloc_tile(shape=[1, 8], dtype=pto.f32)
    loop = pto.for_(0, 8, step=1, unroll_factor=2).carry(acc=acc)
    with loop:
        loop.update(acc=loop.acc)
    _ = loop.final("acc")


# ── Invalid kernels (compile must raise) ─────────────────────────────────────


@pto.jit(target="a5")
def range_hint_unknown_keyword():
    acc = pto.const(0, dtype=pto.i32)
    for i in pto.range(4, unroll_count=4):
        acc = acc + pto.const(1, dtype=pto.i32)
    _ = acc


@pto.jit(target="a5")
def range_hint_with_break():
    acc = pto.const(0, dtype=pto.i32)
    for i in pto.range(8, unroll="full"):
        if i == pto.const(3, dtype=pto.i32):
            break
        acc = acc + pto.const(1, dtype=pto.i32)
    _ = acc


@pto.jit(target="a5")
def range_hint_negative_step():
    acc = pto.const(0, dtype=pto.i32)
    for i in pto.range(4, 0, -1, unroll="full"):
        acc = acc + pto.const(1, dtype=pto.i32)
    _ = acc


@pto.jit(target="a5")
def plain_range_negative_step():
    # No hint: the negative-step rejection is a behavior change of the plain
    # range(...) path itself (previously lowered to a silent zero-trip loop).
    acc = pto.const(0, dtype=pto.i32)
    for i in range(4, 0, -1):
        acc = acc + pto.const(1, dtype=pto.i32)
    _ = acc


@pto.jit(target="a5")
def plain_range_zero_step():
    acc = pto.const(0, dtype=pto.i32)
    for i in range(0, 8, 0):
        acc = acc + pto.const(1, dtype=pto.i32)
    _ = acc


@pto.jit(target="a5")
def plain_range_false_step():
    # bool is an int subclass; step=False must be rejected like step=0 while
    # step=True (== 1) stays legal.
    acc = pto.const(0, dtype=pto.i32)
    for i in range(0, 8, False):
        acc = acc + pto.const(1, dtype=pto.i32)
    _ = acc


@pto.jit(target="a5")
def plain_range_true_step():
    acc = pto.const(0, dtype=pto.i32)
    for i in range(0, 8, True):
        acc = acc + pto.const(1, dtype=pto.i32)
    _ = acc


@pto.jit(target="a5")
def plain_range_no_hint():
    acc = pto.const(0, dtype=pto.i32)
    for i in range(4):
        acc = acc + pto.const(1, dtype=pto.i32)
    _ = acc


@pto.jit(target="a5")
def explicit_for_no_hint():
    acc = pto.const(0, dtype=pto.i32)
    with pto.for_(0, 4, step=1) as i:
        acc = acc + pto.const(1, dtype=pto.i32)
    _ = acc


# ── Tile-template tracing path ───────────────────────────────────────────────
#
# tilelib templates author loops via an @rewrite_jit_function core: the AST
# rewrite turns ``for i in pto.range(..., unroll_factor=N)`` into a
# ``pto.for_(..., unroll_factor=N)`` call, which the template trace session
# (``_TemplateTrace``) builds like any other hinted loop.  This exercises the
# same kwarg forwarding a real template would rely on.

try:
    import ptodsl.tilelib as tilelib
    from ptodsl._ast_rewrite import rewrite_jit_function
    from ptodsl.tilelib import ScalarType, TileSpec

    @rewrite_jit_function
    def _tt_hint_core():
        for i in pto.range(0, 4, unroll_factor=3):
            pass

    @tilelib.tile_template(
        op="test.unroll_hint",
        target="a5",
        name="unroll_hint_loop",
        loop_depth=1,
        register=False,
    )
    def _tt_unroll_hint(src: pto.Tile, dst: pto.Tile):
        _tt_hint_core()

    _TT_SPECS = {
        "src": TileSpec(shape=(8, 64), dtype=ScalarType("f32")),
        "dst": TileSpec(shape=(8, 64), dtype=ScalarType("f32")),
    }
except ImportError:  # tilelib unavailable: the tile-template test is skipped
    _tt_unroll_hint = None


def _expect_raises(exc_types, fn, needle: str, context: str):
    try:
        fn()
    except exc_types as exc:
        assert needle in str(exc), f"{context}: {needle!r} not in {exc}"
        return
    raise AssertionError(f"{context}: expected {exc_types}, no error raised")


def main():
    # 1. pto.range(unroll="full") on a constant loop.
    text = range_hint_full.compile().mlir_text()
    _assert_loop_attr(text, r'pto\.unroll\s*=\s*"full"', "pto.range unroll=full")

    # 2. pto.range(0, 8, 2, unroll_factor=4).
    text = range_hint_factor.compile().mlir_text()
    _assert_loop_attr(text, r"pto\.unroll_factor\s*=\s*4", "pto.range unroll_factor=4")

    # 3. pto.range with a dynamic bound and a loop-carried value.
    text = range_hint_carry.compile().mlir_text()
    _assert_loop_attr(text, r"pto\.unroll_factor\s*=\s*2", "pto.range carry unroll_factor=2")

    # 4. Explicit pto.for_(..., unroll="full").
    text = explicit_for_hint_full.compile().mlir_text()
    _assert_loop_attr(text, r'pto\.unroll\s*=\s*"full"', "pto.for_ unroll=full")

    # 5. Explicit pto.for_(..., unroll_factor=2).carry(...).
    text = explicit_for_carry_hint.compile().mlir_text()
    _assert_loop_attr(text, r"pto\.unroll_factor\s*=\s*2", "pto.for_ carry unroll_factor=2")

    # 6. Hint validation happens eagerly at loop construction.
    _expect_raises(
        (ValueError,),
        lambda: pto.for_(0, 4, step=1, unroll="sometimes"),
        "unroll= expects one of",
        "bad unroll value",
    )
    _expect_raises(
        (ValueError,),
        lambda: pto.for_(0, 4, step=1, unroll="full", unroll_factor=4),
        "mutually exclusive",
        "conflicting hints",
    )
    _expect_raises(
        (ValueError,),
        lambda: pto.for_(0, 4, step=1, unroll_factor=0),
        "positive integer",
        "non-positive factor",
    )
    _expect_raises(
        (TypeError,),
        lambda: pto.for_(0, 4, step=1, unroll_factor="4"),
        "positive Python int",
        "non-int factor",
    )
    # The attribute is a signless i32; factors beyond INT32_MAX would wrap.
    _expect_raises(
        (ValueError,),
        lambda: pto.for_(0, 4, step=1, unroll_factor=2**31),
        "signless i32",
        "factor 2**31",
    )
    _expect_raises(
        (ValueError,),
        lambda: pto.for_(0, 4, step=1, unroll_factor=2**32),
        "signless i32",
        "factor 2**32",
    )

    # 7. pto.range is an AST-rewrite marker: iterating it directly must fail.
    _expect_raises(
        (RuntimeError,),
        lambda: iter(pto.range(3)),
        "may only be used as the iterable of a 'for' loop",
        "pto.range marker misuse",
    )

    # 8. Unknown pto.range keywords are rejected by the AST rewrite.
    _expect_raises(
        (PTODSLAstRewriteError,),
        lambda: range_hint_unknown_keyword.compile(),
        "unroll_count",
        "unknown pto.range keyword",
    )

    # 9. Unroll hints are unsupported on break/continue loops (scf.while path).
    _expect_raises(
        (PTODSLAstRewriteError,),
        lambda: range_hint_with_break.compile(),
        "not supported on loops with",
        "pto.range hint with break",
    )

    # 10. The plain scf.for path requires a positive step: a constant negative
    # step would silently lower to zero iterations.
    _expect_raises(
        (PTODSLAstRewriteError,),
        lambda: range_hint_negative_step.compile(),
        "positive step",
        "pto.range negative step",
    )

    # 11. The same rejection applies to plain range(...) without any hint (the
    # behavior-change regression surface), and step=0 / step=False are
    # rejected like Python's own range, while step=True (== 1) stays legal.
    _expect_raises(
        (PTODSLAstRewriteError,),
        lambda: plain_range_negative_step.compile(),
        "positive step",
        "plain range negative step",
    )
    _expect_raises(
        (PTODSLAstRewriteError,),
        lambda: plain_range_zero_step.compile(),
        "non-zero step",
        "plain range zero step",
    )
    _expect_raises(
        (PTODSLAstRewriteError,),
        lambda: plain_range_false_step.compile(),
        "non-zero step",
        "plain range step=False",
    )
    text = plain_range_true_step.compile().mlir_text()
    _assert_loop_attr(text, r"scf\.for", "plain range step=True compiles")

    # 11b. "enable"/"disable" were removed with the metadata forwarding path;
    # only "full" remains a valid unroll= value.
    _expect_raises(
        (ValueError,),
        lambda: pto.for_(0, 4, step=1, unroll="enable"),
        "unroll= expects one of",
        "removed enable hint",
    )
    _expect_raises(
        (ValueError,),
        lambda: pto.for_(0, 4, step=1, unroll="disable"),
        "unroll= expects one of",
        "removed disable hint",
    )

    # 12. Type traps at eager validation: bool is an int subclass and must not
    # be accepted as a factor; floats and non-str unroll values are rejected.
    _expect_raises(
        (TypeError,),
        lambda: pto.for_(0, 4, step=1, unroll_factor=True),
        "positive Python int",
        "bool factor",
    )
    _expect_raises(
        (TypeError,),
        lambda: pto.for_(0, 4, step=1, unroll_factor=4.0),
        "positive Python int",
        "float factor",
    )
    _expect_raises(
        (ValueError,),
        lambda: pto.for_(0, 4, step=1, unroll=1),
        "unroll= expects one of",
        "non-str unroll value",
    )

    # 13. No-hint contract: loops built without hints carry no pto.unroll*
    # attribute at all.
    for fn, context in (
        (plain_range_no_hint, "plain range without hint"),
        (explicit_for_no_hint, "explicit pto.for_ without hint"),
        (plain_range_true_step, "plain range step=True"),
    ):
        text = fn.compile().mlir_text()
        assert re.search(r"scf\.for", text), f"{context}: no scf.for in MLIR:\n{text}"
        assert not re.search(r'"?pto\.unroll(_factor)?"?\s*=', text), (
            f"{context}: unexpected pto.unroll* attribute in MLIR:\n{text}"
        )

    # 14. Tile-template tracing path: for_ -> _TraceBuilder.for_ -> _enter_for
    # must forward unroll_factor all the way onto the generated scf.for.
    if _tt_unroll_hint is not None:
        text = _tt_unroll_hint.specialize(**_TT_SPECS).mlir_text()
        _assert_loop_attr(
            text, r"pto\.unroll_factor\s*=\s*3", "tile-template for_ unroll_factor=3"
        )

    print("PASS test_loop_unroll_hints")


if __name__ == "__main__":
    main()
