#!/usr/bin/env python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

import ast
import re

from ptodsl import pto
from ptodsl._ast_rewrite import _drop_unreachable_tails


def _while_iter_arg_count(mlir_text: str) -> int:
    """Count the loop-carried state slots of the top-level scf.while op."""
    match = re.search(r"scf\.while\s*\([^)]*\)\s*:\s*\(([^)]*)\)", mlir_text)
    assert match is not None, "scf.while signature not found in MLIR"
    return len([part for part in match.group(1).split(",") if part.strip()])


def _region_close(text: str, open_brace: int) -> int:
    """Return the index just past the ``}`` matching the ``{`` at open_brace."""
    assert text[open_brace] == "{"
    depth = 0
    for i in range(open_brace, len(text)):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                return i + 1
    raise AssertionError("unbalanced braces in MLIR text")


def _is_flag_merge_result(mlir_text: str, cond_token: str) -> bool:
    """True when cond_token (``%X`` or ``%X#k``) is defined by a
    result-producing ``scf.if`` merging i1 control-flag state.

    The break/continue predication contract requires the tail guard's
    condition to be the *merged* ``active`` flag — a result of the
    flag-merge ``scf.if`` — not an arbitrary dynamic condition (e.g. the
    original ``if value > 0`` test, which would leave the tail
    unpredicated).  Single-result merges print as ``%X = scf.if ... ->
    (i1)``; multi-result merges as ``%X:N = scf.if ... -> (i1, ...)`` with
    the condition referenced as ``%X#k``.
    """
    if "#" in cond_token:
        base, idx = cond_token[1:].split("#", 1)
        idx = int(idx)
        m = re.search(
            rf"%{re.escape(base)}:(\d+) = scf\.if [^\n]*-> \(([^)]*)\)", mlir_text)
        if m is None or idx >= int(m.group(1)):
            return False
        types = [t.strip() for t in m.group(2).split(",")]
        return idx < len(types) and types[idx] == "i1"
    base = cond_token[1:]
    m = re.search(
        rf"%{re.escape(base)} = scf\.if [^\n]*-> \(([^)]*)\)", mlir_text)
    return m is not None and m.group(1).strip() == "i1"


def _assert_tail_guarded(mlir_text: str, tail_op_pattern: str):
    """Lock the break/continue tail-skip contract in the emitted IR.

    Statements after a break/continue (or after an if containing one) must be
    predicated on the merged ``active`` flag: the tail operation must appear
    *inside the region* of a result-producing ``scf.if`` whose condition is
    the i1 result of an earlier flag-merge ``scf.if`` (never a constant, a
    block argument, or an unrelated dynamic condition).  Region containment
    is checked by brace depth, not mere text order, and the condition's
    defining op is resolved, so a tail that merely sits inside the original
    dynamic ``if value > 0`` region does not satisfy this contract.  The
    dead constant-true whole-body guard must be gone.
    """
    assert not re.search(r"scf\.if %true", mlir_text), (
        "constant-true guard still present; break/continue tails are unpredicated")
    tail = re.search(tail_op_pattern, mlir_text)
    assert tail is not None, f"tail op {tail_op_pattern!r} not found in MLIR"
    guards = list(re.finditer(r"= scf\.if (%(?!true|false)\w+(?:#\d+)?) -> \(", mlir_text))
    assert guards, "no active-guarded tail region found"
    for guard in guards:
        if not _is_flag_merge_result(mlir_text, guard.group(1)):
            continue
        open_brace = mlir_text.index("{", guard.end())
        if guard.start() < tail.start() < _region_close(mlir_text, open_brace):
            return
    raise AssertionError(
        "tail op is not contained in a region guarded by the merged active "
        "flag (an i1 result of a flag-merge scf.if); statements after "
        "break/continue would execute unconditionally")


@pto.jit(target="a5")
def runtime_while_probe(limit: pto.i32):
    value = pto.const(0, dtype=pto.i32)
    while value < limit:
        value = value + pto.const(1, dtype=pto.i32)
    _ = value


@pto.jit(target="a5")
def runtime_while_break_continue(limit: pto.i32):
    value = pto.const(0, dtype=pto.i32)
    while value < limit:
        value = value + pto.const(1, dtype=pto.i32)
        if value == pto.const(1, dtype=pto.i32):
            continue
        if value == pto.const(3, dtype=pto.i32):
            break
        value = value + pto.const(10, dtype=pto.i32)
    _ = value


@pto.jit(target="a5")
def runtime_while_else(limit: pto.i32):
    value = pto.const(0, dtype=pto.i32)
    while value < limit:
        value = value + pto.const(1, dtype=pto.i32)
    else:
        value = value + pto.const(1, dtype=pto.i32)
    _ = value + pto.const(1, dtype=pto.i32)


@pto.jit(target="a5")
def runtime_for_break_continue(limit: pto.i32):
    value = pto.const(0, dtype=pto.i32)
    for i in range(limit):
        if i == pto.const(1, dtype=pto.i32):
            continue
        if i == pto.const(3, dtype=pto.i32):
            break
        value = value + pto.const(1, dtype=pto.i32)
    else:
        value = value + pto.const(2, dtype=pto.i32)
    _ = value + pto.const(1, dtype=pto.i32)


@pto.jit(target="a5")
def runtime_while_static_break(limit: pto.i32):
    value = pto.const(0, dtype=pto.i32)
    while value < limit:
        for _ in pto.static_range(2):
            break
        value = value + pto.const(1, dtype=pto.i32)
        if value == pto.const(2, dtype=pto.i32):
            break
    _ = value + pto.const(1, dtype=pto.i32)


# ---------------------------------------------------------------------------
# Issue #1256: _rewrite_while must not carry loop-local temporaries.
#
# Before: every body load was treated as loop-carried state, so a temporary
# written before being read each iteration (e.g. ``col = base + index``) was
# read while still unbound at the pto._while(...) setup, raising
# UnboundLocalError during compile.  After: only names read by the test, read
# before assignment inside the body, or live after the loop are carried.
# ---------------------------------------------------------------------------


@pto.jit(target="a5")
def issue_1256_while_local_temp(limit: pto.i32):
    base = pto.const(2, dtype=pto.i32)
    index = pto.const(0, dtype=pto.i32)
    total = pto.const(0, dtype=pto.i32)
    while index < limit:
        col = base + index
        total = total + col
        index = index + 1
    _ = total


@pto.jit(target="a5")
def issue_1256_while_break_flag(limit: pto.i32):
    value = pto.const(0, dtype=pto.i32)
    while value < limit:
        value = value + pto.const(1, dtype=pto.i32)
        should_break = value == pto.const(3, dtype=pto.i32)
        if should_break:
            break
    _ = value


@pto.jit(target="a5")
def issue_1256_while_branch_temp(limit: pto.i32):
    low = pto.const(0, dtype=pto.i32)
    high = limit
    while low < high:
        mid = low + pto.const(2, dtype=pto.i32)
        take_upper = mid < pto.const(4, dtype=pto.i32)
        if take_upper:
            low = mid + pto.const(1, dtype=pto.i32)
        else:
            high = mid + pto.const(0, dtype=pto.i32)
    _ = low
    _ = high


@pto.jit(target="a5")
def issue_1256_while_conditional_carry(limit: pto.i32):
    value = pto.const(0, dtype=pto.i32)
    while value < limit:
        if value < pto.const(2, dtype=pto.i32):
            value = value + pto.const(1, dtype=pto.i32)
    _ = value


@pto.jit(target="a5")
def while_unbound_partial_liveout(limit: pto.i32, cond: pto.i1):
    index = pto.const(0, dtype=pto.i32)
    one = pto.const(1, dtype=pto.i32)
    while index < limit:
        if cond:
            value = one
        index = index + one
    _ = value


@pto.jit(target="a5")
def while_bound_partial_liveout(limit: pto.i32, cond: pto.i1):
    index = pto.const(0, dtype=pto.i32)
    value = pto.const(0, dtype=pto.i32)
    one = pto.const(1, dtype=pto.i32)
    while index < limit:
        if cond:
            value = one
        index = index + one
    _ = value


@pto.jit(target="a5")
def issue_1256_while_break_flags_only(limit: pto.i32, running: pto.i32, probe: pto.i32):
    # Controlled loop with no user-level carry: the test only reads
    # loop-invariant names and the body store is a loop-local temporary,
    # so the active/did_break control flags provide the loop-carried state.
    while running < limit:
        t = probe
        if t < pto.const(3, dtype=pto.i32):
            break
    _ = running


@pto.jit(target="a5")
def issue_1256_while_continue_else_flags_only(
    limit: pto.i32, running: pto.i32, probe: pto.i32
):
    # The loop test is invariant and all authored stores are loop-local.
    # continue and else therefore rely exclusively on the generated control
    # flags for their loop-carried state.
    while running < limit:
        t = probe
        if t < pto.const(3, dtype=pto.i32):
            continue
    else:
        t = probe + pto.const(1, dtype=pto.i32)
    _ = running


# Issue #1256 exact repros (https://github.com/hw-native-sys/PTOAS/issues/1256).
# Case 2 uses a literal ``while True:`` test; the generated condition must
# materialize the literal as an i1 constant so the break/continue guard
# (``active/did_break``) can combine with it through the runtime ``and`` op.


@pto.jit(target="a5")
def issue_1256_exact_while_local_temp(limit: pto.i32, base: pto.i32):
    index = pto.const(0, dtype=pto.i32)
    total = pto.const(0, dtype=pto.i32)
    while index < limit:
        col = base + index
        total = total + col
        index = index + pto.const(1, dtype=pto.i32)
    _ = total


@pto.jit(target="a5")
def issue_1256_exact_while_true_break(limit: pto.i32):
    value = pto.const(0, dtype=pto.i32)
    while True:
        should_break = value >= limit
        if should_break:
            break
        value = value + pto.const(1, dtype=pto.i32)
    _ = value


@pto.jit(target="a5")
def issue_1256_exact_while_branch_cond(limit: pto.i32, pivot: pto.i32):
    low = pto.const(0, dtype=pto.i32)
    high = limit
    mid = pto.const(0, dtype=pto.i32)
    while low < high:
        mid = (low + high) // pto.const(2, dtype=pto.i32)
        take_upper = mid < pivot
        if take_upper:
            low = mid + pto.const(1, dtype=pto.i32)
        else:
            high = mid
    _ = low


# ---------------------------------------------------------------------------
# Break/continue tail predication: statements after a control transfer must
# only execute while the iteration is still active.  Before the fix the whole
# body was wrapped in a single guard on the entry value of ``active`` (a
# constant true, since the prologue resets it), so a tail statement such as
# ``value = value + 1`` after ``if should_break: break`` executed
# unconditionally in the breaking iteration (issue #1256 follow-up).
# ---------------------------------------------------------------------------


@pto.jit(target="a5")
def while_continue_skips_tail(limit: pto.i32):
    value = pto.const(0, dtype=pto.i32)
    while value < limit:
        value = value + pto.const(1, dtype=pto.i32)
        if value == pto.const(2, dtype=pto.i32):
            continue
        value = value + pto.const(10, dtype=pto.i32)
    _ = value


@pto.jit(target="a5")
def while_continue_then_break_flags(limit: pto.i32):
    # The continue guard's tail contains a break: the guard must merge
    # active/did_break back out, or the break would be lost (or its SSA
    # value would leak out of the guard region).
    value = pto.const(0, dtype=pto.i32)
    while value < limit:
        if value == pto.const(1, dtype=pto.i32):
            continue
        if value == pto.const(3, dtype=pto.i32):
            break
        value = value + pto.const(1, dtype=pto.i32)
    _ = value


@pto.jit(target="a5")
def while_nested_if_break_tail(limit: pto.i32):
    # Tails must be predicated at every block level: inside the branch that
    # contains the break, and in the enclosing loop body after the if.
    value = pto.const(0, dtype=pto.i32)
    while value < limit:
        if value > pto.const(0, dtype=pto.i32):
            if value == pto.const(3, dtype=pto.i32):
                break
            value = value + pto.const(1, dtype=pto.i32)
        value = value + pto.const(10, dtype=pto.i32)
    _ = value


@pto.jit(target="a5")
def while_unconditional_break_dead_tail(limit: pto.i32):
    # The tail of an unconditional break is statically dead; it is dropped
    # before analysis instead of being carried, traced, or diagnosed,
    # matching Python's tolerance for unreachable code.
    value = pto.const(0, dtype=pto.i32)
    while value < limit:
        value = value + pto.const(1, dtype=pto.i32)
        break
        value = value + pto.const(10, dtype=pto.i32)
    _ = value


@pto.jit(target="a5")
def for_break_tail_guard(limit: pto.i32):
    # The controlled for-loop lowering shares the same predication scheme.
    value = pto.const(0, dtype=pto.i32)
    for i in range(limit):
        if i == pto.const(3, dtype=pto.i32):
            break
        value = value + pto.const(1, dtype=pto.i32)
    _ = value


@pto.jit(target="a5")
def for_break_tail_carry_merge(limit: pto.i32):
    # The tail assignment to a loop-carried value must be merged out of the
    # active guard even though liveness at the break point no longer sees a
    # later read before the loop update.
    value = pto.const(0, dtype=pto.i32)
    for i in range(limit):
        value = value + pto.const(1, dtype=pto.i32)
        if i == pto.const(3, dtype=pto.i32):
            break
        value = value + pto.const(10, dtype=pto.i32)
    _ = value


@pto.jit(target="a5")
def while_break_dead_unbound_tail(limit: pto.i32):
    # The dead tail references a name Python would never bind.  It must be
    # dropped before analysis instead of entering the carry state — before
    # the truncation fix this surfaced as UnboundLocalError at the
    # pto._while(...) setup.
    value = pto.const(0, dtype=pto.i32)
    while value < limit:
        value = value + pto.const(1, dtype=pto.i32)
        break
        dead = dead + pto.const(1, dtype=pto.i32)
    _ = value


@pto.jit(target="a5")
def while_if_break_dead_branch_tail(limit: pto.i32):
    # Truncation also applies inside branch bodies: the dead store after the
    # break must not be carried or traced.
    value = pto.const(0, dtype=pto.i32)
    while value < limit:
        if value == pto.const(2, dtype=pto.i32):
            break
            dead = dead + pto.const(1, dtype=pto.i32)
        value = value + pto.const(1, dtype=pto.i32)
    _ = value


@pto.jit(target="a5")
def while_break_dead_slot_tail(limit: pto.i32):
    # A dead static-subscript store after break is dropped with the tail, so
    # it no longer trips the "static subscript carries" diagnostic.
    values = [0]
    value = pto.const(0, dtype=pto.i32)
    while value < limit:
        value = value + pto.const(1, dtype=pto.i32)
        break
        values[0] = value
    _ = value


@pto.jit(target="a5")
def while_if_else_break_dead_tail(limit: pto.i32):
    # An if whose both branches break always exits the iteration, so the
    # outer tail after it is dead and must be dropped even though the if
    # itself is not a bare break/continue.
    value = pto.const(0, dtype=pto.i32)
    while value < limit:
        value = value + pto.const(1, dtype=pto.i32)
        if value > pto.const(2, dtype=pto.i32):
            break
        else:
            break
        dead = dead + pto.const(10, dtype=pto.i32)
    _ = value


class _NullContext:
    def __enter__(self):
        return None

    def __exit__(self, *exc):
        return False


@pto.jit(target="a5")
def while_with_break_dead_tail(limit: pto.i32):
    # A with body that always breaks never falls through: the outer tail is
    # dead (the context exit runs natively at trace time, then the transfer
    # propagates).
    value = pto.const(0, dtype=pto.i32)
    while value < limit:
        value = value + pto.const(1, dtype=pto.i32)
        with _NullContext():
            break
        dead = dead + pto.const(10, dtype=pto.i32)
    _ = value


@pto.jit(target="a5")
def while_try_break_dead_tail(limit: pto.i32):
    # try/finally whose body always breaks: the finally block runs (here a
    # real store to a carried name), then the transfer propagates and the
    # outer tail is dead.
    value = pto.const(0, dtype=pto.i32)
    while value < limit:
        value = value + pto.const(1, dtype=pto.i32)
        try:
            break
        finally:
            value = value + pto.const(0, dtype=pto.i32)
        dead = dead + pto.const(10, dtype=pto.i32)
    _ = value


def unsupported_while_subscript(limit: pto.i32):
    values = [0]
    value = pto.const(0, dtype=pto.i32)
    while value < limit:
        values[0] = value
        value = value + pto.const(1, dtype=pto.i32)


def main():
    # Dead-tail truncation must account for exception paths.  A break in the
    # try body does not make the statement transfer on an exception caught by
    # a falling-through handler, while a transferring handler (or finally)
    # does make the whole statement non-fallthrough.
    def cleaned_loop_body(source):
        loop = ast.parse(source).body[0]
        return _drop_unreachable_tails(loop.body)

    assert len(cleaned_loop_body(
        "while cond:\n"
        "    try:\n"
        "        break\n"
        "    except Exception:\n"
        "        value = value + one\n"
        "    dead = dead + ten\n"
    )) == 2, "falling-through except handler must keep the outer tail"
    assert len(cleaned_loop_body(
        "while cond:\n"
        "    try:\n"
        "        break\n"
        "    except Exception:\n"
        "        continue\n"
        "    dead = dead + ten\n"
    )) == 1, "transferring except handlers must drop the outer tail"
    assert len(cleaned_loop_body(
        "while cond:\n"
        "    try:\n"
        "        value = value + one\n"
        "    finally:\n"
        "        continue\n"
        "    dead = dead + ten\n"
    )) == 1, "transferring finally blocks must drop the outer tail"

    text = runtime_while_probe.compile().mlir_text()
    assert "scf.while" in text
    assert "scf.condition" in text
    assert "scf.yield" in text

    for fn in (runtime_while_break_continue, runtime_while_else, runtime_for_break_continue,
               runtime_while_static_break):
        loop_text = fn.compile().mlir_text()
        assert "scf.while" in loop_text
        assert "scf.condition" in loop_text
        assert "scf.yield" in loop_text

    # Issue #1256 regressions: loop-local temporaries must not be carried.
    # issue_1256_while_local_temp / break_flag / branch_temp all used to raise
    # UnboundLocalError at the pto._while(...) setup.  Besides compiling, each
    # case also locks the exact number of loop-carried slots of the emitted
    # scf.while, so a future regression that re-adds a loop-local temporary to
    # the carry state (the #1256 defect) fails this contract.
    carry_contract = {
        issue_1256_while_local_temp: 2,               # index, total
        issue_1256_while_break_flag: 3,               # value + control flags
        issue_1256_while_branch_temp: 2,              # low, high
        issue_1256_while_conditional_carry: 1,        # value
        issue_1256_while_break_flags_only: 2,         # active, did_break
        issue_1256_while_continue_else_flags_only: 2, # active, did_break
        issue_1256_exact_while_local_temp: 2,         # index, total
        issue_1256_exact_while_true_break: 3,         # value + control flags
        issue_1256_exact_while_branch_cond: 2,        # low, high
    }
    for fn, expected_carries in carry_contract.items():
        loop_text = fn.compile().mlir_text()
        assert "scf.while" in loop_text
        assert "scf.condition" in loop_text
        actual = _while_iter_arg_count(loop_text)
        assert actual == expected_carries, (
            f"{fn.__name__}: expected {expected_carries} loop-carried slots, got {actual}; "
            "loop-local temporaries must not enter the carry state")

    # While-loop partial live-outs use the same definite-binding rule as
    # runtime for-loops: an unbound default path is diagnosed explicitly,
    # while a value initialized before the loop becomes an ordinary carry.
    try:
        while_unbound_partial_liveout.compile()
    except Exception as exc:
        assert "last-iteration-only" in str(exc), str(exc)
    else:
        raise AssertionError("unbound partial while live-out must be diagnosed")
    bound_text = while_bound_partial_liveout.compile().mlir_text()
    assert "scf.while" in bound_text
    assert _while_iter_arg_count(bound_text) == 2

    # Break/continue tail predication: the statement after a control transfer
    # must sit inside a region guarded by the merged active flag.
    # issue_1256_exact_while_true_break is the reporter's kernel: the addi
    # after ``if should_break: break`` used to execute unconditionally.
    _assert_tail_guarded(
        issue_1256_exact_while_true_break.compile().mlir_text(),
        r"arith\.addi %[\w#]+, %c1_i32")
    _assert_tail_guarded(
        while_continue_skips_tail.compile().mlir_text(),
        r"arith\.addi %[\w#]+, %c10_i32")
    _assert_tail_guarded(
        while_nested_if_break_tail.compile().mlir_text(),
        r"arith\.addi %[\w#]+, %c10_i32")
    _assert_tail_guarded(
        for_break_tail_guard.compile().mlir_text(),
        r"arith\.addi %[\w#]+, %c1_i32")
    for_tail_carry_text = for_break_tail_carry_merge.compile().mlir_text()
    _assert_tail_guarded(for_tail_carry_text, r"arith\.addi %[\w#]+, %c10_i32")
    assert re.search(r"= scf\.if %\w+(?:#\d+)? -> \(i1, i32\)", for_tail_carry_text), (
        "for break tail guard must merge active + loop-carried value so the "
        "loop update does not read a stale SSA value")

    # The continue-guard's tail contains a break: the guard must merge both
    # control flags back out (value + active + did_break).
    flags_text = while_continue_then_break_flags.compile().mlir_text()
    _assert_tail_guarded(flags_text, r"arith\.addi %[\w#]+, %c1_i32")
    assert re.search(r"= scf\.if %\w+(?:#\d+)? -> \(i1, i1, i32\)", flags_text), (
        "outer guard must merge value + active + did_break so the nested "
        "break survives the guard region")

    # Nested branch tail: the +1 addi inside the branch must also sit inside
    # an active-guarded region (verified by brace-depth containment).
    nested_text = while_nested_if_break_tail.compile().mlir_text()
    _assert_tail_guarded(nested_text, r"arith\.addi %[\w#]+, %c1_i32")

    # The dead tail of an unconditional break is truncated before analysis:
    # the dead addi must not appear in the IR at all (neither executed nor
    # wrapped in a constant-false guard).
    dead_text = while_unconditional_break_dead_tail.compile().mlir_text()
    assert not re.search(r"arith\.addi %[\w#]+, %c10_i32", dead_text), (
        "dead tail after unconditional break must be dropped, not emitted")
    assert not re.search(r"scf\.if %false", dead_text), (
        "dead tail after unconditional break must be dropped, not false-guarded")

    # The new kernels also lock their carry counts (value + control flags;
    # the for-loop additionally carries its induction variable).  The dead-*
    # kernels prove truncation keeps dead names out of the carry state: they
    # compile at all only because the dead tail is dropped before analysis.
    tail_carry_contract = {
        while_continue_skips_tail: 3,
        while_continue_then_break_flags: 3,
        while_nested_if_break_tail: 3,
        while_unconditional_break_dead_tail: 3,
        while_break_dead_unbound_tail: 3,
        while_if_break_dead_branch_tail: 3,
        while_break_dead_slot_tail: 3,
        while_if_else_break_dead_tail: 3,
        while_with_break_dead_tail: 3,
        while_try_break_dead_tail: 3,
        for_break_tail_guard: 4,  # iv + value + control flags
        for_break_tail_carry_merge: 4,  # iv + value + control flags
    }
    for fn, expected_carries in tail_carry_contract.items():
        loop_text = fn.compile().mlir_text()
        actual = _while_iter_arg_count(loop_text)
        assert actual == expected_carries, (
            f"{fn.__name__}: expected {expected_carries} loop-carried slots, got {actual}")

    # Compound-statement transfers (both-branches-break if, with/try bodies
    # that always break) must also have their outer dead tails dropped.
    for fn in (while_if_else_break_dead_tail, while_with_break_dead_tail,
               while_try_break_dead_tail):
        text = fn.compile().mlir_text()
        assert not re.search(r"arith\.addi %[\w#]+, %c10_i32", text), (
            f"{fn.__name__}: dead tail after a guaranteed compound transfer "
            "must be dropped, not emitted")

    # Truncating the dead tail must not weaken the store-less-body diagnostic:
    # a controlled while whose only real statement is the transfer still has
    # nothing to carry the control state in.
    def while_break_first_dead_store(limit: pto.i32):
        value = pto.const(0, dtype=pto.i32)
        while value < limit:
            break
            value = value + pto.const(1, dtype=pto.i32)
        _ = value

    try:
        pto.jit(target="a5")(while_break_first_dead_store).compile()
    except Exception as exc:
        assert "control-state lowering" in str(exc)
    else:
        raise AssertionError("store-less controlled while must be diagnosed")

    def unsupported_break(limit: pto.i32):
        value = pto.const(0, dtype=pto.i32)
        while value < limit:
            break

    try:
        pto.jit(target="a5")(unsupported_break).compile()
    except Exception as exc:
        assert "control-state lowering" in str(exc)
    else:
        raise AssertionError("runtime break must not be silently traced")

    try:
        pto.jit(target="a5")(unsupported_while_subscript).compile()
    except Exception as exc:
        assert "static subscript carries" in str(exc)
    else:
        raise AssertionError("while subscript carry must be diagnosed")

    # A loop-local name used after the loop without an outer initialization
    # must stay on the explicit last-iteration-only diagnostics path rather
    # than being evaluated as an unbound carry at while setup.
    def issue_1256_unbound_live_after(limit: pto.i32):
        value = pto.const(0, dtype=pto.i32)
        while value < limit:
            col = value + pto.const(1, dtype=pto.i32)
            value = col
        _ = col

    try:
        pto.jit(target="a5")(issue_1256_unbound_live_after).compile()
    except Exception as exc:
        assert "last-iteration-only" in str(exc), str(exc)
    else:
        raise AssertionError("unbound loop-local used after while must be diagnosed")

    # Negative fixtures for the checker itself: a tail that merely sits
    # inside an *unrelated* dynamic scf.if (condition is a block argument or
    # an arith result, not the merged active flag) must be rejected — the
    # reviewer's "if value > 0 region without an active guard" case.
    unguarded_fixtures = [
        # condition is a plain block argument
        "func.func @k(%cond: i1) {\n"
        "  %c1_i32 = arith.constant 1 : i32\n"
        "  %0 = scf.if %cond -> (i32) {\n"
        "    %1 = arith.addi %c1_i32, %c1_i32 : i32\n"
        "    scf.yield %1 : i32\n"
        "  }\n"
        "  return\n"
        "}\n",
        # condition is an arith.cmpi result, not a flag-merge scf.if result
        "func.func @k(%a: i32, %b: i32) {\n"
        "  %c1_i32 = arith.constant 1 : i32\n"
        "  %cmp = arith.cmpi slt, %a, %b : i32\n"
        "  %0 = scf.if %cmp -> (i32) {\n"
        "    %1 = arith.addi %c1_i32, %c1_i32 : i32\n"
        "    scf.yield %1 : i32\n"
        "  }\n"
        "  return\n"
        "}\n",
    ]
    for fixture in unguarded_fixtures:
        try:
            _assert_tail_guarded(fixture, r"arith\.addi %[\w#]+, %c1_i32")
        except AssertionError:
            pass
        else:
            raise AssertionError(
                "checker accepted a tail guarded by an unrelated dynamic if "
                "(condition is not the merged active flag)")


if __name__ == "__main__":
    main()
