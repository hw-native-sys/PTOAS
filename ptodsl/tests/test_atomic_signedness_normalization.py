#!/usr/bin/env python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Regression tests for signed/unsigned integer atomic normalization.

``pto.atomic_{add,sub,and,or,xor}`` are sign-agnostic at the ISA level, but
``pto.atomic_*`` only accepts signless integer carriers.  The PTODSL frontend
therefore normalizes an authored ``si32/si64/ui32/ui64`` pointer and value to
the same-width signless type before constructing the op.  Without this the
verifier rejects the access with

    'pto.atomic_add' op operand #1 must be i32, i64, ... but got 'si64'

``atomic_min/max`` keep their authored signedness because comparison semantics
depend on it, and floating-point atomics must stay untouched.
"""

import logging
import re

from ptodsl import pto

logging.basicConfig(level=logging.INFO)

_ATOMIC_LINE = re.compile(r"pto\.atomic_add[^\n]*")
_ATOMIC_AND_LINE = re.compile(r"pto\.atomic_and[^\n]*")
_ATOMIC_MIN_LINE = re.compile(r"pto\.atomic_min[^\n]*")
_ATOMIC_MAX_LINE = re.compile(r"pto\.atomic_max[^\n]*")
_ATOMIC_CAS_LINE = re.compile(r"pto\.atomic_cas[^\n]*")


@pto.jit(name="atomic_si64_add_probe", target="a5")
def atomic_si64_add_probe(counter: pto.ptr(pto.si64, "gm"), value: pto.si64):
    with pto.simt():
        pto.atomic_add(counter, value)


@pto.jit(name="atomic_ui64_add_probe", target="a5")
def atomic_ui64_add_probe(counter: pto.ptr(pto.ui64, "gm"), value: pto.ui64):
    with pto.simt():
        pto.atomic_add(counter, value)


@pto.jit(name="atomic_si32_add_probe", target="a5")
def atomic_si32_add_probe(counter: pto.ptr(pto.si32, "gm"), value: pto.si32):
    with pto.simt():
        pto.atomic_add(counter, value)


@pto.jit(name="atomic_ui32_add_probe", target="a5")
def atomic_ui32_add_probe(counter: pto.ptr(pto.ui32, "gm"), value: pto.ui32):
    with pto.simt():
        pto.atomic_add(counter, value)


@pto.jit(name="atomic_si64_and_probe", target="a5")
def atomic_si64_and_probe(counter: pto.ptr(pto.si64, "gm"), value: pto.si64):
    with pto.simt():
        pto.atomic_and(counter, value)


@pto.jit(name="atomic_f32_add_probe", target="a5")
def atomic_f32_add_probe(counter: pto.ptr(pto.f32, "gm"), value: pto.f32):
    with pto.simt():
        pto.atomic_add(counter, value)


@pto.jit(name="atomic_ui64_old_shift_probe", target="a5")
def atomic_ui64_old_shift_probe(
    counter: pto.ptr(pto.ui64, "gm"),
    out: pto.ptr(pto.ui64, "gm"),
    value: pto.ui64,
):
    with pto.simt():
        old = pto.atomic_add(counter, value)
        pto.store(old >> pto.const(1, dtype=pto.ui64), out, 0)


@pto.jit(name="atomic_si64_cas_probe", target="a5")
def atomic_si64_cas_probe(
    counter: pto.ptr(pto.si64, "gm"),
    out: pto.ptr(pto.si64, "gm"),
    compare: pto.si64,
    value: pto.si64,
):
    with pto.simt():
        old = pto.atomic_cas(counter, compare, value)
        pto.store(old, out, 0)


@pto.jit(name="atomic_ui64_cas_probe", target="a5")
def atomic_ui64_cas_probe(
    counter: pto.ptr(pto.ui64, "gm"),
    out: pto.ptr(pto.ui64, "gm"),
    compare: pto.ui64,
    value: pto.ui64,
):
    with pto.simt():
        old = pto.atomic_cas(counter, compare, value)
        pto.store(old, out, 0)


@pto.jit(name="atomic_si64_min_probe", target="a5")
def atomic_si64_min_probe(counter: pto.ptr(pto.si64, "gm"), value: pto.si64):
    with pto.simt():
        pto.atomic_min(counter, value, signedness="signed")


@pto.jit(name="atomic_ui64_max_probe", target="a5")
def atomic_ui64_max_probe(counter: pto.ptr(pto.ui64, "gm"), value: pto.ui64):
    with pto.simt():
        pto.atomic_max(counter, value, signedness="unsigned")


def _first(pattern, mlir):
    match = pattern.search(mlir)
    assert match is not None, f"expected an atomic op in:\n{mlir}"
    return match.group(0)


def main():
    si64 = _first(_ATOMIC_LINE, atomic_si64_add_probe.compile().mlir_text())
    assert "!pto.ptr<i64, gm>" in si64, si64
    assert "i64 -> i64" in si64, si64
    assert "si64" not in si64, si64

    ui64 = _first(_ATOMIC_LINE, atomic_ui64_add_probe.compile().mlir_text())
    assert "!pto.ptr<i64, gm>" in ui64, ui64
    assert "i64 -> i64" in ui64, ui64
    assert "ui64" not in ui64, ui64

    si32 = _first(_ATOMIC_LINE, atomic_si32_add_probe.compile().mlir_text())
    assert "!pto.ptr<i32, gm>" in si32, si32
    assert "i32 -> i32" in si32, si32
    assert "si32" not in si32, si32

    ui32 = _first(_ATOMIC_LINE, atomic_ui32_add_probe.compile().mlir_text())
    assert "!pto.ptr<i32, gm>" in ui32, ui32
    assert "i32 -> i32" in ui32, ui32
    assert "ui32" not in ui32, ui32

    si64_and = _first(_ATOMIC_AND_LINE, atomic_si64_and_probe.compile().mlir_text())
    assert "!pto.ptr<i64, gm>" in si64_and, si64_and
    assert "si64" not in si64_and, si64_and

    f32 = _first(_ATOMIC_LINE, atomic_f32_add_probe.compile().mlir_text())
    assert "!pto.ptr<f32, gm>" in f32, f32
    assert "f32 -> f32" in f32, f32

    shift_mlir = atomic_ui64_old_shift_probe.compile().mlir_text()
    assert "to ui64" in shift_mlir, shift_mlir
    assert "pto.shr" in shift_mlir, shift_mlir
    shift_line = next(line for line in shift_mlir.splitlines() if "pto.shr" in line)
    assert "unsigned" in shift_line, shift_line

    min_mlir = atomic_si64_min_probe.compile().mlir_text()
    min_line = _first(_ATOMIC_MIN_LINE, min_mlir)
    assert "!pto.ptr<i64, gm>" in min_line, min_line
    assert "si64" not in min_line, min_line
    assert " signed :" in min_line, min_line

    max_mlir = atomic_ui64_max_probe.compile().mlir_text()
    max_line = _first(_ATOMIC_MAX_LINE, max_mlir)
    assert "!pto.ptr<i64, gm>" in max_line, max_line
    assert "ui64" not in max_line, max_line
    assert " unsigned :" in max_line, max_line

    for probe, authored in ((atomic_si64_cas_probe, "si64"), (atomic_ui64_cas_probe, "ui64")):
        cas_mlir = probe.compile().mlir_text()
        cas_line = _first(_ATOMIC_CAS_LINE, cas_mlir)
        assert "!pto.ptr<i64, gm>, i64 -> i64" in cas_line, cas_line
        assert authored not in cas_line, cas_line
        assert f"to {authored}" in cas_mlir, cas_mlir
        assert "!pto.ptr<i64, gm> -> !pto.ptr<i64, gm>" not in cas_mlir, cas_mlir

    logging.info("ptodsl_atomic_signedness_normalization: PASS")


if __name__ == "__main__":
    main()
