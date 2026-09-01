#!/usr/bin/env python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Regression test for signed runtime values used in arithmetic expressions."""

from ptodsl import pto, scalar


@pto.jit(target="a5")
def signed_runtime_arithmetic_probe(
    out_ptr: pto.ptr(pto.si32, "gm"),
    num_tokens: pto.si32,
):
    # This is the shape emitted by TileLang topk_gate: a signed runtime
    # parameter is combined with Python literals and then used as an index.
    waves = (num_tokens + 3) // 4
    index = waves - 1
    scalar.store(num_tokens + 1, out_ptr, index)


def test_signed_runtime_arithmetic_roundtrip():
    text = signed_runtime_arithmetic_probe.compile().mlir_text()
    # Compilation itself performs the MLIR parse/verify round trip.  Keep the
    # ABI signed while requiring signless arith operations in the generated IR.
    assert "num_tokens" not in text
    assert "func.func @signed_runtime_arithmetic_probe" in text
    assert "arith.addi" in text
    assert "arith.floordivsi" in text
    assert "si32" in text
    assert "i32" in text
    # The source of every signless arithmetic constant must be i32 itself;
    # the malformed form was `... %c3_i32 : si32 to i32`.
    import re

    assert not re.search(r"%c\w+ = builtin\.unrealized_conversion_cast .*: si32 to i32", text)


@pto.jit(target="a5")
def unsigned_runtime_arithmetic_probe(
    out_ptr: pto.ptr(pto.ui32, "gm"),
    value: pto.ui32,
):
    quotient = (value + "0x3") // 4
    remainder = value % 4
    is_small = value < 7
    bounded_max = scalar.max(value, 7)
    bounded_min = scalar.min(value, 7)
    scalar.store(quotient + remainder + bounded_max + bounded_min, out_ptr, 0)
    scalar.store(is_small, out_ptr, 1)


def test_unsigned_runtime_arithmetic_preserves_unsigned_ops():
    text = unsigned_runtime_arithmetic_probe.compile().mlir_text()
    assert "arith.divui" in text
    assert "arith.remui" in text
    assert "arith.cmpi ult" in text
    assert "arith.maxui" in text
    assert "arith.minui" in text
