#!/usr/bin/env python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

from ptodsl import pto


@pto.jit(name="issue_1126_bool_merge", target="a5")
def issue_1126_bool_merge(flag: pto.ptr(pto.i8, "gm")) -> pto.i32:
    cond = pto.const(0, dtype=pto.i1)

    if pto.const(1, dtype=pto.i32):
        cond = pto.load(flag, 0)

    if cond:
        return pto.const(42, dtype=pto.i32)
    return pto.const(0, dtype=pto.i32)


def _explicit_integer_bool_merge(value):
    cond = pto.const(0, dtype=pto.i1)
    with pto.if_(pto.const(1, dtype=pto.i1)) as br:
        with br.then_:
            br.assign(value=value)
        with br.else_:
            br.assign(value=cond)
    return br.value


@pto.jit(name="issue_1126_integer_widths", target="a5")
def issue_1126_integer_widths(
    i8_value: pto.i8,
    i16_value: pto.i16,
    i32_value: pto.i32,
    i64_value: pto.i64,
    ui8_value: pto.ui8,
    ui16_value: pto.ui16,
    ui32_value: pto.ui32,
    ui64_value: pto.ui64,
) -> pto.i32:
    flag_i8 = _explicit_integer_bool_merge(i8_value)
    flag_i16 = _explicit_integer_bool_merge(i16_value)
    flag_i32 = _explicit_integer_bool_merge(i32_value)
    flag_i64 = _explicit_integer_bool_merge(i64_value)
    flag_ui8 = _explicit_integer_bool_merge(ui8_value)
    flag_ui16 = _explicit_integer_bool_merge(ui16_value)
    flag_ui32 = _explicit_integer_bool_merge(ui32_value)
    flag_ui64 = _explicit_integer_bool_merge(ui64_value)
    result = pto.const(0, dtype=pto.i32)
    result = pto.select(flag_i8, pto.const(1, dtype=pto.i32), result)
    result = pto.select(flag_i16, pto.const(1, dtype=pto.i32), result)
    result = pto.select(flag_i32, pto.const(1, dtype=pto.i32), result)
    result = pto.select(flag_i64, pto.const(1, dtype=pto.i32), result)
    result = pto.select(flag_ui8, pto.const(1, dtype=pto.i32), result)
    result = pto.select(flag_ui16, pto.const(1, dtype=pto.i32), result)
    result = pto.select(flag_ui32, pto.const(1, dtype=pto.i32), result)
    result = pto.select(flag_ui64, pto.const(1, dtype=pto.i32), result)
    return result


@pto.jit(name="issue_1126_incompatible_merge", target="a5")
def issue_1126_incompatible_merge(value: pto.i32) -> pto.i32:
    with pto.if_(pto.const(1, dtype=pto.i1)) as br:
        with br.then_:
            br.assign(value=value)
        with br.else_:
            br.assign(value=pto.const(0.0, dtype=pto.f32))
    return pto.const(0, dtype=pto.i32)


def main():
    mlir = issue_1126_bool_merge.compile().mlir_text()
    assert "pto.cmpi ne" in mlir
    assert "pto.cmpi ne" in mlir and "i8" in mlir

    widths_mlir = issue_1126_integer_widths.compile().mlir_text()
    assert widths_mlir.count("pto.cmpi ne") == 8
    for width in ("i8", "i16", "i32", "i64"):
        assert width in widths_mlir

    try:
        issue_1126_incompatible_merge.compile()
    except RuntimeError as exc:
        assert "type mismatch for 'value'" in str(exc)
    else:
        raise AssertionError("incompatible integer/float branch merge should still fail")


if __name__ == "__main__":
    main()
