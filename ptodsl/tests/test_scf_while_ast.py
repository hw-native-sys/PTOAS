#!/usr/bin/env python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

from ptodsl import pto


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


def unsupported_while_subscript(limit: pto.i32):
    values = [0]
    value = pto.const(0, dtype=pto.i32)
    while value < limit:
        values[0] = value
        value = value + pto.const(1, dtype=pto.i32)


def main():
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


if __name__ == "__main__":
    main()
