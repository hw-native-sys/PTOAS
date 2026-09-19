#!/usr/bin/env python3
# coding=utf-8
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Python-surface pto.vmi.vunzip / pto.vmi.vzip validation, with no device dependency.

Throwaway MLIR functions expose the public entry points to their argument
validation: reject cases must raise before any op is emitted, accept cases must
build the op with the expected result type.
"""

from ptoas.mlir.ir import (
    BF16Type,
    Context,
    F32Type,
    FunctionType,
    InsertionPoint,
    IntegerType,
    Location,
    Module,
)
from ptoas.mlir.dialects import func, pto as pto_ir
from ptodsl import _vmi_namespace as ns


def _vreg(lanes, elem):
    return pto_ir.VMIVRegType.get(lanes, elem)


def _expect_error(action, needle, error_type=ValueError):
    try:
        action()
    except error_type as exc:
        assert needle in str(exc), (needle, str(exc))
        return
    raise AssertionError(f"expected {error_type.__name__} containing {needle!r}")


def _unzip(ctx, source_elem, to_dtype=None):
    source = _vreg(128, source_elem)
    module = Module.create()
    with InsertionPoint(module.body):
        fn = func.FuncOp("probe", FunctionType.get([source], []))
    entry = fn.add_entry_block()
    with InsertionPoint(entry):
        low, high = ns.vmi.vunzip(entry.arguments[0], to_dtype)
        func.ReturnOp([])
    return ns._raw(low).type, ns._raw(high).type


def _zip(ctx, half_elem, to_dtype=None, high_elem=None):
    low_type = _vreg(128, half_elem)
    high_type = _vreg(128, half_elem if high_elem is None else high_elem)
    module = Module.create()
    with InsertionPoint(module.body):
        fn = func.FuncOp("probe", FunctionType.get([low_type, high_type], []))
    entry = fn.add_entry_block()
    with InsertionPoint(entry):
        result = ns.vmi.vzip(entry.arguments[0], entry.arguments[1], to_dtype)
        func.ReturnOp([])
    return ns._raw(result).type


def _zip_reject_bad_high(ctx, half_elem, high_elem):
    low_type = _vreg(128, half_elem)
    module = Module.create()
    with InsertionPoint(module.body):
        fn = func.FuncOp("probe", FunctionType.get([low_type, high_elem], []))
    entry = fn.add_entry_block()
    with InsertionPoint(entry):
        ns.vmi.vzip(entry.arguments[0], entry.arguments[1])
        func.ReturnOp([])


def main():
    with Context() as ctx:
        pto_ir.register_dialect(ctx)
        with Location.unknown(ctx):
            ui8 = IntegerType.get_unsigned(8)
            ui16 = IntegerType.get_unsigned(16)
            ui32 = IntegerType.get_unsigned(32)
            i32 = IntegerType.get_signless(32)
            f32 = F32Type.get(ctx)
            bf16 = BF16Type.get(ctx)

            _expect_error(
                lambda: _unzip(ctx, ui8),
                "requires a 16- or 32-bit source element",
            )
            _expect_error(
                lambda: _unzip(ctx, f32, f32),
                "requires the target element storage width to be exactly half",
            )

            _expect_error(
                lambda: _zip_reject_bad_high(ctx, ui16, i32),
                "expects a !pto.vmi.vreg value",
                error_type=TypeError,
            )
            _expect_error(
                lambda: _zip(ctx, ui16, high_elem=f32),
                "requires low and high to share one lane count and element type",
            )
            _expect_error(
                lambda: _zip(ctx, ui32),
                "requires an 8- or 16-bit half element",
            )
            _expect_error(
                lambda: _zip(ctx, ui16, ui16),
                "requires the wide element storage width to be exactly twice",
            )

            low_type, high_type = _unzip(ctx, f32)
            assert str(low_type) == "!pto.vmi.vreg<128xui16>", low_type
            assert str(high_type) == "!pto.vmi.vreg<128xui16>", high_type
            low_type, _ = _unzip(ctx, f32, bf16)
            assert str(low_type) == "!pto.vmi.vreg<128xbf16>", low_type
            low_type, _ = _unzip(ctx, ui16)
            assert str(low_type) == "!pto.vmi.vreg<128xui8>", low_type

            assert str(_zip(ctx, ui16)) == "!pto.vmi.vreg<128xui32>"
            assert str(_zip(ctx, ui8)) == "!pto.vmi.vreg<128xui16>"
            assert str(_zip(ctx, ui16, f32)) == "!pto.vmi.vreg<128xf32>"

    print("vunzip/vzip python validation: 6 reject and 6 accept cases passed")


if __name__ == "__main__":
    main()
