#!/usr/bin/env python3
# coding=utf-8
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

from ptodsl import pto


def _compile_vdup(name, dtype, mask_bits, value):
    @pto.jit(name=name, kernel_kind="vector", target="a5", mode="explicit")
    def kernel():
        mask = getattr(pto, f"pset_b{mask_bits}")(pto.MaskPattern.ALL)
        pto.vdup(pto.const(value, dtype=dtype), mask)

    return kernel.compile().mlir_text()


def _compile_vdup_consumer():
    @pto.jit(name="issue_1102_vdup_vor", kernel_kind="vector", target="a5", mode="explicit")
    def kernel():
        base = pto.const(0, dtype=pto.ui64)
        source = pto.castptr(base, pto.ptr(pto.ui16, "ub"))
        mask = pto.pset_b16(pto.MaskPattern.ALL)
        loaded = pto.vlds(source, pto.const(0))
        duplicated = pto.vdup(pto.ui16(1), mask)
        pto.vor(loaded, duplicated, mask)

    return kernel.compile().mlir_text()


def main():
    cases = (
        ("i8", pto.i8, 8, 1, "i8, !pto.mask<b8> -> !pto.vreg<256xi8>"),
        ("si8", pto.si8, 8, 1, "si8, !pto.mask<b8> -> !pto.vreg<256xsi8>"),
        ("ui8", pto.ui8, 8, 1, "ui8, !pto.mask<b8> -> !pto.vreg<256xui8>"),
        ("i16", pto.i16, 16, 1, "i16, !pto.mask<b16> -> !pto.vreg<128xi16>"),
        ("si16", pto.si16, 16, 1, "si16, !pto.mask<b16> -> !pto.vreg<128xsi16>"),
        ("ui16", pto.ui16, 16, 1, "ui16, !pto.mask<b16> -> !pto.vreg<128xui16>"),
        ("i32", pto.i32, 32, 1, "i32, !pto.mask<b32> -> !pto.vreg<64xi32>"),
        ("si32", pto.si32, 32, 1, "si32, !pto.mask<b32> -> !pto.vreg<64xsi32>"),
        ("ui32", pto.ui32, 32, 1, "ui32, !pto.mask<b32> -> !pto.vreg<64xui32>"),
    )
    for name, dtype, mask_bits, value, expected_type in cases:
        text = _compile_vdup(f"issue_1102_vdup_{name}", dtype, mask_bits, value)
        if expected_type not in text:
            raise AssertionError(
                f"{name} vdup should preserve the scalar integer signedness in its result vector; "
                f"expected {expected_type!r} in:\n{text}"
            )
    _compile_vdup_consumer()
    print("issue_1102_vdup: PASS")


if __name__ == "__main__":
    main()
