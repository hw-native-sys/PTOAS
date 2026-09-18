#!/usr/bin/env python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

import re

from ptodsl import pto


@pto.jit(name="issue_1521_missing_simd_interfaces", kernel_kind="vector", target="a5", mode="explicit")
def issue_1521_missing_simd_interfaces():
    base = pto.const(0, dtype=pto.ui64)
    source_u8 = pto.vlds(pto.castptr(base, pto.ptr(pto.ui8, "ub")), pto.const(0))
    source_s8 = pto.vlds(pto.castptr(base, pto.ptr(pto.si8, "ub")), pto.const(0))
    acc_u16 = pto.vlds(pto.castptr(base, pto.ptr(pto.ui16, "ub")), pto.const(0))
    merge_i32 = pto.vlds(pto.castptr(base, pto.ptr(pto.i32, "ub")), pto.const(0))

    mask_b8 = pto.pset_b8(pto.MaskPattern.ALL)
    mask_b32 = pto.pset_b32(pto.MaskPattern.ALL)

    _ = pto.dhistv2(acc_u16, source_u8, mask_b8, pto.const(0, dtype=pto.i32))
    _ = pto.vusqz(merge_i32, mask_b32)

    _ = pto.vzunpack(source_u8, pto.VPackPart.LOWER)
    _ = pto.vunpack(source_u8, pto.VPackPart.HIGHER)
    _ = pto.vsunpack(source_s8, pto.VPackPart.LOWER)
    _ = pto.vunpack(source_s8, pto.VPackPart.HIGHER)


@pto.jit(target="a5", mode="explicit")
def vsunpack_unsigned_source_is_rejected():
    base = pto.const(0, dtype=pto.ui64)
    source = pto.vlds(pto.castptr(base, pto.ptr(pto.ui8, "ub")), pto.const(0))
    _ = pto.vsunpack(source, pto.VPackPart.LOWER)


@pto.jit(target="a5", mode="explicit")
def vzunpack_signed_source_is_rejected():
    base = pto.const(0, dtype=pto.ui64)
    source = pto.vlds(pto.castptr(base, pto.ptr(pto.si8, "ub")), pto.const(0))
    _ = pto.vzunpack(source, pto.VPackPart.LOWER)


@pto.jit(target="a5", mode="explicit")
def vunpack_invalid_part_is_rejected():
    base = pto.const(0, dtype=pto.ui64)
    source = pto.vlds(pto.castptr(base, pto.ptr(pto.ui8, "ub")), pto.const(0))
    _ = pto.vunpack(source, "MIDDLE")


def _expect_compile_error(kernel, needle: str):
    try:
        kernel.compile()
    except (TypeError, ValueError) as exc:
        if needle not in str(exc):
            raise AssertionError(f"expected {needle!r} in {exc!r}") from exc
        return
    raise AssertionError(f"{kernel.__name__} should reject its invalid arguments")


def main():
    for name in ("dhistv2", "vusqz", "vsunpack", "vzunpack", "vunpack"):
        if not hasattr(pto, name):
            raise AssertionError(f"pto.{name} should be publicly exported")

    text = issue_1521_missing_simd_interfaces.compile().mlir_text()
    expected_patterns = (
        r"pto\.dhistv2 .* : !pto\.vreg<128xui16>, !pto\.vreg<256xui8>, !pto\.mask<b8>, i32 -> !pto\.vreg<128xui16>",
        r"pto\.vusqz .* : !pto\.vreg<64xi32>, !pto\.mask<b32> -> !pto\.vreg<64xi32>",
        r"pto\.vzunpack .* : !pto\.vreg<256xui8> -> !pto\.vreg<128xui16>",
        r"pto\.vsunpack .* : !pto\.vreg<256xsi8> -> !pto\.vreg<128xsi16>",
    )
    for pattern in expected_patterns:
        if re.search(pattern, text) is None:
            raise AssertionError(f"expected {pattern!r} in:\n{text}")

    if text.count("pto.vzunpack ") != 2:
        raise AssertionError("unsigned vunpack should dispatch to exactly one pto.vzunpack")
    if text.count("pto.vsunpack ") != 2:
        raise AssertionError("signed vunpack should dispatch to exactly one pto.vsunpack")

    _expect_compile_error(vsunpack_unsigned_source_is_rejected, "signed or signless integer source")
    _expect_compile_error(vzunpack_signed_source_is_rejected, "unsigned integer source")
    _expect_compile_error(vunpack_invalid_part_is_rejected, "expected one of HIGHER, LOWER")

    print("issue_1521_missing_simd_interfaces: PASS")


if __name__ == "__main__":
    main()
