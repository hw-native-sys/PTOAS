#!/usr/bin/env python3
# coding=utf-8
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

import re

from ptodsl import pto


def expect(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


@pto.jit(target="a5", backend="vpto", mode="explicit")
def shift_count_normalization_probe():
    unsigned_tile = pto.alloc_tile(shape=[1, 64], dtype=pto.ui32)
    signed_tile = pto.alloc_tile(shape=[1, 64], dtype=pto.si32)
    signless_tile = pto.alloc_tile(shape=[1, 64], dtype=pto.i32)
    offset = pto.const(0, dtype=pto.index)
    active_lanes = pto.const(64, dtype=pto.i32)
    mask, _ = pto.make_mask(pto.ui32, active_lanes)
    unsigned_value = pto.vlds(unsigned_tile.as_ptr(), offset)
    signless_count = pto.vlds(signless_tile.as_ptr(), offset)
    signed_value = pto.vlds(signed_tile.as_ptr(), offset)
    unsigned_count = pto.vlds(unsigned_tile.as_ptr(), offset)
    _ = pto.vshr(unsigned_value, signless_count, mask)
    _ = pto.vshl(signed_value, unsigned_count, mask)


def main() -> None:
    mlir_text = shift_count_normalization_probe.compile().mlir_text()
    expect(
        mlir_text.count("pto.vbitcast") == 2,
        "each non-signed shift count must be normalized with pto.vbitcast",
    )
    expect(
        re.search(
            r"pto\.vshr [^:]+ : !pto\.vreg<64xui32>, "
            r"!pto\.vreg<64xsi32>, !pto\.mask<b32> -> !pto\.vreg<64xui32>",
            mlir_text,
        ) is not None,
        "vshr must consume a signed i32 shift-count vector",
    )
    expect(
        re.search(
            r"pto\.vshl [^:]+ : !pto\.vreg<64xsi32>, "
            r"!pto\.vreg<64xsi32>, !pto\.mask<b32> -> !pto\.vreg<64xsi32>",
            mlir_text,
        ) is not None,
        "vshl must consume a signed i32 shift-count vector",
    )
    print("ptodsl_shift_count_normalization: PASS")


if __name__ == "__main__":
    main()
