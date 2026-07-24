# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Python DSL form of ``desired_vmi.pto`` — pack UE8M0 into ui16 then store.

Canonical IR: ``desired_vmi.pto`` (``pto.vmi.vpack`` — unknown / not bound today).
"""

from ptodsl import pto


@pto.jit(target="a5", backend="vpto", mode="explicit")
def packed_ue8m0_sf_store():
    src = pto.alloc_tile(shape=[1, 64], dtype=pto.ui8)
    dst = pto.alloc_tile(shape=[1, 32], dtype=pto.ui16)
    off = pto.const(0, dtype=pto.index)
    sf_u8 = pto.vmi.vload(src.as_ptr(), off, size=64)
    # Desired surface: pack_factor=2 ui8 → ui16 (not on PTODSL yet).
    sf_u16 = pto.vmi.vpack(sf_u8)
    pto.vmi.vstore(sf_u16, dst.as_ptr(), off)


if __name__ == "__main__":
    try:
        print(packed_ue8m0_sf_store.compile().mlir_text())
    except Exception as exc:  # noqa: BLE001 — document the gap
        print(f"desired emit failed (expected until pto.vmi.vpack lands): {exc}")
