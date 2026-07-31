# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Python DSL form of ``desired_compact_vmi.pto``.

Desired: in-register ``vbrc(..., group=G)`` expand without pad-32 + ``brc`` reload.
Canonical IR: ``desired_compact_vmi.pto`` (expected FAIL on current tip).
"""

from ptodsl import pto

_G = 4
_STRIP = 128


@pto.jit(target="a5", backend="vpto", mode="explicit")
def desired_compact_vbrc():
    data_ub = pto.alloc_tile(shape=[1, _STRIP], dtype=pto.f32)
    out_ub = pto.alloc_tile(shape=[1, _STRIP], dtype=pto.f32)
    off = pto.const(0, dtype=pto.index)
    mask = pto.vmi.create_mask(_STRIP, size=_STRIP)

    # G-lane recip kept in-register (no pad store).
    recip = pto.vmi.vbrc(pto.f32(0.5), size=_G)
    expanded = pto.vmi.vbrc(recip, size=_STRIP, group=_G)
    data = pto.vmi.vload(data_ub.as_ptr(), off, size=_STRIP)
    out = pto.vmi.vmul(data, expanded, mask)
    pto.vmi.vstore(out, out_ub.as_ptr(), off, mask)


if __name__ == "__main__":
    try:
        print(desired_compact_vbrc.compile().mlir_text())
    except Exception as exc:  # noqa: BLE001 — document the gap
        print(f"desired compact vbrc emit/lower may fail: {exc}")
