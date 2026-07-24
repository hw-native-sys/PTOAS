# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Python DSL form of ``desired_vmi.pto`` — compact inverse ``vbrc`` (no pad round-trip).

Canonical IR: ``desired_vmi.pto``.
"""

from ptodsl import pto


@pto.jit(target="a5", backend="vpto", mode="explicit")
def compact_inv_vbrc():
    inv_ub = pto.alloc_tile(shape=[1, 8], dtype=pto.f32)
    off = pto.const(0, dtype=pto.index)
    inv_compact = pto.vmi.vload(inv_ub.as_ptr(), off, size=8)
    inv = pto.vmi.vbrc(inv_compact, size=64, group=8)
    _ = inv


if __name__ == "__main__":
    print(compact_inv_vbrc.compile().mlir_text())
