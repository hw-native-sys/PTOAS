# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Python DSL form of ``current_slow_vmi.pto`` — L=256 ui8→ui16 ``vcvt``.

Canonical IR: ``current_slow_vmi.pto``.
"""

from ptodsl import pto


@pto.jit(target="a5", backend="vpto", mode="explicit")
def ui8_to_ui16_l256():
    ub = pto.alloc_tile(shape=[1, 256], dtype=pto.ui8)
    off = pto.const(0, dtype=pto.index)
    inp = pto.vmi.vload(ub.as_ptr(), off, size=256)
    wide = pto.vmi.vcvt(inp, pto.ui16)
    _ = wide


if __name__ == "__main__":
    print(ui8_to_ui16_l256.compile().mlir_text())
