# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Python DSL form of ``desired_vmi.pto`` — scalar GM store with dcache bypass.

Canonical IR: ``desired_vmi.pto`` (``{group = 1, dcache_bypass}``).
PTODSL ``vstore`` does not yet expose ``dcache_bypass``; IR remains authoritative.
"""

from ptodsl import pto


@pto.jit(target="a5", backend="vpto", mode="explicit")
def scale_factor_gm_bypass(sf_gm: pto.ptr(pto.f32, "gm")):
    ub = pto.alloc_tile(shape=[1, 8], dtype=pto.f32)
    off = pto.const(0, dtype=pto.index)
    scale = pto.vmi.vload(ub.as_ptr(), off, size=1)
    # Desired IR also sets dcache_bypass on this store.
    pto.vmi.vstore(
        scale,
        sf_gm,
        off,
        group=1,
        stride=pto.const(1, dtype=pto.index),
    )


if __name__ == "__main__":
    print(scale_factor_gm_bypass.compile().mlir_text())
