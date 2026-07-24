# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Python DSL form of ``current_slow_vmi.pto`` — bulk UB→GM MTE for a tiny SF.

Canonical IR: ``current_slow_vmi.pto``. Emit: ``scale_factor_bulk_mte.compile().mlir_text()``.
"""

from ptodsl import pto


@pto.jit(target="a5", backend="vpto", mode="explicit")
def scale_factor_bulk_mte(sf_gm: pto.ptr(pto.f32, "gm")):
    # SF staged in UB; 4B useful / 32B burst — MTE setup dominates.
    sf_ub = pto.alloc_tile(shape=[1, 8], dtype=pto.f32)
    pto.mte_ub_gm(sf_ub.as_ptr(), sf_gm, 32, nburst=(1, 32, 32))


if __name__ == "__main__":
    print(scale_factor_bulk_mte.compile().mlir_text())
