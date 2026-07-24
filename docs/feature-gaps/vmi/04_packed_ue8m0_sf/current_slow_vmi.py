# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Python DSL form of ``current_slow_vmi.pto`` — unpacked ui8 SF store.

Canonical IR: ``current_slow_vmi.pto``.
"""

from ptodsl import pto


@pto.jit(target="a5", backend="vpto", mode="explicit")
def unpacked_ue8m0_sf_store():
    dst = pto.alloc_tile(shape=[1, 64], dtype=pto.ui8)
    off = pto.const(0, dtype=pto.index)
    sf_u8 = pto.vmi.vload(dst.as_ptr(), off, size=64)
    pto.vmi.vstore(sf_u8, dst.as_ptr(), off)


if __name__ == "__main__":
    print(unpacked_ue8m0_sf_store.compile().mlir_text())
