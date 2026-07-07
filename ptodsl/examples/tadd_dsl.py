# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""
TADD kernel – DSL-style builder.

Generates the same IR as expand_tileop_to_vpto_result.pto using the
``@pto.jit`` decorator and the ``pto.*`` namespace.

The Python code maps 1-to-1 to the MLIR IR lines:

    func.func @TADD() {                      # @pto.jit(name="TADD", …)
      %c0_i64    = arith.constant 0 : i64    # pto.const(0, dtype=pto.int64)
      %c16       = arith.constant 16 : index # pto.const(16, dtype=pto.index)
      …
      pto.tileop {                              # with pto.tileop():
        %0 = pto.castptr %c4096_i64 …        #   pto.castptr(c4096_i64, …)
        scf.for %arg0 = %c0 to %c16 … {      #   for i in range(c0, c16, c1):
          %mask, _ = pto.plt_b32 …           #     pto.plt_b32(c64_i32)
          …
        }
      }
    }
"""

from ptodsl import pto, scalar

s = scalar  # arith shorthand alias


@pto.jit(name="TADD", kernel_kind="vector", target="a5")
def TADD():
    c0_i64    = pto.const(0,    dtype=pto.int64)
    c16       = pto.const(16,   dtype=pto.index)
    c4096_i64 = pto.const(4096, dtype=pto.int64)
    c0        = pto.const(0)
    c1        = pto.const(1)
    c64_i32   = pto.const(64,   dtype=pto.int32)
    c64       = pto.const(64)

    with pto.tileop():
        ptr_f32_ub   = pto.ptr(pto.float32, "ub")
        vf32         = pto.vreg_type(64, pto.float32)
        ptr_src      = pto.castptr(c4096_i64, ptr_f32_ub)
        ptr_dst      = pto.castptr(c0_i64,    ptr_f32_ub)

        for tile_idx in range(c0, c16, c1):
            mask, _      = pto.plt_b32(c64_i32)
            tile_off     = s.muli(tile_idx, c64)
            va           = pto.vlds(pto.addptr(ptr_src, tile_off), c0, vf32)
            ptr_dst_tile = pto.addptr(ptr_dst, tile_off)
            vb           = pto.vlds(ptr_dst_tile, c0, vf32)
            vc           = pto.vadd(va, vb, mask)
            pto.vsts(vc, ptr_dst_tile, c0, mask)


def build():
    return TADD.mlir_module()


if __name__ == "__main__":
    print(TADD)
