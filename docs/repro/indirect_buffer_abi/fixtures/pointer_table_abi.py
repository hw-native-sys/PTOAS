# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""Runnable regression for a GM address-table indirection.

Each table element is a 64-bit device address.  The table is therefore an
ordinary ``ptr<i64, gm>``; the loaded address becomes a typed GM pointer only
at the DMA use site.
"""

import argparse

from ptodsl import pto


@pto.jit(target="a5", backend="vpto", mode="explicit", insert_sync=False)
def indirect_address_table(
    table: pto.ptr(pto.i64, "gm"),
    dst: pto.ptr(pto.f32, "gm"),
    entry: pto.i32,
):
    addr = pto.load_scalar(table, entry, bypass_l1=True)
    src = pto.castptr(addr, pto.ptr(pto.f32, "gm"))
    ub = pto.castptr(pto.const(0, dtype=pto.i64), pto.ptr(pto.f32, "ub"))
    pto.mte_gm_ub(src, ub, 0, 64, nburst=(1, 64, 64))
    pto.pipe_barrier(pto.Pipe.ALL)
    pto.mte_ub_gm(ub, dst, 64, nburst=(1, 64, 64))
    pto.pipe_barrier(pto.Pipe.ALL)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--emit-mlir", action="store_true")
    args = parser.parse_args()
    if args.emit_mlir:
        print(indirect_address_table.compile().mlir_text())


if __name__ == "__main__":
    main()
