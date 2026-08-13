# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

import argparse
from ptodsl import pto

ROWS = 10
COLS = 64


@pto.jit(target="a5", backend="vpto", mode="explicit", insert_sync=False)
def fixed_two_buffers(
    src0: pto.ptr(pto.f32, "gm"),
    src1: pto.ptr(pto.f32, "gm"),
    dst: pto.ptr(pto.f32, "gm"),
):
    # Compileable ABI control: every buffer must be a separate formal argument.
    # The reduced issue is the inability to replace src0/src1/... with a runtime
    # table and select one typed GM pointer inside the loop.
    zero = pto.const(0, dtype=pto.i64)
    ub = pto.castptr(zero, pto.ptr(pto.f32, "ub"))
    # One launch performs ten fixed transfers; this matches the pointer-table
    # CCE launch's ten layer iterations.  The ABI remains deliberately fixed-
    # argument because nested GM pointer values are not representable.
    for layer in range(ROWS):
        src_segment = pto.addptr(src0, layer * COLS)
        dst_segment = pto.addptr(dst, layer * COLS)
        pto.mte_gm_ub(src_segment, ub, 0, COLS * 4,
                      nburst=(1, COLS * 4, COLS * 4))
        pto.pipe_barrier(pto.Pipe.ALL)
        pto.mte_ub_gm(ub, dst_segment, COLS * 4,
                      nburst=(1, COLS * 4, COLS * 4))
        pto.pipe_barrier(pto.Pipe.ALL)
    pto.pipe_barrier(pto.Pipe.ALL)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--emit-mlir", action="store_true")
    args = ap.parse_args()
    if args.emit_mlir:
        print(fixed_two_buffers.compile().mlir_text())


if __name__ == "__main__":
    main()
