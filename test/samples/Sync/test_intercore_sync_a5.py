#!/usr/bin/env python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

from mlir.ir import (
    Context,
    F32Type,
    IndexType,
    InsertionPoint,
    Location,
    Module,
)
from mlir.dialects import arith, func, pto


def build():
    with Context() as ctx:
        pto.register_dialect(ctx, load=True)
        with Location.unknown(ctx):
            module = Module.create()
            f32 = F32Type.get(ctx)
            idx = IndexType.get(ctx)
            ptr_f32 = pto.PtrType.get(f32, ctx)
            fn_ty = func.FunctionType.get([ptr_f32], [])

            with InsertionPoint(module.body):
                fn = func.FuncOp("test_intercore_sync_a5", fn_ty)
                entry = fn.add_entry_block()

            with InsertionPoint(entry):
                c0 = arith.ConstantOp(idx, 0).result
                two = arith.ConstantOp(f32, 2.0).result
                evt = 0
                out = entry.arguments[0]
                # PTO-ISA A5 mix-kernel style:
                # cube producer -> vec consumer
                # set: PIPE_FIX, wait: PIPE_MTE3.
                pipe_fix = pto.PipeAttr.get(pto.PIPE.PIPE_FIX, ctx)
                pipe_mte3 = pto.PipeAttr.get(pto.PIPE.PIPE_MTE3, ctx)

                sec_cube = pto.SectionCubeOp()
                with InsertionPoint(sec_cube.body.blocks.append()):
                    pto.sync_set(pipe_fix, evt)

                sec_vec = pto.SectionVectorOp()
                with InsertionPoint(sec_vec.body.blocks.append()):
                    pto.sync_wait(pipe_mte3, evt)
                    pto.store_scalar(out, c0, two)

                func.ReturnOp([])

            module.operation.verify()
            return module


if __name__ == "__main__":
    print(build())
