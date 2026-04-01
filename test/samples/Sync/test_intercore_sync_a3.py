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
    IntegerType,
    Location,
    MemRefType,
    Module,
)
from mlir.dialects import arith, func, pto, scf


def build():
    with Context() as ctx:
        pto.register_dialect(ctx, load=True)
        with Location.unknown(ctx):
            module = Module.create()

            f32 = F32Type.get(ctx)
            idx = IndexType.get(ctx)
            i64 = IntegerType.get_signless(64, ctx)
            i32 = IntegerType.get_signless(32, ctx)
            # Reserve a practical FFTS workspace size instead of a 1-element stub.
            ffts_ty = MemRefType.get([256], i64)
            ptr_f32 = pto.PtrType.get(f32, ctx)
            fn_ty = func.FunctionType.get([ffts_ty, ptr_f32, i32], [])

            with InsertionPoint(module.body):
                fn = func.FuncOp("test_intercore_sync_a3", fn_ty)
                entry = fn.add_entry_block()

            with InsertionPoint(entry):
                c0 = arith.ConstantOp(idx, 0).result
                c0_i32 = arith.ConstantOp(i32, 0).result
                one = arith.ConstantOp(f32, 1.0).result
                pipe_mte3 = pto.PipeAttr.get(pto.PIPE.PIPE_MTE3, ctx)
                pipe_v = pto.PipeAttr.get(pto.PIPE.PIPE_V, ctx)
                pto.set_ffts(entry.arguments[0])
                pto.sync_set(pipe_mte3, 3)
                # Keep wait lowering in IR while avoiding deadlock in single-core
                # functional validation (generated main.cpp initializes scalar to 1).
                should_wait = arith.CmpIOp(
                    arith.CmpIPredicate.eq, entry.arguments[2], c0_i32
                ).result
                if_op = scf.IfOp(should_wait, [], hasElse=False)
                with InsertionPoint(if_op.then_block):
                    pto.sync_wait(pipe_v, 3)
                    scf.YieldOp([])
                pto.store_scalar(entry.arguments[1], c0, one)
                func.ReturnOp([])

            module.operation.verify()
            return module


if __name__ == "__main__":
    print(build())
