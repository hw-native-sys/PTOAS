# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

from mlir.ir import Context, Location, Module, InsertionPoint, F32Type, IntegerType, MemRefType
from mlir.dialects import arith, func, pto, scf


def build():
    with Context() as ctx:
        pto.register_dialect(ctx, load=True)

        with Location.unknown(ctx):
            module = Module.create()

            f32 = F32Type.get(ctx)
            i8 = IntegerType.get_signless(8, ctx)

            gm = pto.AddressSpaceAttr.get(pto.AddressSpace.GM, ctx)
            vec = pto.AddressSpaceAttr.get(pto.AddressSpace.VEC, ctx)

            data_ty = MemRefType.get([128], f32, memory_space=gm)
            workspace_ty = MemRefType.get([1024], i8, memory_space=gm)
            scratch_ty = pto.TileBufType.get([1, 256], i8, vec, [1, 256], None, ctx)

            i32 = IntegerType.get_signless(32, ctx)

            fn_ty = func.FunctionType.get([data_ty, data_ty, workspace_ty, i32], [])
            with InsertionPoint(module.body):
                fn = func.FuncOp("async_comm_kernel", fn_ty)
                entry = fn.add_entry_block()

            with InsertionPoint(entry):
                dst, src, workspace, nranks = entry.arguments
                c1_i32 = arith.ConstantOp(i32, 1).result
                single_rank = arith.CmpIOp(
                    arith.CmpIPredicate.sle, nranks, c1_i32
                ).result
                guarded = scf.IfOp(single_rank, [], hasElse=True)

                with InsertionPoint(guarded.then_block):
                    scf.YieldOp([])

                with InsertionPoint(guarded.else_block):
                    scratch = pto.AllocTileOp(scratch_ty).result
                    session = pto.BuildAsyncSessionOp(scratch, workspace).result
                    put_event = pto.TPutAsyncOp(dst, src, session).result
                    get_event = pto.TGetAsyncOp(src, dst, session).result
                    pto.WaitAsyncEventOp(put_event, session)
                    pto.TestAsyncEventOp(get_event, session)
                    scf.YieldOp([])

                func.ReturnOp([])

            module.operation.verify()
            return module


if __name__ == "__main__":
    print(build())
