# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

from mlir.ir import Context, Location, Module, InsertionPoint, F32Type, IntegerType, IntegerAttr, IndexType, Operation
from mlir.dialects import arith, func, pto, scf


def _build_async_session(scratch, workspace, i32, ctx, sync_id=0):
    if hasattr(pto, "BuildAsyncSessionOp"):
        return pto.BuildAsyncSessionOp(scratch, workspace, sync_id=sync_id).result
    op = Operation.create(
        "pto.build_async_session",
        operands=[scratch, workspace],
        attributes={"sync_id": IntegerAttr.get(i32, sync_id)},
        results=[pto.AsyncSessionType.get(ctx)],
    )
    return op.result


def _async_transfer(op_name, dst, src, session, ctx):
    op_class_name = {
        "pto.tput_async": "TPutAsyncOp",
        "pto.tget_async": "TGetAsyncOp",
    }[op_name]
    if hasattr(pto, op_class_name):
        return getattr(pto, op_class_name)(dst, src, session).result
    op = Operation.create(
        op_name,
        operands=[dst, src, session],
        results=[pto.AsyncEventType.get(ctx)],
    )
    return op.result


def _wait_async_event(event, session):
    if hasattr(pto, "WaitAsyncEventOp"):
        return pto.WaitAsyncEventOp(event, session)
    Operation.create("pto.wait_async_event", operands=[event, session], results=[])

def build():
    with Context() as ctx:
        pto.register_dialect(ctx, load=True)

        with Location.unknown(ctx):
            module = Module.create()

            f32 = F32Type.get(ctx)
            i8 = IntegerType.get_signless(8, ctx)

            vec = pto.AddressSpaceAttr.get(pto.AddressSpace.VEC, ctx)
            ptr_f32 = pto.PtrType.get(f32, ctx)
            ptr_i8 = pto.PtrType.get(i8, ctx)
            tv1_f32 = pto.TensorViewType.get([128], f32, ctx)
            pv1_f32 = pto.PartitionTensorViewType.get([128], f32, ctx)
            scratch_ty = pto.TileBufType.get([1, 256], i8, vec, [1, 256], None, ctx)

            i32 = IntegerType.get_signless(32, ctx)
            idx = IndexType.get(ctx)

            fn_ty = func.FunctionType.get([ptr_f32, ptr_f32, ptr_i8, i32], [])
            with InsertionPoint(module.body):
                fn = func.FuncOp("async_comm_kernel", fn_ty)
                entry = fn.add_entry_block()

            with InsertionPoint(entry):
                dst_ptr, src_ptr, workspace_ptr, nranks = entry.arguments
                c0 = arith.ConstantOp(idx, 0).result
                c128 = arith.ConstantOp(idx, 128).result
                c1 = arith.ConstantOp(idx, 1).result
                dst_tv = pto.MakeTensorViewOp(tv1_f32, dst_ptr, [c128], [c1]).result
                src_tv = pto.MakeTensorViewOp(tv1_f32, src_ptr, [c128], [c1]).result
                dst = pto.PartitionViewOp(pv1_f32, dst_tv, offsets=[c0], sizes=[c128]).result
                src = pto.PartitionViewOp(pv1_f32, src_tv, offsets=[c0], sizes=[c128]).result
                c1_i32 = arith.ConstantOp(i32, 1).result
                single_rank = arith.CmpIOp(
                    arith.CmpIPredicate.sle, nranks, c1_i32
                ).result
                guarded = scf.IfOp(single_rank, [], hasElse=True)

                with InsertionPoint(guarded.then_block):
                    scf.YieldOp([])

                with InsertionPoint(guarded.else_block):
                    scratch = pto.AllocTileOp(scratch_ty).result
                    session = _build_async_session(scratch, workspace_ptr, i32, ctx, sync_id=0)
                    put_event = _async_transfer("pto.tput_async", dst, src, session, ctx)
                    get_event = _async_transfer("pto.tget_async", src, dst, session, ctx)
                    _wait_async_event(put_event, session)
                    if hasattr(pto, "TestAsyncEventOp"):
                        pto.TestAsyncEventOp(get_event, session)
                    else:
                        Operation.create(
                            "pto.test_async_event",
                            operands=[get_event, session],
                            results=[IntegerType.get_signless(1, ctx)],
                        )
                    scf.YieldOp([])

                func.ReturnOp([])

            module.operation.verify()
            return module


if __name__ == "__main__":
    print(build())
