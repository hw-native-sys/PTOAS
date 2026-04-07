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
    IntegerAttr,
    IntegerType,
    Location,
    Module,
    Operation,
    Type,
)
from mlir.dialects import arith, func, pto, scf


def _build_async_session(scratch, workspace, i32, ctx, sync_id=0):
    if hasattr(pto, "BuildAsyncSessionOp"):
        return pto.BuildAsyncSessionOp(scratch, workspace, sync_id=sync_id).result
    if hasattr(pto, "build_async_session"):
        return pto.build_async_session(scratch, workspace, sync_id=sync_id)
    op = Operation.create(
        "pto.build_async_session",
        operands=[scratch, workspace],
        attributes={"sync_id": IntegerAttr.get(i32, sync_id)},
        results=[pto.AsyncSessionType.get(ctx)],
    )
    return op.result


def _tput_async(dst, src, session, ctx):
    if hasattr(pto, "TPutAsyncOp"):
        return pto.TPutAsyncOp(dst, src, session).result
    if hasattr(pto, "tput_async"):
        return pto.tput_async(dst, src, session)
    op = Operation.create(
        "pto.tput_async",
        operands=[dst, src, session],
        results=[pto.AsyncEventType.get(ctx)],
    )
    return op.result


def _wait_async_event(event, session):
    if hasattr(pto, "WaitAsyncEventOp"):
        return pto.WaitAsyncEventOp(event, session).result
    if hasattr(pto, "wait_async_event"):
        return pto.wait_async_event(event, session)
    op = Operation.create(
        "pto.wait_async_event",
        operands=[event, session],
        results=[IntegerType.get_signless(1)],
    )
    return op.result


def _wait_after_async(event, session):
    _wait_async_event(event, session)

def build():
    with Context() as ctx:
        pto.register_dialect(ctx, load=True)

        with Location.unknown(ctx):
            module = Module.create()

            f32 = F32Type.get(ctx)
            i8 = IntegerType.get_signless(8, ctx)
            i32 = IntegerType.get_signless(32, ctx)
            idx = IndexType.get(ctx)

            vec = pto.AddressSpaceAttr.get(pto.AddressSpace.VEC, ctx)
            pipe_all = pto.PipeAttr.get(pto.PIPE.PIPE_ALL, ctx)
            ptr_f32 = pto.PtrType.get(f32, ctx)
            ptr_i8 = pto.PtrType.get(i8, ctx)
            tv1_f32 = pto.TensorViewType.get([256], f32, ctx)
            pv1_f32 = pto.PartitionTensorViewType.get([256], f32, ctx)
            scratch_ty = pto.TileBufType.get([1, 256], i8, vec, [1, 256], None, ctx)

            fn_ty = func.FunctionType.get(
                [
                    ptr_f32,
                    ptr_f32,
                    ptr_f32,
                    ptr_f32,
                    ptr_i8,
                    i32,  # nranks
                    i32,  # root_rank
                    i32,  # my_rank
                    i32,  # elem_offset
                    i32,  # elem_count
                ],
                [],
            )

            with InsertionPoint(module.body):
                fn = func.FuncOp("tput_async_kernel_impl_like", fn_ty)
                entry = fn.add_entry_block()

            with InsertionPoint(entry):
                (
                    dst_rank1_ptr,
                    dst_rank2_ptr,
                    dst_rank3_ptr,
                    src_ptr,
                    workspace_ptr,
                    nranks,
                    root_rank,
                    my_rank,
                    elem_offset,
                    elem_count,
                ) = entry.arguments

                c0 = arith.ConstantOp(idx, 0).result
                c1 = arith.ConstantOp(idx, 1).result
                c0_i32 = arith.ConstantOp(i32, 0).result
                c1_i32 = arith.ConstantOp(i32, 1).result
                c2_i32 = arith.ConstantOp(i32, 2).result
                c3_i32 = arith.ConstantOp(i32, 3).result
                c256_i32 = arith.ConstantOp(i32, 256).result
                c256 = arith.ConstantOp(idx, 256).result
                c1_idx = arith.ConstantOp(idx, 1).result
                dst_rank1_tv = pto.MakeTensorViewOp(tv1_f32, dst_rank1_ptr, [c256], [c1_idx]).result
                dst_rank2_tv = pto.MakeTensorViewOp(tv1_f32, dst_rank2_ptr, [c256], [c1_idx]).result
                dst_rank3_tv = pto.MakeTensorViewOp(tv1_f32, dst_rank3_ptr, [c256], [c1_idx]).result
                src_tv = pto.MakeTensorViewOp(tv1_f32, src_ptr, [c256], [c1_idx]).result
                dst_rank1 = pto.PartitionViewOp(pv1_f32, dst_rank1_tv, offsets=[c0], sizes=[c256]).result
                dst_rank2 = pto.PartitionViewOp(pv1_f32, dst_rank2_tv, offsets=[c0], sizes=[c256]).result
                dst_rank3 = pto.PartitionViewOp(pv1_f32, dst_rank3_tv, offsets=[c0], sizes=[c256]).result
                src = pto.PartitionViewOp(pv1_f32, src_tv, offsets=[c0], sizes=[c256]).result
                count_gt_zero = arith.CmpIOp(
                    arith.CmpIPredicate.sgt, elem_count, c0_i32
                ).result
                offset_ge_zero = arith.CmpIOp(
                    arith.CmpIPredicate.sge, elem_offset, c0_i32
                ).result
                end_index = arith.AddIOp(elem_offset, elem_count).result
                end_le_bound = arith.CmpIOp(
                    arith.CmpIPredicate.sle, end_index, c256_i32
                ).result
                valid = arith.AndIOp(
                    arith.AndIOp(count_gt_zero, offset_ge_zero).result, end_le_bound
                ).result

                valid_if = scf.IfOp(valid, [], hasElse=False)

                with InsertionPoint(valid_if.then_block):
                    scratch = pto.AllocTileOp(scratch_ty).result
                    session = _build_async_session(scratch, workspace_ptr, i32, ctx, sync_id=0)

                    is_root = arith.CmpIOp(
                        arith.CmpIPredicate.eq, my_rank, root_rank
                    ).result
                    root_if = scf.IfOp(is_root, [], hasElse=False)

                    with InsertionPoint(root_if.then_block):
                        nranks_idx = arith.IndexCastOp(idx, nranks).result
                        loop = scf.ForOp(c0, nranks_idx, c1, [])
                        with InsertionPoint(loop.body):
                            target_rank = loop.induction_variable
                            target_rank_i32 = arith.IndexCastOp(i32, target_rank).result
                            is_not_self = arith.CmpIOp(
                                arith.CmpIPredicate.ne, target_rank_i32, root_rank
                            ).result
                            target_if = scf.IfOp(is_not_self, [], hasElse=False)

                            with InsertionPoint(target_if.then_block):
                                is_rank1 = arith.CmpIOp(
                                    arith.CmpIPredicate.eq, target_rank_i32, c1_i32
                                ).result
                                rank1_if = scf.IfOp(is_rank1, [], hasElse=False)
                                with InsertionPoint(rank1_if.then_block):
                                    event1 = _tput_async(dst_rank1, src, session, ctx)
                                    _wait_after_async(event1, session)
                                    scf.YieldOp([])

                                is_rank2 = arith.CmpIOp(
                                    arith.CmpIPredicate.eq, target_rank_i32, c2_i32
                                ).result
                                rank2_if = scf.IfOp(is_rank2, [], hasElse=False)
                                with InsertionPoint(rank2_if.then_block):
                                    event2 = _tput_async(dst_rank2, src, session, ctx)
                                    _wait_after_async(event2, session)
                                    scf.YieldOp([])

                                is_rank3 = arith.CmpIOp(
                                    arith.CmpIPredicate.eq, target_rank_i32, c3_i32
                                ).result
                                rank3_if = scf.IfOp(is_rank3, [], hasElse=False)
                                with InsertionPoint(rank3_if.then_block):
                                    event3 = _tput_async(dst_rank3, src, session, ctx)
                                    _wait_after_async(event3, session)
                                    scf.YieldOp([])

                                scf.YieldOp([])
                            scf.YieldOp([])
                        scf.YieldOp([])

                    scf.YieldOp([])

                pto.barrier(pipe_all)
                func.ReturnOp([])

            module.operation.verify()
            return module


if __name__ == "__main__":
    print(build())
