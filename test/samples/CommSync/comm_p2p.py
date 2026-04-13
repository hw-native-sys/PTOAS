# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

from mlir.ir import Context, F32Type, IndexType, InsertionPoint, IntegerType, Location, Module
from mlir.dialects import arith, func, pto


def build():
    with Context() as ctx:
        pto.register_dialect(ctx, load=True)

        with Location.unknown(ctx):
            module = Module.create()

            f32 = F32Type.get(ctx)
            i32 = IntegerType.get_signless(32, ctx)
            i1 = IntegerType.get_signless(1, ctx)
            idx = IndexType.get(ctx)

            ptr_f32 = pto.PtrType.get(f32, ctx)
            ptr_i32 = pto.PtrType.get(i32, ctx)

            tv1_f32 = pto.TensorViewType.get([128], f32, ctx)
            pv1_f32 = pto.PartitionTensorViewType.get([128], f32, ctx)
            tv1_i32 = pto.TensorViewType.get([1], i32, ctx)
            pv1_i32 = pto.PartitionTensorViewType.get([1], i32, ctx)

            vec = pto.AddressSpaceAttr.get(pto.AddressSpace.VEC, ctx)
            bl = pto.BLayoutAttr.get(pto.BLayout.RowMajor, ctx)
            sl = pto.SLayoutAttr.get(pto.SLayout.NoneBox, ctx)
            pd = pto.PadValueAttr.get(pto.PadValue.Null, ctx)
            cfg = pto.TileBufConfigAttr.get(bl, sl, pto.TileConfig.fractalABSize, pd, ctx)
            tb_f32 = pto.TileBufType.get([1, 128], f32, vec, [1, 128], cfg, ctx)

            fn_ty = func.FunctionType.get([ptr_f32, ptr_f32, ptr_i32], [])
            with InsertionPoint(module.body):
                fn = func.FuncOp("comm_p2p_kernel", fn_ty)
                entry = fn.add_entry_block()

            with InsertionPoint(entry):
                dst_ptr, src_ptr, signal_ptr = entry.arguments
                c0 = arith.ConstantOp(idx, 0).result
                c1 = arith.ConstantOp(idx, 1).result
                c128 = arith.ConstantOp(idx, 128).result
                c7 = arith.ConstantOp(i32, 7).result

                dst_view = pto.MakeTensorViewOp(tv1_f32, dst_ptr, [c128], [c1]).result
                src_view = pto.MakeTensorViewOp(tv1_f32, src_ptr, [c128], [c1]).result
                signal_view = pto.MakeTensorViewOp(tv1_i32, signal_ptr, [c1], [c1]).result

                dst = pto.PartitionViewOp(pv1_f32, dst_view, offsets=[c0], sizes=[c128]).result
                src = pto.PartitionViewOp(pv1_f32, src_view, offsets=[c0], sizes=[c128]).result
                signal = pto.PartitionViewOp(pv1_i32, signal_view, offsets=[c0], sizes=[c1]).result

                ping = pto.AllocTileOp(tb_f32).result
                pong = pto.AllocTileOp(tb_f32).result

                pto.TPutOp(dst, src, ping)
                pto.TPutOp(
                    dst,
                    src,
                    ping,
                    pong=pong,
                    atomicType=pto.AtomicTypeAttr.get(pto.AtomicType.AtomicAdd, ctx),
                )
                pto.TGetOp(dst, src, ping)
                pto.TGetOp(dst, src, ping, pong=pong)
                pto.TNotifyOp(signal, c7, pto.NotifyOpAttr.get(pto.NotifyOp.Set, ctx))
                pto.TWaitOp(signal, c7, pto.WaitCmpAttr.get(pto.WaitCmp.GE, ctx))
                pto.TTestOp(signal, c7, pto.WaitCmpAttr.get(pto.WaitCmp.EQ, ctx))

                func.ReturnOp([])

            module.operation.verify()
            return module


if __name__ == "__main__":
    print(build())
