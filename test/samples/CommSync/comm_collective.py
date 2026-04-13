# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

from mlir.ir import Context, F32Type, IndexType, InsertionPoint, Location, Module
from mlir.dialects import arith, func, pto

def build():
    with Context() as ctx:
        pto.register_dialect(ctx, load=True)

        with Location.unknown(ctx):
            module = Module.create()

            f32 = F32Type.get(ctx)
            idx = IndexType.get(ctx)
            ptr_f32 = pto.PtrType.get(f32, ctx)
            tv1_f32 = pto.TensorViewType.get([128], f32, ctx)
            pv1_f32 = pto.PartitionTensorViewType.get([128], f32, ctx)

            vec = pto.AddressSpaceAttr.get(pto.AddressSpace.VEC, ctx)
            bl = pto.BLayoutAttr.get(pto.BLayout.RowMajor, ctx)
            sl = pto.SLayoutAttr.get(pto.SLayout.NoneBox, ctx)
            pd = pto.PadValueAttr.get(pto.PadValue.Null, ctx)
            cfg = pto.TileBufConfigAttr.get(bl, sl, pto.TileConfig.fractalABSize, pd, ctx)
            tb_f32 = pto.TileBufType.get([1, 128], f32, vec, [1, 128], cfg, ctx)

            fn_ty = func.FunctionType.get(
                [ptr_f32, ptr_f32, ptr_f32, ptr_f32, ptr_f32], []
            )
            with InsertionPoint(module.body):
                fn = func.FuncOp("comm_collective_kernel", fn_ty)
                entry = fn.add_entry_block()

            with InsertionPoint(entry):
                dst_ptr, src_ptr, peer0_ptr, peer1_ptr, peer2_ptr = entry.arguments
                c0 = arith.ConstantOp(idx, 0).result
                c1 = arith.ConstantOp(idx, 1).result
                c128 = arith.ConstantOp(idx, 128).result

                def make_part(arg):
                    view = pto.MakeTensorViewOp(tv1_f32, arg, [c128], [c1]).result
                    return pto.PartitionViewOp(
                        pv1_f32, view, offsets=[c0], sizes=[c128]
                    ).result

                dst = make_part(dst_ptr)
                src = make_part(src_ptr)
                peer0 = make_part(peer0_ptr)
                peer1 = make_part(peer1_ptr)
                peer2 = make_part(peer2_ptr)

                ping = pto.AllocTileOp(tb_f32).result
                pong = pto.AllocTileOp(tb_f32).result
                acc = pto.AllocTileOp(tb_f32).result

                group = [peer0, peer1, peer2]
                root = 1

                pto.TBroadcastOp(src, ping, group, root)
                pto.TBroadcastOp(src, ping, group, root, pong=pong)
                pto.CommTGatherOp(dst, ping, group, root)
                pto.CommTGatherOp(dst, ping, group, root, pong=pong)
                pto.CommTScatterOp(src, ping, group, root)
                pto.CommTScatterOp(src, ping, group, root, pong=pong)
                pto.TReduceOp(
                    dst,
                    acc,
                    ping,
                    group,
                    pto.ReduceOpAttr.get(pto.ReduceOp.Sum, ctx),
                    root,
                )
                pto.TReduceOp(
                    dst,
                    acc,
                    ping,
                    group,
                    pto.ReduceOpAttr.get(pto.ReduceOp.Max, ctx),
                    root,
                    recvPong=pong,
                )

                func.ReturnOp([])

            module.operation.verify()
            return module


if __name__ == "__main__":
    print(build())
