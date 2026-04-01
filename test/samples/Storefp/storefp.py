# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

from mlir.ir import Context, Location, Module, InsertionPoint
from mlir.dialects import func, arith, pto
from mlir.ir import F32Type, IntegerType, IndexType

def build():
    with Context() as ctx:
        pto.register_dialect(ctx, load=True)

        with Location.unknown(ctx):
            m = Module.create()

            f32 = F32Type.get(ctx)
            i8 = IntegerType.get_signless(8, ctx)
            u64 = IntegerType.get_unsigned(64, ctx)
            ptr_i8 = pto.PtrType.get(i8, ctx)
            tv2_i8 = pto.TensorViewType.get(2, i8, ctx)
            tile_view_8 = pto.PartitionTensorViewType.get([32, 32], i8, ctx)

            acc = pto.AddressSpaceAttr.get(pto.AddressSpace.ACC, ctx)
            scaling = pto.AddressSpaceAttr.get(pto.AddressSpace.SCALING, ctx)

            pad = pto.PadValueAttr.get(pto.PadValue.Null, ctx)

            cfg_acc = pto.TileBufConfigAttr.get(
                pto.BLayoutAttr.get(pto.BLayout.ColMajor, ctx),
                pto.SLayoutAttr.get(pto.SLayout.RowMajor, ctx),
                pto.TileConfig.fractalCSize,
                pad,
                ctx,
            )
            cfg_fp = pto.TileBufConfigAttr.get(
                pto.BLayoutAttr.get(pto.BLayout.RowMajor, ctx),
                pto.SLayoutAttr.get(pto.SLayout.NoneBox, ctx),
                pto.TileConfig.fractalABSize,
                pad,
                ctx,
            )

            acc_tile_ty = pto.TileBufType.get([16, 32], f32, acc, [1, 32], cfg_acc, ctx)
            fp_tile_ty = pto.TileBufType.get([1, 16], u64, scaling, [1, 16], cfg_fp, ctx)

            fn_ty = func.FunctionType.get([ptr_i8], [])
            with InsertionPoint(m.body):
                fn = func.FuncOp("tstore_fp_pass", fn_ty)
                entry = fn.add_entry_block()

            with InsertionPoint(entry):
                dst_ptr = entry.arguments[0]
                c0 = arith.ConstantOp(IndexType.get(ctx), 0).result
                c1 = arith.ConstantOp(IndexType.get(ctx), 1).result
                c32 = arith.ConstantOp(IndexType.get(ctx), 32).result

                tv = pto.MakeTensorViewOp(tv2_i8, dst_ptr, [c32, c32], [c32, c1]).result
                sv = pto.PartitionViewOp(tile_view_8, tv, offsets=[c0, c0], sizes=[c32, c32]).result

                acc_tile = pto.AllocTileOp(acc_tile_ty).result
                fp_tile = pto.AllocTileOp(fp_tile_ty).result
                pto.TStoreFPOp(acc_tile, fp_tile, sv)
                func.ReturnOp([])

            m.operation.verify()
            return m


if __name__ == "__main__":
    print(build())
