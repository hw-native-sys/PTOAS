# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

from mlir.ir import Context, Location, Module, InsertionPoint
from mlir.dialects import func, arith, pto
from mlir.ir import IntegerType, IndexType


def build():
    with Context() as ctx:
        pto.register_dialect(ctx, load=True)

        with Location.unknown(ctx):
            m = Module.create()

            i16 = IntegerType.get_signless(16, ctx)
            ptr_i16 = pto.PtrType.get(i16, ctx)

            tv2_i16 = pto.TensorViewType.get(2, i16, ctx)

            tile_view_32x16 = pto.PartitionTensorViewType.get([32, 16], i16, ctx)
            tile_view_32x32 = pto.PartitionTensorViewType.get([32, 32], i16, ctx)

            vec = pto.AddressSpaceAttr.get(pto.AddressSpace.VEC, ctx)
            bl = pto.BLayoutAttr.get(pto.BLayout.RowMajor, ctx)
            sl = pto.SLayoutAttr.get(pto.SLayout.NoneBox, ctx)
            pd = pto.PadValueAttr.get(pto.PadValue.Null, ctx)

            fractal_ab_size = pto.TileConfig.fractalABSize
            cfg = pto.TileBufConfigAttr.get(bl, sl, fractal_ab_size, pd, ctx)

            tb_32x16 = pto.TileBufType.get([32, 16], i16, vec, [32, 16], cfg, ctx)
            tb_32x32 = pto.TileBufType.get([32, 32], i16, vec, [32, 32], cfg, ctx)

            fn_ty = func.FunctionType.get([ptr_i16, ptr_i16, ptr_i16], [])
            with InsertionPoint(m.body):
                fn = func.FuncOp("concat_kernel_2d", fn_ty)
                entry = fn.add_entry_block()

            with InsertionPoint(entry):
                c0 = arith.ConstantOp(IndexType.get(ctx), 0).result
                c1 = arith.ConstantOp(IndexType.get(ctx), 1).result
                c16 = arith.ConstantOp(IndexType.get(ctx), 16).result
                c32 = arith.ConstantOp(IndexType.get(ctx), 32).result

                arg0, arg1, arg2 = entry.arguments

                tv_src0 = pto.MakeTensorViewOp(tv2_i16, arg0, [c32, c16], [c16, c1]).result
                tv_src1 = pto.MakeTensorViewOp(tv2_i16, arg1, [c32, c16], [c16, c1]).result
                tv_dst = pto.MakeTensorViewOp(tv2_i16, arg2, [c32, c32], [c32, c1]).result

                sv_src0 = pto.PartitionViewOp(tile_view_32x16, tv_src0, offsets=[c0, c0], sizes=[c32, c16]).result
                sv_src1 = pto.PartitionViewOp(tile_view_32x16, tv_src1, offsets=[c0, c0], sizes=[c32, c16]).result
                sv_dst = pto.PartitionViewOp(tile_view_32x32, tv_dst, offsets=[c0, c0], sizes=[c32, c32]).result

                tb_src0 = pto.AllocTileOp(tb_32x16).result
                tb_src1 = pto.AllocTileOp(tb_32x16).result
                tb_dst = pto.AllocTileOp(tb_32x32).result

                pto.TLoadOp(None, sv_src0, tb_src0)
                pto.TLoadOp(None, sv_src1, tb_src1)

                pto.TConcatOp(tb_src0, tb_src1, tb_dst)

                pto.TStoreOp(None, tb_dst, sv_dst)
                func.ReturnOp([])

            m.operation.verify()
            return m


if __name__ == "__main__":
    print(build())

