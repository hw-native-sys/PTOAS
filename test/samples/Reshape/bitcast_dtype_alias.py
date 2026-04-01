# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

from mlir.ir import Context, Location, Module, InsertionPoint
from mlir.dialects import func, arith, pto
from mlir.ir import F32Type, IndexType, IntegerType


def build():
    with Context() as ctx:
        pto.register_dialect(ctx, load=True)

        with Location.unknown(ctx):
            m = Module.create()

            f32 = F32Type.get(ctx)
            i32 = IntegerType.get_signless(32, ctx)

            ptr_f32 = pto.PtrType.get(f32, ctx)
            tv2_f32 = pto.TensorViewType.get(2, f32, ctx)
            tile_view_32_f32 = pto.PartitionTensorViewType.get([32, 32], f32, ctx)

            vec = pto.AddressSpaceAttr.get(pto.AddressSpace.VEC, ctx)
            bl = pto.BLayoutAttr.get(pto.BLayout.RowMajor, ctx)
            sl = pto.SLayoutAttr.get(pto.SLayout.NoneBox, ctx)
            pd = pto.PadValueAttr.get(pto.PadValue.Null, ctx)

            fractal_ab_size = pto.TileConfig.fractalABSize
            cfg = pto.TileBufConfigAttr.get(bl, sl, fractal_ab_size, pd, ctx)

            tile_buf_32_f32 = pto.TileBufType.get([32, 32], f32, vec, [32, 32], cfg, ctx)
            tile_buf_32_i32 = pto.TileBufType.get([32, 32], i32, vec, [32, 32], cfg, ctx)

            fn_ty = func.FunctionType.get([ptr_f32, ptr_f32], [])
            with InsertionPoint(m.body):
                fn = func.FuncOp("bitcast_dtype_alias", fn_ty)
                entry = fn.add_entry_block()

            with InsertionPoint(entry):
                c0 = arith.ConstantOp(IndexType.get(ctx), 0).result
                c1 = arith.ConstantOp(IndexType.get(ctx), 1).result
                c32 = arith.ConstantOp(IndexType.get(ctx), 32).result

                arg_in, arg_out = entry.arguments

                tv_in = pto.MakeTensorViewOp(tv2_f32, arg_in, [c32, c32], [c32, c1]).result
                tv_out = pto.MakeTensorViewOp(tv2_f32, arg_out, [c32, c32], [c32, c1]).result

                sv_in = pto.PartitionViewOp(tile_view_32_f32, tv_in, offsets=[c0, c0], sizes=[c32, c32]).result
                sv_out = pto.PartitionViewOp(tile_view_32_f32, tv_out, offsets=[c0, c0], sizes=[c32, c32]).result

                tb_f32 = pto.AllocTileOp(tile_buf_32_f32).result
                pto.TLoadOp(None, sv_in, tb_f32)

                tb_i32 = pto.BitcastOp(tile_buf_32_i32, tb_f32).result
                tb_f32_alias = pto.BitcastOp(tile_buf_32_f32, tb_i32).result

                pto.TStoreOp(None, tb_f32_alias, sv_out)

                func.ReturnOp([])

            m.operation.verify()
            return m


if __name__ == "__main__":
    print(build())
