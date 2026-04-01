# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

from mlir.ir import Context, InsertionPoint, Location, Module
from mlir.dialects import arith, func, pto
from mlir.ir import F32Type, IndexType


def build():
    with Context() as ctx:
        pto.register_dialect(ctx, load=True)

        with Location.unknown(ctx):
            m = Module.create()

            f32 = F32Type.get(ctx)
            ptr_f32 = pto.PtrType.get(f32, ctx)

            tv2_f32 = pto.TensorViewType.get(2, f32, ctx)
            tile_view_32_16 = pto.PartitionTensorViewType.get([32, 16], f32, ctx)
            tile_view_32_32 = pto.PartitionTensorViewType.get([32, 32], f32, ctx)
            vec = pto.AddressSpaceAttr.get(pto.AddressSpace.VEC, ctx)
            bl = pto.BLayoutAttr.get(pto.BLayout.RowMajor, ctx)
            sl = pto.SLayoutAttr.get(pto.SLayout.NoneBox, ctx)
            pd = pto.PadValueAttr.get(pto.PadValue.Zero, ctx)

            fractal_ab_size = pto.TileConfig.fractalABSize
            cfg = pto.TileBufConfigAttr.get(bl, sl, fractal_ab_size, pd, ctx)
            tile_buf_32_16 = pto.TileBufType.get([32, 16], f32, vec, [32, 16], cfg, ctx)
            tile_buf_32_32 = pto.TileBufType.get([32, 32], f32, vec, [32, 32], cfg, ctx)

            fn_ty = func.FunctionType.get([ptr_f32, ptr_f32], [])
            with InsertionPoint(m.body):
                fn = func.FuncOp("fillpad_expand_kernel_2d", fn_ty)
                entry = fn.add_entry_block()

            with InsertionPoint(entry):
                c0 = arith.ConstantOp(IndexType.get(ctx), 0).result
                c1 = arith.ConstantOp(IndexType.get(ctx), 1).result
                c16 = arith.ConstantOp(IndexType.get(ctx), 16).result
                c32 = arith.ConstantOp(IndexType.get(ctx), 32).result

                arg0, arg1 = entry.arguments

                src_tv = pto.MakeTensorViewOp(tv2_f32, arg0, [c32, c16], [c16, c1]).result
                dst_tv = pto.MakeTensorViewOp(tv2_f32, arg1, [c32, c32], [c32, c1]).result

                src_sv = pto.PartitionViewOp(
                    tile_view_32_16, src_tv, offsets=[c0, c0], sizes=[c32, c16]
                ).result
                dst_sv = pto.PartitionViewOp(
                    tile_view_32_32, dst_tv, offsets=[c0, c0], sizes=[c32, c32]
                ).result

                src_tb = pto.AllocTileOp(tile_buf_32_16).result
                dst_tb = pto.AllocTileOp(tile_buf_32_32).result

                pto.TLoadOp(None, src_sv, src_tb)
                pto.TFillPadExpandOp(src_tb, dst_tb)
                pto.TStoreOp(None, dst_tb, dst_sv)

                func.ReturnOp([])

            m.operation.verify()
            return m


if __name__ == "__main__":
    print(build())
