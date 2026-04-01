# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

from mlir.dialects import arith, func, pto
from mlir.ir import Context, F32Type, IndexType, InsertionPoint, IntegerType, Location, Module


def build():
    with Context() as ctx:
        pto.register_dialect(ctx, load=True)

        with Location.unknown(ctx):
            module = Module.create()

            f32 = F32Type.get(ctx)
            u32 = IntegerType.get_unsigned(32, ctx)

            ptr_f32 = pto.PtrType.get(f32, ctx)
            ptr_u32 = pto.PtrType.get(u32, ctx)
            tv2_f32 = pto.TensorViewType.get(2, f32, ctx)
            tv2_u32 = pto.TensorViewType.get(2, u32, ctx)

            src_rows = 2
            src_cols = 128
            offset_rows = 2
            offset_cols = 16

            tile_view_f32 = pto.PartitionTensorViewType.get([src_rows, src_cols], f32, ctx)
            tile_view_u32 = pto.PartitionTensorViewType.get([offset_rows, offset_cols], u32, ctx)

            vec = pto.AddressSpaceAttr.get(pto.AddressSpace.VEC, ctx)
            bl = pto.BLayoutAttr.get(pto.BLayout.RowMajor, ctx)
            sl = pto.SLayoutAttr.get(pto.SLayout.NoneBox, ctx)
            pd = pto.PadValueAttr.get(pto.PadValue.Null, ctx)
            cfg = pto.TileBufConfigAttr.get(bl, sl, pto.TileConfig.fractalABSize, pd, ctx)

            tile_buf_f32 = pto.TileBufType.get([src_rows, src_cols], f32, vec, [src_rows, src_cols], cfg, ctx)
            tile_buf_u32 = pto.TileBufType.get([offset_rows, offset_cols], u32, vec, [offset_rows, offset_cols], cfg, ctx)

            fn_ty = func.FunctionType.get([ptr_f32, ptr_u32, ptr_f32], [])
            with InsertionPoint(module.body):
                fn = func.FuncOp("vec_add_kernel_2d", fn_ty)
                entry = fn.add_entry_block()

            with InsertionPoint(entry):
                c0 = arith.ConstantOp(IndexType.get(ctx), 0).result
                c1 = arith.ConstantOp(IndexType.get(ctx), 1).result
                c2 = arith.ConstantOp(IndexType.get(ctx), src_rows).result
                c128 = arith.ConstantOp(IndexType.get(ctx), src_cols).result
                c16 = arith.ConstantOp(IndexType.get(ctx), offset_cols).result

                src_ptr, offset_ptr, dst_ptr = entry.arguments

                src_tv = pto.MakeTensorViewOp(tv2_f32, src_ptr, [c2, c128], [c128, c1]).result
                offset_tv = pto.MakeTensorViewOp(tv2_u32, offset_ptr, [c2, c16], [c16, c1]).result
                dst_tv = pto.MakeTensorViewOp(tv2_f32, dst_ptr, [c2, c128], [c128, c1]).result

                src_view = pto.PartitionViewOp(tile_view_f32, src_tv, offsets=[c0, c0], sizes=[c2, c128]).result
                offset_view = pto.PartitionViewOp(tile_view_u32, offset_tv, offsets=[c0, c0], sizes=[c2, c16]).result
                dst_view = pto.PartitionViewOp(tile_view_f32, dst_tv, offsets=[c0, c0], sizes=[c2, c128]).result

                src_tile = pto.AllocTileOp(tile_buf_f32).result
                offset_tile = pto.AllocTileOp(tile_buf_u32).result
                dst_tile = pto.AllocTileOp(tile_buf_f32).result

                pto.TLoadOp(None, src_view, src_tile)
                pto.TLoadOp(None, offset_view, offset_tile)
                pto.TGatherBOp(src_tile, offset_tile, dst_tile)
                pto.TStoreOp(None, dst_tile, dst_view)
                func.ReturnOp([])

            module.operation.verify()
            return module


if __name__ == "__main__":
    print(build())
