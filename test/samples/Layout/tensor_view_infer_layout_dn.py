# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""
Infer tensor_view layout sample (DN).

This checks that PTOAS can infer `layout=DN` for a 2D column-vector GM view:
  - shape = (16 x 1)
  - strides = (1, 1)  (contiguous)

The expected emitted C++ should use `pto::Layout::DN` in GlobalTensor<>.
"""

from mlir.ir import Context, Location, Module, InsertionPoint
from mlir.dialects import func, arith, pto
from mlir.ir import F32Type, IndexType


def build():
    with Context() as ctx:
        pto.register_dialect(ctx, load=True)

        with Location.unknown(ctx):
            m = Module.create()

            f32 = F32Type.get(ctx)
            ptr_f32 = pto.PtrType.get(f32, ctx)

            tv2_f32 = pto.TensorViewType.get(2, f32, ctx)
            tile_view_16x1 = pto.PartitionTensorViewType.get([16, 1], f32, ctx)

            vec = pto.AddressSpaceAttr.get(pto.AddressSpace.VEC, ctx)
            bl = pto.BLayoutAttr.get(pto.BLayout.ColMajor, ctx)
            sl = pto.SLayoutAttr.get(pto.SLayout.NoneBox, ctx)
            pd = pto.PadValueAttr.get(pto.PadValue.Null, ctx)

            fractal_ab_size = pto.TileConfig.fractalABSize
            cfg = pto.TileBufConfigAttr.get(bl, sl, fractal_ab_size, pd, ctx)
            tile_buf_16x1 = pto.TileBufType.get([16, 1], f32, vec, [16, 1], cfg, ctx)

            fn_ty = func.FunctionType.get([ptr_f32, ptr_f32], [])
            with InsertionPoint(m.body):
                fn = func.FuncOp("tensor_view_infer_layout_dn_2d", fn_ty)
                entry = fn.add_entry_block()

            with InsertionPoint(entry):
                c0 = arith.ConstantOp(IndexType.get(ctx), 0).result
                c1 = arith.ConstantOp(IndexType.get(ctx), 1).result
                c16 = arith.ConstantOp(IndexType.get(ctx), 16).result

                src, dst = entry.arguments

                src_view = pto.MakeTensorViewOp(tv2_f32, src, [c16, c1], [c1, c1]).result
                src_part = pto.PartitionViewOp(
                    tile_view_16x1, src_view, offsets=[c0, c0], sizes=[c16, c1]
                ).result

                tile = pto.AllocTileOp(tile_buf_16x1).result
                pto.TLoadOp(None, src_part, tile)

                dst_view = pto.MakeTensorViewOp(tv2_f32, dst, [c16, c1], [c1, c1]).result
                dst_part = pto.PartitionViewOp(
                    tile_view_16x1, dst_view, offsets=[c0, c0], sizes=[c16, c1]
                ).result
                pto.TStoreOp(None, tile, dst_part)

                func.ReturnOp([])

            m.operation.verify()
            return m


if __name__ == "__main__":
    print(build())
