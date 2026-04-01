# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""
Elementwise type conversion: f32 -> f16 using pto.tcvt (CAST_RINT rounding mode).

Kernel signature:
    tcvt_kernel_2d(src: ptr<f32>, dst: ptr<f16>)

Pipeline:
    make_tensor_view -> partition_view -> alloc_tile -> tload -> tcvt -> tstore
"""

from mlir.ir import Attribute, Context, F16Type, F32Type, IndexType, InsertionPoint, Location, Module, UnitAttr
from mlir.dialects import arith, func, pto


def build():
    with Context() as ctx:
        pto.register_dialect(ctx, load=True)

        with Location.unknown(ctx):
            m = Module.create()

            f16 = F16Type.get(ctx)
            f32 = F32Type.get(ctx)

            ptr_f32 = pto.PtrType.get(f32, ctx)
            ptr_f16 = pto.PtrType.get(f16, ctx)

            tv2_f32 = pto.TensorViewType.get(2, f32, ctx)
            tv2_f16 = pto.TensorViewType.get(2, f16, ctx)

            part_view_f32 = pto.PartitionTensorViewType.get([32, 32], f32, ctx)
            part_view_f16 = pto.PartitionTensorViewType.get([32, 32], f16, ctx)

            vec = pto.AddressSpaceAttr.get(pto.AddressSpace.VEC, ctx)
            bl  = pto.BLayoutAttr.get(pto.BLayout.RowMajor, ctx)
            sl  = pto.SLayoutAttr.get(pto.SLayout.NoneBox, ctx)
            pd  = pto.PadValueAttr.get(pto.PadValue.Null, ctx)

            fractal_ab_size = pto.TileConfig.fractalABSize
            cfg = pto.TileBufConfigAttr.get(bl, sl, fractal_ab_size, pd, ctx)

            tile_buf_f32 = pto.TileBufType.get([32, 32], f32, vec, [32, 32], cfg, ctx)
            tile_buf_f16 = pto.TileBufType.get([32, 32], f16, vec, [32, 32], cfg, ctx)

            fn_ty = func.FunctionType.get([ptr_f32, ptr_f16], [])
            with InsertionPoint(m.body):
                fn = func.FuncOp("tcvt_kernel_2d", fn_ty)
                fn.operation.attributes["pto.entry"] = UnitAttr.get(ctx)
                fn.operation.attributes["pto.kernel_kind"] = Attribute.parse(
                    "#pto.kernel_kind<vector>", ctx
                )
                entry = fn.add_entry_block()

            with InsertionPoint(entry):
                c0  = arith.ConstantOp(IndexType.get(ctx), 0).result
                c1  = arith.ConstantOp(IndexType.get(ctx), 1).result
                c32 = arith.ConstantOp(IndexType.get(ctx), 32).result

                arg_src, arg_dst = entry.arguments

                # Build tensor views over the raw pointers.
                tv_src = pto.MakeTensorViewOp(tv2_f32, arg_src, [c32, c32], [c32, c1]).result
                tv_dst = pto.MakeTensorViewOp(tv2_f16, arg_dst, [c32, c32], [c32, c1]).result

                # Partition views select the 32x32 tile window.
                sv_src = pto.PartitionViewOp(part_view_f32, tv_src, offsets=[c0, c0], sizes=[c32, c32]).result
                sv_dst = pto.PartitionViewOp(part_view_f16, tv_dst, offsets=[c0, c0], sizes=[c32, c32]).result

                # Allocate tile buffers.
                tb_src = pto.AllocTileOp(tile_buf_f32).result
                tb_dst = pto.AllocTileOp(tile_buf_f16).result

                # Load f32 data from GM into the source tile.
                pto.TLoadOp(None, sv_src, tb_src)

                # Convert f32 -> f16 using the default CAST_RINT rounding mode.
                pto.TCvtOp(tb_src, tb_dst)

                # Store the f16 result back to GM.
                pto.TStoreOp(None, tb_dst, sv_dst)

                func.ReturnOp([])

            m.operation.verify()
            return m


if __name__ == "__main__":
    print(build())
