# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""TTri sample.

Generates one lower-triangular and one upper-triangular 32x32 i32 tile and
stores them into a single output buffer as two consecutive 32x32 slices.
"""

from mlir.ir import Context, Location, Module, InsertionPoint, IndexType, IntegerType
from mlir.dialects import func, arith, pto


def build():
    with Context() as ctx:
        pto.register_dialect(ctx, load=True)

        with Location.unknown(ctx):
            module = Module.create()

            i32 = IntegerType.get_signless(32, ctx)
            idx = IndexType.get(ctx)
            ptr_i32 = pto.PtrType.get(i32, ctx)
            tv2_i32 = pto.TensorViewType.get(2, i32, ctx)
            ptv_32x32 = pto.PartitionTensorViewType.get([32, 32], i32, ctx)
            vec = pto.AddressSpaceAttr.get(pto.AddressSpace.VEC, ctx)
            bl = pto.BLayoutAttr.get(pto.BLayout.RowMajor, ctx)
            sl = pto.SLayoutAttr.get(pto.SLayout.NoneBox, ctx)
            pd = pto.PadValueAttr.get(pto.PadValue.Null, ctx)
            cfg = pto.TileBufConfigAttr.get(bl, sl, pto.TileConfig.fractalABSize, pd, ctx)
            tb_32x32 = pto.TileBufType.get([32, 32], i32, vec, [32, 32], cfg, ctx)

            fn_ty = func.FunctionType.get([ptr_i32], [])
            with InsertionPoint(module.body):
                fn = func.FuncOp("ttri_kernel", fn_ty)
                entry = fn.add_entry_block()

            with InsertionPoint(entry):
                c0 = arith.ConstantOp(idx, 0).result
                c1 = arith.ConstantOp(idx, 1).result
                c32 = arith.ConstantOp(idx, 32).result
                c64 = arith.ConstantOp(idx, 64).result
                d0 = arith.ConstantOp(i32, 0).result
                d16 = arith.ConstantOp(i32, 16).result

                out_ptr = entry.arguments[0]

                out_tv = pto.MakeTensorViewOp(tv2_i32, out_ptr, [c64, c32], [c32, c1]).result
                out_lower = pto.PartitionViewOp(
                    ptv_32x32, out_tv, offsets=[c0, c0], sizes=[c32, c32]
                ).result
                out_upper = pto.PartitionViewOp(
                    ptv_32x32, out_tv, offsets=[c32, c0], sizes=[c32, c32]
                ).result

                lower_tile = pto.AllocTileOp(tb_32x32).result
                upper_tile = pto.AllocTileOp(tb_32x32).result

                pto.TTriOp(d0, lower_tile)
                pto.TTriOp(d16, upper_tile, upperOrLower=1)

                pto.TStoreOp(None, lower_tile, out_lower)
                pto.TStoreOp(None, upper_tile, out_upper)

                func.ReturnOp([])

            module.operation.verify()
            return module


if __name__ == "__main__":
    print(build())
