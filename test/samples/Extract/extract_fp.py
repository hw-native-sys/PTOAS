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
            i8 = IntegerType.get_signless(8, ctx)
            idx = IndexType.get(ctx)

            acc = pto.AddressSpaceAttr.get(pto.AddressSpace.ACC, ctx)
            mat = pto.AddressSpaceAttr.get(pto.AddressSpace.MAT, ctx)
            scaling = pto.AddressSpaceAttr.get(pto.AddressSpace.SCALING, ctx)
            pd = pto.PadValueAttr.get(pto.PadValue.Null, ctx)

            acc_cfg = pto.TileBufConfigAttr.get(
                pto.BLayoutAttr.get(pto.BLayout.ColMajor, ctx),
                pto.SLayoutAttr.get(pto.SLayout.RowMajor, ctx),
                1024,
                pd,
                ctx,
            )
            mat_cfg = pto.TileBufConfigAttr.get(
                pto.BLayoutAttr.get(pto.BLayout.ColMajor, ctx),
                pto.SLayoutAttr.get(pto.SLayout.RowMajor, ctx),
                pto.TileConfig.fractalABSize,
                pd,
                ctx,
            )
            fp_cfg = pto.TileBufConfigAttr.get(
                pto.BLayoutAttr.get(pto.BLayout.RowMajor, ctx),
                pto.SLayoutAttr.get(pto.SLayout.RowMajor, ctx),
                pto.TileConfig.fractalABSize,
                pd,
                ctx,
            )

            src_ty = pto.TileBufType.get([32, 32], f32, acc, [32, 32], acc_cfg, ctx)
            fp_ty = pto.TileBufType.get([32, 32], f32, scaling, [32, 32], fp_cfg, ctx)
            dst_ty = pto.TileBufType.get([32, 32], i8, mat, [32, 32], mat_cfg, ctx)

            fn_ty = func.FunctionType.get([], [])
            with InsertionPoint(m.body):
                fn = func.FuncOp("extract_fp_kernel", fn_ty)
                entry = fn.add_entry_block()

            with InsertionPoint(entry):
                c0 = arith.ConstantOp(idx, 0).result
                src = pto.AllocTileOp(src_ty).result
                fp = pto.AllocTileOp(fp_ty).result
                dst = pto.AllocTileOp(dst_ty).result
                pto.TExtractFPOp(src, fp, c0, c0, dst)
                func.ReturnOp([])

            m.operation.verify()
            return m


if __name__ == "__main__":
    print(build())
