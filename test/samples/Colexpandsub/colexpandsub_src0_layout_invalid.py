# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

from mlir.ir import Context, Location, Module, InsertionPoint
from mlir.dialects import func, pto
from mlir.ir import F32Type


def build():
    with Context() as ctx:
        pto.register_dialect(ctx, load=True)

        with Location.unknown(ctx):
            m = Module.create()

            f32 = F32Type.get(ctx)
            vec = pto.AddressSpaceAttr.get(pto.AddressSpace.VEC, ctx)
            bl_row = pto.BLayoutAttr.get(pto.BLayout.RowMajor, ctx)
            bl_col = pto.BLayoutAttr.get(pto.BLayout.ColMajor, ctx)
            sl = pto.SLayoutAttr.get(pto.SLayout.NoneBox, ctx)
            pd = pto.PadValueAttr.get(pto.PadValue.Null, ctx)

            fractal_ab_size = pto.TileConfig.fractalABSize
            cfg_row = pto.TileBufConfigAttr.get(bl_row, sl, fractal_ab_size, pd, ctx)
            cfg_col = pto.TileBufConfigAttr.get(bl_col, sl, fractal_ab_size, pd, ctx)

            # Invalid on purpose: src0 uses col-major layout.
            src0_ty = pto.TileBufType.get([32, 32], f32, vec, [32, 32], cfg_col, ctx)
            src1_ty = pto.TileBufType.get([1, 32], f32, vec, [1, 32], cfg_row, ctx)
            dst_ty = pto.TileBufType.get([32, 32], f32, vec, [32, 32], cfg_row, ctx)

            fn_ty = func.FunctionType.get([], [])
            with InsertionPoint(m.body):
                fn = func.FuncOp("tcolexpandsub_src0_layout_invalid", fn_ty)
                entry = fn.add_entry_block()

            with InsertionPoint(entry):
                src0 = pto.AllocTileOp(src0_ty).result
                src1 = pto.AllocTileOp(src1_ty).result
                dst = pto.AllocTileOp(dst_ty).result
                pto.TColExpandSubOp(src0, src1, dst)
                func.ReturnOp([])

            ok = m.operation.verify()
            if ok:
                return m
            raise SystemExit(1)


if __name__ == "__main__":
    print(build())
