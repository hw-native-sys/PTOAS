# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""
Test that TGetValOp rejects MAT tile_buf (Ascend hardware does not support
reading from MAT tile_buf to scalar). Verification must fail.
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
            # MAT memory space (invalid for TGetValOp; only VEC is allowed)
            mat = pto.AddressSpaceAttr.get(pto.AddressSpace.MAT, ctx)
            bl = pto.BLayoutAttr.get(pto.BLayout.RowMajor, ctx)
            sl = pto.SLayoutAttr.get(pto.SLayout.NoneBox, ctx)
            pd = pto.PadValueAttr.get(pto.PadValue.Null, ctx)
            fractal_ab_size = pto.TileConfig.fractalABSize
            cfg = pto.TileBufConfigAttr.get(bl, sl, fractal_ab_size, pd, ctx)
            tile_buf_mat_f32 = pto.TileBufType.get(
                [32, 32], f32, mat, [32, 32], cfg, ctx
            )

            fn_ty = func.FunctionType.get([], [])
            with InsertionPoint(m.body):
                fn = func.FuncOp("tgetval_mat_invalid", fn_ty)
                entry = fn.add_entry_block()

            with InsertionPoint(entry):
                c0 = arith.ConstantOp(IndexType.get(ctx), 0).result
                tb_mat = pto.AllocTileOp(tile_buf_mat_f32).result
                # TGetValOp from MAT tile_buf must fail verify
                pto.TGetValOp(f32, tb_mat, c0)
                func.ReturnOp([])

            ok = m.operation.verify()
            if ok:
                raise SystemExit(
                    1,
                    "expected TGetValOp with MAT tile_buf to fail verification",
                )
            return m


if __name__ == "__main__":
    build()
    print("PASS: TGetValOp with MAT tile_buf correctly failed verification")
