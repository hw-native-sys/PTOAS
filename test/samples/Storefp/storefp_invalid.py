# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

from mlir.ir import Context, Location, Module, InsertionPoint
from mlir.dialects import func, pto
from mlir.ir import F16Type, IntegerType, MemRefType


def build():
    with Context() as ctx:
        pto.register_dialect(ctx, load=True)

        with Location.unknown(ctx):
            m = Module.create()

            f16 = F16Type.get(ctx)
            i8 = IntegerType.get_signless(8, ctx)
            u64 = IntegerType.get_unsigned(64, ctx)

            gm = pto.AddressSpaceAttr.get(pto.AddressSpace.GM, ctx)
            vec = pto.AddressSpaceAttr.get(pto.AddressSpace.VEC, ctx)
            scaling = pto.AddressSpaceAttr.get(pto.AddressSpace.SCALING, ctx)

            pad = pto.PadValueAttr.get(pto.PadValue.Null, ctx)
            cfg_vec = pto.TileBufConfigAttr.get(
                pto.BLayoutAttr.get(pto.BLayout.RowMajor, ctx),
                pto.SLayoutAttr.get(pto.SLayout.NoneBox, ctx),
                pto.TileConfig.fractalABSize,
                pad,
                ctx,
            )
            cfg_fp = pto.TileBufConfigAttr.get(
                pto.BLayoutAttr.get(pto.BLayout.RowMajor, ctx),
                pto.SLayoutAttr.get(pto.SLayout.NoneBox, ctx),
                pto.TileConfig.fractalABSize,
                pad,
                ctx,
            )

            src_tile_ty = pto.TileBufType.get([16, 32], f16, vec, [16, 32], cfg_vec, ctx)
            fp_tile_ty = pto.TileBufType.get([1, 16], u64, scaling, [1, 16], cfg_fp, ctx)
            dst_memref_ty = MemRefType.get([16, 32], i8, memory_space=gm)

            fn_ty = func.FunctionType.get([dst_memref_ty], [])
            with InsertionPoint(m.body):
                fn = func.FuncOp("tstore_fp_invalid_vec_f16_to_i8", fn_ty)
                entry = fn.add_entry_block()

            with InsertionPoint(entry):
                dst = entry.arguments[0]
                src_tile = pto.AllocTileOp(src_tile_ty).result
                fp_tile = pto.AllocTileOp(fp_tile_ty).result
                pto.TStoreFPOp(src_tile, fp_tile, dst)
                func.ReturnOp([])

            ok = m.operation.verify()
            if ok:
                return m
            raise SystemExit(1)


if __name__ == "__main__":
    print(build())
