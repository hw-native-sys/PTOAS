#!/usr/bin/env python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

import os

from ptoas.mlir.dialects import arith, func, pto
from ptoas.mlir.ir import (
    BoolAttr,
    Context,
    F32Type,
    InsertionPoint,
    IndexType,
    IntegerAttr,
    IntegerType,
    Location,
    Module,
    StringAttr,
    UnitAttr,
)


def build():
    with Context() as ctx:
        pto.register_dialect(ctx, load=True)
        with Location.unknown(ctx):
            module = Module.create()
            arch = os.environ.get("PTOAS_SAMPLE_ARCH", "a5")
            module.operation.attributes["pto.target_arch"] = StringAttr.get(arch)

            f32 = F32Type.get(ctx)
            i32 = IntegerType.get_signless(32, ctx)
            i64 = IntegerType.get_signless(64, ctx)

            ptr_f32 = pto.PtrType.get(f32, ctx)
            tv5_f32 = pto.TensorViewType.get(5, f32, ctx)
            src_ptv = pto.PartitionTensorViewType.get([1, 1, 3, 4, 8], f32, ctx)

            vec = pto.AddressSpaceAttr.get(pto.AddressSpace.VEC, ctx)
            left = pto.AddressSpaceAttr.get(pto.AddressSpace.LEFT, ctx)
            mat = pto.AddressSpaceAttr.get(pto.AddressSpace.MAT, ctx)
            layout = pto.ConvLayoutAttr.get(pto.ConvLayout.NC1HWC0, ctx)
            col_major = pto.BLayoutAttr.get(pto.BLayout.ColMajor, ctx)
            row_box = pto.SLayoutAttr.get(pto.SLayout.RowMajor, ctx)
            null_pad = pto.PadValueAttr.get(pto.PadValue.Null, ctx)
            tile_cfg = pto.TileBufConfigAttr.get(
                col_major,
                row_box,
                pto.TileConfig.fractalABSize,
                null_pad,
                ctx,
            )
            dst_tile_ty = pto.TileBufType.get([16, 32], f32, left, [9, 32], tile_cfg, ctx)

            conv_cfg = pto.ConvTileConfigAttr.get(
                IntegerAttr.get(i32, 3),
                IntegerAttr.get(i32, 4),
                [1, 0, 1, 0],
                IntegerAttr.get(i32, 2),
                IntegerAttr.get(i32, 2),
                IntegerAttr.get(i32, 1),
                IntegerAttr.get(i32, 1),
                IntegerAttr.get(i32, 1),
                IntegerAttr.get(i32, 1),
                IntegerAttr.get(i32, 0),
                IntegerAttr.get(i32, 8),
                IntegerAttr.get(i32, 0),
                IntegerAttr.get(i32, 1),
                IntegerAttr.get(i32, 0),
                IntegerAttr.get(i32, 1),
                IntegerAttr.get(i32, 0),
                BoolAttr.get(False),
            )
            src_tile_ty = pto.ConvTileBufType.get(
                [1, 1, 3, 4, 8],
                f32,
                IntegerAttr.get(i64, 1 * 1 * 3 * 4 * 8),
                mat,
                layout,
                conv_cfg,
                ctx,
            )

            fn_ty = func.FunctionType.get([ptr_f32], [])
            with InsertionPoint(module.body):
                fn = func.FuncOp("timg2col_runtime_kernel", fn_ty)
                fn.operation.attributes["pto.entry"] = UnitAttr.get(ctx)
                entry = fn.add_entry_block()

            with InsertionPoint(entry):
                c0 = arith.ConstantOp(IndexType.get(ctx), 0).result
                c1 = arith.ConstantOp(IndexType.get(ctx), 1).result
                c3 = arith.ConstantOp(IndexType.get(ctx), 3).result
                c4 = arith.ConstantOp(IndexType.get(ctx), 4).result
                c8 = arith.ConstantOp(IndexType.get(ctx), 8).result
                c32 = arith.ConstantOp(IndexType.get(ctx), 32).result
                c96 = arith.ConstantOp(IndexType.get(ctx), 96).result
                src_ptr = entry.arguments[0]

                src_view = pto.MakeTensorViewOp(
                    tv5_f32, src_ptr, [c1, c1, c3, c4, c8], [c96, c96, c32, c8, c1]
                ).result

                src_part = pto.PartitionViewOp(
                    src_ptv,
                    src_view,
                    offsets=[c0, c0, c0, c0, c0],
                    sizes=[c1, c1, c3, c4, c8],
                ).result

                src_tile = pto.AllocTileOp(src_tile_ty).result
                dst_tile = pto.AllocTileOp(dst_tile_ty).result
                cube_section = pto.SectionCubeOp()
                with InsertionPoint(cube_section.body.blocks.append()):
                    fmatrix_mode = pto.FmatrixModeAttr.get(
                        pto.FmatrixMode.FMATRIX_B_MANUAL, ctx
                    )
                    pto.TLoadOp(None, src_part, src_tile)
                    pto.SetFmatrixOp(src_tile, fmatrixMode=fmatrix_mode)
                    pto.SetImg2colRptOp(src_tile, fmatrixMode=fmatrix_mode)
                    pto.SetImg2colPaddingOp(src_tile, fmatrixMode=fmatrix_mode)
                    pto.TImg2colOp(
                        dst_tile,
                        src_tile,
                        c1,
                        c8,
                        fmatrixMode=fmatrix_mode,
                    )
                func.ReturnOp([])

            module.operation.verify()
            return module


if __name__ == "__main__":
    print(build())
