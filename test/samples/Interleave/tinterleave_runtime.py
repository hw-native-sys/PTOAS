# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

import os

from ptoas.mlir.ir import (
    Context,
    InsertionPoint,
    IndexType,
    Location,
    Module,
    StringAttr,
    UnitAttr,
)
from ptoas.mlir.dialects import arith, func, pto
from ptoas.mlir.ir import F32Type


def build():
    with Context() as ctx:
        pto.register_dialect(ctx, load=True)
        with Location.unknown(ctx):
            module = Module.create()
            arch = os.environ.get("PTOAS_SAMPLE_ARCH", "a5")
            module.operation.attributes["pto.target_arch"] = StringAttr.get(arch)
            f32 = F32Type.get(ctx)
            ptr_f32 = pto.PtrType.get(f32, ctx)
            tensor_view = pto.TensorViewType.get(2, f32, ctx)
            partition_view = pto.PartitionTensorViewType.get([16, 64], f32, ctx)
            vec = pto.AddressSpaceAttr.get(pto.AddressSpace.VEC, ctx)
            config = pto.TileBufConfigAttr.get(
                pto.BLayoutAttr.get(pto.BLayout.RowMajor, ctx),
                pto.SLayoutAttr.get(pto.SLayout.NoneBox, ctx),
                pto.TileConfig.fractalABSize,
                pto.PadValueAttr.get(pto.PadValue.Null, ctx),
                ctx,
            )
            tile_type = pto.TileBufType.get([16, 64], f32, vec, [16, 64], config, ctx)

            function_type = func.FunctionType.get([ptr_f32] * 4, [])
            with InsertionPoint(module.body):
                function = func.FuncOp("tinterleave_runtime_kernel", function_type)
                function.operation.attributes["pto.entry"] = UnitAttr.get(ctx)
                entry = function.add_entry_block()

            with InsertionPoint(entry):
                c0 = arith.ConstantOp(IndexType.get(ctx), 0).result
                c1 = arith.ConstantOp(IndexType.get(ctx), 1).result
                c16 = arith.ConstantOp(IndexType.get(ctx), 16).result
                c64 = arith.ConstantOp(IndexType.get(ctx), 64).result
                src0_ptr, src1_ptr, dst0_ptr, dst1_ptr = entry.arguments

                src0_view = pto.MakeTensorViewOp(
                    tensor_view, src0_ptr, [c16, c64], [c64, c1]
                ).result
                src1_view = pto.MakeTensorViewOp(
                    tensor_view, src1_ptr, [c16, c64], [c64, c1]
                ).result
                dst0_view = pto.MakeTensorViewOp(
                    tensor_view, dst0_ptr, [c16, c64], [c64, c1]
                ).result
                dst1_view = pto.MakeTensorViewOp(
                    tensor_view, dst1_ptr, [c16, c64], [c64, c1]
                ).result

                src0_partition = pto.PartitionViewOp(
                    partition_view, src0_view, offsets=[c0, c0], sizes=[c16, c64]
                ).result
                src1_partition = pto.PartitionViewOp(
                    partition_view, src1_view, offsets=[c0, c0], sizes=[c16, c64]
                ).result
                dst0_partition = pto.PartitionViewOp(
                    partition_view, dst0_view, offsets=[c0, c0], sizes=[c16, c64]
                ).result
                dst1_partition = pto.PartitionViewOp(
                    partition_view, dst1_view, offsets=[c0, c0], sizes=[c16, c64]
                ).result

                src0_tile = pto.AllocTileOp(tile_type).result
                src1_tile = pto.AllocTileOp(tile_type).result
                dst0_tile = pto.AllocTileOp(tile_type).result
                dst1_tile = pto.AllocTileOp(tile_type).result

                pto.TLoadOp(None, src0_partition, src0_tile)
                pto.TLoadOp(None, src1_partition, src1_tile)
                pto.TInterleaveOp(src0_tile, src1_tile, dst0_tile, dst1_tile)
                pto.TStoreOp(None, dst0_tile, dst0_partition)
                pto.TStoreOp(None, dst1_tile, dst1_partition)
                func.ReturnOp([])

            module.operation.verify()
            return module


if __name__ == "__main__":
    print(build())
