# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

from mlir.ir import (
    Context,
    Location,
    InsertionPoint,
    IndexType,
    BF16Type,
    StringAttr,
)
from mlir.dialects import func, arith, pto, builtin


def _idx_const(v: int):
    return arith.ConstantOp(IndexType.get(), v).result


def build():
    with Context() as ctx, Location.unknown():
        pto.register_dialect(ctx, load=True)

        module = builtin.ModuleOp()
        module.attributes["pto.device-spec"] = StringAttr.get("Ascend910B1")

        bf16 = BF16Type.get()
        ptr_bf16 = pto.PtrType.get(bf16)

        tv2 = pto.TensorViewType.get(2, bf16)
        tile_view = pto.PartitionTensorViewType.get([16, 128], bf16)

        mat = pto.AddressSpaceAttr.get(pto.AddressSpace.MAT)
        cfg = pto.TileBufConfigAttr.get(
            pto.BLayoutAttr.get(pto.BLayout.ColMajor),
            pto.SLayoutAttr.get(pto.SLayout.RowMajor),
            pto.TileConfig.fractalABSize,
            pto.PadValueAttr.get(pto.PadValue.Null),
        )
        tile_buf = pto.TileBufType.get([16, 128], bf16, mat, [16, 128], cfg)

        fn_ty = func.FunctionType.get([ptr_bf16], [])
        with InsertionPoint(module.body):
            fn = func.FuncOp("bf16_tile", fn_ty)
            entry = fn.add_entry_block()

        with InsertionPoint(entry):
            (a_ptr,) = entry.arguments

            c0 = _idx_const(0)
            c1 = _idx_const(1)
            c16 = _idx_const(16)
            c128 = _idx_const(128)

            tv = pto.MakeTensorViewOp(tv2, a_ptr, [c16, c128], [c128, c1]).result
            sv = pto.PartitionViewOp(tile_view, tv, offsets=[c0, c0], sizes=[c16, c128]).result

            tile = pto.AllocTileOp(tile_buf).result
            pto.TLoadOp(None, sv, tile)

            func.ReturnOp([])

        module.operation.verify()
        return module


if __name__ == "__main__":
    print(build())
