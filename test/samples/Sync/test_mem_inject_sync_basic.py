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
    Module,
    InsertionPoint,
    F16Type,
    IndexType,
    IntegerType,
)
from mlir.dialects import func, arith, scf, pto


def _idx_const(v: int):
    return arith.ConstantOp(IndexType.get(), v).result


def build():
    with Context() as ctx:
        pto.register_dialect(ctx, load=True)

        with Location.unknown(ctx):
            m = Module.create()

            f16 = F16Type.get(ctx)
            i1 = IntegerType.get_signless(1, ctx)

            ptr_f16 = pto.PtrType.get(f16, ctx)
            tv2 = pto.TensorViewType.get(2, f16, ctx)

            tile_view = pto.PartitionTensorViewType.get([16, 16], f16, ctx)

            vec = pto.AddressSpaceAttr.get(pto.AddressSpace.VEC, ctx)
            bl = pto.BLayoutAttr.get(pto.BLayout.RowMajor, ctx)
            sl = pto.SLayoutAttr.get(pto.SLayout.NoneBox, ctx)
            pd = pto.PadValueAttr.get(pto.PadValue.Null, ctx)
            cfg = pto.TileBufConfigAttr.get(bl, sl, pto.TileConfig.fractalABSize, pd, ctx)
            tile_buf = pto.TileBufType.get([16, 16], f16, vec, [16, 16], cfg, ctx)

            fn_ty = func.FunctionType.get([ptr_f16, ptr_f16, i1], [])
            with InsertionPoint(m.body):
                fn = func.FuncOp("test_basic_pipeline", fn_ty)
                entry = fn.add_entry_block()

            with InsertionPoint(entry):
                inp_ptr, out_ptr, cond = entry.arguments

                c0 = _idx_const(0)
                c1 = _idx_const(1)
                c16 = _idx_const(16)

                tv_in = pto.MakeTensorViewOp(tv2, inp_ptr, [c16, c16], [c16, c1]).result
                tv_out = pto.MakeTensorViewOp(tv2, out_ptr, [c16, c16], [c16, c1]).result

                sv_in = pto.PartitionViewOp(
                    tile_view, tv_in, offsets=[c0, c0], sizes=[c16, c16]
                ).result
                sv_out = pto.PartitionViewOp(
                    tile_view, tv_out, offsets=[c0, c0], sizes=[c16, c16]
                ).result

                ub = pto.AllocTileOp(tile_buf).result

                # Base pipeline:
                #   MTE2(TLOAD) -> V(TADD, conditional) -> MTE3(TSTORE)
                pto.TLoadOp(None, sv_in, ub)

                if_op = scf.IfOp(cond, [], hasElse=False)
                with InsertionPoint(if_op.then_block):
                    pto.TAddOp(ub, ub, ub)
                    scf.YieldOp([])

                pto.TStoreOp(None, ub, sv_out)
                func.ReturnOp([])

            m.operation.verify()
            return m


if __name__ == "__main__":
    print(build())
