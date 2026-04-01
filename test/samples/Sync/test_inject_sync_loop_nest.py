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
)
from mlir.dialects import func, arith, scf, pto
from mlir.dialects.arith import CmpIPredicate


def _idx_const(v: int):
    return arith.ConstantOp(IndexType.get(), v).result


def build():
    with Context() as ctx:
        pto.register_dialect(ctx, load=True)

        with Location.unknown(ctx):
            m = Module.create()

            f16 = F16Type.get(ctx)

            ptr_f16 = pto.PtrType.get(f16, ctx)
            tv2 = pto.TensorViewType.get(2, f16, ctx)

            tile_view = pto.PartitionTensorViewType.get([16, 16], f16, ctx)

            vec = pto.AddressSpaceAttr.get(pto.AddressSpace.VEC, ctx)
            bl = pto.BLayoutAttr.get(pto.BLayout.RowMajor, ctx)
            sl = pto.SLayoutAttr.get(pto.SLayout.NoneBox, ctx)
            pd = pto.PadValueAttr.get(pto.PadValue.Null, ctx)
            cfg = pto.TileBufConfigAttr.get(bl, sl, pto.TileConfig.fractalABSize, pd, ctx)
            tile_buf = pto.TileBufType.get([16, 16], f16, vec, [16, 16], cfg, ctx)

            fn_ty = func.FunctionType.get([ptr_f16, ptr_f16], [])
            with InsertionPoint(m.body):
                fn = func.FuncOp("test_loop_nest_sync", fn_ty)
                entry = fn.add_entry_block()

            with InsertionPoint(entry):
                inp_ptr, out_ptr = entry.arguments

                c0 = _idx_const(0)
                c1 = _idx_const(1)
                c2 = _idx_const(2)
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

                outer = scf.ForOp(c0, c2, c1, [])
                with InsertionPoint(outer.body):
                    inner = scf.ForOp(c0, c2, c1, [])
                    with InsertionPoint(inner.body):
                        j = inner.induction_variable

                        pto.TLoadOp(None, sv_in, ub)

                        is_j0 = arith.CmpIOp(CmpIPredicate.eq, j, c0).result
                        if_op = scf.IfOp(is_j0, [], hasElse=False)
                        with InsertionPoint(if_op.then_block):
                            pto.TAddOp(ub, ub, ub)
                            scf.YieldOp([])

                        pto.TStoreOp(None, ub, sv_out)
                        scf.YieldOp([])

                    scf.YieldOp([])

                func.ReturnOp([])

            m.operation.verify()
            return m


if __name__ == "__main__":
    print(build())
