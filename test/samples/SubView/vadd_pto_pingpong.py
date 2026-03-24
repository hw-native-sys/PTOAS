# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

from mlir.ir import Context, Location, Module, InsertionPoint
from mlir.dialects import func, arith, pto
from mlir.ir import F32Type, IndexType

def build_pingpong():
    with Context() as ctx:
        pto.register_dialect(ctx, load=True)
        
        with Location.unknown(ctx):
            m = Module.create()
            f32 = F32Type.get(ctx)
            idx = IndexType.get(ctx)

            # ======================================================
            # 1. 极简类型定义 (隐藏了所有复杂性)
            # ======================================================
            
            # 使用指针 + make_tensor_view 包装 GM 数据
            ptr_f32 = pto.PtrType.get(f32, ctx)
            
            # 显式指定 UB 空间、valid shape、config（新版接口不再有简化重载）
            vec = pto.AddressSpaceAttr.get(pto.AddressSpace.VEC, ctx)
            bl = pto.BLayoutAttr.get(pto.BLayout.RowMajor, ctx)
            sl = pto.SLayoutAttr.get(pto.SLayout.NoneBox, ctx)
            pd = pto.PadValueAttr.get(pto.PadValue.Null, ctx)
            fractal_ab_size = pto.TileConfig.fractalABSize
            cfg = pto.TileBufConfigAttr.get(bl, sl, fractal_ab_size, pd, ctx)
            ws_type = pto.TileBufType.get([32, 64], f32, vec, [32, 32], cfg, ctx)

            # ======================================================
            # 2. 逻辑主体
            # ======================================================
            fn_ty = func.FunctionType.get([ptr_f32, ptr_f32], [])
            
            with InsertionPoint(m.body):
                fn = func.FuncOp("test_double_buffer_step", fn_ty)
                entry = fn.add_entry_block()
                
            with InsertionPoint(entry):
                gm_src, gm_dst = entry.arguments
                c0 = arith.ConstantOp(idx, 0).result
                c1 = arith.ConstantOp(idx, 1).result
                c32 = arith.ConstantOp(idx, 32).result
                workspace = pto.AllocTileOp(ws_type).result

                # Wrap GM memrefs as tensor_view and create full partitions
                tv_src = pto.MakeTensorViewOp(pto.TensorViewType.get(2, f32, ctx),
                                             gm_src, [c32, c32], [c32, c1]).result
                tv_dst = pto.MakeTensorViewOp(pto.TensorViewType.get(2, f32, ctx),
                                             gm_dst, [c32, c32], [c32, c1]).result
                sv_src = pto.PartitionViewOp(pto.PartitionTensorViewType.get([32,32], f32, ctx),
                                             tv_src, offsets=[c0, c0], sizes=[c32, c32]).result
                sv_dst = pto.PartitionViewOp(pto.PartitionTensorViewType.get([32,32], f32, ctx),
                                             tv_dst, offsets=[c0, c0], sizes=[c32, c32]).result
                
                # SubView: Ping [0,0], Pong [0,32]
                # 这里不需要指定 Result 类型，C++ 会自动推导
                # SubView sizes must be static (I64ArrayAttr); offsets must be SSA.
                ping = pto.SubViewOp(workspace, [c0, c0], sizes=[32, 32]).result
                pong = pto.SubViewOp(workspace, [c0, c32], sizes=[32, 32]).result

                # DPS: Compute, Prefetch, WriteBack
                pto.TLoadOp(None, sv_src, pong)
                pto.TAddOp(pong, pong, ping)
                pto.TStoreOp(None, ping, sv_dst)
                
                func.ReturnOp([])
            
            m.operation.verify()
            return m

if __name__ == "__main__":
    print(build_pingpong())
