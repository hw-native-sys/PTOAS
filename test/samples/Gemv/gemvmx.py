# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

from mlir.ir import Context, Location, Module, InsertionPoint
from mlir.dialects import func, arith, pto
from mlir.ir import F16Type, F32Type, IndexType


def build():
    with Context() as ctx:
        pto.register_dialect(ctx, load=True)

        with Location.unknown(ctx):
            m = Module.create()

            f16 = F16Type.get(ctx)
            f32 = F32Type.get(ctx)
            ptr_f16 = pto.PtrType.get(f16, ctx)
            ptr_f32 = pto.PtrType.get(f32, ctx)

            # TGEMV_MX family:
            #   TGEMV_MX(dst, a, a_scale, b, b_scale)
            #   TGEMV_MX(dst, c_in, a, a_scale, b, b_scale)
            #   TGEMV_MX(dst, a, a_scale, b, b_scale, bias)
            M = 1
            M_ALIGN = 16
            K = 128
            N = 16

            tv2_f16 = pto.TensorViewType.get(2, f16, ctx)
            tv2_f32 = pto.TensorViewType.get(2, f32, ctx)
            tile_view_a = pto.PartitionTensorViewType.get([M, K], f16, ctx)
            tile_view_b = pto.PartitionTensorViewType.get([K, N], f16, ctx)
            tile_view_as = pto.PartitionTensorViewType.get([M, K], f16, ctx)
            tile_view_bs = pto.PartitionTensorViewType.get([K, N], f16, ctx)
            tile_view_c = pto.PartitionTensorViewType.get([M, N], f32, ctx)
            tile_view_bias = pto.PartitionTensorViewType.get([M, N], f32, ctx)

            mat = pto.AddressSpaceAttr.get(pto.AddressSpace.MAT, ctx)
            left = pto.AddressSpaceAttr.get(pto.AddressSpace.LEFT, ctx)
            right = pto.AddressSpaceAttr.get(pto.AddressSpace.RIGHT, ctx)
            scaling = pto.AddressSpaceAttr.get(pto.AddressSpace.SCALING, ctx)
            acc = pto.AddressSpaceAttr.get(pto.AddressSpace.ACC, ctx)
            bias_space = pto.AddressSpaceAttr.get(pto.AddressSpace.BIAS, ctx)

            pd = pto.PadValueAttr.get(pto.PadValue.Null, ctx)

            cfg_a_mat = pto.TileBufConfigAttr.get(
                pto.BLayoutAttr.get(pto.BLayout.RowMajor, ctx),
                pto.SLayoutAttr.get(pto.SLayout.NoneBox, ctx),
                pto.TileConfig.fractalABSize,
                pd,
                ctx,
            )
            cfg_b_mat = pto.TileBufConfigAttr.get(
                pto.BLayoutAttr.get(pto.BLayout.ColMajor, ctx),
                pto.SLayoutAttr.get(pto.SLayout.RowMajor, ctx),
                pto.TileConfig.fractalABSize,
                pd,
                ctx,
            )
            cfg_left = pto.TileBufConfigAttr.get(
                pto.BLayoutAttr.get(pto.BLayout.RowMajor, ctx),
                pto.SLayoutAttr.get(pto.SLayout.RowMajor, ctx),
                pto.TileConfig.fractalABSize,
                pd,
                ctx,
            )
            cfg_right = pto.TileBufConfigAttr.get(
                pto.BLayoutAttr.get(pto.BLayout.RowMajor, ctx),
                pto.SLayoutAttr.get(pto.SLayout.ColMajor, ctx),
                pto.TileConfig.fractalABSize,
                pd,
                ctx,
            )
            cfg_scaling = pto.TileBufConfigAttr.get(
                pto.BLayoutAttr.get(pto.BLayout.RowMajor, ctx),
                pto.SLayoutAttr.get(pto.SLayout.RowMajor, ctx),
                pto.TileConfig.fractalABSize,
                pd,
                ctx,
            )
            cfg_acc = pto.TileBufConfigAttr.get(
                pto.BLayoutAttr.get(pto.BLayout.ColMajor, ctx),
                pto.SLayoutAttr.get(pto.SLayout.RowMajor, ctx),
                pto.TileConfig.fractalCSize,
                pd,
                ctx,
            )
            cfg_bias = pto.TileBufConfigAttr.get(
                pto.BLayoutAttr.get(pto.BLayout.RowMajor, ctx),
                pto.SLayoutAttr.get(pto.SLayout.NoneBox, ctx),
                pto.TileConfig.fractalABSize,
                pd,
                ctx,
            )

            tile_buf_a_mat = pto.TileBufType.get([M, K], f16, mat, [M, K], cfg_a_mat, ctx)
            tile_buf_b_mat = pto.TileBufType.get([K, N], f16, mat, [K, N], cfg_b_mat, ctx)
            tile_buf_as_mat = pto.TileBufType.get([M, K], f16, mat, [M, K], cfg_a_mat, ctx)
            tile_buf_bs_mat = pto.TileBufType.get([K, N], f16, mat, [K, N], cfg_b_mat, ctx)
            tile_buf_bias_mat = pto.TileBufType.get([M, N], f32, mat, [M, N], cfg_bias, ctx)

            tile_buf_a = pto.TileBufType.get([M, K], f16, left, [M, K], cfg_left, ctx)
            tile_buf_b = pto.TileBufType.get([K, N], f16, right, [K, N], cfg_right, ctx)
            tile_buf_as = pto.TileBufType.get([M, K], f16, scaling, [M, K], cfg_scaling, ctx)
            tile_buf_bs = pto.TileBufType.get([K, N], f16, scaling, [K, N], cfg_scaling, ctx)
            tile_buf_bias = pto.TileBufType.get([M, N], f32, bias_space, [M, N], cfg_bias, ctx)
            tile_buf_c = pto.TileBufType.get([M_ALIGN, N], f32, acc, [M, N], cfg_acc, ctx)

            fn_ty = func.FunctionType.get(
                [
                    ptr_f16,  # a
                    ptr_f16,  # b
                    ptr_f16,  # a_scale
                    ptr_f16,  # b_scale
                    ptr_f32,  # bias (for mx.bias)
                    ptr_f32,  # out_mx
                    ptr_f32,  # out_mx_acc
                    ptr_f32,  # out_mx_bias
                ],
                [],
            )
            with InsertionPoint(m.body):
                fn = func.FuncOp("gemvmx_kernel", fn_ty)
                entry = fn.add_entry_block()

            with InsertionPoint(entry):
                c0 = arith.ConstantOp(IndexType.get(ctx), 0).result
                c1 = arith.ConstantOp(IndexType.get(ctx), 1).result
                cM = arith.ConstantOp(IndexType.get(ctx), M).result
                cK = arith.ConstantOp(IndexType.get(ctx), K).result
                cN = arith.ConstantOp(IndexType.get(ctx), N).result

                (
                    arg_a,
                    arg_b,
                    arg_as,
                    arg_bs,
                    arg_bias,
                    arg_out_mx,
                    arg_out_mx_acc,
                    arg_out_mx_bias,
                ) = entry.arguments

                tv_a = pto.MakeTensorViewOp(tv2_f16, arg_a, [cM, cK], [cK, c1]).result
                tv_b = pto.MakeTensorViewOp(tv2_f16, arg_b, [cK, cN], [cN, c1]).result
                tv_as = pto.MakeTensorViewOp(tv2_f16, arg_as, [cM, cK], [cK, c1]).result
                tv_bs = pto.MakeTensorViewOp(tv2_f16, arg_bs, [cK, cN], [cN, c1]).result
                tv_bias = pto.MakeTensorViewOp(tv2_f32, arg_bias, [cM, cN], [cN, c1]).result
                tv_out_mx = pto.MakeTensorViewOp(tv2_f32, arg_out_mx, [cM, cN], [cN, c1]).result
                tv_out_mx_acc = pto.MakeTensorViewOp(tv2_f32, arg_out_mx_acc, [cM, cN], [cN, c1]).result
                tv_out_mx_bias = pto.MakeTensorViewOp(tv2_f32, arg_out_mx_bias, [cM, cN], [cN, c1]).result

                sv_a = pto.PartitionViewOp(tile_view_a, tv_a, offsets=[c0, c0], sizes=[cM, cK]).result
                sv_b = pto.PartitionViewOp(tile_view_b, tv_b, offsets=[c0, c0], sizes=[cK, cN]).result
                sv_as = pto.PartitionViewOp(tile_view_as, tv_as, offsets=[c0, c0], sizes=[cM, cK]).result
                sv_bs = pto.PartitionViewOp(tile_view_bs, tv_bs, offsets=[c0, c0], sizes=[cK, cN]).result
                sv_bias = pto.PartitionViewOp(tile_view_bias, tv_bias, offsets=[c0, c0], sizes=[cM, cN]).result
                sv_out_mx = pto.PartitionViewOp(tile_view_c, tv_out_mx, offsets=[c0, c0], sizes=[cM, cN]).result
                sv_out_mx_acc = pto.PartitionViewOp(tile_view_c, tv_out_mx_acc, offsets=[c0, c0], sizes=[cM, cN]).result
                sv_out_mx_bias = pto.PartitionViewOp(tile_view_c, tv_out_mx_bias, offsets=[c0, c0], sizes=[cM, cN]).result

                a_mat = pto.AllocTileOp(tile_buf_a_mat).result
                b_mat = pto.AllocTileOp(tile_buf_b_mat).result
                as_mat = pto.AllocTileOp(tile_buf_as_mat).result
                bs_mat = pto.AllocTileOp(tile_buf_bs_mat).result
                bias_mat = pto.AllocTileOp(tile_buf_bias_mat).result
                a_tile = pto.AllocTileOp(tile_buf_a).result
                b_tile = pto.AllocTileOp(tile_buf_b).result
                as_tile = pto.AllocTileOp(tile_buf_as).result
                bs_tile = pto.AllocTileOp(tile_buf_bs).result
                bias_tile = pto.AllocTileOp(tile_buf_bias).result
                c_mx_tile = pto.AllocTileOp(tile_buf_c).result
                c_mx_acc_tile = pto.AllocTileOp(tile_buf_c).result
                c_mx_bias_tile = pto.AllocTileOp(tile_buf_c).result

                pto.TLoadOp(None, sv_a, a_mat)
                pto.TLoadOp(None, sv_b, b_mat)
                pto.TLoadOp(None, sv_as, as_mat)
                pto.TLoadOp(None, sv_bs, bs_mat)
                pto.TLoadOp(None, sv_bias, bias_mat)

                pto.TExtractOp(a_mat, c0, c0, a_tile)
                pto.TMovOp(None, b_mat, b_tile)
                pto.TMovOp(None, as_mat, as_tile)
                pto.TMovOp(None, bs_mat, bs_tile)
                pto.TMovOp(None, bias_mat, bias_tile)

                pto.TGemvMxOp(None, a_tile, as_tile, b_tile, bs_tile, c_mx_tile)
                pto.TGemvMxAccOp(None, c_mx_tile, a_tile, as_tile, b_tile, bs_tile, c_mx_acc_tile)
                pto.TGemvMxBiasOp(None, a_tile, as_tile, b_tile, bs_tile, bias_tile, c_mx_bias_tile)

                pto.TStoreOp(None, c_mx_tile, sv_out_mx)
                pto.TStoreOp(None, c_mx_acc_tile, sv_out_mx_acc)
                pto.TStoreOp(None, c_mx_bias_tile, sv_out_mx_bias)

                func.ReturnOp([])

            m.operation.verify()
            return m


if __name__ == "__main__":
    print(build())
