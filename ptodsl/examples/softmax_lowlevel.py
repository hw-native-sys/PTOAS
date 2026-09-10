# coding=utf-8
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""
Low-level builder for the online softmax kernel.

Reconstructs the IR in
  test/tilelang_st/npu/a5/src/st/testcase/softmax/softmax.pto
using raw MLIR Python binding calls, with no additional abstraction layer.
"""

from ptoas.mlir.ir import (
    Attribute,
    Context,
    F32Type,
    InsertionPoint,
    IntegerType,
    IndexType,
    Location,
    Module,
    ShapedType,
    StringAttr,
    Type,
    UnitAttr,
)
from ptoas.mlir.dialects import arith, func, pto, scf


def build():
    with Context() as ctx:
        pto.register_dialect(ctx, load=True)

        with Location.unknown():
            # ── Types ────────────────────────────────────────────────────────
            i1  = IntegerType.get_signless(1)
            i32 = IntegerType.get_signless(32)
            i64 = IntegerType.get_signless(64)
            idx = IndexType.get()
            f32 = F32Type.get()

            # Address-space attributes used in pointer and tile types
            _gm  = pto.AddressSpaceAttr.get(pto.AddressSpace.GM)   # gm = global memory
            _ub  = pto.AddressSpaceAttr.get(pto.AddressSpace.VEC)  # vec = UB (unified buffer)
            # Sentinel value for a dynamic (unknown) dimension
            _dyn = ShapedType.get_dynamic_size()

            # Pointer types built with PtrType.get
            ptr_gm  = pto.PtrType.get(f32, memory_space=_gm)  # !pto.ptr<f32, gm>
            ptr_ub  = pto.PtrType.get(f32, memory_space=_ub)  # !pto.ptr<f32, ub>

            # Tensor-view types built with TensorViewType / PartitionTensorViewType
            tv5d   = pto.TensorViewType.get(5, f32)                    # !pto.tensor_view<?x?x?x?x?xf32>
            ptv5d  = pto.PartitionTensorViewType.get([_dyn] * 5, f32)  # !pto.partition_tensor_view<?x?x?x?x?xf32>

            # Tile-buffer config attributes
            _reduce_cfg = pto.TileBufConfigAttr.get(
                pto.BLayoutAttr.get(pto.BLayout.RowMajor),
                pto.SLayoutAttr.get(pto.SLayout.NoneBox),
                512, pto.PadValueAttr.get(pto.PadValue.Null),
            )
            _row_cfg = pto.TileBufConfigAttr.get(
                pto.BLayoutAttr.get(pto.BLayout.RowMajor),
                pto.SLayoutAttr.get(pto.SLayout.NoneBox),
                512, pto.PadValueAttr.get(pto.PadValue.Null),
            )
            # !pto.tile_buf<vec, 8x1xf32, valid=?x1, blayout=row_major>
            tile_col  = pto.TileBufType.get([8, 1],   f32, _ub, [-1,  1], _reduce_cfg)
            # !pto.tile_buf<vec, 8x128xf32, valid=?x?>
            tile_wide = pto.TileBufType.get([8, 128], f32, _ub, [-1, -1], _row_cfg)

            # VReg and Mask types have no Python-binding constructors yet;
            # Type.parse is the only available path for these two.
            vreg     = Type.parse("!pto.vreg<64xf32>")
            mask_b32 = Type.parse("!pto.mask<b32>")

            # ── Flat single module ────────────────────────────────────────
            m = Module.create()
            m.operation.attributes["pto.target_arch"] = StringAttr.get("a5")
            # FunctionKernelKindAttr has no binding; Attribute.parse is the only path.
            m.operation.attributes["pto.kernel_kind"] = Attribute.parse(
                "#pto.kernel_kind<vector>"
            )

            fn_ty = func.FunctionType.get([ptr_gm] * 7 + [i32, i32], [])
            with InsertionPoint(m.body):
                fn = func.FuncOp("online_softmax_update_kernel_2d", fn_ty)
                fn.attributes["pto.aicore"] = UnitAttr.get()
                entry = fn.add_entry_block()

            with InsertionPoint(entry):
                a0, a1, a2, a3, a4, a5, a6, arg7, arg8 = entry.arguments

                # ── Index constants ───────────────────────────────────────
                c0   = arith.ConstantOp(idx, 0).result
                c1   = arith.ConstantOp(idx, 1).result
                c8   = arith.ConstantOp(idx, 8).result
                c64  = arith.ConstantOp(idx, 64).result
                c128 = arith.ConstantOp(idx, 128).result

                # ── i64 constants ─────────────────────────────────────────
                c0_i64    = arith.ConstantOp(i64,     0).result
                c1_i64    = arith.ConstantOp(i64,     1).result
                c8_i64    = arith.ConstantOp(i64,     8).result
                c16_i64   = arith.ConstantOp(i64,    16).result
                c32_i64   = arith.ConstantOp(i64,    32).result
                c64_i64   = arith.ConstantOp(i64,    64).result
                c128_i64  = arith.ConstantOp(i64,   128).result
                c256_i64  = arith.ConstantOp(i64,   256).result
                c512_i64  = arith.ConstantOp(i64,   512).result
                c8448_i64  = arith.ConstantOp(i64,  8448).result
                c16640_i64 = arith.ConstantOp(i64, 16640).result
                c16768_i64 = arith.ConstantOp(i64, 16768).result
                c16896_i64 = arith.ConstantOp(i64, 16896).result

                # ── i32 constants ─────────────────────────────────────────
                c1_i32  = arith.ConstantOp(i32,  1).result
                c8_i32  = arith.ConstantOp(i32,  8).result
                c64_i32 = arith.ConstantOp(i32, 64).result
                c0_i32  = arith.ConstantOp(i32,  0).result

                # ── Block / row computation ───────────────────────────────
                block         = pto.GetBlockIdxOp().result            # i64
                block_idx     = arith.IndexCastOp(idx, block).result
                row_base      = arith.MulIOp(block_idx, c8).result
                block_rows_i32= arith.IndexCastOp(i32, c8).result
                row_base_i32  = arith.IndexCastOp(i32, row_base).result
                remaining_rows= arith.SubIOp(arg8, row_base_i32).result
                has_rows      = arith.CmpIOp(arith.CmpIPredicate.sgt,
                                             remaining_rows, c0_i32).result
                too_many_rows = arith.CmpIOp(arith.CmpIPredicate.sgt,
                                             remaining_rows, c8_i32).result
                row_count_i32 = arith.SelectOp(too_many_rows, c8_i32,
                                               remaining_rows).result
                row_count     = arith.IndexCastOp(idx, row_count_i32).result
                seq           = arith.IndexCastOp(idx, arg7).result
                rows          = arith.IndexCastOp(idx, arg8).result
                rows_x_128    = arith.MulIOp(rows, c128).result

                # ── scf.if %has_rows ──────────────────────────────────────
                if_rows = scf.IfOp(has_rows)
                with InsertionPoint(if_rows.then_block):

                    # ── Tensor views ──────────────────────────────────────
                    s1 = [rows, rows, rows, c1, rows]
                    s128 = [rows_x_128, rows_x_128, rows_x_128, c128, c1]
                    sh1 = [c1, c1, c1, rows, c1]
                    sh128 = [c1, c1, c1, rows, c128]

                    oldmax_view = pto.MakeTensorViewOp(tv5d, a0, sh1,  s1).result
                    oldsum_view = pto.MakeTensorViewOp(tv5d, a1, sh1,  s1).result
                    qk_view     = pto.MakeTensorViewOp(tv5d, a2, sh128, s128).result
                    newmax_view = pto.MakeTensorViewOp(tv5d, a3, sh1,  s1).result
                    newsum_view = pto.MakeTensorViewOp(tv5d, a4, sh1,  s1).result
                    expmax_view = pto.MakeTensorViewOp(tv5d, a5, sh1,  s1).result
                    out_view    = pto.MakeTensorViewOp(tv5d, a6, sh128, s128).result

                    # ── Partition views ───────────────────────────────────
                    off5 = [c0, c0, c0, row_base, c0]
                    sz1  = [c1, c1, c1, row_count, c1]
                    szs  = [c1, c1, c1, row_count, seq]

                    oldmax_part = pto.PartitionViewOp(ptv5d, oldmax_view, off5, sz1).result
                    oldsum_part = pto.PartitionViewOp(ptv5d, oldsum_view, off5, sz1).result
                    qk_part     = pto.PartitionViewOp(ptv5d, qk_view,     off5, szs).result
                    newmax_part = pto.PartitionViewOp(ptv5d, newmax_view, off5, sz1).result
                    newsum_part = pto.PartitionViewOp(ptv5d, newsum_view, off5, sz1).result
                    expmax_part = pto.PartitionViewOp(ptv5d, expmax_view, off5, sz1).result
                    out_part    = pto.PartitionViewOp(ptv5d, out_view,    off5, szs).result

                    # ── Tile allocation ───────────────────────────────────
                    oldmax_tile = pto.AllocTileOp(tile_col,  addr=c0_i64,    valid_row=row_count).result
                    oldsum_tile = pto.AllocTileOp(tile_col,  addr=c128_i64,  valid_row=row_count).result
                    qk_tile     = pto.AllocTileOp(tile_wide, addr=c256_i64,  valid_row=row_count, valid_col=seq).result
                    out_tile    = pto.AllocTileOp(tile_wide, addr=c8448_i64, valid_row=row_count, valid_col=seq).result
                    newmax_tile = pto.AllocTileOp(tile_col,  addr=c16640_i64, valid_row=row_count).result
                    newsum_tile = pto.AllocTileOp(tile_col,  addr=c16768_i64, valid_row=row_count).result
                    expmax_tile = pto.AllocTileOp(tile_col,  addr=c16896_i64, valid_row=row_count).result

                    # ── Tile loads ────────────────────────────────────────
                    pto.TLoadOp(None, oldmax_part, oldmax_tile)
                    pto.TLoadOp(None, oldsum_part, oldsum_tile)
                    pto.TLoadOp(None, qk_part,     qk_tile)

                    # ── Sync before vecscope ──────────────────────────────
                    pto.set_flag("PIPE_MTE2", "PIPE_V", pto.EVENT_ID0)
                    pto.wait_flag("PIPE_MTE2", "PIPE_V", pto.EVENT_ID0)

                    # ── pto.vecscope ──────────────────────────────────────
                    vs_op    = pto.VecScopeOp()
                    vs_block = vs_op.body.blocks.append()
                    with InsertionPoint(vs_block):

                        # Materialise UB pointers from tile handles
                        ub_oldmax = pto.TileBufAddrOp(ptr_ub, oldmax_tile).result
                        ub_oldsum = pto.TileBufAddrOp(ptr_ub, oldsum_tile).result
                        ub_qk     = pto.TileBufAddrOp(ptr_ub, qk_tile).result
                        ub_out    = pto.TileBufAddrOp(ptr_ub, out_tile).result
                        ub_newmax = pto.TileBufAddrOp(ptr_ub, newmax_tile).result
                        ub_newsum = pto.TileBufAddrOp(ptr_ub, newsum_tile).result
                        ub_expmax = pto.TileBufAddrOp(ptr_ub, expmax_tile).result

                        active   = pto.PsetB32Op(mask_b32, "PAT_ALL").result
                        plt1     = pto.PltB32Op(mask_b32, i32, c1_i32)
                        one_mask = plt1.mask

                        # ── for row in [0, row_count) ─────────────────────
                        row_for = scf.ForOp(c0, row_count, c1)
                        with InsertionPoint(row_for.body):
                            row    = row_for.induction_variable
                            row_qk = arith.MulIOp(row, c128).result

                            oldmax_bc = pto.VldsOp(vreg, ub_oldmax, row,
                                                   dist="BRC_B32").result
                            oldsum_bc = pto.VldsOp(vreg, ub_oldsum, row,
                                                   dist="BRC_B32").result

                            # ── for chunk in [0,128,64) with iter_args ────
                            chunk_for = scf.ForOp(c0, c128, c64,
                                                  [oldmax_bc, oldsum_bc])
                            with InsertionPoint(chunk_for.body):
                                chunk       = chunk_for.induction_variable
                                running_max = chunk_for.inner_iter_args[0]
                                running_sum = chunk_for.inner_iter_args[1]

                                chunk_i32     = arith.IndexCastOp(i32, chunk).result
                                remaining_cols= arith.SubIOp(arg7, chunk_i32).result
                                has_chunk     = arith.CmpIOp(
                                    arith.CmpIPredicate.sgt,
                                    remaining_cols, c0_i32).result

                                # ── if has_chunk -> (vreg, vreg) ──────────
                                c_if = scf.IfOp(has_chunk, [vreg, vreg],
                                                hasElse=True)
                                with InsertionPoint(c_if.then_block):
                                    cplt        = pto.PltB32Op(mask_b32, i32,
                                                               remaining_cols)
                                    chunk_mask  = cplt.mask
                                    chunk_base  = arith.AddIOp(row_qk,
                                                               chunk).result
                                    vec         = pto.VldsOp(vreg, ub_qk,
                                                             chunk_base).result
                                    chunk_max   = pto.VcmaxOp(vreg, vec,
                                                              chunk_mask).result
                                    chunk_max_bc= pto.VdupOp(vreg, chunk_max,
                                                             active,
                                                             position="LOWEST").result
                                    merged_max  = pto.VmaxOp(vreg, running_max,
                                                             chunk_max_bc,
                                                             active).result
                                    scaled_run  = pto.VexpdifOp(vreg,
                                                                running_max,
                                                                merged_max,
                                                                active,
                                                                "ODD").result
                                    run_sum_sc  = pto.VmulOp(vreg, scaled_run,
                                                             running_sum,
                                                             active).result
                                    chunk_exp   = pto.VexpdifOp(vreg, vec,
                                                                merged_max,
                                                                chunk_mask,
                                                                "ODD").result
                                    chunk_sum   = pto.VcaddOp(vreg, chunk_exp,
                                                              chunk_mask).result
                                    chunk_sum_bc= pto.VdupOp(vreg, chunk_sum,
                                                             active,
                                                             position="LOWEST").result
                                    merged_sum  = pto.VaddOp(vreg, run_sum_sc,
                                                             chunk_sum_bc,
                                                             active).result
                                    scf.YieldOp([merged_max, merged_sum])
                                with InsertionPoint(c_if.else_block):
                                    scf.YieldOp([running_max, running_sum])

                                next_max, next_sum = c_if.results
                                scf.YieldOp([next_max, next_sum])

                            final_max, final_sum = chunk_for.results

                            # ── Post-loop: compute expmax ─────────────────
                            raw_expmax    = pto.VexpdifOp(vreg, oldmax_bc,
                                                          final_max, active,
                                                          "ODD").result
                            scaled_oldsum = pto.VmulOp(vreg, raw_expmax,
                                                       oldsum_bc,
                                                       active).result
                            expmax        = pto.VdivOp(vreg, scaled_oldsum,
                                                       final_sum,
                                                       active).result

                            pto.VstsOp(final_max, ub_newmax, row, one_mask,
                                       dist="1PT_B32")
                            pto.VstsOp(final_sum, ub_newsum, row, one_mask,
                                       dist="1PT_B32")
                            pto.VstsOp(expmax,    ub_expmax, row, one_mask,
                                       dist="1PT_B32")

                            # ── Output normalisation loop ─────────────────
                            out_for = scf.ForOp(c0, c128, c64)
                            with InsertionPoint(out_for.body):
                                chunk2        = out_for.induction_variable
                                ci32_2        = arith.IndexCastOp(i32,
                                                                   chunk2).result
                                rem2          = arith.SubIOp(arg7, ci32_2).result
                                has_chunk2    = arith.CmpIOp(
                                    arith.CmpIPredicate.sgt,
                                    rem2, c0_i32).result

                                o_if = scf.IfOp(has_chunk2)
                                with InsertionPoint(o_if.then_block):
                                    oplt       = pto.PltB32Op(mask_b32, i32,
                                                              rem2)
                                    cmask2     = oplt.mask
                                    cbase2     = arith.AddIOp(row_qk,
                                                              chunk2).result
                                    vec2       = pto.VldsOp(vreg, ub_qk,
                                                            cbase2).result
                                    exp2       = pto.VexpdifOp(vreg, vec2,
                                                               final_max,
                                                               cmask2,
                                                               "ODD").result
                                    out2       = pto.VdivOp(vreg, exp2,
                                                            final_sum,
                                                            cmask2).result
                                    pto.VstsOp(out2, ub_out, cbase2, cmask2)
                                    scf.YieldOp([])

                                scf.YieldOp([])   # out_for body

                            scf.YieldOp([])   # row_for body

                    # ── Sync after vecscope ───────────────────────────────
                    pto.set_flag("PIPE_V", "PIPE_MTE3", pto.EVENT_ID0)
                    pto.wait_flag("PIPE_V", "PIPE_MTE3", pto.EVENT_ID0)

                    # ── Tile stores ───────────────────────────────────────
                    pto.TStoreOp(None, newmax_tile, newmax_part)
                    pto.TStoreOp(None, newsum_tile, newsum_part)
                    pto.TStoreOp(None, expmax_tile, expmax_part)
                    pto.TStoreOp(None, out_tile,    out_part)

                    scf.YieldOp([])   # if_rows then_block

                # ── Barrier and return ────────────────────────────────────
                pto.BarrierOp(pto.PipeAttr.get(pto.PIPE.PIPE_ALL))
                func.ReturnOp([])

            m.operation.verify()
            return m


if __name__ == "__main__":
    print(build())
