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
    Type,
    UnitAttr,
)

# Small board-friendly geometry (fmap [1,1,4,6,8] NC1HWC0, 4D FRACTAL_Z
# weight [1,1,16,8] i.e. 16 output channels, filter 2x2, stride 1x1, pad
# (top,bottom,left,right)=(1,0,1,0)). Sizes chosen so the compact-style
# left/right physical tiles fit the level2 planner capacities.
FMAP_N, FMAP_C1, FMAP_H, FMAP_W, FMAP_C0 = 1, 2, 4, 6, 8
WG_C1, WG_FH, WG_FW, WG_N, WG_C0 = 2, 2, 2, 16, 8
FILTER_H, FILTER_W = 2, 2
STRIDE_H, STRIDE_W = 1, 1
PAD_TOP, PAD_BOTTOM, PAD_LEFT, PAD_RIGHT = 1, 0, 1, 0

HEIGHT_OUT = (FMAP_H + PAD_TOP + PAD_BOTTOM - 1 * (FILTER_H - 1) - 1) // STRIDE_H + 1
WIDTH_OUT = (FMAP_W + PAD_LEFT + PAD_RIGHT - 1 * (FILTER_W - 1) - 1) // STRIDE_W + 1
VALID_M = FMAP_N * HEIGHT_OUT * WIDTH_OUT
VALID_N = WG_N
VALID_K = WG_C1 * WG_FH * WG_FW * WG_C0
BASE_K = 64
M = -(-VALID_M // 16) * 16          # CeilAlign(validM, 16)
N = -(-VALID_N // 8) * 8            # CeilAlign(validN, 8)  (blockAlign = 8 for f32)
K = -(-VALID_K // BASE_K) * BASE_K
ITER = K // BASE_K


def build():
    with Context() as ctx:
        pto.register_dialect(ctx, load=True)
        with Location.unknown(ctx):
            module = Module.create()
            arch = os.environ.get("PTOAS_SAMPLE_ARCH", "a5")
            module.operation.attributes["pto.target_arch"] = StringAttr.get(arch)
            module.operation.attributes["pto.kernel_kind"] = StringAttr.get("cube")

            f32 = F32Type.get(ctx)
            i32 = IntegerType.get_signless(32, ctx)
            i64 = IntegerType.get_signless(64, ctx)

            ptr_f32 = pto.PtrType.get(f32, ctx)

            tv5_conv = Type.parse(
                f"!pto.tensor_view<{FMAP_N}x{FMAP_C1}x{FMAP_H}x{FMAP_W}x{FMAP_C0}xf32, "
                f"#pto.conv_layout<nc1hwc0>>",
                context=ctx,
            )
            src_ptv = Type.parse(
                f"!pto.partition_tensor_view<{FMAP_N}x{FMAP_C1}x{FMAP_H}x{FMAP_W}x{FMAP_C0}xf32, "
                f"#pto.conv_layout<nc1hwc0>>",
                context=ctx,
            )
            # 4D FRACTAL_Z weight view: [1, D3, D2, D1, D0]
            wgt_tv5 = Type.parse(
                f"!pto.tensor_view<{WG_C1}x{WG_FH}x{WG_FW}x{WG_N}x{WG_C0}xf32, "
                f"#pto.conv_layout<fractal_z>>",
                context=ctx,
            )
            wgt_ptv = Type.parse(
                f"!pto.partition_tensor_view<{WG_C1}x{WG_FH}x{WG_FW}x{WG_N}x{WG_C0}xf32, "
                f"#pto.conv_layout<fractal_z>>",
                context=ctx,
            )
            out_tv = Type.parse(f"!pto.tensor_view<{VALID_M}x{VALID_N}xf32>", context=ctx)
            out_ptv = Type.parse(f"!pto.partition_tensor_view<{VALID_M}x{VALID_N}xf32>", context=ctx)

            mat = pto.AddressSpaceAttr.get(pto.AddressSpace.MAT, ctx)
            layout_in = pto.ConvLayoutAttr.get(pto.ConvLayout.NC1HWC0, ctx)
            layout_wgt = pto.ConvLayoutAttr.get(pto.ConvLayout.FRACTAL_Z, ctx)

            conv_cfg = pto.ConvTileConfigAttr.get(
                IntegerAttr.get(i32, FMAP_H),
                IntegerAttr.get(i32, FMAP_W),
                [PAD_LEFT, PAD_RIGHT, PAD_TOP, PAD_BOTTOM],
                IntegerAttr.get(i32, FILTER_H),
                IntegerAttr.get(i32, FILTER_W),
                IntegerAttr.get(i32, 1),   # dilationH
                IntegerAttr.get(i32, 1),   # dilationW
                IntegerAttr.get(i32, STRIDE_H),
                IntegerAttr.get(i32, STRIDE_W),
                IntegerAttr.get(i32, 0),   # padValue
                IntegerAttr.get(i32, FMAP_C1 * FMAP_C0),  # channelSize
                IntegerAttr.get(i32, 0),   # repeatStride
                IntegerAttr.get(i32, 1),   # repeatTime
                IntegerAttr.get(i32, 0),   # repeatMode
                IntegerAttr.get(i32, -(-M // 16)),  # dstStride = CeilDivision(M,16)
                IntegerAttr.get(i32, 0),   # dstMposition
                BoolAttr.get(False, ctx),  # transpose
                context=ctx,
            )

            buf_a = FMAP_N * FMAP_C1 * FMAP_H * FMAP_W * FMAP_C0
            src_tile_ty = pto.ConvTileBufType.get(
                [FMAP_N, FMAP_C1, FMAP_H, FMAP_W, FMAP_C0],
                f32,
                IntegerAttr.get(i64, buf_a),
                mat,
                layout_in,
                context=ctx,
            )

            buf_b = WG_C1 * WG_FH * WG_FW * WG_N * WG_C0
            wgt_tile_ty = pto.ConvTileBufType.get(
                [WG_C1, WG_FH, WG_FW, WG_N, WG_C0],
                f32,
                IntegerAttr.get(i64, buf_b),
                mat,
                layout_wgt,
                context=ctx,
            )

            # Left/Right/Acc tiles mirrored from the ST: compact left/right
            # (2*physical), Acc with fractalC=1024.
            left = pto.AddressSpaceAttr.get(pto.AddressSpace.LEFT, ctx)
            right = pto.AddressSpaceAttr.get(pto.AddressSpace.RIGHT, ctx)
            acc = pto.AddressSpaceAttr.get(pto.AddressSpace.ACC, ctx)
            col_major = pto.BLayoutAttr.get(pto.BLayout.ColMajor, ctx)
            row_major = pto.BLayoutAttr.get(pto.BLayout.RowMajor, ctx)
            row_box = pto.SLayoutAttr.get(pto.SLayout.RowMajor, ctx)
            col_box = pto.SLayoutAttr.get(pto.SLayout.ColMajor, ctx)
            null_pad = pto.PadValueAttr.get(pto.PadValue.Null, ctx)
            left_cfg = pto.TileBufConfigAttr.get(col_major, row_box, pto.TileConfig.fractalABSize, null_pad, ctx)
            right_cfg = pto.TileBufConfigAttr.get(row_major, col_box, pto.TileConfig.fractalABSize, null_pad, ctx)
            acc_cfg = pto.TileBufConfigAttr.get(col_major, row_box, pto.TileConfig.fractalCSize, null_pad, ctx)

            left_ty = pto.TileBufType.get([2 * M, 2 * BASE_K], f32, left, [VALID_M, BASE_K], left_cfg, ctx)
            right_ty = pto.TileBufType.get([2 * BASE_K, 2 * N], f32, right, [BASE_K, VALID_N], right_cfg, ctx)
            acc_ty = pto.TileBufType.get([M, N], f32, acc, [VALID_M, VALID_N], acc_cfg, ctx)

            fn_ty = func.FunctionType.get([ptr_f32, ptr_f32, ptr_f32], [])
            with InsertionPoint(module.body):
                fn = func.FuncOp("timg2col_runtime_kernel", fn_ty)
                fn.operation.attributes["pto.entry"] = UnitAttr.get(ctx)
                entry = fn.add_entry_block()

            with InsertionPoint(entry):
                c0 = arith.ConstantOp(IndexType.get(ctx), 0).result
                c1 = arith.ConstantOp(IndexType.get(ctx), 1).result
                cN = arith.ConstantOp(IndexType.get(ctx), FMAP_N).result
                cC1 = arith.ConstantOp(IndexType.get(ctx), FMAP_C1).result
                cH = arith.ConstantOp(IndexType.get(ctx), FMAP_H).result
                cW = arith.ConstantOp(IndexType.get(ctx), FMAP_W).result
                cC0 = arith.ConstantOp(IndexType.get(ctx), FMAP_C0).result
                cW1 = arith.ConstantOp(IndexType.get(ctx), WG_C1).result
                cWH = arith.ConstantOp(IndexType.get(ctx), WG_FH).result
                cWW = arith.ConstantOp(IndexType.get(ctx), WG_FW).result
                cWN = arith.ConstantOp(IndexType.get(ctx), WG_N).result
                cW0 = arith.ConstantOp(IndexType.get(ctx), WG_C0).result
                cVM = arith.ConstantOp(IndexType.get(ctx), VALID_M).result
                cVN = arith.ConstantOp(IndexType.get(ctx), VALID_N).result
                cBK = arith.ConstantOp(IndexType.get(ctx), BASE_K).result
                fmap_ptr, wgt_ptr, out_ptr = entry.arguments

                # strides for fmap: [C1*H*W*C0, H*W*C0, W*C0, C0, 1]
                sA = FMAP_C1 * FMAP_H * FMAP_W * FMAP_C0
                sB = FMAP_H * FMAP_W * FMAP_C0
                sC = FMAP_W * FMAP_C0
                cSA = arith.ConstantOp(IndexType.get(ctx), sA).result
                cSB = arith.ConstantOp(IndexType.get(ctx), sB).result
                cSC = arith.ConstantOp(IndexType.get(ctx), sC).result
                src_view = pto.MakeTensorViewOp(
                    tv5_conv, fmap_ptr,
                    [cN, cC1, cH, cW, cC0], [cSA, cSB, cSC, cC0, c1]).result
                src_part = pto.PartitionViewOp(
                    src_ptv, src_view,
                    offsets=[c0, c0, c0, c0, c0],
                    sizes=[cN, cC1, cH, cW, cC0]).result

                # strides for 5D fractal_z weight [C1,FH,FW,N,C0]:
                # [FH*FW*N*C0, FW*N*C0, N*C0, C0, 1]
                tA = WG_FH * WG_FW * WG_N * WG_C0
                tB = WG_FW * WG_N * WG_C0
                tC = WG_N * WG_C0
                tD = WG_C0
                cTA = arith.ConstantOp(IndexType.get(ctx), tA).result
                cTB = arith.ConstantOp(IndexType.get(ctx), tB).result
                cTC = arith.ConstantOp(IndexType.get(ctx), tC).result
                cTD = arith.ConstantOp(IndexType.get(ctx), tD).result
                wgt_view = pto.MakeTensorViewOp(
                    wgt_tv5, wgt_ptr,
                    [cW1, cWH, cWW, cWN, cW0], [cTA, cTB, cTC, cTD, c1]).result
                wgt_part = pto.PartitionViewOp(
                    wgt_ptv, wgt_view,
                    offsets=[c0, c0, c0, c0, c0],
                    sizes=[cW1, cWH, cWW, cWN, cW0]).result

                out_view = pto.MakeTensorViewOp(
                    out_tv, out_ptr, [cVM, cVN], [cVN, c1]).result
                out_part = pto.PartitionViewOp(
                    out_ptv, out_view, offsets=[c0, c0], sizes=[cVM, cVN]).result

                fmatrix_mode = pto.FmatrixModeAttr.get(
                    pto.FmatrixMode.FMATRIX_B_MANUAL, ctx)

                cube_section = pto.SectionCubeOp()
                with InsertionPoint(cube_section.body.blocks.append()):
                    src_tile = pto.AllocTileOp(src_tile_ty, config=conv_cfg).result
                    wgt_tile = pto.AllocTileOp(wgt_tile_ty).result
                    left_tile = pto.AllocTileOp(left_ty).result
                    right_tile = pto.AllocTileOp(right_ty).result
                    acc_tile = pto.AllocTileOp(acc_ty).result

                    pto.TLoadOp(None, src_part, src_tile)
                    pto.TLoadOp(None, wgt_part, wgt_tile)
                    pto.SetImg2colRptOp(src_tile, fmatrixMode=fmatrix_mode)
                    pto.SetFmatrixOp(src_tile, fmatrixMode=fmatrix_mode)

                    for i in range(ITER):
                        off = arith.ConstantOp(IndexType.get(ctx), i * BASE_K).result
                        pto.TImg2colOp(
                            left_tile, src_tile, posM=c0, posK=off,
                            fmatrixMode=fmatrix_mode)
                        pto.TExtractOp(wgt_tile, off, c0, right_tile)
                        if i == 0:
                            pto.TMatmulOp(None, left_tile, right_tile, acc_tile)
                        else:
                            pto.TMatmulAccOp(None, acc_tile, left_tile, right_tile, acc_tile)

                    pto.TStoreOp(None, acc_tile, out_part)
                func.ReturnOp([])

            module.operation.verify()
            return module


if __name__ == "__main__":
    print(build())
