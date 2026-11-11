# PyPTO tile-op baseline: pto.thistogram golden sample (A5-only op).
# One kernel runs both byte variants (byte=1 default and byte=0) on the same
# ui16 source / ui8 index tiles, storing into two ui32 256-bin histogram tiles.
# Raw-op builder + auto pipeline (samples convention).

import os

from ptoas.mlir.ir import Context, Location, Module, InsertionPoint, UnitAttr, StringAttr
from ptoas.mlir.dialects import func, arith, pto
from ptoas.mlir.ir import IndexType, IntegerType

ROWS, COLS, BINS = 32, 32, 256


def build():
    with Context() as ctx:
        pto.register_dialect(ctx, load=True)

        with Location.unknown(ctx):
            m = Module.create()
            m.operation.attributes["pto.target_arch"] = StringAttr.get(
                os.environ.get("PTOAS_SAMPLE_ARCH", "a5"))

            ui16 = IntegerType.get_unsigned(16, ctx)
            ui8 = IntegerType.get_unsigned(8, ctx)
            ui32 = IntegerType.get_unsigned(32, ctx)
            ptr_ui16 = pto.PtrType.get(ui16, ctx)
            ptr_ui8 = pto.PtrType.get(ui8, ctx)
            ptr_ui32 = pto.PtrType.get(ui32, ctx)
            tv2_ui16 = pto.TensorViewType.get(2, ui16, ctx)
            tv2_ui8 = pto.TensorViewType.get(2, ui8, ctx)
            tv2_ui32 = pto.TensorViewType.get(2, ui32, ctx)
            src_view = pto.PartitionTensorViewType.get([ROWS, COLS], ui16, ctx)
            idx_view = pto.PartitionTensorViewType.get([ROWS, 1], ui8, ctx)
            dst_view = pto.PartitionTensorViewType.get([ROWS, BINS], ui32, ctx)
            vec = pto.AddressSpaceAttr.get(pto.AddressSpace.VEC, ctx)
            bl_row = pto.BLayoutAttr.get(pto.BLayout.RowMajor, ctx)
            bl_col = pto.BLayoutAttr.get(pto.BLayout.ColMajor, ctx)
            sl = pto.SLayoutAttr.get(pto.SLayout.NoneBox, ctx)
            pd = pto.PadValueAttr.get(pto.PadValue.Null, ctx)
            cfg = pto.TileBufConfigAttr.get(bl_row, sl, pto.TileConfig.fractalABSize, pd, ctx)
            cfg_idx = pto.TileBufConfigAttr.get(bl_col, sl, pto.TileConfig.fractalABSize, pd, ctx)
            src_buf_ty = pto.TileBufType.get([ROWS, COLS], ui16, vec, [ROWS, COLS], cfg, ctx)
            idx_buf_ty = pto.TileBufType.get([ROWS, 1], ui8, vec, [ROWS, 1], cfg_idx, ctx)
            dst_buf_ty = pto.TileBufType.get([ROWS, BINS], ui32, vec, [ROWS, BINS], cfg, ctx)

            fn_ty = func.FunctionType.get([ptr_ui16, ptr_ui8, ptr_ui32, ptr_ui32], [])
            with InsertionPoint(m.body):
                fn = func.FuncOp("thistogram_kernel", fn_ty)
                fn.operation.attributes["pto.entry"] = UnitAttr.get(ctx)
                entry = fn.add_entry_block()

            with InsertionPoint(entry):
                src_ptr, idx_ptr, dst0_ptr, dst1_ptr = entry.arguments
                c0 = arith.ConstantOp(IndexType.get(ctx), 0).result
                c1 = arith.ConstantOp(IndexType.get(ctx), 1).result
                cR = arith.ConstantOp(IndexType.get(ctx), ROWS).result
                cC = arith.ConstantOp(IndexType.get(ctx), COLS).result
                cB = arith.ConstantOp(IndexType.get(ctx), BINS).result

                tv_src = pto.MakeTensorViewOp(tv2_ui16, src_ptr, [cR, cC], [cC, c1]).result
                tv_idx = pto.MakeTensorViewOp(tv2_ui8, idx_ptr, [cR, c1], [c1, cR]).result
                tv_d0 = pto.MakeTensorViewOp(tv2_ui32, dst0_ptr, [cR, cB], [cB, c1]).result
                tv_d1 = pto.MakeTensorViewOp(tv2_ui32, dst1_ptr, [cR, cB], [cB, c1]).result
                sv_src = pto.PartitionViewOp(src_view, tv_src, offsets=[c0, c0], sizes=[cR, cC]).result
                sv_idx = pto.PartitionViewOp(idx_view, tv_idx, offsets=[c0, c0], sizes=[cR, c1]).result
                sv_d0 = pto.PartitionViewOp(dst_view, tv_d0, offsets=[c0, c0], sizes=[cR, cB]).result
                sv_d1 = pto.PartitionViewOp(dst_view, tv_d1, offsets=[c0, c0], sizes=[cR, cB]).result

                tb_src = pto.AllocTileOp(src_buf_ty).result
                tb_idx = pto.AllocTileOp(idx_buf_ty).result
                tb_d0 = pto.AllocTileOp(dst_buf_ty).result
                tb_d1 = pto.AllocTileOp(dst_buf_ty).result

                pto.TLoadOp(None, sv_src, tb_src)
                pto.TLoadOp(None, sv_idx, tb_idx)

                # byte=1 (default, high byte) -> dst0; byte=0 (low byte) -> dst1.
                pto.THistogramOp(tb_src, tb_idx, tb_d0, byte=1)
                pto.THistogramOp(tb_src, tb_idx, tb_d1, byte=0)

                pto.TStoreOp(None, tb_d0, sv_d0)
                pto.TStoreOp(None, tb_d1, sv_d1)

                func.ReturnOp([])

            m.operation.verify()
            return m


if __name__ == "__main__":
    print(build())
