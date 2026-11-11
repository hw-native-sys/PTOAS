# PyPTO tile-op baseline: pto.tmov golden sample.
# One kernel exercises two dtypes (f32 and i32, 16x64 each): load -> tmov(vec->vec) -> store.
# Raw-op builder + auto pipeline (samples convention; PyPTO lowers straight to pto.* IR).

from ptoas.mlir.ir import Context, Location, Module, InsertionPoint, UnitAttr
from ptoas.mlir.dialects import func, arith, pto
from ptoas.mlir.ir import F32Type, IndexType, IntegerType

ROWS, COLS = 16, 64


def build():
    with Context() as ctx:
        pto.register_dialect(ctx, load=True)

        with Location.unknown(ctx):
            m = Module.create()

            f32 = F32Type.get(ctx)
            i32 = IntegerType.get_signless(32, ctx)
            ptr_f32 = pto.PtrType.get(f32, ctx)
            ptr_i32 = pto.PtrType.get(i32, ctx)
            tv2_f32 = pto.TensorViewType.get(2, f32, ctx)
            tv2_i32 = pto.TensorViewType.get(2, i32, ctx)
            tile_view_f32 = pto.PartitionTensorViewType.get([ROWS, COLS], f32, ctx)
            tile_view_i32 = pto.PartitionTensorViewType.get([ROWS, COLS], i32, ctx)
            vec = pto.AddressSpaceAttr.get(pto.AddressSpace.VEC, ctx)
            bl = pto.BLayoutAttr.get(pto.BLayout.RowMajor, ctx)
            sl = pto.SLayoutAttr.get(pto.SLayout.NoneBox, ctx)
            pd = pto.PadValueAttr.get(pto.PadValue.Null, ctx)
            cfg = pto.TileBufConfigAttr.get(bl, sl, pto.TileConfig.fractalABSize, pd, ctx)
            tile_buf_f32 = pto.TileBufType.get([ROWS, COLS], f32, vec, [ROWS, COLS], cfg, ctx)
            tile_buf_i32 = pto.TileBufType.get([ROWS, COLS], i32, vec, [ROWS, COLS], cfg, ctx)

            fn_ty = func.FunctionType.get([ptr_f32, ptr_f32, ptr_i32, ptr_i32], [])
            with InsertionPoint(m.body):
                fn = func.FuncOp("tmov_multi_dtype_kernel", fn_ty)
                fn.operation.attributes["pto.entry"] = UnitAttr.get(ctx)
                entry = fn.add_entry_block()

            with InsertionPoint(entry):
                a_f32, b_f32, a_i32, b_i32 = entry.arguments
                c0 = arith.ConstantOp(IndexType.get(ctx), 0).result
                c1 = arith.ConstantOp(IndexType.get(ctx), 1).result
                cR = arith.ConstantOp(IndexType.get(ctx), ROWS).result
                cC = arith.ConstantOp(IndexType.get(ctx), COLS).result

                tv0 = pto.MakeTensorViewOp(tv2_f32, a_f32, [cR, cC], [cC, c1]).result
                tv1 = pto.MakeTensorViewOp(tv2_f32, b_f32, [cR, cC], [cC, c1]).result
                tv2_ = pto.MakeTensorViewOp(tv2_i32, a_i32, [cR, cC], [cC, c1]).result
                tv3 = pto.MakeTensorViewOp(tv2_i32, b_i32, [cR, cC], [cC, c1]).result

                sv0 = pto.PartitionViewOp(tile_view_f32, tv0, offsets=[c0, c0], sizes=[cR, cC]).result
                sv1 = pto.PartitionViewOp(tile_view_f32, tv1, offsets=[c0, c0], sizes=[cR, cC]).result
                sv2 = pto.PartitionViewOp(tile_view_i32, tv2_, offsets=[c0, c0], sizes=[cR, cC]).result
                sv3 = pto.PartitionViewOp(tile_view_i32, tv3, offsets=[c0, c0], sizes=[cR, cC]).result

                tb0 = pto.AllocTileOp(tile_buf_f32).result
                tb1 = pto.AllocTileOp(tile_buf_f32).result
                tb2 = pto.AllocTileOp(tile_buf_i32).result
                tb3 = pto.AllocTileOp(tile_buf_i32).result

                pto.TLoadOp(None, sv0, tb0)
                pto.TLoadOp(None, sv2, tb2)

                pto.TMovOp(None, tb0, tb1)
                pto.TMovOp(None, tb2, tb3)

                pto.TStoreOp(None, tb1, sv1)
                pto.TStoreOp(None, tb3, sv3)

                func.ReturnOp([])

            m.operation.verify()
            return m


if __name__ == "__main__":
    print(build())
