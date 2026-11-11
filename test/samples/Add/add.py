# PyPTO tile-op baseline: pto.tadd golden sample, f32 32x32 (elementwise add)
# Raw-op builder + auto pipeline (samples convention; PyPTO lowers straight to pto.* IR).

from ptoas.mlir.ir import Context, Location, Module, InsertionPoint, UnitAttr
from ptoas.mlir.dialects import func, arith, pto
from ptoas.mlir.ir import F32Type, IndexType


def build():
    with Context() as ctx:
        pto.register_dialect(ctx, load=True)

        with Location.unknown(ctx):
            m = Module.create()

            f32 = F32Type.get(ctx)
            ptr_f32 = pto.PtrType.get(f32, ctx)
            tv2_f32 = pto.TensorViewType.get(2, f32, ctx)
            tile_view_32 = pto.PartitionTensorViewType.get([32, 32], f32, ctx)
            vec = pto.AddressSpaceAttr.get(pto.AddressSpace.VEC, ctx)
            bl = pto.BLayoutAttr.get(pto.BLayout.RowMajor, ctx)
            sl = pto.SLayoutAttr.get(pto.SLayout.NoneBox, ctx)
            pd = pto.PadValueAttr.get(pto.PadValue.Null, ctx)
            cfg = pto.TileBufConfigAttr.get(bl, sl, pto.TileConfig.fractalABSize, pd, ctx)
            tile_buf_32 = pto.TileBufType.get([32, 32], f32, vec, [32, 32], cfg, ctx)

            fn_ty = func.FunctionType.get([ptr_f32, ptr_f32, ptr_f32], [])
            with InsertionPoint(m.body):
                fn = func.FuncOp("tadd_kernel_2d", fn_ty)
                fn.operation.attributes["pto.entry"] = UnitAttr.get(ctx)
                entry = fn.add_entry_block()

            with InsertionPoint(entry):
                arg0, arg1, arg2 = entry.arguments
                c0 = arith.ConstantOp(IndexType.get(ctx), 0).result
                c1 = arith.ConstantOp(IndexType.get(ctx), 1).result
                c32 = arith.ConstantOp(IndexType.get(ctx), 32).result
                tv0 = pto.MakeTensorViewOp(tv2_f32, arg0, [c32, c32], [c32, c1]).result
                tv1 = pto.MakeTensorViewOp(tv2_f32, arg1, [c32, c32], [c32, c1]).result
                tv2_ = pto.MakeTensorViewOp(tv2_f32, arg2, [c32, c32], [c32, c1]).result
                sv0 = pto.PartitionViewOp(tile_view_32, tv0, offsets=[c0, c0], sizes=[c32, c32]).result
                sv1 = pto.PartitionViewOp(tile_view_32, tv1, offsets=[c0, c0], sizes=[c32, c32]).result
                sv2 = pto.PartitionViewOp(tile_view_32, tv2_, offsets=[c0, c0], sizes=[c32, c32]).result
                tb0 = pto.AllocTileOp(tile_buf_32).result
                tb1 = pto.AllocTileOp(tile_buf_32).result
                tb2 = pto.AllocTileOp(tile_buf_32).result
                pto.TLoadOp(None, sv0, tb0)
                pto.TLoadOp(None, sv1, tb1)
                pto.TAddOp(tb0, tb1, tb2)
                pto.TStoreOp(None, tb2, sv2)

                func.ReturnOp([])

            m.operation.verify()
            return m


if __name__ == "__main__":
    print(build())
