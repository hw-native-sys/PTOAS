from mlir.ir import Context, Location, Module, InsertionPoint
from mlir.dialects import func, arith, pto
from mlir.ir import F32Type, IndexType, IntegerType


def build():
    with Context() as ctx:
        pto.register_dialect(ctx, load=True)

        with Location.unknown(ctx):
            m = Module.create()

            f32 = F32Type.get(ctx)
            ptr_f32 = pto.PtrType.get(f32, ctx)

            # Mergesort on a 1x1024 list for TMrgSort.
            #
            # NOTE: A5 pto-isa requires that Vec TLOAD's tile ValidCol matches
            # GlobalTensor's staticShape[4]. Build a 1x1024 tensor_view so the
            # generated GlobalTensor column dimension is 1024.
            tv2_f32 = pto.TensorViewType.get([1, 1024], f32, ctx)
            part_view_1x1024 = pto.PartitionTensorViewType.get([1, 1024], f32, ctx)
            vec = pto.AddressSpaceAttr.get(pto.AddressSpace.VEC, ctx)
            bl = pto.BLayoutAttr.get(pto.BLayout.RowMajor, ctx)
            sl = pto.SLayoutAttr.get(pto.SLayout.NoneBox, ctx)
            pd = pto.PadValueAttr.get(pto.PadValue.Null, ctx)

            fractal_ab_size = pto.TileConfig.fractalABSize
            cfg = pto.TileBufConfigAttr.get(bl, sl, fractal_ab_size, pd, ctx)
            # NOTE: pto.tmrgsort (format1) expects a rank-2 tile with rows == 1.
            tile_buf_1x1024 = pto.TileBufType.get([1, 1024], f32, vec, [1, 1024], cfg, ctx)

            fn_ty = func.FunctionType.get([ptr_f32, ptr_f32], [])
            with InsertionPoint(m.body):
                fn = func.FuncOp("vec_add_scalar_kernel_2d", fn_ty)
                entry = fn.add_entry_block()

            with InsertionPoint(entry):
                # constants
                c0 = arith.ConstantOp(IndexType.get(ctx), 0).result
                c1 = arith.ConstantOp(IndexType.get(ctx), 1).result
                c1024 = arith.ConstantOp(IndexType.get(ctx), 1024).result
                # blockLen for tmrgsort format1: ins(src, blockLen) outs(dst), must be integer (e.g. i32)
                i32 = IntegerType.get_signless(32, ctx)
                c64_i32 = arith.ConstantOp(i32, 64).result

                arg0, arg1 = entry.arguments

                # Flatten as a 1x1024 tensor.
                tv0 = pto.MakeTensorViewOp(tv2_f32, arg0, [c1, c1024], [c1024, c1]).result
                tv1 = pto.MakeTensorViewOp(tv2_f32, arg1, [c1, c1024], [c1024, c1]).result

                sv0 = pto.PartitionViewOp(
                    part_view_1x1024, tv0, offsets=[c0, c0], sizes=[c1, c1024]
                ).result

                # Format1: ins(%src, %blockLen : tile_buf, i32) outs(%dst : tile_buf)
                tb0 = pto.AllocTileOp(tile_buf_1x1024).result
                tb1 = pto.AllocTileOp(tile_buf_1x1024).result

                pto.TLoadOp(None, sv0, tb0)  # result=None
                pto.TMrgSortOp(srcs=[tb0], dsts=[tb1], blockLen=c64_i32)

                sv1 = pto.PartitionViewOp(
                    part_view_1x1024, tv1, offsets=[c0, c0], sizes=[c1, c1024]
                ).result

                pto.TStoreOp(None, tb1, sv1)

                func.ReturnOp([])

            m.operation.verify()

            return m


if __name__ == "__main__":
    print(build())
