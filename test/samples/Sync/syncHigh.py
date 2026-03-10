#!/usr/bin/env python3
from mlir.ir import Context, Location, Module, InsertionPoint
from mlir.dialects import func, arith, pto
from mlir.ir import F32Type, IndexType


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

            fractal_ab_size = pto.TileConfig.fractalABSize
            cfg = pto.TileBufConfigAttr.get(bl, sl, fractal_ab_size, pd, ctx)
            tile_buf_32 = pto.TileBufType.get([32, 32], f32, vec, [32, 32], cfg, ctx)

            # A real kernel that manually inserts set_flag/wait_flag.
            #
            # NOTE(A5): `set_flag/wait_flag` only accept a subset of PIPE enums on A5.
            # Keep this sample in the supported set by using MTE2/V/MTE3 only.
            fn_ty = func.FunctionType.get([ptr_f32, ptr_f32], [])
            with InsertionPoint(m.body):
                fn = func.FuncOp("run_sync_high", fn_ty)
                entry = fn.add_entry_block()

            with InsertionPoint(entry):
                c0 = arith.ConstantOp(IndexType.get(ctx), 0).result
                c1 = arith.ConstantOp(IndexType.get(ctx), 1).result
                c32 = arith.ConstantOp(IndexType.get(ctx), 32).result
                scale = arith.ConstantOp(f32, 3.14).result
                pipe_mte2 = pto.PipeAttr.get(pto.PIPE.PIPE_MTE2, ctx)
                pipe_v = pto.PipeAttr.get(pto.PIPE.PIPE_V, ctx)
                pipe_mte3 = pto.PipeAttr.get(pto.PIPE.PIPE_MTE3, ctx)
                evt0 = pto.EventAttr.get(pto.EVENT.EVENT_ID0, ctx)
                evt1 = pto.EventAttr.get(pto.EVENT.EVENT_ID1, ctx)

                arg0, arg1 = entry.arguments

                tv0 = pto.MakeTensorViewOp(tv2_f32, arg0, [c32, c32], [c32, c1]).result
                tv1 = pto.MakeTensorViewOp(tv2_f32, arg1, [c32, c32], [c32, c1]).result

                sv0 = pto.PartitionViewOp(tile_view_32, tv0, offsets=[c0, c0], sizes=[c32, c32]).result
                sv1 = pto.PartitionViewOp(tile_view_32, tv1, offsets=[c0, c0], sizes=[c32, c32]).result

                tb0 = pto.AllocTileOp(tile_buf_32).result
                tb1 = pto.AllocTileOp(tile_buf_32).result

                # Load (PIPE_MTE2)
                pto.TLoadOp(None, sv0, tb0)
                # MTE2 -> V sync before vector compute.
                pto.set_flag(pipe_mte2, pipe_v, evt0)
                pto.wait_flag(pipe_mte2, pipe_v, evt0)

                # Compute (PIPE_V)
                pto.TAddSOp(tb0, scale, tb1)
                # V -> MTE3 sync before store.
                pto.set_flag(pipe_v, pipe_mte3, evt1)
                pto.wait_flag(pipe_v, pipe_mte3, evt1)

                # Store (PIPE_MTE3)
                pto.TStoreOp(None, tb1, sv1)

                func.ReturnOp([])

            m.operation.verify()
            return m


if __name__ == "__main__":
    print(build())
