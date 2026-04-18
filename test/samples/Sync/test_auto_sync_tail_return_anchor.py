from mlir.ir import (
    Context,
    F32Type,
    InsertionPoint,
    IndexType,
    Location,
    Module,
)
from mlir.dialects import arith, func, pto


def _idx_const(v: int):
    return arith.ConstantOp(IndexType.get(), v).result


def build():
    with Context() as ctx:
        pto.register_dialect(ctx, load=True)

        with Location.unknown(ctx):
            module = Module.create()

            f32 = F32Type.get(ctx)
            ptr_f32 = pto.PtrType.get(f32, ctx)
            tensor_view = pto.TensorViewType.get(2, f32, ctx)
            part_view = pto.PartitionTensorViewType.get([32, 32], f32, ctx)

            vec = pto.AddressSpaceAttr.get(pto.AddressSpace.VEC, ctx)
            blayout = pto.BLayoutAttr.get(pto.BLayout.RowMajor, ctx)
            slayout = pto.SLayoutAttr.get(pto.SLayout.NoneBox, ctx)
            pad = pto.PadValueAttr.get(pto.PadValue.Null, ctx)
            config = pto.TileBufConfigAttr.get(
                blayout, slayout, pto.TileConfig.fractalABSize, pad, ctx
            )
            tile_buf = pto.TileBufType.get([32, 32], f32, vec, [32, 32], config, ctx)

            fn_ty = func.FunctionType.get([ptr_f32, ptr_f32, ptr_f32], [])
            with InsertionPoint(module.body):
                fn = func.FuncOp("test_auto_sync_tail_return_anchor", fn_ty)
                entry = fn.add_entry_block()

            with InsertionPoint(entry):
                inp, aux, out = entry.arguments

                c0 = _idx_const(0)
                c1 = _idx_const(1)
                c32 = _idx_const(32)

                tv_in = pto.MakeTensorViewOp(
                    tensor_view, inp, [c32, c32], [c32, c1]
                ).result
                part_in = pto.PartitionViewOp(
                    part_view, tv_in, offsets=[c0, c0], sizes=[c32, c32]
                ).result
                tile = pto.AllocTileOp(tile_buf).result

                # The last SyncIR pipe op is intentionally not the last op in
                # the function. The auto tail clean still has to be anchored at
                # the actual function return.
                pto.TLoadOp(None, part_in, tile)

                tv_aux = pto.MakeTensorViewOp(
                    tensor_view, aux, [c32, c32], [c32, c1]
                ).result
                part_aux = pto.PartitionViewOp(
                    part_view, tv_aux, offsets=[c0, c0], sizes=[c32, c32]
                ).result
                pto.TLoadOp(None, part_aux, tile)

                # Scalar pointer ops lower to real C++ statements but are not
                # part of SyncIR pipe analysis. This reproduces the old bug
                # where auto tail clean was emitted before trailing non-pipe
                # code.
                scalar = pto.load_scalar(f32, out, c0)
                pto.store_scalar(out, c0, scalar)

                func.ReturnOp([])

            module.operation.verify()
            return module


if __name__ == "__main__":
    print(build())
