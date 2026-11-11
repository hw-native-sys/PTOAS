# PyPTO tile-op baseline: pto.trandom golden sample, i32 4x256, rounds=10 (default).
# Raw-op builder + auto pipeline (samples convention; PyPTO lowers straight to pto.* IR).
# Key/counter words are baked constants so the kernel is deterministic; the golden
# reimplements the A5 Philox-style generator in numpy (ported from
# test/tilelib-st/a5/trandom/case.py).

from ptoas.mlir.ir import Context, Location, Module, InsertionPoint, UnitAttr
from ptoas.mlir.dialects import func, arith, pto
import os

from ptoas.mlir.ir import IndexType, IntegerType, StringAttr

ROWS, COLS = 4, 256
KEY0, KEY1 = 123456789, 987654321
COUNTER = (1, 2, 3, 4)


def build():
    with Context() as ctx:
        pto.register_dialect(ctx, load=True)

        with Location.unknown(ctx):
            m = Module.create()
            m.operation.attributes["pto.target_arch"] = StringAttr.get(
                os.environ.get("PTOAS_SAMPLE_ARCH", "a5"))

            i32 = IntegerType.get_signless(32, ctx)
            ptr_i32 = pto.PtrType.get(i32, ctx)
            tv2_i32 = pto.TensorViewType.get(2, i32, ctx)
            tile_view = pto.PartitionTensorViewType.get([ROWS, COLS], i32, ctx)
            vec = pto.AddressSpaceAttr.get(pto.AddressSpace.VEC, ctx)
            bl = pto.BLayoutAttr.get(pto.BLayout.RowMajor, ctx)
            sl = pto.SLayoutAttr.get(pto.SLayout.NoneBox, ctx)
            pd = pto.PadValueAttr.get(pto.PadValue.Null, ctx)
            cfg = pto.TileBufConfigAttr.get(bl, sl, pto.TileConfig.fractalABSize, pd, ctx)
            tile_buf = pto.TileBufType.get([ROWS, COLS], i32, vec, [ROWS, COLS], cfg, ctx)

            fn_ty = func.FunctionType.get([ptr_i32], [])
            with InsertionPoint(m.body):
                fn = func.FuncOp("trandom_kernel", fn_ty)
                fn.operation.attributes["pto.entry"] = UnitAttr.get(ctx)
                entry = fn.add_entry_block()

            with InsertionPoint(entry):
                arg0 = entry.arguments[0]
                c0 = arith.ConstantOp(IndexType.get(ctx), 0).result
                c1 = arith.ConstantOp(IndexType.get(ctx), 1).result
                cR = arith.ConstantOp(IndexType.get(ctx), ROWS).result
                cC = arith.ConstantOp(IndexType.get(ctx), COLS).result

                k0 = arith.ConstantOp(i32, KEY0).result
                k1 = arith.ConstantOp(i32, KEY1).result
                ctr0 = arith.ConstantOp(i32, COUNTER[0]).result
                ctr1 = arith.ConstantOp(i32, COUNTER[1]).result
                ctr2 = arith.ConstantOp(i32, COUNTER[2]).result
                ctr3 = arith.ConstantOp(i32, COUNTER[3]).result

                tv0 = pto.MakeTensorViewOp(tv2_i32, arg0, [cR, cC], [cC, c1]).result
                sv0 = pto.PartitionViewOp(tile_view, tv0, offsets=[c0, c0], sizes=[cR, cC]).result
                dst = pto.AllocTileOp(tile_buf).result

                pto.TRandomOp(k0, k1, ctr0, ctr1, ctr2, ctr3, dst)
                pto.TStoreOp(None, dst, sv0)

                func.ReturnOp([])

            m.operation.verify()
            return m


if __name__ == "__main__":
    print(build())
