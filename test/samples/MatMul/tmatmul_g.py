# PyPTO tile-op baseline: pto.tmatmul (f32 32x32x32, level2 auto) golden sample (level2 auto: tload mat -> tmov l0 -> tmatmul -> tstore).
from ptoas.mlir.ir import Context, Location, Module, InsertionPoint, UnitAttr
from ptoas.mlir.dialects import func, arith, pto
from ptoas.mlir.ir import F32Type, IndexType

M = K = N = 32


def build():
    with Context() as ctx:
        pto.register_dialect(ctx, load=True)
        with Location.unknown(ctx):
            m = Module.create()
            f32 = F32Type.get(ctx)
            ptr_f32 = pto.PtrType.get(f32, ctx)
            tv2 = pto.TensorViewType.get(2, f32, ctx)
            mat = pto.AddressSpaceAttr.get(pto.AddressSpace.MAT, ctx)
            left = pto.AddressSpaceAttr.get(pto.AddressSpace.LEFT, ctx)
            right = pto.AddressSpaceAttr.get(pto.AddressSpace.RIGHT, ctx)
            acc = pto.AddressSpaceAttr.get(pto.AddressSpace.ACC, ctx)
            vec = pto.AddressSpaceAttr.get(pto.AddressSpace.VEC, ctx)
            bl_col = pto.BLayoutAttr.get(pto.BLayout.ColMajor, ctx)
            sl_row = pto.SLayoutAttr.get(pto.SLayout.RowMajor, ctx)
            sl_col = pto.SLayoutAttr.get(pto.SLayout.ColMajor, ctx)
            bl_row = pto.BLayoutAttr.get(pto.BLayout.RowMajor, ctx)
            sl_none = pto.SLayoutAttr.get(pto.SLayout.NoneBox, ctx)
            pd = pto.PadValueAttr.get(pto.PadValue.Null, ctx)
            fab = pto.TileConfig.fractalABSize
            cfg_mat = pto.TileBufConfigAttr.get(bl_col, sl_row, fab, pd, ctx)
            cfg_mat_b = pto.TileBufConfigAttr.get(bl_row, sl_row, fab, pd, ctx)
            cfg_left = pto.TileBufConfigAttr.get(bl_col, sl_row, fab, pd, ctx)
            cfg_right = pto.TileBufConfigAttr.get(bl_row, sl_col, fab, pd, ctx)
            cfg_acc = pto.TileBufConfigAttr.get(bl_col, sl_row, 1024, pd, ctx)
            cfg_vec = pto.TileBufConfigAttr.get(bl_row, sl_none, fab, pd, ctx)
            tb_a_mat = pto.TileBufType.get([M, K], f32, mat, [M, K], cfg_mat, ctx)
            tb_b_mat = pto.TileBufType.get([K, N], f32, mat, [K, N], cfg_mat, ctx)
            tb_a = pto.TileBufType.get([M, K], f32, left, [M, K], cfg_left, ctx)
            tb_b = pto.TileBufType.get([K, N], f32, right, [K, N], cfg_right, ctx)
            tb_c = pto.TileBufType.get([M, N], f32, acc, [M, N], cfg_acc, ctx)
            tb_out = pto.TileBufType.get([M, N], f32, vec, [M, N], cfg_vec, ctx)
            pv_a = pto.PartitionTensorViewType.get([M, K], f32, ctx)
            pv_b = pto.PartitionTensorViewType.get([K, N], f32, ctx)
            pv_c = pto.PartitionTensorViewType.get([M, N], f32, ctx)

            sig = [ptr_f32, ptr_f32, ptr_f32]

            fn_ty = func.FunctionType.get(sig, [])
            with InsertionPoint(m.body):
                fn = func.FuncOp("tmatmul_golden_kernel", fn_ty)
                fn.operation.attributes["pto.entry"] = UnitAttr.get(ctx)
                entry = fn.add_entry_block()

            with InsertionPoint(entry):
                a_ptr, b_ptr, o_ptr = entry.arguments
                c0 = arith.ConstantOp(IndexType.get(ctx), 0).result
                c1 = arith.ConstantOp(IndexType.get(ctx), 1).result
                cM = arith.ConstantOp(IndexType.get(ctx), M).result
                cK = arith.ConstantOp(IndexType.get(ctx), K).result
                cN = arith.ConstantOp(IndexType.get(ctx), N).result

                tv_a = pto.MakeTensorViewOp(tv2, a_ptr, [cM, cK], [cK, c1]).result
                tv_b = pto.MakeTensorViewOp(tv2, b_ptr, [cK, cN], [cN, c1]).result
                tv_o = pto.MakeTensorViewOp(tv2, o_ptr, [cM, cN], [cN, c1]).result

                sv_a = pto.PartitionViewOp(pv_a, tv_a, offsets=[c0, c0], sizes=[cM, cK]).result
                sv_b = pto.PartitionViewOp(pv_b, tv_b, offsets=[c0, c0], sizes=[cK, cN]).result
                sv_o = pto.PartitionViewOp(pv_c, tv_o, offsets=[c0, c0], sizes=[cM, cN]).result

                a_mat = pto.AllocTileOp(tb_a_mat).result
                b_mat = pto.AllocTileOp(tb_b_mat).result
                a_l = pto.AllocTileOp(tb_a).result
                b_r = pto.AllocTileOp(tb_b).result
                c_acc = pto.AllocTileOp(tb_c).result

                pto.TLoadOp(None, sv_a, a_mat)
                pto.TLoadOp(None, sv_b, b_mat)

                pto.TMovOp(None, a_mat, a_l)
                pto.TMovOp(None, b_mat, b_r)

                pto.TMatmulOp(None, a_l, b_r, c_acc)
                pto.TStoreOp(None, c_acc, sv_o)

                func.ReturnOp([])

            m.operation.verify()
            return m


if __name__ == "__main__":
    print(build())
