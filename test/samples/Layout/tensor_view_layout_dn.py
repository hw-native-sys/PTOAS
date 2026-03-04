"""
Explicit tensor_view layout sample (DN).

This checks that:
  - User-specified `layout` on pto.make_tensor_view is preserved through
    view lowering (make_tensor_view -> reinterpret_cast -> subview)
  - The generated C++ GlobalTensor uses pto::Layout::DN.
"""

from mlir.ir import Context, Location, InsertionPoint, Module, IndexType
from mlir.dialects import arith, func, pto, builtin


def idx(val: int):
    return arith.ConstantOp(IndexType.get(), val).result


def build_module():
    with Context() as ctx, Location.unknown():
        pto.register_dialect(ctx, load=True)

        f32 = builtin.F32Type.get()
        mat = pto.AddressSpaceAttr.get(pto.AddressSpace.MAT, ctx)

        tensor_view_ty = pto.TensorViewType.get([1, 1, 16, 1024, 1024], f32)
        part_view_ty = pto.PartitionTensorViewType.get([1, 1, 16, 16, 16], f32)
        tile_buf_ty = pto.TileBufType.get(
            [256, 16], f32, mat, [256, 16], pto.TileBufConfigAttr.get_default(ctx)
        )

        ptr_f32 = pto.PtrType.get(f32)
        layout_dn = pto.LayoutAttr.get(pto.Layout.DN, ctx)

        m = Module.create()
        with InsertionPoint(m.body):

            @func.FuncOp.from_py_func(ptr_f32, ptr_f32)
            def run(src, dst):
                c0 = idx(0)

                shape = [idx(1), idx(1), idx(16), idx(1024), idx(1024)]
                # DN (col-major) for the minor 2D dims (rows x cols):
                #   addr(r, c) = base + r * 1 + c * rows
                # so we want strides [..., 1, rows] for the last two dims.
                strides = [idx(1048576), idx(1048576), idx(1048576), idx(1), idx(1024)]

                src_view = pto.MakeTensorViewOp(
                    tensor_view_ty, src, shape, strides, layout=layout_dn
                ).result
                src_part = pto.PartitionViewOp(
                    part_view_ty,
                    src_view,
                    offsets=[c0, c0, c0, c0, c0],
                    sizes=[idx(1), idx(1), idx(16), idx(16), idx(16)],
                ).result

                tile = pto.AllocTileOp(tile_buf_ty).result
                pto.TLoadOp(None, src_part, tile)

                dst_view = pto.MakeTensorViewOp(
                    tensor_view_ty, dst, shape, strides, layout=layout_dn
                ).result
                dst_part = pto.PartitionViewOp(
                    part_view_ty,
                    dst_view,
                    offsets=[c0, c0, c0, c0, c0],
                    sizes=[idx(1), idx(1), idx(16), idx(16), idx(16)],
                ).result

                pto.TStoreOp(None, tile, dst_part)
                return

        return m


if __name__ == "__main__":
    print(build_module())
