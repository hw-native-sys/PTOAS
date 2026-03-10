from mlir.ir import Context, Location, Module, InsertionPoint
from mlir.ir import F16Type, MemRefType
from mlir.dialects import func, memref, pto


def build():
    with Context() as ctx:
        pto.register_dialect(ctx, load=True)

        with Location.unknown(ctx):
            m = Module.create()

            f16 = F16Type.get(ctx)
            vec = pto.AddressSpaceAttr.get(pto.AddressSpace.VEC, ctx)
            ub_ty = MemRefType.get([16, 16, 16], f16, memory_space=vec)

            fn_ty = func.FunctionType.get([], [])
            with InsertionPoint(m.body):
                fn = func.FuncOp("test_intra_pipe_barrier_py", fn_ty)
                entry = fn.add_entry_block()

            with InsertionPoint(entry):
                ub0 = memref.AllocOp(ub_ty, [], []).result
                ub1 = memref.AllocOp(ub_ty, [], []).result

                # Two dependent PIPE_V ops on the same buffer should be
                # serialized by an intra-pipe barrier when auto insert sync is
                # enabled.
                pto.TAddOp(ub0, ub0, ub0)
                pto.TAddOp(ub0, ub0, ub1)

                func.ReturnOp([])

            m.operation.verify()
            return m


if __name__ == "__main__":
    print(build())
