#!/usr/bin/env python3
from mlir.ir import Context, F32Type, IndexType, InsertionPoint, Location, Module
from mlir.dialects import arith, func, pto


def build():
    with Context() as ctx:
        pto.register_dialect(ctx, load=True)
        with Location.unknown(ctx):
            module = Module.create()
            f32 = F32Type.get(ctx)
            idx = IndexType.get(ctx)
            ptr_f32 = pto.PtrType.get(f32, ctx)
            fn_ty = func.FunctionType.get([ptr_f32], [])

            with InsertionPoint(module.body):
                fn = func.FuncOp("test_intercore_sync_a5_dyn", fn_ty)
                entry = fn.add_entry_block()

            with InsertionPoint(entry):
                c0 = arith.ConstantOp(idx, 0).result
                c5 = arith.ConstantOp(idx, 5).result
                two = arith.ConstantOp(f32, 2.0).result
                pipe_mte3 = pto.PipeAttr.get(pto.PIPE.PIPE_MTE3, ctx)
                pipe_v = pto.PipeAttr.get(pto.PIPE.PIPE_V, ctx)

                pto.sync_set(pipe_mte3, c5)
                pto.sync_wait(pipe_v, c5)
                pto.store_scalar(entry.arguments[0], c0, two)
                func.ReturnOp([])

            module.operation.verify()
            return module


if __name__ == "__main__":
    print(build())
