#!/usr/bin/env python3
from mlir.ir import Context, IndexType, InsertionPoint, IntegerType, Location, MemRefType, Module
from mlir.dialects import func, pto


def build():
    with Context() as ctx:
        pto.register_dialect(ctx, load=True)
        with Location.unknown(ctx):
            module = Module.create()

            idx = IndexType.get(ctx)
            i64 = IntegerType.get_signless(64, ctx)
            ffts_ty = MemRefType.get([256], i64)
            fn_ty = func.FunctionType.get([ffts_ty, idx], [])

            with InsertionPoint(module.body):
                fn = func.FuncOp("test_intercore_sync_a3_modes", fn_ty)
                entry = fn.add_entry_block()

            with InsertionPoint(entry):
                pipe_mte3 = pto.PipeAttr.get(pto.PIPE.PIPE_MTE3, ctx)
                pto.set_ffts(entry.arguments[0])
                pto.sync_set(pipe_mte3, 3, ffts_mode=0)
                pto.sync_set(pipe_mte3, entry.arguments[1], ffts_mode=1)
                func.ReturnOp([])

            module.operation.verify()
            return module


if __name__ == "__main__":
    print(build())
