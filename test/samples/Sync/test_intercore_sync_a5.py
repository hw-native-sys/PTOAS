#!/usr/bin/env python3
from mlir.ir import (
    Context,
    F32Type,
    IndexType,
    InsertionPoint,
    IntegerType,
    Location,
    Module,
)
from mlir.dialects import arith, func, pto, scf


def build():
    with Context() as ctx:
        pto.register_dialect(ctx, load=True)
        with Location.unknown(ctx):
            module = Module.create()
            f32 = F32Type.get(ctx)
            idx = IndexType.get(ctx)
            i32 = IntegerType.get_signless(32, ctx)
            ptr_f32 = pto.PtrType.get(f32, ctx)
            fn_ty = func.FunctionType.get([ptr_f32, i32], [])

            with InsertionPoint(module.body):
                fn = func.FuncOp("test_intercore_sync_a5", fn_ty)
                entry = fn.add_entry_block()

            with InsertionPoint(entry):
                c0 = arith.ConstantOp(idx, 0).result
                c0_i32 = arith.ConstantOp(i32, 0).result
                two = arith.ConstantOp(f32, 2.0).result
                pipe_mte3 = pto.PipeAttr.get(pto.PIPE.PIPE_MTE3, ctx)
                pipe_v = pto.PipeAttr.get(pto.PIPE.PIPE_V, ctx)
                pto.sync_set(pipe_mte3, 5)
                # Keep sync.wait in generated code shape checks, but avoid
                # unconditional wait deadlock in single-core functional runs.
                should_wait = arith.CmpIOp(
                    arith.CmpIPredicate.eq, entry.arguments[1], c0_i32
                ).result
                if_op = scf.IfOp(should_wait, [], hasElse=False)
                with InsertionPoint(if_op.then_block):
                    pto.sync_wait(pipe_v, 5)
                    scf.YieldOp([])
                pto.store_scalar(entry.arguments[0], c0, two)
                func.ReturnOp([])

            module.operation.verify()
            return module


if __name__ == "__main__":
    print(build())
