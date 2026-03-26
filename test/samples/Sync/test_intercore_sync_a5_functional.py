#!/usr/bin/env python3
from mlir.ir import Context, F32Type, IndexType, InsertionPoint, IntegerType, Location, Module
from mlir.dialects import arith, func, pto, scf


def build():
    with Context() as ctx:
        pto.register_dialect(ctx, load=True)
        with Location.unknown(ctx):
            module = Module.create()

            f32 = F32Type.get(ctx)
            i64 = IntegerType.get_signless(64, ctx)
            idx = IndexType.get(ctx)
            ptr_f32 = pto.PtrType.get(f32, ctx)
            fn_ty = func.FunctionType.get([ptr_f32], [])

            with InsertionPoint(module.body):
                fn = func.FuncOp("test_intercore_sync_a5_functional", fn_ty)
                entry = fn.add_entry_block()

            with InsertionPoint(entry):
                out = entry.arguments[0]

                c0_idx = arith.ConstantOp(idx, 0).result
                c1_idx = arith.ConstantOp(idx, 1).result
                c0_i64 = arith.ConstantOp(i64, 0).result
                c1_i64 = arith.ConstantOp(i64, 1).result
                c2 = arith.ConstantOp(f32, 2.0).result

                bid = pto.GetBlockIdxOp().result
                pipe_mte3 = pto.PipeAttr.get(pto.PIPE.PIPE_MTE3, ctx)
                pipe_v = pto.PipeAttr.get(pto.PIPE.PIPE_V, ctx)

                is_producer = arith.CmpIOp(arith.CmpIPredicate.eq, bid, c0_i64).result
                producer_if = scf.IfOp(is_producer, [], hasElse=False)
                with InsertionPoint(producer_if.then_block):
                    # producer core: publish data then signal event 5.
                    pto.store_scalar(out, c0_idx, c2)
                    pto.sync_set(pipe_mte3, 5)
                    scf.YieldOp([])

                is_consumer = arith.CmpIOp(arith.CmpIPredicate.eq, bid, c1_i64).result
                consumer_if = scf.IfOp(is_consumer, [], hasElse=False)
                with InsertionPoint(consumer_if.then_block):
                    # consumer core: wait event 5 then observe producer write.
                    pto.sync_wait(pipe_v, 5)
                    loaded = pto.load_scalar(f32, out, c0_idx)
                    pto.store_scalar(out, c1_idx, loaded)
                    scf.YieldOp([])

                func.ReturnOp([])

            module.operation.verify()
            return module


if __name__ == "__main__":
    print(build())
