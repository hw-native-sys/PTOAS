# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

from mlir.ir import Context, Location, Module, InsertionPoint
from mlir.dialects import arith, func, pto
from mlir.ir import IndexType


def build():
    with Context() as ctx:
        pto.register_dialect(ctx, load=True)

        with Location.unknown(ctx):
            m = Module.create()

            idx = IndexType.get(ctx)
            pipe_mte2 = pto.PipeAttr.get(pto.PIPE.PIPE_MTE2, ctx)
            pipe_mte3 = pto.PipeAttr.get(pto.PIPE.PIPE_MTE3, ctx)

            fn_ty = func.FunctionType.get([], [])
            with InsertionPoint(m.body):
                fn = func.FuncOp("test_set_wait_unified_api_py", fn_ty)
                entry = fn.add_entry_block()

            with InsertionPoint(entry):
                c2 = arith.ConstantOp(idx, 2).result

                # Static path: int -> EVENT_ID2.
                pto.set_flag(pipe_mte2, pipe_mte3, 2)
                pto.wait_flag(pipe_mte2, pipe_mte3, 2)

                # Dynamic path: SSA index value.
                pto.set_flag(pipe_mte2, pipe_mte3, c2)
                pto.wait_flag(pipe_mte2, pipe_mte3, c2)

                func.ReturnOp([])

            m.operation.verify()
            return m


if __name__ == "__main__":
    print(build())
