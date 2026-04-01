#!/usr/bin/env python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

from mlir.ir import Context, InsertionPoint, Location, Module
from mlir.dialects import func, pto


def build():
    with Context() as ctx:
        pto.register_dialect(ctx, load=True)
        with Location.unknown(ctx):
            module = Module.create()
            fn_ty = func.FunctionType.get([], [])

            with InsertionPoint(module.body):
                fn = func.FuncOp("test_intercore_sync_a3_missing_setffts", fn_ty)
                entry = fn.add_entry_block()

            with InsertionPoint(entry):
                pipe_fix = pto.PipeAttr.get(pto.PIPE.PIPE_FIX, ctx)
                pipe_v = pto.PipeAttr.get(pto.PIPE.PIPE_V, ctx)
                pto.sync_set(pipe_fix, 7)
                pto.sync_wait(pipe_v, 7)
                func.ReturnOp([])

            module.operation.verify()
            return module


if __name__ == "__main__":
    print(build())
