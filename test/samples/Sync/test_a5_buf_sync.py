# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

from mlir.ir import Context, Location, Module, InsertionPoint
from mlir.dialects import func, pto


def build():
    with Context() as ctx:
        pto.register_dialect(ctx, load=True)

        with Location.unknown(ctx):
            module = Module.create()
            with InsertionPoint(module.body):
                fn = func.FuncOp("test_a5_buf_sync", func.FunctionType.get([], []))
                entry = fn.add_entry_block()

            with InsertionPoint(entry):
                # Acquire/release buffer-id token on MTE2.
                pto.get_buf(pto.TLOAD, 0)
                pto.rls_buf(pto.TLOAD, 0)
                # Acquire/release buffer-id token on VEC.
                pto.get_buf(pto.TVEC, 0)
                pto.rls_buf(pto.TVEC, 0)
                func.ReturnOp([])

            module.operation.verify()
            return module


if __name__ == "__main__":
    print(build())
