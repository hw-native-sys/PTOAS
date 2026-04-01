# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Test pto.print: format is string attribute, scalar is operand. Generates PRINTF("format", scalar) in C++."""
from mlir.ir import Context, Location, Module, InsertionPoint
from mlir.dialects import func, pto
from mlir.ir import F32Type


def build():
    with Context() as ctx:
        pto.register_dialect(ctx, load=True)

        with Location.unknown(ctx):
            m = Module.create()

            f32 = F32Type.get(ctx)
            fn_ty = func.FunctionType.get([f32], [])
            with InsertionPoint(m.body):
                fn = func.FuncOp("print_scalar_kernel", fn_ty)
                entry = fn.add_entry_block()

            with InsertionPoint(entry):
                scalar = entry.arguments[0]
                pto.PrintOp("scalar = %+08.3f", scalar)
                func.ReturnOp([])

            m.operation.verify()
            return m


if __name__ == "__main__":
    print(build())
