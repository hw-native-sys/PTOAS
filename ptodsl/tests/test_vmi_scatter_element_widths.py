# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""The public scatter builder must fail IR verification for illegal widths."""
from ptoas.mlir import ir
from ptoas.mlir.dialects import func
from ptoas.mlir.dialects import pto as dialect

from ptodsl import pto


def verify_scatter(elem, index, lanes):
    with ir.Context() as ctx, ir.Location.unknown():
        dialect.register_dialect(ctx)
        types = [ir.Type.parse(f"!pto.vmi.vreg<{lanes}x{elem}>"),
                 ir.Type.parse(f"!pto.ptr<{elem}, ub>"),
                 ir.Type.parse(f"!pto.vmi.vreg<{lanes}x{index}>"),
                 dialect.VMIMaskType.get(lanes, "pred")]
        module = ir.Module.create()
        with ir.InsertionPoint(module.body):
            fn = func.FuncOp("probe", (types, []))
        block = fn.add_entry_block()
        with ir.InsertionPoint(block):
            pto.vmi.vscatter(*block.arguments)
            func.ReturnOp([])
        module.operation.verify()


def verify_rejected_scatter(elem, index, lanes, bits):
    try:
        verify_scatter(elem, index, lanes)
    except ir.MLIRError as exc:
        expected = 32 if bits == 32 else 16
        assert (f"requires {expected}-bit offset vector elements for "
                f"{bits}-bit values") in str(exc), str(exc)
    else:
        raise AssertionError(f"vscatter accepted {elem}/{index}, VL={lanes}")


def main():
    rows = (("i8", 8, ("i16", "ui16"), ("i32", "ui32")),
            ("i16", 16, ("i16", "ui16"), ("i32", "ui32")),
            ("f16", 16, ("i16", "ui16"), ("i32", "ui32")),
            ("bf16", 16, ("i16", "ui16"), ("i32", "ui32")),
            ("i32", 32, ("i32", "ui32"), ("i16", "ui16")),
            ("f32", 32, ("i32", "ui32"), ("i16", "ui16")))
    for elem, bits, good, bad in rows:
        for lanes in (1, 2, 4, 8, 64, 128, 256):
            for index in good:
                verify_scatter(elem, index, lanes)
            for index in bad:
                verify_rejected_scatter(elem, index, lanes, bits)


if __name__ == "__main__":
    main()
