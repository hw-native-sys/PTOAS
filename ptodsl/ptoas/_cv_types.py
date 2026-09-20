# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.


"""Typed MLIR-to-wire values; opaque text is never a modeling fallback."""
from ptoas.mlir import ir
from ptoas.mlir.dialects import pto
from ptoas._cv_ir import tile_info
from pto_costmodel.wire import ContractError, require

ENUMS = {
    "#pto.kernel_kind<cube>": "cube", "#pto.kernel_kind<vector>": "vector",
    "#pto<relu_pre_mode no_relu>": "no_relu", "#pto<acc_phase unspecified>": "unspecified",
    "#pto<atomic_type atomic_none>": "atomic_none", "#pto<st_phase unspecified>": "unspecified",
    "#arith.overflow<none>": "none",
}
SPACES = {"#pto.address_space<" + name + ">": name for name in ("gm", "mat", "left", "right", "acc", "vec")}


def type_record(typ, profile):
    if pto.TileBufType.isinstance(typ):
        tile = pto.TileBufType(typ)
        return dict(kind="tile", padding=tile.pad_value, compact_mode=tile.compact_mode, **tile_info(typ, profile))
    if pto.PtrType.isinstance(typ):
        pointer = pto.PtrType(typ)
        require(str(pointer.memory_space) == "#pto.address_space<gm>", "UNSUPPORTED_TYPE",
                "kernel pointers must address GM")
        return dict(kind="pointer", dtype=str(pointer.element_type), memory_space="gm")
    for cls, kind in ((pto.TensorViewType, "tensor_view"), (pto.PartitionTensorViewType, "partition_view")):
        if cls.isinstance(typ):
            view = cls(typ)
            return dict(kind=kind, shape=[d if d >= 0 else None for d in view.shape], dtype=str(view.element_type))
    if ir.FunctionType.isinstance(typ):
        function = ir.FunctionType(typ)
        return dict(kind="function", inputs=[type_record(t, profile) for t in function.inputs],
                    results=[type_record(t, profile) for t in function.results])
    if ir.IntegerType.isinstance(typ):
        integer = ir.IntegerType(typ)
        signedness = "signless" if integer.is_signless else "signed" if integer.is_signed else "unsigned"
        return dict(kind="integer", bits=integer.width, signedness=signedness)
    if ir.IndexType.isinstance(typ):
        return dict(kind="index")
    if str(typ) in ("f16", "bf16", "f32", "f64"):
        return dict(kind="float", dtype=str(typ))
    if str(typ) == "!pto.pipe":
        return dict(kind="pipe")
    raise ContractError("UNSUPPORTED_TYPE", str(typ))


def attribute_record(value, profile):
    if ir.BoolAttr.isinstance(value):
        return ir.BoolAttr(value).value
    if ir.IntegerAttr.isinstance(value):
        return ir.IntegerAttr(value).value
    if ir.FloatAttr.isinstance(value):
        return ir.FloatAttr(value).value
    if ir.StringAttr.isinstance(value):
        return ir.StringAttr(value).value
    if ir.FlatSymbolRefAttr.isinstance(value):
        return dict(symbol=ir.FlatSymbolRefAttr(value).value)
    if ir.DenseI32ArrayAttr.isinstance(value):
        return list(ir.DenseI32ArrayAttr(value))
    if ir.DenseI64ArrayAttr.isinstance(value):
        return list(ir.DenseI64ArrayAttr(value))
    if ir.TypeAttr.isinstance(value):
        return type_record(ir.TypeAttr(value).value, profile)
    if str(value) in ENUMS:
        return ENUMS[str(value)]
    if str(value) in SPACES:
        return SPACES[str(value)]
    raise ContractError("UNSUPPORTED_ATTRIBUTE", str(value))
