# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.


"""Read actual MLIR objects at the canonical CV exchange checkpoint."""
from __future__ import annotations

import math

from ptoas.mlir import ir
from ptoas.mlir.dialects import pto
from ptoas._cv_common import OUTPUT_ATTRS, ContractError, integer, require

ALLOWED = frozenset({
    "builtin.module", "func.func", "func.return", "scf.for", "scf.yield",
    "arith.constant", "arith.addi", "arith.muli", "arith.subi", "arith.remui",
    "arith.cmpi", "arith.select", "arith.index_cast",
    "pto.alloc_tile", "pto.declare_tile", "pto.reserve_buffer", "pto.import_reserved_buffer",
    "pto.initialize_l2l_pipe", "pto.tpush", "pto.tpop", "pto.tfree",
    "pto.get_subblock_idx", "pto.make_tensor_view", "pto.partition_view", "pto.treshape", "pto.subview",
    "pto.tload", "pto.tstore", "pto.tmov", "pto.tmatmul", "pto.tmatmul.acc",
    "pto.tadd", "pto.tsub", "pto.tmul", "pto.tdiv", "pto.tmuls", "pto.tadds",
    "pto.tneg", "pto.texp",
})
ALIASES = frozenset({"pto.treshape", "pto.subview"})


def walk(operation):
    op = operation.operation
    yield op
    for region in op.regions:
        for block in region.blocks:
            for child in block.operations:
                yield from walk(child)


def attr(op, name, default=None):
    if name not in op.attributes:
        return default
    value = op.attributes[name]
    if ir.StringAttr.isinstance(value):
        return ir.StringAttr(value).value
    if ir.IntegerAttr.isinstance(value):
        return ir.IntegerAttr(value).value
    return str(value)


def semantic_attrs(op):
    return {item.name: str(item.attr) for item in op.attributes
            if item.name not in OUTPUT_ATTRS}


def owner_operation(value):
    owner = value.owner
    return owner.operation if hasattr(owner, "operation") else owner


def constant(value):
    owner = owner_operation(value)
    require(isinstance(owner, ir.Operation) and owner.name == "arith.constant",
            "UNSUPPORTED", "v1 requires constant bounds, strides and offsets")
    return integer(attr(owner, "value"), "constant")


def space_name(value):
    text = str(value)
    for space in ("mat", "left", "right", "acc", "vec"):
        if text == f"#pto.address_space<{space}>":
            return space
    raise ContractError("UNSUPPORTED", f"unsupported memory space: {text}")


def tile_info(typ, profile):
    require(pto.TileBufType.isinstance(typ), "UNSUPPORTED", "expected a tile_buf")
    tile = pto.TileBufType(typ)
    shape = list(tile.shape)
    valid = list(tile.valid_shape)
    require(len(shape) == 2 and len(valid) == 2, "UNSUPPORTED", "v1 requires rank-2 tiles")
    for dimension in shape:
        integer(dimension, "tile dimension", 1)
    for extent, dimension in zip(valid, shape):
        integer(extent, "valid extent")
        require(extent <= dimension, "UNSUPPORTED", "valid extent exceeds tile")
    dtype = str(tile.element_type)
    require(dtype in ("f16", "bf16", "f32", "i32"), "UNSUPPORTED", "unsupported dtype")
    require(tile.compact_mode == 0, "UNSUPPORTED", "compact tile storage is not supported in v1")
    elem_bytes = 2 if dtype in ("f16", "bf16") else 4
    size = integer(math.prod(shape) * elem_bytes, "allocation bytes", 1)
    space = space_name(tile.memory_space)
    alignment = profile["alignment_bytes"][space]
    return dict(shape=shape, valid_shape=valid, dtype=dtype, memory_space=space,
                logical_bytes=math.prod(valid) * elem_bytes, allocation_bytes=size,
                slot_stride_bytes=((size + alignment - 1) // alignment) * alignment,
                layout=dict(blayout=tile.blayout_value, slayout=tile.slayout_value,
                            fractal_bytes=tile.s_fractal_size))


class IRIndex:
    """Stable traversal IDs and a location/SSA-spelling independent fingerprint input."""

    def __init__(self, module):
        self.module = module
        self.operations = list(walk(module.operation))
        self.ids = {op: f"op{index}" for index, op in enumerate(self.operations)}
        self.values = {}
        for op in self.operations:
            for index, result in enumerate(op.results):
                self.values[result] = f"{self.ids[op]}.r{index}"
            for ri, region in enumerate(op.regions):
                for bi, block in enumerate(region.blocks):
                    for ai, value in enumerate(block.arguments):
                        self.values[value] = f"{self.ids[op]}.b{ri}.{bi}.a{ai}"

    def signature(self):
        rows = []
        for op in self.operations:
            require(op.name in ALLOWED, "UNSUPPORTED", f"unsupported op: {op.name}")
            require("pto.multi_buffer_addrs" not in op.attributes,
                    "PHYSICAL_ADDRESS", "physical address annotations are internal")
            parent_id = None if op == self.module.operation else self.ids[op.parent]
            rows.append(dict(id=self.ids[op], parent_id=parent_id, name=op.name, attrs=semantic_attrs(op),
                             operands=[self.values[v] for v in op.operands],
                             results=[str(v.type) for v in op.results],
                             blocks=[[[str(a.type) for a in b.arguments] for b in r.blocks]
                                     for r in op.regions]))
        return rows

    def clean_annotations(self):
        for op in self.operations:
            for key in OUTPUT_ATTRS:
                if key in op.attributes:
                    del op.attributes[key]

    def set_string(self, op, name, value):
        op.attributes[name] = ir.StringAttr.get(value, self.module.context)

    def set_integer(self, op, name, value):
        typ = ir.IntegerType.get_signless(64, self.module.context)
        op.attributes[name] = ir.IntegerAttr.get(typ, value)

    def asm(self):
        return self.module.operation.get_asm(enable_debug_info=False, print_generic_op_form=True) + "\n"
