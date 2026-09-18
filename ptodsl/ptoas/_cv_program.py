# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.


"""Export typed, model-independent facts from the verified checkpoint."""
from copy import deepcopy

from ptoas._cv_common import OUTPUT_ATTRS
from ptoas._cv_types import attribute_record, type_record
from pto_costmodel.contract import SEMANTICS
from pto_costmodel.wire import require


def program_from_graph(graph):
    manifest = graph.manifest
    keys = ("pipeline_id", "loops", "buffers", "pipes", "tasks", "transactions", "memory_accesses")
    program = {k: deepcopy(manifest[k]) for k in keys}
    program.update(semantics_version=SEMANTICS, operations=[], values={}, functions=[])
    for kind, (func, _, scope) in graph.functions.items():
        args = func.regions[0].blocks[0].arguments
        program["functions"].append(dict(id=scope, core=kind, arguments=[graph.index.values[v] for v in args]))
    for op in graph.index.operations:
        record = _operation(op, graph)
        program["operations"].append(record)
        for value in op.results:
            program["values"][graph.index.values[value]] = type_record(value.type, graph.profile)
        for region in op.regions:
            for block in region.blocks:
                for value in block.arguments:
                    program["values"][graph.index.values[value]] = type_record(value.type, graph.profile)
    for buffer in program["buffers"]:
        buffer["root_id"] = buffer["id"]
        buffer["value_id"] = buffer.pop("debug_value", None)
        core = next(f["core"] for f in program["functions"] if f["id"] == buffer["source_scope"])
        buffer["physical_instances"] = ["AIC0"] if core == "cube" else ["AIV0", "AIV1"]
    program["gm_aliasing"] = "not_proven"
    program["dependency_coverage"] = "static_local_and_pipe; gm_requires_bindings"
    return program


def _operation(op, graph):
    attributes = {a.name: attribute_record(a.attr, graph.profile) for a in op.attributes
                  if a.name not in OUTPUT_ATTRS and not a.name.startswith("pto.costmodel.")}
    return dict(id=graph.index.ids[op], name=op.name, attributes=attributes,
                operands=[graph.index.values[v] for v in op.operands],
                results=[graph.index.values[v] for v in op.results],
                parent_id=None if op == graph.index.module.operation else graph.index.ids[op.parent])


def validate_bindings(bindings, program):
    from pto_costmodel.wire import fields, integer
    fields(bindings, ("arguments", "alias_contract", "scenario"))
    require(bindings["alias_contract"] in ("unknown", "disjoint"), "BINDINGS", "unknown alias contract")
    require(isinstance(bindings["scenario"], str), "BINDINGS", "scenario must be a string")
    expected = {a for f in program["functions"] for a in f["arguments"]}
    fields(bindings["arguments"], expected)
    for key, row in bindings["arguments"].items():
        fields(row, ("shape", "strides", "dtype"))
        require(row["dtype"] == program["values"][key]["dtype"], "BINDINGS", "argument dtype mismatch")
        require(isinstance(row["shape"], list) and isinstance(row["strides"], list)
                and len(row["shape"]) == len(row["strides"]) == 2,
                "BINDINGS", "rank-2 argument shape and strides required")
        for dimension in row["shape"] + row["strides"]:
            integer(dimension, "argument shape/stride", 1)


def inferred_bindings(program):
    ops = {o["results"][0]: o for o in program["operations"] if len(o["results"]) == 1}
    arguments = {}
    for function in program["functions"]:
        for arg in function["arguments"]:
            views = [o for o in program["operations"] if o["name"] == "pto.make_tensor_view"
                     and o["operands"][0] == arg]
            require(len(views) == 1, "BINDINGS", "explicit bindings needed for argument views")
            dims = []
            for value in views[0]["operands"][1:]:
                require(value in ops and ops[value]["name"] == "arith.constant",
                        "BINDINGS", "explicit bindings needed for dynamic view")
                dims.append(ops[value]["attributes"]["value"])
            arguments[arg] = dict(shape=dims[:2], strides=dims[2:], dtype=program["values"][arg]["dtype"])
    result = dict(arguments=arguments, alias_contract="unknown", scenario="static_inferred")
    validate_bindings(result, program)
    return result
