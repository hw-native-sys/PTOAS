# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.


"""Bounded static candidate materialization using the exported schedule."""
from ptoas.mlir import ir
from ptoas._cv_common import OUTPUT_ATTRS
from pto_costmodel.wire import require
from pto_costmodel.semantics import verify_global_accesses


def serial_baseline(graph, package, candidate):
    require(candidate["configuration"]["preload_count"] == 0
            and all(r["count"] == 1 for r in candidate["configuration"]["buffers"]),
            "BASELINE", "serial baseline requires P=0 and one slot per buffer")
    require(package["bindings"]["alias_contract"] == "disjoint", "ALIAS_UNKNOWN", "disjoint GM arguments required")
    verify_global_accesses(package)
    _slot_allocations(graph, candidate)
    for handle, pipe in graph.handles.items():
        handle.owner.attributes["pto.costmodel.pipe_id"] = ir.StringAttr.get(pipe["id"])
    for _, loop, _ in graph.functions.values():
        for op in loop.regions[0].blocks[0].operations:
            op.attributes["pto.costmodel.event_id"] = ir.StringAttr.get(graph.index.ids[op.operation] + "@loop")
    return graph.index.asm(), []


def materialize(graph, package, candidate):
    require(package["bindings"]["alias_contract"] == "disjoint", "ALIAS_UNKNOWN",
            "compile mode requires explicit disjoint GM arguments")
    program = package["program"]
    for handle, pipe in graph.handles.items():
        handle.owner.attributes["pto.costmodel.pipe_id"] = ir.StringAttr.get(pipe["id"])
    verify_global_accesses(package)
    require(not any(b.get("aliases") for b in program["buffers"]), "UNSUPPORTED",
            "static materialization does not yet support tile subviews/reshapes")
    operations = {graph.index.ids[o]: o for o in graph.index.operations}
    events = {e["id"]: e for e in candidate["schedule"]["events"]}
    slots, replaced = _slot_allocations(graph, candidate)
    accesses = {}
    for row in program["memory_accesses"]:
        accesses.setdefault(row["op_id"], {})[row["operand"]] = row
    trace = []
    for kind, (_, loop, _) in graph.functions.items():
        values = {}
        induction = loop.regions[0].blocks[0].arguments[0]
        bounds = next(l for l in program["loops"] if l["op_id"] == graph.index.ids[loop])
        with ir.InsertionPoint(loop), ir.Location.unknown():
            for key in candidate["schedule"]["core_order"][kind]:
                event = events[key]
                iteration = event["iteration"]
                induction_value = bounds["lower"] + iteration * bounds["step"]
                index = ir.Operation.create("arith.constant", results=[induction.type],
                                            attributes={"value": ir.IntegerAttr.get(induction.type, induction_value)})
                values[(induction, iteration)] = index.results[0]
                _clone_event(operations[event["op_id"]], event, values, accesses, slots)
                trace.append(dict(core=kind, event=key))
        loop.erase()
    for allocation in replaced:
        allocation.erase()
    require(graph.index.module.operation.verify(), "IR", "materialized candidate failed MLIR verification")
    module = graph.index.module.operation
    module.attributes["pto.costmodel.status"] = ir.StringAttr.get("materialized_unverified")
    return graph.index.asm(), trace


def _slot_allocations(graph, candidate):
    result = {}
    replaced = []
    counts = {r["buffer_id"]: r["count"] for r in candidate["configuration"]["buffers"]}
    for key, count in counts.items():
        original = graph.buffer_ops[key]
        original.attributes["pto.costmodel.buffer_id"] = ir.StringAttr.get(key)
        original.attributes["pto.costmodel.slot"] = ir.IntegerAttr.get(ir.IntegerType.get_signless(64), 0)
        require(original.parent.name == "func.func", "UNSUPPORTED", "local alloc must dominate the loop")
        result[key] = [original.results[0]]
        if count == 1:
            continue
        replaced.append(original)
        with ir.InsertionPoint(original), ir.Location.unknown():
            typ = original.results[0].type
            multi = ir.Type.parse(f"!pto.multi_tile_buf<{typ}, count={count}>")
            attrs = {a.name: a.attr for a in original.attributes if a.name not in OUTPUT_ATTRS}
            attrs["pto.costmodel.buffer_id"] = ir.StringAttr.get(key)
            attrs.pop("pto.costmodel.slot", None)
            alloc = ir.Operation.create("pto.alloc_multi_tile", results=[multi], attributes=attrs)
            result[key] = []
            for slot in range(count):
                index_type = ir.IndexType.get()
                index = ir.Operation.create("arith.constant", results=[index_type],
                                            attributes={"value": ir.IntegerAttr.get(index_type, slot)})
                selected = ir.Operation.create("pto.multi_tile_get", operands=[alloc.results[0], index.results[0]],
                                               results=[typ])
                result[key].append(selected.results[0])
    return result, replaced


def _clone_event(op, event, values, accesses, slots):
    require(not op.regions, "UNSUPPORTED", "nested region in stage")
    args = []
    for index, operand in enumerate(op.operands):
        access = accesses.get(event["op_id"], {}).get(index)
        if access is not None and access["buffer_id"] in slots:
            root = access["buffer_id"]
            read = next((r for r in event["reads"] if r["buffer_id"] == root), None)
            write = next((w for w in event["writes"] if w["buffer_id"] == root), None)
            if access["access"] == "read_write":
                require(read["slot"] == write["slot"], "UNSUPPORTED", "in-place update needs different slots")
            selected = read if access["access"] == "read" else write
            args.append(slots[root][selected["slot"]])
        else:
            args.append(values.get((operand, event["iteration"]), operand))
    attrs = {a.name: a.attr for a in op.attributes if a.name not in OUTPUT_ATTRS}
    attrs["pto.costmodel.event_id"] = ir.StringAttr.get(event["id"])
    clone = ir.Operation.create(op.name, results=[v.type for v in op.results], operands=args, attributes=attrs)
    for old, new in zip(op.results, clone.results):
        values[(old, event["iteration"])] = new
