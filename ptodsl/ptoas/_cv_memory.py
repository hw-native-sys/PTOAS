# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.


"""Compiler-produced physical storage provenance at the final PTO checkpoint."""
from ptoas.mlir import ir
from ptoas._cv_ir import attr, tile_info, walk
from pto_costmodel.wire import require


def address(value):
    op = value.owner
    if op.name == "arith.constant":
        return attr(op, "value")
    return None


def physical_memory(module, package, candidate):
    buffers = {b["id"]: b for b in package["program"]["buffers"]}
    profile = package["target"]["compiler_budget"]
    rows, mapped, endpoints = [], {}, {}
    layouts = module.operation.attributes
    if "pto.costmodel.slot_layouts" in layouts:
        for raw in ir.ArrayAttr(layouts["pto.costmodel.slot_layouts"]):
            row = ir.DictAttr(raw)
            key = ir.StringAttr(row["buffer_id"]).value
            for slot, offset in enumerate(ir.DenseI64ArrayAttr(row["addresses"])):
                mapped[(key, slot)] = int(offset)
    for func in module.body.operations:
        if func.operation.name != "func.func":
            continue
        core = "cube" if "cube" in str(func.attributes["pto.kernel_kind"]) else "vector"
        for op in walk(func):
            if op.name == "pto.alloc_tile":
                _allocation(op, core, buffers, profile, rows, mapped)
            elif op.name == "pto.initialize_l2l_pipe":
                _backing(op, core, package, buffers, rows, endpoints)
    for pipe in package["program"]["pipes"]:
        addresses = endpoints.get(pipe["id"], {})
        require(set(addresses) == {"cube", "vector"} and len(set(addresses.values())) == 1,
                "PROVENANCE", "pipe endpoint backing addresses differ or are missing")
    for (key, slot), offset in mapped.items():
        if not any(r["source_buffer_id"] == key and r["slot"] == slot for r in rows):
            rows.append(_row(buffers[key], slot, offset, "reserved_unused"))
    counts = {b["buffer_id"]: b["count"] for b in candidate["configuration"]["buffers"]}
    for key, count in counts.items():
        for slot in range(count):
            if not any(r["source_buffer_id"] == key and r["slot"] == slot for r in rows):
                rows.append(_row(buffers[key], slot, None, "eliminated"))
    rows.sort(key=lambda r: r["allocation_id"])
    return dict(schema_version="pto.compiler_memory.v1", allocations=rows,
                versions=candidate["schedule"]["buffer_versions"],
                borrowed_versions=_borrowed_versions(buffers, rows, package, candidate),
                accesses=candidate["schedule"]["events"],
                ownership={k: b["owner"] for k, b in buffers.items()},
                usage=_usage(rows, profile), read_only=True)


def _borrowed_versions(buffers, rows, package, candidate):
    result = []
    for version in candidate["schedule"]["buffer_versions"]:
        buffer = buffers[version["buffer_id"]]
        if buffer["owner"] != "borrowed_entry":
            continue
        pipe = next(p for p in package["program"]["pipes"] if p["backing_buffer_id"] == buffer["backing_buffer_id"])
        backing = next(r for r in rows if r["source_buffer_id"] == buffer["backing_buffer_id"])
        iteration = int(version["writer"].split("@")[-1])
        slot = iteration % pipe["effective_slot_num"]
        result.append(dict(version=version["id"], buffer_id=buffer["id"],
                           backing_allocation_id=backing["allocation_id"], slot=slot,
                           offset_bytes=backing["offset_bytes"] + slot * pipe["slot_size_bytes"],
                           payload_bytes=buffer["allocation_bytes"], memory_space=buffer["memory_space"],
                           physical_instances=buffer["physical_instances"]))
    return result


def _row(buffer, slot, offset, state):
    size = buffer["allocation_bytes"]
    return dict(allocation_id=f"{buffer['id']}/slot{slot}", source_buffer_id=buffer["id"],
                slot=slot, state=state, physical_instances=buffer["physical_instances"],
                memory_space=buffer["memory_space"], offset_bytes=offset, payload_bytes=size,
                reserved_bytes=buffer.get("slot_stride_bytes", size), owner=buffer["owner"])


def _allocation(op, core, buffers, profile, rows, mapped):
    key = attr(op, "pto.costmodel.buffer_id")
    slot = attr(op, "pto.costmodel.slot", 0)
    offset = address(op.operands[0]) if op.operands else None
    info = tile_info(op.results[0].type, profile)
    if key is None:
        key = f"compiler.{core}.temporary{len(rows)}"
        buffer = dict(id=key, owner="compiler_generated", **info,
                      physical_instances=["AIC0"] if core == "cube" else ["AIV0", "AIV1"])
    else:
        require(key in buffers, "PROVENANCE", "allocation has unknown source buffer")
        buffer = buffers[key]
        require(info["allocation_bytes"] == buffer["allocation_bytes"]
                and info["memory_space"] == buffer["memory_space"], "PROVENANCE", "allocation layout changed")
    require(offset is not None and offset >= 0, "LAYOUT_UNKNOWN", "nonconstant local address")
    if (key, slot) in mapped:
        require(mapped[(key, slot)] == offset, "PROVENANCE", "slot address differs from planner record")
    row = _row(buffer, slot, offset, "materialized")
    old = next((r for r in rows if r["allocation_id"] == row["allocation_id"]), None)
    require(old is None or old == row, "PROVENANCE", "conflicting allocation provenance")
    if old is None:
        rows.append(row)


def _backing(op, core, package, buffers, rows, endpoints):
    key = attr(op, "pto.costmodel.pipe_id")
    require(key is not None, "PROVENANCE", "pipe origin missing")
    pipe = next(p for p in package["program"]["pipes"] if p["id"] == key)
    require(attr(op, "slot_num") == pipe["effective_slot_num"]
            and attr(op, "slot_size") == pipe["slot_size_bytes"]
            and attr(op, "dir_mask") == (1 if pipe["direction"] == "c2v" else 2),
            "PROVENANCE", "lowered pipe differs from candidate contract")
    offset = address(op.operands[0])
    require(offset is not None and offset >= 0, "LAYOUT_UNKNOWN", "nonconstant pipe backing")
    require(core not in endpoints.setdefault(key, {}), "PROVENANCE", "duplicate pipe endpoint")
    endpoints[key][core] = offset
    owner = "vector" if pipe["direction"] == "c2v" else "cube"
    if core != owner:
        return
    buffer = buffers[pipe["backing_buffer_id"]]
    rows.append(_row(buffer, 0, offset, "materialized"))


def _usage(rows, profile):
    domains = {}
    for row in rows:
        if row["offset_bytes"] is None:
            continue
        space = row["memory_space"]
        alignment = profile["alignment_bytes"][space]
        require(row["offset_bytes"] % alignment == 0, "ALIGNMENT", "misaligned physical allocation")
        for core in row["physical_instances"]:
            domains.setdefault((core, space), []).append(row)
    result = []
    for (core, space), allocations in sorted(domains.items()):
        intervals = sorted((r["offset_bytes"], r["offset_bytes"] + r["reserved_bytes"]) for r in allocations)
        end, union = 0, 0
        for start, stop in intervals:
            union += max(0, stop - max(end, start))
            end = max(end, stop)
        capacity = profile["capacity_bytes"][space]
        require(end <= capacity, "CAPACITY", "realized arena exceeds compiler budget")
        result.append(dict(core=core, memory_space=space, arena_extent_bytes=end,
                           union_reserved_bytes=union, capacity_bytes=capacity,
                           basis="compiler_realized_layout", coverage="final_PTO_checkpoint"))
    return result
