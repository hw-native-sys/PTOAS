# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.


"""Compiler-owned expanded schedule and versioned local-memory dependencies."""
from __future__ import annotations

from pto_costmodel.contract import MAX_TRIPS, SCHEDULE, validate_configuration
from pto_costmodel.wire import fingerprint, integer, require


def stage_order(program, preload):
    count = program["loops"][0]["trip_count"]
    integer(count, "trip count")
    require(count <= MAX_TRIPS, "UNSUPPORTED", "expanded schedules support at most 256 iterations")
    require(sum(len(t["operation_ids"]) for t in program["tasks"]) * count <= 65536,
            "RANGE", "expanded schedule exceeds 65536 operation instances")
    effective = min(preload, count)
    sequence = []
    for step in range(count + effective):
        if step < count:
            sequence.extend((stage, step) for stage in ("C_QK", "V_P"))
        if step >= effective:
            sequence.extend((stage, step - effective) for stage in ("C_PV", "V_O"))
    return sequence


def serial_versions(program):
    """Resolve RAW values before rescheduling; preserve every write generation."""
    require(not any(b.get("aliases") for b in program["buffers"]), "UNSUPPORTED",
            "partial tile aliases require region-aware version analysis")
    count = program["loops"][0]["trip_count"]
    accesses = {}
    for row in program["memory_accesses"]:
        accesses.setdefault(row["op_id"], []).append(row)
    tasks = {t["id"]: t for t in program["tasks"]}
    current = {}
    events = {}
    generations = []
    for stage, iteration in stage_order(program, 0):
        for op in tasks[stage]["operation_ids"]:
            key = f"{op}@{iteration}"
            entry = dict(id=key, op_id=op, iteration=iteration, stage=stage, reads=[], writes=[])
            for access in accesses.get(op, []):
                root = access["buffer_id"]
                if access["access"] in ("read", "read_write"):
                    require(root in current, "UNSUPPORTED", f"uninitialized/recursive buffer read: {root}")
                    entry["reads"].append(dict(buffer_id=root, version=current[root]))
                if access["access"] in ("write", "read_write"):
                    version = f"{root}/{key}"
                    current[root] = version
                    entry["writes"].append(dict(buffer_id=root, version=version))
                    generations.append(dict(id=version, buffer_id=root, writer=key, readers=[]))
            events[key] = entry
    versions = {g["id"]: g for g in generations}
    for event in events.values():
        for read in event["reads"]:
            versions[read["version"]]["readers"].append(event["id"])
    return events, versions


def build_schedule(program, config):
    validate_configuration(config, program)
    events, versions = serial_versions(program)
    tasks = {t["id"]: t for t in program["tasks"]}
    order = stage_order(program, config["preload_count"])
    cores = {"cube": [], "vector": []}
    for stage, iteration in order:
        task = tasks[stage]
        cores[task["core"]].extend(f"{op}@{iteration}" for op in task["operation_ids"])
    positions = {key: i for keys in cores.values() for i, key in enumerate(keys)}
    slots, needs = _allocate_versions(program, config, versions, positions)
    for event in events.values():
        for access in event["reads"] + event["writes"]:
            access["slot"] = slots.get(access["version"])
    edges = [dict(source=v["writer"], target=r, kind="RAW")
             for v in versions.values() for r in v["readers"] if r != v["writer"]]
    edges.extend(_pipe_edges(program))
    schedule = dict(semantics_version=SCHEDULE, requested_preload=config["preload_count"],
                    effective_preload=min(config["preload_count"], program["loops"][0]["trip_count"]),
                    stage_instances=[dict(stage=s, iteration=i) for s, i in order],
                    core_order=cores, events=list(events.values()), dependencies=edges,
                    buffer_requirements=needs, buffer_versions=list(versions.values()))
    schedule["fingerprint"] = fingerprint(schedule)
    return schedule


def _allocate_versions(program, config, versions, positions):
    local = {b["id"] for b in program["buffers"] if b["multi_buffer_eligible"]}
    counts = {b["buffer_id"]: b["count"] for b in config["buffers"]}
    slots = {}
    needs = {}
    for root in sorted(local):
        intervals = []
        for value in versions.values():
            if value["buffer_id"] == root:
                start = positions[value["writer"]]
                readers = [positions[r] for r in value["readers"]]
                require(all(p >= start for p in readers), "SCHEDULE", "reschedule reverses RAW dependency")
                intervals.append((start, max([start] + readers), value["id"]))
        ends = []
        for start, end, key in sorted(intervals):
            # Sharing at an operation boundary is allowed for explicit in-place updates.
            slot = next((i for i, last in enumerate(ends) if last <= start), len(ends))
            if slot == len(ends):
                ends.append(end)
            else:
                ends[slot] = end
            slots[key] = slot
        needs[root] = max(1, len(ends))
        require(counts[root] >= needs[root], "INSUFFICIENT_SLOTS", f"{root} needs {needs[root]} slots")
        slots.update(_rotate_slots(intervals, versions, counts[root]))
    return slots, needs


def _rotate_slots(intervals, versions, count):
    ends = [-1] * count
    assigned = {}
    cursor = 0
    for start, end, key in sorted(intervals):
        writer = versions[key]["writer"]
        prior = next((old for old in assigned if writer in versions[old]["readers"]), None)
        reusable = [index for index in range(count) if ends[index] <= start]
        require(reusable, "INSUFFICIENT_SLOTS", "no free slot for write generation")
        if prior is not None and assigned[prior] in reusable:
            slot = assigned[prior]
        else:
            slot = min(reusable, key=lambda index: (index - cursor) % count)
        assigned[key] = slot
        ends[slot] = end
        cursor = (slot + 1) % count
    return assigned


def _pipe_edges(program):
    edges = []
    for pipe in program["pipes"]:
        actions = {t["action"]: t["op_id"] for t in program["transactions"] if t["pipe_id"] == pipe["id"]}
        capacity = pipe["effective_slot_num"]
        for i in range(program["loops"][0]["trip_count"]):
            edges.append(dict(source=f"{actions['tpush']}@{i}", target=f"{actions['tpop']}@{i}", kind="pipe"))
            if i >= capacity:
                edges.append(dict(source=f"{actions['tfree']}@{i-capacity}",
                                  target=f"{actions['tpush']}@{i}", kind="fifo_capacity"))
    return edges
