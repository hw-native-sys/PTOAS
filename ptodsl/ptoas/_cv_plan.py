# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.


"""Validate a plan fully before attaching annotation-only metadata."""
from __future__ import annotations

from ptoas._cv_common import SCHEMA, fields, fingerprint, integer, require


def validate_plan(plan, graph):
    required = ("schema_version", "input_fingerprint", "target_profile_fingerprint",
                "bindings_fingerprint", "pipeline_id", "schedule_kind", "preload_semantics",
                "preload_count", "local_schedule", "buffers")
    fields(plan, required, ("metrics", "model_version"))
    require(plan["schema_version"] == SCHEMA, "SCHEMA", "unsupported plan schema")
    for key in ("input_fingerprint", "target_profile_fingerprint", "bindings_fingerprint"):
        require(plan[key] == graph.manifest[key], "STALE_PLAN", f"{key} mismatch")
    require(plan["pipeline_id"] == "cv0", "UNKNOWN_ID", "unknown pipeline_id")
    require(plan["schedule_kind"] == "prefix_suffix_v1"
            and plan["preload_semantics"] == "iteration_distance"
            and plan["local_schedule"] == "off", "SCHEDULE", "unsupported scheduling semantics")
    requested = integer(plan["preload_count"], "preload_count")
    effective = min(requested, graph.loops[0]["trip_count"])
    count = graph.loops[0]["trip_count"]
    if count:
        pipe_p = next(p for p in graph.pipes.values() if p["direction"] == "v2c")
        require(pipe_p["effective_slot_num"] >= max(effective, 1),
                "CAPACITY", "pipe_p cannot support the effective preload distance")
    buffers = _validate_buffers(plan["buffers"], graph)
    selection = {key: plan[key] for key in required}
    selection["buffers"] = sorted(selection["buffers"], key=lambda b: b["buffer_id"])
    return dict(plan_id=fingerprint(selection), pipeline_id=graph.manifest["pipeline_id"],
                input_fingerprint=graph.manifest["input_fingerprint"], requested_preload=requested,
                effective_preload=effective, buffers=buffers, status="annotation_only",
                optimization_applied=False, physical_feasibility="not_checked",
                schedule_legality="not_proven", local_schedule="off")


def _validate_buffers(rows, graph):
    require(isinstance(rows, list), "SCHEMA", "buffers must be an array")
    expected = {b["id"]: b for b in graph.buffers if b["multi_buffer_eligible"]}
    all_ids = {b["id"] for b in graph.buffers}
    selected = {}
    for row in rows:
        fields(row, ("buffer_id", "count"))
        key = row["buffer_id"]
        require(isinstance(key, str) and key in all_ids, "UNKNOWN_ID", f"unknown buffer_id: {key}")
        require(key in expected, "OWNERSHIP", f"{key} is not a local allocation")
        require(key not in selected, "DUPLICATE_ID", f"duplicate buffer_id: {key}")
        selected[key] = integer(row["count"], f"{key}.multi_buffer_count", 1)
        descriptor = expected[key]
        minimum = (selected[key] - 1) * descriptor["slot_stride_bytes"] + descriptor["allocation_bytes"]
        capacity = graph.profile["capacity_bytes"][descriptor["memory_space"]]
        require(minimum <= capacity, "CAPACITY", f"{key} alone exceeds the target budget")
    require(set(selected) == set(expected), "MISSING_BUFFER",
            f"missing local allocations: {sorted(set(expected) - set(selected))}")
    return [dict(buffer_id=key, count=value, op_id=expected[key]["op_id"],
                 allocation_bytes=expected[key]["allocation_bytes"])
            for key, value in sorted(selected.items())]


def annotate(graph, report):
    graph.index.clean_annotations()
    for key, op in graph.buffer_ops.items():
        graph.index.set_string(op, "pto.costmodel.buffer_id", key)
    for loop, info in zip(graph.loop_ops, graph.loops):
        graph.index.set_string(loop, "pto.costmodel.loop_id", info["id"])
        graph.index.set_integer(loop, "pto.cv_preload_count", report["requested_preload"])
    for row in report["buffers"]:
        graph.index.set_integer(graph.buffer_ops[row["buffer_id"]],
                                "pto.pipeline.multi_buffer_count", row["count"])
    graph.index.set_string(graph.index.module.operation, "pto.costmodel.plan_id", report["plan_id"])
    graph.index.set_string(graph.index.module.operation, "pto.costmodel.status", "annotation_only")


def annotate_ids(graph):
    graph.index.clean_annotations()
    for key, op in graph.buffer_ops.items():
        graph.index.set_string(op, "pto.costmodel.buffer_id", key)
    for loop, info in zip(graph.loop_ops, graph.loops):
        graph.index.set_string(loop, "pto.costmodel.loop_id", info["id"])
