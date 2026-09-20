# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.


"""Version negotiation, identity, and complete candidate configuration."""
from __future__ import annotations

import math
from copy import deepcopy

from pto_costmodel.wire import fields, fingerprint, integer, require

VERSION = "2.0"
SEMANTICS = "pto.static_cv.1"
SCHEDULE = "prefix_suffix_v1"
FEATURES = ["static_cv", "typed_operations", "buffer_ownership", "schedule_binding"]
MAX_TRIPS = 256
MAX_CANDIDATES = 32


def envelope(value, kind):
    require(isinstance(value, dict), "SCHEMA", "expected object")
    version = value.get("protocol_version")
    require(isinstance(version, str) and len(version.split(".")) == 2,
            "VERSION", "expected major.minor protocol version")
    major, minor = version.split(".")
    require(major == "2" and minor.isdigit(), "VERSION", "unsupported protocol major")
    require(value.get("kind") == kind, "SCHEMA", f"expected {kind}")
    required = value.get("required_features")
    require(isinstance(required, list) and all(isinstance(x, str) for x in required),
            "SCHEMA", "required_features must be strings")
    require(set(required) <= set(FEATURES), "UNSUPPORTED_FEATURE", str(required))
    extensions = value.get("extensions", {})
    require(isinstance(extensions, dict) and all("." in k for k in extensions),
            "SCHEMA", "extensions require namespaced keys")


def header(kind):
    return dict(protocol_version=VERSION, kind=kind, required_features=list(FEATURES))


def identity(package):
    return {key: package[key] for key in
            ("program_fingerprint", "bindings_fingerprint", "target_fingerprint",
             "checkpoint_version", "semantics_version")}


def configuration(program, preload, counts=None):
    locals_ = [b for b in program["buffers"] if b["multi_buffer_eligible"]]
    slots = {} if counts is None else counts
    return dict(pipeline_id=program["pipeline_id"], schedule_kind=SCHEDULE,
                preload_semantics="iteration_distance", preload_count=preload, local_schedule="off",
                buffers=[dict(buffer_id=b["id"], count=slots.get(b["id"], 1)) for b in locals_])


def validate_configuration(config, program):
    fields(config, ("pipeline_id", "schedule_kind", "preload_semantics", "preload_count",
                    "local_schedule", "buffers"))
    require(config["pipeline_id"] == program["pipeline_id"], "UNKNOWN_ID", "pipeline mismatch")
    require(config["schedule_kind"] == SCHEDULE and config["preload_semantics"] == "iteration_distance"
            and config["local_schedule"] == "off", "SCHEDULE", "unsupported schedule semantics")
    integer(config["preload_count"], "preload_count")
    expected = {b["id"] for b in program["buffers"] if b["multi_buffer_eligible"]}
    require(isinstance(config["buffers"], list), "SCHEMA", "buffers must be an array")
    seen = set()
    for row in config["buffers"]:
        fields(row, ("buffer_id", "count"))
        key = row["buffer_id"]
        require(isinstance(key, str) and key in expected, "OWNERSHIP", "unknown or non-local buffer")
        require(key not in seen, "DUPLICATE_ID", key)
        seen.add(key)
        integer(row["count"], "buffer count", 1)
        require(row["count"] <= MAX_TRIPS, "RANGE", "static slot expansion exceeds 256")
    require(seen == expected, "MISSING_BUFFER", str(sorted(expected - seen)))


def candidate_id(binding, config):
    selected = dict(config, buffers=sorted(config["buffers"], key=lambda row: row["buffer_id"]))
    return fingerprint(dict(identity=binding, configuration=selected))


def metric(value):
    fields(value, ("value", "unit"))
    require(value["unit"] == "us", "METRIC", "latency must use us")
    number = value["value"]
    require(number is None or (type(number) in (int, float) and math.isfinite(number) and number >= 0),
            "METRIC", "latency must be finite and nonnegative, or null")


def validate_model(model):
    fields(model, ("name", "revision", "adapter_version", "config_fingerprint"))
    require(all(isinstance(v, str) and v for v in model.values()), "SCHEMA", "model provenance missing")


def capabilities(adapter="ptoas"):
    return dict(**header("capabilities"), adapter=adapter, targets=["a5"],
                semantics_versions=[SEMANTICS], schedule_kinds=[SCHEDULE],
                limits=dict(max_trip_count=MAX_TRIPS, max_candidates=MAX_CANDIDATES),
                apply_modes=["annotation_only", "compile"], automatic_application=False)


def plan_from_result(package, candidate, model, result_fingerprint):
    return deepcopy(dict(**header("plan"), identity=package["identity"], candidate_id=candidate["candidate_id"],
                         configuration=candidate["configuration"],
                         schedule_fingerprint=candidate["schedule"]["fingerprint"],
                         model=model, result_fingerprint=result_fingerprint))


def make_request(package, candidates):
    require(candidates and candidates[0]["configuration"]["preload_count"] == 0,
            "BASELINE", "the first candidate must be the serial baseline")
    return dict(**header("request"), identity=package["identity"], candidates=candidates,
                baseline_candidate_id=candidates[0]["candidate_id"], objective="latency", memory_constraint="hard",
                execution_mode="recommendation", search_budget=dict(max_candidates=len(candidates)))
