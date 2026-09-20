# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.


"""Compatibility boundary for the unchanged v1 exchange."""
from pto_costmodel.wire import (ContractError, MAX_BYTES, MAX_INTEGER, encode, fields,
                                fingerprint, integer, publish, read_json, read_text, require)

SCHEMA = "pto.cv_costmodel.v1"
OUTPUT_ATTRS = frozenset({
    "pto.cv_preload_count", "pto.pipeline.multi_buffer_count",
    "pto.costmodel.buffer_id", "pto.costmodel.loop_id",
    "pto.costmodel.plan_id", "pto.costmodel.status",
})

def validate_profile(profile):
    fields(profile, ("schema_version", "profile_id", "arch", "aic_count", "aiv_count",
                     "capacity_bytes", "alignment_bytes", "l2"))
    require(profile["schema_version"] == SCHEMA and profile["arch"] == "a5",
            "TARGET", "v1 requires an A5 profile")
    require(type(profile["aic_count"]) is int and profile["aic_count"] == 1
            and type(profile["aiv_count"]) is int and profile["aiv_count"] == 2,
            "TARGET", "v1 requires 1 AIC / 2 AIV")
    require(isinstance(profile["profile_id"], str) and bool(profile["profile_id"]),
            "SCHEMA", "profile_id must be a nonempty string")
    for table in ("capacity_bytes", "alignment_bytes"):
        fields(profile[table], ("mat", "left", "right", "acc", "vec"))
        for space, value in profile[table].items():
            integer(value, f"{table}.{space}", 1)
    for value in profile["alignment_bytes"].values():
        require(value & (value - 1) == 0, "TARGET", "alignment must be a power of two")
    fields(profile["l2"], ("policy", "capacity_bytes"))
    require(profile["l2"]["policy"] in ("off", "lru"), "TARGET", "unsupported L2 policy")
    integer(profile["l2"]["capacity_bytes"], "L2 capacity")
