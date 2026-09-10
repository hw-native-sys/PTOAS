# coding=utf-8
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Case listing and verification for the rope kernel."""

from __future__ import annotations

import os

from kernel_test.results import CaseResult

from .tile_config import DEFAULT_TILE, DTYPES, MODES, TOLERANCE, TileConfig


def tile_from_env() -> TileConfig | None:
    """Build a rope tile override from environment variables if present."""

    s = os.environ.get("ROPE_VF_S")
    n = os.environ.get("ROPE_VF_N")
    if s is None or n is None:
        return None
    return TileConfig(name="env", s=int(s), n=int(n))


def default_tile_from_env() -> TileConfig:
    """Return the configured tile, falling back to the default rope tile."""

    return tile_from_env() or DEFAULT_TILE


def _lightweight_cases(tile: TileConfig) -> dict[str, object]:
    return {
        f"{dtype}_{mode}": {"mode": mode, "dtype": dtype, "tile": tile}
        for dtype in DTYPES
        for mode in MODES
    }


def list_cases(workflow: str) -> dict[str, object]:
    """Return the rope case matrix for the requested workflow."""

    if workflow not in {"correctness", "cycle"}:
        raise ValueError(f"unsupported rope workflow: {workflow}")

    tile = default_tile_from_env()
    try:
        from .reference import generate_all
    except ModuleNotFoundError as exc:
        if exc.name not in {"numpy", "torch"}:
            raise
        return _lightweight_cases(tile)

    return generate_all(tile=tile)


def verify_case(case_id: str, case: object, output: object) -> CaseResult:
    """Verify a rope backend output against the generated golden tensors."""

    y_host = output.cpu()
    if case["dtype"] == "bf16":
        got = y_host.float().numpy()
    else:
        got = y_host.numpy()

    max_diff = float(abs(got.astype("float32") - case["y"].astype("float32")).max())
    tile = case["tile"]
    message = (
        f"{case['dtype']}/{case['mode']}: maxDiff={max_diff:.6f} "
        f"tile=s{tile.s}_n{tile.n}"
    )
    return CaseResult(
        ok=max_diff < TOLERANCE[case["dtype"]],
        message=message,
    )


def cycle_fields(case_id: str, case: object, backend: object) -> dict[str, object]:
    """Build stable cycle marker fields for one rope case."""

    del backend, case_id
    tile = case["tile"]
    return {
        "mode": case["mode"],
        "dtype": case["dtype"],
        "s": tile.s,
        "n": tile.n,
        "vf_inner": tile.vf_inner_iters,
    }
