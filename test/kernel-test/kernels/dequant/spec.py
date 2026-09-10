# coding=utf-8
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Case listing and verification for dequant."""

from __future__ import annotations

from kernel_test.results import CaseResult

from .runtime import ENTRY_SYMBOL


def list_cases(workflow: str) -> dict[str, object]:
    """Return the dequant case matrix for the requested workflow."""

    if workflow == "cycle":
        return {}
    if workflow != "correctness":
        raise ValueError(f"unsupported dequant workflow: {workflow}")

    try:
        from .reference import generate_all
    except ModuleNotFoundError as exc:
        if exc.name not in {"numpy", "torch"}:
            raise
        cases = {
            f"{src_fmt}_{dst_fmt}": {
                "src_fmt": src_fmt,
                "scale_fmt": "e8m0",
                "dst_fmt": dst_fmt,
                "row_block_num": 4,
                "col_block_num": 4,
                "loop_num2vf": 1,
                "default_alias": src_fmt == "e4m3" and dst_fmt == "f32",
            }
            for src_fmt in ("e4m3", "e5m2")
            for dst_fmt in ("f32", "bf16", "f16")
        }
    else:
        cases = generate_all()

    for case in cases.values():
        case["entry_symbol"] = ENTRY_SYMBOL
    return cases


def verify_case(case_id: str, case: object, output: object) -> CaseResult:
    """Verify one dequant case against the CPU golden."""

    import numpy as np

    y_host = output["y"].cpu()
    if case["dst_fmt"] == "f32":
        got = y_host.numpy().astype(np.float32)
    else:
        got = y_host.float().numpy().astype(np.float32)
    expected = np.asarray(case["y_expected"], dtype=np.float32)
    max_diff = float(np.max(np.abs(got - expected)))
    tol = float(case.get("tolerance", 1e-3))
    if max_diff > tol:
        return CaseResult(
            ok=False,
            message=(
                f"maxDiff={max_diff:.6g} exceeds tol={tol:.6g} "
                f"for {case['src_fmt']}+{case['scale_fmt']}->{case['dst_fmt']}"
            ),
        )

    return CaseResult(
        ok=True,
        message=(
            f"{case['src_fmt']}+{case['scale_fmt']}->{case['dst_fmt']}: "
            f"maxDiff={max_diff:.6g}"
        ),
    )


def cycle_fields(case_id: str, case: object, backend: object) -> dict[str, object]:
    """Build stable case fields for future cycle support."""

    del case_id, backend
    return {
        "src": case["src_fmt"],
        "scale": case["scale_fmt"],
        "dst": case["dst_fmt"],
        "rb": case["row_block_num"],
        "cb": case["col_block_num"],
    }
