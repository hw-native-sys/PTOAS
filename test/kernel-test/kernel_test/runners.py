# coding=utf-8
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Shared correctness and cycle runners for the kernel-test framework."""

from __future__ import annotations

import os
import time
from collections.abc import Callable, Mapping

from .backends import ArtifactOutputs, BackendAdapter
from .pto_artifacts import materialize_artifact_plan
from .results import CaseResult, RunSummary


def run_correctness_suite(
    cases: Mapping[str, object],
    *,
    backend: BackendAdapter,
    verify_case: Callable[[str, object, object], CaseResult],
    print_prefix: str = "",
    flush: bool = True,
) -> RunSummary:
    """Run a correctness suite with stable PASS/FAIL/SKIP output."""

    total = 0
    passed = 0
    failed = 0
    skipped = 0

    for case_id in sorted(cases.keys()):
        total += 1
        case = cases[case_id]
        supported, reason = backend.is_supported(case, purpose="correctness")
        if not supported:
            skipped += 1
            result = CaseResult(
                ok=True,
                skipped=True,
                message=reason or "backend not wired for this case",
            )
        else:
            outputs = backend.launch(case, purpose="correctness")
            result = verify_case(case_id, case, outputs)
            if result.ok:
                passed += 1
            else:
                failed += 1

        status = "SKIP" if result.skipped else ("PASS" if result.ok else "FAIL")
        print(f"{print_prefix}[{case_id}] {status}: {result.message}", flush=flush)

    return RunSummary(total=total, passed=passed, failed=failed, skipped=skipped)


def _emit_backend_artifacts(
    backend: BackendAdapter,
    case_id: str,
    case: object,
    *,
    root_alias: bool,
) -> ArtifactOutputs | None:
    build_plan = getattr(backend, "build_artifact_plan", None)
    if build_plan is None:
        return None
    return materialize_artifact_plan(case_id, build_plan(case_id, case), root_alias=root_alias)


def run_artifact_suite(
    cases: Mapping[str, object],
    *,
    backend: BackendAdapter,
    print_prefix: str = "",
    flush: bool = True,
) -> RunSummary:
    """Generate PTO artifacts for one or more cases with stable output."""

    total = 0
    passed = 0
    failed = 0
    skipped = 0
    root_alias = len(cases) == 1

    for case_id in sorted(cases.keys()):
        total += 1
        case = cases[case_id]
        supported, reason = backend.is_supported(case, purpose="correctness")
        if not supported:
            skipped += 1
            result = CaseResult(
                ok=True,
                skipped=True,
                message=reason or "backend not wired for this case",
            )
        else:
            artifacts = _emit_backend_artifacts(backend, case_id, case, root_alias=root_alias)
            if artifacts is None:
                skipped += 1
                result = CaseResult(
                    ok=True,
                    skipped=True,
                    message=f"backend={backend.name} does not implement PTO artifact emission",
                )
            else:
                passed += 1
                result = CaseResult(ok=True, message=artifacts.message)

        status = "SKIP" if result.skipped else ("PASS" if result.ok else "FAIL")
        print(f"{print_prefix}[{case_id}] {status}: {result.message}", flush=flush)

    return RunSummary(total=total, passed=passed, failed=failed, skipped=skipped)


def format_cycle_fields(**fields: object) -> str:
    """Format key/value fields in a stable single-line marker payload."""

    return " ".join(f"{key}={value}" for key, value in fields.items())


def run_cycle_probe(
    *,
    case_id: str,
    case: object,
    backend: BackendAdapter,
    marker_fields: Mapping[str, object],
    flush_wait_env: str = "KERNEL_TEST_RECORD_FLUSH_WAIT",
    default_flush_wait_s: float = 0.0,
) -> int:
    """Run one cycle probe with consistent marker and skip output."""

    supported, reason = backend.is_supported(case, purpose="cycle")
    if not supported:
        payload = dict(marker_fields)
        if reason:
            payload["reason"] = reason
        print(f"CYCLE_SKIP {format_cycle_fields(**payload)}", flush=True)
        return 0

    marker = {"case": case_id, **dict(marker_fields)}
    rendered = format_cycle_fields(**marker)
    print(f"CYCLE_MARKER {rendered}", flush=True)
    backend.launch(case, purpose="cycle")
    print(f"CYCLE_DONE {rendered}", flush=True)

    wait_s = float(os.environ.get(flush_wait_env, str(default_flush_wait_s)))
    if wait_s > 0.0:
        time.sleep(wait_s)
    return 0
