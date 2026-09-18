# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.


"""Version 2.0 CLI; independent from the legacy v1 command grammar."""
import argparse
from pathlib import Path
import subprocess
import sys

from ptoas.mlir import ir
from ptoas._cv_exchange import export_v2
from ptoas._cv_session import apply_candidate, run_model
from pto_costmodel.contract import capabilities, envelope, validate_model
from pto_costmodel.package import read_package
from pto_costmodel.validation import prepare_candidate
from pto_costmodel.wire import ContractError, encode, fields, read_json, require


def parser():
    root = argparse.ArgumentParser(description=__doc__)
    sub = root.add_subparsers(dest="action", required=True)
    sub.add_parser("capabilities")
    certification = sub.add_parser("certify")
    certification.add_argument("--policy", type=Path, required=True)
    certification.add_argument("--evidence", type=Path, required=True)
    certification.add_argument("--output", type=Path, required=True)
    export = sub.add_parser("export")
    export.add_argument("input", type=Path)
    export.add_argument("--profile", type=Path, required=True)
    export.add_argument("--bindings", type=Path)
    export.add_argument("--output", type=Path, required=True)
    for name in ("evaluate", "propose", "validate", "apply"):
        cmd = sub.add_parser(name)
        cmd.add_argument("package", type=Path)
        cmd.add_argument("--plan", type=Path, required=name in ("validate", "apply"))
        if name in ("evaluate", "propose"):
            cmd.add_argument("--adapter", type=Path, required=True)
            cmd.add_argument("--preloads", type=int, nargs="+", default=[0, 1, 2])
            cmd.add_argument("--cache", type=Path)
        if name == "apply":
            cmd.add_argument("--mode", choices=("annotation_only", "compile", "compile_serial"),
                             default="annotation_only")
            cmd.add_argument("--input", type=Path)
        if name != "validate":
            cmd.add_argument("--output", type=Path, required=True)
    return root


def selected_plan(path, package):
    plan = read_json(path)
    envelope(plan, "plan")
    fields(plan, ("protocol_version", "kind", "required_features", "identity", "candidate_id", "configuration",
                  "schedule_fingerprint", "model", "result_fingerprint"), ("extensions",))
    validate_model(plan["model"])
    require(plan.get("identity") == package["identity"], "STALE_PLAN", "plan identity mismatch")
    candidate = prepare_candidate(package, plan["configuration"])
    require(plan.get("candidate_id") == candidate["candidate_id"], "CANDIDATE", "candidate identity mismatch")
    supplied = plan["schedule_fingerprint"]
    require(supplied == candidate["schedule"]["fingerprint"], "SCHEDULE", "candidate schedule mismatch")
    return candidate


def dispatch(args):
    if args.action == "capabilities":
        return capabilities()
    if args.action == "certify":
        from pto_costmodel.certification import certify
        from pto_costmodel.wire import publish
        policy, evidence = read_json(args.policy), read_json(args.evidence)
        report = certify(policy, evidence)
        publish(args.output, {"certification.json": encode(report), "policy.json": encode(policy),
                              "measurement_evidence.json": encode(evidence)})
        return report
    if args.action == "export":
        return export_v2(args.input, read_json(args.profile), args.output,
                         None if args.bindings is None else read_json(args.bindings))
    package = read_package(args.package)
    candidate = selected_plan(args.plan, package) if args.plan else None
    if args.action == "validate":
        from ptoas._cv_exchange import verified_graph
        from ptoas.mlir.dialects import pto
        with ir.Context() as context:
            pto.register_dialect(context, load=True)
            verified_graph(args.package, package)
        return dict(status="validated", candidate_id=candidate["candidate_id"], validation_scope="G1")
    if args.action == "apply":
        return apply_candidate(args.package, candidate["configuration"], args.output, args.mode, args.input)
    require(args.action != "evaluate" or candidate is not None, "SCHEMA", "evaluate requires --plan")
    return run_model(args.package, read_json(args.adapter), args.output, args.preloads,
                     None if candidate is None else candidate["configuration"], args.cache)


def main(argv=None):
    args = parser().parse_args(argv)
    try:
        print(encode(dispatch(args)), end="")
        return 0
    except ContractError as exc:
        report = dict(status="rejected", code=exc.code, message=str(exc))
    except (OSError, ValueError, TypeError, KeyError, ir.MLIRError, subprocess.TimeoutExpired) as exc:
        report = dict(status="rejected", code="INPUT_OR_IO", message=str(exc))
    print(encode(report), end="", file=sys.stderr)
    return 1
