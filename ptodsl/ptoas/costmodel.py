# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.


"""PTOAS CV cost model file exchange; actual transformations are a later stage.

CLI: ptoas costmodel export INPUT --profile PROFILE --output PACKAGE
     ptoas costmodel import PACKAGE --plan PLAN --output RESULT [--input CURRENT]
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import subprocess
import sys
import tempfile

from ptoas.mlir import ir
from ptoas.mlir.dialects import pto
from ptoas._cv_common import ContractError, encode, fingerprint
from ptoas._cv_common import publish, read_json, read_text, require, validate_profile
from ptoas._cv_graph import Graph
from ptoas._cv_ir import attr
from ptoas._cv_plan import annotate, annotate_ids, validate_plan
from ptoas._loader import ensure_core


def run_native(arguments):
    """Invoke the native compiler belonging to this Python package."""
    package_root = Path(ensure_core().__file__).resolve().parent.parent
    environment = dict(os.environ)
    environment["PYTHONPATH"] = str(package_root) + os.pathsep + environment.get("PYTHONPATH", "")
    bootstrap = ("import sys; "
                 "sys.meta_path[:] = [f for f in sys.meta_path if 'editable' not in repr(f).lower()]; "
                 "from pathlib import Path; from ptoas import _cli; "
                 "raise SystemExit(_cli.launch(sys.argv[1:], wrapper=Path(_cli.__file__)))")
    result = subprocess.run([sys.executable, "-c", bootstrap] + list(arguments), env=environment,
                            capture_output=True, text=True, timeout=120, check=False)
    require(result.returncode == 0, "COMPILER", result.stderr)
    return result


def canonicalize(source):
    """Use the current package's native compiler, never a PATH-selected binary."""
    with tempfile.TemporaryDirectory(prefix="pto-cv-seam-") as tmp:
        output = Path(tmp) / "canonical.pto"
        run_native(["--pto-arch=a5", "--emit-cv-costmodel-ir", str(Path(source).resolve()), "-o", str(output)])
        return read_text(output)


def parse_graph(text, profile, bindings):
    module = ir.Module.parse(text)
    require(module.operation.verify(), "IR", "invalid canonical IR")
    require("pto.target_arch" in module.operation.attributes
            and ir.StringAttr(module.operation.attributes["pto.target_arch"]).value == "a5",
            "TARGET", "canonical IR must target A5")
    require(attr(module.operation, "pto.costmodel.checkpoint_version") == 1,
            "SCHEMA", "unsupported or missing native checkpoint version")
    graph = Graph(module, profile, bindings)
    return graph


def export_package(source, profile, bindings, output):
    """Normalize PTO and publish a versioned, fingerprint-bound input package."""
    validate_profile(profile)
    require(bindings == {}, "UNSUPPORTED", "v1 only accepts static pointer-only kernels")
    canonical = canonicalize(source)
    with ir.Context() as context:
        pto.register_dialect(context, load=True)
        graph = parse_graph(canonical, profile, bindings)
        annotate_ids(graph)
        publish(output, {"canonical.pto": graph.index.asm(),
                         "manifest.json": encode(graph.manifest),
                         "runtime_bindings.json": encode(bindings),
                         "target_profile.json": encode(profile)})
    return dict(status="exported", pipeline_id="cv0", input_fingerprint=graph.manifest["input_fingerprint"])


def import_plan(package, plan, output, current_input=None, profile=None, bindings=None):
    """Recompute package identity and attach a validated annotation-only plan."""
    package = Path(package).resolve()
    saved_manifest = read_json(package / "manifest.json")
    saved_profile = read_json(package / "target_profile.json")
    saved_bindings = read_json(package / "runtime_bindings.json")
    validate_profile(saved_profile)
    selected_profile = saved_profile if profile is None else profile
    selected_bindings = saved_bindings if bindings is None else bindings
    validate_profile(selected_profile)
    text = read_text(package / "canonical.pto")
    with ir.Context() as context:
        pto.register_dialect(context, load=True)
        graph = parse_graph(text, saved_profile, saved_bindings)
        require(saved_manifest == graph.manifest, "PACKAGE_INTEGRITY", "manifest does not match canonical IR")
        require(fingerprint(selected_profile) == saved_manifest["target_profile_fingerprint"],
                "STALE_PLAN", "target profile changed")
        require(fingerprint(selected_bindings) == saved_manifest["bindings_fingerprint"],
                "STALE_PLAN", "runtime bindings changed")
        if current_input is not None:
            graph = parse_graph(canonicalize(current_input), selected_profile, selected_bindings)
            require(graph.manifest["input_fingerprint"] == saved_manifest["input_fingerprint"],
                    "STALE_PLAN", "current input differs from the exported package")
        report = validate_plan(plan, graph)
        annotate(graph, report)
        require(graph.index.module.operation.verify(), "IR", "annotated IR verification failed")
        publish(output, {"annotated.pto": graph.index.asm(), "apply_report.json": encode(report)})
    return report


def main(argv=None):
    selected = sys.argv[1:] if argv is None else list(argv)
    if selected and selected[0] == "v2":
        from ptoas._cv_cli import main as v2_main
        return v2_main(selected[1:])
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    export = sub.add_parser("export")
    export.add_argument("input", type=Path)
    export.add_argument("--profile", type=Path, required=True)
    export.add_argument("--bindings", type=Path)
    export.add_argument("--output", type=Path, required=True)
    load = sub.add_parser("import")
    load.add_argument("package", type=Path)
    load.add_argument("--plan", type=Path, required=True)
    load.add_argument("--input", type=Path)
    load.add_argument("--profile", type=Path)
    load.add_argument("--bindings", type=Path)
    load.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        profile = read_json(args.profile) if args.profile else None
        bindings = read_json(args.bindings) if args.bindings else None
        if args.command == "export":
            report = export_package(args.input, profile, {} if bindings is None else bindings, args.output)
        else:
            report = import_plan(args.package, read_json(args.plan), args.output, args.input, profile, bindings)
        print(encode(report), end="")
        return 0
    except ContractError as exc:
        print(encode(dict(status="rejected", code=exc.code, message=str(exc))), file=sys.stderr, end="")
    except (OSError, UnicodeError, ValueError, ir.MLIRError, subprocess.TimeoutExpired) as exc:
        print(encode(dict(status="rejected", code="INPUT_OR_IO", message=str(exc))), file=sys.stderr, end="")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
