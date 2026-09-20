# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""Freeze TileSim recommendations, then build bound A/P0/B PTOAS artifacts."""
import argparse
import hashlib
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

from ptoas._cv_exchange import export_v2
from ptoas._cv_session import apply_candidate, run_model
from pto_costmodel.contract import configuration
from pto_costmodel.package import read_package
from pto_costmodel.wire import encode, read_json, require
from prepare_runtime import wrapper
from runtime_fixture import input_arrays, source_text


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def git_state(root):
    def git(*args):
        return subprocess.run(["git", "-C", str(root), *args], capture_output=True,
                              text=True, timeout=10, check=True).stdout
    paths = git("ls-files", "--cached", "--others", "--exclude-standard").splitlines()
    files = {name: sha(root / name) for name in paths if (root / name).is_file()}
    return dict(commit=git("rev-parse", "HEAD").strip(), dirty=bool(git("status", "--porcelain")),
                source_fingerprint=hashlib.sha256(encode(files).encode()).hexdigest())


def prepare_case(root, count, crossing, adapter, run_id):
    case = root / (("crossing" if crossing else "basic") + f"-n{count}")
    case.mkdir()
    source = case / "serial.pto"
    source.write_text(source_text(count, crossing), encoding="utf-8")
    profile = read_json(Path(__file__).with_name("a5_profile.json"))
    with tempfile.TemporaryDirectory(prefix="pto-ab-bindings-") as temporary:
        initial = Path(temporary) / "package"
        export_v2(source, profile, initial)
        bindings = read_package(initial)["bindings"]
    bindings["alias_contract"] = "disjoint"
    export_v2(source, profile, case / "package", bindings)
    package = read_package(case / "package")

    # This call must finish before any A5 result exists.  Its immutable output
    # is the sole source of the B configuration.
    run_model(case / "package", adapter, case / "model",
              preloads=sorted({0, 1, 2, count, count + 1}))
    request = read_json(case / "model" / "request.json")
    result = read_json(case / "model" / "result.json")
    report = read_json(case / "model" / "selection_report.json")
    selection = result.get("extensions", {}).get("tilesim.selection.v1")
    require(selection is not None and report["selection"] == selection, "SELECTION", "frozen selection missing")
    candidates = {row["candidate_id"]: row for row in request["candidates"]}
    require(selection["baseline_candidate_id"] == request["baseline_candidate_id"],
            "SELECTION", "baseline identity mismatch")
    baseline = candidates[selection["baseline_candidate_id"]]

    variants = [
        _variant(case, "A", baseline["configuration"], "compile_serial", run_id),
        _variant(case, "P0", baseline["configuration"], "compile", run_id),
    ]
    if selection["action"] == "optimize":
        selected = candidates[selection["recommended_candidate_id"]]
        plan = read_json(case / "model" / "selected-plan.json")
        require(plan["candidate_id"] == selected["candidate_id"]
                and plan["configuration"] == selected["configuration"]
                and plan["schedule_fingerprint"] == selected["schedule"]["fingerprint"],
                "SELECTION", "Plan/candidate/schedule mismatch")
        variants.append(_variant(case, "B", selected["configuration"], "compile", run_id))
    else:
        require(selection["recommended_candidate_id"] == baseline["candidate_id"],
                "SELECTION", "baseline retention chose a non-baseline candidate")

    for seed in (0, 1, 2):
        inputs = case / f"seed{seed}"
        inputs.mkdir()
        for name, data in input_arrays(count, seed, crossing).items():
            data.tofile(inputs / (name + ".bin"))
    binding = dict(schema_version="ptoas.tilesim.ab.v1", case=case.name,
                   identity=package["identity"], request_fingerprint=hashlib.sha256(
                       (case / "model" / "request.json").read_bytes()).hexdigest(),
                   result_fingerprint=hashlib.sha256((case / "model" / "result.json").read_bytes()).hexdigest(),
                   model=result["model"], selection=selection,
                   frozen_before_device_measurement=True, variants=variants,
                   status="BASELINE_RETAINED" if selection["action"] == "baseline" else "READY_FOR_G3")
    (case / "ab_manifest.json").write_text(encode(binding), encoding="utf-8")
    return dict(case=case.name, count=count, crossing=crossing, candidates=variants,
                ab_manifest_sha256=sha(case / "ab_manifest.json"), selection=selection,
                status=binding["status"])


def _variant(case, name, config, mode, run_id):
    output = case / name
    report = apply_candidate(case / "package", config, output, mode)
    require(report["validation"]["G2"] == "pass", "G2", "A/B artifact failed compiler validation")
    symbol = "cv_ab_" + hashlib.sha256((run_id + case.name + name).encode()).hexdigest()[:20]
    cpp = wrapper((output / "candidate.cpp").read_text(), symbol)
    (output / "kernel.cpp").write_text(cpp, encoding="utf-8")
    return dict(variant=name, G2="pass", symbol=symbol, candidate_id=report["plan_id"],
                schedule_fingerprint=report["schedule_fingerprint"], configuration=config,
                wrapper_sha256=hashlib.sha256(cpp.encode()).hexdigest())


def provenance(tilesim_root):
    import ptoas
    from ptoas._cv_compile import compiler_identity
    from ptoas._loader import ensure_core
    root = Path(__file__).resolve().parents[3]
    binary = Path(ensure_core().__file__)
    runtime_tools = {name: sha(Path(__file__).with_name(name)) for name in (
        "runtime_main.cpp", "build_runtime.py", "run_ab.py", "summarize_ab.py", "prepare_ab.py")}
    return dict(**git_state(root), compiler_identity=compiler_identity(), compiler_path=str(binary),
                python=sys.executable, python_version=sys.version, ptoas_path=str(ptoas.__file__),
                tilesim=git_state(tilesim_root), runtime_tools=runtime_tools)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--tilesim-root", type=Path, required=True)
    parser.add_argument("--tilesim-python", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    adapter = dict(argv=[str(args.tilesim_python), "-m", "core.frontend.adaptor.ptoas_exchange"],
                   cwd=str(args.tilesim_root), environment={"PYTHONPATH": str(Path(__file__).resolve().parents[3] / "ptodsl")})
    rows = [prepare_case(args.output, count, crossing, adapter, str(args.output.resolve()))
            for crossing in (False, True) for count in (4, 8)]
    prov = provenance(args.tilesim_root)
    require(not prov["dirty"] and not prov["tilesim"]["dirty"], "PROVENANCE",
            "freeze clean PTOAS and TileSim commits before preparing A/B")
    files = {str(path.relative_to(args.output)): sha(path)
             for path in sorted(args.output.rglob("*")) if path.is_file()}
    (args.output / "manifest.json").write_text(encode(dict(schema_version="ptoas.tilesim.ab.v1",
        cases=rows, files=files, provenance=prov, G3="not_run", G4="not_run")), encoding="utf-8")


if __name__ == "__main__":
    main()
