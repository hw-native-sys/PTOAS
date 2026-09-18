# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""Freeze one stress recommendation and build bound A/P0/M/B artifacts."""
import argparse
from copy import deepcopy
import hashlib
from pathlib import Path
import shutil
import sys
import tempfile

from ptoas._cv_exchange import export_v2
from ptoas._cv_session import apply_candidate, run_model
from pto_costmodel.package import read_package
from pto_costmodel.wire import ContractError, encode, read_json, require
from prepare_ab import git_state, sha
from prepare_runtime import wrapper
from runtime_fixture import input_arrays, multibuffer_stress_source_text

REPETITIONS = (129, 257, 513)
PRELOADS = (0, 1, 2, 8, 9)


def export_package(source, output):
    profile = read_json(Path(__file__).with_name("a5_profile.json"))
    with tempfile.TemporaryDirectory(prefix="pto-mechanism-bindings-") as temporary:
        initial = Path(temporary) / "package"
        export_v2(source, profile, initial)
        bindings = read_package(initial)["bindings"]
    bindings["alias_contract"] = "disjoint"
    export_v2(source, profile, output, bindings)


def predict_choice(root, repetitions, adapter):
    attempt = root / f"r{repetitions}"
    attempt.mkdir(parents=True)
    source = attempt / "serial.pto"
    source.write_text(multibuffer_stress_source_text(repetitions), encoding="utf-8")
    export_package(source, attempt / "package")
    run_model(attempt / "package", adapter, attempt / "model", preloads=list(PRELOADS))
    request = read_json(attempt / "model" / "request.json")
    result = read_json(attempt / "model" / "result.json")
    selection = result.get("extensions", {}).get("tilesim.selection.v1")
    require(selection is not None, "SELECTION", "TileSim selection extension is required")
    candidates = {row["candidate_id"]: row for row in request["candidates"]}
    selected = candidates[selection["recommended_candidate_id"]]
    valid = (selection["action"] == "optimize"
             and selection["predicted_gain"] >= 0.10
             and selected["configuration"]["preload_count"] > 0
             and any(row["count"] > 1 for row in selected["configuration"]["buffers"]))
    return attempt, request, result, selection, selected, valid


def variant(case, name, config, mode, run_id):
    output = case / name
    report = apply_candidate(case / "package", config, output, mode)
    require(report["validation"]["G2"] == "pass", "G2", name + " failed compiler validation")
    symbol = "cv_mechanism_" + hashlib.sha256((run_id + case.name + name).encode()).hexdigest()[:16]
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
    names = ("runtime_main.cpp", "build_runtime.py", "prepare_mechanism.py",
             "run_mechanism.py", "summarize_mechanism.py", "runtime_fixture.py")
    return dict(**git_state(root), compiler_identity=compiler_identity(), compiler_path=str(binary),
                python=sys.executable, python_version=sys.version, ptoas_path=str(ptoas.__file__),
                tilesim=git_state(tilesim_root),
                runtime_tools={name: sha(Path(__file__).with_name(name)) for name in names})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--tilesim-root", type=Path, required=True)
    parser.add_argument("--tilesim-python", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    adapter = dict(argv=[str(args.tilesim_python), "-m", "core.frontend.adaptor.ptoas_exchange"],
                   cwd=str(args.tilesim_root), environment={"PYTHONPATH": str(
                       Path(__file__).resolve().parents[3] / "ptodsl")})

    with tempfile.TemporaryDirectory(prefix="pto-mechanism-model-") as temporary:
        design = Path(temporary)
        attempts, chosen = [], None
        for repetitions in REPETITIONS:
            attempt, request, result, selection, selected, valid = predict_choice(design, repetitions, adapter)
            attempts.append(dict(repetitions=repetitions, action=selection["action"],
                                 predicted_gain=selection["predicted_gain"],
                                 recommended_candidate_id=selection["recommended_candidate_id"],
                                 valid=valid))
            if valid:
                chosen = (repetitions, attempt, request, result, selection, selected)
                break
        if chosen is None:
            (args.output / "model_search.json").write_text(encode(dict(
                schema_version="ptoas.tilesim.mechanism.search.v1", status="MODEL_NO_OPTIMIZABLE_FIXTURE",
                attempts=attempts)), encoding="utf-8")
            raise ContractError("MODEL_NO_OPTIMIZABLE_FIXTURE", "no frozen repeat count predicts >=10%")
        repetitions, attempt, request, result, selection, selected = chosen
        case = args.output / f"multibuffer-stress-n8-r{repetitions}"
        shutil.copytree(attempt, case)

    package = read_package(case / "package")
    candidates = {row["candidate_id"]: row for row in request["candidates"]}
    baseline = candidates[selection["baseline_candidate_id"]]
    plan = read_json(case / "model" / "selected-plan.json")
    require(plan["candidate_id"] == selected["candidate_id"]
            and plan["configuration"] == selected["configuration"]
            and plan["schedule_fingerprint"] == selected["schedule"]["fingerprint"],
            "SELECTION", "Plan/candidate/schedule mismatch")

    vector_locals = [row["id"] for row in package["program"]["buffers"]
                     if row["owner"] == "local" and row["source_scope"] == "vector"]
    require(vector_locals, "FIXTURE", "P local Buffer is missing")
    p_buffer_id = vector_locals[0]
    selected_counts = {row["buffer_id"]: row["count"] for row in selected["configuration"]["buffers"]}
    require(selected_counts[p_buffer_id] > 1, "SELECTION", "recommended P Buffer is not multi-buffered")

    m_config = deepcopy(baseline["configuration"])
    next(row for row in m_config["buffers"] if row["buffer_id"] == p_buffer_id)["count"] = selected_counts[p_buffer_id]
    run_id = str(args.output.resolve())
    variants = [
        variant(case, "A", baseline["configuration"], "compile_serial", run_id),
        variant(case, "P0", baseline["configuration"], "compile", run_id),
        variant(case, "M", m_config, "compile", run_id),
        variant(case, "B", selected["configuration"], "compile", run_id),
    ]

    invalid = deepcopy(selected["configuration"])
    next(row for row in invalid["buffers"] if row["buffer_id"] == p_buffer_id)["count"] = 1
    try:
        apply_candidate(case / "package", invalid, case / "P-invalid", "compile")
    except ContractError as error:
        require(error.code == "INSUFFICIENT_SLOTS", "NEGATIVE", "unexpected invalid-candidate rejection")
        invalid_record = dict(status="rejected", reason=error.code, configuration=invalid)
    else:
        raise ContractError("NEGATIVE", "P-invalid unexpectedly passed G2")

    for seed in (0, 1, 2):
        inputs = case / f"seed{seed}"
        inputs.mkdir()
        for name, data in input_arrays(8, seed, True).items():
            data.tofile(inputs / (name + ".bin"))
    search = dict(schema_version="ptoas.tilesim.mechanism.search.v1", status="selected",
                  rule="smallest frozen repetition count with predicted gain >= 10%", attempts=attempts,
                  selected_repetitions=repetitions)
    (args.output / "model_search.json").write_text(encode(search), encoding="utf-8")
    binding = dict(schema_version="ptoas.tilesim.mechanism.v1", case=case.name,
                   identity=package["identity"], repetitions=repetitions, p_buffer_id=p_buffer_id,
                   request_fingerprint=sha(case / "model" / "request.json"),
                   result_fingerprint=sha(case / "model" / "result.json"), model=result["model"],
                   selection=selection, invalid=invalid_record, frozen_before_device_measurement=True,
                   variants=variants, status="READY_FOR_G3")
    (case / "mechanism_manifest.json").write_text(encode(binding), encoding="utf-8")
    prov = provenance(args.tilesim_root)
    require(not prov["dirty"] and not prov["tilesim"]["dirty"], "PROVENANCE",
            "freeze clean PTOAS and TileSim commits before preparing the mechanism matrix")
    files = {str(path.relative_to(args.output)): sha(path)
             for path in sorted(args.output.rglob("*")) if path.is_file()}
    manifest = dict(schema_version="ptoas.tilesim.mechanism.v1", cases=[dict(
        case=case.name, count=8, crossing=True, repetitions=repetitions, candidates=variants,
        selection=selection, p_buffer_id=p_buffer_id, invalid=invalid_record,
        mechanism_manifest_sha256=sha(case / "mechanism_manifest.json"), status="READY_FOR_G3")],
        files=files, provenance=prov, G3="not_run", mechanism_benefit="not_run")
    (args.output / "manifest.json").write_text(encode(manifest), encoding="utf-8")


if __name__ == "__main__":
    main()
