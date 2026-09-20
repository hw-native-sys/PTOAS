# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.


"""Reproducible candidate evaluation, strict application, and baseline fallback."""
from pathlib import Path
import tempfile

from ptoas.mlir import ir
from ptoas.mlir.dialects import pto
from ptoas._cv_exchange import verified_graph
from ptoas._cv_plan import annotate
from pto_costmodel.contract import MAX_CANDIDATES, MAX_TRIPS, configuration, make_request, plan_from_result
from pto_costmodel.package import read_package
from pto_costmodel.runner import invoke, negotiate
from pto_costmodel.schedule import build_schedule
from pto_costmodel.validation import prepare_candidate, validate_result
from pto_costmodel.wire import ContractError, encode, fingerprint, integer, publish, read_json, require


def search_candidates(package, preloads):
    require(isinstance(preloads, list) and 0 < len(preloads) <= MAX_CANDIDATES,
            "RANGE", "provide 1..32 preload candidates")
    result, rejected = [], []
    selected_preloads = sorted(set([0] + [integer(p, "preload") for p in preloads]))
    require(len(selected_preloads) <= MAX_CANDIDATES, "RANGE", "candidate budget includes baseline")
    for preload in selected_preloads:
        program = package["program"]
        generous = {b["id"]: MAX_TRIPS for b in program["buffers"] if b["multi_buffer_eligible"]}
        try:
            schedule = build_schedule(program, configuration(program, preload, generous))
            selected = configuration(program, preload, schedule["buffer_requirements"])
            result.append(prepare_candidate(package, selected))
        except ContractError as exc:
            if exc.code not in ("CAPACITY", "FIFO_DEADLOCK", "INSUFFICIENT_SLOTS"):
                raise
            rejected.append(dict(preload=preload, code=exc.code, message=str(exc)))
    require(any(c["configuration"]["preload_count"] == 0 for c in result), "BASELINE", "baseline is infeasible")
    return result, rejected


def run_model(path, adapter, output, preloads=None, config=None, cache=None):
    package = read_package(path)
    with ir.Context() as context:
        pto.register_dialect(context, load=True)
        verified_graph(path, package)
    capabilities = negotiate(adapter)
    if config is not None:
        candidates, rejected = search_candidates(package, [0])
        selected = prepare_candidate(package, config)
        if selected["candidate_id"] != candidates[0]["candidate_id"]:
            candidates.append(selected)
    else:
        candidates, rejected = search_candidates(package, [0, 1, 2] if preloads is None else preloads)
    action = "evaluate" if config is not None else "propose"
    request = make_request(package, candidates)
    result, cache_hit = _evaluate_cached(adapter, action, path, request, capabilities["model"], cache)
    validate_result(result, package, candidates)
    require(result["model"] == capabilities["model"], "MODEL_CHANGED", "model changed after capability negotiation")
    report = dict(status="recommendation_only", selected="baseline", optimization_applied=False,
                  reason="no_performance_certification", rejected_candidates=rejected,
                  input_identity=package["identity"], model=result["model"], prediction_cache_hit=cache_hit)
    files = {"request.json": encode(request), "result.json": encode(result),
             "capabilities.json": encode(capabilities), "selection_report.json": encode(report)}
    plans = {}
    for index, candidate in enumerate(candidates):
        plan = plan_from_result(package, candidate, result["model"], fingerprint(result))
        plans[candidate["candidate_id"]] = plan
        files[f"plan-{index}.json"] = encode(plan)
    selection = result.get("extensions", {}).get("tilesim.selection.v1")
    if selection is not None:
        selected_id = selection["recommended_candidate_id"]
        report.update(status="frozen_recommendation", selected=selected_id,
                      action=selection["action"], reason="tilesim.selection.v1",
                      selection=selection, result_fingerprint=fingerprint(result))
        files["selected-plan.json"] = encode(plans[selected_id])
        files["selection_report.json"] = encode(report)
    publish(output, files)
    if cache is not None:
        key = fingerprint(dict(request=request, model=result["model"]))
        destination = Path(cache) / "predictions" / key
        if not destination.exists():
            publish(destination, {"result.json": encode(result), "request.json": encode(request)})
    return report


def _evaluate_cached(adapter, action, path, request, model, cache):
    key = fingerprint(dict(request=request, model=model))
    if cache is not None:
        directory = Path(cache) / "predictions" / key
        if directory.exists():
            require(read_json(directory / "request.json") == request, "CACHE_INTEGRITY", "cached request mismatch")
            return read_json(directory / "result.json"), True
    with tempfile.TemporaryDirectory(prefix="pto-cv-request-") as tmp:
        request_path = Path(tmp) / "request.json"
        request_path.write_text(encode(request), encoding="utf-8")
        return invoke(adapter, action, path, request_path), False


def apply_candidate(path, config, output, mode="annotation_only", current_input=None):
    require(mode in ("annotation_only", "compile", "compile_serial"), "MODE", "unknown application mode")
    package = read_package(path)
    candidate = prepare_candidate(package, config)
    with ir.Context() as context:
        pto.register_dialect(context, load=True)
        graph = verified_graph(path, package, current_input)
        report = dict(plan_id=candidate["candidate_id"], requested_preload=config["preload_count"],
                      buffers=[dict(buffer_id=r["buffer_id"], count=r["count"]) for r in config["buffers"]],
                      status="annotation_only", optimization_applied=False, physical_feasibility="not_checked",
                      identity=package["identity"], schedule_fingerprint=candidate["schedule"]["fingerprint"],
                      validation=dict(G1="pass", G2="not_run", G3="not_run", G4="not_run"))
        annotate(graph, report)
        files = {"annotated.pto": graph.index.asm(), "candidate.json": encode(candidate)}
        if mode in ("compile", "compile_serial"):
            from ptoas._cv_compile import compile_candidate
            files.update(compile_candidate(graph, package, candidate, report, serial=mode == "compile_serial"))
        files["apply_report.json"] = encode(report)
        publish(output, files)
    return report
