# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.


"""Compiler-backed exchange 2.0 contracts and actual candidate materialization."""
from copy import deepcopy
from pathlib import Path
import sys
import tempfile
import unittest

from ptoas._cv_exchange import export_v2
from ptoas._cv_session import apply_candidate, run_model, search_candidates
from ptoas._cv_cli import selected_plan
from pto_costmodel.contract import configuration, envelope, header, make_request, plan_from_result
from pto_costmodel.package import read_package
from pto_costmodel.reference import evaluate, model_info
from pto_costmodel.schedule import build_schedule
from pto_costmodel.validation import prepare_candidate, validate_result
from pto_costmodel.wire import ContractError, encode, read_json

SAMPLE = Path(__file__).resolve().parents[3] / "samples" / "CVCostModel"


class ExchangeV2Test(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temporary = tempfile.TemporaryDirectory(prefix="pto-v2-test-")
        cls.root = Path(cls.temporary.name)
        cls.profile = read_json(SAMPLE / "a5_profile.json")
        cls.source = SAMPLE / "four_stage_serial.pto"
        export_v2(cls.source, cls.profile, cls.root / "package")
        cls.package = read_package(cls.root / "package")
        bindings = deepcopy(cls.package["bindings"])
        bindings["alias_contract"] = "disjoint"
        export_v2(cls.source, cls.profile, cls.root / "disjoint", bindings)

    @classmethod
    def tearDownClass(cls):
        cls.temporary.cleanup()

    def output(self, name="result"):
        return self.root / (self.id().split(".")[-1] + "-" + name)

    def config(self, preload=1):
        return configuration(self.package["program"], preload)

    def reject(self, code, function, *args):
        with self.assertRaises(ContractError) as failure:
            function(*args)
        self.assertEqual(failure.exception.code, code)

    def test_typed_program(self):
        program = self.package["program"]
        self.assertEqual([l["trip_count"] for l in program["loops"]], [2, 2])
        self.assertEqual(len(program["pipes"]), 3)
        self.assertEqual(len([b for b in program["buffers"] if b["owner"] == "local"]), 8)
        matmul = next(o for o in program["operations"] if o["name"] == "pto.tmatmul")
        self.assertEqual(matmul["attributes"]["accPhase"], "unspecified")
        self.assertEqual(program["values"][matmul["operands"][0]]["shape"], [16, 16])
        self.assertNotIn("!pto", encode(program))

    def test_no_native_dependency(self):
        import subprocess
        sdk = Path(__file__).resolve().parents[4] / "ptodsl"
        result = subprocess.run([sys.executable, "-c", "import pto_costmodel.reference,sys; "
                                 "print('ptoas' in sys.modules)"], cwd=sdk, capture_output=True,
                                text=True, timeout=30, check=False)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout.strip(), "False")

    def test_schema_major_rejected(self):
        value = header("plan")
        value["protocol_version"] = "3.0"
        self.reject("VERSION", envelope, value, "plan")

    def test_minor_and_extensions(self):
        value = dict(header("plan"), protocol_version="2.1", extensions={"vendor.diagnostic": "ok"})
        envelope(value, "plan")

    def test_required_feature_rejected(self):
        value = header("plan")
        value["required_features"].append("dynamic_shapes")
        self.reject("UNSUPPORTED_FEATURE", envelope, value, "plan")

    def test_preload_saturation(self):
        a = build_schedule(self.package["program"], self.config(2))
        b = build_schedule(self.package["program"], self.config(3))
        self.assertEqual(a["effective_preload"], 2)
        self.assertEqual(a["core_order"], b["core_order"])
        self.assertNotEqual(a["fingerprint"], b["fingerprint"])

    def test_zero_and_one_trip(self):
        for count in (0, 1):
            program = deepcopy(self.package["program"])
            for loop in program["loops"]:
                loop["trip_count"] = count
            schedule = build_schedule(program, configuration(program, 3))
            self.assertEqual(len(schedule["stage_instances"]), 4 * count)
            self.assertEqual(schedule["effective_preload"], count)

    def test_prefix_before_suffix(self):
        program = deepcopy(self.package["program"])
        for loop in program["loops"]:
            loop["trip_count"] = 4
        schedule = build_schedule(program, configuration(program, 1))
        cube = [(s["stage"], s["iteration"]) for s in schedule["stage_instances"] if s["stage"].startswith("C_")]
        self.assertEqual(cube[:4], [("C_QK", 0), ("C_QK", 1), ("C_PV", 0), ("C_QK", 2)])

    def test_illegal_counts(self):
        for value in (0, -1, True, 257, 1.5):
            config = self.config()
            config["buffers"][0]["count"] = value
            self.reject("RANGE", prepare_candidate, self.package, config)

    def test_duplicate_and_missing_buffer(self):
        config = self.config()
        config["buffers"].append(config["buffers"][0])
        self.reject("DUPLICATE_ID", prepare_candidate, self.package, config)
        config = self.config()
        config["buffers"].pop()
        self.reject("MISSING_BUFFER", prepare_candidate, self.package, config)

    def test_borrowed_and_backing_not_local(self):
        for owner in ("borrowed_entry", "pipe_backing"):
            config = self.config()
            config["buffers"][0]["buffer_id"] = next(b["id"] for b in self.package["program"]["buffers"]
                                                       if b["owner"] == owner)
            self.reject("OWNERSHIP", prepare_candidate, self.package, config)

    def test_aggregate_overflow(self):
        package = deepcopy(self.package)
        package["target"]["compiler_budget"]["capacity_bytes"]["vec"] = 8192
        self.reject("CAPACITY", prepare_candidate, package, self.config())

    def test_fifo_progress(self):
        package = deepcopy(self.package)
        for pipe in package["program"]["pipes"]:
            pipe["effective_slot_num"] = 1
        self.reject("CAPACITY", prepare_candidate, package, self.config(2))

    def test_reference_result_binding(self):
        candidates, _ = search_candidates(self.package, [0, 1, 2])
        request = make_request(self.package, candidates)
        result = evaluate(self.package, request)
        validate_result(result, self.package, candidates)
        result["candidates"][0]["schedule_fingerprint"] = "wrong"
        self.reject("SCHEDULE", validate_result, result, self.package, candidates)

    def test_tilesim_selection_extension_is_strictly_bound(self):
        candidates, _ = search_candidates(self.package, [0, 1])
        result = evaluate(self.package, make_request(self.package, candidates))
        ranking = []
        for index, row in enumerate(result["candidates"]):
            row["coverage"] = dict(status="complete", operations="complete", communication="complete",
                                   approximations=[], unsupported=[])
            row["latency"] = dict(value=10.0 - index, unit="us")
            candidate = candidates[index]
            ranking.append(dict(rank=index + 1, candidate_id=row["candidate_id"],
                predicted_latency_us=row["latency"]["value"],
                total_memory_bytes=sum(value for spaces in row["resources"]["per_core"].values()
                                       for value in spaces.values()),
                preload_count=row["configuration"]["preload_count"],
                effective_preload=candidate["schedule"]["effective_preload"]))
        result["extensions"] = {"tilesim.selection.v1": dict(schema_version="tilesim.selection.v1",
            recommended_candidate_id=ranking[-1]["candidate_id"], action="optimize",
            baseline_candidate_id=ranking[0]["candidate_id"], predicted_gain=0.1,
            recommendation_threshold=0.02, tie_threshold=0.005, ranking=ranking, rejected=[],
            tie_break=["total_memory_bytes", "effective_preload", "candidate_id"])}
        validate_result(result, self.package, candidates)
        result["extensions"]["tilesim.selection.v1"]["ranking"][0]["candidate_id"] = "unknown"
        self.reject("SELECTION", validate_result, result, self.package, candidates)

    def test_partial_model_cannot_claim_latency(self):
        candidates, _ = search_candidates(self.package, [0])
        request = make_request(self.package, candidates)
        result = evaluate(self.package, request)
        result["candidates"][0]["latency"]["value"] = 0
        self.reject("COVERAGE", validate_result, result, self.package, candidates)

    def test_physical_address_injection(self):
        config = self.config()
        config["physical_address"] = 4096
        self.reject("SCHEMA", prepare_candidate, self.package, config)

    def test_plan_identity_and_schema(self):
        candidate = prepare_candidate(self.package, self.config())
        plan = deepcopy(plan_from_result(self.package, candidate, model_info(), "manual"))
        path = self.output("plan.json")
        path.write_text(encode(plan), encoding="utf-8")
        self.assertEqual(selected_plan(path, self.package), candidate)
        plan["identity"]["bindings_fingerprint"] = "stale"
        path.write_text(encode(plan), encoding="utf-8")
        # Mutate a copy to avoid changing the shared fixture.
        fresh = read_package(self.root / "package")
        self.reject("STALE_PLAN", selected_plan, path, fresh)
        self.package = fresh

    def test_model_process_and_plan_apply(self):
        sdk = Path(__file__).resolve().parents[4] / "ptodsl"
        adapter = dict(argv=[sys.executable, "-m", "pto_costmodel.reference"], cwd=str(sdk))
        output = self.output()
        report = run_model(self.root / "package", adapter, output, [0, 1])
        self.assertFalse(report["optimization_applied"])
        candidate = selected_plan(output / "plan-1.json", read_package(self.root / "package"))
        applied = apply_candidate(self.root / "package", candidate["configuration"], self.output("applied"))
        self.assertEqual(applied["status"], "annotation_only")

    def test_idempotent_annotations(self):
        apply_candidate(self.root / "package", self.config(), self.output("a"))
        apply_candidate(self.root / "package", self.config(), self.output("b"))
        self.assertEqual((self.output("a") / "annotated.pto").read_text(),
                         (self.output("b") / "annotated.pto").read_text())

    def test_alias_contract_required_for_compile(self):
        self.reject("ALIAS_UNKNOWN", apply_candidate, self.root / "package", self.config(), self.output(), "compile")
        self.assertFalse(self.output().exists())

    def test_compile_actual_slots(self):
        config = self.config(2)
        vector = [r for r in config["buffers"] if r["buffer_id"].startswith("vector.")]
        for row, count in zip(vector, (1, 2, 3)):
            row["count"] = count
        report = apply_candidate(self.root / "disjoint", config, self.output(), "compile")
        self.assertEqual(report["status"], "compiled_candidate")
        text = (self.output() / "candidate.pto").read_text()
        self.assertNotIn('"scf.for"', text)
        self.assertEqual(text.count('"pto.alloc_tile"'), 6)
        self.assertEqual(text.count('"pto.alloc_multi_tile"'), 2)
        self.assertEqual(text.count('"pto.multi_tile_get"'), 5)
        self.assertTrue((self.output() / "candidate.cpp").stat().st_size > 0)
        self.assertGreater(report["lowered_operation_counts"].get("pto.set_flag", 0), 0)
        self.assertGreater(report["lowered_operation_counts"].get("pto.wait_flag", 0), 0)

    def test_crossing_state_requires_versions(self):
        text = self.source.read_text().replace("pto.tadd ins(%pv,%pv", "pto.tadd ins(%pv,%a")
        text = text.replace("%c2 = arith.constant 2 : index", "%c2 = arith.constant 4 : index")
        source = self.output("crossing.pto")
        source.write_text(text, encoding="utf-8")
        path = self.output("package")
        export_v2(source, self.profile, path)
        package = read_package(path)
        self.reject("INSUFFICIENT_SLOTS", prepare_candidate, package, configuration(package["program"], 2))
        candidates, _ = search_candidates(package, [2])
        selected = next(c for c in candidates if c["configuration"]["preload_count"] == 2)
        self.assertEqual(max(selected["schedule"]["buffer_requirements"].values()), 3)

    def test_ssa_rename_stable(self):
        import re
        source = self.output("renamed.pto")
        source.write_text(re.sub(r"%([a-zA-Z][a-zA-Z0-9_]*)", r"%renamed_\1", self.source.read_text()),
                          encoding="utf-8")
        path = self.output("package")
        export_v2(source, self.profile, path)
        self.assertEqual(read_package(path)["identity"], read_package(self.root / "package")["identity"])

    def test_current_shape_changed(self):
        source = self.output("changed.pto")
        source.write_text(self.source.read_text().replace("%c2 = arith.constant 2 : index",
                                                         "%c2 = arith.constant 3 : index"), encoding="utf-8")
        self.reject("PACKAGE_INTEGRITY", apply_candidate, self.root / "package", self.config(),
                    self.output(), "annotation_only", source)

    def test_cache_roundtrip(self):
        sdk = Path(__file__).resolve().parents[4] / "ptodsl"
        adapter = dict(argv=[sys.executable, "-m", "pto_costmodel.reference"], cwd=str(sdk))
        cache = self.output("cache")
        first = run_model(self.root / "package", adapter, self.output("a"), [0, 1], cache=cache)
        second = run_model(self.root / "package", adapter, self.output("b"), [0, 1], cache=cache)
        self.assertFalse(first["prediction_cache_hit"])
        self.assertTrue(second["prediction_cache_hit"])

    def test_cross_core_gm_overlap(self):
        source = self.output("overlap.pto")
        source.write_text(self.source.read_text().replace("offsets=[%row,%c0]", "offsets=[%c0,%c0]"),
                          encoding="utf-8")
        path = self.output("package")
        export_v2(source, self.profile, path, deepcopy(read_package(self.root / "disjoint")["bindings"]))
        self.reject("ALIAS", apply_candidate, path, self.config(), self.output(), "compile")

    def test_gm_bounds(self):
        from pto_costmodel.semantics import verify_global_accesses
        package = deepcopy(read_package(self.root / "disjoint"))
        arg = package["program"]["functions"][-1]["arguments"][0]
        package["bindings"]["arguments"][arg]["shape"][0] = 8
        self.reject("BINDINGS", verify_global_accesses, package)

    def test_finite_metrics(self):
        candidates, _ = search_candidates(self.package, [0])
        request = make_request(self.package, candidates)
        for value in (-1, float("inf"), float("nan"), True):
            result = evaluate(self.package, request)
            result["candidates"][0]["latency"]["value"] = value
            self.reject("METRIC", validate_result, result, self.package, candidates)

    def test_tampered_package(self):
        import shutil
        path = self.output("package")
        shutil.copytree(self.root / "package", path)
        (path / "program.json").write_text("{}", encoding="utf-8")
        self.reject("PACKAGE_INTEGRITY", read_package, path)

    def test_model_timeout_or_failure_does_not_publish(self):
        adapter = dict(argv=[sys.executable, "-c", "raise SystemExit(3)"], cwd=str(self.root))
        self.reject("ADAPTER_FAILURE", run_model, self.root / "package", adapter, self.output())
        self.assertFalse(self.output().exists())

    def test_certification_requires_correctness_and_gain(self):
        from pto_costmodel.certification import certify
        policy = dict(minimum_samples=3, minimum_speedup=0.05, maximum_regression=0.01,
                      maximum_prediction_error=0.1, absolute_tolerance=0, relative_tolerance=0)
        # Synthetic evidence validates the gate only; it is not a measurement.
        evidence = dict(identity=self.package["identity"], candidate_id="fixture", schedule_fingerprint="fixture",
                        model=model_info(), device="synthetic", measurement_environment="unit_test",
                        baseline_artifact_fingerprint="fixture_base", candidate_artifact_fingerprint="fixture_opt",
                        predicted_latency_us=8, baseline_us=[10, 10, 10], candidate_us=[8, 8, 8],
                        runner_revision="fixture", correctness=dict(golden_fingerprint="fixture",
                            passed=True, absolute_tolerance=0, relative_tolerance=0,
                            deadlock_free=True, bounds_checked=True))
        self.assertEqual(certify(policy, evidence)["status"], "certified_exact_workload")
        evidence["candidate_us"] = [11, 11, 11]
        self.assertEqual(certify(policy, evidence)["status"], "not_certified")
        evidence["correctness"]["deadlock_free"] = False
        self.reject("CORRECTNESS", certify, policy, evidence)

    def test_paired_bootstrap_certification_is_reproducible(self):
        from pto_costmodel.certification import certify
        policy = dict(minimum_samples=3, minimum_speedup=0.02, maximum_regression=0.02,
                      maximum_prediction_error=0.1, confidence_level=0.95,
                      bootstrap_resamples=1000, bootstrap_seed=20260916,
                      absolute_tolerance=0, relative_tolerance=0)
        evidence = dict(identity=self.package["identity"], candidate_id="fixture", schedule_fingerprint="fixture",
                        model=model_info(), device="synthetic", measurement_environment="unit_test",
                        baseline_artifact_fingerprint="fixture_base", candidate_artifact_fingerprint="fixture_opt",
                        predicted_latency_us=8, baseline_us=[10, 10, 10], candidate_us=[8, 8, 8],
                        runner_revision="fixture", correctness=dict(golden_fingerprint="fixture", passed=True,
                            absolute_tolerance=0, relative_tolerance=0, deadlock_free=True, bounds_checked=True))
        first = certify(policy, evidence)
        second = certify(policy, evidence)
        self.assertEqual(first, second)
        self.assertGreater(first["metrics"]["bootstrap_confidence_interval"][0], 0)


if __name__ == "__main__":
    unittest.main()
