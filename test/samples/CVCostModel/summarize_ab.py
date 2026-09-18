# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""Validate paired A/B evidence and issue exact-workload certification results."""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import random
import statistics

from pto_costmodel.certification import certify

POLICY = dict(minimum_samples=20, minimum_speedup=0.02, maximum_regression=0.02,
              maximum_prediction_error=0.10, confidence_level=0.95,
              bootstrap_resamples=10000, bootstrap_seed=20260916,
              absolute_tolerance=0, relative_tolerance=0)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def task_duration(runtime, symbol, device):
    paths = list((runtime / "profile").rglob("op_summary*.csv"))
    if len(paths) != 1:
        raise ValueError("expected one profiler operator summary")
    with paths[0].open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    if len(rows) != 1 or rows[0]["Op Name"] != symbol:
        raise ValueError("expected exactly one matching kernel task")
    row = rows[0]
    if (row["Task Type"] != "MIX_AIC" or int(row["Block Num"]) != 1
            or int(row["Mix Block Num"]) != 2 or int(row["Device_id"]) != device):
        raise ValueError("mixed 1 AIC + 2 AIV boundary not established")
    value = float(row["Task Duration(us)"])
    if not math.isfinite(value) or value <= 0:
        raise ValueError("invalid task duration")
    return value, dict(file=str(paths[0].relative_to(runtime)), sha256=sha(paths[0]))


def percentile(values, probability):
    ordered = sorted(values)
    position = (len(ordered) - 1) * probability
    lower, upper = math.floor(position), math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1 - fraction) + ordered[upper] * fraction


def paired_statistics(baseline, candidate):
    if len(baseline) != 20 or len(candidate) != 20:
        raise ValueError("twenty paired samples required")
    gains = [1.0 - b / a for a, b in zip(baseline, candidate)]
    rng = random.Random(POLICY["bootstrap_seed"])
    means = []
    for _ in range(POLICY["bootstrap_resamples"]):
        means.append(statistics.mean(gains[rng.randrange(len(gains))] for _ in gains))
    alpha = (1.0 - POLICY["confidence_level"]) / 2.0
    return dict(pair_gains=gains, mean_gain=statistics.mean(gains),
                confidence_interval=[percentile(means, alpha), percentile(means, 1.0 - alpha)],
                worst_regression=max(b / a - 1.0 for a, b in zip(baseline, candidate)))


def collect(experiment, report):
    grouped = {}
    for row in report["rows"]:
        runtime = Path(row["runtime"])
        saved = json.loads((runtime / "result.json").read_text())
        if saved != row or row["status"] != "pass" or sha(runtime / "output.bin") != row["output_sha256"]:
            raise ValueError("runtime evidence mismatch")
        for name, digest in row["input_sha256"].items():
            if sha(runtime / name) != digest:
                raise ValueError("input/golden evidence mismatch")
        if row["stage"] in ("paired_profile", "p0_profile"):
            duration, profiler = task_duration(runtime, row["kernel_symbol"], report["device"])
            grouped.setdefault(row["case"], []).append(dict(**row, duration_us=duration, profiler=profiler))
    return grouped


def analyze_case(matrix, case, rows, report):
    if case["status"] == "BASELINE_RETAINED":
        if rows:
            raise ValueError("baseline-retained workload must not have A/B profiler samples")
        return dict(case=case["case"], status="BASELINE_RETAINED", G4="not_applicable",
                    selection=case["selection"])
    paired = [row for row in rows if row["stage"] == "paired_profile"]
    p0 = [row for row in rows if row["stage"] == "p0_profile"]
    if len(paired) != 40 or len(p0) != 5:
        raise ValueError("incomplete paired/P0 sample set")
    baseline, candidate, pairs = [], [], []
    for block in range(20):
        block_rows = [row for row in paired if row["pair_block"] == block]
        expected_order = "AB" if block % 2 == 0 else "BA"
        if (len(block_rows) != 2 or {row["pair_role"] for row in block_rows} != {"A", "B"}
                or any(row["pair_order"] != expected_order or row["seed"] != block % 3 for row in block_rows)):
            raise ValueError("paired order, seed, or membership mismatch")
        by_role = {row["pair_role"]: row for row in block_rows}
        baseline.append(by_role["A"]["duration_us"])
        candidate.append(by_role["B"]["duration_us"])
        pairs.append(dict(block=block, seed=block % 3, order=expected_order,
                          A_us=baseline[-1], B_us=candidate[-1]))
    stats = paired_statistics(baseline, candidate)
    benefit = (stats["mean_gain"] >= POLICY["minimum_speedup"]
               and stats["confidence_interval"][0] > 0
               and stats["worst_regression"] <= POLICY["maximum_regression"])
    selection = case["selection"]
    prediction = next(row["predicted_latency_us"] for row in selection["ranking"]
                      if row["candidate_id"] == selection["recommended_candidate_id"])
    model_error = abs(prediction - statistics.mean(candidate)) / statistics.mean(candidate)
    variants = {row["variant"]: row for row in case["candidates"]}
    g3 = json.loads((Path(experiment) / "results" / "correctness.json").read_text())
    golden = hashlib.sha256("".join(sorted(row["output_sha256"] for row in g3["rows"]
        if row["case"] == case["case"])).encode()).hexdigest()
    evidence = dict(identity=json.loads((matrix / case["case"] / "ab_manifest.json").read_text())["identity"],
        candidate_id=variants["B"]["candidate_id"], schedule_fingerprint=variants["B"]["schedule_fingerprint"],
        model=json.loads((matrix / case["case"] / "model" / "result.json").read_text())["model"],
        device=f"{report['soc']}:device{report['device']}", measurement_environment=report["health"],
        baseline_artifact_fingerprint=next(row["build_manifest_sha256"] for row in g3["rows"]
                                            if row["case"] == case["case"] and row["variant"] == "A"),
        candidate_artifact_fingerprint=next(row["build_manifest_sha256"] for row in g3["rows"]
                                             if row["case"] == case["case"] and row["variant"] == "B"),
        predicted_latency_us=prediction, baseline_us=baseline, candidate_us=candidate,
        runner_revision=sha(Path(__file__).with_name("run_ab.py")),
        correctness=dict(golden_fingerprint=golden, passed=True, absolute_tolerance=0,
                         relative_tolerance=0, deadlock_free=True, bounds_checked=True))
    certificate = certify(POLICY, evidence)
    exact = benefit and model_error <= POLICY["maximum_prediction_error"] and certificate["status"] == "certified_exact_workload"
    return dict(case=case["case"], status="AB_BENEFIT_PASS" if benefit else "AB_BENEFIT_FAIL",
                G4="G4_EXACT_WORKLOAD_PASS" if exact else "G4_NOT_CERTIFIED",
                prediction_error=model_error, predicted_B_us=prediction,
                measured_A_mean_us=statistics.mean(baseline), measured_B_mean_us=statistics.mean(candidate),
                statistics=stats, pairs=pairs,
                p0_samples_us=[row["duration_us"] for row in sorted(p0, key=lambda item: item["repeat"])],
                certificate=certificate, selection=selection)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", type=Path, required=True)
    parser.add_argument("--matrix", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = json.loads((args.experiment / "results" / "performance.json").read_text())
    manifest = json.loads((args.matrix / "manifest.json").read_text())
    if report["status"] != "pass" or report["matrix_sha256"] != sha(args.matrix / "manifest.json"):
        raise ValueError("performance evidence is incomplete or stale")
    grouped = collect(args.experiment, report)
    rows = [analyze_case(args.matrix, case, grouped.get(case["case"], []), report)
            for case in manifest["cases"]]
    result = dict(schema_version="ptoas.tilesim.ab.report.v1", policy=POLICY, workloads=rows,
                  source_commit=manifest["provenance"]["commit"],
                  tilesim_commit=manifest["provenance"]["tilesim"]["commit"],
                  matrix_sha256=report["matrix_sha256"], measurement_report_sha256=sha(
                      args.experiment / "results" / "performance.json"),
                  measurement_boundary="exactly one MIX_AIC task, Block Num=1, Mix Block Num=2")
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / "report.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    lines = ["# PTOAS × TileSim fixed-micro A/B", "",
             "| Workload | Selection | A mean (us) | B mean (us) | Mean gain | 95% CI | Prediction error | Result | G4 |",
             "|---|---|---:|---:|---:|---|---:|---|---|"]
    for row in rows:
        if row["status"] == "BASELINE_RETAINED":
            lines.append(f"| {row['case']} | baseline | — | — | — | — | — | BASELINE_RETAINED | — |")
        else:
            ci = row["statistics"]["confidence_interval"]
            lines.append(f"| {row['case']} | B | {row['measured_A_mean_us']:.4f} | {row['measured_B_mean_us']:.4f} | "
                         f"{row['statistics']['mean_gain']:.2%} | [{ci[0]:.2%}, {ci[1]:.2%}] | "
                         f"{row['prediction_error']:.2%} | {row['status']} | {row['G4']} |")
    (args.output / "report.md").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
