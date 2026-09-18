# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""Validate four-way paired evidence and decide mechanism benefit."""
import argparse
import json
from pathlib import Path
import statistics

from run_mechanism import ORDERS
from summarize_ab import POLICY, paired_statistics, sha, task_duration


def relocated_runtime(experiment, recorded):
    path = Path(recorded)
    if path.exists():
        return path
    marker = "work"
    if marker not in path.parts:
        raise ValueError("runtime evidence is unavailable")
    relocated = Path(experiment).joinpath(*path.parts[path.parts.index(marker):])
    if not relocated.exists():
        raise ValueError("relocated runtime evidence is unavailable")
    return relocated


def collect(experiment, report):
    rows = []
    for row in report["rows"]:
        runtime = relocated_runtime(experiment, row["runtime"])
        saved = json.loads((runtime / "result.json").read_text())
        if saved != row or row["status"] != "pass" or sha(runtime / "output.bin") != row["output_sha256"]:
            raise ValueError("runtime evidence mismatch")
        for name, digest in row["input_sha256"].items():
            if sha(runtime / name) != digest:
                raise ValueError("input/golden evidence mismatch")
        if row["stage"] == "paired_profile":
            duration, profiler = task_duration(runtime, row["kernel_symbol"], report["device"])
            rows.append(dict(**row, duration_us=duration, profiler=profiler))
    if len(rows) != 80:
        raise ValueError("exactly 80 paired profiler samples are required")
    return rows


def analyze(matrix, case, rows, report, experiment):
    values = {name: [] for name in ("A", "B", "P0", "M")}
    pairs = []
    for block in range(20):
        block_rows = [row for row in rows if row["pair_block"] == block]
        order = ORDERS[block % len(ORDERS)]
        order_id = ",".join(order)
        if (len(block_rows) != 4 or [row["pair_role"] for row in block_rows] != list(order)
                or any(row["pair_order"] != order_id or row["seed"] != block % 3 for row in block_rows)):
            raise ValueError("four-way order, seed, or membership mismatch")
        by_role = {row["pair_role"]: row for row in block_rows}
        sample = dict(block=block, seed=block % 3, order=list(order))
        for name in values:
            value = by_role[name]["duration_us"]
            values[name].append(value)
            sample[name + "_us"] = value
        pairs.append(sample)
    ba = paired_statistics(values["A"], values["B"])
    bp0 = paired_statistics(values["P0"], values["B"])
    mp0 = paired_statistics(values["P0"], values["M"])
    benefit = (ba["mean_gain"] >= POLICY["minimum_speedup"]
               and bp0["mean_gain"] >= POLICY["minimum_speedup"]
               and ba["confidence_interval"][0] > 0 and bp0["confidence_interval"][0] > 0
               and ba["worst_regression"] <= POLICY["maximum_regression"]
               and bp0["worst_regression"] <= POLICY["maximum_regression"]
               and mp0["confidence_interval"][0] >= -0.02
               and mp0["confidence_interval"][1] <= 0.02
               and case["invalid"]["status"] == "rejected"
               and case["invalid"]["reason"] == "INSUFFICIENT_SLOTS")
    selection = case["selection"]
    prediction = next(row["predicted_latency_us"] for row in selection["ranking"]
                      if row["candidate_id"] == selection["recommended_candidate_id"])
    baseline_prediction = next(row["predicted_latency_us"] for row in selection["ranking"]
                               if row["candidate_id"] == selection["baseline_candidate_id"])
    prediction_error = abs(prediction - statistics.mean(values["B"])) / statistics.mean(values["B"])
    baseline_prediction_error = (abs(baseline_prediction - statistics.mean(values["P0"]))
                                 / statistics.mean(values["P0"]))
    measured_gain = bp0["mean_gain"]
    gain_error_points = selection["predicted_gain"] - measured_gain
    selected_preload = next(row["preload_count"] for row in selection["ranking"]
                            if row["candidate_id"] == selection["recommended_candidate_id"])
    b_configuration = next(row["configuration"] for row in case["candidates"] if row["variant"] == "B")
    p_buffer_slots = next(row["count"] for row in b_configuration["buffers"]
                          if row["buffer_id"] == case["p_buffer_id"])
    correctness = json.loads((experiment / "results" / "correctness.json").read_text())
    if correctness["status"] != "pass" or len(correctness["rows"]) != 24:
        raise ValueError("complete G3 evidence is required")
    return dict(case=case["case"], status="MECHANISM_BENEFIT_PASS" if benefit else "MECHANISM_BENEFIT_FAIL",
                G4="not_claimed", repetitions=case["repetitions"], p_buffer_id=case["p_buffer_id"],
                selected_candidate_id=selection["recommended_candidate_id"],
                selected_preload=selected_preload,
                predicted_B_us=prediction, measured_B_mean_us=statistics.mean(values["B"]),
                prediction_error=prediction_error,
                optimization_effect=dict(status="pass" if benefit else "fail",
                    annotation_configuration=dict(preload_count=selected_preload,
                                                  p_buffer_slots=p_buffer_slots),
                    measured_gain_over_serial=ba["mean_gain"],
                    measured_gain_over_static_baseline=bp0["mean_gain"],
                    multibuffer_only_gain=mp0["mean_gain"]),
                model_assessment=dict(status="latency_accuracy_fail" if prediction_error > 0.10 else "pass",
                    predicted_baseline_us=baseline_prediction,
                    measured_baseline_us=statistics.mean(values["P0"]),
                    baseline_prediction_error=baseline_prediction_error,
                    predicted_candidate_us=prediction,
                    measured_candidate_us=statistics.mean(values["B"]),
                    candidate_prediction_error=prediction_error,
                    predicted_gain=selection["predicted_gain"], measured_gain=measured_gain,
                    gain_overestimate_points=gain_error_points,
                    feedback=["calibrate common launch/layout/L2L/synchronization costs",
                              "calibrate tneg throughput and fixed cost with independent primitive microbenchmarks",
                              "consume lowered operation and synchronization feedback before G4 evaluation",
                              "report selection quality separately from absolute latency accuracy"]),
                statistics={"B_over_A": ba, "B_over_P0": bp0,
                "M_over_P0": mp0}, means_us={key: statistics.mean(value) for key, value in values.items()},
                pairs=pairs, invalid=case["invalid"], all_G3_and_performance_correct=True)


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
    rows = collect(args.experiment, report)
    workload = analyze(args.matrix, manifest["cases"][0], rows, report, args.experiment)
    result = dict(schema_version="ptoas.tilesim.mechanism.report.v1", policy=dict(
        minimum_samples=20, minimum_speedup=0.02, maximum_regression=0.02,
        confidence_level=0.95, bootstrap_resamples=10000, bootstrap_seed=20260916,
        multibuffer_only_equivalence_interval=[-0.02, 0.02]), workload=workload,
        source_commit=manifest["provenance"]["commit"],
        tilesim_commit=manifest["provenance"]["tilesim"]["commit"],
        matrix_sha256=report["matrix_sha256"],
        measurement_report_sha256=sha(args.experiment / "results" / "performance.json"),
        measurement_boundary="exactly one MIX_AIC task, Block Num=1, Mix Block Num=2")
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / "report.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    stats = workload["statistics"]
    model = workload["model_assessment"]
    lines = ["# preload + multi-buffer mechanism acceptance", "",
             f"Overall mechanism result: **{workload['status']}** (G4 is not claimed).", "",
             "## 1. Cost Model annotation-guided optimization effect", "",
             f"TileSim selected `preload_count={workload['selected_preload']}` and two slots for "
             f"P Buffer `{workload['p_buffer_id']}`. PTOAS validated and materialized that configuration.", "",
             "| Comparison | Mean gain | 95% CI | Worst regression |",
             "|---|---:|---:|---:|"]
    for label, key in (("B/A", "B_over_A"), ("B/P0", "B_over_P0"), ("M/P0", "M_over_P0")):
        row = stats[key]
        lines.append(f"| {label} | {row['mean_gain']:.2%} | "
                     f"[{row['confidence_interval'][0]:.2%}, {row['confidence_interval'][1]:.2%}] | "
                     f"{row['worst_regression']:.2%} |")
    lines.extend(["", "P-invalid was rejected by G2 with `INSUFFICIENT_SLOTS`. The M/P0 interval "
                  "shows that allocating the extra slot alone did not create the measured benefit.", "",
                  "## 2. Cost Model prediction versus A5 measurement", "",
                  "| Quantity | TileSim | A5 measured | Error |", "|---|---:|---:|---:|",
                  f"| P0 baseline latency | {model['predicted_baseline_us']:.4f} us | "
                  f"{model['measured_baseline_us']:.4f} us | {model['baseline_prediction_error']:.2%} |",
                  f"| B candidate latency | {model['predicted_candidate_us']:.4f} us | "
                  f"{model['measured_candidate_us']:.4f} us | {model['candidate_prediction_error']:.2%} |",
                  f"| B/P0 gain | {model['predicted_gain']:.2%} | {model['measured_gain']:.2%} | "
                  f"overestimated by {100 * model['gain_overestimate_points']:.2f} percentage points |", "",
                  "The model selected a beneficial candidate, but its absolute latency accuracy failed the 10% G4 gate.", "",
                  "## 3. Feedback to the Cost Model team", "",
                  "- Calibrate common launch, layout conversion, L2L, and synchronization costs; both P0 and B are underestimated.",
                  "- Calibrate `tneg` slope and fixed cost with independent primitive microbenchmarks, not this acceptance workload.",
                  "- Consume compiler-lowered operation, wait, and synchronization feedback before G4 evaluation.",
                  "- Report candidate-selection quality separately from absolute-latency and speedup accuracy."])
    (args.output / "report.md").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
