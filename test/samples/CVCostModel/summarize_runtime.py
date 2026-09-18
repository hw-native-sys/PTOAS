# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.


"""Summarize verified mixed-kernel msprof samples without issuing G4 certification."""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import statistics


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def duration(runtime, symbol, device):
    paths = list((runtime / "profile").rglob("op_summary*.csv"))
    if len(paths) != 1:
        raise ValueError("expected one profiler operator summary")
    with paths[0].open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    if len(rows) != 1 or rows[0]["Op Name"] != symbol:
        raise ValueError("unexpected kernel identity or task count")
    row = rows[0]
    if (row["Task Type"] != "MIX_AIC" or int(row["Block Num"]) != 1
            or int(row["Mix Block Num"]) != 2 or int(row["Device_id"]) != device):
        raise ValueError("mixed 1 AIC + 2 AIV measurement boundary not established")
    value = float(row["Task Duration(us)"])
    if not math.isfinite(value) or value <= 0:
        raise ValueError("unknown/invalid profiler duration")
    return dict(duration_us=value, profiler_file=str(paths[0].relative_to(runtime)),
                profiler_sha256=sha(paths[0]))


def samples(root, report):
    grouped = {}
    for row in report["rows"]:
        case, variant = row["case"], row["variant"]
        if any(Path(s).name != s or s in (".", "..") for s in (case, variant)):
            raise ValueError("invalid case/variant path")
        label = f"{row['stage']}-seed{row['seed']}-r{row['repeat']}"
        runtime = root / "work" / case / variant / label
        saved = json.loads((runtime / "result.json").read_text())
        if saved != row or row["status"] != "pass" or sha(runtime / "output.bin") != row["output_sha256"]:
            raise ValueError("runtime evidence or output mismatch")
        for name, digest in row["input_sha256"].items():
            if Path(name).name != name or sha(runtime / name) != digest:
                raise ValueError("input/golden evidence mismatch")
        key = (case, variant)
        bucket = grouped.setdefault(key, dict(warmup=0, repeats=set(), samples=[]))
        if row["stage"] == "warmup":
            bucket["warmup"] += 1
        elif row["stage"] == "profile":
            if row["repeat"] in bucket["repeats"]:
                raise ValueError("duplicate profiler repeat")
            bucket["repeats"].add(row["repeat"])
            sample = duration(runtime, row["kernel_symbol"], report["device"])
            bucket["samples"].append(dict(repeat=row["repeat"], **sample))
        else:
            raise ValueError("unexpected performance stage")
    for bucket in grouped.values():
        if bucket["warmup"] != 1 or bucket["repeats"] != {1, 2, 3, 4, 5}:
            raise ValueError("one warmup and five profiles are required")
    return grouped


def summarize(grouped):
    rows = {}
    for key, bucket in grouped.items():
        values = [s["duration_us"] for s in bucket["samples"]]
        rows[key] = dict(case=key[0], variant=key[1], median_us=statistics.median(values),
                         min_us=min(values), max_us=max(values), stddev_us=statistics.pstdev(values),
                         samples=bucket["samples"])
    for (case, variant), row in rows.items():
        for base in ("serial", "p0"):
            baseline = rows[(case, base)]
            row["ratio_to_" + base] = row["median_us"] / baseline["median_us"]
            overlap = max(row["min_us"], baseline["min_us"]) <= min(row["max_us"], baseline["max_us"])
            row["comparison_to_" + base] = "self" if variant == base else (
                "cannot_reliably_distinguish" if overlap else "diagnostic_difference_only")
    return list(rows.values())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", type=Path, required=True)
    parser.add_argument("--matrix", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = json.loads((args.experiment / "results" / "performance.json").read_text())
    matrix = json.loads((args.matrix / "manifest.json").read_text())
    if report["status"] != "pass" or report["matrix_sha256"] != sha(args.matrix / "manifest.json"):
        raise ValueError("performance collection is incomplete or stale")
    grouped = samples(args.experiment, report)
    expected = {(c["case"], v["variant"]) for c in matrix["cases"] if c["count"] in (4, 8)
                for v in c["candidates"] if v["G2"] == "pass"}
    if set(grouped) != expected:
        raise ValueError("incomplete performance matrix")
    rows = summarize(grouped)
    result = dict(G4="not_run", certification=False, rows=rows,
                  analysis_tool_sha256=sha(Path(__file__)),
                  measurement_report_sha256=sha(args.experiment / "results" / "performance.json"),
                  source_commit=matrix["provenance"]["commit"], matrix_sha256=report["matrix_sha256"],
                  measurement_boundary="Task Duration(us): exactly one MIX_AIC, Block Num=1, Mix Block Num=2",
                  limitations=[report["limitation"], "five diagnostic samples do not establish certified ranking",
                               "no complete TileSim timing or prediction error assessment"])
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
    lines = ["# Diagnostic mixed C/V timing", "", "No G4 certification or automatic enablement.", "",
             "| Case | Variant | Median (us) | Range (us) | /serial | /P0 | vs P0 |",
             "|---|---|---:|---|---:|---:|---|"]
    for row in rows:
        lines.append(f"| {row['case']} | {row['variant']} | {row['median_us']:.3f} | "
                     f"{row['min_us']:.3f}–{row['max_us']:.3f} | {row['ratio_to_serial']:.3f} | "
                     f"{row['ratio_to_p0']:.3f} | {row['comparison_to_p0']} |")
    lines.extend(["", *result["limitations"]])
    (args.output / "summary.md").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
