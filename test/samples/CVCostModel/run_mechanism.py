# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""Build and replay a frozen A/P0/M/B mechanism matrix on one A5."""
import argparse
import json
from pathlib import Path

from run_ab import build_all, check_matrix, run_one, sha, write

ORDERS = (("A", "B", "P0", "M"), ("B", "A", "M", "P0"),
          ("P0", "M", "A", "B"), ("M", "P0", "B", "A"))


def correctness_order(manifest):
    for case in manifest["cases"]:
        variants = {row["variant"]: row for row in case["candidates"]}
        forward = [variants[name] for name in ("B", "M", "P0", "A")]
        for repeat, ordered in enumerate((forward, list(reversed(forward)))):
            for seed in (0, 1, 2):
                for variant in ordered:
                    yield case, variant, seed, repeat, "correctness", None, None


def performance_order(manifest):
    for case in manifest["cases"]:
        variants = {row["variant"]: row for row in case["candidates"]}
        for name in ("A", "B", "P0", "M"):
            yield case, variants[name], 0, 0, "warmup", None, name
        for block in range(20):
            order = ORDERS[block % len(ORDERS)]
            order_id = ",".join(order)
            for name in order:
                yield case, variants[name], block % 3, block, "paired_profile", order_id, name


def run_matrix(args, manifest):
    performance = args.mode == "performance"
    if performance:
        accepted = json.loads(args.g3_report.read_text())
        if (accepted["status"] != "pass"
                or accepted["matrix_sha256"] != sha(args.matrix / "manifest.json")
                or len(accepted["rows"]) != 24):
            raise ValueError("performance requires matching complete 24-run G3 evidence")
        if accepted["device"] != args.device or accepted["soc"] != args.soc:
            raise ValueError("do not combine devices or SOCs")
        for row in accepted["rows"]:
            path = args.artifacts / row["case"] / row["variant"] / "build_manifest.json"
            if sha(path) != row["build_manifest_sha256"]:
                raise ValueError("G3 and performance build identities differ")
    report = dict(status="running", mode=args.mode, device=args.device, soc=args.soc,
                  matrix_sha256=sha(args.matrix / "manifest.json"), rows=[],
                  health="ACL_RUNTIME_AVAILABLE", npu_smi="unavailable", exclusive_task_queue=True,
                  limitation="no full health telemetry or occupancy proof for non-cooperating processes")
    path = args.experiment / "results" / (args.mode + ".json")
    if path.exists():
        raise ValueError("never overwrite a previous experiment report")
    write(path, report)
    hashes = {}
    sequence = performance_order(manifest) if performance else correctness_order(manifest)
    try:
        for item in sequence:
            case, variant, seed, repeat, stage, order, role = item
            print(stage, case["case"], variant["variant"], seed, repeat, order or "", flush=True)
            row = run_one(args, *item)
            key = (row["case"], row["variant"], seed)
            if key in hashes and hashes[key] != row["output_sha256"]:
                raise ValueError("unstable repeated output")
            hashes[key] = row["output_sha256"]
            report["rows"].append(row)
            write(path, report)
        check_matrix(args.matrix)
        expected = 84 if performance else 24
        if len(report["rows"]) != expected:
            raise ValueError(f"expected {expected} completed runs")
        report["status"] = "pass"
    except Exception as error:
        report.update(status="fail", error=str(error))
        write(path, report)
        raise
    write(path, report)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("build", "correctness", "performance"), required=True)
    parser.add_argument("--matrix", type=Path, required=True)
    parser.add_argument("--artifacts", type=Path, required=True)
    parser.add_argument("--experiment", type=Path, required=True)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--soc", required=True)
    parser.add_argument("--g3-report", type=Path)
    args = parser.parse_args()
    if args.mode == "performance" and args.g3_report is None:
        parser.error("--g3-report is required for performance")
    manifest = check_matrix(args.matrix)
    provenance = manifest["provenance"]
    if provenance["dirty"] or provenance["tilesim"]["dirty"]:
        raise ValueError("matrix requires clean frozen PTOAS and TileSim sources")
    if provenance["runtime_tools"]["run_mechanism.py"] != sha(Path(__file__)):
        raise ValueError("matrix requires its frozen mechanism runner")
    if args.mode == "build":
        build_all(args.matrix, args.artifacts, manifest)
    else:
        run_matrix(args, manifest)


if __name__ == "__main__":
    main()
