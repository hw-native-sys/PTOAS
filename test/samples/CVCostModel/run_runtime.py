# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.


"""Replay a frozen matrix on one exclusively queued A5 device, fail fast."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shlex
import shutil
import signal
import subprocess
import sys
import time

from build_runtime import build, sha, verify_artifacts


def write(path, value):
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def check_matrix(matrix):
    manifest = json.loads((matrix / "manifest.json").read_text())
    for name, digest in manifest["files"].items():
        if sha(matrix / name) != digest:
            raise ValueError("input fingerprint mismatch: " + name)
    return manifest


def build_all(matrix, artifacts, manifest):
    for case in manifest["cases"]:
        for variant in case["candidates"]:
            if variant["G2"] == "pass":
                print("BUILD", case["case"], variant["variant"], flush=True)
                build(matrix / case["case"] / variant["variant"], artifacts / case["case"] / variant["variant"])


def execution_order(manifest, performance):
    for case in manifest["cases"]:
        variants = [v for v in case["candidates"] if v["G2"] == "pass"]
        if performance:
            if case["count"] not in (4, 8):
                continue
            for variant in variants:
                for repeat in range(6):
                    yield case, variant, 0, repeat, "warmup" if repeat == 0 else "profile"
        else:
            # First round proves the serial baseline, second starts with a candidate.
            for repeat, order in enumerate((variants, list(reversed(variants)))):
                for seed in (0, 1, 2):
                    for variant in order:
                        yield case, variant, seed, repeat, "correctness"


def verify_binary(variant, binary):
    binding = verify_artifacts(variant)
    record = json.loads((binary / "build_manifest.json").read_text())
    if record["binding"] != binding or record["runner_sha256"] != sha(binary / "runner"):
        raise ValueError("launcher/compiler identity mismatch")
    if record["object_sha256"] != sha(binary / "kernel.o"):
        raise ValueError("device object identity mismatch")
    return record


def execute(command, runtime):
    with (runtime / "runtime.log").open("w") as log:
        with subprocess.Popen(command, cwd=runtime, stdout=log, stderr=subprocess.STDOUT,
                              start_new_session=True) as process:
            try:
                return process.wait(timeout=60)
            except subprocess.TimeoutExpired:
                # Only the new process group created for this invocation is terminated.
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
                raise


def run_one(args, case, variant, seed, repeat, stage):
    case_name, name = case["case"], variant["variant"]
    source = args.matrix / case_name / name
    binary = args.artifacts / case_name / name
    record = verify_binary(source, binary)
    helper = args.experiment.parents[1] / "bin" / "new_run_dir.sh"
    label = f"{stage}-seed{seed}-r{repeat}"
    allocation = subprocess.run([str(helper), str(args.experiment), case_name, name, label],
                                capture_output=True, text=True, check=True, timeout=10)
    runtime = Path(allocation.stdout.strip())
    for path in (args.matrix / case_name / f"seed{seed}").iterdir():
        shutil.copyfile(path, runtime / path.name)
    command = [str(binary / "runner"), str(args.device), str(case["count"]), str(runtime)]
    if stage == "profile":
        application = shlex.join(command)
        command = [shutil.which("msprof") or "msprof", "--application=" + application,
                   "--output=" + str(runtime / "profile"), "--aic-mode=task-based", "--aic-metrics=PipeUtilization"]
    write(runtime / "command.json", command)
    result = dict(case=case_name, variant=name, seed=seed, repeat=repeat, stage=stage,
                  kernel_symbol=record["kernel_symbol"], candidate_id=record["binding"]["candidate_id"],
                  build_manifest_sha256=sha(binary / "build_manifest.json"),
                  input_sha256={p.name: sha(p) for p in runtime.glob("*.bin")}, runtime=str(runtime))
    started = time.monotonic()
    try:
        returncode = execute(command, runtime)
        text = (runtime / "runtime.log").read_text()
        result.update(exit_code=returncode, wall_seconds=time.monotonic()-started)
        if returncode != 0 or "GOLDEN_PASS" not in text or ("soc=" + args.soc) not in text:
            raise ValueError("runtime/golden/SOC gate failed")
        for name, digest in result["input_sha256"].items():
            if sha(runtime / name) != digest:
                raise ValueError("input snapshot modified")
        result.update(status="pass", output_sha256=sha(runtime / "output.bin"))
    except (subprocess.TimeoutExpired, ValueError, OSError) as error:
        result.update(status="fail", error=str(error), wall_seconds=time.monotonic()-started)
        write(runtime / "result.json", result)
        raise RuntimeError(str(runtime) + ": " + str(error)) from error
    write(runtime / "result.json", result)
    return result


def run_matrix(args, manifest):
    performance = args.mode == "performance"
    if performance:
        accepted = json.loads(args.g3_report.read_text())
        if accepted["status"] != "pass" or accepted["matrix_sha256"] != sha(args.matrix / "manifest.json"):
            raise ValueError("performance requires matching complete G3 evidence")
        if accepted["device"] != args.device or accepted["soc"] != args.soc:
            raise ValueError("do not combine devices or SOCs")
        for row in accepted["rows"]:
            path = args.artifacts / row["case"] / row["variant"] / "build_manifest.json"
            if sha(path) != row["build_manifest_sha256"]:
                raise ValueError("G3 and performance build identities differ")
    report = dict(status="running", G4="not_run", mode=args.mode, device=args.device, soc=args.soc,
                  matrix_sha256=sha(args.matrix / "manifest.json"), rows=[],
                  health="ACL_RUNTIME_AVAILABLE", npu_smi="unavailable", exclusive_task_queue=True,
                  limitation="no full health telemetry or occupancy proof for non-cooperating processes")
    path = args.experiment / "results" / (args.mode + ".json")
    if path.exists():
        raise ValueError("never overwrite a previous experiment report")
    write(path, report)
    hashes = {}
    try:
        for case, variant, seed, repeat, stage in execution_order(manifest, performance):
            print(stage, case["case"], variant["variant"], seed, repeat, flush=True)
            row = run_one(args, case, variant, seed, repeat, stage)
            key = (row["case"], row["variant"], seed)
            if key in hashes and hashes[key] != row["output_sha256"]:
                raise ValueError("unstable repeated output")
            hashes[key] = row["output_sha256"]
            report["rows"].append(row)
            write(path, report)
        check_matrix(args.matrix)
        report["status"] = "pass"
    except (ValueError, OSError, RuntimeError, subprocess.SubprocessError) as error:
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
    if provenance["dirty"] or provenance["runtime_tools"]["run_runtime.py"] != sha(Path(__file__)):
        raise ValueError("matrix requires clean frozen source and matching runtime tool")
    if args.mode == "build":
        build_all(args.matrix, args.artifacts, manifest)
    else:
        run_matrix(args, manifest)


if __name__ == "__main__":
    main()
