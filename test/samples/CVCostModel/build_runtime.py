# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.


"""Build one G2-verified mixed C/V variant using the activated A5 toolkit."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess

BUILD_TIMEOUT_SECONDS = 900


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify_artifacts(variant):
    root = variant.parent.parent
    manifest = json.loads((root / "manifest.json").read_text())
    for path in variant.iterdir():
        if path.is_file():
            key = str(path.relative_to(root))
            if manifest["files"].get(key) != sha(path):
                raise ValueError("prepared artifact fingerprint mismatch: " + key)
    report = json.loads((variant / "validation_report.json").read_text())
    memory = json.loads((variant / "memory_plan.json").read_text())
    if report["G2"] != "pass" or report["completion"]["status"] != "pass":
        raise ValueError("candidate G2 has not passed")
    binding = report["binding"]
    digest = memory.pop("fingerprint")
    encoded = json.dumps(memory, ensure_ascii=True, sort_keys=True, indent=2, allow_nan=False) + "\n"
    if (hashlib.sha256(encoded.encode()).hexdigest() != digest
            or report["memory_plan_fingerprint"] != digest or memory["binding"] != binding):
        raise ValueError("compiler feedback fingerprint mismatch")
    if sha(variant / "candidate.cpp") != binding["cpp_fingerprint"]:
        raise ValueError("candidate C++ fingerprint mismatch")
    if sha(variant / "lowered.pto") != binding["final_ir_fingerprint"]:
        raise ValueError("candidate IR fingerprint mismatch")
    return binding


def build(variant, output):
    binding = verify_artifacts(variant)
    provenance = json.loads((variant.parent.parent / "manifest.json").read_text()).get("provenance", {})
    expected = provenance.get("runtime_tools", {})
    for name in ("runtime_main.cpp", "build_runtime.py"):
        if expected and expected.get(name) != sha(Path(__file__).with_name(name)):
            raise ValueError("runtime tool differs from the frozen source")
    compiler = shutil.which("bisheng")
    if compiler is None or not os.environ.get("ASCEND_HOME_PATH"):
        raise ValueError("activate the verified A5 toolchain before building")
    toolkit = Path(os.environ["ASCEND_HOME_PATH"]).resolve()
    include, library = toolkit / "aarch64-linux/include", toolkit / "aarch64-linux/lib64"
    output.mkdir(parents=True, exist_ok=False)
    common = [compiler, "-std=c++17", "-O2", "-fPIC", "-I" + str(include)]
    compile_args = common + ["-xcce", "-fenable-matrix",
                            "--cce-aicore-arch=dav-c310", "-DREGISTER_BASE",
                            "-mllvm", "-cce-aicore-stack-size=0x8000",
                            "-mllvm", "-cce-aicore-function-stack-size=0x8000",
                            "-mllvm", "-cce-aicore-record-overflow=true",
                            "-mllvm", "-cce-aicore-addr-transform",
                            "-mllvm", "-cce-aicore-dcci-insert-for-scalar=false",
                            "-c", str(variant / "kernel.cpp"), "-o", str(output / "kernel.o")]
    main = Path(__file__).with_name("runtime_main.cpp").resolve()
    link_args = common + [str(main), str(output / "kernel.o"), "--cce-fatobj-link",
                          "-L" + str(library), "-Wl,-rpath," + str(library),
                          "-lruntime", "-lascendcl", "-lstdc++", "-ldl", "-lpthread", "-o", str(output / "runner")]
    commands = [compile_args, link_args]
    (output / "commands.json").write_text(json.dumps(commands, indent=2))
    for index, command in enumerate(commands):
        with (output / f"build-{index}.log").open("w") as log:
            subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True,
                           timeout=BUILD_TIMEOUT_SECONDS)
    symbols = [v["symbol"] for c in json.loads((variant.parent.parent / "manifest.json").read_text())["cases"]
               if c["case"] == variant.parent.name for v in c["candidates"] if v["variant"] == variant.name]
    if len(symbols) != 1 or symbols[0].encode() not in (output / "kernel.o").read_bytes():
        raise ValueError("device object kernel symbol mismatch")
    headers = {str(p.relative_to(include)): sha(p) for p in sorted((include / "pto").rglob("*")) if p.is_file()}
    evidence = dict(binding=binding, compiler=compiler, compiler_sha256=sha(Path(compiler)),
                    headers=headers, kernel_symbol=symbols[0], main_sha256=sha(main),
                    wrapper_sha256=sha(variant / "kernel.cpp"), object_sha256=sha(output / "kernel.o"),
                    runner_sha256=sha(output / "runner"), aicore_arch="dav-c310", aic=1, aiv=2)
    (output / "build_manifest.json").write_text(json.dumps(evidence, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    build(args.variant.resolve(), args.output.resolve())
