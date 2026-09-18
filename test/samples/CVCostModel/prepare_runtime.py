# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.


"""Prepare immutable G2-checked cases; never execute an unverified candidate."""
import argparse
import hashlib
from pathlib import Path
import sys
import tempfile
import subprocess

from ptoas._cv_exchange import export_v2
from ptoas._cv_session import apply_candidate, search_candidates
from pto_costmodel.contract import configuration
from pto_costmodel.package import read_package
from pto_costmodel.wire import encode, read_json, require
from runtime_fixture import input_arrays, source_text


def wrapper(cpp, symbol):
    helpers = Path(__file__).resolve().parents[2] / "npu_validation" / "scripts"
    sys.path.insert(0, str(helpers))
    from generate_testcase import _extract_aicore_functions
    functions = {f["name"]: f for f in _extract_aicore_functions(cpp)}
    require(set(functions) == {"cube", "vector"}, "FIXTURE", "unexpected C/V function ABI")
    for name, macro in (("cube", "__DAV_CUBE__"), ("vector", "__DAV_VEC__")):
        original = functions[name]["text"]
        cpp = cpp.replace(original, f"#if defined({macro})\n{original}\n#endif")
    return cpp + f'''
extern "C" __global__ AICORE void {symbol}(
    __gm__ float *q, __gm__ float *k, __gm__ float *v, __gm__ float *out) {{
#if defined(__DAV_CUBE__)
    cube(q, k, v);
#endif
#if defined(__DAV_VEC__)
    vector(out);
#endif
}}
extern "C" void launch_cv(float *q, float *k, float *v, float *out, void *stream) {{
    {symbol}<<<1, nullptr, stream>>>(q, k, v, out);
}}
'''


def prepare_case(root, count, crossing, run_id):
    case = root / (("crossing" if crossing else "basic") + f"-n{count}")
    case.mkdir()
    source = case / "serial.pto"
    source.write_text(source_text(count, crossing), encoding="utf-8")
    profile = read_json(Path(__file__).with_name("a5_profile.json"))
    with tempfile.TemporaryDirectory(prefix="pto-runtime-bindings-") as temporary:
        initial = Path(temporary) / "package"
        export_v2(source, profile, initial)
        bindings = read_package(initial)["bindings"]
    bindings["alias_contract"] = "disjoint"
    export_v2(source, profile, case / "package", bindings)
    package = read_package(case / "package")
    candidates, rejected = search_candidates(package, sorted({0, 1, 2, count, count + 1}))
    configs = [("serial", configuration(package["program"], 0), "compile_serial")]
    configs.extend((f"p{c['configuration']['preload_count']}", c["configuration"], "compile") for c in candidates)
    extra = configuration(package["program"], min(2, count))
    for row, slots in zip([r for r in extra["buffers"] if r["buffer_id"].startswith("vector.")], (1, 2, 3)):
        row["count"] = slots
    configs.append(("slots123", extra, "compile"))
    rows = []
    for name, config, mode in configs:
        rows.append(_variant(case, name, config, mode, run_id))
    for seed in (0, 1, 2):
        inputs = case / f"seed{seed}"
        inputs.mkdir()
        for name, data in input_arrays(count, seed, crossing).items():
            data.tofile(inputs / (name + ".bin"))
    return dict(case=case.name, count=count, crossing=crossing, candidates=rows, rejected=rejected)


def _variant(case, name, config, mode, run_id):
    from pto_costmodel.wire import ContractError
    output = case / name
    try:
        report = apply_candidate(case / "package", config, output, mode)
    except ContractError as error:
        if error.code != "INSUFFICIENT_SLOTS":
            raise
        return dict(variant=name, G2="rejected", reason=error.code)
    symbol = "cv_" + hashlib.sha256((run_id + case.name + name).encode()).hexdigest()[:20]
    cpp = wrapper((output / "candidate.cpp").read_text(), symbol)
    (output / "kernel.cpp").write_text(cpp, encoding="utf-8")
    return dict(variant=name, G2=report["validation"]["G2"], symbol=symbol,
                configuration=config, wrapper_sha256=hashlib.sha256(cpp.encode()).hexdigest())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--counts", type=int, nargs="+", default=[1, 2, 4, 8])
    args = parser.parse_args()
    require(all(n in (1, 2, 4, 8) for n in args.counts), "FIXTURE", "unsupported count")
    args.output.mkdir(parents=True, exist_ok=False)
    rows = [prepare_case(args.output, n, crossing, str(args.output.resolve()))
            for crossing in (False, True) for n in args.counts]
    hashes = {str(p.relative_to(args.output)): hashlib.sha256(p.read_bytes()).hexdigest()
              for p in sorted(args.output.rglob("*")) if p.is_file()}
    (args.output / "manifest.json").write_text(encode(dict(cases=rows, files=hashes,
                                                         provenance=source_provenance(), G3="not_run", G4="not_run")))


def source_provenance():
    import ptoas
    from ptoas._loader import ensure_core
    from ptoas._cv_compile import compiler_identity
    root = Path(__file__).resolve().parents[3]
    def git(*args):
        return subprocess.run(["git", "-C", str(root), *args], capture_output=True,
                              text=True, timeout=10, check=True).stdout
    paths = git("ls-files", "--cached", "--others", "--exclude-standard").splitlines()
    files = {p: hashlib.sha256((root / p).read_bytes()).hexdigest() for p in paths if (root / p).is_file()}
    binary = Path(ensure_core().__file__)
    return dict(commit=git("rev-parse", "HEAD").strip(), dirty=bool(git("status", "--porcelain")),
                source_fingerprint=hashlib.sha256(encode(files).encode()).hexdigest(),
                compiler_identity=compiler_identity(), compiler_path=str(binary),
                python=sys.executable, python_version=sys.version, ptoas_path=str(ptoas.__file__),
                runtime_tools={name: files["test/samples/CVCostModel/" + name]
                               for name in ("runtime_main.cpp", "build_runtime.py", "run_runtime.py")})


if __name__ == "__main__":
    main()
