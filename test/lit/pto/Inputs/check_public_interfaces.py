# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Audit complete generated translation units, including optional helper bodies."""

import argparse
import re
import shutil
import subprocess
import tempfile
from pathlib import Path

# Include arithmetic, conversion, memory, compute, control, communication and
# generated preamble helpers. Existing tests retain their precise semantic checks.
CASES = (
    ("part_arithmetic_tile_native", "a3", "level3", "TPARTADD"),
    ("minmax_scalar_tile_native", "a3", "level3", "TMAXS"),
    ("addmul_scalar_tile_native", "a3", "level3", "TADDS"),
    ("matmul_tile_native", "a5", "level3", "TMATMUL"),
    ("tmatmul_mx_emitc", "a5", "level2", "TMATMUL_MX"),
    ("trandom_emitc", "a5", "level2", "PTOAS__TRANDOM"),
    ("eventid_array_dyn_sync", "a3", "level2", "PTOAS_EventIdArray"),
    ("comm_p2p_emitc", "a3", "level2", "pto::comm::"),
    ("comm_collective_emitc", "a3", "level2", "pto::comm::"),
    ("pipe_tile_native", "a3", "level3", "TPUSH"),
    ("tprefetch_emitc", "a3", "level2", "TPREFETCH"),
    ("issue1478_scalar_addf_emitc", "a3", "level3", "scalar_addf"),
    ("issue1478_scalar_addf_emitc", "a5", "level3", "scalar_addf"),
    ("fixpipe_frontend_emitc_scalar_vector_quant_a3", "a3", "level2", "SET_QUANT_SCALAR"),
    ("fixpipe_frontend_emitc_vector_quant_a5", "a5", "level2", "SET_QUANT_VECTOR"),
    ("scf_while_generic_domains_emitc", "a5", "level2", "while"),
    ("public_scalar_helpers", "a3", "level2", "ptoas_bitcast"),
    ("public_scalar_helpers", "a5", "level3", "__builtin_fmodf"),
)


def run(command):
    """Run trusted test tools with bounded execution and useful failure output."""
    result = subprocess.run(
        command, capture_output=True, text=True, timeout=120, check=False
    )
    if result.returncode:
        raise RuntimeError(f"{command!r}\n{result.stdout}\n{result.stderr}")


def generate(ptoas, source, output, arch, level, expected):
    """Check final source after lowering and driver postprocessing."""
    run(
        [
            ptoas,
            "--pto-backend=emitc",
            f"--pto-arch={arch}",
            f"--pto-level={level}",
            str(source),
            "-o",
            str(output),
        ]
    )
    generated = output.read_text(encoding="utf-8")
    private = re.findall(r"\b__builtin_cce_\w+", generated)
    if private or expected not in generated:
        raise RuntimeError(
            f"{source.name} ({arch}, {level}): private={private}, expected={expected}"
        )


def check_sync(ptoas, cxx, tests, scratch):
    """Cover every architecture, build level and kernel role without private declarations."""
    original = (tests / "public_sync_interfaces.pto").read_text(encoding="utf-8")
    for role in ("cube", "vector"):
        source = scratch / f"sync_{role}.pto"
        source_text = original.replace("kernel_kind<cube>", f"kernel_kind<{role}>")
        if role == "vector":
            source_text = source_text.replace("PIPE_FIX", "PIPE_MTE3").replace(
                "PIPE_MTE1", "PIPE_MTE3"
            )
        source.write_text(source_text, encoding="utf-8")
        for arch in ("a2", "a3", "a5"):
            for level in ("level1", "level2", "level3"):
                output = scratch / f"sync_{role}_{arch}_{level}.cpp"
                generate(ptoas, source, output, arch, level, "ffts_cross_core_sync")
                run(
                    [
                        cxx,
                        "-std=c++17",
                        "-fsyntax-only",
                        "-I",
                        str(tests / "Inputs/public_sync"),
                        "-DPTOAS_TEST_A5" if arch == "a5" else "-DPTOAS_TEST_A23",
                        "-D__DAV_CUBE__" if role == "cube" else "-D__DAV_VEC__",
                        str(output),
                    ]
                )


def main():
    """Run the representative EmitC output matrix."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("ptoas")
    parser.add_argument("--cxx", default="c++")
    args = parser.parse_args()
    ptoas = shutil.which(args.ptoas)
    cxx = shutil.which(args.cxx)
    if ptoas is None or cxx is None:
        raise RuntimeError(
            "The generated-interface audit requires ptoas and a host C++ compiler"
        )
    tests = Path(__file__).resolve().parent.parent
    with tempfile.TemporaryDirectory(prefix="ptoas-public-interfaces-") as directory:
        scratch = Path(directory)
        check_sync(ptoas, cxx, tests, scratch)
        for name, arch, level, expected in CASES:
            generate(
                ptoas,
                tests / f"{name}.pto",
                scratch / f"{name}_{arch}.cpp",
                arch,
                level,
                expected,
            )
    print(
        "Public-interface audit passed: 18 synchronization consumer compilations and 18 other generated sources"
    )


if __name__ == "__main__":
    main()
