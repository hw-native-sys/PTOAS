# coding=utf-8
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Compile the public lane matrix, with no device dependencies."""
from pathlib import Path
import subprocess
import sys

TOOL, OUT = sys.argv[1], Path(sys.argv[2])
OUT.mkdir(parents=True, exist_ok=True)
LANES = (1, 2, 4, 8, 64, 128, 256)
PIPE = ["-vmi-lower-unified-to-legacy", "-vmi-mask-granularity-assignment",
        "-vmi-layout-assignment", "-pto-validate-vmi-layout-ir"]


def compile_ir(name, text, succeeds=True, load_safety=None):
    path = OUT / (name + ".pto")
    path.write_text(text)
    lowering = ("-vmi-to-vpto" if load_safety is None
                else f"-vmi-to-vpto=load-safety={load_safety}")
    result = subprocess.run([TOOL, str(path), *PIPE, lowering],
                            capture_output=True, text=True)
    (OUT / (name + ".out")).write_text(result.stdout)
    assert (result.returncode == 0) == succeeds, name + "\n" + result.stderr
    if succeeds:
        assert "pto.vmi." not in result.stdout, result.stdout
    return result.stdout


for n in (*LANES, 96):  # Public whitelist must not constrain internal IR.
    for elem, index, bits in (("f32", "i32", 32), ("f16", "ui16", 16),
                              ("ui8", "ui16", 8), ("i8", "ui16", 8)):
        v = f"!pto.vmi.vreg<{n}x{elem}>"
        i = f"!pto.vmi.vreg<{n}x{index}>"
        m = f"!pto.vmi.mask<{n}xpred>"
        out = compile_ir(f"scatter_{elem}_{n}", f"""module {{
  func.func @probe(%v: {v}, %p: !pto.ptr<{elem}, ub>, %i: {i}, %m: {m}) {{
    pto.vmi.vscatter %v, %p, %i, %m : {v}, !pto.ptr<{elem}, ub>, {i}, {m}
    return
  }}
}}""")
        capacity = 64 if bits == 32 else 128
        requests = (n + capacity - 1) // capacity
        assert out.count("pto.vscatter ") == requests, out
        assert ("pto.pand " in out) == (n % capacity != 0), out
        if bits == 8:
            assert out.count("pto.vzunpack ") == requests, out
            assert out.count("pto.punpack ") == requests, out
            assert ('"HIGHER"' in out) == (n > 128), out
            for line in out.splitlines():
                if "pto.vscatter " in line:
                    assert "!pto.mask<b16>" in line, line

for n in LANES:
    for bins in (128, 256):
        for op in ("vdhist", "vchist"):
            acc = f"!pto.vmi.vreg<{bins}xui16>"
            src = f"!pto.vmi.vreg<{n}xui8>"
            mask = f"!pto.vmi.mask<{n}xpred>"
            out = compile_ir(f"{op}_{bins}_{n}", f"""module {{
  func.func @probe(%p: !pto.ptr<ui8, ub>, %a: {acc}, %m: {mask}) -> {acc} {{
    %c0 = arith.constant 0 : index
    %s = pto.vmi.vload %p[%c0] : !pto.ptr<ui8, ub> -> {src}
    %r = pto.vmi.{op} %a, %s, %m : {acc}, {src}, {mask} -> {acc}
    return %r : {acc}
  }}
}}""")
            physical = "pto.dhistv2 " if op == "vdhist" else "pto.chistv2 "
            assert out.count(physical) == bins // 128, out
            if n in (64, 128):
                assert out.count("pto.vsldb ") == 1 and "pto.vlds " not in out, out
                pattern = "PAT_VL64" if n == 64 else "PAT_VL128"
                assert f'pto.pset_b8 "{pattern}"' in out, out
                assert "arith.constant 1 : i16" in out, out
            if n == 256:
                assert "pto.vlds " in out and "pto.vsldb " not in out, out

# 33 bytes is not a bounded contiguous load: it falls back to a full-carrier
# read. Strict load safety rejects the unproven physical read; the default
# policy accepts the same shape. This 33-lane shape covers IR robustness only;
# PTODSL's public VMI lane whitelist excludes 33.
compile_ir("unsafe_load_33_0", """module {
  func.func @probe(%p: !pto.ptr<ui8, ub>) -> !pto.vmi.vreg<33xui8> {
    %off = arith.constant 0 : index
    %s = pto.vmi.vload %p[%off] : !pto.ptr<ui8, ub> -> !pto.vmi.vreg<33xui8>
    return %s : !pto.vmi.vreg<33xui8>
  }
}""", succeeds=False, load_safety="error")

# PtrType sources with non-32-byte-aligned offsets now use the stateful
# vldas+vldus fallback (introduced by 5eb87c2), so these succeed with the
# default load-safety=warn policy.
for n, offset in ((64, "1"), (128, "16"), (64, None)):
    offset_ir = "" if offset is None else f"%off = arith.constant {offset} : index"
    arg = ", %off: index" if offset is None else ""
    compile_ir(f"unsafe_load_{n}_{offset}", f"""module {{
  func.func @probe(%p: !pto.ptr<ui8, ub>{arg}) -> !pto.vmi.vreg<{n}xui8> {{
    {offset_ir}
    %s = pto.vmi.vload %p[%off] : !pto.ptr<ui8, ub> -> !pto.vmi.vreg<{n}xui8>
    return %s : !pto.vmi.vreg<{n}xui8>
  }}
}}""")

print("64 histogram/scatter/load compilation cases passed")
