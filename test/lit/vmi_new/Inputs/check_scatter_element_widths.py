# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Check scatter width pairing before passes and preserve legal VPTO lowering."""
import subprocess
import sys
from pathlib import Path

TOOL, OUT = sys.argv[1], Path(sys.argv[2])
OUT.mkdir(parents=True, exist_ok=True)
# Explicit ISA rows: value type, bit width, legal offsets, illegal offsets.
ROWS = (
    ("i8", 8, ("i16", "ui16"), ("i32", "ui32")),
    ("i16", 16, ("i16", "ui16"), ("i32", "ui32")),
    ("f16", 16, ("i16", "ui16"), ("i32", "ui32")),
    ("bf16", 16, ("i16", "ui16"), ("i32", "ui32")),
    ("i32", 32, ("i32", "ui32"), ("i16", "ui16")),
    ("f32", 32, ("i32", "ui32"), ("i16", "ui16")),
)
negative, positive, expected = [], [], {}
for op in ("vscatter", "scatter"):
    for elem, bits, good, bad in ROWS:
        for index in (*good, *bad):
            for assigned in (False, True):
                for lanes in (8, 128):
                    name = f"{op}_{elem}_{index}_{assigned}_{lanes}"
                    layout = ", #pto.vmi.layout<contiguous>" if assigned else ""
                    granularity = f"b{bits}" if assigned else "pred"
                    v = f"!pto.vmi.vreg<{lanes}x{elem}{layout}>"
                    i = f"!pto.vmi.vreg<{lanes}x{index}{layout}>"
                    m = f"!pto.vmi.mask<{lanes}x{granularity}{layout}>"
                    p = f"!pto.ptr<{elem}, ub>"
                    operands = "%v, %p, %i, %m" if op == "vscatter" else "%v, %p[%i], %m"
                    diagnostic = ""
                    if index in bad:
                        width = 32 if bits == 32 else 16
                        message = f"requires {width}-bit offset vector elements for {bits}-bit values"
                        diagnostic = "// expected-error @+1 {{" + message + "}}"
                    function = f"""func.func @{name}(%v: {v}, %p: {p}, %i: {i}, %m: {m}) {{
  {diagnostic}
  pto.vmi.{op} {operands} : {v}, {p}, {i}, {m}
  return
}}
"""
                    if index in bad:
                        negative.append("module {\n" + function + "}\n")
                    else:
                        positive.append(function)
                        expected[name] = 2 if bits == 32 and lanes == 128 else 1


def compile_text(name, text, *flags):
    path = OUT / (name + ".pto")
    path.write_text(text)
    result = subprocess.run([TOOL, str(path), *flags], check=False,
                            capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr
    return result.stdout


compile_text("invalid", "\n// -----\n".join(negative),
             "-split-input-file", "-verify-diagnostics")
valid = "module {\n" + "\n".join(positive) + "}\n"
compile_text("valid", valid)
lowered = compile_text("lowered", valid, "-vmi-lower-unified-to-legacy",
                       "-vmi-mask-granularity-assignment", "-vmi-layout-assignment",
                       "-pto-validate-vmi-layout-ir", "-vmi-to-vpto")
assert "pto.vmi." not in lowered, lowered
for function in lowered.split("func.func @")[1:]:
    name = function.split("(", 1)[0]
    assert function.count("pto.vscatter ") == expected.pop(name), function
assert not expected, expected
print(f"{len(negative)} invalid pairings rejected; {len(positive)} legal pairings verified and lowered")
