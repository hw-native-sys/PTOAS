#!/usr/bin/env python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

from pathlib import Path
import re

from ptodsl import pto
from ptoas.mlir.dialects import pto as generated_pto


ROOT = Path(__file__).resolve().parents[2]
INDEX = ROOT / "docs/isa/vmi-isa/10-appendices.md"
ODS = ROOT / "include/PTO/IR/VMIOps.td"
EMISSION_TEST = ROOT / "ptodsl/tests/test_jit_compile.py"

# All indexed ops currently lower through the A5 VMI-to-VPTO pipeline. Keep
# this manifest explicit so a new ISA entry cannot silently inherit support.
BACKEND_CAPABILITY = {
    name: "a5-vpto"
    for name in (
        "vload vstore vsstb vci vadd vaddc vaddcs vsub vmul vdiv vmax vmin vabs vneg "
        "vrelu vexp vln vsqrt vand vor vxor vnot vshl vshr vadds vmuls "
        "vmaxs vmins vshls vshrs vcmp vcmps vsel vselr vbrc vcadd vcmax "
        "vcmin vcvt vinterpret_cast vexpdif vaxpy vlrelu vprelu vmull "
        "vmula vchist vdhist vgather vgatherb vscatter create_mask "
        "create_group_mask vintlv vdintlv"
    ).split()
}

# PTODSL spells the grouped-mask generator as create_mask(..., group=...).
PTODSL_ALIASES = {"create_group_mask": "create_mask"}


def _indexed_ops():
    rows = re.findall(
        r"^\|\s*(\d+)\s*\|\s*`pto\.vmi\.([a-z0-9_]+)`",
        INDEX.read_text(encoding="utf-8"),
        re.MULTILINE,
    )
    return [(int(number), name) for number, name in rows]


def main() -> None:
    indexed = _indexed_ops()
    assert [number for number, _ in indexed] == list(range(1, 56))
    names = [name for _, name in indexed]
    assert len(names) == len(set(names)) == 55
    assert set(BACKEND_CAPABILITY) == set(names)
    assert set(PTODSL_ALIASES) <= set(names)

    ods = ODS.read_text(encoding="utf-8")
    emission_test = EMISSION_TEST.read_text(encoding="utf-8")
    for name in names:
        wrapper = PTODSL_ALIASES.get(name, name)
        assert hasattr(pto.vmi, wrapper), f"missing PTODSL wrapper for pto.vmi.{name}"
        assert hasattr(generated_pto, f"vmi_{name}"), f"missing generated binding for pto.vmi.{name}"
        definition = re.search(
            rf'^def\s+\w+\s*:\s*(VMI_Op|VMI_VecScalarOp|VMIHistogramOp)<"{re.escape(name)}"[\s\S]*?(?=^def\s|\Z)',
            ods,
            re.MULTILINE,
        )
        assert definition, f"missing ODS op for pto.vmi.{name}"
        verifier_text = definition.group(0)
        if definition.group(1) == "VMI_VecScalarOp":
            verifier_text += re.search(r'class VMI_VecScalarOp[\s\S]*?^}', ods, re.MULTILINE).group(0)
        elif definition.group(1) == "VMIHistogramOp":
            verifier_text += re.search(r'class VMIHistogramOp[\s\S]*?^}', ods, re.MULTILINE).group(0)
        assert "let hasVerifier = 1;" in verifier_text, f"missing verifier for pto.vmi.{name}"
        assert f'"pto.vmi.{name}"' in emission_test, f"missing emission assertion for pto.vmi.{name}"


if __name__ == "__main__":
    main()
