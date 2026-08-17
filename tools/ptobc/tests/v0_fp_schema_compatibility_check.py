#!/usr/bin/env python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

import re
import sys
from pathlib import Path


EXPECTED = {
    0x1021: ("pto.textract", 0x00, 4),
    0x1022: ("pto.textract", 0x00, 5),
    0x1023: ("pto.tfillpad", 0x00, 2),
    0x1024: ("pto.tfillpad", 0x00, 2),
    0x1025: ("pto.tfillpad", 0x00, 2),
    0x102D: ("pto.tinsert", 0x00, 4),
    0x102E: ("pto.tinsert", 0x00, 5),
    0x1038: ("pto.tmov", 0x00, 2),
    0x1039: ("pto.tmov", 0x00, 3),
    0x1047: ("pto.tquant", 0x02, 3),
    0x1065: ("pto.tstore", 0x02, 0),
    0x1066: ("pto.tstore", 0x00, 3),
}


def main() -> int:
    if len(sys.argv) != 2:
        print(f"usage: {sys.argv[0]} <ptobc_opcodes_v0.h>", file=sys.stderr)
        return 2

    text = Path(sys.argv[1]).read_text(encoding="utf-8")
    rows = {}
    pattern = re.compile(
        r'\{0x([0-9A-Fa-f]+),\s*"([^"]+)",\s*\d+,\s*0x[0-9A-Fa-f]+,\s*'
        r'0x([0-9A-Fa-f]+),\s*(\d+),'
    )
    for opcode, name, operand_mode, num_operands in pattern.findall(text):
        rows[int(opcode, 16)] = (name, int(operand_mode, 16), int(num_operands))

    mismatches = {
        f"0x{opcode:04X}": (rows.get(opcode), expected)
        for opcode, expected in EXPECTED.items()
        if rows.get(opcode) != expected
    }
    if mismatches:
        for opcode, (actual, expected) in mismatches.items():
            print(f"{opcode}: actual={actual}, expected={expected}", file=sys.stderr)
        return 1
    if re.search(r'"pto\.tquant\.mx"', text):
        print("pto.tquant.mx must not be added to the v0 known-op table",
              file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
