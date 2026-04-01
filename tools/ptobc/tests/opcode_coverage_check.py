#!/usr/bin/env python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

import logging
import re
import sys
from pathlib import Path


def parse_td_mnemonics(td_path: Path):
    td = td_path.read_text(encoding="utf-8", errors="ignore")
    mns = set(re.findall(r"\bmnemonic\s*=\s*\"([^\"]+)\"", td))
    return {f"pto.{m}" for m in mns}


def parse_header_names(h_path: Path):
    txt = h_path.read_text(encoding="utf-8", errors="ignore")
    return set(re.findall(r"\{0x[0-9A-Fa-f]+,\s*\"([^\"]+)\"", txt))


def main():
    if len(sys.argv) != 3:
        logging.error("usage: %s <PTOOps.td> <ptobc_opcodes_v0.h>", sys.argv[0])
        return 2

    td_path = Path(sys.argv[1])
    h_path = Path(sys.argv[2])

    td_ops = parse_td_mnemonics(td_path)
    hdr_ops = parse_header_names(h_path)

    missing = sorted(op for op in td_ops if op not in hdr_ops)
    if missing:
        logging.error("ptobc v0 opcode table is missing ops present in PTOOps.td:")
        for op in missing:
            logging.error("  - %s", op)
        logging.error(
            "Fix: extend docs/bytecode/tools/gen_v0_tables.py "
            "(or table source) and regenerate ptobc_opcodes_v0.h"
        )
        return 1

    logging.info(
        "OK: opcode coverage check passed (PTOOps.td ops=%d, table ops=%d)",
        len(td_ops),
        len(hdr_ops),
    )
    return 0


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    raise SystemExit(main())
