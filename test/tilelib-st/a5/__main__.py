#!/usr/bin/env python3
# coding=utf-8
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

# Directory runner for the A5 TileLib operator-level ST cases.
#
# Each operator suite (e.g. tadd, tmatmul, ...) is a subdirectory with a
# ``case.py`` module that authors its kernels with PTODSL and builds its CASES
# list through the helpers in the parent ``test/tilelib-st/common.py`` module.
# Running this directory discovers every nested ``case.py`` module and executes
# the cases against the torch_npu / simulator runtime.
#
# See test/tilelib-st/README.md for the authoring conventions.

from pathlib import Path
import sys


if __package__ in {None, ""}:
    # common.py lives one level up, in test/tilelib-st/.
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from common import run_discovered_cases


if __name__ == "__main__":
    raise SystemExit(run_discovered_cases(Path(__file__).resolve().parent))
