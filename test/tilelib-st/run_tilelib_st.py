#!/usr/bin/env python3
# coding=utf-8
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

from pathlib import Path
import sys


if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parent))

from common import run_discovered_cases


def _resolve_case_root(arg: str | None) -> Path:
    if arg is None:
        return Path(__file__).resolve().parent
    root = Path(arg)
    if root.is_absolute():
        return root
    repo_root = Path(__file__).resolve().parents[2]
    return (repo_root / root).resolve()


def main() -> int:
    root = _resolve_case_root(sys.argv[1] if len(sys.argv) > 1 else None)
    return run_discovered_cases(root, argv=sys.argv[2:])


if __name__ == "__main__":
    raise SystemExit(main())
