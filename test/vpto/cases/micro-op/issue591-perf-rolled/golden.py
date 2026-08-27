# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

# Issue #591 performance acceptance: generate the input payload only. The
# acceptance criterion is the CA-model instruction mix and RVEC cycle count,
# not the numeric result (correctness is covered by lit + differential tests).

import argparse
from pathlib import Path

import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=Path("."))
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    rng = np.random.default_rng(args.seed)
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    elems = 16 * 128
    v1 = rng.integers(0, 1024, size=elems, dtype=np.uint16)
    v1.tofile(out / "v1.bin")
    golden = np.zeros(elems, dtype=np.uint16)
    golden.tofile(out / "golden_v3.bin")


if __name__ == "__main__":
    main()
