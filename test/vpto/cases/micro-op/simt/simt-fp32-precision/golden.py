#!/usr/bin/env python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software; you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OR
# CONDITIONS OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the license.

import argparse
import ctypes
from pathlib import Path

import numpy as np

N = 256


def generate(output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(20260803)
    a = rng.standard_normal(N).astype(np.float32)
    b = rng.standard_normal(N).astype(np.float32)
    c = rng.standard_normal(N).astype(np.float32)
    lhs = (rng.random(N, dtype=np.float32) * np.float32(2.0)).astype(np.float32)
    rhs = (rng.random(N, dtype=np.float32) * np.float32(128.0) + np.float32(0.25)).astype(np.float32)

    separate = np.add(np.multiply(a, b, dtype=np.float32), c, dtype=np.float32)
    libm = ctypes.CDLL("libm.so.6")
    fmaf = libm.fmaf
    fmaf.argtypes = [ctypes.c_float, ctypes.c_float, ctypes.c_float]
    fmaf.restype = ctypes.c_float
    fused = np.asarray([fmaf(x, y, z) for x, y, z in zip(a, b, c)], dtype=np.float32)
    quotient = np.divide(lhs, rhs, dtype=np.float32)

    for name, value in (("a", a), ("b", b), ("c", c), ("lhs", lhs), ("rhs", rhs)):
        value.tofile(output_dir / f"{name}.bin")
    np.concatenate((separate, fused, quotient)).tofile(output_dir / "golden_out.bin")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=Path("."))
    generate(parser.parse_args().output_dir)
