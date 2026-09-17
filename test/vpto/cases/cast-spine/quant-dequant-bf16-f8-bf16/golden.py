#!/usr/bin/env python3
#!/usr/bin/env python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Golden data for quant-dequant-bf16-f8-bf16.

Chain:  bf16 --vcvt--> f32 --*1.5--> f8E4M3FN (stored)
        f8E4M3FN --vcvt--> f32 --*2.0--> bf16 (stored)

Every input value multiplied by 1.5 stays exactly representable in f8E4M3FN
(magnitude <= 8, mantissa <= 3 bits) and the dequantised product is exact in
bf16, so both outputs are lossless for this input set and the oracle is
independent of the hardware rounding mode.
"""

import argparse
from pathlib import Path

import ml_dtypes
import numpy as np

ELEMS = 16384
QUANT_SCALE = np.float32(1.5)
DEQUANT_SCALE = np.float32(2.0)
BF16 = ml_dtypes.bfloat16
F8E4M3FN = ml_dtypes.float8_e4m3fn
EXACT_VALUES = np.array(
    [0.0, 1.0, -1.0, 0.5, -0.5, 1.5, -1.5, 2.0, -2.0, 3.0, -3.0, 4.0, -4.0],
    dtype=np.float32,
)


def generate(output_dir: Path) -> None:
    rng = np.random.default_rng(1337)
    src = EXACT_VALUES[rng.integers(0, EXACT_VALUES.size, ELEMS)].astype(BF16)

    quantized = (src.astype(np.float32) * QUANT_SCALE).astype(F8E4M3FN)
    restored = (quantized.astype(np.float32) * DEQUANT_SCALE).astype(BF16)

    output_dir.mkdir(parents=True, exist_ok=True)
    src.view(np.uint16).tofile(output_dir / "v1.bin")
    np.zeros(ELEMS, dtype=np.uint8).tofile(output_dir / "v2.bin")
    np.zeros(ELEMS, dtype=np.uint16).tofile(output_dir / "v3.bin")
    quantized.view(np.uint8).tofile(output_dir / "golden_v2.bin")
    restored.view(np.uint16).tofile(output_dir / "golden_v3.bin")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=Path("."))
    args = parser.parse_args()
    generate(args.output_dir)


if __name__ == "__main__":
    main()
