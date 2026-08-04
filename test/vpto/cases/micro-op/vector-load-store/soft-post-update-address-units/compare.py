#!/usr/bin/env python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

import numpy as np


def check(name: str, dtype: np.dtype) -> None:
    suffix = "" if name == "main" else f"_{name}"
    golden = np.fromfile(f"golden_output{suffix}.bin", dtype=dtype)
    output = np.fromfile(f"output{suffix}.bin", dtype=dtype)
    if golden.shape != output.shape or not np.array_equal(golden, output):
        mismatch = np.flatnonzero(golden != output)
        idx = int(mismatch[0]) if mismatch.size else 0
        print(
            f"[ERROR] {name} mismatch: shape={output.shape}, idx={idx}, "
            f"golden={golden[idx] if golden.size else 'n/a'}, "
            f"output={output[idx] if output.size else 'n/a'}"
        )
        raise SystemExit(2)


def main() -> None:
    check("main", np.float32)
    check("spr", np.uint32)
    print("[INFO] compare passed")


if __name__ == "__main__":
    main()
