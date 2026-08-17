#!/usr/bin/env python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

from ptoas.mlir.ir import Context
from ptoas.mlir.dialects import pto


def assert_contains(text: str, needle: str) -> None:
    if needle not in text:
        raise AssertionError(f"missing {needle!r} in {text!r}")


def main() -> None:
    with Context() as ctx:
        pto.register_dialect(ctx)

        hif8 = pto.HiF8Type.get(ctx)
        hif8x2 = pto.HiF8x2Type.get(ctx)
        f8e8m0 = pto.F8E8M0Type.get(ctx)
        f4e1 = pto.F4E1M2x2Type.get(ctx)
        f4e2 = pto.F4E2M1x2Type.get(ctx)
        bf16x2 = pto.BF16x2Type.get(ctx)

        assert_contains(str(hif8), "hif8")
        assert_contains(str(hif8x2), "hif8x2")
        assert_contains(str(f8e8m0), "f8E8M0")
        assert_contains(str(f4e1), "f4E1M2x2")
        assert_contains(str(f4e2), "f4E2M1x2")
        assert_contains(str(bf16x2), "bf16x2")

    print("low_precision_types: PASS")


if __name__ == "__main__":
    main()
