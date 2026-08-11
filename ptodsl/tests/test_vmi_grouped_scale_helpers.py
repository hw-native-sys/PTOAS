#!/usr/bin/env python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""Keep the VMI grouped-scale reduction aligned with the direct CCE path."""

from pathlib import Path
import tempfile

from ptoas import _core
from ptodsl import pto


def bit_and(lhs, rhs, mask):
    return pto.vmi.vand(lhs, rhs, mask)


def max(lhs, rhs, mask):
    return pto.vmi.vmax(lhs, rhs, mask)


def grouped_max(source, mask):
    return pto.vmi.vcmax(source, mask, group=8)


@pto.jit(
    name="vmi_grouped_scale_helpers",
    kernel_kind="vector",
    target="a5",
    backend="vpto",
    mode="explicit",
    insert_sync=False,
)
def vmi_grouped_scale_helpers(
    source_gm: pto.ptr(pto.bf16, "gm"),
    scale_gm: pto.ptr(pto.bf16, "gm"),
):
    source_ub = pto.castptr(0, pto.ptr(pto.bf16, "ub"))
    scale_ub = pto.castptr(512, pto.ptr(pto.bf16, "ub"))

    pto.copy_gm_to_ubuf(
        source_gm, source_ub, 0, 1, 512, 0, 0, False, 0, 512, 512
    )
    pto.set_flag("PIPE_MTE2", "PIPE_V", "EVENT_ID0")
    pto.wait_flag("PIPE_MTE2", "PIPE_V", "EVENT_ID0")

    offset = pto.const(0, dtype=pto.index)
    mask = pto.vmi.create_mask(128, size=128)
    low, high = pto.vmi.vload(
        source_ub, offset, size=128, dist_mode="dintlv"
    )
    low_bits = pto.vmi.vinterpret_cast(low, to_dtype=pto.ui16)
    high_bits = pto.vmi.vinterpret_cast(high, to_dtype=pto.ui16)
    abs_bits = pto.vmi.vbrc(pto.ui16(0x7FFF), size=128)

    # Direct CCE peer:
    #   a0 = bit_and(pair.lo, abs_bits, m16)
    #   a1 = bit_and(pair.hi, abs_bits, m16)
    #   maxima = grouped_max(max(a0, a1, m16), m16)
    a0 = bit_and(low_bits, abs_bits, mask)
    a1 = bit_and(high_bits, abs_bits, mask)
    maxima = grouped_max(max(a0, a1, mask), mask)
    scale = pto.vmi.vinterpret_cast(maxima, to_dtype=pto.bf16)
    pto.vmi.vstore(scale, scale_ub, offset, group=8, stride=1)

    pto.set_flag("PIPE_V", "PIPE_MTE3", "EVENT_ID1")
    pto.wait_flag("PIPE_V", "PIPE_MTE3", "EVENT_ID1")
    pto.copy_ubuf_to_gm(scale_ub, scale_gm, 0, 1, 16, 0, 16, 16)


def expect_count(text: str, op_name: str, expected: int) -> None:
    actual = text.count(f"{op_name} ")
    if actual != expected:
        raise AssertionError(
            f"expected {expected} occurrence(s) of {op_name}, got {actual}:\n{text}"
        )


def main() -> None:
    compiled = vmi_grouped_scale_helpers.compile()
    compiled.verify()
    source = compiled.mlir_text()

    expect_count(source, "pto.vmi.vand", 2)
    expect_count(source, "pto.vmi.vmax", 1)
    expect_count(source, "pto.vmi.vcmax", 1)

    with tempfile.TemporaryDirectory() as temp_dir:
        input_path = Path(temp_dir) / "grouped_scale.pto"
        output_path = Path(temp_dir) / "grouped_scale.vpto"
        input_path.write_text(source, encoding="utf-8")
        result = _core.main(
            [
                "ptoas",
                "--pto-arch=a5",
                "--pto-backend=vpto",
                "--emit-vpto",
                str(input_path),
                "-o",
                str(output_path),
            ]
        )
        if result != 0:
            raise AssertionError(f"PTOAS VPTO lowering failed with exit code {result}")
        vpto = output_path.read_text(encoding="utf-8")

    expect_count(vpto, "pto.vldsx2", 1)
    expect_count(vpto, "pto.vand", 2)
    expect_count(vpto, "pto.vmax", 1)
    expect_count(vpto, "pto.vcgmax", 1)
    expect_count(vpto, "pto.vbitcast", 3)
    if "DINTLV_B16" not in vpto:
        raise AssertionError(f"expected one deinterleaved BF16 load:\n{vpto}")
    if "pto.vcvt" in vpto:
        raise AssertionError(f"grouped-scale reduction must not convert through F32:\n{vpto}")
    if "func.call" in vpto or "@bit_and" in vpto or "@grouped_max" in vpto:
        raise AssertionError(f"Python helpers must be traced inline:\n{vpto}")

    print("ptodsl_vmi_grouped_scale_helpers: PASS")


if __name__ == "__main__":
    main()
