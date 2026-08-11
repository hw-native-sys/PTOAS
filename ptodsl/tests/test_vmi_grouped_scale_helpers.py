#!/usr/bin/env python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""Keep the complete VMI grouped-scale pipeline aligned with direct CCE."""

from pathlib import Path
import tempfile

from ptoas import _core
from ptodsl import pto


def bit_and(lhs, rhs, mask):
    return pto.vmi.vand(lhs, rhs, mask)


def grouped_max(source, mask):
    return pto.vmi.vcmax(source, mask, group=8)


@pto.jit(
    name="vmi_grouped_scale_pipeline",
    kernel_kind="vector",
    target="a5",
    backend="vpto",
    mode="explicit",
    insert_sync=False,
)
def vmi_grouped_scale_pipeline(
    source_gm: pto.ptr(pto.bf16, "gm"),
    destination_gm: pto.ptr(pto.f8e4m3, "gm"),
    scale_gm: pto.ptr(pto.bf16, "gm"),
):
    source_ub = pto.castptr(pto.ui64(0), pto.ptr(pto.bf16, "ub"))
    destination_ub = pto.castptr(pto.ui64(32768), pto.ptr(pto.f8e4m3, "ub"))
    scale_ub = pto.castptr(pto.ui64(33024), pto.ptr(pto.bf16, "ub"))

    pto.mte_gm_ub(source_gm, source_ub, 0, 512, nburst=(1, 512, 512))
    pto.set_flag("MTE2", "V", event_id=0)
    pto.wait_flag("MTE2", "V", event_id=0)

    offset = pto.const(0, dtype=pto.index)
    mask = pto.vmi.create_mask(256, size=256)
    source = pto.vmi.vload(source_ub, offset, size=256)
    source_bits = pto.vmi.vinterpret_cast(source, to_dtype=pto.ui16)
    abs_bits = pto.vmi.vbrc(pto.ui16(0x7FFF), size=256)

    # VMI carries 256 logical lanes here. Lowering splits the bitwise abs into
    # the CCE peer's two vand operations, combines the physical halves with
    # vmax, and then emits vcgmax for the eight groups.
    absolute = bit_and(source_bits, abs_bits, mask)
    maxima = grouped_max(absolute, mask)
    scale = pto.vmi.vinterpret_cast(maxima, to_dtype=pto.bf16)
    expanded_scale = pto.vmi.vbrc(scale, size=256, group=8)
    scaled = pto.vmi.vmul(source, expanded_scale, mask)
    scaled_f32 = pto.vmi.vcvt(scaled, to_dtype=pto.f32)
    quantized = pto.vmi.vcvt(
        scaled_f32,
        to_dtype=pto.f8e4m3,
        rounding="R",
        saturate="SAT",
    )
    pto.vmi.vstore(quantized, destination_ub, offset)
    pto.vmi.vstore(scale, scale_ub, offset, group=8, stride=1)

    pto.set_flag("V", "MTE3", event_id=1)
    pto.wait_flag("V", "MTE3", event_id=1)
    pto.mte_ub_gm(destination_ub, destination_gm, 256, nburst=(1, 256, 256))
    pto.mte_ub_gm(scale_ub, scale_gm, 16, nburst=(1, 16, 16))


def expect_count(text: str, op_name: str, expected: int) -> None:
    actual = text.count(f"{op_name} ")
    if actual != expected:
        raise AssertionError(
            f"expected {expected} occurrence(s) of {op_name}, got {actual}:\n{text}"
        )


def main() -> None:
    compiled = vmi_grouped_scale_pipeline.compile()
    compiled.verify()
    source = compiled.mlir_text()

    expect_count(source, "pto.vmi.vand", 1)
    expect_count(source, "pto.vmi.vcmax", 1)
    expect_count(source, "pto.vmi.vmul", 1)
    expect_count(source, "pto.vmi.vcvt", 2)

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
    expect_count(vpto, "pto.vmul", 2)
    expect_count(vpto, "pto.vcvt", 8)
    expect_count(vpto, "pto.vor", 3)
    expect_count(vpto, "pto.vsts", 2)
    if "DINTLV_B16" not in vpto:
        raise AssertionError(f"expected one deinterleaved BF16 load:\n{vpto}")
    reduction = vpto[: vpto.index("pto.vcgmax")]
    if "pto.vcvt" in reduction or "pto.vabs" in reduction:
        raise AssertionError(
            f"grouped-scale reduction must not convert through F32:\n{vpto}"
        )
    broadcast = vpto[vpto.index("pto.vcgmax") : vpto.index("pto.vmul")]
    if "pto.vsts" in broadcast or "pto.vlds" in broadcast:
        raise AssertionError(
            f"grouped scale must stay in registers through broadcast:\n{vpto}"
        )
    if "func.call" in vpto or "@bit_and" in vpto or "@grouped_max" in vpto:
        raise AssertionError(f"Python helpers must be traced inline:\n{vpto}")

    print("ptodsl_vmi_grouped_scale_pipeline: PASS")


if __name__ == "__main__":
    main()
