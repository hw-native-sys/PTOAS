#!/usr/bin/env python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Indexed DSL offsets preserve bits while normalizing authored signedness."""

from itertools import product
import logging

from ptoas.mlir import ir
from ptoas.mlir.dialects import func
from ptoas.mlir.dialects import pto as dialect
from ptodsl._surface_values import unwrap_surface_value
from ptodsl._vmi_namespace import _normalize_indexed_offsets

from ptodsl import pto

LOGGER = logging.getLogger(__name__)


def indexed_argument_types(offset_dtype, payload_dtype, *, layout, lanes):
    parsed_layout = ir.Attribute.parse(layout) if layout else None
    offset_type = dialect.VMIVRegType.get(
        lanes, ir.Type.parse(offset_dtype), layout=parsed_layout
    )
    payload_type = dialect.VMIVRegType.get(
        lanes, ir.Type.parse(payload_dtype), layout=parsed_layout
    )
    pointer_type = ir.Type.parse(f"!pto.ptr<{payload_dtype}, ub>")
    mask_type = (
        dialect.VMIMaskType.get(
            lanes, "b16" if payload_dtype == "f16" else "b32", layout=parsed_layout
        )
        if layout
        else dialect.VMIMaskType.get(lanes, "pred")
    )
    return [payload_type, pointer_type, offset_type, mask_type]


def emit_indexed(kind, offset_dtype, payload_dtype, *, layout=None, lanes=128):
    with ir.Context() as ctx, ir.Location.unknown():
        dialect.register_dialect(ctx)
        location = ir.Location.file("signed_offsets.py", 17, 9)
        types = indexed_argument_types(
            offset_dtype, payload_dtype, layout=layout, lanes=lanes
        )
        module = ir.Module.create()
        with ir.InsertionPoint(module.body):
            fn = func.FuncOp("probe", (types, []))
        block = fn.add_entry_block()
        value, pointer, offsets, mask = block.arguments
        with ir.InsertionPoint(block) as insertion_point:
            if kind == "normalize":
                result = _normalize_indexed_offsets(
                    offsets, context="test", loc=location, ip=insertion_point
                )
                result_type = unwrap_surface_value(result).type
            elif kind == "vgather":
                result = pto.vmi.vgather(
                    pointer,
                    offsets,
                    mask,
                    pmode="zero",
                    loc=location,
                    ip=insertion_point,
                )
                result_type = unwrap_surface_value(result).type
            else:
                pto.vmi.vscatter(
                    value, pointer, offsets, mask, loc=location, ip=insertion_point
                )
                result_type = None
            func.ReturnOp([])
        module.operation.verify()
        return (
            str(module),
            str(result_type),
            module.operation.get_asm(enable_debug_info=True),
        )


def check_offset_normalization():
    # Test the DSL adaptation itself at every authored integer width, including
    # 8-bit carriers whose unsigned result is not a legal hardware index width.
    for bits in (8, 16, 32):
        for dtype in (f"i{bits}", f"si{bits}", f"ui{bits}"):
            text, result_type, debug = emit_indexed("normalize", dtype, "f32")
            assert result_type == f"!pto.vmi.vreg<128xui{bits}>", result_type
            count = text.count("pto.vmi.vinterpret_cast")
            assert count == (0 if dtype.startswith("ui") else 1), text
            assert "pto.vmi.vcvt" not in text, text
            if count:
                assert 'signed_offsets.py":17:9' in debug, debug


def check_memory_offset_signedness():
    # Public memory builders accept si16/si32 and signless i16/i32 after the
    # same-width DSL conversion; the generated memory ops receive unsigned.
    for kind in ("vgather", "vscatter"):
        for bits, payload in ((16, "f16"), (32, "f32")):
            for dtype in (f"i{bits}", f"si{bits}", f"ui{bits}"):
                text, _, debug = emit_indexed(kind, dtype, payload)
                line = next(
                    line for line in text.splitlines() if f"pto.vmi.{kind} " in line
                )
                assert f"!pto.vmi.vreg<128xui{bits}>" in line, line
                assert text.count("pto.vmi.vinterpret_cast") == (
                    0 if dtype.startswith("ui") else 1
                ), text
                assert 'signed_offsets.py":17:9' in debug, debug


def check_byte_offsets():
    # Byte carriers become unsigned bits before they are zero-extended to
    # the required raw VMI memory width. Native ui8 needs only that extension.
    for kind in ("vgather", "vscatter"):
        for payload, bits in (("f16", 16), ("f32", 32)):
            for dtype in ("i8", "si8", "ui8"):
                text, _, debug = emit_indexed(kind, dtype, payload)
                memory = next(
                    line for line in text.splitlines() if f"pto.vmi.{kind} " in line
                )
                assert f"128xui{bits}" in memory, memory
                assert text.count("pto.vmi.vcvt") == 1, text
                assert text.count("pto.vmi.vinterpret_cast") == (
                    0 if dtype == "ui8" else 1
                ), text
                extension = next(
                    line for line in text.splitlines() if "pto.vmi.vcvt " in line
                )
                assert "128xui8" in extension, extension
                assert 'signed_offsets.py":17:9' in debug, debug


def check_byte_payload_offsets():
    for kind, payload in (("vscatter", "i8"), ("vgather", "i8")):
        text, _, _ = emit_indexed(kind, "si8", payload)
        assert "128xui16" in text and "pto.vmi.vcvt" in text, text


def check_low_precision_scatter_offsets():
    for payload, dtype in product(
        ("!pto.hif8", "!pto.hif8x2", "!pto.f8E8M0"),
        ("i8", "si8", "ui8", "i16", "si16", "ui16"),
    ):
        text, _, debug = emit_indexed("vscatter", dtype, payload)
        memory = next(line for line in text.splitlines() if "pto.vmi.vscatter " in line)
        assert f"128x{payload}" in memory, memory
        assert "128xui16" in memory, memory
        assert text.count("pto.vmi.vinterpret_cast") == (not dtype.startswith("ui")), text
        assert text.count("pto.vmi.vcvt") == dtype.endswith("8"), text
        assert 'signed_offsets.py":17:9' in debug, debug


def check_halfword_offsets_for_b32():
    for kind, payload, dtype in product(
        ("vgather", "vscatter"),
        ("f32", "i32", "si32", "ui32"),
        ("i16", "si16", "ui16"),
    ):
        text, _, debug = emit_indexed(kind, dtype, payload)
        memory = next(
            line for line in text.splitlines() if f"pto.vmi.{kind} " in line
        )
        assert "128xui32" in memory, memory
        assert text.count("pto.vmi.vinterpret_cast") == (dtype != "ui16"), text
        assert text.count("pto.vmi.vcvt") == 1, text
        extension = next(
            line for line in text.splitlines() if "pto.vmi.vcvt " in line
        )
        assert "128xui16" in extension and "128xui32" in extension, extension
        if dtype != "ui16":
            assert text.index("pto.vmi.vinterpret_cast") < text.index("pto.vmi.vcvt"), text
        assert 'signed_offsets.py":17:9' in debug, debug


def check_assigned_layout():
    # Assigned layout and lane count survive the inserted same-width cast.
    text, _, _ = emit_indexed(
        "vgather", "si32", "f32", layout="#pto.vmi.layout<contiguous>"
    )
    assert "128xui32" in text and "layout<contiguous>" in text, text
    for kind in ("vgather", "vscatter"):
        text, _, _ = emit_indexed(
            kind, "si16", "f32", layout="#pto.vmi.layout<contiguous>"
        )
        extension = next(line for line in text.splitlines() if "pto.vmi.vcvt " in line)
        assert extension.count("layout<contiguous>") == 2, extension


def check_invalid_offset_widths():
    # Normalization does not silently narrow a mismatched offset width.
    for kind, index, payload in (
        ("vscatter", "si32", "f16"),
        ("vscatter", "si32", "i8"),
        ("vscatter", "ui32", "!pto.hif8"),
        ("vscatter", "ui32", "!pto.hif8x2"),
        ("vscatter", "ui32", "!pto.f8E8M0"),
    ):
        try:
            emit_indexed(kind, index, payload)
        except ir.MLIRError:
            pass
        else:
            raise AssertionError(
                f"{kind} accepted an illegal {index}/{payload} index/payload width pair"
            )


def main():
    check_offset_normalization()
    check_memory_offset_signedness()
    check_byte_offsets()
    check_byte_payload_offsets()
    check_low_precision_scatter_offsets()
    check_halfword_offsets_for_b32()
    check_assigned_layout()
    check_invalid_offset_widths()
    LOGGER.info("ptodsl_vmi_indexed_offset_signedness: PASS")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    main()
