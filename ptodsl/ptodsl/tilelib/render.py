# coding=utf-8
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Render a selected TileLib template to MLIR.

``render_best`` is the seam ``ExpandTileOp`` will call: select the legal/best template
for an op+target+operand specialization, then render it via ptodsl's engine.

CLI (standalone):

    python3 -m ptodsl.tilelib.render --op pto.tadd --target a5 \\
        --tile dst=8x64@ub:f32 --tile src0=8x64@ub:f32 --tile src1=8x64@ub:f32 \\
        -o /tmp/ptodsl_tadd_tilelib.mlir

    python3 -m ptodsl.tilelib.render --op pto.tadds --target a5 \\
        --tile src=8x64@ub:f32 --scalar scalar:f32:1 --tile dst=8x64@ub:f32
"""

from __future__ import annotations

import argparse

from . import registry as _registry
from .metadata import ScalarSpec, ScalarType, TileSpec


def select_and_specialize(op: str, target: str, tile_specs: dict,
                          context_attrs: dict | None = None,
                          candidate_id: str | None = None):
    # Registry selection lazily imports only the module for this (target, op).
    descriptor = _registry.select(op, target, tile_specs, context_attrs, candidate_id)
    return descriptor.specialize(context_attrs=context_attrs or {}, **tile_specs)


def render_best(op: str, target: str, tile_specs: dict,
                context_attrs: dict | None = None,
                candidate_id: str | None = None) -> str:
    return select_and_specialize(op, target, tile_specs, context_attrs, candidate_id).mlir_text()


# ── CLI ─────────────────────────────────────────────────────────────────────────────

_DTYPES = {
    "f32", "f16", "bf16",
    "f8e4m3", "f8e5m2", "hif8", "f4e1m2x2", "f4e2m1x2",
    "i64", "i32", "i16", "i8",
    "si64", "si32", "si16", "si8",
    "ui64", "ui32", "ui16", "ui8",
}


def _parse_tile_arg(spec: str):
    """Parse ``name=RxCxMEM:dtype`` like ``dst=8x64@ub:f32``."""
    name, _, rest = spec.partition("=")
    if not name or not rest:
        raise argparse.ArgumentTypeError(f"invalid --tile {spec!r}; expected name=RxC@mem:dtype")
    shape_mem, _, dtype = rest.partition(":")
    shape_str, _, mem = shape_mem.partition("@")
    mem = mem or "ub"
    dtype = dtype or "f32"
    if dtype not in _DTYPES:
        raise argparse.ArgumentTypeError(f"unsupported dtype {dtype!r} in --tile {spec!r}")
    dims = tuple(int(d) for d in shape_str.split("x"))
    return name, TileSpec(shape=dims, dtype=ScalarType(dtype), memory_space=mem)


def _parse_scalar_arg(spec: str):
    """Parse ``name:dtype[:value]`` like ``scalar:f32:1``."""
    parts = spec.split(":")
    if len(parts) not in {2, 3} or not parts[0]:
        raise argparse.ArgumentTypeError(f"invalid --scalar {spec!r}; expected name:dtype[:value]")
    name, dtype = parts[0], parts[1]
    if dtype not in _DTYPES:
        raise argparse.ArgumentTypeError(f"unsupported dtype {dtype!r} in --scalar {spec!r}")
    value = parts[2] if len(parts) == 3 else None
    return name, ScalarSpec(dtype=ScalarType(dtype), value=value)


def main(argv=None):
    parser = argparse.ArgumentParser(prog="ptodsl.tilelib.render")
    parser.add_argument("--op", required=True)
    parser.add_argument("--target", default="a5")
    parser.add_argument("--tile", action="append", default=[], help="name=RxC@mem:dtype")
    parser.add_argument("--scalar", action="append", default=[], help="name:dtype[:value]")
    parser.add_argument("-o", "--output", default=None)
    args = parser.parse_args(argv)

    operand_specs = dict(_parse_tile_arg(t) for t in args.tile)
    operand_specs.update(_parse_scalar_arg(s) for s in args.scalar)
    text = render_best(args.op, args.target, operand_specs)

    if args.output:
        with open(args.output, "w", encoding="utf-8") as handle:
            handle.write(text)
        print(f"wrote {args.output}")
    else:
        print(text)


if __name__ == "__main__":
    main()


__all__ = ["render_best", "select_and_specialize"]
