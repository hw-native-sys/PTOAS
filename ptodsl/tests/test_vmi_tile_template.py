#!/usr/bin/env python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

from pathlib import Path
import shutil
import subprocess
import sys
from tempfile import TemporaryDirectory
from types import ModuleType


sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "ptodsl"))

from ptodsl._tile_template_tracing import (
    CanonicalBlockMap,
    Tile,
    TileSpec,
    bf16,
    f16,
    f32,
    for_,
    i32,
    make_mask,
    scalar_const,
    tile_template,
    vadd,
    vecscope,
    vlds,
    vsts,
)
from ptodsl.tilelib.registry import TileTemplateRegistry
from ptodsl.tilelib.constraints import evaluate_candidate
from ptodsl.vmi_tilelib import (
    VMI_TILELIB_REGISTRY,
    vmi_tadd_block64,
    vmi_tadds,
    vmi_tcolmax,
    vmi_tcolsum,
    vmi_tcolexpand,
    vmi_tcolexpandmul,
    vmi_tcolexpandsub,
    vmi_tcvt,
    vmi_tabs,
    vmi_texp_block64,
    vmi_texpands,
    vmi_texpands_bf16,
    vmi_texpands_f16,
    vmi_texpands_i32,
    vmi_tneg,
    vmi_tmul,
    vmi_tmuls,
    vmi_trowexpanddiv,
    vmi_trowexpandmul,
    vmi_trowmax,
    vmi_trowmax_row,
    vmi_trowsum,
    vmi_trowsum_row,
    vmi_tsub,
    vmi_tsubs,
)
from ptodsl.vmi_tilelib_helper import instantiate_candidate


TILE_SHAPE = (32, 64)
WIDE_TILE_SHAPE = (32, 128)
NARROW_TILE_SHAPE = (1, 32)
ROPE_TILE_SHAPE = (64, 32)
RMSNORM_TILE_SHAPE = (8, 128)


@tile_template(op="tadd", name="legacy_vpto_tadd")
def legacy_vpto_tadd(src0: Tile, src1: Tile, dst: Tile):
    with vecscope():
        rows, cols = dst.valid_shape
        with for_(0, rows, step=1) as row:
            remained = scalar_const(256, i32)
            with for_(0, cols, step=64) as col:
                mask, _ = make_mask(dst.element_type, remained)
                lhs = vlds(src0[row, col:])
                rhs = vlds(src1[row, col:])
                vsts(vadd(lhs, rhs, mask), dst[row, col:], mask)


def expect(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def expect_raises(callback, exc_type, *message_fragments: str) -> None:
    try:
        callback()
    except exc_type as exc:
        text = str(exc)
        for fragment in message_fragments:
            expect(fragment in text, f"expected diagnostic fragment {fragment!r} in {text!r}")
    else:
        raise AssertionError(f"expected {exc_type.__name__} to be raised")


def specialize_tadd(dtype=f32, shape=TILE_SHAPE):
    spec = TileSpec(shape, dtype)
    return vmi_tadd_block64.specialize(src0=spec, src1=spec, dst=spec)


def specialize_texp(dtype=f32, shape=TILE_SHAPE):
    spec = TileSpec(shape, dtype)
    return vmi_texp_block64.specialize(src=spec, dst=spec)


def check_canonical_block_map() -> None:
    block_map = CanonicalBlockMap(TILE_SHAPE, logical_lanes=64)
    expect(block_map.blocks_per_row == 1, "[32,64]xf32 should contain one block per row")
    expect(block_map.logical_block_count == 32, "[32,64]xf32 should contain 32 blocks")

    coordinate = block_map.coordinate(17)
    expect(coordinate.row == 17, "logical block 17 should map directly to row 17")
    expect(coordinate.block_in_row == 0, "each row should contain only block 0")
    expect(coordinate.col_start == 0, "the row-local block should start at column 0")
    expect(coordinate.linear_offset == 1088, "logical block 17 should start at offset 1088")
    expect(coordinate.active_lanes == 64, "the f32 contract should activate 64 lanes")

    wide_block_map = CanonicalBlockMap(WIDE_TILE_SHAPE, logical_lanes=128)
    expect(wide_block_map.blocks_per_row == 1, "[32,128]xf32 should contain one block per row")
    expect(wide_block_map.logical_block_count == 32, "[32,128]xf32 should contain 32 blocks")
    wide_coordinate = wide_block_map.coordinate(17)
    expect(wide_coordinate.row == 17, "wide logical block 17 should map directly to row 17")
    expect(wide_coordinate.col_start == 0, "the wide row-local block should start at column 0")
    expect(wide_coordinate.linear_offset == 2176, "wide logical block 17 should start at offset 2176")
    expect(wide_coordinate.active_lanes == 128, "wide rows should activate their full inner width")

    narrow_block_map = CanonicalBlockMap(NARROW_TILE_SHAPE, logical_lanes=32)
    expect(narrow_block_map.blocks_per_row == 1, "[1,32]xf32 should contain one block per row")
    expect(narrow_block_map.logical_block_count == 1, "[1,32]xf32 should contain one block")

    expect_raises(
        lambda: CanonicalBlockMap((32, 128), logical_lanes=64),
        ValueError,
        "exactly one logical VL block per row",
    )
    expect_raises(
        lambda: CanonicalBlockMap((32, 32), logical_lanes=64),
        ValueError,
        "exactly one logical VL block per row",
    )


def check_candidate_ir() -> tuple[str, str, str]:
    tadd = specialize_tadd()
    tadd.verify()
    tadd_text = tadd.mlir_text()
    expect("pto.vecscope" not in tadd_text, "VMI templates must remain scope-free")
    expect(tadd_text.count("scf.for") == 1, "tadd candidate should contain one flat loop")
    expect("arith.constant 32 : index" in tadd_text, "tadd should iterate once per row")
    expect(tadd_text.count("pto.vmi.vload") == 2, "tadd should issue two VMI loads")
    expect(tadd_text.count("pto.vmi.vadd") == 1, "tadd should issue one VMI add")
    expect(tadd_text.count("pto.vmi.vstore") == 1, "tadd should issue one VMI store")
    expect(tadd_text.count("pto.tile_buf_addr") == 3, "tadd should materialize three tile pointers")
    expect(
        tadd_text.rfind("pto.tile_buf_addr") < tadd_text.index("scf.for"),
        "tadd tile pointers should be materialized before the logical-block loop",
    )
    expect("!pto.vmi.vreg<64xf32>" in tadd_text, "tadd should use 64 logical f32 lanes")
    expect("pto.vlds" not in tadd_text, "VMI candidate should not emit physical vlds")
    expect("pto.vsts" not in tadd_text, "VMI candidate should not emit physical vsts")

    wide_tadd = specialize_tadd(shape=WIDE_TILE_SHAPE)
    wide_tadd.verify()
    wide_tadd_text = wide_tadd.mlir_text()
    expect(
        wide_tadd_text.count("scf.for") == 1,
        "wide tadd should contain one chunk loop",
    )
    expect(
        "arith.constant 4096 : index" in wide_tadd_text
        and "arith.constant 64 : index" in wide_tadd_text,
        "wide tadd should traverse its full storage in native chunks",
    )
    expect(
        "!pto.vmi.vreg<64xf32>" in wide_tadd_text,
        "wide tadd should use native f32 vregs",
    )
    expect(
        "!pto.vmi.vreg<128xf32>" not in wide_tadd_text,
        "wide tadd should avoid split vregs",
    )
    expect(
        wide_tadd_text.count("pto.vmi.vadd") == 1,
        "wide tadd should issue one chunk add",
    )

    texp = specialize_texp()
    texp.verify()
    texp_text = texp.mlir_text()
    expect("pto.vecscope" not in texp_text, "VMI templates must remain scope-free")
    expect(texp_text.count("scf.for") == 1, "texp candidate should contain one flat loop")
    expect(texp_text.count("pto.vmi.vload") == 1, "texp should issue one VMI load")
    expect(texp_text.count("pto.vmi.vexp") == 1, "texp should issue one VMI exp")
    expect(texp_text.count("pto.vmi.vstore") == 1, "texp should issue one VMI store")

    f16_tadd = specialize_tadd(dtype=f16)
    f16_tadd.verify()
    f16_text = f16_tadd.mlir_text()
    expect(
        "pto.vmi.vadd" in f16_text,
        "vmi_tadd_block64 should accept f16 tiles",
    )
    wide_texp = specialize_texp(shape=WIDE_TILE_SHAPE)
    wide_texp.verify()
    wide_texp_text = wide_texp.mlir_text()
    expect(wide_texp_text.count("scf.for") == 1, "wide texp should contain one chunk loop")
    expect(
        "!pto.vmi.vreg<64xf32>" in wide_texp_text,
        "wide texp should use native f32 vregs",
    )
    expect(
        "!pto.vmi.vreg<128xf32>" not in wide_texp_text,
        "wide texp should avoid split vregs",
    )

    one_row = specialize_tadd(shape=(1, 256))
    one_row.verify()
    one_row_text = one_row.mlir_text()
    expect(
        one_row_text.count("scf.for") == 1,
        "one-row multi-VL tadd should contain one flat chunk loop",
    )
    expect(
        "arith.constant 256 : index" in one_row_text
        and "arith.constant 64 : index" in one_row_text,
        "one-row 256-lane tadd should step through native-sized offsets",
    )
    expect(
        "!pto.vmi.vreg<64xf32>" in one_row_text,
        "one-row multi-VL tadd should keep native f32 vregs",
    )
    expect(
        "!pto.vmi.vreg<256xf32>" not in one_row_text,
        "one-row multi-VL tadd should not materialize a wide logical vreg",
    )

    non_divisible = specialize_tadd(shape=(1, 96))
    non_divisible.verify()
    non_divisible_text = non_divisible.mlir_text()
    expect(
        "!pto.vmi.vreg<128xf32>" in non_divisible_text,
        "non-divisible one-row widths should snap to the next legal VMI vreg",
    )
    expect(
        "arith.constant 1 : index" in non_divisible_text,
        "non-divisible one-row widths should still iterate once by row",
    )

    one_row_fill_spec = TileSpec((1, 256), f32)
    one_row_fill = vmi_texpands.specialize(
        scalar=f32, dst=one_row_fill_spec
    )
    one_row_fill.verify()
    one_row_fill_text = one_row_fill.mlir_text()
    expect(
        "arith.constant 256 : index" in one_row_fill_text
        and "arith.constant 64 : index" in one_row_fill_text,
        "one-row scalar fill should step through native-sized offsets",
    )
    expect(
        "!pto.vmi.vreg<64xf32>" in one_row_fill_text,
        "one-row scalar fill should broadcast one native f32 vreg",
    )
    expect(
        one_row_fill_text.index("pto.vmi.vbrc")
        < one_row_fill_text.index("scf.for"),
        "one-row scalar fill should hoist its native broadcast",
    )
    return tadd_text, wide_tadd_text, texp_text


def check_rope_128b_candidates() -> dict[str, tuple[str, str]]:
    """DSv4 RoPE full-shape elementwise candidates use native chunks."""

    wide = TileSpec(ROPE_TILE_SHAPE, f32)
    column = TileSpec((1, ROPE_TILE_SHAPE[1]), f32)
    scalar_candidates = (
        ("vmi_tmuls_rope128", vmi_tmuls.specialize(src=wide, scale=f32, dst=wide), "pto.vmuls"),
        ("vmi_tadds_rope128", vmi_tadds.specialize(src=wide, scalar=f32, dst=wide), "pto.vadds"),
    )
    binary_candidates = (
        ("vmi_tsub_rope128", vmi_tsub.specialize(src0=wide, src1=wide, dst=wide), "pto.vsub"),
        ("vmi_tmul_rope128", vmi_tmul.specialize(src0=wide, src1=wide, dst=wide), "pto.vmul"),
        ("vmi_tadd_rope128", vmi_tadd_block64.specialize(src0=wide, src1=wide, dst=wide), "pto.vadd"),
    )
    candidates = (*scalar_candidates, *binary_candidates)
    lowering_cases = {}
    for name, artifact, expected_op in candidates:
        artifact.verify()
        text = artifact.mlir_text()
        expect(text.count("scf.for") == 1, f"{name} should contain one chunk loop")
        expect(
            "arith.constant 2048 : index" in text
            and "arith.constant 64 : index" in text,
            f"{name} should cover the full tile in native f32 chunks",
        )
        expect(
            "!pto.vmi.vreg<64xf32>" in text,
            f"{name} should use one native f32 vector per iteration",
        )
        expect(
            "arith.muli" not in text,
            f"{name} should use its chunk induction variable as the linear offset",
        )
        lowering_cases[name] = (text, expected_op)

    fill = vmi_texpands.specialize(scalar=f32, dst=wide)
    fill.verify()
    fill_text = fill.mlir_text()
    expect(fill_text.count("scf.for") == 1, "RoPE texpands should contain one chunk loop")
    expect(
        "arith.constant 2048 : index" in fill_text
        and "!pto.vmi.vreg<64xf32>" in fill_text,
        "RoPE texpands should fill the full tile in native f32 chunks",
    )
    lowering_cases["vmi_texpands_rope128"] = (fill_text, "pto.vdup")

    col_mul = vmi_tcolexpandmul.specialize(src=wide, col_values=column, dst=wide)
    col_mul.verify()
    col_mul_text = col_mul.mlir_text()
    expect(col_mul_text.count("scf.for") == 1, "RoPE tcolexpandmul should contain one row loop")
    expect(
        col_mul_text[: col_mul_text.index("scf.for")].count("pto.vmi.vload") == 1,
        "RoPE tcolexpandmul should hoist its column vector load",
    )
    expect(
        "!pto.vmi.vreg<64xf32>" in col_mul_text,
        "RoPE column multiply should snap the 32-column row to a legal 64-lane vreg",
    )
    lowering_cases["vmi_tcolexpandmul_rope128"] = (col_mul_text, "pto.vmul")

    tail = TileSpec(ROPE_TILE_SHAPE, f32, valid_shape=(63, 32))
    tail_tmuls = vmi_tmuls.specialize(src=tail, scale=f32, dst=tail)
    tail_tmuls.verify()
    tail_text = tail_tmuls.mlir_text()
    expect(
        "!pto.vmi.vreg<64xf32>" in tail_text
        and "arith.constant 2048 : index" not in tail_text,
        "RoPE tails must retain the row-aware masked form",
    )

    i32_tile = TileSpec(ROPE_TILE_SHAPE, i32)
    bf16_tile = TileSpec(ROPE_TILE_SHAPE, bf16)
    conversions = (
        (
            "vmi_tcvt_i32_f32_rope128",
            vmi_tcvt.specialize(
                src=i32_tile,
                dst=wide,
                context_attrs={"round_mode": "ROUND", "sat_mode": "OFF"},
            ),
        ),
        (
            "vmi_tcvt_f32_i32_rope128",
            vmi_tcvt.specialize(
                src=wide,
                dst=i32_tile,
                context_attrs={"round_mode": "TRUNC", "sat_mode": "OFF"},
            ),
        ),
        (
            "vmi_tcvt_f32_bf16_rope128",
            vmi_tcvt.specialize(
                src=wide,
                dst=bf16_tile,
                context_attrs={"round_mode": "RINT", "sat_mode": "OFF"},
            ),
        ),
    )
    for name, artifact in conversions:
        artifact.verify()
        text = artifact.mlir_text()
        expect(text.count("scf.for") == 1, f"{name} should contain one chunk loop")
        expect("pto.vmi.vcvt" in text, f"{name} should emit a VMI conversion")
        expect(
            "arith.constant 2048 : index" in text and "vreg<64x" in text,
            f"{name} should convert the full tile in 64-lane chunks",
        )
        if name == "vmi_tcvt_i32_f32_rope128":
            expect(
                "rounding" not in text,
                "integer-to-float widening must not carry VMI rounding",
            )
        if name == "vmi_tcvt_f32_i32_rope128":
            expect(
                'saturate = "NOSAT"' in text,
                "RoPE f32->i32 should preserve the TileOp default saturation mode",
            )
        lowering_cases[name] = (text, "pto.vcvt")
    return lowering_cases


def check_rmsnorm_256b_row_candidates() -> dict[str, tuple[str, str]]:
    """Full RMSNorm rows use one native f32 chunk per loop iteration."""

    f32_tile = TileSpec(RMSNORM_TILE_SHAPE, f32)
    bf16_tile = TileSpec(RMSNORM_TILE_SHAPE, bf16)
    candidates = (
        (
            "vmi_tmul_rmsnorm256",
            vmi_tmul.specialize(src0=f32_tile, src1=f32_tile, dst=f32_tile),
            "pto.vmul",
        ),
        (
            "vmi_tcvt_bf16_f32_rmsnorm256",
            vmi_tcvt.specialize(
                src=bf16_tile,
                dst=f32_tile,
                context_attrs={"round_mode": "ROUND", "sat_mode": "OFF"},
            ),
            "pto.vcvt",
        ),
    )
    lowering_cases = {}
    for name, artifact, expected_op in candidates:
        artifact.verify()
        text = artifact.mlir_text()
        expect(text.count("scf.for") == 1, f"{name} should contain one chunk loop")
        expect(
            "arith.constant 1024 : index" in text
            and "arith.constant 64 : index" in text,
            f"{name} should cover the full tile in native f32 chunks",
        )
        expect(
            "!pto.vmi.vreg<64xf32>" in text,
            f"{name} should avoid a split 128-lane f32 value",
        )
        expect(
            "!pto.vmi.vreg<128xf32>" not in text,
            f"{name} should not materialize a wide f32 row",
        )
        lowering_cases[name] = (text, expected_op)
    return lowering_cases


def check_local_elementwise_candidates() -> dict[str, tuple[str, str]]:
    shape = (8, 512)
    fill_candidates = (
        ("vmi_texpands", vmi_texpands, f32),
        ("vmi_texpands_f16", vmi_texpands_f16, f16),
        ("vmi_texpands_bf16", vmi_texpands_bf16, bf16),
        ("vmi_texpands_i32", vmi_texpands_i32, i32),
    )
    lowering_cases = {}
    for name, candidate, dtype in fill_candidates:
        spec = TileSpec(shape, dtype)
        artifact = candidate.specialize(scalar=dtype, dst=spec)
        artifact.verify()
        text = artifact.mlir_text()
        expect(text.count("scf.for") == 1, f"{name} should contain one row loop")
        expect(text.count("pto.vmi.vbrc") == 1, f"{name} should broadcast once")
        expect(text.count("pto.vmi.vstore") == 1, f"{name} should store one logical row")
        expect(
            text.index("pto.vmi.vbrc") < text.index("scf.for"),
            f"{name} should hoist its invariant broadcast",
        )
        lowering_cases[name] = (text, "pto.vdup")

    f32_spec = TileSpec(shape, f32)
    elementwise = (
        (
            "vmi_tsubs",
            vmi_tsubs.specialize(src=f32_spec, scalar=f32, dst=f32_spec),
            "pto.vmi.vadds",
            "pto.vadds",
        ),
        (
            "vmi_tabs",
            vmi_tabs.specialize(src=f32_spec, dst=f32_spec),
            "pto.vmi.vabs",
            "pto.vabs",
        ),
        (
            "vmi_tneg",
            vmi_tneg.specialize(src=f32_spec, dst=f32_spec),
            "pto.vmi.vneg",
            "pto.vneg",
        ),
    )
    for name, artifact, vmi_op, vpto_op in elementwise:
        artifact.verify()
        text = artifact.mlir_text()
        expect(text.count("scf.for") == 1, f"{name} should contain one row loop")
        expect(text.count("pto.vmi.vload") == 1, f"{name} should load one logical row")
        expect(text.count(vmi_op) == 1, f"{name} should emit {vmi_op}")
        expect(text.count("pto.vmi.vstore") == 1, f"{name} should store one logical row")
        lowering_cases[name] = (text, vpto_op)

    tsubs_text = lowering_cases["vmi_tsubs"][0]
    expect(tsubs_text.count("arith.subf") == 1, "tsubs should negate its scalar once")
    expect(
        tsubs_text.index("arith.subf") < tsubs_text.index("scf.for"),
        "tsubs scalar negation should be loop invariant",
    )
    for op in ("texpands", "tsubs", "tabs", "tneg"):
        candidates = VMI_TILELIB_REGISTRY.lookup(op, "a5")
        expect(candidates, f"{op} should register at least one VMI candidate")
        for candidate in candidates:
            expect(
                candidate.metadata.tags[:3]
                == ("vmi", "fusion_eligible", "single_logical_row_loop"),
                f"{candidate.name} should carry the canonical VMI fusion tags",
            )
    return lowering_cases


def check_local_broadcast_candidates() -> dict[str, tuple[str, str]]:
    rows, cols = 8, 512
    wide = TileSpec((rows, cols), f32)
    compact = TileSpec((rows, 1), f32, b_layout="col_major")
    column = TileSpec((1, cols), f32)
    binary = (
        (
            "vmi_trowexpandmul",
            vmi_trowexpandmul.specialize(src=wide, row_values=compact, dst=wide),
            "pto.vmi.vmul",
            "pto.vmul",
        ),
        (
            "vmi_trowexpanddiv",
            vmi_trowexpanddiv.specialize(
                src=wide,
                row_values=compact,
                dst=wide,
                context_attrs={"precisionType": "default"},
            ),
            "pto.vmi.vdiv",
            "pto.vdiv",
        ),
    )
    lowering_cases = {}
    for name, artifact, vmi_op, vpto_op in binary:
        artifact.verify()
        text = artifact.mlir_text()
        expect(text.count("scf.for") == 1, f"{name} should contain one row loop")
        expect(
            text.count("pto.vmi.vload") == 2,
            f"{name} should load one data row and one compact row state",
        )
        expect(text.count("pto.vmi.vgather") == 0, f"{name} should not use gather for compact row state")
        expect(text.count("pto.vmi.vbrc") == 0, f"{name} should use a native broadcast load")
        expect(text.count('dist_mode = "brc"') == 1, f"{name} should broadcast-load one compact row state")
        expect(text.count(vmi_op) == 1, f"{name} should emit {vmi_op}")
        expect(text.count("pto.vmi.vstore") == 1, f"{name} should store one row")
        lowering_cases[name] = (text, vpto_op)

    artifact = vmi_tcolexpand.specialize(src=column, dst=wide)
    artifact.verify()
    text = artifact.mlir_text()
    expect(text.count("scf.for") == 1, "tcolexpand should contain one row loop")
    expect(text.count("pto.vmi.vload") == 1, "tcolexpand should load once")
    expect(
        text.index("pto.vmi.vload") < text.index("scf.for"),
        "tcolexpand source should be loop invariant",
    )
    expect(text.count("pto.vmi.vstore") == 1, "tcolexpand should store one row")
    lowering_cases["vmi_tcolexpand"] = (text, "pto.vsts")

    for op in ("trowexpandmul", "trowexpanddiv", "tcolexpand"):
        candidates = VMI_TILELIB_REGISTRY.lookup(op, "a5")
        expect(candidates, f"{op} should register at least one VMI candidate")
        for candidate in candidates:
            expect(
                candidate.metadata.tags[:3]
                == ("vmi", "fusion_eligible", "single_logical_row_loop"),
                f"{candidate.name} should carry canonical VMI fusion tags",
            )

    row_specs = {"src": wide, "row_values": compact, "dst": wide}
    # The VMI row-expand emit path loads each row with a single VMI vreg, which
    # maxes out at 256 lanes.  The P1-2 gating keeps shapes whose logical row
    # exceeds that ceiling on the ordinary fallback instead of silently
    # truncating the trailing columns (a 512-column f32 row would drop the
    # second half), matching the row-reduce/streaming and col-expand gates.
    expect(
        not evaluate_candidate(
            vmi_trowexpanddiv,
            row_specs,
            "a5",
            "pto.trowexpanddiv",
            {"precisionType": "default"},
        ).legal,
        "a 512-column row expand must remain a fallback under the 256-lane VMI ceiling",
    )
    expect(
        not evaluate_candidate(
            vmi_trowexpanddiv,
            row_specs,
            "a5",
            "pto.trowexpanddiv",
            {"precisionType": "high_precision"},
        ).legal,
        "high-precision row expand must remain a fallback",
    )
    row_major_state = TileSpec((rows, 1), f32)
    expect(
        not evaluate_candidate(
            vmi_trowexpandmul,
            {"src": wide, "row_values": row_major_state, "dst": wide},
            "a5",
            "pto.trowexpandmul",
        ).legal,
        "the VMI row-expand form must require col-major [rows, 1] state",
    )
    tail_wide = TileSpec((rows, cols), f32, valid_shape=(rows - 1, cols))
    tail_compact = TileSpec(
        (rows, 1),
        f32,
        valid_shape=(rows - 1, 1),
        b_layout="col_major",
    )
    expect(
        not evaluate_candidate(
            vmi_trowexpandmul,
            {"src": tail_wide, "row_values": tail_compact, "dst": tail_wide},
            "a5",
            "pto.trowexpandmul",
        ).legal,
        "tail row-expand form must remain a fallback",
    )
    static_subregion = TileSpec(
        (rows, cols), f32, valid_shape=(rows, 448)
    )
    static_subregion_dst = TileSpec((rows, 448), f32)
    subregion_specs = {
        "src": static_subregion,
        "row_values": compact,
        "dst": static_subregion_dst,
    }
    expect(
        not evaluate_candidate(
            vmi_trowexpandmul,
            subregion_specs,
            "a5",
            "pto.trowexpandmul",
        ).legal,
        "an unregistered storage subregion should remain a fallback",
    )
    unsafe_prefix = TileSpec((rows, 32), f32, valid_shape=(rows, 16))
    expect(
        not evaluate_candidate(
            vmi_trowexpandmul,
            {
                "src": unsafe_prefix,
                "row_values": compact,
                "dst": TileSpec((rows, 16), f32),
            },
            "a5",
            "pto.trowexpandmul",
        ).legal,
        "a prefix that overreads its physical row must remain a fallback",
    )
    return lowering_cases


def check_provider_helper() -> None:
    registered_tadd = VMI_TILELIB_REGISTRY.lookup("tadd", "a5")
    expect(
        vmi_tadd_block64 in registered_tadd,
        "tadd must retain its canonical wide VMI template",
    )
    expect(
        len({candidate.name for candidate in registered_tadd})
        == len(registered_tadd),
        "tadd VMI semantic forms must use unique candidate names",
    )
    expect(
        dict(vmi_texp_block64.context_constraints)
        == {"precisionType": ("default",)},
        "texp must declare its supported context attrs on the candidate",
    )

    raw_tile_spec = {
        "kind": "tile",
        "dtype": "f32",
        "shape": [32, 64],
        "valid_shape": [32, 64],
        "memory_space": "ub",
        "config": {
            "b_layout": "row_major",
            "s_layout": "none_box",
            "s_fractal_size": 512,
            "pad_value": "0x0",
        },
    }
    artifact = instantiate_candidate(
        target="a5",
        op_name="pto.tadd",
        operand_specs=[raw_tile_spec, raw_tile_spec, raw_tile_spec],
        provider_module="ptodsl.vmi_tilelib",
        context_attrs={},
    )
    text = artifact.mlir_text()
    expect("pto.vmi.vadd" in text, "provider helper should instantiate the tadd VMI candidate")
    expect(text.count("scf.for") == 1, "provider helper should preserve one logical-block loop")

    f16_tile_spec = {
        **raw_tile_spec,
        "dtype": "f16",
        "shape": [32, 128],
        "valid_shape": [32, 128],
    }
    f16_artifact = instantiate_candidate(
        target="a5",
        op_name="pto.tadd",
        operand_specs=[f16_tile_spec, f16_tile_spec, f16_tile_spec],
        provider_module="ptodsl.vmi_tilelib",
        context_attrs={},
    )
    f16_text = f16_artifact.mlir_text()
    expect(
        "pto.vmi.vadd" in f16_text,
        "provider helper should instantiate the f16 tadd VMI candidate",
    )
    expect(
        "!pto.vmi.vreg<128xf16>" in f16_text,
        "f16 multi-row tadd should chunk per 128-lane native vreg",
    )

    exp_artifact = instantiate_candidate(
        target="a5",
        op_name="pto.texp",
        operand_specs=[raw_tile_spec, raw_tile_spec],
        provider_module="ptodsl.vmi_tilelib",
        context_attrs={"precisionType": "default"},
    )
    expect(
        "pto.vmi.vexp" in exp_artifact.mlir_text(),
        "provider helper should accept the default texp precision contract",
    )
    expect_raises(
        lambda: instantiate_candidate(
            target="a5",
            op_name="pto.tadd",
            operand_specs=[raw_tile_spec, raw_tile_spec, raw_tile_spec],
            provider_module="ptodsl.vmi_tilelib",
            context_attrs={"precisionType": "default"},
        ),
        ValueError,
        "does not support context attrs",
    )

    tmul_artifact = instantiate_candidate(
        target="a5",
        op_name="pto.tmul",
        operand_specs=[raw_tile_spec, raw_tile_spec, raw_tile_spec],
        provider_module="ptodsl.vmi_tilelib",
        context_attrs={},
    )
    expect("pto.vmi.vmul" in tmul_artifact.mlir_text(), "tmul should lower to VMI")

    below_min_row_spec = {
        **raw_tile_spec,
        "shape": [1, 16],
        "valid_shape": [1, 16],
    }
    expect_raises(
        lambda: instantiate_candidate(
            target="a5",
            op_name="pto.tadd",
            operand_specs=[below_min_row_spec, below_min_row_spec, below_min_row_spec],
            provider_module="ptodsl.vmi_tilelib",
            context_attrs={},
        ),
        LookupError,
        "custom constraints are not satisfied",
    )

    scalar_spec = {"kind": "scalar", "dtype": "f32"}
    tmuls = instantiate_candidate(
        target="a5",
        op_name="pto.tmuls",
        operand_specs=[raw_tile_spec, scalar_spec, raw_tile_spec],
        provider_module="ptodsl.vmi_tilelib",
        context_attrs={},
    ).mlir_text()
    expect("%arg1: f32" in tmuls, "tmuls should preserve its runtime scalar parameter")
    expect("pto.vmi.vmuls" in tmuls, "tmuls should lower to VMI scalar multiply")

    scalar_expectations = {
        "tadds": "pto.vmi.vadds",
        "tmaxs": "pto.vmi.vmaxs",
        "tmins": "pto.vmi.vmins",
    }
    for op_name, expected_op in scalar_expectations.items():
        text = instantiate_candidate(
            target="a5",
            op_name=f"pto.{op_name}",
            operand_specs=[raw_tile_spec, scalar_spec, raw_tile_spec],
            provider_module="ptodsl.vmi_tilelib",
            context_attrs={},
        ).mlir_text()
        expect(expected_op in text, f"{op_name} should lower to {expected_op}")

    tdivs = instantiate_candidate(
        target="a5",
        op_name="pto.tdivs",
        operand_specs=[raw_tile_spec, scalar_spec, raw_tile_spec],
        provider_module="ptodsl.vmi_tilelib",
        context_attrs={"precisionType": "default"},
    ).mlir_text()
    expect("pto.vmi.vbrc" in tdivs, "tdivs should broadcast its scalar operand")
    expect("pto.vmi.vdiv" in tdivs, "tdivs should lower to VMI vector divide")
    tdivs_hp = instantiate_candidate(
        target="a5",
        op_name="pto.tdivs",
        operand_specs=[raw_tile_spec, scalar_spec, raw_tile_spec],
        provider_module="ptodsl.vmi_tilelib",
        context_attrs={"precisionType": "high_precision"},
    ).mlir_text()
    expect(
        tdivs_hp.count("scf.for") == 1,
        "high-precision tdivs should still emit one logical row loop",
    )
    expect(
        "pto.vmi.vmula" in tdivs_hp,
        "high-precision tdivs should lower to the VMI refinement sequence",
    )

    rowmax_src_spec = {**raw_tile_spec, "shape": [8, 32], "valid_shape": [8, 32]}
    reduced_tile_spec = {
        **raw_tile_spec,
        "shape": [8, 1],
        "valid_shape": [8, 1],
        "config": {**raw_tile_spec["config"], "b_layout": "col_major"},
    }
    rowmax = instantiate_candidate(
        target="a5",
        op_name="pto.trowmax",
        operand_specs=[rowmax_src_spec, rowmax_src_spec, reduced_tile_spec],
        provider_module="ptodsl.vmi_tilelib",
        context_attrs={},
    ).mlir_text()
    expect("scf.for" not in rowmax, "rowmax should emit one grouped reduction")
    expect(rowmax.count("pto.vmi.vcmax") == 1, "rowmax should reduce all row groups")
    expect("group = 8" in rowmax, "rowmax should preserve 8 compact row groups")
    expect("!pto.vmi.vreg<8xf32>" in rowmax, "rowmax should produce one value per row")

    row_expand = instantiate_candidate(
        target="a5",
        op_name="pto.trowexpandsub",
        operand_specs=[rowmax_src_spec, reduced_tile_spec, rowmax_src_spec],
        provider_module="ptodsl.vmi_tilelib",
        context_attrs={},
    ).mlir_text()
    expect(
        'dist_mode = "brc"' in row_expand,
        "row expand should broadcast-load one scalar value per row",
    )
    expect(
        "pto.vmi.vbrc" not in row_expand,
        "row expand should use the native broadcast-load form",
    )

    convert_src_spec = {
        **raw_tile_spec,
        "shape": [32, 128],
        "valid_shape": [32, 128],
    }
    f16_tile_spec = {**convert_src_spec, "dtype": "f16"}
    tcvt = instantiate_candidate(
        target="a5",
        op_name="pto.tcvt",
        operand_specs=[convert_src_spec, f16_tile_spec],
        provider_module="ptodsl.vmi_tilelib",
        context_attrs={"round_mode": "RINT", "sat_mode": "OFF"},
    ).mlir_text()
    expect("pto.vmi.vcvt" in tcvt, "tcvt should lower to VMI conversion")

    tdiv = instantiate_candidate(
        target="a5",
        op_name="pto.tdiv",
        operand_specs=[raw_tile_spec, raw_tile_spec, raw_tile_spec],
        provider_module="ptodsl.vmi_tilelib",
        context_attrs={"precisionType": "default"},
    ).mlir_text()
    expect("pto.vmi.vdiv" in tdiv, "default tdiv should lower to VMI vector divide")
    tdiv_hp = instantiate_candidate(
        target="a5",
        op_name="pto.tdiv",
        operand_specs=[raw_tile_spec, raw_tile_spec, raw_tile_spec],
        provider_module="ptodsl.vmi_tilelib",
        context_attrs={"precisionType": "high_precision"},
    ).mlir_text()
    expect(
        "pto.vmi.vmula" in tdiv_hp,
        "high-precision tdiv should lower to the VMI refinement sequence",
    )
    expect_raises(
        lambda: instantiate_candidate(
            target="a5",
            op_name="pto.texp",
            operand_specs=[raw_tile_spec, raw_tile_spec],
            provider_module="ptodsl.vmi_tilelib",
            context_attrs={"precisionType": "high"},
        ),
        LookupError,
        "no legal PTODSL VMI candidate",
    )

    duplicate_module = ModuleType("ptodsl_test_duplicate_vmi_candidates")
    duplicate_module.VMI_TILELIB_REGISTRY = TileTemplateRegistry()

    @tile_template(target="a5", op="tadd", name="duplicate_tadd_a", ir_level="vmi")
    def duplicate_tadd_a(src0: Tile, src1: Tile, dst: Tile):
        pass

    @tile_template(target="a5", op="tadd", name="duplicate_tadd_b", ir_level="vmi")
    def duplicate_tadd_b(src0: Tile, src1: Tile, dst: Tile):
        pass

    duplicate_module.VMI_TILELIB_REGISTRY.register(duplicate_tadd_a)
    duplicate_module.VMI_TILELIB_REGISTRY.register(duplicate_tadd_b)
    sys.modules[duplicate_module.__name__] = duplicate_module
    try:
        expect_raises(
            lambda: instantiate_candidate(
                target="a5",
                op_name="pto.tadd",
                operand_specs=[raw_tile_spec, raw_tile_spec, raw_tile_spec],
                provider_module=duplicate_module.__name__,
                context_attrs={},
            ),
            LookupError,
            "requires exactly one canonical candidate",
            "found 2",
        )
    finally:
        del sys.modules[duplicate_module.__name__]


def check_col_reduce_candidate() -> tuple[str, str, str]:
    """ColReduce (tcolmax / tcolsum) candidates must lower to one runtime
    ``scf.for`` carrying a VL-wide accumulator as a ``vreg`` iter_arg — mirroring
    the pto-isa ``TColReduceInstr_NoPostUpdate`` repeat loop — and must NOT
    statically unroll one merge per row.

    Each candidate runs over a single-VL-block column tile: src is
    [rows, VL] row-major, dst is [1, VL] row-major (the surviving column axis).
    """
    col_tile_spec = {
        "kind": "tile",
        "dtype": "f32",
        "shape": [32, 64],
        "valid_shape": [32, 64],
        "memory_space": "ub",
        "config": {
            "b_layout": "row_major",
            "s_layout": "none_box",
            "s_fractal_size": 512,
            "pad_value": "0x0",
        },
    }
    reduced_col_spec = {
        **col_tile_spec,
        "shape": [1, 64],
        "valid_shape": [1, 64],
    }

    colmax = instantiate_candidate(
        target="a5",
        op_name="pto.tcolmax",
        operand_specs=[col_tile_spec, reduced_col_spec],
        provider_module="ptodsl.vmi_tilelib",
        context_attrs={},
    ).mlir_text()
    expect("pto.vecscope" not in colmax, "colmax template must remain scope-free")
    expect(colmax.count("scf.for") == 1, "colmax should emit one runtime reduce loop")
    expect(colmax.count("scf.yield") == 1, "colmax should yield the merged accumulator")
    # The vmi_tcolmax candidate uses split=4 (rows=32 is divisible by 4): the
    # loop carries 4 independent VL-wide accumulators (step=4), each loaded and
    # merged once per iteration (4 vload + 4 vmax inside the loop), then merged
    # by a 3-way vmax tree outside the loop (3 more vmax). The accumulator seed
    # is a vbr of the identity, not a dummy vload (a vload carries a Read memory
    # effect and cannot be DCE'd, so a dummy load would duplicate the row-0
    # read).
    expect(
        "step %c4" in colmax and "iter_args" in colmax,
        "colmax split=4 should step by 4 and carry loop-carried accumulators",
    )
    expect(
        colmax.count("!pto.vmi.vreg<64xf32>") >= 4,
        "colmax split=4 should carry at least 4 VL-wide vreg accumulators",
    )
    expect(colmax.count("pto.vmi.vmax") == 7, "colmax split=4 should issue 4 in-loop + 3 merge vmax")
    expect(colmax.count("pto.vmi.vload") == 4, "colmax split=4 should load 4 rows per iteration")
    expect(colmax.count("pto.vmi.vstore") == 1, "colmax should store the reduced result once")
    expect("pto.vmi.vcmax" not in colmax, "colmax must not collapse to a 1-lane vcmax")
    expect("pto.vmi.vreduce_max" not in colmax, "colmax must not collapse to a 1-lane vreduce")

    colsum = instantiate_candidate(
        target="a5",
        op_name="pto.tcolsum",
        operand_specs=[col_tile_spec, reduced_col_spec],
        provider_module="ptodsl.vmi_tilelib",
        context_attrs={},
    ).mlir_text()
    expect(colsum.count("scf.for") == 1, "colsum should emit one runtime reduce loop")
    expect(
        "iter_args" in colsum and "!pto.vmi.vreg<64xf32>" in colsum,
        "colsum should carry a VL-wide vreg accumulator through the loop",
    )
    expect(colsum.count("pto.vmi.vadd") == 1, "colsum should issue one VMI add inside the loop")

    # A non-binary colsum must not accept the binary 3-operand form (it has no
    # fallback path); the two-operand form is the only supported lowering.
    expect_raises(
        lambda: instantiate_candidate(
            target="a5",
            op_name="pto.tcolsum",
            operand_specs=[col_tile_spec, reduced_col_spec, reduced_col_spec],
            provider_module="ptodsl.vmi_tilelib",
            context_attrs={},
        ),
        LookupError,
        "expects 2 operands, got 3",
    )
    return colmax, colsum, reduced_col_spec


def check_col_reduce_split() -> None:
    """The vmi_tcolmax / vmi_tcolmin candidates default to split=4, running 4
    independent VL-wide accumulators when ``rows % 4 == 0``. When the row count
    is NOT divisible by 4, split silently falls back to 1 (single-way, always
    correct) — this exercises both paths so a regression that breaks the
    fallback (e.g. emitting a step=4 loop over a non-divisible trip, which would
    OOB the tail rows) is caught here, not on a real kernel.

    tcolmin mirrors tcolmax (vmax->vmin, -inf identity -> +inf identity).
    """
    base_spec = {
        "kind": "tile",
        "dtype": "f32",
        "memory_space": "ub",
        "config": {
            "b_layout": "row_major",
            "s_layout": "none_box",
            "s_fractal_size": 512,
            "pad_value": "0x0",
        },
    }

    # rows=128 is divisible by 4 -> split=4 active (step=4, 4 accumulators).
    divisible_src = {**base_spec, "shape": [128, 64], "valid_shape": [128, 64]}
    dst_spec = {**base_spec, "shape": [1, 64], "valid_shape": [1, 64]}

    colmax = instantiate_candidate(
        target="a5",
        op_name="pto.tcolmax",
        operand_specs=[divisible_src, dst_spec],
        provider_module="ptodsl.vmi_tilelib",
        context_attrs={},
    ).mlir_text()
    expect(
        "step %c4" in colmax,
        "tcolmax split=4 should step by 4 when rows % 4 == 0",
    )
    expect(
        colmax.count("pto.vmi.vmax") == 4 + 3,
        "tcolmax split=4 should issue 4 in-loop vmax + 3 merge vmax (128 rows)",
    )

    colmin = instantiate_candidate(
        target="a5",
        op_name="pto.tcolmin",
        operand_specs=[divisible_src, dst_spec],
        provider_module="ptodsl.vmi_tilelib",
        context_attrs={},
    ).mlir_text()
    expect(
        "step %c4" in colmin,
        "tcolmin split=4 should step by 4 when rows % 4 == 0",
    )
    expect(
        colmin.count("pto.vmi.vmin") == 4 + 3,
        "tcolmin split=4 should issue 4 in-loop vmin + 3 merge vmin (128 rows)",
    )

    # rows=10 is NOT divisible by 4 -> fallback to split=1 (single-way, step=1,
    # one accumulator, one merge per row). This must not emit step=4 (which
    # would skip the tail rows / OOB the half-open scf.for).
    nondivisible_src = {**base_spec, "shape": [10, 64], "valid_shape": [10, 64]}
    colmax_fb = instantiate_candidate(
        target="a5",
        op_name="pto.tcolmax",
        operand_specs=[nondivisible_src, dst_spec],
        provider_module="ptodsl.vmi_tilelib",
        context_attrs={},
    ).mlir_text()
    expect(
        "step %c4" not in colmax_fb,
        "tcolmax should NOT use step=4 when rows % 4 != 0 (fallback to split=1)",
    )
    expect(
        "step %c1" in colmax_fb,
        "tcolmax fallback should step by 1 (split=1 single-way)",
    )
    expect(
        colmax_fb.count("pto.vmi.vmax") == 1,
        "tcolmax split=1 fallback should issue one vmax op in the loop body "
        "(the loop runs 10 iterations but the body is a template, not unrolled)",
    )
    expect(
        colmax_fb.count("pto.vmi.vload") == 1,
        "tcolmax split=1 fallback should load one row per iteration in the loop body",
    )

    colmin_fb = instantiate_candidate(
        target="a5",
        op_name="pto.tcolmin",
        operand_specs=[nondivisible_src, dst_spec],
        provider_module="ptodsl.vmi_tilelib",
        context_attrs={},
    ).mlir_text()
    expect(
        "step %c4" not in colmin_fb,
        "tcolmin should NOT use step=4 when rows % 4 != 0 (fallback to split=1)",
    )
    expect(
        "step %c1" in colmin_fb,
        "tcolmin fallback should step by 1 (split=1 single-way)",
    )

def check_row_reduce_candidates() -> dict[str, tuple[str, str, int]]:
    lowering_cases = {}
    # The grouped row-reduce emit loads the whole tile as one vector
    # (total_lanes = rows * physical_cols), so the shape is limited to a
    # single 256-lane VMI vreg.  8x32 is the widest legal f32 grouped form;
    # wider tiles take the row_streaming candidates instead.
    for op_name, candidate, physical_op in (
        ("trowmax", vmi_trowmax, "pto.vcmax"),
        ("trowsum", vmi_trowsum, "pto.vcadd"),
    ):
        cols = 32
        src = TileSpec((8, cols), f32)
        workspace = TileSpec((8, max(cols, 128)), f32)
        dst = TileSpec((8, 1), f32, b_layout="col_major")
        artifact = candidate.specialize(src=src, workspace=workspace, dst=dst)
        artifact.verify()
        text = artifact.mlir_text()
        name = f"vmi_{op_name}_{cols}lanes"
        expect("scf.for" not in text, f"{name} should use one grouped reduction")
        expect(
            f"!pto.vmi.vreg<{8 * cols}xf32>" in text,
            f"{name} should reduce the complete logical row group",
        )
        expect(text.count("pto.vmi.vload") == 1, f"{name} should load one row")
        expect("group = 8" in text, f"{name} should preserve eight row groups")
        expected_stores = 1 if cols < f32.lanes else 0
        expect(
            text.count("pto.vmi.vstore") == expected_stores,
            f"{name} should use allocation-safe row-result stores",
        )
        if expected_stores == 0:
            expect(
                "pto.vmi.vscatter" in text,
                f"{name} should scatter compact rows from an aligned base",
            )
        expected_physical_op = (
            physical_op.replace("pto.vc", "pto.vcg")
            if cols < f32.lanes
            else physical_op
        )
        lowering_cases[name] = (text, expected_physical_op, 0)

    # A 8x128 grouped reduction would need a 1024-lane mask, which no VMI vreg
    # can represent (P1-2: VMI vregs max out at 256 lanes).  The row_streaming
    # candidates cover rows wider than one vreg; the grouped form must fall
    # back.  Guard that boundary at the constraint level so the emit path never
    # builds an impossible mask.
    src = TileSpec((8, 128), f32)
    workspace = TileSpec((8, 128), f32)
    dst = TileSpec((8, 1), f32, b_layout="col_major")
    row_specs = {"src": src, "workspace": workspace, "dst": dst}
    for op_name, candidate in (("trowmax", vmi_trowmax), ("trowsum", vmi_trowsum)):
        expect(
            not evaluate_candidate(
                candidate,
                row_specs,
                "a5",
                f"pto.{op_name}",
            ).legal,
            f"8x128 grouped {op_name} must remain a fallback under the 256-lane total ceiling",
        )

    workspace = TileSpec((8, 128), f32)
    dst = TileSpec((8, 1), f32, b_layout="col_major")
    safe_subregion_src = TileSpec((8, 512), f32, valid_shape=(8, 128))
    expect_raises(
        lambda: vmi_trowmax.specialize(
            src=safe_subregion_src, workspace=workspace, dst=dst
        ).mlir_text(),
        ValueError,
        "grouped row-reduce requires a full static source tile",
    )

    partial_src = TileSpec((8, 32), f32, valid_shape=(8, 16))
    expect_raises(
        lambda: vmi_trowsum.specialize(
            src=partial_src, workspace=workspace, dst=dst
        ).mlir_text(),
        ValueError,
        "every physical lane",
    )

    return lowering_cases


def check_row_streaming_reduce_candidates() -> dict[str, tuple[str, str, int]]:
    lowering_cases = {}
    src = TileSpec((8, 128), f32)
    workspace = TileSpec((8, 128), f32)
    dst = TileSpec((8, 1), f32, b_layout="col_major")
    for op_name, candidate, vmi_op, physical_op in (
        ("trowmax", vmi_trowmax_row, "pto.vmi.vcmax", "pto.vcmax"),
        ("trowsum", vmi_trowsum_row, "pto.vmi.vcadd", "pto.vcadd"),
    ):
        artifact = candidate.specialize(src=src, workspace=workspace, dst=dst)
        artifact.verify()
        text = artifact.mlir_text()
        name = f"vmi_{op_name}_row_streaming"
        expect(candidate.metadata.id == 1001, f"{name} should have a unique id")
        expect(
            candidate.metadata.fusible
            and "row_streaming" in candidate.metadata.tags
            and "single_logical_row_loop" in candidate.metadata.tags,
            f"{name} should advertise its row-loop fusion contract",
        )
        expect(text.count("scf.for") == 1, f"{name} should emit one row loop")
        expect(
            "!pto.vmi.vreg<1024xf32>" not in text
            and "!pto.vmi.vreg<128xf32>" in text,
            f"{name} should materialize one row rather than the full tile",
        )
        expect(text.count(vmi_op) == 1, f"{name} should issue one reduction")
        expect(
            text.count("pto.vmi.vstore") == 1 and "group = 1" in text,
            f"{name} should use an unaligned-safe one-point group store",
        )
        lowering_cases[name] = (text, physical_op, 1)
    return lowering_cases


def check_col_expand_candidate() -> None:
    """ColExpandBinary (tcolexpandsub/add/mul/div) broadcasts a [1, VL] column
    result across every row of a [rows, VL] tile, mirroring pto-isa
    ``TColExpandBinOp`` (reload the same VL block per row, not a 1-lane vbrc).
    """
    col_tile_spec = {
        "kind": "tile",
        "dtype": "f32",
        "shape": [32, 64],
        "valid_shape": [32, 64],
        "memory_space": "ub",
        "config": {
            "b_layout": "row_major",
            "s_layout": "none_box",
            "s_fractal_size": 512,
            "pad_value": "0x0",
        },
    }
    reduced_col_spec = {
        **col_tile_spec,
        "shape": [1, 64],
        "valid_shape": [1, 64],
    }
    binops = {
        "pto.tcolexpandsub": "pto.vmi.vsub",
        "pto.tcolexpandadd": "pto.vmi.vadd",
        "pto.tcolexpandmul": "pto.vmi.vmul",
        "pto.tcolexpanddiv": "pto.vmi.vdiv",
    }
    for op_name, expected_op in binops.items():
        # tcolexpanddiv is the only ColExpandBinary op that ExpandTileOp
        # decorates with a `precisionType` context attr (even at default). Real
        # TileOp -> PTODSL VMI provider selection passes that attr; a candidate
        # that didn't declare it under context_constraints would be rejected by
        # validate_context_attrs. Instantiate with the real attr here so the
        # candidate is exercised through the same path.
        ctx_attrs = (
            {"precisionType": "default"} if op_name == "pto.tcolexpanddiv" else {}
        )
        text = instantiate_candidate(
            target="a5",
            op_name=op_name,
            operand_specs=[col_tile_spec, reduced_col_spec, col_tile_spec],
            provider_module="ptodsl.vmi_tilelib",
            context_attrs=ctx_attrs,
        ).mlir_text()
        expect(text.count("scf.for") == 1, f"{op_name} should emit one runtime row loop")
        expect(expected_op in text, f"{op_name} should lower to {expected_op}")
        expect("pto.vmi.vbrc" not in text, f"{op_name} must reload the VL block, not 1-lane vbrc")
        expect(
            text.count("pto.vmi.vload") == 2,
            f"{op_name} should load one source row plus the broadcast VL block",
        )
        # The broadcast VL block is loop-invariant (col_values is [1, VL]); it
        # must be hoisted out of the row loop so a later mem2reg can forward the
        # ColMax result straight to the consumer without a per-row reload. So
        # exactly one vload precedes scf.for (the broadcast) and one sits inside
        # (the source row).
        for_pos = text.find("scf.for")
        expect(
            for_pos > 0 and text[:for_pos].count("pto.vmi.vload") == 1,
            f"{op_name} should hoist the broadcast vload out of the row loop",
        )
        expect(
            text[for_pos:].count("pto.vmi.vload") == 1,
            f"{op_name} should keep only the source-row vload inside the loop",
        )

def check_tcvt_bf16_candidate() -> None:
    """tcvt covers the static DSv4 conversion forms on one chunk loop."""
    raw_tile_spec = {
        "kind": "tile",
        "dtype": "f32",
        "shape": [32, 128],
        "valid_shape": [32, 128],
        "memory_space": "ub",
        "config": {
            "b_layout": "row_major",
            "s_layout": "none_box",
            "s_fractal_size": 512,
            "pad_value": "0x0",
        },
    }
    f16_dst_spec = {**raw_tile_spec, "dtype": "f16"}
    bf16_dst_spec = {**raw_tile_spec, "dtype": "bf16"}
    f16_text = instantiate_candidate(
        target="a5",
        op_name="pto.tcvt",
        operand_specs=[raw_tile_spec, f16_dst_spec],
        provider_module="ptodsl.vmi_tilelib",
        context_attrs={"round_mode": "RINT", "sat_mode": "OFF"},
    ).mlir_text()
    expect("pto.vmi.vcvt" in f16_text, "tcvt f32->f16 should lower to VMI conversion")
    expect(
        "vreg<64xf16>" in f16_text,
        "tcvt f32->f16 should use native f32-sized chunks",
    )
    expect(
        "vreg<128xf16>" not in f16_text,
        "tcvt f32->f16 should avoid split input rows",
    )

    bf16_text = instantiate_candidate(
        target="a5",
        op_name="pto.tcvt",
        operand_specs=[raw_tile_spec, bf16_dst_spec],
        provider_module="ptodsl.vmi_tilelib",
        context_attrs={"round_mode": "RINT", "sat_mode": "OFF"},
    ).mlir_text()
    expect("pto.vmi.vcvt" in bf16_text, "tcvt f32->bf16 should lower to VMI conversion")
    expect(
        "vreg<64xbf16>" in bf16_text,
        "tcvt f32->bf16 should use native f32-sized chunks",
    )
    expect(
        "vreg<128xbf16>" not in bf16_text,
        "tcvt f32->bf16 should avoid split input rows",
    )

    default_bf16_text = instantiate_candidate(
        target="a5",
        op_name="pto.tcvt",
        operand_specs=[raw_tile_spec, bf16_dst_spec],
        provider_module="ptodsl.vmi_tilelib",
        context_attrs={"round_mode": "RINT", "sat_mode": "DEFAULT"},
    ).mlir_text()
    expect(
        'saturate = "SAT"' in default_bf16_text,
        "omitted f32->bf16 saturation should use the A5 TCVT default",
    )
    expect(
        'saturate = "NOSAT"' in bf16_text,
        "explicit f32->bf16 saturation OFF should remain non-saturating",
    )

    half_vl_src_spec = {
        **raw_tile_spec,
        "shape": [32, 64],
        "valid_shape": [32, 64],
    }
    half_vl_bf16_dst_spec = {**half_vl_src_spec, "dtype": "bf16"}
    half_vl_text = instantiate_candidate(
        target="a5",
        op_name="pto.tcvt",
        operand_specs=[half_vl_src_spec, half_vl_bf16_dst_spec],
        provider_module="ptodsl.vmi_tilelib",
        context_attrs={"round_mode": "RINT", "sat_mode": "OFF"},
    ).mlir_text()
    expect(
        "!pto.vmi.vreg<64xbf16>" in half_vl_text,
        "a 256B f32 row may narrow to a 128B bf16 row",
    )

    below_min_src_spec = {
        **raw_tile_spec,
        "shape": [32, 16],
        "valid_shape": [32, 16],
    }
    below_min_bf16_dst_spec = {**below_min_src_spec, "dtype": "bf16"}
    expect_raises(
        lambda: instantiate_candidate(
            target="a5",
            op_name="pto.tcvt",
            operand_specs=[below_min_src_spec, below_min_bf16_dst_spec],
            provider_module="ptodsl.vmi_tilelib",
            context_attrs={"round_mode": "RINT", "sat_mode": "OFF"},
        ),
        LookupError,
        "custom constraints are not satisfied",
    )

    forms = (
        ("bf16", "f32", "ROUND", 1),
        ("f16", "f32", "ROUND", 1),
        ("i32", "f32", "ROUND", 1),
        ("f32", "i32", "TRUNC", 1),
        ("i32", "f16", "ROUND", 2),
    )
    for src_dtype, dst_dtype, round_mode, vcvt_count in forms:
        src_spec = {**raw_tile_spec, "dtype": src_dtype}
        dst_spec = {**raw_tile_spec, "dtype": dst_dtype}
        text = instantiate_candidate(
            target="a5",
            op_name="pto.tcvt",
            operand_specs=[src_spec, dst_spec],
            provider_module="ptodsl.vmi_tilelib",
            context_attrs={
                "round_mode": round_mode,
                "sat_mode": "ON" if (src_dtype, dst_dtype) == ("f32", "i32") else "OFF",
            },
        ).mlir_text()
        expect(text.count("scf.for") == 1, f"tcvt {src_dtype}->{dst_dtype} needs one row loop")
        expect(
            text.count("pto.vmi.vcvt") == vcvt_count,
            f"tcvt {src_dtype}->{dst_dtype} should emit {vcvt_count} conversion op(s)",
        )
        if (src_dtype, dst_dtype) == ("f32", "i32"):
            expect(
                'rounding = "Z"' in text,
                "f32->i32 TRUNC must lower to the physical toward-zero mode",
            )
        elif dst_dtype == "f32":
            expect(
                "rounding" not in text,
                f"{src_dtype}->f32 widening must not carry VMI rounding",
            )
        elif (src_dtype, dst_dtype) == ("i32", "f16"):
            expect(
                text.count('rounding = "A"') == 1,
                "i32->f16 must apply rounding only to the f32->f16 narrowing step",
            )
        elif round_mode == "TRUNC":
            expect(
                'rounding = "Z"' in text,
                f"tcvt {src_dtype}->{dst_dtype} should preserve truncation",
            )
        expect(
            'saturate = "SAT"' in text
            if (src_dtype, dst_dtype) == ("f32", "i32")
            else True,
            "tcvt f32->i32 should preserve saturation mode",
        )

    f32_to_i32 = {**raw_tile_spec, "dtype": "i32"}
    expect_raises(
        lambda: instantiate_candidate(
            target="a5",
            op_name="pto.tcvt",
            operand_specs=[raw_tile_spec, f32_to_i32],
            provider_module="ptodsl.vmi_tilelib",
            context_attrs={"round_mode": "RINT", "sat_mode": "OFF"},
        ),
        LookupError,
        "custom constraints are not satisfied",
    )


def check_col_reduce_vmi_to_vpto_lowering() -> None:
    """The vreg-carrying ColReduce loop must survive VMI->VPTO lowering as a
    real physical ``scf.for iter_args(%acc = ...) -> !pto.vreg<...>`` (the seed
    loaded once before the loop, one vlds+vmax per iteration), proving the
    pto-isa reduce loop shape reaches the physical layer."""
    col_tile_spec = {
        "kind": "tile",
        "dtype": "f32",
        "shape": [32, 64],
        "valid_shape": [32, 64],
        "memory_space": "ub",
        "config": {
            "b_layout": "row_major",
            "s_layout": "none_box",
            "s_fractal_size": 512,
            "pad_value": "0x0",
        },
    }
    reduced_col_spec = {
        **col_tile_spec,
        "shape": [1, 64],
        "valid_shape": [1, 64],
    }
    colmax = instantiate_candidate(
        target="a5",
        op_name="pto.tcolmax",
        operand_specs=[col_tile_spec, reduced_col_spec],
        provider_module="ptodsl.vmi_tilelib",
        context_attrs={},
    ).mlir_text()
    colsum = instantiate_candidate(
        target="a5",
        op_name="pto.tcolsum",
        operand_specs=[col_tile_spec, reduced_col_spec],
        provider_module="ptodsl.vmi_tilelib",
        context_attrs={},
    ).mlir_text()
    check_vmi_to_vpto_lowering("vmi_tcolmax", colmax, "pto.vmax")
    check_vmi_to_vpto_lowering("vmi_tcolsum", colsum, "pto.vadd")


def check_tmov_nd2nz() -> None:
    """tmov dispatches on dst layout: ND row-major -> elementwise move;
    NZ col-major -> single-VL ND->NZ block-strided vstore loop that lowers to
    pto.vsstb (one row scf.for, constant block_stride, explicit 32-byte row
    offsets, no last-block branch). Mirrors the hand-written softmax ND->NZ
    path's single-layer constant-stride form."""
    nd_tile_spec = {
        "kind": "tile",
        "dtype": "f16",
        "shape": [16, 128],
        "valid_shape": [16, 128],
        "memory_space": "ub",
        "config": {
            "b_layout": "row_major",
            "s_layout": "none_box",
            "s_fractal_size": 512,
            "pad_value": "0x0",
        },
    }
    nz_tile_spec = {
        **nd_tile_spec,
        "config": {
            "b_layout": "col_major",
            "s_layout": "row_major",
            "s_fractal_size": 512,
            "pad_value": "0x0",
        },
    }

    # ND -> NZ: helper must accept the NZ dst config (s_layout=row_major) and
    # the candidate must render one row loop with block-strided vmi.vstore.
    nz_text = instantiate_candidate(
        target="a5",
        op_name="pto.tmov",
        operand_specs=[nd_tile_spec, nz_tile_spec],
        provider_module="ptodsl.vmi_tilelib",
        context_attrs={},
    ).mlir_text()
    expect("pto.vmi.vload" in nz_text, "tmov ND->NZ should load via pto.vmi.vload")
    expect("pto.vmi.vstore" in nz_text, "tmov ND->NZ should store via pto.vmi.vstore")
    expect(nz_text.count("scf.for") == 1, "tmov ND->NZ should render one row loop")
    expect("blayout=col_major" in nz_text, "tmov ND->NZ dst should be a col-major NZ tile")
    # constant block stride, no second (tail-block) loop; VMI vstore no longer
    # carries a repeat_stride operand (upstream dropped it). The explicit
    # destination offset advances one 32-byte block, or 16 f16 elements, per
    # row instead of relying on a post-update result.
    expect("arith.constant 16 : i16" in nz_text, "tmov ND->NZ block_stride should be a constant 16")
    expect("arith.constant 16 : index" in nz_text, "tmov ND->NZ should advance one f16 block per row")
    expect(
        all(" -> !pto.ptr" not in line for line in nz_text.splitlines() if "pto.vmi.vstore" in line),
        "tmov ND->NZ should not use a post-update destination",
    )

    # Lower to VPTO and confirm it reaches a single pto.vsstb with constant
    # block_stride/repeat_stride inside one scf.for (the pto-isa single-VL form).
    nz_lowered = check_vmi_to_vpto_lowering("vmi_tmov_nd2nz", nz_text, "pto.vsstb")
    expect("pto.addptr" in nz_lowered, "tmov ND->NZ should lower its explicit row offset")

    # 1/2-VL case (bf16 cols=64 < lanes=128): the partial tail is handled by a
    # count predicate (pto-isa CreatePredicate(count)); lowering should reach a
    # full-VL vlds + a half-VL mask (PAT_VL64) vsstb, still one row loop, still
    # constant strides. Mirrors fa_dn_softmax [128,64] bf16 x_exp_buf -> NZ.
    half_nd_spec = {
        "kind": "tile",
        "dtype": "bf16",
        "shape": [128, 64],
        "valid_shape": [128, 64],
        "memory_space": "ub",
        "config": {
            "b_layout": "row_major",
            "s_layout": "none_box",
            "s_fractal_size": 512,
            "pad_value": "0x0",
        },
    }
    half_nz_spec = {
        **half_nd_spec,
        "config": {
            "b_layout": "col_major",
            "s_layout": "row_major",
            "s_fractal_size": 512,
            "pad_value": "0x0",
        },
    }
    half_text = instantiate_candidate(
        target="a5",
        op_name="pto.tmov",
        operand_specs=[half_nd_spec, half_nz_spec],
        provider_module="ptodsl.vmi_tilelib",
        context_attrs={},
    ).mlir_text()
    expect(half_text.count("scf.for") == 1, "tmov 1/2-VL ND->NZ should render one row loop")
    expect(
        "arith.constant 128 : i16" in half_text,
        "tmov 1/2-VL block_stride should still be a constant 128",
    )
    expect(
        "arith.constant 16 : index" in half_text,
        "tmov 1/2-VL should advance one bf16 block per row",
    )
    expect(
        all(" -> !pto.ptr" not in line for line in half_text.splitlines() if "pto.vmi.vstore" in line),
        "tmov 1/2-VL should not use a post-update destination",
    )
    # Lower to VPTO and confirm the half-VL mask form (full-VL vlds + half-VL
    # block-mask vsstb) reaches pto.vsstb inside one scf.for.
    check_vmi_to_vpto_lowering("vmi_tmov_nd2nz_halfvl", half_text, "pto.vsstb")

    # RowPlusOne keeps 129 physical rows for block_stride while iterating only
    # 128 valid rows. This is the runtime regression shape used by
    # fa-softmax-dn-init-rowplusone.
    rowplusone_nz_spec = {
        **half_nd_spec,
        "shape": [129, 64],
        "valid_shape": [128, 64],
        "config": half_nz_spec["config"],
    }
    rowplusone_text = instantiate_candidate(
        target="a5",
        op_name="pto.tmov",
        operand_specs=[half_nd_spec, rowplusone_nz_spec],
        provider_module="ptodsl.vmi_tilelib",
        context_attrs={},
    ).mlir_text()
    expect(
        "arith.constant 129 : i16" in rowplusone_text,
        "tmov RowPlusOne block_stride should preserve 129 physical rows",
    )
    expect(
        "arith.constant 16 : index" in rowplusone_text,
        "tmov RowPlusOne should advance one bf16 block per valid row",
    )
    expect(
        all(
            " -> !pto.ptr" not in line
            for line in rowplusone_text.splitlines()
            if "pto.vmi.vstore" in line
        ),
        "tmov RowPlusOne should not use a post-update destination",
    )

    # Regression: ND -> ND still selects the elementwise move path (no
    # block-strided store), so the dispatch did not break the plain-move case.
    # The elementwise VMI candidate is f32 / 64-lane, so use an f32 tile here.
    nd_f32_spec = {
        "kind": "tile",
        "dtype": "f32",
        "shape": [16, 64],
        "valid_shape": [16, 64],
        "memory_space": "ub",
        "config": {
            "b_layout": "row_major",
            "s_layout": "none_box",
            "s_fractal_size": 512,
            "pad_value": "0x0",
        },
    }
    nd_text = instantiate_candidate(
        target="a5",
        op_name="pto.tmov",
        operand_specs=[nd_f32_spec, nd_f32_spec],
        provider_module="ptodsl.vmi_tilelib",
        context_attrs={},
    ).mlir_text()
    expect("pto.vmi.vstore" in nd_text, "tmov ND->ND should still emit a vmi store")
    expect("arith.constant 1 : i16" not in nd_text, "tmov ND->ND must not emit block-stride operands")

    # Regression: ordinary ND -> ND moves must honor the dtype set declared by
    # the VMI candidate. DSv4 uses bf16 moves in this form; the helper used to
    # reject them because emit_elementwise_vmi defaults to f32.
    nd_bf16_spec = {
        **half_nd_spec,
        "shape": [128, 128],
        "valid_shape": [128, 128],
        "config": {
            "b_layout": "row_major",
            "s_layout": "none_box",
            "s_fractal_size": 512,
            "pad_value": "0x0",
        },
    }
    nd_bf16_text = instantiate_candidate(
        target="a5",
        op_name="pto.tmov",
        operand_specs=[nd_bf16_spec, nd_bf16_spec],
        provider_module="ptodsl.vmi_tilelib",
        context_attrs={},
    ).mlir_text()
    expect(
        "pto.vmi.vload" in nd_bf16_text and "pto.vmi.vstore" in nd_bf16_text,
        "tmov bf16 ND->ND should instantiate the VMI elementwise move",
    )
    expect(
        "!pto.vmi.vreg<128xbf16>" in nd_bf16_text,
        "tmov bf16 ND->ND should use a full physical bf16 row",
    )

    expect_raises(
        lambda: instantiate_candidate(
            target="a5",
            op_name="pto.tmov",
            operand_specs=[half_nd_spec, half_nd_spec],
            provider_module="ptodsl.vmi_tilelib",
            context_attrs={},
        ),
        LookupError,
        "custom constraints are not satisfied",
    )


def check_legacy_vpto_compatibility() -> None:
    spec = TileSpec(TILE_SHAPE, f32)
    artifact = legacy_vpto_tadd.specialize(src0=spec, src1=spec, dst=spec)
    artifact.verify()
    text = artifact.mlir_text()
    expect(text.count("scf.for") == 2, "legacy VPTO template should retain its two loops")
    expect("pto.vlds" in text, "legacy VPTO template should still emit vlds")
    expect("pto.vadd" in text, "legacy VPTO template should still emit vadd")
    expect("pto.vsts" in text, "legacy VPTO template should still emit vsts")


def _wrap_template_with_alloc_driver(mlir_text: str) -> str:
    """Add a driver that allocates the template's tile args and inlines it.

    The standalone template render uses tile_buf function arguments, which
    FoldTileBufIntrinsics cannot bridge on the VPTO path (it requires
    alloc_tile/treshape-defined handles). Rewrite the render into one driver
    function that allocates each tile argument locally and runs the template
    body with those handles — the shape ExpandTileOp produces after inlining
    the helper into the caller.
    """
    import re as _re

    matches = list(_re.finditer(
        r"func\.func @([\w.]+)\(([^)]*)\)", mlir_text
    ))
    expect(len(matches) == 1, "template render should contain exactly one function")
    func_name = matches[0].group(1)
    params_text = matches[0].group(2)

    def _split_params(text: str) -> list[str]:
        """Split on top-level commas (tile types contain ``<..., ...>``)."""
        parts = []
        depth = 0
        current = []
        for ch in text:
            if ch == "<":
                depth += 1
            elif ch == ">":
                depth -= 1
            if ch == "," and depth == 0:
                parts.append("".join(current).strip())
                current = []
            else:
                current.append(ch)
        tail = "".join(current).strip()
        if tail:
            parts.append(tail)
        return [part for part in parts if part]

    param_names = []
    tile_args = []
    scalar_args = []
    for param in _split_params(params_text):
        match = _re.match(r"%(\w+):\s*((?:!pto\.tile_buf)<[^)]+>)", param)
        if match:
            param_names.append(match.group(1))
            tile_args.append((match.group(1), match.group(2)))
            continue
        match = _re.match(r"%(\w+):\s*(\w+)", param)
        if match:
            param_names.append(match.group(1))
            scalar_args.append((match.group(1), match.group(2)))

    alloc_lines = []
    renames = {}
    addr_consts = []
    for index, (arg_name, tile_type) in enumerate(tile_args):
        # VPTO helpers require alloc_tile to carry an explicit addr operand
        # (PlanMemory normally assigns it); give each tile a distinct base.
        addr_consts.append(
            f"    %addr{index} = arith.constant {index * 4096} : i64"
        )
        alloc_lines.append(
            f"    %tile{index} = pto.alloc_tile addr = %addr{index} : {tile_type}"
        )
        renames[arg_name] = f"%tile{index}"
    for index, (arg_name, scalar_type) in enumerate(scalar_args):
        value = "1.0" if scalar_type in ("f32", "f16", "bf16") else "1"
        # The body already materializes locals for scalar args; bind the SSA
        # name to a printed constant so remaining uses resolve.
        alloc_lines.append(
            f"    %{arg_name} = arith.constant {value} : {scalar_type}"
        )
        renames[arg_name] = f"%{arg_name}"

    # Extract the template function body (from its `{` to the matching `}`)
    # and substitute the argument SSA names with the driver locals.
    body_start = mlir_text.find("{", matches[0].start())
    depth = 0
    body_end = None
    for index in range(body_start, len(mlir_text)):
        if mlir_text[index] == "{":
            depth += 1
        elif mlir_text[index] == "}":
            depth -= 1
            if depth == 0:
                body_end = index
                break
    expect(body_end is not None, "template render should have a balanced body")
    body = mlir_text[body_start + 1 : body_end]
    for arg_name in param_names:
        body = body.replace(f"%{arg_name}", renames[arg_name])
    body_indented = "\n".join(
        f"    {line}" if line.strip() else line for line in body.splitlines()
    )

    driver = f"""module attributes {{pto.target_arch = "a5"}} {{
  module attributes {{pto.backend = "vpto", pto.kernel_kind = #pto.kernel_kind<vector>, pto.target_arch = "a5"}} {{
    func.func @{func_name}() {{
{chr(10).join(addr_consts)}
{chr(10).join(alloc_lines)}
{body_indented}
    }}
  }}
}}
"""
    return driver


def check_vmi_to_vpto_lowering(
    name: str,
    mlir_text: str,
    expected_op: str,
    expected_loop_count: int = 1,
) -> str:
    ptoas = shutil.which("ptoas")
    expect(ptoas is not None, "ptoas must be available for VMI-to-VPTO regression coverage")
    # The rendered text is a template function whose tile arguments are not
    # materialized tile handles. FoldTileBufIntrinsics requires every tile_buf
    # used by tile_buf_addr to come from alloc_tile/treshape, so wrap the
    # template in a driver that allocates the operand tiles and calls it —
    # the same usage pattern ExpandTileOp generates in the real pipeline.
    driver = _wrap_template_with_alloc_driver(mlir_text)
    with TemporaryDirectory() as temp_dir:
        input_path = Path(temp_dir) / f"{name}.pto"
        input_path.write_text(driver, encoding="utf-8")
        completed = subprocess.run(
            [
                ptoas,
                "--pto-arch=a5",
                "--pto-backend=vpto",
                "--pto-level=level3",
                "--enable-vmi",
                "--emit-vpto",
                str(input_path),
                "-o",
                "-",
            ],
            check=False,
            capture_output=True,
            text=True,
        )
    expect(
        completed.returncode == 0,
        f"VMI-to-VPTO lowering failed for {name}:\n{completed.stderr}",
    )
    expect("pto.vmi." not in completed.stdout, f"{name} should contain no VMI ops after lowering")
    expect(expected_op in completed.stdout, f"{name} should lower to {expected_op}")
    expect(
        completed.stdout.count("scf.for") == expected_loop_count,
        f"{name} should lower to {expected_loop_count} principal loop(s)",
    )
    return completed.stdout


def main() -> None:
    check_canonical_block_map()
    check_legacy_vpto_compatibility()
    check_provider_helper()
    tadd_text, wide_tadd_text, texp_text = check_candidate_ir()
    check_vmi_to_vpto_lowering("vmi_tadd_block64", tadd_text, "pto.vadd")
    check_vmi_to_vpto_lowering("vmi_tadd_block128", wide_tadd_text, "pto.vadd")
    check_vmi_to_vpto_lowering("vmi_texp_block64", texp_text, "pto.vexp")
    for name, (text, expected_op) in check_local_elementwise_candidates().items():
        check_vmi_to_vpto_lowering(name, text, expected_op)
    for name, (text, expected_op) in check_rope_128b_candidates().items():
        check_vmi_to_vpto_lowering(name, text, expected_op)
    for name, (text, expected_op) in check_rmsnorm_256b_row_candidates().items():
        check_vmi_to_vpto_lowering(name, text, expected_op)
    for name, (text, expected_op) in check_local_broadcast_candidates().items():
        lowered = check_vmi_to_vpto_lowering(name, text, expected_op)
        if name.startswith("vmi_trowexpand"):
            expect(
                'pto.vlds' in lowered and 'dist = "BRC_B32"' in lowered,
                f"{name} should lower compact state access to native broadcast load",
            )
    for name, (text, expected_op, expected_loop_count) in (
        check_row_reduce_candidates().items()
    ):
        lowered = check_vmi_to_vpto_lowering(
            name, text, expected_op, expected_loop_count
        )
        expected_store = "pto.vscatter" if name.endswith("_128lanes") else "pto.vsts"
        expect(
            expected_store in lowered,
            f"{name} should lower row results with {expected_store}",
        )
    for name, (text, expected_op, expected_loop_count) in (
        check_row_streaming_reduce_candidates().items()
    ):
        check_vmi_to_vpto_lowering(name, text, expected_op, expected_loop_count)
    check_col_reduce_candidate()
    check_col_reduce_split()
    check_col_expand_candidate()
    check_tcvt_bf16_candidate()
    check_col_reduce_vmi_to_vpto_lowering()
    check_tmov_nd2nz()
    print("ptodsl_vmi_tile_template: PASS")


if __name__ == "__main__":
    main()
