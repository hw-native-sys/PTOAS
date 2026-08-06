# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""PTODSL TileLib templates for A5 ``pto.tcvt`` 1D/2D paths.

Conversion-specific instruction sequences stay in this module. Flattened
forms use typed source and destination pointers so dtype width changes retain
their existing logical-element addressing and distribution modes.
"""

from dataclasses import replace

from ptodsl import pto
from ptodsl._ast_rewrite import rewrite_jit_function
from ptodsl._surface_values import unwrap_surface_value, wrap_surface_value
import ptodsl.tilelib as tilelib
from ptoas.mlir.dialects import pto as _pto

from ._elementwise import (
    FALLBACK_TRAVERSAL_PRIORITY,
    PREFERRED_TRAVERSAL_PRIORITY,
)


_PENDING_TCVT_1D = []


def _defer_tcvt_1d(candidate):
    _PENDING_TCVT_1D.append(candidate)
    return candidate


def _rowwise(src_shape, src_valid_shape, dst_shape, dst_valid_shape, src_config, dst_config, **_):
    return (
        tuple(src_shape) == tuple(dst_shape)
        and tuple(src_valid_shape) == tuple(dst_valid_shape)
        and src_config.b_layout == "row_major"
        and dst_config.b_layout == "row_major"
        and src_config.s_layout == "none_box"
        and dst_config.s_layout == "none_box"
    )


def _rowwise_bf16_to_fp4(src_shape, src_valid_shape, dst_shape, dst_valid_shape, src_config, dst_config, **_):
    return (
        len(src_shape) == 2
        and len(dst_shape) == 2
        and src_shape[0] == dst_shape[0]
        and src_shape[1] == dst_shape[1] * 2
        and src_valid_shape[0] == dst_valid_shape[0]
        and src_valid_shape[1] == dst_valid_shape[1] * 2
        and src_config.b_layout == "row_major"
        and dst_config.b_layout == "row_major"
        and src_config.s_layout == "none_box"
        and dst_config.s_layout == "none_box"
    )


def _round_mode():
    round_mode = pto.get_op_attr("round_mode", "RINT")
    if round_mode == "ROUND":
        return pto.VcvtRoundMode.A
    if round_mode == "FLOOR":
        return pto.VcvtRoundMode.F
    if round_mode == "CEIL":
        return pto.VcvtRoundMode.C
    if round_mode == "TRUNC":
        return pto.VcvtRoundMode.Z
    if round_mode == "ODD":
        return pto.VcvtRoundMode.O
    return pto.VcvtRoundMode.R


def _sat_mode(token):
    if token == "nosat":
        return pto.VcvtSatMode.NOSAT
    if token == "sat":
        return pto.VcvtSatMode.SAT
    return None


def _part_mode(token):
    if token == "even":
        return pto.VcvtPartMode.EVEN
    if token == "p0":
        return pto.VcvtPartMode.P0
    return None


def _vselr_low_precision(src, idx):
    raw_src = unwrap_surface_value(src)
    return wrap_surface_value(_pto.VselrOp(raw_src.type, raw_src, unwrap_surface_value(idx)).result)


def _tcvt_conversion_mask(src, mask, mode):
    if mode == "src_full":
        return pto.make_mask(src.dtype, pto.PAT.ALL)
    return mask


def _tcvt_load_2d(src, row, col, dist):
    if dist:
        return pto.vlds(src[row, col:], dist=dist)
    return pto.vlds(src[row, col:])


def _tcvt_convert(vec, dtype, mask, *, rnd, sat, part):
    kwargs = {}
    if rnd:
        kwargs["rnd"] = _round_mode()
    sat_mode = _sat_mode(sat)
    if sat_mode is not None:
        kwargs["sat"] = sat_mode
    part_mode = _part_mode(part)
    if part_mode is not None:
        kwargs["part"] = part_mode
    return pto.vcvt(vec, dtype, mask, **kwargs)


def _tcvt_store_2d(converted, dst, row, col, mask, dist):
    if dist:
        pto.vsts(converted, dst[row, col:], mask, dist=dist)
    else:
        pto.vsts(converted, dst[row, col:], mask)


@rewrite_jit_function
def _emit_tcvt_1d(dst, step_dtype, mask_dtype, remaining_scale, emit_chunk):
    """Run one conversion vector loop over the flattened destination range."""

    valid_rows, valid_cols = dst.valid_shape
    total_elements = valid_rows * valid_cols
    lanes = pto.elements_per_vreg(step_dtype)
    remained = total_elements * remaining_scale
    for offset in range(0, total_elements, lanes):
        mask, remained = pto.make_mask(mask_dtype, remained)
        emit_chunk(offset, mask)


@rewrite_jit_function
def _render_tcvt(
    src,
    dst,
    *,
    rnd=False,
    sat=None,
    part=None,
    load_dist=None,
    store_dist=None,
    mask_dtype="dst",
    convert_mask="store",
):
    valid_rows, valid_cols = dst.valid_shape
    dtype = dst.dtype
    loop_dtype = src.dtype if mask_dtype == "src" else dtype
    lanes = pto.elements_per_vreg(loop_dtype)
    for row in range(0, valid_rows, 1):
        remained = valid_cols
        for col in range(0, valid_cols, lanes):
            mask, remained = pto.make_mask(loop_dtype, remained)
            convert_mask_value = _tcvt_conversion_mask(
                src,
                mask,
                convert_mask,
            )
            vec = _tcvt_load_2d(src, row, col, load_dist)
            converted = _tcvt_convert(
                vec,
                dtype,
                convert_mask_value,
                rnd=rnd,
                sat=sat,
                part=part,
            )
            _tcvt_store_2d(
                converted,
                dst,
                row,
                col,
                mask,
                store_dist,
            )


def _render_tcvt_1d(
    src,
    dst,
    *,
    rnd=False,
    sat=None,
    part=None,
    load_dist=None,
    store_dist=None,
    mask_dtype="dst",
    convert_mask="store",
):
    dtype = dst.dtype
    loop_dtype = src.dtype if mask_dtype == "src" else dtype
    src_ptr = src.as_ptr()
    dst_ptr = dst.as_ptr()

    def emit_chunk(offset, mask):
        convert_mask_value = mask
        if convert_mask == "src_full":
            convert_mask_value = pto.make_mask(src.dtype, pto.PAT.ALL)
        vec = (
            pto.vlds(src_ptr, offset, dist=load_dist)
            if load_dist
            else pto.vlds(src_ptr, offset)
        )
        kwargs = {}
        if rnd:
            kwargs["rnd"] = _round_mode()
        sat_mode = _sat_mode(sat)
        if sat_mode is not None:
            kwargs["sat"] = sat_mode
        part_mode = _part_mode(part)
        if part_mode is not None:
            kwargs["part"] = part_mode
        converted = pto.vcvt(vec, dtype, convert_mask_value, **kwargs)
        if store_dist:
            pto.vsts(converted, dst_ptr, offset, mask, dist=store_dist)
        else:
            pto.vsts(converted, dst_ptr, offset, mask)

    _emit_tcvt_1d(dst, loop_dtype, loop_dtype, 1, emit_chunk)


def _register_tcvt(
    *,
    name,
    dtypes,
    idx,
    rnd,
    sat=None,
    part=None,
    load_dist=None,
    store_dist=None,
    mask_dtype="dst",
    convert_mask="store",
):
    def register_form(
        *, traversal, candidate_name, candidate_id, priority, register=True
    ):
        constraints = [_rowwise]
        if traversal == "1d":
            constraints.append(tilelib.require_conversion_1d())

        @tilelib.tile_template(
            op="pto.tcvt",
            target="a5",
            name=candidate_name,
            dtypes=[dtypes],
            iteration_axis="none",
            op_engine="vector",
            op_class="other",
            constraints=constraints,
            priority=priority,
            id=candidate_id,
            loop_depth=1 if traversal == "1d" else 2,
            is_post_update=False,
            tags=("convert", traversal),
            register=register,
        )
        def template(src: pto.Tile, dst: pto.Tile):
            renderer = _render_tcvt_1d if traversal == "1d" else _render_tcvt
            renderer(
                src,
                dst,
                rnd=rnd,
                sat=sat,
                part=part,
                load_dist=load_dist,
                store_dist=store_dist,
                mask_dtype=mask_dtype,
                convert_mask=convert_mask,
            )

        return template

    fallback = register_form(
        traversal="2d",
        candidate_name=name,
        candidate_id=idx,
        priority=FALLBACK_TRAVERSAL_PRIORITY,
    )
    _defer_tcvt_1d(
        register_form(
            traversal="1d",
            candidate_name=f"{name}_1d",
            candidate_id=None,
            priority=PREFERRED_TRAVERSAL_PRIORITY,
            register=False,
        )
    )
    return fallback


def _register_tcvt_1d(
    *,
    name,
    dtypes,
    renderer,
    source_elements_per_destination=1,
    tags=(),
):
    """Register a preferred flattened form for a bespoke conversion body."""

    dtype_signatures = (
        [dtypes]
        if dtypes and isinstance(dtypes[0], str)
        else list(dtypes)
    )
    shape_constraint = (
        _rowwise_bf16_to_fp4
        if source_elements_per_destination == 2
        else _rowwise
    )

    @tilelib.tile_template(
        op="pto.tcvt",
        target="a5",
        name=f"{name}_1d",
        dtypes=dtype_signatures,
        iteration_axis="none",
        op_engine="vector",
        op_class="other",
        constraints=[
            shape_constraint,
            tilelib.require_conversion_1d(
                source_elements_per_destination=(
                    source_elements_per_destination
                ),
            ),
        ],
        priority=PREFERRED_TRAVERSAL_PRIORITY,
        id=None,
        loop_depth=1,
        is_post_update=False,
        tags=("convert", "1d", *tags),
        register=False,
    )
    def template(src: pto.Tile, dst: pto.Tile):
        renderer(src, dst)

    return _defer_tcvt_1d(template)


template_tcvt_f32_to_i32 = _register_tcvt(
    name="template_tcvt_f32_to_i32",
    dtypes=("f32", "i32"),
    idx=0,
    rnd=True,
    sat="sat",
)

template_tcvt_i32_to_f32 = _register_tcvt(
    name="template_tcvt_i32_to_f32",
    dtypes=("i32", "f32"),
    idx=1,
    rnd=True,
)

template_tcvt_i16_to_f16 = _register_tcvt(
    name="template_tcvt_i16_to_f16",
    dtypes=("i16", "f16"),
    idx=2,
    rnd=True,
)

@tilelib.tile_template(
    op="pto.tcvt",
    target="a5",
    name="template_tcvt_f16_to_i16",
    dtypes=[("f16", "i16")],
    iteration_axis="none",
    op_engine="vector",
    op_class="other",
    constraints=[_rowwise],
    id=3,
    loop_depth=2,
    is_post_update=False,
    tags=("convert", "rowwise"),
)
def template_tcvt_f16_to_i16(src: pto.Tile, dst: pto.Tile):
    valid_rows, valid_cols = dst.valid_shape
    lanes_f32 = pto.elements_per_vreg(pto.f32)
    full_mask_b16 = pto.make_mask(src.dtype, pto.PAT.ALL)
    full_mask_b32 = pto.make_mask(pto.i32, pto.PAT.ALL)
    for row in range(0, valid_rows, 1):
        remained = valid_cols
        for col in range(0, valid_cols, lanes_f32):
            store_mask, remained = pto.make_mask(pto.i32, remained)
            vec_f16 = pto.vlds(src[row, col:], dist="UNPK_B16")
            vec_i32 = pto.vcvt(
                vec_f16,
                pto.i32,
                full_mask_b16,
                rnd=_round_mode(),
                part=pto.VcvtPartMode.EVEN,
            )
            vec_i16 = pto.vcvt(
                vec_i32,
                pto.i16,
                full_mask_b32,
                sat=pto.VcvtSatMode.NOSAT,
                part=pto.VcvtPartMode.EVEN,
            )
            pto.vsts(vec_i16, dst[row, col:], store_mask, dist=pto.VStoreDist.PK_B32)


def _render_tcvt_f16_to_i16_1d(src: pto.Tile, dst: pto.Tile):
    src_ptr = src.as_ptr()
    dst_ptr = dst.as_ptr()
    full_mask_b16 = pto.make_mask(src.dtype, pto.PAT.ALL)
    full_mask_b32 = pto.make_mask(pto.i32, pto.PAT.ALL)

    def emit_chunk(offset, store_mask):
        vec_f16 = pto.vlds(src_ptr, offset, dist="UNPK_B16")
        vec_i32 = pto.vcvt(
            vec_f16,
            pto.i32,
            full_mask_b16,
            rnd=_round_mode(),
            part=pto.VcvtPartMode.EVEN,
        )
        vec_i16 = pto.vcvt(
            vec_i32,
            pto.i16,
            full_mask_b32,
            sat=pto.VcvtSatMode.NOSAT,
            part=pto.VcvtPartMode.EVEN,
        )
        pto.vsts(
            vec_i16,
            dst_ptr,
            offset,
            store_mask,
            dist=pto.VStoreDist.PK_B32,
        )

    _emit_tcvt_1d(dst, pto.f32, pto.i32, 1, emit_chunk)


template_tcvt_f16_to_i16_1d = _register_tcvt_1d(
    name="template_tcvt_f16_to_i16",
    dtypes=("f16", "i16"),
    renderer=_render_tcvt_f16_to_i16_1d,
)

template_tcvt_bf16_to_f16 = _register_tcvt(
    name="template_tcvt_bf16_to_f16",
    dtypes=("bf16", "f16"),
    idx=4,
    rnd=True,
    sat="sat",
)

template_tcvt_f32_to_f16 = _register_tcvt(
    name="template_tcvt_f32_to_f16",
    dtypes=("f32", "f16"),
    idx=5,
    rnd=True,
    sat="sat",
    part="even",
    store_dist=pto.VStoreDist.PK_B32,
    mask_dtype="src",
)

template_tcvt_f32_to_bf16 = _register_tcvt(
    name="template_tcvt_f32_to_bf16",
    dtypes=("f32", "bf16"),
    idx=6,
    rnd=True,
    sat="sat",
    part="even",
    store_dist=pto.VStoreDist.PK_B32,
    mask_dtype="src",
)


@tilelib.tile_template(
    op="pto.tcvt",
    target="a5",
    name="template_tcvt_f32_to_f32",
    dtypes=[("f32", "f32")],
    iteration_axis="none",
    op_engine="vector",
    op_class="other",
    constraints=[_rowwise],
    id=17,
    loop_depth=2,
    is_post_update=False,
    tags=("convert", "rowwise"),
)
def template_tcvt_f32_to_f32(src: pto.Tile, dst: pto.Tile):
    valid_rows, valid_cols = dst.valid_shape
    lanes_f32 = pto.elements_per_vreg(src.dtype)
    for row in range(0, valid_rows, 1):
        remained = valid_cols
        for col in range(0, valid_cols, lanes_f32):
            mask, remained = pto.make_mask(src.dtype, remained)
            vec = pto.vlds(src[row, col:])
            converted = pto.vtrc(vec, mask, rnd=_round_mode())
            pto.vsts(converted, dst[row, col:], mask)


def _render_tcvt_f32_to_f32_1d(src: pto.Tile, dst: pto.Tile):
    src_ptr = src.as_ptr()
    dst_ptr = dst.as_ptr()

    def emit_chunk(offset, mask):
        vec = pto.vlds(src_ptr, offset)
        converted = pto.vtrc(vec, mask, rnd=_round_mode())
        pto.vsts(converted, dst_ptr, offset, mask)

    _emit_tcvt_1d(dst, src.dtype, src.dtype, 1, emit_chunk)


template_tcvt_f32_to_f32_1d = _register_tcvt_1d(
    name="template_tcvt_f32_to_f32",
    dtypes=("f32", "f32"),
    renderer=_render_tcvt_f32_to_f32_1d,
)


@tilelib.tile_template(
    op="pto.tcvt",
    target="a5",
    name="template_tcvt_f32_to_i16",
    dtypes=[("f32", "i16")],
    iteration_axis="none",
    op_engine="vector",
    op_class="other",
    constraints=[_rowwise],
    id=15,
    loop_depth=2,
    is_post_update=False,
    tags=("convert", "rowwise"),
)
def template_tcvt_f32_to_i16(src: pto.Tile, dst: pto.Tile):
    valid_rows, valid_cols = dst.valid_shape
    lanes_f32 = pto.elements_per_vreg(src.dtype)
    full_mask = pto.make_mask(src.dtype, pto.PAT.ALL)
    for row in range(0, valid_rows, 1):
        remained = valid_cols
        for col in range(0, valid_cols, lanes_f32):
            store_mask, remained = pto.make_mask(src.dtype, remained)
            vec_f32 = pto.vlds(src[row, col:])
            vec_i32 = pto.vcvt(
                vec_f32,
                pto.i32,
                full_mask,
                rnd=_round_mode(),
                sat=pto.VcvtSatMode.NOSAT,
            )
            vec_i16 = pto.vcvt(
                vec_i32,
                pto.i16,
                full_mask,
                sat=pto.VcvtSatMode.NOSAT,
                part=pto.VcvtPartMode.EVEN,
            )
            pto.vsts(vec_i16, dst[row, col:], store_mask, dist=pto.VStoreDist.PK_B32)


def _render_tcvt_f32_to_i16_1d(src: pto.Tile, dst: pto.Tile):
    src_ptr = src.as_ptr()
    dst_ptr = dst.as_ptr()
    full_mask = pto.make_mask(src.dtype, pto.PAT.ALL)

    def emit_chunk(offset, store_mask):
        vec_f32 = pto.vlds(src_ptr, offset)
        vec_i32 = pto.vcvt(
            vec_f32,
            pto.i32,
            full_mask,
            rnd=_round_mode(),
            sat=pto.VcvtSatMode.NOSAT,
        )
        vec_i16 = pto.vcvt(
            vec_i32,
            pto.i16,
            full_mask,
            sat=pto.VcvtSatMode.NOSAT,
            part=pto.VcvtPartMode.EVEN,
        )
        pto.vsts(
            vec_i16,
            dst_ptr,
            offset,
            store_mask,
            dist=pto.VStoreDist.PK_B32,
        )

    _emit_tcvt_1d(dst, src.dtype, src.dtype, 1, emit_chunk)


template_tcvt_f32_to_i16_1d = _register_tcvt_1d(
    name="template_tcvt_f32_to_i16",
    dtypes=("f32", "i16"),
    renderer=_render_tcvt_f32_to_i16_1d,
)


@tilelib.tile_template(
    op="pto.tcvt",
    target="a5",
    name="template_tcvt_f32_to_i64",
    dtypes=[("f32", "i64")],
    iteration_axis="none",
    op_engine="vector",
    op_class="other",
    constraints=[_rowwise],
    id=16,
    loop_depth=2,
    is_post_update=False,
    tags=("convert", "rowwise"),
)
def template_tcvt_f32_to_i64(src: pto.Tile, dst: pto.Tile):
    valid_rows, valid_cols = dst.valid_shape
    lanes_i64 = pto.elements_per_vreg(dst.dtype)
    full_mask = pto.make_mask(src.dtype, pto.PAT.ALL)
    for row in range(0, valid_rows, 1):
        remained = valid_cols * 2
        for col in range(0, valid_cols, lanes_i64):
            store_mask, remained = pto.make_mask(pto.i32, remained)
            vec = pto.vlds(src[row, col:], dist="UNPK_B32")
            converted = pto.vcvt(
                vec,
                pto.i64,
                full_mask,
                rnd=_round_mode(),
                sat=pto.VcvtSatMode.SAT,
                part=pto.VcvtPartMode.EVEN,
            )
            pto.vsts(converted, dst[row, col:], store_mask, dist=pto.VStoreDist.NORM_B32)


def _render_tcvt_f32_to_i64_1d(src: pto.Tile, dst: pto.Tile):
    src_ptr = src.as_ptr()
    dst_ptr = dst.as_ptr()
    full_mask = pto.make_mask(src.dtype, pto.PAT.ALL)

    def emit_chunk(offset, store_mask):
        vec = pto.vlds(src_ptr, offset, dist="UNPK_B32")
        converted = pto.vcvt(
            vec,
            pto.i64,
            full_mask,
            rnd=_round_mode(),
            sat=pto.VcvtSatMode.SAT,
            part=pto.VcvtPartMode.EVEN,
        )
        pto.vsts(
            converted,
            dst_ptr,
            offset,
            store_mask,
            dist=pto.VStoreDist.NORM_B32,
        )

    _emit_tcvt_1d(dst, dst.dtype, pto.i32, 2, emit_chunk)


template_tcvt_f32_to_i64_1d = _register_tcvt_1d(
    name="template_tcvt_f32_to_i64",
    dtypes=("f32", "i64"),
    renderer=_render_tcvt_f32_to_i64_1d,
)

template_tcvt_f16_to_i32 = _register_tcvt(
    name="template_tcvt_f16_to_i32",
    dtypes=("f16", "i32"),
    idx=7,
    rnd=True,
    part="even",
    load_dist="UNPK_B16",
    convert_mask="src_full",
)

template_tcvt_f16_to_f32 = _register_tcvt(
    name="template_tcvt_f16_to_f32",
    dtypes=("f16", "f32"),
    idx=8,
    rnd=False,
    part="even",
    load_dist="UNPK_B16",
    convert_mask="src_full",
)


@tilelib.tile_template(
    op="pto.tcvt",
    target="a5",
    name="template_tcvt_f16_to_ui8",
    dtypes=[("f16", "ui8")],
    iteration_axis="none",
    op_engine="vector",
    op_class="other",
    constraints=[_rowwise],
    id=18,
    loop_depth=2,
    is_post_update=False,
    tags=("convert", "rowwise"),
)
def template_tcvt_f16_to_ui8(src: pto.Tile, dst: pto.Tile):
    valid_rows, valid_cols = dst.valid_shape
    lanes_f16 = pto.elements_per_vreg(src.dtype)
    full_mask = pto.make_mask(src.dtype, pto.PAT.ALL)
    for row in range(0, valid_rows, 1):
        remained = valid_cols
        for col in range(0, valid_cols, lanes_f16):
            store_mask, remained = pto.make_mask(src.dtype, remained)
            vec = pto.vlds(src[row, col:])
            converted = pto.vcvt(
                vec,
                pto.ui8,
                full_mask,
                rnd=_round_mode(),
                sat=pto.VcvtSatMode.NOSAT,
                part=pto.VcvtPartMode.EVEN,
            )
            pto.vsts(converted, dst[row, col:], store_mask, dist=pto.VStoreDist.PK_B16)


def _render_tcvt_f16_to_ui8_1d(src: pto.Tile, dst: pto.Tile):
    src_ptr = src.as_ptr()
    dst_ptr = dst.as_ptr()
    full_mask = pto.make_mask(src.dtype, pto.PAT.ALL)

    def emit_chunk(offset, store_mask):
        vec = pto.vlds(src_ptr, offset)
        converted = pto.vcvt(
            vec,
            pto.ui8,
            full_mask,
            rnd=_round_mode(),
            sat=pto.VcvtSatMode.NOSAT,
            part=pto.VcvtPartMode.EVEN,
        )
        pto.vsts(
            converted,
            dst_ptr,
            offset,
            store_mask,
            dist=pto.VStoreDist.PK_B16,
        )

    _emit_tcvt_1d(dst, src.dtype, src.dtype, 1, emit_chunk)


template_tcvt_f16_to_ui8_1d = _register_tcvt_1d(
    name="template_tcvt_f16_to_ui8",
    dtypes=("f16", "ui8"),
    renderer=_render_tcvt_f16_to_ui8_1d,
)


@tilelib.tile_template(
    op="pto.tcvt",
    target="a5",
    name="template_tcvt_f16_to_si8",
    dtypes=[("f16", "si8")],
    iteration_axis="none",
    op_engine="vector",
    op_class="other",
    constraints=[_rowwise],
    id=19,
    loop_depth=2,
    is_post_update=False,
    tags=("convert", "rowwise"),
)
def template_tcvt_f16_to_si8(src: pto.Tile, dst: pto.Tile):
    valid_rows, valid_cols = dst.valid_shape
    lanes_f16 = pto.elements_per_vreg(src.dtype)
    pg = pto.make_mask(src.dtype, pto.PAT.ALL)
    for row in range(0, valid_rows, 1):
        remained = valid_cols
        for col in range(0, valid_cols, lanes_f16):
            full_mask, _ = pto.make_mask(src.dtype, lanes_f16)
            store_mask, remained = pto.make_mask(src.dtype, remained)
            vec_f16 = pto.vlds(src[row, col:])
            vec_i16 = pto.vcvt(
                vec_f16,
                pto.i16,
                full_mask,
                rnd=_round_mode(),
                sat=pto.VcvtSatMode.NOSAT,
            )
            v_mask = pto.vdup(pto.i16(255), pg)
            vec_i16_and = pto.vand(vec_i16, v_mask, store_mask)
            vec_f16_temp = pto.vcvt(
                vec_i16_and,
                pto.f16,
                full_mask,
                rnd=_round_mode(),
            )
            vec_si8 = pto.vcvt(
                vec_f16_temp,
                pto.si8,
                full_mask,
                rnd=_round_mode(),
                sat=pto.VcvtSatMode.NOSAT,
                part=pto.VcvtPartMode.EVEN,
            )
            pto.vsts(vec_si8, dst[row, col:], store_mask, dist=pto.VStoreDist.PK_B16)


def _render_tcvt_f16_to_si8_1d(src: pto.Tile, dst: pto.Tile):
    src_ptr = src.as_ptr()
    dst_ptr = dst.as_ptr()
    pg = pto.make_mask(src.dtype, pto.PAT.ALL)
    full_mask, _ = pto.make_mask(
        src.dtype,
        pto.elements_per_vreg(src.dtype),
    )

    def emit_chunk(offset, store_mask):
        vec_f16 = pto.vlds(src_ptr, offset)
        vec_i16 = pto.vcvt(
            vec_f16,
            pto.i16,
            full_mask,
            rnd=_round_mode(),
            sat=pto.VcvtSatMode.NOSAT,
        )
        v_mask = pto.vdup(pto.i16(255), pg)
        vec_i16_and = pto.vand(vec_i16, v_mask, store_mask)
        vec_f16_temp = pto.vcvt(
            vec_i16_and,
            pto.f16,
            full_mask,
            rnd=_round_mode(),
        )
        vec_si8 = pto.vcvt(
            vec_f16_temp,
            pto.si8,
            full_mask,
            rnd=_round_mode(),
            sat=pto.VcvtSatMode.NOSAT,
            part=pto.VcvtPartMode.EVEN,
        )
        pto.vsts(
            vec_si8,
            dst_ptr,
            offset,
            store_mask,
            dist=pto.VStoreDist.PK_B16,
        )

    _emit_tcvt_1d(dst, src.dtype, src.dtype, 1, emit_chunk)


template_tcvt_f16_to_si8_1d = _register_tcvt_1d(
    name="template_tcvt_f16_to_si8",
    dtypes=("f16", "si8"),
    renderer=_render_tcvt_f16_to_si8_1d,
)


template_tcvt_bf16_to_f32 = _register_tcvt(
    name="template_tcvt_bf16_to_f32",
    dtypes=("bf16", "f32"),
    idx=20,
    rnd=False,
    part="even",
    load_dist="UNPK_B16",
    convert_mask="src_full",
)

template_tcvt_i16_to_f32 = _register_tcvt(
    name="template_tcvt_i16_to_f32",
    dtypes=("i16", "f32"),
    idx=21,
    rnd=False,
    part="even",
    load_dist="UNPK_B16",
    convert_mask="src_full",
)

template_tcvt_i16_to_i32 = _register_tcvt(
    name="template_tcvt_i16_to_i32",
    dtypes=("i16", "i32"),
    idx=22,
    rnd=False,
    part="even",
    load_dist="UNPK_B16",
    convert_mask="src_full",
)

template_tcvt_i16_to_ui32 = _register_tcvt(
    name="template_tcvt_i16_to_ui32",
    dtypes=("i16", "ui32"),
    idx=23,
    rnd=False,
    part="even",
    load_dist="UNPK_B16",
    convert_mask="src_full",
)

template_tcvt_bf16_to_i32 = _register_tcvt(
    name="template_tcvt_bf16_to_i32",
    dtypes=("bf16", "i32"),
    idx=9,
    rnd=True,
    sat="sat",
    part="even",
    load_dist="UNPK_B16",
    convert_mask="src_full",
)

template_tcvt_ui8_to_ui16 = _register_tcvt(
    name="template_tcvt_ui8_to_ui16",
    dtypes=("ui8", "ui16"),
    idx=10,
    rnd=False,
    part="even",
    load_dist="UNPK_B8",
    convert_mask="src_full",
)

template_tcvt_ui8_to_f16 = _register_tcvt(
    name="template_tcvt_ui8_to_f16",
    dtypes=("ui8", "f16"),
    idx=24,
    rnd=False,
    part="even",
    load_dist="UNPK_B8",
    convert_mask="src_full",
)

template_tcvt_si8_to_f16 = _register_tcvt(
    name="template_tcvt_si8_to_f16",
    dtypes=("si8", "f16"),
    idx=25,
    rnd=False,
    part="even",
    load_dist="UNPK_B8",
    convert_mask="src_full",
)

template_tcvt_si8_to_si16 = _register_tcvt(
    name="template_tcvt_si8_to_si16",
    dtypes=("si8", "si16"),
    idx=26,
    rnd=False,
    part="even",
    load_dist="UNPK_B8",
    store_dist=pto.VStoreDist.NORM_B16,
    convert_mask="src_full",
)

template_tcvt_i32_to_i16 = _register_tcvt(
    name="template_tcvt_i32_to_i16",
    dtypes=("i32", "i16"),
    idx=28,
    rnd=False,
    sat="nosat",
    part="even",
    store_dist=pto.VStoreDist.PK_B32,
    mask_dtype="src",
    convert_mask="src_full",
)

template_tcvt_i32_to_ui16 = _register_tcvt(
    name="template_tcvt_i32_to_ui16",
    dtypes=("i32", "ui16"),
    idx=29,
    rnd=False,
    sat="sat",
    part="even",
    store_dist=pto.VStoreDist.PK_B32,
    mask_dtype="src",
    convert_mask="src_full",
)

template_tcvt_ui32_to_i16 = _register_tcvt(
    name="template_tcvt_ui32_to_i16",
    dtypes=("ui32", "i16"),
    idx=30,
    rnd=False,
    sat="sat",
    part="even",
    store_dist=pto.VStoreDist.PK_B32,
    mask_dtype="src",
    convert_mask="src_full",
)

template_tcvt_ui32_to_ui16 = _register_tcvt(
    name="template_tcvt_ui32_to_ui16",
    dtypes=("ui32", "ui16"),
    idx=31,
    rnd=False,
    sat="sat",
    part="even",
    store_dist=pto.VStoreDist.PK_B32,
    mask_dtype="src",
    convert_mask="src_full",
)


@tilelib.tile_template(
    op="pto.tcvt",
    target="a5",
    name="template_tcvt_si8_to_i32",
    dtypes=[("si8", "i32")],
    iteration_axis="none",
    op_engine="vector",
    op_class="other",
    constraints=[_rowwise],
    id=32,
    loop_depth=2,
    is_post_update=False,
    tags=("convert", "rowwise"),
)
def template_tcvt_si8_to_i32(src: pto.Tile, dst: pto.Tile):
    valid_rows, valid_cols = dst.valid_shape
    b8_mask = pto.make_mask(pto.ui8, pto.PAT.ALL)
    v_zero = pto.vbitcast(pto.vdup(pto.i8(0), b8_mask), pto.ui8)
    lanes_i16 = pto.elements_per_vreg(pto.i16)
    lanes_i32 = pto.elements_per_vreg(pto.i32)
    for row in range(0, valid_rows, 1):
        remained = valid_cols
        next_remained = valid_cols - lanes_i32
        for col in range(0, valid_cols, lanes_i16):
            mask_b16_cur, remained = pto.make_mask(pto.i16, remained)
            mask_b16_next, next_remained = pto.make_mask(pto.i16, next_remained)
            mask_b32_cur = pto.punpack(mask_b16_cur, pto.PredicatePart.LOWER, to_type=pto.mask_b32)
            mask_b32_next = pto.punpack(mask_b16_next, pto.PredicatePart.LOWER, to_type=pto.mask_b32)
            vec_si8_0 = pto.vlds(src[row, col:], dist="UNPK_B8")
            vec_ui8_0 = pto.vbitcast(vec_si8_0, pto.ui8)
            vec_ui8_1, vec_ui8_2 = pto.vintlv(vec_ui8_0, v_zero)
            vec_si8_1 = pto.vbitcast(vec_ui8_1, pto.si8)
            vec_si8_2 = pto.vbitcast(vec_ui8_2, pto.si8)
            output_0 = pto.vcvt(vec_si8_1, pto.i32, b8_mask, part=pto.VcvtPartMode.P0)
            output_1 = pto.vcvt(vec_si8_2, pto.i32, b8_mask, part=pto.VcvtPartMode.P0)
            pto.vsts(output_0, dst[row, col:], mask_b32_cur, dist=pto.VStoreDist.NORM_B32)
            pto.vsts(output_1, dst[row, col + lanes_i32:], mask_b32_next, dist=pto.VStoreDist.NORM_B32)


@rewrite_jit_function
def _render_tcvt_si8_to_i32_1d(src: pto.Tile, dst: pto.Tile):
    valid_rows, valid_cols = dst.valid_shape
    total_elements = valid_rows * valid_cols
    src_ptr = src.as_ptr()
    dst_ptr = dst.as_ptr()
    b8_mask = pto.make_mask(pto.ui8, pto.PAT.ALL)
    v_zero = pto.vbitcast(pto.vdup(pto.i8(0), b8_mask), pto.ui8)
    lanes_i16 = pto.elements_per_vreg(pto.i16)
    lanes_i32 = pto.elements_per_vreg(pto.i32)
    remained = total_elements
    next_remained = total_elements - lanes_i32
    for offset in range(0, total_elements, lanes_i16):
        mask_b16_cur, remained = pto.make_mask(pto.i16, remained)
        mask_b16_next, next_remained = pto.make_mask(
            pto.i16,
            next_remained,
        )
        mask_b32_cur = pto.punpack(
            mask_b16_cur,
            pto.PredicatePart.LOWER,
            to_type=pto.mask_b32,
        )
        mask_b32_next = pto.punpack(
            mask_b16_next,
            pto.PredicatePart.LOWER,
            to_type=pto.mask_b32,
        )
        vec_si8_0 = pto.vlds(src_ptr, offset, dist="UNPK_B8")
        vec_ui8_0 = pto.vbitcast(vec_si8_0, pto.ui8)
        vec_ui8_1, vec_ui8_2 = pto.vintlv(vec_ui8_0, v_zero)
        vec_si8_1 = pto.vbitcast(vec_ui8_1, pto.si8)
        vec_si8_2 = pto.vbitcast(vec_ui8_2, pto.si8)
        output_0 = pto.vcvt(
            vec_si8_1,
            pto.i32,
            b8_mask,
            part=pto.VcvtPartMode.P0,
        )
        output_1 = pto.vcvt(
            vec_si8_2,
            pto.i32,
            b8_mask,
            part=pto.VcvtPartMode.P0,
        )
        pto.vsts(
            output_0,
            dst_ptr,
            offset,
            mask_b32_cur,
            dist=pto.VStoreDist.NORM_B32,
        )
        pto.vsts(
            output_1,
            dst_ptr,
            offset + lanes_i32,
            mask_b32_next,
            dist=pto.VStoreDist.NORM_B32,
        )


template_tcvt_si8_to_i32_1d = _register_tcvt_1d(
    name="template_tcvt_si8_to_i32",
    dtypes=("si8", "i32"),
    renderer=_render_tcvt_si8_to_i32_1d,
)


@rewrite_jit_function
def _render_32_to_ui8(src: pto.Tile, dst: pto.Tile):
    valid_rows, valid_cols = dst.valid_shape
    full_mask = pto.make_mask(src.dtype, pto.PAT.ALL)
    idx_mask_b8 = pto.pset_b8(pto.PAT.ALL)
    idx_mask_b16 = pto.pbitcast(idx_mask_b8, pto.mask_b16)
    lanes = pto.elements_per_vreg(src.dtype)
    v_idx = pto.vci(pto.i8(0), "ASC")
    v_idx_i16 = pto.vbitcast(v_idx, pto.i16)
    v_idx_i16 = pto.vmuls(v_idx_i16, pto.i16(4), idx_mask_b16)
    v_idx_ui8 = pto.vbitcast(v_idx_i16, pto.ui8)
    for row in range(0, valid_rows, 1):
        remained = valid_cols
        for col in range(0, valid_cols, lanes):
            # One source register contributes only 64 compacted bytes. Building
            # this directly as a b8 mask would consume up to 256 elements.
            iteration_mask, remained = pto.make_mask(src.dtype, remained)
            store_mask = pto.pbitcast(iteration_mask, pto.mask_b8)
            vec = pto.vlds(src[row, col:])
            converted = pto.vcvt(
                vec,
                pto.ui8,
                full_mask,
                sat=pto.VcvtSatMode.SAT,
                part=pto.VcvtPartMode.P0,
            )
            result = pto.vselr(converted, v_idx_ui8)
            pto.mem_bar(pto.BarrierType.VST_VST)
            pto.vsts(result, dst[row, col:], store_mask, dist=pto.VStoreDist.NORM_B8)


def _render_32_to_ui8_1d(src: pto.Tile, dst: pto.Tile):
    src_ptr = src.as_ptr()
    dst_ptr = dst.as_ptr()
    full_mask = pto.make_mask(src.dtype, pto.PAT.ALL)
    idx_mask_b8 = pto.pset_b8(pto.PAT.ALL)
    idx_mask_b16 = pto.pbitcast(idx_mask_b8, pto.mask_b16)
    v_idx = pto.vci(pto.i8(0), "ASC")
    v_idx_i16 = pto.vbitcast(v_idx, pto.i16)
    v_idx_i16 = pto.vmuls(v_idx_i16, pto.i16(4), idx_mask_b16)
    v_idx_ui8 = pto.vbitcast(v_idx_i16, pto.ui8)

    def emit_chunk(offset, iteration_mask):
        # Preserve the 64-element b32 loop granularity for the b8 store.
        store_mask = pto.pbitcast(iteration_mask, pto.mask_b8)
        vec = pto.vlds(src_ptr, offset)
        converted = pto.vcvt(
            vec,
            pto.ui8,
            full_mask,
            sat=pto.VcvtSatMode.SAT,
            part=pto.VcvtPartMode.P0,
        )
        result = pto.vselr(converted, v_idx_ui8)
        pto.mem_bar(pto.BarrierType.VST_VST)
        pto.vsts(
            result,
            dst_ptr,
            offset,
            store_mask,
            dist=pto.VStoreDist.NORM_B8,
        )

    _emit_tcvt_1d(dst, src.dtype, src.dtype, 1, emit_chunk)


@tilelib.tile_template(
    op="pto.tcvt",
    target="a5",
    name="template_tcvt_i32_to_ui8",
    dtypes=[("i32", "ui8")],
    iteration_axis="none",
    op_engine="vector",
    op_class="other",
    constraints=[_rowwise],
    id=33,
    loop_depth=2,
    is_post_update=False,
    tags=("convert", "rowwise"),
)
def template_tcvt_i32_to_ui8(src: pto.Tile, dst: pto.Tile):
    _render_32_to_ui8(src, dst)


@tilelib.tile_template(
    op="pto.tcvt",
    target="a5",
    name="template_tcvt_ui32_to_ui8",
    dtypes=[("ui32", "ui8")],
    iteration_axis="none",
    op_engine="vector",
    op_class="other",
    constraints=[_rowwise],
    id=34,
    loop_depth=2,
    is_post_update=False,
    tags=("convert", "rowwise"),
)
def template_tcvt_ui32_to_ui8(src: pto.Tile, dst: pto.Tile):
    _render_32_to_ui8(src, dst)


template_tcvt_i32_to_ui8_1d = _register_tcvt_1d(
    name="template_tcvt_i32_to_ui8",
    dtypes=("i32", "ui8"),
    renderer=_render_32_to_ui8_1d,
)

template_tcvt_ui32_to_ui8_1d = _register_tcvt_1d(
    name="template_tcvt_ui32_to_ui8",
    dtypes=("ui32", "ui8"),
    renderer=_render_32_to_ui8_1d,
)


@tilelib.tile_template(
    op="pto.tcvt",
    target="a5",
    name="template_tcvt_i16_to_ui8",
    dtypes=[("i16", "ui8")],
    iteration_axis="none",
    op_engine="vector",
    op_class="other",
    constraints=[_rowwise],
    id=35,
    loop_depth=2,
    is_post_update=False,
    tags=("convert", "rowwise"),
)
def template_tcvt_i16_to_ui8(src: pto.Tile, dst: pto.Tile):
    valid_rows, valid_cols = dst.valid_shape
    lanes_i16 = pto.elements_per_vreg(src.dtype)
    full_mask = pto.make_mask(src.dtype, pto.PAT.ALL)
    for row in range(0, valid_rows, 1):
        remained = valid_cols
        for col in range(0, valid_cols, lanes_i16):
            store_mask, remained = pto.make_mask(src.dtype, remained)
            vec = pto.vlds(src[row, col:])
            converted = pto.vcvt(
                vec,
                pto.ui8,
                full_mask,
                sat=pto.VcvtSatMode.SAT,
                part=pto.VcvtPartMode.EVEN,
            )
            pto.vsts(converted, dst[row, col:], store_mask, dist=pto.VStoreDist.PK_B16)


def _render_tcvt_i16_to_ui8_1d(src: pto.Tile, dst: pto.Tile):
    src_ptr = src.as_ptr()
    dst_ptr = dst.as_ptr()
    full_mask = pto.make_mask(src.dtype, pto.PAT.ALL)

    def emit_chunk(offset, store_mask):
        vec = pto.vlds(src_ptr, offset)
        converted = pto.vcvt(
            vec,
            pto.ui8,
            full_mask,
            sat=pto.VcvtSatMode.SAT,
            part=pto.VcvtPartMode.EVEN,
        )
        pto.vsts(
            converted,
            dst_ptr,
            offset,
            store_mask,
            dist=pto.VStoreDist.PK_B16,
        )

    _emit_tcvt_1d(dst, src.dtype, src.dtype, 1, emit_chunk)


template_tcvt_i16_to_ui8_1d = _register_tcvt_1d(
    name="template_tcvt_i16_to_ui8",
    dtypes=("i16", "ui8"),
    renderer=_render_tcvt_i16_to_ui8_1d,
)


@tilelib.tile_template(
    op="pto.tcvt",
    target="a5",
    name="template_tcvt_i32_to_i64",
    dtypes=[("i32", "i64")],
    iteration_axis="none",
    op_engine="vector",
    op_class="other",
    constraints=[_rowwise],
    id=27,
    loop_depth=2,
    is_post_update=False,
    tags=("convert", "rowwise"),
)
def template_tcvt_i32_to_i64(src: pto.Tile, dst: pto.Tile):
    valid_rows, valid_cols = dst.valid_shape
    lanes_i64 = pto.elements_per_vreg(dst.dtype)
    full_mask = pto.make_mask(src.dtype, pto.PAT.ALL)
    for row in range(0, valid_rows, 1):
        remained = valid_cols * 2
        for col in range(0, valid_cols, lanes_i64):
            store_mask, remained = pto.make_mask(pto.i32, remained)
            vec = pto.vlds(src[row, col:], dist="UNPK_B32")
            converted = pto.vcvt(
                vec,
                pto.i64,
                full_mask,
                part=pto.VcvtPartMode.EVEN,
            )
            pto.vsts(converted, dst[row, col:], store_mask, dist=pto.VStoreDist.NORM_B32)


def _render_tcvt_i32_to_i64_1d(src: pto.Tile, dst: pto.Tile):
    src_ptr = src.as_ptr()
    dst_ptr = dst.as_ptr()
    full_mask = pto.make_mask(src.dtype, pto.PAT.ALL)

    def emit_chunk(offset, store_mask):
        vec = pto.vlds(src_ptr, offset, dist="UNPK_B32")
        converted = pto.vcvt(
            vec,
            pto.i64,
            full_mask,
            part=pto.VcvtPartMode.EVEN,
        )
        pto.vsts(
            converted,
            dst_ptr,
            offset,
            store_mask,
            dist=pto.VStoreDist.NORM_B32,
        )

    _emit_tcvt_1d(dst, dst.dtype, pto.i32, 2, emit_chunk)


template_tcvt_i32_to_i64_1d = _register_tcvt_1d(
    name="template_tcvt_i32_to_i64",
    dtypes=("i32", "i64"),
    renderer=_render_tcvt_i32_to_i64_1d,
)


@rewrite_jit_function
def _render_i64_to_32(src: pto.Tile, dst: pto.Tile, *, use_rounding: bool):
    valid_rows, valid_cols = dst.valid_shape
    lanes_i64 = pto.elements_per_vreg(src.dtype)
    for row in range(0, valid_rows, 1):
        remained = valid_cols * 2
        full_mask, _ = pto.make_mask(pto.i32, remained)
        for col in range(0, valid_cols, lanes_i64):
            store_mask, remained = pto.make_mask(dst.dtype, remained)
            vec = pto.vlds(src[row, col:])
            if use_rounding:
                converted = pto.vcvt(
                    vec,
                    dst.dtype,
                    full_mask,
                    rnd=_round_mode(),
                    part=pto.VcvtPartMode.EVEN,
                )
            else:
                converted = pto.vcvt(
                    vec,
                    dst.dtype,
                    full_mask,
                    sat=pto.VcvtSatMode.NOSAT,
                    part=pto.VcvtPartMode.EVEN,
                )
            pto.vsts(converted, dst[row, col:], store_mask, dist=pto.VStoreDist.PK_B64)


def _render_i64_to_32_1d(src: pto.Tile, dst: pto.Tile, *, use_rounding: bool):
    src_ptr = src.as_ptr()
    dst_ptr = dst.as_ptr()
    full_mask = pto.make_mask(pto.i32, pto.PAT.ALL)

    def emit_chunk(offset, store_mask):
        vec = pto.vlds(src_ptr, offset)
        if use_rounding:
            converted = pto.vcvt(
                vec,
                dst.dtype,
                full_mask,
                rnd=_round_mode(),
                part=pto.VcvtPartMode.EVEN,
            )
        else:
            converted = pto.vcvt(
                vec,
                dst.dtype,
                full_mask,
                sat=pto.VcvtSatMode.NOSAT,
                part=pto.VcvtPartMode.EVEN,
            )
        pto.vsts(
            converted,
            dst_ptr,
            offset,
            store_mask,
            dist=pto.VStoreDist.PK_B64,
        )

    _emit_tcvt_1d(dst, src.dtype, dst.dtype, 2, emit_chunk)


def _render_i64_to_f32_1d(src: pto.Tile, dst: pto.Tile):
    _render_i64_to_32_1d(src, dst, use_rounding=True)


def _render_i64_to_i32_1d(src: pto.Tile, dst: pto.Tile):
    _render_i64_to_32_1d(src, dst, use_rounding=False)


@tilelib.tile_template(
    op="pto.tcvt",
    target="a5",
    name="template_tcvt_i64_to_f32",
    dtypes=[("i64", "f32")],
    iteration_axis="none",
    op_engine="vector",
    op_class="other",
    constraints=[_rowwise],
    id=36,
    loop_depth=2,
    is_post_update=False,
    tags=("convert", "rowwise"),
)
def template_tcvt_i64_to_f32(src: pto.Tile, dst: pto.Tile):
    _render_i64_to_32(src, dst, use_rounding=True)


@tilelib.tile_template(
    op="pto.tcvt",
    target="a5",
    name="template_tcvt_i64_to_i32",
    dtypes=[("i64", "i32")],
    iteration_axis="none",
    op_engine="vector",
    op_class="other",
    constraints=[_rowwise],
    id=37,
    loop_depth=2,
    is_post_update=False,
    tags=("convert", "rowwise"),
)
def template_tcvt_i64_to_i32(src: pto.Tile, dst: pto.Tile):
    _render_i64_to_32(src, dst, use_rounding=False)


template_tcvt_i64_to_f32_1d = _register_tcvt_1d(
    name="template_tcvt_i64_to_f32",
    dtypes=("i64", "f32"),
    renderer=_render_i64_to_f32_1d,
)

template_tcvt_i64_to_i32_1d = _register_tcvt_1d(
    name="template_tcvt_i64_to_i32",
    dtypes=("i64", "i32"),
    renderer=_render_i64_to_i32_1d,
)


@tilelib.tile_template(
    op="pto.tcvt",
    target="a5",
    name="template_tcvt_f32_to_fp8",
    dtypes=[("f32", "f8e4m3"), ("f32", "f8e5m2")],
    iteration_axis="none",
    op_engine="vector",
    op_class="other",
    constraints=[_rowwise],
    id=11,
    loop_depth=2,
    is_post_update=False,
    tags=("convert", "rowwise", "low_precision"),
)
def template_tcvt_f32_to_fp8(src: pto.Tile, dst: pto.Tile):
    valid_rows, valid_cols = dst.valid_shape
    dst_dtype = dst.dtype
    lanes_f32 = pto.elements_per_vreg(src.dtype)
    src_mask = pto.make_mask(src.dtype, pto.PAT.ALL)
    idx_mask_b8 = pto.pset_b8(pto.PAT.ALL)
    idx_mask_b16 = pto.pbitcast(idx_mask_b8, pto.mask_b16)
    v_idx = pto.vci(pto.i8(0), "ASC")
    v_idx_i16 = pto.vbitcast(v_idx, pto.i16)
    v_idx_i16 = pto.vmuls(v_idx_i16, pto.i16(4), idx_mask_b16)
    v_idx_ui8 = pto.vbitcast(v_idx_i16, pto.ui8)
    for row in range(0, valid_rows, 1):
        remained = valid_cols
        for col in range(0, valid_cols, lanes_f32):
            dst_mask, remained = pto.make_mask(dst_dtype, remained)
            vec = pto.vlds(src[row, col:])
            converted = pto.vcvt(
                vec,
                dst_dtype,
                src_mask,
                rnd=_round_mode(),
                sat=pto.VcvtSatMode.SAT,
                part=pto.VcvtPartMode.P0,
            )
            result = _vselr_low_precision(converted, v_idx_ui8)
            pto.mem_bar(pto.BarrierType.VST_VST)
            pto.vsts(result, dst[row, col:], dst_mask, dist=pto.VStoreDist.NORM_B8)


def _render_tcvt_f32_to_fp8_1d(src: pto.Tile, dst: pto.Tile):
    src_ptr = src.as_ptr()
    dst_ptr = dst.as_ptr()
    dst_dtype = dst.dtype
    src_mask = pto.make_mask(src.dtype, pto.PAT.ALL)
    idx_mask_b8 = pto.pset_b8(pto.PAT.ALL)
    idx_mask_b16 = pto.pbitcast(idx_mask_b8, pto.mask_b16)
    v_idx = pto.vci(pto.i8(0), "ASC")
    v_idx_i16 = pto.vbitcast(v_idx, pto.i16)
    v_idx_i16 = pto.vmuls(v_idx_i16, pto.i16(4), idx_mask_b16)
    v_idx_ui8 = pto.vbitcast(v_idx_i16, pto.ui8)

    def emit_chunk(offset, dst_mask):
        vec = pto.vlds(src_ptr, offset)
        converted = pto.vcvt(
            vec,
            dst_dtype,
            src_mask,
            rnd=_round_mode(),
            sat=pto.VcvtSatMode.SAT,
            part=pto.VcvtPartMode.P0,
        )
        result = _vselr_low_precision(converted, v_idx_ui8)
        pto.mem_bar(pto.BarrierType.VST_VST)
        pto.vsts(
            result,
            dst_ptr,
            offset,
            dst_mask,
            dist=pto.VStoreDist.NORM_B8,
        )

    _emit_tcvt_1d(dst, src.dtype, dst_dtype, 1, emit_chunk)


template_tcvt_f32_to_fp8_1d = _register_tcvt_1d(
    name="template_tcvt_f32_to_fp8",
    dtypes=(("f32", "f8e4m3"), ("f32", "f8e5m2")),
    renderer=_render_tcvt_f32_to_fp8_1d,
    tags=("low_precision",),
)


@tilelib.tile_template(
    op="pto.tcvt",
    target="a5",
    name="template_tcvt_f32_to_hif8",
    dtypes=[("f32", "hif8")],
    iteration_axis="none",
    op_engine="vector",
    op_class="other",
    constraints=[_rowwise],
    id=12,
    loop_depth=2,
    is_post_update=False,
    tags=("convert", "rowwise", "low_precision"),
)
def template_tcvt_f32_to_hif8(src: pto.Tile, dst: pto.Tile):
    valid_rows, valid_cols = dst.valid_shape
    dst_dtype = dst.dtype
    lanes_f32 = pto.elements_per_vreg(src.dtype)
    src_mask = pto.make_mask(src.dtype, pto.PAT.ALL)
    idx_mask_b8 = pto.pset_b8(pto.PAT.ALL)
    idx_mask_b16 = pto.pbitcast(idx_mask_b8, pto.mask_b16)
    v_idx = pto.vci(pto.i8(0), "ASC")
    v_idx_i16 = pto.vbitcast(v_idx, pto.i16)
    v_idx_i16 = pto.vmuls(v_idx_i16, pto.i16(4), idx_mask_b16)
    v_idx_ui8 = pto.vbitcast(v_idx_i16, pto.ui8)
    for row in range(0, valid_rows, 1):
        remained = valid_cols
        for col in range(0, valid_cols, lanes_f32):
            dst_mask, remained = pto.make_mask(dst_dtype, remained)
            vec = pto.vlds(src[row, col:])
            converted = pto.vcvt(
                vec,
                dst_dtype,
                src_mask,
                rnd=pto.VcvtRoundMode.A,
                sat=pto.VcvtSatMode.NOSAT,
                part=pto.VcvtPartMode.P0,
            )
            result = _vselr_low_precision(converted, v_idx_ui8)
            pto.mem_bar(pto.BarrierType.VST_VST)
            pto.vsts(result, dst[row, col:], dst_mask, dist=pto.VStoreDist.NORM_B8)


def _render_tcvt_f32_to_hif8_1d(src: pto.Tile, dst: pto.Tile):
    src_ptr = src.as_ptr()
    dst_ptr = dst.as_ptr()
    dst_dtype = dst.dtype
    src_mask = pto.make_mask(src.dtype, pto.PAT.ALL)
    idx_mask_b8 = pto.pset_b8(pto.PAT.ALL)
    idx_mask_b16 = pto.pbitcast(idx_mask_b8, pto.mask_b16)
    v_idx = pto.vci(pto.i8(0), "ASC")
    v_idx_i16 = pto.vbitcast(v_idx, pto.i16)
    v_idx_i16 = pto.vmuls(v_idx_i16, pto.i16(4), idx_mask_b16)
    v_idx_ui8 = pto.vbitcast(v_idx_i16, pto.ui8)

    def emit_chunk(offset, dst_mask):
        vec = pto.vlds(src_ptr, offset)
        converted = pto.vcvt(
            vec,
            dst_dtype,
            src_mask,
            rnd=pto.VcvtRoundMode.A,
            sat=pto.VcvtSatMode.NOSAT,
            part=pto.VcvtPartMode.P0,
        )
        result = _vselr_low_precision(converted, v_idx_ui8)
        pto.mem_bar(pto.BarrierType.VST_VST)
        pto.vsts(
            result,
            dst_ptr,
            offset,
            dst_mask,
            dist=pto.VStoreDist.NORM_B8,
        )

    _emit_tcvt_1d(dst, src.dtype, dst_dtype, 1, emit_chunk)


template_tcvt_f32_to_hif8_1d = _register_tcvt_1d(
    name="template_tcvt_f32_to_hif8",
    dtypes=("f32", "hif8"),
    renderer=_render_tcvt_f32_to_hif8_1d,
    tags=("low_precision",),
)


@tilelib.tile_template(
    op="pto.tcvt",
    target="a5",
    name="template_tcvt_f16_to_hif8",
    dtypes=[("f16", "hif8")],
    iteration_axis="none",
    op_engine="vector",
    op_class="other",
    constraints=[_rowwise],
    id=13,
    loop_depth=2,
    is_post_update=False,
    tags=("convert", "rowwise", "low_precision"),
)
def template_tcvt_f16_to_hif8(src: pto.Tile, dst: pto.Tile):
    valid_rows, valid_cols = dst.valid_shape
    dst_dtype = dst.dtype
    lanes_f16 = pto.elements_per_vreg(src.dtype)
    src_mask = pto.make_mask(src.dtype, pto.PAT.ALL)
    for row in range(0, valid_rows, 1):
        remained = valid_cols
        for col in range(0, valid_cols, lanes_f16):
            dst_mask, remained = pto.make_mask(src.dtype, remained)
            vec = pto.vlds(src[row, col:])
            converted = pto.vcvt(
                vec,
                dst_dtype,
                src_mask,
                rnd=pto.VcvtRoundMode.A,
                sat=pto.VcvtSatMode.NOSAT,
                part=pto.VcvtPartMode.EVEN,
            )
            pto.vsts(converted, dst[row, col:], dst_mask, dist=pto.VStoreDist.PK_B16)


def _render_tcvt_f16_to_hif8_1d(src: pto.Tile, dst: pto.Tile):
    src_ptr = src.as_ptr()
    dst_ptr = dst.as_ptr()
    dst_dtype = dst.dtype
    src_mask = pto.make_mask(src.dtype, pto.PAT.ALL)

    def emit_chunk(offset, dst_mask):
        vec = pto.vlds(src_ptr, offset)
        converted = pto.vcvt(
            vec,
            dst_dtype,
            src_mask,
            rnd=pto.VcvtRoundMode.A,
            sat=pto.VcvtSatMode.NOSAT,
            part=pto.VcvtPartMode.EVEN,
        )
        pto.vsts(
            converted,
            dst_ptr,
            offset,
            dst_mask,
            dist=pto.VStoreDist.PK_B16,
        )

    _emit_tcvt_1d(dst, src.dtype, src.dtype, 1, emit_chunk)


template_tcvt_f16_to_hif8_1d = _register_tcvt_1d(
    name="template_tcvt_f16_to_hif8",
    dtypes=("f16", "hif8"),
    renderer=_render_tcvt_f16_to_hif8_1d,
    tags=("low_precision",),
)


@tilelib.tile_template(
    op="pto.tcvt",
    target="a5",
    name="template_tcvt_bf16_to_fp4",
    dtypes=[("bf16", "f4e1m2x2"), ("bf16", "f4e2m1x2")],
    iteration_axis="none",
    op_engine="vector",
    op_class="other",
    constraints=[_rowwise_bf16_to_fp4],
    id=14,
    loop_depth=2,
    is_post_update=False,
    tags=("convert", "rowwise", "low_precision"),
)
def template_tcvt_bf16_to_fp4(src: pto.Tile, dst: pto.Tile):
    valid_rows, valid_cols = dst.valid_shape
    dst_dtype = dst.dtype
    lanes_bf16 = pto.elements_per_vreg(src.dtype)
    dst_chunk_cols = lanes_bf16 // 2
    idx_mask_b8 = pto.pset_b8(pto.PAT.ALL)
    idx_mask_b16 = pto.pbitcast(idx_mask_b8, pto.mask_b16)
    v_idx = pto.vci(pto.i8(0), "ASC")
    v_idx_i16 = pto.vbitcast(v_idx, pto.i16)
    v_idx_i16 = pto.vmuls(v_idx_i16, pto.i16(4), idx_mask_b16)
    v_idx_ui8 = pto.vbitcast(v_idx_i16, pto.ui8)
    for row in range(0, valid_rows, 1):
        remained = valid_cols
        src_remained = valid_cols * 2
        for col in range(0, valid_cols, dst_chunk_cols):
            dst_mask, remained = pto.make_mask(dst_dtype, remained)
            src_mask, src_remained = pto.make_mask(src.dtype, src_remained)
            vec = pto.vlds(src[row, col * 2:])
            converted = pto.vcvt(
                vec,
                dst_dtype,
                src_mask,
                rnd=_round_mode(),
                part=pto.VcvtPartMode.P0,
            )
            result = _vselr_low_precision(converted, v_idx_ui8)
            pto.mem_bar(pto.BarrierType.VST_VST)
            pto.vsts(result, dst[row, col:], dst_mask, dist=pto.VStoreDist.NORM_B8)


@rewrite_jit_function
def _render_tcvt_bf16_to_fp4_1d(src: pto.Tile, dst: pto.Tile):
    valid_rows, valid_cols = dst.valid_shape
    total_destination_elements = valid_rows * valid_cols
    src_ptr = src.as_ptr()
    dst_ptr = dst.as_ptr()
    dst_dtype = dst.dtype
    lanes_bf16 = pto.elements_per_vreg(src.dtype)
    dst_chunk_cols = lanes_bf16 // 2
    idx_mask_b8 = pto.pset_b8(pto.PAT.ALL)
    idx_mask_b16 = pto.pbitcast(idx_mask_b8, pto.mask_b16)
    v_idx = pto.vci(pto.i8(0), "ASC")
    v_idx_i16 = pto.vbitcast(v_idx, pto.i16)
    v_idx_i16 = pto.vmuls(v_idx_i16, pto.i16(4), idx_mask_b16)
    v_idx_ui8 = pto.vbitcast(v_idx_i16, pto.ui8)
    remained = total_destination_elements
    src_remained = total_destination_elements * 2
    for offset in range(0, total_destination_elements, dst_chunk_cols):
        dst_mask, remained = pto.make_mask(dst_dtype, remained)
        src_mask, src_remained = pto.make_mask(src.dtype, src_remained)
        vec = pto.vlds(src_ptr, offset * 2)
        converted = pto.vcvt(
            vec,
            dst_dtype,
            src_mask,
            rnd=_round_mode(),
            part=pto.VcvtPartMode.P0,
        )
        result = _vselr_low_precision(converted, v_idx_ui8)
        pto.mem_bar(pto.BarrierType.VST_VST)
        pto.vsts(
            result,
            dst_ptr,
            offset,
            dst_mask,
            dist=pto.VStoreDist.NORM_B8,
        )


template_tcvt_bf16_to_fp4_1d = _register_tcvt_1d(
    name="template_tcvt_bf16_to_fp4",
    dtypes=(
        ("bf16", "f4e1m2x2"),
        ("bf16", "f4e2m1x2"),
    ),
    renderer=_render_tcvt_bf16_to_fp4_1d,
    source_elements_per_destination=2,
    tags=("low_precision",),
)


def _register_deferred_tcvt_1d():
    """Assign stable 1D IDs after every 2D fallback is registered."""

    registry = tilelib.default_registry()
    fallbacks = {
        candidate.name: candidate
        for candidate in registry.lookup("pto.tcvt", "a5")
        if candidate.metadata.loop_depth == 2
    }
    next_candidate_id = max(
        candidate.metadata.id for candidate in fallbacks.values()
    ) + 1
    replacements = {}
    for candidate in sorted(
        _PENDING_TCVT_1D,
        key=lambda item: fallbacks[
            item.name.removesuffix("_1d")
        ].metadata.id,
    ):
        assigned = replace(
            candidate,
            metadata=replace(
                candidate.metadata,
                id=next_candidate_id,
            ),
        )
        registry.register(assigned)
        replacements[id(candidate)] = assigned
        next_candidate_id += 1

    for global_name, value in tuple(globals().items()):
        assigned = replacements.get(id(value))
        if assigned is not None:
            globals()[global_name] = assigned
    _PENDING_TCVT_1D.clear()


_register_deferred_tcvt_1d()
