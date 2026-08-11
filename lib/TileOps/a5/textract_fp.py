# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""PTODSL TileLib templates for the ``pto.textract`` fp form."""

from ptodsl import pto
import ptodsl.tilelib as tilelib


def _textract_fp_constraint(src_kind, src_memory_space, fp_kind, fp_memory_space, dst_kind, dst_memory_space, **_):
    return (
        src_kind == "tile"
        and fp_kind == "tile"
        and dst_kind == "tile"
        and src_memory_space == "acc"
        and fp_memory_space == "scaling"
        and dst_memory_space == "mat"
    )


def _register_textract_fp(name, signatures, quant_mode, template_id):
    if isinstance(signatures, tuple) and signatures and isinstance(signatures[0], str):
        signatures = (signatures,)

    @tilelib.tile_template(
        op="pto.textract",
        target="a5",
        name=name,
        dtypes=tuple(signatures),
        iteration_axis="none",
        op_engine="other",
        op_class="movement",
        constraints=[_textract_fp_constraint],
        id=template_id,
        loop_depth=0,
        is_post_update=False,
        tags=("extract", "acc", "mat", "fp"),
    )
    def _template(src: pto.Tile, index_row: pto.i32, index_col: pto.i32, dst: pto.Tile, fp: pto.Tile):
        m, n = dst.valid_shape
        src_ptr = src.as_ptr()
        if str(src.dtype) == "si32":
            src_ptr = pto.castptr(src_ptr, pto.ptr(pto.i32, "acc"))
        pto.mte_l0c_l1(
            src_ptr,
            dst.as_ptr(),
            m,
            n,
            src.shape[0],
            dst.shape[1],
            pre_quant=(fp.as_ptr(), quant_mode),
        )

    return _template


template_textract_fp_f32_si8 = _register_textract_fp(
    "template_textract_fp_f32_si8",
    ("f32", "i32", "i32", "si8", "f32"),
    "qf322b8_pre_vec",
    0,
)
template_textract_fp_f32_ui8 = _register_textract_fp(
    "template_textract_fp_f32_ui8",
    ("f32", "i32", "i32", "ui8", "f32"),
    "qf322b8_pre_vec",
    1,
)
template_textract_fp_f32_f16 = _register_textract_fp(
    "template_textract_fp_f32_f16",
    ("f32", "i32", "i32", "f16", "f32"),
    "qf322f16_pre_vec",
    2,
)
template_textract_fp_f32_bf16 = _register_textract_fp(
    "template_textract_fp_f32_bf16",
    ("f32", "i32", "i32", "bf16", "f32"),
    "qf322bf16_pre_vec",
    3,
)
template_textract_fp_f32_f32 = _register_textract_fp(
    "template_textract_fp_f32_f32",
    ("f32", "i32", "i32", "f32", "f32"),
    "qf322f32_pre_vec",
    4,
)
template_textract_fp_si32_si8 = _register_textract_fp(
    "template_textract_fp_si32_si8",
    (
        ("si32", "i32", "i32", "si8", "f32"),
        ("i32", "i32", "i32", "si8", "f32"),
    ),
    "req8_vec",
    5,
)
template_textract_fp_si32_ui8 = _register_textract_fp(
    "template_textract_fp_si32_ui8",
    (
        ("si32", "i32", "i32", "ui8", "f32"),
        ("i32", "i32", "i32", "ui8", "f32"),
    ),
    "req8_vec",
    6,
)
template_textract_fp_si32_f16 = _register_textract_fp(
    "template_textract_fp_si32_f16",
    (
        ("si32", "i32", "i32", "f16", "f32"),
        ("i32", "i32", "i32", "f16", "f32"),
    ),
    "deqf16_vec",
    7,
)
template_textract_fp_si32_bf16 = _register_textract_fp(
    "template_textract_fp_si32_bf16",
    (
        ("si32", "i32", "i32", "bf16", "f32"),
        ("i32", "i32", "i32", "bf16", "f32"),
    ),
    "qs322bf16_pre_vec",
    8,
)
