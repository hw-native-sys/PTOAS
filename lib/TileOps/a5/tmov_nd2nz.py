# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""PTODSL TileLib template for ``pto.tmov`` UB ND -> UB NZ.

Covers both column-repeat shapes of pto-isa ``TMovToVecNd2Nz``:

- **single-column-repeat** (``valid_col <= lanes``): one row ``scf.for``
  carrying dst_ptr, body = ``vlds`` (fixed src, per-row explicit offset) ->
  ``vsstb`` with constant ``block_stride`` / ``repeat_stride=1`` (plain NZ),
  no ``cfgVsstbLast`` last-block special case. Mirrors pto-isa
  ``TMovNd2NzLoopRepeat1``. The 1/2-VL tail (cols < lanes) uses a count
  predicate (``CreatePredicate(validCol)``); no ``vpack`` — the dtype
  conversion/pack is owned by the preceding ``tcvt``.
- **multi-column-repeat** (``valid_col > lanes``): nested ``scf.for`` — an
  outer loop over column-block groups (``repeatTimes = ceil(cols/lanes)``,
  carrying dst_ptr + a runtime ``remained`` count for the per-group
  predicate) + an inner row loop + a trailing ``cfgVsstbLast`` beat per group
  (large ``repeat_stride_last`` that repositions dst to the next group's
  head). Mirrors pto-isa ``TMovNd2NzLoop``. dst_ptr threads group j's trailing
  beat into group j+1's loop init; the per-group count predicate
  (``min(remained, lanes)``) is full VL except the final partial group.

Serves two roles:
- InsertTemplateAttributes (the metadata pass) queries this candidate for
  legality (ND->NZ dst layout) so the ptodsl.tilelib daemon returns a legal
  candidate for tmov ND->NZ; without it the metadata pass fails before the VMI
  provider can render.
- Non-VMI backends (ptodsl/tilelang) fall back to rendering this VPTO-level
  body (the VMI backend renders its own mirror in ``vmi_tilelib.py``).

The load/store pointer discipline matches pto-isa: ``src_ptr`` is
loop-invariant and ``vlds`` uses an explicit per-beat offset
(``RowStride * i [+ j * lanes]``) in NORM mode (NO ``POST_UPDATE``);
only ``dst_ptr`` is loop-carried, auto-advanced by ``vsstb POST_UPDATE``
(advance by ``repeatStride`` each beat). The src-fixed + explicit-offset form
is used uniformly — pto-isa's ``vlds POST_UPDATE`` src-advance semantics are
sim-unreliable, and the explicit form is arithmetically equivalent (the
reference's src POST_UPDATE + ``srcOffset`` rewind advances src by exactly
``lanes`` per outer iteration).

Correctness is verified bit-exact against the pto-isa ``nd_to_nz`` golden
(``atol=0, rtol=0``) for the single-column path (full-VL and half-VL); the
multi-column path is new ground — pto-isa's own ``TMovNd2NzLoop`` is never
exercised with ``repeatTimes > 1`` in its tests, so this template + the
``fa-softmax-dn-init-multirepeat`` case provide the first end-to-end bit-exact
evidence for the multi-column-repeat path. See
``ND2NZ实现与精度问题记录.md``.
"""

from ptodsl import pto, scalar
import ptodsl.tilelib as tilelib


def _nd_src_nz_dst(src_kind, src_memory_space, src_config,
                   dst_kind, dst_memory_space, dst_config, **_):
    # src: UB/vec RowMajor+NoneBox (ND); dst: UB/vec ColMajor+RowMajor (NZ)
    return (
        src_kind == "tile" and dst_kind == "tile"
        and src_memory_space in {"ub", "vec"} and dst_memory_space in {"ub", "vec"}
        and src_config.b_layout == "row_major" and src_config.s_layout == "none_box"
        and dst_config.b_layout != "row_major" and dst_config.s_layout == "row_major"
    )


# ISA byte constants (pto/npu/a5: REPEAT_BYTE=256, BLOCK_BYTE_SIZE=32).
_REPEAT_BYTE = 256
_BLOCK_BYTE_SIZE = 32


def _nd2nz_repeat_stride_last(virtual_row, inner_loop_num):
    """vsstb repeat_stride for the last beat of a column-block group.

    Mirrors pto-isa ``TMovToVecNd2Nz`` ``repeatStrideLast = (REPEAT_BYTE *
    virtualRow - innerLoopNum * BLOCK_BYTE_SIZE) / BLOCK_BYTE_SIZE``: the large
    stride that repositions dst from the tail of one column-block group to the
    head of the next one (consumed by the NEXT outer iteration).
    """
    return (_REPEAT_BYTE * virtual_row - inner_loop_num * _BLOCK_BYTE_SIZE) // _BLOCK_BYTE_SIZE


@tilelib.tile_template(
    op="pto.tmov",
    target="a5",
    name="template_tmov_nd2nz",
    dtypes=[("f32", "f32"), ("f16", "f16"), ("bf16", "bf16"),
            ("i32", "i32"), ("i16", "i16"), ("i8", "i8"), ("ui8", "ui8")],
    iteration_axis="none",
    op_engine="vector",
    op_class="movement",
    constraints=[_nd_src_nz_dst,
                 tilelib.require_same_valid_shape("src", "dst")],
    id=8,
    loop_depth=1,
    is_post_update=True,
    tags=("move", "ub", "ub", "nd2nz", "nz"),
)
def template_tmov_nd2nz(src: pto.Tile, dst: pto.Tile):
    dtype = src.dtype
    valid_rows, valid_cols = src.valid_shape
    block_stride = dst.shape[0]  # NZ dst physical (aligned) row count (plain NZ)
    repeat_stride = 1
    row_stride = src.shape[1]    # ND row-major: contiguous cols (RowStride)
    src_ptr = src.as_ptr()
    dst_ptr = dst.as_ptr()
    lanes = pto.elements_per_vreg(dtype)
    # Static column count for the single/multi-column branch decision (the
    # valid_shape is a dynamic SSA value, not a Python int, so it cannot
    # drive a trace-time Python `if`). For ND->NZ the valid cols == physical
    # cols (cols are never padded, only rows can be — RowPlusOne), so the
    # static dst physical cols is the right count to branch on.
    static_cols = dst.shape[1]

    if pto.const_expr(static_cols <= lanes):
        # Single-column-repeat: one row scf.for (cfgVsstbLast not needed — no
        # following column-repeat to reposition for). Mirrors pto-isa
        # TMovNd2NzLoopRepeat1. Fixed predicate: CreatePredicate<T>(validCol).
        # Full-VL data + full-VL mask when cols == lanes; count predicate (data
        # + mask both sized to cols) when cols < lanes. Either way data and mask
        # share the active-lane count so the block-strided store verifier holds.
        preg, _ = pto.make_mask(dtype, valid_cols)
        # Loop carries ONLY dst_ptr (vsstb POST_UPDATE auto-advances dst by
        # repeatStride each beat). src_ptr is loop-invariant: vlds uses an
        # explicit per-iteration offset = RowStride * i (NORM, no POST_UPDATE),
        # matching pto-isa TMovNd2NzLoopRepeat1.
        loop = pto.for_(0, valid_rows, step=1).carry(dst_ptr=dst_ptr)
        with loop:
            src_off = scalar.muli(loop.iv, pto.const(row_stride))
            vec = pto.vlds(src_ptr, src_off, dist="NORM")
            d_next = pto.vsstb(vec, loop.dst_ptr, block_stride, repeat_stride,
                               preg, post_update="ON")
            loop.update(dst_ptr=d_next)
        return

    # Multi-column-repeat (valid_cols > lanes): nested scf.for — outer j over
    # column-block groups (repeatTimes = ceil(cols/lanes)) + inner row i loop +
    # a trailing cfgVsstbLast beat per group. Mirrors pto-isa TMovNd2NzLoop.
    #
    # Both loops use the explicit pto.for_(...).carry(...) form (NOT `for x in
    # range(...)`, which the AST rewriter would convert and cannot carry a
    # last-iteration dst_ptr across the outer loop). The outer loop carries
    # dst_ptr across column-block groups and a runtime `remained` count for the
    # per-group predicate: count = min(remained, lanes), recomputed each group
    # (full VL except the final partial group when cols % lanes != 0). src_ptr
    # is loop-invariant; each beat's src offset = i*row_stride + j*lanes.
    repeat_times = -(-static_cols // lanes)  # ceil(cols / lanes), Python int
    static_rows = dst.shape[0]  # physical rows (plain NZ); valid rows == for plain
    inner_loop_num = static_rows - 1
    virtual_row = dst.shape[0]  # plain NZ = aligned rows; RowPlusOne = aligned+1
    repeat_stride_last = _nd2nz_repeat_stride_last(virtual_row, inner_loop_num)
    lanes_const = pto.const(lanes)
    inner_num_const = pto.const(inner_loop_num)
    row_stride_const = pto.const(row_stride)

    # Outer loop over column-block groups: carries dst_ptr (advanced by each
    # group's trailing cfgVsstbLast beat) and remained (cols still to process).
    # `remained` starts at the static cols count (a Python int -> pto.const),
    # then decrements by `lanes` each group; the per-group count predicate is
    # min(remained, lanes) (full VL except the final partial group).
    outer = pto.for_(0, repeat_times, step=1).carry(dst_ptr=dst_ptr,
                                                    remained=pto.const(static_cols))
    with outer:
        # Per-group runtime predicate: count = min(remained, lanes). Full-VL
        # groups (remained >= lanes) get a full mask; the final partial group
        # (remained < lanes) gets a count predicate sized to the remainder.
        count = scalar.min(outer.remained, lanes_const)
        preg, _ = pto.make_mask(dtype, count)
        col_block_off = scalar.muli(outer.iv, lanes_const)
        # Inner row loop: innerLoopNum beats with cfgVsstb (repeat_stride=1).
        # valid_rows >= FRACTAL_NZ_ROW (16) for a legal NZ dst, so innerLoopNum
        # = valid_rows - 1 >= 15 > 0 always — no need to guard the zero case.
        inner = pto.for_(0, inner_loop_num, step=1).carry(dst_ptr=outer.dst_ptr)
        with inner:
            row_off = scalar.muli(inner.iv, row_stride_const)
            src_off = scalar.addi(row_off, col_block_off)
            vec = pto.vlds(src_ptr, src_off, dist="NORM")
            d_next = pto.vsstb(vec, inner.dst_ptr, block_stride, repeat_stride,
                               preg, post_update="ON")
            inner.update(dst_ptr=d_next)
        group_dst = inner.final("dst_ptr")
        # Trailing beat for this group: cfgVsstbLast (repeat_stride_last jumps
        # dst to the next column-block group's head). Row index = innerLoopNum.
        last_row_off = scalar.muli(inner_num_const, row_stride_const)
        last_src_off = scalar.addi(last_row_off, col_block_off)
        last_vec = pto.vlds(src_ptr, last_src_off, dist="NORM")
        next_dst = pto.vsstb(last_vec, group_dst, block_stride,
                             repeat_stride_last, preg, post_update="ON")
        next_remained = outer.remained - lanes_const
        outer.update(dst_ptr=next_dst, remained=next_remained)
