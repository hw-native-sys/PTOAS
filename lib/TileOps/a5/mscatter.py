# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""PTODSL TileLib templates for pto.mscatter."""

from ptodsl import pto, scalar
import ptodsl.tilelib as tilelib
from ._common import NUMERIC_DTYPES


SCATTER_DTYPES = NUMERIC_DTYPES
INDEX_DTYPES = ("i32", "ui32")
SCATTER_ATOMIC_ADD_DTYPES = ("i32", "ui32", "f16", "f32")
SCATTER_ATOMIC_MINMAX_DTYPES = ("i32", "ui32", "f32")


def mscatter_dtype_signatures():
    return [
        (dtype, idx_dtype, dtype)
        for dtype in SCATTER_DTYPES
        for idx_dtype in INDEX_DTYPES
    ]


def _ub_or_vec_row_major(
    src_memory_space, src_config,
    indices_memory_space, indices_config,
    table_kind, table_memory_space,
    **_
):
    return (
        src_memory_space in {"ub", "vec"}
        and src_config.b_layout == "row_major"
        and src_config.s_layout == "none_box"
        and indices_memory_space in {"ub", "vec"}
        and indices_config.b_layout == "row_major"
        and indices_config.s_layout == "none_box"
        and table_kind == "view"
        and table_memory_space == "gm"
    )


def _is_unknown_dim(value):
    return value is None or value in {-1, -(2**63)}


def _row_mode_constraint(src_valid_shape=(), indices_valid_shape=(), table_shape=(), **_):
    if len(src_valid_shape) != 2 or len(indices_valid_shape) != 2:
        return False
    src_rows, src_cols = src_valid_shape
    idx_rows, idx_cols = indices_valid_shape
    if not (
        _is_unknown_dim(src_rows)
        or _is_unknown_dim(idx_rows)
        or _is_unknown_dim(idx_cols)
    ):
        if not ((idx_rows == src_rows and idx_cols == 1) or (
            idx_rows == 1 and idx_cols == src_rows)):
            return False
    if table_shape and not _is_unknown_dim(src_cols):
        table_cols = table_shape[-1]
        if not _is_unknown_dim(table_cols) and table_cols != src_cols:
            return False
    return True


def _elem_mode_constraint(src_valid_shape=(), indices_valid_shape=(), **_):
    if len(src_valid_shape) != 2 or len(indices_valid_shape) != 2:
        return False
    src_rows, src_cols = src_valid_shape
    idx_rows, idx_cols = indices_valid_shape
    rows_match = (
        src_rows == idx_rows or _is_unknown_dim(src_rows) or _is_unknown_dim(idx_rows)
    )
    cols_match = (
        src_cols == idx_cols or _is_unknown_dim(src_cols) or _is_unknown_dim(idx_cols)
    )
    return rows_match and cols_match


def _no_atomic_constraint(scatter_atomic_op=None, **_):
    """Constraint that matches when no atomic operation is specified."""
    return scatter_atomic_op is None or scatter_atomic_op == "none"


def _atomic_add_constraint(scatter_atomic_op=None, **_):
    """Constraint that matches when atomic add operation is specified."""
    return scatter_atomic_op == "add"


def _atomic_max_constraint(scatter_atomic_op=None, **_):
    """Constraint that matches when atomic max operation is specified."""
    return scatter_atomic_op == "max"


def _atomic_min_constraint(scatter_atomic_op=None, **_):
    """Constraint that matches when atomic min operation is specified."""
    return scatter_atomic_op == "min"


def _no_conflict_constraint(scatter_conflict=None, **_):
    """Constraint that matches when no conflict mode or default is specified."""
    return scatter_conflict is None or scatter_conflict == "default"


def _conflict_last_constraint(scatter_conflict=None, **_):
    """Constraint that matches when conflict=last is specified."""
    return scatter_conflict == "last"


def _scatter_remap(raw_idx, cap, oob):
    zero_idx = pto.const(0, dtype=raw_idx.dtype)
    if oob == "undefined":
        return raw_idx, None
    if oob == "skip":
        return raw_idx, (raw_idx >= zero_idx) & (raw_idx < cap)
    if oob == "clamp":
        with pto.if_(raw_idx < zero_idx) as br_lo:
            with br_lo.then_:
                br_lo.assign(safe=zero_idx)
            with br_lo.else_:
                with pto.if_(raw_idx >= cap) as br_hi:
                    with br_hi.then_:
                        br_hi.assign(inner=cap - 1)
                    with br_hi.else_:
                        br_hi.assign(inner=raw_idx)
                br_lo.assign(safe=br_hi.inner)
        return br_lo.safe, None
    safe = raw_idx % cap
    with pto.if_(safe < zero_idx) as br:
        with br.then_:
            br.assign(adj=safe + cap)
        with br.else_:
            br.assign(adj=safe)
    return br.adj, None


def _load_row_idx(idx_ptr, idx_cols, row):
    if idx_cols == 1:
        return scalar.load(pto.addptr(idx_ptr, row * idx_cols))
    return scalar.load(pto.addptr(idx_ptr, row))


# SIMT helper for atomic row scatter
@pto.simt
def _simt_atomic_row_scatter(
    table_ptr, src_ptr, idx_ptr,
    valid_rows, valid_cols, table_rows, src_cols, idx_cols, table_row_stride,
    atomic_op_type: pto.const_expr,
    oob_mode: pto.const_expr,
):
    tx = pto.get_tid_x()
    ty = pto.get_tid_y()

    for r in range(ty, valid_rows, 8):
        if idx_cols == 1:
            idx_addr = pto.addptr(idx_ptr, r * idx_cols)
        else:
            idx_addr = pto.addptr(idx_ptr, r)
        raw_idx = scalar.load(idx_addr, 0)

        if oob_mode == "undefined":
            safe_idx = raw_idx
            do_write = True
        elif oob_mode == "skip":
            safe_idx = raw_idx
            do_write = (raw_idx >= 0) & (raw_idx < table_rows)
        elif oob_mode == "clamp":
            clamped = scalar.select(raw_idx < 0, 0, raw_idx)
            safe_idx = scalar.select(clamped >= table_rows, table_rows - 1, clamped)
            do_write = True
        else:
            mod_idx = raw_idx % table_rows
            safe_idx = scalar.select(mod_idx < 0, mod_idx + table_rows, mod_idx)
            do_write = True

        if do_write:
            dst_base = pto.addptr(table_ptr, safe_idx * table_row_stride)

            for col in range(tx, valid_cols, 32):
                src_addr = pto.addptr(src_ptr, r * src_cols + col)
                value = scalar.load(src_addr, 0)
                dst_addr = pto.addptr(dst_base, col)

                if atomic_op_type == "add":
                    pto.atomic_add(dst_addr, value)
                elif atomic_op_type == "max":
                    pto.atomic_max(dst_addr, value)
                elif atomic_op_type == "min":
                    pto.atomic_min(dst_addr, value)


def _check_last_row_writer(row, idx_ptr, idx_cols, cur_safe, valid_rows, table_rows, oob, cur_do_write=None):
    one = pto.const(1, dtype=pto.i1)
    zero = pto.const(0, dtype=pto.i1)
    check_loop = pto.for_(row + 1, valid_rows, step=1).carry(is_last=one)
    with check_loop:
        later = check_loop.iv
        cur_is_last = check_loop.is_last
        later_raw = _load_row_idx(idx_ptr, idx_cols, later)
        later_safe, later_do_write = _scatter_remap(later_raw, table_rows, oob)
        same_slot = (cur_safe == later_safe)
        if oob == "skip":
            will_override = same_slot & later_do_write
        else:
            will_override = same_slot
        with pto.if_(will_override) as br:
            with br.then_:
                br.assign(new_is_last=zero)
            with br.else_:
                br.assign(new_is_last=cur_is_last)
        check_loop.update(is_last=br.new_is_last)
    result = check_loop.final("is_last")
    if cur_do_write is not None:
        result = result & cur_do_write
    return result


def _check_last_elem_writer(row, col, idx_ptr, idx_cols, cur_safe, valid_rows, valid_cols, table_size, oob, cur_do_write=None):
    total_elems = valid_rows * valid_cols
    flat_i = row * valid_cols + col
    one = pto.const(1, dtype=pto.i1)
    zero = pto.const(0, dtype=pto.i1)
    check_loop = pto.for_(flat_i + 1, total_elems, step=1).carry(is_last=one)
    with check_loop:
        j = check_loop.iv
        cur_is_last = check_loop.is_last
        later_r = j // valid_cols
        later_c = j - later_r * valid_cols
        later_raw = scalar.load(pto.addptr(idx_ptr, later_r * idx_cols + later_c))
        later_safe, later_do_write = _scatter_remap(later_raw, table_size, oob)
        same_slot = (cur_safe == later_safe)
        if oob == "skip":
            will_override = same_slot & later_do_write
        else:
            will_override = same_slot
        with pto.if_(will_override) as br:
            with br.then_:
                br.assign(new_is_last=zero)
            with br.else_:
                br.assign(new_is_last=cur_is_last)
        check_loop.update(is_last=br.new_is_last)
    result = check_loop.final("is_last")
    if cur_do_write is not None:
        result = result & cur_do_write
    return result


def _resolve_table_row_stride(table, fallback_cols):
    strides = getattr(table, "strides", None)
    if strides is not None and len(strides) >= 2:
        s = strides[-2]
        if s is not None:
            return s
    return fallback_cols


@pto.jit(entry=False)
def _emit_row_scatter_body(src, indices, table, atomic_op=None,
                           conflict_mode="default"):
    dtype = src.element_type
    src_ptr = src.as_ptr()
    idx_ptr = indices.as_ptr()
    table_ptr = table.as_ptr()
    valid_rows, valid_cols = src.valid_shape
    src_cols = src.shape[1]
    idx_cols = indices.shape[1]

    oob = pto.get_op_attr("scatter_oob", "undefined")
    table_rows = table.shape[-2]
    table_row_stride = _resolve_table_row_stride(table, valid_cols)

    if atomic_op is not None:
        if atomic_op == _atomic_add_op:
            atomic_type = "add"
        elif atomic_op == _atomic_max_op:
            atomic_type = "max"
        elif atomic_op == _atomic_min_op:
            atomic_type = "min"
        else:
            raise ValueError(f"Unknown atomic operation: {atomic_op}")

        table_rows_val = pto.const(table_rows, dtype=pto.index)
        src_cols_val = pto.const(src_cols, dtype=pto.index)
        idx_cols_val = pto.const(idx_cols, dtype=pto.index)
        stride_val = pto.const(table_row_stride, dtype=pto.index)

        pto.simt_launch(
            _simt_atomic_row_scatter,
            table_ptr, src_ptr, idx_ptr,
            valid_rows, valid_cols, table_rows_val, src_cols_val, idx_cols_val,
            stride_val,
            atomic_op_type=atomic_type,
            oob_mode=oob,
            dims=(32, 8, 1)
        )
    else:
        pto.set_flag("V", "S", event_id=0)
        pto.wait_flag("V", "S", event_id=0)

        for row in range(valid_rows):
            raw_idx = _load_row_idx(idx_ptr, idx_cols, row)
            safe_idx, write_cond = _scatter_remap(raw_idx, table_rows, oob)
            dst_addr = pto.addptr(table_ptr, safe_idx * table_row_stride)

            if conflict_mode == "last":
                is_last = _check_last_row_writer(
                    row, idx_ptr, idx_cols, safe_idx, valid_rows, table_rows, oob,
                    cur_do_write=write_cond)
                should_write = is_last
            else:
                should_write = write_cond if write_cond is not None else pto.const(1, dtype=pto.i1)

            if should_write:
                for col in range(valid_cols):
                    src_addr = pto.addptr(src_ptr, row * src_cols + col)
                    value = scalar.load(src_addr)
                    pto.store_scalar(dst_addr, col, value)

        pto.set_flag("S", "V", event_id=0)
        pto.wait_flag("S", "V", event_id=0)


@pto.jit(entry=False)
def _emit_elem_scatter_body(src, indices, table, atomic_op=None,
                            conflict_mode="default"):
    src_ptr = src.as_ptr()
    idx_ptr = indices.as_ptr()
    table_ptr = table.as_ptr()
    valid_rows, valid_cols = src.valid_shape
    src_cols = src.shape[1]
    idx_cols = indices.shape[1]

    oob = pto.get_op_attr("scatter_oob", "undefined")
    table_size = 1
    for d in table.shape:
        table_size *= d

    pto.set_flag("V", "S", event_id=0)
    pto.wait_flag("V", "S", event_id=0)

    for row in range(valid_rows):
        for col in range(valid_cols):
            idx_addr = pto.addptr(idx_ptr, row * idx_cols + col)
            raw_idx = scalar.load(idx_addr)
            src_addr = pto.addptr(src_ptr, row * src_cols + col)
            value = scalar.load(src_addr)

            safe_idx, write_cond = _scatter_remap(raw_idx, table_size, oob)
            dst_addr = pto.addptr(table_ptr, safe_idx)

            if atomic_op is None:
                if conflict_mode == "last":
                    is_last = _check_last_elem_writer(
                        row, col, idx_ptr, idx_cols, safe_idx,
                        valid_rows, valid_cols, table_size, oob,
                        cur_do_write=write_cond)
                    should_write = is_last
                else:
                    should_write = write_cond if write_cond is not None else pto.const(1, dtype=pto.i1)
                if should_write:
                    pto.store_scalar(dst_addr, 0, value)
            else:
                if write_cond is not None:
                    if write_cond:
                        atomic_op(dst_addr, value)
                else:
                    atomic_op(dst_addr, value)

    pto.set_flag("S", "V", event_id=0)
    pto.wait_flag("S", "V", event_id=0)


def _atomic_add_op(ptr, value):
    pto.atomic_add(ptr, value)


def _atomic_max_op(ptr, value):
    pto.atomic_max(ptr, value)


def _atomic_min_op(ptr, value):
    pto.atomic_min(ptr, value)


@tilelib.tile_template(
    op="pto.mscatter",
    target="a5",
    name="template_mscatter_row",
    dtypes=mscatter_dtype_signatures(),
    iteration_axis="none",
    op_engine="vector",
    op_class="other",
    constraints=[_ub_or_vec_row_major, _row_mode_constraint, _no_atomic_constraint, _no_conflict_constraint],
    id=0,
    loop_depth=2,
    is_post_update=False,
    tags=("scatter", "row"),
)
def template_mscatter_row(
    src: pto.Tile, indices: pto.Tile, table: pto.PartitionTensorView
):
    _emit_row_scatter_body(src, indices, table)


@tilelib.tile_template(
    op="pto.mscatter",
    target="a5",
    name="template_mscatter_elem",
    dtypes=mscatter_dtype_signatures(),
    iteration_axis="none",
    op_engine="vector",
    op_class="other",
    constraints=[_ub_or_vec_row_major, _elem_mode_constraint, _no_atomic_constraint, _no_conflict_constraint],
    id=1,
    loop_depth=2,
    is_post_update=False,
    tags=("scatter", "elem"),
)
def template_mscatter_elem(
    src: pto.Tile, indices: pto.Tile, table: pto.PartitionTensorView
):
    _emit_elem_scatter_body(src, indices, table)


@tilelib.tile_template(
    op="pto.mscatter",
    target="a5",
    name="template_mscatter_row_atomic_add",
    dtypes=[
        (dtype, idx_dtype, dtype)
        for dtype in SCATTER_ATOMIC_ADD_DTYPES
        for idx_dtype in INDEX_DTYPES
    ],
    iteration_axis="none",
    op_engine="vector",
    op_class="other",
    constraints=[_ub_or_vec_row_major, _row_mode_constraint, _atomic_add_constraint, _no_conflict_constraint],
    id=2,
    loop_depth=2,
    is_post_update=False,
    tags=("scatter", "row", "atomic_add"),
)
def template_mscatter_row_atomic_add(
    src: pto.Tile, indices: pto.Tile, table: pto.PartitionTensorView
):
    _emit_row_scatter_body(src, indices, table, atomic_op=_atomic_add_op)


@tilelib.tile_template(
    op="pto.mscatter",
    target="a5",
    name="template_mscatter_elem_atomic_add",
    dtypes=[
        (dtype, idx_dtype, dtype)
        for dtype in SCATTER_ATOMIC_ADD_DTYPES
        for idx_dtype in INDEX_DTYPES
    ],
    iteration_axis="none",
    op_engine="vector",
    op_class="other",
    constraints=[_ub_or_vec_row_major, _elem_mode_constraint, _atomic_add_constraint, _no_conflict_constraint],
    id=3,
    loop_depth=2,
    is_post_update=False,
    tags=("scatter", "elem", "atomic_add"),
)
def template_mscatter_elem_atomic_add(
    src: pto.Tile, indices: pto.Tile, table: pto.PartitionTensorView
):
    _emit_elem_scatter_body(src, indices, table, atomic_op=_atomic_add_op)


@tilelib.tile_template(
    op="pto.mscatter",
    target="a5",
    name="template_mscatter_row_atomic_max",
    dtypes=[
        (dtype, idx_dtype, dtype)
        for dtype in SCATTER_ATOMIC_MINMAX_DTYPES
        for idx_dtype in INDEX_DTYPES
    ],
    iteration_axis="none",
    op_engine="vector",
    op_class="other",
    constraints=[_ub_or_vec_row_major, _row_mode_constraint, _atomic_max_constraint, _no_conflict_constraint],
    id=4,
    loop_depth=2,
    is_post_update=False,
    tags=("scatter", "row", "atomic_max"),
)
def template_mscatter_row_atomic_max(
    src: pto.Tile, indices: pto.Tile, table: pto.PartitionTensorView
):
    _emit_row_scatter_body(src, indices, table, atomic_op=_atomic_max_op)


@tilelib.tile_template(
    op="pto.mscatter",
    target="a5",
    name="template_mscatter_elem_atomic_max",
    dtypes=[
        (dtype, idx_dtype, dtype)
        for dtype in SCATTER_ATOMIC_MINMAX_DTYPES
        for idx_dtype in INDEX_DTYPES
    ],
    iteration_axis="none",
    op_engine="vector",
    op_class="other",
    constraints=[_ub_or_vec_row_major, _elem_mode_constraint, _atomic_max_constraint, _no_conflict_constraint],
    id=5,
    loop_depth=2,
    is_post_update=False,
    tags=("scatter", "elem", "atomic_max"),
)
def template_mscatter_elem_atomic_max(
    src: pto.Tile, indices: pto.Tile, table: pto.PartitionTensorView
):
    _emit_elem_scatter_body(src, indices, table, atomic_op=_atomic_max_op)


@tilelib.tile_template(
    op="pto.mscatter",
    target="a5",
    name="template_mscatter_row_atomic_min",
    dtypes=[
        (dtype, idx_dtype, dtype)
        for dtype in SCATTER_ATOMIC_MINMAX_DTYPES
        for idx_dtype in INDEX_DTYPES
    ],
    iteration_axis="none",
    op_engine="vector",
    op_class="other",
    constraints=[_ub_or_vec_row_major, _row_mode_constraint, _atomic_min_constraint, _no_conflict_constraint],
    id=6,
    loop_depth=2,
    is_post_update=False,
    tags=("scatter", "row", "atomic_min"),
)
def template_mscatter_row_atomic_min(
    src: pto.Tile, indices: pto.Tile, table: pto.PartitionTensorView
):
    _emit_row_scatter_body(src, indices, table, atomic_op=_atomic_min_op)


@tilelib.tile_template(
    op="pto.mscatter",
    target="a5",
    name="template_mscatter_elem_atomic_min",
    dtypes=[
        (dtype, idx_dtype, dtype)
        for dtype in SCATTER_ATOMIC_MINMAX_DTYPES
        for idx_dtype in INDEX_DTYPES
    ],
    iteration_axis="none",
    op_engine="vector",
    op_class="other",
    constraints=[_ub_or_vec_row_major, _elem_mode_constraint, _atomic_min_constraint, _no_conflict_constraint],
    id=7,
    loop_depth=2,
    is_post_update=False,
    tags=("scatter", "elem", "atomic_min"),
)
def template_mscatter_elem_atomic_min(
    src: pto.Tile, indices: pto.Tile, table: pto.PartitionTensorView
):
    _emit_elem_scatter_body(src, indices, table, atomic_op=_atomic_min_op)


@tilelib.tile_template(
    op="pto.mscatter",
    target="a5",
    name="template_mscatter_row_conflict_last",
    dtypes=mscatter_dtype_signatures(),
    iteration_axis="none",
    op_engine="vector",
    op_class="other",
    constraints=[_ub_or_vec_row_major, _row_mode_constraint, _no_atomic_constraint, _conflict_last_constraint],
    id=8,
    loop_depth=2,
    is_post_update=False,
    tags=("scatter", "row", "conflict_last"),
)
def template_mscatter_row_conflict_last(
    src: pto.Tile, indices: pto.Tile, table: pto.PartitionTensorView
):
    _emit_row_scatter_body(src, indices, table, conflict_mode="last")


@tilelib.tile_template(
    op="pto.mscatter",
    target="a5",
    name="template_mscatter_elem_conflict_last",
    dtypes=mscatter_dtype_signatures(),
    iteration_axis="none",
    op_engine="vector",
    op_class="other",
    constraints=[_ub_or_vec_row_major, _elem_mode_constraint, _no_atomic_constraint, _conflict_last_constraint],
    id=9,
    loop_depth=2,
    is_post_update=False,
    tags=("scatter", "elem", "conflict_last"),
)
def template_mscatter_elem_conflict_last(
    src: pto.Tile, indices: pto.Tile, table: pto.PartitionTensorView
):
    _emit_elem_scatter_body(src, indices, table, conflict_mode="last")
