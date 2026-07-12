# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""TileLang DSL template for `pto.tload`"""

import tilelang_dsl as pto


def _constraint_scalar(value):
    return value.value if hasattr(value, "value") else value


def _known_eq(lhs, rhs) -> bool:
    lhs_value = _constraint_scalar(lhs)
    rhs_value = _constraint_scalar(rhs)
    if lhs_value is None or rhs_value is None:
        return True
    return lhs_value == rhs_value


def _known_le(lhs, rhs) -> bool:
    lhs_value = _constraint_scalar(lhs)
    rhs_value = _constraint_scalar(rhs)
    if lhs_value is None or rhs_value is None:
        return True
    return lhs_value <= rhs_value


def _match_tile_layout(dst, *, row_major: bool, s_layout) -> bool:
    b_layout_ok = (
        dst.config.b_layout == pto.BLayout.ROW_MAJOR
        if row_major
        else dst.config.b_layout != pto.BLayout.ROW_MAJOR
    )
    return b_layout_ok and dst.config.s_layout == s_layout


def _check_load_bounds(src, dst, *, logical_rows, logical_cols=None, stride_axis=None) -> bool:
    if src.rank != 5:
        return False
    if stride_axis is not None and not _known_eq(src.strides[stride_axis], 1):
        return False
    if not _known_le(dst.valid_shape[0], logical_rows):
        return False
    if not _known_le(logical_rows, dst.shape[0]):
        return False
    if not _known_le(dst.valid_shape[0], dst.shape[0]):
        return False
    if logical_cols is not None:
        if not _known_le(dst.valid_shape[1], logical_cols):
            return False
        if not _known_le(logical_cols, dst.shape[1]):
            return False
    if not _known_le(dst.valid_shape[1], dst.shape[1]):
        return False
    return True


def _tload_preconditions_nd2nd(src, dst) -> bool:
    logical_rows = src.shape[0] * src.shape[1] * src.shape[2] * src.shape[3]
    logical_cols = src.shape[4]
    return _match_tile_layout(
        dst, row_major=True, s_layout=pto.SLayout.NONE_BOX
    ) and _check_load_bounds(
        src, dst, logical_rows=logical_rows, logical_cols=logical_cols, stride_axis=4
    )


def _tload_preconditions_dn2dn(src, dst) -> bool:
    logical_rows = src.shape[3]
    logical_cols = src.shape[0] * src.shape[1] * src.shape[2] * src.shape[4]
    return _match_tile_layout(
        dst, row_major=False, s_layout=pto.SLayout.NONE_BOX
    ) and _check_load_bounds(
        src, dst, logical_rows=logical_rows, logical_cols=logical_cols, stride_axis=3
    )

def _tload_preconditions_nz2nz(src, dst) -> bool:
    logical_rows = src.shape[2]
    return _match_tile_layout(
        dst, row_major=False, s_layout=pto.SLayout.ROW_MAJOR
    ) and _check_load_bounds(
        src, dst, logical_rows=logical_rows
    )


@pto.vkernel(
    target="a5",
    op="pto.tload",
    advanced=True,
    constraints=[_tload_preconditions_nd2nd],
)
def template_tload_nd2nd(src: pto.PartitionTensorView, dst: pto.Tile):
    dtype = dst.element_type
    elem_bytes = pto.bytewidth(dtype)
    if pto.constexpr(dst.pad_value != pto.PadValue.NULL):
        pto.set_mov_pad_val(dst.pad_value.eval())

    g0, g1, g2, g3, g4 = src.shape
    s0, s1, s2, s3, s4 = src.strides

    valid_rows, valid_cols = dst.valid_shape
    ub_rows, ub_cols = dst.shape

    n_burst = g3
    len_burst = g4 * elem_bytes
    gm_stride = s3 * elem_bytes
    ub_stride = ub_cols * elem_bytes

    dst_stride2 = g3 * ub_cols
    dst_stride1 = g2 * dst_stride2
    dst_stride0 = g1 * dst_stride1

    loop1 = g2
    loop2 = g1
    loop1_src_stride = s2 * elem_bytes
    loop1_dst_stride = dst_stride2 * elem_bytes
    loop2_src_stride = s1 * elem_bytes
    loop2_dst_stride = dst_stride1 * elem_bytes

    gm_ptr = src.as_ptr()
    ub_ptr = dst.as_ptr()

    if loop1 != 1 or loop2 != 1:
        pto.set_loop2_stride_outtoub(
            src_stride=loop2_src_stride, dst_stride=loop2_dst_stride
        )
        pto.set_loop1_stride_outtoub(
            src_stride=loop1_src_stride, dst_stride=loop1_dst_stride
        )
        pto.set_loop_size_outtoub(loop1=loop1, loop2=loop2)

    for i in range(0, g0, 1):
        src_i = pto.addptr(gm_ptr, i * s0)
        dst_i = pto.addptr(ub_ptr, i * dst_stride0)
        if pto.constexpr(dst.pad_value != pto.PadValue.NULL):
            pto.copy_gm_to_ubuf(
                dst=dst_i,
                src=src_i,
                n_burst=n_burst,
                len_burst=len_burst,
                gm_stride=gm_stride,
                ub_stride=ub_stride,
                enable_ub_pad=True,
            )
        else:
            pto.copy_gm_to_ubuf(
                dst=dst_i,
                src=src_i,
                n_burst=n_burst,
                len_burst=len_burst,
                gm_stride=gm_stride,
                ub_stride=ub_stride,
                enable_ub_pad=False,
            )

    if loop1 != 1 or loop2 != 1:
        pto.set_loop_size_outtoub(loop1=1, loop2=1)
    return

@pto.vkernel(
    target="a5",
    op="pto.tload",
    advanced=True,
    constraints=[_tload_preconditions_dn2dn],
)
def template_tload_dn2dn(src: pto.PartitionTensorView, dst: pto.Tile):
    dtype = dst.element_type
    elem_bytes = pto.bytewidth(dtype)
    if pto.constexpr(dst.pad_value != pto.PadValue.NULL):
        pto.set_mov_pad_val(dst.pad_value.eval())

    # rank-5 partition view metadata.
    g0, g1, g2, g3, g4 = src.shape
    s0, s1, s2, s3, s4 = src.strides

    tile_rows, tile_cols = dst.shape
    valid_rows, valid_cols = dst.valid_shape

    n_burst = g4
    len_burst = valid_rows * elem_bytes
    gm_stride = s4 * elem_bytes
    ub_stride = tile_rows * elem_bytes

    # The UB destination tile has a compact col-major layout with column
    # height `tile_rows`. From the innermost `g4 × tile_rows` block, three
    # levels of stride are derived recursively.
    dst_stride2 = g4 * tile_rows
    dst_stride1 = g2 * dst_stride2
    dst_stride0 = g1 * dst_stride1

    # loop1 <-> g2 (inner), loop2 <-> g1 (outer), software for <-> g0.
    loop1 = g2
    loop2 = g1
    loop1_src_stride = s2 * elem_bytes
    loop1_dst_stride = dst_stride2 * elem_bytes
    loop2_src_stride = s1 * elem_bytes
    loop2_dst_stride = dst_stride1 * elem_bytes

    gm_ptr = src.as_ptr()
    ub_ptr = dst.as_ptr()

    if loop1 != 1 or loop2 != 1:
        pto.set_loop2_stride_outtoub(
            src_stride=loop2_src_stride, dst_stride=loop2_dst_stride
        )
        pto.set_loop1_stride_outtoub(
            src_stride=loop1_src_stride, dst_stride=loop1_dst_stride
        )
        pto.set_loop_size_outtoub(loop1=loop1, loop2=loop2)

    for i in range(0, g0, 1):
        src_i = pto.addptr(gm_ptr, i * s0)
        dst_i = pto.addptr(ub_ptr, i * dst_stride0)
        if pto.constexpr(dst.pad_value != pto.PadValue.NULL):
            pto.copy_gm_to_ubuf(
                dst=dst_i,
                src=src_i,
                n_burst=n_burst,
                len_burst=len_burst,
                gm_stride=gm_stride,
                ub_stride=ub_stride,
                enable_ub_pad=True,
            )
        else:
            pto.copy_gm_to_ubuf(
                dst=dst_i,
                src=src_i,
                n_burst=n_burst,
                len_burst=len_burst,
                gm_stride=gm_stride,
                ub_stride=ub_stride,
                enable_ub_pad=False,
            )

    if loop1 != 1 or loop2 != 1:
        pto.set_loop_size_outtoub(loop1=1, loop2=1)
    return

@pto.vkernel(
    target="a5",
    op="pto.tload",
    advanced=True,
    constraints=[_tload_preconditions_nz2nz],
)
def template_tload_nz2nz(src: pto.PartitionTensorView, dst: pto.Tile):
    dtype = dst.element_type
    elem_bytes = pto.bytewidth(dtype)

    if pto.constexpr(dst.pad_value != pto.PadValue.NULL):
        pto.set_mov_pad_val(dst.pad_value.eval())

    # rank-5 partition view metadata. NZ static tile constraints (g3/g4 vs
    # dtype relationship) are enforced by higher-layer schema / static checks;
    # here we only keep the runtime DMA formula.
    g0, g1, g2, g3, g4 = src.shape
    s0, s1, s2, s3, s4 = src.strides

    tile_rows, tile_cols = dst.shape
    valid_rows, valid_cols = dst.valid_shape

    c0_size_bytes = 32
    n_burst = g1
    len_burst = valid_rows * c0_size_bytes
    gm_stride = s1 * elem_bytes
    ub_stride = tile_rows * c0_size_bytes

    # Each g0 block in UB contains `g1` NZ sub-blocks; each sub-block has `g4` columns.
    tile_stride = g1 * tile_rows * g4

    gm_ptr = src.as_ptr()
    ub_ptr = dst.as_ptr()

    # NZ2NZ always uses normal mode; do not reuse loop1/loop2 registers.
    pto.set_loop_size_outtoub(loop1=1, loop2=1)
    for i in range(0, g0, 1):
        src_i = pto.addptr(gm_ptr, i * s0)
        dst_i = pto.addptr(ub_ptr, i * tile_stride)
        if pto.constexpr(dst.pad_value != pto.PadValue.NULL):
            pto.copy_gm_to_ubuf(
                dst=dst_i,
                src=src_i,
                n_burst=n_burst,
                len_burst=len_burst,
                gm_stride=gm_stride,
                ub_stride=ub_stride,
                enable_ub_pad=True,
            )
        else:
            pto.copy_gm_to_ubuf(
                dst=dst_i,
                src=src_i,
                n_burst=n_burst,
                len_burst=len_burst,
                gm_stride=gm_stride,
                ub_stride=ub_stride,
                enable_ub_pad=False,
            )
    return


# ============================================================================
# Cube Matrix Templates: TLOAD.MAT (GM → L1)
# ============================================================================

def _constraint_tload_mat_base(src, dst) -> bool:
    """TLOAD.MAT base constraint check"""
    # src must be GM
    src_space = src.memory_space
    src_space_value = "gm" if src_space is None else (src_space.value if hasattr(src_space, "value") else src_space)
    if src_space_value not in {"gm", "GM"}:
        return False
    # dst must be MemorySpace.MAT
    dst_space = dst.memory_space
    if dst_space is None:
        return False
    dst_space_value = dst_space.value if hasattr(dst_space, "value") else dst_space
    if dst_space_value not in {"mat", "MAT"}:
        return False
    # dst must be a 2D Tile
    if dst.rank != 2:
        return False
    # dtype check
    dst_dtype = dst.dtype
    if dst_dtype is None:
        return False
    dtype_name = dst_dtype.name if hasattr(dst_dtype, "name") else str(dst_dtype)
    supported_dtypes = {"f16", "bf16", "f32", "i8", "si8", "ui8", "i16", "si16", "ui16", "i32", "si32"}
    if dtype_name not in supported_dtypes:
        return False
    return True


def _constraint_tload_mat_nd2nz(src, dst) -> bool:
    """TLOAD.MAT ND2NZ fractal load constraint"""
    if not _constraint_tload_mat_base(src, dst):
        return False
    # dst layout must be col_major + row_major slayout (NZ format)
    config = dst.config
    if config is None:
        return False
    b_layout = config.b_layout
    s_layout = config.s_layout
    if b_layout is None or s_layout is None:
        return False
    b_layout_value = b_layout.value if hasattr(b_layout, "value") else b_layout
    s_layout_value = s_layout.value if hasattr(s_layout, "value") else s_layout
    # COL_MAJOR + ROW_MAJOR corresponds to NZ format
    if b_layout_value not in {"col_major", "COL_MAJOR"} or s_layout_value not in {"row_major", "ROW_MAJOR"}:
        return False
    # ND2NZ: source is in ND (row-major) format where the inner dimension (g4)
    # corresponds to the tile column count. Disambiguates from DN format where
    # g4 corresponds to the tile row count. Use the *static* dst.shape
    # (rows/cols from the tile_buf type) rather than dst.valid_shape, which
    # may be dynamic ([null, null]) and would otherwise let both ND2NZ and
    # DN2NZ match when the shape comparison is skipped.
    if not (hasattr(src, 'rank') and src.rank == 5):
        return False
    dst_cols = dst.shape[1] if hasattr(dst, 'shape') and dst.shape is not None else None
    if dst_cols is not None and hasattr(src, 'shape') and src.shape is not None:
        src_inner = src.shape[4] if len(src.shape) >= 5 else None
        if src_inner is not None:
            if not _known_eq(dst_cols, src_inner):
                return False
    return True


def _constraint_tload_mat_dn2nz(src, dst) -> bool:
    """TLOAD.MAT DN2NZ fractal load constraint"""
    if not _constraint_tload_mat_base(src, dst):
        return False
    config = dst.config
    if config is None:
        return False
    b_layout = config.b_layout
    s_layout = config.s_layout
    if b_layout is None or s_layout is None:
        return False
    b_layout_value = b_layout.value if hasattr(b_layout, "value") else b_layout
    s_layout_value = s_layout.value if hasattr(s_layout, "value") else s_layout
    if b_layout_value not in {"col_major", "COL_MAJOR"} or s_layout_value not in {"row_major", "ROW_MAJOR"}:
        return False
    # DN2NZ: source is in DN (col-major) format where the inner dimension (g4)
    # corresponds to the tile row count. Disambiguates from ND format where
    # g4 corresponds to the tile column count. Use the *static* dst.shape
    # (rows/cols from the tile_buf type) rather than dst.valid_shape, which
    # may be dynamic and would otherwise let both templates match.
    if not (hasattr(src, 'rank') and src.rank == 5):
        return False
    dst_rows = dst.shape[0] if hasattr(dst, 'shape') and dst.shape is not None else None
    if dst_rows is not None and hasattr(src, 'shape') and src.shape is not None:
        src_inner = src.shape[4] if len(src.shape) >= 5 else None
        if src_inner is not None:
            if not _known_eq(dst_rows, src_inner):
                return False
    return True


@pto.ckernel(
    target="a5",
    op="pto.tload",
    priority=1,
    dtypes=[
        (pto.f16, pto.f16),
        (pto.bf16, pto.bf16),
        (pto.f32, pto.f32),
    ],
    constraints=[_constraint_tload_mat_nd2nz],
    name="tload_gm_to_mat_nd2nz",
)
def template_tload_gm_to_mat_nd2nz(src: pto.PartitionTensorView, dst: pto.Tile):
    """GM -> MAT ND2NZ fractal load template

    Load Row-Major (ND) format data from GM into L1 MAT Buffer in NZ format.

    Args:
        src: PartitionTensorView with GM memory_space (row-major source)
        dst: Tile with MAT memory_space, shape=(M, K), col_major layout

    Uses:
        pto.mte_gm_l1_frac with mode=FractalMode.ND2NZ
    """
    m, k = dst.valid_shape

    gm_ptr = src.as_ptr()
    mat_ptr = dst.as_ptr()

    # Element byte width (src and dst share dtype per the template dtypes table).
    elem_bytes = pto.bytewidth(dst.element_type)

    # rank-5 partition view metadata (g3 = M rows, g4 = K cols for ND source).
    g0, g1, g2, g3, g4 = src.shape
    s0, s1, s2, s3, s4 = src.strides

    # ND2NZ parameter calculation
    # n_value = M (row count), d_value = K (column count)
    n_value = m
    d_value = k

    # src_layout: inner stride in BYTES = src physical row stride (s3).
    # Hardware loop1_src_stride is in bytes (pto-isa a5/TLoad.hpp:
    # GetByteSize<DType>(gStride3)); a strided GM slice has s3 != K, so the
    # element stride must be scaled by elem_bytes.
    src_inner_stride = s3 * elem_bytes

    # dst_group: (group_count, loop2_stride, loop3_stride, loop4_stride)
    # For simple single-block case: (1, 1, m, 0)

    # ctrl: (l2_cache_ctrl, smallc0_en)
    # (0, False) → l2_cache_ctrl=0, smallc0_en=False

    pto.mte_gm_l1_frac(
        gm_ptr, mat_ptr, pto.FractalMode.ND2NZ,
        shape=(n_value, d_value),
        src_layout=(src_inner_stride,),
        dst_group=(1, 1, m, 0),
        ctrl=(0, False)
    )
    return


@pto.ckernel(
    target="a5",
    op="pto.tload",
    priority=1,
    dtypes=[
        (pto.f16, pto.f16),
        (pto.bf16, pto.bf16),
        (pto.f32, pto.f32),
    ],
    constraints=[_constraint_tload_mat_dn2nz],
    name="tload_gm_to_mat_dn2nz",
)
def template_tload_gm_to_mat_dn2nz(src: pto.PartitionTensorView, dst: pto.Tile):
    """GM -> MAT DN2NZ fractal load template

    Load Col-Major (DN) format data from GM into L1 MAT Buffer in NZ format.
    The output is still logically N x D; only the memory layout changes to NZ
    (fractal). No logical shape conversion is needed.

    Args:
        src: PartitionTensorView with GM memory_space (col-major source)
        dst: Tile with MAT memory_space, shape=(M, K), col_major layout

    Uses:
        pto.mte_gm_l1_frac with mode=FractalMode.DN2NZ
    """
    m, k = dst.valid_shape

    gm_ptr = src.as_ptr()
    mat_ptr = dst.as_ptr()

    # Element byte width (src and dst share dtype per the template dtypes table).
    elem_bytes = pto.bytewidth(dst.element_type)

    # rank-5 partition view metadata (g3 = K, g4 = M for DN col-major source).
    g0, g1, g2, g3, g4 = src.shape
    s0, s1, s2, s3, s4 = src.strides

    # DN2NZ parameter calculation
    # For DN format, the original shape is (K, M) -- no logical conversion
    # needed. dn2nz writes the same logical N x D result into NZ layout.
    # n_value = K, d_value = M
    n_value = k
    d_value = m

    # src_layout: inner stride in BYTES = src physical stride along the M
    # direction (s4). pto-isa TLoadCubeND2NZ uses gStride4 when layout == DN.
    src_inner_stride = s4 * elem_bytes

    # dst_group: (group_count, loop2_stride, loop3_stride, loop4_stride)
    # (1, 1, k, 0)
    # ctrl: (l2_cache_ctrl, smallc0_en)
    # (0, False)

    pto.mte_gm_l1_frac(
        gm_ptr, mat_ptr, pto.FractalMode.DN2NZ,
        shape=(n_value, d_value),
        src_layout=(src_inner_stride,),
        dst_group=(1, 1, k, 0),
        ctrl=(0, False)
    )
    return
