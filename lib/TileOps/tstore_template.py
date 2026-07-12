# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""TileLang DSL template for `pto.tstore`"""

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


def _match_store_tile_layout(src, *, row_major: bool, s_layout) -> bool:
    b_layout_ok = (
        src.config.b_layout == pto.BLayout.ROW_MAJOR
        if row_major
        else src.config.b_layout != pto.BLayout.ROW_MAJOR
    )
    return b_layout_ok and src.config.s_layout == s_layout


def _check_store_bounds(src, dst, *, logical_rows, logical_cols, stride_axis=None) -> bool:
    if dst.rank != 5:
        return False
    if stride_axis is not None and not _known_eq(dst.strides[stride_axis], 1):
        return False
    if not _known_eq(src.valid_shape[0], logical_rows):
        return False
    if not _known_eq(src.valid_shape[1], logical_cols):
        return False
    if not _known_le(src.valid_shape[0], src.shape[0]):
        return False
    if not _known_le(src.valid_shape[1], src.shape[1]):
        return False
    return True


def _tstore_preconditions_nd(src, dst) -> bool:
    logical_rows = dst.shape[0] * dst.shape[1] * dst.shape[2] * dst.shape[3]
    logical_cols = dst.shape[4]
    return _match_store_tile_layout(
        src, row_major=True, s_layout=pto.SLayout.NONE_BOX
    ) and _check_store_bounds(
        src, dst, logical_rows=logical_rows, logical_cols=logical_cols, stride_axis=4
    )
    
def _tstore_preconditions_dn(src, dst) -> bool:
    logical_rows = dst.shape[3]
    logical_cols = dst.shape[0] * dst.shape[1] * dst.shape[2] * dst.shape[4]
    return _match_store_tile_layout(
        src, row_major=False, s_layout=pto.SLayout.NONE_BOX
    ) and _check_store_bounds(
        src, dst, logical_rows=logical_rows, logical_cols=logical_cols, stride_axis=3
    )

def _tstore_preconditions_nz(src, dst) -> bool:
    logical_rows = dst.shape[2] * dst.shape[3]
    logical_cols = dst.shape[0] * dst.shape[1] * dst.shape[4]
    return _match_store_tile_layout(
        src, row_major=False, s_layout=pto.SLayout.ROW_MAJOR
    ) and _check_store_bounds(
        src, dst, logical_rows=logical_rows, logical_cols=logical_cols
    )

@pto.vkernel(
    target="a5",
    op="pto.tstore",
    advanced=True,
    constraints=[_tstore_preconditions_nd],
)
def template_tstore_nd(src: pto.Tile, dst: pto.PartitionTensorView):
    dtype = src.element_type
    elem_bytes = pto.bytewidth(dtype)

    g0, g1, g2, g3, g4 = dst.shape
    s0, s1, s2, s3, s4 = dst.strides

    valid_rows, valid_cols = src.valid_shape
    ub_rows, ub_cols = src.shape

    # These preconditions are expressed through the descriptor-level constraint
    # callable above, using direct `src.*` / `dst.*` metadata syntax.

    n_burst = g3
    len_burst = valid_cols * elem_bytes
    ub_stride = ub_cols * elem_bytes
    gm_stride = s3 * elem_bytes

    src_stride2 = g3 * ub_cols
    src_stride1 = g2 * src_stride2
    src_stride0 = g1 * src_stride1

    loop1 = g2
    loop2 = g1
    loop1_src_stride = src_stride2 * elem_bytes
    loop1_dst_stride = s2 * elem_bytes
    loop2_src_stride = src_stride1 * elem_bytes
    loop2_dst_stride = s1 * elem_bytes

    ub_ptr = src.as_ptr()
    gm_ptr = dst.as_ptr()

    if loop1 != 1 or loop2 != 1:
        pto.set_loop2_stride_ubtoout(
            src_stride=loop2_src_stride, dst_stride=loop2_dst_stride
        )
        pto.set_loop1_stride_ubtoout(
            src_stride=loop1_src_stride, dst_stride=loop1_dst_stride
        )
        pto.set_loop_size_ubtoout(loop1=loop1, loop2=loop2)

    for i in range(0, g0, 1):
        src_i = pto.addptr(ub_ptr, i * src_stride0)
        dst_i = pto.addptr(gm_ptr, i * s0)
        pto.copy_ubuf_to_gm(
            dst=dst_i,
            src=src_i,
            n_burst=n_burst,
            len_burst=len_burst,
            gm_stride=gm_stride,
            ub_stride=ub_stride,
        )

    if loop1 != 1 or loop2 != 1:
        pto.set_loop_size_ubtoout(loop1=1, loop2=1)
    return

@pto.vkernel(
    target="a5",
    op="pto.tstore",
    advanced=True,
    constraints=[_tstore_preconditions_dn],
)
def template_tstore_dn(src: pto.Tile, dst: pto.PartitionTensorView):
    dtype = src.element_type
    elem_bytes = pto.bytewidth(dtype)

    g0, g1, g2, g3, g4 = dst.shape
    s0, s1, s2, s3, s4 = dst.strides

    valid_rows, valid_cols = src.valid_shape
    ub_rows, ub_cols = src.shape

    n_burst = g4
    len_burst = valid_rows * elem_bytes
    gm_stride = s4 * elem_bytes
    ub_stride = ub_rows * elem_bytes

    # The UB source tile has a compact col-major layout with column height
    # `ub_rows`. Like `TStoreVecDN`, three levels of stride are derived from
    # `g4` / `g2` / `g1`.
    src_stride2 = ub_rows * g4
    src_stride1 = g2 * src_stride2
    src_stride0 = g1 * src_stride1

    loop1 = g2
    loop2 = g1
    loop1_src_stride = src_stride2 * elem_bytes
    loop1_dst_stride = s2 * elem_bytes
    loop2_src_stride = src_stride1 * elem_bytes
    loop2_dst_stride = s1 * elem_bytes

    ub_ptr = src.as_ptr()
    gm_ptr = dst.as_ptr()

    if loop1 != 1 or loop2 != 1:
        pto.set_loop2_stride_ubtoout(
            src_stride=loop2_src_stride, dst_stride=loop2_dst_stride
        )
        pto.set_loop1_stride_ubtoout(
            src_stride=loop1_src_stride, dst_stride=loop1_dst_stride
        )
        pto.set_loop_size_ubtoout(loop1=loop1, loop2=loop2)

    for i in range(0, g0, 1):
        src_i = pto.addptr(ub_ptr, i * src_stride0)
        dst_i = pto.addptr(gm_ptr, i * s0)
        pto.copy_ubuf_to_gm(
            dst=dst_i,
            src=src_i,
            n_burst=n_burst,
            len_burst=len_burst,
            gm_stride=gm_stride,
            ub_stride=ub_stride,        
        )

    if loop1 != 1 or loop2 != 1:
        pto.set_loop_size_ubtoout(loop1=1, loop2=1)
    return

@pto.vkernel(
    target="a5",
    op="pto.tstore",
    advanced=True,
    constraints=[_tstore_preconditions_nz],
)
def template_tstore_nz(src: pto.Tile, dst: pto.PartitionTensorView):
    dtype = src.element_type
    elem_bytes = pto.bytewidth(dtype)

    g0, g1, g2, g3, g4 = dst.shape
    s0, s1, s2, s3, s4 = dst.strides

    valid_rows, valid_cols = src.valid_shape
    ub_rows, ub_cols = src.shape

    # Corresponds to C++ `C0_SIZE_BYTE`. Each NZ burst always writes one
    # complete C0 block.
    c0_size_bytes = 32
    n_burst = g1
    len_burst = valid_rows * c0_size_bytes
    gm_stride = s1 * elem_bytes
    ub_stride = ub_rows * c0_size_bytes

    # Each g0 block in UB is composed of `g1` NZ blocks concatenated together.
    tile_stride = g1 * ub_rows * g4

    ub_ptr = src.as_ptr()
    gm_ptr = dst.as_ptr()

    # NZ path itself does not use loop1/loop2; explicitly reset to normal mode
    # to avoid inheriting stale state.
    pto.set_loop_size_ubtoout(loop1=1, loop2=1)
    for i in range(0, g0, 1):
        src_i = pto.addptr(ub_ptr, i * tile_stride)
        dst_i = pto.addptr(gm_ptr, i * s0)
        pto.copy_ubuf_to_gm(
            dst=dst_i,
            src=src_i,
            n_burst=n_burst,
            len_burst=len_burst,
            gm_stride=gm_stride,
            ub_stride=ub_stride,
        )
    return


# ============================================================================
# Cube Templates: TSTORE.ACC (ACC → GM)
# ============================================================================

def _constraint_tstore_acc_base(src, dst) -> bool:
    """TSTORE.ACC base constraint check"""
    # src must be MemorySpace.ACC
    src_space = src.memory_space
    if src_space is None:
        return False
    src_space_value = src_space.value if hasattr(src_space, "value") else src_space
    if src_space_value not in {"acc", "ACC"}:
        return False
    # dst must be GM (via PartitionTensorView)
    dst_space = dst.memory_space
    if dst_space is None:
        dst_space_value = "gm"  # PartitionTensorView defaults to GM
    else:
        dst_space_value = dst_space.value if hasattr(dst_space, "value") else dst_space
    if dst_space_value not in {"gm", "GM"}:
        return False
    # ACC dtype must be f32 or i32
    src_dtype = src.dtype
    if src_dtype is None:
        return False
    dtype_name = src_dtype.name if hasattr(src_dtype, "name") else str(src_dtype)
    if dtype_name not in {"f32", "i32"}:
        return False
    # dst dtype can be f32, f16, bf16, i32
    dst_dtype = dst.dtype
    if dst_dtype is None:
        return True  # allow dst dtype to be unspecified
    dst_dtype_name = dst_dtype.name if hasattr(dst_dtype, "name") else str(dst_dtype)
    supported_dst_dtypes = {"f32", "f16", "bf16", "i32"}
    if dst_dtype_name not in supported_dst_dtypes:
        return False
    return True


def _extract_b_layout_value(config) -> str | None:
    """Extract the effective b_layout string from a config object.

    Handles both TileConfig (with b_layout/s_layout attributes) and
    ViewConfig (with a single layout attribute like 'nd' or 'dn').
    Returns a canonical layout string: 'row_major' or 'col_major', or None.
    """
    if config is None:
        return None
    # TileConfig has explicit b_layout attribute
    if hasattr(config, "b_layout") and config.b_layout is not None:
        bl = config.b_layout
        return bl.value if hasattr(bl, "value") else bl
    # ViewConfig uses a single 'layout' attribute:
    #   'nd' -> row_major (ND/row-major format)
    #   'dn' -> col_major (DN/col-major format)
    #   'nz' -> col_major (NZ/fractal format, same block layout as col_major)
    if hasattr(config, "layout") and config.layout is not None:
        layout = config.layout
        layout_str = layout.value if hasattr(layout, "value") else str(layout)
        normalized = layout_str.strip().lower().replace("-", "_")
        if normalized in {"nd", "row_major"}:
            return "row_major"
        if normalized in {"dn", "col_major"}:
            return "col_major"
        if normalized in {"nz"}:
            return "col_major"  # NZ fractal uses col-major block layout
    return None


def _extract_s_layout_value(config) -> str | None:
    """Extract the effective s_layout string from a config object.

    For TileConfig this is the s_layout attribute.
    For ViewConfig there is no slayout distinction; returns None.
    """
    if config is None:
        return None
    if hasattr(config, "s_layout") and config.s_layout is not None:
        sl = config.s_layout
        return sl.value if hasattr(sl, "value") else sl
    return None


def _constraint_tstore_acc_nz2nd(src, dst) -> bool:
    """TSTORE.ACC NZ2ND constraint"""
    if not _constraint_tstore_acc_base(src, dst):
        return False
    # dst must be row-major layout (ND format)
    b_layout_value = _extract_b_layout_value(dst.config)
    if b_layout_value is None:
        return True  # default is row-major
    # ROW_MAJOR corresponds to ND format
    if b_layout_value not in {"row_major", "ROW_MAJOR"}:
        return False
    return True


def _constraint_tstore_acc_nz2dn(src, dst) -> bool:
    """TSTORE.ACC NZ2DN constraint"""
    if not _constraint_tstore_acc_base(src, dst):
        return False
    b_layout_value = _extract_b_layout_value(dst.config)
    if b_layout_value is None:
        return False
    if b_layout_value not in {"col_major", "COL_MAJOR"}:
        return False
    return True


def _constraint_tstore_acc_nz2nz(src, dst) -> bool:
    """TSTORE.ACC NZ2NZ constraint"""
    if not _constraint_tstore_acc_base(src, dst):
        return False
    # dst must be NZ layout (fractal): b_layout=col_major + s_layout=row_major
    b_layout_value = _extract_b_layout_value(dst.config)
    s_layout_value = _extract_s_layout_value(dst.config)
    if b_layout_value is None:
        return False
    if s_layout_value is None:
        return False
    if b_layout_value not in {"col_major", "COL_MAJOR"}:
        return False
    if s_layout_value not in {"row_major", "ROW_MAJOR"}:
        return False
    return True


@pto.ckernel(
    target="a5",
    op="pto.tstore",
    priority=1,
    dtypes=[
        (pto.f32, pto.f32),
        (pto.f32, pto.f16),
        (pto.f32, pto.bf16),
        (pto.i32, pto.i32),
    ],
    constraints=[_constraint_tstore_acc_nz2nd],
    name="tstore_acc_to_gm_nz2nd",
)
def template_tstore_acc_to_gm_nz2nd(src: pto.Tile, dst: pto.PartitionTensorView):
    """ACC -> GM (NZ2ND mode)

    Write NZ-format data from L0C Accumulator Buffer back to GM in
    Row-Major (ND) format.

    Args:
        src: Tile with ACC memory_space, shape=(M, N), dtype=f32/i32
        dst: PartitionTensorView with GM memory_space, row-major (ND) format

    Uses:
        pto.mte_l0c_gm with layout="nz2nd"
    """
    m, n = src.valid_shape

    acc_ptr = src.as_ptr()
    gm_ptr = dst.as_ptr()

    # rank-5 partition view metadata (ND: g3 = M rows, g4 = N cols).
    g0, g1, g2, g3, g4 = dst.shape
    s0, s1, s2, s3, s4 = dst.strides

    # Strides are in ELEMENT units (pto-isa a5/TStore.hpp: TStoreAccND passes
    # srcStride/dstD to copy_matrix_cc_to_gm without GetByteSize; the hardware
    # scales by dtype internally). The earlier `src_stride = dst_stride = n`
    # filled both fields with the tile column count, which is wrong for any
    # strided GM slice and produces sparse/shifted GM writes.
    #
    #   src_stride = TileData::Rows = src.shape[0]  (NZ row stride of L0C acc)
    #   dst_stride = gStride3       = s3            (GM physical row stride)
    src_stride = src.shape[0]
    dst_stride = s3

    pto.mte_l0c_gm(
        acc_ptr, gm_ptr,
        m, n, src_stride, dst_stride,
        0, 0,
        layout="nz2nd"
    )
    return


@pto.ckernel(
    target="a5",
    op="pto.tstore",
    priority=1,
    dtypes=[
        (pto.f32, pto.f32),
        (pto.f32, pto.f16),
        (pto.f32, pto.bf16),
        (pto.i32, pto.i32),
    ],
    constraints=[_constraint_tstore_acc_nz2dn],
    name="tstore_acc_to_gm_nz2dn",
)
def template_tstore_acc_to_gm_nz2dn(src: pto.Tile, dst: pto.PartitionTensorView):
    """ACC -> GM (NZ2DN mode)

    Write NZ-format data from L0C Accumulator Buffer back to GM in
    Col-Major (DN) format.

    Args:
        src: Tile with ACC memory_space, shape=(M, N)
        dst: PartitionTensorView with GM memory_space, col-major (DN) format

    Uses:
        pto.mte_l0c_gm with layout="nz2dn"
    """
    m, n = src.valid_shape

    acc_ptr = src.as_ptr()
    gm_ptr = dst.as_ptr()

    # rank-5 partition view metadata (DN: g3 = M, g4 = N, col-major).
    g0, g1, g2, g3, g4 = dst.shape
    s0, s1, s2, s3, s4 = dst.strides

    # Strides are in ELEMENT units (see template_tstore_acc_to_gm_nz2nd).
    # DN (col-major): adjacent rows step along the M direction, whose physical
    # stride is s4 (PTOAS 5D DN view strides collapse to [M, M, M, 1, M]).
    #   src_stride = TileData::Rows = src.shape[0]  (NZ row stride of L0C acc)
    #   dst_stride = s4                             (GM physical stride along M)
    src_stride = src.shape[0]
    dst_stride = s4
    loop0_src_stride = 1  # loop0_src_stride for NZ2DN (NZ source)

    pto.mte_l0c_gm(
        acc_ptr, gm_ptr,
        m, n, src_stride, dst_stride,
        0, 0,
        layout=("nz2dn", loop0_src_stride)
    )
    return


@pto.ckernel(
    target="a5",
    op="pto.tstore",
    dtypes=[
        (pto.f32, pto.f32),
        (pto.f32, pto.f16),
        (pto.f32, pto.bf16),
        (pto.i32, pto.i32),
    ],
    constraints=[_constraint_tstore_acc_nz2nz],
    name="tstore_acc_to_gm_nz2nz",
)
def template_tstore_acc_to_gm_nz2nz(src: pto.Tile, dst: pto.PartitionTensorView):
    """ACC -> GM (NZ2NZ mode)

    Write NZ-format data from L0C Accumulator Buffer back to GM in NZ
    format (no layout conversion).

    Args:
        src: Tile with ACC memory_space, shape=(M, N)
        dst: PartitionTensorView with GM memory_space, NZ (fractal) format

    Uses:
        pto.mte_l0c_gm with layout="nz2nz"
    """
    m, n = src.valid_shape

    acc_ptr = src.as_ptr()
    gm_ptr = dst.as_ptr()

    # src_stride is in ELEMENT units (pto-isa a5/TStore.hpp: TStoreAccNZ sets
    # srcStride = TileData::Rows). The earlier `src_stride = n` filled the L0C
    # source stride with the column count, which is wrong whenever N differs
    # from the acc tile row count.
    #   src_stride = TileData::Rows = src.shape[0]  (NZ row stride of L0C acc)
    # NOTE: dst_stride for NZ2NZ is the NZ-fractal physical row stride
    # (ceil(M, FRACTAL_NZ_ROW) * c0 in pto-isa TStoreAccNZ). The compact NZ
    # views used today set this equal to n, so the legacy value is retained
    # until a strided NZ destination case is exercised to validate the exact
    # stride index.
    src_stride = src.shape[0]
    dst_stride = n
    split = 1  # NZ2NZ requires a split parameter

    pto.mte_l0c_gm(
        acc_ptr, gm_ptr,
        m, n, src_stride, dst_stride,
        0, 0,
        layout=("nz2nz", split)
    )
    return


# ============================================================================
# Cube Templates: TSTORE.MAT (MAT → GM) — NOT YET IMPLEMENTED
# ============================================================================
# NOTE: There is no direct MAT → GM DMA path (mte_l1_gm does not exist).
# The correct implementation requires a two-step intermediate path:
#   1. Copy from MAT to UB via mte_l1_ub (allocating a temporary UB tile)
#   2. Copy from UB to GM via copy_ubuf_to_gm
#
# However, UB buffer allocation is not yet supported in ckernel mode.
# This template is intentionally NOT registered until a proper
# MAT → UB → GM implementation can be provided.
# Adding it with dtypes=[] would crash the module import (_freeze_dtypes
# requires at least one signature tuple), so we omit it entirely.


# ============================================================================
# Cube Templates: TSTORE_FP (ACC + FP → GM)
# ============================================================================

def _constraint_tstore_fp(src, fp, dst) -> bool:
    """TSTORE_FP constraint check"""
    # src must be MemorySpace.ACC
    src_space = src.memory_space
    if src_space is None:
        return False
    src_space_value = src_space.value if hasattr(src_space, "value") else src_space
    if src_space_value not in {"acc", "ACC"}:
        return False
    # fp must be SCALING memory space or specific buffer
    fp_space = fp.memory_space
    if fp_space is None:
        return False
    fp_space_value = fp_space.value if hasattr(fp_space, "value") else fp_space
    if fp_space_value not in {"scaling", "SCALING", "ub", "UB"}:
        return False
    # dst must be GM
    dst_space = dst.memory_space
    if dst_space is None:
        dst_space_value = "gm"
    else:
        dst_space_value = dst_space.value if hasattr(dst_space, "value") else dst_space
    if dst_space_value not in {"gm", "GM"}:
        return False
    # src dtype must be f32
    src_dtype = src.dtype
    if src_dtype is None:
        return False
    dtype_name = src_dtype.name if hasattr(src_dtype, "name") else str(src_dtype)
    if dtype_name != "f32":
        return False
    return True


@pto.ckernel(
    target="a5",
    op="pto.tstore_fp",
    dtypes=[
        (pto.f32, pto.f16, pto.f16),
        (pto.f32, pto.bf16, pto.bf16),
    ],
    constraints=[_constraint_tstore_fp],
    name="tstore_fp_acc_to_gm",
)
def template_tstore_fp_acc_to_gm(src: pto.Tile, fp: pto.Tile, dst: pto.PartitionTensorView):
    """ACC + FP -> GM with floating-point conversion (TSTORE_FP)

    Write f32 data from L0C Accumulator Buffer, combined with FP (scaling)
    parameters, back to GM in f16/bf16 format.

    Args:
        src: Tile with ACC memory_space, dtype=f32
        fp: Tile with SCALING/UB memory_space, dtype=f16/bf16
        dst: PartitionTensorView with GM memory_space, dtype=f16/bf16

    Note:
        Uses qf322f16_pre_vec / qf322bf16_pre_vec mode for pre_quant, which
        supports vector (per-channel) scaling rows per the hardware spec.
        The f32_f16/f32_bf16 modes only accept scalar payload and cannot be
        used for per-channel quantization via a scaling buffer.
    """
    m, n = src.valid_shape

    acc_ptr = src.as_ptr()
    fp_ptr = fp.as_ptr()
    gm_ptr = dst.as_ptr()

    # Determine pre_quant mode based on fp (scaling) buffer dtype:
    # Use qf322*_pre_vec modes which support vector scale rows (scaling buffer).
    # f32_f16/f32_bf16 modes only accept scalar payload and cannot be used here.
    # Note: dst.dtype is not supported by the TileLang DSL v1 frontend for
    # PartitionTensorView, so we use fp.element_type instead.
    fp_dtype = fp.element_type

    if pto.constexpr(fp_dtype == pto.bf16):
        quant_mode = "qf322bf16_pre_vec"
    else:
        quant_mode = "qf322f16_pre_vec"

    # TODO: Replace with tstore_fp DSL surface once available.
    # Currently using mte_l0c_gm + pre_quant as a temporary workaround.
    # Strides are in ELEMENT units (see template_tstore_acc_to_gm_nz2nd).
    #   src_stride = TileData::Rows = src.shape[0]  (NZ row stride of L0C acc)
    #   dst_stride = gStride3       = s3            (GM physical row stride, ND)
    g0, g1, g2, g3, g4 = dst.shape
    s0, s1, s2, s3, s4 = dst.strides
    src_stride = src.shape[0]
    dst_stride = s3

    pto.mte_l0c_gm(
        acc_ptr, gm_ptr,
        m, n, src_stride, dst_stride,
        0, 0,
        layout="nz2nd",
        pre_quant=(fp_ptr, quant_mode)
    )
    return
