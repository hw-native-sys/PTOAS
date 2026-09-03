# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""
``pto`` – the public DSL namespace.

Import as::

    import pto

or as the sub-namespace ``pto`` from the ptodsl package::

    from ptodsl import pto

All user-facing symbols live here.  Low-level MLIR bindings are accessed
internally as ``_pto`` (``from ptoas.mlir.dialects import pto as _pto``).
"""

from ._diagnostics import deprecated, unsupported_public_surface_error

# ── Types ─────────────────────────────────────────────────────────────────────
from ._types import (           # noqa: F401
    float32, float16, bf16,
    f8e4m3, f8e5m2, f8e8m0, hif8, f4e1m2x2, f4e2m1x2,
    f16x2, bf16x2, f32x2,
    f8e4m3x2, f8e4m3x4, f8e4m3x8,
    f8e5m2x2, f8e5m2x4, f8e5m2x8, hif8x2,
    i8x2, i16x2, i32x2,
    int1, int8, int16, int32, int64,
    si8, si16, si32, si64,
    ui8, ui16, ui32, ui64,
    index,
    ptr, vreg_type, vec_type, mask_type, struct_type,
    _resolve,
)
from ._builtin_vector import Vec  # noqa: F401
from ._surface_types import (   # noqa: F401
    const_expr,
    BarrierType,
    Pipe,
    MemorySpace,
    BLayout,
    SLayout,
    MaskPattern,
    CmpMode,
    PredicatePart,
    PredicateDist,
    VStoreDist,
    DeinterleaveDist,
    InterleaveDist,
    PostUpdate,
    FractalMode,
    AccStoreUnitFlagCtrl,
    MadUnitFlagMode,
    SatMode,
    Tf32Mode,
    SplitMode,
    PartMode,
    PositionMode,
    VPackPart,
    VcvtRoundMode,
    VcvtSatMode,
    VcvtPartMode,
    AlignType,
    RoundMode,
    Precision,
    DivPrecision,
    RemPrecision,
    ExpPrecision,
    LogPrecision,
    RecipPrecision,
    RsqrtPrecision,
    SqrtPrecision,
    TensorView,
    PartitionTensorView,
    Tile,
)
from ._tensor_factories import empty_like  # noqa: F401
from ._tile_namespace import tile  # noqa: F401
from ._vmi_namespace import vmi  # noqa: F401

# ── Operations ────────────────────────────────────────────────────────────────
from ._ops import (             # noqa: F401
    const,
    declare_struct, struct_get, struct_set,
    get_op_attr,
    castptr, addptr,
    vlds, vldas, vldus, vldsx2, vsts, vstsx2,
    init_align,
    plt_b8, plt_b16, plt_b32,
    pset_b8, pset_b16, pset_b32,
    pge_b8, pge_b16, pge_b32,
    make_mask, bytewidth, elements_per_vreg,
    pand, por, pxor, pnot, psel,
    pbitcast,
    vcvt, vpack, vmulscvt,
    ppack, punpack,
    pintlv_b8, pintlv_b16, pintlv_b32,
    pdintlv_b8, pdintlv_b16, pdintlv_b32,
    vintlv, vdintlv,
    vgather2, vgather2_bc, vgatherb, vscatter, vsldb, vsstb,
    vcmp, vcmps,
    plds, psts, pstu, vstar, vstas, vstur, vstus,
    vbitcast,
    vbr,
    vadd, vsub, vmul, vdiv, vmax, vmin,
    vand, vor, vxor, vshl, vshr, vshls, vshrs,
    vcmax, vcadd, vcmin, vdup, vexpdif,
    vexp, vln, vsqrt, vabs, vneg, vrec, vrsqrt, vrelu, vnot, vsqz,
    vcgmax, vcgadd, vcgmin, vcpadd,
    vtrc, vprelu, vintlv, vdintlv, vselr,
    chistv2,
    vci, vaddc, vaddcs, vmull, vbitsort, vmrgsort4,
    load_scalar, store_scalar, print,
    vadds, vsubs, vmuls, vmaxs, vmins, vlrelu, vands, vors, vxors,
    vaxpy, vaddrelu, vsubrelu,
    vmula, vmadd,
    vsel,
    make_tensor_view, partition_view,
    alloc_buffer, alloc_tile,
    tsort32, tmrgsort, tgather, tscatter, tprint,
    trem, trems, tfmod, tfmods, tprelu,
    trandom,
    mte_load, mte_store, mte_gm_ub, mte_ub_gm, mte_ub_ub, mte_ub_l1,
    mte_gm_l1, raw_fill_l1, mte_l1_ub, mte_gm_l1_frac, mte_l1_bt, mte_l1_fb, mem_bar,
    set_store_atomic_cfg,
    set_atomic_add, set_atomic_max, set_atomic_min, set_atomic_none,
    set_atomic_f32, set_atomic_f16, set_atomic_bf16,
    set_atomic_s32, set_atomic_s16, set_atomic_s8,
    mte_l1_l0a, mte_l1_l0b, mte_l1_l0a_mx, mte_l1_l0b_mx,
    mte_l0c_l1, mte_l0c_gm, mte_l0c_ub,
    mad, mad_acc, mad_bias, mad_mx, mad_mx_acc, mad_mx_bias,
    tgemv, tgemv_acc, tgemv_bias,
    get_block_idx, get_block_num, get_subblock_idx, get_subblock_num,
    store_vfsimt_info, simt_launch,
    get_tid, get_tid_x, get_tid_y, get_tid_z,
    get_block_dim, get_block_dim_x, get_block_dim_y, get_block_dim_z,
    get_grid_dim, get_grid_dim_x, get_grid_dim_y, get_grid_dim_z,
    get_block_idx_x, get_block_idx_y, get_block_idx_z,
    get_veccoreid, get_clock32, get_clock64,
    get_laneid, get_lanemask_eq, get_lanemask_le, get_lanemask_lt,
    get_lanemask_ge, get_lanemask_gt,
    vote_all, vote_any, vote_uni, vote_ballot,
    shuffle_idx, shuffle_up, shuffle_down, shuffle_bfly,
    redux_add, redux_max, redux_min,
    ldg, stg,
    atomic_exch, atomic_add, atomic_sub, atomic_min, atomic_max,
    atomic_and, atomic_or, atomic_xor, atomic_cas,
    prmt, mulhi, mul_i32toi64,
    absf, sqrt, exp, log, sin, cos, pow, ceil, floor, rint, round,
    fmin, fmax, fma, convert,
    syncthreads, threadfence, threadfence_block, trap, keep, resume,
    pipe_barrier,
    get_buf, rls_buf,
    set_cross_block, wait_cross_block, set_intra_block, wait_intra_block,
    set_flag, wait_flag,
    reserve_buffer, import_reserved_buffer,
)

# ── Control flow ──────────────────────────────────────────────────────────────
from ._control_flow import (    # noqa: F401
    section, vecscope,
    for_, while_, _while, if_, yield_,
    static_range, range,
    LoopHandle, BranchHandle,
    _short_circuit_and, _short_circuit_or,
)

# ── All-reduce ─────────────────────────────────────────────────────────────────
from ._allreduce import simt_allreduce_max, simt_allreduce_min, simt_allreduce_sum  # noqa: F401

# ── Decorator ─────────────────────────────────────────────────────────────────
from ._jit import jit, KernelHandle, merge_jit_modules      # noqa: F401
from ._func import func  # noqa: F401
from ._subkernels import cube, simd, simt, tileop     # noqa: F401
from ._pipe_namespace import pipe  # noqa: F401

# ── Shorthand dtype aliases ───────────────────────────────────────────────────
def gm_ptr(elem):
    return ptr(elem, "gm")


f32 = float32
f16 = float16
i1 = int1
i8 = int8
i16 = int16
i32 = int32
i64 = int64
mask_b8 = mask_type("b8")
mask_b16 = mask_type("b16")
mask_b32 = mask_type("b32")
PAT = MaskPattern


def __getattr__(name):
    if name in {"ukernel", "tile_buf_type", "as_ptr", "vbrc_load", "vsts_1pt", "constexpr", "copy_ubuf_to_ubuf", "tensor_spec", "TensorSpec"}:
        raise unsupported_public_surface_error(name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
