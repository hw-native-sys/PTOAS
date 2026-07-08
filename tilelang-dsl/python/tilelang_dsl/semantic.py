# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Semantic model for TileLang DSL descriptor lowering."""

from __future__ import annotations

import ast
import struct
from dataclasses import dataclass
from typing import Any

from .frontend_ast import (
    FrontendAssignStmt,
    FrontendAttributeExpr,
    FrontendBinaryExpr,
    FrontendCallExpr,
    FrontendConstantExpr,
    FrontendExprNode,
    FrontendExprStmt,
    FrontendForStmt,
    FrontendIfStmt,
    FrontendInlineProcNode,
    FrontendKernelNode,
    FrontendNameExpr,
    FrontendNameTarget,
    FrontendNoOpStmt,
    FrontendReturnStmt,
    FrontendSliceExpr,
    FrontendSourceLocation,
    FrontendStrictVecscopeStmt,
    FrontendStmtNode,
    FrontendSubscriptExpr,
    FrontendSymbolExpr,
    FrontendTargetNode,
    FrontendTupleExpr,
    FrontendTupleTarget,
    FrontendVecscopeStmt,
)
from .support_matrix import (
    DEFERRED_PTO_SURFACES,
    advanced_mode_message,
    deferred_surface_message,
    unsupported_feature_message,
)
from .types import (
    AlignType,
    BarrierType,
    BLayout,
    CmpMode,
    CompactMode,
    DeinterleaveDist,
    Event,
    InterleaveDist,
    MaskType,
    MaskPattern,
    MemorySpace,
    OrderMode,
    PadMode,
    PadValue,
    PredicateDist,
    PredicatePart,
    Pipe,
    PostUpdateMode,
    PositionMode,
    PointerType,
    ScalarType,
    SLayout,
    TileConfig,
    FractalMode,
    VcvtPartMode,
    VcvtRoundMode,
    VcvtSatMode,
    VLoadDist,
    VRegType,
    VStoreDist,
    bf16,
    bytewidth,
    f16,
    f32,
    hif8,
    f8e4m3,
    f8e5m2,
    f4e1m2x2,
    f4e2m1x2,
    i1,
    i8,
    i16,
    i32,
    i64,
    integer_bitwidth,
    integer_signedness,
    is_float_dtype,
    is_integer_dtype,
    is_low_precision_dtype,
    is_storage_only_dtype,
    si8,
    si16,
    si32,
    si64,
    align,
    ui8,
    ui16,
    ui32,
    ui64,
    VectorType,
)


_DTYPE_SYMBOLS = {
    "i1": i1,
    "i8": i8,
    "si8": si8,
    "ui8": ui8,
    "i16": i16,
    "si16": si16,
    "ui16": ui16,
    "i32": i32,
    "si32": si32,
    "ui32": ui32,
    "i64": i64,
    "si64": si64,
    "ui64": ui64,
    "f16": f16,
    "bf16": bf16,
    "f32": f32,
    "hif8": hif8,
    "f8e4m3": f8e4m3,
    "f8e5m2": f8e5m2,
    "f4e1m2x2": f4e1m2x2,
    "f4e2m1x2": f4e2m1x2,
}
_MASK_TYPE_SYMBOLS = {
    "mask_b8": MaskType("b8"),
    "mask_b16": MaskType("b16"),
    "mask_b32": MaskType("b32"),
}
_PATTERN_SYMBOLS = {pattern.name: pattern for pattern in MaskPattern}
_PIPE_SYMBOLS = {pipe.name: pipe for pipe in Pipe}
_EVENT_SYMBOLS = {event.name: event for event in Event}
_BARRIER_TYPE_SYMBOLS = {barrier_type.name: barrier_type for barrier_type in BarrierType}
_MEMORY_SPACE_SYMBOLS = {memory_space.name: memory_space for memory_space in MemorySpace}
_PAD_MODE_SYMBOLS = {pad_mode.name: pad_mode for pad_mode in PadMode}
_B_LAYOUT_SYMBOLS = {layout.name: layout for layout in BLayout}
_S_LAYOUT_SYMBOLS = {layout.name: layout for layout in SLayout}
_COMPACT_MODE_SYMBOLS = {compact_mode.name: compact_mode for compact_mode in CompactMode}
_PAD_VALUE_SYMBOLS = {
    pad_value.name: pad_value
    for pad_value in (PadValue.NULL, PadValue.ZERO, PadValue.MAX, PadValue.MIN)
}
_PREDICATE_DIST_SYMBOLS = {dist.name: dist for dist in PredicateDist}
_VLOAD_DIST_SYMBOLS = {dist.name: dist for dist in VLoadDist}
_VSTORE_DIST_SYMBOLS = {dist.name: dist for dist in VStoreDist}
_PREDICATE_PART_SYMBOLS = {part.name: part for part in PredicatePart}
_CMP_MODE_SYMBOLS = {mode.name: mode for mode in CmpMode}
_DEINTERLEAVE_DIST_SYMBOLS = dict(DeinterleaveDist.__members__)
_INTERLEAVE_DIST_SYMBOLS = dict(InterleaveDist.__members__)
_POSITION_MODE_SYMBOLS = {position_mode.name: position_mode for position_mode in PositionMode}
_ORDER_MODE_SYMBOLS = {order_mode.name: order_mode for order_mode in OrderMode}
_VCVT_ROUND_MODE_SYMBOLS = {mode.name: mode for mode in VcvtRoundMode}
_VCVT_SAT_MODE_SYMBOLS = {mode.name: mode for mode in VcvtSatMode}
_VCVT_PART_MODE_SYMBOLS = {mode.name: mode for mode in VcvtPartMode}
_POST_UPDATE_MODE_SYMBOLS = {mode.name: mode for mode in PostUpdateMode}
_FRACTAL_MODE_SYMBOLS = {mode.name: mode for mode in FractalMode}
_TILE_CONSTRUCTOR_ALLOWED_KEYWORDS = frozenset(
    {
        "valid_shape",
        "blayout",
        "slayout",
        "fractal_size",
        "pad_value",
        "compact_mode",
        "addr",
    }
)
_VCVT_ATTR_CONTRACTS: dict[tuple[str, str], tuple[bool, bool, bool]] = {
    # (src_kind, dst_kind): (requires_rnd, requires_sat, requires_part)
    ("f32", "f16"): (True, True, True),
    ("f32", "bf16"): (True, True, True),
    ("f32", "s16"): (True, True, True),
    ("f32", "s64"): (True, True, True),
    ("f32", "s32"): (True, True, False),
    ("f16", "f32"): (False, False, True),
    ("f16", "s32"): (True, False, True),
    ("f16", "s16"): (True, True, False),
    ("f16", "s8"): (True, True, True),
    ("f16", "u8"): (True, True, True),
    ("bf16", "f16"): (True, True, False),
    ("bf16", "f32"): (False, False, True),
    ("bf16", "s32"): (True, True, True),
    ("u8", "f16"): (False, False, True),
    ("u8", "u16"): (False, False, True),
    ("u8", "u32"): (False, False, True),
    ("s8", "f16"): (False, False, True),
    ("s8", "s16"): (False, False, True),
    ("s8", "s32"): (False, False, True),
    ("u16", "u8"): (False, True, True),
    ("u16", "u32"): (False, False, True),
    ("s16", "f16"): (True, False, False),
    ("s16", "f32"): (False, False, True),
    ("s16", "u32"): (False, False, True),
    ("s16", "s32"): (False, False, True),
    ("s16", "u8"): (False, True, True),
    ("u32", "u8"): (False, True, True),
    ("u32", "u16"): (False, True, True),
    ("u32", "s16"): (False, True, True),
    ("s32", "f32"): (True, False, False),
    ("s32", "u8"): (False, True, True),
    ("s32", "u16"): (False, True, True),
    ("s32", "s16"): (False, True, True),
    ("s32", "s64"): (False, False, True),
    ("s64", "f32"): (True, False, True),
    ("s64", "s32"): (False, True, True),
    ("f32", "fp8"): (True, True, True),
    ("f32", "hif8"): (True, True, True),
    ("f16", "hif8"): (True, True, True),
    ("bf16", "fp4x2"): (True, False, True),
    ("fp8", "f32"): (False, False, True),
    ("hif8", "f32"): (False, False, True),
    ("fp4x2", "bf16"): (False, False, True),
}


def _classify_vcvt_elem_kind(dtype: ScalarType) -> str | None:
    if dtype == f16:
        return "f16"
    if dtype == bf16:
        return "bf16"
    if dtype == f32:
        return "f32"
    if dtype in {f8e4m3, f8e5m2}:
        return "fp8"
    if dtype == hif8:
        return "hif8"
    if dtype in {f4e1m2x2, f4e2m1x2}:
        return "fp4x2"
    if not is_integer_dtype(dtype):
        return None
    width = integer_bitwidth(dtype)
    sign = integer_signedness(dtype)
    is_unsigned = sign == "unsigned"
    if width == 8:
        return "u8" if is_unsigned else "s8"
    if width == 16:
        return "u16" if is_unsigned else "s16"
    if width == 32:
        return "u32" if is_unsigned else "s32"
    if width == 64:
        return None if is_unsigned else "s64"
    return None
_UNARY_VECTOR_OPS = {
    "vabs",
    "vrelu",
    "vexp",
    "vln",
    "vsqrt",
    "vrec",
    "vnot",
    "vcadd",
    "vcmax",
    "vbcnt",
    "vneg",
    "vcls",
    "vcmin",
    "vrsqrt",
    "vmov",
    "vsunpack",
    "vzunpack",
    "vusqz",
    "vsqz",
    "vtrc",
    "vcgadd",
    "vcgmax",
    "vcgmin",
    "vcpadd",
    "vsort32",
}
_BINARY_VECTOR_OPS = {
    "vadd",
    "vsub",
    "vmul",
    "vdiv",
    "vmod",
    "vmax",
    "vmin",
    "vand",
    "vor",
    "vxor",
    "vaddrelu",
    "vaddreluconv",
    "vsubrelu",
    "vmulconv",
    "vshl",
    "vshr",
    "vprelu",
    "vpack",
    "vperm",
    "vmrgsort",
}
_CUBE_MATMUL_OPS = {
    "mad",
    "mad_acc",
    "mad_bias",
    "mad_mx",
    "mad_mx_acc",
    "mad_mx_bias",
}
_CUBE_TRANSFER_OPS = {
    "mte_gm_l1",
    "mte_l1_ub",
    "mte_gm_l1_frac",
    "mte_l1_bt",
    "mte_l1_fb",
    "mte_l1_l0a",
    "mte_l1_l0b",
    "mte_l1_l0a_mx",
    "mte_l1_l0b_mx",
    "mte_l0c_l1",
    "mte_l0c_gm",
    "mte_l0c_ub",
}
_LEGACY_CUBE_ALIAS_CANONICAL = {
    "cube_load": "mte_gm_l1",
    "cube_store": "mte_l1_ub",
    "cube_load_frac": "mte_gm_l1_frac",
    "bias_load": "mte_l1_bt",
    "left_load": "mte_l1_l0a",
    "right_load": "mte_l1_l0b",
    "left_load_mx": "mte_l1_l0a_mx",
    "right_load_mx": "mte_l1_l0b_mx",
    "acc_store": "mte_l0c_l1",
    "acc_store_gm": "mte_l0c_gm",
    "acc_store_ub": "mte_l0c_ub",
}
_CUBE_CALL_OPS = _CUBE_MATMUL_OPS | _CUBE_TRANSFER_OPS | set(_LEGACY_CUBE_ALIAS_CANONICAL)
_VECTOR_SCALAR_OPS = {
    "vadds",
    "vsubs",
    "vmuls",
    "vdivs",
    "vmaxs",
    "vmins",
    "vlrelu",
    "vshls",
    "vshrs",
    "vands",
    "vors",
    "vxors",
}
_VECTOR_IMMEDIATE_OPS = {"vshift", "vslide"}
_TERNARY_VECTOR_OPS = {"vaxpy", "vmula"}
_MULTI_RESULT_VECTOR_OPS = {"vmull", "vldsx2", "vldus", "pstu"}
_BROADCAST_VECTOR_OPS = {"vbr", "vdup", "vci"}
_VEXPDIF_OP_ALIASES = {"vexpdif", "vexpdiff"}
_LOW_LEVEL_DMA_UNARY_CONFIG_OPS = {"set_mov_pad_val"}
_LOW_LEVEL_DMA_CONFIG_OPS = {
    "set_loop2_stride_outtoub",
    "set_loop1_stride_outtoub",
    "set_loop_size_outtoub",
    "set_loop2_stride_ubtoout",
    "set_loop1_stride_ubtoout",
    "set_loop_size_ubtoout",
}
_LOW_LEVEL_DMA_COPY_OPS = {
    "copy_gm_to_ubuf",
    "copy_ubuf_to_gm",
    "copy_ubuf_to_ubuf",
}
_GROUPED_MTE_DMA_OPS = {
    "mte_gm_ub",
    "mte_ub_gm",
    "mte_ub_ub",
    "mte_ub_l1",
}


def _is_supported_mov_pad_scalar_dtype(dtype: ScalarType) -> bool:
    if is_integer_dtype(dtype):
        return integer_bitwidth(dtype) in {8, 16, 32}
    return dtype.name in {"f16", "bf16", "f32"}


_UB_HELPER_OPS = {"vbitsort", "vmrgsort4"}
_TENSORVIEW_RANK = 5


class SemanticType:
    """Base class for semantic value types."""


@dataclass(frozen=True)
class SemanticTensorViewType(SemanticType):
    element_dtype: ScalarType
    rank: int = _TENSORVIEW_RANK


@dataclass(frozen=True)
class SemanticPartitionTensorViewType(SemanticType):
    element_dtype: ScalarType
    rank: int = _TENSORVIEW_RANK


@dataclass(frozen=True)
class SemanticTensorSliceType(SemanticType):
    element_dtype: ScalarType
    rank: int
    extents: tuple[int | None, ...]
    physical_axes: tuple[int, ...]


@dataclass(frozen=True)
class SemanticTileType(SemanticType):
    element_dtype: ScalarType
    rank: int
    shape: tuple[int, ...] | None
    valid_shape: tuple[int | None, ...] | None
    memory_space: str | None
    config: TileConfig | None


@dataclass(frozen=True)
class SemanticTileConfigType(SemanticType):
    element_dtype: ScalarType | None = None


@dataclass(frozen=True)
class SemanticScalarType(SemanticType):
    dtype: ScalarType


@dataclass(frozen=True)
class SemanticPtrType(SemanticType):
    element_dtype: ScalarType
    memory_space: str


@dataclass(frozen=True)
class SemanticIndexType(SemanticType):
    pass


@dataclass(frozen=True)
class SemanticShapeType(SemanticType):
    rank: int


@dataclass(frozen=True)
class SemanticSliceType(SemanticType):
    pass


@dataclass(frozen=True)
class SemanticTupleType(SemanticType):
    elements: tuple[SemanticType, ...]


@dataclass(frozen=True)
class SemanticMetaType(SemanticType):
    kind: str


@dataclass(frozen=True)
class SemanticPadValueType(SemanticType):
    element_dtype: ScalarType | None = None


@dataclass(frozen=True)
class SemanticAlignType(SemanticType):
    pass


@dataclass(frozen=True)
class SemanticMaskType(SemanticType):
    granularity: str


@dataclass(frozen=True)
class SemanticVRegType(SemanticType):
    element_dtype: ScalarType
    lanes: int


@dataclass(frozen=True)
class SemanticVectorType(SemanticType):
    element_dtype: ScalarType
    shape: tuple[int, ...]


_I32_TYPE = SemanticScalarType(dtype=i32)


@dataclass(frozen=True)
class SemanticBinding:
    name: str
    ssa_name: str
    type: SemanticType
    origin: str
    value: Any | None = None


@dataclass(frozen=True)
class SemanticTileBinding:
    name: str
    shape: tuple[int, ...]
    valid_shape: tuple[int | None, ...] | None
    memory_space: str
    config: Any


class SemanticExpr:
    """Base class for typed semantic expressions."""


@dataclass(frozen=True)
class SemanticBindingRef(SemanticExpr):
    binding: SemanticBinding
    type: SemanticType


@dataclass(frozen=True)
class SemanticLiteralExpr(SemanticExpr):
    value: Any
    type: SemanticType


@dataclass(frozen=True)
class SemanticSymbolExpr(SemanticExpr):
    namespace: str
    name: str
    value: Any
    type: SemanticType


@dataclass(frozen=True)
class SemanticSliceExpr(SemanticExpr):
    start: SemanticExpr | None
    stop: SemanticExpr | None
    step: SemanticExpr | None
    type: SemanticSliceType


@dataclass(frozen=True)
class SemanticTensorSliceAxis:
    start: SemanticExpr
    stop: SemanticExpr
    step: SemanticExpr
    extent: int | None


@dataclass(frozen=True)
class SemanticTupleExpr(SemanticExpr):
    elements: tuple[SemanticExpr, ...]
    type: SemanticTupleType


@dataclass(frozen=True)
class SemanticAttributeAccess(SemanticExpr):
    base: SemanticExpr
    attr: str
    type: SemanticType


@dataclass(frozen=True)
class SemanticSubscriptAccess(SemanticExpr):
    base: SemanticExpr
    index: SemanticExpr
    type: SemanticType


@dataclass(frozen=True)
class SemanticTensorSliceExpr(SemanticExpr):
    base: SemanticExpr
    slices: tuple[SemanticTensorSliceAxis, ...]
    type: SemanticTensorSliceType


@dataclass(frozen=True)
class SemanticBinaryExpr(SemanticExpr):
    lhs: SemanticExpr
    op: str
    rhs: SemanticExpr
    type: SemanticType


@dataclass(frozen=True)
class SemanticIndexCastExpr(SemanticExpr):
    value: SemanticExpr
    type: SemanticIndexType


@dataclass(frozen=True)
class SemanticCallExpr(SemanticExpr):
    namespace: str | None
    name: str
    args: tuple[SemanticExpr, ...]
    type: SemanticType | None


class SemanticStmt:
    """Base class for semantic statements."""


@dataclass(frozen=True)
class SemanticAssignStmt(SemanticStmt):
    targets: tuple[SemanticBinding, ...]
    value: SemanticExpr
    annotation: Any | None = None


@dataclass(frozen=True)
class SemanticExprStmt(SemanticStmt):
    expr: SemanticExpr


@dataclass(frozen=True)
class SemanticVectorStoreStmt(SemanticStmt):
    value: SemanticExpr
    destination: SemanticExpr
    indices: tuple[SemanticExpr, ...]
    dist: SemanticExpr | None
    mask: SemanticExpr


@dataclass(frozen=True)
class SemanticVectorPairStoreStmt(SemanticStmt):
    low: SemanticExpr
    high: SemanticExpr
    destination: SemanticExpr
    indices: tuple[SemanticExpr, ...]
    dist: SemanticExpr
    mask: SemanticExpr


@dataclass(frozen=True)
class SemanticVScatterStmt(SemanticStmt):
    value: SemanticExpr
    destination: SemanticExpr
    offsets: SemanticExpr
    mask: SemanticExpr


@dataclass(frozen=True)
class SemanticPredicateStoreStmt(SemanticStmt):
    op_name: str
    value: SemanticExpr
    destination: SemanticExpr
    indices: tuple[SemanticExpr, ...]
    dist: SemanticExpr


@dataclass(frozen=True)
class SemanticAlignStoreStmt(SemanticStmt):
    op_name: str
    value: SemanticExpr
    destination: SemanticExpr
    indices: tuple[SemanticExpr, ...] = ()
    offset: SemanticExpr | None = None


@dataclass(frozen=True)
class SemanticScalarStoreStmt(SemanticStmt):
    value: SemanticExpr
    destination: SemanticExpr
    offset: SemanticExpr


@dataclass(frozen=True)
class SemanticVecscopeStmt(SemanticStmt):
    body: tuple[SemanticStmt, ...]


@dataclass(frozen=True)
class SemanticSetFlagStmt(SemanticStmt):
    src_pipe: str
    dst_pipe: str
    event: str


@dataclass(frozen=True)
class SemanticWaitFlagStmt(SemanticStmt):
    src_pipe: str
    dst_pipe: str
    event: str


@dataclass(frozen=True)
class SemanticPipeBarrierStmt(SemanticStmt):
    pipe: str


@dataclass(frozen=True)
class SemanticGetBufStmt(SemanticStmt):
    pipe: str
    buf_id: SemanticExpr
    mode: SemanticExpr


@dataclass(frozen=True)
class SemanticRlsBufStmt(SemanticStmt):
    pipe: str
    buf_id: SemanticExpr
    mode: SemanticExpr


@dataclass(frozen=True)
class SemanticMemBarStmt(SemanticStmt):
    barrier_type: str


@dataclass(frozen=True)
class SemanticSetCrossCoreStmt(SemanticStmt):
    core_id: SemanticExpr
    event_id: SemanticExpr


@dataclass(frozen=True)
class SemanticSetIntraBlockStmt(SemanticStmt):
    block_id: SemanticExpr
    event_id: SemanticExpr


@dataclass(frozen=True)
class SemanticSetIntraCoreStmt(SemanticStmt):
    config: SemanticExpr


@dataclass(frozen=True)
class SemanticWaitFlagDevStmt(SemanticStmt):
    core_id: SemanticExpr
    event_id: SemanticExpr


@dataclass(frozen=True)
class SemanticWaitIntraCoreStmt(SemanticStmt):
    block_id: SemanticExpr
    event_id: SemanticExpr


@dataclass(frozen=True)
class SemanticDmaConfigStmt(SemanticStmt):
    name: str
    first: SemanticExpr
    second: SemanticExpr


@dataclass(frozen=True)
class SemanticDmaUnaryConfigStmt(SemanticStmt):
    name: str
    value: SemanticExpr


@dataclass(frozen=True)
class SemanticLowLevelCopyStmt(SemanticStmt):
    name: str
    source: SemanticExpr
    destination: SemanticExpr
    operands: tuple[SemanticExpr, ...]


@dataclass(frozen=True)
class SemanticGroupedMteDmaStmt(SemanticStmt):
    name: str
    args: tuple[SemanticExpr, ...]


@dataclass(frozen=True)
class SemanticIfResult:
    result_binding: SemanticBinding
    then_binding: SemanticBinding
    else_binding: SemanticBinding


@dataclass(frozen=True)
class SemanticIfStmt(SemanticStmt):
    condition: SemanticExpr
    then_body: tuple[SemanticStmt, ...]
    else_body: tuple[SemanticStmt, ...]
    results: tuple[SemanticIfResult, ...]


@dataclass(frozen=True)
class SemanticReturnStmt(SemanticStmt):
    value: SemanticExpr | None


@dataclass(frozen=True)
class SemanticForStmt(SemanticStmt):
    induction_variable: SemanticBinding
    lower_bound: SemanticExpr
    upper_bound: SemanticExpr
    step: SemanticExpr
    body: tuple[SemanticStmt, ...]
    loop_carried: tuple[SemanticBinding, ...]


@dataclass(frozen=True)
class SemanticStrictVecscopeStmt(SemanticStmt):
    captures: tuple[SemanticExpr, ...]
    block_arguments: tuple[SemanticBinding, ...]
    body: tuple[SemanticStmt, ...]


@dataclass(frozen=True)
class SemanticParameter:
    binding: SemanticBinding

    @property
    def name(self) -> str:
        return self.binding.name

    @property
    def kind(self) -> str:
        return self.binding.origin

    @property
    def type(self) -> SemanticType:
        return self.binding.type

    @property
    def ssa_name(self) -> str:
        return self.binding.ssa_name


@dataclass(frozen=True)
class SemanticKernel:
    target: str
    op: str
    symbol_name: str
    kernel_family: str
    verify_enabled: bool
    advanced_enabled: bool
    dtype_signature: tuple[Any, ...] | None
    parameters: tuple[SemanticParameter, ...]
    tile_bindings: tuple[SemanticTileBinding, ...]
    body: tuple[SemanticStmt, ...]
    inline_helpers: tuple["SemanticKernel", ...] = ()


class _SemanticAnalyzer:
    def __init__(self, node: FrontendKernelNode):
        self.node = node
        self._context_attrs = dict(node.context_attrs)
        self._counter = 0
        self._tile_specializations = {
            spec.name: spec for spec in node.tile_specializations
        }
        self._hidden_parameters: list[SemanticParameter] = []
        self._inline_proc_nodes: dict[str, FrontendInlineProcNode] = {
            inline_proc.name: inline_proc for inline_proc in node.inline_procs
        }
        self._internal_inline_proc_nodes: dict[str, FrontendInlineProcNode] = {
            inline_proc.name: inline_proc for inline_proc in node.internal_inline_procs
        }
        self._inline_proc_specializations: dict[
            tuple[str, tuple[tuple[SemanticType, object], ...]], SemanticKernel
        ] = {}
        self._inline_proc_return_types: dict[
            tuple[str, tuple[tuple[SemanticType, object], ...]], SemanticType | None
        ] = {}
        self._inline_proc_order: list[tuple[str, tuple[tuple[SemanticType, object], ...]]] = []
        self._inline_proc_active_stack: list[tuple[str, tuple[tuple[SemanticType, object], ...]]] = []

    def _expr_source_location(
        self,
        expr: FrontendExprNode | SemanticExpr,
    ) -> FrontendSourceLocation | None:
        return getattr(expr, "source_location", None)

    def _attach_expr_source_location(
        self,
        semantic_expr: SemanticExpr,
        frontend_expr: FrontendExprNode,
    ) -> SemanticExpr:
        source_location = self._expr_source_location(frontend_expr)
        if source_location is not None:
            object.__setattr__(semantic_expr, "source_location", source_location)
        return semantic_expr

    def _format_source_message(
        self,
        message: str,
        expr: FrontendExprNode | SemanticExpr | None = None,
    ) -> str:
        if expr is None:
            return message
        source_location = self._expr_source_location(expr)
        if source_location is None:
            return message
        return (
            f"{source_location.path}:{source_location.line}:{source_location.column}: "
            f"{message}"
        )

    def _raise_expr_type_error(
        self,
        message: str,
        expr: FrontendExprNode | SemanticExpr | None = None,
    ) -> None:
        raise TypeError(self._format_source_message(message, expr))

    def analyze(self) -> SemanticKernel:
        env: dict[str, SemanticBinding] = {}
        parameters = []
        for index, param in enumerate(self.node.parameters):
            binding = SemanticBinding(
                name=param.name,
                ssa_name=f"%arg{index}",
                type=self._parameter_type(param),
                origin=param.kind,
            )
            env[param.name] = binding
            parameters.append(SemanticParameter(binding=binding))
        body, _ = self._analyze_kernel_body(env)
        parameters.extend(self._hidden_parameters)
        tile_bindings = tuple(
            SemanticTileBinding(
                name=spec.name,
                shape=spec.shape,
                valid_shape=spec.valid_shape,
                memory_space=spec.memory_space,
                config=spec.config,
            )
            for spec in self.node.tile_specializations
        )
        return SemanticKernel(
            target=self.node.target,
            op=self.node.op,
            symbol_name=self.node.name,
            kernel_family=self.node.kernel_family,
            verify_enabled=self.node.verify_enabled,
            advanced_enabled=self.node.advanced_enabled,
            dtype_signature=self.node.dtype_signature,
            parameters=tuple(parameters),
            tile_bindings=tile_bindings,
            body=body,
            inline_helpers=tuple(
                self._inline_proc_specializations[key]
                for key in self._inline_proc_order
            ),
        )

    def _analyze_kernel_body(
        self,
        env: dict[str, SemanticBinding],
    ) -> tuple[tuple[SemanticStmt, ...], dict[str, SemanticBinding]]:
        return self._analyze_block(
            self.node.body,
            env,
            allow_outer_lookup=True,
        )

    def _parameter_type(self, param: Any) -> SemanticType:
        if param.kind == "tensorview":
            return SemanticTensorViewType(
                element_dtype=param.dtype,
                rank=_TENSORVIEW_RANK,
            )
        if param.kind == "partition_tensor_view":
            return SemanticPartitionTensorViewType(
                element_dtype=param.dtype,
                rank=_TENSORVIEW_RANK,
            )
        if param.kind == "tile":
            spec = self._tile_specializations.get(param.name)
            rank = 2 if spec is None else len(spec.shape)
            shape = None if spec is None else spec.shape
            valid_shape = None if spec is None else (
                spec.shape if spec.valid_shape is None else spec.valid_shape
            )
            memory_space = None if spec is None else spec.memory_space
            return SemanticTileType(
                element_dtype=param.dtype,
                rank=rank,
                shape=shape,
                valid_shape=valid_shape,
                memory_space=memory_space,
                config=None if spec is None else (spec.config or TileConfig()),
            )
        if param.kind == "vector":
            vector_type = param.annotation
            return SemanticVectorType(
                element_dtype=param.dtype,
                shape=vector_type.shape,
            )
        if param.kind == "ptr":
            memory_space = param.annotation.memory_space.value
            return SemanticPtrType(
                element_dtype=param.dtype,
                memory_space=memory_space,
            )
        if param.kind == "mask":
            return SemanticMaskType(granularity=param.dtype.granularity)
        if param.kind == "scalar":
            self._require_non_low_precision_dtype(param.dtype, "scalar parameter type")
            return SemanticScalarType(dtype=param.dtype)
        raise ValueError(f"unsupported parameter kind {param.kind!r}")

    def _new_ssa_name(self, stem: str) -> str:
        name = f"%{stem}_{self._counter}"
        self._counter += 1
        return name

    def _tensor_shape_binding_name(self, tensor_name: str, axis: int) -> str:
        return f"__shape_{tensor_name}_{axis}"

    def _tensor_stride_binding_name(self, tensor_name: str, axis: int) -> str:
        return f"__stride_{tensor_name}_{axis}"

    def _tile_valid_shape_binding_name(self, tile_name: str, axis: int) -> str:
        return f"__valid_shape_{tile_name}_{axis}"

    def _ensure_hidden_parameter(
        self,
        hidden_name: str,
        origin: str,
    ) -> SemanticBinding:
        for parameter in self._hidden_parameters:
            if parameter.name == hidden_name:
                return parameter.binding
        binding = SemanticBinding(
            name=hidden_name,
            ssa_name=f"%arg{len(self.node.parameters) + len(self._hidden_parameters)}",
            type=SemanticIndexType(),
            origin=origin,
        )
        self._hidden_parameters.append(SemanticParameter(binding=binding))
        return binding

    def _ensure_tensor_shape_parameter(
        self,
        tensor_binding: SemanticBinding,
        axis: int,
    ) -> SemanticBinding:
        hidden_name = self._tensor_shape_binding_name(tensor_binding.name, axis)
        return self._ensure_hidden_parameter(hidden_name, "tensorview_shape")

    def _ensure_tensor_stride_parameter(
        self,
        tensor_binding: SemanticBinding,
        axis: int,
    ) -> SemanticBinding:
        hidden_name = self._tensor_stride_binding_name(tensor_binding.name, axis)
        return self._ensure_hidden_parameter(hidden_name, "tensorview_stride")

    def _ensure_tile_valid_shape_parameter(
        self,
        tile_binding: SemanticBinding,
        axis: int,
    ) -> SemanticBinding:
        hidden_name = self._tile_valid_shape_binding_name(tile_binding.name, axis)
        return self._ensure_hidden_parameter(hidden_name, "tile_valid_shape")

    def _make_binding(
        self,
        name: str,
        ty: SemanticType,
        origin: str,
        *,
        value: Any | None = None,
    ) -> SemanticBinding:
        stem = name if name.isidentifier() else "v"
        return SemanticBinding(
            name=name,
            ssa_name=self._new_ssa_name(stem),
            type=ty,
            origin=origin,
            value=value,
        )

    def _analyze_block(
        self,
        statements: tuple[FrontendStmtNode, ...],
        env: dict[str, SemanticBinding],
        *,
        allow_outer_lookup: bool,
    ) -> tuple[tuple[SemanticStmt, ...], dict[str, SemanticBinding]]:
        current_env = dict(env)
        semantic_statements = []
        for stmt in statements:
            emitted_stmts, current_env = self._analyze_stmt_or_inline(
                stmt,
                current_env,
                allow_outer_lookup=allow_outer_lookup,
            )
            semantic_statements.extend(emitted_stmts)
        return tuple(semantic_statements), current_env

    def _analyze_stmt_or_inline(
        self,
        stmt: FrontendStmtNode,
        env: dict[str, SemanticBinding],
        *,
        allow_outer_lookup: bool,
    ) -> tuple[tuple[SemanticStmt, ...], dict[str, SemanticBinding]]:
        if isinstance(stmt, FrontendNoOpStmt):
            # Python `pass` lowers to a frontend no-op and does not materialize semantic IR.
            return tuple(), dict(env)
        if (
            isinstance(stmt, FrontendExprStmt)
            and isinstance(stmt.expr, FrontendConstantExpr)
            and isinstance(stmt.expr.value, str)
        ):
            # Treat Python docstring-style string expression statements as no-op.
            return tuple(), dict(env)
        if isinstance(stmt, FrontendIfStmt) and stmt.is_constexpr:
            return self._analyze_constexpr_if(
                stmt,
                env,
                allow_outer_lookup=allow_outer_lookup,
            )
        semantic_stmt, updated_env = self._analyze_stmt(
            stmt,
            env,
            allow_outer_lookup=allow_outer_lookup,
        )
        return (semantic_stmt,), updated_env

    def _analyze_stmt(
        self,
        stmt: FrontendStmtNode,
        env: dict[str, SemanticBinding],
        *,
        allow_outer_lookup: bool,
    ) -> tuple[SemanticStmt, dict[str, SemanticBinding]]:
        if isinstance(stmt, FrontendAssignStmt):
            value = self._analyze_expr(stmt.value, env, allow_outer_lookup=allow_outer_lookup)
            updated_env = dict(env)
            targets = self._bind_assignment_target(
                stmt.target,
                value,
                updated_env,
                stmt.annotation,
            )
            return (
                SemanticAssignStmt(targets=targets, value=value, annotation=stmt.annotation),
                updated_env,
            )
        if isinstance(stmt, FrontendExprStmt):
            if self._is_sync_call(stmt.expr):
                return self._analyze_sync_stmt(stmt.expr, env, allow_outer_lookup=allow_outer_lookup)
            if self._is_grouped_mte_dma_call(stmt.expr):
                return self._analyze_grouped_mte_dma_stmt(
                    stmt.expr,
                    env,
                    allow_outer_lookup=allow_outer_lookup,
                )
            if self._is_low_level_dma_call(stmt.expr):
                return self._analyze_low_level_dma_stmt(
                    stmt.expr,
                    env,
                    allow_outer_lookup=allow_outer_lookup,
                )
            if self._is_vector_store_call(stmt.expr):
                return self._analyze_vector_store_stmt(stmt.expr, env, allow_outer_lookup=allow_outer_lookup)
            if self._is_scalar_store_call(stmt.expr):
                return self._analyze_scalar_store_stmt(stmt.expr, env, allow_outer_lookup=allow_outer_lookup)
            expr = self._analyze_expr(stmt.expr, env, allow_outer_lookup=allow_outer_lookup)
            return SemanticExprStmt(expr=expr), dict(env)
        if isinstance(stmt, FrontendReturnStmt):
            value = None
            if stmt.value is not None:
                value = self._analyze_expr(stmt.value, env, allow_outer_lookup=allow_outer_lookup)
            return SemanticReturnStmt(value=value), dict(env)
        if isinstance(stmt, FrontendForStmt):
            return self._analyze_for(stmt, env, allow_outer_lookup=allow_outer_lookup)
        if isinstance(stmt, FrontendIfStmt):
            return self._analyze_if(stmt, env, allow_outer_lookup=allow_outer_lookup)
        if isinstance(stmt, FrontendVecscopeStmt):
            return self._analyze_explicit_vecscope(stmt, env, allow_outer_lookup=allow_outer_lookup)
        if isinstance(stmt, FrontendStrictVecscopeStmt):
            return self._analyze_strict_vecscope(stmt, env)
        raise ValueError(f"unsupported frontend statement {type(stmt).__name__}")

    def _inline_proc_specialization_key(
        self,
        name: str,
        args: tuple[SemanticExpr, ...],
        *,
        internal: bool = False,
    ) -> tuple[str, tuple[tuple[SemanticType, object], ...]]:
        specialization_name = f"__internal__::{name}" if internal else name
        return (
            specialization_name,
            tuple(
                (arg.type, self._inline_proc_static_specialization_token(arg))
                for arg in args
            ),
        )

    def _inline_proc_static_specialization_token(
        self,
        expr: SemanticExpr,
    ) -> object:
        if isinstance(expr, SemanticLiteralExpr) and expr.value is None:
            return ("none",)

        if isinstance(expr.type, SemanticMetaType) and expr.type.kind in {
            "dtype",
            "ptr_type",
            "mask_type",
        }:
            value = self._try_static_value(expr)
            if value is not None:
                return ("meta", expr.type.kind, value)

        value = self._try_static_value(expr)
        if isinstance(value, bool):
            return ("bool", value)
        if isinstance(value, int) and not isinstance(value, bool):
            return ("int", value)
        if value is None:
            return ("dynamic",)
        return ("dynamic",)

    def _inline_proc_bound_static_value(
        self,
        expr: SemanticExpr,
    ) -> Any | None:
        token = self._inline_proc_static_specialization_token(expr)
        kind = token[0]
        if kind == "meta":
            return token[2]
        if kind in {"bool", "int"}:
            return token[1]
        if kind == "none":
            return None
        return None

    def _inline_proc_symbol_name(
        self,
        name: str,
        index: int,
    ) -> str:
        sanitized = "".join(char if char.isalnum() else "_" for char in name)
        return f"__tl_inline_{sanitized}_{index}"

    def _collect_inline_helper_tile_bindings(
        self,
        parameters: tuple[SemanticParameter, ...],
    ) -> tuple[SemanticTileBinding, ...]:
        tile_bindings: list[SemanticTileBinding] = []
        for parameter in parameters:
            if not isinstance(parameter.type, SemanticTileType):
                continue
            if parameter.type.shape is None:
                continue
            tile_bindings.append(
                SemanticTileBinding(
                    name=parameter.name,
                    shape=parameter.type.shape,
                    valid_shape=parameter.type.valid_shape,
                    memory_space=parameter.type.memory_space or "ub",
                    config=parameter.type.config or TileConfig(),
                )
            )
        return tuple(tile_bindings)

    def _materialize_inline_proc_specialization(
        self,
        name: str,
        args: tuple[SemanticExpr, ...],
        *,
        internal: bool = False,
    ) -> SemanticKernel:
        inline_proc_nodes = (
            self._internal_inline_proc_nodes if internal else self._inline_proc_nodes
        )
        inline_proc_node = inline_proc_nodes.get(name)
        if inline_proc_node is None:
            raise TypeError(f"inline_proc `{name}` is not registered in the current TileLang module")

        key = self._inline_proc_specialization_key(name, args, internal=internal)
        existing = self._inline_proc_specializations.get(key)
        if existing is not None:
            return existing
        if key in self._inline_proc_active_stack:
            raise TypeError(
                f"recursive inline_proc call `{name}` is not supported in TileLang DSL v1"
            )

        if len(inline_proc_node.parameters) != len(args):
            raise TypeError(
                f"inline_proc `{name}` expects {len(inline_proc_node.parameters)} arguments in TileLang DSL v1"
            )

        helper_env: dict[str, SemanticBinding] = {}
        helper_parameters: list[SemanticParameter] = []
        for index, (param, arg_expr) in enumerate(zip(inline_proc_node.parameters, args)):
            binding = SemanticBinding(
                name=param.name,
                ssa_name=f"%arg{index}",
                type=arg_expr.type,
                origin="inline_param",
                value=self._inline_proc_bound_static_value(arg_expr),
            )
            helper_env[param.name] = binding
            helper_parameters.append(SemanticParameter(binding=binding))

        saved_hidden_parameters = self._hidden_parameters
        self._hidden_parameters = []
        self._inline_proc_active_stack.append(key)
        try:
            body, _ = self._analyze_block(
                inline_proc_node.body,
                helper_env,
                allow_outer_lookup=False,
            )
        finally:
            self._inline_proc_active_stack.pop()
        helper_hidden_parameters = tuple(self._hidden_parameters)
        self._hidden_parameters = saved_hidden_parameters

        if helper_hidden_parameters:
            raise TypeError(
                f"inline_proc `{name}` currently does not support dynamic shape metadata captures in TileLang DSL v1"
            )

        return_type: SemanticType | None = None
        if body and isinstance(body[-1], SemanticReturnStmt):
            return_type = None if body[-1].value is None else body[-1].value.type

        helper_index = len(self._inline_proc_order)
        helper_kernel = SemanticKernel(
            target=self.node.target,
            op=self.node.op,
            symbol_name=self._inline_proc_symbol_name(name, helper_index),
            kernel_family=self.node.kernel_family,
            verify_enabled=False,
            advanced_enabled=self.node.advanced_enabled,
            dtype_signature=self.node.dtype_signature,
            parameters=tuple(helper_parameters),
            tile_bindings=self._collect_inline_helper_tile_bindings(tuple(helper_parameters)),
            body=body,
            inline_helpers=(),
        )
        self._inline_proc_specializations[key] = helper_kernel
        self._inline_proc_return_types[key] = return_type
        self._inline_proc_order.append(key)
        return helper_kernel

    def _analyze_inline_proc_call_expr(
        self,
        name: str,
        args: tuple[SemanticExpr, ...],
    ) -> SemanticExpr:
        helper_kernel = self._materialize_inline_proc_specialization(name, args)
        key = self._inline_proc_specialization_key(name, args)
        return SemanticCallExpr(
            namespace=None,
            name=helper_kernel.symbol_name,
            args=args,
            type=self._inline_proc_return_types.get(key),
        )

    def _analyze_internal_inline_proc_call_expr(
        self,
        name: str,
        args: tuple[SemanticExpr, ...],
    ) -> SemanticExpr:
        helper_kernel = self._materialize_inline_proc_specialization(
            name,
            args,
            internal=True,
        )
        key = self._inline_proc_specialization_key(name, args, internal=True)
        return SemanticCallExpr(
            namespace=None,
            name=helper_kernel.symbol_name,
            args=args,
            type=self._inline_proc_return_types.get(key),
        )

    def _is_internal_inline_proc_context(self) -> bool:
        return any(key[0].startswith("__internal__::") for key in self._inline_proc_active_stack)

    def _analyze_explicit_vecscope(
        self,
        stmt: FrontendVecscopeStmt,
        env: dict[str, SemanticBinding],
        *,
        allow_outer_lookup: bool,
    ) -> tuple[SemanticStmt, dict[str, SemanticBinding]]:
        body, updated_env = self._analyze_block(
            stmt.body,
            dict(env),
            allow_outer_lookup=allow_outer_lookup,
        )
        return SemanticVecscopeStmt(body=body), updated_env

    def _is_vector_store_call(self, expr: FrontendExprNode) -> bool:
        return (
            isinstance(expr, FrontendCallExpr)
            and expr.namespace == "pto"
            and expr.name in {"psts", "pst", "psti", "vsst", "vsta", "vstas", "vstar", "vscatter", "vsts", "vstsx2"}
        )

    def _is_scalar_store_call(self, expr: FrontendExprNode) -> bool:
        return (
            isinstance(expr, FrontendCallExpr)
            and expr.namespace == "pto"
            and expr.name == "store_scalar"
        )

    def _is_sync_call(self, expr: FrontendExprNode) -> bool:
        return (
            isinstance(expr, FrontendCallExpr)
            and expr.namespace == "pto"
            and expr.name in {
                "set_flag",
                "wait_flag",
                "pipe_barrier",
                "barrier",
                "get_buf",
                "rls_buf",
                "mem_bar",
                "set_cross_core",
                "set_intra_block",
                "set_intra_core",
                "wait_flag_dev",
                "wait_intra_core",
            }
        )

    def _is_ub_helper_call(self, expr: FrontendExprNode) -> bool:
        return (
            isinstance(expr, FrontendCallExpr)
            and expr.namespace == "pto"
            and expr.name in _UB_HELPER_OPS
        )

    def _is_low_level_dma_call(self, expr: FrontendExprNode) -> bool:
        return (
            isinstance(expr, FrontendCallExpr)
            and expr.namespace == "pto"
            and expr.name in _LOW_LEVEL_DMA_UNARY_CONFIG_OPS | _LOW_LEVEL_DMA_CONFIG_OPS | _LOW_LEVEL_DMA_COPY_OPS
        )

    def _is_grouped_mte_dma_call(self, expr: FrontendExprNode) -> bool:
        return (
            isinstance(expr, FrontendCallExpr)
            and expr.namespace == "pto"
            and expr.name in _GROUPED_MTE_DMA_OPS
        )

    def _analyze_vector_store_stmt(
        self,
        expr: FrontendCallExpr,
        env: dict[str, SemanticBinding],
        *,
        allow_outer_lookup: bool,
    ) -> tuple[SemanticStmt, dict[str, SemanticBinding]]:
        if expr.name in {"psts", "pst", "psti"}:
            canonical_name = "psts" if expr.name == "pst" else expr.name
            if len(expr.args) in {2, 3} and isinstance(expr.args[1], FrontendSubscriptExpr):
                raise TypeError(
                    f"pto.{expr.name} does not support Tile element-indexing syntax in TileLang DSL v1; "
                    f"use explicit pointer form `pto.{expr.name}(mask, buf, offset[, dist])`"
                )

            args = tuple(
                self._analyze_expr(arg, env, allow_outer_lookup=allow_outer_lookup)
                for arg in expr.args
            )
            dist_expr: SemanticExpr | None = None
            if len(args) == 3:
                value, destination, offset = args
                indices = (offset,)
            elif len(args) == 4:
                value, destination, offset, dist_expr = args
                indices = (offset,)
            else:
                raise TypeError(
                    f"pto.{expr.name} expects 3 or 4 positional arguments in TileLang DSL v1: "
                    f"`pto.{expr.name}(mask, buf, offset[, dist])`"
                )
            self._require_mask_expr(value, f"pto.{expr.name} value")
            self._require_vector_pointer_expr(destination, f"pto.{expr.name} destination")
            normalized_indices = []
            for index in indices:
                if expr.name == "psti":
                    self._require_i32_like_expr(index, "pto.psti offset")
                else:
                    index = self._require_index_typed_expr(index)
                normalized_indices.append(index)
            indices = tuple(normalized_indices)
            dist = self._normalize_predicate_store_dist(dist_expr, f"pto.{expr.name} dist")
            return (
                SemanticPredicateStoreStmt(
                    op_name=canonical_name,
                    value=value,
                    destination=destination,
                    indices=indices,
                    dist=dist,
                ),
                dict(env),
            )

        if expr.name in {"vsta", "vstas", "vstar"}:
            offset: SemanticExpr | None = None
            op_name = "vstas" if expr.name == "vsta" else expr.name
            if expr.name == "vsta":
                if len(expr.args) == 2:
                    value = self._analyze_expr(expr.args[0], env, allow_outer_lookup=allow_outer_lookup)
                    destination, indices = self._analyze_tile_vector_access(
                        expr.args[1],
                        env,
                        allow_outer_lookup=allow_outer_lookup,
                        context="pto.vsta destination",
                    )
                    offset = SemanticLiteralExpr(value=0, type=SemanticScalarType(dtype=i32))
                else:
                    args = tuple(
                        self._analyze_expr(arg, env, allow_outer_lookup=allow_outer_lookup)
                        for arg in expr.args
                    )
                    if len(args) != 3:
                        raise TypeError("pto.vsta expects 2 or 3 positional arguments in TileLang DSL v1")
                    value, destination, offset = args
                    indices = ()
            elif expr.name == "vstas":
                if len(expr.args) == 3 and isinstance(expr.args[1], FrontendSubscriptExpr):
                    value = self._analyze_expr(expr.args[0], env, allow_outer_lookup=allow_outer_lookup)
                    destination, indices = self._analyze_tile_vector_access(
                        expr.args[1],
                        env,
                        allow_outer_lookup=allow_outer_lookup,
                        context="pto.vstas destination",
                    )
                    offset = self._analyze_expr(expr.args[2], env, allow_outer_lookup=allow_outer_lookup)
                else:
                    args = tuple(
                        self._analyze_expr(arg, env, allow_outer_lookup=allow_outer_lookup)
                        for arg in expr.args
                    )
                    if len(args) != 3:
                        raise TypeError("pto.vstas expects exactly 3 positional arguments in TileLang DSL v1")
                    value, destination, offset = args
                    indices = ()
            else:
                if len(expr.args) == 2 and isinstance(expr.args[1], FrontendSubscriptExpr):
                    value = self._analyze_expr(expr.args[0], env, allow_outer_lookup=allow_outer_lookup)
                    destination, indices = self._analyze_tile_vector_access(
                        expr.args[1],
                        env,
                        allow_outer_lookup=allow_outer_lookup,
                        context="pto.vstar destination",
                    )
                else:
                    args = tuple(
                        self._analyze_expr(arg, env, allow_outer_lookup=allow_outer_lookup)
                        for arg in expr.args
                    )
                    if len(args) != 2:
                        raise TypeError("pto.vstar expects exactly 2 positional arguments in TileLang DSL v1")
                    value, destination = args
                    indices = ()
            self._require_align_expr(value, f"pto.{expr.name} value")
            self._require_vector_pointer_expr(destination, f"pto.{expr.name} destination")
            indices = tuple(self._require_index_typed_expr(index) for index in indices)
            if offset is not None:
                self._require_i32_like_expr(offset, f"pto.{expr.name} offset")
            return (
                SemanticAlignStoreStmt(
                    op_name=op_name,
                    value=value,
                    destination=destination,
                    indices=indices,
                    offset=offset,
                ),
                dict(env),
            )

        if expr.name == "vscatter":
            args = tuple(
                self._analyze_expr(arg, env, allow_outer_lookup=allow_outer_lookup)
                for arg in expr.args
            )
            if len(args) != 4:
                raise TypeError("pto.vscatter expects exactly 4 positional arguments in TileLang DSL v1")
            value, destination, offsets, mask = args
            value_type = self._require_vreg_expr(value, "pto.vscatter value")
            self._require_vector_pointer_expr(destination, "pto.vscatter destination")
            offsets_type = self._require_vreg_expr(offsets, "pto.vscatter offsets")
            if not is_integer_dtype(offsets_type.element_dtype):
                raise TypeError("pto.vscatter offsets must use an integer vector type in TileLang DSL v1")
            if integer_bitwidth(offsets_type.element_dtype) != 32:
                raise TypeError("pto.vscatter currently requires i32 offset vectors in TileLang DSL v1")
            if value_type.lanes != offsets_type.lanes:
                raise TypeError("pto.vscatter value and offsets must use the same lane count in TileLang DSL v1")
            self._require_matching_vector_pointer(value_type, destination.type, "pto.vscatter")
            self._require_mask_for_vreg(mask, value_type, "pto.vscatter")
            return (
                SemanticVScatterStmt(
                    value=value,
                    destination=destination,
                    offsets=offsets,
                    mask=mask,
                ),
                dict(env),
            )

        if expr.name == "vsst":
            if len(expr.args) == 3:
                scalar = self._analyze_expr(expr.args[0], env, allow_outer_lookup=allow_outer_lookup)
                destination, indices = self._analyze_tile_vector_access(
                    expr.args[1],
                    env,
                    allow_outer_lookup=allow_outer_lookup,
                    context="pto.vsst destination",
                )
                mask = self._analyze_expr(expr.args[2], env, allow_outer_lookup=allow_outer_lookup)
            else:
                args = tuple(
                    self._analyze_expr(arg, env, allow_outer_lookup=allow_outer_lookup)
                    for arg in expr.args
                )
                if len(args) != 4:
                    raise TypeError("pto.vsst expects 3 or 4 positional arguments in TileLang DSL v1")
                scalar, destination, offset, mask = args
                indices = (offset,)
            scalar_type = self._require_scalar_expr(scalar, "pto.vsst scalar")
            self._require_vector_pointer_expr(destination, "pto.vsst destination")
            indices = tuple(self._require_index_typed_expr(index) for index in indices)
            destination_dtype = destination.type.element_dtype
            if scalar_type.dtype != destination_dtype:
                raise TypeError("pto.vsst scalar dtype must match destination element dtype in TileLang DSL v1")
            value = SemanticCallExpr(
                namespace="pto",
                name="vbr",
                args=(scalar,),
                type=self._vreg_type_for_dtype(destination_dtype),
            )
            self._require_mask_for_vreg(mask, value.type, "pto.vsst")
            self._require_matching_vector_pointer(value.type, destination.type, "pto.vsst")
            return (
                SemanticVectorStoreStmt(
                    value=value,
                    destination=destination,
                    indices=indices,
                    dist=None,
                    mask=mask,
                ),
                dict(env),
            )

        if expr.name == "vsts":
            analyzed_keywords = {
                name: self._analyze_expr(value, env, allow_outer_lookup=allow_outer_lookup)
                for name, value in expr.keywords
            }
            unexpected_keywords = sorted(set(analyzed_keywords) - {"dist"})
            if unexpected_keywords:
                keyword_text = ", ".join(unexpected_keywords)
                raise TypeError(
                    "pto.vsts only accepts keyword attr `dist`; "
                    f"got unsupported keyword(s): {keyword_text}"
                )
            dist = self._normalize_vsts_dist(analyzed_keywords.get("dist"), "pto.vsts dist")
            if len(expr.args) == 3:
                value = self._analyze_expr(expr.args[0], env, allow_outer_lookup=allow_outer_lookup)
                destination, indices = self._analyze_tile_vector_access(
                    expr.args[1],
                    env,
                    allow_outer_lookup=allow_outer_lookup,
                    context="pto.vsts destination",
                )
                mask = self._analyze_expr(expr.args[2], env, allow_outer_lookup=allow_outer_lookup)
            else:
                args = tuple(
                    self._analyze_expr(arg, env, allow_outer_lookup=allow_outer_lookup)
                    for arg in expr.args
                )
                if len(args) != 4:
                    raise TypeError("pto.vsts expects 3 or 4 positional arguments in TileLang DSL v1")
                value, destination, offset, mask = args
                indices = (offset,)
            self._require_vreg_expr(value, "pto.vsts value")
            self._require_vector_pointer_expr(destination, "pto.vsts destination")
            indices = tuple(self._require_index_typed_expr(index) for index in indices)
            self._require_mask_for_vsts(mask, value.type, dist, "pto.vsts")
            self._require_matching_vector_pointer(value.type, destination.type, "pto.vsts")
            return (
                SemanticVectorStoreStmt(
                    value=value,
                    destination=destination,
                    indices=indices,
                    dist=dist,
                    mask=mask,
                ),
                dict(env),
            )

        if len(expr.args) == 5:
            low = self._analyze_expr(expr.args[0], env, allow_outer_lookup=allow_outer_lookup)
            high = self._analyze_expr(expr.args[1], env, allow_outer_lookup=allow_outer_lookup)
            destination, indices = self._analyze_tile_vector_access(
                expr.args[2],
                env,
                allow_outer_lookup=allow_outer_lookup,
                context="pto.vstsx2 destination",
            )
            dist = self._analyze_expr(expr.args[3], env, allow_outer_lookup=allow_outer_lookup)
            mask = self._analyze_expr(expr.args[4], env, allow_outer_lookup=allow_outer_lookup)
        else:
            args = tuple(
                self._analyze_expr(arg, env, allow_outer_lookup=allow_outer_lookup)
                for arg in expr.args
            )
            if len(args) != 6:
                raise TypeError("pto.vstsx2 expects 5 or 6 positional arguments in TileLang DSL v1")
            low, high, destination, offset, dist, mask = args
            indices = (offset,)
        low_type = self._require_vreg_expr(low, "pto.vstsx2 low")
        high_type = self._require_vreg_expr(high, "pto.vstsx2 high")
        if low_type != high_type:
            raise TypeError("pto.vstsx2 requires low/high vectors to use the same vector type")
        self._require_vector_pointer_expr(destination, "pto.vstsx2 destination")
        indices = tuple(self._require_index_typed_expr(index) for index in indices)
        dist = self._normalize_vstsx2_dist(dist)
        self._require_mask_for_vreg(mask, low_type, "pto.vstsx2")
        self._require_matching_vector_pointer(low_type, destination.type, "pto.vstsx2")
        return (
            SemanticVectorPairStoreStmt(
                low=low,
                high=high,
                destination=destination,
                indices=indices,
                dist=dist,
                mask=mask,
            ),
            dict(env),
        )

    def _analyze_scalar_store_stmt(
        self,
        expr: FrontendCallExpr,
        env: dict[str, SemanticBinding],
        *,
        allow_outer_lookup: bool,
    ) -> tuple[SemanticStmt, dict[str, SemanticBinding]]:
        args = tuple(
            self._analyze_expr(arg, env, allow_outer_lookup=allow_outer_lookup)
            for arg in expr.args
        )
        if len(args) != 3:
            raise TypeError("pto.store_scalar expects exactly 3 positional arguments in TileLang DSL v1")
        if isinstance(args[0].type, SemanticPtrType):
            destination = self._require_pointer_expr(args[0], "pto.store_scalar destination")
            offset = args[1]
            value = args[2]
        else:
            value = args[0]
            destination = self._require_pointer_expr(args[1], "pto.store_scalar destination")
            offset = args[2]
        offset = self._require_index_typed_expr(offset)
        value_type = self._require_scalar_expr(value, "pto.store_scalar value")
        if value_type.dtype != destination.type.element_dtype:
            raise TypeError("pto.store_scalar value dtype must match destination pointer element dtype")
        return (
            SemanticScalarStoreStmt(value=value, destination=destination, offset=offset),
            dict(env),
        )

    def _analyze_sync_stmt(
        self,
        expr: FrontendCallExpr,
        env: dict[str, SemanticBinding],
        *,
        allow_outer_lookup: bool,
    ) -> tuple[SemanticStmt, dict[str, SemanticBinding]]:
        args = tuple(
            self._analyze_expr(arg, env, allow_outer_lookup=allow_outer_lookup)
            for arg in expr.args
        )
        if expr.name in {"set_flag", "wait_flag"}:
            if len(args) != 3:
                raise TypeError(f"pto.{expr.name} expects exactly 3 positional arguments in TileLang DSL v1")
            src_pipe = self._require_sync_pipe(args[0], f"pto.{expr.name} source pipe")
            dst_pipe = self._require_sync_pipe(args[1], f"pto.{expr.name} destination pipe")
            event = self._require_sync_event(args[2], f"pto.{expr.name} event")
            if expr.name == "set_flag":
                return SemanticSetFlagStmt(src_pipe=src_pipe, dst_pipe=dst_pipe, event=event), dict(env)
            return SemanticWaitFlagStmt(src_pipe=src_pipe, dst_pipe=dst_pipe, event=event), dict(env)
        if expr.name in {"get_buf", "rls_buf"}:
            if len(args) not in {2, 3}:
                raise TypeError(f"pto.{expr.name} expects 2 or 3 positional arguments in TileLang DSL v1")
            pipe = self._require_sync_pipe(args[0], f"pto.{expr.name} pipe")
            self._require_i64_like_expr(args[1], f"pto.{expr.name} buf_id")
            mode = args[2] if len(args) == 3 else SemanticLiteralExpr(value=0, type=SemanticScalarType(dtype=i64))
            self._require_i64_like_expr(mode, f"pto.{expr.name} mode")
            if expr.name == "get_buf":
                return SemanticGetBufStmt(pipe=pipe, buf_id=args[1], mode=mode), dict(env)
            return SemanticRlsBufStmt(pipe=pipe, buf_id=args[1], mode=mode), dict(env)
        if expr.name == "mem_bar":
            if len(args) != 1:
                raise TypeError("pto.mem_bar expects exactly 1 positional argument in TileLang DSL v1")
            barrier_type = self._require_barrier_type(args[0], "pto.mem_bar barrier_type")
            return SemanticMemBarStmt(barrier_type=barrier_type), dict(env)
        if expr.name in {"set_cross_core", "set_intra_block", "wait_flag_dev", "wait_intra_core"}:
            if len(args) != 2:
                raise TypeError(f"pto.{expr.name} expects exactly 2 positional arguments in TileLang DSL v1")
            identifier = self._require_scalar_or_index_expr(args[0], f"pto.{expr.name} first operand")
            self._require_i64_like_expr(identifier, f"pto.{expr.name} first operand")
            event_id = self._normalize_event_id_expr(args[1], f"pto.{expr.name} event_id")
            if expr.name == "set_cross_core":
                return SemanticSetCrossCoreStmt(core_id=identifier, event_id=event_id), dict(env)
            if expr.name == "set_intra_block":
                return SemanticSetIntraBlockStmt(block_id=identifier, event_id=event_id), dict(env)
            if expr.name == "wait_flag_dev":
                return SemanticWaitFlagDevStmt(core_id=identifier, event_id=event_id), dict(env)
            return SemanticWaitIntraCoreStmt(block_id=identifier, event_id=event_id), dict(env)
        if expr.name == "set_intra_core":
            if len(args) != 1:
                raise TypeError("pto.set_intra_core expects exactly 1 positional argument in TileLang DSL v1")
            config = self._require_scalar_or_index_expr(args[0], "pto.set_intra_core config")
            self._require_i32_like_expr(config, "pto.set_intra_core config")
            return SemanticSetIntraCoreStmt(config=config), dict(env)
        if expr.name in {"pipe_barrier", "barrier"}:
            if len(args) != 1:
                raise TypeError(f"pto.{expr.name} expects exactly 1 positional argument in TileLang DSL v1")
            pipe = self._require_sync_pipe(args[0], f"pto.{expr.name} pipe")
            return SemanticPipeBarrierStmt(pipe=pipe), dict(env)
        raise ValueError(f"unsupported sync stmt pto.{expr.name}")

    def _analyze_low_level_dma_stmt(
        self,
        expr: FrontendCallExpr,
        env: dict[str, SemanticBinding],
        *,
        allow_outer_lookup: bool,
    ) -> tuple[SemanticStmt, dict[str, SemanticBinding]]:
        args = self._analyze_low_level_dma_operands(
            expr,
            env,
            allow_outer_lookup=allow_outer_lookup,
        )
        if expr.name in _LOW_LEVEL_DMA_UNARY_CONFIG_OPS:
            if len(args) != 1:
                raise TypeError(f"pto.{expr.name} expects exactly 1 positional argument in TileLang DSL")
            scalar = self._require_scalar_expr(args[0], f"pto.{expr.name} pad_value")
            if not _is_supported_mov_pad_scalar_dtype(scalar.dtype):
                raise TypeError(
                    "pto.set_mov_pad_val pad_value must be an 8/16/32-bit integer or f16/bf16/f32 in TileLang DSL v1"
                )
            return (
                SemanticDmaUnaryConfigStmt(
                    name=expr.name,
                    value=args[0],
                ),
                dict(env),
            )
        if expr.name in _LOW_LEVEL_DMA_CONFIG_OPS:
            if len(args) != 2:
                raise TypeError(f"pto.{expr.name} expects exactly 2 positional arguments in TileLang DSL")
            self._require_i64_like_expr(args[0], f"pto.{expr.name} first operand")
            self._require_i64_like_expr(args[1], f"pto.{expr.name} second operand")
            return (
                SemanticDmaConfigStmt(
                    name=expr.name,
                    first=args[0],
                    second=args[1],
                ),
                dict(env),
            )
        if expr.name == "copy_gm_to_ubuf":
            if len(args) != 11:
                raise TypeError("pto.copy_gm_to_ubuf expects exactly 11 positional arguments in TileLang DSL")
            source = self._require_pointer_expr(args[0], "pto.copy_gm_to_ubuf source", memory_space="gm")
            destination = self._require_pointer_expr(args[1], "pto.copy_gm_to_ubuf destination", memory_space="ub")
            for operand, label in zip(
                args[2:7] + args[8:],
                (
                    "sid",
                    "n_burst",
                    "len_burst",
                    "left_padding_count",
                    "right_padding_count",
                    "l2_cache_ctl",
                    "gm_stride",
                    "ub_stride",
                ),
            ):
                self._require_i64_like_expr(operand, f"pto.copy_gm_to_ubuf {label}")
            self._require_i1_expr(args[7], "pto.copy_gm_to_ubuf data_select_bit")
            return (
                SemanticLowLevelCopyStmt(
                    name=expr.name,
                    source=source,
                    destination=destination,
                    operands=args[2:],
                ),
                dict(env),
            )
        if expr.name == "copy_ubuf_to_gm":
            if len(args) != 8:
                raise TypeError("pto.copy_ubuf_to_gm expects exactly 8 positional arguments in TileLang DSL")
            source = self._require_pointer_expr(args[0], "pto.copy_ubuf_to_gm source", memory_space="ub")
            destination = self._require_pointer_expr(args[1], "pto.copy_ubuf_to_gm destination", memory_space="gm")
            for operand, label in zip(
                args[2:],
                (
                    "sid",
                    "n_burst",
                    "len_burst",
                    "reserved",
                    "burst_dst_stride",
                    "burst_src_stride",
                ),
            ):
                self._require_i64_like_expr(operand, f"pto.copy_ubuf_to_gm {label}")
            return (
                SemanticLowLevelCopyStmt(
                    name=expr.name,
                    source=source,
                    destination=destination,
                    operands=args[2:],
                ),
                dict(env),
            )
        if expr.name == "copy_ubuf_to_ubuf":
            if len(args) != 7:
                raise TypeError("pto.copy_ubuf_to_ubuf expects exactly 7 positional arguments in TileLang DSL")
            source = self._require_pointer_expr(args[0], "pto.copy_ubuf_to_ubuf source", memory_space="ub")
            destination = self._require_pointer_expr(args[1], "pto.copy_ubuf_to_ubuf destination", memory_space="ub")
            for operand, label in zip(
                args[2:],
                ("sid", "n_burst", "len_burst", "src_stride", "dst_stride"),
            ):
                self._require_i64_like_expr(operand, f"pto.copy_ubuf_to_ubuf {label}")
            return (
                SemanticLowLevelCopyStmt(
                    name=expr.name,
                    source=source,
                    destination=destination,
                    operands=args[2:],
                ),
                dict(env),
            )
        raise ValueError(f"unsupported low-level DMA stmt pto.{expr.name}")

    def _analyze_grouped_mte_dma_stmt(
        self,
        expr: FrontendCallExpr,
        env: dict[str, SemanticBinding],
        *,
        allow_outer_lookup: bool,
    ) -> tuple[SemanticStmt, dict[str, SemanticBinding]]:
        semantic_expr = self._analyze_grouped_mte_dma_expr(
            expr,
            env,
            allow_outer_lookup=allow_outer_lookup,
        )
        return SemanticGroupedMteDmaStmt(name=expr.name, args=semantic_expr.args), dict(env)

    def _analyze_grouped_mte_dma_expr(
        self,
        expr: FrontendCallExpr,
        env: dict[str, SemanticBinding],
        *,
        allow_outer_lookup: bool,
    ) -> SemanticCallExpr:
        args = tuple(
            self._analyze_expr(arg, env, allow_outer_lookup=allow_outer_lookup)
            for arg in expr.args
        )
        keywords = self._analyze_keyword_args(
            expr.keywords,
            env,
            allow_outer_lookup=allow_outer_lookup,
            context=f"pto.{expr.name}",
        )
        if expr.name == "mte_gm_ub":
            return self._analyze_mte_gm_ub(args, keywords)
        if expr.name == "mte_ub_gm":
            return self._analyze_mte_ub_gm(args, keywords)
        if expr.name == "mte_ub_ub":
            return self._analyze_mte_ub_copy(expr.name, args, keywords, destination_space="ub")
        if expr.name == "mte_ub_l1":
            return self._analyze_mte_ub_copy(expr.name, args, keywords, destination_space="mat")
        raise ValueError(f"unsupported grouped MTE DMA stmt pto.{expr.name}")

    def _analyze_low_level_dma_operands(
        self,
        expr: FrontendCallExpr,
        env: dict[str, SemanticBinding],
        *,
        allow_outer_lookup: bool,
    ) -> tuple[SemanticExpr, ...]:
        if expr.args and expr.keywords:
            raise TypeError(
                f"pto.{expr.name} does not support mixing positional and keyword operands in TileLang DSL v1"
            )
        if not expr.keywords:
            return tuple(
                self._analyze_expr(arg, env, allow_outer_lookup=allow_outer_lookup)
                for arg in expr.args
            )

        analyzed_keywords: dict[str, SemanticExpr] = {
            name: self._analyze_expr(value, env, allow_outer_lookup=allow_outer_lookup)
            for name, value in expr.keywords
        }

        def index_literal(value: int) -> SemanticLiteralExpr:
            return SemanticLiteralExpr(value=value, type=SemanticIndexType())

        def bool_literal(value: bool) -> SemanticLiteralExpr:
            return SemanticLiteralExpr(value=value, type=SemanticScalarType(dtype=i1))

        if expr.name == "set_mov_pad_val":
            return (analyzed_keywords["pad_value"],)
        if expr.name in {
            "set_loop2_stride_outtoub",
            "set_loop1_stride_outtoub",
            "set_loop2_stride_ubtoout",
            "set_loop1_stride_ubtoout",
        }:
            return (
                analyzed_keywords["src_stride"],
                analyzed_keywords["dst_stride"],
            )
        if expr.name in {"set_loop_size_outtoub", "set_loop_size_ubtoout"}:
            return (
                analyzed_keywords["loop1"],
                analyzed_keywords["loop2"],
            )
        if expr.name == "copy_gm_to_ubuf":
            if "data_select_bit" in analyzed_keywords and "enable_ub_pad" in analyzed_keywords:
                raise TypeError(
                    "pto.copy_gm_to_ubuf keyword form accepts either `data_select_bit` or `enable_ub_pad`, not both"
                )
            return (
                analyzed_keywords["src"],
                analyzed_keywords["dst"],
                analyzed_keywords.get("sid", index_literal(0)),
                analyzed_keywords["n_burst"],
                analyzed_keywords["len_burst"],
                analyzed_keywords.get("left_padding_count", index_literal(0)),
                analyzed_keywords.get("right_padding_count", index_literal(0)),
                analyzed_keywords.get(
                    "data_select_bit",
                    analyzed_keywords.get("enable_ub_pad", bool_literal(False)),
                ),
                analyzed_keywords.get("l2_cache_ctl", index_literal(0)),
                analyzed_keywords["gm_stride"],
                analyzed_keywords["ub_stride"],
            )
        if expr.name == "copy_ubuf_to_gm":
            if "burst_dst_stride" in analyzed_keywords and "gm_stride" in analyzed_keywords:
                raise TypeError(
                    "pto.copy_ubuf_to_gm keyword form accepts either `burst_dst_stride` or `gm_stride`, not both"
                )
            if "burst_src_stride" in analyzed_keywords and "ub_stride" in analyzed_keywords:
                raise TypeError(
                    "pto.copy_ubuf_to_gm keyword form accepts either `burst_src_stride` or `ub_stride`, not both"
                )
            return (
                analyzed_keywords["src"],
                analyzed_keywords["dst"],
                analyzed_keywords.get("sid", index_literal(0)),
                analyzed_keywords["n_burst"],
                analyzed_keywords["len_burst"],
                analyzed_keywords.get("reserved", index_literal(0)),
                analyzed_keywords.get(
                    "burst_dst_stride",
                    analyzed_keywords["gm_stride"],
                ),
                analyzed_keywords.get(
                    "burst_src_stride",
                    analyzed_keywords["ub_stride"],
                ),
            )
        raise TypeError(
            f"pto.{expr.name} keyword form is not implemented in TileLang DSL v1"
        )

    def _require_tensor_slice(
        self,
        expr: SemanticExpr,
        context: str,
    ) -> SemanticTensorSliceExpr:
        if not isinstance(expr, SemanticTensorSliceExpr):
            raise TypeError(f"{context} must be a TensorView slice in TileLang DSL v1")
        return expr

    def _require_tile_expr(self, expr: SemanticExpr, context: str) -> SemanticExpr:
        if not isinstance(expr.type, SemanticTileType):
            raise TypeError(f"{context} must be a Tile value in TileLang DSL v1")
        if expr.type.rank != 2:
            raise TypeError(f"{context} currently only supports rank-2 Tile values in TileLang DSL v1")
        if expr.type.shape is None:
            raise TypeError(f"{context} requires a statically specialized Tile shape in TileLang DSL v1")
        if expr.type.memory_space != "ub":
            raise TypeError(f"{context} currently only supports MemorySpace.UB Tile values in TileLang DSL v1")
        return expr

    def _require_pointer_expr(
        self,
        expr: SemanticExpr,
        context: str,
        *,
        memory_space: str | None = None,
    ) -> SemanticExpr:
        if not isinstance(expr.type, SemanticPtrType):
            raise TypeError(f"{context} must be a pointer value in TileLang DSL")
        if memory_space is not None and expr.type.memory_space != memory_space:
            raise TypeError(f"{context} requires MemorySpace.{memory_space.upper()} pointers in TileLang DSL")
        return expr

    def _require_vector_pointer_expr(self, expr: SemanticExpr, context: str) -> SemanticExpr:
        if isinstance(expr.type, SemanticTileType):
            return self._require_tile_expr(expr, context)
        return self._require_pointer_expr(expr, context, memory_space="ub")

    def _bind_assignment_target(
        self,
        target: FrontendTargetNode,
        value: SemanticExpr,
        env: dict[str, SemanticBinding],
        annotation: Any | None,
    ) -> tuple[SemanticBinding, ...]:
        if isinstance(target, FrontendNameTarget):
            if isinstance(value.type, SemanticTupleType):
                raise ValueError("multi-result call assignment requires tuple binding in TileLang DSL v1")
            inferred_type: SemanticType = value.type
            if isinstance(value.type, SemanticTensorSliceType):
                # Tensor slicing materializes a logical partition descriptor value in IR.
                inferred_type = SemanticPartitionTensorViewType(
                    element_dtype=value.type.element_dtype,
                    rank=value.type.rank,
                )
            annotated_type = self._annotation_type(annotation, inferred_type, env)
            binding = self._make_binding(
                target.name,
                annotated_type if annotated_type is not None else inferred_type,
                "ssa",
                value=self._binding_value_for_expr(value),
            )
            env[target.name] = binding
            return (binding,)
        if isinstance(target, FrontendTupleTarget):
            if isinstance(value.type, SemanticTupleType):
                element_types = value.type.elements
            elif isinstance(value.type, SemanticShapeType):
                element_types = tuple(SemanticIndexType() for _ in range(value.type.rank))
            else:
                raise ValueError("tuple assignment expects a tuple-typed value")
            if annotation is not None:
                raise TypeError("annotated tuple assignment is not supported in TileLang DSL v1")
            if len(target.elements) != len(element_types):
                raise ValueError("tuple assignment arity must match the tuple value")
            tuple_values: tuple[SemanticExpr, ...]
            if isinstance(value, SemanticTupleExpr):
                tuple_values = value.elements
            elif isinstance(value, SemanticAttributeAccess) and isinstance(value.type, SemanticShapeType):
                if isinstance(value.base, SemanticBindingRef):
                    if isinstance(value.base.type, SemanticTileType) and value.attr == "valid_shape":
                        valid_shape = value.base.type.valid_shape
                        if valid_shape is not None:
                            for axis, dim in enumerate(valid_shape):
                                if dim is None:
                                    self._ensure_tile_valid_shape_parameter(value.base.binding, axis)
                tuple_values = tuple(
                    SemanticSubscriptAccess(
                        base=value,
                        index=SemanticLiteralExpr(value=axis, type=SemanticIndexType()),
                        type=SemanticIndexType(),
                    )
                    for axis in range(value.type.rank)
                )
            elif isinstance(value, SemanticCallExpr):
                if len(value.args) == len(element_types):
                    tuple_values = value.args
                else:
                    tuple_values = tuple(
                        SemanticLiteralExpr(value=None, type=element_type) for element_type in element_types
                    )
            else:
                tuple_values = tuple(
                    SemanticLiteralExpr(value=None, type=element_type) for element_type in element_types
                )
            bindings = []
            for element, element_type, element_value in zip(target.elements, element_types, tuple_values):
                binding = self._make_binding(
                    element.name,
                    element_type,
                    "ssa",
                    value=self._binding_value_for_expr(element_value),
                )
                env[element.name] = binding
                bindings.append(binding)
            return tuple(bindings)
        raise ValueError(f"unsupported frontend assignment target {type(target).__name__}")

    def _binding_value_for_expr(self, expr: SemanticExpr) -> Any | None:
        return self._try_static_value(expr)

    def _annotation_type(
        self,
        annotation: Any | None,
        inferred_type: SemanticType | None,
        env: dict[str, SemanticBinding],
    ) -> SemanticType | None:
        if annotation is None:
            return inferred_type
        annotation_expr = self._analyze_annotation_expr(annotation, env)
        if isinstance(annotation_expr.type, SemanticMetaType):
            if annotation_expr.type.kind == "dtype" and isinstance(inferred_type, SemanticScalarType):
                dtype = self._require_dtype_symbol(annotation_expr, "annotated scalar type")
                if inferred_type.dtype != dtype:
                    raise TypeError(
                        f"annotated scalar type `{dtype!r}` does not match inferred {inferred_type.dtype!r}"
                    )
                return inferred_type
            if annotation_expr.type.kind == "ptr_type" and isinstance(inferred_type, SemanticPtrType):
                ptr_type = self._require_ptr_type_expr(annotation_expr, "annotated pointer type")
                if inferred_type.element_dtype != ptr_type.element_dtype:
                    raise TypeError(
                        f"annotated pointer type `{ptr_type!r}` does not match inferred pointer element type {inferred_type.element_dtype!r}"
                    )
                if inferred_type.memory_space != ptr_type.memory_space.value:
                    raise TypeError(
                        f"annotated pointer type `{ptr_type!r}` does not match inferred pointer memory space `{inferred_type.memory_space}`"
                    )
                return inferred_type
            if annotation_expr.type.kind == "vreg_type" and isinstance(inferred_type, SemanticVRegType):
                vreg_type = self._require_vreg_type_expr(annotation_expr, "annotated vector type")
                if inferred_type.element_dtype != vreg_type.element_dtype or inferred_type.lanes != vreg_type.lanes:
                    raise TypeError(
                        f"annotated vector type `{vreg_type!r}` does not match inferred !pto.vreg<{inferred_type.lanes}x{inferred_type.element_dtype.name}>"
                    )
                return inferred_type
            if annotation_expr.type.kind == "vector_type" and isinstance(inferred_type, SemanticVectorType):
                vector_type = self._require_vector_type_expr(annotation_expr, "annotated builtin vector type")
                if (
                    inferred_type.element_dtype != vector_type.element_dtype
                    or inferred_type.shape != vector_type.shape
                ):
                    shape_text = "x".join(str(dim) for dim in inferred_type.shape)
                    raise TypeError(
                        f"annotated builtin vector type `{vector_type!r}` does not match inferred !pto.vector<{shape_text}x{inferred_type.element_dtype.name}>"
                    )
                return inferred_type
            if annotation_expr.type.kind == "mask_type" and isinstance(inferred_type, SemanticMaskType):
                mask_type = self._require_mask_type_expr(annotation_expr, "annotated mask type")
                if inferred_type.granularity != mask_type.granularity:
                    raise TypeError(
                        f"annotated mask type `{mask_type!r}` does not match inferred !pto.mask<{inferred_type.granularity}>"
                    )
                return inferred_type
            if annotation_expr.type.kind == "align_type" and isinstance(inferred_type, SemanticAlignType):
                return inferred_type
            if (
                annotation_expr.type.kind == "partition_tensor_view_type"
                and isinstance(inferred_type, SemanticPartitionTensorViewType)
            ):
                return inferred_type
        raise TypeError("unsupported annotated assignment type in TileLang DSL v1")

    def _analyze_annotation_expr(
        self,
        annotation: ast.AST,
        env: dict[str, SemanticBinding],
    ) -> SemanticExpr:
        frontend_expr = self._build_frontend_annotation_expr(annotation)
        return self._analyze_expr(frontend_expr, env, allow_outer_lookup=True)

    def _build_frontend_annotation_expr(self, node: ast.AST) -> FrontendExprNode:
        if isinstance(node, ast.Name):
            return FrontendNameExpr(name=node.id)
        if isinstance(node, ast.Constant):
            return FrontendConstantExpr(value=node.value)
        if isinstance(node, ast.Attribute):
            path = self._annotation_attribute_path(node)
            if path is not None and path[0] in {"pto", "PAT", "PIPE", "EVENT"} and len(path) >= 2:
                return FrontendSymbolExpr(namespace=".".join(path[:-1]), name=path[-1])
            return FrontendAttributeExpr(
                base=self._build_frontend_annotation_expr(node.value),
                attr=node.attr,
            )
        if isinstance(node, ast.Call):
            if any(keyword.arg is None for keyword in node.keywords):
                raise TypeError("annotated assignment type does not support keyword unpacking in TileLang DSL v1")
            if node.keywords:
                raise TypeError("annotated assignment type does not support keyword arguments in TileLang DSL v1")
            if isinstance(node.func, ast.Name):
                return FrontendCallExpr(
                    namespace=None,
                    name=node.func.id,
                    args=tuple(self._build_frontend_annotation_expr(arg) for arg in node.args),
                    keywords=(),
                )
            if isinstance(node.func, ast.Attribute) and isinstance(node.func.value, ast.Name):
                return FrontendCallExpr(
                    namespace=node.func.value.id,
                    name=node.func.attr,
                    args=tuple(self._build_frontend_annotation_expr(arg) for arg in node.args),
                    keywords=(),
                )
        raise TypeError("unsupported annotated assignment type in TileLang DSL v1")

    def _annotation_attribute_path(self, node: ast.AST) -> tuple[str, ...] | None:
        if isinstance(node, ast.Name):
            return (node.id,)
        if isinstance(node, ast.Attribute):
            base_path = self._annotation_attribute_path(node.value)
            if base_path is None:
                return None
            return base_path + (node.attr,)
        return None

    def _analyze_for(
        self,
        stmt: FrontendForStmt,
        env: dict[str, SemanticBinding],
        *,
        allow_outer_lookup: bool,
    ) -> tuple[SemanticStmt, dict[str, SemanticBinding]]:
        lower_bound = self._analyze_expr(stmt.lower_bound, env, allow_outer_lookup=allow_outer_lookup)
        upper_bound = self._analyze_expr(stmt.upper_bound, env, allow_outer_lookup=allow_outer_lookup)
        step = self._analyze_expr(stmt.step, env, allow_outer_lookup=allow_outer_lookup)
        for expr in (lower_bound, upper_bound, step):
            self._require_loop_bound_type(expr.type)

        body_env = dict(env)
        induction_variable = self._make_binding(stmt.target, SemanticIndexType(), "loop_iv")
        body_env[stmt.target] = induction_variable
        body, final_body_env = self._analyze_block(
            stmt.body,
            body_env,
            allow_outer_lookup=allow_outer_lookup,
        )

        updated_env = dict(env)
        loop_carried = []
        for name, outer_binding in env.items():
            final_binding = final_body_env.get(name)
            if final_binding is None or final_binding is outer_binding:
                continue
            merged_type = self._merge_loop_carried_types(outer_binding.type, final_binding.type)
            if merged_type is None:
                raise TypeError(
                    f"loop-carried binding '{name}' changes type from {outer_binding.type!r} to {final_binding.type!r}"
                )
            merged = self._make_binding(name, merged_type, "loop_result")
            updated_env[name] = merged
            loop_carried.append(merged)

        return (
            SemanticForStmt(
                induction_variable=induction_variable,
                lower_bound=lower_bound,
                upper_bound=upper_bound,
                step=step,
                body=body,
                loop_carried=tuple(loop_carried),
            ),
            updated_env,
        )

    def _analyze_if(
        self,
        stmt: FrontendIfStmt,
        env: dict[str, SemanticBinding],
        *,
        allow_outer_lookup: bool,
    ) -> tuple[SemanticStmt, dict[str, SemanticBinding]]:
        condition = self._analyze_expr(stmt.condition, env, allow_outer_lookup=allow_outer_lookup)
        self._require_condition_type(condition.type)
        if self._contains_meta_condition_operand(condition):
            raise TypeError(
                "if condition comparing meta values requires wrapping the condition with pto.constexpr(...) "
                "in TileLang DSL v1"
            )

        then_body, then_env = self._analyze_block(
            stmt.then_body,
            dict(env),
            allow_outer_lookup=allow_outer_lookup,
        )
        else_body, else_env = self._analyze_block(
            stmt.else_body,
            dict(env),
            allow_outer_lookup=allow_outer_lookup,
        )

        updated_env = dict(env)
        merged_results: list[SemanticIfResult] = []
        merge_names = list(env)
        merge_names.extend(
            name
            for name in then_env
            if name not in env and name in else_env
        )
        for name in merge_names:
            outer_binding = env.get(name)
            then_binding = then_env.get(name, outer_binding)
            else_binding = else_env.get(name, outer_binding)
            if outer_binding is None:
                if then_binding is None or else_binding is None:
                    continue
            else:
                if then_binding is outer_binding and else_binding is outer_binding:
                    continue
            if then_binding is None or else_binding is None:
                continue
            if then_binding.type != else_binding.type:
                raise TypeError(
                    f"if/else merge for '{name}' changes type between branches: "
                    f"{then_binding.type!r} vs {else_binding.type!r}"
                )
            merged_binding = self._make_binding(name, then_binding.type, "if_result")
            updated_env[name] = merged_binding
            merged_results.append(
                SemanticIfResult(
                    result_binding=merged_binding,
                    then_binding=then_binding,
                    else_binding=else_binding,
                )
            )

        return (
            SemanticIfStmt(
                condition=condition,
                then_body=then_body,
                else_body=else_body,
                results=tuple(merged_results),
            ),
            updated_env,
        )

    def _contains_meta_condition_operand(self, expr: SemanticExpr) -> bool:
        if isinstance(expr, SemanticBinaryExpr):
            if expr.op in {"eq", "ne"} and (
                isinstance(expr.lhs.type, SemanticMetaType) or isinstance(expr.rhs.type, SemanticMetaType)
            ):
                return True
            if expr.op in {"and", "or"}:
                return self._contains_meta_condition_operand(expr.lhs) or self._contains_meta_condition_operand(expr.rhs)
        return False

    def _analyze_constexpr_if(
        self,
        stmt: FrontendIfStmt,
        env: dict[str, SemanticBinding],
        *,
        allow_outer_lookup: bool,
    ) -> tuple[tuple[SemanticStmt, ...], dict[str, SemanticBinding]]:
        condition = self._analyze_expr(stmt.condition, env, allow_outer_lookup=allow_outer_lookup)
        self._require_condition_type(condition.type)
        static_value = self._require_constexpr_condition_bool(
            condition,
            context="if pto.constexpr(...) condition",
        )
        selected_body = stmt.then_body if static_value else stmt.else_body
        return self._analyze_block(
            selected_body,
            dict(env),
            allow_outer_lookup=allow_outer_lookup,
        )

    def _analyze_strict_vecscope(
        self,
        stmt: FrontendStrictVecscopeStmt,
        env: dict[str, SemanticBinding],
    ) -> tuple[SemanticStmt, dict[str, SemanticBinding]]:
        if not self.node.advanced_enabled:
            raise TypeError(advanced_mode_message("strict_vecscope"))
        if len(stmt.captures) != len(stmt.block_arguments):
            raise ValueError("strict_vecscope capture arity must match block arguments")

        captures = tuple(
            self._analyze_expr(expr, env, allow_outer_lookup=True)
            for expr in stmt.captures
        )
        scope_env: dict[str, SemanticBinding] = {}
        block_arguments = []
        for name, capture in zip(stmt.block_arguments, captures):
            if capture.type is None:
                raise TypeError(
                    f"strict_vecscope block argument '{name}' type could not be inferred"
                )
            block_binding = self._make_binding(name, capture.type, "strict_vecscope_arg")
            scope_env[name] = block_binding
            block_arguments.append(block_binding)
        body, _ = self._analyze_block(
            stmt.body,
            scope_env,
            allow_outer_lookup=False,
        )
        return (
            SemanticStrictVecscopeStmt(
                captures=captures,
                block_arguments=tuple(block_arguments),
                body=body,
            ),
            dict(env),
        )

    def _analyze_expr(
        self,
        expr: FrontendExprNode,
        env: dict[str, SemanticBinding],
        *,
        allow_outer_lookup: bool,
    ) -> SemanticExpr:
        if isinstance(expr, FrontendNameExpr):
            binding = env.get(expr.name)
            if binding is None:
                if allow_outer_lookup:
                    raise ValueError(f"unknown name '{expr.name}'")
                raise ValueError(
                    f"implicit capture of '{expr.name}' is not allowed in pto.strict_vecscope"
                )
            return self._attach_expr_source_location(
                SemanticBindingRef(binding=binding, type=binding.type),
                expr,
            )
        if isinstance(expr, FrontendConstantExpr):
            if isinstance(expr.value, bool):
                return self._attach_expr_source_location(
                    SemanticLiteralExpr(value=expr.value, type=SemanticScalarType(dtype=i1)),
                    expr,
                )
            if isinstance(expr.value, int):
                return self._attach_expr_source_location(
                    SemanticLiteralExpr(value=expr.value, type=SemanticIndexType()),
                    expr,
                )
            if isinstance(expr.value, float):
                return self._attach_expr_source_location(
                    SemanticLiteralExpr(
                        value=expr.value,
                        type=SemanticScalarType(dtype=f32),
                    ),
                    expr,
                )
            if isinstance(expr.value, str):
                return self._attach_expr_source_location(
                    SemanticLiteralExpr(
                        value=expr.value,
                        type=SemanticMetaType(kind="string"),
                    ),
                    expr,
                )
            if expr.value is None:
                return self._attach_expr_source_location(
                    SemanticLiteralExpr(value=None, type=SemanticIndexType()),
                    expr,
                )
            raise TypeError(f"unsupported constant {expr.value!r} in TileLang DSL v1")
        if isinstance(expr, FrontendSymbolExpr):
            return self._attach_expr_source_location(
                self._analyze_symbol_expr(expr),
                expr,
            )
        if isinstance(expr, FrontendSliceExpr):
            start = None if expr.start is None else self._analyze_expr(expr.start, env, allow_outer_lookup=allow_outer_lookup)
            stop = None if expr.stop is None else self._analyze_expr(expr.stop, env, allow_outer_lookup=allow_outer_lookup)
            step = None if expr.step is None else self._analyze_expr(expr.step, env, allow_outer_lookup=allow_outer_lookup)
            if start is not None:
                start = self._require_index_typed_expr(start)
            if stop is not None:
                stop = self._require_index_typed_expr(stop)
            if step is not None:
                step = self._require_index_typed_expr(step)
            return self._attach_expr_source_location(
                SemanticSliceExpr(
                    start=start,
                    stop=stop,
                    step=step,
                    type=SemanticSliceType(),
                ),
                expr,
            )
        if isinstance(expr, FrontendTupleExpr):
            elements = tuple(
                self._analyze_expr(element, env, allow_outer_lookup=allow_outer_lookup)
                for element in expr.elements
            )
            return self._attach_expr_source_location(
                SemanticTupleExpr(
                    elements=elements,
                    type=SemanticTupleType(elements=tuple(element.type for element in elements)),
                ),
                expr,
            )
        if isinstance(expr, FrontendAttributeExpr):
            base = self._analyze_expr(expr.base, env, allow_outer_lookup=allow_outer_lookup)
            if expr.attr == "element_type":
                return self._attach_expr_source_location(self._element_type_expr(base), expr)
            if expr.attr == "rank":
                return self._attach_expr_source_location(self._rank_expr(base), expr)
            if expr.attr == "memory_space":
                return self._attach_expr_source_location(self._memory_space_expr(base), expr)
            if expr.attr == "pad_value" and isinstance(base.type, SemanticTileType):
                return self._attach_expr_source_location(self._tile_pad_value_expr(base), expr)
            if expr.attr == "compact_mode" and isinstance(base.type, SemanticTileType):
                return self._attach_expr_source_location(self._tile_compact_mode_expr(base), expr)
            if expr.attr == "config":
                return self._attach_expr_source_location(self._tile_config_expr(base), expr)
            if expr.attr == "valid_shape":
                return self._attach_expr_source_location(self._valid_shape_expr(base), expr)
            if expr.attr == "strides":
                return self._attach_expr_source_location(self._strides_expr(base), expr)
            if isinstance(base.type, SemanticTileConfigType):
                return self._attach_expr_source_location(self._tile_config_attr_expr(base, expr.attr), expr)
            attr_type = self._attribute_type(base, expr.attr)
            return self._attach_expr_source_location(
                SemanticAttributeAccess(base=base, attr=expr.attr, type=attr_type),
                expr,
            )
        if isinstance(expr, FrontendSubscriptExpr):
            base = self._analyze_expr(expr.base, env, allow_outer_lookup=allow_outer_lookup)
            index = self._analyze_expr(expr.index, env, allow_outer_lookup=allow_outer_lookup)
            if isinstance(base.type, (SemanticShapeType, SemanticTupleType)):
                index = self._require_index_typed_expr(index)
            result_type = self._subscript_type(base, index)
            if isinstance(result_type, SemanticTensorSliceType):
                slices = self._normalize_tensor_slice(index, base.type.rank)
                return self._attach_expr_source_location(
                    SemanticTensorSliceExpr(base=base, slices=slices, type=result_type),
                    expr,
                )
            return self._attach_expr_source_location(
                SemanticSubscriptAccess(base=base, index=index, type=result_type),
                expr,
            )
        if isinstance(expr, FrontendBinaryExpr):
            lhs = self._analyze_expr(expr.lhs, env, allow_outer_lookup=allow_outer_lookup)
            rhs = self._analyze_expr(expr.rhs, env, allow_outer_lookup=allow_outer_lookup)
            lhs, rhs = self._retarget_literals_for_binary_op(lhs, rhs, expr.op)
            result_type = self._binary_type(lhs, rhs, expr.op)
            return self._attach_expr_source_location(
                SemanticBinaryExpr(lhs=lhs, op=expr.op, rhs=rhs, type=result_type),
                expr,
            )
        if isinstance(expr, FrontendCallExpr):
            if expr.namespace is None:
                binding = env.get(expr.name)
                if (
                    binding is not None
                    and isinstance(binding.type, SemanticMetaType)
                    and binding.type.kind == "dtype"
                    and isinstance(binding.value, ScalarType)
                ):
                    if expr.keywords:
                        raise TypeError(
                            f"`{expr.name}` does not support keyword arguments in TileLang DSL v1"
                        )
                    args = tuple(
                        self._analyze_expr(arg, env, allow_outer_lookup=allow_outer_lookup)
                        for arg in expr.args
                    )
                    return self._analyze_scalar_constructor_for_dtype(
                        binding.value,
                        args,
                        surface_name=expr.name,
                    )
            if expr.namespace is None and expr.name in self._inline_proc_nodes:
                if expr.keywords:
                    raise TypeError(
                        f"inline_proc call `{expr.name}` reached semantic analysis with unresolved keywords in TileLang DSL v1"
                    )
                args = tuple(
                    self._analyze_expr(arg, env, allow_outer_lookup=allow_outer_lookup)
                    for arg in expr.args
                )
                return self._analyze_inline_proc_call_expr(expr.name, args)
            if expr.namespace is None and expr.name == "eval":
                if expr.keywords:
                    raise TypeError("method call `eval` does not support keyword arguments in TileLang DSL v1")
                if not expr.args:
                    raise TypeError("`eval()` expects a receiver in TileLang DSL v1")
                base = self._analyze_expr(expr.args[0], env, allow_outer_lookup=allow_outer_lookup)
                args = tuple(
                    self._analyze_expr(arg, env, allow_outer_lookup=allow_outer_lookup)
                    for arg in expr.args[1:]
                )
                return self._analyze_eval_method(base, args)
            if expr.namespace is None and expr.name == "astype":
                if expr.keywords:
                    raise TypeError("method call `astype` does not support keyword arguments in TileLang DSL v1")
                if not expr.args:
                    raise TypeError("`astype()` expects a receiver in TileLang DSL v1")
                base = self._analyze_expr(expr.args[0], env, allow_outer_lookup=allow_outer_lookup)
                args = tuple(
                    self._analyze_expr(arg, env, allow_outer_lookup=allow_outer_lookup)
                    for arg in expr.args[1:]
                )
                return self._analyze_astype_method(base, args)
            if expr.namespace not in {None, "pto"} and expr.name == "eval":
                if expr.keywords:
                    raise TypeError("method call `eval` does not support keyword arguments in TileLang DSL v1")
                binding = env.get(expr.namespace)
                if binding is None:
                    if allow_outer_lookup:
                        raise ValueError(f"unknown name '{expr.namespace}'")
                    raise ValueError(
                        f"implicit capture of '{expr.namespace}' is not allowed in pto.strict_vecscope"
                    )
                base = SemanticBindingRef(binding=binding, type=binding.type)
                args = tuple(
                    self._analyze_expr(arg, env, allow_outer_lookup=allow_outer_lookup)
                    for arg in expr.args
                )
                return self._analyze_eval_method(base, args)
            if expr.namespace not in {None, "pto"} and expr.name == "as_ptr":
                if expr.keywords:
                    raise TypeError("method call `as_ptr` does not support keyword arguments in TileLang DSL v1")
                binding = env.get(expr.namespace)
                if binding is None:
                    if allow_outer_lookup:
                        raise ValueError(f"unknown name '{expr.namespace}'")
                    raise ValueError(
                        f"implicit capture of '{expr.namespace}' is not allowed in pto.strict_vecscope"
                    )
                base = SemanticBindingRef(binding=binding, type=binding.type)
                return self._analyze_as_ptr_method(base)
            if expr.namespace not in {None, "pto"} and expr.name == "astype":
                if expr.keywords:
                    raise TypeError("method call `astype` does not support keyword arguments in TileLang DSL v1")
                binding = env.get(expr.namespace)
                if binding is None:
                    if allow_outer_lookup:
                        raise ValueError(f"unknown name '{expr.namespace}'")
                    raise ValueError(
                        f"implicit capture of '{expr.namespace}' is not allowed in pto.strict_vecscope"
                    )
                base = SemanticBindingRef(binding=binding, type=binding.type)
                args = tuple(
                    self._analyze_expr(arg, env, allow_outer_lookup=allow_outer_lookup)
                    for arg in expr.args
                )
                return self._analyze_astype_method(base, args)
            if expr.namespace == "pto" and expr.name == "vlds":
                return self._analyze_vlds_frontend_call(
                    expr,
                    env,
                    allow_outer_lookup=allow_outer_lookup,
                )
            if (
                expr.namespace == "pto"
                and expr.name == "vldas"
                and len(expr.args) == 1
                and isinstance(expr.args[0], FrontendSubscriptExpr)
            ):
                base, indices = self._analyze_tile_vector_access(
                    expr.args[0],
                    env,
                    allow_outer_lookup=allow_outer_lookup,
                    context="pto.vldas source",
                )
                return self._analyze_vldas((base, *indices))
            if (
                expr.namespace == "pto"
                and expr.name == "vldus"
                and len(expr.args) == 2
                and isinstance(expr.args[0], FrontendSubscriptExpr)
            ):
                base, indices = self._analyze_tile_vector_access(
                    expr.args[0],
                    env,
                    allow_outer_lookup=allow_outer_lookup,
                    context="pto.vldus source",
                )
                align_expr = self._analyze_expr(expr.args[1], env, allow_outer_lookup=allow_outer_lookup)
                return self._analyze_vldus((base, *indices, align_expr))
            if expr.namespace == "pto" and expr.name == "vldsx2" and len(expr.args) == 2:
                base, indices = self._analyze_tile_vector_access(
                    expr.args[0],
                    env,
                    allow_outer_lookup=allow_outer_lookup,
                    context="pto.vldsx2 source",
                )
                dist = self._analyze_expr(expr.args[1], env, allow_outer_lookup=allow_outer_lookup)
                return self._analyze_vldsx2((base, *indices, dist))
            if expr.namespace == "pto" and expr.name == "vcvt":
                return self._analyze_vcvt_frontend_call(
                    expr,
                    env,
                    allow_outer_lookup=allow_outer_lookup,
                )
            if expr.namespace == "pto" and expr.name == "vtrc":
                return self._analyze_vtrc_frontend_call(
                    expr,
                    env,
                    allow_outer_lookup=allow_outer_lookup,
                )
            if expr.namespace == "pto" and expr.name == "Tile":
                return self._analyze_tile_frontend_call(
                    expr,
                    env,
                    allow_outer_lookup=allow_outer_lookup,
                )
            if expr.namespace == "pto" and expr.name in _CUBE_CALL_OPS:
                return self._analyze_cube_frontend_call_expr(
                    expr,
                    env,
                    allow_outer_lookup=allow_outer_lookup,
                )
            if expr.keywords:
                raise TypeError(
                    f"call surface `{expr.namespace + '.' if expr.namespace else ''}{expr.name}` "
                    "carries keyword arguments, but semantic keyword handling is not implemented "
                    "in TileLang DSL v1 yet"
                )
            args = tuple(
                self._analyze_expr(arg, env, allow_outer_lookup=allow_outer_lookup)
                for arg in expr.args
            )
            return self._analyze_call_expr(expr.namespace, expr.name, args)
        raise ValueError(f"unsupported frontend expression {type(expr).__name__}")

    def _analyze_symbol_expr(self, expr: FrontendSymbolExpr) -> SemanticExpr:
        if expr.namespace == "pto":
            dtype = _DTYPE_SYMBOLS.get(expr.name)
            if dtype is not None:
                return SemanticSymbolExpr(
                    namespace=expr.namespace,
                    name=expr.name,
                    value=dtype,
                    type=SemanticMetaType(kind="dtype"),
                )
            mask_type = _MASK_TYPE_SYMBOLS.get(expr.name)
            if mask_type is not None:
                return SemanticSymbolExpr(
                    namespace=expr.namespace,
                    name=expr.name,
                    value=mask_type,
                    type=SemanticMetaType(kind="mask_type"),
                )
            if expr.name == "align":
                return SemanticSymbolExpr(
                    namespace=expr.namespace,
                    name=expr.name,
                    value=align,
                    type=SemanticMetaType(kind="align_type"),
                )
            if expr.name == "PartitionTensorView":
                return SemanticSymbolExpr(
                    namespace=expr.namespace,
                    name=expr.name,
                    value=expr.name,
                    type=SemanticMetaType(kind="partition_tensor_view_type"),
                )
        if expr.namespace in {"PAT", "pto.PAT", "pto.MaskPattern"}:
            pattern = _PATTERN_SYMBOLS.get(expr.name)
            if pattern is None and expr.name.startswith("PAT_"):
                canonical = expr.name[len("PAT_") :]
                pattern = _PATTERN_SYMBOLS.get(canonical)
            if pattern is not None:
                return SemanticSymbolExpr(
                    namespace=expr.namespace,
                    name=expr.name,
                    value=pattern,
                    type=SemanticMetaType(kind="mask_pattern"),
                )
        if expr.namespace in {"PIPE", "pto.PIPE", "Pipe", "pto.Pipe"}:
            pipe = _PIPE_SYMBOLS.get(expr.name)
            if pipe is not None:
                return SemanticSymbolExpr(
                    namespace=expr.namespace,
                    name=expr.name,
                    value=pipe,
                    type=SemanticMetaType(kind="pipe"),
                )
        if expr.namespace in {"EVENT", "pto.EVENT", "Event", "pto.Event"}:
            event = _EVENT_SYMBOLS.get(expr.name)
            if event is not None:
                return SemanticSymbolExpr(
                    namespace=expr.namespace,
                    name=expr.name,
                    value=event,
                    type=SemanticMetaType(kind="event"),
                )
        if expr.namespace in {"BarrierType", "pto.BarrierType"}:
            barrier_type = _BARRIER_TYPE_SYMBOLS.get(expr.name)
            if barrier_type is not None:
                return SemanticSymbolExpr(
                    namespace=expr.namespace,
                    name=expr.name,
                    value=barrier_type,
                    type=SemanticMetaType(kind="barrier_type"),
                )
        if expr.namespace in {"MemorySpace", "pto.MemorySpace"}:
            memory_space = _MEMORY_SPACE_SYMBOLS.get(expr.name)
            if memory_space is not None:
                return SemanticSymbolExpr(
                    namespace=expr.namespace,
                    name=expr.name,
                    value=memory_space,
                    type=SemanticMetaType(kind="memory_space"),
                )
        if expr.namespace in {"PadMode", "pto.PadMode"}:
            pad_mode = _PAD_MODE_SYMBOLS.get(expr.name)
            if pad_mode is not None:
                return SemanticSymbolExpr(
                    namespace=expr.namespace,
                    name=expr.name,
                    value=pad_mode,
                    type=SemanticMetaType(kind="pad_mode"),
                )
        if expr.namespace in {"BLayout", "pto.BLayout"}:
            b_layout = _B_LAYOUT_SYMBOLS.get(expr.name)
            if b_layout is not None:
                return SemanticSymbolExpr(
                    namespace=expr.namespace,
                    name=expr.name,
                    value=b_layout,
                    type=SemanticMetaType(kind="b_layout"),
                )
        if expr.namespace in {"SLayout", "pto.SLayout"}:
            s_layout = _S_LAYOUT_SYMBOLS.get(expr.name)
            if s_layout is not None:
                return SemanticSymbolExpr(
                    namespace=expr.namespace,
                    name=expr.name,
                    value=s_layout,
                    type=SemanticMetaType(kind="s_layout"),
                )
        if expr.namespace in {"CompactMode", "pto.CompactMode"}:
            compact_mode = _COMPACT_MODE_SYMBOLS.get(expr.name)
            if compact_mode is not None:
                return SemanticSymbolExpr(
                    namespace=expr.namespace,
                    name=expr.name,
                    value=compact_mode,
                    type=SemanticMetaType(kind="compact_mode"),
                )
        if expr.namespace in {"PadValue", "pto.PadValue"}:
            pad_value = _PAD_VALUE_SYMBOLS.get(expr.name)
            if pad_value is not None:
                return SemanticSymbolExpr(
                    namespace=expr.namespace,
                    name=expr.name,
                    value=pad_value,
                    type=SemanticPadValueType(),
                )
        if expr.namespace in {"PredicateDist", "pto.PredicateDist"}:
            predicate_dist = _PREDICATE_DIST_SYMBOLS.get(expr.name)
            if predicate_dist is not None:
                return SemanticSymbolExpr(
                    namespace=expr.namespace,
                    name=expr.name,
                    value=predicate_dist,
                    type=SemanticMetaType(kind="predicate_dist"),
                )
        if expr.namespace in {"VLoadDist", "pto.VLoadDist"}:
            vload_dist = _VLOAD_DIST_SYMBOLS.get(expr.name)
            if vload_dist is not None:
                return SemanticSymbolExpr(
                    namespace=expr.namespace,
                    name=expr.name,
                    value=vload_dist,
                    type=SemanticMetaType(kind="vload_dist"),
                )
        if expr.namespace in {"VStoreDist", "pto.VStoreDist"}:
            vstore_dist = _VSTORE_DIST_SYMBOLS.get(expr.name)
            if vstore_dist is not None:
                return SemanticSymbolExpr(
                    namespace=expr.namespace,
                    name=expr.name,
                    value=vstore_dist,
                    type=SemanticMetaType(kind="vstore_dist"),
                )
        if expr.namespace in {"PredicatePart", "pto.PredicatePart"}:
            predicate_part = _PREDICATE_PART_SYMBOLS.get(expr.name)
            if predicate_part is not None:
                return SemanticSymbolExpr(
                    namespace=expr.namespace,
                    name=expr.name,
                    value=predicate_part,
                    type=SemanticMetaType(kind="predicate_part"),
                )
        if expr.namespace in {"CmpMode", "pto.CmpMode"}:
            cmp_mode = _CMP_MODE_SYMBOLS.get(expr.name)
            if cmp_mode is not None:
                return SemanticSymbolExpr(
                    namespace=expr.namespace,
                    name=expr.name,
                    value=cmp_mode,
                    type=SemanticMetaType(kind="cmp_mode"),
                )
        if expr.namespace in {"DeinterleaveDist", "pto.DeinterleaveDist"}:
            dist = _DEINTERLEAVE_DIST_SYMBOLS.get(expr.name)
            if dist is not None:
                return SemanticSymbolExpr(
                    namespace=expr.namespace,
                    name=expr.name,
                    value=dist,
                    type=SemanticMetaType(kind="deinterleave_dist"),
                )
        if expr.namespace in {"InterleaveDist", "pto.InterleaveDist"}:
            dist = _INTERLEAVE_DIST_SYMBOLS.get(expr.name)
            if dist is not None:
                return SemanticSymbolExpr(
                    namespace=expr.namespace,
                    name=expr.name,
                    value=dist,
                    type=SemanticMetaType(kind="interleave_dist"),
                )
        if expr.namespace in {"PositionMode", "pto.PositionMode"}:
            position_mode = _POSITION_MODE_SYMBOLS.get(expr.name)
            if position_mode is not None:
                return SemanticSymbolExpr(
                    namespace=expr.namespace,
                    name=expr.name,
                    value=position_mode,
                    type=SemanticMetaType(kind="position_mode"),
                )
        if expr.namespace in {"OrderMode", "pto.OrderMode"}:
            order_mode = _ORDER_MODE_SYMBOLS.get(expr.name)
            if order_mode is not None:
                return SemanticSymbolExpr(
                    namespace=expr.namespace,
                    name=expr.name,
                    value=order_mode,
                    type=SemanticMetaType(kind="order_mode"),
                )
        if expr.namespace in {"VcvtRoundMode", "pto.VcvtRoundMode"}:
            round_mode = _VCVT_ROUND_MODE_SYMBOLS.get(expr.name)
            if round_mode is not None:
                return SemanticSymbolExpr(
                    namespace=expr.namespace,
                    name=expr.name,
                    value=round_mode,
                    type=SemanticMetaType(kind="vcvt_round_mode"),
                )
        if expr.namespace in {"VcvtSatMode", "pto.VcvtSatMode"}:
            sat_mode = _VCVT_SAT_MODE_SYMBOLS.get(expr.name)
            if sat_mode is not None:
                return SemanticSymbolExpr(
                    namespace=expr.namespace,
                    name=expr.name,
                    value=sat_mode,
                    type=SemanticMetaType(kind="vcvt_sat_mode"),
                )
        if expr.namespace in {"VcvtPartMode", "pto.VcvtPartMode"}:
            part_mode = _VCVT_PART_MODE_SYMBOLS.get(expr.name)
            if part_mode is not None:
                return SemanticSymbolExpr(
                    namespace=expr.namespace,
                    name=expr.name,
                    value=part_mode,
                    type=SemanticMetaType(kind="vcvt_part_mode"),
                )
        if expr.namespace in {"PostUpdateMode", "pto.PostUpdateMode"}:
            post_update_mode = _POST_UPDATE_MODE_SYMBOLS.get(expr.name)
            if post_update_mode is not None:
                return SemanticSymbolExpr(
                    namespace=expr.namespace,
                    name=expr.name,
                    value=post_update_mode,
                    type=SemanticMetaType(kind="post_update_mode"),
                )
        if expr.namespace in {"FractalMode", "pto.FractalMode"}:
            fractal_mode = _FRACTAL_MODE_SYMBOLS.get(expr.name)
            if fractal_mode is not None:
                return SemanticSymbolExpr(
                    namespace=expr.namespace,
                    name=expr.name,
                    value=fractal_mode,
                    type=SemanticMetaType(kind="cube_mode"),
                )
        raise TypeError(
            f"symbol `{expr.namespace}.{expr.name}` is not supported in TileLang DSL v1"
        )

    def _attribute_type(self, base: SemanticExpr, attr: str) -> SemanticType:
        base_type = base.type
        if isinstance(base_type, (SemanticTensorViewType, SemanticPartitionTensorViewType)) and attr == "shape":
            return SemanticShapeType(rank=base_type.rank)
        if isinstance(base_type, (SemanticTensorViewType, SemanticPartitionTensorViewType)) and attr == "strides":
            return SemanticShapeType(rank=base_type.rank)
        if isinstance(base_type, SemanticTileType) and attr == "shape":
            return SemanticShapeType(rank=base_type.rank)
        if isinstance(base_type, (SemanticTensorViewType, SemanticPartitionTensorViewType, SemanticTileType)) and attr == "valid_shape":
            return SemanticShapeType(rank=base_type.rank)
        raise TypeError(f"unsupported attribute access '{attr}' in TileLang DSL v1")

    def _element_type_expr(self, base: SemanticExpr) -> SemanticExpr:
        base_type = base.type
        if isinstance(base_type, (SemanticTensorViewType, SemanticPartitionTensorViewType, SemanticTileType)):
            return SemanticSymbolExpr(
                namespace="pto",
                name=base_type.element_dtype.name,
                value=base_type.element_dtype,
                type=SemanticMetaType(kind="dtype"),
            )
        raise TypeError("unsupported attribute access 'element_type' in TileLang DSL v1")

    def _rank_expr(self, base: SemanticExpr) -> SemanticExpr:
        base_type = base.type
        if isinstance(base_type, (SemanticTensorViewType, SemanticPartitionTensorViewType, SemanticTileType)):
            return SemanticLiteralExpr(value=base_type.rank, type=SemanticIndexType())
        raise TypeError("unsupported attribute access 'rank' in TileLang DSL v1")

    def _memory_space_expr(self, base: SemanticExpr) -> SemanticExpr:
        base_type = base.type
        if isinstance(base_type, (SemanticTensorViewType, SemanticPartitionTensorViewType)):
            return SemanticSymbolExpr(
                namespace="pto",
                name=MemorySpace.GM.name,
                value=MemorySpace.GM,
                type=SemanticMetaType(kind="memory_space"),
            )
        if isinstance(base_type, SemanticTileType):
            memory_space = MemorySpace.UB if base_type.memory_space is None else MemorySpace(base_type.memory_space)
            return SemanticSymbolExpr(
                namespace="pto",
                name=memory_space.name,
                value=memory_space,
                type=SemanticMetaType(kind="memory_space"),
            )
        raise TypeError("unsupported attribute access 'memory_space' in TileLang DSL v1")

    def _tile_config_expr(self, base: SemanticExpr) -> SemanticExpr:
        base_type = base.type
        if isinstance(base_type, SemanticTileType):
            return SemanticLiteralExpr(
                value=base_type.config or TileConfig(),
                type=SemanticTileConfigType(element_dtype=base_type.element_dtype),
            )
        raise TypeError("unsupported attribute access 'config' in TileLang DSL v1")

    def _tile_pad_value_expr(self, base: SemanticExpr) -> SemanticExpr:
        base_type = base.type
        if not isinstance(base_type, SemanticTileType):
            raise TypeError("unsupported attribute access 'pad_value' in TileLang DSL v1")
        config = base_type.config or TileConfig()
        return SemanticSymbolExpr(
            namespace="pto",
            name=config.pad_value.name,
            value=config.pad_value,
            type=SemanticPadValueType(element_dtype=base_type.element_dtype),
        )

    def _tile_compact_mode_expr(self, base: SemanticExpr) -> SemanticExpr:
        base_type = base.type
        if not isinstance(base_type, SemanticTileType):
            raise TypeError("unsupported attribute access 'compact_mode' in TileLang DSL v1")
        config = base_type.config or TileConfig()
        return SemanticSymbolExpr(
            namespace="pto",
            name=config.compact_mode.name,
            value=config.compact_mode,
            type=SemanticMetaType(kind="compact_mode"),
        )

    def _pad_value_eval_expr(
        self,
        base: SemanticExpr,
        dtype_expr: SemanticExpr | None = None,
    ) -> SemanticExpr:
        if not isinstance(base.type, SemanticPadValueType):
            raise TypeError("`eval()` expects a PadValue descriptor in TileLang DSL v1")
        element_dtype = base.type.element_dtype
        if dtype_expr is not None:
            explicit_dtype = self._try_static_value(dtype_expr)
            if not isinstance(explicit_dtype, ScalarType):
                raise TypeError("PadValue.eval(dtype) expects a TileLang scalar dtype symbol in TileLang DSL v1")
            element_dtype = explicit_dtype
        if element_dtype is None:
            raise TypeError(
                "PadValue.eval() requires either a Tile-bound pad descriptor or an explicit dtype argument "
                "in TileLang DSL v1"
            )
        pad_value = self._try_static_value(base)
        if not isinstance(pad_value, PadValue):
            raise TypeError("PadValue.eval() expects a statically known PadValue enum in TileLang DSL v1")
        pad_scalar = pad_value.eval(element_dtype)
        if pad_scalar is None:
            raise TypeError(
                "PadValue.NULL.eval() is invalid in TileLang DSL v1; "
                "guard it with `pto.constexpr(tile.pad_value != pto.PadValue.NULL)` before calling `.eval()`"
            )
        return SemanticLiteralExpr(
            value=pad_scalar,
            type=SemanticScalarType(dtype=element_dtype),
        )

    def _analyze_eval_method(
        self,
        base: SemanticExpr,
        args: tuple[SemanticExpr, ...],
    ) -> SemanticExpr:
        if len(args) > 1:
            raise TypeError("`eval()` accepts at most one positional dtype argument in TileLang DSL v1")
        return self._pad_value_eval_expr(base, args[0] if args else None)

    def _tile_config_attr_expr(self, base: SemanticExpr, attr: str) -> SemanticExpr:
        config = self._try_static_value(base)
        if not isinstance(config, TileConfig):
            raise TypeError("Tile config metadata must be statically known in TileLang DSL v1")
        if attr == "b_layout":
            return SemanticSymbolExpr(
                namespace="pto",
                name=config.b_layout.name,
                value=config.b_layout,
                type=SemanticMetaType(kind="b_layout"),
            )
        if attr == "s_layout":
            return SemanticSymbolExpr(
                namespace="pto",
                name=config.s_layout.name,
                value=config.s_layout,
                type=SemanticMetaType(kind="s_layout"),
            )
        if attr == "s_fractal_size":
            return SemanticLiteralExpr(
                value=config.s_fractal_size,
                type=SemanticScalarType(dtype=i32),
            )
        if attr == "pad_value":
            if not isinstance(base.type, SemanticTileConfigType):
                raise TypeError(
                    "TileConfig.pad_value expects a TileConfig value in TileLang DSL v1"
                )
            return SemanticSymbolExpr(
                namespace="pto",
                name=config.pad_value.name,
                value=config.pad_value,
                type=SemanticPadValueType(element_dtype=base.type.element_dtype),
            )
        if attr == "compact_mode":
            return SemanticSymbolExpr(
                namespace="pto",
                name=config.compact_mode.name,
                value=config.compact_mode,
                type=SemanticMetaType(kind="compact_mode"),
            )
        raise TypeError(f"unsupported TileConfig attribute access '{attr}' in TileLang DSL v1")

    def _analyze_as_ptr_method(self, base: SemanticExpr) -> SemanticExpr:
        base_type = base.type
        if isinstance(base_type, (SemanticTensorViewType, SemanticPartitionTensorViewType)):
            return SemanticCallExpr(
                namespace="pto",
                name="tensor_view_as_ptr",
                args=(base,),
                type=SemanticPtrType(
                    element_dtype=base_type.element_dtype,
                    memory_space="gm",
                ),
            )
        if isinstance(base_type, SemanticTileType):
            return SemanticCallExpr(
                namespace="pto",
                name="tile_as_ptr",
                args=(base,),
                type=SemanticPtrType(
                    element_dtype=base_type.element_dtype,
                    memory_space=base_type.memory_space or "ub",
                ),
            )
        raise TypeError("`as_ptr()` expects a TensorView/PartitionTensorView or Tile value in TileLang DSL v1")

    def _analyze_astype_method(self, base: SemanticExpr, args: tuple[SemanticExpr, ...]) -> SemanticExpr:
        if len(args) != 1:
            raise TypeError("`astype()` expects exactly 1 positional argument (target dtype) in TileLang DSL v1")
        if isinstance(base.type, SemanticVRegType):
            target_dtype = self._require_dtype_symbol(args[0], "astype target dtype")
            return SemanticCallExpr(
                namespace="pto",
                name="vbitcast",
                args=(base, args[0]),
                type=self._vreg_type_for_dtype(target_dtype),
            )
        if isinstance(base.type, SemanticMaskType):
            target_mask_type = self._require_mask_type_expr(args[0], "astype target dtype")
            return SemanticCallExpr(
                namespace="pto",
                name="pbitcast",
                args=(base, args[0]),
                type=SemanticMaskType(granularity=target_mask_type.granularity),
            )
        raise TypeError("`astype()` expects a vector register or mask value in TileLang DSL v1")

    def _valid_shape_expr(self, base: SemanticExpr) -> SemanticExpr:
        base_type = base.type
        if not isinstance(base_type, (SemanticTensorViewType, SemanticPartitionTensorViewType, SemanticTileType)):
            raise TypeError("unsupported attribute access 'valid_shape' in TileLang DSL v1")
        shape_access = SemanticAttributeAccess(
            base=base,
            attr="valid_shape",
            type=SemanticShapeType(rank=base_type.rank),
        )
        elements = []
        for axis in range(base_type.rank):
            if (
                isinstance(base, SemanticBindingRef)
                and isinstance(base.type, SemanticTileType)
                and base.type.valid_shape is not None
                and base.type.valid_shape[axis] is None
            ):
                self._ensure_tile_valid_shape_parameter(base.binding, axis)
            elements.append(
                SemanticSubscriptAccess(
                    base=shape_access,
                    index=SemanticLiteralExpr(value=axis, type=SemanticIndexType()),
                    type=SemanticIndexType(),
                )
            )
        return SemanticTupleExpr(
            elements=tuple(elements),
            type=SemanticTupleType(elements=tuple(SemanticIndexType() for _ in elements)),
        )

    def _strides_expr(self, base: SemanticExpr) -> SemanticExpr:
        base_type = base.type
        if not isinstance(base_type, (SemanticTensorViewType, SemanticPartitionTensorViewType)):
            raise TypeError("unsupported attribute access 'strides' in TileLang DSL v1")
        stride_access = SemanticAttributeAccess(
            base=base,
            attr="strides",
            type=SemanticShapeType(rank=base_type.rank),
        )
        elements = []
        for axis in range(base_type.rank):
            elements.append(
                SemanticSubscriptAccess(
                    base=stride_access,
                    index=SemanticLiteralExpr(value=axis, type=SemanticIndexType()),
                    type=SemanticIndexType(),
                )
            )
        return SemanticTupleExpr(
            elements=tuple(elements),
            type=SemanticTupleType(elements=tuple(SemanticIndexType() for _ in elements)),
        )

    def _subscript_type(self, base: SemanticExpr, index: SemanticExpr) -> SemanticType:
        if isinstance(base.type, SemanticShapeType):
            if not isinstance(index.type, SemanticIndexType):
                raise TypeError("shape subscript index must be an index value in TileLang DSL v1")
            if not isinstance(index, SemanticLiteralExpr) or not isinstance(index.value, int):
                raise TypeError(
                    "shape/stride/valid_shape subscript index must be an integer literal in TileLang DSL v1"
                )
            if index.value < 0 or index.value >= base.type.rank:
                raise TypeError(
                    f"shape subscript index {index.value} is out of bounds for rank {base.type.rank}"
                )
            return SemanticIndexType()
        if isinstance(base.type, SemanticTupleType):
            if not isinstance(index.type, SemanticIndexType):
                raise TypeError("tuple subscript index must be an index value in TileLang DSL v1")
            if not isinstance(base, SemanticTupleExpr):
                raise TypeError(
                    "tuple subscripting currently requires a shape-like tuple expression in TileLang DSL v1"
                )
            if not base.type.elements:
                raise TypeError("cannot subscript an empty tuple in TileLang DSL v1")
            if not isinstance(index, SemanticLiteralExpr) or not isinstance(index.value, int):
                raise TypeError("tuple subscript index must be an integer literal in TileLang DSL v1")

            if index.value < 0 or index.value >= len(base.type.elements):
                raise TypeError(
                    f"tuple subscript index {index.value} is out of bounds for tuple length {len(base.type.elements)}"
                )
            return base.type.elements[index.value]
        if isinstance(base.type, (SemanticTensorViewType, SemanticPartitionTensorViewType)):
            if not isinstance(index, SemanticTupleExpr):
                raise TypeError("TensorView slicing expects a tuple of slices in TileLang DSL v1")
            return self._tensor_slice_type(base.type, index)
        raise TypeError("unsupported subscript base in TileLang DSL v1")

    def _analyze_tile_vector_access(
        self,
        expr: FrontendExprNode,
        env: dict[str, SemanticBinding],
        *,
        allow_outer_lookup: bool,
        context: str,
    ) -> tuple[SemanticExpr, tuple[SemanticExpr, ...]]:
        if not isinstance(expr, FrontendSubscriptExpr):
            raise TypeError(
                f"{context} expects Tile element-indexing syntax in TileLang DSL v1"
            )
        base = self._analyze_expr(expr.base, env, allow_outer_lookup=allow_outer_lookup)
        tile = self._require_tile_expr(base, context)
        indices = self._tile_vector_indices(
            expr.index,
            tile.type,
            env,
            allow_outer_lookup=allow_outer_lookup,
            context=context,
        )
        return base, indices

    def _tile_vector_indices(
        self,
        index_expr: FrontendExprNode,
        tile_type: SemanticTileType,
        env: dict[str, SemanticBinding],
        *,
        allow_outer_lookup: bool,
        context: str,
    ) -> tuple[SemanticExpr, ...]:
        if tile_type.rank == 1:
            if not isinstance(index_expr, FrontendSliceExpr):
                raise TypeError(f"{context} expects Tile[start:] syntax for rank-1 Tile values")
            if index_expr.stop is not None:
                raise TypeError(f"{context} does not support explicit slice stop in TileLang DSL advanced mode")
            if index_expr.step is not None:
                raise TypeError(f"{context} does not support stepped Tile vector slices in TileLang DSL advanced mode")
            if index_expr.start is None:
                return (SemanticLiteralExpr(value=0, type=SemanticIndexType()),)
            start = self._analyze_expr(index_expr.start, env, allow_outer_lookup=allow_outer_lookup)
            start = self._require_index_typed_expr(start)
            return (start,)

        if tile_type.rank != 2 or tile_type.shape is None:
            raise TypeError(f"{context} currently only supports statically specialized rank-1 or rank-2 Tiles")
        if not isinstance(index_expr, FrontendTupleExpr) or len(index_expr.elements) != 2:
            raise TypeError(f"{context} expects Tile[row, col:] syntax for rank-2 Tile values")

        row_expr, col_expr = index_expr.elements
        if not isinstance(col_expr, FrontendSliceExpr):
            raise TypeError(f"{context} expects Tile[row, col:] syntax for rank-2 Tile values")
        if col_expr.stop is not None:
            raise TypeError(f"{context} does not support explicit slice stop in TileLang DSL advanced mode")
        if col_expr.step is not None:
            raise TypeError(f"{context} does not support stepped Tile vector slices in TileLang DSL advanced mode")

        row = self._analyze_expr(row_expr, env, allow_outer_lookup=allow_outer_lookup)
        row = self._require_index_typed_expr(row)
        if col_expr.start is None:
            col = SemanticLiteralExpr(value=0, type=SemanticIndexType())
        else:
            col = self._analyze_expr(col_expr.start, env, allow_outer_lookup=allow_outer_lookup)
            col = self._require_index_typed_expr(col)
        return (row, col)

    def _tensor_slice_type(
        self,
        tensor_type: SemanticTensorViewType | SemanticPartitionTensorViewType,
        index: SemanticTupleExpr,
    ) -> SemanticTensorSliceType:
        if not 1 <= len(index.elements) <= tensor_type.rank:
            raise TypeError(
                f"TensorView slice rank {len(index.elements)} must be between 1 and "
                f"{tensor_type.rank} in TileLang DSL v1"
            )
        axis_offset = tensor_type.rank - len(index.elements)
        extents = []
        for axis, element in enumerate(index.elements):
            if not isinstance(element, SemanticSliceExpr):
                raise TypeError(
                    f"TensorView slicing axis {axis} must use a Python slice in TileLang DSL v1"
                )
            self._require_optional_index_typed_expr(element.start)
            self._require_optional_index_typed_expr(element.stop)
            self._require_optional_index_typed_expr(element.step)

            if element.stop is None:
                raise TypeError("TensorView slicing requires explicit stop bounds in TileLang DSL v1")
            extents.append(self._normalized_tensor_slice_extent(element))
        return SemanticTensorSliceType(
            element_dtype=tensor_type.element_dtype,
            rank=len(index.elements),
            extents=tuple(extents),
            physical_axes=tuple(range(axis_offset, tensor_type.rank)),
        )

    def _normalize_tensor_slice(
        self,
        index: SemanticExpr,
        rank: int,
    ) -> tuple[SemanticTensorSliceAxis, ...]:
        if not isinstance(index, SemanticTupleExpr):
            raise TypeError("TensorView slicing expects a tuple index in TileLang DSL v1")
        if not 1 <= len(index.elements) <= rank:
            raise TypeError(
                f"TensorView slicing expects between 1 and {rank} slice elements in TileLang DSL v1"
            )
        slices = []
        for element in index.elements:
            if not isinstance(element, SemanticSliceExpr):
                raise TypeError("TensorView slicing only supports slice syntax in TileLang DSL v1")
            if element.stop is None:
                raise TypeError("TensorView slicing requires explicit stop bounds in TileLang DSL v1")
            start = self._normalize_optional_index_expr(element.start, default=0)
            stop = element.stop
            step = self._normalize_optional_index_expr(element.step, default=1)
            slices.append(
                SemanticTensorSliceAxis(
                    start=start,
                    stop=stop,
                    step=step,
                    extent=self._normalized_tensor_slice_extent(element),
                )
            )
        return tuple(slices)

    def _binary_type(
        self,
        lhs: SemanticExpr,
        rhs: SemanticExpr,
        op: str,
    ) -> SemanticType:
        mixed_index_scalar_type = self._mixed_index_integer_scalar_type(lhs.type, rhs.type)
        if op in {"add", "sub", "mul", "mod", "floordiv", "bitand", "bitor", "bitxor", "lshift", "rshift"}:
            if isinstance(lhs.type, SemanticIndexType) and isinstance(rhs.type, SemanticIndexType):
                if op in {"add", "sub", "mul", "mod", "floordiv"}:
                    return SemanticIndexType()
            if isinstance(lhs.type, SemanticScalarType) and lhs.type == rhs.type:
                dtype = lhs.type.dtype
                if op in {"add", "sub", "mul"} and (is_integer_dtype(dtype) or is_float_dtype(dtype)):
                    return SemanticScalarType(dtype=dtype)
                if op in {"mod", "floordiv"} and is_integer_dtype(dtype):
                    return SemanticScalarType(dtype=dtype)
                if op in {"bitand", "bitor", "bitxor", "lshift", "rshift"} and is_integer_dtype(dtype):
                    return SemanticScalarType(dtype=dtype)
            if mixed_index_scalar_type is not None and op in {"add", "sub", "mul", "mod", "floordiv"}:
                return mixed_index_scalar_type
            raise TypeError(
                "binary expressions currently require matching index operands, "
                "matching scalar operands (add/sub/mul for integer/float; "
                "mod/floordiv/bitwise/shift for integer), or index operands "
                "mixed with integer scalars for add/sub/mul/mod/floordiv"
            )
        if op in {"eq", "ne"}:
            if isinstance(lhs.type, SemanticIndexType) and isinstance(rhs.type, SemanticIndexType):
                return SemanticScalarType(dtype=i1)
            if isinstance(lhs.type, SemanticScalarType) and lhs.type == rhs.type:
                return SemanticScalarType(dtype=i1)
            if mixed_index_scalar_type is not None:
                return SemanticScalarType(dtype=i1)
            if isinstance(lhs.type, SemanticPadValueType) and isinstance(rhs.type, SemanticPadValueType):
                return SemanticScalarType(dtype=i1)
            if isinstance(lhs.type, SemanticMetaType) and lhs.type == rhs.type:
                return SemanticScalarType(dtype=i1)
            raise TypeError(
                "comparison expressions currently require matching scalar/meta types, "
                "index-typed operands, or index operands mixed with integer scalars"
            )
        if op in {"gt", "lt", "ge", "le"}:
            if isinstance(lhs.type, SemanticIndexType) and isinstance(rhs.type, SemanticIndexType):
                return SemanticScalarType(dtype=i1)
            if isinstance(lhs.type, SemanticScalarType) and lhs.type == rhs.type:
                return SemanticScalarType(dtype=i1)
            if mixed_index_scalar_type is not None:
                return SemanticScalarType(dtype=i1)
            raise TypeError(
                "ordered comparison expressions currently require matching scalar types, "
                "index-typed operands, or index operands mixed with integer scalars"
            )
        if op in {"and", "or"}:
            self._require_condition_type(lhs.type)
            self._require_condition_type(rhs.type)
            return SemanticScalarType(dtype=i1)
        raise TypeError(f"unsupported binary operator '{op}' in TileLang DSL v1")

    def _retarget_literals_for_binary_op(
        self,
        lhs: SemanticExpr,
        rhs: SemanticExpr,
        op: str,
    ) -> tuple[SemanticExpr, SemanticExpr]:
        if isinstance(lhs.type, SemanticScalarType):
            rhs = self._retarget_literal_to_scalar_type_for_binary_op(rhs, lhs.type.dtype, op)
        if isinstance(rhs.type, SemanticScalarType):
            lhs = self._retarget_literal_to_scalar_type_for_binary_op(lhs, rhs.type.dtype, op)
        return lhs, rhs

    def _retarget_literal_to_scalar_type_for_binary_op(
        self,
        expr: SemanticExpr,
        target_dtype: ScalarType,
        op: str,
    ) -> SemanticExpr:
        if not isinstance(expr, SemanticLiteralExpr):
            return expr
        if not self._binary_op_supports_scalar_dtype(op, target_dtype):
            return expr
        if is_integer_dtype(target_dtype):
            if not isinstance(expr.type, SemanticIndexType):
                return expr
            if not isinstance(expr.value, int) or isinstance(expr.value, bool):
                return expr
            checked = self._check_integer_literal_range(expr.value, target_dtype, f"{target_dtype!r} literal")
            retargeted = SemanticLiteralExpr(
                value=checked,
                type=SemanticScalarType(dtype=target_dtype),
            )
            source_location = self._expr_source_location(expr)
            if source_location is not None:
                object.__setattr__(retargeted, "source_location", source_location)
            return retargeted
        if is_float_dtype(target_dtype):
            if isinstance(expr.value, bool) or not isinstance(expr.value, (int, float)):
                return expr
            if isinstance(expr.type, SemanticScalarType) and not is_float_dtype(expr.type.dtype):
                return expr
            if not isinstance(expr.type, (SemanticIndexType, SemanticScalarType)):
                return expr
            retargeted = SemanticLiteralExpr(
                value=float(expr.value),
                type=SemanticScalarType(dtype=target_dtype),
            )
            source_location = self._expr_source_location(expr)
            if source_location is not None:
                object.__setattr__(retargeted, "source_location", source_location)
            return retargeted
        return expr

    def _binary_op_supports_scalar_dtype(self, op: str, dtype: ScalarType) -> bool:
        if is_integer_dtype(dtype):
            if op in {"add", "sub", "mul", "eq", "ne", "gt", "lt", "ge", "le"}:
                return True
            if op in {"mod", "floordiv", "bitand", "bitor", "bitxor", "lshift", "rshift"}:
                return True
            return False
        if is_float_dtype(dtype):
            return op in {"add", "sub", "mul", "eq", "ne", "gt", "lt", "ge", "le"}
        return False

    def _mixed_index_integer_scalar_type(
        self,
        lhs_type: SemanticType,
        rhs_type: SemanticType,
    ) -> SemanticScalarType | None:
        scalar_type: SemanticScalarType | None = None
        if isinstance(lhs_type, SemanticIndexType) and isinstance(rhs_type, SemanticScalarType):
            scalar_type = rhs_type
        elif isinstance(rhs_type, SemanticIndexType) and isinstance(lhs_type, SemanticScalarType):
            scalar_type = lhs_type
        if scalar_type is None or not is_integer_dtype(scalar_type.dtype):
            return None
        if integer_bitwidth(scalar_type.dtype) not in {8, 16, 32, 64}:
            return None
        return scalar_type

    def _analyze_call_expr(
        self,
        namespace: str | None,
        name: str,
        args: tuple[SemanticExpr, ...],
    ) -> SemanticExpr:
        if namespace is None and name == "range":
            return SemanticCallExpr(namespace=namespace, name=name, args=args, type=None)
        if namespace is None:
            if name in self._inline_proc_nodes:
                return self._analyze_inline_proc_call_expr(name, args)
            if name in self._internal_inline_proc_nodes and self._is_internal_inline_proc_context():
                return self._analyze_internal_inline_proc_call_expr(name, args)
            raise TypeError(
                f"call surface `{name}` is not supported in TileLang DSL v1"
            )
        if namespace != "pto":
            raise TypeError(
                f"call surface `{namespace + '.' if namespace else ''}{name}` is not supported in TileLang DSL v1 yet"
            )
        if name in DEFERRED_PTO_SURFACES:
            raise TypeError(deferred_surface_message(name))
        if name in _DTYPE_SYMBOLS:
            return self._analyze_scalar_constructor(name, args)
        if name == "Tile":
            raise TypeError(
                "pto.Tile(...) requires dedicated keyword-aware semantic handling in TileLang DSL v1"
            )
        if name == "ptr":
            return self._analyze_ptr_type(args)
        if name == "vreg":
            return self._analyze_vreg_type(args)
        if name == "vector":
            return self._analyze_vector_type(args)
        if name == "castptr":
            return self._analyze_castptr(args)
        if name == "addptr":
            return self._analyze_addptr(args)
        if name == "bytewidth":
            return self._analyze_bytewidth(args)
        if name in {"get_lanes", "elements_per_vreg"}:
            return self._analyze_get_lanes(args, call_name=name)
        if name == "get_op_attr":
            return self._analyze_get_op_attr(args)
        if name == "constexpr":
            raise TypeError(
                "pto.constexpr(...) is only supported as an if-condition wrapper in TileLang DSL v1"
            )
        if name == "make_mask":
            return self._analyze_make_mask(args)
        if name in {
            "get_block_idx",
            "get_subblock_idx",
            "get_block_num",
            "get_subblock_num",
        }:
            return self._analyze_runtime_block_query(name, args)
        if name == "init_align":
            return self._analyze_init_align(args)
        if name == "vlds":
            return self._analyze_vlds(args)
        if name == "vldas":
            return self._analyze_vldas(args)
        if name == "vldus":
            return self._analyze_vldus(args)
        if name == "vldsx2":
            return self._analyze_vldsx2(args)
        if name in {"pset_b8", "pset_b16", "pset_b32", "pge_b8", "pge_b16", "pge_b32"}:
            return self._analyze_predicate_pattern_op(name, args)
        if name in {"plt_b8", "plt_b16", "plt_b32"}:
            return self._analyze_predicate_tail_op(name, args)
        if name in {"plds", "pld", "pldi"}:
            return self._analyze_predicate_load_op(name, args)
        if name == "pstu":
            return self._analyze_pstu(args)
        if name == "vstus":
            return self._analyze_vstus(args)
        if name == "vstur":
            return self._analyze_vstur(args)
        if name == "load_scalar":
            return self._analyze_load_scalar(args)
        if name in {"ppack", "punpack"}:
            return self._analyze_mask_part_op(name, args)
        if name in {"pnot", "psel", "pand", "por", "pxor"}:
            return self._analyze_mask_logic_op(name, args)
        if name in {"pdintlv_b8", "pdintlv_b16", "pdintlv_b32", "pintlv_b8", "pintlv_b16", "pintlv_b32"}:
            return self._analyze_predicate_reorder_op(name, args)
        if name in {"vcmp", "vcmps"}:
            return self._analyze_compare_op(name, args)
        if name in {"vsel", "vselr", "vselrv2"}:
            return self._analyze_select_op(name, args)
        if name in {"vaddc", "vsubc", "vaddcs", "vsubcs"}:
            return self._analyze_carry_op(name, args)
        if name in {"vintlv", "vdintlv", "vintlvv2", "vdintlvv2"}:
            return self._analyze_rearrangement_op(name, args)
        if name == "vpack":
            return self._analyze_vpack_op(args)
        if name == "vcvt":
            return self._analyze_vcvt(args)
        if name == "vbitcast":
            return self._analyze_vbitcast(args)
        if name == "pbitcast":
            return self._analyze_pbitcast(args)
        if name == "vtrc":
            return self._analyze_vtrc(args)
        if name == "vbitsort":
            return self._analyze_vbitsort(args)
        if name == "vmrgsort4":
            return self._analyze_vmrgsort4(args)
        if name == "get_vms4_sr":
            return self._analyze_get_vms4_sr(args)
        if name in _BROADCAST_VECTOR_OPS:
            return self._analyze_broadcast_vector_op(name, args)
        if name in _MULTI_RESULT_VECTOR_OPS:
            return self._analyze_multi_result_vector_op(name, args)
        if name in _VEXPDIF_OP_ALIASES:
            return self._analyze_vexpdif_op(args)
        if name in _UNARY_VECTOR_OPS:
            return self._analyze_unary_vector_op(name, args)
        if name in _BINARY_VECTOR_OPS:
            return self._analyze_binary_vector_op(name, args)
        if name in _VECTOR_SCALAR_OPS:
            return self._analyze_vector_scalar_op(name, args)
        if name in _VECTOR_IMMEDIATE_OPS:
            return self._analyze_vector_immediate_op(name, args)
        if name in _TERNARY_VECTOR_OPS:
            return self._analyze_ternary_vector_op(name, args)
        raise TypeError(f"call surface `pto.{name}` is not supported in TileLang DSL v1 yet")

    def _analyze_cube_frontend_call_expr(
        self,
        expr: FrontendCallExpr,
        env: dict[str, SemanticBinding],
        *,
        allow_outer_lookup: bool,
    ) -> SemanticExpr:
        args = tuple(
            self._analyze_expr(arg, env, allow_outer_lookup=allow_outer_lookup)
            for arg in expr.args
        )
        keywords = self._analyze_keyword_args(
            expr.keywords,
            env,
            allow_outer_lookup=allow_outer_lookup,
            context=f"pto.{expr.name}",
        )

        if expr.name in _LEGACY_CUBE_ALIAS_CANONICAL:
            self._reject_legacy_cube_alias(expr.name)
        if expr.name in _CUBE_MATMUL_OPS:
            return self._analyze_cube_mad_like_op(expr.name, args, keywords)
        if expr.name in {"mte_gm_l1", "mte_l1_ub"}:
            return self._analyze_mte_load_store(expr.name, args, keywords)
        if expr.name == "mte_gm_l1_frac":
            return self._analyze_mte_gm_l1_frac(args, keywords)
        if expr.name == "mte_l1_bt":
            return self._analyze_mte_l1_bt(args, keywords)
        if expr.name == "mte_l1_fb":
            return self._analyze_mte_l1_fb(args, keywords)
        if expr.name in {"mte_l1_l0a", "mte_l1_l0b", "mte_l1_l0a_mx", "mte_l1_l0b_mx"}:
            return self._analyze_mte_stage_load(expr.name, args, keywords)
        if expr.name in {"mte_l0c_l1", "mte_l0c_gm", "mte_l0c_ub"}:
            return self._analyze_mte_l0c_store(expr.name, args, keywords)
        raise TypeError(f"call surface `pto.{expr.name}` is not supported in TileLang DSL v1 yet")

    def _reject_legacy_cube_alias(self, name: str) -> None:
        canonical = _LEGACY_CUBE_ALIAS_CANONICAL[name]
        raise TypeError(
            f"legacy cube surface `pto.{name}` is not part of the current TileLang DSL v1 public contract; "
            f"use canonical `pto.{canonical}` instead"
        )

    def _analyze_keyword_args(
        self,
        keywords: tuple[tuple[str, FrontendExprNode], ...],
        env: dict[str, SemanticBinding],
        *,
        allow_outer_lookup: bool,
        context: str,
        allowed_keywords: set[str] | None = None,
    ) -> dict[str, SemanticExpr]:
        analyzed: dict[str, SemanticExpr] = {}
        seen: set[str] = set()
        for keyword_name, keyword_value in keywords:
            if keyword_name in seen:
                raise TypeError(f"duplicate keyword `{keyword_name}` for {context} in TileLang DSL v1")
            if allowed_keywords is not None and keyword_name not in allowed_keywords:
                allowed_text = ", ".join(sorted(allowed_keywords))
                raise TypeError(
                    f"{context} only accepts keyword(s) {allowed_text} in TileLang DSL v1; "
                    f"got unsupported keyword `{keyword_name}`"
                )
            analyzed[keyword_name] = self._analyze_expr(
                keyword_value,
                env,
                allow_outer_lookup=allow_outer_lookup,
            )
            seen.add(keyword_name)
        return analyzed

    def _require_semantic_tuple_expr(
        self,
        expr: SemanticExpr,
        context: str,
        *,
        exact_len: int | None = None,
        min_len: int | None = None,
        max_len: int | None = None,
    ) -> tuple[SemanticExpr, ...]:
        if not isinstance(expr, SemanticTupleExpr):
            raise TypeError(f"{context} must be a tuple or list literal in TileLang DSL v1")
        elements = expr.elements
        if exact_len is not None and len(elements) != exact_len:
            raise TypeError(f"{context} expects exactly {exact_len} elements in TileLang DSL v1")
        if min_len is not None and len(elements) < min_len:
            raise TypeError(f"{context} expects at least {min_len} elements in TileLang DSL v1")
        if max_len is not None and len(elements) > max_len:
            raise TypeError(f"{context} expects at most {max_len} elements in TileLang DSL v1")
        return elements

    def _require_cube_pointer_expr(
        self,
        expr: SemanticExpr,
        context: str,
        *,
        memory_space: str,
    ) -> SemanticPtrType:
        ptr = self._require_pointer_expr(expr, context, memory_space=memory_space)
        return ptr.type

    def _require_matching_cube_pointer_element_dtypes(
        self,
        lhs: SemanticExpr,
        rhs: SemanticExpr,
        context: str,
        allow_mixed_low_precision: bool = False,
    ) -> None:
        lhs_dtype = lhs.type.element_dtype
        rhs_dtype = rhs.type.element_dtype
        if lhs_dtype is None or rhs_dtype is None:
            return
        if lhs_dtype != rhs_dtype:
            if allow_mixed_low_precision:
                lhs_name = getattr(lhs_dtype, "name", "")
                rhs_name = getattr(rhs_dtype, "name", "")
                allowed_mixed_sets = (
                    {"f8e4m3", "f8e5m2"},
                    {"f4e1m2x2", "f4e2m1x2"},
                )
                if any(lhs_name in dtype_set and rhs_name in dtype_set for dtype_set in allowed_mixed_sets):
                    return
            raise TypeError(f"{context} requires source/destination pointer element dtypes to match")

    def _require_cube_i64_tuple(
        self,
        expr: SemanticExpr,
        context: str,
        *,
        exact_len: int | None = None,
        min_len: int | None = None,
        max_len: int | None = None,
    ) -> SemanticTupleExpr:
        elements = self._require_semantic_tuple_expr(
            expr,
            context,
            exact_len=exact_len,
            min_len=min_len,
            max_len=max_len,
        )
        for element in elements:
            self._require_i64_like_expr(element, context)
        return SemanticTupleExpr(elements=elements, type=SemanticTupleType(elements=tuple(element.type for element in elements)))

    def _require_cube_optional_none(
        self,
        expr: SemanticExpr | None,
        context: str,
    ) -> SemanticExpr | None:
        if expr is None:
            return self._missing_optional_meta_expr()
        if isinstance(expr, SemanticLiteralExpr) and expr.value is None:
            return self._missing_optional_meta_expr()
        raise TypeError(f"{context} must be omitted or `None` in TileLang DSL v1")

    def _cube_keyword_or_default(
        self,
        keywords: dict[str, SemanticExpr],
        name: str,
        default: SemanticExpr,
    ) -> SemanticExpr:
        return keywords.get(name, default)

    def _is_none_literal_expr(self, expr: SemanticExpr | None) -> bool:
        return isinstance(expr, SemanticLiteralExpr) and expr.value is None

    def _normalize_cube_mode(
        self,
        expr: SemanticExpr,
        context: str,
        allowed_modes: set[str],
    ) -> SemanticExpr:
        if (
            isinstance(expr, SemanticSymbolExpr)
            and isinstance(expr.type, SemanticMetaType)
            and expr.type.kind == "cube_mode"
            and isinstance(expr.value, FractalMode)
        ):
            mode = expr.value.value
        elif (
            isinstance(expr, SemanticBindingRef)
            and isinstance(expr.type, SemanticMetaType)
            and expr.type.kind == "cube_mode"
            and isinstance(expr.binding.value, FractalMode)
        ):
            mode = expr.binding.value.value
        else:
            raise TypeError(f"{context} must be a FractalMode enum in TileLang DSL v1")
        if mode not in allowed_modes:
            allowed_text = " or ".join(f'\"{value}\"' for value in sorted(allowed_modes))
            raise TypeError(f"{context} must be {allowed_text} in TileLang DSL v1")
        return SemanticLiteralExpr(value=mode, type=SemanticMetaType(kind="string"))

    def _normalize_cube_loop_groups(
        self,
        expr: SemanticExpr | None,
        context: str,
    ) -> SemanticExpr:
        if expr is None:
            return self._missing_optional_meta_expr()
        if isinstance(expr, SemanticLiteralExpr) and expr.value is None:
            return self._missing_optional_meta_expr()
        if isinstance(expr, SemanticTupleExpr):
            elements = expr.elements
            if len(elements) == 3 and not any(isinstance(element, SemanticTupleExpr) for element in elements):
                single_loop = self._require_cube_i64_tuple(expr, context, exact_len=3)
                return SemanticTupleExpr(
                    elements=(single_loop,),
                    type=SemanticTupleType(elements=(single_loop.type,)),
                )
        loops = self._require_semantic_tuple_expr(expr, context)
        normalized_loops = []
        for index, loop_expr in enumerate(loops):
            loop_context = f"{context}[{index}]"
            normalized_loops.append(self._require_cube_i64_tuple(loop_expr, loop_context, exact_len=3))
        return SemanticTupleExpr(elements=tuple(normalized_loops), type=SemanticTupleType(elements=tuple(loop.type for loop in normalized_loops)))

    def _analyze_cube_mad_like_op(
        self,
        name: str,
        args: tuple[SemanticExpr, ...],
        keywords: dict[str, SemanticExpr],
    ) -> SemanticExpr:
        expected_argc = 7 if "bias" in name else 6
        if len(args) != expected_argc:
            raise TypeError(f"pto.{name} expects exactly {expected_argc} positional arguments in TileLang DSL v1")
        lhs = self._require_pointer_expr(args[0], f"pto.{name} lhs", memory_space="left")
        rhs = self._require_pointer_expr(args[1], f"pto.{name} rhs", memory_space="right")
        dst = self._require_pointer_expr(args[2], f"pto.{name} dst", memory_space="acc")
        self._require_matching_cube_pointer_element_dtypes(
            lhs,
            rhs,
            f"pto.{name}",
            allow_mixed_low_precision=True,
        )
        if "bias" in name:
            bias = self._require_pointer_expr(args[3], f"pto.{name} bias", memory_space="bias")
            self._require_matching_cube_pointer_element_dtypes(
                bias,
                dst,
                f"pto.{name}",
            )
        m_index = 4 if "bias" in name else 3
        self._require_i64_like_expr(args[m_index], f"pto.{name} m")
        self._require_i64_like_expr(args[m_index + 1], f"pto.{name} n")
        self._require_i64_like_expr(args[m_index + 2], f"pto.{name} k")
        allowed_keywords = {"unit_flag", "disable_gemv", "sat", "tf32_mode", "n_dir"}
        unsupported_keywords = sorted(set(keywords) - allowed_keywords)
        if unsupported_keywords:
            raise TypeError(
                f"pto.{name} only accepts keyword(s) unit_flag, disable_gemv, sat, tf32_mode, n_dir in TileLang DSL v1; "
                f"got unsupported keyword(s): {', '.join(unsupported_keywords)}"
            )
        unit_flag_expr = self._normalize_cube_keyword_string(
            keywords.get("unit_flag"),
            f"pto.{name} unit_flag",
            {"check_only", "check_and_set"},
            allow_none=True,
        )
        disable_gemv_expr = self._cube_keyword_or_default(
            keywords,
            "disable_gemv",
            SemanticLiteralExpr(value=False, type=SemanticScalarType(dtype=i1)),
        )
        if not isinstance(disable_gemv_expr.type, SemanticScalarType) or disable_gemv_expr.type.dtype != i1:
            raise TypeError(f"pto.{name} disable_gemv must be an i1/bool value in TileLang DSL v1")
        sat_expr = self._normalize_cube_keyword_string(
            keywords.get("sat"),
            f"pto.{name} sat",
            {"sat", "nosat"},
            allow_none=True,
        )
        tf32_mode_expr = self._normalize_cube_keyword_string(
            keywords.get("tf32_mode"),
            f"pto.{name} tf32_mode",
            {"round_even", "round_away"},
            allow_none=True,
        )
        n_dir_expr = self._cube_keyword_or_default(
            keywords,
            "n_dir",
            SemanticLiteralExpr(value=False, type=SemanticScalarType(dtype=i1)),
        )
        if not isinstance(n_dir_expr.type, SemanticScalarType) or n_dir_expr.type.dtype != i1:
            raise TypeError(f"pto.{name} n_dir must be an i1/bool value in TileLang DSL v1")
        lhs_dtype = lhs.type.element_dtype
        rhs_dtype = rhs.type.element_dtype
        dst_dtype = dst.type.element_dtype
        if not self._is_none_literal_expr(tf32_mode_expr):
            if "_mx" in name:
                raise TypeError(f"pto.{name} does not support tf32_mode in TileLang DSL v1")
            if lhs_dtype != f32 or rhs_dtype != f32 or dst_dtype != f32:
                raise TypeError(f"pto.{name} tf32_mode requires f32 lhs, rhs, and dst in TileLang DSL v1")
        if not self._is_none_literal_expr(sat_expr):
            def is_mad_sat_compatible_dtype(dtype: ScalarType) -> bool:
                if is_float_dtype(dtype):
                    return True
                if "_mx" not in name:
                    return False
                return dtype in {hif8, f8e4m3, f8e5m2, f4e1m2x2, f4e2m1x2}

            if not (
                is_mad_sat_compatible_dtype(lhs_dtype)
                and is_mad_sat_compatible_dtype(rhs_dtype)
                and is_mad_sat_compatible_dtype(dst_dtype)
            ):
                raise TypeError(f"pto.{name} sat requires a floating lhs/rhs/dst dtype combination in TileLang DSL v1")
        return SemanticCallExpr(
            namespace="pto",
            name=name,
            args=args + (unit_flag_expr, disable_gemv_expr, sat_expr, tf32_mode_expr, n_dir_expr),
            type=None,
        )

    def _extract_cube_static_int(self, expr: SemanticExpr, context: str) -> int:
        if isinstance(expr, SemanticLiteralExpr) and isinstance(expr.value, int) and not isinstance(expr.value, bool):
            return expr.value
        raise TypeError(f"{context} must be an integer constant in TileLang DSL v1")

    def _normalize_cube_keyword_string(
        self,
        expr: SemanticExpr | None,
        context: str,
        allowed_values: set[str],
        *,
        allow_none: bool = False,
    ) -> SemanticExpr:
        if expr is None:
            return self._missing_optional_meta_expr() if allow_none else None  # type: ignore[return-value]
        if self._is_none_literal_expr(expr):
            if allow_none:
                return self._missing_optional_meta_expr()
            raise TypeError(f"{context} cannot be None in TileLang DSL v1")
        value = self._require_string_expr(expr, context)
        if value not in allowed_values:
            allowed_text = ", ".join(sorted(allowed_values))
            raise TypeError(f"{context} must be one of {allowed_text} in TileLang DSL v1")
        return SemanticLiteralExpr(value=value, type=SemanticMetaType(kind="string"))

    def _normalize_cube_optional_bool_keyword(
        self,
        expr: SemanticExpr | None,
        context: str,
    ) -> SemanticExpr:
        if expr is None or self._is_none_literal_expr(expr):
            return self._missing_optional_meta_expr()
        if not isinstance(expr.type, SemanticScalarType) or expr.type.dtype != i1:
            raise TypeError(f"{context} must be an i1/bool value in TileLang DSL v1")
        return expr

    def _analyze_mte_load_store(
        self,
        name: str,
        args: tuple[SemanticExpr, ...],
        keywords: dict[str, SemanticExpr],
    ) -> SemanticExpr:
        if len(args) != 3:
            raise TypeError(f"pto.{name} expects exactly 3 positional arguments in TileLang DSL v1")
        src = self._require_pointer_expr(
            args[0],
            f"pto.{name} source",
            memory_space="gm" if name == "mte_gm_l1" else "mat",
        )
        dst = self._require_pointer_expr(
            args[1],
            f"pto.{name} destination",
            memory_space="mat" if name == "mte_gm_l1" else "ub",
        )
        self._require_matching_cube_pointer_element_dtypes(src, dst, f"pto.{name}")
        self._require_i64_like_expr(args[2], f"pto.{name} len_burst")
        allowed_keywords = {"nburst", "loops"}
        unsupported_keywords = sorted(set(keywords) - allowed_keywords)
        if unsupported_keywords:
            raise TypeError(
                f"pto.{name} only accepts keyword(s) nburst, loops in TileLang DSL v1; "
                f"got unsupported keyword(s): {', '.join(unsupported_keywords)}"
            )
        if "nburst" not in keywords:
            raise TypeError(f"pto.{name} requires keyword `nburst` in TileLang DSL v1")
        nburst_expr = self._require_cube_i64_tuple(keywords["nburst"], f"pto.{name} nburst", exact_len=3)
        loops_expr = self._normalize_cube_loop_groups(keywords.get("loops"), f"pto.{name} loops")
        return SemanticCallExpr(namespace="pto", name=name, args=(args[0], args[1], args[2], nburst_expr, loops_expr), type=None)

    def _require_grouped_mte_nburst(
        self,
        keywords: dict[str, SemanticExpr],
        context: str,
    ) -> SemanticExpr:
        if "nburst" not in keywords:
            raise TypeError(f"{context} requires keyword `nburst` in TileLang DSL v1")
        return self._require_cube_i64_tuple(keywords["nburst"], f"{context} nburst", exact_len=3)

    def _normalize_grouped_mte_pad(
        self,
        expr: SemanticExpr | None,
        context: str,
    ) -> SemanticExpr:
        if expr is None or self._is_none_literal_expr(expr):
            return self._missing_optional_meta_expr()
        tuple_expr = self._require_semantic_tuple_expr(expr, context, min_len=1, max_len=3)
        if len(tuple_expr) == 2:
            raise TypeError(f"{context} must be `pad(value)` or `pad(value, left, right)` in TileLang DSL v1")
        self._require_scalar_expr(tuple_expr[0], f"{context} value")
        if len(tuple_expr) == 3:
            self._require_i64_like_expr(tuple_expr[1], f"{context} left_padding_count")
            self._require_i64_like_expr(tuple_expr[2], f"{context} right_padding_count")
        return SemanticTupleExpr(
            elements=tuple(tuple_expr),
            type=SemanticTupleType(elements=tuple(element.type for element in tuple_expr)),
        )

    def _analyze_mte_gm_ub(
        self,
        args: tuple[SemanticExpr, ...],
        keywords: dict[str, SemanticExpr],
    ) -> SemanticCallExpr:
        if len(args) != 4:
            raise TypeError("pto.mte_gm_ub expects exactly 4 positional arguments in TileLang DSL v1")
        src = self._require_pointer_expr(args[0], "pto.mte_gm_ub source", memory_space="gm")
        dst = self._require_pointer_expr(args[1], "pto.mte_gm_ub destination", memory_space="ub")
        self._require_matching_pointer_element_dtypes(src, dst, "pto.mte_gm_ub")
        self._require_i64_like_expr(args[2], "pto.mte_gm_ub l2_cache_ctl")
        self._require_i64_like_expr(args[3], "pto.mte_gm_ub len_burst")
        allowed_keywords = {"nburst", "loops", "pad"}
        unsupported = sorted(set(keywords) - allowed_keywords)
        if unsupported:
            raise TypeError(
                "pto.mte_gm_ub only accepts keyword(s) nburst, loops, pad in TileLang DSL v1; "
                f"got unsupported keyword(s): {', '.join(unsupported)}"
            )
        nburst_expr = self._require_grouped_mte_nburst(keywords, "pto.mte_gm_ub")
        loops_expr = self._normalize_cube_loop_groups(keywords.get("loops"), "pto.mte_gm_ub loops")
        pad_expr = self._normalize_grouped_mte_pad(keywords.get("pad"), "pto.mte_gm_ub pad")
        return SemanticCallExpr(
            namespace="pto",
            name="mte_gm_ub",
            args=(args[0], args[1], args[2], args[3], nburst_expr, loops_expr, pad_expr),
            type=None,
        )

    def _analyze_mte_ub_gm(
        self,
        args: tuple[SemanticExpr, ...],
        keywords: dict[str, SemanticExpr],
    ) -> SemanticCallExpr:
        if len(args) != 3:
            raise TypeError("pto.mte_ub_gm expects exactly 3 positional arguments in TileLang DSL v1")
        src = self._require_pointer_expr(args[0], "pto.mte_ub_gm source", memory_space="ub")
        dst = self._require_pointer_expr(args[1], "pto.mte_ub_gm destination", memory_space="gm")
        self._require_matching_pointer_element_dtypes(src, dst, "pto.mte_ub_gm")
        self._require_i64_like_expr(args[2], "pto.mte_ub_gm len_burst")
        allowed_keywords = {"nburst", "loops"}
        unsupported = sorted(set(keywords) - allowed_keywords)
        if unsupported:
            raise TypeError(
                "pto.mte_ub_gm only accepts keyword(s) nburst, loops in TileLang DSL v1; "
                f"got unsupported keyword(s): {', '.join(unsupported)}"
            )
        nburst_expr = self._require_grouped_mte_nburst(keywords, "pto.mte_ub_gm")
        loops_expr = self._normalize_cube_loop_groups(keywords.get("loops"), "pto.mte_ub_gm loops")
        return SemanticCallExpr(
            namespace="pto",
            name="mte_ub_gm",
            args=(args[0], args[1], args[2], nburst_expr, loops_expr),
            type=None,
        )

    def _analyze_mte_ub_copy(
        self,
        name: str,
        args: tuple[SemanticExpr, ...],
        keywords: dict[str, SemanticExpr],
        *,
        destination_space: str,
    ) -> SemanticCallExpr:
        if len(args) != 3:
            raise TypeError(f"pto.{name} expects exactly 3 positional arguments in TileLang DSL v1")
        src = self._require_pointer_expr(args[0], f"pto.{name} source", memory_space="ub")
        dst = self._require_pointer_expr(args[1], f"pto.{name} destination", memory_space=destination_space)
        self._require_matching_pointer_element_dtypes(src, dst, f"pto.{name}")
        self._require_i64_like_expr(args[2], f"pto.{name} len_burst")
        allowed_keywords = {"nburst"}
        unsupported = sorted(set(keywords) - allowed_keywords)
        if unsupported:
            raise TypeError(
                f"pto.{name} only accepts keyword(s) nburst in TileLang DSL v1; "
                f"got unsupported keyword(s): {', '.join(unsupported)}"
            )
        nburst_expr = self._require_grouped_mte_nburst(keywords, f"pto.{name}")
        return SemanticCallExpr(
            namespace="pto",
            name=name,
            args=(args[0], args[1], args[2], nburst_expr),
            type=None,
        )

    def _analyze_mte_gm_l1_frac(
        self,
        args: tuple[SemanticExpr, ...],
        keywords: dict[str, SemanticExpr],
    ) -> SemanticExpr:
        if len(args) != 3:
            raise TypeError("pto.mte_gm_l1_frac expects exactly 3 positional arguments in TileLang DSL v1")
        src = self._require_pointer_expr(args[0], "pto.mte_gm_l1_frac source", memory_space="gm")
        dst = self._require_pointer_expr(args[1], "pto.mte_gm_l1_frac destination", memory_space="mat")
        mode = self._normalize_cube_mode(args[2], "pto.mte_gm_l1_frac mode", {FractalMode.ND2NZ.value, FractalMode.DN2NZ.value})
        self._require_matching_cube_pointer_element_dtypes(src, dst, "pto.mte_gm_l1_frac")
        allowed_keywords = {"shape", "src_layout", "dst_group", "ctrl"}
        unsupported = ", ".join(sorted(set(keywords) - allowed_keywords))
        if unsupported:
            raise TypeError(
                "pto.mte_gm_l1_frac only accepts keyword(s) shape, src_layout, dst_group, ctrl "
                f"in TileLang DSL v1; got unsupported keyword(s): {unsupported}"
            )
        missing = sorted(allowed_keywords - set(keywords))
        if missing:
            raise TypeError(f"pto.mte_gm_l1_frac requires keyword(s) {', '.join(missing)} in TileLang DSL v1")
        shape = self._require_cube_i64_tuple(keywords["shape"], "pto.mte_gm_l1_frac shape", exact_len=2)
        src_layout = self._require_cube_i64_tuple(keywords["src_layout"], "pto.mte_gm_l1_frac src_layout", min_len=1, max_len=2)
        dst_group = self._require_cube_i64_tuple(keywords["dst_group"], "pto.mte_gm_l1_frac dst_group", exact_len=4)
        ctrl = self._require_semantic_tuple_expr(keywords["ctrl"], "pto.mte_gm_l1_frac ctrl", exact_len=2)
        self._require_i64_like_expr(ctrl[0], "pto.mte_gm_l1_frac ctrl")
        if not (isinstance(ctrl[1].type, SemanticScalarType) and ctrl[1].type.dtype == i1):
            raise TypeError("pto.mte_gm_l1_frac ctrl smallc0_en must be an i1/bool value in TileLang DSL v1")
        self._validate_frac_smallc0_shape(shape, ctrl[1], "pto.mte_gm_l1_frac")
        ctrl_expr = SemanticTupleExpr(elements=ctrl, type=SemanticTupleType(elements=tuple(element.type for element in ctrl)))
        return SemanticCallExpr(namespace="pto", name="mte_gm_l1_frac", args=(args[0], args[1], mode, shape, src_layout, dst_group, ctrl_expr), type=None)

    def _analyze_mte_l1_bt(
        self,
        args: tuple[SemanticExpr, ...],
        keywords: dict[str, SemanticExpr],
    ) -> SemanticExpr:
        if len(args) != 3:
            raise TypeError("pto.mte_l1_bt expects exactly 3 positional arguments in TileLang DSL v1")
        src = self._require_pointer_expr(args[0], "pto.mte_l1_bt source", memory_space="mat")
        dst = self._require_pointer_expr(args[1], "pto.mte_l1_bt destination", memory_space="bias")
        allowed_pairs = {("f32", "f32"), ("i32", "i32"), ("f16", "f32"), ("bf16", "f32")}
        if src.type.element_dtype is not None and dst.type.element_dtype is not None and (src.type.element_dtype.name, dst.type.element_dtype.name) not in allowed_pairs:
            raise TypeError("pto.mte_l1_bt only supports f32->f32, i32->i32, f16->f32, and bf16->f32 in TileLang DSL v1")
        self._require_i64_like_expr(args[2], "pto.mte_l1_bt len_burst")
        allowed_keywords = {"nburst"}
        unsupported_keywords = sorted(set(keywords) - allowed_keywords)
        if unsupported_keywords:
            raise TypeError(
                f"pto.mte_l1_bt only accepts keyword(s) nburst in TileLang DSL v1; "
                f"got unsupported keyword(s): {', '.join(unsupported_keywords)}"
            )
        if "nburst" not in keywords:
            raise TypeError("pto.mte_l1_bt requires keyword `nburst` in TileLang DSL v1")
        nburst_expr = self._require_cube_i64_tuple(keywords["nburst"], "pto.mte_l1_bt nburst", exact_len=3)
        return SemanticCallExpr(namespace="pto", name="mte_l1_bt", args=(args[0], args[1], args[2], nburst_expr), type=None)

    def _analyze_mte_l1_fb(
        self,
        args: tuple[SemanticExpr, ...],
        keywords: dict[str, SemanticExpr],
    ) -> SemanticExpr:
        if len(args) != 3:
            raise TypeError("pto.mte_l1_fb expects exactly 3 positional arguments in TileLang DSL v1")
        src = self._require_pointer_expr(args[0], "pto.mte_l1_fb source", memory_space="mat")
        self._require_pointer_expr(args[1], "pto.mte_l1_fb destination", memory_space="scaling")
        self._require_i64_like_expr(args[2], "pto.mte_l1_fb len_burst")
        allowed_keywords = {"nburst"}
        unsupported_keywords = sorted(set(keywords) - allowed_keywords)
        if unsupported_keywords:
            raise TypeError(
                f"pto.mte_l1_fb only accepts keyword(s) nburst in TileLang DSL v1; "
                f"got unsupported keyword(s): {', '.join(unsupported_keywords)}"
            )
        if "nburst" not in keywords:
            raise TypeError("pto.mte_l1_fb requires keyword `nburst` in TileLang DSL v1")
        nburst_expr = self._require_cube_i64_tuple(keywords["nburst"], "pto.mte_l1_fb nburst", exact_len=3)
        return SemanticCallExpr(namespace="pto", name="mte_l1_fb", args=(args[0], args[1], args[2], nburst_expr), type=None)

    def _analyze_mte_stage_load(
        self,
        name: str,
        args: tuple[SemanticExpr, ...],
        keywords: dict[str, SemanticExpr],
    ) -> SemanticExpr:
        if len(args) != 4:
            raise TypeError(f"pto.{name} expects exactly 4 positional arguments in TileLang DSL v1")
        allowed_keywords: set[str] = set()
        supports_transpose = name in {"mte_l1_l0a", "mte_l1_l0b"}
        if supports_transpose:
            allowed_keywords.add("transpose")
        supports_start_position = name in {
            "mte_l1_l0a",
            "mte_l1_l0b",
            "mte_l1_l0a_mx",
            "mte_l1_l0b_mx",
        }
        if supports_start_position:
            allowed_keywords |= {"start_row", "start_col"}
        unsupported = sorted(set(keywords) - allowed_keywords)
        if unsupported:
            raise TypeError(
                f"pto.{name} only accepts keyword(s) {', '.join(sorted(allowed_keywords))} in TileLang DSL v1; "
                f"got unsupported keyword(s): {', '.join(unsupported)}"
            )
        src = self._require_pointer_expr(args[0], f"pto.{name} source", memory_space="mat")
        dst_space = "left" if name in {"mte_l1_l0a", "mte_l1_l0a_mx"} else "right"
        dst = self._require_pointer_expr(args[1], f"pto.{name} destination", memory_space=dst_space)
        self._require_matching_cube_pointer_element_dtypes(src, dst, f"pto.{name}")
        self._require_i64_like_expr(args[2], f"pto.{name} first dimension")
        self._require_i64_like_expr(args[3], f"pto.{name} second dimension")
        start_row = self._cube_keyword_or_default(
            keywords,
            "start_row",
            SemanticLiteralExpr(value=0, type=SemanticIndexType()),
        )
        start_col = self._cube_keyword_or_default(
            keywords,
            "start_col",
            SemanticLiteralExpr(value=0, type=SemanticIndexType()),
        )
        self._require_i64_like_expr(start_row, f"pto.{name} start_row")
        self._require_i64_like_expr(start_col, f"pto.{name} start_col")
        if supports_transpose:
            transpose = self._cube_keyword_or_default(
                keywords,
                "transpose",
                SemanticLiteralExpr(value=False, type=SemanticScalarType(dtype=i1)),
            )
            if not isinstance(transpose.type, SemanticScalarType) or transpose.type.dtype != i1:
                raise TypeError(f"pto.{name} transpose must be an i1/bool value in TileLang DSL v1")
            return SemanticCallExpr(
                namespace="pto",
                name=name,
                args=args + (start_row, start_col, transpose),
                type=None,
            )
        return SemanticCallExpr(
            namespace="pto",
            name=name,
            args=args + (start_row, start_col),
            type=None,
        )

    def _analyze_mte_l0c_store(
        self,
        name: str,
        args: tuple[SemanticExpr, ...],
        keywords: dict[str, SemanticExpr],
    ) -> SemanticExpr:
        expected_argc = 8 if name == "mte_l0c_gm" else 7 if name == "mte_l0c_ub" else 6
        if len(args) != expected_argc:
            raise TypeError(f"pto.{name} expects exactly {expected_argc} positional arguments in TileLang DSL v1")
        src = self._require_pointer_expr(args[0], f"pto.{name} source", memory_space="acc")
        dst_space = "mat" if name == "mte_l0c_l1" else "gm" if name == "mte_l0c_gm" else "ub"
        dst = self._require_pointer_expr(args[1], f"pto.{name} destination", memory_space=dst_space)
        for index, label in enumerate(("m", "n", "src_stride", "dst_stride"), start=2):
            self._require_i64_like_expr(args[index], f"pto.{name} {label}")
        cursor = 6
        extra_args: list[SemanticExpr] = []
        if name == "mte_l0c_gm":
            self._require_i64_like_expr(args[cursor], f"pto.{name} sid")
            self._require_i64_like_expr(args[cursor + 1], f"pto.{name} l2_cache_ctrl")
            sid_value = self._cube_static_int_if_available(args[cursor])
            if sid_value is not None and not 0 <= sid_value <= 3:
                raise TypeError(f"pto.{name} sid constant must be in [0, 3] in TileLang DSL v1")
            l2_value = self._cube_static_int_if_available(args[cursor + 1])
            if l2_value is not None and not 0 <= l2_value <= 15:
                raise TypeError(f"pto.{name} l2_cache_ctrl constant must be in [0, 15] in TileLang DSL v1")
            extra_args.extend([args[cursor], args[cursor + 1]])
            cursor += 2
        elif name == "mte_l0c_ub":
            dst_mode_raw = args[cursor]
            dst_mode_kind = "sub_blockid"
            dst_mode_value: SemanticExpr = dst_mode_raw
            if not self._is_none_literal_expr(dst_mode_raw):
                try:
                    mode_text = self._require_string_expr(dst_mode_raw, f"pto.{name} dst_mode")
                    if mode_text not in {"split_m", "split_n"}:
                        raise TypeError
                    dst_mode_kind = mode_text
                    dst_mode_value = self._missing_optional_meta_expr()
                except TypeError:
                    self._require_i64_like_expr(dst_mode_raw, f"pto.{name} dst_mode")
                    sub_blockid_value = self._cube_static_int_if_available(dst_mode_raw)
                    if sub_blockid_value is not None and sub_blockid_value not in {0, 1}:
                        raise TypeError(f"pto.{name} dst_mode sub_blockid constant must be 0 or 1 in TileLang DSL v1")
            extra_args.extend(
                [
                    SemanticLiteralExpr(value=dst_mode_kind, type=SemanticMetaType(kind="string")),
                    dst_mode_value,
                ]
            )
            cursor += 1

        allowed_keywords = {"unit_flag", "pre_quant", "pre_relu", "layout", "loop3", "sat"}
        if name == "mte_l0c_gm":
            allowed_keywords.add("atomic")
        unsupported = sorted(set(keywords) - allowed_keywords)
        if unsupported:
            raise TypeError(
                f"pto.{name} only accepts keyword(s) {', '.join(sorted(allowed_keywords))} in TileLang DSL v1; "
                f"got unsupported keyword(s): {', '.join(unsupported)}"
            )
        unit_flag = self._normalize_cube_keyword_string(
            keywords.get("unit_flag"),
            f"pto.{name} unit_flag",
            {"check_only", "check_and_clear"},
            allow_none=True,
        )
        pre_quant_payload = self._missing_optional_meta_expr()
        pre_quant_mode = self._missing_optional_meta_expr()
        if "pre_quant" in keywords:
            pre_quant = self._require_semantic_tuple_expr(keywords["pre_quant"], f"pto.{name} pre_quant", exact_len=2)
            pre_quant_payload = pre_quant[0]
            pre_quant_mode = self._normalize_cube_keyword_string(
                pre_quant[1],
                f"pto.{name} pre_quant mode",
                {
                    "f32_f16", "qf322hif8_pre_vec", "qf322hif8_pre_scalar", "qf322hif8_pre_hybrid_vec",
                    "qf322hif8_pre_hybrid_scalar", "deqs32_int_vec", "deqs32_int_scalar", "req8_vec",
                    "req8_scalar", "deqf16_vec", "deqf16_scalar", "qf322fp8_pre_vec", "qf322fp8_pre_scalar",
                    "qf322f32_pre_vec", "qf322f32_pre_scalar", "f32_bf16", "qf162b8_pre_vec",
                    "qf162b8_pre_scalar", "qf162s4_pre_vec", "qf162s4_pre_scalar", "req4_vec",
                    "req4_scalar", "qf322b8_pre_vec", "qf322b8_pre_scalar", "qf322s4_pre_vec",
                    "qf322s4_pre_scalar", "deqs16_vec", "deqs16_scalar", "qf162s16_pre_vec",
                    "qf162s16_pre_scalar", "qf322f16_pre_vec", "qf322f16_pre_scalar", "qf322bf16_pre_vec",
                    "qf322bf16_pre_scalar", "qs322bf16_pre_vec", "qs322bf16_pre_scalar",
                },
            )
            if src.type.element_dtype.name not in {"f32", "i32", "si32"}:
                raise TypeError(f"pto.{name} pre_quant requires f32, i32, or si32 source elements in TileLang DSL v1")
            self._validate_fixpipe_payload(
                pre_quant_payload,
                pre_quant_mode,
                context=f"pto.{name} pre_quant",
                scalar_required_suffix="_scalar",
                vector_required_suffix="_vec",
            )
            self._validate_fixpipe_pre_quant_mode_compatibility(
                src_dtype=src.type.element_dtype,
                dst_dtype=dst.type.element_dtype,
                mode=pre_quant_mode,
                context=f"pto.{name} pre_quant",
            )
        pre_relu_payload = self._missing_optional_meta_expr()
        pre_relu_mode = self._missing_optional_meta_expr()
        clip_expr = self._missing_optional_meta_expr()
        if "pre_relu" in keywords:
            pre_relu = self._require_semantic_tuple_expr(keywords["pre_relu"], f"pto.{name} pre_relu", exact_len=3)
            pre_relu_mode = self._normalize_cube_keyword_string(
                pre_relu[0],
                f"pto.{name} pre_relu mode",
                {"no_relu", "normal_relu", "scalar_relu", "vector_relu"},
            )
            if not self._is_none_literal_expr(pre_relu[1]):
                pre_relu_payload = pre_relu[1]
            if not self._is_none_literal_expr(pre_relu[2]):
                clip_expr = pre_relu[2]
            pre_relu_mode_value = self._require_string_expr(pre_relu_mode, f"pto.{name} pre_relu mode")
            if pre_relu_mode_value in {"no_relu", "normal_relu"}:
                if not self._is_none_literal_expr(pre_relu_payload):
                    raise TypeError(f"pto.{name} pre_relu mode {pre_relu_mode_value} does not accept a payload in TileLang DSL v1")
            elif pre_relu_mode_value == "scalar_relu":
                if self._is_none_literal_expr(pre_relu_payload):
                    raise TypeError(f"pto.{name} pre_relu mode scalar_relu requires a scalar payload in TileLang DSL v1")
                self._require_fixpipe_scalar_payload(pre_relu_payload, f"pto.{name} pre_relu scalar payload")
            elif pre_relu_mode_value == "vector_relu":
                if self._is_none_literal_expr(pre_relu_payload):
                    raise TypeError(f"pto.{name} pre_relu mode vector_relu requires an fb payload in TileLang DSL v1")
                self._require_fixpipe_vector_payload(pre_relu_payload, f"pto.{name} pre_relu vector payload")
            self._validate_fixpipe_clip_compatibility(
                clip_expr,
                dst_dtype=dst.type.element_dtype,
                context=f"pto.{name} pre_relu clip",
            )
        layout_mode = self._missing_optional_meta_expr()
        layout_arg = self._missing_optional_meta_expr()
        if "layout" in keywords:
            layout_expr = keywords["layout"]
            if self._is_none_literal_expr(layout_expr):
                pass
            elif isinstance(layout_expr, SemanticTupleExpr):
                layout_parts = self._require_semantic_tuple_expr(layout_expr, f"pto.{name} layout", exact_len=2)
                layout_mode = self._normalize_cube_keyword_string(
                    layout_parts[0],
                    f"pto.{name} layout mode",
                    {"nz2dn", "nz2nz"},
                )
                layout_arg = layout_parts[1]
                self._require_i64_like_expr(layout_arg, f"pto.{name} layout operand")
            else:
                layout_mode = self._normalize_cube_keyword_string(
                    layout_expr,
                    f"pto.{name} layout mode",
                    {"nz2nd"},
                )
        loop3_expr = self._normalize_cube_loop_groups(keywords.get("loop3"), f"pto.{name} loop3")
        if isinstance(loop3_expr, SemanticTupleExpr) and len(loop3_expr.elements) != 1:
            raise TypeError(f"pto.{name} loop3 expects exactly one loop triple in TileLang DSL v1")
        loop3_value: SemanticExpr
        if isinstance(loop3_expr, SemanticTupleExpr):
            loop3_value = loop3_expr.elements[0]
        else:
            loop3_value = loop3_expr
        sat_expr = self._normalize_cube_keyword_string(
            keywords.get("sat"),
            f"pto.{name} sat",
            {"sat", "sat(preserve_nan)", "nosat"},
            allow_none=True,
        )
        atomic_type = self._missing_optional_meta_expr()
        atomic_op = self._missing_optional_meta_expr()
        if "atomic" in keywords:
            if name != "mte_l0c_gm":
                raise TypeError(f"pto.{name} does not support atomic in TileLang DSL v1")
            atomic_tuple = self._require_semantic_tuple_expr(keywords["atomic"], f"pto.{name} atomic", exact_len=2)
            atomic_type = self._normalize_cube_keyword_string(
                atomic_tuple[0],
                f"pto.{name} atomic type",
                {"f32", "f16", "bf16", "s32", "s16", "s8"},
            )
            atomic_op = self._normalize_cube_keyword_string(
                atomic_tuple[1],
                f"pto.{name} atomic op",
                {"add", "max", "min"},
            )
        layout_mode_value = None if self._is_none_literal_expr(layout_mode) else self._require_string_expr(layout_mode, f"pto.{name} layout mode")
        layout_arg_value = self._cube_static_int_if_available(layout_arg)
        if layout_mode_value == "nz2dn" and not self._is_none_literal_expr(unit_flag):
            if layout_arg_value is not None and layout_arg_value != 1:
                raise TypeError(f"pto.{name} unit_flag must be omitted when nz2dn loop0_src_stride is not 1 in TileLang DSL v1")
        if layout_mode_value == "nz2nz":
            if dst.type.element_dtype != f32:
                raise TypeError(f"pto.{name} nz2nz requires an f32 destination in TileLang DSL v1")
            if not self._is_none_literal_expr(loop3_value):
                raise TypeError(f"pto.{name} nz2nz does not support loop3 in TileLang DSL v1")
        if name == "mte_l0c_ub":
            dst_mode_kind_expr = extra_args[0]
            dst_mode_kind_value = self._require_string_expr(dst_mode_kind_expr, f"pto.{name} dst_mode kind")
            if dst_mode_kind_value in {"split_m", "split_n"}:
                m_value = self._cube_static_int_if_available(args[2])
                n_value = self._cube_static_int_if_available(args[3])
                if dst_mode_kind_value == "split_m" and m_value is not None and m_value % 2 != 0:
                    raise TypeError("pto.mte_l0c_ub dst_mode split_m requires even m in TileLang DSL v1")
                if dst_mode_kind_value == "split_n" and n_value is not None and n_value % 32 != 0:
                    raise TypeError("pto.mte_l0c_ub dst_mode split_n requires n to be a multiple of 32 in TileLang DSL v1")
                if layout_mode_value not in {None, "nz2nd"}:
                    raise TypeError("pto.mte_l0c_ub split_m/split_n only support normal or nz2nd layout in TileLang DSL v1")
                if not (
                    self._is_none_literal_expr(unit_flag)
                    and self._is_none_literal_expr(pre_quant_mode)
                    and self._is_none_literal_expr(pre_relu_mode)
                    and self._is_none_literal_expr(clip_expr)
                    and self._is_none_literal_expr(loop3_value)
                    and self._is_none_literal_expr(sat_expr)
                ):
                    raise TypeError(
                        "pto.mte_l0c_ub split_m/split_n do not support unit_flag, pre_quant, pre_relu, clip, loop3, or sat in TileLang DSL v1"
                    )
        base_args = args[:6] if name in {"mte_l0c_gm", "mte_l0c_ub"} else args[:cursor]
        return SemanticCallExpr(
            namespace="pto",
            name=name,
            args=tuple(base_args) + tuple(extra_args) + (
                unit_flag,
                pre_quant_payload,
                pre_quant_mode,
                pre_relu_payload,
                pre_relu_mode,
                clip_expr,
                layout_mode,
                layout_arg,
                loop3_value,
                sat_expr,
                atomic_type,
                atomic_op,
            ),
            type=None,
        )

    def _cube_static_int_if_available(self, expr: SemanticExpr) -> int | None:
        if isinstance(expr, SemanticLiteralExpr) and isinstance(expr.value, int) and not isinstance(expr.value, bool):
            return expr.value
        return None

    def _validate_frac_smallc0_shape(
        self,
        shape: SemanticExpr,
        smallc0_en_expr: SemanticExpr,
        op_name: str,
    ) -> None:
        smallc0_en = self._try_static_value(smallc0_en_expr)
        if smallc0_en is not True:
            return
        if not isinstance(shape, SemanticTupleExpr) or len(shape.elements) != 2:
            return
        d_value = self._cube_static_int_if_available(shape.elements[1])
        if d_value is not None and d_value > 4:
            raise TypeError(
                f"{op_name} small-C0 mode requires shape d_value <= 4 when `ctrl(..., True)` is statically known in TileLang DSL v1"
            )

    def _require_fixpipe_scalar_payload(self, expr: SemanticExpr, context: str) -> None:
        if not isinstance(expr.type, SemanticScalarType) or expr.type.dtype not in {f16, bf16, f32}:
            raise TypeError(f"{context} must be an f16, bf16, or f32 scalar in TileLang DSL v1")

    def _require_fixpipe_vector_payload(self, expr: SemanticExpr, context: str) -> None:
        ptr = self._require_pointer_expr(expr, context, memory_space="scaling")
        if ptr.type.element_dtype not in {f16, bf16, f32, ui64}:
            raise TypeError(
                f"{context} must be an fb pointer with f16, bf16, f32 elements or packed ui64 Quant_PRE entries in TileLang DSL v1"
            )

    def _validate_fixpipe_payload(
        self,
        payload: SemanticExpr,
        mode: SemanticExpr,
        *,
        context: str,
        scalar_required_suffix: str,
        vector_required_suffix: str,
    ) -> None:
        mode_value = self._require_string_expr(mode, f"{context} mode")
        if mode_value in {"f32_f16", "f32_bf16"}:
            # f32_f16 / f32_bf16 only accept scalar payload per hardware spec.
            # Vector/scaling payload is NOT supported for these modes.
            self._require_fixpipe_scalar_payload(payload, f"{context} payload")
        elif mode_value.endswith(scalar_required_suffix):
            self._require_fixpipe_scalar_payload(payload, f"{context} payload")
        elif mode_value.endswith(vector_required_suffix):
            self._require_fixpipe_vector_payload(payload, f"{context} payload")

    def _fixpipe_pre_quant_mode_families(self, mode_value: str) -> tuple[str | None, str | None]:
        if mode_value in {"f32_f16", "f32_bf16"}:
            return ("f32", mode_value.split("_", 1)[1])
        if mode_value.startswith("qf322") or mode_value.startswith("qf162"):
            if "f32_pre" in mode_value:
                return ("f32", "f32")
            if "bf16_pre" in mode_value:
                return ("f32", "bf16")
            if "f16_pre" in mode_value:
                return ("f32", "f16")
            if "b8" in mode_value:
                return ("f32", "i8_family")
            if "s4" in mode_value:
                return ("f32", "i4_family")
            if "s16" in mode_value:
                return ("i32", "i16_family")
            if "hif8" in mode_value or "fp8" in mode_value:
                return ("f32", "fp8_family")
        if mode_value.startswith("deqs32_int"):
            return ("i32", "i32")
        if mode_value.startswith("deqs16"):
            return ("i32", "i16_family")
        if mode_value.startswith("req8"):
            return ("i32", "i8_family")
        if mode_value.startswith("req4"):
            return ("i32", "i4_family")
        if mode_value.startswith("deqf16"):
            return ("i32", "f16")
        if mode_value.startswith("qs322bf16"):
            return ("i32", "bf16")
        return (None, None)

    def _fixpipe_pre_quant_src_family_matches(self, src_dtype: ScalarType, src_family: str | None) -> bool:
        if src_family is None:
            return True
        if src_family == "f32":
            return src_dtype.name == "f32"
        if src_family == "i32":
            return src_dtype.name in {"i32", "si32"}
        return src_dtype.name == src_family

    def _fixpipe_pre_quant_dst_family_matches(self, dst_dtype: ScalarType, dst_family: str | None) -> bool:
        if dst_family is None:
            return True
        if dst_family in {"f16", "bf16", "f32", "i32"}:
            return dst_dtype.name == dst_family
        if dst_family == "i16_family":
            return is_integer_dtype(dst_dtype) and integer_bitwidth(dst_dtype) == 16 and integer_signedness(dst_dtype) != "unsigned"
        if dst_family == "i8_family":
            return is_integer_dtype(dst_dtype) and integer_bitwidth(dst_dtype) == 8
        if dst_family == "i4_family":
            return False
        if dst_family == "fp8_family":
            return False
        return False

    def _validate_fixpipe_pre_quant_mode_compatibility(
        self,
        *,
        src_dtype: ScalarType,
        dst_dtype: ScalarType,
        mode: SemanticExpr,
        context: str,
    ) -> None:
        mode_value = self._require_string_expr(mode, f"{context} mode")
        expected_src_family, expected_dst_family = self._fixpipe_pre_quant_mode_families(mode_value)
        if not self._fixpipe_pre_quant_src_family_matches(src_dtype, expected_src_family):
            raise TypeError(
                f"{context} mode {mode_value} requires {expected_src_family} source elements in TileLang DSL v1"
            )
        if not self._fixpipe_pre_quant_dst_family_matches(dst_dtype, expected_dst_family):
            raise TypeError(
                f"{context} mode {mode_value} is not compatible with destination dtype {dst_dtype.name} in TileLang DSL v1"
            )

    def _validate_fixpipe_clip_compatibility(
        self,
        clip_expr: SemanticExpr,
        *,
        dst_dtype: ScalarType,
        context: str,
    ) -> None:
        if self._is_none_literal_expr(clip_expr):
            return
        if dst_dtype == f16:
            if not isinstance(clip_expr.type, SemanticScalarType) or clip_expr.type.dtype != f16:
                raise TypeError(f"{context} requires an f16 clip payload for f16 destinations in TileLang DSL v1")
            return
        if dst_dtype == ui8:
            if not isinstance(clip_expr.type, SemanticScalarType) or clip_expr.type.dtype not in {ui16, i16}:
                raise TypeError(f"{context} requires a ui16/i16 clip payload for ui8 destinations in TileLang DSL v1")
            return
        if is_integer_dtype(dst_dtype):
            bits = integer_bitwidth(dst_dtype)
            signedness = integer_signedness(dst_dtype)
            if bits in {8, 16} and signedness != "unsigned":
                if not isinstance(clip_expr.type, SemanticScalarType):
                    raise TypeError(f"{context} requires an integer clip payload matching the destination family in TileLang DSL v1")
                clip_dtype = clip_expr.type.dtype
                if not (
                    is_integer_dtype(clip_dtype)
                    and integer_bitwidth(clip_dtype) == bits
                    and integer_signedness(clip_dtype) != "unsigned"
                ):
                    raise TypeError(
                        f"{context} requires an i{bits}/si{bits} clip payload for {dst_dtype.name} destinations in TileLang DSL v1"
                    )
                return
        raise TypeError(
            f"{context} is only supported for destination f16, ui8, and signed/signless 8/16-bit integer dtypes in TileLang DSL v1"
        )

    def _analyze_cube_load_store(
        self,
        name: str,
        args: tuple[SemanticExpr, ...],
        keywords: dict[str, SemanticExpr],
    ) -> SemanticExpr:
        if len(args) != 3:
            raise TypeError(f"pto.{name} expects exactly 3 positional arguments in TileLang DSL v1")
        src = self._require_pointer_expr(
            args[0],
            f"pto.{name} source",
            memory_space="gm" if name == "cube_load" else "mat",
        )
        dst = self._require_pointer_expr(
            args[1],
            f"pto.{name} destination",
            memory_space="mat" if name == "cube_load" else "ub",
        )
        self._require_matching_cube_pointer_element_dtypes(
            src,
            dst,
            f"pto.{name}",
        )
        self._require_i64_like_expr(args[2], f"pto.{name} len_burst")
        allowed_keywords = {"nburst", "loops"}
        unsupported_keywords = sorted(set(keywords) - allowed_keywords)
        if unsupported_keywords:
            raise TypeError(
                f"pto.{name} only accepts keyword(s) nburst, loops in TileLang DSL v1; "
                f"got unsupported keyword(s): {', '.join(unsupported_keywords)}"
            )
        nburst_expr = keywords.get("nburst", SemanticTupleExpr(
            elements=(
                SemanticLiteralExpr(value=1, type=SemanticIndexType()),
                SemanticLiteralExpr(value=0, type=SemanticIndexType()),
                SemanticLiteralExpr(value=0, type=SemanticIndexType()),
            ),
            type=SemanticTupleType(elements=(SemanticIndexType(), SemanticIndexType(), SemanticIndexType())),
        ))
        if "nburst" in keywords:
            nburst_expr = self._require_cube_i64_tuple(keywords["nburst"], f"pto.{name} nburst", exact_len=3)
        loops_expr = self._normalize_cube_loop_groups(keywords.get("loops"), f"pto.{name} loops")
        return SemanticCallExpr(
            namespace="pto",
            name=name,
            args=(args[0], args[1], args[2], nburst_expr, loops_expr),
            type=None,
        )

    def _analyze_cube_load_frac(
        self,
        args: tuple[SemanticExpr, ...],
        keywords: dict[str, SemanticExpr],
    ) -> SemanticExpr:
        if len(args) != 3:
            raise TypeError("pto.cube_load_frac expects exactly 3 positional arguments in TileLang DSL v1")
        src = self._require_pointer_expr(args[0], "pto.cube_load_frac source", memory_space="gm")
        dst = self._require_pointer_expr(args[1], "pto.cube_load_frac destination", memory_space="mat")
        mode = self._normalize_cube_mode(
            args[2],
            "pto.cube_load_frac mode",
            {FractalMode.ND2NZ.value, FractalMode.DN2NZ.value},
        )
        self._require_matching_cube_pointer_element_dtypes(
            src,
            dst,
            "pto.cube_load_frac",
        )
        allowed_keywords = {"shape", "src_layout", "dst_group", "ctrl"}
        unsupported = ", ".join(sorted(set(keywords) - allowed_keywords))
        if unsupported:
            raise TypeError(
                "pto.cube_load_frac only accepts keyword(s) shape, src_layout, dst_group, ctrl "
                f"in TileLang DSL v1; got unsupported keyword(s): {unsupported}"
            )
        missing = sorted(allowed_keywords - set(keywords))
        if missing:
            raise TypeError(
                f"pto.cube_load_frac requires keyword(s) {', '.join(missing)} in TileLang DSL v1"
            )
        shape = self._require_cube_i64_tuple(keywords["shape"], "pto.cube_load_frac shape", exact_len=2)
        src_layout = self._require_cube_i64_tuple(keywords["src_layout"], "pto.cube_load_frac src_layout", min_len=1, max_len=2)
        dst_group = self._require_cube_i64_tuple(keywords["dst_group"], "pto.cube_load_frac dst_group", exact_len=4)
        ctrl = self._require_semantic_tuple_expr(keywords["ctrl"], "pto.cube_load_frac ctrl", exact_len=2)
        self._require_i64_like_expr(ctrl[0], "pto.cube_load_frac ctrl")
        if not (
            isinstance(ctrl[1].type, SemanticScalarType)
            and ctrl[1].type.dtype == i1
        ):
            raise TypeError("pto.cube_load_frac ctrl smallc0_en must be an i1/bool value in TileLang DSL v1")
        self._validate_frac_smallc0_shape(shape, ctrl[1], "pto.cube_load_frac")
        ctrl_expr = SemanticTupleExpr(
            elements=ctrl,
            type=SemanticTupleType(elements=tuple(element.type for element in ctrl)),
        )
        return SemanticCallExpr(
            namespace="pto",
            name="cube_load_frac",
            args=(args[0], args[1], mode, shape, src_layout, dst_group, ctrl_expr),
            type=None,
        )

    def _analyze_cube_bias_load(
        self,
        args: tuple[SemanticExpr, ...],
        keywords: dict[str, SemanticExpr],
    ) -> SemanticExpr:
        if len(args) != 3:
            raise TypeError("pto.bias_load expects exactly 3 positional arguments in TileLang DSL v1")
        src = self._require_pointer_expr(args[0], "pto.bias_load source", memory_space="mat")
        dst = self._require_pointer_expr(args[1], "pto.bias_load destination", memory_space="bias")
        allowed_pairs = {
            ("f32", "f32"),
            ("i32", "i32"),
            ("f16", "f32"),
            ("bf16", "f32"),
        }
        if src.type.element_dtype is not None and dst.type.element_dtype is not None and (
            src.type.element_dtype.name,
            dst.type.element_dtype.name,
        ) not in allowed_pairs:
            raise TypeError(
                "pto.bias_load only supports f32->f32, i32->i32, f16->f32, and bf16->f32 in TileLang DSL v1"
            )
        self._require_i64_like_expr(args[2], "pto.bias_load len_burst")
        allowed_keywords = {"nburst"}
        unsupported_keywords = sorted(set(keywords) - allowed_keywords)
        if unsupported_keywords:
            raise TypeError(
                f"pto.bias_load only accepts keyword(s) nburst in TileLang DSL v1; "
                f"got unsupported keyword(s): {', '.join(unsupported_keywords)}"
            )
        nburst_expr = keywords.get("nburst", SemanticTupleExpr(
            elements=(
                SemanticLiteralExpr(value=1, type=SemanticIndexType()),
                SemanticLiteralExpr(value=0, type=SemanticIndexType()),
                SemanticLiteralExpr(value=0, type=SemanticIndexType()),
            ),
            type=SemanticTupleType(elements=(SemanticIndexType(), SemanticIndexType(), SemanticIndexType())),
        ))
        if "nburst" in keywords:
            nburst_expr = self._require_cube_i64_tuple(keywords["nburst"], "pto.bias_load nburst", exact_len=3)
        return SemanticCallExpr(
            namespace="pto",
            name="bias_load",
            args=(args[0], args[1], args[2], nburst_expr),
            type=None,
        )

    def _analyze_cube_stage_load(
        self,
        name: str,
        args: tuple[SemanticExpr, ...],
        keywords: dict[str, SemanticExpr],
    ) -> SemanticExpr:
        if keywords:
            raise TypeError(f"pto.{name} does not accept keyword arguments in TileLang DSL v1")
        if len(args) != 4:
            raise TypeError(f"pto.{name} expects exactly 4 positional arguments in TileLang DSL v1")
        src = self._require_pointer_expr(args[0], f"pto.{name} source", memory_space="mat")
        dst_space = "left" if name.startswith("left") else "right"
        dst = self._require_pointer_expr(args[1], f"pto.{name} destination", memory_space=dst_space)
        self._require_matching_cube_pointer_element_dtypes(
            src,
            dst,
            f"pto.{name}",
        )
        self._require_i64_like_expr(args[2], f"pto.{name} first dimension")
        self._require_i64_like_expr(args[3], f"pto.{name} second dimension")
        return SemanticCallExpr(namespace="pto", name=name, args=args, type=None)

    def _analyze_cube_acc_store(
        self,
        name: str,
        args: tuple[SemanticExpr, ...],
        keywords: dict[str, SemanticExpr],
    ) -> SemanticExpr:
        if len(args) != 6:
            raise TypeError(f"pto.{name} expects exactly 6 positional arguments in TileLang DSL v1")
        src = self._require_pointer_expr(args[0], f"pto.{name} source", memory_space="acc")
        dst_space = "mat" if name == "acc_store" else "gm" if name == "acc_store_gm" else "ub"
        dst = self._require_pointer_expr(args[1], f"pto.{name} destination", memory_space=dst_space)
        for index, label in enumerate(("m", "n", "src_stride", "dst_stride"), start=2):
            self._require_i64_like_expr(args[index], f"pto.{name} {label}")

        allowed_keywords = {
            "mode",
            "loop0_src_stride",
            "split",
            "loop3",
        }
        if name == "acc_store_gm":
            allowed_keywords |= {"sid", "l2_cache_ctrl"}
        if name == "acc_store_ub":
            allowed_keywords = {
                "mode",
                "loop0_src_stride",
                "channel_split_en",
                "loop3",
                "dual_dst_mode",
                "sub_blockid",
            }
        unsupported = sorted(set(keywords) - allowed_keywords)
        if unsupported:
            raise TypeError(
                f"pto.{name} only accepts keyword(s) {', '.join(sorted(allowed_keywords))} in TileLang DSL v1; "
                f"got unsupported keyword(s): {', '.join(unsupported)}"
            )

        mode = self._normalize_cube_mode(
            keywords.get(
                "mode",
                SemanticSymbolExpr(
                    namespace="pto.FractalMode",
                    name="NZ2ND",
                    value=FractalMode.NZ2ND,
                    type=SemanticMetaType(kind="cube_mode"),
                ),
            ),
            f"pto.{name} mode",
            {FractalMode.NZ2ND.value, FractalMode.NZ2DN.value, FractalMode.NZ2NZ.value},
        )
        mode_text = self._require_string_expr(mode, f"pto.{name} mode")

        loop0_src_stride = keywords.get("loop0_src_stride")
        split_key = "channel_split_en" if name == "acc_store_ub" else "split"
        split_expr = keywords.get(split_key)
        loop3_expr = keywords.get("loop3")

        if self._is_none_literal_expr(loop0_src_stride):
            loop0_src_stride = None
        if self._is_none_literal_expr(split_expr):
            split_expr = None
        if self._is_none_literal_expr(loop3_expr):
            loop3_expr = None

        if mode_text == "nz2nd":
            if loop0_src_stride is not None:
                raise TypeError(f"pto.{name} mode \"nz2nd\" does not accept loop0_src_stride in TileLang DSL v1")
            if split_expr is not None:
                raise TypeError(f"pto.{name} mode \"nz2nd\" does not accept {split_key} in TileLang DSL v1")
        elif mode_text == "nz2dn":
            if split_expr is not None:
                raise TypeError(f"pto.{name} mode \"nz2dn\" does not accept {split_key} in TileLang DSL v1")
        elif mode_text == "nz2nz":
            if loop0_src_stride is not None:
                raise TypeError(f"pto.{name} mode \"nz2nz\" does not accept loop0_src_stride in TileLang DSL v1")
            if split_expr is None:
                raise TypeError(f"pto.{name} mode \"nz2nz\" requires {split_key} in TileLang DSL v1")
            if loop3_expr is not None:
                raise TypeError(f"pto.{name} mode \"nz2nz\" does not accept loop3(...) in TileLang DSL v1")

        if loop0_src_stride is not None:
            self._require_i64_like_expr(loop0_src_stride, f"pto.{name} loop0_src_stride")
        if split_expr is not None:
            self._require_i64_like_expr(split_expr, f"pto.{name} {split_key}")
        if loop3_expr is not None:
            loop3_expr = self._require_cube_i64_tuple(loop3_expr, f"pto.{name} loop3", exact_len=3)

        tail_args: list[SemanticExpr] = []
        if name == "acc_store_gm":
            sid_expr = keywords.get("sid", SemanticLiteralExpr(value=0, type=SemanticIndexType()))
            l2_cache_ctrl_expr = keywords.get("l2_cache_ctrl", SemanticLiteralExpr(value=0, type=SemanticIndexType()))
            self._require_i64_like_expr(sid_expr, f"pto.{name} sid")
            self._require_i64_like_expr(l2_cache_ctrl_expr, f"pto.{name} l2_cache_ctrl")
            tail_args.extend([sid_expr, l2_cache_ctrl_expr])
        elif name == "acc_store_ub":
            dual_dst_mode_expr = keywords.get("dual_dst_mode", SemanticLiteralExpr(value=0, type=SemanticIndexType()))
            sub_blockid_expr = keywords.get("sub_blockid", SemanticLiteralExpr(value=0, type=SemanticIndexType()))
            self._require_i64_like_expr(dual_dst_mode_expr, f"pto.{name} dual_dst_mode")
            self._require_i64_like_expr(sub_blockid_expr, f"pto.{name} sub_blockid")
            tail_args.extend([dual_dst_mode_expr, sub_blockid_expr])
        tail_args.append(mode)
        if loop0_src_stride is not None:
            tail_args.append(loop0_src_stride)
        if split_expr is not None:
            tail_args.append(split_expr)
        if loop3_expr is not None:
            tail_args.append(loop3_expr)

        return SemanticCallExpr(namespace="pto", name=name, args=tuple(args) + tuple(tail_args), type=None)

    def _analyze_make_mask(self, args: tuple[SemanticExpr, ...]) -> SemanticExpr:
        if len(args) != 2:
            raise TypeError("pto.make_mask expects exactly 2 positional arguments in TileLang DSL v1")
        dtype_expr, value_expr = args
        dtype = self._require_dtype_symbol(dtype_expr, "pto.make_mask element type")
        if isinstance(value_expr, SemanticSymbolExpr) and value_expr.type.kind == "mask_pattern":
            return SemanticCallExpr(
                namespace="pto",
                name="make_mask",
                args=args,
                type=SemanticMaskType(granularity=self._mask_granularity_for_dtype(dtype)),
            )
        self._require_tail_remaining_expr(value_expr, "pto.make_mask tail remaining")
        return SemanticCallExpr(
            namespace="pto",
            name="make_mask",
            args=args,
            type=SemanticTupleType(
                elements=(
                    SemanticMaskType(granularity=self._mask_granularity_for_dtype(dtype)),
                    _I32_TYPE,
                )
            ),
        )

    def _analyze_predicate_pattern_op(
        self,
        name: str,
        args: tuple[SemanticExpr, ...],
    ) -> SemanticExpr:
        if len(args) != 1:
            raise TypeError(f"pto.{name} expects exactly 1 positional argument in TileLang DSL v1")
        pattern = args[0]
        if not (
            isinstance(pattern, SemanticSymbolExpr)
            and isinstance(pattern.type, SemanticMetaType)
            and pattern.type.kind == "mask_pattern"
            and isinstance(pattern.value, MaskPattern)
        ):
            raise TypeError(f"pto.{name} pattern must be a MaskPattern symbol such as `pto.PAT.ALL`")
        granularity = name.rsplit("_", 1)[-1]
        return SemanticCallExpr(
            namespace="pto",
            name=name,
            args=args,
            type=SemanticMaskType(granularity=granularity),
        )

    def _analyze_predicate_tail_op(
        self,
        name: str,
        args: tuple[SemanticExpr, ...],
    ) -> SemanticExpr:
        if len(args) != 1:
            raise TypeError(f"pto.{name} expects exactly 1 positional argument in TileLang DSL v1")
        self._require_tail_remaining_expr(args[0], f"pto.{name} scalar")
        granularity = name.rsplit("_", 1)[-1]
        mask_type = SemanticMaskType(granularity=granularity)
        return SemanticCallExpr(
            namespace="pto",
            name=name,
            args=args,
            type=SemanticTupleType(elements=(mask_type, _I32_TYPE)),
        )

    def _literal_expr_from_context_value(self, value: object, context: str) -> SemanticExpr:
        if isinstance(value, bool):
            return SemanticLiteralExpr(value=value, type=SemanticScalarType(dtype=i1))
        if isinstance(value, int) and not isinstance(value, bool):
            return SemanticLiteralExpr(value=value, type=SemanticIndexType())
        if isinstance(value, float):
            return SemanticLiteralExpr(value=value, type=SemanticScalarType(dtype=f32))
        if isinstance(value, str):
            return SemanticLiteralExpr(value=value, type=SemanticMetaType(kind="string"))
        if isinstance(value, ScalarType):
            return SemanticSymbolExpr(
                namespace="pto",
                name=value.name,
                value=value,
                type=SemanticMetaType(kind="dtype"),
            )
        if isinstance(value, MemorySpace):
            return SemanticSymbolExpr(
                namespace="pto",
                name=value.name,
                value=value,
                type=SemanticMetaType(kind="memory_space"),
            )
        if isinstance(value, CmpMode):
            return SemanticSymbolExpr(
                namespace="pto",
                name=value.name,
                value=value,
                type=SemanticMetaType(kind="cmp_mode"),
            )
        if isinstance(value, PredicatePart):
            return SemanticSymbolExpr(
                namespace="pto",
                name=value.name,
                value=value,
                type=SemanticMetaType(kind="predicate_part"),
            )
        raise TypeError(
            f"{context} resolved to unsupported static value {value!r} in TileLang DSL v1"
        )

    def _analyze_get_op_attr(self, args: tuple[SemanticExpr, ...]) -> SemanticExpr:
        if len(args) not in {1, 2}:
            raise TypeError(
                "pto.get_op_attr expects 1 or 2 positional arguments `(name, default?)` in TileLang DSL v1"
            )
        attr_name = self._require_string_expr(args[0], "pto.get_op_attr name")
        if attr_name in self._context_attrs:
            return self._literal_expr_from_context_value(
                self._context_attrs[attr_name],
                f"pto.get_op_attr({attr_name!r})",
            )
        if len(args) == 2:
            return args[1]
        raise TypeError(
            f"pto.get_op_attr could not resolve attribute {attr_name!r} and no default was provided"
        )

    def _analyze_scalar_constructor(
        self,
        name: str,
        args: tuple[SemanticExpr, ...],
    ) -> SemanticExpr:
        return self._analyze_scalar_constructor_for_dtype(
            _DTYPE_SYMBOLS[name],
            args,
            surface_name=f"pto.{name}",
        )

    def _analyze_scalar_constructor_for_dtype(
        self,
        target_dtype: ScalarType,
        args: tuple[SemanticExpr, ...],
        *,
        surface_name: str,
    ) -> SemanticExpr:
        self._require_non_low_precision_dtype(target_dtype, f"{surface_name} scalar constructor")
        if len(args) != 1:
            raise TypeError(f"{surface_name} expects exactly 1 positional argument in TileLang DSL v1")

        if (
            target_dtype.name in {"f16", "bf16", "f32"}
            and isinstance(args[0], SemanticLiteralExpr)
            and isinstance(args[0].type, SemanticMetaType)
            and args[0].type.kind == "string"
        ):
            parsed = self._parse_float_literal_string(args[0].value, target_dtype, f"{surface_name} value")
            return SemanticLiteralExpr(
                value=parsed,
                type=SemanticScalarType(dtype=target_dtype),
            )
        if (
            is_integer_dtype(target_dtype)
            and isinstance(args[0], SemanticLiteralExpr)
            and isinstance(args[0].type, SemanticMetaType)
            and args[0].type.kind == "string"
        ):
            parsed = self._parse_integer_literal_string(
                args[0].value,
                target_dtype,
                f"{surface_name} value",
            )
            return SemanticLiteralExpr(
                value=parsed,
                type=SemanticScalarType(dtype=target_dtype),
            )

        value = self._require_scalar_or_index_expr(args[0], f"{surface_name} value")

        if isinstance(value.type, SemanticScalarType) and value.type.dtype == target_dtype:
            return value

        if isinstance(value, SemanticLiteralExpr):
            literal_value = value.value
            if target_dtype == i1:
                if isinstance(literal_value, bool):
                    return SemanticLiteralExpr(value=literal_value, type=SemanticScalarType(dtype=i1))
                if isinstance(literal_value, int):
                    return SemanticLiteralExpr(value=bool(literal_value), type=SemanticScalarType(dtype=i1))
                if isinstance(literal_value, float):
                    return SemanticLiteralExpr(value=bool(literal_value), type=SemanticScalarType(dtype=i1))
            elif is_integer_dtype(target_dtype):
                if isinstance(literal_value, bool):
                    casted = int(literal_value)
                elif isinstance(literal_value, (int, float)):
                    casted = int(literal_value)
                else:
                    casted = None
                if casted is not None:
                    checked = self._check_integer_literal_range(
                        casted,
                        target_dtype,
                        f"{surface_name} value",
                    )
                    return SemanticLiteralExpr(value=checked, type=SemanticScalarType(dtype=target_dtype))
            else:
                if isinstance(literal_value, (bool, int, float)):
                    return SemanticLiteralExpr(
                        value=float(literal_value),
                        type=SemanticScalarType(dtype=target_dtype),
                    )

        return SemanticCallExpr(
            namespace="pto",
            name=target_dtype.name,
            args=(value,),
            type=SemanticScalarType(dtype=target_dtype),
        )

    def _parse_float_literal_string(
        self,
        literal: str,
        target_dtype: ScalarType,
        context: str,
    ) -> float:
        text = literal.strip().lower()
        if text in {"inf", "+inf", "infinity", "+infinity"}:
            return float("inf")
        if text in {"-inf", "-infinity"}:
            return float("-inf")
        if text in {"nan", "+nan", "-nan"}:
            return float("nan")

        if text.startswith("0x"):
            try:
                bit_pattern = int(text, 16)
            except ValueError as exc:
                raise TypeError(
                    f"{context} string literal {literal!r} is not a valid hex bit-pattern"
                ) from exc
            return self._float_from_bit_pattern(bit_pattern, target_dtype, context=context)

        try:
            return float(text)
        except ValueError as exc:
            raise TypeError(
                f"{context} string literal {literal!r} is not a valid float literal"
            ) from exc

    def _parse_integer_literal_string(
        self,
        literal: str,
        target_dtype: ScalarType,
        context: str,
    ) -> int:
        text = literal.strip().lower()
        bits = integer_bitwidth(target_dtype)
        signedness = integer_signedness(target_dtype)
        assert bits is not None
        signless_or_signed = signedness != "unsigned"
        if not text.startswith("0x"):
            raise TypeError(
                f"{context} string literals must use hex bit-pattern form like \"0xFF\" in TileLang DSL v1"
            )
        try:
            parsed = int(text, 16)
        except ValueError as exc:
            raise TypeError(
                f"{context} string literal {literal!r} is not a valid hex bit-pattern"
            ) from exc
        if parsed >= (1 << bits):
            raise TypeError(
                f"{context} bit-pattern literal {literal!r} exceeds {bits}-bit width for {target_dtype.name}"
            )
        if signless_or_signed and parsed >= (1 << (bits - 1)):
            parsed -= 1 << bits
        return self._check_integer_literal_range(parsed, target_dtype, context)

    def _check_integer_literal_range(
        self,
        value: int,
        target_dtype: ScalarType,
        context: str,
    ) -> int:
        bits = integer_bitwidth(target_dtype)
        signedness = integer_signedness(target_dtype)
        assert bits is not None
        if signedness == "unsigned":
            min_value = 0
            max_value = (1 << bits) - 1
        else:
            min_value = -(1 << (bits - 1))
            max_value = (1 << (bits - 1)) - 1
        if value < min_value or value > max_value:
            raise TypeError(
                f"{context} {value} is out of range for {target_dtype.name} in TileLang DSL v1"
            )
        return value

    def _float_from_bit_pattern(
        self,
        bit_pattern: int,
        target_dtype: ScalarType,
        *,
        context: str,
    ) -> float:
        if target_dtype.name == "f16":
            if bit_pattern < 0 or bit_pattern > 0xFFFF:
                raise TypeError(f"{context} f16 bit-pattern must be in [0x0, 0xFFFF]")
            return float(struct.unpack(">e", struct.pack(">H", bit_pattern))[0])
        if target_dtype.name == "bf16":
            if bit_pattern < 0 or bit_pattern > 0xFFFF:
                raise TypeError(f"{context} bf16 bit-pattern must be in [0x0, 0xFFFF]")
            widened = bit_pattern << 16
            return float(struct.unpack(">f", struct.pack(">I", widened))[0])
        if target_dtype.name == "f32":
            if bit_pattern < 0 or bit_pattern > 0xFFFFFFFF:
                raise TypeError(f"{context} f32 bit-pattern must be in [0x0, 0xFFFFFFFF]")
            return float(struct.unpack(">f", struct.pack(">I", bit_pattern))[0])
        raise TypeError(f"{context} bit-pattern literals are not supported for dtype {target_dtype.name}")

    def _analyze_ptr_type(self, args: tuple[SemanticExpr, ...]) -> SemanticExpr:
        if len(args) != 2:
            raise TypeError("pto.ptr expects exactly 2 positional arguments in TileLang DSL")
        dtype = self._require_dtype_symbol(args[0], "pto.ptr element type")
        memory_space = self._require_memory_space_symbol(args[1], "pto.ptr memory space")
        return SemanticLiteralExpr(
            value=PointerType(element_dtype=dtype, memory_space=memory_space),
            type=SemanticMetaType(kind="ptr_type"),
        )

    def _analyze_vreg_type(self, args: tuple[SemanticExpr, ...]) -> SemanticExpr:
        if len(args) != 1:
            raise TypeError("pto.vreg expects exactly 1 positional argument in TileLang DSL v1")
        dtype = self._require_dtype_symbol(args[0], "pto.vreg element type")
        self._require_non_storage_only_dtype(dtype, "pto.vreg element type")
        vreg_type = self._vreg_type_for_dtype(dtype)
        return SemanticLiteralExpr(
            value=VRegType(element_dtype=dtype, lanes=vreg_type.lanes),
            type=SemanticMetaType(kind="vreg_type"),
        )

    def _analyze_vector_type(self, args: tuple[SemanticExpr, ...]) -> SemanticExpr:
        if len(args) != 2:
            raise TypeError("pto.vector expects exactly 2 positional arguments in TileLang DSL v1")
        dtype = self._require_dtype_symbol(args[0], "pto.vector element type")
        self._require_non_storage_only_dtype(dtype, "pto.vector element type")
        shape = self._require_vector_shape_expr(args[1], "pto.vector shape")
        return SemanticLiteralExpr(
            value=VectorType(element_dtype=dtype, shape=shape),
            type=SemanticMetaType(kind="vector_type"),
        )

    def _analyze_castptr(self, args: tuple[SemanticExpr, ...]) -> SemanticExpr:
        if len(args) != 2:
            raise TypeError("pto.castptr expects exactly 2 positional arguments in TileLang DSL")
        value, target = args
        target_type = self._require_cast_target_type(target)
        if isinstance(target_type, SemanticPtrType):
            self._require_castptr_input(value, target_type)
        else:
            self._require_pointer_expr(value, "pto.castptr input")
        return SemanticCallExpr(namespace="pto", name="castptr", args=args, type=target_type)

    def _analyze_addptr(self, args: tuple[SemanticExpr, ...]) -> SemanticExpr:
        if len(args) != 2:
            raise TypeError("pto.addptr expects exactly 2 positional arguments in TileLang DSL")
        pointer, offset = args
        ptr = self._require_pointer_expr(pointer, "pto.addptr pointer")
        offset = self._require_index_typed_expr(offset)
        return SemanticCallExpr(namespace="pto", name="addptr", args=(ptr, offset), type=ptr.type)

    def _analyze_tile_frontend_call(
        self,
        expr: FrontendCallExpr,
        env: dict[str, SemanticBinding],
        *,
        allow_outer_lookup: bool,
    ) -> SemanticExpr:
        analyzed_keywords = {
            name: self._analyze_expr(value, env, allow_outer_lookup=allow_outer_lookup)
            for name, value in expr.keywords
        }
        unexpected_keywords = sorted(set(analyzed_keywords) - _TILE_CONSTRUCTOR_ALLOWED_KEYWORDS)
        if unexpected_keywords:
            keyword_text = ", ".join(unexpected_keywords)
            raise TypeError(
                "pto.Tile only accepts keyword args "
                "`valid_shape`, `blayout`, `slayout`, `fractal_size`, `pad_value`, `compact_mode`, and `addr`; "
                f"got unsupported keyword(s): {keyword_text}"
            )
        if len(expr.args) != 3:
            raise TypeError(
                "pto.Tile expects exactly 3 positional arguments `(shape, dtype, memory_space)` in TileLang DSL v1"
            )

        shape_expr = self._analyze_expr(expr.args[0], env, allow_outer_lookup=allow_outer_lookup)
        dtype_expr = self._analyze_expr(expr.args[1], env, allow_outer_lookup=allow_outer_lookup)
        memory_space_expr = self._analyze_expr(expr.args[2], env, allow_outer_lookup=allow_outer_lookup)
        return self._analyze_tile_constructor(
            shape_expr,
            dtype_expr,
            memory_space_expr,
            valid_shape_expr=analyzed_keywords.get("valid_shape"),
            b_layout_expr=analyzed_keywords.get("blayout"),
            s_layout_expr=analyzed_keywords.get("slayout"),
            fractal_size_expr=analyzed_keywords.get("fractal_size"),
            pad_value_expr=analyzed_keywords.get("pad_value"),
            compact_mode_expr=analyzed_keywords.get("compact_mode"),
            addr_expr=analyzed_keywords.get("addr"),
        )

    def _analyze_tile_constructor(
        self,
        shape_expr: SemanticExpr,
        dtype_expr: SemanticExpr,
        memory_space_expr: SemanticExpr,
        *,
        valid_shape_expr: SemanticExpr | None,
        b_layout_expr: SemanticExpr | None,
        s_layout_expr: SemanticExpr | None,
        fractal_size_expr: SemanticExpr | None,
        pad_value_expr: SemanticExpr | None,
        compact_mode_expr: SemanticExpr | None,
        addr_expr: SemanticExpr | None,
    ) -> SemanticExpr:
        if addr_expr is not None:
            self._require_i64_like_expr(addr_expr, "pto.Tile addr")

        shape = self._require_static_shape_tuple(shape_expr, "pto.Tile shape")
        if not shape:
            raise TypeError("pto.Tile shape must be non-empty in TileLang DSL v1")
        if len(shape) not in {1, 2}:
            raise TypeError("pto.Tile only supports rank-1 or rank-2 shapes in TileLang DSL v1")
        dtype = self._require_dtype_symbol(dtype_expr, "pto.Tile dtype")
        memory_space = self._require_memory_space_symbol(memory_space_expr, "pto.Tile memory_space")
        valid_shape = self._normalize_tile_valid_shape_expr(valid_shape_expr, shape, "pto.Tile valid_shape")
        config = self._build_tile_constructor_config(
            memory_space,
            b_layout_expr=b_layout_expr,
            s_layout_expr=s_layout_expr,
            fractal_size_expr=fractal_size_expr,
            pad_value_expr=pad_value_expr,
            compact_mode_expr=compact_mode_expr,
        )
        lowered_args: list[SemanticExpr] = []
        if valid_shape_expr is not None or addr_expr is not None:
            lowered_args.append(
                valid_shape_expr
                if valid_shape_expr is not None
                else SemanticLiteralExpr(value=None, type=SemanticMetaType(kind="none"))
            )
        if addr_expr is not None:
            lowered_args.append(addr_expr)
        return SemanticCallExpr(
            namespace="pto",
            name="alloc_tile",
            args=tuple(lowered_args),
            type=SemanticTileType(
                element_dtype=dtype,
                rank=len(shape),
                shape=shape,
                valid_shape=valid_shape,
                memory_space=memory_space.value,
                config=config,
            ),
        )

    def _require_static_shape_tuple(
        self,
        expr: SemanticExpr,
        context: str,
    ) -> tuple[int, ...]:
        value = self._try_static_value(expr)
        if not isinstance(value, tuple):
            raise TypeError(f"{context} must be a statically known tuple/list of integers in TileLang DSL v1")
        dims: list[int] = []
        for index, dim in enumerate(value):
            if isinstance(dim, bool) or not isinstance(dim, int):
                raise TypeError(f"{context}[{index}] must be a positive integer in TileLang DSL v1")
            if dim <= 0:
                raise TypeError(f"{context}[{index}] must be a positive integer in TileLang DSL v1")
            dims.append(dim)
        return tuple(dims)

    def _normalize_tile_valid_shape_expr(
        self,
        expr: SemanticExpr | None,
        shape: tuple[int, ...],
        context: str,
    ) -> tuple[int | None, ...]:
        if expr is None:
            return shape
        value = self._try_static_value(expr)
        if not isinstance(value, tuple):
            raise TypeError(f"{context} must be a statically known tuple/list in TileLang DSL v1")
        if len(value) != len(shape):
            raise TypeError(f"{context} rank must match tile shape rank in TileLang DSL v1")
        dims: list[int | None] = []
        for index, (dim, bound) in enumerate(zip(value, shape)):
            if dim is None:
                dims.append(None)
                continue
            if isinstance(dim, bool) or not isinstance(dim, int):
                raise TypeError(f"{context}[{index}] must be an integer or None in TileLang DSL v1")
            if dim <= 0:
                raise TypeError(f"{context}[{index}] must be positive when provided in TileLang DSL v1")
            if dim > bound:
                raise TypeError(f"{context}[{index}] must be <= shape[{index}] in TileLang DSL v1")
            dims.append(dim)
        return tuple(dims)

    def _build_tile_constructor_config(
        self,
        memory_space: MemorySpace,
        *,
        b_layout_expr: SemanticExpr | None,
        s_layout_expr: SemanticExpr | None,
        fractal_size_expr: SemanticExpr | None,
        pad_value_expr: SemanticExpr | None,
        compact_mode_expr: SemanticExpr | None,
    ) -> TileConfig:
        defaults = dict(TileConfig.for_memory_space(memory_space).fields)
        if b_layout_expr is not None:
            defaults["b_layout"] = self._require_b_layout_symbol(b_layout_expr, "pto.Tile blayout")
        if s_layout_expr is not None:
            defaults["s_layout"] = self._require_s_layout_symbol(s_layout_expr, "pto.Tile slayout")
        if fractal_size_expr is not None:
            fractal = self._try_static_value(fractal_size_expr)
            if isinstance(fractal, bool) or not isinstance(fractal, int):
                raise TypeError("pto.Tile fractal_size must be a static integer in TileLang DSL v1")
            defaults["s_fractal_size"] = fractal
        if pad_value_expr is not None:
            pad_value = self._try_static_value(pad_value_expr)
            if not isinstance(pad_value, PadValue):
                raise TypeError("pto.Tile pad_value must be a PadValue symbol in TileLang DSL v1")
            defaults["pad_value"] = pad_value
        if compact_mode_expr is not None:
            defaults["compact_mode"] = self._require_compact_mode_symbol(
                compact_mode_expr, "pto.Tile compact_mode"
            )
        return TileConfig(tuple(sorted(defaults.items())))

    def _require_b_layout_symbol(self, expr: SemanticExpr, context: str) -> BLayout:
        if (
            isinstance(expr, SemanticSymbolExpr)
            and isinstance(expr.type, SemanticMetaType)
            and expr.type.kind == "b_layout"
            and isinstance(expr.value, BLayout)
        ):
            return expr.value
        if (
            isinstance(expr, SemanticBindingRef)
            and isinstance(expr.type, SemanticMetaType)
            and expr.type.kind == "b_layout"
            and isinstance(expr.binding.value, BLayout)
        ):
            return expr.binding.value
        raise TypeError(f"{context} must be a BLayout symbol in TileLang DSL v1")

    def _require_s_layout_symbol(self, expr: SemanticExpr, context: str) -> SLayout:
        if (
            isinstance(expr, SemanticSymbolExpr)
            and isinstance(expr.type, SemanticMetaType)
            and expr.type.kind == "s_layout"
            and isinstance(expr.value, SLayout)
        ):
            return expr.value
        if (
            isinstance(expr, SemanticBindingRef)
            and isinstance(expr.type, SemanticMetaType)
            and expr.type.kind == "s_layout"
            and isinstance(expr.binding.value, SLayout)
        ):
            return expr.binding.value
        raise TypeError(f"{context} must be an SLayout symbol in TileLang DSL v1")

    def _require_compact_mode_symbol(self, expr: SemanticExpr, context: str) -> CompactMode:
        if (
            isinstance(expr, SemanticSymbolExpr)
            and isinstance(expr.type, SemanticMetaType)
            and expr.type.kind == "compact_mode"
            and isinstance(expr.value, CompactMode)
        ):
            return expr.value
        if (
            isinstance(expr, SemanticBindingRef)
            and isinstance(expr.type, SemanticMetaType)
            and expr.type.kind == "compact_mode"
            and isinstance(expr.binding.value, CompactMode)
        ):
            return expr.binding.value
        raise TypeError(f"{context} must be a CompactMode symbol in TileLang DSL v1")

    def _analyze_get_lanes(
        self,
        args: tuple[SemanticExpr, ...],
        *,
        call_name: str = "get_lanes",
    ) -> SemanticExpr:
        if len(args) != 1:
            raise TypeError(
                f"pto.{call_name} expects exactly 1 positional argument in TileLang DSL v1"
            )
        dtype = self._require_dtype_symbol(args[0], f"pto.{call_name} dtype")
        return SemanticLiteralExpr(value=self._vreg_type_for_dtype(dtype).lanes, type=SemanticIndexType())

    def _analyze_bytewidth(self, args: tuple[SemanticExpr, ...]) -> SemanticExpr:
        if len(args) != 1:
            raise TypeError("pto.bytewidth expects exactly 1 positional argument in TileLang DSL v1")
        dtype = self._require_dtype_symbol(args[0], "pto.bytewidth dtype")
        return SemanticLiteralExpr(value=bytewidth(dtype), type=SemanticIndexType())

    def _analyze_init_align(self, args: tuple[SemanticExpr, ...]) -> SemanticExpr:
        if args:
            raise TypeError("pto.init_align does not accept positional arguments in TileLang DSL v1")
        return SemanticCallExpr(namespace="pto", name="init_align", args=(), type=SemanticAlignType())

    def _analyze_vlds(
        self,
        args: tuple[SemanticExpr, ...],
        *,
        dist: SemanticExpr | None = None,
    ) -> SemanticExpr:
        if len(args) < 2:
            raise TypeError("pto.vlds expects at least 2 positional arguments in TileLang DSL v1")
        source, *indices = args
        source_type = source.type
        if isinstance(source_type, SemanticTileType):
            source = self._require_tile_expr(source, "pto.vlds source")
        else:
            source = self._require_pointer_expr(source, "pto.vlds source", memory_space="ub")
        indices = tuple(self._require_index_typed_expr(index) for index in indices)
        lowered_args: tuple[SemanticExpr, ...]
        if dist is not None:
            lowered_args = (source, *indices, dist)
        else:
            lowered_args = (source, *indices)
        return SemanticCallExpr(
            namespace="pto",
            name="vlds",
            args=lowered_args,
            type=self._vreg_type_for_dtype(source.type.element_dtype),
        )

    def _analyze_vlds_frontend_call(
        self,
        expr: FrontendCallExpr,
        env: dict[str, SemanticBinding],
        *,
        allow_outer_lookup: bool,
    ) -> SemanticExpr:
        analyzed_keywords = {
            name: self._analyze_expr(value, env, allow_outer_lookup=allow_outer_lookup)
            for name, value in expr.keywords
        }
        unexpected_keywords = sorted(set(analyzed_keywords) - {"dist"})
        if unexpected_keywords:
            keyword_text = ", ".join(unexpected_keywords)
            raise TypeError(
                "pto.vlds only accepts keyword attr `dist`; "
                f"got unsupported keyword(s): {keyword_text}"
            )
        dist = self._normalize_vlds_dist(analyzed_keywords.get("dist"), "pto.vlds dist")
        if len(expr.args) == 1 and isinstance(expr.args[0], FrontendSubscriptExpr):
            base, indices = self._analyze_tile_vector_access(
                expr.args[0],
                env,
                allow_outer_lookup=allow_outer_lookup,
                context="pto.vlds source",
            )
            return self._analyze_vlds((base, *indices), dist=dist)

        args = tuple(
            self._analyze_expr(arg, env, allow_outer_lookup=allow_outer_lookup)
            for arg in expr.args
        )
        return self._analyze_vlds(args, dist=dist)

    def _analyze_vldas(self, args: tuple[SemanticExpr, ...]) -> SemanticExpr:
        if len(args) not in {1, 2, 3}:
            raise TypeError("pto.vldas expects 1 positional source or Tile[start:]/Tile[row, col:] in TileLang DSL v1")
        source, *indices = args
        source_type = source.type
        if isinstance(source_type, SemanticTileType):
            source = self._require_tile_expr(source, "pto.vldas source")
            indices = tuple(self._require_index_typed_expr(index) for index in indices)
        else:
            if indices:
                raise TypeError("pto.vldas pointer syntax does not accept explicit indices in TileLang DSL v1")
            source = self._require_pointer_expr(source, "pto.vldas source", memory_space="ub")
        return SemanticCallExpr(
            namespace="pto",
            name="vldas",
            args=(source, *indices),
            type=SemanticAlignType(),
        )

    def _analyze_vldus(self, args: tuple[SemanticExpr, ...]) -> SemanticExpr:
        if len(args) not in {2, 3, 4}:
            raise TypeError("pto.vldus expects (source, align) or Tile element-indexing syntax in TileLang DSL v1")
        source, *rest = args
        align_expr = rest[-1]
        index_args = rest[:-1]
        source_type = source.type
        if isinstance(source_type, SemanticTileType):
            source = self._require_tile_expr(source, "pto.vldus source")
            index_args = tuple(self._require_index_typed_expr(index) for index in index_args)
        else:
            if index_args:
                raise TypeError("pto.vldus pointer syntax does not accept explicit indices in TileLang DSL v1")
            source = self._require_pointer_expr(source, "pto.vldus source", memory_space="ub")
        self._require_align_expr(align_expr, "pto.vldus align")
        return SemanticCallExpr(
            namespace="pto",
            name="vldus",
            args=(source, *index_args, align_expr),
            type=SemanticTupleType(elements=(self._vreg_type_for_dtype(source.type.element_dtype), SemanticAlignType())),
        )

    def _analyze_vldsx2(self, args: tuple[SemanticExpr, ...]) -> SemanticExpr:
        if len(args) not in {3, 4}:
            raise TypeError("pto.vldsx2 expects 3 or 4 positional arguments in TileLang DSL v1")
        source, *rest = args
        if len(rest) == 2:
            index_args = rest[:1]
            dist = rest[1]
        else:
            index_args = rest[:2]
            dist = rest[2]
        source_type = source.type
        if isinstance(source_type, SemanticTileType):
            source = self._require_tile_expr(source, "pto.vldsx2 source")
        else:
            source = self._require_pointer_expr(source, "pto.vldsx2 source", memory_space="ub")
        index_args = tuple(self._require_index_typed_expr(index) for index in index_args)
        dist = self._normalize_vldsx2_dist(dist)
        vreg_type = self._vreg_type_for_dtype(source.type.element_dtype)
        return SemanticCallExpr(
            namespace="pto",
            name="vldsx2",
            args=(source, *index_args, dist),
            type=SemanticTupleType(elements=(vreg_type, vreg_type)),
        )

    def _analyze_predicate_load_op(
        self,
        name: str,
        args: tuple[SemanticExpr, ...],
    ) -> SemanticExpr:
        expects_i32_immediate = name == "pldi"
        canonical_name = "plds" if name == "pld" else name
        if len(args) not in {2, 3}:
            raise TypeError(
                f"pto.{name} expects 2 or 3 positional arguments in TileLang DSL v1: "
                f"`pto.{name}(buf, offset[, dist])`"
            )

        source, offset = args[:2]
        source = self._require_pointer_expr(source, f"pto.{name} source", memory_space="ub")
        if expects_i32_immediate:
            self._require_i32_like_expr(offset, "pto.pldi offset")
        else:
            offset = self._require_index_typed_expr(offset)
        dist = self._normalize_predicate_load_dist(
            args[2] if len(args) == 3 else None,
            f"pto.{name} dist",
        )

        if source.type.element_dtype == ui8:
            granularity = "b8"
        elif source.type.element_dtype == ui16:
            granularity = "b16"
        elif source.type.element_dtype == ui32:
            granularity = "b32"
        else:
            raise TypeError(
                f"pto.{name} source must be !pto.ptr<ui8/ui16/ui32, ub> in TileLang DSL v1"
            )

        return SemanticCallExpr(
            namespace="pto",
            name=canonical_name,
            args=(source, offset, dist),
            type=SemanticMaskType(granularity=granularity),
        )

    def _analyze_pstu(self, args: tuple[SemanticExpr, ...]) -> SemanticExpr:
        if len(args) != 3:
            raise TypeError("pto.pstu expects exactly 3 positional arguments in TileLang DSL v1")
        align_expr, value, base = args
        self._require_align_expr(align_expr, "pto.pstu align_in")
        mask_type = self._require_mask_expr(value, "pto.pstu value")
        base = self._require_pointer_expr(base, "pto.pstu base", memory_space="ub")
        if mask_type.granularity == "b16":
            expected = ui16
        elif mask_type.granularity == "b32":
            expected = ui32
        else:
            raise TypeError("pto.pstu only supports !pto.mask<b16> and !pto.mask<b32> in TileLang DSL v1")
        if base.type.element_dtype != expected:
            raise TypeError(
                f"pto.pstu requires !pto.ptr<{expected.name}, ub> for mask granularity {mask_type.granularity}"
            )
        return SemanticCallExpr(
            namespace="pto",
            name="pstu",
            args=(align_expr, value, base),
            type=SemanticTupleType(elements=(SemanticAlignType(), base.type)),
        )

    def _analyze_vstus(self, args: tuple[SemanticExpr, ...]) -> SemanticExpr:
        if len(args) != 4:
            raise TypeError("pto.vstus expects exactly 4 positional arguments in TileLang DSL v1")
        align_expr, offset, value, base = args
        self._require_align_expr(align_expr, "pto.vstus align_in")
        self._require_i32_like_expr(offset, "pto.vstus offset")
        self._require_vreg_expr(value, "pto.vstus value")
        base = self._require_pointer_expr(base, "pto.vstus base", memory_space="ub")
        return SemanticCallExpr(
            namespace="pto",
            name="vstus",
            args=(align_expr, offset, value, base),
            type=SemanticAlignType(),
        )

    def _analyze_vstur(self, args: tuple[SemanticExpr, ...]) -> SemanticExpr:
        if len(args) not in {3, 4}:
            raise TypeError("pto.vstur expects 3 or 4 positional arguments in TileLang DSL v1")
        align_expr, value, base = args[:3]
        mode = self._normalize_post_update_mode(args[3] if len(args) == 4 else None, "pto.vstur mode")
        self._require_align_expr(align_expr, "pto.vstur align_in")
        self._require_vreg_expr(value, "pto.vstur value")
        base = self._require_pointer_expr(base, "pto.vstur base", memory_space="ub")
        return SemanticCallExpr(
            namespace="pto",
            name="vstur",
            args=(align_expr, value, base, mode),
            type=SemanticAlignType(),
        )

    def _analyze_load_scalar(self, args: tuple[SemanticExpr, ...]) -> SemanticExpr:
        if len(args) == 2:
            destination_dtype = None
            pointer, offset = args
        elif len(args) == 3:
            destination_dtype = self._require_dtype_symbol(args[0], "pto.load_scalar result type")
            pointer, offset = args[1:]
        else:
            raise TypeError("pto.load_scalar expects 2 or 3 positional arguments in TileLang DSL v1")
        pointer = self._require_pointer_expr(pointer, "pto.load_scalar source")
        offset = self._require_index_typed_expr(offset)
        if destination_dtype is not None and destination_dtype != pointer.type.element_dtype:
            raise TypeError("pto.load_scalar result type must match source pointer element dtype")
        return SemanticCallExpr(
            namespace="pto",
            name="load_scalar",
            args=(pointer, offset),
            type=SemanticScalarType(dtype=pointer.type.element_dtype),
        )

    def _analyze_runtime_block_query(
        self,
        name: str,
        args: tuple[SemanticExpr, ...],
    ) -> SemanticExpr:
        if args:
            raise TypeError(f"pto.{name} does not accept positional arguments in TileLang DSL v1")
        return SemanticCallExpr(
            namespace="pto",
            name=name,
            args=(),
            type=SemanticScalarType(dtype=i64),
        )

    def _analyze_broadcast_vector_op(
        self,
        name: str,
        args: tuple[SemanticExpr, ...],
    ) -> SemanticExpr:
        if name == "vbr":
            if len(args) != 1:
                raise TypeError("pto.vbr expects exactly 1 positional argument in TileLang DSL v1")
            value = args[0]
            vec_type = self._vreg_type_for_scalar_or_index(value, "pto.vbr value")
            return SemanticCallExpr(namespace="pto", name=name, args=args, type=vec_type)

        if name == "vdup":
            if len(args) not in {2, 3}:
                raise TypeError("pto.vdup expects 2 or 3 positional arguments in TileLang DSL v1")
            value = args[0]
            if isinstance(value.type, SemanticVRegType):
                vec_type = value.type
                mask = args[1]
                self._require_mask_for_vreg(mask, vec_type, "pto.vdup")
                position_arg = args[2] if len(args) == 3 else None
                position = self._normalize_position_mode(position_arg, "pto.vdup position")
                return SemanticCallExpr(
                    namespace="pto",
                    name=name,
                    args=(value, mask, position),
                    type=vec_type,
                )

            if len(args) == 3:
                raise TypeError(
                    "pto.vdup scalar input does not accept `position`; use `pto.vdup(input, mask)` "
                    "in TileLang DSL v1"
                )
            vec_type = self._vreg_type_for_scalar_or_index(value, "pto.vdup input")
            mask = args[1]
            self._require_mask_for_vreg(mask, vec_type, "pto.vdup")
            return SemanticCallExpr(
                namespace="pto",
                name=name,
                args=(value, mask),
                type=vec_type,
            )

        if name == "vci":
            if len(args) not in {1, 2}:
                raise TypeError("pto.vci expects 1 or 2 positional arguments in TileLang DSL v1")
            index = self._require_scalar_or_index_expr(args[0], "pto.vci index")
            index_dtype = i32 if isinstance(index.type, SemanticIndexType) else index.type.dtype
            if not (is_integer_dtype(index_dtype) and integer_bitwidth(index_dtype) in {8, 16, 32}):
                raise TypeError("pto.vci index only supports 8/16/32-bit integer dtypes in TileLang DSL v1")
            order_arg = args[1] if len(args) == 2 else None
            order = self._normalize_order_mode(order_arg, "pto.vci order")
            return SemanticCallExpr(
                namespace="pto",
                name=name,
                args=(index, order),
                type=self._vreg_type_for_dtype(index_dtype),
            )

        raise TypeError(f"call surface `pto.{name}` is not supported in TileLang DSL v1 yet")

    def _analyze_unary_vector_op(
        self,
        name: str,
        args: tuple[SemanticExpr, ...],
    ) -> SemanticExpr:
        if name in {"vsunpack", "vzunpack"}:
            if len(args) != 2:
                raise TypeError(f"pto.{name} expects exactly 2 positional arguments in TileLang DSL v1")
            value, part = args
            vreg = self._require_vreg_expr(value, f"pto.{name} value")
            self._require_i32_like_expr(part, f"pto.{name} part")
            self._validate_unary_dtype(name, vreg.element_dtype)
            result_dtype = self._unpack_result_dtype(name, vreg.element_dtype)
            return SemanticCallExpr(
                namespace="pto",
                name=name,
                args=args,
                type=SemanticVRegType(element_dtype=result_dtype, lanes=vreg.lanes // 2),
            )
        if len(args) != 2:
            raise TypeError(f"pto.{name} expects exactly 2 positional arguments in TileLang DSL v1")
        value, mask = args
        vreg = self._require_vreg_expr(value, f"pto.{name} value")
        self._require_mask_for_vreg(mask, vreg, f"pto.{name}")
        self._validate_unary_dtype(name, vreg.element_dtype)
        result_type = vreg
        if name == "vcadd":
            result_type = self._vcadd_result_vreg_type(vreg)
        return SemanticCallExpr(namespace="pto", name=name, args=args, type=result_type)

    def _analyze_vexpdif_op(
        self,
        args: tuple[SemanticExpr, ...],
    ) -> SemanticExpr:
        if len(args) != 4:
            raise TypeError("pto.vexpdif expects exactly 4 positional arguments in TileLang DSL v1")
        input_expr, max_expr, mask_expr, part_expr = args
        input_type = self._require_vreg_expr(input_expr, "pto.vexpdif input")
        max_type = self._require_vreg_expr(max_expr, "pto.vexpdif max")
        if input_type != max_type:
            raise TypeError("pto.vexpdif requires input/max vector types to match")
        self._validate_vexpdif_dtype(input_type.element_dtype)
        self._require_mask_for_vreg(mask_expr, input_type, "pto.vexpdif")
        part = self._normalize_vexpdif_part(part_expr, "pto.vexpdif part")
        return SemanticCallExpr(
            namespace="pto",
            name="vexpdif",
            args=(input_expr, max_expr, mask_expr, part),
            type=self._vexpdif_result_vreg_type(input_type),
        )

    def _analyze_binary_vector_op(
        self,
        name: str,
        args: tuple[SemanticExpr, ...],
    ) -> SemanticExpr:
        if len(args) != 3:
            raise TypeError(f"pto.{name} expects exactly 3 positional arguments in TileLang DSL v1")
        lhs_expr, rhs_expr, mask = args
        lhs = self._require_vreg_expr(lhs_expr, f"pto.{name} lhs")
        rhs = self._require_vreg_expr(rhs_expr, f"pto.{name} rhs")
        if name == "vperm":
            if not (is_integer_dtype(rhs.element_dtype) and integer_bitwidth(rhs.element_dtype) in {8, 16, 32}):
                raise TypeError("pto.vperm indices vector only supports integer vector dtypes in TileLang DSL v1")
            if lhs.lanes != rhs.lanes:
                raise TypeError("pto.vperm requires data/indices vectors to use the same lane width")
        elif lhs != rhs:
            raise TypeError(f"pto.{name} requires lhs/rhs vector types to match")
        self._require_mask_for_vreg(mask, lhs, f"pto.{name}")
        self._validate_binary_dtype(name, lhs.element_dtype)
        if (
            name in {"vdiv", "vmod"}
            and is_integer_dtype(lhs.element_dtype)
            and integer_bitwidth(lhs.element_dtype) in {8, 16, 32}
        ):
            return self._analyze_internal_inline_proc_call_expr(
                "_tl_soft_vdiv" if name == "vdiv" else "_tl_soft_vmod",
                (
                    lhs_expr,
                    rhs_expr,
                    mask,
                    self._dtype_symbol_expr(lhs.element_dtype),
                ),
            )
        return SemanticCallExpr(namespace="pto", name=name, args=args, type=lhs)

    def _analyze_vector_scalar_op(
        self,
        name: str,
        args: tuple[SemanticExpr, ...],
    ) -> SemanticExpr:
        if len(args) != 3:
            raise TypeError(f"pto.{name} expects exactly 3 positional arguments in TileLang DSL v1")
        vector_expr, scalar_expr, mask = args
        vreg = self._require_vreg_expr(vector_expr, f"pto.{name} vector")
        scalar = self._require_scalar_expr(scalar_expr, f"pto.{name} scalar")
        if name in {"vshls", "vshrs"}:
            if scalar.dtype != i16:
                raise TypeError(f"pto.{name} scalar dtype must be i16")
        elif scalar.dtype != vreg.element_dtype:
            raise TypeError(f"pto.{name} scalar dtype must match vector element dtype")
        self._require_mask_for_vreg(mask, vreg, f"pto.{name}")
        self._validate_vector_scalar_dtype(name, vreg.element_dtype)
        return SemanticCallExpr(namespace="pto", name=name, args=args, type=vreg)

    def _analyze_vector_immediate_op(
        self,
        name: str,
        args: tuple[SemanticExpr, ...],
    ) -> SemanticExpr:
        if len(args) != 3:
            raise TypeError(f"pto.{name} expects exactly 3 positional arguments in TileLang DSL v1")
        vector = self._require_vreg_expr(args[0], f"pto.{name} vector")
        immediate = self._require_scalar_or_index_expr(args[1], f"pto.{name} immediate")
        if isinstance(immediate.type, SemanticScalarType) and not (
            is_integer_dtype(immediate.type.dtype) and integer_bitwidth(immediate.type.dtype) in {8, 16, 32}
        ):
            raise TypeError(f"pto.{name} immediate only supports 8/16/32-bit integer dtypes in TileLang DSL v1")
        self._require_mask_for_vreg(args[2], vector, f"pto.{name}")
        self._validate_vector_immediate_dtype(name, vector.element_dtype)
        return SemanticCallExpr(namespace="pto", name=name, args=args, type=vector)

    def _analyze_ternary_vector_op(
        self,
        name: str,
        args: tuple[SemanticExpr, ...],
    ) -> SemanticExpr:
        if len(args) != 4:
            raise TypeError(f"pto.{name} expects exactly 4 positional arguments in TileLang DSL v1")
        vec0 = self._require_vreg_expr(args[0], f"pto.{name} vec0")
        vec1 = self._require_vreg_expr(args[1], f"pto.{name} vec1")
        vec2 = self._require_vreg_expr(args[2], f"pto.{name} vec2")
        if not (vec0 == vec1 == vec2):
            raise TypeError(f"pto.{name} requires all vector operands to use the same vector type")
        self._require_mask_for_vreg(args[3], vec0, f"pto.{name}")
        self._validate_ternary_vector_dtype(name, vec0.element_dtype)
        return SemanticCallExpr(namespace="pto", name=name, args=args, type=vec0)

    def _analyze_multi_result_vector_op(
        self,
        name: str,
        args: tuple[SemanticExpr, ...],
    ) -> SemanticExpr:
        if name != "vmull":
            raise TypeError(f"call surface `pto.{name}` is not supported in TileLang DSL v1 yet")
        if len(args) != 3:
            raise TypeError("pto.vmull expects exactly 3 positional arguments in TileLang DSL")
        lhs = self._require_vreg_expr(args[0], "pto.vmull lhs")
        rhs = self._require_vreg_expr(args[1], "pto.vmull rhs")
        if lhs != rhs:
            raise TypeError("pto.vmull requires lhs/rhs vector types to match")
        self._require_mask_for_vreg(args[2], lhs, "pto.vmull")
        self._validate_multi_result_vector_dtype(name, lhs.element_dtype)
        return SemanticCallExpr(
            namespace="pto",
            name=name,
            args=args,
            type=SemanticTupleType(elements=(lhs, lhs)),
        )

    def _analyze_mask_part_op(
        self,
        name: str,
        args: tuple[SemanticExpr, ...],
    ) -> SemanticExpr:
        if len(args) != 2:
            raise TypeError(f"pto.{name} expects exactly 2 positional arguments in TileLang DSL")
        mask = self._require_mask_expr(args[0], f"pto.{name} mask")
        part = self._normalize_predicate_part(args[1], f"pto.{name} part")
        result_granularity = mask.granularity
        if name == "punpack":
            if mask.granularity == "b8":
                result_granularity = "b16"
            elif mask.granularity == "b16":
                result_granularity = "b32"
        return SemanticCallExpr(
            namespace="pto",
            name=name,
            args=(args[0], part),
            type=SemanticMaskType(granularity=result_granularity),
        )

    def _analyze_mask_logic_op(
        self,
        name: str,
        args: tuple[SemanticExpr, ...],
    ) -> SemanticExpr:
        if name == "pnot":
            if len(args) != 2:
                raise TypeError("pto.pnot expects exactly 2 positional arguments in TileLang DSL")
            value = self._require_mask_expr(args[0], "pto.pnot input")
            mask = self._require_mask_expr(args[1], "pto.pnot mask")
            self._require_matching_mask_types(value, mask, "pto.pnot")
            return SemanticCallExpr(namespace="pto", name=name, args=args, type=value)
        if name in {"pand", "por", "pxor"}:
            if len(args) != 3:
                raise TypeError(f"pto.{name} expects exactly 3 positional arguments in TileLang DSL")
            src0 = self._require_mask_expr(args[0], f"pto.{name} src0")
            src1 = self._require_mask_expr(args[1], f"pto.{name} src1")
            mask = self._require_mask_expr(args[2], f"pto.{name} mask")
            self._require_matching_mask_types(src0, src1, f"pto.{name}")
            self._require_matching_mask_types(src0, mask, f"pto.{name}")
            return SemanticCallExpr(namespace="pto", name=name, args=args, type=src0)
        if len(args) != 3:
            raise TypeError("pto.psel expects exactly 3 positional arguments in TileLang DSL")
        src0 = self._require_mask_expr(args[0], "pto.psel src0")
        src1 = self._require_mask_expr(args[1], "pto.psel src1")
        mask = self._require_mask_expr(args[2], "pto.psel mask")
        self._require_matching_mask_types(src0, src1, "pto.psel")
        self._require_matching_mask_types(src0, mask, "pto.psel")
        return SemanticCallExpr(namespace="pto", name=name, args=args, type=src0)

    def _analyze_predicate_reorder_op(
        self,
        name: str,
        args: tuple[SemanticExpr, ...],
    ) -> SemanticExpr:
        if len(args) != 2:
            raise TypeError(f"pto.{name} expects exactly 2 positional arguments in TileLang DSL v1")
        lhs = self._require_mask_expr(args[0], f"pto.{name} src0")
        rhs = self._require_mask_expr(args[1], f"pto.{name} src1")
        expected_granularity = {
            "pdintlv_b8": "b8",
            "pdintlv_b16": "b16",
            "pdintlv_b32": "b32",
            "pintlv_b8": "b8",
            "pintlv_b16": "b16",
            "pintlv_b32": "b32",
        }[name]
        if lhs.granularity != expected_granularity or rhs.granularity != expected_granularity:
            raise TypeError(f"pto.{name} expects !pto.mask<{expected_granularity}> operands")
        return SemanticCallExpr(
            namespace="pto",
            name=name,
            args=args,
            type=SemanticTupleType(
                elements=(
                    SemanticMaskType(granularity=expected_granularity),
                    SemanticMaskType(granularity=expected_granularity),
                )
            ),
        )

    def _analyze_compare_op(
        self,
        name: str,
        args: tuple[SemanticExpr, ...],
    ) -> SemanticExpr:
        if name == "vcmp":
            if len(args) != 4:
                raise TypeError("pto.vcmp expects exactly 4 positional arguments in TileLang DSL")
            lhs = self._require_vreg_expr(args[0], "pto.vcmp lhs")
            rhs = self._require_vreg_expr(args[1], "pto.vcmp rhs")
            if lhs != rhs:
                raise TypeError("pto.vcmp requires lhs/rhs vector types to match")
            seed = self._require_mask_expr(args[2], "pto.vcmp seed mask")
            self._require_mask_for_vreg(args[2], lhs, "pto.vcmp")
            cmp_mode = self._normalize_cmp_mode(args[3], "pto.vcmp compare mode")
            return SemanticCallExpr(
                namespace="pto",
                name=name,
                args=(args[0], args[1], args[2], cmp_mode),
                type=SemanticMaskType(granularity=seed.granularity),
            )

        if len(args) != 4:
            raise TypeError("pto.vcmps expects exactly 4 positional arguments in TileLang DSL")
        vector = self._require_vreg_expr(args[0], "pto.vcmps vector")
        scalar = self._require_scalar_expr(args[1], "pto.vcmps scalar")
        if scalar.dtype != vector.element_dtype:
            raise TypeError("pto.vcmps scalar dtype must match vector element dtype")
        seed = self._require_mask_expr(args[2], "pto.vcmps seed mask")
        self._require_mask_for_vreg(args[2], vector, "pto.vcmps")
        cmp_mode = self._normalize_cmp_mode(args[3], "pto.vcmps compare mode")
        return SemanticCallExpr(
            namespace="pto",
            name=name,
            args=(args[0], args[1], args[2], cmp_mode),
            type=SemanticMaskType(granularity=seed.granularity),
        )

    def _analyze_select_op(
        self,
        name: str,
        args: tuple[SemanticExpr, ...],
    ) -> SemanticExpr:
        if name == "vsel":
            if len(args) != 3:
                raise TypeError("pto.vsel expects exactly 3 positional arguments in TileLang DSL")
            src0 = self._require_vreg_expr(args[0], "pto.vsel src0")
            src1 = self._require_vreg_expr(args[1], "pto.vsel src1")
            if src0 != src1:
                raise TypeError("pto.vsel requires src0/src1 vector types to match")
            self._require_mask_for_vreg(args[2], src0, "pto.vsel")
            return SemanticCallExpr(namespace="pto", name=name, args=args, type=src0)

        if len(args) != 2:
            raise TypeError(f"pto.{name} expects exactly 2 positional arguments in TileLang DSL")
        src0 = self._require_vreg_expr(args[0], f"pto.{name} src0")
        src1 = self._require_vreg_expr(args[1], f"pto.{name} src1")
        if src1.lanes != src0.lanes:
            raise TypeError(f"pto.{name} requires src0/src1 to have identical lane counts")
        src1_bits = integer_bitwidth(src1.element_dtype)
        if src1_bits is None:
            raise TypeError(f"pto.{name} requires src1 to use integer vector elements")
        if src1_bits != bytewidth(src0.element_dtype) * 8:
            raise TypeError(f"pto.{name} requires src1 integer element width to match src0 element width")
        return SemanticCallExpr(namespace="pto", name=name, args=args, type=src0)

    def _analyze_carry_op(
        self,
        name: str,
        args: tuple[SemanticExpr, ...],
    ) -> SemanticExpr:
        if name in {"vaddc", "vsubc"}:
            if len(args) != 3:
                raise TypeError(f"pto.{name} expects exactly 3 positional arguments in TileLang DSL")
            lhs = self._require_vreg_expr(args[0], f"pto.{name} lhs")
            rhs = self._require_vreg_expr(args[1], f"pto.{name} rhs")
            if lhs != rhs:
                raise TypeError(f"pto.{name} requires lhs/rhs vector types to match")
            self._require_mask_for_vreg(args[2], lhs, f"pto.{name}")
            carry_type = args[2].type
            return SemanticCallExpr(
                namespace="pto",
                name=name,
                args=args,
                type=SemanticTupleType(elements=(lhs, carry_type)),
            )

        if len(args) != 4:
            raise TypeError(f"pto.{name} expects exactly 4 positional arguments in TileLang DSL")
        lhs = self._require_vreg_expr(args[0], f"pto.{name} lhs")
        rhs = self._require_vreg_expr(args[1], f"pto.{name} rhs")
        if lhs != rhs:
            raise TypeError(f"pto.{name} requires lhs/rhs vector types to match")
        carry_in = self._require_mask_expr(args[2], f"pto.{name} carry_in")
        self._require_mask_for_vreg(args[3], lhs, f"pto.{name}")
        carry_mask = self._require_mask_expr(args[3], f"pto.{name} mask")
        self._require_matching_mask_types(carry_in, carry_mask, f"pto.{name}")
        return SemanticCallExpr(
            namespace="pto",
            name=name,
            args=args,
            type=SemanticTupleType(elements=(lhs, carry_in)),
        )

    def _analyze_rearrangement_op(
        self,
        name: str,
        args: tuple[SemanticExpr, ...],
    ) -> SemanticExpr:
        if name in {"vintlv", "vdintlv"}:
            if len(args) != 2:
                raise TypeError(f"pto.{name} expects exactly 2 positional arguments in TileLang DSL")
            lhs = self._require_vreg_expr(args[0], f"pto.{name} lhs")
            rhs = self._require_vreg_expr(args[1], f"pto.{name} rhs")
            if lhs != rhs:
                raise TypeError(f"pto.{name} requires lhs/rhs vector types to match")
            return SemanticCallExpr(
                namespace="pto",
                name=name,
                args=args,
                type=SemanticTupleType(elements=(lhs, lhs)),
            )

        if len(args) != 3:
            raise TypeError(f"pto.{name} expects exactly 3 positional arguments in TileLang DSL")
        lhs = self._require_vreg_expr(args[0], f"pto.{name} lhs")
        rhs = self._require_vreg_expr(args[1], f"pto.{name} rhs")
        if lhs != rhs:
            raise TypeError(f"pto.{name} requires lhs/rhs vector types to match")
        self._require_string_expr(args[2], f"pto.{name} part")
        return SemanticCallExpr(namespace="pto", name=name, args=args, type=lhs)

    def _missing_optional_meta_expr(self) -> SemanticLiteralExpr:
        return SemanticLiteralExpr(value=None, type=SemanticMetaType(kind="none"))

    def _analyze_vcvt_frontend_call(
        self,
        expr: FrontendCallExpr,
        env: dict[str, SemanticBinding],
        *,
        allow_outer_lookup: bool,
    ) -> SemanticExpr:
        if len(expr.args) != 3:
            raise TypeError(
                "pto.vcvt expects exactly 3 positional operands `(vec, to_type, mask)` "
                "before optional keyword attrs in TileLang DSL v1"
            )
        args = tuple(
            self._analyze_expr(arg, env, allow_outer_lookup=allow_outer_lookup)
            for arg in expr.args
        )
        analyzed_keywords = {
            name: self._analyze_expr(value, env, allow_outer_lookup=allow_outer_lookup)
            for name, value in expr.keywords
        }
        allowed_keywords = {"rnd", "sat", "part"}
        unexpected_keywords = sorted(set(analyzed_keywords) - allowed_keywords)
        if unexpected_keywords:
            keyword_text = ", ".join(unexpected_keywords)
            raise TypeError(
                "pto.vcvt only accepts keyword attrs `rnd`, `sat`, and `part`; "
                f"got unsupported keyword(s): {keyword_text}"
            )
        return self._analyze_vcvt(
            args,
            rnd=self._normalize_vcvt_round_mode(analyzed_keywords.get("rnd")),
            sat=self._normalize_vcvt_sat_mode(analyzed_keywords.get("sat")),
            part=self._normalize_vcvt_part_mode(analyzed_keywords.get("part")),
            rnd_explicit="rnd" in analyzed_keywords,
            sat_explicit="sat" in analyzed_keywords,
            part_explicit="part" in analyzed_keywords,
        )

    def _analyze_vtrc_frontend_call(
        self,
        expr: FrontendCallExpr,
        env: dict[str, SemanticBinding],
        *,
        allow_outer_lookup: bool,
    ) -> SemanticExpr:
        if len(expr.args) != 2:
            raise TypeError(
                "pto.vtrc expects exactly 2 positional operands `(vec, mask)` "
                "before optional keyword attrs in TileLang DSL v1"
            )
        args = tuple(
            self._analyze_expr(arg, env, allow_outer_lookup=allow_outer_lookup)
            for arg in expr.args
        )
        analyzed_keywords = {
            name: self._analyze_expr(value, env, allow_outer_lookup=allow_outer_lookup)
            for name, value in expr.keywords
        }
        allowed_keywords = {"rnd"}
        unexpected_keywords = sorted(set(analyzed_keywords) - allowed_keywords)
        if unexpected_keywords:
            keyword_text = ", ".join(unexpected_keywords)
            raise TypeError(
                "pto.vtrc only accepts keyword attr `rnd`; "
                f"got unsupported keyword(s): {keyword_text}"
            )
        return self._analyze_vtrc(
            args,
            rnd=self._normalize_vtrc_round_mode(analyzed_keywords.get("rnd")),
        )

    def _analyze_vcvt(
        self,
        args: tuple[SemanticExpr, ...],
        *,
        rnd: SemanticExpr | None = None,
        sat: SemanticExpr | None = None,
        part: SemanticExpr | None = None,
        rnd_explicit: bool = False,
        sat_explicit: bool = False,
        part_explicit: bool = False,
    ) -> SemanticExpr:
        if len(args) != 3:
            raise TypeError("pto.vcvt expects exactly 3 positional arguments in TileLang DSL")
        vector = self._require_vreg_expr(args[0], "pto.vcvt vector")
        target_dtype = self._require_dtype_symbol(args[1], "pto.vcvt to_type")
        self._require_mask_for_vreg(args[2], vector, "pto.vcvt")
        contract = self._lookup_vcvt_attr_contract(vector.element_dtype, target_dtype)
        if contract is not None:
            self._require_explicit_vcvt_attrs(
                src_dtype=vector.element_dtype,
                dst_dtype=target_dtype,
                rnd_required=contract[0],
                sat_required=contract[1],
                part_required=contract[2],
                rnd_explicit=rnd_explicit,
                sat_explicit=sat_explicit,
                part_explicit=part_explicit,
            )
        return SemanticCallExpr(
            namespace="pto",
            name="vcvt",
            args=(
                args[0],
                args[1],
                args[2],
                rnd if rnd is not None else self._missing_optional_meta_expr(),
                sat if sat is not None else self._missing_optional_meta_expr(),
                part if part is not None else self._missing_optional_meta_expr(),
            ),
            type=self._vreg_type_for_dtype(target_dtype),
        )

    def _analyze_vpack_op(
        self,
        args: tuple[SemanticExpr, ...],
    ) -> SemanticExpr:
        if len(args) != 2:
            raise TypeError("pto.vpack expects exactly 2 positional arguments in TileLang DSL")
        vector = self._require_vreg_expr(args[0], "pto.vpack vector")
        part = self._normalize_predicate_part(args[1], "pto.vpack part")
        self._validate_binary_dtype("vpack", vector.element_dtype)
        result_dtype = self._pack_result_dtype(vector.element_dtype)
        return SemanticCallExpr(
            namespace="pto",
            name="vpack",
            args=(args[0], part),
            type=SemanticVRegType(element_dtype=result_dtype, lanes=vector.lanes * 2),
        )

    def _analyze_vtrc(
        self,
        args: tuple[SemanticExpr, ...],
        *,
        rnd: SemanticExpr | None = None,
    ) -> SemanticExpr:
        if len(args) != 2:
            raise TypeError("pto.vtrc expects exactly 2 positional arguments in TileLang DSL v1")
        vector = self._require_vreg_expr(args[0], "pto.vtrc vector")
        self._require_mask_for_vreg(args[1], vector, "pto.vtrc")
        if vector.element_dtype not in {f16, bf16, f32}:
            raise TypeError("pto.vtrc only supports f16/bf16/f32 vector element types in TileLang DSL v1")
        return SemanticCallExpr(
            namespace="pto",
            name="vtrc",
            args=(
                args[0],
                args[1],
                rnd
                if rnd is not None
                else SemanticLiteralExpr(value="R", type=SemanticMetaType(kind="string")),
            ),
            type=vector,
        )

    def _analyze_vbitcast(self, args: tuple[SemanticExpr, ...]) -> SemanticExpr:
        if len(args) != 2:
            raise TypeError("pto.vbitcast expects exactly 2 positional arguments in TileLang DSL")
        vector = self._require_vreg_expr(args[0], "pto.vbitcast vector")
        target_dtype = self._require_dtype_symbol(args[1], "pto.vbitcast to_type")
        # No mask for vbitcast (pure type conversion)
        return SemanticCallExpr(
            namespace="pto",
            name="vbitcast",
            args=(
                args[0],
                args[1],
            ),
            type=self._vreg_type_for_dtype(target_dtype),
        )

    def _analyze_pbitcast(self, args: tuple[SemanticExpr, ...]) -> SemanticExpr:
        if len(args) != 2:
            raise TypeError("pto.pbitcast expects exactly 2 positional arguments in TileLang DSL")
        self._require_mask_expr(args[0], "pto.pbitcast mask")
        target_mask_type = self._require_mask_type_expr(args[1], "pto.pbitcast to_type")
        return SemanticCallExpr(
            namespace="pto",
            name="pbitcast",
            args=(
                args[0],
                args[1],
            ),
            type=SemanticMaskType(granularity=target_mask_type.granularity),
        )

    def _analyze_vbitsort(self, args: tuple[SemanticExpr, ...]) -> SemanticExpr:
        if len(args) != 4:
            raise TypeError("pto.vbitsort expects exactly 4 positional arguments in TileLang DSL v1")
        destination = self._require_pointer_expr(args[0], "pto.vbitsort destination", memory_space="ub")
        source = self._require_pointer_expr(args[1], "pto.vbitsort source", memory_space="ub")
        indices = self._require_pointer_expr(args[2], "pto.vbitsort indices", memory_space="ub")
        count = self._require_index_typed_expr(args[3])
        return SemanticCallExpr(
            namespace="pto",
            name="vbitsort",
            args=(destination, source, indices, count),
            type=None,
        )

    def _analyze_vmrgsort4(self, args: tuple[SemanticExpr, ...]) -> SemanticExpr:
        if len(args) != 7:
            raise TypeError("pto.vmrgsort4 expects exactly 7 positional arguments in TileLang DSL v1")
        destination = self._require_pointer_expr(args[0], "pto.vmrgsort4 destination", memory_space="ub")
        source0 = self._require_pointer_expr(args[1], "pto.vmrgsort4 src0", memory_space="ub")
        source1 = self._require_pointer_expr(args[2], "pto.vmrgsort4 src1", memory_space="ub")
        source2 = self._require_pointer_expr(args[3], "pto.vmrgsort4 src2", memory_space="ub")
        source3 = self._require_pointer_expr(args[4], "pto.vmrgsort4 src3", memory_space="ub")
        self._require_i64_like_expr(args[5], "pto.vmrgsort4 count")
        self._require_i64_like_expr(args[6], "pto.vmrgsort4 config")
        return SemanticCallExpr(
            namespace="pto",
            name="vmrgsort4",
            args=(destination, source0, source1, source2, source3, args[5], args[6]),
            type=None,
        )

    def _analyze_get_vms4_sr(self, args: tuple[SemanticExpr, ...]) -> SemanticExpr:
        if args:
            raise TypeError("pto.get_vms4_sr does not accept positional arguments in TileLang DSL v1")
        count_type = SemanticScalarType(dtype=i16)
        return SemanticCallExpr(
            namespace="pto",
            name="get_vms4_sr",
            args=(),
            type=SemanticTupleType(elements=(count_type, count_type, count_type, count_type)),
        )

    def _require_dtype_symbol(self, expr: SemanticExpr, context: str) -> ScalarType:
        if not (
            isinstance(expr, SemanticSymbolExpr)
            and expr.type.kind == "dtype"
            and isinstance(expr.value, ScalarType)
        ):
            if (
                isinstance(expr, SemanticBindingRef)
                and isinstance(expr.type, SemanticMetaType)
                and expr.type.kind == "dtype"
                and isinstance(expr.binding.value, ScalarType)
            ):
                return expr.binding.value
            raise TypeError(f"{context} must be a TileLang scalar dtype symbol in TileLang DSL v1")
        return expr.value

    def _require_non_storage_only_dtype(self, dtype: ScalarType, context: str) -> None:
        if is_storage_only_dtype(dtype):
            raise TypeError(
                f"{context} does not accept storage-only low-precision dtype `{dtype.name}`; "
                "these dtypes are only supported by storage and ptr surfaces in TileLang DSL v1"
            )

    def _require_non_low_precision_dtype(self, dtype: ScalarType, context: str) -> None:
        if is_low_precision_dtype(dtype):
            raise TypeError(
                f"{context} does not accept low-precision dtype `{dtype.name}`; "
                "these dtypes are not scalar element dtypes in TileLang DSL v1"
            )

    def _dtype_symbol_expr(self, dtype: ScalarType) -> SemanticSymbolExpr:
        return SemanticSymbolExpr(
            namespace="pto",
            name=dtype.name,
            value=dtype,
            type=SemanticMetaType(kind="dtype"),
        )

    def _require_memory_space_symbol(self, expr: SemanticExpr, context: str) -> MemorySpace:
        if (
            isinstance(expr, SemanticSymbolExpr)
            and expr.type.kind == "memory_space"
            and isinstance(expr.value, MemorySpace)
        ):
            return expr.value
        if (
            isinstance(expr, SemanticBindingRef)
            and isinstance(expr.type, SemanticMetaType)
            and expr.type.kind == "memory_space"
            and isinstance(expr.binding.value, MemorySpace)
        ):
            return expr.binding.value
        raise TypeError(f"{context} must be a TileLang MemorySpace symbol")

    def _require_ptr_type_expr(self, expr: SemanticExpr, context: str) -> PointerType:
        if (
            isinstance(expr, SemanticLiteralExpr)
            and isinstance(expr.type, SemanticMetaType)
            and expr.type.kind == "ptr_type"
            and isinstance(expr.value, PointerType)
        ):
            return expr.value
        if (
            isinstance(expr, SemanticBindingRef)
            and isinstance(expr.type, SemanticMetaType)
            and expr.type.kind == "ptr_type"
            and isinstance(expr.binding.value, PointerType)
        ):
            return expr.binding.value
        raise TypeError(f"{context} must be a pointer type constructed with pto.ptr(...)")

    def _require_vreg_type_expr(self, expr: SemanticExpr, context: str) -> VRegType:
        if (
            isinstance(expr, SemanticLiteralExpr)
            and isinstance(expr.type, SemanticMetaType)
            and expr.type.kind == "vreg_type"
            and isinstance(expr.value, VRegType)
        ):
            return expr.value
        if (
            isinstance(expr, SemanticBindingRef)
            and isinstance(expr.type, SemanticMetaType)
            and expr.type.kind == "vreg_type"
            and isinstance(expr.binding.value, VRegType)
        ):
            return expr.binding.value
        raise TypeError(f"{context} must be a vector type constructed with pto.vreg(...)")

    def _require_vector_type_expr(self, expr: SemanticExpr, context: str) -> VectorType:
        if (
            isinstance(expr, SemanticLiteralExpr)
            and isinstance(expr.type, SemanticMetaType)
            and expr.type.kind == "vector_type"
            and isinstance(expr.value, VectorType)
        ):
            return expr.value
        if (
            isinstance(expr, SemanticBindingRef)
            and isinstance(expr.type, SemanticMetaType)
            and expr.type.kind == "vector_type"
            and isinstance(expr.binding.value, VectorType)
        ):
            return expr.binding.value
        raise TypeError(f"{context} must be a builtin vector type constructed with pto.vector(...)")

    def _require_vector_shape_expr(self, expr: SemanticExpr, context: str) -> tuple[int, ...]:
        if not isinstance(expr, SemanticTupleExpr):
            dim = self._static_index_value(expr, default=None)
            if dim is None:
                raise TypeError(f"{context} must be a static integer or tuple of static integers")
            if dim <= 0:
                raise TypeError(f"{context} shape entries must be positive")
            return (dim,)
        if isinstance(expr, SemanticTupleExpr):
            shape: list[int] = []
            for element in expr.elements:
                dim = self._static_index_value(element, default=None)
                if dim is None:
                    raise TypeError(f"{context} tuple entries must be static integers")
                if dim <= 0:
                    raise TypeError(f"{context} shape entries must be positive")
                shape.append(dim)
            if not shape:
                raise TypeError(f"{context} must be a non-empty shape")
            return tuple(shape)
        raise TypeError(f"{context} must be a static integer or tuple of static integers")

    def _require_mask_type_expr(self, expr: SemanticExpr, context: str) -> MaskType:
        if (
            isinstance(expr, SemanticSymbolExpr)
            and isinstance(expr.type, SemanticMetaType)
            and expr.type.kind == "mask_type"
            and isinstance(expr.value, MaskType)
        ):
            return expr.value
        if (
            isinstance(expr, SemanticBindingRef)
            and isinstance(expr.type, SemanticMetaType)
            and expr.type.kind == "mask_type"
            and isinstance(expr.binding.value, MaskType)
        ):
            return expr.binding.value
        raise TypeError(f"{context} must be a mask type such as pto.mask_b32")

    def _require_cast_target_type(self, expr: SemanticExpr) -> SemanticType:
        if self._is_i64_dtype_expr(expr):
            return SemanticScalarType(dtype=i64)
        ptr_type = self._require_ptr_type_expr(expr, "pto.castptr target type")
        return SemanticPtrType(
            element_dtype=ptr_type.element_dtype,
            memory_space=ptr_type.memory_space.value,
        )

    def _require_castptr_input(self, expr: SemanticExpr, target_type: SemanticPtrType) -> None:
        if isinstance(expr.type, SemanticIndexType):
            return
        if isinstance(expr.type, SemanticScalarType) and expr.type.dtype == i64:
            return
        if isinstance(expr.type, SemanticPtrType):
            if expr.type.memory_space != target_type.memory_space:
                raise TypeError("pto.castptr pointer-to-pointer casts must stay within one PTO memory space")
            return
        raise TypeError("pto.castptr input must be an index/i64, pointer, or memref-backed address value")

    def _is_i64_dtype_expr(self, expr: SemanticExpr) -> bool:
        if isinstance(expr, SemanticSymbolExpr):
            return expr.type.kind == "dtype" and expr.value == i64
        if isinstance(expr, SemanticBindingRef):
            return (
                isinstance(expr.type, SemanticMetaType)
                and expr.type.kind == "dtype"
                and expr.binding.value == i64
            )
        return False

    def _require_vreg_expr(self, expr: SemanticExpr, context: str) -> SemanticVRegType:
        if not isinstance(expr.type, SemanticVRegType):
            raise TypeError(f"{context} must be a vector register value in TileLang DSL v1")
        return expr.type

    def _require_scalar_expr(self, expr: SemanticExpr, context: str) -> SemanticScalarType:
        if not isinstance(expr.type, SemanticScalarType):
            raise TypeError(f"{context} must be a scalar value in TileLang DSL v1")
        return expr.type

    def _require_scalar_or_index_expr(self, expr: SemanticExpr, context: str) -> SemanticExpr:
        if isinstance(expr.type, (SemanticScalarType, SemanticIndexType)):
            return expr
        raise TypeError(f"{context} must be a scalar or index value in TileLang DSL v1")

    def _vreg_type_for_scalar_or_index(self, expr: SemanticExpr, context: str) -> SemanticVRegType:
        value = self._require_scalar_or_index_expr(expr, context)
        if isinstance(value.type, SemanticScalarType):
            return self._vreg_type_for_dtype(value.type.dtype)
        return self._vreg_type_for_dtype(i32)

    def _vcadd_result_vreg_type(self, vreg_type: SemanticVRegType) -> SemanticVRegType:
        dtype = vreg_type.element_dtype
        if not is_integer_dtype(dtype):
            return vreg_type
        signedness = integer_signedness(dtype)
        bitwidth = integer_bitwidth(dtype)
        if bitwidth == 8:
            widened_dtype = ui16 if signedness == "unsigned" else i16
            return self._vreg_type_for_dtype(widened_dtype)
        if bitwidth == 16:
            widened_dtype = ui32 if signedness == "unsigned" else i32
            return self._vreg_type_for_dtype(widened_dtype)
        return vreg_type

    def _vexpdif_result_vreg_type(self, vreg_type: SemanticVRegType) -> SemanticVRegType:
        if vreg_type.element_dtype.name == "f32":
            return vreg_type
        return SemanticVRegType(element_dtype=f32, lanes=vreg_type.lanes // 2)

    def _normalize_position_mode(
        self,
        expr: SemanticExpr | None,
        context: str,
    ) -> SemanticExpr:
        if expr is None:
            return SemanticLiteralExpr(value=PositionMode.LOWEST.value, type=SemanticMetaType(kind="string"))
        if (
            isinstance(expr, SemanticSymbolExpr)
            and isinstance(expr.type, SemanticMetaType)
            and expr.type.kind == "position_mode"
            and isinstance(expr.value, PositionMode)
        ):
            return SemanticLiteralExpr(value=expr.value.value, type=SemanticMetaType(kind="string"))
        if (
            isinstance(expr, SemanticBindingRef)
            and isinstance(expr.type, SemanticMetaType)
            and expr.type.kind == "position_mode"
            and isinstance(expr.binding.value, PositionMode)
        ):
            return SemanticLiteralExpr(value=expr.binding.value.value, type=SemanticMetaType(kind="string"))
        position = self._require_string_expr(expr, context)
        if position == "POS_LOWEST":
            position = PositionMode.LOWEST.value
        if position not in {PositionMode.LOWEST.value, PositionMode.HIGHEST.value}:
            raise TypeError(
                "pto.vdup position must be `PositionMode.LOWEST` or `PositionMode.HIGHEST` in TileLang DSL v1"
            )
        return SemanticLiteralExpr(value=position, type=SemanticMetaType(kind="string"))

    def _normalize_order_mode(
        self,
        expr: SemanticExpr | None,
        context: str,
    ) -> SemanticExpr:
        if expr is None:
            return SemanticLiteralExpr(value=OrderMode.ASC.value, type=SemanticMetaType(kind="string"))
        if (
            isinstance(expr, SemanticSymbolExpr)
            and isinstance(expr.type, SemanticMetaType)
            and expr.type.kind == "order_mode"
            and isinstance(expr.value, OrderMode)
        ):
            return SemanticLiteralExpr(value=expr.value.value, type=SemanticMetaType(kind="string"))
        if (
            isinstance(expr, SemanticBindingRef)
            and isinstance(expr.type, SemanticMetaType)
            and expr.type.kind == "order_mode"
            and isinstance(expr.binding.value, OrderMode)
        ):
            return SemanticLiteralExpr(value=expr.binding.value.value, type=SemanticMetaType(kind="string"))
        order = self._require_string_expr(expr, context)
        if order not in {OrderMode.ASC.value, OrderMode.DESC.value}:
            raise TypeError(
                "pto.vci currently only supports order `OrderMode.ASC` or `OrderMode.DESC` in TileLang DSL v1"
            )
        return SemanticLiteralExpr(value=order, type=SemanticMetaType(kind="string"))

    def _normalize_vcvt_round_mode(self, expr: SemanticExpr | None) -> SemanticExpr | None:
        if expr is None:
            return None
        if (
            isinstance(expr, SemanticSymbolExpr)
            and isinstance(expr.type, SemanticMetaType)
            and expr.type.kind == "vcvt_round_mode"
            and isinstance(expr.value, VcvtRoundMode)
        ):
            return SemanticLiteralExpr(value=expr.value.value, type=SemanticMetaType(kind="string"))
        if (
            isinstance(expr, SemanticBindingRef)
            and isinstance(expr.type, SemanticMetaType)
            and expr.type.kind == "vcvt_round_mode"
            and isinstance(expr.binding.value, VcvtRoundMode)
        ):
            return SemanticLiteralExpr(value=expr.binding.value.value, type=SemanticMetaType(kind="string"))
        round_mode = self._require_string_expr(expr, "pto.vcvt rnd")
        if round_mode not in {mode.value for mode in VcvtRoundMode}:
            raise TypeError(
                "pto.vcvt rnd must be a VcvtRoundMode enum such as "
                "`pto.VcvtRoundMode.R` or one of the canonical strings "
                '`"R"`, `"A"`, `"F"`, `"C"`, `"Z"`, `"O"` in TileLang DSL v1'
            )
        return SemanticLiteralExpr(value=round_mode, type=SemanticMetaType(kind="string"))

    def _normalize_vtrc_round_mode(self, expr: SemanticExpr | None) -> SemanticExpr | None:
        normalized = self._normalize_vcvt_round_mode(expr)
        if normalized is None:
            return None
        round_mode = self._require_string_expr(normalized, "pto.vtrc rnd")
        if round_mode == VcvtRoundMode.O.value:
            raise TypeError(
                "pto.vtrc rnd must be one of "
                '`"R"`, `"A"`, `"F"`, `"C"`, `"Z"` or a matching '
                "VcvtRoundMode enum in TileLang DSL v1"
            )
        return SemanticLiteralExpr(value=round_mode, type=SemanticMetaType(kind="string"))

    def _normalize_vcvt_sat_mode(self, expr: SemanticExpr | None) -> SemanticExpr | None:
        if expr is None:
            return None
        if (
            isinstance(expr, SemanticSymbolExpr)
            and isinstance(expr.type, SemanticMetaType)
            and expr.type.kind == "vcvt_sat_mode"
            and isinstance(expr.value, VcvtSatMode)
        ):
            return SemanticLiteralExpr(value=expr.value.value, type=SemanticMetaType(kind="string"))
        if (
            isinstance(expr, SemanticBindingRef)
            and isinstance(expr.type, SemanticMetaType)
            and expr.type.kind == "vcvt_sat_mode"
            and isinstance(expr.binding.value, VcvtSatMode)
        ):
            return SemanticLiteralExpr(value=expr.binding.value.value, type=SemanticMetaType(kind="string"))
        sat_mode = self._require_string_expr(expr, "pto.vcvt sat")
        if sat_mode not in {mode.value for mode in VcvtSatMode}:
            raise TypeError(
                "pto.vcvt sat must be a VcvtSatMode enum such as "
                "`pto.VcvtSatMode.SAT` or `pto.VcvtSatMode.NOSAT`, or one of the "
                'canonical strings `"SAT"` / `"NOSAT"` in TileLang DSL v1'
            )
        return SemanticLiteralExpr(value=sat_mode, type=SemanticMetaType(kind="string"))

    def _normalize_vcvt_part_mode(self, expr: SemanticExpr | None) -> SemanticExpr | None:
        if expr is None:
            return None
        if (
            isinstance(expr, SemanticSymbolExpr)
            and isinstance(expr.type, SemanticMetaType)
            and expr.type.kind == "vcvt_part_mode"
            and isinstance(expr.value, VcvtPartMode)
        ):
            return SemanticLiteralExpr(value=expr.value.value, type=SemanticMetaType(kind="string"))
        if (
            isinstance(expr, SemanticBindingRef)
            and isinstance(expr.type, SemanticMetaType)
            and expr.type.kind == "vcvt_part_mode"
            and isinstance(expr.binding.value, VcvtPartMode)
        ):
            return SemanticLiteralExpr(value=expr.binding.value.value, type=SemanticMetaType(kind="string"))
        part_mode = self._require_string_expr(expr, "pto.vcvt part")
        if part_mode not in {mode.value for mode in VcvtPartMode}:
            raise TypeError(
                "pto.vcvt part must be a VcvtPartMode enum such as "
                "`pto.VcvtPartMode.EVEN`, `pto.VcvtPartMode.ODD`, or "
                "`pto.VcvtPartMode.P0`..`pto.VcvtPartMode.P3`, or one of the "
                'canonical strings `"EVEN"`, `"ODD"`, `"P0"`, `"P1"`, `"P2"`, or `"P3"` '
                "in TileLang DSL v1"
            )
        return SemanticLiteralExpr(value=part_mode, type=SemanticMetaType(kind="string"))

    def _lookup_vcvt_attr_contract(
        self, src_dtype: ScalarType, dst_dtype: ScalarType
    ) -> tuple[bool, bool, bool] | None:
        src_kind = _classify_vcvt_elem_kind(src_dtype)
        dst_kind = _classify_vcvt_elem_kind(dst_dtype)
        if src_kind is None or dst_kind is None:
            return None
        return _VCVT_ATTR_CONTRACTS.get((src_kind, dst_kind))

    def _require_explicit_vcvt_attrs(
        self,
        *,
        src_dtype: ScalarType,
        dst_dtype: ScalarType,
        rnd_required: bool,
        sat_required: bool,
        part_required: bool,
        rnd_explicit: bool,
        sat_explicit: bool,
        part_explicit: bool,
    ) -> None:
        pair = f"{src_dtype.name}->{dst_dtype.name}"

        def _check(attr_name: str, required: bool, explicit: bool) -> None:
            if required and not explicit:
                raise TypeError(
                    f"pto.vcvt {pair} requires explicit `{attr_name}=` in TileLang DSL v1"
                )
            if not required and explicit:
                raise TypeError(
                    f"pto.vcvt {pair} does not accept `{attr_name}=` for this type pair in TileLang DSL v1"
                )

        _check("rnd", rnd_required, rnd_explicit)
        _check("sat", sat_required, sat_explicit)
        _check("part", part_required, part_explicit)

    def _require_mask_expr(self, expr: SemanticExpr, context: str) -> SemanticMaskType:
        if not isinstance(expr.type, SemanticMaskType):
            raise TypeError(f"{context} must be a mask value in TileLang DSL")
        return expr.type

    def _require_align_expr(self, expr: SemanticExpr, context: str) -> None:
        if not isinstance(expr.type, SemanticAlignType):
            raise TypeError(f"{context} must be a pto.align value in TileLang DSL v1")

    def _require_matching_mask_types(
        self,
        lhs: SemanticMaskType,
        rhs: SemanticMaskType,
        context: str,
    ) -> None:
        if lhs != rhs:
            raise TypeError(f"{context} requires all mask operands to use the same mask granularity")

    def _require_string_expr(self, expr: SemanticExpr, context: str) -> str:
        if isinstance(expr, SemanticLiteralExpr) and isinstance(expr.type, SemanticMetaType) and expr.type.kind == "string":
            return expr.value
        if (
            isinstance(expr, SemanticBindingRef)
            and isinstance(expr.type, SemanticMetaType)
            and expr.type.kind == "string"
            and isinstance(expr.binding.value, str)
        ):
            return expr.binding.value
        raise TypeError(f"{context} must be a string literal in TileLang DSL")

    def _normalize_vexpdif_part(self, expr: SemanticExpr, context: str) -> SemanticExpr:
        if (
            isinstance(expr, SemanticSymbolExpr)
            and isinstance(expr.type, SemanticMetaType)
            and expr.type.kind == "vcvt_part_mode"
            and isinstance(expr.value, VcvtPartMode)
        ):
            part = expr.value.value
        elif (
            isinstance(expr, SemanticBindingRef)
            and isinstance(expr.type, SemanticMetaType)
            and expr.type.kind == "vcvt_part_mode"
            and isinstance(expr.binding.value, VcvtPartMode)
        ):
            part = expr.binding.value.value
        else:
            part = self._require_string_expr(expr, context)
        if part not in {VcvtPartMode.EVEN.value, VcvtPartMode.ODD.value}:
            raise TypeError(
                "pto.vexpdif part must be `pto.VcvtPartMode.EVEN` or "
                "`pto.VcvtPartMode.ODD`, or one of the canonical strings "
                '`"EVEN"` / `"ODD"` in TileLang DSL v1'
            )
        return SemanticLiteralExpr(value=part, type=SemanticMetaType(kind="string"))

    def _normalize_cmp_mode(self, expr: SemanticExpr, context: str) -> SemanticExpr:
        if (
            isinstance(expr, SemanticSymbolExpr)
            and isinstance(expr.type, SemanticMetaType)
            and expr.type.kind == "cmp_mode"
            and isinstance(expr.value, CmpMode)
        ):
            return SemanticLiteralExpr(value=expr.value.value, type=SemanticMetaType(kind="string"))
        if (
            isinstance(expr, SemanticBindingRef)
            and isinstance(expr.type, SemanticMetaType)
            and expr.type.kind == "cmp_mode"
            and isinstance(expr.binding.value, CmpMode)
        ):
            return SemanticLiteralExpr(value=expr.binding.value.value, type=SemanticMetaType(kind="string"))
        cmp_mode = self._require_string_expr(expr, context)
        if cmp_mode not in {mode.value for mode in CmpMode}:
            raise TypeError(
                f"{context} must be a CmpMode enum such as `pto.CmpMode.LT`, "
                'or one of the canonical strings `"eq"`, `"ne"`, `"lt"`, `"le"`, `"gt"`, `"ge"` '
                "in TileLang DSL v1"
            )
        return SemanticLiteralExpr(value=cmp_mode, type=SemanticMetaType(kind="string"))

    def _normalize_predicate_part(self, expr: SemanticExpr, context: str) -> SemanticExpr:
        if (
            isinstance(expr, SemanticSymbolExpr)
            and isinstance(expr.type, SemanticMetaType)
            and expr.type.kind == "predicate_part"
            and isinstance(expr.value, PredicatePart)
        ):
            return SemanticLiteralExpr(value=expr.value.value, type=SemanticMetaType(kind="string"))
        if (
            isinstance(expr, SemanticBindingRef)
            and isinstance(expr.type, SemanticMetaType)
            and expr.type.kind == "predicate_part"
            and isinstance(expr.binding.value, PredicatePart)
        ):
            return SemanticLiteralExpr(value=expr.binding.value.value, type=SemanticMetaType(kind="string"))
        part = self._require_string_expr(expr, context)
        if part not in {token.value for token in PredicatePart}:
            raise TypeError(
                f"{context} must be a PredicatePart enum such as `pto.PredicatePart.LOWER`, "
                'or one of the canonical strings `"LOWER"`, `"HIGHER"` in TileLang DSL v1'
            )
        return SemanticLiteralExpr(value=part, type=SemanticMetaType(kind="string"))

    def _normalize_post_update_mode(
        self,
        expr: SemanticExpr | None,
        context: str,
    ) -> SemanticExpr:
        if expr is None:
            return SemanticLiteralExpr(value="NO_POST_UPDATE", type=SemanticMetaType(kind="string"))
        if (
            isinstance(expr, SemanticSymbolExpr)
            and isinstance(expr.type, SemanticMetaType)
            and expr.type.kind == "post_update_mode"
            and isinstance(expr.value, PostUpdateMode)
        ):
            return SemanticLiteralExpr(value=expr.value.value, type=SemanticMetaType(kind="string"))
        if (
            isinstance(expr, SemanticBindingRef)
            and isinstance(expr.type, SemanticMetaType)
            and expr.type.kind == "post_update_mode"
            and isinstance(expr.binding.value, PostUpdateMode)
        ):
            return SemanticLiteralExpr(value=expr.binding.value.value, type=SemanticMetaType(kind="string"))
        raise TypeError(
            "pto.vstur mode must be a PostUpdateMode enum such as "
            "`pto.PostUpdateMode.NO_POST_UPDATE` or `pto.PostUpdateMode.POST_UPDATE` in TileLang DSL v1"
        )

    def _normalize_predicate_store_dist(
        self,
        expr: SemanticExpr | None,
        context: str,
    ) -> SemanticExpr:
        if expr is None:
            return SemanticLiteralExpr(value="NORM", type=SemanticMetaType(kind="string"))
        if (
            isinstance(expr, SemanticSymbolExpr)
            and isinstance(expr.type, SemanticMetaType)
            and expr.type.kind == "predicate_dist"
            and isinstance(expr.value, PredicateDist)
        ):
            dist = expr.value.value
        elif (
            isinstance(expr, SemanticBindingRef)
            and isinstance(expr.type, SemanticMetaType)
            and expr.type.kind == "predicate_dist"
            and isinstance(expr.binding.value, PredicateDist)
        ):
            dist = expr.binding.value.value
        else:
            raise TypeError(
                "predicate store dist must be a PredicateDist enum such as "
                "`pto.PredicateDist.NORM` or `pto.PredicateDist.PK` in TileLang DSL v1"
            )
        if dist not in {"NORM", "PK"}:
            raise TypeError(
                "predicate store dist must be one of "
                "`pto.PredicateDist.NORM` or `pto.PredicateDist.PK` in TileLang DSL v1"
            )
        return SemanticLiteralExpr(value=dist, type=SemanticMetaType(kind="string"))

    def _normalize_predicate_load_dist(
        self,
        expr: SemanticExpr | None,
        context: str,
    ) -> SemanticExpr:
        if expr is None:
            return SemanticLiteralExpr(value="NORM", type=SemanticMetaType(kind="string"))
        if (
            isinstance(expr, SemanticSymbolExpr)
            and isinstance(expr.type, SemanticMetaType)
            and expr.type.kind == "predicate_dist"
            and isinstance(expr.value, PredicateDist)
        ):
            dist = expr.value.value
        elif (
            isinstance(expr, SemanticBindingRef)
            and isinstance(expr.type, SemanticMetaType)
            and expr.type.kind == "predicate_dist"
            and isinstance(expr.binding.value, PredicateDist)
        ):
            dist = expr.binding.value.value
        else:
            raise TypeError(
                "predicate load dist must be a PredicateDist enum such as "
                "`pto.PredicateDist.NORM`, `pto.PredicateDist.US`, or `pto.PredicateDist.DS` in TileLang DSL v1"
            )
        if dist not in {"NORM", "US", "DS"}:
            raise TypeError(
                "predicate load dist must be one of "
                "`pto.PredicateDist.NORM`, `pto.PredicateDist.US`, or `pto.PredicateDist.DS` in TileLang DSL v1"
            )
        return SemanticLiteralExpr(value=dist, type=SemanticMetaType(kind="string"))

    def _normalize_vlds_dist(
        self,
        expr: SemanticExpr | None,
        context: str,
    ) -> SemanticExpr | None:
        if expr is None:
            return None
        if (
            isinstance(expr, SemanticSymbolExpr)
            and isinstance(expr.type, SemanticMetaType)
            and expr.type.kind == "vload_dist"
            and isinstance(expr.value, VLoadDist)
        ):
            dist = expr.value.value
        elif (
            isinstance(expr, SemanticBindingRef)
            and isinstance(expr.type, SemanticMetaType)
            and expr.type.kind == "vload_dist"
            and isinstance(expr.binding.value, VLoadDist)
        ):
            dist = expr.binding.value.value
        else:
            raise TypeError(
                "pto.vlds dist must be a VLoadDist enum such as "
                "`pto.VLoadDist.NORM`, `pto.VLoadDist.UNPK_B16`, or "
                "`pto.VLoadDist.BRC_B32` in TileLang DSL v1"
            )
        return SemanticLiteralExpr(value=dist, type=SemanticMetaType(kind="string"))

    def _normalize_vsts_dist(
        self,
        expr: SemanticExpr | None,
        context: str,
    ) -> SemanticExpr | None:
        if expr is None:
            return None
        if (
            isinstance(expr, SemanticSymbolExpr)
            and isinstance(expr.type, SemanticMetaType)
            and expr.type.kind == "vstore_dist"
            and isinstance(expr.value, VStoreDist)
        ):
            dist = expr.value.value
        elif (
            isinstance(expr, SemanticBindingRef)
            and isinstance(expr.type, SemanticMetaType)
            and expr.type.kind == "vstore_dist"
            and isinstance(expr.binding.value, VStoreDist)
        ):
            dist = expr.binding.value.value
        else:
            raise TypeError(
                "pto.vsts dist must be a VStoreDist enum such as "
                "`pto.VStoreDist.NORM_B32`, `pto.VStoreDist.PK_B32`, or "
                "`pto.VStoreDist.ONE_POINT_B8` in TileLang DSL v1"
            )
        return SemanticLiteralExpr(value=dist, type=SemanticMetaType(kind="string"))

    def _require_i1_expr(self, expr: SemanticExpr, context: str) -> None:
        scalar = self._require_scalar_expr(expr, context)
        if scalar.dtype != i1:
            raise TypeError(f"{context} must be an i1 value in TileLang DSL")

    def _require_i32_like_expr(self, expr: SemanticExpr, context: str) -> None:
        if isinstance(expr.type, SemanticIndexType):
            return
        scalar = self._require_scalar_expr(expr, context)
        if scalar.dtype != i32:
            raise TypeError(f"{context} must be an i32 or index value in TileLang DSL")

    def _require_i64_like_expr(self, expr: SemanticExpr, context: str) -> None:
        if isinstance(expr.type, SemanticIndexType):
            return
        scalar = self._require_scalar_expr(expr, context)
        if scalar.dtype not in (i64, i32):
            raise TypeError(f"{context} must be an i64, i32, or index value in TileLang DSL")

    def _require_tail_remaining_expr(self, expr: SemanticExpr, context: str) -> None:
        if isinstance(expr.type, SemanticIndexType):
            return
        if isinstance(expr.type, SemanticScalarType) and expr.type.dtype.name == "i32":
            return
        raise TypeError(f"{context} must be an i32 or index value in TileLang DSL v1")

    def _require_mask_for_vreg(
        self,
        mask_expr: SemanticExpr,
        vreg_type: SemanticVRegType,
        context: str,
    ) -> None:
        if not isinstance(mask_expr.type, SemanticMaskType):
            raise TypeError(f"{context} requires a mask operand in TileLang DSL v1")
        expected = self._mask_granularity_for_dtype(vreg_type.element_dtype)
        if mask_expr.type.granularity != expected:
            raise TypeError(
                f"{context} requires mask granularity {expected} for vector dtype {vreg_type.element_dtype!r}"
            )

    def _require_mask_for_vsts(
        self,
        mask_expr: SemanticExpr,
        vreg_type: SemanticVRegType,
        dist_expr: SemanticExpr | None,
        context: str,
    ) -> None:
        if not isinstance(mask_expr.type, SemanticMaskType):
            raise TypeError(f"{context} requires a mask operand in TileLang DSL v1")
        expected = self._mask_granularity_for_dtype(vreg_type.element_dtype)
        if dist_expr is not None:
            dist = self._require_string_expr(dist_expr, f"{context} dist")
            if dist == "PK_B16":
                expected = "b16"
            elif dist == "PK_B32":
                expected = "b32"
            elif dist == "PK_B64":
                expected = "b32"
            elif dist == "MRG4CHN_B8":
                expected = "b32"
            elif dist in {"MRG2CHN_B8", "MRG2CHN_B16"}:
                expected = "b16" if dist == "MRG2CHN_B8" else "b32"
        if mask_expr.type.granularity != expected:
            raise TypeError(
                f"{context} requires mask granularity {expected} for store dist "
                f"{self._require_string_expr(dist_expr, f'{context} dist') if dist_expr is not None else 'default'}"
            )

    def _require_matching_vector_pointer(
        self,
        vreg_type: SemanticVRegType,
        pointer_type: SemanticType,
        context: str,
    ) -> None:
        if isinstance(pointer_type, SemanticTileType):
            if pointer_type.element_dtype != vreg_type.element_dtype:
                raise TypeError(f"{context} requires destination Tile dtype to match vector dtype")
            return
        if isinstance(pointer_type, SemanticPtrType):
            if pointer_type.memory_space != "ub":
                raise TypeError(f"{context} requires a UB pointer destination in TileLang DSL")
            if pointer_type.element_dtype != vreg_type.element_dtype:
                raise TypeError(f"{context} requires destination pointer dtype to match vector dtype")
            return
        raise TypeError(f"{context} requires a Tile or pointer destination in TileLang DSL")

    def _normalize_vldsx2_dist(self, expr: SemanticExpr) -> SemanticExpr:
        if (
            isinstance(expr, SemanticSymbolExpr)
            and isinstance(expr.type, SemanticMetaType)
            and expr.type.kind == "deinterleave_dist"
            and isinstance(expr.value, DeinterleaveDist)
        ):
            dist = expr.value.value
        elif (
            isinstance(expr, SemanticBindingRef)
            and isinstance(expr.type, SemanticMetaType)
            and expr.type.kind == "deinterleave_dist"
            and isinstance(expr.binding.value, DeinterleaveDist)
        ):
            dist = expr.binding.value.value
        else:
            dist = self._require_string_expr(expr, "pto.vldsx2 dist")
        legacy_map = {
            "BD": "BDINTLV",
        }
        normalized = legacy_map.get(dist, dist)
        if normalized not in {"DINTLV", "DINTLV_B8", "DINTLV_B16", "DINTLV_B32", "BDINTLV"}:
            raise TypeError(
                "pto.vldsx2 dist must be one of \"DINTLV\", \"DINTLV_B8\", \"DINTLV_B16\", \"DINTLV_B32\", or \"BDINTLV\" in TileLang DSL v1"
            )
        return SemanticLiteralExpr(value=normalized, type=SemanticMetaType(kind="string"))

    def _normalize_vstsx2_dist(self, expr: SemanticExpr) -> SemanticExpr:
        if (
            isinstance(expr, SemanticSymbolExpr)
            and isinstance(expr.type, SemanticMetaType)
            and expr.type.kind == "interleave_dist"
            and isinstance(expr.value, InterleaveDist)
        ):
            dist = expr.value.value
        elif (
            isinstance(expr, SemanticBindingRef)
            and isinstance(expr.type, SemanticMetaType)
            and expr.type.kind == "interleave_dist"
            and isinstance(expr.binding.value, InterleaveDist)
        ):
            dist = expr.binding.value.value
        else:
            dist = self._require_string_expr(expr, "pto.vstsx2 dist")
        legacy_map = {
            "INTLV_B8": "INTLV",
            "INTLV_B16": "INTLV",
            "INTLV_B32": "INTLV",
        }
        normalized = legacy_map.get(dist, dist)
        if normalized != "INTLV":
            raise TypeError("pto.vstsx2 dist must be \"INTLV\" in TileLang DSL v1")
        return SemanticLiteralExpr(value=normalized, type=SemanticMetaType(kind="string"))

    def _mask_granularity_for_dtype(self, dtype: ScalarType) -> str:
        int_bits = integer_bitwidth(dtype)
        if dtype.name == "f32" or int_bits in {32, 64}:
            return "b32"
        if dtype.name in {"f16", "bf16"} or int_bits == 16:
            return "b16"
        if dtype.name in {"f8e4m3", "f8e5m2", "hif8", "f4e1m2x2", "f4e2m1x2"}:
            return "b8"
        if int_bits == 8:
            return "b8"
        raise TypeError(f"dtype `{dtype.name}` is not supported by make_mask/vector lowering in TileLang DSL v1")

    def _vreg_type_for_dtype(self, dtype: ScalarType) -> SemanticVRegType:
        width = bytewidth(dtype)
        if width not in {1, 2, 4, 8}:
            raise TypeError(f"dtype `{dtype.name}` is not supported by vlds/vsts in TileLang DSL v1")
        return SemanticVRegType(element_dtype=dtype, lanes=256 // width)

    def _unpack_result_dtype(self, name: str, dtype: ScalarType) -> ScalarType:
        if not is_integer_dtype(dtype):
            raise TypeError(f"pto.{name} only supports integer vector dtypes in TileLang DSL v1")
        width = integer_bitwidth(dtype)
        if width not in {8, 16, 32}:
            raise TypeError(f"pto.{name} only supports 8/16/32-bit integer vector dtypes in TileLang DSL v1")

        if name == "vzunpack":
            mapping = {
                "i8": ui16,
                "si8": ui16,
                "ui8": ui16,
                "i16": ui32,
                "si16": ui32,
                "ui16": ui32,
                "i32": ui64,
                "si32": ui64,
                "ui32": ui64,
            }
            return mapping[dtype.name]

        mapping = {
            "i8": i16,
            "si8": si16,
            "i16": i32,
            "si16": si32,
            "i32": i64,
            "si32": si64,
        }
        if dtype.name not in mapping:
            raise TypeError(f"pto.{name} requires signed/signless integer vector dtypes in TileLang DSL v1")
        return mapping[dtype.name]

    def _pack_result_dtype(self, dtype: ScalarType) -> ScalarType:
        if not is_integer_dtype(dtype):
            raise TypeError("pto.vpack only supports integer vector dtypes in TileLang DSL v1")
        mapping = {
            "i32": ui16,
            "si32": ui16,
            "ui32": ui16,
            "i16": ui8,
            "si16": ui8,
            "ui16": ui8,
        }
        if dtype.name not in mapping:
            raise TypeError("pto.vpack only supports 32->16 and 16->8 integer packing in TileLang DSL v1")
        return mapping[dtype.name]

    def _validate_unary_dtype(self, name: str, dtype: ScalarType) -> None:
        if name in {"vexp", "vln", "vsqrt", "vrec", "vrsqrt"} and dtype.name not in {"f16", "f32"}:
            raise TypeError(f"pto.{name} only supports f16/f32 in TileLang DSL v1")
        if name == "vrelu" and not (
            dtype.name in {"f16", "f32"}
            or (is_integer_dtype(dtype) and integer_bitwidth(dtype) == 32)
        ):
            raise TypeError("pto.vrelu only supports i32/f16/f32 in TileLang DSL v1")
        if name in {"vnot", "vbcnt", "vcls", "vsunpack", "vzunpack", "vusqz", "vsqz"} and not (
            is_integer_dtype(dtype) and integer_bitwidth(dtype) in {8, 16, 32}
        ):
            raise TypeError(f"pto.{name} only supports integer vector dtypes in TileLang DSL v1")
        if name in {"vabs", "vneg", "vmov", "vtrc", "vcadd", "vcmax", "vcmin"} and not (
            (is_integer_dtype(dtype) and integer_bitwidth(dtype) in {8, 16, 32}) or dtype.name in {"f16", "bf16", "f32"}
        ):
            raise TypeError(f"pto.{name} does not support this dtype in TileLang DSL v1")

    def _validate_binary_dtype(self, name: str, dtype: ScalarType) -> None:
        if name == "vdiv" and not (
            (is_integer_dtype(dtype) and integer_bitwidth(dtype) in {8, 16, 32})
            or dtype.name in {"f16", "f32"}
        ):
            raise TypeError(
                "pto.vdiv only supports 8/16/32-bit integer families and f16/f32 in TileLang DSL v1"
            )
        if name == "vmod" and not (
            is_integer_dtype(dtype) and integer_bitwidth(dtype) in {8, 16, 32}
        ):
            raise TypeError(
                "pto.vmod only supports 8/16/32-bit integer families in TileLang DSL v1"
            )
        if name == "vprelu" and dtype.name not in {"f16", "f32"}:
            raise TypeError("pto.vprelu only supports f16/f32 in TileLang DSL v1")
        if name in {"vaddreluconv", "vmulconv"} and dtype.name not in {"f16", "bf16", "f32"}:
            raise TypeError(f"pto.{name} only supports f16/bf16/f32 in TileLang DSL v1")
        if name in {"vand", "vxor"} and not (
            is_integer_dtype(dtype) and integer_bitwidth(dtype) in {8, 16, 32}
        ):
            raise TypeError(f"pto.{name} only supports integer vector dtypes in TileLang DSL v1")
        if name == "vor" and not (
            (is_integer_dtype(dtype) and integer_bitwidth(dtype) in {8, 16, 32})
            or dtype.name in {"f16", "bf16", "f32"}
        ):
            raise TypeError("pto.vor only supports integer vector dtypes and f16/bf16/f32 in TileLang DSL v1")
        if name in {"vshl", "vshr"} and not (
            is_integer_dtype(dtype) and integer_bitwidth(dtype) in {8, 16, 32}
        ):
            raise TypeError(f"pto.{name} only supports integer vector dtypes in TileLang DSL v1")
        if name == "vmul" and not (
            (is_integer_dtype(dtype) and integer_bitwidth(dtype) in {16, 32}) or dtype.name in {"f16", "f32"}
        ):
            raise TypeError("pto.vmul only supports 16/32-bit integer families and f16/f32 in TileLang DSL v1")
        if name == "vperm" and not (
            (is_integer_dtype(dtype) and integer_bitwidth(dtype) in {8, 16, 32}) or dtype.name in {"f16", "bf16", "f32"}
        ):
            raise TypeError("pto.vperm does not support this data vector dtype in TileLang DSL v1")
        if name in {"vadd", "vsub", "vmax", "vmin", "vaddrelu", "vsubrelu"} and not (
            (is_integer_dtype(dtype) and integer_bitwidth(dtype) in {8, 16, 32}) or dtype.name in {"f16", "bf16", "f32"}
        ):
            raise TypeError(f"pto.{name} does not support this dtype in TileLang DSL v1")
        if name in {"vpack", "vmrgsort"} and not (
            (is_integer_dtype(dtype) and integer_bitwidth(dtype) in {8, 16, 32}) or dtype.name in {"f16", "bf16", "f32"}
        ):
            raise TypeError(f"pto.{name} does not support this dtype in TileLang DSL v1")

    def _validate_vexpdif_dtype(self, dtype: ScalarType) -> None:
        if dtype.name not in {"f16", "f32"}:
            raise TypeError("pto.vexpdif only supports f16/f32 in TileLang DSL v1")

    def _validate_vector_scalar_dtype(self, name: str, dtype: ScalarType) -> None:
        if name == "vdivs" and dtype.name not in {"f16", "f32"}:
            raise TypeError("pto.vdivs only supports f16/f32 in TileLang DSL v1")
        if name == "vlrelu" and dtype.name not in {"f16", "f32"}:
            raise TypeError("pto.vlrelu only supports f16/f32 in TileLang DSL v1")
        if name in {"vshls", "vshrs"} and not (
            is_integer_dtype(dtype) and integer_bitwidth(dtype) in {8, 16, 32}
        ):
            raise TypeError(f"pto.{name} only supports integer vector dtypes in TileLang DSL v1")
        if name in {"vands", "vors", "vxors"} and not (
            is_integer_dtype(dtype) and integer_bitwidth(dtype) in {8, 16, 32}
        ):
            raise TypeError(f"pto.{name} only supports integer vector dtypes in TileLang DSL v1")
        if name in {"vadds", "vsubs", "vmuls", "vmaxs", "vmins"} and not (
            (is_integer_dtype(dtype) and integer_bitwidth(dtype) in {8, 16, 32}) or dtype.name in {"f16", "bf16", "f32"}
        ):
            raise TypeError(f"pto.{name} does not support this dtype in TileLang DSL v1")

    def _validate_vector_immediate_dtype(self, name: str, dtype: ScalarType) -> None:
        if name in {"vshift", "vslide"} and not (
            (is_integer_dtype(dtype) and integer_bitwidth(dtype) in {8, 16, 32}) or dtype.name in {"f16", "bf16", "f32"}
        ):
            raise TypeError(f"pto.{name} does not support this vector dtype in TileLang DSL v1")

    def _validate_ternary_vector_dtype(self, name: str, dtype: ScalarType) -> None:
        if name == "vaxpy" and not (
            (is_integer_dtype(dtype) and integer_bitwidth(dtype) in {16, 32}) or dtype.name in {"f16", "f32"}
        ):
            raise TypeError("pto.vaxpy only supports 16/32-bit integer families and f16/f32 in TileLang DSL v1")
        if name == "vmula" and not (
            (is_integer_dtype(dtype) and integer_bitwidth(dtype) in {16, 32}) or dtype.name in {"f16", "f32"}
        ):
            raise TypeError("pto.vmula only supports 16/32-bit integer families and f16/f32 in TileLang DSL v1")

    def _validate_multi_result_vector_dtype(self, name: str, dtype: ScalarType) -> None:
        if name == "vmull" and not (is_integer_dtype(dtype) and integer_bitwidth(dtype) == 32):
            raise TypeError("pto.vmull only supports 32-bit integer vector families in TileLang DSL v1")

    def _require_sync_pipe(self, expr: SemanticExpr, context: str) -> str:
        if isinstance(expr, SemanticSymbolExpr) and expr.type.kind == "pipe":
            return expr.value.value
        if isinstance(expr, SemanticLiteralExpr) and isinstance(expr.type, SemanticMetaType) and expr.type.kind == "string":
            return expr.value
        raise TypeError(f"{context} must be a PIPE symbol or pipe string in TileLang DSL v1")

    def _require_sync_event(self, expr: SemanticExpr, context: str) -> str:
        if isinstance(expr, SemanticSymbolExpr) and expr.type.kind == "event":
            return expr.value.value
        if isinstance(expr, SemanticLiteralExpr) and isinstance(expr.type, SemanticMetaType) and expr.type.kind == "string":
            return expr.value
        raise TypeError(f"{context} must be an EVENT symbol or event string in TileLang DSL v1")

    def _require_barrier_type(self, expr: SemanticExpr, context: str) -> str:
        if isinstance(expr, SemanticSymbolExpr) and expr.type.kind == "barrier_type":
            return expr.value.value
        if isinstance(expr, SemanticBindingRef) and isinstance(expr.type, SemanticMetaType):
            if expr.type.kind == "barrier_type" and isinstance(expr.binding.value, BarrierType):
                return expr.binding.value.value
        if isinstance(expr, SemanticLiteralExpr) and isinstance(expr.type, SemanticMetaType) and expr.type.kind == "string":
            if expr.value in {barrier_type.value for barrier_type in BarrierType}:
                return expr.value
        raise TypeError(
            f"{context} must be a BarrierType symbol or canonical barrier string "
            "(`VV_ALL`, `VST_VLD`, `VLD_VST`, `VST_VST`, `VS_ALL`, `VST_LD`, "
            "`VLD_ST`, `VST_ST`, `SV_ALL`, `ST_VLD`, `LD_VST`, or `ST_VST`) "
            "in TileLang DSL v1"
        )

    def _normalize_event_id_expr(self, expr: SemanticExpr, context: str) -> SemanticExpr:
        if isinstance(expr, SemanticSymbolExpr) and expr.type.kind == "event" and isinstance(expr.value, Event):
            return SemanticLiteralExpr(
                value=int(expr.value.name[2:]),
                type=SemanticScalarType(dtype=i64),
            )
        if isinstance(expr, SemanticBindingRef) and isinstance(expr.type, SemanticMetaType):
            if expr.type.kind == "event" and isinstance(expr.binding.value, Event):
                return SemanticLiteralExpr(
                    value=int(expr.binding.value.name[2:]),
                    type=SemanticScalarType(dtype=i64),
                )
        self._require_i64_like_expr(expr, context)
        return expr

    def _require_loop_bound_type(self, ty: SemanticType) -> None:
        if isinstance(ty, (SemanticIndexType, SemanticScalarType)):
            return
        raise TypeError(f"loop bound must be scalar/index typed, got {ty!r}")

    def _require_condition_type(self, ty: SemanticType) -> None:
        if isinstance(ty, SemanticIndexType):
            return
        if isinstance(ty, SemanticScalarType):
            return
        raise TypeError(f"if condition must be scalar/index typed, got {ty!r}")

    def _merge_loop_carried_types(
        self,
        outer_type: SemanticType,
        final_type: SemanticType,
    ) -> SemanticType | None:
        if final_type == outer_type:
            return outer_type
        if (
            isinstance(outer_type, SemanticIndexType)
            and isinstance(final_type, SemanticScalarType)
            and final_type.dtype == i32
        ):
            return final_type
        if (
            isinstance(final_type, SemanticIndexType)
            and isinstance(outer_type, SemanticScalarType)
            and outer_type.dtype == i32
        ):
            return outer_type
        return None

    def _require_index_typed_expr(self, expr: SemanticExpr) -> SemanticExpr:
        if isinstance(expr.type, SemanticIndexType):
            return expr
        if isinstance(expr.type, SemanticScalarType) and is_integer_dtype(expr.type.dtype):
            bits = integer_bitwidth(expr.type.dtype)
            if bits in {8, 16, 32, 64}:
                if isinstance(expr, SemanticLiteralExpr) and isinstance(expr.value, int) and not isinstance(expr.value, bool):
                    coerced: SemanticExpr = SemanticLiteralExpr(value=expr.value, type=SemanticIndexType())
                else:
                    coerced = SemanticIndexCastExpr(value=expr, type=SemanticIndexType())
                source_location = self._expr_source_location(expr)
                if source_location is not None:
                    object.__setattr__(coerced, "source_location", source_location)
                return coerced
        self._raise_expr_type_error(
            "slice bounds and vector offsets must be index-typed in TileLang DSL v1",
            expr,
        )

    def _try_static_dtype(self, expr: SemanticExpr) -> ScalarType | None:
        if (
            isinstance(expr, SemanticSymbolExpr)
            and expr.type.kind == "dtype"
            and isinstance(expr.value, ScalarType)
        ):
            return expr.value
        if (
            isinstance(expr, SemanticBindingRef)
            and isinstance(expr.type, SemanticMetaType)
            and expr.type.kind == "dtype"
            and isinstance(expr.binding.value, ScalarType)
        ):
            return expr.binding.value
        return None

    def _try_static_subscript_value(self, expr: SemanticSubscriptAccess) -> Any | None:
        index_value = self._try_static_value(expr.index)
        if not isinstance(index_value, int):
            return None

        base = expr.base
        if isinstance(base, SemanticAttributeAccess) and isinstance(base.base, SemanticBindingRef):
            binding_ref = base.base
            binding_type = binding_ref.type
            if isinstance(binding_type, SemanticTileType):
                if base.attr == "shape" and binding_type.shape is not None:
                    if 0 <= index_value < len(binding_type.shape):
                        return binding_type.shape[index_value]
                if base.attr == "valid_shape" and binding_type.valid_shape is not None:
                    if 0 <= index_value < len(binding_type.valid_shape):
                        return binding_type.valid_shape[index_value]
                return None
            if isinstance(binding_type, (SemanticTensorViewType, SemanticPartitionTensorViewType)):
                return None

        base_value = self._try_static_value(base)
        if isinstance(base_value, (tuple, list)):
            if 0 <= index_value < len(base_value):
                return base_value[index_value]
            return None
        return None

    def _try_static_value(self, expr: SemanticExpr | None) -> Any | None:
        if expr is None:
            return None
        if isinstance(expr, SemanticSymbolExpr):
            return expr.value
        if isinstance(expr, SemanticLiteralExpr):
            return expr.value
        if isinstance(expr, SemanticIndexCastExpr):
            value = self._try_static_value(expr.value)
            if isinstance(value, int) and not isinstance(value, bool):
                return value
            return None
        if isinstance(expr, SemanticBindingRef):
            return expr.binding.value
        if isinstance(expr, SemanticTupleExpr):
            elements = []
            for element in expr.elements:
                static_element = self._try_static_value(element)
                if static_element is None:
                    return None
                elements.append(static_element)
            return tuple(elements)
        if isinstance(expr, SemanticSubscriptAccess):
            return self._try_static_subscript_value(expr)
        if isinstance(expr, SemanticBinaryExpr):
            if expr.op in {"and", "or"}:
                lhs_bool = self._try_static_condition_bool(expr.lhs)
                rhs_bool = self._try_static_condition_bool(expr.rhs)
                if lhs_bool is None or rhs_bool is None:
                    return None
                if expr.op == "and":
                    return lhs_bool and rhs_bool
                return lhs_bool or rhs_bool
            lhs = self._try_static_value(expr.lhs)
            rhs = self._try_static_value(expr.rhs)
            if lhs is None or rhs is None:
                return None
            if expr.op == "add":
                if (
                    isinstance(lhs, (int, float))
                    and isinstance(rhs, (int, float))
                    and not isinstance(lhs, bool)
                    and not isinstance(rhs, bool)
                ):
                    return lhs + rhs
                return None
            if expr.op == "sub":
                if (
                    isinstance(lhs, (int, float))
                    and isinstance(rhs, (int, float))
                    and not isinstance(lhs, bool)
                    and not isinstance(rhs, bool)
                ):
                    return lhs - rhs
                return None
            if expr.op == "mul":
                if (
                    isinstance(lhs, (int, float))
                    and isinstance(rhs, (int, float))
                    and not isinstance(lhs, bool)
                    and not isinstance(rhs, bool)
                ):
                    return lhs * rhs
                return None
            if expr.op == "mod":
                if isinstance(lhs, int) and isinstance(rhs, int):
                    if rhs == 0:
                        return None
                    return lhs % rhs
                return None
            if expr.op == "floordiv":
                if isinstance(lhs, int) and isinstance(rhs, int):
                    if rhs == 0:
                        return None
                    return lhs // rhs
                return None
            if expr.op == "bitand":
                if isinstance(lhs, int) and isinstance(rhs, int):
                    if isinstance(lhs, bool) or isinstance(rhs, bool):
                        return None
                    return lhs & rhs
                return None
            if expr.op == "bitor":
                if isinstance(lhs, int) and isinstance(rhs, int):
                    if isinstance(lhs, bool) or isinstance(rhs, bool):
                        return None
                    return lhs | rhs
                return None
            if expr.op == "bitxor":
                if isinstance(lhs, int) and isinstance(rhs, int):
                    if isinstance(lhs, bool) or isinstance(rhs, bool):
                        return None
                    return lhs ^ rhs
                return None
            if expr.op == "lshift":
                if isinstance(lhs, int) and isinstance(rhs, int):
                    if isinstance(lhs, bool) or isinstance(rhs, bool) or rhs < 0:
                        return None
                    return lhs << rhs
                return None
            if expr.op == "rshift":
                if isinstance(lhs, int) and isinstance(rhs, int):
                    if isinstance(lhs, bool) or isinstance(rhs, bool) or rhs < 0:
                        return None
                    return lhs >> rhs
                return None
            if expr.op == "eq":
                return lhs == rhs
            if expr.op == "ne":
                return lhs != rhs
            if expr.op == "gt":
                try:
                    return lhs > rhs
                except TypeError:
                    return None
            if expr.op == "lt":
                try:
                    return lhs < rhs
                except TypeError:
                    return None
            if expr.op == "ge":
                try:
                    return lhs >= rhs
                except TypeError:
                    return None
            if expr.op == "le":
                try:
                    return lhs <= rhs
                except TypeError:
                    return None
            return None
        if isinstance(expr, SemanticCallExpr):
            if expr.namespace != "pto":
                return None
            if expr.name == "bytewidth":
                if len(expr.args) != 1:
                    return None
                dtype = self._try_static_dtype(expr.args[0])
                if dtype is None:
                    return None
                return bytewidth(dtype)
            if expr.name in {"get_lanes", "elements_per_vreg"}:
                if len(expr.args) != 1:
                    return None
                dtype = self._try_static_dtype(expr.args[0])
                if dtype is None:
                    return None
                return self._vreg_type_for_dtype(dtype).lanes
        return None

    def _try_static_condition_bool(self, expr: SemanticExpr | None) -> bool | None:
        value = self._try_static_value(expr)
        if isinstance(value, bool):
            return value
        if isinstance(value, int):
            return value != 0
        return None

    def _require_constexpr_condition_bool(
        self,
        expr: SemanticExpr,
        *,
        context: str,
    ) -> bool:
        value = self._try_static_condition_bool(expr)
        if value is None:
            raise TypeError(
                f"{context} must be a compile-time bool in TileLang DSL v1"
            )
        return value

    def _static_index_value(self, expr: SemanticExpr | None, *, default: int | None) -> int | None:
        if expr is None:
            return default
        value = self._try_static_value(expr)
        if isinstance(value, int) and not isinstance(value, bool):
            return value
        return None

    def _require_optional_index_typed_expr(self, expr: SemanticExpr | None) -> SemanticExpr | None:
        if expr is None:
            return None
        return self._require_index_typed_expr(expr)

    def _static_bool_value(self, expr: SemanticExpr | None, *, default: bool | None) -> bool | None:
        if expr is None:
            return default
        if isinstance(expr, SemanticLiteralExpr):
            if (
                isinstance(expr.type, SemanticScalarType)
                and expr.type.dtype == i1
                and isinstance(expr.value, bool)
            ):
                return expr.value
            return None
        if (
            isinstance(expr, SemanticBindingRef)
            and isinstance(expr.type, SemanticScalarType)
            and expr.type.dtype == i1
            and isinstance(expr.binding.value, bool)
        ):
            return expr.binding.value
        return None

    def _require_static_bool_value(
        self,
        expr: SemanticExpr | None,
        *,
        context: str,
        default: bool,
    ) -> bool:
        value = self._static_bool_value(expr, default=default)
        if value is None:
            raise TypeError(
                f"{context} must be a compile-time bool in the stable frontend-only DMA profile"
            )
        return value

    def _require_static_non_negative_index_value(
        self,
        expr: SemanticExpr | None,
        *,
        context: str,
        default: int,
    ) -> int:
        value = self._static_index_value(expr, default=default)
        if value is None:
            raise TypeError(
                f"{context} must be a static non-negative index in the stable frontend-only DMA profile"
            )
        if value < 0:
            raise TypeError(
                f"{context} must be a non-negative index in the stable frontend-only DMA profile"
            )
        return value

    def _normalize_optional_index_expr(
        self,
        expr: SemanticExpr | None,
        *,
        default: int,
    ) -> SemanticExpr:
        if expr is not None:
            return expr
        return SemanticLiteralExpr(value=default, type=SemanticIndexType())

    def _normalized_tensor_slice_extent(self, expr: SemanticSliceExpr) -> int | None:
        start = self._static_index_value(expr.start, default=0)
        stop = self._static_index_value(expr.stop, default=None)
        step = self._static_index_value(expr.step, default=1)
        if stop is None or start is None or step is None:
            return None
        if step <= 0:
            raise TypeError("TensorView slicing requires a positive static step in TileLang DSL v1")
        distance = stop - start
        if distance <= 0:
            raise TypeError("TensorView slicing requires positive extents in TileLang DSL v1")
        return (distance + step - 1) // step


def analyze_frontend_kernel(node: FrontendKernelNode) -> SemanticKernel:
    """Normalize descriptor-owned AST into a lowering semantic model."""

    return _SemanticAnalyzer(node).analyze()


__all__ = [
    "SemanticAssignStmt",
    "SemanticAttributeAccess",
    "SemanticBinaryExpr",
    "SemanticBinding",
    "SemanticBindingRef",
    "SemanticCallExpr",
    "SemanticExpr",
    "SemanticExprStmt",
    "SemanticForStmt",
    "SemanticGetBufStmt",
    "SemanticAlignStoreStmt",
    "SemanticAlignType",
    "SemanticIfResult",
    "SemanticIfStmt",
    "SemanticIndexCastExpr",
    "SemanticIndexType",
    "SemanticKernel",
    "SemanticLiteralExpr",
    "SemanticMemBarStmt",
    "SemanticMaskType",
    "SemanticPadValueType",
    "SemanticParameter",
    "SemanticPipeBarrierStmt",
    "SemanticPredicateStoreStmt",
    "SemanticRlsBufStmt",
    "SemanticReturnStmt",
    "SemanticScalarType",
    "SemanticSetCrossCoreStmt",
    "SemanticSetFlagStmt",
    "SemanticSetIntraBlockStmt",
    "SemanticSetIntraCoreStmt",
    "SemanticShapeType",
    "SemanticSliceExpr",
    "SemanticSliceType",
    "SemanticStmt",
    "SemanticVecscopeStmt",
    "SemanticStrictVecscopeStmt",
    "SemanticSubscriptAccess",
    "SemanticSymbolExpr",
    "SemanticTensorSliceAxis",
    "SemanticTensorSliceExpr",
    "SemanticTensorSliceType",
    "SemanticTensorViewType",
    "SemanticPartitionTensorViewType",
    "SemanticTileBinding",
    "SemanticTileConfigType",
    "SemanticTileType",
    "SemanticTupleExpr",
    "SemanticTupleType",
    "SemanticType",
    "SemanticVectorType",
    "SemanticVRegType",
    "SemanticVScatterStmt",
    "SemanticVectorPairStoreStmt",
    "SemanticVectorStoreStmt",
    "SemanticWaitFlagDevStmt",
    "SemanticWaitFlagStmt",
    "SemanticWaitIntraCoreStmt",
    "analyze_frontend_kernel",
]
