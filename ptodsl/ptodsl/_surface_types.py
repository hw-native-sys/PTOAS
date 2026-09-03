# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""Public PTODSL surface markers and enums."""

from enum import Enum

from ptoas.mlir.dialects import pto as _pto


class _ConstExprHelper:
    """Callable marker for PTODSL compile-time specialization parameters and branches."""

    def __repr__(self):
        return "pto.const_expr"

    def __call__(self, value):
        return bool(value)


const_expr = _ConstExprHelper()


class MemorySpace:
    """Public PTODSL memory-space enum aliases."""

    GM = _pto.AddressSpace.GM
    UB = _pto.AddressSpace.VEC
    VEC = _pto.AddressSpace.VEC
    MAT = _pto.AddressSpace.MAT
    LEFT = _pto.AddressSpace.LEFT
    RIGHT = _pto.AddressSpace.RIGHT
    ACC = _pto.AddressSpace.ACC
    BIAS = _pto.AddressSpace.BIAS
    SCALING = _pto.AddressSpace.SCALING


class BLayout(str, Enum):
    """Public tile block-layout aliases used by TileLib constraints."""

    ROW_MAJOR = "row_major"
    COL_MAJOR = "col_major"


class SLayout(str, Enum):
    """Public tile storage-layout aliases used by TileLib constraints."""

    NONE_BOX = "none_box"
    ROW_MAJOR = "row_major"
    COL_MAJOR = "col_major"


class BarrierType:
    """Public PTODSL memory-barrier kind aliases."""

    VV_ALL = "VV_ALL"
    VST_VLD = "VST_VLD"
    VLD_VST = "VLD_VST"
    VST_VST = "VST_VST"
    VS_ALL = "VS_ALL"
    VST_LD = "VST_LD"
    VLD_ST = "VLD_ST"
    VST_ST = "VST_ST"
    SV_ALL = "SV_ALL"
    ST_VLD = "ST_VLD"
    LD_VST = "LD_VST"
    ST_VST = "ST_VST"
    SS_ALL = "SS_ALL"
    ST_LD = "ST_LD"
    LD_ST = "LD_ST"
    ST_ST = "ST_ST"


class Pipe:
    """Public PTODSL pipeline aliases for pipeline-level sync ops."""

    S = _pto.PIPE.PIPE_S
    V = _pto.PIPE.PIPE_V
    M = _pto.PIPE.PIPE_M
    MTE1 = _pto.PIPE.PIPE_MTE1
    MTE2 = _pto.PIPE.PIPE_MTE2
    MTE3 = _pto.PIPE.PIPE_MTE3
    MTE4 = _pto.PIPE.PIPE_MTE4
    MTE5 = _pto.PIPE.PIPE_MTE5
    V2 = _pto.PIPE.PIPE_V2
    FIX = _pto.PIPE.PIPE_FIX
    ALL = _pto.PIPE.PIPE_ALL


class MaskPattern:
    """Public PTODSL mask-pattern tokens."""

    ALL = "PAT_ALL"
    ALLF = "PAT_ALLF"
    H = "PAT_H"
    Q = "PAT_Q"
    M3 = "PAT_M3"
    M4 = "PAT_M4"


for _vl in range(1, 129):
    setattr(MaskPattern, f"VL{_vl}", f"PAT_VL{_vl}")


class CmpMode:
    """Public PTODSL compare-mode tokens."""

    EQ = "eq"
    NE = "ne"
    LT = "lt"
    LE = "le"
    GT = "gt"
    GE = "ge"


class PredicatePart:
    """Public PTODSL predicate pack/unpack part tokens."""

    LOWER = "LOWER"
    HIGHER = "HIGHER"


class PredicateDist:
    """Public PTODSL predicate load/store distribution tokens."""

    NORM = "NORM"
    US = "US"
    DS = "DS"
    PK = "PK"


class VStoreDist:
    """Public PTODSL vector-store distribution tokens."""

    NORM_B8 = "NORM_B8"
    NORM_B16 = "NORM_B16"
    NORM_B32 = "NORM_B32"
    _1PT_B8 = "1PT_B8"
    _1PT_B16 = "1PT_B16"
    _1PT_B32 = "1PT_B32"
    PK_B16 = "PK_B16"
    PK_B32 = "PK_B32"
    PK_B64 = "PK_B64"
    PK4_B32 = "PK4_B32"
    MRG4CHN_B8 = "MRG4CHN_B8"
    MRG2CHN_B8 = "MRG2CHN_B8"
    MRG2CHN_B16 = "MRG2CHN_B16"


setattr(VStoreDist, "1PT_B8", VStoreDist._1PT_B8)
setattr(VStoreDist, "1PT_B16", VStoreDist._1PT_B16)
setattr(VStoreDist, "1PT_B32", VStoreDist._1PT_B32)


class DeinterleaveDist:
    """Public PTODSL dual-load distribution tokens."""

    DINTLV_B8 = "DINTLV_B8"
    DINTLV_B16 = "DINTLV_B16"
    DINTLV_B32 = "DINTLV_B32"
    BDINTLV = "BDINTLV"


class InterleaveDist:
    """Public PTODSL dual-store distribution tokens."""

    INTLV_B8 = "INTLV_B8"
    INTLV_B16 = "INTLV_B16"
    INTLV_B32 = "INTLV_B32"


class PostUpdate:
    """Public PTODSL post-update mode tokens for stateful stores."""

    OFF = "NO_POST_UPDATE"
    ON = "POST_UPDATE"


class FractalMode(str, Enum):
    """Cube fractal-layout conversion modes."""

    ND2NZ = "nd2nz"
    DN2NZ = "dn2nz"
    NZ2ND = "nz2nd"
    NZ2DN = "nz2dn"
    NZ2NZ = "nz2nz"


class AccStoreUnitFlagCtrl(str, Enum):
    """Accumulator-store unit-flag modes."""

    CHECK_ONLY = "check_only"
    CHECK_AND_CLEAR = "check_and_clear"


class MadUnitFlagMode(str, Enum):
    """MAD producer unit-flag modes."""

    CHECK_ONLY = "check_only"
    CHECK_AND_SET = "check_and_set"


class SatMode(str, Enum):
    """Saturation policy values used by cube compute/store surfaces."""

    ON = "on"
    OFF = "off"
    PRESERVE_NAN = "preserve_nan"


class Tf32Mode(str, Enum):
    """TF32 rounding mode for f32 cube MAD operations."""

    ROUND_EVEN = "round_even"
    ROUND_AWAY = "round_away"


class SplitMode(str, Enum):
    """Dual-destination split axis for accumulator stores to UB."""

    M = "M"
    N = "N"

class PartMode:
    """Public PTODSL EVEN/ODD part selectors."""

    EVEN = "EVEN"
    ODD = "ODD"


class PositionMode:
    """Public PTODSL source-lane selectors for vdup vector input."""

    LOWEST = "LOWEST"
    HIGHEST = "HIGHEST"


class VPackPart:
    """Public PTODSL pack-half selectors."""

    LOWER = "LOWER"
    HIGHER = "HIGHER"


class VcvtRoundMode:
    """Public PTODSL vcvt/vtrc rounding-mode tokens."""

    R = "R"
    A = "A"
    F = "F"
    C = "C"
    Z = "Z"
    O = "O"
    H = "H"


class VcvtSatMode:
    """Public PTODSL vcvt saturation-mode tokens."""

    SAT = "SAT"
    NOSAT = "NOSAT"
    RS_ENABLE = "SAT"
    RS_DISABLE = "NOSAT"


class VcvtPartMode:
    """Public PTODSL vcvt part selectors."""

    EVEN = "EVEN"
    ODD = "ODD"
    P0 = "P0"
    P1 = "P1"
    P2 = "P2"
    P3 = "P3"


AlignType = _pto.AlignType
RoundMode = _pto.RoundMode
# Public tile precision modes share one enum.  The generated dialect keeps
# operation-specific attribute classes internally, so the wrappers translate
# this enum to the attribute type required by each operation.
Precision = _pto.DivPrecision
DivPrecision = Precision
RemPrecision = Precision
ExpPrecision = Precision
LogPrecision = Precision
RecipPrecision = Precision
RsqrtPrecision = Precision
SqrtPrecision = Precision


class TensorView:
    """Authoring-time marker for a tensor-view descriptor value."""


class PartitionTensorView:
    """Authoring-time marker for a partitioned tensor-view descriptor value."""


class Tile:
    """Authoring-time marker for an on-chip tile value."""


__all__ = [
    "const_expr",
    "MemorySpace",
    "BLayout",
    "SLayout",
    "BarrierType",
    "Pipe",
    "MaskPattern",
    "CmpMode",
    "PredicatePart",
    "PredicateDist",
    "VStoreDist",
    "DeinterleaveDist",
    "InterleaveDist",
    "PostUpdate",
    "FractalMode",
    "AccStoreUnitFlagCtrl",
    "MadUnitFlagMode",
    "SatMode",
    "Tf32Mode",
    "SplitMode",
    "PartMode",
    "PositionMode",
    "VPackPart",
    "VcvtRoundMode",
    "VcvtSatMode",
    "VcvtPartMode",
    "AlignType",
    "RoundMode",
    "Precision",
    "DivPrecision",
    "RemPrecision",
    "ExpPrecision",
    "LogPrecision",
    "RecipPrecision",
    "RsqrtPrecision",
    "SqrtPrecision",
    "TensorView",
    "PartitionTensorView",
    "Tile",
]
