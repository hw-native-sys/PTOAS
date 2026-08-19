# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""Compatibility import for A5 VMI TileLib candidates.

The implementations live next to their ordinary A5 TileLib templates, one file
per TileOp.  Importing this module registers all built-in A5 VMI candidates and
keeps the legacy ``ptodsl.vmi_tilelib`` provider path stable.
"""

from .tilelib.templates.a5._vmi_common import (
    VMI_TILELIB_REGISTRY,
    canonical_vmi_template,
    emit_elementwise_vmi,
)
from .tilelib.templates.a5.tadd import vmi_tadd_block64
from .tilelib.templates.a5.tadds import vmi_tadds
from .tilelib.templates.a5.tcvt import vmi_tcvt
from .tilelib.templates.a5.tcolmax import vmi_tcolmax
from .tilelib.templates.a5.tcolmin import vmi_tcolmin
from .tilelib.templates.a5.tcolsum import vmi_tcolsum
from .tilelib.templates.a5.tcolexpand import vmi_tcolexpand
from .tilelib.templates.a5.tcolexpandadd import vmi_tcolexpandadd
from .tilelib.templates.a5.tcolexpanddiv import vmi_tcolexpanddiv
from .tilelib.templates.a5.tcolexpandmul import vmi_tcolexpandmul
from .tilelib.templates.a5.tcolexpandsub import vmi_tcolexpandsub
from .tilelib.templates.a5.tdiv import vmi_tdiv
from .tilelib.templates.a5.tdivs import vmi_tdivs, vmi_tdivs_scalar_tile
from .tilelib.templates.a5.texp import vmi_texp_block64
from .tilelib.templates.a5.texpand import (
    vmi_texpands,
    vmi_texpands_bf16,
    vmi_texpands_f16,
    vmi_texpands_i32,
)
from .tilelib.templates.a5.tabs import vmi_tabs
from .tilelib.templates.a5.tmax import vmi_tmax
from .tilelib.templates.a5.tmaxs import vmi_tmaxs
from .tilelib.templates.a5.tmins import vmi_tmins
from .tilelib.templates.a5.tmov import vmi_tmov
from .tilelib.templates.a5.tmul import vmi_tmul
from .tilelib.templates.a5.tmuls import vmi_tmuls
from .tilelib.templates.a5.tneg import vmi_tneg
from .tilelib.templates.a5.trecip import vmi_trecip
from .tilelib.templates.a5.trsqrt import vmi_trsqrt, vmi_trsqrt_with_tmp
from .tilelib.templates.a5.trowexpandsub import vmi_trowexpandsub
from .tilelib.templates.a5.trowexpanddiv import vmi_trowexpanddiv
from .tilelib.templates.a5.trowexpandmul import vmi_trowexpandmul
from .tilelib.templates.a5.trowmax import vmi_trowmax, vmi_trowmax_row
from .tilelib.templates.a5.trowsum import vmi_trowsum, vmi_trowsum_row
from .tilelib.templates.a5.tsqrt import vmi_tsqrt
from .tilelib.templates.a5.tsub import vmi_tsub
from .tilelib.templates.a5.tsubs import vmi_tsubs


__all__ = [
    "VMI_TILELIB_REGISTRY",
    "canonical_vmi_template",
    "emit_elementwise_vmi",
    "vmi_tadd_block64",
    "vmi_tadds",
    "vmi_tcvt",
    "vmi_tcolmax",
    "vmi_tcolmin",
    "vmi_tcolsum",
    "vmi_tcolexpand",
    "vmi_tcolexpandadd",
    "vmi_tcolexpanddiv",
    "vmi_tcolexpandmul",
    "vmi_tcolexpandsub",
    "vmi_tdiv",
    "vmi_tdivs",
    "vmi_tdivs_scalar_tile",
    "vmi_texp_block64",
    "vmi_texpands",
    "vmi_texpands_bf16",
    "vmi_texpands_f16",
    "vmi_texpands_i32",
    "vmi_tabs",
    "vmi_tmax",
    "vmi_tmaxs",
    "vmi_tmins",
    "vmi_tmov",
    "vmi_tmul",
    "vmi_tmuls",
    "vmi_tneg",
    "vmi_trecip",
    "vmi_trsqrt",
    "vmi_trsqrt_with_tmp",
    "vmi_trowexpandsub",
    "vmi_trowexpanddiv",
    "vmi_trowexpandmul",
    "vmi_trowmax",
    "vmi_trowmax_row",
    "vmi_trowsum",
    "vmi_trowsum_row",
    "vmi_tsqrt",
    "vmi_tsub",
    "vmi_tsubs",
]
