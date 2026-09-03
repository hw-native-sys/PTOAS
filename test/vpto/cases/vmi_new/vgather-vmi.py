#!/usr/bin/env python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""PTODSL source-backed cases for the 512-lane vgather VMI kernels.

The MLIR is embedded inline so these cases are self-contained and do not depend
on the old board-test directories.
"""

from pathlib import Path
import sys

import numpy as np


def _bootstrap_dsl_st_common() -> None:
    here = Path(__file__).resolve()
    for candidate in here.parents:
        common_dir = candidate / "test" / "dsl-st"
        if (common_dir / "common.py").exists():
            sys.path.insert(0, str(common_dir))
            return
    raise RuntimeError("Unable to locate test/dsl-st/common.py from vgather-vmi.py")


_bootstrap_dsl_st_common()

from common import auto_main, golden_output_case
from ptodsl import pto


BF16_SRC = r'''// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

module attributes {pto.target_arch = "a5", pto.kernel_kind = #pto.kernel_kind<vector>} {
  func.func @vmi_vgather_bf16_to_bf16_512_kernel(%src_gm: !pto.ptr<bf16, gm>,
                                               %idx_gm: !pto.ptr<ui16, gm>,
                                               %dst_gm: !pto.ptr<bf16, gm>) attributes {pto.kernel} {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c512 = arith.constant 512 : index
    %c0_i64 = arith.constant 0 : i64
    %c1_i64 = arith.constant 1 : i64
    %c512_i64 = arith.constant 512 : i64
    %c1024_i64 = arith.constant 1024 : i64
    %c4096_i64 = arith.constant 4096 : i64
    %c8192_i64 = arith.constant 8192 : i64

    %ub_src = pto.castptr %c0_i64 : i64 -> !pto.ptr<bf16, ub>
    %ub_idx = pto.castptr %c4096_i64 : i64 -> !pto.ptr<ui16, ub>
    %ub_dst = pto.castptr %c8192_i64 : i64 -> !pto.ptr<bf16, ub>

    pto.mte_gm_ub %src_gm, %ub_src, %c0_i64, %c1024_i64
      nburst(%c1_i64, %c1024_i64, %c1024_i64)
      : !pto.ptr<bf16, gm>, !pto.ptr<bf16, ub>, i64, i64, i64, i64, i64
    pto.mte_gm_ub %idx_gm, %ub_idx, %c0_i64, %c1024_i64
      nburst(%c1_i64, %c1024_i64, %c1024_i64)
      : !pto.ptr<ui16, gm>, !pto.ptr<ui16, ub>, i64, i64, i64, i64, i64

    pto.set_flag["PIPE_MTE2", "PIPE_V", "EVENT_ID0"]
    pto.wait_flag["PIPE_MTE2", "PIPE_V", "EVENT_ID0"]

    pto.vecscope {
      %idx = pto.vmi.vload %ub_idx[%c0]
          : !pto.ptr<ui16, ub> -> !pto.vmi.vreg<512xui16>
      %mask = pto.vmi.create_mask %c512 : index -> !pto.vmi.mask<512xpred>
      %out = pto.vmi.vgather %ub_src, %idx, %mask
          : !pto.ptr<bf16, ub>, !pto.vmi.vreg<512xui16>,
            !pto.vmi.mask<512xpred> -> !pto.vmi.vreg<512xbf16>
      pto.vmi.vstore %out, %ub_dst[%c0]
          : !pto.vmi.vreg<512xbf16>, !pto.ptr<bf16, ub>
    }

    pto.set_flag["PIPE_V", "PIPE_MTE3", "EVENT_ID0"]
    pto.wait_flag["PIPE_V", "PIPE_MTE3", "EVENT_ID0"]
    pto.mte_ub_gm %ub_dst, %dst_gm, %c1024_i64
      nburst(%c1_i64, %c1024_i64, %c1024_i64)
      : !pto.ptr<bf16, ub>, !pto.ptr<bf16, gm>, i64, i64, i64, i64
    pto.barrier #pto.pipe<PIPE_ALL>
    return
  }
}
'''
F16_SRC = r'''// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

module attributes {pto.target_arch = "a5", pto.kernel_kind = #pto.kernel_kind<vector>} {
  func.func @vmi_vgather_f16_to_f16_512_kernel(%src_gm: !pto.ptr<f16, gm>,
                                               %idx_gm: !pto.ptr<ui16, gm>,
                                               %dst_gm: !pto.ptr<f16, gm>) attributes {pto.kernel} {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c512 = arith.constant 512 : index
    %c0_i64 = arith.constant 0 : i64
    %c1_i64 = arith.constant 1 : i64
    %c512_i64 = arith.constant 512 : i64
    %c1024_i64 = arith.constant 1024 : i64
    %c4096_i64 = arith.constant 4096 : i64
    %c8192_i64 = arith.constant 8192 : i64

    %ub_src = pto.castptr %c0_i64 : i64 -> !pto.ptr<f16, ub>
    %ub_idx = pto.castptr %c4096_i64 : i64 -> !pto.ptr<ui16, ub>
    %ub_dst = pto.castptr %c8192_i64 : i64 -> !pto.ptr<f16, ub>

    pto.mte_gm_ub %src_gm, %ub_src, %c0_i64, %c1024_i64
      nburst(%c1_i64, %c1024_i64, %c1024_i64)
      : !pto.ptr<f16, gm>, !pto.ptr<f16, ub>, i64, i64, i64, i64, i64
    pto.mte_gm_ub %idx_gm, %ub_idx, %c0_i64, %c1024_i64
      nburst(%c1_i64, %c1024_i64, %c1024_i64)
      : !pto.ptr<ui16, gm>, !pto.ptr<ui16, ub>, i64, i64, i64, i64, i64

    pto.set_flag["PIPE_MTE2", "PIPE_V", "EVENT_ID0"]
    pto.wait_flag["PIPE_MTE2", "PIPE_V", "EVENT_ID0"]

    pto.vecscope {
      %idx = pto.vmi.vload %ub_idx[%c0]
          : !pto.ptr<ui16, ub> -> !pto.vmi.vreg<512xui16>
      %mask = pto.vmi.create_mask %c512 : index -> !pto.vmi.mask<512xpred>
      %out = pto.vmi.vgather %ub_src, %idx, %mask
          : !pto.ptr<f16, ub>, !pto.vmi.vreg<512xui16>,
            !pto.vmi.mask<512xpred> -> !pto.vmi.vreg<512xf16>
      pto.vmi.vstore %out, %ub_dst[%c0]
          : !pto.vmi.vreg<512xf16>, !pto.ptr<f16, ub>
    }

    pto.set_flag["PIPE_V", "PIPE_MTE3", "EVENT_ID0"]
    pto.wait_flag["PIPE_V", "PIPE_MTE3", "EVENT_ID0"]
    pto.mte_ub_gm %ub_dst, %dst_gm, %c1024_i64
      nburst(%c1_i64, %c1024_i64, %c1024_i64)
      : !pto.ptr<f16, ub>, !pto.ptr<f16, gm>, i64, i64, i64, i64
    pto.barrier #pto.pipe<PIPE_ALL>
    return
  }
}
'''
I8_512_SRC = r'''// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

module attributes {pto.target_arch = "a5", pto.kernel_kind = #pto.kernel_kind<vector>} {
  func.func @vmi_vgather_i8_to_i16_512_kernel(%src_gm: !pto.ptr<i8, gm>,
                                               %idx_gm: !pto.ptr<ui16, gm>,
                                               %dst_gm: !pto.ptr<i16, gm>) attributes {pto.kernel} {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c512 = arith.constant 512 : index
    %c0_i64 = arith.constant 0 : i64
    %c1_i64 = arith.constant 1 : i64
    %c512_i64 = arith.constant 512 : i64
    %c1024_i64 = arith.constant 1024 : i64
    %c4096_i64 = arith.constant 4096 : i64
    %c8192_i64 = arith.constant 8192 : i64

    %ub_src = pto.castptr %c0_i64 : i64 -> !pto.ptr<i8, ub>
    %ub_idx = pto.castptr %c4096_i64 : i64 -> !pto.ptr<ui16, ub>
    %ub_dst = pto.castptr %c8192_i64 : i64 -> !pto.ptr<i16, ub>

    pto.mte_gm_ub %src_gm, %ub_src, %c0_i64, %c512_i64
      nburst(%c1_i64, %c512_i64, %c512_i64)
      : !pto.ptr<i8, gm>, !pto.ptr<i8, ub>, i64, i64, i64, i64, i64
    pto.mte_gm_ub %idx_gm, %ub_idx, %c0_i64, %c1024_i64
      nburst(%c1_i64, %c1024_i64, %c1024_i64)
      : !pto.ptr<ui16, gm>, !pto.ptr<ui16, ub>, i64, i64, i64, i64, i64

    pto.set_flag["PIPE_MTE2", "PIPE_V", "EVENT_ID0"]
    pto.wait_flag["PIPE_MTE2", "PIPE_V", "EVENT_ID0"]

    pto.vecscope {
      %idx = pto.vmi.vload %ub_idx[%c0]
          : !pto.ptr<ui16, ub> -> !pto.vmi.vreg<512xui16>
      %mask = pto.vmi.create_mask %c512 : index -> !pto.vmi.mask<512xpred>
      %out = pto.vmi.vgather %ub_src, %idx, %mask
          : !pto.ptr<i8, ub>, !pto.vmi.vreg<512xui16>,
            !pto.vmi.mask<512xpred> -> !pto.vmi.vreg<512xi16>
      pto.vmi.vstore %out, %ub_dst[%c0]
          : !pto.vmi.vreg<512xi16>, !pto.ptr<i16, ub>
    }

    pto.set_flag["PIPE_V", "PIPE_MTE3", "EVENT_ID0"]
    pto.wait_flag["PIPE_V", "PIPE_MTE3", "EVENT_ID0"]
    pto.mte_ub_gm %ub_dst, %dst_gm, %c1024_i64
      nburst(%c1_i64, %c1024_i64, %c1024_i64)
      : !pto.ptr<i16, ub>, !pto.ptr<i16, gm>, i64, i64, i64, i64
    pto.barrier #pto.pipe<PIPE_ALL>
    return
  }
}
'''
U8_512_SRC = r'''// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

module attributes {pto.target_arch = "a5", pto.kernel_kind = #pto.kernel_kind<vector>} {
  func.func @vmi_vgather_u8_to_u16_512_kernel(%src_gm: !pto.ptr<ui8, gm>,
                                               %idx_gm: !pto.ptr<ui16, gm>,
                                               %dst_gm: !pto.ptr<ui16, gm>) attributes {pto.kernel} {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c512 = arith.constant 512 : index
    %c0_i64 = arith.constant 0 : i64
    %c1_i64 = arith.constant 1 : i64
    %c512_i64 = arith.constant 512 : i64
    %c1024_i64 = arith.constant 1024 : i64
    %c4096_i64 = arith.constant 4096 : i64
    %c8192_i64 = arith.constant 8192 : i64

    %ub_src = pto.castptr %c0_i64 : i64 -> !pto.ptr<ui8, ub>
    %ub_idx = pto.castptr %c4096_i64 : i64 -> !pto.ptr<ui16, ub>
    %ub_dst = pto.castptr %c8192_i64 : i64 -> !pto.ptr<ui16, ub>

    pto.mte_gm_ub %src_gm, %ub_src, %c0_i64, %c512_i64
      nburst(%c1_i64, %c512_i64, %c512_i64)
      : !pto.ptr<ui8, gm>, !pto.ptr<ui8, ub>, i64, i64, i64, i64, i64
    pto.mte_gm_ub %idx_gm, %ub_idx, %c0_i64, %c1024_i64
      nburst(%c1_i64, %c1024_i64, %c1024_i64)
      : !pto.ptr<ui16, gm>, !pto.ptr<ui16, ub>, i64, i64, i64, i64, i64

    pto.set_flag["PIPE_MTE2", "PIPE_V", "EVENT_ID0"]
    pto.wait_flag["PIPE_MTE2", "PIPE_V", "EVENT_ID0"]

    pto.vecscope {
      %idx = pto.vmi.vload %ub_idx[%c0]
          : !pto.ptr<ui16, ub> -> !pto.vmi.vreg<512xui16>
      %mask = pto.vmi.create_mask %c512 : index -> !pto.vmi.mask<512xpred>
      %out = pto.vmi.vgather %ub_src, %idx, %mask
          : !pto.ptr<ui8, ub>, !pto.vmi.vreg<512xui16>,
            !pto.vmi.mask<512xpred> -> !pto.vmi.vreg<512xui16>
      pto.vmi.vstore %out, %ub_dst[%c0]
          : !pto.vmi.vreg<512xui16>, !pto.ptr<ui16, ub>
    }

    pto.set_flag["PIPE_V", "PIPE_MTE3", "EVENT_ID0"]
    pto.wait_flag["PIPE_V", "PIPE_MTE3", "EVENT_ID0"]
    pto.mte_ub_gm %ub_dst, %dst_gm, %c1024_i64
      nburst(%c1_i64, %c1024_i64, %c1024_i64)
      : !pto.ptr<ui16, ub>, !pto.ptr<ui16, gm>, i64, i64, i64, i64
    pto.barrier #pto.pipe<PIPE_ALL>
    return
  }
}
'''
UI16_512_SRC = r'''// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

module attributes {pto.target_arch = "a5", pto.kernel_kind = #pto.kernel_kind<vector>} {
  func.func @vmi_vgather_ui16_to_ui16_512_kernel(%src_gm: !pto.ptr<ui16, gm>,
                                               %idx_gm: !pto.ptr<ui16, gm>,
                                               %dst_gm: !pto.ptr<ui16, gm>) attributes {pto.kernel} {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c512 = arith.constant 512 : index
    %c0_i64 = arith.constant 0 : i64
    %c1_i64 = arith.constant 1 : i64
    %c512_i64 = arith.constant 512 : i64
    %c1024_i64 = arith.constant 1024 : i64
    %c4096_i64 = arith.constant 4096 : i64
    %c8192_i64 = arith.constant 8192 : i64

    %ub_src = pto.castptr %c0_i64 : i64 -> !pto.ptr<ui16, ub>
    %ub_idx = pto.castptr %c4096_i64 : i64 -> !pto.ptr<ui16, ub>
    %ub_dst = pto.castptr %c8192_i64 : i64 -> !pto.ptr<ui16, ub>

    pto.mte_gm_ub %src_gm, %ub_src, %c0_i64, %c1024_i64
      nburst(%c1_i64, %c1024_i64, %c1024_i64)
      : !pto.ptr<ui16, gm>, !pto.ptr<ui16, ub>, i64, i64, i64, i64, i64
    pto.mte_gm_ub %idx_gm, %ub_idx, %c0_i64, %c1024_i64
      nburst(%c1_i64, %c1024_i64, %c1024_i64)
      : !pto.ptr<ui16, gm>, !pto.ptr<ui16, ub>, i64, i64, i64, i64, i64

    pto.set_flag["PIPE_MTE2", "PIPE_V", "EVENT_ID0"]
    pto.wait_flag["PIPE_MTE2", "PIPE_V", "EVENT_ID0"]

    pto.vecscope {
      %idx = pto.vmi.vload %ub_idx[%c0]
          : !pto.ptr<ui16, ub> -> !pto.vmi.vreg<512xui16>
      %mask = pto.vmi.create_mask %c512 : index -> !pto.vmi.mask<512xpred>
      %out = pto.vmi.vgather %ub_src, %idx, %mask
          : !pto.ptr<ui16, ub>, !pto.vmi.vreg<512xui16>,
            !pto.vmi.mask<512xpred> -> !pto.vmi.vreg<512xui16>
      pto.vmi.vstore %out, %ub_dst[%c0]
          : !pto.vmi.vreg<512xui16>, !pto.ptr<ui16, ub>
    }

    pto.set_flag["PIPE_V", "PIPE_MTE3", "EVENT_ID0"]
    pto.wait_flag["PIPE_V", "PIPE_MTE3", "EVENT_ID0"]
    pto.mte_ub_gm %ub_dst, %dst_gm, %c1024_i64
      nburst(%c1_i64, %c1024_i64, %c1024_i64)
      : !pto.ptr<ui16, ub>, !pto.ptr<ui16, gm>, i64, i64, i64, i64
    pto.barrier #pto.pipe<PIPE_ALL>
    return
  }
}
'''
@pto.jit(
    name="vmi_vgather_bf16_to_bf16_512_kernel",
    target="a5",
    backend="vpto",
    mode="explicit",
    source=BF16_SRC,
)
def vmi_vgather_bf16_to_bf16_512_kernel(
    src_gm: pto.ptr(pto.bf16, "gm"),
    idx_gm: pto.ptr(pto.ui16, "gm"),
    dst_gm: pto.ptr(pto.bf16, "gm"),
):
    pass


@pto.jit(
    name="vmi_vgather_f16_to_f16_512_kernel",
    target="a5",
    backend="vpto",
    mode="explicit",
    source=F16_SRC,
)
def vmi_vgather_f16_to_f16_512_kernel(
    src_gm: pto.ptr(pto.f16, "gm"),
    idx_gm: pto.ptr(pto.ui16, "gm"),
    dst_gm: pto.ptr(pto.f16, "gm"),
):
    pass


@pto.jit(
    name="vmi_vgather_i8_to_i16_512_kernel",
    target="a5",
    backend="vpto",
    mode="explicit",
    source=I8_512_SRC,
)
def vmi_vgather_i8_to_i16_512_kernel(
    src_gm: pto.ptr(pto.i8, "gm"),
    idx_gm: pto.ptr(pto.ui16, "gm"),
    dst_gm: pto.ptr(pto.i16, "gm"),
):
    pass


@pto.jit(
    name="vmi_vgather_u8_to_u16_512_kernel",
    target="a5",
    backend="vpto",
    mode="explicit",
    source=U8_512_SRC,
)
def vmi_vgather_u8_to_u16_512_kernel(
    src_gm: pto.ptr(pto.ui8, "gm"),
    idx_gm: pto.ptr(pto.ui16, "gm"),
    dst_gm: pto.ptr(pto.ui16, "gm"),
):
    pass


@pto.jit(
    name="vmi_vgather_ui16_to_ui16_512_kernel",
    target="a5",
    backend="vpto",
    mode="explicit",
    source=UI16_512_SRC,
)
def vmi_vgather_ui16_to_ui16_512_kernel(
    src_gm: pto.ptr(pto.ui16, "gm"),
    idx_gm: pto.ptr(pto.ui16, "gm"),
    dst_gm: pto.ptr(pto.ui16, "gm"),
):
    pass


def _non_identity_indices(elems: int) -> np.ndarray:
    # Non-identity deterministic gather (idx[i] != i for most lanes).
    return ((np.arange(elems, dtype=np.uint16) * 7 + 13) % elems).astype(np.uint16)


def _bf16_512_inputs():
    elems = 512
    src = np.arange(elems, dtype=np.float16)
    return [src, _non_identity_indices(elems)]


def _bf16_512_expected(src, idx):
    return src[idx].astype(np.float16)


def _f16_512_inputs():
    elems = 512
    src = np.arange(elems, dtype=np.float16)
    return [src, _non_identity_indices(elems)]


def _f16_512_expected(src, idx):
    return src[idx].astype(np.float16)


def _i8_512_inputs():
    elems = 512
    src = np.arange(elems, dtype=np.uint8)
    return [src, _non_identity_indices(elems)]


def _i8_512_expected(src, idx):
    return src.astype(np.uint16)[idx].astype(np.uint16)


def _u8_512_inputs():
    elems = 512
    src = np.arange(elems, dtype=np.uint8)
    return [src, _non_identity_indices(elems)]


def _u8_512_expected(src, idx):
    return src.astype(np.uint16)[idx].astype(np.uint16)


def _ui16_512_inputs():
    elems = 512
    src = np.arange(elems, dtype=np.uint16)
    return [src, _non_identity_indices(elems)]


def _ui16_512_expected(src, idx):
    return src[idx].astype(np.uint16)


CASES = [
    golden_output_case(
        "vmi_vgather_bf16_to_bf16_512",
        vmi_vgather_bf16_to_bf16_512_kernel,
        inputs=_bf16_512_inputs,
        expected=_bf16_512_expected,
        output_shape=(512,),
        output_dtype=np.float16,
        rtol=0.0,
        atol=0.0,
    ),
    golden_output_case(
        "vmi_vgather_f16_to_f16_512",
        vmi_vgather_f16_to_f16_512_kernel,
        inputs=_f16_512_inputs,
        expected=_f16_512_expected,
        output_shape=(512,),
        output_dtype=np.float16,
        rtol=0.0,
        atol=0.0,
    ),
    golden_output_case(
        "vmi_vgather_i8_to_i16_512",
        vmi_vgather_i8_to_i16_512_kernel,
        inputs=_i8_512_inputs,
        expected=_i8_512_expected,
        output_shape=(512,),
        output_dtype=np.uint16,
        rtol=0.0,
        atol=0.0,
    ),
    golden_output_case(
        "vmi_vgather_u8_to_u16_512",
        vmi_vgather_u8_to_u16_512_kernel,
        inputs=_u8_512_inputs,
        expected=_u8_512_expected,
        output_shape=(512,),
        output_dtype=np.uint16,
        rtol=0.0,
        atol=0.0,
    ),
    golden_output_case(
        "vmi_vgather_ui16_to_ui16_512",
        vmi_vgather_ui16_to_ui16_512_kernel,
        inputs=_ui16_512_inputs,
        expected=_ui16_512_expected,
        output_shape=(512,),
        output_dtype=np.uint16,
        rtol=0.0,
        atol=0.0,
    ),
]


KERNELS = [
    vmi_vgather_bf16_to_bf16_512_kernel,
    vmi_vgather_f16_to_f16_512_kernel,
    vmi_vgather_i8_to_i16_512_kernel,
    vmi_vgather_u8_to_u16_512_kernel,
    vmi_vgather_ui16_to_ui16_512_kernel,
]


auto_main(globals())
