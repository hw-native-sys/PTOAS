# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""TileLang DSL template for pto.tmov - Mat to Scaling/FB buffer movement.

This template implements the TMOV_M2S scenario for cube kernels:
  - Source: L1 Mat buffer (memory_space="mat", 1xN row-major layout)
  - Destination: L0 Fixpipe Buffer (memory_space="scaling")
  - Uses mte_l1_fb (fixpipe buffer load) intrinsic operation

The Fixpipe Buffer (FB) is a 4KB buffer used for storing quantization
scale parameters in the fixpipe quantization flow. Requirements:
  - Row dimension must be 1
  - Column dimension * sizeof(dtype) must be aligned to 128 bits (16 bytes)
  - Total size must not exceed 4KB (4096 bytes)

Constraint: This template is selected when dst.memory_space == SCALING.

Data format:
  - The source and destination tile element types match.
  - len_burst is expressed in 64-byte copy_cbuf_to_fbuf burst units.
"""

import tilelang_dsl as pto


def _tmov_m2s_constraint(src: pto.Tile, dst: pto.Tile) -> bool:
    """Constraint: Mat to Scaling transfer scenario.

    Supported scenario:
      - src.memory_space == MAT
      - dst.memory_space == SCALING
    """
    src_ms = src.memory_space
    dst_ms = dst.memory_space

    # Check src is MAT
    if isinstance(src_ms, str):
        src_is_mat = src_ms == "mat"
    elif isinstance(src_ms, pto.MemorySpace):
        src_is_mat = src_ms == pto.MemorySpace.MAT
    else:
        src_is_mat = hasattr(src_ms, "value") and src_ms.value == "mat"

    # Check dst is SCALING
    if isinstance(dst_ms, str):
        dst_is_scaling = dst_ms == "scaling"
    elif isinstance(dst_ms, pto.MemorySpace):
        dst_is_scaling = dst_ms == pto.MemorySpace.SCALING
    else:
        dst_is_scaling = hasattr(dst_ms, "value") and dst_ms.value == "scaling"

    return src_is_mat and dst_is_scaling


@pto.ckernel(
    target="a5",
    op="pto.tmov",
    constraints=[_tmov_m2s_constraint],
    dtypes=[
        (pto.f16, pto.f16),
        (pto.bf16, pto.bf16),
        (pto.f32, pto.f32),
        (pto.ui64, pto.ui64),
    ],
)
def template_tmov_m2s(src: pto.Tile, dst: pto.Tile):
    """Move data from Mat buffer to Fixpipe Buffer (Scaling).

    Args:
        src: Source tile in L1 Mat location (1xN row-major)
        dst: Destination tile in Scaling/FB location

    The scale parameters are loaded into FB for fixpipe quantization.

    mte_l1_fb semantics:
      - len_burst = rows * cols * sizeof(dtype) / 64
      - Each row contains one set of scale parameters for all columns

    nburst = (n_burst, src_gap, dst_gap) - single burst with no gaps
    """
    # copy_cbuf_to_fbuf uses 64-byte burst units.
    m, n = src.valid_shape
    len_burst = (m * n * pto.bytewidth(src.element_type)) // 64
    n_burst = 1
    src_gap = 0
    dst_gap = 0
    pto.mte_l1_fb(src.as_ptr(), dst.as_ptr(), len_burst, nburst=(n_burst, src_gap, dst_gap))
    return
