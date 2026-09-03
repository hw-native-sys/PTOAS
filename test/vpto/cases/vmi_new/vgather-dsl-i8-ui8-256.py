#!/usr/bin/env python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""PTODSL Python DSL cases for the 256-lane vgather 8-bit promotion paths.

These exercise ``pto.vmi.vgather`` directly and cover the i8 -> i16 and
ui8 -> ui16 result-type inference added with the vgather board-test commit.
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
    raise RuntimeError("Unable to locate test/dsl-st/common.py from vgather-dsl-i8-ui8-256.py")


_bootstrap_dsl_st_common()

from common import auto_main, golden_output_case
from ptodsl import pto


ELEMS = 256
SRC_BYTES = 256
IDX_BYTES = 512
DST_BYTES = 512


@pto.jit(
    name="vmi_vgather_i8_to_i16_256_kernel",
    target="a5",
    backend="vpto",
    mode="explicit",
    kernel_kind="vector",
    insert_sync=False,
)
def vmi_vgather_i8_to_i16_256_kernel(
    src_gm: pto.ptr(pto.i8, "gm"),
    idx_gm: pto.ptr(pto.ui16, "gm"),
    dst_gm: pto.ptr(pto.i16, "gm"),
):
    ub_src = pto.castptr(pto.const(0, dtype=pto.i64), pto.ptr(pto.i8, "ub"))
    ub_idx = pto.castptr(pto.const(4096, dtype=pto.i64), pto.ptr(pto.ui16, "ub"))
    ub_dst = pto.castptr(pto.const(8192, dtype=pto.i64), pto.ptr(pto.i16, "ub"))

    pto.mte_gm_ub(src_gm, ub_src, 0, SRC_BYTES, nburst=(1, SRC_BYTES, SRC_BYTES))
    pto.mte_gm_ub(idx_gm, ub_idx, 0, IDX_BYTES, nburst=(1, IDX_BYTES, IDX_BYTES))

    pto.set_flag(pto.Pipe.MTE2, pto.Pipe.V, event_id=0)
    pto.wait_flag(pto.Pipe.MTE2, pto.Pipe.V, event_id=0)

    offset = pto.const(0, dtype=pto.index)
    idx = pto.vmi.vload(ub_idx, offset, size=ELEMS)
    mask = pto.vmi.create_mask(pto.const(ELEMS, dtype=pto.index), size=ELEMS)
    out = pto.vmi.vgather(ub_src, idx, mask)
    pto.vmi.vstore(out, ub_dst, offset, mask)

    pto.set_flag(pto.Pipe.V, pto.Pipe.MTE3, event_id=0)
    pto.wait_flag(pto.Pipe.V, pto.Pipe.MTE3, event_id=0)
    pto.mte_ub_gm(ub_dst, dst_gm, DST_BYTES, nburst=(1, DST_BYTES, DST_BYTES))
    pto.pipe_barrier(pto.Pipe.ALL)


@pto.jit(
    name="vmi_vgather_u8_to_u16_256_kernel",
    target="a5",
    backend="vpto",
    mode="explicit",
    kernel_kind="vector",
    insert_sync=False,
)
def vmi_vgather_u8_to_u16_256_kernel(
    src_gm: pto.ptr(pto.ui8, "gm"),
    idx_gm: pto.ptr(pto.ui16, "gm"),
    dst_gm: pto.ptr(pto.ui16, "gm"),
):
    ub_src = pto.castptr(pto.const(0, dtype=pto.i64), pto.ptr(pto.ui8, "ub"))
    ub_idx = pto.castptr(pto.const(4096, dtype=pto.i64), pto.ptr(pto.ui16, "ub"))
    ub_dst = pto.castptr(pto.const(8192, dtype=pto.i64), pto.ptr(pto.ui16, "ub"))

    pto.mte_gm_ub(src_gm, ub_src, 0, SRC_BYTES, nburst=(1, SRC_BYTES, SRC_BYTES))
    pto.mte_gm_ub(idx_gm, ub_idx, 0, IDX_BYTES, nburst=(1, IDX_BYTES, IDX_BYTES))

    pto.set_flag(pto.Pipe.MTE2, pto.Pipe.V, event_id=0)
    pto.wait_flag(pto.Pipe.MTE2, pto.Pipe.V, event_id=0)

    offset = pto.const(0, dtype=pto.index)
    idx = pto.vmi.vload(ub_idx, offset, size=ELEMS)
    mask = pto.vmi.create_mask(pto.const(ELEMS, dtype=pto.index), size=ELEMS)
    out = pto.vmi.vgather(ub_src, idx, mask)
    pto.vmi.vstore(out, ub_dst, offset, mask)

    pto.set_flag(pto.Pipe.V, pto.Pipe.MTE3, event_id=0)
    pto.wait_flag(pto.Pipe.V, pto.Pipe.MTE3, event_id=0)
    pto.mte_ub_gm(ub_dst, dst_gm, DST_BYTES, nburst=(1, DST_BYTES, DST_BYTES))
    pto.pipe_barrier(pto.Pipe.ALL)


def _non_identity_indices() -> np.ndarray:
    return ((np.arange(ELEMS, dtype=np.uint16) * 7 + 13) % ELEMS).astype(np.uint16)


def _i8_inputs():
    src = np.arange(ELEMS, dtype=np.uint8)
    return [src, _non_identity_indices()]


def _i8_expected(src, idx):
    return src.astype(np.uint16)[idx].astype(np.uint16)


def _u8_inputs():
    src = np.arange(ELEMS, dtype=np.uint8)
    return [src, _non_identity_indices()]


def _u8_expected(src, idx):
    return src.astype(np.uint16)[idx].astype(np.uint16)


CASES = [
    golden_output_case(
        "vmi_vgather_i8_to_i16_256",
        vmi_vgather_i8_to_i16_256_kernel,
        inputs=_i8_inputs,
        expected=_i8_expected,
        output_shape=(ELEMS,),
        output_dtype=np.uint16,
        rtol=0.0,
        atol=0.0,
    ),
    golden_output_case(
        "vmi_vgather_u8_to_u16_256",
        vmi_vgather_u8_to_u16_256_kernel,
        inputs=_u8_inputs,
        expected=_u8_expected,
        output_shape=(ELEMS,),
        output_dtype=np.uint16,
        rtol=0.0,
        atol=0.0,
    ),
]


KERNELS = [
    vmi_vgather_i8_to_i16_256_kernel,
    vmi_vgather_u8_to_u16_256_kernel,
]


auto_main(globals())
