#!/usr/bin/env python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

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
    raise RuntimeError("Unable to locate test/dsl-st/common.py")


_bootstrap_dsl_st_common()

from common import auto_main, golden_output_case
from ptodsl import pto


ROWS = 16
COLS = 16
BYTES = ROWS * COLS * 2


def _transpose_kernel(name, element_type):
    @pto.jit(
        name=name,
        target="a5",
        backend="vpto",
        mode="explicit",
        kernel_kind="vector",
        insert_sync=False,
    )
    def kernel(src: pto.ptr(element_type, "gm"), dst: pto.ptr(element_type, "gm")):
        zero = pto.const(0, dtype=pto.i64)
        ub_src = pto.castptr(zero, pto.ptr(element_type, "ub"))
        ub_dst = pto.castptr(
            pto.const(BYTES, dtype=pto.i64), pto.ptr(element_type, "ub")
        )
        pto.mte_gm_ub(src, ub_src, 0, BYTES, nburst=(1, BYTES, BYTES))
        pto.set_flag(pto.Pipe.MTE2, pto.Pipe.V, event_id=0)
        pto.wait_flag(pto.Pipe.MTE2, pto.Pipe.V, event_id=0)
        pto.vtranspose(ub_dst, ub_src)
        pto.set_flag(pto.Pipe.V, pto.Pipe.MTE3, event_id=0)
        pto.wait_flag(pto.Pipe.V, pto.Pipe.MTE3, event_id=0)
        pto.mte_ub_gm(ub_dst, dst, BYTES, nburst=(1, BYTES, BYTES))
        pto.pipe_barrier(pto.Pipe.ALL)

    return kernel


# Runtime execution remains available for the non-overlapping case once the
# simulator/compiler provides VTRANSPOSE support.
vtranspose_ui16_kernel = _transpose_kernel("vtranspose_ui16_kernel", pto.ui16)


def make_unsigned_inputs():
    source = (np.arange(ROWS * COLS, dtype=np.uint32) * 257 % 65536).astype(np.uint16)
    return [source]


def transpose_expected(source):
    return source.reshape(ROWS, COLS).T.reshape(-1)


CASES = [
    golden_output_case(
        "vtranspose_ui16",
        vtranspose_ui16_kernel,
        inputs=make_unsigned_inputs,
        expected=transpose_expected,
        rtol=0.0,
        atol=0.0,
    ),
]


auto_main(globals())
