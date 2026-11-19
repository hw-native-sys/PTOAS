#!/usr/bin/env python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""All BF16 encodings: exact finite/infinity widening and preserved NaN class."""

import numpy as np

from common import auto_main
from ptodsl import pto

ELEMENTS = 65536


def _build_kernel(lanes):
    @pto.jit(name=f"vmi_deinterleave_extf_{lanes}_kernel", target="a5", backend="vpto",
             mode="explicit", kernel_kind="vector", insert_sync=False)
    def _kernel(src: pto.ptr(pto.bf16, "gm"), dst: pto.ptr(pto.ui32, "gm")):
        src_ub = pto.castptr(pto.i64(0), pto.ptr(pto.bf16, "ub"))
        dst_ub = pto.castptr(pto.i64(1024), pto.ptr(pto.ui32, "ub"))
        for offset in pto.range(0, ELEMENTS, lanes):
            pto.mte_gm_ub(pto.addptr(src, offset), src_ub, 0, lanes * 2,
                          nburst=(1, lanes * 2, lanes * 2))
            pto.set_flag("MTE2", "V", event_id=0)
            pto.wait_flag("MTE2", "V", event_id=0)
            packed = pto.vmi.vload(src_ub, 0, size=lanes)
            words = pto.vmi.vinterpret_cast(packed, pto.ui32)
            even_bf16, odd_bf16 = pto.vmi.vunzip(words, pto.bf16)
            even = pto.vmi.vcvt(even_bf16, pto.f32)
            odd = pto.vmi.vcvt(odd_bf16, pto.f32)
            even_bits = pto.vmi.vinterpret_cast(even, pto.ui32)
            odd_bits = pto.vmi.vinterpret_cast(odd, pto.ui32)
            pto.vmi.vstore(even_bits, dst_ub, 0)
            pto.vmi.vstore(odd_bits, dst_ub, lanes // 2)
            pto.set_flag("V", "MTE3", event_id=0)
            pto.wait_flag("V", "MTE3", event_id=0)
            pto.mte_ub_gm(dst_ub, pto.addptr(dst, offset), lanes * 4,
                          nburst=(1, lanes * 4, lanes * 4))
            pto.pipe_barrier(pto.Pipe.ALL)
    return _kernel


def _make_case(lanes):
    source = np.arange(ELEMENTS, dtype=np.uint16)
    # Outputs are channel-major within every source packet. Finite values,
    # signed zero, infinities and subnormals must retain their exact bits.
    # FP conversion may canonicalize NaNs; their payload is not a cast contract.
    packets = source.astype(np.uint32).reshape(-1, lanes)
    expected = np.concatenate((packets[:, ::2], packets[:, 1::2]), axis=1)
    expected = (expected << np.uint32(16)).reshape(-1)
    return [source, np.zeros(ELEMENTS, dtype=np.uint32)], expected


def _check_case(device_inputs, expected):
    actual = device_inputs[1].cpu().numpy()
    nan = ((expected & 0x7F800000) == 0x7F800000) & ((expected & 0x007FFFFF) != 0)
    np.testing.assert_array_equal(actual[~nan], expected[~nan])
    np.testing.assert_array_equal(np.isnan(actual[nan].view(np.float32)), True)


def _case(lanes):
    def make_case():
        return _make_case(lanes)

    return {"name": f"vmi_deinterleave_extf_{lanes}_all_bf16_bits",
            "kernel": _build_kernel(lanes), "make_case": make_case, "check": _check_case}


CASES = [_case(128), _case(256)]


auto_main(globals())
