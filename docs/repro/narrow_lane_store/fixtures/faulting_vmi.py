# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Faulting VMI fixture: 1-lane vstore into a packed (B,) float32 UB buffer.

For odd k the destination is 4-byte aligned but not 32-byte aligned. Current
builds fault the vector cores (ACL 507035).
"""

import os

import numpy as np
from ptodsl import pto

_B = 8
_DEVICE = os.environ.get("NPU_TEST_DEVICE", "npu:1")


@pto.jit(
    name="narrow_lane_store_packed",
    kernel_kind="vector",
    target="a5",
    backend="vpto",
    mode="explicit",
)
def narrow_lane_store_packed(
    val_gm: pto.ptr(pto.f32, "gm"),
    out_gm: pto.ptr(pto.f32, "gm"),
):
    c0 = pto.const(0)
    c1 = pto.const(1)
    c_b = pto.const(_B)
    shape = [c1, c1, c1, c1, c_b]
    strides = [c_b, c_b, c_b, c_b, c1]
    off = [c0, c0, c0, c0, c0]

    val_view = pto.make_tensor_view(val_gm, shape=shape, strides=strides)
    out_view = pto.make_tensor_view(out_gm, shape=shape, strides=strides)
    val_part = pto.partition_view(val_view, offsets=off, sizes=shape)
    out_part = pto.partition_view(out_view, offsets=off, sizes=shape)

    val_ub = pto.alloc_tile(shape=[1, _B], dtype=pto.f32)
    out_ub = pto.alloc_tile(shape=[1, _B], dtype=pto.f32)
    pto.tile.load(val_part, val_ub)

    one = pto.vmi.create_mask(1, size=1)
    for k in range(_B):
        src = pto.vmi.vload(
            pto.addptr(val_ub.as_ptr(), k),
            pto.const(0, dtype=pto.index),
            size=1,
        )
        pto.vmi.vstore(
            src,
            pto.addptr(out_ub.as_ptr(), k),
            pto.const(0, dtype=pto.index),
            one,
        )

    pto.tile.store(out_ub, out_part)


def run_numeric(device=None):
    import torch
    import torch_npu  # noqa: F401

    device = device or _DEVICE
    if not torch.npu.is_available() or torch.npu.device_count() < 1:
        raise RuntimeError("NPU not available")
    torch.npu.set_device(device)
    vals = (np.arange(_B, dtype=np.float32) + 1.0) * 10.0
    a = torch.from_numpy(vals).to(device)
    out = torch.full((_B,), float("nan"), dtype=torch.float32, device=device)
    stream = torch.npu.current_stream()._as_parameter_  # noqa: SLF001
    compiled = narrow_lane_store_packed.compile()
    compiled[1, stream](a.data_ptr(), out.data_ptr())
    torch.npu.synchronize()
    got = out.cpu().numpy()
    return {"ok": bool(np.allclose(got, vals)), "got": got.tolist()}


if __name__ == "__main__":
    print(narrow_lane_store_packed.compile().mlir_text()[:400])
