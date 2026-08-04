# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Faulting VMI fixture: 1-lane vstore into a packed (B,) float32 UB buffer.

Input is staged as (B, 8) so every 1-lane *load* address is 32-byte aligned.
Only the destination of the 1-lane *store* is packed — for odd k that address
is 4-byte aligned but not 32-byte aligned. Current builds fault the vector
cores (ACL 507035).
"""

import os

import numpy as np
from ptodsl import pto

_B = 8
_ALIGN = 8
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
    ub = pto.castptr(pto.const(0, dtype=pto.i64), pto.ptr(pto.f32, "ub"))
    pad_in = ub
    out_ub = pto.addptr(ub, _B * _ALIGN)
    in_bytes = _B * _ALIGN * 4
    out_bytes = _B * 4

    pto.mte_gm_ub(val_gm, pad_in, 0, in_bytes, nburst=(1, in_bytes, in_bytes))
    pto.set_flag("MTE2", "V", event_id=0)
    pto.wait_flag("MTE2", "V", event_id=0)

    one = pto.vmi.create_mask(1, size=1)
    for k in range(_B):
        src = pto.vmi.vload(
            pto.addptr(pad_in, k * _ALIGN),
            pto.const(0, dtype=pto.index),
            size=1,
        )
        pto.vmi.vstore(
            src,
            pto.addptr(out_ub, k),
            pto.const(0, dtype=pto.index),
            one,
        )

    pto.set_flag("V", "MTE3", event_id=0)
    pto.wait_flag("V", "MTE3", event_id=0)
    pto.mte_ub_gm(out_ub, out_gm, out_bytes, nburst=(1, out_bytes, out_bytes))


def run_numeric(device=None):
    import torch
    import torch_npu  # noqa: F401

    device = device or _DEVICE
    if not torch.npu.is_available() or torch.npu.device_count() < 1:
        raise RuntimeError("NPU not available")
    torch.npu.set_device(device)
    vals = (np.arange(_B, dtype=np.float32) + 1.0) * 10.0
    pad = np.zeros((_B, _ALIGN), dtype=np.float32)
    pad[:, 0] = vals
    a = torch.from_numpy(pad).to(device)
    out = torch.full((_B,), float("nan"), dtype=torch.float32, device=device)
    stream = torch.npu.current_stream()._as_parameter_  # noqa: SLF001
    compiled = narrow_lane_store_packed.compile()
    compiled[1, stream](a.data_ptr(), out.data_ptr())
    torch.npu.synchronize()
    got = out.cpu().numpy()
    return {"ok": bool(np.allclose(got, vals)), "got": got.tolist()}


if __name__ == "__main__":
    print(narrow_lane_store_packed.compile().mlir_text()[:400])
