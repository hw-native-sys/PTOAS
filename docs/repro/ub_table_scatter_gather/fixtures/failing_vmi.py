# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Failing VMI fixture: UB-resident table via vscatter then vgather.

Algorithm:
  1. seed pool[0..63] with -inf via contiguous vstore (known-good write path)
  2. pipe_barrier("V")
  3. vscatter host values at identity offsets
  4. pipe_barrier("V")
  5. vgather those offsets back to out

Expected: out == vals. Observed on current builds: every lane returns -inf
(the seed), as if the vscatter never landed.

Note: seeding with vscatter itself also fails to land (gather then returns
NaN from untouched UB). The vstore seed makes the stale-sentinel signature
unambiguous. Contiguous vstore/vload and MTE→vgather of the same addresses
are exact — only vscatter is broken.
"""

import os

import numpy as np
from ptodsl import pto

_VL = 64
_POOL = 256
_DEVICE = os.environ.get("NPU_TEST_DEVICE", "npu:1")


@pto.jit(
    name="ub_table_scatter_gather",
    kernel_kind="vector",
    target="a5",
    backend="vpto",
    mode="explicit",
)
def ub_table_scatter_gather(
    val_gm: pto.ptr(pto.f32, "gm"),
    out_gm: pto.ptr(pto.f32, "gm"),
):
    ub = pto.castptr(pto.const(0, dtype=pto.i64), pto.ptr(pto.f32, "ub"))
    pool = ub
    val_ub = pto.addptr(ub, _POOL)
    out_ub = pto.addptr(ub, _POOL + _VL)
    nbytes = _VL * 4

    pto.mte_gm_ub(val_gm, val_ub, 0, nbytes, nburst=(1, nbytes, nbytes))
    pto.set_flag("MTE2", "V", event_id=0)
    pto.wait_flag("MTE2", "V", event_id=0)

    mask = pto.vmi.create_mask(_VL, size=_VL)
    idx = pto.vmi.vci(pto.i32(0), size=_VL, order="ASC")
    off0 = pto.const(0, dtype=pto.index)

    neg_inf = pto.vmi.vbrc(pto.f32(float("-inf")), size=_VL)
    pto.vmi.vstore(neg_inf, pool, off0, mask)
    pto.pipe_barrier("V")

    val = pto.vmi.vload(val_ub, off0, size=_VL)
    pto.vmi.vscatter(val, pool, idx, mask)
    pto.pipe_barrier("V")

    got = pto.vmi.vgather(pool, idx, mask)
    pto.vmi.vstore(got, out_ub, off0, mask)

    pto.set_flag("V", "MTE3", event_id=0)
    pto.wait_flag("V", "MTE3", event_id=0)
    pto.mte_ub_gm(out_ub, out_gm, nbytes, nburst=(1, nbytes, nbytes))


def run_numeric(device=None):
    """Compile, launch, return a result dict. Raises if NPU unavailable."""
    import torch
    import torch_npu  # noqa: F401

    device = device or _DEVICE
    if not torch.npu.is_available() or torch.npu.device_count() < 1:
        raise RuntimeError("NPU not available")
    torch.npu.set_device(device)

    vals = np.arange(_VL, dtype=np.float32) + 1.0
    a = torch.from_numpy(vals).to(device)
    out = torch.full((_VL,), float("nan"), dtype=torch.float32, device=device)
    stream = torch.npu.current_stream()._as_parameter_  # noqa: SLF001
    compiled = ub_table_scatter_gather.compile()
    compiled[1, stream](a.data_ptr(), out.data_ptr())
    torch.npu.synchronize()
    got = out.cpu().numpy()
    ok = bool(np.allclose(got, vals))
    all_neg_inf = bool(np.all(np.isneginf(got)))
    return {
        "ok": ok,
        "all_neg_inf": all_neg_inf,
        "got_head": got[:8].tolist(),
        "expected_head": vals[:8].tolist(),
    }


if __name__ == "__main__":
    print(ub_table_scatter_gather.compile().mlir_text()[:500])
