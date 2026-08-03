# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Failing VMI fixture: UB-resident table via vscatter then vgather.

Algorithm:
  1. seed pool[0..63] with -inf via vscatter
  2. pipe_barrier("V")
  3. scatter host values at the same offsets
  4. pipe_barrier("V")
  5. gather those offsets back to out

Expected: out == vals. Observed on current builds: every lane returns -inf
(the seed), as if the second scatter never landed.
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
    c0 = pto.const(0)
    c1 = pto.const(1)
    c_vl = pto.const(_VL)
    shape = [c1, c1, c1, c1, c_vl]
    strides = [c_vl, c_vl, c_vl, c_vl, c1]
    off = [c0, c0, c0, c0, c0]

    val_view = pto.make_tensor_view(val_gm, shape=shape, strides=strides)
    out_view = pto.make_tensor_view(out_gm, shape=shape, strides=strides)
    val_part = pto.partition_view(val_view, offsets=off, sizes=shape)
    out_part = pto.partition_view(out_view, offsets=off, sizes=shape)

    pool = pto.alloc_tile(shape=[1, _POOL], dtype=pto.f32)
    val_ub = pto.alloc_tile(shape=[1, _VL], dtype=pto.f32)
    out_ub = pto.alloc_tile(shape=[1, _VL], dtype=pto.f32)

    pto.tile.load(val_part, val_ub)

    mask = pto.vmi.create_mask(_VL, size=_VL)
    idx = pto.vmi.vci(pto.i32(0), size=_VL, order="ASC")
    off0 = pto.const(0, dtype=pto.index)

    neg_inf = pto.vmi.vbrc(pto.f32(float("-inf")), size=_VL)
    pto.vmi.vscatter(neg_inf, pool.as_ptr(), idx, mask)
    pto.pipe_barrier("V")

    val = pto.vmi.vload(val_ub.as_ptr(), off0, size=_VL)
    pto.vmi.vscatter(val, pool.as_ptr(), idx, mask)
    pto.pipe_barrier("V")

    got = pto.vmi.vgather(pool.as_ptr(), idx, mask)
    pto.vmi.vstore(got, out_ub.as_ptr(), off0, mask)

    pto.tile.store(out_ub, out_part)


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
