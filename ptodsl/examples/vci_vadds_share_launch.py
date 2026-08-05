# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Shared ``vci`` + VL ``vadds`` + ``vsts`` probe (group=1 / group=2).

Replaces topk-heavy validation for the dynamic contiguous iota share path:

  idx = vci(0, size=vl, group=G)   # G=2 → one VL64 ramp reused as 0..63|0..63
  out = vadds(idx, 1000, mask)     # VL-scale compute on that shared value
  vstore(out)                      # → 1000..1063  (G=1)
                                   # → 1000..1063|1000..1063  (G=2)

Run under the CPU / camodel simulator:

  scripts/sim_dsl.sh --output /tmp/vci_vadds_g1_out \\
    ptodsl/examples/vci_vadds_share_launch.py -- --groups 1
  scripts/sim_dsl.sh --output /tmp/vci_vadds_g2_out \\
    ptodsl/examples/vci_vadds_share_launch.py -- --groups 2
"""

import argparse
import sys
import time

import numpy as np

from ptodsl import pto

_DEVICE = "npu:0"
PHYS_VL = 64
ADD_SCALAR = 1000
K_REMAT = 2


def _expected(num_groups: int) -> np.ndarray:
    ramp = np.arange(ADD_SCALAR, ADD_SCALAR + PHYS_VL, dtype=np.int32)
    if num_groups == 1:
        return ramp.reshape(1, PHYS_VL)
    out = np.zeros((1, PHYS_VL * 2), dtype=np.int32)
    out[0, :PHYS_VL] = ramp
    out[0, PHYS_VL:] = ramp
    return out


def _make_kernel(num_groups: int):
    cols = PHYS_VL if num_groups == 1 else PHYS_VL * 2
    vl = PHYS_VL * num_groups
    nbytes = cols * 4

    @pto.jit(
        name=f"vci_vadds_share_g{num_groups}",
        kernel_kind="vector",
        target="a5",
        backend="vpto",
        mode="explicit",
    )
    def kernel(out_ptr: pto.ptr(pto.i32, "gm")):
        ub = pto.castptr(pto.i64(0), pto.ptr(pto.i32, "ub"))
        mask = pto.vmi.create_mask(vl, size=vl)
        for _k in range(K_REMAT):
            idx = pto.vmi.vci(pto.i32(0), size=vl, group=num_groups)
            out_idx = pto.vmi.vadds(idx, pto.i32(ADD_SCALAR), mask)
            pto.vmi.vstore(out_idx, ub, pto.const(0, dtype=pto.index))
        pto.set_flag("V", "MTE3", event_id=0)
        pto.wait_flag("V", "MTE3", event_id=0)
        pto.mte_ub_gm(
            ub,
            out_ptr,
            nbytes,
            nburst=(1, 0, 0),
        )

    return kernel


def init_torch_npu():
    import torch
    import torch_npu  # noqa: F401

    torch.npu.config.allow_internal_format = False
    torch_npu.npu.set_compile_mode(jit_compile=False)
    torch.npu.set_device(_DEVICE)
    return torch


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--groups", type=int, choices=(1, 2), default=2)
    argv = [a for a in sys.argv[1:] if a != "--"]
    args = ap.parse_args(argv)

    torch = init_torch_npu()
    kernel = _make_kernel(args.groups)
    expect = _expected(args.groups)

    t0 = time.perf_counter()
    compiled = kernel.compile()
    compile_s = time.perf_counter() - t0
    mlir = compiled.mlir_text()
    if "pto.vmi.vci" not in mlir:
        raise SystemExit("FAIL: expected pto.vmi.vci in frontend MLIR")
    if "pto.vmi.vadds" not in mlir and "vadds" not in mlir:
        raise SystemExit("FAIL: expected vadds in frontend MLIR")

    out = torch.zeros(expect.shape, dtype=torch.int32, device=_DEVICE)
    stream = torch.npu.current_stream()._as_parameter_  # noqa: SLF001
    t0 = time.perf_counter()
    compiled(out.data_ptr(), stream=stream)
    torch.npu.synchronize()
    launch_s = time.perf_counter() - t0

    got = out.cpu().numpy()
    if not np.array_equal(got, expect):
        diff = np.argwhere(got != expect)
        i, j = (int(x) for x in diff[0])
        print(f"FAIL groups={args.groups} num_mismatch={len(diff)}")
        print(f"  first bad lane={j} got={got[i, j]} expected={expect[i, j]}")
        print(f"  got={got[i].tolist()}")
        print(f"  exp={expect[i].tolist()}")
        raise SystemExit(1)

    half = ""
    if args.groups == 2:
        half = (
            f" half0={got[0, :4].tolist()}"
            f"|half1={got[0, PHYS_VL:PHYS_VL + 4].tolist()}"
        )
    print(
        f"PASS groups={args.groups} remat={K_REMAT} "
        f"vci(0)+vadds({ADD_SCALAR})+vsts "
        f"compile={compile_s:.2f}s launch={launch_s:.2f}s"
        f"{half}"
    )


if __name__ == "__main__":
    main()
