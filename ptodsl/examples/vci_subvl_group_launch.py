# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Sub-VL group-periodic ``vci`` + ``vadds`` + ``vsts`` camodel probe.

Cases (S < physVL, physVL % S == 0):

  i32 size=64  group=2 → [0..31|0..31] then +1000
  i32 size=64  group=4 → [0..15]x4 then +1000
  i16 size=128 group=2 → [0..63|0..63] then +1000

Run:

  scripts/sim_dsl.sh --output /tmp/vci_subvl_i32_g2_out \\
    ptodsl/examples/vci_subvl_group_launch.py -- --case i32_g2
  scripts/sim_dsl.sh --output /tmp/vci_subvl_i32_g4_out \\
    ptodsl/examples/vci_subvl_group_launch.py -- --case i32_g4
  scripts/sim_dsl.sh --output /tmp/vci_subvl_i16_g2_out \\
    ptodsl/examples/vci_subvl_group_launch.py -- --case i16_g2
"""

import argparse
import sys
import time

import numpy as np

from ptodsl import pto

_DEVICE = "npu:0"
ADD_SCALAR = 1000

# name -> (dtype, size, group, np_dtype, elem_bytes)
_CASES = {
    "i32_g2": (pto.i32, 64, 2, np.int32, 4),
    "i32_g4": (pto.i32, 64, 4, np.int32, 4),
    "i16_g2": (pto.i16, 128, 2, np.int16, 2),
}


def _expected(size: int, group: int, np_dtype) -> np.ndarray:
    s = size // group
    ramp = np.arange(ADD_SCALAR, ADD_SCALAR + s, dtype=np_dtype)
    out = np.tile(ramp, group).reshape(1, size)
    return out


def _make_kernel(case: str):
    dtype, size, group, _np_dtype, elem_bytes = _CASES[case]
    nbytes = size * elem_bytes

    @pto.jit(
        name=f"vci_subvl_{case}",
        kernel_kind="vector",
        target="a5",
        backend="vpto",
        mode="explicit",
    )
    def kernel(out_ptr: pto.ptr(dtype, "gm")):
        ub = pto.castptr(pto.i64(0), pto.ptr(dtype, "ub"))
        mask = pto.vmi.create_mask(size, size=size)
        idx = pto.vmi.vci(dtype(0), size=size, group=group)
        out_idx = pto.vmi.vadds(idx, dtype(ADD_SCALAR), mask)
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
    ap.add_argument("--case", choices=sorted(_CASES), default="i32_g2")
    argv = [a for a in sys.argv[1:] if a != "--"]
    args = ap.parse_args(argv)

    dtype, size, group, np_dtype, _ = _CASES[args.case]
    torch = init_torch_npu()
    kernel = _make_kernel(args.case)
    expect = _expected(size, group, np_dtype)

    t0 = time.perf_counter()
    compiled = kernel.compile()
    compile_s = time.perf_counter() - t0
    mlir = compiled.mlir_text()
    if "pto.vmi.vci" not in mlir:
        raise SystemExit("FAIL: expected pto.vmi.vci in frontend MLIR")
    if f"group = {group}" not in mlir and f"{{group = {group}" not in mlir:
        raise SystemExit(f"FAIL: expected group={group} in frontend MLIR:\n{mlir[:1500]}")

    torch_dtype = torch.int32 if np_dtype == np.int32 else torch.int16
    out = torch.zeros(expect.shape, dtype=torch_dtype, device=_DEVICE)
    stream = torch.npu.current_stream()._as_parameter_  # noqa: SLF001
    t0 = time.perf_counter()
    compiled(out.data_ptr(), stream=stream)
    torch.npu.synchronize()
    launch_s = time.perf_counter() - t0

    got = out.cpu().numpy()
    if not np.array_equal(got, expect):
        diff = np.argwhere(got != expect)
        i, j = (int(x) for x in diff[0])
        s = size // group
        print(f"FAIL case={args.case} num_mismatch={len(diff)}")
        print(f"  first bad lane={j} got={got[i, j]} expected={expect[i, j]}")
        print(f"  got groups={[got[0, g * s:(g + 1) * s].tolist() for g in range(group)]}")
        print(f"  exp groups={[expect[0, g * s:(g + 1) * s].tolist() for g in range(group)]}")
        raise SystemExit(1)

    s = size // group
    preview = " | ".join(
        f"g{g}={got[0, g * s:g * s + min(4, s)].tolist()}" for g in range(group)
    )
    print(
        f"PASS case={args.case} size={size} group={group} "
        f"vci(0)+vadds({ADD_SCALAR})+vsts "
        f"compile={compile_s:.2f}s launch={launch_s:.2f}s {preview}"
    )


if __name__ == "__main__":
    main()
