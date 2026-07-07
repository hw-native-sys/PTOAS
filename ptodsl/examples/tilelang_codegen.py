# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""
TileLang-generated explicit PTODSL kernel.

This file keeps the original generated kernel body essentially intact and only
adds the minimum wrapper needed to make it usable as a compile/test target:

- public `@pto.jit` host ABI via explicit GM pointers
- `--emit-mlir` entry point
- compile smoke path for regression tests
"""

import argparse
import time

import numpy as np

from ptodsl import pto


_DEVICE = "npu:0"


def _tilelang_generated_body(
    A,
    B,
    C,
):
    bx = pto.get_block_idx()
    buf_dyn_shmem = pto.const(0, dtype=pto.int64)
    with pto.for_(0, 2, step=1) as f:
        pto.set_flag("MTE3", "V", event_id=f)
        pto.set_flag("V", "MTE2", event_id=f)
    with pto.for_(0, 2048, step=1) as iter:
        pto.wait_flag("V", "MTE2", event_id=iter % 2)
        pto.mte_gm_ub(
            pto.addptr(A, (iter * 524288) + (bx * 8192)),
            pto.addptr(
                pto.castptr(buf_dyn_shmem, pto.ptr(pto.float32, "ub")),
                (iter % 2) * 8192,
            ),
            0,
            32768,
            nburst=(1, 0, 0),
        )
        pto.mte_gm_ub(
            pto.addptr(B, (iter * 524288) + (bx * 8192)),
            pto.addptr(
                pto.castptr(buf_dyn_shmem, pto.ptr(pto.float32, "ub")),
                ((iter % 2) * 8192) + 16384,
            ),
            0,
            32768,
            nburst=(1, 0, 0),
        )
        pto.set_flag("MTE2", "V", event_id=iter % 2)
        pto.wait_flag("MTE2", "V", event_id=iter % 2)
        pto.wait_flag("MTE3", "V", event_id=iter % 2)
        with pto.tileop():
            mask_cnt = 8192
            with pto.for_(0, 128, step=1) as i:
                mask = pto.pset_b32("PAT_ALL")
                r0 = pto.vlds(
                    pto.addptr(
                        pto.castptr(buf_dyn_shmem, pto.ptr(pto.float32, "ub")),
                        ((iter % 2) * 8192) + (i * 64),
                    ),
                    pto.const(0),
                    pto.vreg_type(64, pto.float32),
                )
                r1 = pto.vlds(
                    pto.addptr(
                        pto.castptr(buf_dyn_shmem, pto.ptr(pto.float32, "ub")),
                        (((iter % 2) * 8192) + (i * 64)) + 16384,
                    ),
                    pto.const(0),
                    pto.vreg_type(64, pto.float32),
                )
                r0 = pto.vadd(r0, r1, mask)
                pto.vsts(
                    r0,
                    pto.addptr(
                        pto.castptr(buf_dyn_shmem, pto.ptr(pto.float32, "ub")),
                        (((iter % 2) * 8192) + (i * 64)) + 32768,
                    ),
                    pto.const(0),
                    mask,
                )
        pto.set_flag("V", "MTE3", event_id=iter % 2)
        pto.set_flag("V", "MTE2", event_id=iter % 2)
        pto.wait_flag("V", "MTE3", event_id=iter % 2)
        pto.mte_ub_gm(
            pto.addptr(
                pto.castptr(buf_dyn_shmem, pto.ptr(pto.float32, "ub")),
                ((iter % 2) * 8192) + 32768,
            ),
            pto.addptr(C, (iter * 524288) + (bx * 8192)),
            32768,
            nburst=(1, 0, 0),
        )
        pto.set_flag("MTE3", "V", event_id=iter % 2)
    with pto.for_(0, 2, step=1) as f_1:
        pto.wait_flag("MTE3", "V", event_id=f_1)
        pto.wait_flag("V", "MTE2", event_id=f_1)


@pto.jit(
    name="main_kernel",
    kernel_kind="vector",
    target="a5",
    mode="explicit",
    insert_sync=False,
)
def main_kernel(
    A_ptr: pto.ptr(pto.f32, "gm"),
    B_ptr: pto.ptr(pto.f32, "gm"),
    C_ptr: pto.ptr(pto.f32, "gm"),
):
    _tilelang_generated_body(A_ptr, B_ptr, C_ptr)


def _tilelang_generated_body_small(A, B, C):
    bx = pto.get_block_idx()
    buf_dyn_shmem = pto.const(0, dtype=pto.int64)
    with pto.for_(0, 2, step=1) as f:
        pto.set_flag("MTE3", "V", event_id=f)
        pto.set_flag("V", "MTE2", event_id=f)
    with pto.for_(0, 2, step=1) as iter:
        pto.wait_flag("V", "MTE2", event_id=iter % 2)
        pto.mte_gm_ub(
            pto.addptr(A, (iter * 128) + (bx * 128)),
            pto.addptr(
                pto.castptr(buf_dyn_shmem, pto.ptr(pto.float32, "ub")),
                (iter % 2) * 128,
            ),
            0,
            512,
            nburst=(1, 0, 0),
        )
        pto.mte_gm_ub(
            pto.addptr(B, (iter * 128) + (bx * 128)),
            pto.addptr(
                pto.castptr(buf_dyn_shmem, pto.ptr(pto.float32, "ub")),
                ((iter % 2) * 128) + 256,
            ),
            0,
            512,
            nburst=(1, 0, 0),
        )
        pto.set_flag("MTE2", "V", event_id=iter % 2)
        pto.wait_flag("MTE2", "V", event_id=iter % 2)
        pto.wait_flag("MTE3", "V", event_id=iter % 2)
        with pto.tileop():
            with pto.for_(0, 2, step=1) as i:
                mask = pto.pset_b32("PAT_ALL")
                r0 = pto.vlds(
                    pto.addptr(
                        pto.castptr(buf_dyn_shmem, pto.ptr(pto.float32, "ub")),
                        ((iter % 2) * 128) + (i * 64),
                    ),
                    pto.const(0),
                    pto.vreg_type(64, pto.float32),
                )
                r1 = pto.vlds(
                    pto.addptr(
                        pto.castptr(buf_dyn_shmem, pto.ptr(pto.float32, "ub")),
                        (((iter % 2) * 128) + (i * 64)) + 256,
                    ),
                    pto.const(0),
                    pto.vreg_type(64, pto.float32),
                )
                r0 = pto.vadd(r0, r1, mask)
                pto.vsts(
                    r0,
                    pto.addptr(
                        pto.castptr(buf_dyn_shmem, pto.ptr(pto.float32, "ub")),
                        (((iter % 2) * 128) + (i * 64)) + 512,
                    ),
                    pto.const(0),
                    mask,
                )
        pto.set_flag("V", "MTE3", event_id=iter % 2)
        pto.set_flag("V", "MTE2", event_id=iter % 2)
        pto.wait_flag("V", "MTE3", event_id=iter % 2)
        pto.mte_ub_gm(
            pto.addptr(
                pto.castptr(buf_dyn_shmem, pto.ptr(pto.float32, "ub")),
                ((iter % 2) * 128) + 512,
            ),
            pto.addptr(C, (iter * 128) + (bx * 128)),
            512,
            nburst=(1, 0, 0),
        )
        pto.set_flag("MTE3", "V", event_id=iter % 2)
    with pto.for_(0, 2, step=1) as f_1:
        pto.wait_flag("MTE3", "V", event_id=f_1)
        pto.wait_flag("V", "MTE2", event_id=f_1)


@pto.jit(
    name="main_kernel_precision_test",
    kernel_kind="vector",
    target="a5",
    mode="explicit",
    insert_sync=False,
)
def main_kernel_precision_test(
    A_ptr: pto.ptr(pto.f32, "gm"),
    B_ptr: pto.ptr(pto.f32, "gm"),
    C_ptr: pto.ptr(pto.f32, "gm"),
):
    _tilelang_generated_body_small(A_ptr, B_ptr, C_ptr)


def emit_mlir():
    return main_kernel.mlir_text()


def compile_kernel():
    compiled = main_kernel.compile()
    compiled.verify()
    return compiled


def init_torch_npu():
    import torch
    import torch_npu  # noqa: F401

    torch.npu.config.allow_internal_format = False
    torch_npu.npu.set_compile_mode(jit_compile=False)
    torch.npu.set_device(_DEVICE)
    return torch


def npu_stream(torch):
    return torch.npu.current_stream()._as_parameter_  # noqa: SLF001


def make_case_inputs():
    total = 256
    rng = np.random.RandomState(20260524)
    a = rng.uniform(-3.0, 3.0, size=(total,)).astype(np.float32)
    b = rng.uniform(-3.0, 3.0, size=(total,)).astype(np.float32)
    c = np.full((total,), np.nan, dtype=np.float32)
    return a, b, c


def run_precision_case(torch) -> None:
    a_np, b_np, c_np = make_case_inputs()
    ref = a_np + b_np

    a_t = torch.from_numpy(a_np).to(_DEVICE)
    b_t = torch.from_numpy(b_np).to(_DEVICE)
    c_t = torch.from_numpy(c_np).to(_DEVICE)
    stream = npu_stream(torch)

    t0 = time.perf_counter()
    compiled = main_kernel_precision_test.compile()
    compile_s = time.perf_counter() - t0

    t0 = time.perf_counter()
    compiled[1, stream](a_t.data_ptr(), b_t.data_ptr(), c_t.data_ptr())
    torch.npu.synchronize()
    launch_s = time.perf_counter() - t0

    np.testing.assert_allclose(c_t.cpu().numpy(), ref, rtol=1e-6, atol=1e-6)
    print(f"PASS tilelang_codegen  compile={compile_s:.3f}s launch={launch_s:.3f}s")


def test_tilelang_codegen() -> None:
    torch = init_torch_npu()
    run_precision_case(torch)
    print("All cases passed.")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--emit-mlir",
        action="store_true",
        help="print compiled MLIR and exit",
    )
    args = parser.parse_args(argv)

    if args.emit_mlir:
        print(emit_mlir())
        return 0

    test_tilelang_codegen()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
