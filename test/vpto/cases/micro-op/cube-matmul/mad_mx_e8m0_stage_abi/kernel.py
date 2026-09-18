#!/usr/bin/env python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""E8M0 MX scale staging address-ABI regression for the PTODSL VPTO path."""

import re
import time

import numpy as np

from ptodsl import pto


K = 256
TILE_K_SUB = 128
SCALE_GROUP_K = 32
SCALE_X_BLOCK = 16
SCALE_SRC_STRIDE = 8
SCALE_STORAGE_ELEMENT_BYTES = 2
SCALE_PAIR_COUNT = K // (2 * SCALE_GROUP_K)
RUNTIME_SHAPES = ((256, 256),)
NON_SQUARE_COMPILE_SHAPE = (256, 192)

L1_A_DATA_ADDR = 0
L1_B_DATA_ADDR = 65536
L1_A_SCALE_ADDR = 131072
L1_B_SCALE_ADDR = 135168
L0_STAGE1_ADDR = 32768

_DEVICE = "npu:0"


@pto.jit(
    name="mad_mx_e8m0_stage_abi_kernel",
    target="a5",
    kernel_kind="cube",
    mode="explicit",
    insert_sync=False,
)
def mad_mx_e8m0_stage_abi_kernel(
    a_gm: pto.ptr(pto.f8e4m3, "gm"),
    b_gm: pto.ptr(pto.f8e4m3, "gm"),
    a_scale_gm: pto.ptr(pto.ui16, "gm"),
    b_scale_gm: pto.ptr(pto.ui16, "gm"),
    c_gm: pto.ptr(pto.f32, "gm"),
    *,
    M: pto.const_expr = 256,
    N: pto.const_expr = 256,
):
    # PTODSL passes real L0 byte addresses. The MX wrapper derives the internal
    # scale-stage destination required by the LOAD.MX ABI.
    a_l1 = pto.castptr(pto.ui64(L1_A_DATA_ADDR), pto.ptr(pto.f8e4m3, "mat"))
    b_l1 = pto.castptr(pto.ui64(L1_B_DATA_ADDR), pto.ptr(pto.f8e4m3, "mat"))
    a_scale_storage_l1 = pto.castptr(pto.ui64(L1_A_SCALE_ADDR), pto.ptr(pto.ui16, "mat"))
    b_scale_storage_l1 = pto.castptr(pto.ui64(L1_B_SCALE_ADDR), pto.ptr(pto.ui16, "mat"))
    a_scale_l1 = pto.castptr(pto.ui64(L1_A_SCALE_ADDR), pto.ptr(pto.f8e8m0, "mat"))
    b_scale_l1 = pto.castptr(pto.ui64(L1_B_SCALE_ADDR), pto.ptr(pto.f8e8m0, "mat"))
    # The second K half follows the first Mx128 / 128xN tile in L1.
    a_l1_stage1 = pto.castptr(
        pto.ui64(L1_A_DATA_ADDR + M * TILE_K_SUB), pto.ptr(pto.f8e4m3, "mat")
    )
    b_l1_stage1 = pto.castptr(
        pto.ui64(L1_B_DATA_ADDR + TILE_K_SUB * N), pto.ptr(pto.f8e4m3, "mat")
    )

    a_l0_stage0 = pto.castptr(pto.ui64(0), pto.ptr(pto.f8e4m3, "left"))
    b_l0_stage0 = pto.castptr(pto.ui64(0), pto.ptr(pto.f8e4m3, "right"))
    a_l0_stage1 = pto.castptr(pto.ui64(L0_STAGE1_ADDR), pto.ptr(pto.f8e4m3, "left"))
    b_l0_stage1 = pto.castptr(pto.ui64(L0_STAGE1_ADDR), pto.ptr(pto.f8e4m3, "right"))
    c_l0 = pto.castptr(pto.ui64(0), pto.ptr(pto.f32, "acc"))

    pto.mte_gm_l1(a_gm, a_l1, M * K, nburst=(1, 0, 0))
    pto.mte_gm_l1(b_gm, b_l1, K * N, nburst=(1, 0, 0))
    pto.mte_gm_l1(
        a_scale_gm,
        a_scale_storage_l1,
        M * SCALE_SRC_STRIDE * SCALE_STORAGE_ELEMENT_BYTES,
        nburst=(1, 0, 0),
    )
    pto.mte_gm_l1(
        b_scale_gm,
        b_scale_storage_l1,
        N * SCALE_SRC_STRIDE * SCALE_STORAGE_ELEMENT_BYTES,
        nburst=(1, 0, 0),
    )

    pto.set_flag(pto.Pipe.MTE2, pto.Pipe.MTE1, event_id=0)
    pto.wait_flag(pto.Pipe.MTE2, pto.Pipe.MTE1, event_id=0)

    pto.mte_l1_l0a(a_l1, a_l0_stage0, M, TILE_K_SUB)
    pto.mte_l1_l0b(b_l1, b_l0_stage0, TILE_K_SUB, N, transpose=True)
    pto.mte_l1_l0a_mx(
        a_scale_l1, a_l0_stage0,
        x_start=0, y_start=0, x_step=M // 16, y_step=2,
        src_stride=SCALE_SRC_STRIDE, dst_stride=2,
    )
    pto.mte_l1_l0b_mx(
        b_scale_l1, b_l0_stage0,
        x_start=0, y_start=0, x_step=N // 16, y_step=2,
        src_stride=SCALE_SRC_STRIDE, dst_stride=2,
    )
    pto.set_flag(pto.Pipe.MTE1, pto.Pipe.M, event_id=0)
    pto.wait_flag(pto.Pipe.MTE1, pto.Pipe.M, event_id=0)
    pto.mad_mx(a_l0_stage0, b_l0_stage0, c_l0, M, N, TILE_K_SUB, disable_gemv=True, sat="sat")
    pto.pipe_barrier(pto.Pipe.ALL)

    pto.mte_l1_l0a(a_l1_stage1, a_l0_stage1, M, TILE_K_SUB)
    pto.mte_l1_l0b(b_l1_stage1, b_l0_stage1, TILE_K_SUB, N, transpose=True)
    # y_start=2 selects the second K=128 scale half. Together with the nonzero
    # L0 destination, this distinguishes raw addresses from pre-divided ones.
    pto.mte_l1_l0a_mx(
        a_scale_l1, a_l0_stage1,
        x_start=0, y_start=2, x_step=M // 16, y_step=2,
        src_stride=SCALE_SRC_STRIDE, dst_stride=2,
    )
    pto.mte_l1_l0b_mx(
        b_scale_l1, b_l0_stage1,
        x_start=0, y_start=2, x_step=N // 16, y_step=2,
        src_stride=SCALE_SRC_STRIDE, dst_stride=2,
    )
    pto.set_flag(pto.Pipe.MTE1, pto.Pipe.M, event_id=1)
    pto.wait_flag(pto.Pipe.MTE1, pto.Pipe.M, event_id=1)
    pto.mad_mx_acc(a_l0_stage1, b_l0_stage1, c_l0, M, N, TILE_K_SUB, disable_gemv=True, sat="sat")

    pto.set_flag(pto.Pipe.M, pto.Pipe.FIX, event_id=1)
    pto.wait_flag(pto.Pipe.M, pto.Pipe.FIX, event_id=1)
    pto.mte_l0c_gm(c_l0, c_gm, M, N, N, N, 0, 0, layout="nz2nd")
    pto.pipe_barrier(pto.Pipe.ALL)


def _e8m0_to_f32(bits):
    return np.exp2(bits.astype(np.int32) - 127).astype(np.float32)


def _pack_scale_pairs(scale_bytes):
    """Pack [MN, K/32] E8M0 bytes into stride-8 uint16 MX source storage."""
    assert scale_bytes.ndim == 2
    assert scale_bytes.shape[1] == K // SCALE_GROUP_K

    mn = scale_bytes.shape[0]
    pairs = np.ascontiguousarray(
        scale_bytes.reshape(mn, SCALE_PAIR_COUNT, 2)
    ).view(np.uint16).reshape(mn, SCALE_PAIR_COUNT)
    assert mn % SCALE_X_BLOCK == 0
    packed = np.full(
        (mn // SCALE_X_BLOCK, SCALE_SRC_STRIDE, SCALE_X_BLOCK),
        0x7F7F,
        dtype=np.uint16,
    )
    packed[:, :SCALE_PAIR_COUNT, :] = pairs.reshape(
        mn // SCALE_X_BLOCK, SCALE_X_BLOCK, SCALE_PAIR_COUNT
    ).transpose(0, 2, 1)
    return packed.reshape(-1)


def _make_case(m, n):
    # E4M3 0x38 is exactly 1.0. Scale values vary by M/N lane and K/32 group,
    # exposing a wrong source stride, scale half, or stage-1 target address.
    a = np.full((m, K), 0x38, dtype=np.uint8)
    b = np.full((K, n), 0x38, dtype=np.uint8)
    group_ids = np.arange(K // SCALE_GROUP_K, dtype=np.uint8)
    m_ids = np.arange(m, dtype=np.uint8)[:, None]
    n_ids = np.arange(n, dtype=np.uint8)[None, :]
    a_scale = (126 + ((m_ids + group_ids) % 3)).astype(np.uint8)
    b_scale = (126 + ((group_ids[:, None] + 2 * n_ids) % 3)).astype(np.uint8)

    reference = np.zeros((m, n), dtype=np.float32)
    for group in range(K // SCALE_GROUP_K):
        reference += (
            SCALE_GROUP_K
            * _e8m0_to_f32(a_scale[:, group : group + 1])
            * _e8m0_to_f32(b_scale[group : group + 1, :])
        )

    return (
        a,
        b,
        _pack_scale_pairs(a_scale),
        _pack_scale_pairs(b_scale.T),
        reference,
    )


def _init_runtime():
    import torch
    import torch_npu  # noqa: F401

    torch.npu.config.allow_internal_format = False
    torch_npu.npu.set_compile_mode(jit_compile=False)
    torch.npu.set_device(_DEVICE)
    return torch


def _npu_stream(torch):
    return torch.npu.current_stream()._as_parameter_  # noqa: SLF001


def _mx_x_steps(mlir_text, op_name):
    constants = {
        match.group("ssa"): int(match.group("value"))
        for match in re.finditer(
            r"(?P<ssa>%[A-Za-z0-9_.$]+) = (?:arith|pto).constant "
            r"(?P<value>\d+) : i64",
            mlir_text,
        )
    }
    x_steps = []
    for line in mlir_text.splitlines():
        if op_name not in line:
            continue
        operands = line.split(op_name, 1)[1].split(" :", 1)[0].split(", ")
        x_steps.append(constants[operands[4]])
    return x_steps


def _verify_non_square_mx_controls():
    m, n = NON_SQUARE_COMPILE_SHAPE
    compiled = mad_mx_e8m0_stage_abi_kernel.compile(M=m, N=n)
    mlir_text = compiled.mlir_text()
    assert _mx_x_steps(mlir_text, "pto.mte_l1_l0a_mx") == [m // 16, m // 16]
    assert _mx_x_steps(mlir_text, "pto.mte_l1_l0b_mx") == [n // 16, n // 16]
    print(f"PASS mad_mx_e8m0_stage_abi controls={m}x{n}")


def main():
    torch = _init_runtime()
    _verify_non_square_mx_controls()
    for m, n in RUNTIME_SHAPES:
        a, b, a_scale, b_scale, reference = _make_case(m, n)
        a_dev = torch.from_numpy(a).to(_DEVICE)
        b_dev = torch.from_numpy(b).to(_DEVICE)
        a_scale_dev = torch.from_numpy(a_scale).to(_DEVICE)
        b_scale_dev = torch.from_numpy(b_scale).to(_DEVICE)
        c_dev = torch.from_numpy(np.zeros((m, n), dtype=np.float32)).to(_DEVICE)

        start = time.perf_counter()
        compiled = mad_mx_e8m0_stage_abi_kernel.compile(M=m, N=n)
        compile_seconds = time.perf_counter() - start

        start = time.perf_counter()
        compiled[1, _npu_stream(torch)](
            a_dev.data_ptr(),
            b_dev.data_ptr(),
            a_scale_dev.data_ptr(),
            b_scale_dev.data_ptr(),
            c_dev.data_ptr(),
        )
        torch.npu.synchronize()
        launch_seconds = time.perf_counter() - start

        np.testing.assert_allclose(c_dev.cpu().numpy(), reference, rtol=1e-3, atol=1e-3)
        print(
            "PASS mad_mx_e8m0_stage_abi "
            f"shape={m}x{n} compile={compile_seconds:.3f}s launch={launch_seconds:.3f}s"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
