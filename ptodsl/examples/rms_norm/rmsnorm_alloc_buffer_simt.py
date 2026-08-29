# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""
RMSNorm compile-only PTODSL example for issue 483.

The example exercises the PTODSL surfaces needed by the RMSNorm SimtVF kernel:

- ``pto.alloc_buffer(...)`` for lane-local SIMT fragment storage
- hand-authored dynamic UB scratch layout via ``pto.castptr`` / ``pto.addptr``
- contiguous scalar ``load`` / ``store`` vector accesses
- ``pto.simt_allreduce_sum(...)`` for cross-workitem sum reduction
- W stays in UB after the GM->UB preload and is read directly by the token SIMT body
- runtime ``range(...)`` for the token loop so the AST rewrite emits ``scf.for``
- Python ``range(...)`` loops inside SIMT helpers to emit compact runtime loops

Run this file directly to print the emitted MLIR for one specialization.
"""

import argparse


from ptodsl import pto


@pto.simt
def rmsnorm_simt_token_body(
    x_ub,
    y_ub,
    rstd_ub,
    reduce_scratch,
    w_ub,
    eps: pto.f32,
    pingpong: pto.i32,
    *,
    threads: pto.const_expr = 128,
    rounds: pto.const_expr = 16,
    lanes: pto.const_expr = 2,
    hidden_size: pto.const_expr = 4096,
):
    tx = pto.get_tid_x()
    frag_elems: pto.const_expr = rounds * lanes
    x_frag = pto.alloc_buffer((frag_elems,), pto.f32)
    sum_sq = pto.alloc_buffer((1,), pto.f32)

    for r in range(0, rounds):
        lane_offset = r * threads * lanes + tx * lanes
        x_offset = pingpong * hidden_size + lane_offset
        frag_offset = r * lanes

        x_vec = pto.load(x_ub, x_offset, contiguous=lanes)
        pto.store(x_vec, x_frag, frag_offset)

    pto.store(pto.const(0.0, dtype=pto.f32), sum_sq, 0)

    for i in range(0, frag_elems):
        local_sum = pto.load(sum_sq, 0)
        x = pto.load(x_frag, i)
        local_sum = local_sum + x * x
        pto.store(local_sum, sum_sq, 0)

    local_sum = pto.load(sum_sq, 0)

    sum_sq = pto.simt_allreduce_sum(
        local_sum,
        threads=threads,
        scale=1,
        thread_offset=0,
        scratch=reduce_scratch,
    )

    rstd = 1.0 / pto.sqrt(sum_sq / hidden_size + eps)

    pto.store(rstd, rstd_ub, pingpong * 8)

    for r in range(0, rounds):
        round_offset = r * threads * lanes
        thread_offset = tx * lanes
        lane_base = round_offset + thread_offset
        y_offset = pingpong * hidden_size + lane_base
        frag_offset = r * lanes

        x_vec = pto.load(x_frag, frag_offset, contiguous=lanes)
        w_vec = pto.load(w_ub, lane_base, contiguous=lanes)
        rstd_vec = pto.Vec(pto.f32, lanes, init=rstd)
        y_vec = x_vec * rstd_vec * w_vec
        pto.store(y_vec, y_ub, y_offset)


@pto.jit(target="a5", mode="explicit", kernel_kind="vector")
def rmsnorm_4096_alloc_buffer_simt_context_kernel(
    X: pto.ptr(pto.f32, "gm"),
    Y: pto.ptr(pto.f32, "gm"),
    W: pto.ptr(pto.f32, "gm"),
    RSTD: pto.ptr(pto.f32, "gm"),
    eps: pto.f32,
    *,
    threads: pto.const_expr = 128,
    rounds: pto.const_expr = 8,
    lanes: pto.const_expr = 4,
    hidden_size: pto.const_expr = 4096,
    n_cores: pto.const_expr = 64,
    tokens_per_core: pto.const_expr = 64,
    f32_bytes: pto.const_expr = 4,
):
    assert threads * rounds * lanes == hidden_size, (
        "threads * rounds * lanes must equal hidden_size for RMSNorm SIMT partitioning"
    )

    core_id = pto.get_block_idx()

    ub_base = pto.castptr(pto.const(0, dtype=pto.ui64), pto.ptr(pto.f32, "ub"))
    w_ub = pto.addptr(ub_base, 0)
    reduce_scratch = pto.addptr(ub_base, hidden_size)
    x_ub = pto.addptr(ub_base, hidden_size + 128)
    y_ub = pto.addptr(ub_base, hidden_size + 128 + 2 * hidden_size)
    rstd_ub = pto.addptr(ub_base, hidden_size + 128 + 4 * hidden_size)

    pto.mte_gm_ub(
        W,
        w_ub,
        0,
        hidden_size * f32_bytes,
        nburst=(1, hidden_size * f32_bytes, hidden_size * f32_bytes),
    )
    pto.set_flag("MTE2", "V", event_id=3)
    pto.wait_flag("MTE2", "V", event_id=3)

    pto.set_flag("V", "MTE2", event_id=0)
    pto.set_flag("MTE3", "V", event_id=0)
    pto.set_flag("V", "MTE2", event_id=1)
    pto.set_flag("MTE3", "V", event_id=1)

    for local_token in range(0, tokens_per_core):
        token_id = local_token * n_cores + core_id
        pingpong = local_token % 2

        pto.wait_flag("V", "MTE2", event_id=pingpong)
        pto.mte_gm_ub(
            pto.addptr(X, token_id * hidden_size),
            pto.addptr(x_ub, pingpong * hidden_size),
            0,
            hidden_size * f32_bytes,
            nburst=(1, hidden_size * f32_bytes, hidden_size * f32_bytes),
        )
        pto.set_flag("MTE2", "V", event_id=pingpong)

        pto.wait_flag("MTE2", "V", event_id=pingpong)
        pto.wait_flag("MTE3", "V", event_id=pingpong)
        rmsnorm_simt_token_body[threads, 1, 1](
            x_ub,
            y_ub,
            rstd_ub,
            reduce_scratch,
            w_ub,
            eps,
            pingpong,
            threads=threads,
            rounds=rounds,
            lanes=lanes,
            hidden_size=hidden_size,
        )
        pto.set_flag("V", "MTE2", event_id=pingpong)
        pto.set_flag("V", "MTE3", event_id=pingpong)

        pto.wait_flag("V", "MTE3", event_id=pingpong)
        pto.mte_ub_gm(
            pto.addptr(y_ub, pingpong * hidden_size),
            pto.addptr(Y, token_id * hidden_size),
            hidden_size * f32_bytes,
            nburst=(1, hidden_size * f32_bytes, hidden_size * f32_bytes),
        )

        pto.mte_ub_gm(
            pto.addptr(rstd_ub, pingpong * 8),
            pto.addptr(RSTD, token_id),
            f32_bytes,
            nburst=(1, f32_bytes, f32_bytes),
        )
        pto.set_flag("MTE3", "V", event_id=pingpong)

    pto.wait_flag("V", "MTE2", event_id=0)
    pto.wait_flag("V", "MTE2", event_id=1)
    pto.wait_flag("MTE3", "V", event_id=0)
    pto.wait_flag("MTE3", "V", event_id=1)


def build_x128():
    return rmsnorm_4096_alloc_buffer_simt_context_kernel.compile(
        threads=128,
        rounds=8,
        lanes=4,
        tokens_per_core=64,
    )


def build_x64():
    return rmsnorm_4096_alloc_buffer_simt_context_kernel.compile(
        threads=64,
        rounds=16,
        lanes=4,
        tokens_per_core=64,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Emit RMSNorm PTODSL MLIR")
    parser.add_argument("--variant", choices=("x128", "x64"), default="x128")
    args = parser.parse_args()

    compiled = build_x128() if args.variant == "x128" else build_x64()
    compiled.verify()
    print(compiled.mlir_text())


if __name__ == "__main__":
    main()
