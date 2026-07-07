# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""
Row-wise softmax kernel – compile-only DSL builder.

This sample mirrors the launchable softmax demo. It uses native Python
``if`` / ``for range(...)`` in the JIT body; the default PTODSL AST rewrite
turns those into runtime control flow. Use ``pto.const_expr`` /
``pto.static_range`` only when a branch or loop is intentionally compile-time.
"""

from ptodsl import pto


def _make_softmax_kernel(name: str, *, rows: int, seq: int):
    if rows <= 0:
        raise ValueError("rows must be positive")
    if seq <= 0:
        raise ValueError("seq must be positive")

    @pto.jit(
        name=name,
        kernel_kind="vector",
        target="a5",
        mode="explicit",
        insert_sync=False,
    )
    def kernel(
        scores_ptr: pto.ptr(pto.f32, "gm"),
        out_ptr: pto.ptr(pto.f32, "gm"),
        runtime_rows: pto.i32,
        runtime_seq: pto.i32,
    ):
        packed_rows = pto.elements_per_vreg(pto.f32)
        physical_rows = ((rows + packed_rows - 1) // packed_rows) * packed_rows
        scores_tile_bytes = seq * physical_rows * pto.bytewidth(pto.f32)
        has_rows = runtime_rows > 0

        if has_rows:
            scores_view = pto.make_tensor_view(
                scores_ptr,
                shape=[seq, rows],
                strides=[1, seq],
            )
            out_view = pto.make_tensor_view(
                out_ptr,
                shape=[seq, rows],
                strides=[1, seq],
            )
            scores_tile = pto.alloc_tile(
                shape=[seq, physical_rows],
                dtype=pto.float32,
                addr=pto.const(0, dtype=pto.i64),
                valid_shape=[runtime_seq, runtime_rows],
            )
            out_tile = pto.alloc_tile(
                shape=[seq, physical_rows],
                dtype=pto.float32,
                addr=pto.const(scores_tile_bytes, dtype=pto.i64),
                valid_shape=[runtime_seq, runtime_rows],
            )

            pto.tile.load(scores_view, scores_tile)

            pto.set_flag("MTE2", "V", event_id=0)
            pto.wait_flag("MTE2", "V", event_id=0)

            with pto.tileop():
                remaining_rows = runtime_rows
                for row_base in range(0, runtime_rows, packed_rows):
                    active_rows, remaining_rows = pto.make_mask(pto.f32, remaining_rows)
                    running_max = pto.vlds(scores_tile[0, row_base:])
                    running_sum = pto.vbr(1.0)

                    for col in range(1, runtime_seq, 1):
                        col_vec = pto.vlds(scores_tile[col, row_base:])
                        merged_max = pto.vmax(running_max, col_vec, active_rows)
                        running_delta = pto.vsub(running_max, merged_max, active_rows)
                        scaled_running = pto.vexp(running_delta, active_rows)
                        running_sum_scaled = pto.vmul(scaled_running, running_sum, active_rows)
                        col_delta = pto.vsub(col_vec, merged_max, active_rows)
                        col_exp = pto.vexp(col_delta, active_rows)
                        running_sum = pto.vadd(running_sum_scaled, col_exp, active_rows)
                        running_max = merged_max

                    final_max = running_max
                    final_sum = running_sum

                    for col in range(0, runtime_seq, 1):
                        col_vec = pto.vlds(scores_tile[col, row_base:])
                        out_delta = pto.vsub(col_vec, final_max, active_rows)
                        exp_vec = pto.vexp(out_delta, active_rows)
                        out_vec = pto.vdiv(exp_vec, final_sum, active_rows)
                        pto.vsts(out_vec, out_tile[col, row_base:], active_rows)

            pto.set_flag("V", "MTE3", event_id=0)
            pto.wait_flag("V", "MTE3", event_id=0)

            pto.tile.store(out_tile, out_view)
            pto.pipe_barrier(pto.Pipe.ALL)

    return kernel


SOFTMAX_ROWS64_SEQ128 = _make_softmax_kernel(
    "softmax_rows64_seq128_dsl",
    rows=64,
    seq=128,
)
SOFTMAX_ROWS81_SEQ96 = _make_softmax_kernel(
    "softmax_rows81_seq96_dsl",
    rows=81,
    seq=96,
)


def build():
    return pto.merge_jit_modules(SOFTMAX_ROWS64_SEQ128, SOFTMAX_ROWS81_SEQ96)


if __name__ == "__main__":
    print(build())
