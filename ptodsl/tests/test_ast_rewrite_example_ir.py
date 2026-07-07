#!/usr/bin/env python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
import re


REPO_ROOT = Path(__file__).resolve().parents[2]
from ptodsl import pto
from ptodsl._bootstrap import make_context
from mlir.ir import Module


def expect(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def expect_parse_roundtrip_and_verify(text: str, label: str) -> None:
    with make_context() as ctx:
        parsed = Module.parse(text, ctx)
        parsed.operation.verify()
        roundtrip_text = str(parsed)
    expect(
        roundtrip_text == text,
        f"{label} should survive Module.parse(...) round-trip without textual drift",
    )


def mlir_op_sequence(text: str) -> list[str]:
    ops = []
    for line in text.splitlines():
        stripped = line.strip()
        match = re.search(r"(?:%[\w#]+(?:\s*:\s*[^=]+)?\s*=\s*)?([a-z][\w]*\.[\w_]+)", stripped)
        if match is not None:
            ops.append(match.group(1))
    return ops


def load_example(filename: str, module_name: str):
    demo_path = REPO_ROOT / "ptodsl" / "examples" / filename
    spec = spec_from_file_location(module_name, demo_path)
    expect(spec is not None and spec.loader is not None, f"unable to create import spec for {demo_path}")
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def compare_modules(
    ast_rewrite_text: str,
    explicit_text: str,
    *,
    label: str,
    required_patterns: tuple[str, ...],
) -> None:
    expect_parse_roundtrip_and_verify(ast_rewrite_text, f"AST-rewrite {label} example MLIR")
    expect_parse_roundtrip_and_verify(explicit_text, f"explicit {label} baseline MLIR")
    expect(
        mlir_op_sequence(ast_rewrite_text) == mlir_op_sequence(explicit_text),
        f"{label} AST-rewrite example should emit the same operation sequence as explicit control-flow baseline",
    )
    for pattern in required_patterns:
        expect(
            ast_rewrite_text.count(pattern) == explicit_text.count(pattern),
            f"{label} AST-rewrite example should match explicit baseline count for {pattern}",
        )


def make_explicit_tadd_kernel():
    from ptodsl import scalar

    s = scalar

    @pto.jit(name="explicit_TADD", kernel_kind="vector", target="a5")
    def kernel():
        c0_i64 = pto.const(0, dtype=pto.int64)
        c16 = pto.const(16, dtype=pto.index)
        c4096_i64 = pto.const(4096, dtype=pto.int64)
        c0 = pto.const(0)
        c1 = pto.const(1)
        c64_i32 = pto.const(64, dtype=pto.int32)
        c64 = pto.const(64)

        with pto.tileop():
            ptr_f32_ub = pto.ptr(pto.float32, "ub")
            vf32 = pto.vreg_type(64, pto.float32)
            ptr_src = pto.castptr(c4096_i64, ptr_f32_ub)
            ptr_dst = pto.castptr(c0_i64, ptr_f32_ub)

            with pto.for_(c0, c16, step=c1) as tile_idx:
                mask, _ = pto.plt_b32(c64_i32)
                tile_off = s.muli(tile_idx, c64)
                va = pto.vlds(pto.addptr(ptr_src, tile_off), c0, vf32)
                ptr_dst_tile = pto.addptr(ptr_dst, tile_off)
                vb = pto.vlds(ptr_dst_tile, c0, vf32)
                vc = pto.vadd(va, vb, mask)
                pto.vsts(vc, ptr_dst_tile, c0, mask)

    return kernel


def make_explicit_softmax_kernel(name: str, *, rows: int, seq: int):
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

        with pto.if_(has_rows) as has_rows_br:
            with has_rows_br.then_:
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
                    row_loop = pto.for_(0, runtime_rows, step=packed_rows).carry(remained=runtime_rows)
                    with row_loop:
                        row_base = row_loop.iv
                        remaining_rows = row_loop.remained
                        active_rows, remaining_after_pack = pto.make_mask(pto.f32, remaining_rows)
                        running_max = pto.vlds(scores_tile[0, row_base:])
                        running_sum = pto.vbr(1.0)

                        softmax_loop = pto.for_(1, runtime_seq, step=1).carry(
                            running_max=running_max,
                            running_sum=running_sum,
                        )
                        with softmax_loop:
                            col = softmax_loop.iv
                            running_max = softmax_loop.running_max
                            running_sum = softmax_loop.running_sum
                            col_vec = pto.vlds(scores_tile[col, row_base:])
                            merged_max = pto.vmax(running_max, col_vec, active_rows)
                            running_delta = pto.vsub(running_max, merged_max, active_rows)
                            scaled_running = pto.vexp(running_delta, active_rows)
                            running_sum_scaled = pto.vmul(scaled_running, running_sum, active_rows)
                            col_delta = pto.vsub(col_vec, merged_max, active_rows)
                            col_exp = pto.vexp(col_delta, active_rows)
                            merged_sum = pto.vadd(running_sum_scaled, col_exp, active_rows)
                            softmax_loop.update(running_max=merged_max, running_sum=merged_sum)

                        final_max = softmax_loop.final("running_max")
                        final_sum = softmax_loop.final("running_sum")

                        with pto.for_(0, runtime_seq, step=1) as col:
                            col_vec = pto.vlds(scores_tile[col, row_base:])
                            out_delta = pto.vsub(col_vec, final_max, active_rows)
                            exp_vec = pto.vexp(out_delta, active_rows)
                            out_vec = pto.vdiv(exp_vec, final_sum, active_rows)
                            pto.vsts(out_vec, out_tile[col, row_base:], active_rows)

                        row_loop.update(remained=remaining_after_pack)

                pto.set_flag("V", "MTE3", event_id=0)
                pto.wait_flag("V", "MTE3", event_id=0)

                pto.tile.store(out_tile, out_view)
                pto.pipe_barrier(pto.Pipe.ALL)

    return kernel


def make_explicit_launch_softmax_kernel(name: str, *, rows: int, seq: int):
    @pto.jit(
        name=name,
        target="a5",
        mode="explicit",
        insert_sync=False,
    )
    def kernel(
        scores_ptr: pto.ptr(pto.f32, "gm"),
        out_ptr: pto.ptr(pto.f32, "gm"),
        runtime_seq: pto.i32,
        runtime_rows: pto.i32,
    ):
        lane_num = pto.elements_per_vreg(pto.f32)
        physical_rows = ((rows + lane_num - 1) // lane_num) * lane_num
        scores_tile_bytes = seq * physical_rows * pto.bytewidth(pto.f32)
        total_elems = runtime_rows * runtime_seq

        scores_view = pto.make_tensor_view(
            scores_ptr,
            shape=[1, 1, 1, runtime_seq, runtime_rows],
            strides=[total_elems, total_elems, total_elems, runtime_rows, 1],
        )
        out_view = pto.make_tensor_view(
            out_ptr,
            shape=[1, 1, 1, runtime_seq, runtime_rows],
            strides=[total_elems, total_elems, total_elems, runtime_rows, 1],
        )
        scores_part = pto.partition_view(
            scores_view,
            offsets=[0, 0, 0, 0, 0],
            sizes=[1, 1, 1, runtime_seq, runtime_rows],
        )
        out_part = pto.partition_view(
            out_view,
            offsets=[0, 0, 0, 0, 0],
            sizes=[1, 1, 1, runtime_seq, runtime_rows],
        )

        scores_tile = pto.alloc_tile(
            shape=[seq, physical_rows],
            dtype=pto.float32,
            addr=0,
            valid_shape=[runtime_seq, runtime_rows],
            blayout="RowMajor",
        )
        out_tile = pto.alloc_tile(
            shape=[seq, physical_rows],
            dtype=pto.float32,
            addr=scores_tile_bytes,
            valid_shape=[runtime_seq, runtime_rows],
            blayout="RowMajor",
        )

        pto.tile.load(scores_part, scores_tile)
        out_tile.fill(0.0)

        pto.set_flag("MTE2", "V", event_id=0)
        pto.wait_flag("MTE2", "V", event_id=0)

        with pto.tileop():
            row_loop = pto.for_(0, runtime_rows, step=lane_num).carry(remained=runtime_rows)
            with row_loop:
                row_base = row_loop.iv
                remaining_rows = row_loop.remained
                active_rows, remaining_after_pack = pto.make_mask(pto.f32, remaining_rows)
                running_max = pto.vlds(scores_tile[0, row_base:])
                running_sum = pto.vbr(1.0)

                softmax_loop = pto.for_(1, runtime_seq, step=1).carry(
                    running_max=running_max,
                    running_sum=running_sum,
                )
                with softmax_loop:
                    col = softmax_loop.iv
                    running_max = softmax_loop.running_max
                    running_sum = softmax_loop.running_sum
                    col_vec = pto.vlds(scores_tile[col, row_base:])
                    merged_max = pto.vmax(running_max, col_vec, active_rows)
                    running_delta = pto.vsub(running_max, merged_max, active_rows)
                    scaled_running = pto.vexp(running_delta, active_rows)
                    running_sum_scaled = pto.vmul(scaled_running, running_sum, active_rows)
                    col_delta = pto.vsub(col_vec, merged_max, active_rows)
                    col_exp = pto.vexp(col_delta, active_rows)
                    merged_sum = pto.vadd(running_sum_scaled, col_exp, active_rows)
                    softmax_loop.update(running_max=merged_max, running_sum=merged_sum)

                final_max = softmax_loop.final("running_max")
                final_sum = softmax_loop.final("running_sum")

                with pto.for_(0, runtime_seq, step=1) as col:
                    col_vec = pto.vlds(scores_tile[col, row_base:])
                    out_delta = pto.vsub(col_vec, final_max, active_rows)
                    exp_vec = pto.vexp(out_delta, active_rows)
                    out_vec = pto.vdiv(exp_vec, final_sum, active_rows)
                    pto.vsts(out_vec, out_tile[col, row_base:], active_rows)

                row_loop.update(remained=remaining_after_pack)

        pto.set_flag("V", "MTE3", event_id=0)
        pto.wait_flag("V", "MTE3", event_id=0)

        pto.tile.store(out_tile, out_part)
        pto.pipe_barrier(pto.Pipe.ALL)

    return kernel


def make_explicit_flash_attention_kernel(example):
    @pto.jit(
        name="explicit_flash_attention_kernel",
        target="a5",
        mode="explicit",
        insert_sync=False,
    )
    def kernel(
        Q_ptr: pto.ptr(pto.f32, "gm"),
        K_ptr: pto.ptr(pto.f32, "gm"),
        V_ptr: pto.ptr(pto.f32, "gm"),
        O_ptr: pto.ptr(pto.f32, "gm"),
        batch: pto.i32,
        seq_q: pto.i32,
        seq_k: pto.i32,
        heads: pto.i32,
        dim: pto.i32,
        *,
        BLOCK_Q: pto.const_expr = 128,
        BLOCK_KV: pto.const_expr = 128,
        HEAD_DIM: pto.const_expr = 128,
        CAUSAL: pto.const_expr = False,
        NUM_STAGES: pto.const_expr = 2,
    ):
        q_strides = [seq_q * heads * dim, heads * dim, dim, 1]
        kv_strides = [seq_k * heads * dim, heads * dim, dim, 1]
        o_strides = [seq_q * heads * dim, heads * dim, dim, 1]

        q_view = pto.make_tensor_view(Q_ptr, shape=[batch, seq_q, heads, dim], strides=q_strides)
        k_view = pto.make_tensor_view(K_ptr, shape=[batch, seq_k, heads, dim], strides=kv_strides)
        v_view = pto.make_tensor_view(V_ptr, shape=[batch, seq_k, heads, dim], strides=kv_strides)
        o_view = pto.make_tensor_view(O_ptr, shape=[batch, seq_q, heads, dim], strides=o_strides)

        block_idx = pto.get_block_idx()
        block_num = pto.get_block_num()
        subblock_idx = pto.get_subblock_idx()
        subblock_num = pto.get_subblock_num()
        _ = block_num
        _ = subblock_idx
        _ = subblock_num

        batch_idx = block_idx // heads
        head_idx = block_idx % heads

        q_head = pto.partition_view(q_view, offsets=[batch_idx, 0, head_idx, 0], sizes=[1, seq_q, 1, dim])
        k_head = pto.partition_view(k_view, offsets=[batch_idx, 0, head_idx, 0], sizes=[1, seq_k, 1, dim])
        v_head = pto.partition_view(v_view, offsets=[batch_idx, 0, head_idx, 0], sizes=[1, seq_k, 1, dim])
        o_head = pto.partition_view(o_view, offsets=[batch_idx, 0, head_idx, 0], sizes=[1, seq_q, 1, dim])

        Br = BLOCK_Q
        Bc = BLOCK_KV
        D = HEAD_DIM
        full_br = pto.const(Br)
        full_bc = pto.const(Bc)
        one = pto.const(1)

        q_blocks = (seq_q + Br - 1) // Br
        kv_blocks = (seq_k + Bc - 1) // Bc

        q_mat = pto.alloc_tile(
            shape=[Br, D],
            dtype=pto.f32,
            memory_space=pto.MemorySpace.MAT,
            valid_shape=[full_br, dim],
            blayout="ColMajor",
            slayout="RowMajor",
        )
        k_mat = pto.alloc_tile(
            shape=[Bc, D],
            dtype=pto.f32,
            memory_space=pto.MemorySpace.MAT,
            valid_shape=[full_bc, dim],
            blayout="ColMajor",
            slayout="RowMajor",
        )
        v_mat = pto.alloc_tile(
            shape=[Bc, D],
            dtype=pto.f32,
            memory_space=pto.MemorySpace.MAT,
            valid_shape=[full_bc, dim],
            blayout="ColMajor",
            slayout="RowMajor",
        )

        o_prev_tile = pto.alloc_tile(shape=[Br, D], dtype=pto.f32, valid_shape=[full_br, dim])
        o_next_tile = pto.alloc_tile(shape=[Br, D], dtype=pto.f32, valid_shape=[full_br, dim])
        m_prev_tile = pto.alloc_tile(shape=[Br, 1], dtype=pto.f32, valid_shape=[full_br, one], blayout="ColMajor")
        m_next_tile = pto.alloc_tile(shape=[Br, 1], dtype=pto.f32, valid_shape=[full_br, one], blayout="ColMajor")
        l_prev_tile = pto.alloc_tile(shape=[Br, 1], dtype=pto.f32, valid_shape=[full_br, one], blayout="ColMajor")
        l_next_tile = pto.alloc_tile(shape=[Br, 1], dtype=pto.f32, valid_shape=[full_br, one], blayout="ColMajor")

        s_tile = pto.alloc_tile(shape=[Br, Bc], dtype=pto.f32, valid_shape=[full_br, full_bc])
        p_tile = pto.alloc_tile(shape=[Br, Bc], dtype=pto.f32, valid_shape=[full_br, full_bc])
        p_mat = pto.alloc_tile(
            shape=[Br, Bc],
            dtype=pto.f32,
            memory_space=pto.MemorySpace.MAT,
            valid_shape=[full_br, full_bc],
            blayout="ColMajor",
            slayout="RowMajor",
        )
        pv_tile = pto.alloc_tile(shape=[Br, D], dtype=pto.f32, valid_shape=[full_br, dim])
        alpha_tile = pto.alloc_tile(shape=[Br, 1], dtype=pto.f32, valid_shape=[full_br, one], blayout="ColMajor")
        beta_tile = pto.alloc_tile(shape=[Br, 1], dtype=pto.f32, valid_shape=[full_br, one], blayout="ColMajor")

        q_l0a = pto.alloc_tile(
            shape=[Br, D],
            dtype=pto.f32,
            memory_space=pto.MemorySpace.LEFT,
            valid_shape=[full_br, dim],
        )
        p_l0a = pto.alloc_tile(
            shape=[Br, Bc],
            dtype=pto.f32,
            memory_space=pto.MemorySpace.LEFT,
            valid_shape=[full_br, full_bc],
        )
        rhs_l0b = pto.alloc_tile(
            shape=[Bc, D],
            dtype=pto.f32,
            memory_space=pto.MemorySpace.RIGHT,
            valid_shape=[full_bc, dim],
        )
        qk_acc_tile = pto.alloc_tile(
            shape=[Br, Bc],
            dtype=pto.f32,
            memory_space=pto.MemorySpace.ACC,
            valid_shape=[full_br, full_bc],
        )
        pv_acc_tile = pto.alloc_tile(
            shape=[Br, D],
            dtype=pto.f32,
            memory_space=pto.MemorySpace.ACC,
            valid_shape=[full_br, dim],
        )

        meta_tile = pto.alloc_tile(shape=[1, 8], dtype=pto.i32, valid_shape=[1, 3])
        meta_ptr = meta_tile.as_ptr()

        with pto.for_(0, q_blocks, step=1) as qi:
            q_rows = example._block_valid_extent(seq_q, qi, Br)
            q_part = pto.partition_view(q_head, offsets=[0, qi * Br, 0, 0], sizes=[1, q_rows, 1, dim])
            o_part = pto.partition_view(o_head, offsets=[0, qi * Br, 0, 0], sizes=[1, q_rows, 1, dim])

            q_mat.valid_shape = [q_rows, dim]
            o_prev_tile.valid_shape = [q_rows, dim]
            o_next_tile.valid_shape = [q_rows, dim]
            m_prev_tile.valid_shape = [q_rows, one]
            m_next_tile.valid_shape = [q_rows, one]
            l_prev_tile.valid_shape = [q_rows, one]
            l_next_tile.valid_shape = [q_rows, one]
            alpha_tile.valid_shape = [q_rows, one]
            beta_tile.valid_shape = [q_rows, one]
            p_mat.valid_shape = [q_rows, full_bc]
            pv_tile.valid_shape = [q_rows, dim]
            q_l0a.valid_shape = [q_rows, dim]

            pto.tile.load(q_part, q_mat)
            m_prev_tile.fill(float("-inf"))
            l_prev_tile.fill(0.0)
            o_prev_tile.fill(0.0)

            kv_loop = pto.for_(0, kv_blocks, step=1).carry(
                m=m_prev_tile,
                l=l_prev_tile,
                o=o_prev_tile,
            )
            with kv_loop:
                kj = kv_loop.iv
                m_cur = kv_loop.m
                l_cur = kv_loop.l
                o_cur = kv_loop.o
                kv_rows = example._block_valid_extent(seq_k, kj, Bc)
                k_part = pto.partition_view(k_head, offsets=[0, kj * Bc, 0, 0], sizes=[1, kv_rows, 1, dim])
                v_part = pto.partition_view(v_head, offsets=[0, kj * Bc, 0, 0], sizes=[1, kv_rows, 1, dim])

                k_mat.valid_shape = [kv_rows, dim]
                v_mat.valid_shape = [kv_rows, dim]
                s_tile.valid_shape = [q_rows, kv_rows]
                p_tile.valid_shape = [q_rows, kv_rows]
                p_mat.valid_shape = [q_rows, kv_rows]
                pv_tile.valid_shape = [q_rows, dim]
                p_l0a.valid_shape = [q_rows, kv_rows]
                rhs_l0b.valid_shape = [kv_rows, dim]
                qk_acc_tile.valid_shape = [q_rows, kv_rows]
                pv_acc_tile.valid_shape = [q_rows, dim]

                example.kv_block_process(
                    q_mat,
                    k_part,
                    v_part,
                    k_mat,
                    v_mat,
                    o_cur,
                    o_next_tile,
                    m_cur,
                    l_cur,
                    m_next_tile,
                    l_next_tile,
                    s_tile,
                    p_tile,
                    p_mat,
                    pv_tile,
                    alpha_tile,
                    beta_tile,
                    q_l0a,
                    p_l0a,
                    rhs_l0b,
                    qk_acc_tile,
                    pv_acc_tile,
                    meta_ptr,
                )
                kv_loop.update(m=m_next_tile, l=l_next_tile, o=o_next_tile)

            pto.tile.store(kv_loop.final("o"), o_part)

        _ = CAUSAL
        _ = NUM_STAGES

    return kernel


EXPLICIT_TADD = make_explicit_tadd_kernel()
EXPLICIT_SOFTMAX_ROWS64_SEQ128 = make_explicit_softmax_kernel(
    "explicit_softmax_rows64_seq128",
    rows=64,
    seq=128,
)
EXPLICIT_SOFTMAX_ROWS81_SEQ96 = make_explicit_softmax_kernel(
    "explicit_softmax_rows81_seq96",
    rows=81,
    seq=96,
)
EXPLICIT_LAUNCH_SOFTMAX_ROWS64_SEQ128 = make_explicit_launch_softmax_kernel(
    "explicit_launch_softmax_rows64_seq128",
    rows=64,
    seq=128,
)
EXPLICIT_LAUNCH_SOFTMAX_ROWS81_SEQ96 = make_explicit_launch_softmax_kernel(
    "explicit_launch_softmax_rows81_seq96",
    rows=81,
    seq=96,
)


def explicit_softmax_module():
    return pto.merge_jit_modules(EXPLICIT_SOFTMAX_ROWS64_SEQ128, EXPLICIT_SOFTMAX_ROWS81_SEQ96)


def explicit_launch_softmax_module():
    return pto.merge_jit_modules(EXPLICIT_LAUNCH_SOFTMAX_ROWS64_SEQ128, EXPLICIT_LAUNCH_SOFTMAX_ROWS81_SEQ96)


def main() -> None:
    tadd_example = load_example("tadd_dsl.py", "ptodsl_tadd_ast_rewrite_example")
    compare_modules(
        str(tadd_example.build()),
        str(EXPLICIT_TADD.mlir_module()),
        label="tadd_dsl",
        required_patterns=("scf.for", "pto.vadd", "pto.vsts"),
    )

    softmax_example = load_example("softmax_dsl.py", "ptodsl_softmax_ast_rewrite_example")
    compare_modules(
        str(softmax_example.build()),
        str(explicit_softmax_module()),
        label="softmax_dsl",
        required_patterns=("scf.if", "scf.for", "iter_args(", "scf.yield", "pto.vexp", "pto.vdiv"),
    )

    launch_softmax_example = load_example(
        "flash_attention_softmax_launch.py",
        "ptodsl_flash_attention_softmax_launch_ast_rewrite_example",
    )
    compare_modules(
        str(launch_softmax_example.emit_mlir()),
        str(explicit_launch_softmax_module()),
        label="flash_attention_softmax_launch",
        required_patterns=("scf.for", "iter_args(", "scf.yield", "pto.vexp", "pto.vdiv"),
    )

    flash_attention_example = load_example(
        "flash_attention_sketch.py",
        "ptodsl_flash_attention_ast_rewrite_example",
    )
    explicit_flash_attention_kernel = make_explicit_flash_attention_kernel(flash_attention_example)
    compare_modules(
        flash_attention_example.flash_attention_kernel.compile(
            BLOCK_Q=64,
            BLOCK_KV=128,
            HEAD_DIM=128,
            CAUSAL=True,
        ).mlir_text(),
        explicit_flash_attention_kernel.compile(
            BLOCK_Q=64,
            BLOCK_KV=128,
            HEAD_DIM=128,
            CAUSAL=True,
        ).mlir_text(),
        label="flash_attention_sketch",
        required_patterns=("scf.for", "iter_args(", "scf.yield", "func.call", "pto.mad"),
    )

    print("ptodsl_ast_rewrite_example_ir: PASS")


if __name__ == "__main__":
    main()
