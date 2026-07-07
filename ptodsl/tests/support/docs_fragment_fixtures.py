#!/usr/bin/env python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

from dataclasses import dataclass
from textwrap import dedent


SNIPPET_PLACEHOLDER = "__PTODSL_DOC_SNIPPET__"


@dataclass(frozen=True)
class FragmentFixture:
    template: str


def _fixture(template: str) -> FragmentFixture:
    return FragmentFixture(template=dedent(template).strip("\n"))


def render_fragment_fixture(fixture: FragmentFixture, snippet: str) -> str:
    rendered_lines: list[str] = []
    placeholder_count = 0
    snippet_lines = snippet.rstrip("\n").splitlines()

    for line in fixture.template.splitlines():
        if SNIPPET_PLACEHOLDER not in line:
            rendered_lines.append(line)
            continue

        placeholder_count += 1
        if line.strip() != SNIPPET_PLACEHOLDER:
            raise ValueError(
                f"fixture placeholder must occupy its own line: {line!r}"
            )

        indent = line[: line.index(SNIPPET_PLACEHOLDER)]
        rendered_lines.extend(
            f"{indent}{snippet_line}" if snippet_line else ""
            for snippet_line in snippet_lines
        )

    if placeholder_count != 1:
        raise ValueError(
            f"fixture must contain exactly one placeholder line, found {placeholder_count}"
        )

    return "\n".join(rendered_lines) + "\n"


FRAGMENT_FIXTURES = {
    "type_system.scalar_expr": _fixture(
        f"""
        @pto.jit(target="a5")
        def type_system_scalar_expr_probe():
            {SNIPPET_PLACEHOLDER}
        """
    ),
    "type_system.low_precision_types": _fixture(
        f"""
        @pto.jit(target="a5")
        def type_system_low_precision_types_probe(
            *,
            BLOCK: pto.const_expr = 128,
        ):
            {SNIPPET_PLACEHOLDER}
            _ = lp_ptr
            _ = lp_vreg_ty
        """
    ),
    "type_system.tensor_view": _fixture(
        f"""
        @pto.jit(target="a5")
        def type_system_tensor_view_probe(
            A_ptr: pto.ptr(pto.f32, "gm"),
            rows: pto.i32,
            cols: pto.i32,
            *,
            BLOCK: pto.const_expr = 128,
        ):
            N = rows
            A = pto.make_tensor_view(A_ptr, shape=[rows, cols], strides=[cols, 1])
            {SNIPPET_PLACEHOLDER}
        """
    ),
    "type_system.partition_view": _fixture(
        f"""
        @pto.jit(target="a5")
        def type_system_partition_view_probe(
            A_ptr: pto.ptr(pto.f32, "gm"),
            rows: pto.i32,
            cols: pto.i32,
            *,
            BLOCK: pto.const_expr = 128,
        ):
            dim = cols
            row_offset = 0
            tv = pto.make_tensor_view(A_ptr, shape=[rows, cols], strides=[cols, 1])
            {SNIPPET_PLACEHOLDER}
        """
    ),
    "type_system.tile_alloc": _fixture(
        f"""
        @pto.jit(target="a5")
        def type_system_tile_alloc_probe(
            *,
            BLOCK: pto.const_expr = 128,
            Br: pto.const_expr = 16,
            Bc: pto.const_expr = 16,
            dim: pto.const_expr = 16,
        ):
            {SNIPPET_PLACEHOLDER}
        """
    ),
    "type_system.tile_methods": _fixture(
        f"""
        @pto.jit(target="a5")
        def type_system_tile_methods_probe(
            *,
            Br: pto.const_expr = 16,
            Bc: pto.const_expr = 16,
            dim: pto.const_expr = 16,
        ):
            m_prev_tile = pto.alloc_tile(shape=[Br, 1], dtype=pto.f32, blayout="ColMajor")
            l_prev_tile = pto.alloc_tile(shape=[Br, 1], dtype=pto.f32, blayout="ColMajor")
            q_tile = pto.alloc_tile(shape=[Br, dim], dtype=pto.f32)
            k_tile = pto.alloc_tile(shape=[Bc, dim], dtype=pto.f32)
            meta_tile = pto.alloc_tile(shape=[1, 8], dtype=pto.i32, valid_shape=[pto.const(1), pto.const(3)])
            tail_tile = pto.alloc_tile(shape=[dim], dtype=pto.f32, valid_shape=[pto.const(dim)])
            {SNIPPET_PLACEHOLDER}
        """
    ),
    "type_system.tile_reshape": _fixture(
        f"""
        @pto.jit(target="a5")
        def type_system_tile_reshape_probe(
            *,
            BR: pto.const_expr = 8,
            BC: pto.const_expr = 64,
        ):
            {SNIPPET_PLACEHOLDER}
        """
    ),
    "type_system.vreg_bitcast": _fixture(
        f"""
        @pto.jit(target="a5")
        def type_system_vreg_bitcast_probe(
            *,
            BLOCK: pto.const_expr = 128,
        ):
            tile = pto.alloc_tile(shape=[2, BLOCK], dtype=pto.f32)
            row = 0
            fvec = pto.vlds(tile[row, 0:])
            {SNIPPET_PLACEHOLDER}
        """
    ),
    "type_system.vreg_bitcast_ptr": _fixture(
        f"""
        @pto.jit(target="a5")
        def type_system_vreg_bitcast_ptr_probe(
            *,
            BLOCK: pto.const_expr = 128,
        ):
            tile = pto.alloc_tile(shape=[2, BLOCK], dtype=pto.f32)
            ptr = tile.as_ptr()
            offset = pto.const(0)
            {SNIPPET_PLACEHOLDER}
        """
    ),
    "type_system.mask_bitcast": _fixture(
        f"""
        @pto.jit(target="a5")
        def type_system_mask_bitcast_probe():
            mask_b8, _ = pto.make_mask(pto.i8, pto.const(256, dtype=pto.i32))
            mask16, _ = pto.make_mask(pto.f16, pto.const(128, dtype=pto.i32))
            {SNIPPET_PLACEHOLDER}
        """
    ),
    "type_system.make_mask": _fixture(
        f"""
        @pto.jit(target="a5")
        def type_system_make_mask_probe():
            tail_count = pto.const(16, dtype=pto.i32)
            {SNIPPET_PLACEHOLDER}
        """
    ),
    "quick_start.make_tensor_view": _fixture(
        f"""
        @pto.jit(target="a5")
        def quick_start_make_tensor_view_probe(
            A_ptr: pto.ptr(pto.f32, "gm"),
            rows: pto.i32,
            cols: pto.i32,
        ):
            {SNIPPET_PLACEHOLDER}
        """
    ),
    "quick_start.alloc_tile": _fixture(
        f"""
        @pto.jit(target="a5")
        def quick_start_alloc_tile_probe(
            *,
            BLOCK: pto.const_expr = 128,
        ):
            {SNIPPET_PLACEHOLDER}
        """
    ),
    "quick_start.partition_view": _fixture(
        f"""
        @pto.jit(target="a5")
        def quick_start_partition_view_probe(
            A_ptr: pto.ptr(pto.f32, "gm"),
            rows: pto.i32,
            cols: pto.i32,
        ):
            a_view = pto.make_tensor_view(A_ptr, shape=[rows, cols], strides=[cols, 1])
            {SNIPPET_PLACEHOLDER}
        """
    ),
    "quick_start.tile_io": _fixture(
        f"""
        @pto.jit(target="a5")
        def quick_start_tile_io_probe(
            A_ptr: pto.ptr(pto.f32, "gm"),
            O_ptr: pto.ptr(pto.f32, "gm"),
            rows: pto.i32,
            cols: pto.i32,
            *,
            BLOCK: pto.const_expr = 128,
        ):
            a_view = pto.make_tensor_view(A_ptr, shape=[rows, cols], strides=[cols, 1])
            o_view = pto.make_tensor_view(O_ptr, shape=[rows, cols], strides=[cols, 1])
            a_part = pto.partition_view(a_view, offsets=[0, 0], sizes=[rows, cols])
            o_part = pto.partition_view(o_view, offsets=[0, 0], sizes=[rows, cols])
            a_tile = pto.alloc_tile(shape=[1, BLOCK], dtype=pto.f32)
            o_tile = pto.alloc_tile(shape=[1, BLOCK], dtype=pto.f32)
            {SNIPPET_PLACEHOLDER}
        """
    ),
    "launch.flash_attention_wrapper": _fixture(
        f"""
        class _FakeTensor:
            _next_ptr = 4096

            def __init__(self, shape):
                self.shape = tuple(shape)
                self._ptr = _FakeTensor._next_ptr
                _FakeTensor._next_ptr += 4096

            def data_ptr(self):
                return self._ptr

            def new_empty(self, shape):
                return _FakeTensor(shape)


        @pto.jit(target="a5")
        def flash_attention_kernel(
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
            CAUSAL: pto.const_expr = False,
        ):
            pto.get_block_idx()


        batch = 2
        heads = 3
        seq_q = 4
        seq_k = 4
        dim = 8
        stream = object()
        Q = _FakeTensor((batch, seq_q, heads, dim))
        K = _FakeTensor((batch, seq_k, heads, dim))
        V = _FakeTensor((batch, seq_k, heads, dim))

        {SNIPPET_PLACEHOLDER}

        O = flash_attention(Q, K, V, causal=False)
        assert O.shape == Q.shape
        assert len(PTODSL_DOC_LAUNCH_RECORDS) == 1
        record = PTODSL_DOC_LAUNCH_RECORDS[0]
        assert record.grid == batch * heads
        assert record.stream is stream
        assert len(record.args) == 9
        assert record.args[0] == Q.data_ptr()
        assert record.args[1] == K.data_ptr()
        assert record.args[2] == V.data_ptr()
        assert record.args[3] == O.data_ptr()
        assert record.args[4:] == (batch, seq_q, seq_k, heads, dim)
        assert record.marshaled_arg_count == 9
        """
    ),
    "launch.blocked_copy_compile_and_launch": _fixture(
        f"""
        @pto.jit(target="a5")
        def blocked_copy(
            A_ptr: pto.ptr(pto.f32, "gm"),
            O_ptr: pto.ptr(pto.f32, "gm"),
            rows: pto.i32,
            cols: pto.i32,
            *,
            BLOCK: pto.const_expr = 128,
        ):
            a_view = pto.make_tensor_view(A_ptr, shape=[rows, cols], strides=[cols, 1])
            o_view = pto.make_tensor_view(O_ptr, shape=[rows, cols], strides=[cols, 1])
            tile = pto.alloc_tile(shape=[1, BLOCK], dtype=pto.f32)
            for row in range(0, rows, 1):
                a_part = pto.partition_view(a_view, offsets=[row, 0], sizes=[1, cols])
                o_part = pto.partition_view(o_view, offsets=[row, 0], sizes=[1, cols])
                pto.tile.load(a_part, tile)
                pto.tile.store(tile, o_part)


        {SNIPPET_PLACEHOLDER}

        assert len(PTODSL_DOC_LAUNCH_RECORDS) == 1
        record = PTODSL_DOC_LAUNCH_RECORDS[0]
        assert record.grid == 1
        assert record.stream is None
        assert len(record.args) == 4
        assert record.args == (A.ctypes.data, O.ctypes.data, 4, 128)
        assert record.marshaled_arg_count == 4
        """
    ),
    "launch.generic_compile_and_launch": _fixture(
        f"""
        import numpy as np


        @pto.jit(target="a5")
        def kernel_name(
            tensor_1_ptr: pto.ptr(pto.f32, "gm"),
            tensor_2_ptr: pto.ptr(pto.f32, "gm"),
            rows: pto.i32,
            cols: pto.i32,
            *,
            CONST_A: pto.const_expr = 128,
            CONST_B: pto.const_expr = 64,
        ):
            pto.get_block_idx()


        grid = 2
        stream = object()

        {SNIPPET_PLACEHOLDER}

        assert len(PTODSL_DOC_LAUNCH_RECORDS) == 1
        record = PTODSL_DOC_LAUNCH_RECORDS[0]
        assert record.grid == grid
        assert record.stream is stream
        assert len(record.args) == 4
        assert record.args == (A.ctypes.data, O.ctypes.data, 4, 128)
        assert record.marshaled_arg_count == 4
        """
    ),
    "launch.source_backed_tadd": _fixture(
        f"""
        grid = 2
        stream = object()
        A = 0x1000
        B = 0x2000
        O = 0x3000
        numel = 128

        {SNIPPET_PLACEHOLDER}

        assert compiled.ir_function_name == "tadd_entry"
        assert compiled.build_metadata()["source_path"].endswith("kernels/tadd_entry.pto")
        assert len(PTODSL_DOC_LAUNCH_RECORDS) == 1
        record = PTODSL_DOC_LAUNCH_RECORDS[0]
        assert record.grid == grid
        assert record.stream is stream
        assert record.args == (A, B, O, numel)
        assert record.marshaled_arg_count == 4
        """
    ),
    "launch.mat_add_wrapper": _fixture(
        f"""
        import numpy as np


        @pto.jit(target="a5")
        def mat_add(
            A_ptr: pto.ptr(pto.f32, "gm"),
            B_ptr: pto.ptr(pto.f32, "gm"),
            O_ptr: pto.ptr(pto.f32, "gm"),
            batch: pto.i32,
            rows: pto.i32,
            cols: pto.i32,
            *,
            BLOCK_M: pto.const_expr = 64,
            BLOCK_N: pto.const_expr = 128,
        ):
            pto.get_block_idx()


        {SNIPPET_PLACEHOLDER}

        A = np.random.randn(2, 64, 128).astype(np.float32)
        B = np.random.randn(2, 64, 128).astype(np.float32)
        O = mat_add_wrapper(A, B, stream=None)
        assert O.shape == A.shape
        assert len(PTODSL_DOC_LAUNCH_RECORDS) == 1
        record = PTODSL_DOC_LAUNCH_RECORDS[0]
        assert record.grid == A.shape[0]
        assert record.stream is None
        assert len(record.args) == 6
        assert record.args == (A.ctypes.data, B.ctypes.data, O.ctypes.data, 2, 64, 128)
        assert record.marshaled_arg_count == 6
        """
    ),
    "launch.gemm_wrapper": _fixture(
        f"""
        import numpy as np


        @pto.jit(target="a5")
        def gemm(
            A_ptr: pto.ptr(pto.f32, "gm"),
            B_ptr: pto.ptr(pto.f32, "gm"),
            O_ptr: pto.ptr(pto.f32, "gm"),
            rows: pto.i32,
            reduce_dim: pto.i32,
            cols: pto.i32,
            *,
            BLOCK_M: pto.const_expr = 64,
            BLOCK_K: pto.const_expr = 64,
            BLOCK_N: pto.const_expr = 64,
        ):
            pto.get_block_idx()


        {SNIPPET_PLACEHOLDER}

        A = np.random.randn(64, 32).astype(np.float32)
        B = np.random.randn(32, 16).astype(np.float32)
        O = gemm_wrapper(A, B, stream=None)
        assert O.shape == (A.shape[0], B.shape[1])
        assert len(PTODSL_DOC_LAUNCH_RECORDS) == 1
        record = PTODSL_DOC_LAUNCH_RECORDS[0]
        assert record.grid == 1
        assert record.stream is None
        assert len(record.args) == 6
        assert record.args == (A.ctypes.data, B.ctypes.data, O.ctypes.data, 64, 32, 16)
        assert record.marshaled_arg_count == 6
        """
    ),
    "control_flow.basic_for": _fixture(
        f"""
        @pto.jit(target="a5")
        def control_flow_basic_for_probe(
            A_ptr: pto.ptr(pto.f32, "gm"),
            O_ptr: pto.ptr(pto.f32, "gm"),
            rows: pto.i32,
            cols: pto.i32,
            *,
            BLOCK: pto.const_expr = 8,
        ):
            start = pto.const(0, dtype=pto.i32)
            stop = pto.const(BLOCK, dtype=pto.i32)
            step = 1
            a_view = pto.make_tensor_view(A_ptr, shape=[rows, cols], strides=[cols, 1])
            o_view = pto.make_tensor_view(O_ptr, shape=[rows, cols], strides=[cols, 1])
            tile = pto.alloc_tile(shape=[1, BLOCK], dtype=pto.f32)
            {SNIPPET_PLACEHOLDER}
        """
    ),
    "control_flow.compare_loops": _fixture(
        f"""
        @pto.jit(target="a5")
        def control_flow_compare_loops_probe(
            A_ptr: pto.ptr(pto.f32, "gm"),
            O_ptr: pto.ptr(pto.f32, "gm"),
            rows: pto.i32,
            cols: pto.i32,
            *,
            BLOCK: pto.const_expr = 8,
        ):
            num_blocks = rows
            a_view = pto.make_tensor_view(A_ptr, shape=[rows, cols], strides=[cols, 1])
            o_view = pto.make_tensor_view(O_ptr, shape=[rows, cols], strides=[cols, 1])
            tile = pto.alloc_tile(shape=[1, BLOCK], dtype=pto.f32)
            {SNIPPET_PLACEHOLDER}
        """
    ),
    "control_flow.nested_loops": _fixture(
        f"""
        @pto.jit(target="a5")
        def control_flow_nested_loops_probe(
            rows: pto.i32,
            cols: pto.i32,
            *,
            BLOCK: pto.const_expr = 8,
        ):
            tile = pto.alloc_tile(shape=[2, BLOCK], dtype=pto.f32, valid_shape=[rows, cols])
            {SNIPPET_PLACEHOLDER}
        """
    ),
    "control_flow.carry_pingpong": _fixture(
        f"""
        @pto.jit(target="a5")
        def control_flow_carry_pingpong_probe(
            *,
            Br: pto.const_expr = 16,
            num_blocks: pto.const_expr = 4,
        ):
            {SNIPPET_PLACEHOLDER}
        """
    ),
    "scalar_ops.tile_access": _fixture(
        f"""
        @pto.jit(target="a5")
        def scalar_ops_tile_access_probe():
            tile = pto.alloc_tile(shape=[1, 8], dtype=pto.f32, valid_shape=[1, 4])
            row = 0
            col = 0
            value = pto.const(1.0, dtype=pto.f32)
            ptr = tile.as_ptr()
            offset = 0
            {SNIPPET_PLACEHOLDER}
        """
    ),
    "tail.chunked_inner_loop": _fixture(
        f"""
        @pto.jit(target="a5")
        def tail_chunked_inner_loop_probe(*, BLOCK: pto.const_expr = 128):
            cols = pto.const(BLOCK, dtype=pto.i32)
            tile = pto.alloc_tile(shape=[1, BLOCK], dtype=pto.f32)
            out_tile = pto.alloc_tile(shape=[1, BLOCK], dtype=pto.f32)
            for r in pto.static_range(1):
                {SNIPPET_PLACEHOLDER}
        """
    ),
    "tail.vector_pattern": _fixture(
        f"""
        @pto.jit(target="a5")
        def tail_vector_pattern_probe(*, BLOCK: pto.const_expr = 128):
            rows = pto.const(1, dtype=pto.i32)
            cols = pto.const(BLOCK, dtype=pto.i32)
            tile = pto.alloc_tile(shape=[1, BLOCK], dtype=pto.f32)
            out_tile = pto.alloc_tile(shape=[1, BLOCK], dtype=pto.f32)
            {SNIPPET_PLACEHOLDER}
        """
    ),
    "tail.tileop_helper": _fixture(
        f"""
        {SNIPPET_PLACEHOLDER}


        @pto.jit(target="a5")
        def tail_simd_helper_probe(*, BLOCK: pto.const_expr = 128):
            a_tile = pto.alloc_tile(shape=[1, BLOCK], dtype=pto.f32)
            b_tile = pto.alloc_tile(shape=[1, BLOCK], dtype=pto.f32)
            o_tile = pto.alloc_tile(shape=[1, BLOCK], dtype=pto.f32)
            add_rows_with_tail(
                a_tile,
                b_tile,
                o_tile,
                pto.const(1, dtype=pto.i32),
                pto.const(BLOCK, dtype=pto.i32),
            )
        """
    ),
    "kernel_entry.direct_l3_call": _fixture(
        f"""
        {SNIPPET_PLACEHOLDER}


        kernel_entry_direct_l3_call_probe = my_kernel
        """
    ),
    "kernel_entry.explicit_signature": _fixture(
        f"""
        {SNIPPET_PLACEHOLDER}


        @pto.jit(target="a5", mode="explicit")
        def kernel_entry_explicit_signature_probe(
            A_ptr: pto.ptr(pto.f32, "gm"),
            rows: pto.i32,
            cols: pto.i32,
            *,
            BLOCK: pto.const_expr = 16,
        ):
            view = pto.make_tensor_view(A_ptr, shape=[rows, cols], strides=[cols, 1])
            part = pto.partition_view(view, offsets=[0, 0], sizes=[1, BLOCK])
            tile = pto.alloc_tile(shape=[1, BLOCK], dtype=pto.f32)
            scratch = pto.alloc_tile(shape=[8, BLOCK], dtype=pto.f32, memory_space=pto.MemorySpace.LEFT)
            my_orchestration_helper(part, tile, scratch, tile.as_ptr(), pto.const(0, dtype=pto.i32))
        """
    ),
    "kernel_entry.explicit_body": _fixture(
        f"""
        @pto.tileop
        def qk_matmul(q_tile: pto.Tile, k_tile: pto.Tile, s_tile: pto.Tile):
            return


        @pto.tileop
        def online_softmax(s_tile: pto.Tile, o_tile: pto.Tile, rows: pto.i32, cols: pto.i32):
            return


        {SNIPPET_PLACEHOLDER}


        @pto.jit(target="a5", mode="explicit")
        def kernel_entry_explicit_body_probe(
            K_ptr: pto.ptr(pto.f16, "gm"),
            V_ptr: pto.ptr(pto.f16, "gm"),
            O_ptr: pto.ptr(pto.f32, "gm"),
            *,
            ROWS: pto.const_expr = 8,
            COLS: pto.const_expr = 16,
        ):
            k_view = pto.make_tensor_view(K_ptr, shape=[ROWS, COLS], strides=[COLS, 1])
            v_view = pto.make_tensor_view(V_ptr, shape=[ROWS, COLS], strides=[COLS, 1])
            o_view = pto.make_tensor_view(O_ptr, shape=[ROWS, COLS], strides=[COLS, 1])
            q_tile = pto.alloc_tile(shape=[ROWS, COLS], dtype=pto.f16)
            k_tile = pto.alloc_tile(shape=[ROWS, COLS], dtype=pto.f16)
            v_tile = pto.alloc_tile(shape=[ROWS, COLS], dtype=pto.f16)
            s_tile = pto.alloc_tile(shape=[ROWS, COLS], dtype=pto.f32)
            o_tile = pto.alloc_tile(shape=[ROWS, COLS], dtype=pto.f32)
            k_part = pto.partition_view(k_view, offsets=[0, 0], sizes=[ROWS, COLS])
            v_part = pto.partition_view(v_view, offsets=[0, 0], sizes=[ROWS, COLS])
            o_part = pto.partition_view(o_view, offsets=[0, 0], sizes=[ROWS, COLS])
            process_block(q_tile, k_part, v_part, k_tile, v_tile, s_tile, o_tile, o_part, ROWS, COLS)
        """
    ),
    "kernel_entry.inline_explicit_scope": _fixture(
        f"""
        @pto.jit(target="a5", mode="explicit")
        def kernel_entry_inline_explicit_scope_probe(
            A_ptr: pto.ptr(pto.f32, "gm"),
            O_ptr: pto.ptr(pto.f32, "gm"),
            *,
            BLOCK: pto.const_expr = 16,
        ):
            a_view = pto.make_tensor_view(A_ptr, shape=[1, BLOCK], strides=[BLOCK, 1])
            o_view = pto.make_tensor_view(O_ptr, shape=[1, BLOCK], strides=[BLOCK, 1])
            part = pto.partition_view(a_view, offsets=[0, 0], sizes=[1, BLOCK])
            out_part = pto.partition_view(o_view, offsets=[0, 0], sizes=[1, BLOCK])
            tile = pto.alloc_tile(shape=[1, BLOCK], dtype=pto.f32, valid_shape=[1, BLOCK])
            row_bytes = BLOCK * pto.bytewidth(pto.f32)
            pto.mte_load(part.as_ptr(), tile.as_ptr(), 0, row_bytes,
                         nburst=(1, row_bytes, row_bytes))
            pto.pipe_barrier(pto.Pipe.ALL)
            pto.mte_store(tile.as_ptr(), out_part.as_ptr(), row_bytes,
                          nburst=(1, row_bytes, row_bytes))
        """
    ),
    "kernel_entry.cube_signature": _fixture(
        f"""
        {SNIPPET_PLACEHOLDER}


        @pto.jit(target="a5", mode="explicit")
        def kernel_entry_cube_signature_probe(
            *,
            BLOCK_M: pto.const_expr = 16,
            BLOCK_K: pto.const_expr = 16,
            BLOCK_N: pto.const_expr = 16,
        ):
            input_tile = pto.alloc_tile(shape=[BLOCK_M, BLOCK_K], dtype=pto.f16, valid_shape=[BLOCK_M, BLOCK_K])
            output_tile = pto.alloc_tile(shape=[BLOCK_M, BLOCK_N], dtype=pto.f32, valid_shape=[BLOCK_M, BLOCK_N])
            left_scratch = pto.alloc_tile(shape=[BLOCK_M, BLOCK_K], dtype=pto.f16, memory_space=pto.MemorySpace.LEFT, valid_shape=[BLOCK_M, BLOCK_K])
            right_scratch = pto.alloc_tile(shape=[BLOCK_K, BLOCK_N], dtype=pto.f16, memory_space=pto.MemorySpace.RIGHT, valid_shape=[BLOCK_K, BLOCK_N])
            acc_scratch = pto.alloc_tile(shape=[BLOCK_M, BLOCK_N], dtype=pto.f32, memory_space=pto.MemorySpace.ACC, valid_shape=[BLOCK_M, BLOCK_N])
            my_cube_kernel(input_tile, output_tile, left_scratch, right_scratch, acc_scratch)
        """
    ),
    "kernel_entry.tileop_signature": _fixture(
        f"""
        {SNIPPET_PLACEHOLDER}


        @pto.jit(target="a5")
        def kernel_entry_simd_signature_probe(*, BLOCK: pto.const_expr = 128):
            input_tile = pto.alloc_tile(shape=[1, BLOCK], dtype=pto.f32)
            output_tile = pto.alloc_tile(shape=[1, BLOCK], dtype=pto.f32)
            my_tileop_kernel(input_tile, output_tile, pto.const(1, dtype=pto.i32), pto.const(BLOCK, dtype=pto.i32))
        """
    ),
    "kernel_entry.tileop_body": _fixture(
        f"""
        {SNIPPET_PLACEHOLDER}


        @pto.jit(target="a5")
        def kernel_entry_simd_body_probe(*, BLOCK: pto.const_expr = 128):
            a_tile = pto.alloc_tile(shape=[1, BLOCK], dtype=pto.f32)
            b_tile = pto.alloc_tile(shape=[1, BLOCK], dtype=pto.f32)
            o_tile = pto.alloc_tile(shape=[1, BLOCK], dtype=pto.f32)
            add_rows(
                a_tile,
                b_tile,
                o_tile,
                pto.const(1, dtype=pto.index),
                pto.const(BLOCK, dtype=pto.index),
            )
        """
    ),
    "kernel_entry.simt_signature": _fixture(
        f"""
        {SNIPPET_PLACEHOLDER}


        @pto.jit(target="a5")
        def kernel_entry_simt_signature_probe(*, BLOCK: pto.const_expr = 8):
            tile = pto.alloc_tile(shape=[1, BLOCK], dtype=pto.f32, valid_shape=[1, BLOCK])
            my_simt_kernel(tile, tile.as_ptr(), pto.const(0, dtype=pto.i32))
        """
    ),
    "kernel_entry.inline_simd_scope": _fixture(
        f"""
        def kernel_entry_inline_simd_scope(
            a_tile: pto.Tile,
            b_tile: pto.Tile,
            o_tile: pto.Tile,
        ):
            for r in range(0, 1, 1):
                c = pto.const(0, dtype=pto.index)
                mask, _ = pto.make_mask(pto.f32, pto.const(16, dtype=pto.i32))
                {SNIPPET_PLACEHOLDER}


        @pto.jit(target="a5")
        def kernel_entry_inline_simd_scope_probe(*, BLOCK: pto.const_expr = 128):
            a_tile = pto.alloc_tile(shape=[1, BLOCK], dtype=pto.f32)
            b_tile = pto.alloc_tile(shape=[1, BLOCK], dtype=pto.f32)
            o_tile = pto.alloc_tile(shape=[1, BLOCK], dtype=pto.f32)
            kernel_entry_inline_simd_scope(a_tile, b_tile, o_tile)
        """
    ),
    "kernel_entry.inline_simt_scope": _fixture(
        f"""
        def kernel_entry_inline_simt_scope(
            o_prev_tile: pto.Tile,
            pv_tile: pto.Tile,
            alpha_tile: pto.Tile,
            beta_tile: pto.Tile,
            o_next_tile: pto.Tile,
        ):
            for row in range(0, 1, 1):
                col = pto.const(0, dtype=pto.index)
                o_prev = scalar.load(o_prev_tile[row, col])
                pv_val = scalar.load(pv_tile[row, col])
                {SNIPPET_PLACEHOLDER}


        @pto.jit(target="a5")
        def kernel_entry_inline_simt_scope_probe(*, BLOCK: pto.const_expr = 8):
            one = 1
            o_prev_tile = pto.alloc_tile(shape=[8, BLOCK], dtype=pto.f32, valid_shape=[2, BLOCK])
            pv_tile = pto.alloc_tile(shape=[8, BLOCK], dtype=pto.f32, valid_shape=[2, BLOCK])
            alpha_tile = pto.alloc_tile(shape=[8, 1], dtype=pto.f32, valid_shape=[2, one], blayout="ColMajor")
            beta_tile = pto.alloc_tile(shape=[8, 1], dtype=pto.f32, valid_shape=[2, one], blayout="ColMajor")
            o_next_tile = pto.alloc_tile(shape=[8, BLOCK], dtype=pto.f32, valid_shape=[2, BLOCK])
            kernel_entry_inline_simt_scope(o_prev_tile, pv_tile, alpha_tile, beta_tile, o_next_tile)
        """
    ),
    "kernel_entry.inline_cube_scope": _fixture(
        f"""
        @pto.jit(target="a5", mode="explicit")
        def kernel_entry_inline_cube_scope_probe(
            *,
            BLOCK_M: pto.const_expr = 16,
            BLOCK_K: pto.const_expr = 16,
            BLOCK_N: pto.const_expr = 16,
        ):
            q_tile = pto.alloc_tile(shape=[BLOCK_M, BLOCK_K], dtype=pto.f16, memory_space=pto.MemorySpace.MAT, valid_shape=[BLOCK_M, BLOCK_K])
            k_tile = pto.alloc_tile(shape=[BLOCK_K, BLOCK_N], dtype=pto.f16, memory_space=pto.MemorySpace.MAT, valid_shape=[BLOCK_K, BLOCK_N])
            s_tile = pto.alloc_tile(shape=[BLOCK_M, BLOCK_N], dtype=pto.f32, valid_shape=[BLOCK_M, BLOCK_N])
            q_l0a = pto.alloc_tile(shape=[BLOCK_M, BLOCK_K], dtype=pto.f16, memory_space=pto.MemorySpace.LEFT, valid_shape=[BLOCK_M, BLOCK_K])
            k_l0b = pto.alloc_tile(shape=[BLOCK_K, BLOCK_N], dtype=pto.f16, memory_space=pto.MemorySpace.RIGHT, valid_shape=[BLOCK_K, BLOCK_N])
            s_acc = pto.alloc_tile(shape=[BLOCK_M, BLOCK_N], dtype=pto.f32, memory_space=pto.MemorySpace.ACC, valid_shape=[BLOCK_M, BLOCK_N])
            m = q_tile.valid_shape[0]
            k = q_tile.valid_shape[1]
            n = k_tile.valid_shape[1]
            {SNIPPET_PLACEHOLDER}
        """
    ),
    "scalar_ops.simt_pointer": _fixture(
        f"""
        @pto.jit(target="a5")
        def scalar_ops_simt_pointer_probe():
            meta_tile = pto.alloc_tile(shape=[1, 8], dtype=pto.i32, valid_shape=[1, 4])
            meta_ptr = meta_tile.as_ptr()
            {SNIPPET_PLACEHOLDER}
        """
    ),
    "scalar_ops.helper_queries": _fixture(
        f"""
        @pto.jit(target="a5")
        def scalar_ops_helper_queries_probe():
            {SNIPPET_PLACEHOLDER}
        """
    ),
    "scalar_ops.chunk_loop": _fixture(
        f"""
        @pto.jit(target="a5")
        def scalar_ops_chunk_loop_probe(*, BLOCK: pto.const_expr = 128):
            cols = pto.const(BLOCK, dtype=pto.i32)
            {SNIPPET_PLACEHOLDER}
        """
    ),
    "scalar_ops.simt_scale": _fixture(
        f"""
        {SNIPPET_PLACEHOLDER}


        @pto.jit(target="a5")
        def scalar_ops_simt_scale_probe(*, BLOCK: pto.const_expr = 8):
            src_tile = pto.alloc_tile(shape=[8, BLOCK], dtype=pto.f32, valid_shape=[2, BLOCK])
            dst_tile = pto.alloc_tile(shape=[8, BLOCK], dtype=pto.f32, valid_shape=[2, BLOCK])
            elementwise_scale(
                src_tile,
                dst_tile,
                pto.f32(2.0),
                pto.const(2, dtype=pto.i32),
                pto.const(BLOCK, dtype=pto.i32),
            )
        """
    ),
    "scalar_ops.simt_row_coeffs": _fixture(
        f"""
        {SNIPPET_PLACEHOLDER}


        @pto.jit(target="a5")
        def scalar_ops_simt_row_coeffs_probe(*, BLOCK: pto.const_expr = 8):
            one = 1
            o_prev_tile = pto.alloc_tile(shape=[8, BLOCK], dtype=pto.f32, valid_shape=[2, BLOCK])
            pv_tile = pto.alloc_tile(shape=[8, BLOCK], dtype=pto.f32, valid_shape=[2, BLOCK])
            alpha_tile = pto.alloc_tile(shape=[8, 1], dtype=pto.f32, valid_shape=[2, one], blayout="ColMajor")
            beta_tile = pto.alloc_tile(shape=[8, 1], dtype=pto.f32, valid_shape=[2, one], blayout="ColMajor")
            o_next_tile = pto.alloc_tile(shape=[8, BLOCK], dtype=pto.f32, valid_shape=[2, BLOCK])
            blend_with_per_row_coeffs(
                o_prev_tile,
                pv_tile,
                alpha_tile,
                beta_tile,
                o_next_tile,
                pto.const(2, dtype=pto.i32),
                pto.const(BLOCK, dtype=pto.i32),
            )
        """
    ),
    "scalar_ops.math": _fixture(
        f"""
        @pto.jit(target="a5")
        def scalar_ops_math_probe():
            tile = pto.alloc_tile(shape=[1, 8], dtype=pto.f32, valid_shape=[1, 4])
            alpha = scalar.load(tile[0, 0])
            o_prev = scalar.load(tile[0, 1])
            beta = scalar.load(tile[0, 2])
            pv_val = scalar.load(tile[0, 3])
            m_prev = scalar.load(tile[0, 0])
            row_max = scalar.load(tile[0, 1])
            l_prev = scalar.load(tile[0, 2])
            m_next = scalar.load(tile[0, 3])
            val = scalar.load(tile[0, 0])
            threshold = pto.const(0.0, dtype=pto.f32)
            tail_count = scalar.load(tile[0, 1])
            N = pto.const(16, dtype=pto.i32)
            BLOCK = 8
            {SNIPPET_PLACEHOLDER}
        """
    ),
    "scalar_ops.value_adaptation": _fixture(
        f"""
        @pto.jit(target="a5")
        def scalar_ops_value_adaptation_probe():
            int_tile = pto.alloc_tile(shape=[1, 8], dtype=pto.i32, valid_shape=[1, 4])
            f16_tile = pto.alloc_tile(shape=[1, 16], dtype=pto.f16, valid_shape=[1, 1])
            f32_tile = pto.alloc_tile(shape=[1, 8], dtype=pto.f32, valid_shape=[1, 2])
            {SNIPPET_PLACEHOLDER}
        """
    ),
    "scalar_ops.pointer_sources": _fixture(
        f"""
        @pto.jit(target="a5")
        def scalar_ops_pointer_sources_probe(
            A_ptr: pto.ptr(pto.f32, "gm"),
            rows: pto.i32,
            cols: pto.i32,
            *,
            BLOCK: pto.const_expr = 8,
        ):
            a_view = pto.make_tensor_view(A_ptr, shape=[rows, cols], strides=[cols, 1])
            partition = pto.partition_view(a_view, offsets=[0, 0], sizes=[rows, cols])
            tile = pto.alloc_tile(shape=[1, BLOCK], dtype=pto.f32)
            {SNIPPET_PLACEHOLDER}
        """
    ),
    "scalar_ops.pointer_manip": _fixture(
        f"""
        @pto.jit(target="a5")
        def scalar_ops_pointer_manip_probe():
            base_tile = pto.alloc_tile(shape=[1, 8], dtype=pto.i32, valid_shape=[1, 4])
            base_ptr = base_tile.as_ptr()
            addr = pto.const(0, dtype=pto.i64)
            {SNIPPET_PLACEHOLDER}
        """
    ),
    "data_movement.tload": _fixture(
        f"""
        @pto.jit(target="a5")
        def data_movement_tload_probe(
            A_ptr: pto.ptr(pto.f32, "gm"),
            rows: pto.i32,
            cols: pto.i32,
            *,
            BLOCK: pto.const_expr = 128,
        ):
            offset = 0
            a_view = pto.make_tensor_view(A_ptr, shape=[rows, cols], strides=[cols, 1])
            {SNIPPET_PLACEHOLDER}
        """
    ),
    "data_movement.explicit_dma": _fixture(
        f"""
        def process_block(
            k_part: pto.PartitionTensorView,
            v_part: pto.PartitionTensorView,
            k_tile: pto.Tile,
            v_tile: pto.Tile,
            o_tile: pto.Tile,
            o_part: pto.PartitionTensorView,
            rows: pto.i32,
            cols: pto.i32,
        ):
            {SNIPPET_PLACEHOLDER}


        @pto.jit(target="a5", mode="explicit")
        def data_movement_explicit_dma_probe(
            K_ptr: pto.ptr(pto.f16, "gm"),
            V_ptr: pto.ptr(pto.f16, "gm"),
            O_ptr: pto.ptr(pto.f32, "gm"),
            *,
            ROWS: pto.const_expr = 8,
            COLS: pto.const_expr = 16,
        ):
            k_view = pto.make_tensor_view(K_ptr, shape=[ROWS, COLS], strides=[COLS, 1])
            v_view = pto.make_tensor_view(V_ptr, shape=[ROWS, COLS], strides=[COLS, 1])
            o_view = pto.make_tensor_view(O_ptr, shape=[ROWS, COLS], strides=[COLS, 1])
            k_part = pto.partition_view(k_view, offsets=[0, 0], sizes=[ROWS, COLS])
            v_part = pto.partition_view(v_view, offsets=[0, 0], sizes=[ROWS, COLS])
            o_part = pto.partition_view(o_view, offsets=[0, 0], sizes=[ROWS, COLS])
            k_tile = pto.alloc_tile(shape=[ROWS, COLS], dtype=pto.f16)
            v_tile = pto.alloc_tile(shape=[ROWS, COLS], dtype=pto.f16)
            o_tile = pto.alloc_tile(shape=[ROWS, COLS], dtype=pto.f32)
            process_block(k_part, v_part, k_tile, v_tile, o_tile, o_part, ROWS, COLS)
        """
    ),
    "sync_ops.flag_pattern_explicit": _fixture(
        f"""
        @pto.tileop
        def qk_matmul(q_tile: pto.Tile, k_tile: pto.Tile, p_tile: pto.Tile):
            return


        @pto.tileop
        def pv_matmul(p_tile: pto.Tile, v_tile: pto.Tile, o_tile: pto.Tile):
            return


        {SNIPPET_PLACEHOLDER}


        @pto.jit(target="a5", mode="explicit")
        def sync_ops_flag_pattern_explicit_probe(
            K_ptr: pto.ptr(pto.f16, "gm"),
            V_ptr: pto.ptr(pto.f16, "gm"),
            O_ptr: pto.ptr(pto.f32, "gm"),
            *,
            ROWS: pto.const_expr = 8,
            COLS: pto.const_expr = 16,
        ):
            k_view = pto.make_tensor_view(K_ptr, shape=[ROWS, COLS], strides=[COLS, 1])
            v_view = pto.make_tensor_view(V_ptr, shape=[ROWS, COLS], strides=[COLS, 1])
            o_view = pto.make_tensor_view(O_ptr, shape=[ROWS, COLS], strides=[COLS, 1])
            q_tile = pto.alloc_tile(shape=[ROWS, COLS], dtype=pto.f16)
            k_tile = pto.alloc_tile(shape=[ROWS, COLS], dtype=pto.f16)
            v_tile = pto.alloc_tile(shape=[ROWS, COLS], dtype=pto.f16)
            p_tile = pto.alloc_tile(shape=[ROWS, COLS], dtype=pto.f32)
            o_tile = pto.alloc_tile(shape=[ROWS, COLS], dtype=pto.f32)
            k_part = pto.partition_view(k_view, offsets=[0, 0], sizes=[ROWS, COLS])
            v_part = pto.partition_view(v_view, offsets=[0, 0], sizes=[ROWS, COLS])
            o_part = pto.partition_view(o_view, offsets=[0, 0], sizes=[ROWS, COLS])
            gemm_block(
                q_tile,
                k_part,
                v_part,
                k_tile,
                v_tile,
                p_tile,
                o_tile,
                o_part,
                pto.const(ROWS, dtype=pto.i32),
                pto.const(COLS, dtype=pto.i32),
            )
        """
    ),
    "sync_ops.phase_barrier_explicit": _fixture(
        f"""
        @pto.tileop
        def qk_matmul(q_tile: pto.Tile, k_tile: pto.Tile, s_tile: pto.Tile):
            return


        @pto.tileop
        def online_softmax(s_tile: pto.Tile, p_tile: pto.Tile, rows: pto.i32, cols: pto.i32):
            return


        @pto.tileop
        def pv_matmul(p_tile: pto.Tile, v_tile: pto.Tile, pv_tile: pto.Tile):
            return


        @pto.simt
        def blend_output(o_prev_tile: pto.Tile, pv_tile: pto.Tile, o_next_tile: pto.Tile, rows: pto.i32, cols: pto.i32):
            return


        {SNIPPET_PLACEHOLDER}


        @pto.jit(target="a5", mode="explicit")
        def sync_ops_phase_barrier_explicit_probe(
            K_ptr: pto.ptr(pto.f16, "gm"),
            V_ptr: pto.ptr(pto.f16, "gm"),
            *,
            ROWS: pto.const_expr = 8,
            COLS: pto.const_expr = 16,
        ):
            k_view = pto.make_tensor_view(K_ptr, shape=[ROWS, COLS], strides=[COLS, 1])
            v_view = pto.make_tensor_view(V_ptr, shape=[ROWS, COLS], strides=[COLS, 1])
            q_tile = pto.alloc_tile(shape=[ROWS, COLS], dtype=pto.f16)
            k_tile = pto.alloc_tile(shape=[ROWS, COLS], dtype=pto.f16)
            v_tile = pto.alloc_tile(shape=[ROWS, COLS], dtype=pto.f16)
            s_tile = pto.alloc_tile(shape=[ROWS, COLS], dtype=pto.f32)
            p_tile = pto.alloc_tile(shape=[ROWS, COLS], dtype=pto.f32)
            pv_tile = pto.alloc_tile(shape=[ROWS, COLS], dtype=pto.f32)
            o_prev_tile = pto.alloc_tile(shape=[ROWS, COLS], dtype=pto.f32)
            o_next_tile = pto.alloc_tile(shape=[ROWS, COLS], dtype=pto.f32)
            k_part = pto.partition_view(k_view, offsets=[0, 0], sizes=[ROWS, COLS])
            v_part = pto.partition_view(v_view, offsets=[0, 0], sizes=[ROWS, COLS])
            flash_attention_block(
                q_tile,
                k_part,
                v_part,
                k_tile,
                v_tile,
                s_tile,
                p_tile,
                pv_tile,
                o_prev_tile,
                o_next_tile,
                pto.const(ROWS, dtype=pto.i32),
                pto.const(COLS, dtype=pto.i32),
            )
        """
    ),
    "data_movement.grouped_dma_ptrs": _fixture(
        f"""
        @pto.jit(target="a5", mode="explicit")
        def data_movement_grouped_dma_ptrs_probe():
            gm_src = pto.castptr(pto.ui64(0), pto.ptr(pto.f16, "gm"))
            gm_dst = pto.castptr(pto.ui64(0), pto.ptr(pto.f16, "gm"))
            gm_src_f32 = pto.castptr(pto.ui64(0), pto.ptr(pto.f32, "gm"))
            gm_dst_f32 = pto.castptr(pto.ui64(0), pto.ptr(pto.f32, "gm"))
            ub_src = pto.castptr(pto.ui64(0), pto.ptr(pto.f16, "ub"))
            ub_dst = pto.castptr(pto.ui64(0), pto.ptr(pto.f16, "ub"))
            ub_src_f32 = pto.castptr(pto.ui64(0), pto.ptr(pto.f32, "ub"))
            ub_dst_f32 = pto.castptr(pto.ui64(0), pto.ptr(pto.f32, "ub"))
            l1_dst = pto.castptr(pto.ui64(0), pto.ptr(pto.f16, "left"))
            {SNIPPET_PLACEHOLDER}
        """
    ),
    "data_movement.low_precision_vector_memory": _fixture(
        f"""
        @pto.jit(target="a5", mode="explicit")
        def data_movement_low_precision_vector_memory_probe():
            f8_src = pto.castptr(pto.ui64(0), pto.ptr(pto.f8e4m3, "ub"))
            f8_dst = pto.castptr(pto.ui64(0), pto.ptr(pto.f8e4m3, "ub"))
            fp4_src = pto.castptr(pto.ui64(0), pto.ptr(pto.f4e1m2x2, "ub"))
            fp4_dst = pto.castptr(pto.ui64(0), pto.ptr(pto.f4e1m2x2, "ub"))
            {SNIPPET_PLACEHOLDER}
        """
    ),
    "data_movement.tile_slice_2d": _fixture(
        f"""
        @pto.jit(target="a5")
        def data_movement_tile_slice_2d_probe(
            *,
            BLOCK: pto.const_expr = 128,
        ):
            tile = pto.alloc_tile(shape=[2, BLOCK], dtype=pto.f32)
            col = 0
            for row in range(0, 1, 1):
                {SNIPPET_PLACEHOLDER}
        """
    ),
    "data_movement.tile_slice_1d": _fixture(
        f"""
        @pto.jit(target="a5")
        def data_movement_tile_slice_1d_probe(
            *,
            BLOCK: pto.const_expr = 128,
        ):
            tile = pto.alloc_tile(shape=[BLOCK], dtype=pto.f32)
            start = pto.const(0, dtype=pto.i32)
            {SNIPPET_PLACEHOLDER}
        """
    ),
    "data_movement.cube_helper": _fixture(
        f"""
        @pto.tileop
        def qk_matmul(
            q_tile: pto.Tile,
            k_tile: pto.Tile,
            q_l0a: pto.Tile,
            k_l0b: pto.Tile,
            s_acc: pto.Tile,
            s_tile: pto.Tile,
        ):
            {SNIPPET_PLACEHOLDER}


        @pto.jit(target="a5", mode="explicit")
        def data_movement_cube_helper_probe(
            *,
            BLOCK_M: pto.const_expr = 16,
            BLOCK_K: pto.const_expr = 16,
            BLOCK_N: pto.const_expr = 16,
        ):
            q_tile = pto.alloc_tile(shape=[BLOCK_M, BLOCK_K], dtype=pto.f16, memory_space=pto.MemorySpace.MAT, valid_shape=[BLOCK_M, BLOCK_K])
            k_tile = pto.alloc_tile(shape=[BLOCK_K, BLOCK_N], dtype=pto.f16, memory_space=pto.MemorySpace.MAT, valid_shape=[BLOCK_K, BLOCK_N])
            s_tile = pto.alloc_tile(shape=[BLOCK_M, BLOCK_N], dtype=pto.f32, valid_shape=[BLOCK_M, BLOCK_N])
            q_l0a = pto.alloc_tile(shape=[BLOCK_M, BLOCK_K], dtype=pto.f16, memory_space=pto.MemorySpace.LEFT, valid_shape=[BLOCK_M, BLOCK_K])
            k_l0b = pto.alloc_tile(shape=[BLOCK_K, BLOCK_N], dtype=pto.f16, memory_space=pto.MemorySpace.RIGHT, valid_shape=[BLOCK_K, BLOCK_N])
            s_acc = pto.alloc_tile(shape=[BLOCK_M, BLOCK_N], dtype=pto.f32, memory_space=pto.MemorySpace.ACC, valid_shape=[BLOCK_M, BLOCK_N])
            qk_matmul(q_tile, k_tile, q_l0a, k_l0b, s_acc, s_tile)
        """
    ),
    "compute_ops.vector_compute": _fixture(
        f"""
        @pto.tileop
        def compute_ops_vector_helper(inp_tile: pto.Tile, out_tile: pto.Tile, row: pto.index):
            col_mask = pto.make_mask(pto.f32, pto.const(16, dtype=pto.i32))
            s_row = pto.vlds(inp_tile[row, 0:])
            p_row = pto.vexp(s_row, col_mask)
            m_next = pto.vcgmax(s_row, col_mask)
            mask32_full = pto.pset_b32(pto.MaskPattern.ALL)
            int_tile = pto.alloc_tile(shape=[1, 64], dtype=pto.i32, valid_shape=[1, 64])
            vec_f32 = pto.vlds(inp_tile[row, 0:])
            vec_i32 = pto.vlds(int_tile.as_ptr(), pto.const(0))
            exp_f32_even = vec_f32
            exp_f32_odd = vec_f32
            mask = mask32_full
            {SNIPPET_PLACEHOLDER}


        @pto.jit(target="a5")
        def compute_ops_vector_probe(*, BLOCK: pto.const_expr = 128):
            inp_tile = pto.alloc_tile(shape=[2, BLOCK], dtype=pto.f32)
            out_tile = pto.alloc_tile(shape=[2, BLOCK], dtype=pto.f32)
            for row in range(0, 1, 1):
                compute_ops_vector_helper(inp_tile, out_tile, row)
        """
    ),
    "compute_ops.tile_low_precision_cvt": _fixture(
        f"""
        @pto.jit(target="a5")
        def compute_ops_tile_low_precision_cvt_probe():
            {SNIPPET_PLACEHOLDER}
        """
    ),
    "compute_ops.tile_window_matmul": _fixture(
        f"""
        @pto.jit(target="a5")
        def compute_ops_tile_window_matmul_probe(
            *,
            BLOCK_M: pto.const_expr = 16,
            BLOCK_K: pto.const_expr = 16,
            BLOCK_N: pto.const_expr = 16,
            CARRIER_M: pto.const_expr = 64,
            CARRIER_N: pto.const_expr = 64,
        ):
            src_mat = pto.alloc_tile(shape=[CARRIER_M, CARRIER_N], dtype=pto.f32, memory_space=pto.MemorySpace.MAT)
            dst_mat = pto.alloc_tile(
                shape=[CARRIER_M, CARRIER_N],
                dtype=pto.f32,
                memory_space=pto.MemorySpace.MAT,
                blayout="ColMajor",
                slayout="RowMajor",
            )
            lhs_l0a = pto.alloc_tile(
                shape=[BLOCK_M, BLOCK_K],
                dtype=pto.f16,
                memory_space=pto.MemorySpace.LEFT,
                blayout="ColMajor",
                slayout="RowMajor",
            )
            rhs_l0b = pto.alloc_tile(
                shape=[BLOCK_K, BLOCK_N],
                dtype=pto.f16,
                memory_space=pto.MemorySpace.RIGHT,
                blayout="RowMajor",
                slayout="ColMajor",
            )
            acc_prev = pto.alloc_tile(
                shape=[BLOCK_M, BLOCK_N],
                dtype=pto.f32,
                memory_space=pto.MemorySpace.ACC,
                blayout="ColMajor",
                slayout="RowMajor",
            )
            acc_out = pto.alloc_tile(
                shape=[BLOCK_M, BLOCK_N],
                dtype=pto.f32,
                memory_space=pto.MemorySpace.ACC,
                blayout="ColMajor",
                slayout="RowMajor",
            )
            {SNIPPET_PLACEHOLDER}
        """
    ),
    "compute_ops.tile_mx_compute": _fixture(
        f"""
        @pto.jit(target="a5")
        def compute_ops_tile_mx_compute_probe():
            {SNIPPET_PLACEHOLDER}
        """
    ),
    "mask_ops.creation": _fixture(
        f"""
        @pto.jit(target="a5")
        def mask_ops_creation_probe():
            remained = pto.const(16, dtype=pto.i32)
            seed = pto.pset_b32("PAT_ALL")
            {SNIPPET_PLACEHOLDER}
        """
    ),
    "mask_ops.logical": _fixture(
        f"""
        @pto.jit(target="a5")
        def mask_ops_logical_probe():
            src0 = pto.pset_b32(pto.MaskPattern.ALL)
            src1 = pto.pge_b32(pto.MaskPattern.VL16)
            gate = pto.pset_b32(pto.MaskPattern.ALL)
            {SNIPPET_PLACEHOLDER}
        """
    ),
    "mask_ops.compare": _fixture(
        f"""
        @pto.jit(target="a5")
        def mask_ops_compare_probe():
            seed = pto.pset_b32(pto.MaskPattern.ALL)
            vec_tile = pto.alloc_tile(shape=[1, 64], dtype=pto.f32, valid_shape=[1, 64])
            scores = pto.vlds(vec_tile.as_ptr(), pto.const(0))
            threshold = pto.f32(0.0)
            {SNIPPET_PLACEHOLDER}
        """
    ),
    "mask_ops.reorg": _fixture(
        f"""
        @pto.jit(target="a5")
        def mask_ops_reorg_probe():
            mask32 = pto.pset_b32(pto.MaskPattern.ALL)
            {SNIPPET_PLACEHOLDER}
        """
    ),
    "sync_ops.basic": _fixture(
        f"""
        @pto.jit(target="a5")
        def sync_ops_basic_probe():
            {SNIPPET_PLACEHOLDER}
        """
    ),
    "flash_attention.l1_tensor_views": _fixture(
        f"""
        @pto.jit(target="a5")
        def flash_attention_l1_tensor_views_probe(
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
            CAUSAL: pto.const_expr = False,
            NUM_STAGES: pto.const_expr = 2,
        ):
            Br = BLOCK_Q
            Bc = BLOCK_KV
            D = dim
            full_br = Br
            full_bc = Bc
            one = 1
            {SNIPPET_PLACEHOLDER}
        """
    ),
    "flash_attention.l1_partitions": _fixture(
        f"""
        @pto.jit(target="a5")
        def flash_attention_l1_partitions_probe(
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
            CAUSAL: pto.const_expr = False,
            NUM_STAGES: pto.const_expr = 2,
        ):
            q_view = pto.make_tensor_view(Q_ptr, shape=[batch, seq_q, heads, dim], strides=[seq_q * heads * dim, heads * dim, dim, 1])
            k_view = pto.make_tensor_view(K_ptr, shape=[batch, seq_k, heads, dim], strides=[seq_k * heads * dim, heads * dim, dim, 1])
            v_view = pto.make_tensor_view(V_ptr, shape=[batch, seq_k, heads, dim], strides=[seq_k * heads * dim, heads * dim, dim, 1])
            o_view = pto.make_tensor_view(O_ptr, shape=[batch, seq_q, heads, dim], strides=[seq_q * heads * dim, heads * dim, dim, 1])
            block_idx = pto.get_block_idx()
            batch_idx = block_idx // heads
            head_idx = block_idx % heads
            {SNIPPET_PLACEHOLDER}
        """
    ),
    "flash_attention.l1_tiles": _fixture(
        f"""
        @pto.jit(target="a5")
        def flash_attention_l1_tiles_probe(
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
        ):
            Br = BLOCK_Q
            Bc = BLOCK_KV
            D = HEAD_DIM
            full_br = Br
            full_bc = Bc
            one = 1
            {SNIPPET_PLACEHOLDER}
        """
    ),
    "flash_attention.l1_loop_body": _fixture(
        f"""
        def _min_index(lhs, rhs):
            return scalar.select(lhs < rhs, lhs, rhs)


        def _block_valid_extent(total, block_index, block_size):
            return _min_index(total - block_index * block_size, pto.const(block_size))


        def kv_block_process(
            q_mat: pto.Tile,
            k_part: pto.PartitionTensorView,
            v_part: pto.PartitionTensorView,
            k_mat: pto.Tile,
            v_mat: pto.Tile,
            o_prev_tile: pto.Tile,
            o_next_tile: pto.Tile,
            m_prev_tile: pto.Tile,
            l_prev_tile: pto.Tile,
            m_next_tile: pto.Tile,
            l_next_tile: pto.Tile,
            s_tile: pto.Tile,
            p_tile: pto.Tile,
            p_mat: pto.Tile,
            pv_tile: pto.Tile,
            alpha_tile: pto.Tile,
            beta_tile: pto.Tile,
            q_l0a: pto.Tile,
            p_l0a: pto.Tile,
            rhs_l0b: pto.Tile,
            qk_acc_tile: pto.Tile,
            pv_acc_tile: pto.Tile,
            meta_ptr,
        ):
            pto.pipe_barrier(pto.Pipe.ALL)


        @pto.jit(target="a5", mode="explicit")
        def flash_attention_l1_loop_body_probe(
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
            q_view = pto.make_tensor_view(Q_ptr, shape=[batch, seq_q, heads, dim], strides=[seq_q * heads * dim, heads * dim, dim, 1])
            k_view = pto.make_tensor_view(K_ptr, shape=[batch, seq_k, heads, dim], strides=[seq_k * heads * dim, heads * dim, dim, 1])
            v_view = pto.make_tensor_view(V_ptr, shape=[batch, seq_k, heads, dim], strides=[seq_k * heads * dim, heads * dim, dim, 1])
            o_view = pto.make_tensor_view(O_ptr, shape=[batch, seq_q, heads, dim], strides=[seq_q * heads * dim, heads * dim, dim, 1])
            block_idx = pto.get_block_idx()
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
            q_l0a = pto.alloc_tile(shape=[Br, D], dtype=pto.f32, memory_space=pto.MemorySpace.LEFT, valid_shape=[full_br, dim])
            p_l0a = pto.alloc_tile(shape=[Br, Bc], dtype=pto.f32, memory_space=pto.MemorySpace.LEFT, valid_shape=[full_br, full_bc])
            rhs_l0b = pto.alloc_tile(shape=[Bc, D], dtype=pto.f32, memory_space=pto.MemorySpace.RIGHT, valid_shape=[full_bc, dim])
            qk_acc_tile = pto.alloc_tile(shape=[Br, Bc], dtype=pto.f32, memory_space=pto.MemorySpace.ACC, valid_shape=[full_br, full_bc])
            pv_acc_tile = pto.alloc_tile(shape=[Br, D], dtype=pto.f32, memory_space=pto.MemorySpace.ACC, valid_shape=[full_br, dim])
            meta_tile = pto.alloc_tile(shape=[1, 8], dtype=pto.i32, valid_shape=[1, 3])
            meta_ptr = meta_tile.as_ptr()
            _ = CAUSAL
            _ = NUM_STAGES
            {SNIPPET_PLACEHOLDER}
        """
    ),
    "flash_attention.explicit_phase": _fixture(
        f"""
        @pto.tileop
        def qk_matmul(
            q_mat: pto.Tile,
            k_mat: pto.Tile,
            q_l0a: pto.Tile,
            rhs_l0b: pto.Tile,
            qk_acc_tile: pto.Tile,
            s_tile: pto.Tile,
        ):
            return


        @pto.tileop
        def online_softmax_rows(
            s_tile: pto.Tile,
            p_tile: pto.Tile,
            m_prev_tile: pto.Tile,
            l_prev_tile: pto.Tile,
            m_next_tile: pto.Tile,
            l_next_tile: pto.Tile,
            alpha_tile: pto.Tile,
            beta_tile: pto.Tile,
            row_start: pto.i32,
            row_stop: pto.i32,
            valid_cols: pto.i32,
        ):
            return


        @pto.tileop
        def pv_matmul(
            p_mat: pto.Tile,
            v_mat: pto.Tile,
            p_l0a: pto.Tile,
            rhs_l0b: pto.Tile,
            pv_acc_tile: pto.Tile,
            pv_tile: pto.Tile,
        ):
            return


        @pto.simt
        def blend_output_rows(
            o_prev_tile: pto.Tile,
            pv_tile: pto.Tile,
            alpha_tile: pto.Tile,
            beta_tile: pto.Tile,
            o_next_tile: pto.Tile,
            row_start: pto.i32,
            row_stop: pto.i32,
            valid_dim: pto.i32,
        ):
            return


        @pto.simt
        def materialize_tile_bounds(
            meta_ptr: pto.ptr(pto.i32, pto.MemorySpace.UB),
            valid_rows: pto.i32,
            valid_cols: pto.i32,
        ):
            scalar.store(0, meta_ptr + 0)
            scalar.store(valid_rows, meta_ptr + 1)
            scalar.store(valid_cols, meta_ptr + 2)


        def flash_attention_explicit_phase(
            q_mat: pto.Tile,
            k_part: pto.PartitionTensorView,
            v_part: pto.PartitionTensorView,
            k_mat: pto.Tile,
            v_mat: pto.Tile,
            o_prev_tile: pto.Tile,
            o_next_tile: pto.Tile,
            m_prev_tile: pto.Tile,
            l_prev_tile: pto.Tile,
            m_next_tile: pto.Tile,
            l_next_tile: pto.Tile,
            s_tile: pto.Tile,
            p_tile: pto.Tile,
            p_mat: pto.Tile,
            pv_tile: pto.Tile,
            alpha_tile: pto.Tile,
            beta_tile: pto.Tile,
            q_l0a: pto.Tile,
            p_l0a: pto.Tile,
            rhs_l0b: pto.Tile,
            qk_acc_tile: pto.Tile,
            pv_acc_tile: pto.Tile,
            meta_ptr,
        ):
            row_start = pto.const(0, dtype=pto.i32)
            row_stop = pto.const(16, dtype=pto.i32)
            valid_cols = pto.const(16, dtype=pto.i32)
            {SNIPPET_PLACEHOLDER}


        @pto.jit(target="a5", mode="explicit")
        def flash_attention_explicit_phase_probe(
            K_ptr: pto.ptr(pto.f32, "gm"),
            V_ptr: pto.ptr(pto.f32, "gm"),
            seq_k: pto.i32,
            *,
            BLOCK_Q: pto.const_expr = 16,
            BLOCK_KV: pto.const_expr = 16,
        ):
            Br = BLOCK_Q
            Bc = BLOCK_KV
            D = 16
            one = 1
            k_view = pto.make_tensor_view(K_ptr, shape=[1, seq_k, 1, D], strides=[seq_k * D, D, D, 1])
            v_view = pto.make_tensor_view(V_ptr, shape=[1, seq_k, 1, D], strides=[seq_k * D, D, D, 1])
            k_part = pto.partition_view(k_view, offsets=[0, 0, 0, 0], sizes=[1, Bc, 1, D])
            v_part = pto.partition_view(v_view, offsets=[0, 0, 0, 0], sizes=[1, Bc, 1, D])
            q_mat = pto.alloc_tile(shape=[Br, D], dtype=pto.f32, memory_space=pto.MemorySpace.MAT, valid_shape=[Br, D], blayout="ColMajor", slayout="RowMajor")
            k_mat = pto.alloc_tile(shape=[Bc, D], dtype=pto.f32, memory_space=pto.MemorySpace.MAT, valid_shape=[Bc, D], blayout="ColMajor", slayout="RowMajor")
            v_mat = pto.alloc_tile(shape=[Bc, D], dtype=pto.f32, memory_space=pto.MemorySpace.MAT, valid_shape=[Bc, D], blayout="ColMajor", slayout="RowMajor")
            o_prev_tile = pto.alloc_tile(shape=[Br, D], dtype=pto.f32, valid_shape=[Br, D])
            o_next_tile = pto.alloc_tile(shape=[Br, D], dtype=pto.f32, valid_shape=[Br, D])
            m_prev_tile = pto.alloc_tile(shape=[Br, 1], dtype=pto.f32, valid_shape=[Br, one], blayout="ColMajor")
            l_prev_tile = pto.alloc_tile(shape=[Br, 1], dtype=pto.f32, valid_shape=[Br, one], blayout="ColMajor")
            m_next_tile = pto.alloc_tile(shape=[Br, 1], dtype=pto.f32, valid_shape=[Br, one], blayout="ColMajor")
            l_next_tile = pto.alloc_tile(shape=[Br, 1], dtype=pto.f32, valid_shape=[Br, one], blayout="ColMajor")
            alpha_tile = pto.alloc_tile(shape=[Br, 1], dtype=pto.f32, valid_shape=[Br, one], blayout="ColMajor")
            beta_tile = pto.alloc_tile(shape=[Br, 1], dtype=pto.f32, valid_shape=[Br, one], blayout="ColMajor")
            s_tile = pto.alloc_tile(shape=[Br, Bc], dtype=pto.f32, valid_shape=[Br, Bc])
            p_tile = pto.alloc_tile(shape=[Br, Bc], dtype=pto.f32, valid_shape=[Br, Bc])
            p_mat = pto.alloc_tile(shape=[Br, Bc], dtype=pto.f32, memory_space=pto.MemorySpace.MAT, valid_shape=[Br, Bc], blayout="ColMajor", slayout="RowMajor")
            pv_tile = pto.alloc_tile(shape=[Br, D], dtype=pto.f32, valid_shape=[Br, D])
            q_l0a = pto.alloc_tile(shape=[Br, D], dtype=pto.f32, memory_space=pto.MemorySpace.LEFT, valid_shape=[Br, D])
            p_l0a = pto.alloc_tile(shape=[Br, Bc], dtype=pto.f32, memory_space=pto.MemorySpace.LEFT, valid_shape=[Br, Bc])
            rhs_l0b = pto.alloc_tile(shape=[Bc, D], dtype=pto.f32, memory_space=pto.MemorySpace.RIGHT, valid_shape=[Bc, D])
            qk_acc_tile = pto.alloc_tile(shape=[Br, Bc], dtype=pto.f32, memory_space=pto.MemorySpace.ACC, valid_shape=[Br, Bc])
            pv_acc_tile = pto.alloc_tile(shape=[Br, D], dtype=pto.f32, memory_space=pto.MemorySpace.ACC, valid_shape=[Br, D])
            meta_tile = pto.alloc_tile(shape=[1, 8], dtype=pto.i32, valid_shape=[1, 3])
            meta_ptr = meta_tile.as_ptr()
            flash_attention_explicit_phase(
                q_mat, k_part, v_part, k_mat, v_mat,
                o_prev_tile, o_next_tile,
                m_prev_tile, l_prev_tile, m_next_tile, l_next_tile,
                s_tile, p_tile, p_mat, pv_tile,
                alpha_tile, beta_tile,
                q_l0a, p_l0a, rhs_l0b,
                qk_acc_tile, pv_acc_tile,
                meta_ptr,
            )
        """
    ),
    "flash_attention.qk_cube_helper": _fixture(
        f"""
        {SNIPPET_PLACEHOLDER}


        @pto.jit(target="a5", mode="explicit")
        def flash_attention_qk_cube_helper_probe(*, BLOCK_Q: pto.const_expr = 16, BLOCK_KV: pto.const_expr = 16):
            Br = BLOCK_Q
            Bc = BLOCK_KV
            D = 16
            q_mat = pto.alloc_tile(shape=[Br, D], dtype=pto.f32, memory_space=pto.MemorySpace.MAT, valid_shape=[Br, D], blayout="ColMajor", slayout="RowMajor")
            k_mat = pto.alloc_tile(shape=[Bc, D], dtype=pto.f32, memory_space=pto.MemorySpace.MAT, valid_shape=[Bc, D], blayout="ColMajor", slayout="RowMajor")
            q_l0a = pto.alloc_tile(shape=[Br, D], dtype=pto.f32, memory_space=pto.MemorySpace.LEFT, valid_shape=[Br, D])
            rhs_l0b = pto.alloc_tile(shape=[Bc, D], dtype=pto.f32, memory_space=pto.MemorySpace.RIGHT, valid_shape=[Bc, D])
            qk_acc_tile = pto.alloc_tile(shape=[Br, Bc], dtype=pto.f32, memory_space=pto.MemorySpace.ACC, valid_shape=[Br, Bc])
            s_tile = pto.alloc_tile(shape=[Br, Bc], dtype=pto.f32, valid_shape=[Br, Bc])
            qk_matmul(q_mat, k_mat, q_l0a, rhs_l0b, qk_acc_tile, s_tile)
        """
    ),
    "flash_attention.pv_cube_helper": _fixture(
        f"""
        {SNIPPET_PLACEHOLDER}


        @pto.jit(target="a5", mode="explicit")
        def flash_attention_pv_cube_helper_probe(*, BLOCK_Q: pto.const_expr = 16, BLOCK_KV: pto.const_expr = 16):
            Br = BLOCK_Q
            Bc = BLOCK_KV
            D = 16
            p_mat = pto.alloc_tile(shape=[Br, Bc], dtype=pto.f32, memory_space=pto.MemorySpace.MAT, valid_shape=[Br, Bc], blayout="ColMajor", slayout="RowMajor")
            v_mat = pto.alloc_tile(shape=[Bc, D], dtype=pto.f32, memory_space=pto.MemorySpace.MAT, valid_shape=[Bc, D], blayout="ColMajor", slayout="RowMajor")
            p_l0a = pto.alloc_tile(shape=[Br, Bc], dtype=pto.f32, memory_space=pto.MemorySpace.LEFT, valid_shape=[Br, Bc])
            rhs_l0b = pto.alloc_tile(shape=[Bc, D], dtype=pto.f32, memory_space=pto.MemorySpace.RIGHT, valid_shape=[Bc, D])
            pv_acc_tile = pto.alloc_tile(shape=[Br, D], dtype=pto.f32, memory_space=pto.MemorySpace.ACC, valid_shape=[Br, D])
            pv_tile = pto.alloc_tile(shape=[Br, D], dtype=pto.f32, valid_shape=[Br, D])
            pv_matmul(p_mat, v_mat, p_l0a, rhs_l0b, pv_acc_tile, pv_tile)
        """
    ),
    "flash_attention.inline_simt_scope": _fixture(
        f"""
        def flash_attention_inline_simt_scope(
            q_mat: pto.Tile,
            k_mat: pto.Tile,
            meta_ptr,
        ):
            {SNIPPET_PLACEHOLDER}


        @pto.jit(target="a5")
        def flash_attention_inline_simt_scope_probe(*, BLOCK_Q: pto.const_expr = 16, BLOCK_KV: pto.const_expr = 16):
            Br = BLOCK_Q
            Bc = BLOCK_KV
            D = 16
            q_mat = pto.alloc_tile(shape=[Br, D], dtype=pto.f32, memory_space=pto.MemorySpace.MAT, valid_shape=[Br, D], blayout="ColMajor", slayout="RowMajor")
            k_mat = pto.alloc_tile(shape=[Bc, D], dtype=pto.f32, memory_space=pto.MemorySpace.MAT, valid_shape=[Bc, D], blayout="ColMajor", slayout="RowMajor")
            meta_tile = pto.alloc_tile(shape=[1, 8], dtype=pto.i32, valid_shape=[1, 3])
            flash_attention_inline_simt_scope(q_mat, k_mat, meta_tile.as_ptr())
        """
    ),
    "flash_attention.online_softmax_loop": _fixture(
        f"""
        @pto.tileop
        def flash_attention_online_softmax_loop_helper(
            s_tile: pto.Tile,
            p_tile: pto.Tile,
            m_prev_tile: pto.Tile,
            l_prev_tile: pto.Tile,
            m_next_tile: pto.Tile,
            l_next_tile: pto.Tile,
            alpha_tile: pto.Tile,
            beta_tile: pto.Tile,
            row_start: pto.i32,
            row_stop: pto.i32,
            valid_cols: pto.i32,
        ):
            {SNIPPET_PLACEHOLDER}


        @pto.jit(target="a5")
        def flash_attention_online_softmax_loop_probe(*, BLOCK: pto.const_expr = 16):
            one = 1
            s_tile = pto.alloc_tile(shape=[8, BLOCK], dtype=pto.f32, valid_shape=[2, BLOCK])
            p_tile = pto.alloc_tile(shape=[8, BLOCK], dtype=pto.f32, valid_shape=[2, BLOCK])
            m_prev_tile = pto.alloc_tile(shape=[8, 1], dtype=pto.f32, valid_shape=[2, one], blayout="ColMajor")
            l_prev_tile = pto.alloc_tile(shape=[8, 1], dtype=pto.f32, valid_shape=[2, one], blayout="ColMajor")
            m_next_tile = pto.alloc_tile(shape=[8, 1], dtype=pto.f32, valid_shape=[2, one], blayout="ColMajor")
            l_next_tile = pto.alloc_tile(shape=[8, 1], dtype=pto.f32, valid_shape=[2, one], blayout="ColMajor")
            alpha_tile = pto.alloc_tile(shape=[8, 1], dtype=pto.f32, valid_shape=[2, one], blayout="ColMajor")
            beta_tile = pto.alloc_tile(shape=[8, 1], dtype=pto.f32, valid_shape=[2, one], blayout="ColMajor")
            flash_attention_online_softmax_loop_helper(
                s_tile, p_tile,
                m_prev_tile, l_prev_tile,
                m_next_tile, l_next_tile,
                alpha_tile, beta_tile,
                0, 2, BLOCK,
            )
        """
    ),
    "flash_attention.online_softmax_compute": _fixture(
        f"""
        @pto.tileop
        def flash_attention_online_softmax_compute_helper(
            s_tile: pto.Tile,
            p_tile: pto.Tile,
            m_prev_tile: pto.Tile,
            l_prev_tile: pto.Tile,
            m_next_tile: pto.Tile,
            l_next_tile: pto.Tile,
            alpha_tile: pto.Tile,
            beta_tile: pto.Tile,
            row_start: pto.i32,
            row_stop: pto.i32,
            valid_cols: pto.i32,
        ):
            for row in range(row_start, row_stop, 1):
                col_mask = pto.make_mask(pto.f32, valid_cols)
                s_row = pto.vlds(s_tile[row, 0:])
                m_prev = scalar.load(m_prev_tile[row, 0])
                l_prev = scalar.load(l_prev_tile[row, 0])
            {SNIPPET_PLACEHOLDER}


        @pto.jit(target="a5")
        def flash_attention_online_softmax_compute_probe(*, BLOCK: pto.const_expr = 16):
            one = 1
            s_tile = pto.alloc_tile(shape=[8, BLOCK], dtype=pto.f32, valid_shape=[2, BLOCK])
            p_tile = pto.alloc_tile(shape=[8, BLOCK], dtype=pto.f32, valid_shape=[2, BLOCK])
            m_prev_tile = pto.alloc_tile(shape=[8, 1], dtype=pto.f32, valid_shape=[2, one], blayout="ColMajor")
            l_prev_tile = pto.alloc_tile(shape=[8, 1], dtype=pto.f32, valid_shape=[2, one], blayout="ColMajor")
            m_next_tile = pto.alloc_tile(shape=[8, 1], dtype=pto.f32, valid_shape=[2, one], blayout="ColMajor")
            l_next_tile = pto.alloc_tile(shape=[8, 1], dtype=pto.f32, valid_shape=[2, one], blayout="ColMajor")
            alpha_tile = pto.alloc_tile(shape=[8, 1], dtype=pto.f32, valid_shape=[2, one], blayout="ColMajor")
            beta_tile = pto.alloc_tile(shape=[8, 1], dtype=pto.f32, valid_shape=[2, one], blayout="ColMajor")
            flash_attention_online_softmax_compute_helper(
                s_tile, p_tile,
                m_prev_tile, l_prev_tile,
                m_next_tile, l_next_tile,
                alpha_tile, beta_tile,
                0, 2, BLOCK,
            )
        """
    ),
    "flash_attention.online_softmax_store": _fixture(
        f"""
        @pto.tileop
        def flash_attention_online_softmax_store_helper(
            s_tile: pto.Tile,
            p_tile: pto.Tile,
            m_prev_tile: pto.Tile,
            l_prev_tile: pto.Tile,
            m_next_tile: pto.Tile,
            l_next_tile: pto.Tile,
            alpha_tile: pto.Tile,
            beta_tile: pto.Tile,
            row_start: pto.i32,
            row_stop: pto.i32,
            valid_cols: pto.i32,
        ):
            for row in range(row_start, row_stop, 1):
                col_mask = pto.make_mask(pto.f32, valid_cols)
                p_row = pto.vexp(pto.vlds(s_tile[row, 0:]), col_mask)
                m_next = scalar.load(m_prev_tile[row, 0])
                l_next = scalar.load(l_prev_tile[row, 0])
                alpha = scalar.load(alpha_tile[row, 0])
                beta = scalar.load(beta_tile[row, 0])
            {SNIPPET_PLACEHOLDER}


        @pto.jit(target="a5")
        def flash_attention_online_softmax_store_probe(*, BLOCK: pto.const_expr = 16):
            one = 1
            s_tile = pto.alloc_tile(shape=[8, BLOCK], dtype=pto.f32, valid_shape=[2, BLOCK])
            p_tile = pto.alloc_tile(shape=[8, BLOCK], dtype=pto.f32, valid_shape=[2, BLOCK])
            m_prev_tile = pto.alloc_tile(shape=[8, 1], dtype=pto.f32, valid_shape=[2, one], blayout="ColMajor")
            l_prev_tile = pto.alloc_tile(shape=[8, 1], dtype=pto.f32, valid_shape=[2, one], blayout="ColMajor")
            m_next_tile = pto.alloc_tile(shape=[8, 1], dtype=pto.f32, valid_shape=[2, one], blayout="ColMajor")
            l_next_tile = pto.alloc_tile(shape=[8, 1], dtype=pto.f32, valid_shape=[2, one], blayout="ColMajor")
            alpha_tile = pto.alloc_tile(shape=[8, 1], dtype=pto.f32, valid_shape=[2, one], blayout="ColMajor")
            beta_tile = pto.alloc_tile(shape=[8, 1], dtype=pto.f32, valid_shape=[2, one], blayout="ColMajor")
            flash_attention_online_softmax_store_helper(
                s_tile, p_tile,
                m_prev_tile, l_prev_tile,
                m_next_tile, l_next_tile,
                alpha_tile, beta_tile,
                0, 2, BLOCK,
            )
        """
    ),
    "flash_attention.simt_materialize": _fixture(
        f"""
        {SNIPPET_PLACEHOLDER}


        @pto.jit(target="a5")
        def flash_attention_simt_materialize_probe():
            meta_tile = pto.alloc_tile(shape=[1, 8], dtype=pto.i32, valid_shape=[1, 3])
            meta_ptr = meta_tile.as_ptr()
            valid_rows = pto.const(1, dtype=pto.i32)
            valid_cols = pto.const(2, dtype=pto.i32)
            materialize_tile_bounds(meta_ptr, valid_rows, valid_cols)
        """
    ),
    "flash_attention.simt_blend": _fixture(
        f"""
        {SNIPPET_PLACEHOLDER}


        @pto.jit(target="a5")
        def flash_attention_simt_blend_probe(*, BLOCK: pto.const_expr = 8):
            one = 1
            o_prev_tile = pto.alloc_tile(shape=[8, BLOCK], dtype=pto.f32, valid_shape=[2, BLOCK])
            pv_tile = pto.alloc_tile(shape=[8, BLOCK], dtype=pto.f32, valid_shape=[2, BLOCK])
            alpha_tile = pto.alloc_tile(shape=[8, 1], dtype=pto.f32, valid_shape=[2, one], blayout="ColMajor")
            beta_tile = pto.alloc_tile(shape=[8, 1], dtype=pto.f32, valid_shape=[2, one], blayout="ColMajor")
            o_next_tile = pto.alloc_tile(shape=[8, BLOCK], dtype=pto.f32, valid_shape=[2, BLOCK])
            blend_output_rows(
                o_prev_tile,
                pv_tile,
                alpha_tile,
                beta_tile,
                o_next_tile,
                pto.const(0, dtype=pto.i32),
                pto.const(2, dtype=pto.i32),
                pto.const(BLOCK, dtype=pto.i32),
            )
        """
    ),
    "gemm.cube_helper": _fixture(
        f"""
        @pto.tileop
        def gemm_tile(
            a_mat: pto.Tile,
            b_mat: pto.Tile,
            o_tile: pto.Tile,
            a_l0a: pto.Tile,
            b_l0b: pto.Tile,
            o_acc: pto.Tile,
        ):
            {SNIPPET_PLACEHOLDER}


        @pto.jit(target="a5", mode="explicit")
        def gemm_tile_probe(*, BLOCK_M: pto.const_expr = 64, BLOCK_K: pto.const_expr = 64, BLOCK_N: pto.const_expr = 64):
            a_mat = pto.alloc_tile(shape=[BLOCK_M, BLOCK_K], dtype=pto.f32, memory_space=pto.MemorySpace.MAT, valid_shape=[BLOCK_M, BLOCK_K])
            b_mat = pto.alloc_tile(shape=[BLOCK_K, BLOCK_N], dtype=pto.f32, memory_space=pto.MemorySpace.MAT, valid_shape=[BLOCK_K, BLOCK_N])
            o_tile = pto.alloc_tile(shape=[BLOCK_M, BLOCK_N], dtype=pto.f32, valid_shape=[BLOCK_M, BLOCK_N])
            a_l0a = pto.alloc_tile(shape=[BLOCK_M, BLOCK_K], dtype=pto.f32, memory_space=pto.MemorySpace.LEFT, valid_shape=[BLOCK_M, BLOCK_K])
            b_l0b = pto.alloc_tile(shape=[BLOCK_K, BLOCK_N], dtype=pto.f32, memory_space=pto.MemorySpace.RIGHT, valid_shape=[BLOCK_K, BLOCK_N])
            o_acc = pto.alloc_tile(shape=[BLOCK_M, BLOCK_N], dtype=pto.f32, memory_space=pto.MemorySpace.ACC, valid_shape=[BLOCK_M, BLOCK_N])
            gemm_tile(a_mat, b_mat, o_tile, a_l0a, b_l0b, o_acc)
        """
    ),
    "gemm.jit_kernel": _fixture(
        f"""
        @pto.tileop
        def gemm_tile(
            a_mat: pto.Tile,
            b_mat: pto.Tile,
            o_tile: pto.Tile,
            a_l0a: pto.Tile,
            b_l0b: pto.Tile,
            o_acc: pto.Tile,
        ):
            m = a_mat.valid_shape[0]
            k = a_mat.valid_shape[1]
            n = b_mat.valid_shape[1]
            pto.mte_l1_l0a(a_mat.as_ptr(), a_l0a.as_ptr(), m, k)
            pto.mte_l1_l0b(b_mat.as_ptr(), b_l0b.as_ptr(), k, n)
            pto.mad(a_l0a.as_ptr(), b_l0b.as_ptr(), o_acc.as_ptr(), m, n, k)
            pto.mte_l0c_ub(o_acc.as_ptr(), o_tile.as_ptr(), m, n, n, n, 0)


        {SNIPPET_PLACEHOLDER}
        """
    ),
    "pipe_communication.c2v_global_declaration": _fixture(
        f"""
        @pto.jit(target="a3")
        def pipe_communication_c2v_global_declaration_probe(
            gm_slot_buffer: pto.gm_ptr(pto.f32),
        ):
            gm_slots = pto.make_tensor_view(gm_slot_buffer, shape=[16, 16], strides=[16, 1])
            {SNIPPET_PLACEHOLDER}
        """
    ),
    "pipe_communication.c2v_global_producer": _fixture(
        f"""
        @pto.jit(target="a3")
        def pipe_communication_c2v_global_producer_probe(
            gm_slot_buffer: pto.gm_ptr(pto.f32),
            src: pto.gm_ptr(pto.f32),
        ):
            gm_view = pto.make_tensor_view(gm_slot_buffer, shape=[16, 16], strides=[16, 1])
            c2v = pto.pipe.c2v(
                gm_slot_tensor=gm_view,
                id=0,
                nosplit=True,
            )

            a_part = pto.partition_view(
                pto.make_tensor_view(src, shape=[16, 16], strides=[16, 1]),
                offsets=[0, 0], sizes=[16, 16])
            a_tile = pto.alloc_tile(shape=[16, 16], dtype=pto.f32)

            {SNIPPET_PLACEHOLDER}
        """
    ),
    "pipe_communication.c2v_global_consumer": _fixture(
        f"""
        @pto.jit(target="a3")
        def pipe_communication_c2v_global_consumer_probe(
            gm_slot_buffer: pto.gm_ptr(pto.f32),
            dst: pto.gm_ptr(pto.f32),
        ):
            gm_view = pto.make_tensor_view(gm_slot_buffer, shape=[16, 16], strides=[16, 1])
            c2v = pto.pipe.c2v(
                gm_slot_tensor=gm_view,
                id=0,
                nosplit=True,
            )

            b_tile = pto.alloc_tile(shape=[16, 16], dtype=pto.f32)
            b_part = pto.partition_view(
                pto.make_tensor_view(dst, shape=[16, 16], strides=[16, 1]),
                offsets=[0, 0], sizes=[16, 16])

            {SNIPPET_PLACEHOLDER}
        """
    ),
    "pipe_communication.v2c_global_declaration": _fixture(
        f"""
        @pto.jit(target="a3")
        def pipe_communication_v2c_global_declaration_probe(
            gm_slot_buffer: pto.gm_ptr(pto.f32),
        ):
            gm_slots = pto.make_tensor_view(gm_slot_buffer, shape=[16, 16], strides=[16, 1])
            {SNIPPET_PLACEHOLDER}
        """
    ),
    "pipe_communication.v2c_global_producer": _fixture(
        f"""
        @pto.jit(target="a3")
        def pipe_communication_v2c_global_producer_probe(
            gm_slot_buffer: pto.gm_ptr(pto.f32),
            src: pto.gm_ptr(pto.f32),
        ):
            gm_view = pto.make_tensor_view(gm_slot_buffer, shape=[16, 16], strides=[16, 1])
            v2c = pto.pipe.v2c(
                gm_slot_tensor=gm_view,
                id=0,
                nosplit=True,
            )

            src_tile = pto.alloc_tile(shape=[16, 16], dtype=pto.f32)

            {SNIPPET_PLACEHOLDER}
        """
    ),
    "pipe_communication.v2c_global_consumer": _fixture(
        f"""
        @pto.jit(target="a3")
        def pipe_communication_v2c_global_consumer_probe(
            gm_slot_buffer: pto.gm_ptr(pto.f32),
            dst: pto.gm_ptr(pto.f32),
        ):
            gm_view = pto.make_tensor_view(gm_slot_buffer, shape=[16, 16], strides=[16, 1])
            v2c = pto.pipe.v2c(
                gm_slot_tensor=gm_view,
                id=0,
                nosplit=True,
            )

            dst_tile = pto.alloc_tile(shape=[16, 16], dtype=pto.f32)

            {SNIPPET_PLACEHOLDER}
        """
    ),
    "pipe_communication.c2v_local_declaration": _fixture(
        f"""
        @pto.jit(target="a5")
        def pipe_communication_c2v_local_declaration_probe():
            {SNIPPET_PLACEHOLDER}
        """
    ),
    "pipe_communication.c2v_local_import": _fixture(
        f"""
        from mlir.dialects import func
        from mlir.ir import InsertionPoint
        from ptodsl._tracing import current_session


        def declare_vector_kernel_peer():
            session = current_session()
            fn_ty = func.FunctionType.get([], [])
            with InsertionPoint(session._function_symbol_table):
                peer = func.FuncOp("vector_kernel", fn_ty)
            entry = peer.add_entry_block()
            with session.enter_function(peer), InsertionPoint(entry):
                pto.reserve_buffer("c2v_fifo", size=8192, location="vec")
                func.ReturnOp([])


        @pto.jit(target="a5")
        def pipe_communication_c2v_local_import_probe():
            declare_vector_kernel_peer()
            {SNIPPET_PLACEHOLDER}
        """
    ),
    "pipe_communication.c2v_local_producer": _fixture(
        f"""
        from mlir.dialects import func
        from mlir.ir import InsertionPoint
        from ptodsl._tracing import current_session


        def declare_vector_kernel_peer():
            session = current_session()
            fn_ty = func.FunctionType.get([], [])
            with InsertionPoint(session._function_symbol_table):
                peer = func.FuncOp("vector_kernel", fn_ty)
            entry = peer.add_entry_block()
            with session.enter_function(peer), InsertionPoint(entry):
                pto.reserve_buffer("c2v_fifo", size=8192, location="vec")
                func.ReturnOp([])


        @pto.jit(target="a5")
        def pipe_communication_c2v_local_producer_probe(
            src: pto.gm_ptr(pto.f32),
        ):
            declare_vector_kernel_peer()
            c2v_buf = pto.import_reserved_buffer("c2v_fifo", peer_func="vector_kernel")
            c2v_peer = pto.pipe.c2v(
                slot_size=1024,
                consumer_buf=c2v_buf,
                id=0,
            )
            src_tile = pto.alloc_tile(shape=[16, 16], dtype=pto.f32)

            {SNIPPET_PLACEHOLDER}
        """
    ),
    "pipe_communication.c2v_local_consumer": _fixture(
        f"""
        @pto.jit(target="a5")
        def pipe_communication_c2v_local_consumer_probe(
            dst: pto.gm_ptr(pto.f32),
        ):
            c2v_buf = pto.reserve_buffer("c2v_fifo", size=8192, location="vec")
            c2v = pto.pipe.c2v(
                slot_size=1024,
                consumer_buf=c2v_buf,
                id=0,
            )
            dst_part = pto.partition_view(
                pto.make_tensor_view(dst, shape=[16, 16], strides=[16, 1]),
                offsets=[0, 0], sizes=[16, 16])
            dst_tile = pto.alloc_tile(shape=[16, 16], dtype=pto.f32)

            {SNIPPET_PLACEHOLDER}
        """
    ),
    "pipe_communication.bidirectional_local_declaration": _fixture(
        f"""
        @pto.jit(target="a5")
        def pipe_communication_bidirectional_local_declaration_probe():
            c2v_buf = pto.reserve_buffer("c2v_fifo", size=8192, location="vec")
            v2c_buf = pto.reserve_buffer("v2c_fifo", size=8192, location="mat")
            {SNIPPET_PLACEHOLDER}
        """
    ),
    "pipe_communication.c2v_global": _fixture(
        f"""
        {SNIPPET_PLACEHOLDER}


        class _PipeCommunicationC2VGlobalProbe:
            def compile(self, **kwargs):
                cube_compiled = cube_producer.compile(**kwargs)
                cube_compiled.verify()
                return vector_consumer.compile(**kwargs)


        pipe_communication_c2v_global_probe = _PipeCommunicationC2VGlobalProbe()
        """
    ),
}
