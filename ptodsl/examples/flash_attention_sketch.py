# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""
Flash Attention compile-only demo.

This file is a compileable PTODSL demo whose current milestone is MLIR
emission, inspection, and API review. The goal is to make the intended API
layering explicit and keep the semantic contracts clean:

    emit_flash_attention_mlir(...) compile/inspect wrapper
      └─ flash_attention_kernel   (@pto.jit, mode="explicit")
           ├─ Tile Ops                 tile.load / tile.store at the GM↔UB boundary
           ├─ explicit orchestration   mte_load / pipe_barrier / pointer sequencing
           ├─ @pto.tileop             matrix products (QK^T and P@V, cube-style)
           ├─ @pto.tileop             row-wise online softmax (vector-style)
           └─ @pto.simt               scalar metadata and output blending

Design rules illustrated here:

1. ``@pto.jit`` marks a launchable kernel template.  It owns JIT compilation,
   cache lookup, and artifact emission, instead of forcing users to hop through
   extra builder objects for common cases.
2. The Python wrapper owns compile/inspection concerns such as selecting
   specialization knobs and returning the emitted MLIR text for review.
3. ``@pto.jit`` also owns the top-level logical tiling, tile allocation, and
   loop scheduling for one already-selected per-head 2D slice.  The per-block
   DMA and barrier choreography is delegated to explicit orchestration.
4. explicit mode owns the per-block execution sandwich: stage the current K/V
   block with explicit micro-instructions, synchronize, call hardware-bound
   sub-kernels, and manage scratch/state.
5. ``@pto.jit`` may use tile ops such as ``tile.load`` / ``tile.store`` at the logical
   scheduling boundary, but explicit mode can also express GM<->UB movement
   directly. Once execution enters explicit orchestration, MTE micro-instructions
   such as ``mte_load`` are used instead of tile ops where needed.
   ``mte_load`` / ``mte_store`` accept partitions and tiles directly,
   deriving strides and burst sizes from the type metadata.
6. ``tileop`` / ``simt`` are the public helper boundaries. ``@pto.tileop``
   covers both vector-style and cube-style helper bodies, but transient
   hardware-local values still do not cross the function boundary. Data crosses
   the boundary through UB-backed tiles or typed UB pointers only.
7. Named sub-kernels are reusable wherever their parameter contract is
   satisfied. This sketch uses the explicit ``@pto.jit(mode="explicit")`` path
   because it needs user-ordered DMA and phase barriers; smaller kernels can
   stay in auto mode and rely on tile-atomic staging instead.
8. Online-softmax state is made explicit with ping-pong tiles
   (``m_prev``/``m_next``, ``l_prev``/``l_next``, ``o_prev``/``o_next``).
   Hiding these dependencies with in-place aliases makes the algorithm harder
   to read and obscures what the DSL needs to express.

Because PTODSL rewrites JIT function ASTs by default, runtime Python
``if`` / ``for range(...)`` in this demo lowers to structured MLIR control
flow. Use ``pto.const_expr`` / ``pto.static_range`` only for intentionally
compile-time control flow.

Scalar literals and simple index/integer conversions are also written in the
authored PTODSL surface. The current frontend lowers these through tracing
instead of forcing authors to spell ``pto.const(...)`` or ``index_cast(...)``
at every use site.
"""

from ptodsl import pto, scalar


def _min_index(lhs, rhs):
    return scalar.select(
        lhs < rhs,
        lhs,
        rhs,
    )


def _block_valid_extent(total, block_index, block_size):
    return _min_index(total - block_index * block_size, pto.const(block_size))


# ═══════════════════════════════════════════════════════════════════════════════
# Public API sketch
# ═══════════════════════════════════════════════════════════════════════════════
#
# This section shows the current compile-only public surface. The split follows
# the common industry pattern:
#
# - a user-facing tensor wrapper
# - a launchable JIT kernel entry
# - hardware-bound sub-kernels below it
#
# The low-level kernel body should not double as the user-facing runtime API.
#
# Two intended usage styles for the current compile-only milestone:
#
# 1. One-shot MLIR emission:
#      mlir_text = emit_flash_attention_mlir(head_dim=128, causal=True)
#
# 2. Compile first, then inspect:
#      compiled = flash_attention_kernel.compile(BLOCK_Q=128, BLOCK_KV=128, CAUSAL=True)
#      mlir_text = compiled.mlir_text()

def emit_flash_attention_mlir(
    *,
    head_dim=128,
    causal=False,
    block_q=128,
    block_kv=128,
):
    """
    Compile the flash-attention sketch and return its MLIR text.

    The current milestone for this demo is compile / inspect / review, not
    runtime launch. The wrapper therefore only specializes the JIT kernel and
    returns the emitted MLIR text.
    """
    compiled = flash_attention_kernel.compile(
        BLOCK_Q=block_q,
        BLOCK_KV=block_kv,
        HEAD_DIM=head_dim,
        CAUSAL=causal,
    )
    return compiled.mlir_text()

@pto.jit(target="a5", mode="explicit")
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
    HEAD_DIM: pto.const_expr = 128,
    CAUSAL: pto.const_expr = False,
    NUM_STAGES: pto.const_expr = 2,
):
    """
    Launchable device entry.

    ``@pto.jit`` is the compile boundary.  Inputs/outputs at this
    boundary are explicit GM pointers plus runtime shape metadata; PTO-specific
    ``TensorView`` descriptors are materialized inside the JIT body rather than
    exposed in the public signature.  Tile sizes and specialization knobs
    remain constexpr metadata.

    A launch instance is responsible for one ``(batch, head)`` slice.  The
    per-slice logical tiling is expressed directly in this top-level JIT entry.
    """
    q_strides = [seq_q * heads * dim, heads * dim, dim, 1]
    kv_strides = [seq_k * heads * dim, heads * dim, dim, 1]
    o_strides = [seq_q * heads * dim, heads * dim, dim, 1]

    q_view = pto.make_tensor_view(Q_ptr, shape=[batch, seq_q, heads, dim], strides=q_strides)
    k_view = pto.make_tensor_view(K_ptr, shape=[batch, seq_k, heads, dim], strides=kv_strides)
    v_view = pto.make_tensor_view(V_ptr, shape=[batch, seq_k, heads, dim], strides=kv_strides)
    o_view = pto.make_tensor_view(O_ptr, shape=[batch, seq_q, heads, dim], strides=o_strides)

    # Make the SPMD launch contract explicit in the authored surface.
    # This sketch uses one block per (batch, head) slice and does not further
    # split work across subblocks, but the runtime indices still belong in a
    # realistic launchable entry.
    block_idx = pto.get_block_idx()
    block_num = pto.get_block_num()
    subblock_idx = pto.get_subblock_idx()
    subblock_num = pto.get_subblock_num()

    # Current mapping:
    # - launch grid = batch * heads
    # - block_idx selects one (batch, head) slice
    # - subblock_idx is queried explicitly, but no extra intra-block partition
    #   is modeled in this sketch yet
    _ = block_num
    _ = subblock_idx
    _ = subblock_num

    batch_idx = block_idx // heads
    head_idx = block_idx % heads

    q_head = pto.partition_view(
        q_view,
        offsets=[batch_idx, 0, head_idx, 0],
        sizes=[1, seq_q, 1, dim],
    )
    k_head = pto.partition_view(
        k_view,
        offsets=[batch_idx, 0, head_idx, 0],
        sizes=[1, seq_k, 1, dim],
    )
    v_head = pto.partition_view(
        v_view,
        offsets=[batch_idx, 0, head_idx, 0],
        sizes=[1, seq_k, 1, dim],
    )
    o_head = pto.partition_view(
        o_view,
        offsets=[batch_idx, 0, head_idx, 0],
        sizes=[1, seq_q, 1, dim],
    )

    Br = BLOCK_Q
    Bc = BLOCK_KV
    D = HEAD_DIM
    full_br = pto.const(Br)
    full_bc = pto.const(Bc)
    one = pto.const(1)

    q_blocks = (seq_q + Br - 1) // Br
    kv_blocks = (seq_k + Bc - 1) // Bc

    # Physical tile shape remains static. Runtime tails live in valid_shape.
    # Cube bridge sources are MAT-backed so they can feed LEFT/RIGHT staging.
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

    # Cube-local scratch is explicit; it should not be conflated with UB tiles.
    q_l0a = pto.alloc_tile(shape=[Br, D], dtype=pto.f32, memory_space=pto.MemorySpace.LEFT, valid_shape=[full_br, dim])
    p_l0a = pto.alloc_tile(shape=[Br, Bc], dtype=pto.f32, memory_space=pto.MemorySpace.LEFT, valid_shape=[full_br, full_bc])
    rhs_l0b = pto.alloc_tile(shape=[Bc, D], dtype=pto.f32, memory_space=pto.MemorySpace.RIGHT, valid_shape=[full_bc, dim])
    qk_acc_tile = pto.alloc_tile(shape=[Br, Bc], dtype=pto.f32, memory_space=pto.MemorySpace.ACC, valid_shape=[full_br, full_bc])
    pv_acc_tile = pto.alloc_tile(shape=[Br, D], dtype=pto.f32, memory_space=pto.MemorySpace.ACC, valid_shape=[full_br, dim])

    # SIMT metadata buffer.  A tiny raw-pointer island is acceptable at the
    # explicit-orchestration boundary because this is scalar control data, not
    # user-facing math.
    meta_tile = pto.alloc_tile(shape=[1, 8], dtype=pto.i32, valid_shape=[1, 3])
    meta_ptr = meta_tile.as_ptr()

    for qi in range(0, q_blocks, 1):
        q_rows = _block_valid_extent(seq_q, qi, Br)
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

        # Initial online-softmax state for this Q block.
        # ``CAUSAL`` is threaded at the API boundary even though the masking
        # details are intentionally omitted from this design-focused sketch.
        m_prev_tile.fill(float("-inf"))
        l_prev_tile.fill(0.0)
        o_prev_tile.fill(0.0)

        m_cur = m_prev_tile
        l_cur = l_prev_tile
        o_cur = o_prev_tile
        for kj in range(0, kv_blocks, 1):
            kv_rows = _block_valid_extent(seq_k, kj, Bc)
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

            kv_block_process(
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

            # Loop-carried state stays visible in Python while AST rewrite
            # lowers it to scf.iter_args / scf.yield.
            m_cur = m_next_tile
            l_cur = l_next_tile
            o_cur = o_next_tile

        pto.tile.store(o_cur, o_part)


# ═══════════════════════════════════════════════════════════════════════════════
# Hardware-bound sub-kernels
# ═══════════════════════════════════════════════════════════════════════════════
#
# Boundary contract:
# - Tile arguments are UB-backed or cube-local buffers carrying addressable
#   storage.
# - No vector register escapes a tile-op helper.
# - No implicit global-memory access happens inside these kernels.


@pto.tileop
def qk_matmul(
    q_mat: pto.Tile,       # MAT, [Br, dim]
    k_mat: pto.Tile,       # MAT, [Bc, dim]
    q_l0a: pto.Tile,       # LEFT scratch
    k_l0b: pto.Tile,       # RIGHT scratch
    s_acc: pto.Tile,       # ACC scratch
    s_tile: pto.Tile,      # UB, [Br, Bc] output
):
    """
    Compute ``S = Q @ K^T`` for one attention block.

    The key point for the redesign is that the cube kernel consumes MAT tiles and
    explicit cube-local scratch, rather than pretending a logical scheduling tile can also stand
    in for LEFT/RIGHT/ACC state.
    """
    m = q_mat.valid_shape[0]
    k = q_mat.valid_shape[1]
    n = k_mat.valid_shape[0]

    # Caller owns scratch lifetime.  The cube kernel only expresses dataflow.
    pto.mte_l1_l0a(q_mat.as_ptr(), q_l0a.as_ptr(), m, k)
    pto.mte_l1_l0b(k_mat.as_ptr(), k_l0b.as_ptr(), k, n, transpose=True)
    pto.mad(q_l0a.as_ptr(), k_l0b.as_ptr(), s_acc.as_ptr(), m, n, k)
    pto.mte_l0c_ub(s_acc.as_ptr(), s_tile.as_ptr(), m, n, n, n, 0)


@pto.tileop
def pv_matmul(
    p_mat: pto.Tile,       # MAT, [Br, Bc]
    v_mat: pto.Tile,       # MAT, [Bc, dim]
    p_l0a: pto.Tile,       # LEFT scratch (reused)
    v_l0b: pto.Tile,       # RIGHT scratch (reused)
    pv_acc: pto.Tile,      # ACC scratch (reused)
    pv_tile: pto.Tile,     # UB, [Br, dim] output
):
    """
    Compute ``PV = P @ V`` for the current block.

    This keeps the second matrix product on the cube path as well, instead of
    accidentally collapsing it into an elementwise vector expression.
    """
    m = p_mat.valid_shape[0]
    k = p_mat.valid_shape[1]
    n = v_mat.valid_shape[1]

    pto.mte_l1_l0a(p_mat.as_ptr(), p_l0a.as_ptr(), m, k)
    pto.mte_l1_l0b(v_mat.as_ptr(), v_l0b.as_ptr(), k, n)
    pto.mad(p_l0a.as_ptr(), v_l0b.as_ptr(), pv_acc.as_ptr(), m, n, k)
    pto.mte_l0c_ub(pv_acc.as_ptr(), pv_tile.as_ptr(), m, n, n, n, 0)


@pto.tileop
def online_softmax_rows(
    s_tile: pto.Tile,          # UB, [Br, Bc]
    p_tile: pto.Tile,          # UB, [Br, Bc], output
    m_prev_tile: pto.Tile,     # UB, [Br, 1]
    l_prev_tile: pto.Tile,     # UB, [Br, 1]
    m_next_tile: pto.Tile,     # UB, [Br, 1], output
    l_next_tile: pto.Tile,     # UB, [Br, 1], output
    alpha_tile: pto.Tile,      # UB, [Br, 1], output
    beta_tile: pto.Tile,       # UB, [Br, 1], output
    row_start: pto.i32,
    row_stop: pto.i32,
    valid_cols: pto.i32,
):
    """
    Per-row online softmax update.

    For each active row::

        m_next = max(m_prev, row_max(S))
        P      = exp(S - m_next)
        l_next = l_prev * exp(m_prev - m_next) + row_sum(P)
        alpha  = l_prev * exp(m_prev - m_next) / l_next
        beta   = 1 / l_next

    ``alpha`` and ``beta`` are kept explicitly because the output update needs
    both the old accumulator and the newly computed ``P @ V`` contribution.
    """
    for row in range(row_start, row_stop, 1):
        col_mask = pto.make_mask(pto.f32, valid_cols)

        s_row = pto.vlds(s_tile[row, 0:])
        m_prev = scalar.load(m_prev_tile[row, 0])
        l_prev = scalar.load(l_prev_tile[row, 0])

        row_max = pto.vcgmax(s_row, col_mask)
        m_next = scalar.max(m_prev, row_max)

        s_shifted = pto.vsubs(s_row, m_next, col_mask)
        p_row = pto.vexp(s_shifted, col_mask)

        row_sum = pto.vcgadd(p_row, col_mask)
        l_scaled = l_prev * scalar.exp(m_prev - m_next)
        l_next = l_scaled + row_sum

        alpha = l_scaled / l_next
        beta = 1.0 / l_next

        pto.vsts(p_row, p_tile[row, 0:], col_mask)
        scalar.store(m_next, m_next_tile[row, 0])
        scalar.store(l_next, l_next_tile[row, 0])
        scalar.store(alpha, alpha_tile[row, 0])
        scalar.store(beta, beta_tile[row, 0])


@pto.simt
def blend_output_rows(
    o_prev_tile: pto.Tile,      # UB, [Br, dim]
    pv_tile: pto.Tile,          # UB, [Br, dim]
    alpha_tile: pto.Tile,       # UB, [Br, 1]
    beta_tile: pto.Tile,        # UB, [Br, 1]
    o_next_tile: pto.Tile,      # UB, [Br, dim], output
    row_start: pto.i32,
    row_stop: pto.i32,
    valid_dim: pto.i32,
):
    """
    Update the output accumulator with SIMT-style scalar element work::

        O_next[row, col] = alpha[row] * O_prev[row, col] + beta[row] * PV[row, col]

    This intentionally contrasts with ``online_softmax_rows``: the softmax step
    stays on the SIMD path because it is dominated by row-wise vector math,
    while the final blend is expressed here as explicit scalar work-items over
    the tile domain.
    """
    for row in range(row_start, row_stop, 1):
        alpha = scalar.load(alpha_tile[row, 0])
        beta = scalar.load(beta_tile[row, 0])

        for col in range(0, valid_dim, 1):
            o_prev = scalar.load(o_prev_tile[row, col])
            pv_val = scalar.load(pv_tile[row, col])

            o_next = alpha * o_prev + beta * pv_val
            scalar.store(o_next, o_next_tile[row, col])


@pto.simt
def materialize_tile_bounds(
    meta_ptr: pto.ptr(pto.i32, pto.MemorySpace.UB),   # [out] {row_start, row_stop, valid_cols}
    valid_rows: pto.i32,
    valid_cols: pto.i32,
):
    """
    Materialize tile-local loop bounds for the current block.

    The SIMT kernel stays intentionally small here: it is responsible for
    scalar control metadata, not for rewriting the vector or cube logic.
    """
    scalar.store(0, meta_ptr + 0)
    scalar.store(valid_rows, meta_ptr + 1)
    scalar.store(valid_cols, meta_ptr + 2)


# ═══════════════════════════════════════════════════════════════════════════════
# Level 2: explicit orchestration — one KV block worth of execution
# ═══════════════════════════════════════════════════════════════════════════════


def kv_block_process(
    q_mat: pto.Tile,                 # MAT, reused across inner KV loop
    k_part: pto.PartitionTensorView, # GM view for current K block
    v_part: pto.PartitionTensorView, # GM view for current V block
    k_mat: pto.Tile,                 # MAT scratch
    v_mat: pto.Tile,                 # MAT scratch
    o_prev_tile: pto.Tile,           # UB state
    o_next_tile: pto.Tile,           # UB state
    m_prev_tile: pto.Tile,           # UB state
    l_prev_tile: pto.Tile,           # UB state
    m_next_tile: pto.Tile,           # UB state
    l_next_tile: pto.Tile,           # UB state
    s_tile: pto.Tile,                # UB scratch for QK^T
    p_tile: pto.Tile,                # UB scratch for probabilities
    p_mat: pto.Tile,                 # MAT scratch for probabilities
    pv_tile: pto.Tile,               # UB scratch for P@V
    alpha_tile: pto.Tile,            # UB scratch
    beta_tile: pto.Tile,             # UB scratch
    q_l0a: pto.Tile,                 # LEFT scratch for Q
    p_l0a: pto.Tile,                 # LEFT scratch for P
    rhs_l0b: pto.Tile,               # RIGHT scratch, reused by K/V
    qk_acc_tile: pto.Tile,           # ACC scratch for QK^T
    pv_acc_tile: pto.Tile,           # ACC scratch for P@V
    meta_ptr: pto.ptr(pto.i32, pto.MemorySpace.UB),
):
    """
    Process one KV block against an already-loaded Q tile.

    The explicit-mode body owns:
    - staging the current K/V block into reusable UB scratch with explicit
      DMA-style micro-instructions,
    - synchronizing the hand-off between MTE, cube, simd, and simt stages,
    - wiring together the explicit state transition
      (prev -> next for m/l/o).
    """
    # Current-block GM->MAT staging via explicit ptr-based DMA parameters.
    rows = k_mat.valid_shape[0]
    cols = k_mat.valid_shape[1]
    row_bytes = cols * pto.bytewidth(pto.f32)
    gm_row_stride = k_part.strides[0] * pto.bytewidth(pto.f32)
    mat_row_stride = k_mat.shape[1] * pto.bytewidth(pto.f32)
    pto.mte_load(
        k_part.as_ptr(),
        k_mat.as_ptr(),
        0,
        row_bytes,
        nburst=(rows, gm_row_stride, mat_row_stride),
    )
    pto.mte_load(
        v_part.as_ptr(),
        v_mat.as_ptr(),
        0,
        row_bytes,
        nburst=(rows, gm_row_stride, mat_row_stride),
    )
    pto.pipe_barrier(pto.Pipe.ALL)

    materialize_tile_bounds(
        meta_ptr,
        q_mat.valid_shape[0],
        k_mat.valid_shape[0],
    )
    row_start = scalar.load(meta_ptr + 0)
    row_stop = scalar.load(meta_ptr + 1)
    valid_cols = scalar.load(meta_ptr + 2)

    # 1. S = Q @ K^T
    qk_matmul(q_mat, k_mat, q_l0a, rhs_l0b, qk_acc_tile, s_tile)
    pto.pipe_barrier(pto.Pipe.ALL)

    # 2. Row-wise online softmax over S
    online_softmax_rows(
        s_tile,
        p_tile,
        m_prev_tile,
        l_prev_tile,
        m_next_tile,
        l_next_tile,
        alpha_tile,
        beta_tile,
        row_start,
        row_stop,
        valid_cols,
    )
    pto.pipe_barrier(pto.Pipe.ALL)

    # Stage the probability tile onto the cube MAT path.
    pto.tile.mov(p_tile, p_mat)
    pto.pipe_barrier(pto.Pipe.ALL)

    # 3. PV = P @ V
    pv_matmul(p_mat, v_mat, p_l0a, rhs_l0b, pv_acc_tile, pv_tile)
    pto.pipe_barrier(pto.Pipe.ALL)

    # 4. O_next = alpha * O_prev + beta * PV
    blend_output_rows(
        o_prev_tile,
        pv_tile,
        alpha_tile,
        beta_tile,
        o_next_tile,
        row_start,
        row_stop,
        v_mat.valid_shape[1],
    )
    pto.pipe_barrier(pto.Pipe.ALL)


# ═══════════════════════════════════════════════════════════════════════════════
# Layer summary
# ═══════════════════════════════════════════════════════════════════════════════
#
# ┌──────────────────────────────────────────────────────────────────────────┐
# │ L0  Python wrapper   emit_flash_attention_mlir(...)                       │
# │                                                                            │
# │   specialize kernel parameters, compile, emit MLIR text                   │
# │                                                                            │
# │   Key idea: current demo goal is compile/inspect, not runtime launch.     │
# ├──────────────────────────────────────────────────────────────────────────┤
# │ L1  @pto.jit(mode="explicit") flash_attention_kernel                      │
# │                                                                            │
# │   flash_attention_kernel.compile(...).mlir_text()                         │
# │   TensorView metadata / alloc_tile / partition_view / tile.load / tile.store      │
# │   outer Q loop + inner KV loop + ping-pong state ownership                │
# │                                                                            │
# │   Key idea: one launchable entry owns both runtime binding and logical     │
# │   tile scheduling.                                                         │
# ├──────────────────────────────────────────────────────────────────────────┤
# │ L2  explicit orchestration  Per-block execution sandwich                  │
# │                                                                            │
# │   explicit mte_load(part, tile) staging for current K/V block,           │
# │   pipe_barrier, call cube/simd/simt sub-kernels,                          │
# │   manage scratch/state hand-off                                            │
# │                                                                            │
# │   Key idea: one place owns the "how this block runs on hardware" story.   │
# ├──────────────────────────────────────────────────────────────────────────┤
# │ @pto.tileop         Matrix-product kernels (cube-style bodies)             │
# │                                                                            │
# │   qk_matmul: Q @ K^T                                                       │
# │   pv_matmul: P @ V                                                         │
# │   explicit LEFT/RIGHT/ACC scratch + UB output                              │
# │                                                                            │
# │   Key idea: UB tiles are inputs/outputs; cube-local state is explicit.    │
# ├──────────────────────────────────────────────────────────────────────────┤
# │ @pto.tileop         Row-wise vector math                                   │
# │                                                                            │
# │   online_softmax_rows                                                      │
# │   vreg stays local; persistent state is written back to UB tiles           │
# │                                                                            │
# │   Key idea: no cross-kernel vreg values, only UB-backed state.            │
# ├──────────────────────────────────────────────────────────────────────────┤
# │ @pto.simt           Scalar metadata and pointwise blend                    │
# │                                                                            │
# │   materialize_tile_bounds / blend_output_rows                              │
# │                                                                            │
# │   Key idea: SIMT handles scalar control facts and scalar tile walks.      │
# └──────────────────────────────────────────────────────────────────────────┘
#
#                       dataflow for one KV block
#
#   jit kernel alloc/schedule
#          │
#          ▼
#   explicit orchestration loads K/V block and sequences the pipeline
#          │
#          ├─ cube:  Q + K  ───────────────► S
#          ├─ simd:  S + (m_prev, l_prev) ─► P, (m_next, l_next), alpha, beta
#          ├─ cube:  P + V  ───────────────► PV
#          └─ simt:  (o_prev, PV, alpha, beta) ─► o_next
#
#   After each KV block:
#     (m_prev, l_prev, o_prev) := (m_next, l_next, o_next)
#
# The important part for the demo is that every cross-stage dependency is
# visible in the surface language and the whole kernel can already be traced to
# MLIR for review.


def main():
    print(emit_flash_attention_mlir())


if __name__ == "__main__":
    main()
