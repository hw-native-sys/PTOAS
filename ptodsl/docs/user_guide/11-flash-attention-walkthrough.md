# 11. Flash Attention Complete Walkthrough

This chapter walks through `examples/flash_attention_sketch.py` layer by layer, tracing a complete flash attention implementation from the user-facing Python wrapper down to hardware-bound sub-kernels. Every API discussed in Chapters 1–10 appears in context here.

The sketch computes **online-softmax flash attention** for one `(batch, head)` slice per launch instance. It partitions Q into blocks along the sequence dimension, iterates over KV blocks for each Q block, and maintains rolling softmax state across KV iterations.

## 11.1 Architecture overview

```
flash_attention(...)           L0  user-facing wrapper
  └─ @pto.jit(entry=True, mode="explicit") flash_attention_kernel
       ├─ Tile Ops                 tile.load / tile.store at the GM↔UB boundary
       ├─ explicit orchestration   mte_load / pipe_barrier / pointer sequencing
       ├─ @pto.tileop               qk_matmul / pv_matmul
       ├─ @pto.tileop               online_softmax_rows
       └─ @pto.simt               materialize_tile_bounds / blend_output_rows
```

The dataflow for one KV block:

```
explicit-mode orchestration loads the K/V block and sequences the pipeline
       │
       ├─ cube:  Q + K  ───────────────► S
       ├─ simd:  S + (m_prev, l_prev) ─► P, (m_next, l_next), alpha, beta
       ├─ cube:  P + V  ───────────────► PV
       └─ simt:  (o_prev, PV, alpha, beta) ─► o_next

After each KV block:
  (m_prev, l_prev, o_prev) := (m_next, l_next, o_next)
```

## 11.2 The Python wrapper

```python
def flash_attention(Q, K, V, *, O=None, causal=False,
                    block_q=128, block_kv=128, stream=None):
    if O is None:
        O = pto.empty_like(Q)

    batch, seq_q, heads, dim = Q.shape
    _, seq_k, _, _ = K.shape

    compiled = flash_attention_kernel.compile(
        BLOCK_Q=block_q, BLOCK_KV=block_kv, CAUSAL=causal,
    )
    compiled[batch * heads, stream](
        Q.data_ptr(), K.data_ptr(), V.data_ptr(), O.data_ptr(),
        batch, seq_q, seq_k, heads, dim,
    )
    return O
```

This is plain Python — no PTO types, no IR. It handles ergonomic runtime concerns:

- **Output allocation**: `pto.empty_like(Q)` when the caller doesn't provide one.
- **Shape extraction**: reads `batch`, `seq_q`, `heads`, `dim` from the framework tensors.
- **Compile + launch**: `flash_attention_kernel.compile(...)` JIT-compiles the kernel with the given constexpr parameters, then launches it with a `[batch * heads]` grid — one block per `(batch, head)` slice.

The wrapper knows nothing about tiles, UB, or pipelines. It is the boundary between the user's tensor world and the PTO device world.

## 11.3 Top-level `@pto.jit(entry=True, mode="explicit")` kernel entry

<!-- ptodsl-doc-test: {"mode":"compile","symbol":"flash_attention_kernel","compile":{"BLOCK_Q":128,"BLOCK_KV":128,"CAUSAL":false,"NUM_STAGES":2}} -->
```python
@pto.jit(target="a5", entry=True, mode="explicit")
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
    NUM_STAGES: pto.const_expr = 2,
):
    # Walkthrough body omitted in this signature overview.
    return
```

The `@pto.jit(entry=True, mode="explicit")` decorator marks the compile + launch boundary. Inputs and outputs arrive as explicit GM pointers plus runtime shape metadata; keyword-only `const_expr` parameters (`BLOCK_Q`, `BLOCK_KV`, `CAUSAL`) are baked at compile time.

### 11.3.1 TensorView construction

<!-- ptodsl-doc-test: {"mode":"compile_fragment","fixture":"flash_attention.l1_tensor_views","symbol":"flash_attention_l1_tensor_views_probe","compile":{"BLOCK_Q":128,"BLOCK_KV":128,"CAUSAL":false,"NUM_STAGES":2}} -->
```python
q_view = pto.make_tensor_view(
    Q_ptr,
    shape=[batch, seq_q, heads, dim],
    strides=[seq_q * heads * dim, heads * dim, dim, 1],
)
k_view = pto.make_tensor_view(
    K_ptr,
    shape=[batch, seq_k, heads, dim],
    strides=[seq_k * heads * dim, heads * dim, dim, 1],
)
v_view = pto.make_tensor_view(
    V_ptr,
    shape=[batch, seq_k, heads, dim],
    strides=[seq_k * heads * dim, heads * dim, dim, 1],
)
o_view = pto.make_tensor_view(
    O_ptr,
    shape=[batch, seq_q, heads, dim],
    strides=[seq_q * heads * dim, heads * dim, dim, 1],
)
```

`make_tensor_view` wraps each explicit GM pointer with a PTO TensorView descriptor — a GM pointer paired with authored shape and stride metadata. These descriptors are what the rest of the kernel uses to address global memory. No data moves yet.

### 11.3.2 SPMD launch contract

<!-- ptodsl-doc-test: {"mode":"compile_fragment","fixture":"flash_attention.l1_tensor_views","symbol":"flash_attention_l1_tensor_views_probe","compile":{"BLOCK_Q":128,"BLOCK_KV":128,"CAUSAL":false,"NUM_STAGES":2}} -->
```python
block_idx = pto.get_block_idx()
block_num = pto.get_block_num()
subblock_idx = pto.get_subblock_idx()
subblock_num = pto.get_subblock_num()

batch_idx = block_idx // heads
head_idx = block_idx % heads
```

The launch grid is `[batch * heads]`. Each block computes one `(batch, head)` slice. `get_block_idx()` returns the current block's linear index; dividing by `heads` recovers the batch and head indices.

### 11.3.3 Per-head view partitioning

<!-- ptodsl-doc-test: {"mode":"compile_fragment","fixture":"flash_attention.l1_partitions","symbol":"flash_attention_l1_partitions_probe","compile":{"BLOCK_Q":128,"BLOCK_KV":128,"CAUSAL":false,"NUM_STAGES":2}} -->
```python
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
```

There is no dedicated `select_head_view` public helper anymore. Each `(batch, head)` working set is sliced from the 4D TensorView with the standard `partition_view(...)` surface, and further logical slicing composes on top of the same primitive.

### 11.3.4 Tile allocation

Three categories of tiles are allocated:

**MAT-backed bridge tiles** — the logical Q/K/V/P blocks that feed the cube path:

<!-- ptodsl-doc-test: {"mode":"compile_fragment","fixture":"flash_attention.l1_tiles","symbol":"flash_attention_l1_tiles_probe","compile":{"BLOCK_Q":128,"BLOCK_KV":128,"HEAD_DIM":128}} -->
```python
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
p_mat = pto.alloc_tile(
    shape=[Br, Bc],
    dtype=pto.f32,
    memory_space=pto.MemorySpace.MAT,
    valid_shape=[full_br, full_bc],
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
pv_tile = pto.alloc_tile(shape=[Br, D], dtype=pto.f32, valid_shape=[full_br, dim])
alpha_tile = pto.alloc_tile(shape=[Br, 1], dtype=pto.f32, valid_shape=[full_br, one], blayout="ColMajor")
beta_tile = pto.alloc_tile(shape=[Br, 1], dtype=pto.f32, valid_shape=[full_br, one], blayout="ColMajor")
```

The walkthrough keeps Q/K/V/P on the MAT path so Cube-kind TileOps consume the same tile objects that the top-level kernel owns. Runtime tails still live in `valid_shape`; the physical tile shapes stay static.

**UB-resident state and scratch tiles** — among the allocations above, the
online-softmax state lives in `o_prev_tile` / `o_next_tile`, `m_prev_tile` /
`m_next_tile`, `l_prev_tile` / `l_next_tile`, plus the scratch tiles
`s_tile`, `p_tile`, `pv_tile`, `alpha_tile`, and `beta_tile`.

The online-softmax algorithm requires **ping-pong state tiles**: `m_prev`/`m_next`, `l_prev`/`l_next`, `o_prev`/`o_next`. After each KV block, `next` becomes `prev` for the following iteration.

**Cube-local scratch tiles** — allocated in specific memory spaces:

<!-- ptodsl-doc-test: {"mode":"compile_fragment","fixture":"flash_attention.l1_tiles","symbol":"flash_attention_l1_tiles_probe","compile":{"BLOCK_Q":128,"BLOCK_KV":128,"HEAD_DIM":128}} -->
```python
q_l0a = pto.alloc_tile(shape=[Br, D], dtype=pto.f32,
                       memory_space=pto.MemorySpace.LEFT, valid_shape=[full_br, dim])
p_l0a = pto.alloc_tile(shape=[Br, Bc], dtype=pto.f32,
                       memory_space=pto.MemorySpace.LEFT, valid_shape=[full_br, full_bc])
rhs_l0b = pto.alloc_tile(shape=[Bc, D], dtype=pto.f32,
                         memory_space=pto.MemorySpace.RIGHT, valid_shape=[full_bc, dim])
qk_acc_tile = pto.alloc_tile(shape=[Br, Bc], dtype=pto.f32,
                             memory_space=pto.MemorySpace.ACC, valid_shape=[full_br, full_bc])
pv_acc_tile = pto.alloc_tile(shape=[Br, D], dtype=pto.f32,
                             memory_space=pto.MemorySpace.ACC, valid_shape=[full_br, dim])
```

Cube scratch tiles are NOT UB buffers. `LEFT`, `RIGHT`, and `ACC` are distinct hardware memory spaces inside the Cube unit. They serve as staging for matrix operands and accumulators.

### 11.3.5 SIMT metadata buffer

<!-- ptodsl-doc-test: {"mode":"compile_fragment","fixture":"flash_attention.l1_tiles","symbol":"flash_attention_l1_tiles_probe","compile":{"BLOCK_Q":128,"BLOCK_KV":128}} -->
```python
meta_tile = pto.alloc_tile(shape=[1, 8], dtype=pto.i32, valid_shape=[1, 3])
meta_ptr = meta_tile.as_ptr()
```

A small UB tile stores three scalar loop bounds (`row_start`, `row_stop`, `valid_cols`). `meta_tile.as_ptr()` materializes a typed UB pointer into it, which is passed to the explicit-mode orchestration as scalar control metadata.

Notice that the row-wise softmax state tiles (`m_*`, `l_*`, `alpha_tile`,
`beta_tile`) are authored as `blayout="ColMajor"`. This is the intended public
surface for logical column vectors; it avoids forcing users to manufacture a
row-major padded physical width just to satisfy row-byte alignment.

### 11.3.6 Outer Q loop + inner KV loop

<!-- ptodsl-doc-test: {"mode":"compile_fragment","fixture":"flash_attention.l1_loop_body","symbol":"flash_attention_l1_loop_body_probe","compile":{"BLOCK_Q":128,"BLOCK_KV":128,"HEAD_DIM":128,"CAUSAL":false,"NUM_STAGES":2}} -->
```python
for qi in range(0, q_blocks, 1):
    q_rows = _block_valid_extent(seq_q, qi, Br)
    q_part = pto.partition_view(q_head, offsets=[0, qi * Br, 0, 0],
                                sizes=[1, q_rows, 1, dim])
    o_part = pto.partition_view(o_head, offsets=[0, qi * Br, 0, 0],
                                sizes=[1, q_rows, 1, dim])

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

    m_cur = m_prev_tile
    l_cur = l_prev_tile
    o_cur = o_prev_tile
    for kj in range(0, kv_blocks, 1):
        kv_rows = _block_valid_extent(seq_k, kj, Bc)
        k_part = pto.partition_view(k_head,
                    offsets=[0, kj * Bc, 0, 0], sizes=[1, kv_rows, 1, dim])
        v_part = pto.partition_view(v_head,
                    offsets=[0, kj * Bc, 0, 0], sizes=[1, kv_rows, 1, dim])

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
            q_mat, k_part, v_part, k_mat, v_mat,
            o_cur, o_next_tile,
            m_cur, l_cur, m_next_tile, l_next_tile,
            s_tile, p_tile, p_mat, pv_tile,
            alpha_tile, beta_tile,
            q_l0a, p_l0a, rhs_l0b,
            qk_acc_tile, pv_acc_tile,
            meta_ptr,
        )

        m_cur = m_next_tile
        l_cur = l_next_tile
        o_cur = o_next_tile

    pto.tile.store(o_cur, o_part)
```

Key points:

- **Static physical shape, dynamic valid extent**: `alloc_tile(shape=...)` stays constexpr. Tail handling is expressed by updating `valid_shape` before each block load and sub-kernel call.
- **`tile.load` at the kernel entry boundary**: Q is loaded once per Q block using a tile op into the MAT-backed bridge tile `q_mat`. The compiler auto-inserts the necessary `set_flag`/`wait_flag` pairs.
- **State initialization**: `fill(float("-inf"))` and `fill(0.0)` initialize the online-softmax accumulators before the first KV block.
- **Carry state**: the inner Python loop carries three ping-pong tiles (`m_cur`, `l_cur`, `o_cur`) across iterations. After each KV block, assignments to the `_next` tiles become loop-carried state after AST rewrite. After the loop, `o_cur` is the final output accumulator.
- **`tile.store` at the kernel entry boundary**: writes the final result for this Q block back to GM.

## 11.4 Explicit orchestration

```python
# Explicit orchestration helper used by flash_attention_kernel:
def kv_block_process(
    q_mat, k_part, v_part, k_mat, v_mat,
    o_prev_tile, o_next_tile,
    m_prev_tile, l_prev_tile, m_next_tile, l_next_tile,
    s_tile, p_tile, p_mat, pv_tile,
    alpha_tile, beta_tile,
    q_l0a, p_l0a, rhs_l0b,
    qk_acc_tile, pv_acc_tile,
    meta_ptr,
):
```

The explicit-mode body processes one KV block against an already-loaded Q tile. It owns the execution sandwich:

### Phase 0 — Stage K/V data

<!-- ptodsl-doc-test: {"mode":"compile_fragment","fixture":"flash_attention.explicit_phase","symbol":"flash_attention_explicit_phase_probe","compile":{"BLOCK_Q":16,"BLOCK_KV":16}} -->
```python
rows = k_mat.valid_shape[0]
cols = k_mat.valid_shape[1]
row_bytes = cols * pto.bytewidth(pto.f32)
gm_row_stride = k_part.strides[0] * pto.bytewidth(pto.f32)
mat_row_stride = k_mat.shape[1] * pto.bytewidth(pto.f32)
pto.mte_load(k_part.as_ptr(), k_mat.as_ptr(), 0, row_bytes,
             nburst=(rows, gm_row_stride, mat_row_stride))
pto.mte_load(v_part.as_ptr(), v_mat.as_ptr(), 0, row_bytes,
             nburst=(rows, gm_row_stride, mat_row_stride))
pto.pipe_barrier(pto.Pipe.ALL)
```

`mte_load` is the ptr-based GM→MAT DMA wrapper used by this walkthrough. Explicit mode passes GM/MAT pointers plus the DMA grouping parameters, and `pipe_barrier(Pipe.ALL)` makes the phase boundary explicit before the cube unit reads `k_mat`/`v_mat`.

### Phase 0b — Materialize loop bounds

<!-- ptodsl-doc-test: {"mode":"compile_fragment","fixture":"flash_attention.explicit_phase","symbol":"flash_attention_explicit_phase_probe","compile":{"BLOCK_Q":16,"BLOCK_KV":16}} -->
```python
materialize_tile_bounds(meta_ptr,
    q_mat.valid_shape[0],
    k_mat.valid_shape[0])
row_start = pto.load(meta_ptr + 0)
row_stop  = pto.load(meta_ptr + 1)
valid_cols = pto.load(meta_ptr + 2)
```

The SIMT sub-kernel `materialize_tile_bounds` writes `{0, valid_rows, valid_cols}` into the metadata buffer. The explicit-mode body then loads these scalars. They control the row iteration range in subsequent sub-kernels, handling partial tail blocks.

### Phase 1 — `S = Q @ K^T`

<!-- ptodsl-doc-test: {"mode":"compile_fragment","fixture":"flash_attention.explicit_phase","symbol":"flash_attention_explicit_phase_probe","compile":{"BLOCK_Q":16,"BLOCK_KV":16}} -->
```python
qk_matmul(q_mat, k_mat, q_l0a, rhs_l0b, qk_acc_tile, s_tile)
pto.pipe_barrier(pto.Pipe.ALL)
```

Dispatches the Cube-kind TileOp. `pipe_barrier(Pipe.ALL)` separates the matrix multiply from the subsequent softmax.

### Phase 2 — Online softmax

<!-- ptodsl-doc-test: {"mode":"compile_fragment","fixture":"flash_attention.explicit_phase","symbol":"flash_attention_explicit_phase_probe","compile":{"BLOCK_Q":16,"BLOCK_KV":16}} -->
```python
online_softmax_rows(
    s_tile, p_tile,
    m_prev_tile, l_prev_tile,
    m_next_tile, l_next_tile,
    alpha_tile, beta_tile,
    row_start, row_stop, valid_cols,
)
pto.pipe_barrier(pto.Pipe.ALL)
```

The Vector-kind TileOp computes per-row softmax on `S`, updates the running `m`/`l` state, and writes `P`, `alpha`, and `beta`.

### Phase 3 — `PV = P @ V`

<!-- ptodsl-doc-test: {"mode":"compile_fragment","fixture":"flash_attention.explicit_phase","symbol":"flash_attention_explicit_phase_probe","compile":{"BLOCK_Q":16,"BLOCK_KV":16}} -->
```python
pto.tile.mov(p_tile, p_mat)
pto.pipe_barrier(pto.Pipe.ALL)

pv_matmul(p_mat, v_mat, p_l0a, rhs_l0b, pv_acc_tile, pv_tile)
pto.pipe_barrier(pto.Pipe.ALL)
```

The probability tile is first staged onto the MAT path with `pto.tile.mov(p_tile, p_mat)`. Then the second cube dispatch reuses `rhs_l0b` for `V` and `pv_acc_tile` for the accumulator.

### Phase 4 — Blend output

<!-- ptodsl-doc-test: {"mode":"compile_fragment","fixture":"flash_attention.explicit_phase","symbol":"flash_attention_explicit_phase_probe","compile":{"BLOCK_Q":16,"BLOCK_KV":16}} -->
```python
blend_output_rows(
    o_prev_tile, pv_tile, alpha_tile, beta_tile,
    o_next_tile, row_start, row_stop,
    v_mat.valid_shape[1],
)
pto.pipe_barrier(pto.Pipe.ALL)
```

The simt sub-kernel blends the old output accumulator with the new PV contribution, weighted by `alpha` and `beta`.

### Why explicit mode owns sync ordering

Each `pipe_barrier(Pipe.ALL)` between phases is explicit in the orchestration body. This is intentional: at the orchestration boundary, the user controls pipeline ordering. Auto mode may still use synchronization primitives where needed, but it does so around compiler-managed tile staging rather than user-authored instruction scheduling.

## 11.5 Cube-kind TileOps

### `qk_matmul` — `S = Q @ K^T`

<!-- ptodsl-doc-test: {"mode":"compile_fragment","fixture":"flash_attention.qk_cube_helper","symbol":"flash_attention_qk_cube_helper_probe","compile":{"BLOCK_Q":16,"BLOCK_KV":16}} -->
```python
@pto.tileop
def qk_matmul(
    q_l0a: pto.Tile,
    k_l0b: pto.Tile,
    s_acc: pto.Tile,
    m: pto.index,
    n: pto.index,
    k: pto.index,
):
    pto.mad(q_l0a.as_ptr(), k_l0b.as_ptr(), s_acc.as_ptr(), m, n, k)
```

The caller stages Q and K into LEFT/RIGHT, invokes this helper for `mad`, and
writes ACC back to the score Tile. Scratch allocation and all MTE operations
remain in the orchestration kernel.

### `pv_matmul` — `PV = P @ V`

<!-- ptodsl-doc-test: {"mode":"compile_fragment","fixture":"flash_attention.pv_cube_helper","symbol":"flash_attention_pv_cube_helper_probe","compile":{"BLOCK_Q":16,"BLOCK_KV":16}} -->
```python
@pto.tileop
def pv_matmul(
    p_l0a: pto.Tile,
    v_l0b: pto.Tile,
    pv_acc: pto.Tile,
    m: pto.index,
    n: pto.index,
    k: pto.index,
):
    pto.mad(p_l0a.as_ptr(), v_l0b.as_ptr(), pv_acc.as_ptr(), m, n, k)
```

Structurally identical to `qk_matmul`. The caller stages P and V, reuses the
scratch Tiles across KV blocks, and performs writeback after the helper call.

## 11.6 Vector-kind TileOp — online softmax

```python
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
```

The simd kernel iterates over rows with Python `for range(...)`, and AST rewrite lowers the loop to runtime IR:

<!-- ptodsl-doc-test: {"mode":"compile_fragment","fixture":"flash_attention.online_softmax_loop","symbol":"flash_attention_online_softmax_loop_probe","compile":{"BLOCK":16}} -->
```python
for row in range(row_start, row_stop, 1):
    col_mask = pto.make_mask(pto.f32, valid_cols)

    s_row  = pto.vlds(s_tile[row, 0:])
    m_prev = pto.vlds(m_prev_tile.as_ptr(), row, dist="BRC_B32")
    l_prev = pto.vlds(l_prev_tile.as_ptr(), row, dist="BRC_B32")
```

- **Mask creation**: `make_mask(pto.f32, valid_cols)` generates a tail mask for the column dimension. On the last KV block, `valid_cols` may be less than the full block width.
- **Vector load**: `vlds(s_tile[row, 0:])` loads one entire row of `S` from UB into a vector register. The slice syntax `[row, 0:]` selects the full row.
- **Broadcast load**: `vlds(..., dist="BRC_B32")` reads each per-row state value and broadcasts it across the vector lanes.

### Softmax computation

<!-- ptodsl-doc-test: {"mode":"compile_fragment","fixture":"flash_attention.online_softmax_compute","symbol":"flash_attention_online_softmax_compute_probe","compile":{"BLOCK":16}} -->
```python
    row_max = pto.vdup(pto.vcmax(s_row, col_mask), col_mask)
    m_next  = pto.vmax(m_prev, row_max, col_mask)

    s_shifted = pto.vsub(s_row, m_next, col_mask)
    p_row     = pto.vexp(s_shifted, col_mask)

    row_sum  = pto.vdup(pto.vcadd(p_row, col_mask), col_mask)
    exp_scale = pto.vexp(pto.vsub(m_prev, m_next, col_mask), col_mask)
    l_scaled = pto.vmul(l_prev, exp_scale, col_mask)
    l_next   = pto.vadd(l_scaled, row_sum, col_mask)

    alpha = pto.vdiv(l_scaled, l_next, col_mask)
    beta  = pto.vrec(l_next, col_mask)
```

This implements the online-softmax update from the Flash Attention paper:

- `vcmax` finds the row maximum and `vdup` broadcasts lane 0.
- `vmax` combines the broadcast row maximum with the running maximum.
- `vsub` subtracts the broadcast `m_next` from every lane.
- `vexp` computes `exp(s_shifted)` element-wise.
- `vcadd` computes the row sum and `vdup` broadcasts lane 0.
- `l_scaled` rescales the previous sum with the running-max correction factor.
- `alpha` and `beta` are the blending coefficients for the output update.

### Store results

<!-- ptodsl-doc-test: {"mode":"compile_fragment","fixture":"flash_attention.online_softmax_store","symbol":"flash_attention_online_softmax_store_probe","compile":{"BLOCK":16}} -->
```python
    pto.vsts(p_row, p_tile[row, 0:], col_mask)
    pto.vsts(m_next, m_next_tile.as_ptr(), row, col_mask, dist="1PT_B32")
    pto.vsts(l_next, l_next_tile.as_ptr(), row, col_mask, dist="1PT_B32")
    pto.vsts(alpha, alpha_tile.as_ptr(), row, col_mask, dist="1PT_B32")
    pto.vsts(beta, beta_tile.as_ptr(), row, col_mask, dist="1PT_B32")
```

- `vsts` stores the vector `p_row` back to UB under the column mask.
- `sts` stores each scalar to its respective UB tile.

**Boundary contract**: vreg values (`s_row`, `p_row`, `row_max`, `row_sum`) never escape the simd kernel. All persistent state is written to UB tiles.

## 11.7 SIMT sub-kernel — blend output

### `materialize_tile_bounds` — scalar metadata

<!-- ptodsl-doc-test: {"mode":"compile_fragment","fixture":"flash_attention.simt_materialize","symbol":"flash_attention_simt_materialize_probe","compile":{}} -->
```python
@pto.simt
def materialize_tile_bounds(
    meta_ptr: pto.ptr(pto.i32, pto.MemorySpace.UB),
    valid_rows: pto.i32,
    valid_cols: pto.i32,
):
    pto.store(0, meta_ptr + 0)
    pto.store(valid_rows, meta_ptr + 1)
    pto.store(valid_cols, meta_ptr + 2)
```

Three scalar stores write the loop bounds into the metadata buffer. `meta_ptr` is a typed UB pointer; `+ 0`, `+ 1`, `+ 2` are element offsets into `i32` storage, not byte offsets. This is the simplest sub-kernel in the sketch — it handles scalar control metadata, not vector math.

### `blend_output_rows` — output accumulation

<!-- ptodsl-doc-test: {"mode":"compile_fragment","fixture":"flash_attention.simt_blend","symbol":"flash_attention_simt_blend_probe","compile":{"BLOCK":8}} -->
```python
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
    for row in range(row_start, row_stop, 1):
        alpha = pto.load(alpha_tile[row, 0])
        beta  = pto.load(beta_tile[row, 0])

        for col in range(0, valid_dim, 1):
            o_prev = pto.load(o_prev_tile[row, col])
            pv_val = pto.load(pv_tile[row, col])
            o_next = alpha * o_prev + beta * pv_val
            pto.store(o_next, o_next_tile[row, col])
```

This is a scalar element-wise blend over the tile domain:

```
O_next[row, col] = alpha[row] * O_prev[row, col] + beta[row] * PV[row, col]
```

The SIMT kernel walks the tile element by element with nested Python
`for range(...)` loops. Under the default AST rewrite path these become
device-side runtime loops. Each iteration loads two scalars (`o_prev` and
`pv_val`), computes the weighted sum, and stores the result. The `alpha` /
`beta` coefficients are per-row (loaded once per row), while the blend is
per-element.

**Why SIMT instead of SIMD?** The intent is to contrast with `online_softmax_rows`: softmax is dominated by row-wise vector reductions and exponentials — natural SIMD work. The final blend is a simple linear combination with per-row coefficients — expressing it as explicit scalar work-items makes the per-element access pattern explicit and leaves the compiler free to vectorize or fuse as it sees fit.

### Context manager alternative

For trivial sub-kernels like `materialize_tile_bounds`, a named function is overkill — the context manager form keeps the logic inline where it's used. The inline SIMT scope itself looks like this:

<!-- ptodsl-doc-test: {"mode":"compile_fragment","fixture":"flash_attention.inline_simt_scope","symbol":"flash_attention_inline_simt_scope_probe","compile":{"BLOCK_Q":16,"BLOCK_KV":16}} -->
```python
with pto.simt():
    pto.store(0, meta_ptr + 0)
    pto.store(q_mat.valid_shape[0], meta_ptr + 1)
    pto.store(k_mat.valid_shape[0], meta_ptr + 2)
```

The `with pto.simt():` block acts as an anonymous inline sub-kernel scope. For 3-line helpers that have no reuse, the context manager avoids the indirection of a separate function. For complex, reusable logic like `online_softmax_rows` or `qk_matmul`, the named decorator form remains the better fit.

## 11.8 Putting it all together: one KV block execution

For one KV block, the full execution sequence is:

| Step | Layer | Operation | Hardware |
|------|-------|-----------|----------|
| 1 | explicit | `tile.load(q_part, q_mat)` | GM → MAT |
| 2 | explicit | `mte_load(k_part.as_ptr(), k_mat.as_ptr(), ...)` | GM → MAT |
| 3 | explicit | `mte_load(v_part.as_ptr(), v_mat.as_ptr(), ...)` | GM → MAT |
| 4 | explicit | `pipe_barrier(Pipe.ALL)` | — |
| 5 | simt | `materialize_tile_bounds` | SIMT |
| 6 | cube | `qk_matmul` (mte_l1_l0a, mte_l1_l0b, mad, mte_l0c_ub) | Cube |
| 7 | explicit | `pipe_barrier(Pipe.ALL)` | — |
| 8 | tileop | `online_softmax_rows` (vlds, vcmax, vexp, vcadd, vsts, ...) | Vector |
| 9 | explicit | `pipe_barrier(Pipe.ALL)` | — |
| 10 | explicit | `tile.mov(p_tile, p_mat)` | Tile copy |
| 11 | explicit | `pipe_barrier(Pipe.ALL)` | — |
| 12 | cube | `pv_matmul` | Cube |
| 13 | explicit | `pipe_barrier(Pipe.ALL)` | — |
| 14 | simt | `blend_output_rows` | SIMT |
| 15 | explicit | `pipe_barrier(Pipe.ALL)` | — |

After all KV blocks: the top-level kernel issues `tile.store(o_final_tile, o_part)` to write the result back to GM.

## 11.9 Design patterns in this sketch

**Ping-pong state for online accumulators**: `m_prev`/`m_next`, `l_prev`/`l_next`, `o_prev`/`o_next` make the state transition explicit. After each KV block, the caller swaps the ping-pong pair by assigning the `_next` tiles to the loop-carried Python variables rather than aliasing in place.

**Scratch reuse**: `rhs_l0b` serves both `K` (in `qk_matmul`) and `V` (in `pv_matmul`). `pv_acc_tile` reuses the accumulator from QK^T. The caller allocates once and passes the scratch Tiles to both Cube-kind TileOps.

**Tile-level boundary vs micro-instruction boundary**: `tile.load`/`tile.store` are the tile-atomic surface used in auto mode and at the top-level tile boundary of this sketch. `mte_load` appears in explicit orchestration, authored as individual pointer-based instructions. The abstraction split is auto mode as tile-centric authoring, explicit mode as user-ordered orchestration.

**No vreg across sub-kernel boundaries**: vector registers are local to each `@pto.tileop` kernel. Data crosses sub-kernel boundaries through UB tiles — the boundary contract is enforced by the type system.

**Invocation flexibility**: This sketch uses the explicit `@pto.jit(entry=True, mode="explicit")` path for full micro-instruction control and keeps its Vector compute directly in the kernel body. The same named sub-kernels can also be reused from `@pto.jit(mode="auto")` when the body stays within the auto-mode contract. The orchestration logic could be extracted into `@pto.jit(entry=False)` kernel modules for reuse across multiple entry kernels. See Chapter 3 for details.
