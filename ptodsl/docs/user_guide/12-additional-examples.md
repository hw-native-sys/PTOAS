# 12. Additional Examples

This chapter presents four self-contained examples that build on the concepts introduced in Chapters 1–11. Each example demonstrates a specific pattern: blocked 2D processing, tail handling with masks, matrix multiplication on the Cube unit, and loop-carried state for online normalization.

## 12.1 Blocked 2D elementwise addition

Chapter 2 showed a 1D vector add with a single blocking dimension. Real workloads often involve 2D tensors — matrices — where blocking happens along both rows and columns.

```python
@pto.jit(target="a5")
def mat_add(
    A_ptr: pto.ptr(pto.f32, "gm"),
    B_ptr: pto.ptr(pto.f32, "gm"),
    O_ptr: pto.ptr(pto.f32, "gm"),
    batch: pto.i32,
    M: pto.i32,
    N_: pto.i32,
    *,
    BLOCK_M: pto.const_expr = 64,
    BLOCK_N: pto.const_expr = 128,
):
    a_view = pto.make_tensor_view(A_ptr, shape=[batch, M, N_], strides=[M * N_, N_, 1])
    b_view = pto.make_tensor_view(B_ptr, shape=[batch, M, N_], strides=[M * N_, N_, 1])
    o_view = pto.make_tensor_view(O_ptr, shape=[batch, M, N_], strides=[M * N_, N_, 1])

    a_tile = pto.alloc_tile(shape=[BLOCK_M, BLOCK_N], dtype=pto.f32)
    b_tile = pto.alloc_tile(shape=[BLOCK_M, BLOCK_N], dtype=pto.f32)
    o_tile = pto.alloc_tile(shape=[BLOCK_M, BLOCK_N], dtype=pto.f32)

    block_idx = pto.get_block_idx()
    num_m = (M + BLOCK_M - 1) // BLOCK_M
    num_n = (N_ + BLOCK_N - 1) // BLOCK_N

    for mi in range(0, num_m, 1):
        m_off = mi * BLOCK_M
        for ni in range(0, num_n, 1):
            n_off = ni * BLOCK_N

            a_part = pto.partition_view(a_view, offsets=[block_idx, m_off, n_off], sizes=[1, BLOCK_M, BLOCK_N])
            b_part = pto.partition_view(b_view, offsets=[block_idx, m_off, n_off], sizes=[1, BLOCK_M, BLOCK_N])
            o_part = pto.partition_view(o_view, offsets=[block_idx, m_off, n_off], sizes=[1, BLOCK_M, BLOCK_N])

            pto.tile.load(a_part, a_tile)
            pto.tile.load(b_part, b_tile)
            pto.tile.add(a_tile, b_tile, o_tile)
            pto.tile.store(o_tile, o_part)
```

**Key points**:

- Nested Python `for range(...)` loops produce a 2D block traversal. Under the
  default AST rewrite path they are recorded as device-side control flow, so
  they adapt to the runtime shapes `M` and `N_`.
- Tile shape `[BLOCK_M, BLOCK_N]` is 2D; all three tiles use the same shape so `tile.add` is elementwise.
- `partition_view` takes 2D offsets and sizes.
- `BLOCK_M` and `BLOCK_N` are `const_expr` — the compiler specializes the kernel per tile shape.

The Python wrapper follows the same pattern as Chapter 2:

<!-- ptodsl-doc-test: {"mode":"launch_fragment","fixture":"launch.mat_add_wrapper","symbol":"mat_add_wrapper"} -->
```python
def mat_add_wrapper(A, B, O=None, stream=None):
    if O is None:
        O = pto.empty_like(A)
    compiled = mat_add.compile(BLOCK_M=64, BLOCK_N=128)
    batch, m, n = A.shape
    compiled[batch, stream](A.ctypes.data, B.ctypes.data, O.ctypes.data, batch, m, n)
    return O
```

The grid is `A.shape[0]` so each SPMD block processes one slice of the leading batch dimension.

## 12.2 Vector operations with tail handling

When a data dimension is not evenly divisible by the tile size or the hardware vector width, the last iteration must operate on fewer elements. PTODSL provides masks for this — `make_mask` produces a predicate that guards loads, computes, and stores so out-of-bounds lanes are not touched.

### 12.2.1 Tail handling in a SIMD kernel

Below is a self-contained `@pto.tileop` kernel that adds two tiles row by row, handling column tails with `make_mask`:

<!-- ptodsl-doc-test: {"mode":"compile_fragment","fixture":"tail.tileop_helper","symbol":"tail_simd_helper_probe","compile":{"BLOCK":128}} -->
```python
@pto.tileop
def add_rows_with_tail(a_tile: pto.Tile, b_tile: pto.Tile, o_tile: pto.Tile,
                       rows: pto.i32, cols: pto.i32):
    VEC = pto.elements_per_vreg(pto.f32)          # 64 for f32

    for r in range(0, rows, 1):
        remained = cols
        for c in range(0, cols, VEC):
            mask, remained = pto.make_mask(pto.f32, remained)

            a_vec = pto.vlds(a_tile[r, c:])       # load under mask
            b_vec = pto.vlds(b_tile[r, c:])
            o_vec = pto.vadd(a_vec, b_vec, mask)  # compute under mask
            pto.vsts(o_vec, o_tile[r, c:], mask)  # store under mask
```

The pattern:

1. **Chunk**: Each iteration processes `VEC` elements (one vector register's worth).
2. **Mask**: `make_mask` returns a predicate and the updated remainder. On the last iteration, where `remained < VEC`, the mask has `remained` valid lanes followed by inactive lanes.
3. **Guard**: `vlds`, `vadd`, and `vsts` all accept the mask — inactive lanes are neither loaded, computed, nor stored.
4. **Carry**: assigning `remained` in the Python loop body makes it loop-carried state after AST rewrite.

### 12.2.2 Tile-level tail handling

At the Tile Op level, tail handling is built into `tile.load` and `tile.store`. When a partition size along a dimension is smaller than the tile size, the tile's `valid_shape` tracks the actual data extent:

<!-- ptodsl-doc-test: {"mode":"compile","symbol":"vec_add_with_tail","compile":{"BLOCK":128}} -->
```python
@pto.jit(target="a5")
def vec_add_with_tail(
    A_ptr: pto.ptr(pto.f32, "gm"),
    B_ptr: pto.ptr(pto.f32, "gm"),
    O_ptr: pto.ptr(pto.f32, "gm"),
    N: pto.i32,
    *,
    BLOCK: pto.const_expr = 128,
):
    a_view = pto.make_tensor_view(A_ptr, shape=[N], strides=[1])
    b_view = pto.make_tensor_view(B_ptr, shape=[N], strides=[1])
    o_view = pto.make_tensor_view(O_ptr, shape=[N], strides=[1])

    a_tile = pto.alloc_tile(shape=[BLOCK], dtype=pto.f32, valid_shape=[pto.const(BLOCK)])
    b_tile = pto.alloc_tile(shape=[BLOCK], dtype=pto.f32, valid_shape=[pto.const(BLOCK)])
    o_tile = pto.alloc_tile(shape=[BLOCK], dtype=pto.f32, valid_shape=[pto.const(BLOCK)])

    num_blocks = (N + BLOCK - 1) // BLOCK

    for i in range(0, num_blocks, 1):
        offset = i * BLOCK
        this_block = scalar.min(BLOCK, N - offset)

        a_part = pto.partition_view(a_view, offsets=[offset], sizes=[this_block])
        b_part = pto.partition_view(b_view, offsets=[offset], sizes=[this_block])
        o_part = pto.partition_view(o_view, offsets=[offset], sizes=[this_block])

        pto.tile.load(a_part, a_tile)
        pto.tile.load(b_part, b_tile)

        a_tile.valid_shape = [this_block]
        b_tile.valid_shape = [this_block]
        o_tile.valid_shape = [this_block]

        pto.tile.add(a_tile, b_tile, o_tile)
        pto.tile.store(o_tile, o_part)
```

- `this_block = scalar.min(BLOCK, N - offset)` computes the actual block size for the tail iteration on the device side.
- `sizes=[this_block]` on the partition and `tile.valid_shape = [...]` on the tile tell `tile.load`/`tile.add`/`tile.store` how many elements are live.

### 12.2.3 The general rule

| Tail scenario | Mechanism |
|---------------|-----------|
| Tile Op boundary (tile.load/tile.store) | `valid_shape` on tile + smaller `sizes` on partition |
| SIMD vector boundary (vlds/vadd/vsts) | `make_mask` + mask parameter on op |
| SIMT scalar loop boundary | `min(BLOCK, N - offset)` in loop bound |

## 12.3 GEMM: matrix multiplication on the Cube unit

This example demonstrates a complete GEMM kernel: `C = A @ B` where A is
`[M, K]` and B is `[K, N]`. It uses `@pto.jit` for tile allocation and loop
scheduling, and a cube-style `@pto.tileop` helper for the actual matrix
multiply.

### 12.3.1 Cube sub-kernel

<!-- ptodsl-doc-test: {"mode":"compile_fragment","fixture":"gemm.cube_helper","symbol":"gemm_tile_probe","compile":{"BLOCK_M":64,"BLOCK_K":64,"BLOCK_N":64}} -->
```python
@pto.tileop
def gemm_tile(a_mat: pto.Tile, b_mat: pto.Tile, o_tile: pto.Tile,
              a_l0a: pto.Tile, b_l0b: pto.Tile, o_acc: pto.Tile):
    m = a_mat.valid_shape[0]
    k = a_mat.valid_shape[1]
    n = b_mat.valid_shape[1]

    pto.mte_l1_l0a(a_mat.as_ptr(), a_l0a.as_ptr(), m, k)
    pto.mte_l1_l0b(b_mat.as_ptr(), b_l0b.as_ptr(), k, n)
    pto.mad(a_l0a.as_ptr(), b_l0b.as_ptr(), o_acc.as_ptr(), m, n, k)
    pto.mte_l0c_ub(o_acc.as_ptr(), o_tile.as_ptr(), m, n, n, n, 0)
```

The cube sub-kernel consumes MAT staging tiles plus cube-local scratch buffers. The four-step sequence — stage left operand, stage right operand, multiply, writeback — is the canonical cube compute pattern.

### 12.3.2 Tile orchestration

<!-- ptodsl-doc-test: {"mode":"compile_fragment","fixture":"gemm.jit_kernel","symbol":"gemm","compile":{"BLOCK_M":64,"BLOCK_K":64,"BLOCK_N":64}} -->
```python
@pto.jit(target="a5", mode="explicit")
def gemm(
    A_ptr: pto.ptr(pto.f32, "gm"),
    B_ptr: pto.ptr(pto.f32, "gm"),
    O_ptr: pto.ptr(pto.f32, "gm"),
    M: pto.i32,
    K_: pto.i32,
    N_: pto.i32,
    *,
    BLOCK_M: pto.const_expr = 64,
    BLOCK_K: pto.const_expr = 64,
    BLOCK_N: pto.const_expr = 64,
):
    a_view = pto.make_tensor_view(A_ptr, shape=[M, K_], strides=[K_, 1])
    b_view = pto.make_tensor_view(B_ptr, shape=[K_, N_], strides=[N_, 1])
    o_view = pto.make_tensor_view(O_ptr, shape=[M, N_], strides=[N_, 1])

    a_mat = pto.alloc_tile(shape=[BLOCK_M, BLOCK_K], dtype=pto.f32,
                           memory_space=pto.MemorySpace.MAT)
    b_mat = pto.alloc_tile(shape=[BLOCK_K, BLOCK_N], dtype=pto.f32,
                           memory_space=pto.MemorySpace.MAT)
    o_tile = pto.alloc_tile(shape=[BLOCK_M, BLOCK_N], dtype=pto.f32)

    a_l0a = pto.alloc_tile(shape=[BLOCK_M, BLOCK_K], dtype=pto.f32,
                           memory_space=pto.MemorySpace.LEFT)
    b_l0b = pto.alloc_tile(shape=[BLOCK_K, BLOCK_N], dtype=pto.f32,
                           memory_space=pto.MemorySpace.RIGHT)
    o_acc = pto.alloc_tile(shape=[BLOCK_M, BLOCK_N], dtype=pto.f32,
                           memory_space=pto.MemorySpace.ACC)

    num_m = (M + BLOCK_M - 1) // BLOCK_M
    num_n = (N_ + BLOCK_N - 1) // BLOCK_N
    num_k = (K_ + BLOCK_K - 1) // BLOCK_K

    for mi in range(0, num_m, 1):
        m_off = mi * BLOCK_M
        for ni in range(0, num_n, 1):
            n_off = ni * BLOCK_N
            o_part = pto.partition_view(o_view, offsets=[m_off, n_off],
                                        sizes=[BLOCK_M, BLOCK_N])

            o_tile.fill(0.0)

            for ki in range(0, num_k, 1):
                k_off = ki * BLOCK_K

                a_part = pto.partition_view(a_view, offsets=[m_off, k_off],
                                            sizes=[BLOCK_M, BLOCK_K])
                b_part = pto.partition_view(b_view, offsets=[k_off, n_off],
                                            sizes=[BLOCK_K, BLOCK_N])

                pto.tile.load(a_part, a_mat)
                pto.tile.load(b_part, b_mat)

                gemm_tile(a_mat, b_mat, o_tile, a_l0a, b_l0b, o_acc)

            pto.tile.store(o_tile, o_part)
```

**Key points**:

- **Triply nested loops**: M, N, and K dimensions are all blocked. The K loop accumulates partial results into `o_tile`.
- **Accumulation**: `o_tile.fill(0.0)` resets the accumulator before the K loop. Each K-block calls `gemm_tile` which writes its partial product back to `o_tile`. The Cube unit accumulates implicitly via `mad` — each K-block's partial result is added to the running total in `o_acc`.
- **MAT staging + cube-local scratch**: `a_mat` and `b_mat` are explicit MAT tiles that satisfy the `mte_l1_l0a` / `mte_l1_l0b` source contract. `a_l0a`, `b_l0b`, and `o_acc` are cube-local scratch (`LEFT`, `RIGHT`, `ACC`).
- **Direct sub-kernel call**: `gemm_tile` is called directly from `@pto.jit` — no separate orchestration layer needed. The compiler handles sync between `tile.load` and the Cube sub-kernel.
- **Cube sub-kernel reuse**: the same `gemm_tile` function is called for every K-block — the named decorator form enables reuse.

### 12.3.3 Python wrapper

<!-- ptodsl-doc-test: {"mode":"launch_fragment","fixture":"launch.gemm_wrapper","symbol":"gemm_wrapper"} -->
```python
import numpy as np


def gemm_wrapper(A, B, O=None, stream=None):
    if O is None:
        O = np.empty((A.shape[0], B.shape[1]), dtype=A.dtype)
    compiled = gemm.compile(BLOCK_M=64, BLOCK_K=64, BLOCK_N=64)
    compiled[1, stream](A.ctypes.data, B.ctypes.data, O.ctypes.data, A.shape[0], A.shape[1], B.shape[1])
    return O
```

This pattern extends directly to batch-GEMM: pass a grid of `batch` and use `pto.get_block_idx()` to select the per-batch slice from `A` and `B`.

### 12.3.4 Comparison with explicit-mode orchestration

This example keeps `mode="explicit"` because the named Cube helper directly
authors explicit-only surfaces such as `mte_l1_l0a`, `mte_l1_l0b`, and
`mte_l0c_ub`. Even though the top-level `@pto.jit` body itself stays fairly
tile-centric, the enclosing kernel still has to opt into explicit mode so that
the called sub-kernel is legal.

For most users, the direct-call structure shown above is still the recommended
pattern: keep the orchestration simple, let the named Cube helper own the
micro-instruction sequence, and only add more top-level explicit scheduling
when you truly need hand-authored DMA ordering or phase control.

## 12.4 Online normalization with loop-carried state

Chapter 11 demonstrated online softmax with ping-pong state tiles. A simpler but instructive case is **online layer normalization** — computing mean and variance incrementally across blocks while carrying only scalar state between iterations.

Given a vector `X` of length `N`, the streaming Welford algorithm updates the running mean `mu` and variance `var` as each new element `x` arrives:

```
n_next    = n_prev + 1
delta     = x - mu_prev
mu_next   = mu_prev + delta / n_next
m2_next   = m2_prev + delta * (x - mu_next)
```

The example below keeps the whole pattern inside one `@pto.jit` kernel. The first pass carries `mu`, `n`, and `m2` across blocks; the second pass reloads each block and applies the normalization explicitly with scalar loads and stores. This version assumes `N > 0`.

### 12.4.1 JIT example with loop-carried Welford state

<!-- ptodsl-doc-test: {"mode":"compile","symbol":"online_layernorm","compile":{"BLOCK":128}} -->
```python
@pto.jit(target="a5")
def online_layernorm(
    X_ptr: pto.ptr(pto.f32, "gm"),
    O_ptr: pto.ptr(pto.f32, "gm"),
    N: pto.i32,
    *,
    BLOCK: pto.const_expr = 128,
):
    x_view = pto.make_tensor_view(X_ptr, shape=[N], strides=[1])
    o_view = pto.make_tensor_view(O_ptr, shape=[N], strides=[1])

    x_tile = pto.alloc_tile(shape=[BLOCK], dtype=pto.f32, valid_shape=[pto.const(BLOCK)])
    o_tile = pto.alloc_tile(shape=[BLOCK], dtype=pto.f32, valid_shape=[pto.const(BLOCK)])

    num_blocks = (N + BLOCK - 1) // BLOCK

    # Pass 1: running Welford state across blocks.
    mu = pto.f32(0.0)
    n = pto.f32(0.0)
    m2 = pto.f32(0.0)
    for i in range(0, num_blocks, 1):
        offset = i * BLOCK
        this_block = scalar.min(BLOCK, N - offset)
        x_part = pto.partition_view(x_view, offsets=[offset], sizes=[this_block])
        pto.tile.load(x_part, x_tile)
        x_tile.valid_shape = [this_block]

        for j in range(0, this_block, 1):
            x = scalar.load(x_tile.as_ptr(), j)
            n_next = n + 1.0
            delta = x - mu
            mu_next = mu + delta / n_next
            delta2 = x - mu_next
            m2_next = m2 + delta * delta2

            mu = mu_next
            n = n_next
            m2 = m2_next

    mean = mu
    count = n
    inv_std = 1.0 / scalar.sqrt(m2 / count + pto.f32(1.0e-5))

    # Pass 2: apply (x - mean) / sqrt(var + eps) block by block.
    for i in range(0, num_blocks, 1):
        offset = i * BLOCK
        this_block = scalar.min(BLOCK, N - offset)
        x_part = pto.partition_view(x_view, offsets=[offset], sizes=[this_block])
        o_part = pto.partition_view(o_view, offsets=[offset], sizes=[this_block])

        pto.tile.load(x_part, x_tile)
        x_tile.valid_shape = [this_block]
        o_tile.valid_shape = [this_block]

        for j in range(0, this_block, 1):
            x = scalar.load(x_tile.as_ptr(), j)
            y = (x - mean) * inv_std
            scalar.store(y, o_tile.as_ptr(), j)

        pto.tile.store(o_tile, o_part)
```

**Key points**:

- **Carry state**: assigning `mu`, `n`, and `m2` inside the Python loops makes them loop-carried state after AST rewrite. The outer loop carries state across blocks; the inner loop carries state across elements inside one block.
- **Tail handling**: `scalar.min(BLOCK, N - offset)` computes the live width of the current block, and `tile.valid_shape = [this_block]` keeps the tile contract aligned with that tail.
- **No special tile op required**: the normalization pass is written explicitly with `scalar.load(...)`, scalar arithmetic, `scalar.sqrt(...)`, and `scalar.store(...)`. There is no dependency on a dedicated `tnormalize` op.
- **Compare to flash attention**: the flash attention carry in Chapter 11 moves several tiles through ping-pong buffers. Here the carried state is only three scalars, so the rewritten Python surface reads like a conventional streaming reduction.

## 12.5 Design guidelines

**Start simple, refine later.** Begin with `@pto.jit` + Tile Ops. If Tile Ops don't cover the computation (e.g., custom softmax, specialized activation), add a sub-kernel. If you need micro-instruction-level control, switch the kernel to `mode="explicit"`.

**Choose the right entry for each piece:**

| Goal | Use |
|------|-----|
| Whole-kernel orchestration, GM↔UB boundary | `@pto.jit` |
| Tile-level data movement | `tile.load` / `tile.store` |
| Custom row-wise vector math | `@pto.tileop` |
| Custom per-element logic | `@pto.simt` |
| Matrix multiply | cube-style `@pto.tileop` |
| Micro-instruction-level control | `mode="explicit"` |
| Inline compute for quick prototyping | `with pto.tileop():` etc. |

**Respect boundary contracts.** Vregs don't cross `@pto.tileop` boundaries. Cube-local state doesn't leak into UB. Tile Ops and MTE Ops belong to different programming models — use Tile Ops in `mode="auto"`, and micro-instructions in `mode="explicit"`.
