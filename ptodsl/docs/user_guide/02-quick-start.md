# 2. Quick Start

This chapter walks through a minimal but complete PTODSL kernel — a tiled copy from one GM tensor to another — covering the essential concepts you need to start writing your own kernels.

## 2.1 A first kernel: tiled copy

<!-- ptodsl-doc-test: {"mode":"compile","symbol":"tile_copy","compile":{"BLOCK":128}} -->
```python
from ptodsl import pto


@pto.jit(target="a5")
def tile_copy(
    A_ptr: pto.ptr(pto.f32, "gm"),
    O_ptr: pto.ptr(pto.f32, "gm"),
    rows: pto.i32,
    cols: pto.i32,
    *,
    BLOCK: pto.const_expr = 128,
):
    """Copy one 2D tensor tile from A to O."""

    # Describe the GM tensors explicitly.
    a_view = pto.make_tensor_view(A_ptr, shape=[rows, cols], strides=[cols, 1])
    o_view = pto.make_tensor_view(O_ptr, shape=[rows, cols], strides=[cols, 1])

    # Allocate a UB tile for one row-strip block.
    a_tile = pto.alloc_tile(shape=[1, BLOCK], dtype=pto.f32)

    # Partition the GM views to cover the current logical slice.
    a_part = pto.partition_view(a_view, offsets=[0, 0], sizes=[rows, cols])
    o_part = pto.partition_view(o_view, offsets=[0, 0], sizes=[rows, cols])

    # Load from GM into UB, then store back out.
    pto.tile.load(a_part, a_tile)
    pto.tile.store(a_tile, o_part)
```

Let us step through each piece.

### The entry point

```python
@pto.jit(target="a5")
def tile_copy(
    A_ptr: pto.ptr(pto.f32, "gm"),
    O_ptr: pto.ptr(pto.f32, "gm"),
    rows: pto.i32,
    cols: pto.i32,
    *,
    BLOCK: pto.const_expr = 128,
):
```

`@pto.jit` marks this function as a launchable PTO kernel. By default `entry=True`, which means it uses the host-visible pointer-first ABI: explicit GM pointers, runtime scalars, and keyword-only `pto.const_expr` compile-time constants. The default compilation backend is `backend="vpto"`. The positional parameters `A_ptr` and `O_ptr` are explicit GM pointers, while `rows` and `cols` are runtime scalar metadata passed at launch time. The keyword-only argument `BLOCK` is a compile-time constant declared with `pto.const_expr`; the compiler specializes the kernel for each tile width.

### Describing GM tensors

```python
a_view = pto.make_tensor_view(A_ptr, shape=[rows, cols], strides=[cols, 1])
```

`make_tensor_view` wraps an explicit GM pointer into a `TensorView` — a descriptor that tells the kernel how to address the tensor in global memory. You provide the logical shape and the stride (in elements) of each dimension.

### Allocating on-chip buffers

```python
a_tile = pto.alloc_tile(shape=[1, BLOCK], dtype=pto.f32)
```

`alloc_tile` reserves space in the Unified Buffer (UB). A `Tile` is a 2D buffer that lives on-chip during kernel execution. Every tile has a `shape` and a `dtype`.

### Partitioning GM views

```python
a_part = pto.partition_view(a_view, offsets=[0, 0], sizes=[rows, cols])
```

`partition_view` creates a sub-view of a `TensorView` at a given offset and size. It describes *which part* of the GM tensor a `tile.load` or `tile.store` should operate on. For this simple whole-tensor example the offset is zero and the size matches the logical tensor extent; in a blocked kernel you would slide the offset through a loop.

### Moving data: tile.load and tile.store

```python
pto.tile.load(a_part, a_tile)   # GM → UB
pto.tile.store(a_tile, o_part)  # UB → GM
```

`tile.load` copies a block of data from GM (described by a partition) into a UB tile. `tile.store` copies a UB tile back to GM. These are **Tile Ops** — they operate on entire tile buffers at once.

### Why start with copy

```python
pto.tile.load(a_part, a_tile)
pto.tile.store(a_tile, o_part)
```

A copy kernel strips the example down to the essential PTODSL boundary objects:

- explicit GM pointers and runtime scalars entering `@pto.jit`
- `TensorView` descriptors over GM tensors
- UB `Tile` allocation
- `PartitionTensorView` slices
- tile-level movement with `tile.load` / `tile.store`

Once these pieces are clear, arithmetic and sub-kernel orchestration become much easier to layer on.

## 2.2 A blocked version with a loop

The kernel above touches one logical slice directly. To introduce device-side control flow, we can iterate over the rows of a 2D tensor and copy one row-strip at a time:

<!-- ptodsl-doc-test: {"mode":"compile","symbol":"blocked_copy","compile":{"BLOCK":128}} -->
```python
from ptodsl import pto


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
```

Here `rows` and `cols` are dynamic launch-time scalars. The loop bound depends on `rows`, so the default AST rewrite records a structured loop in the IR rather than unrolling at trace time. The `BLOCK` parameter stays `const_expr` because it is a tuning knob, not data-dependent. Chapter 5 covers this distinction in detail.

## 2.3 Compile and launch

Once the kernel is defined, you compile it and then launch it:

<!-- ptodsl-doc-test: {"mode":"launch_fragment","fixture":"launch.blocked_copy_compile_and_launch","symbol":"blocked_copy"} -->
```python
# Compile once, cache the result.
compiled = blocked_copy.compile(BLOCK=128)

# Allocate or obtain input/output tensors (NumPy, torch-npu, ...).
import numpy as np
A = np.random.randn(4, 128).astype(np.float32)
O = np.empty_like(A)

# Launch on the NPU.
compiled[1, None](A.ctypes.data, O.ctypes.data, 4, 128)
```

- `.compile(**constexprs)` traces the kernel body, lowers it through the PTOAS pipeline, and returns a compiled handle. Repeated calls with the same tensor ABI contract and constexpr configuration hit the cache.
- `compiled[grid, stream](args...)` launches the compiled kernel. `grid` is the number of SPMD blocks; `stream` is the NPU stream (or `None` for the default).

## 2.4 SPMD launch

For workloads that can be parallelized across multiple blocks, specify a grid:

```python
# Process batch * heads slices in parallel.
compiled[batch * heads, stream](
    Q.ctypes.data,
    K.ctypes.data,
    V.ctypes.data,
    O.ctypes.data,
    batch,
    seq_q,
    seq_k,
    heads,
    dim,
)
```

Inside the kernel, each block queries its index:

```python
block_idx = pto.get_block_idx()
block_num = pto.get_block_num()
```

This lets you map different data slices to different blocks — for example, one block per (batch, head) pair in flash attention.

## 2.5 Adding sub-kernels and explicit orchestration

The examples above used Tile Ops (`tile.load` / `tile.store` here, and
arithmetic Tile Ops in later chapters), which operate on entire tiles at once.
When you need finer control — for instance, writing a custom softmax or an
activation that maps directly to vector hardware — you can keep the same
`@pto.jit` entry and add sub-kernels. If you also need micro-instruction control,
switch that kernel to `mode="explicit"`:

<!-- ptodsl-doc-test: {"mode":"compile","symbol":"vec_add_micro","compile":{"BLOCK":128}} -->
```python
# Tile-op helper — vector instructions on individual rows.
@pto.tileop
def add_rows(a_tile: pto.Tile, b_tile: pto.Tile, o_tile: pto.Tile,
             rows: pto.index, cols: pto.index):
    VEC = pto.elements_per_vreg(pto.f32)
    for r in range(0, rows, 1):
        remained = cols
        for c in range(0, cols, VEC):
            mask, remained = pto.make_mask(pto.f32, remained)
            a_vec = pto.vlds(a_tile[r, c:])
            b_vec = pto.vlds(b_tile[r, c:])
            o_vec = pto.vadd(a_vec, b_vec, mask)
            pto.vsts(o_vec, o_tile[r, c:], mask)

# Single kernel entry in explicit mode — micro-instruction staging plus a tile-op helper.
@pto.jit(target="a5", mode="explicit")
def vec_add_micro(
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

    a_tile = pto.alloc_tile(shape=[1, BLOCK], dtype=pto.f32)
    b_tile = pto.alloc_tile(shape=[1, BLOCK], dtype=pto.f32)
    o_tile = pto.alloc_tile(shape=[1, BLOCK], dtype=pto.f32)

    num_blocks = (N + BLOCK - 1) // BLOCK
    for i in range(0, num_blocks, 1):
        offset = i * BLOCK
        this_block = scalar.min(N - offset, BLOCK)
        a_part = pto.partition_view(a_view, offsets=[offset], sizes=[this_block])
        b_part = pto.partition_view(b_view, offsets=[offset], sizes=[this_block])
        o_part = pto.partition_view(o_view, offsets=[offset], sizes=[this_block])
        row_bytes = this_block * pto.bytewidth(pto.f32)
        pto.mte_load(a_part.as_ptr(), a_tile.as_ptr(), 0, row_bytes,
                     nburst=(1, 0, 0))
        pto.mte_load(b_part.as_ptr(), b_tile.as_ptr(), 0, row_bytes,
                     nburst=(1, 0, 0))
        pto.pipe_barrier(pto.Pipe.ALL)
        add_rows(a_tile, b_tile, o_tile, 1, this_block)
        pto.pipe_barrier(pto.Pipe.ALL)
        pto.mte_store(o_tile.as_ptr(), o_part.as_ptr(), row_bytes,
                      nburst=(1, 0, 0))
```

- **`@pto.jit(mode="explicit")`**: allocates tiles, partitions the GM views,
  loops over blocks, and directly authors the micro-instruction schedule for
  each block.

- **`@pto.tileop` sub-kernel**: the top-level kernel calls a vector custom OP
  for the row-wise vector work while keeping instruction staging in the
  explicit entry body.

- **Inside `@pto.tileop`**: the outer Python `for range(...)` iterates over rows,
  and the inner Python `for range(...)` iterates over column chunks of the hardware vector width
  (`elements_per_vreg`). Each iteration loads a vector-width slice into a
  `vreg`, does the addition under a mask (for tail elements), and stores the
  result back. Both loops are recorded as structured control flow IR — the
  compiler decides whether to keep them or unroll them.

The same pattern also has an `auto` counterpart: keep `@pto.jit` in its
default mode and replace the explicit `mte_*` sequence with `tile.load` /
`tile.store`. Chapter 3 covers the full entry model; Chapters 7–10 cover each
operation family in detail.

## 2.6 Next steps

You now have a complete mental model of a PTODSL kernel: entry points, tile
allocation, data movement with `tile.load` / `tile.store`, compile, and
launch. From here:

- **Chapter 3** covers the full kernel entry and module system: `auto` vs
  `explicit` modes, `vpto` vs `emitc` backends, sub-kernels for custom tile
  ops, and kernel modules (`@pto.jit(entry=False)`) for decomposing large
  kernels across independent compilation boundaries.
- **Chapter 4** details the type system: scalars, tiles, tensor views, and
  buffer management.
- **Chapter 5** explains control flow: device-side loops and conditionals,
  trace-time unrolling, and the AST rewrite model.
