# 10. Synchronization Operations

Chapters 7 and 8 covered data movement and computation. This chapter covers the synchronization primitives that keep those operations correctly ordered across the NPU's concurrent hardware pipelines.

The Ascend NPU executes work across multiple independent pipelines — MTE (DMA), Vector, and Cube — each with its own instruction stream. Synchronization operations coordinate these pipelines: a DMA must finish loading data before the vector unit starts computing on it; a matrix multiply must complete before the result is stored. These operations are available in both `mode="auto"` and `mode="explicit"` when the kernel needs them. Without correct synchronization, pipelines race, and results are undefined.

## 10.1 Enum types for synchronization

PTODSL provides three enum types for type-safe specification of synchronization parameters.

### `BarrierType`

Memory barrier types used with `pto.mem_bar`. Each value specifies which category of prior instruction must complete before which category of subsequent instruction may proceed.

| Member | Meaning |
|--------|---------|
| `VV_ALL` | All vector ops before → all vector ops after |
| `VST_VLD` | Vector stores before → vector loads after |
| `VLD_VST` | Vector loads before → vector stores after |
| `VST_VST` | Vector stores before → vector stores after |
| `VS_ALL` | All vector ops before → all scalar ops after |
| `VST_LD` | Vector stores before → scalar loads after |
| `VLD_ST` | Vector loads before → scalar stores after |
| `VST_ST` | Vector stores before → scalar stores after |
| `SV_ALL` | All scalar ops before → all vector ops after |
| `ST_VLD` | Scalar stores before → vector loads after |
| `LD_VST` | Scalar loads before → vector stores after |
| `ST_VST` | Scalar stores before → vector stores after |

The naming convention: `V` = vector, `S` = scalar, `ST` = store, `LD` = load. `VST_VLD` reads "Vector STore before Vector LoaD."

### `Pipe`

Hardware pipeline identifiers used with `pto.set_flag`, `pto.wait_flag`, and `pto.pipe_barrier`.

| Member | Pipeline |
|--------|----------|
| `S` | Scalar / control pipeline |
| `V` | Vector pipeline (SIMD) |
| `M` | Matrix / Cube pipeline |
| `MTE1` | Memory Transfer Engine 1 |
| `MTE2` | Memory Transfer Engine 2 |
| `MTE3` | Memory Transfer Engine 3 |
| `MTE4` | Memory Transfer Engine 4 |
| `ALL` | All pipelines (for barrier operations) |

The most commonly used pipes in synchronization are `MTE2` (GM ↔ UB DMA), `MTE3` (UB ↔ UB DMA), `V` (vector compute), and `M` (matrix compute).

### `event_id`

Event identifiers for pipeline synchronization flags. The hardware provides 8 event IDs (`0`–`7`) per pipeline pair, supporting up to 8 concurrent in-flight DMA/compute sequences.

In PTODSL, `event_id` may be either:

- a Python integer literal in `0`–`7`
- a runtime index-like PTO scalar value

Events are per-pipeline-pair: the same `event_id=0` used between `MTE2 → V` is independent from `event_id=0` used between `MTE3 → V`.

Runtime loop indices can be used to select alternating event slots. This is useful for double-buffered pipelines:

<!-- ptodsl-doc-test: {"mode":"compile_fragment","fixture":"sync_ops.basic","symbol":"sync_ops_basic_probe","compile":{}} -->
```python
with pto.for_(0, 4, step=1) as stage:
    event = stage & 1
    pto.set_flag(pto.Pipe.MTE2, pto.Pipe.V, event_id=event)
    pto.wait_flag(pto.Pipe.MTE2, pto.Pipe.V, event_id=event)
```

---

## 10.2 Pipeline synchronization: `set_flag`, `wait_flag`, `pipe_barrier`

Pipeline synchronization is the primary mechanism for ordering work across pipelines. The pattern is always **signal then wait**: the producer pipeline sets a flag when its work is done; the consumer pipeline waits on that flag before proceeding.

### `pto.set_flag(pipe_from, pipe_to, *, event_id=0)`

**Description**: Sets a synchronization flag between two hardware pipelines. The producing pipeline signals that work up to this point is complete.

**Parameters**:

| Parameter | Type | Description |
|-----------|------|-------------|
| `pipe_from` | `Pipe` | Source pipeline — the pipeline that has completed its work |
| `pipe_to` | `Pipe` | Destination pipeline — the pipeline being notified |
| `event_id` | `int` or index-like PTO scalar | Event identifier for this specific synchronization point (`0`–`7`) |

**Returns**: None (side-effect operation).

**Example**:

<!-- ptodsl-doc-test: {"mode":"compile_fragment","fixture":"sync_ops.basic","symbol":"sync_ops_basic_probe","compile":{}} -->
```python
# MTE2 has finished loading tile data — signal Vector pipeline
pto.set_flag(pto.Pipe.MTE2, pto.Pipe.V, event_id=0)
```

### `pto.wait_flag(pipe_from, pipe_to, *, event_id=0)`

**Description**: Waits for a synchronization flag. The consuming pipeline blocks until the flag is set by the producing pipeline.

**Parameters**:

| Parameter | Type | Description |
|-----------|------|-------------|
| `pipe_from` | `Pipe` | Source pipeline that set the flag |
| `pipe_to` | `Pipe` | Destination pipeline — the pipeline that is waiting |
| `event_id` | `int` or index-like PTO scalar | Event identifier matching the corresponding `set_flag` (`0`–`7`) |

**Returns**: None (side-effect operation).

**Example**:

<!-- ptodsl-doc-test: {"mode":"compile_fragment","fixture":"sync_ops.basic","symbol":"sync_ops_basic_probe","compile":{}} -->
```python
# Vector pipeline waits for MTE2 to finish loading
pto.wait_flag(pto.Pipe.MTE2, pto.Pipe.V, event_id=0)
```

### `pto.pipe_barrier(pipes)`

**Description**: Executes a barrier across the specified pipelines. All work before the barrier in the named pipelines must complete before any work after the barrier may begin.

**Parameters**:

| Parameter | Type | Description |
|-----------|------|-------------|
| `pipes` | `Pipe` | Pipeline specification — typically `Pipe.ALL` for a full barrier |

**Returns**: None (side-effect operation).

**Example**:

<!-- ptodsl-doc-test: {"mode":"compile_fragment","fixture":"sync_ops.basic","symbol":"sync_ops_basic_probe","compile":{}} -->
```python
# Full hardware barrier — all pipelines synchronize
pto.pipe_barrier(pto.Pipe.ALL)
```

### Typical explicit-mode usage pattern

A common explicit-mode pattern interleaves DMA and compute with `set_flag` /
`wait_flag` pairs:

<!-- ptodsl-doc-test: {"mode":"compile_fragment","fixture":"sync_ops.flag_pattern_explicit","symbol":"sync_ops_flag_pattern_explicit_probe","compile":{"ROWS":8,"COLS":16}} -->
```python
# Inside a @pto.jit(mode="explicit") body:
def gemm_block(
    q_tile: pto.Tile,
    k_part: pto.PartitionTensorView,
    v_part: pto.PartitionTensorView,
    k_tile: pto.Tile,
    v_tile: pto.Tile,
    p_tile: pto.Tile,
    o_tile: pto.Tile,
    o_part: pto.PartitionTensorView,
    rows: pto.i32,
    cols: pto.i32,
):
    # DMA: load K and V tiles from GM to UB
    row_bytes = cols * pto.bytewidth(pto.f16)
    gm_row_stride = k_part.strides[0] * pto.bytewidth(pto.f16)
    ub_row_stride = k_tile.shape[1] * pto.bytewidth(pto.f16)
    out_row_bytes = cols * pto.bytewidth(pto.f32)
    out_gm_row_stride = o_part.strides[0] * pto.bytewidth(pto.f32)
    out_ub_row_stride = o_tile.shape[1] * pto.bytewidth(pto.f32)
    pto.mte_load(k_part.as_ptr(), k_tile.as_ptr(), 0, row_bytes,
                 nburst=(rows, gm_row_stride, ub_row_stride))
    pto.mte_load(v_part.as_ptr(), v_tile.as_ptr(), 0, row_bytes,
                 nburst=(rows, gm_row_stride, ub_row_stride))

    # Signal: DMA done, UB data ready
    pto.set_flag(pto.Pipe.MTE2, pto.Pipe.V, event_id=0)

    # Wait: vector pipeline stalls until data arrives
    pto.wait_flag(pto.Pipe.MTE2, pto.Pipe.V, event_id=0)

    # Compute: now safe to use k_tile and v_tile
    qk_matmul(q_tile, k_tile, p_tile)
    pv_matmul(p_tile, v_tile, o_tile)

    # Signal: compute done, results ready for store
    pto.set_flag(pto.Pipe.V, pto.Pipe.MTE3, event_id=1)
    pto.wait_flag(pto.Pipe.V, pto.Pipe.MTE3, event_id=1)

    # DMA: store results back to GM
    pto.mte_store(o_tile.as_ptr(), o_part.as_ptr(), out_row_bytes,
                  nburst=(rows, out_ub_row_stride, out_gm_row_stride))
```

---

## 10.3 Buffer management: `get_buf`, `rls_buf`

Double-buffering is a common optimization in NPU kernels: while one buffer is being computed on, the other is being loaded with the next block of data. The `get_buf` / `rls_buf` pair coordinates buffer ownership between pipelines.

### `pto.get_buf(pipe, buf_id, mode=0)`

**Description**: Acquire a buffer slot for inter-pipeline double-buffering coordination. The calling pipeline claims ownership of the buffer, blocking if the buffer is still in use by another pipeline.

**Parameters**:

| Parameter | Type | Description |
|-----------|------|-------------|
| `pipe` | `Pipe` | Pipeline identifier of the acquiring pipeline |
| `buf_id` | `pto.i64` | Buffer identifier (0-based index into the buffer pool) |
| `mode` | `pto.i64` | Acquisition mode (default 0) |

**Returns**: None (side-effect operation).

### `pto.rls_buf(pipe, buf_id, mode=0)`

**Description**: Release a buffer slot, allowing another pipeline to acquire it.

**Parameters**:

| Parameter | Type | Description |
|-----------|------|-------------|
| `pipe` | `Pipe` | Pipeline identifier of the releasing pipeline |
| `buf_id` | `pto.i64` | Buffer identifier matching the corresponding `get_buf` |
| `mode` | `pto.i64` | Release mode (default 0) |

**Returns**: None (side-effect operation).

### Double-buffering example

<!-- ptodsl-doc-test: {"mode":"compile_fragment","fixture":"sync_ops.basic","symbol":"sync_ops_basic_probe","compile":{}} -->
```python
# Pipeline V acquires buffer 0 for compute
pto.get_buf(pto.Pipe.V, 0, 0)

# ... compute into buffer 0 ...

# Release buffer 0 — DMA can now refill it
pto.rls_buf(pto.Pipe.V, 0, 0)

# Pipeline MTE2 acquires buffer 0 for reload
pto.get_buf(pto.Pipe.MTE2, 0, 0)

# ... DMA loads next block into buffer 0 ...

pto.rls_buf(pto.Pipe.MTE2, 0, 0)
```

---

## 10.4 Memory barriers: `mem_bar`

Within a single pipeline, load and store instructions may be reordered by the hardware. `mem_bar` enforces ordering when UB addresses alias between operations — for example, when a store to a region must be visible to a subsequent load from the same region.

### `pto.mem_bar(barrier_type)`

**Description**: Inserts a memory barrier that enforces ordering of prior and subsequent instructions within the same pipeline.

**Parameters**:

| Parameter | Type | Description |
|-----------|------|-------------|
| `barrier_type` | `BarrierType` | Barrier type controlling which categories of prior instructions must complete before which categories of subsequent instructions may proceed |

**Returns**: None (side-effect operation).

**Example**:

<!-- ptodsl-doc-test: {"mode":"compile_fragment","fixture":"sync_ops.basic","symbol":"sync_ops_basic_probe","compile":{}} -->
```python
# Ensure all prior vector stores are visible before any subsequent vector loads
pto.mem_bar(pto.BarrierType.VST_VLD)
```

The most commonly used barrier types in practice:

| Use case | Barrier type |
|----------|--------------|
| General vector ordering | `BarrierType.VV_ALL` |
| Store-then-load to same UB region | `BarrierType.VST_VLD` |
| Vector → scalar handoff | `BarrierType.VS_ALL` |
| Scalar → vector handoff | `BarrierType.SV_ALL` |

### Usage in explicit orchestration blocks

In explicit-mode kernels, phase boundaries use `pipe_barrier(Pipe.ALL)`, while
`mem_bar` remains the tool for narrower intra-pipeline ordering:

<!-- ptodsl-doc-test: {"mode":"compile_fragment","fixture":"sync_ops.phase_barrier_explicit","symbol":"sync_ops_phase_barrier_explicit_probe","compile":{"ROWS":8,"COLS":16}} -->
```python
# Inside a @pto.jit(mode="explicit") body:
def flash_attention_block(
    q_tile: pto.Tile,
    k_part: pto.PartitionTensorView,
    v_part: pto.PartitionTensorView,
    k_tile: pto.Tile,
    v_tile: pto.Tile,
    s_tile: pto.Tile,
    p_tile: pto.Tile,
    pv_tile: pto.Tile,
    o_prev_tile: pto.Tile,
    o_next_tile: pto.Tile,
    rows: pto.i32,
    cols: pto.i32,
):
    # Phase 1: load K/V
    row_bytes = cols * pto.bytewidth(pto.f16)
    gm_row_stride = k_part.strides[0] * pto.bytewidth(pto.f16)
    ub_row_stride = k_tile.shape[1] * pto.bytewidth(pto.f16)
    pto.mte_load(k_part.as_ptr(), k_tile.as_ptr(), 0, row_bytes,
                 nburst=(rows, gm_row_stride, ub_row_stride))
    pto.mte_load(v_part.as_ptr(), v_tile.as_ptr(), 0, row_bytes,
                 nburst=(rows, gm_row_stride, ub_row_stride))
    pto.pipe_barrier(pto.Pipe.ALL)

    # Phase 2: S = Q @ K^T
    qk_matmul(q_tile, k_tile, s_tile)
    pto.pipe_barrier(pto.Pipe.ALL)

    # Phase 3: softmax(S)
    online_softmax(s_tile, p_tile, rows, cols)
    pto.mem_bar(pto.BarrierType.VV_ALL)
    pto.pipe_barrier(pto.Pipe.ALL)

    # Phase 4: PV = P @ V
    pv_matmul(p_tile, v_tile, pv_tile)
    pto.pipe_barrier(pto.Pipe.ALL)

    # Phase 5: blend output
    blend_output(o_prev_tile, pv_tile, o_next_tile, rows, cols)
    pto.pipe_barrier(pto.Pipe.ALL)
```

---

## 10.5 Cross-core and intra-block synchronization

Section 10.2 covers the general pipe-to-pipe sync mechanism (`set_flag`/`wait_flag`). This section covers two additional sync domains that the pipe-flag mechanism does not address: **cross-core** communication between separate NPU cores, and **intra-block** synchronization between the Cube and Vector units within a block.

### 10.5.1 Cross-core sync: `set_cross_flag`, `wait_cross_flag`

When a kernel spans multiple cores, cores need to coordinate through shared resources. `set_cross_flag` sends a signal to another core; `wait_cross_flag` blocks the calling core until the expected signal arrives.

These are core-level (SU) operations — `wait_cross_flag` stalls the entire core, not just a single pipeline. Use them sparingly: splitting work so that each core operates independently for as long as possible minimises cross-core sync overhead.

#### `pto.set_cross_flag(pipe, event_id)`

**Description**: Signal an event on a synchronization endpoint. In the current PTODSL surface this is authored with a `Pipe`; the backend maps it to the architecture-specific cross-core / intra-block builtin during lowering.

**Parameters**:

| Parameter | Type | Description |
|-----------|------|-------------|
| `pipe` | `Pipe` | Producing endpoint for the synchronization event. The public DSL accepts `Pipe.FIX` here. |
| `event_id` | `int` | Cross-core event identifier (`0`–`7`) |

**Returns**: None (side-effect operation).

**Example**:

<!-- ptodsl-doc-test: {"mode":"compile_fragment","fixture":"sync_ops.basic","symbol":"sync_ops_basic_probe","compile":{}} -->
```python
# Signal from the FIX/Cube-side endpoint
pto.set_cross_flag(pto.Pipe.FIX, 0)
```

#### `pto.wait_cross_flag(pipe, event_id)`

**Description**: Wait for an event on a synchronization endpoint. On architectures that lower this surface to the backend `sync.wait` primitive, the wait is core-level (SU) blocking.

**Parameters**:

| Parameter | Type | Description |
|-----------|------|-------------|
| `pipe` | `Pipe` | Waiting endpoint for the synchronization event. The public DSL accepts `Pipe.FIX` here. |
| `event_id` | `int` | Event identifier to wait on (`0`–`7`) |

**Returns**: None (side-effect operation).

**Example**:

<!-- ptodsl-doc-test: {"mode":"compile_fragment","fixture":"sync_ops.basic","symbol":"sync_ops_basic_probe","compile":{}} -->
```python
# Wait on the FIX/Cube-side endpoint
pto.wait_cross_flag(pto.Pipe.FIX, 0)
```

### 10.5.2 Intra-block sync: `set_intra_flag`, `wait_intra_flag`

The Cube unit (matrix pipeline) has a dedicated synchronization channel separate from the standard pipe-flag mechanism used by MTE and Vector pipelines. `set_intra_flag` and `wait_intra_flag` synchronize Cube and Vector within the same block, ensuring that shared UB tile data is not accessed before the producer finishes.

Unlike `wait_cross_flag`, `wait_intra_flag` only stalls the specified pipeline — the SU and other pipelines continue executing.

#### `pto.set_intra_flag(pipe, event_id)`

**Description**: Signal a synchronization event within a block. The current PTODSL surface authors the trigger endpoint explicitly as a `Pipe`.

**Parameters**:

| Parameter | Type | Description |
|-----------|------|-------------|
| `pipe` | `Pipe` | Trigger endpoint for the synchronization event. The public DSL accepts `Pipe.MTE3` here. |
| `event_id` | `int` | Event identifier (`0`–`7`) |

**Returns**: None (side-effect operation).

**Example**:

<!-- ptodsl-doc-test: {"mode":"compile_fragment","fixture":"sync_ops.basic","symbol":"sync_ops_basic_probe","compile":{}} -->
```python
# Signal event ID0 from the MTE3-side endpoint
pto.set_intra_flag(pto.Pipe.MTE3, 0)
```

#### `pto.wait_intra_flag(pipe, event_id)`

**Description**: Wait for an intra-block event. Only the specified pipeline stalls — the SU and other pipelines continue executing independently.

**Parameters**:

| Parameter | Type | Description |
|-----------|------|-------------|
| `pipe` | `Pipe` | Waiting endpoint for the synchronization event. The public DSL accepts `Pipe.V` here. |
| `event_id` | `int` | Event identifier to wait on (`0`–`7`) |

**Returns**: None (side-effect operation).

**Example**:

<!-- ptodsl-doc-test: {"mode":"compile_fragment","fixture":"sync_ops.basic","symbol":"sync_ops_basic_probe","compile":{}} -->
```python
# Vector-side endpoint waits for event ID0
pto.wait_intra_flag(pto.Pipe.V, 0)
```

## 10.6 Synchronization in the authoring model

Where do sync operations belong in PTODSL's public entry model?

| Surface | Sync responsibility |
|---------|---------------------|
| `@pto.jit(mode="auto")` | Users can write sync explicitly when needed. PTOAS also provides an `--enable-insert-sync` option that auto-inserts `set_flag`/`wait_flag` pairs based on op-to-pipe mapping. |
| `@pto.jit(mode="explicit")` | The compiler does not insert sync — the user is fully responsible. Place `set_flag`/`wait_flag` between MTE and compute, `mem_bar` between compute phases, `pipe_barrier` at phase boundaries. |
| Shared `@pto.tileop` / `@pto.simt` helpers | Cross-pipeline ordering is provided by the surrounding `@pto.jit` schedule. Helpers may still use `mem_bar` for intra-pipeline ordering when UB addresses alias. |

**Rule of thumb**: in `mode="auto"`, think in tiles and let the compiler handle
orchestration. In `mode="explicit"`, think in micro-instructions and place the
required sync yourself.

### Auto-sync at the tile level

In auto mode, users can still write sync operations directly — `set_flag`/`wait_flag`, `pipe_barrier`, `mem_bar` are available in both modes. For convenience, PTOAS also provides an `--enable-insert-sync` pass: each tile op carries a pipe assignment (e.g., `tile.load` → `PIPE_MTE2`, `tile.add` → `PIPE_V`), and the pass analyzes the op sequence, infers the necessary `set_flag`/`wait_flag` pairs from pipe transitions, and injects them into the lowered code.

### Quick reference: which sync for which scenario

| Scenario | Sync primitive |
|----------|----------------|
| DMA load must finish before compute | `set_flag(MTE2, V, event_id=id)` + `wait_flag(MTE2, V, event_id=id)` |
| Compute must finish before DMA store | `set_flag(V, MTE3, event_id=id)` + `wait_flag(V, MTE3, event_id=id)` |
| Two compute phases must not overlap | `mem_bar(BarrierType.VV_ALL)` |
| Store must be visible to later load (same UB) | `mem_bar(BarrierType.VST_VLD)` |
| Full pipeline sync point | `pipe_barrier(Pipe.ALL)` |
| Double-buffer handoff (compute → DMA) | `rls_buf(V, id)` + `get_buf(MTE2, id)` |
| Double-buffer handoff (DMA → compute) | `rls_buf(MTE2, id)` + `get_buf(V, id)` |
| Core A notifies core B | `set_cross_flag(B, id)` + `wait_cross_flag(A, id)` |
