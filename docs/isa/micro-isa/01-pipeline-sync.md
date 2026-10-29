# 1. Pipeline Synchronization

> **Category:** Synchronization primitives for coordinating pipeline execution
> **Pipelines:** MTE2 (GM→UB), PIPE_V (Vector), MTE3 (UB→GM)

The PTO micro Instruction model operates on the Ascend 950's **Decoupled Access-Execute** architecture. The MTE and Vector pipelines run asynchronously, requiring explicit synchronization to prevent data hazards.

---

## Intra-Core Pipeline Sync

These ops coordinate data flow between pipelines within a single vector core.

### `pto.set_flag`

- **syntax:** `pto.set_flag["SRC_PIPE", "DST_PIPE", "EVENT_ID"]`
- **semantics:** Signal event from source pipe to destination pipe.

```c
set_flag(src_pipe, dst_pipe, event_id);
```

**Example:** After MTE2 completes GM→UB transfer, signal Vector pipe:
```mlir
pto.set_flag["PIPE_MTE2", "PIPE_V", "EVENT_ID0"]
```

---

### `pto.wait_flag`

- **syntax:** `pto.wait_flag["SRC_PIPE", "DST_PIPE", "EVENT_ID"]`
- **semantics:** Block destination pipe until source pipe signals event.

```c
wait_flag(src_pipe, dst_pipe, event_id);
```

**Example:** Vector pipe waits for MTE2 data to arrive:
```mlir
pto.wait_flag["PIPE_MTE2", "PIPE_V", "EVENT_ID0"]
```

---

### `pto.pipe_barrier`

- **syntax:** `pto.pipe_barrier "PIPE_*"`
- **semantics:** Drain all pending ops in the specified pipe. All previously issued operations on that pipe complete before any subsequent operation begins.

```c
pipe_barrier(pipe);
```

**Pipe identifiers:** `PIPE_MTE2`, `PIPE_V`, `PIPE_MTE3`

**Example:** Two back-to-back `pto.mte_ub_gm` calls writing to the same GM address. Without a barrier, MTE3 may reorder them and the final GM value is non-deterministic:

```mlir
// Both stores target the same GM address — order matters!
pto.mte_ub_gm %ub_partial_0, %gm_result, %len_burst ...
// Without pipe_barrier, MTE3 could execute the second copy before the first
// completes, producing a non-deterministic result at %gm_result.
pto.pipe_barrier "PIPE_MTE3"
// After barrier: first copy is guaranteed complete. Second copy overwrites deterministically.
pto.mte_ub_gm %ub_partial_1, %gm_result, %len_burst ...
```

---

### `pto.get_buf`

- **syntax:** `pto.get_buf "PIPE_*", %buf_id, %mode : i64, i64`
- **semantics:** Acquire buffer slot for inter-pipeline double-buffering coordination.

```c
get_buf(pipe, buf_id, mode);
```

---

### `pto.rls_buf`

- **syntax:** `pto.rls_buf "PIPE_*", %buf_id, %mode : i64, i64`
- **semantics:** Release buffer slot to allow other pipeline to proceed.

```c
rls_buf(pipe, buf_id, mode);
```

---

### Mode Parameter for `get_buf` / `rls_buf`

The `mode` parameter controls how `get_buf` and `rls_buf` interact with pipeline execution and dependency tracking:

| Mode | `get_buf` Behavior | `rls_buf` Behavior | Use Case |
|------|-------------------|-------------------|----------|
| **0** (default) | **Blocking acquire**: waits for all previous `rls_buf` with same `buf_id` from all pipelines (in program order) before the specified pipe can proceed | **Immediate release**: signals completion for only the instructions related to the specified pipe | **Automatic ping/pong dependency** — recommended for double/multi-buffering |
| **1** | **Non-blocking acquire**: does not wait; pipe execution proceeds immediately | **Deferred release**: waits for all instructions across all pipelines with same `buf_id` to retire before signaling | **Backward compatibility** with `set_flag`/`wait_flag` semantics |

**Mode 0 (Default — Recommended):**
- `get_buf`: The specified pipeline blocks until all previous `rls_buf` operations for the same buffer ID (from any pipeline) have completed, respecting program order.
- `rls_buf`: Immediately signals that the specified pipeline has finished using the buffer — only waits for that pipe's related instructions.
- This mode provides **automatic RAW/WAR/WAW dependency resolution** based on buffer ID and program order, making it ideal for ping/pong and N-buffer patterns.

**Mode 1 (Legacy Compatibility):**
- `get_buf`: Does not block — the pipeline proceeds immediately without waiting.
- `rls_buf`: Waits for **all** previous instructions across **all** pipelines with the same buffer ID to retire before signaling release.
- This mode emulates `set_flag`/`wait_flag` behavior and is provided for backward compatibility with existing code patterns.

> **Note:** A5 supports both `set_flag`/`wait_flag` and `get_buf`/`rls_buf` mechanisms. Mode 1 is rarely needed since mode 0 provides a more programmer-friendly approach for buffer-based synchronization.

---

### `pto.get_buf_dyn` / `pto.rls_buf_dyn`

Dynamic variants of `get_buf`/`rls_buf` where `buf_id` is provided as an SSA value instead of a static integer attribute. This enables runtime-computed buf_id patterns such as SIMT ping-pong buffering.

- **syntax:**
  - String shorthand: `pto.get_buf_dyn "PIPE_MTE2", %buf_id, 0`
  - Bracket form: `pto.get_buf_dyn [TLOAD, %buf_id, 0]`
- **semantics:** Same as `get_buf`/`rls_buf`, but the buffer-id is an `index`-typed SSA value resolved at runtime.
- **inputs:**
  - `op_type`: same pipe-like attribute as the static form
  - `buf_id`: an SSA value of `index` type (e.g. `iter & 1` for ping-pong)
  - `mode`: same mode parameter (default `0`)
- **constraints and limitations:** The BufidSync auto-insertion pass only uses the static form (`get_buf`/`rls_buf`). Use the dynamic form (`get_buf_dyn`/`rls_buf_dyn`) when buf_id must be computed at runtime.

Example (SIMT double-buffering with `iter & 1`):

```mlir
  %c1 = arith.constant 1 : index
  %buf_id = arith.andi %iter, %c1 : index
  pto.get_buf_dyn [TLOAD, %buf_id, 0]
  // ... tload to ubuf slot %buf_id ...
  pto.rls_buf_dyn [TLOAD, %buf_id, 0]
```

---

### `pto.mem_bar`

- **syntax:** `pto.mem_bar "BARRIER_TYPE"`
- **semantics:** Shared-memory (UB address space) memory fence within `__VEC_SCOPE__`. Required when UB addresses alias between memory operations. The barrier type selects which classes of prior instructions must complete before which classes of subsequent instructions may proceed.

```c
mem_bar(barrier_type);
```

**Barrier types** are organized into four families by the scope of prior vs. subsequent instructions:

| Family | Barrier type | Prior instructions | Subsequent instructions |
|--------|-------------|-------------------|------------------------|
| **VV** (vector→vector) | `VV_ALL` | All vector load/store | All vector load/store |
| | `VST_VLD` | All vector store | All vector load |
| | `VLD_VST` | All vector load | All vector store |
| | `VST_VST` | All vector store | All vector store |
| **VS** (vector→scalar) | `VS_ALL` | All vector load/store | All scalar load/store |
| | `VST_LD` | All vector store | All scalar load |
| | `VLD_ST` | All vector load | All scalar store |
| | `VST_ST` | All vector store | All scalar store |
| **SV** (scalar→vector) | `SV_ALL` | All scalar load/store | All vector load/store |
| | `ST_VLD` | All scalar store | All vector load |
| | `LD_VST` | All scalar load | All vector store |
| | `ST_VST` | All scalar store | All vector store |
| **SS** (scalar→scalar) | `SS_ALL` | All scalar load/store | All scalar load/store |
| | `ST_LD` | All scalar store | All scalar load |
| | `LD_ST` | All scalar load | All scalar store |
| | `ST_ST` | All scalar store | All scalar store |

**Example:** Ensure vector stores are visible before subsequent vector loads to the same UB region:
```mlir
pto.vsts %v0, %ub[%c0] : !pto.vreg<64xf32>, !pto.ptr<f32, ub>
pto.mem_bar "VST_VLD"
%v1 = pto.vlds %ub[%c0] : !pto.ptr<f32, ub> -> !pto.vreg<64xf32>
```

**Scope and provision of barriers.** `pto.mem_bar` only orders operations inside a single `__VEC_SCOPE__` region. Making one vecscope's UB writes visible to a later sibling vecscope is a cross-region ordering concern and must be managed with `pto.set_flag` / `pto.wait_flag` (or `pto.get_buf` / `pto.rls_buf`); a `pto.mem_bar` cannot express it.

Barrier insertion modes:

- **Manual:** written in the source. In the ordinary automatic mode, the ordering established by a manual barrier counts as already provided when its type and placement cover the dependency. An adjacent barrier can be combined with a newly required barrier of the same family into that family’s `ALL` barrier. The current insertion pass does not generally remove standalone redundant manual barriers. Conservative insertion can add barriers even when a manual barrier already orders the accesses.
- **Automatic insertion (VPTO):** `--enable-vecscope-mem-bar` is enabled by default. PTOAS analyzes UB access footprints and inserts barriers where overlapping accesses still require ordering. Scalar-to-scalar accesses are ordered by the scalar pipeline and do not require automatically inserted barriers; use `--enable-vecscope-mem-bar=false` to disable it.
- **Conservative insertion (VPTO):** `--enable-vecscope-mem-bar-all` defaults to off and takes precedence over the ordinary mode when enabled. It inserts `SV_ALL` and `VV_ALL` before each UB vector access, and only `VS_ALL` before each UB scalar access. Scalar-to-scalar accesses are already ordered by the scalar pipeline, so no `SS_ALL` is inserted. This mode performs no dependence, footprint, or redundancy analysis and preserves manual barriers.

**Device-compiler automatic synchronization.** The device compiler (Bisheng) can also insert vector-pipeline synchronization on its own. Who owns that insertion depends on the compilation route:

- When the kernel is compiled through the VMI tile-fusion pipeline (`--enable-vmi` with `--enable-op-fusion=true` on the A5 VPTO backend at `--pto-level=level2` or `level3`), the program PTOAS emits already carries its synchronization, so PTOAS disables the device compiler's automatic vector-pipeline synchronization and its own vector-fusion transformations for that device compilation.
- On every other route, device-compiler automatic vector-pipeline synchronization also defaults to off. PTOAS barrier insertion remains controlled independently by the options above.
- `--bisheng-vf-auto-sync=off|fused|global` selects the device-compiler synchronization mode (default: `off`). An explicit choice is honored on any route, including the VMI route.
- `--bisheng-vec-misched=auto|on|off` selects the device compiler's vector-instruction scheduling policy: `off` disables the vector scheduler, `on` keeps the device compiler's default scheduling, and `auto` (the default) keeps default scheduling and retries with the scheduler disabled when compilation fails or the reported vector-stack usage is nonzero, preferring the smaller vector stack. On the VMI tile-fusion route the scheduler likewise defaults to off; an explicit choice is honored on any route.

---

### `pto.dsb`

- **syntax:** `pto.dsb "MEM_DOMAIN"`
- **semantics:** Issues a data synchronization barrier for the selected memory domain. All prior memory effects covered by the domain must become complete before subsequent memory effects proceed.

**Memory domains:**

| Domain | Meaning |
|--------|---------|
| `ALL` | Wait for all memory-access classes covered by the target |
| `DDR` | Wait for DDR/GM memory-access effects |
| `UB` | Wait for UB memory-access effects |
| `SEQ` | Wait for sequencer-visible memory-access effects |

**Example:** Ensure prior GM stores are complete before publishing a scalar signal:

```mlir
pto.dsb "DDR"
```

---

### `pto.dcci`

- **syntax:** `pto.dcci %ptr "CACHE_SCOPE" : !pto.ptr<T, gm|ub>`
- **syntax:** `pto.dcci %ptr "CACHE_SCOPE", "CACHE_DST" : !pto.ptr<T, gm|ub>`
- **semantics:** Performs data-cache clean/invalidate maintenance for the selected cache scope. The pointer address space selects whether the operation applies to GM or UB-backed cache state. The optional destination domain further restricts which cache-line class is affected.

**Cache scopes:**

| Scope | Meaning |
|-------|---------|
| `SINGLE_CACHE_LINE` | Apply maintenance to the cache line containing `%ptr` |
| `ENTIRE_DATA_CACHE` | Apply maintenance to the entire data cache; `%ptr` is still required by the IR form |

**Destination domains:**

| Destination | Meaning |
|-------------|---------|
| `CACHELINE_ALL` | All supported cache-line domains |
| `CACHELINE_UB` | UB cache-line domain |
| `CACHELINE_OUT` | Output/GM-visible cache-line domain |
| `CACHELINE_ATOMIC` | Atomic cache-line domain |

**Constraints:**
- `%ptr` must be a PTO pointer or buffer-like value in GM or UB address space.
- Omitting `CACHE_DST` uses the target's default destination-domain form.

**Example:** Flush GM-visible cache state after scalar GM stores:

```mlir
pto.dcci %gm "ENTIRE_DATA_CACHE", "CACHELINE_OUT" : !pto.ptr<i8, gm>
pto.dsb "ALL"
```

---

## Why `get_buf` / `rls_buf` is More Programmer-Friendly

The buffer-based synchronization (`get_buf`/`rls_buf`) provides the **same functional capability** as `set_flag`/`wait_flag` for maintaining correct ordering of RAW/WAR/WAW dependencies across pipelines, but with significant usability advantages:

### 1. No Manual Priming or Draining

With `set_flag`/`wait_flag`, ping/pong loops require:
- **Pre-loop priming**: 4× `set_flag` to initialize reverse-dependency signals (otherwise first iteration deadlocks)
- **Post-loop draining**: 4× `wait_flag` to consume leftover signals from final iterations

With `get_buf`/`rls_buf`:
- **First iteration**: Buffer is initially free, so `get_buf` proceeds immediately — no priming needed
- **Final iteration**: Last `rls_buf` simply completes — no draining required

### 2. No Loop Peeling for Complex Dependencies

For non-1:1 producer-consumer ratios (e.g., 1 MTE2 load : N Vector compute slices), `set_flag`/`wait_flag` requires **peeling the set_flag outside the loop**:

```mlir
// set_flag/wait_flag: 1 MTE2 load, 8 Vector computes on slices
// MTE2 loads large tile once
pto.mte_gm_ub %gm_ptr, %ub_tile, ...
pto.set_flag["PIPE_MTE2", "PIPE_V", "EVT_TILE_READY"]  // ◀ MUST be outside loop

// Vector consumes in 8 slices — but wait_flag can only fire ONCE
pto.wait_flag["PIPE_MTE2", "PIPE_V", "EVT_TILE_READY"] // ◀ MUST peel before loop
scf.for %slice = %c0 to %c8 step %c1 {
  // compute on %ub_tile[%slice]
  // Cannot put wait_flag here — would deadlock on iteration 1+
}
```

With `get_buf`/`rls_buf`, acquire/release can be **inside the loop** — no peeling needed:

```mlir
// get_buf/rls_buf: same 1:8 pattern, acquire/release inside loop works fine
// MTE2 loads large tile
pto.get_buf "PIPE_MTE2", %bufid_tile, %c0 : i64, i64
pto.mte_gm_ub %gm_ptr, %ub_tile, ...
pto.rls_buf "PIPE_MTE2", %bufid_tile, %c0 : i64, i64

// Vector acquires/releases per slice — all 8 iterations work correctly
scf.for %slice = %c0 to %c8 step %c1 {
  pto.get_buf "PIPE_V", %bufid_tile, %c0 : i64, i64  // iteration 0: blocks until MTE2 done
                                                      // iteration 1-7: proceeds immediately (already acquired)
  // compute on %ub_tile[%slice]
  pto.rls_buf "PIPE_V", %bufid_tile, %c0 : i64, i64
}
// No peeling required — get_buf handles the MTE2→V dependency automatically
```

### 3. Simpler Mental Model

| Aspect | `set_flag`/`wait_flag` | `get_buf`/`rls_buf` |
|--------|------------------------|---------------------|
| **Dependency tracking** | Manual: track event IDs, signal directions, pair every set with wait | Automatic: buffer ID + program order |
| **Event ID management** | **8 IDs per pipe-pair direction** (HW limit); each buffer occupies 1 ID per direction | **1 buffer ID per shared resource** (HW limit: 32 global); same ID used across all pipelines |
| **Error-prone areas** | Forgetting prime/drain, mismatched IDs, wrong direction | Forgetting release (but compile-time checkable) |

### Quick Example Comparison

**Problem:** MTE2 loads into `buf[i%2]`, Vector processes, MTE3 stores — standard ping/pong.

**set_flag/wait_flag approach:**
```mlir
// BEFORE loop: prime 4 reverse-dep signals
pto.set_flag["PIPE_V", "PIPE_MTE2", "EVT_IN_REV_0"]
pto.set_flag["PIPE_V", "PIPE_MTE2", "EVT_IN_REV_1"]
pto.set_flag["PIPE_MTE3", "PIPE_V", "EVT_OUT_REV_0"]
pto.set_flag["PIPE_MTE3", "PIPE_V", "EVT_OUT_REV_1"]

scf.for %i = ... {
  // 4 set_flag + 4 wait_flag inside loop
  // Must track 4 IDs: 2 pipe-pair directions × 2 ping/pong buffers
}

// AFTER loop: drain 4 signals
pto.wait_flag["PIPE_V", "PIPE_MTE2", "EVT_IN_REV_0"]
pto.wait_flag["PIPE_V", "PIPE_MTE2", "EVT_IN_REV_1"]
pto.wait_flag["PIPE_MTE3", "PIPE_V", "EVT_OUT_REV_0"]
pto.wait_flag["PIPE_MTE3", "PIPE_V", "EVT_OUT_REV_1"]
```

**get_buf/rls_buf approach:**
```mlir
scf.for %i = ... {
  pto.get_buf %bufid_in[%pp], "PIPE_MTE2"
  // ... MTE2 work ...
  pto.rls_buf %bufid_in[%pp], "PIPE_MTE2"

  pto.get_buf %bufid_in[%pp], "PIPE_V"
  pto.get_buf %bufid_out[%pp], "PIPE_V"
  // ... Vector work ...
  pto.rls_buf %bufid_in[%pp], "PIPE_V"
  pto.rls_buf %bufid_out[%pp], "PIPE_V"

  pto.get_buf %bufid_out[%pp], "PIPE_MTE3"
  // ... MTE3 work ...
  pto.rls_buf %bufid_out[%pp], "PIPE_MTE3"
}
// Done. No prime. No drain. Dependencies resolved by buffer ID + program order.
```

---

## Intra-Core Sync Patterns & Examples

### Example 1: `set_flag` / `wait_flag` (Explicit Events)

Each cross-pipeline data dependency requires an explicit signal/wait pair. The programmer must manually insert `set_flag` after the producer and `wait_flag` before the consumer.

```mlir
// ─── Stage 1: MTE2 loads data from GM into UB ───
pto.mte_gm_ub %gm_ptr, %ub_ptr, ...

// MTE2 signals: "UB data is ready for Vector pipe"
pto.set_flag["PIPE_MTE2", "PIPE_V", "EVENT_ID0"]

// ─── Stage 2: Vector pipe consumes UB data ───
// Vector waits until MTE2's signal arrives
pto.wait_flag["PIPE_MTE2", "PIPE_V", "EVENT_ID0"]

pto.vecscope {
  %v   = pto.vlds %ub_ptr[%lane] : !pto.ptr<f32, ub> -> !pto.vreg<64xf32>
  %mask = pto.pset_b32 "PAT_ALL" : !pto.mask<b32>
  %abs = pto.vabs %v, %mask : !pto.vreg<64xf32>, !pto.mask<b32> -> !pto.vreg<64xf32>
  pto.vsts %abs, %ub_out[%lane], %mask : !pto.vreg<64xf32>, !pto.ptr<f32, ub>, !pto.mask<b32>
}

// Vector signals: "UB output is ready for MTE3"
pto.set_flag["PIPE_V", "PIPE_MTE3", "EVENT_ID0"]

// ─── Stage 3: MTE3 stores result from UB back to GM ───
// MTE3 waits until Vector's signal arrives
pto.wait_flag["PIPE_V", "PIPE_MTE3", "EVENT_ID0"]

pto.mte_ub_gm %ub_out, %gm_out, %len_burst ...
```

**Key property:** Every cross-pipeline edge is an explicit `(set_flag, wait_flag)` pair. Simple for straight-line code, but gets verbose in loops (see Example 3).

---

### Example 2: `get_buf` / `rls_buf` (Resource-Based)

Instead of naming events, each pipeline declares when it **acquires** (`get_buf`) and **releases** (`rls_buf`) a shared UB buffer. Cross-pipeline RAW/WAR dependencies are resolved implicitly by program order — if MTE2 releases `buf_A` and Vector later acquires `buf_A`, the hardware ensures the acquire cannot proceed until the release completes.

```mlir
// ─── Stage 1: MTE2 loads data into UB ───
// MTE2 acquires ub_ptr — blocks if Vector hasn't released it from a prior iteration
pto.get_buf "PIPE_MTE2", %bufid_ub_ptr, %c0 : i64, i64   // mode=0 (default)
pto.mte_gm_ub %gm_ptr, %ub_ptr, ...
// MTE2 done writing ub_ptr — release it so Vector can consume
pto.rls_buf "PIPE_MTE2", %bufid_ub_ptr, %c0 : i64, i64

// ─── Stage 2: Vector computation ───
// Vector acquires ub_ptr (input) — blocks until MTE2 releases it (RAW: MTE2 write → V read)
pto.get_buf "PIPE_V", %bufid_ub_ptr, %c0 : i64, i64
// Vector acquires ub_out (output) — blocks until MTE3 releases it from a prior iteration (WAR: MTE3 read → V write)
pto.get_buf "PIPE_V", %bufid_ub_out, %c0 : i64, i64

pto.vecscope {
  %mask = pto.pset_b32 "PAT_ALL" : !pto.mask<b32>
  %v   = pto.vlds %ub_ptr[%lane] : !pto.ptr<f32, ub> -> !pto.vreg<64xf32>
  %abs = pto.vabs %v, %mask : !pto.vreg<64xf32>, !pto.mask<b32> -> !pto.vreg<64xf32>
  pto.vsts %abs, %ub_out[%lane], %mask : !pto.vreg<64xf32>, !pto.ptr<f32, ub>, !pto.mask<b32>
}

// Vector done reading ub_ptr — release so MTE2 can reuse it in next iteration
pto.rls_buf "PIPE_V", %bufid_ub_ptr, %c0 : i64, i64
// Vector done writing ub_out — release so MTE3 can consume
pto.rls_buf "PIPE_V", %bufid_ub_out, %c0 : i64, i64

// ─── Stage 3: MTE3 stores result to GM ───
// MTE3 acquires ub_out — blocks until Vector releases it (RAW: V write → MTE3 read)
pto.get_buf "PIPE_MTE3", %bufid_ub_out, %c0 : i64, i64
pto.mte_ub_gm %ub_out, %gm_out, %len_burst ...
// MTE3 done reading ub_out — release so Vector can reuse it in next iteration
pto.rls_buf "PIPE_MTE3", %bufid_ub_out, %c0 : i64, i64
```

**Key property:** No event IDs needed. Dependencies are implicit from program order of `get_buf`/`rls_buf` on the same buffer ID. This becomes much more convenient in multi-iteration loops (see Example 3).

---

### Example 3: Ping/Pong Double-Buffering Loop

Double-buffering overlaps DMA and compute by using two UB buffers alternately. All three stages (MTE2, Vector, MTE3) appear in the **same iteration** — the hardware pipelines them across iterations because different iterations operate on different buffers (`buf[i%2]`).

#### Event ID scheme (`set_flag` / `wait_flag`)

With 2 ping/pong buffers and 2 pipeline pairs (MTE2↔V, V↔MTE3), `set_flag`/`wait_flag` needs **8 event IDs** = 2 pipe-pairs × 2 buffers × (forward + reverse):

**MTE2 ↔ Vector (input buffers):**

| Event ID | Direction | Purpose |
|----------|-----------|---------|
| `EVT_IN_FWD_0` | MTE2 → V | RAW: buf_in[0] data ready |
| `EVT_IN_FWD_1` | MTE2 → V | RAW: buf_in[1] data ready |
| `EVT_IN_REV_0` | V → MTE2 | WAR: Vector done reading buf_in[0] |
| `EVT_IN_REV_1` | V → MTE2 | WAR: Vector done reading buf_in[1] |

**Vector ↔ MTE3 (output buffers):**

| Event ID | Direction | Purpose |
|----------|-----------|---------|
| `EVT_OUT_FWD_0` | V → MTE3 | RAW: buf_out[0] result ready |
| `EVT_OUT_FWD_1` | V → MTE3 | RAW: buf_out[1] result ready |
| `EVT_OUT_REV_0` | MTE3 → V | WAR: MTE3 done reading buf_out[0] |
| `EVT_OUT_REV_1` | MTE3 → V | WAR: MTE3 done reading buf_out[1] |

#### 3a. `set_flag` / `wait_flag` version

```mlir
// ═══ Pre-loop: prime ALL reverse-dependency signals ═══
// Both input and output buffers start unused. We must pre-send
// reverse-dep signals so the first iteration's wait_flags don't deadlock.
pto.set_flag["PIPE_V",    "PIPE_MTE2", "EVT_IN_REV_0"]   // ◀ PRIME: buf_in[0] "free"
pto.set_flag["PIPE_V",    "PIPE_MTE2", "EVT_IN_REV_1"]   // ◀ PRIME: buf_in[1] "free"
pto.set_flag["PIPE_MTE3", "PIPE_V",    "EVT_OUT_REV_0"]  // ◀ PRIME: buf_out[0] "free"
pto.set_flag["PIPE_MTE3", "PIPE_V",    "EVT_OUT_REV_1"]  // ◀ PRIME: buf_out[1] "free"

scf.for %i = %c0 to %N step %c1 {
  // ── All 3 stages in same iteration, indexed by i%2 ──
  // %pp = i % 2  (ping/pong selector for buffer & event IDs)

  // ── MTE2: load tile[i] into buf_in[i%2] ──
  // WAR: wait until Vector has released buf_in[i%2] from iteration i-2
  pto.wait_flag["PIPE_V", "PIPE_MTE2", "EVT_IN_REV_{pp}"]
  pto.mte_gm_ub %gm_ptr[%i], %ub_in[%pp], ...
  // RAW: signal Vector that buf_in[i%2] data is ready
  pto.set_flag["PIPE_MTE2", "PIPE_V", "EVT_IN_FWD_{pp}"]

  // ── Vector: compute buf_in[i%2] → buf_out[i%2] ──
  // RAW: wait for MTE2 to finish loading buf_in[i%2]
  pto.wait_flag["PIPE_MTE2", "PIPE_V", "EVT_IN_FWD_{pp}"]
  // WAR: wait for MTE3 to finish reading buf_out[i%2] from iteration i-2
  pto.wait_flag["PIPE_MTE3", "PIPE_V", "EVT_OUT_REV_{pp}"]
  scf.for %dummy = %c0 to %c1 step %c1 {
    %v   = pto.vlds %ub_in[%pp][%lane] : !pto.ptr<f32, ub> -> !pto.vreg<64xf32>
    %mask = pto.pset_b32 "PAT_ALL" : !pto.mask<b32>
    %abs = pto.vabs %v, %mask : !pto.vreg<64xf32>, !pto.mask<b32> -> !pto.vreg<64xf32>
    pto.vsts %abs, %ub_out[%pp][%lane], %mask : !pto.vreg<64xf32>, !pto.ptr<f32, ub>, !pto.mask<b32>
  } {llvm.loop.aivector_scope}
  // WAR: tell MTE2 "done reading buf_in[i%2]"
  pto.set_flag["PIPE_V", "PIPE_MTE2", "EVT_IN_REV_{pp}"]
  // RAW: tell MTE3 "buf_out[i%2] result ready"
  pto.set_flag["PIPE_V", "PIPE_MTE3", "EVT_OUT_FWD_{pp}"]

  // ── MTE3: store result from buf_out[i%2] to GM ──
  // RAW: wait for Vector to finish writing buf_out[i%2]
  pto.wait_flag["PIPE_V", "PIPE_MTE3", "EVT_OUT_FWD_{pp}"]
  pto.mte_ub_gm %ub_out[%pp], %gm_out[%i], %len_burst ...
  // WAR: tell Vector "done reading buf_out[i%2]"
  pto.set_flag["PIPE_MTE3", "PIPE_V", "EVT_OUT_REV_{pp}"]
}

// ═══ Post-loop: drain — match every pre-loop prime with a wait ═══
// Each priming set_flag must be paired. The last loop iteration's
// set_flags are consumed by wait_flags that will never fire inside the
// loop (there is no iteration i+2). Drain them here.
pto.wait_flag["PIPE_V",    "PIPE_MTE2", "EVT_IN_REV_{(N-1)%2}"]  // ◀ DRAIN
pto.wait_flag["PIPE_V",    "PIPE_MTE2", "EVT_IN_REV_{(N-2)%2}"]  // ◀ DRAIN
pto.wait_flag["PIPE_MTE3", "PIPE_V",    "EVT_OUT_REV_{(N-1)%2}"] // ◀ DRAIN
pto.wait_flag["PIPE_MTE3", "PIPE_V",    "EVT_OUT_REV_{(N-2)%2}"] // ◀ DRAIN
```

**What `set_flag`/`wait_flag` requires outside the loop:**
- **Before the loop (4 × `set_flag`):** Prime every reverse-dependency event ID — one per buffer per pipe-pair. Without this, the first iteration's `wait_flag` for reverse deps would deadlock (no signal was ever sent).
- **After the loop (4 × `wait_flag`):** Drain the matching reverse-dep signals from the last iterations. Every `set_flag` must be paired with a `wait_flag` — the last loop iterations produce signals that no subsequent iteration consumes, so they must be drained explicitly.

#### 3b. `get_buf` / `rls_buf` version

Same ping/pong double-buffering, but **no pre-loop priming or post-loop draining needed.** Buffer acquire/release semantics handle everything.

```mlir
scf.for %i = %c0 to %N step %c1 {
  // %pp = i % 2  (ping/pong selector)

  // ── MTE2: load tile[i] into buf[i%2] ──
  // Acquires buf[i%2] — on first iteration, buffer is free so proceeds immediately.
  // On later iterations, blocks until Vector releases buf[i%2] (WAR: automatic).
  pto.get_buf "PIPE_MTE2", %bufid_buf[%pp], %c0 : i64, i64   // mode=0
  pto.mte_gm_ub %gm_ptr[%i], %ub_buf[%pp], ...
  pto.rls_buf "PIPE_MTE2", %bufid_buf[%pp], %c0 : i64, i64

  // ── Vector: compute on buf[i%2] ──
  // Acquires buf[i%2] — blocks until MTE2 releases it (RAW: automatic)
  pto.get_buf "PIPE_V", %bufid_buf[%pp], %c0 : i64, i64
  pto.get_buf "PIPE_V", %bufid_out[%pp], %c0 : i64, i64
  scf.for %dummy = %c0 to %c1 step %c1 {
    %v   = pto.vlds %ub_buf[%pp][%lane] : !pto.ptr<f32, ub> -> !pto.vreg<64xf32>
    %mask = pto.pset_b32 "PAT_ALL" : !pto.mask<b32>
    %abs = pto.vabs %v, %mask : !pto.vreg<64xf32>, !pto.mask<b32> -> !pto.vreg<64xf32>
    pto.vsts %abs, %ub_out[%pp][%lane], %mask : !pto.vreg<64xf32>, !pto.ptr<f32, ub>, !pto.mask<b32>
  } {llvm.loop.aivector_scope}
  // Release buf[i%2] — MTE2 can reuse in iteration i+2 (WAR resolved)
  pto.rls_buf "PIPE_V", %bufid_buf[%pp], %c0 : i64, i64
  pto.rls_buf "PIPE_V", %bufid_out[%pp], %c0 : i64, i64

  // ── MTE3: store result ──
  // Acquires out[i%2] — blocks until Vector releases it (RAW: automatic)
  pto.get_buf "PIPE_MTE3", %bufid_out[%pp], %c0 : i64, i64
  pto.mte_ub_gm %ub_out[%pp], %gm_out[%i], %len_burst ...
  pto.rls_buf "PIPE_MTE3", %bufid_out[%pp], %c0 : i64, i64
}
// No post-loop drain needed — last rls_buf completes the pipeline.
```

**No priming, no draining, no event IDs.** The acquire/release protocol on buffer IDs indexed by `i%2` implicitly resolves all cross-pipeline dependencies:
- **RAW** (MTE2→V): Vector's `get_buf` blocks until MTE2's `rls_buf` on `buf[i%2]`
- **WAR** (V→MTE2): MTE2's `get_buf` in iteration `i+2` blocks until Vector's `rls_buf` in iteration `i` (same buffer)
- **First iteration:** Buffer is initially free, so `get_buf` proceeds without blocking — no priming needed

---

## Comparison Summary

| Aspect | `set_flag` / `wait_flag` | `get_buf` / `rls_buf` |
|--------|--------------------------|------------------------|
| Dependency model | Explicit event signals | Implicit via buffer acquire/release |
| IDs per pipe-pair | 2 IDs per buffer: 1 for forward (e.g., MTE2→V) + 1 for reverse (V→MTE2) | **1 ID per buffer** (handles both directions automatically) |
| Total HW IDs | **8 per pipe-pair** (hardware limit) | **32 global** across all pipes |
| Reverse (WAR) deps | Extra `set_flag`/`wait_flag` pair per buffer | Handled automatically |
| Pre-loop setup | `set_flag` to prime each reverse dep | **None** |
| Post-loop teardown | `wait_flag` to drain all primed signals | **None** |
| Loop peeling for complex deps | Required for non-1:1 or nested loops | **Not required** |
| Straight-line code | Simple, clear | Slightly more verbose (bracket each stage) |
| Ping/pong loops | 8 event IDs + 4 prime + 4 drain | Same pattern, **no overhead** |
| Best used for | Simple pipelines, fine-grained control | Double/multi-buffering, complex loops |

---

## Inter-Core Sync

> **Note:** Inter-core sync is only needed for **mixed Cube+Vector tasks** where Cube produces data that Vector consumes (or vice versa). **Vec-only tasks can ignore this section entirely.**

These ops coordinate execution across the Cube block and Vector subblocks within a cluster. Each core cluster consists of **1 Cube block : 2 Vector subblocks**, each with its own **SU (Sequencer Unit)** running independent instruction streams.

```
Core Cluster (1:2 ratio)
┌─────────────────────────────────────────────┐
│  ┌──────────────┐    ┌──────────────┐       │
│  │  AIC (Cube)  │    │  AIV0 (Vec)  │       │
│  │  ┌────────┐  │    │  ┌────────┐  │       │
│  │  │   SU   │──┼────┼──│   SU   │  │       │
│  │  └────────┘  │    │  └────────┘  │       │
│  │  CUBE pipe   │    │  MTE2/V/MTE3 │       │
│  │  L0C buffer  │    │  UB (256KB)  │       │
│  └──────────────┘    └──────────────┘       │
│                      ┌──────────────┐       │
│                      │  AIV1 (Vec)  │       │
│                      │  ┌────────┐  │       │
│                      │  │   SU   │  │       │
│                      │  └────────┘  │       │
│                      │  MTE2/V/MTE3 │       │
│                      │  UB (256KB)  │       │
│                      └──────────────┘       │
└─────────────────────────────────────────────┘
```

### Platform Comparison

| Aspect | A2A3 (Ascend 910) | A5 (Ascend 950) |
|--------|-------------------|-----------------|
| **Signal op** | `set_cross_core` (mode2) | `set_intra_block` |
| **Wait op** | `wait_flag_dev` | `wait_intra_core` |
| **Wait behavior** | SU-level blocking (entire core stalls) | Per-pipeline (only named pipe stalls) |
| **Semaphore pool** | 16 IDs per cluster, 4-bit counter | 16 IDs, but 32-ID address space (see below) |
| **C→V** | **Broadcast**: one `set` reaches both AIV0+AIV1 | **1:1**: separate `set` per subblock required |
| **V→C** | **Reduce**: Cube waits for both subblocks in one `wait` | **1:1**: Cube needs separate `wait` per subblock |

### A2A3: `set_cross_core` / `wait_flag_dev`

```c
// mode2 broadcast/reduce semantics for 1:2 cluster
set_cross_core(pipe, semaphore_id);   // pipe: VEC/MTE2/CUBE/FIX
wait_flag_dev(semaphore_id);          // SU-level blocking
```

```
C→V Broadcast (one set reaches both):
    AIC ──set_cross_core──┬──> AIV0 sema++
                          └──> AIV1 sema++

V→C Reduce (one wait for both):
    AIV0 ──set_cross_core──┐
                           ├──> AIC wait_flag_dev (blocks until BOTH)
    AIV1 ──set_cross_core──┘
```

### `pto.set_cross_core`

- **syntax:** `pto.set_cross_core %core_id, %event_id : i64, i64`
- **semantics:** Signal event to another core. Uses **mode2** for 1:2 cluster on A2A3.

### `pto.wait_flag_dev`

- **syntax:** `pto.wait_flag_dev %core_id, %event_id : i64, i64`
- **semantics:** Wait for event from another core. **SU-level blocking** — entire core stalls.

### A5: `set_intra_block` / `wait_intra_core`

```c
set_intra_block(trigger_pipe, semaphore_id);
wait_intra_core(wait_pipe, semaphore_id);   // only named pipe stalls
```

**A5 semaphore address space:** The hardware has **16 physical semaphore IDs** but exposes a **32-ID address space** to support 1:1 signaling to each subblock:

| ID Range | Target |
|----------|--------|
| 0–15 | AIV0 (subblock 0) |
| 16–31 (+15 offset) | AIV1 (subblock 1) |

This means C→V requires **separate `set_intra_block` calls** per subblock (no broadcast), and V→C requires **separate `wait_intra_core` calls** per subblock (no hardware reduce).

```
C→V on A5 (1:1, no broadcast — need two sets):
    AIC ──set_intra_block(pipe, sema_id)────> AIV0
    AIC ──set_intra_block(pipe, sema_id+15)──> AIV1

V→C on A5 (1:1, no reduce — need two waits):
    AIV0 ──set_intra_block──> AIC wait_intra_core(pipe, sema_id)
    AIV1 ──set_intra_block──> AIC wait_intra_core(pipe, sema_id+15)  // extra wait
```

### `pto.set_intra_block`

- **syntax:** `pto.set_intra_block %block_id, %event_id : i64, i64`
- **semantics:** Signal event within a block (A5). Specifies **trigger pipe**. 1:1 per subblock.

### `pto.wait_intra_core`

- **syntax:** `pto.wait_intra_core %block_id, %event_id : i64, i64`
- **semantics:** Wait for event within block (A5). Specifies **which pipeline should wait** — only that pipe stalls, SU and other pipes continue.

### Wait Granularity Comparison

```
A2A3 wait_flag_dev (SU-level stall):
    SU ──┬── PIPE_MTE2 ───╳ ALL STALLED
         ├── PIPE_V    ───╳ ALL STALLED
         └── PIPE_MTE3 ───╳ ALL STALLED

A5 wait_intra_core "PIPE_MTE2" (per-pipe stall):
    SU ──┬── PIPE_MTE2 ───╳ STALLED (waiting for Cube)
         ├── PIPE_V    ─── ✓ RUNNING
         └── PIPE_MTE3 ─── ✓ RUNNING
```
