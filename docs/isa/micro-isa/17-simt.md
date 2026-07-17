# 17. SIMT Ops

> **Category:** SIMT scalar execution, lane collectives, scalar memory, and
> memory-reduction operations
> **Pipeline:** Vector-side SIMT execution

SIMT ops are scalar operations executed by a group of workitems. A VPTO SIMT
program has an outer `pto.aicore` kernel that configures a VF subtask launch and
calls a SIMT body function marked with `pto.simt_entry`. The body is executed by
the logical workitems in the configured `dim_x * dim_y * dim_z` launch space.

---

## Common SIMT Execution Model

- The outer non-SIMT kernel configures launch dimensions with
  `pto.store_vfsimt_info`.
- The SIMT body is a normal `func.func` with the `pto.simt_entry` attribute.
- Each active workitem executes the same SIMT body with its own scalar SSA
  values, thread coordinates, lane id, and lane-mask state.
- SIMT scalar memory offsets are element offsets, not byte offsets.
- Vector-register ops such as `pto.vlds`, `pto.vadd`, and `pto.vsts` belong to
  normal vector code, not to the SIMT body.

Example SIMT body:

```mlir
func.func @body(%dst: !pto.ptr<i32, ub>) attributes {pto.simt_entry} {
  %tx = pto.get_tid_x : i32
  %idx = arith.index_castui %tx : i32 to index
  pto.store %tx, %dst[%idx] : !pto.ptr<i32, ub>, i32
  return
}
```

### Supported PTO SIMT Operation Surface

The current PTO SIMT surface supports these operation families:

| Family | Ops |
|--------|-----|
| Launch configuration | `pto.store_vfsimt_info`, `pto.simt_launch` |
| Thread and lane queries | `pto.get_tid_x`, `pto.get_tid_y`, `pto.get_tid_z`, `pto.get_block_dim_x`, `pto.get_block_dim_y`, `pto.get_block_dim_z`, `pto.get_grid_dim_x`, `pto.get_grid_dim_y`, `pto.get_grid_dim_z`, `pto.get_block_idx_x`, `pto.get_block_idx_y`, `pto.get_block_idx_z`, `pto.get_veccoreid`, `pto.get_clock32`, `pto.get_clock64`, `pto.get_laneid`, `pto.get_lanemask_eq`, `pto.get_lanemask_le`, `pto.get_lanemask_lt`, `pto.get_lanemask_ge`, `pto.get_lanemask_gt` |
| Lane collectives | `pto.vote_all`, `pto.vote_any`, `pto.vote_uni`, `pto.vote_ballot`, `pto.shuffle_idx`, `pto.shuffle_up`, `pto.shuffle_down`, `pto.shuffle_bfly`, `pto.redux_add`, `pto.redux_max`, `pto.redux_min` |
| Scalar memory | `pto.load`, `pto.store`, `pto.ldg`, `pto.stg` |
| Atomic memory | `pto.atomic_exch`, `pto.atomic_add`, `pto.atomic_sub`, `pto.atomic_min`, `pto.atomic_max`, `pto.atomic_and`, `pto.atomic_or`, `pto.atomic_xor`, `pto.atomic_cas` |
| Scalar math | `pto.prmt`, `pto.mulhi`, `pto.mul_i32toi64`, `pto.absf`, `pto.sqrt`, `pto.exp`, `pto.log`, `pto.pow`, `pto.ceil`, `pto.floor`, `pto.rint`, `pto.round`, `pto.fmin`, `pto.fmax`, `pto.fma` |
| Conversion | `pto.convert` |
| Entry synchronization and state | `pto.syncthreads`, `pto.threadfence`, `pto.threadfence_block`, `pto.keep`, `pto.resume` |

One optional function attribute may be attached to a `pto.simt_entry` function:

| Function attribute | Type | Default | Meaning |
|--------------------|------|---------|---------|
| `pto.simt_max_threads` | signless `i32` integer attribute | `1024` | Compile-time launch envelope. It should cover the largest `dim_x * dim_y * dim_z` launch count used for this entry. |

The attribute is optional. If present, it must be a positive `i32` attribute
and may only appear on functions that also carry `pto.simt_entry`. It does not
launch work by itself; the actual workitem count comes from
`pto.store_vfsimt_info` or `pto.simt_launch`.

The per-workitem register budget (`simt-max-registers`) is **automatically
derived** from `pto.simt_max_threads` according to the hardware resource
partition:

| `max_threads` | Derived `max_regs` |
|---|---|
| ≤ 256 | 128 |
| ≤ 512 | 64 |
| ≤ 1024 | 32 |
| ≤ 2048 | 16 |

The `pto.simt_max_regs` attribute is **deprecated** as a public interface.
Existing IR that carries it will still parse, but the emitter ignores its
value and always computes `simt-max-registers` from `pto.simt_max_threads`.

```mlir
func.func @body(%dst: !pto.ptr<i32, ub>)
    attributes {pto.simt_entry,
                pto.simt_max_threads = 256 : i32} {
  return
}
```

---

## Launch Configuration

### `pto.store_vfsimt_info`

- **syntax:** `pto.store_vfsimt_info %dim_z, %dim_y, %dim_x : i32, i32, i32`
- **semantics:** Configure the launch descriptor consumed by a subsequent SIMT
  entry call sequence in the current outer vector-side kernel.

```text
configured_dim_z = dim_z
configured_dim_y = dim_y
configured_dim_x = dim_x
logical_workitems = dim_x * dim_y * dim_z
call one or more simt_entry_body(...) functions
```

- **inputs:** `%dim_z`, `%dim_y`, and `%dim_x` are `i32` workitem counts in
  `z, y, x` order.
- **outputs:** None.
- **constraints and limitations:** This op belongs in the outer non-SIMT
  caller and must not appear inside a function marked with `pto.simt_entry`.
  SIMT entry calls that use the descriptor must be dominated by the matching
  launch configuration. On the current SIMT VF model, the launch count is
  bounded by 2048.
  If `pto.simt_max_threads` is present on the callee, it should be at least the
  largest launch count used for that callee.

Typical outer-kernel pattern:

```mlir
%dim_z = arith.constant 1 : i32
%dim_y = arith.constant 1 : i32
%dim_x = arith.constant 32 : i32
pto.store_vfsimt_info %dim_z, %dim_y, %dim_x : i32, i32, i32
func.call @body(%ub_out) : (!pto.ptr<i32, ub>) -> ()
```

### `pto.simt_launch`

- **syntax:** `pto.simt_launch @body<<<%dim_x, %dim_y, %dim_z>>>(%arg0, ...) : (arg_types...) -> ()`
- **semantics:** Launch the SIMT body `@body` using the workitem dimensions
  `%dim_x`, `%dim_y`, and `%dim_z`. The dimension order follows the launch-site
  order `x, y, z`; each active workitem in the body observes coordinates in the
  ranges `tid_x in [0, dim_x)`, `tid_y in [0, dim_y)`, and
  `tid_z in [0, dim_z)`.
- **inputs:** `%dim_x`, `%dim_y`, and `%dim_z` are `i32` workitem counts. The
  remaining operands are passed to the SIMT body and must match the callee
  function signature.
- **outputs:** None. The SIMT body must return no values.
- **constraints and limitations:** The callee must be a `func.func` marked with
  `pto.simt_entry`. The launch op belongs in the outer non-SIMT caller and must
  not appear inside a function marked with `pto.simt_entry`. The launch count is
  `dim_x * dim_y * dim_z` and is bounded by the same limits as
  `pto.store_vfsimt_info`.

Example launch-site pattern:

```mlir
%dim_x = arith.constant 32 : i32
%dim_y = arith.constant 1 : i32
%dim_z = arith.constant 1 : i32
pto.simt_launch @body<<<%dim_x, %dim_y, %dim_z>>>(%ub_out)
  : (!pto.ptr<i32, ub>) -> ()
```

---

## Thread and Lane Query Ops

Thread and lane query ops are nullary pure scalar ops. They return the value
visible to the current workitem.

### `pto.get_tid_x` / `pto.get_tid_y` / `pto.get_tid_z`

- **syntax:** `%tx = pto.get_tid_x : i32`
- **semantics:** Return the current workitem coordinate in the selected launch
  dimension.

```text
0 <= tid_x < dim_x
0 <= tid_y < dim_y
0 <= tid_z < dim_z
linear_tid = tid_x + dim_x * (tid_y + dim_y * tid_z)
```

- **inputs:** None.
- **outputs:** One `i32` coordinate.
- **constraints and limitations:** Use these coordinates for logical indexing.
  They are launch coordinates, not necessarily the same value as the physical
  lane id.

### `pto.get_block_dim_x` / `pto.get_block_dim_y` / `pto.get_block_dim_z`

- **syntax:** `%v = pto.get_block_dim_x : i32`
- **semantics:** Return the block dimension visible to the current workitem in
  the selected dimension.
- **inputs:** None.
- **outputs:** One `i32` block dimension.
- **constraints and limitations:** For single-block VF launches, block
  dimensions match the configured launch dimensions.

### `pto.get_grid_dim_x` / `pto.get_grid_dim_y` / `pto.get_grid_dim_z`

- **syntax:** `%v = pto.get_grid_dim_x : i32`
- **semantics:** Return the grid dimension visible to the current workitem in
  the selected dimension.
- **inputs:** None.
- **outputs:** One `i32` grid dimension.
- **constraints and limitations:** Use grid dimensions with block dimensions and
  block indices when deriving global workitem coordinates.

### `pto.get_block_idx_x` / `pto.get_block_idx_y` / `pto.get_block_idx_z`

- **syntax:** `%v = pto.get_block_idx_x : i32`
- **semantics:** Return the current block index in the selected dimension.
- **inputs:** None.
- **outputs:** One `i32` block index.
- **constraints and limitations:** For single-block VF launches, block indices
  are normally zero.

### `pto.get_veccoreid`

- **syntax:** `%core = pto.get_veccoreid : i32`
- **semantics:** Return the vector-core id visible to the current workitem.
- **inputs:** None.
- **outputs:** One `i32` vector-core id.
- **constraints and limitations:** The value is target scoped; use it only when
  the algorithm intentionally depends on the executing vector core.

### `pto.get_clock32` / `pto.get_clock64`

- **syntax:** `%c32 = pto.get_clock32 : i32`, `%c64 = pto.get_clock64 : i64`
- **semantics:** Sample the target clock counter visible to the current
  workitem.
- **inputs:** None.
- **outputs:** `pto.get_clock32` returns `i32`; `pto.get_clock64` returns `i64`.
- **constraints and limitations:** Use `get_clock64` when 32-bit wraparound
  could make elapsed-time comparisons ambiguous.

### `pto.get_laneid`

- **syntax:** `%lane = pto.get_laneid : i32`
- **semantics:** Return the physical SIMT lane id for the current workitem.
- **inputs:** None.
- **outputs:** One `i32` lane id.
- **constraints and limitations:** Use lane id for lane-mask, vote, shuffle,
  and reduction logic. Use `get_tid_x/y/z` for logical tensor indexing.

### `pto.get_lanemask_eq` / `pto.get_lanemask_le` / `pto.get_lanemask_lt` / `pto.get_lanemask_ge` / `pto.get_lanemask_gt`

- **syntax:** `%mask = pto.get_lanemask_lt : i32`
- **semantics:** Return a 32-bit mask derived from the current lane id.

```text
get_lanemask_eq = 1 << laneid
get_lanemask_lt = bits for lanes 0 .. laneid-1
get_lanemask_le = bits for lanes 0 .. laneid
get_lanemask_gt = bits for lanes laneid+1 .. subgroup_width-1
get_lanemask_ge = bits for lanes laneid .. subgroup_width-1
```

- **inputs:** None.
- **outputs:** One `i32` mask value.
- **constraints and limitations:** The mask is indexed by physical lane id.

---

## Vote Ops

Vote ops consume one `i1` predicate from each participating active lane and
return a collective result to each participating active lane.

### `pto.vote_all` / `pto.vote_any` / `pto.vote_uni` / `pto.vote_ballot`

- **syntax:**
```mlir
%all = pto.vote_all %pred : i1 -> i1
%any = pto.vote_any %pred : i1 -> i1
%uni = pto.vote_uni %pred : i1 -> i1
%bits = pto.vote_ballot %pred : i1 -> i32
```
- **semantics:**
```text
active = participating active lanes
vote_all    = forall lane in active: pred[lane]
vote_any    = exists lane in active: pred[lane]
vote_uni    = all pred[lane] values in active are equal
vote_ballot = bitset of lanes in active where pred[lane] is true
```
- **inputs:** `%pred` is the current lane's `i1` predicate.
- **outputs:** `vote_all`, `vote_any`, and `vote_uni` return `i1`.
  `vote_ballot` returns an `i32` lane bit mask.
- **constraints and limitations:** Inactive lanes do not contribute predicate
  values to the vote.

Example:

```mlir
%lane = pto.get_laneid : i32
%one = arith.constant 1 : i32
%low = arith.andi %lane, %one : i32
%is_odd = arith.cmpi eq, %low, %one : i32
%odd_mask = pto.vote_ballot %is_odd : i1 -> i32
```

---

## Shuffle Ops

Shuffle ops exchange values between participating lanes. The source value and
result have the same type.

### `pto.shuffle_idx`

- **syntax:** `%r = pto.shuffle_idx %value, %index {width = 16 : i32} : T, i32 -> T`
- **semantics:** Read `%value` from absolute `%index` inside the current
  subgroup.
- **inputs:** `%value` is the current lane's payload. `%index` is the
  source lane index inside the subgroup.
- **outputs:** `%r` is the selected source lane's value.
- **constraints and limitations:** `T` is `i32`, `i64`, `f16`, `f32`, or
  `vector<2xf16>`. `%index` is `i32`. `width` is `16` or `32` and defaults to
  `32`.

### `pto.shuffle_up` / `pto.shuffle_down` / `pto.shuffle_bfly`

- **syntax:**
```mlir
%u = pto.shuffle_up %value, %offset : T, i32 -> T
%d = pto.shuffle_down %value, %offset : T, i32 -> T
%b = pto.shuffle_bfly %value, %mask : T, i32 -> T
```
- **semantics:**
```text
group_base = floor(lane / width) * width
local_lane = lane - group_base
shuffle_up:   source = group_base + local_lane - offset
shuffle_down: source = group_base + local_lane + offset
shuffle_bfly: source = group_base + (local_lane xor mask)
result        = value[source] when source is a valid participating lane
```
- **inputs:** `%value` is the current lane's payload. `%offset` is the
  relative lane distance. `%mask` is the XOR mask for butterfly selection.
- **outputs:** The selected source lane's value.
- **constraints and limitations:** `T` is `i32`, `i64`, `f16`, `f32`, or
  `vector<2xf16>`. Control operands are `i32`. The optional `width` attribute is
  `16` or `32` and defaults to `32`. Out-of-range source-lane behavior is
  target-scoped and should not be used for portable algorithms.

---

## Lane Redux Ops

Redux ops reduce one scalar value from each participating active lane and
return the reduction result to each participating active lane.

### `pto.redux_add` / `pto.redux_max` / `pto.redux_min`

- **syntax:**
```mlir
%sum_i = pto.redux_add %v signed : i32 -> i32
%max_u = pto.redux_max %v unsigned : i32 -> i32
%sum_f = pto.redux_add %f : f32 -> f32
```
- **semantics:**
```text
redux_add = sum(value[lane] for lane in active)
redux_max = max(value[lane] for lane in active)
redux_min = min(value[lane] for lane in active)
result    = the selected reduction value in every participating active lane
```
- **inputs:** `%value` is `i32`, `f16`, or `f32`.
- **outputs:** The result type matches `%value`.
- **constraints and limitations:** Floating-point forms do not accept
  signedness. For `i32`, `pto.redux_max` and `pto.redux_min` require explicit
  `signed` or `unsigned`. `pto.redux_add` accepts signedness for consistency
  with integer authoring, but addition has the same two's-complement bit result
  for signed and unsigned inputs.

---

## Common SIMT Memory Attributes

**L1 cache.**

`l1cache(cache)` and `l1cache(uncache)` are accepted on GM scalar `pto.ldg` /
`pto.stg` forms.

| Attribute | Meaning |
|-----------|---------|
| `l1cache(cache)` | Request cacheable GM scalar access |
| `l1cache(uncache)` | Request uncacheable GM scalar access |

The L1 cache clause selects the GM access path. It does not change the scalar value
being loaded or stored.

**L2 cache.**

L2 cache clauses select the memory hierarchy behavior attached to GM `pto.ldg`,
GM `pto.stg`, and atomic ops. They do not select the
mathematical operation; the op mnemonic still determines the load, store,
or atomic update.

Load `l2cache(...)` uses these tokens:

| Token | Meaning |
|-------|---------|
| `nmfv` | Normal allocation, first-victim replacement priority |
| `nmlv` | Normal allocation, last-victim replacement priority |
| `nmprs` | Normal allocation, persistent cache residency hint |
| `nmpref` | Normal allocation, prefetch-oriented hint |
| `nakeep` | Not-allocate, keep existing cache line state |
| `naclean` | Not-allocate, clean cache line state |
| `nadrop` | Not-allocate, drop cache line state |
| `idsfv` | Inter-domain-share, first-victim replacement priority |
| `idslv` | Inter-domain-share, last-victim replacement priority |
| `idsprs` | Inter-domain-share, persistent cache residency hint |
| `idspref` | Inter-domain-share, prefetch-oriented hint |
| `exfv` | Exclusive, first-victim replacement priority |
| `exlv` | Exclusive, last-victim replacement priority |
| `exprs` | Exclusive, persistent cache residency hint |
| `expref` | Exclusive, prefetch-oriented hint |

Store and atomic `l2cache(...)` uses these tokens:

| Token | Meaning |
|-------|---------|
| `nmfv` | Normal allocation, first-victim replacement priority |
| `nmlv` | Normal allocation, last-victim replacement priority |
| `nmprs` | Normal allocation, persistent cache residency hint |
| `nmred` | Normal allocation, reduce-oriented update hint |
| `naci` | Not-allocate, clean-invalid cache line state |
| `napw` | Not-allocate, clean pre-writeback cache line state |
| `napi` | Not-allocate, pre-invalid cache line state |
| `nared` | Not-allocate, reduce-oriented update hint |
| `wbhfv` | Write-back-home, first-victim replacement priority |
| `wbhlv` | Write-back-home, last-victim replacement priority |
| `wbhprs` | Write-back-home, persistent cache residency hint |
| `wbhred` | Write-back-home, reduce-oriented update hint |
| `wtsfv` | Write-through-share, first-victim replacement priority |
| `wtslv` | Write-through-share, last-victim replacement priority |
| `wtsprs` | Write-through-share, persistent cache residency hint |
| `wtsred` | Write-through-share, reduce-oriented update hint |

For scalar GM `pto.ldg` / `pto.stg` and atomic syntax, write
`l2cache(...)`. Omitted `l2cache(...)` means `l2cache(nmfv)`. On `pto.ldg` /
`pto.stg`, omitted `l1cache(...)` means `l1cache(cache)`.

In syntax summaries, `<ld-l2cache>` means one token from the load L2 cache
table, `<st-l2cache>` means one token from the store/atomic L2 cache table,
`?` marks an optional clause, and `signedness?` means either
`signed`, `unsigned`, or no signedness clause.

---

## SIMT Scalar Memory Ops

### `pto.load`

- **syntax:** `%value = pto.load %ptr[%offset] : !pto.ptr<T, space> -> T`
- **accepted forms:**

```mlir
// Plain scalar load. Uses the ordinary scalar memory path.
%value = pto.load %ptr[%offset] : !pto.ptr<T, space> -> T
```

- **semantics:** Load one scalar element from `%ptr + %offset`.

```text
effective_element = ptr + offset
result = memory[effective_element]
```

- **inputs:** `%ptr` is a `!pto.ptr<T, space>` or memref. `%offset` is an
  `index` element offset, not a byte offset.
- **outputs:** One scalar value of type `T`.
- **constraints and limitations:** The result type must match the pointer
  element type. This op does not accept cache-control clauses; use `pto.ldg`
  for GM scalar loads that need `l1cache(...)` or `l2cache(...)`.

### `pto.store`

- **syntax:** `pto.store %value, %ptr[%offset] : !pto.ptr<T, space>, T`
- **accepted forms:**

```mlir
// Plain scalar store. Uses the ordinary scalar memory path.
pto.store %value, %ptr[%offset] : !pto.ptr<T, space>, T
```

- **semantics:** Store one scalar element to `%ptr + %offset`.

```text
effective_element = ptr + offset
memory[effective_element] = value
```

- **inputs:** `%value` is the scalar element to write. `%ptr` is a
  `!pto.ptr<T, space>` or memref. `%offset` is an `index` element offset.
- **outputs:** None.
- **constraints and limitations:** `%value` type must match the pointer element
  type. This op does not accept cache-control clauses; use `pto.stg` for GM
  scalar stores that need `l1cache(...)` or `l2cache(...)`.

### `pto.ldg`

- **syntax:** `%value = pto.ldg %ptr[%offset] l1cache(...)? l2cache(...)? attr-dict : !pto.ptr<T, gm> -> T`
- **accepted forms:**

```mlir
// GM load with default cache controls: l1cache(cache) and l2cache(nmfv).
%value = pto.ldg %gm[%offset] : !pto.ptr<T, gm> -> T

// GM load with an explicit L1 cache control.
%value = pto.ldg %gm[%offset] l1cache(cache) : !pto.ptr<T, gm> -> T

// GM load with explicit L1 and L2 cache controls.
%value = pto.ldg %gm[%offset] l1cache(uncache) l2cache(nmpref) : !pto.ptr<T, gm> -> T
```

- **semantics:** Load one element from GM at `%ptr + %offset` using the
  selected cache controls.
- **inputs:** `%ptr` is a `!pto.ptr<T, gm>`. `%offset` is an `index` element
  offset, not a byte offset.  For ``!pto.ptr<vector<2xf32>, gm>``, offset 1
  advances by 8 bytes (2 × sizeof(f32)).
- **attributes:** `l1cache` may be `l1cache(cache)` or `l1cache(uncache)` and
  defaults to `cache`. `l2cache(...)` uses the load L2 cache table and defaults
  to `nmfv`.
- **outputs:** One value of type `T`.
- **constraints and limitations:** `pto.ldg` supports 8/16/32/64-bit integer
  values, `f16`, `bf16`, `f32`, `f64`, `fp8`, `hif8`, and packed vectors
  `vector<2xf16>`, `vector<2xbf16>`, `vector<2xf32>`, `vector<2xf8E4M3FN>`,
  `vector<2xf8E5M2>`, `vector<2xi8>`, `vector<2xi16>`, `vector<2xi32>`,
  and `!pto.hif8x2`.  Vector loads from GM
  use the same-width load path as scalars (e.g. ``vector<2xf32>`` uses a 64-bit
  GM load) and reinterpret the loaded bits as the requested vector type.  The
  effective address for ``vector<2xf32>`` must satisfy 8-byte alignment
  (enforced by the call-site contract; the op does not carry an alignment
  operand).

### `pto.stg`

- **syntax:** `pto.stg %value, %ptr[%offset] l1cache(...)? l2cache(...)? attr-dict : !pto.ptr<T, gm>, T`
- **accepted forms:**

```mlir
// GM store with default cache controls: l1cache(cache) and l2cache(nmfv).
pto.stg %value, %gm[%offset] : !pto.ptr<T, gm>, T

// GM store with an explicit L1 cache control.
pto.stg %value, %gm[%offset] l1cache(cache) : !pto.ptr<T, gm>, T

// GM store with explicit L1 and L2 cache controls.
pto.stg %value, %gm[%offset] l1cache(uncache) l2cache(wtsred) : !pto.ptr<T, gm>, T
```

- **semantics:** Store one element to GM at `%ptr + %offset` using the
  selected cache controls.
- **inputs:** `%value` is the element to write. `%ptr` is a
  `!pto.ptr<T, gm>`. `%offset` is an `index` element offset
  (same element-level semantics as `pto.ldg`).
- **attributes:** `l1cache` may be `l1cache(cache)` or `l1cache(uncache)` and
  defaults to `cache`. `l2cache(...)` uses the store/atomic L2 cache table and
  defaults to `nmfv`.
- **outputs:** None.
- **constraints and limitations:** `%value` type must match the pointer element
  type.  Supported types and alignment requirements are the same as `pto.ldg`
  (see above).

Example:

```mlir
%tx = pto.get_tid_x : i32
%idx = arith.index_castui %tx : i32 to index
%loaded = pto.load %gm[%idx] : !pto.ptr<i32, gm> -> i32
%sum = arith.addi %loaded, %tx : i32
pto.store %sum, %gm[%idx] : !pto.ptr<i32, gm>, i32
```

---

## Atomic Memory Ops

Atomic ops update one scalar or supported packed memory location and return the
old value observed by the current workitem. The read, update, and returned old
value form one atomic read-modify-write at `%ptr`.

### `pto.atomic_exch` / `pto.atomic_add` / `pto.atomic_sub`

- **syntax:**
```mlir
%old = pto.atomic_exch %ptr, %value l2cache(<st-l2cache>)? signedness? : !pto.ptr<T, space>, T -> T
%old = pto.atomic_add  %ptr, %value l2cache(<st-l2cache>)? signedness? : !pto.ptr<T, space>, T -> T
%old = pto.atomic_sub  %ptr, %value l2cache(<st-l2cache>)? signedness? : !pto.ptr<T, space>, T -> T
```
- **accepted forms:**

```mlir
// Signed integer atomic with default nmfv L2 cache.
%old = pto.atomic_add %ptr, %value signed : !pto.ptr<i32, space>, i32 -> i32

// Signed integer atomic with an explicit store/atomic L2 cache.
%old = pto.atomic_add %ptr, %value l2cache(wtsred) signed : !pto.ptr<i32, space>, i32 -> i32

// Floating-point atomic. Floating-point atomics do not take signedness.
%old = pto.atomic_add %ptr, %value l2cache(nmfv) : !pto.ptr<f32, space>, f32 -> f32

// Packed two-lane atomic. Packed atomics do not take signedness.
%old = pto.atomic_add %ptr, %value : !pto.ptr<vector<2xf16>, space>, vector<2xf16> -> vector<2xf16>
```
- **semantics:**
```text
old = *ptr
atomic_exch: *ptr = value
atomic_add:  *ptr = old + value
atomic_sub:  *ptr = old - value
return old
```
- **inputs:** `%ptr` is `!pto.ptr<T, gm>` or `!pto.ptr<T, ub>`. `%value` is
  `i32`, `i64`, `f16`, `bf16`, `f32`, `vector<2xf16>`, or
  `vector<2xbf16>`.
  `l2cache(...)` selects the store/atomic L2 cache control and defaults to
  `nmfv`.
- **outputs:** `%old` has the same type as `%value`. For packed
  `vector<2xf16>` and `vector<2xbf16>` atomics on beta.1, `%old` must be left
  unused; beta.1 can compile the packed atomic update but cannot compile a
  consumed packed old-value result.
- **constraints and limitations:** UB-space atomics do not support `i64`.
  Floating-point and packed atomics do not accept `signed` or `unsigned`.
  Packed atomics must be placed inside a `pto.simt_entry` function on beta.1.

### `pto.atomic_min` / `pto.atomic_max`

- **syntax:**
```mlir
%old = pto.atomic_min %ptr, %value l2cache(<st-l2cache>)? signedness? : !pto.ptr<T, space>, T -> T
%old = pto.atomic_max %ptr, %value l2cache(<st-l2cache>)? signedness? : !pto.ptr<T, space>, T -> T
```
- **accepted forms:**

```mlir
// Signed integer comparison.
%old = pto.atomic_min %ptr, %value signed : !pto.ptr<i32, space>, i32 -> i32

// Unsigned integer comparison.
%old = pto.atomic_min %ptr, %value unsigned : !pto.ptr<i32, space>, i32 -> i32

// Floating-point comparison. Floating-point atomics do not take signedness.
%old = pto.atomic_min %ptr, %value l2cache(nmlv) : !pto.ptr<f32, space>, f32 -> f32

// Packed two-lane comparison.
%old = pto.atomic_min %ptr, %value : !pto.ptr<vector<2xbf16>, space>, vector<2xbf16> -> vector<2xbf16>
```
- **semantics:**
```text
old = *ptr
atomic_min: *ptr = min(old, value)
atomic_max: *ptr = max(old, value)
return old
```
- **inputs:** Same as `pto.atomic_add`. For integer values, `signed` or
  `unsigned` selects the comparison interpretation.
- **outputs:** `%old` has the same type as `%value`. For packed
  `vector<2xf16>` and `vector<2xbf16>` atomics on beta.1, `%old` must be left
  unused; beta.1 can compile the packed atomic update but cannot compile a
  consumed packed old-value result.
- **constraints and limitations:** Floating-point and packed atomics do not
  accept signedness. UB-space atomics do not support `i64`. Packed atomics
  must be placed inside a `pto.simt_entry` function on beta.1.

### `pto.atomic_and` / `pto.atomic_or` / `pto.atomic_xor`

- **syntax:**
```mlir
%old = pto.atomic_and %ptr, %value l2cache(<st-l2cache>)? signedness? : !pto.ptr<T, space>, T -> T
%old = pto.atomic_or  %ptr, %value l2cache(<st-l2cache>)? signedness? : !pto.ptr<T, space>, T -> T
%old = pto.atomic_xor %ptr, %value l2cache(<st-l2cache>)? signedness? : !pto.ptr<T, space>, T -> T
```
- **accepted forms:**

```mlir
// Unsigned bitwise atomic with default nmfv L2 cache.
%old = pto.atomic_and %ptr, %value unsigned : !pto.ptr<i32, space>, i32 -> i32

// Signedness is accepted for integer authoring consistency; the bit operation
// itself is bitwise and does not reinterpret arithmetic magnitude.
%old = pto.atomic_and %ptr, %value l2cache(napw) signed : !pto.ptr<i32, space>, i32 -> i32
```
- **semantics:**
```text
old = *ptr
atomic_and: *ptr = old & value
atomic_or:  *ptr = old | value
atomic_xor: *ptr = old ^ value
return old
```
- **inputs:** `%ptr` points to an integer scalar element. `%value` is `i32` or
  `i64`.
- **outputs:** `%old` has the same type as `%value`.
- **constraints and limitations:** Bitwise atomics require integer types.
  UB-space bitwise atomics do not support `i64`.

### `pto.atomic_cas`

- **syntax:** `%old = pto.atomic_cas %ptr, %compare, %value l2cache(<st-l2cache>)? signedness? : !pto.ptr<T, space>, T -> T`
- **accepted forms:**

```mlir
// Integer CAS with default nmfv L2 cache.
%old = pto.atomic_cas %ptr, %compare, %value signed : !pto.ptr<i32, space>, i32 -> i32

// Integer CAS with an explicit store/atomic L2 cache.
%old = pto.atomic_cas %ptr, %compare, %value l2cache(wbhred) signed : !pto.ptr<i32, space>, i32 -> i32

// Floating-point CAS. Floating-point atomics do not take signedness.
%old = pto.atomic_cas %ptr, %compare, %value : !pto.ptr<f32, space>, f32 -> f32

// Packed two-lane CAS. Packed atomics do not take signedness.
%old = pto.atomic_cas %ptr, %compare, %value : !pto.ptr<vector<2xbf16>, space>, vector<2xbf16> -> vector<2xbf16>
```

- **semantics:**
```text
old = *ptr
if old == compare:
  *ptr = value
return old
```
- **inputs:** `%ptr` is the atomic address. `%compare` is the expected old
  value. `%value` is the replacement value.
- **outputs:** `%old` is the value observed before the conditional update. For
  packed `vector<2xf16>` and `vector<2xbf16>` CAS on beta.1, `%old` must be
  left unused; beta.1 can compile the packed CAS update but cannot compile a
  consumed packed old-value result.
- **constraints and limitations:** `%compare`, `%value`, pointer element type,
  and result type must match. `T` is `i32`, `i64`, `f32`,
  `vector<2xf16>`, or `vector<2xbf16>`; UB-space `i64` is not supported.
  Packed CAS must be placed inside a `pto.simt_entry` function on beta.1.

When multiple workitems target the same address, each workitem observes one
serialized old value from the total order chosen by the target. Algorithms must
not rely on any particular tie order beyond atomicity.

---

## SIMT Scalar Math Ops

### `pto.prmt`

- **syntax:** `%r = pto.prmt %lhs, %rhs, %selector : i32, i32, i32 -> i32`
- **semantics:** Build the `i32` result byte-by-byte from the eight source bytes
  in `%lhs:%rhs` according to `%selector`.
- **inputs:** `%lhs` and `%rhs` provide the source bytes. `%selector` selects
  which source byte is copied into each destination byte.
- **outputs:** One `i32` result.
- **constraints and limitations:** All operands and the result are `i32`.

### `pto.mulhi`

- **syntax:**
```mlir
%s32 = pto.mulhi %lhs, %rhs signed : i32, i32 -> i32
%u32 = pto.mulhi %lhs, %rhs unsigned : i32, i32 -> i32
%s64 = pto.mulhi %lhs64, %rhs64 signed : i64, i64 -> i64
%u64 = pto.mulhi %lhs64, %rhs64 unsigned : i64, i64 -> i64
```
- **semantics:**
```text
N = bitwidth(lhs)
if signed:
  product = signed_N(lhs) * signed_N(rhs)
else:
  product = unsigned_N(lhs) * unsigned_N(rhs)
result = high_N_bits(product)
```
- **inputs:** `%lhs` and `%rhs` are scalar integer operands with the same type.
- **outputs:** One scalar integer result with the same type as the inputs.
- **attributes:** The required `signed` or `unsigned` clause selects whether
  the operands are interpreted as signed two's-complement values or unsigned
  values before forming the double-width product.
- **constraints and limitations:** The operands and result must all be `i32` or
  all be `i64`.

### `pto.mul_i32toi64`

- **syntax:**
```mlir
%s = pto.mul_i32toi64 %lhs, %rhs signed : i32, i32 -> i64
%u = pto.mul_i32toi64 %lhs, %rhs unsigned : i32, i32 -> i64
```
- **semantics:**
```text
if signed:
  result = sign_extend_i64(lhs) * sign_extend_i64(rhs)
else:
  result = zero_extend_i64(lhs) * zero_extend_i64(rhs)
```
- **inputs:** `%lhs` and `%rhs` are `i32` scalar operands.
- **outputs:** One `i64` widened-product result.
- **attributes:** The required `signed` or `unsigned` clause selects the
  extension rule before multiplication.
- **constraints and limitations:** The operand types are fixed to `i32`, and
  the result type is fixed to `i64`.

### `pto.absf`

- **syntax:** `%r = pto.absf %x : T -> T`
- **semantics:** Return `abs(x)`. For `vector<2xT>`, absolute value is applied
  independently to each element.
- **inputs:** `%x` is an `f32` scalar, `vector<2xf16>`, or `vector<2xbf16>`.
- **outputs:** One value with the same type as `%x`.
- **constraints and limitations:** Scalar `f16` and scalar `bf16` are not
  accepted by this op; use the packed form only for `vector<2xT>`.

### `pto.sqrt`

- **syntax:** `%r = pto.sqrt %x : T -> T`
- **semantics:** Return `sqrt(x)`. For `vector<2xT>`, square root is applied
  independently to each element.
- **inputs:** `%x` is `f16`, `f32`, or `vector<2xf16>`.
- **outputs:** One value with the same type as `%x`.
- **constraints and limitations:** `T` is `f16`, `f32`, or `vector<2xf16>`.

### `pto.exp`

- **syntax:** `%r = pto.exp %x : T -> T`
- **semantics:** Return the natural exponential `e ** x`. For `vector<2xT>`,
  exponentiation is applied independently to each element.
- **inputs:** `%x` is an `f16` scalar, `f32` scalar, or `vector<2xf16>`.
- **outputs:** One value with the same type as `%x`.
- **constraints and limitations:** `T` is `f16`, `f32`, or `vector<2xf16>`.
  Overflow, underflow, infinities, and NaNs follow the target floating-point
  rules.

### `pto.log`

- **syntax:** `%r = pto.log %x : T -> T`
- **semantics:** Return the natural logarithm `ln(x)`. For `vector<2xT>`,
  logarithm is applied independently to each element.
- **inputs:** `%x` is an `f16` scalar, `f32` scalar, or `vector<2xf16>`.
- **outputs:** One value with the same type as `%x`.
- **constraints and limitations:** `T` is `f16`, `f32`, or `vector<2xf16>`.
  For real-number semantics, each element should be positive; non-positive
  inputs follow the target floating-point rules.

### `pto.pow`

- **syntax:** `%r = pto.pow %a, %b : T, T -> T`
- **semantics:** Return `%a ** %b`. For `vector<2xT>`, power is applied
  independently to each element pair.
- **inputs:** `%a` is the base and `%b` is the exponent. Both operands have the
  same type.
- **outputs:** One value with the same type as the inputs.
- **constraints and limitations:** `T` is `f16`, `f32`, or `vector<2xf16>`.
  Exceptional inputs follow the target floating-point rules.

### `pto.ceil`

- **syntax:** `%r = pto.ceil %x : T -> T`
- **semantics:** Return the smallest integral floating value not less than
  `%x`.
- **inputs:** `%x` is an `f16`, `bf16`, or `f32` scalar.
- **outputs:** One scalar with the same type as `%x`.
- **constraints and limitations:** `T` is `f16`, `bf16`, or `f32`.

### `pto.floor`

- **syntax:** `%r = pto.floor %x : T -> T`
- **semantics:** Return the largest integral floating value not greater than
  `%x`.
- **inputs:** `%x` is an `f16`, `bf16`, or `f32` scalar.
- **outputs:** One scalar with the same type as `%x`.
- **constraints and limitations:** `T` is `f16`, `bf16`, or `f32`.

### `pto.rint`

- **syntax:** `%r = pto.rint %x : T -> T`
- **semantics:** Return the integral floating value selected by the target's
  current floating rounding rule.
- **inputs:** `%x` is an `f16`, `bf16`, or `f32` scalar.
- **outputs:** One scalar with the same type as `%x`.
- **constraints and limitations:** `T` is `f16`, `bf16`, or `f32`.

### `pto.round`

- **syntax:** `%r = pto.round %x : T -> T`
- **semantics:** Return the nearest integral floating value using the target
  round operation's tie rule.
- **inputs:** `%x` is an `f16`, `bf16`, or `f32` scalar.
- **outputs:** One scalar with the same type as `%x`.
- **constraints and limitations:** `T` is `f16`, `bf16`, or `f32`.

### `pto.fmin`

- **syntax:** `%r = pto.fmin %a, %b : T, T -> T`
- **semantics:** Return the floating minimum of `%a` and `%b`.
- **inputs:** `%a` and `%b` have the same type.
- **outputs:** One value with the same type as the inputs.
- **constraints and limitations:** `T` is `f32`, `bf16`, `vector<2xf16>`, or
  `vector<2xbf16>`. For vector types, the minimum is computed element-wise. NaN
  handling follows the target floating-point minimum rule.

### `pto.fmax`

- **syntax:** `%r = pto.fmax %a, %b : T, T -> T`
- **semantics:** Return the floating maximum of `%a` and `%b`.
- **inputs:** `%a` and `%b` have the same type.
- **outputs:** One value with the same type as the inputs.
- **constraints and limitations:** `T` is `f32`, `bf16`, `vector<2xf16>`, or
  `vector<2xbf16>`. For vector types, the maximum is computed element-wise. NaN
  handling follows the target floating-point maximum rule.

### `pto.fma`

- **syntax:** `%r = pto.fma %a, %b, %acc : T, T, T -> T`
- **semantics:** Return fused `a * b + acc` with one final rounding.
- **inputs:** `%a`, `%b`, and `%acc` have the same type.
- **outputs:** One value with the same type as the inputs.
- **constraints and limitations:** `T` is `f16`, `bf16`, `f32`,
  `vector<2xf16>`, or `vector<2xbf16>`. For vector types, fused multiply-add is
  computed element-wise.

---

## SIMT Conversion Op

### `pto.convert`

- **syntax:** `%dst = pto.convert %src round(R) sat|nosat [signed|unsigned] : SrcType -> DstType`
- **semantics:** Convert one scalar or packed two-element value from `SrcType` to
  `DstType` using the specified rounding, saturation, and signedness controls.

```mlir
%as_f32 = pto.convert %i round(r) nosat signed : i32 -> f32
%as_i32 = pto.convert %f round(z) sat signed : f32 -> i32
%as_h2 = pto.convert %f2 round(r) nosat : vector<2xf32> -> vector<2xf16>
```

- **inputs:** `%src` is `i32`, `i64`, `f16`, `bf16`, `f32`,
  `vector<2xf16>`, `vector<2xbf16>`, or `vector<2xf32>`.
  `round(R)` selects the rounding rule. `sat` or `nosat` selects whether
  finite overflow is clamped to the destination range. `signed` or `unsigned`
  is required when converting to or from an integer type and is omitted for
  floating-to-floating and packed vector conversion.
- **outputs:** `%dst` is `i32`, `i64`, `f16`, `bf16`, `f32`,
  `vector<2xf16>`, `vector<2xbf16>`, or `vector<2xf32>`.
- **constraints and limitations:** Integer-to-integer conversion is not
  supported by `pto.convert`. Scalar floating-to-floating conversion supports
  `f32`, `f16`, and `bf16` source/destination pairs. `i64` source conversion is
  supported only to `f32`; conversion to `i64` is supported only from `f32`.
  `i32` can convert to `f32`, `f16`, or `bf16`, with `signed` or `unsigned`
  selecting the source interpretation. Floating-to-integer conversion supports
  `i32` destinations, plus `f32 -> i64`, and requires `sat`. Packed conversion
  supports only
  `vector<2xf32> -> vector<2xf16>`, `vector<2xf16> -> vector<2xf32>`,
  `vector<2xf32> -> vector<2xbf16>`, and
  `vector<2xbf16> -> vector<2xf32>`.

Rounding selectors:

| Selector | Meaning |
|----------|---------|
| `round(r)` | Round to nearest, ties to even |
| `round(a)` | Round away from zero |
| `round(f)` | Round toward minus infinity |
| `round(c)` | Round toward plus infinity |
| `round(z)` | Round toward zero |
| `round(o)` | Round to odd |
| `round(h)` | Cast-ceil mode for the target conversion slice that supports it |

Saturation selectors:

| Selector | Meaning |
|----------|---------|
| `nosat` | Do not clamp finite overflow to the destination range |
| `sat` | Clamp finite overflow to the destination range |

---

## SIMT Entry Synchronization and State Ops

### `pto.syncthreads`

- **syntax:** `pto.syncthreads attr-dict`
- **semantics:** Synchronize all active workitems in the current SIMT entry.
  Memory effects issued before the barrier by participating workitems are
  ordered before memory effects issued after the barrier by those workitems.
- **inputs:** None.
- **outputs:** None.
- **constraints and limitations:** `pto.syncthreads` must appear inside a
  function marked with `pto.simt_entry`. It synchronizes workitems in the
  active SIMT entry; it is not a substitute for outer pipeline synchronization
  between vector, MTE, cube, and scalar host-visible effects.

Example:

```mlir
func.func @body(%ub: !pto.ptr<i32, ub>) attributes {pto.simt_entry} {
  %tx = pto.get_tid_x : i32
  %idx = arith.index_castui %tx : i32 to index
  pto.store %tx, %ub[%idx] : !pto.ptr<i32, ub>, i32
  pto.syncthreads
  %v = pto.load %ub[%idx] : !pto.ptr<i32, ub> -> i32
  pto.store %v, %ub[%idx] : !pto.ptr<i32, ub>, i32
  return
}
```

### `pto.threadfence` / `pto.threadfence_block`

- **syntax:** `pto.threadfence attr-dict` or `pto.threadfence_block attr-dict`
- **semantics:** Issue a memory fence for memory effects from the current SIMT
  workitem. `pto.threadfence` uses the target workitem fence operation;
  `pto.threadfence_block` uses the target block-scoped workitem fence
  operation.
- **inputs:** None.
- **outputs:** None.
- **constraints and limitations:** These ops must appear inside a function
  marked with `pto.simt_entry`. They order memory effects but do not by
  themselves make other workitems wait; use `pto.syncthreads` when a workitem
  barrier is required.

### `pto.keep` / `pto.resume`

- **syntax:**

```mlir
pto.keep %value {slot = N : i64} : T
%value = pto.resume {slot = N : i64} : T
```

- **semantics:** Preserve and restore one per-workitem scalar payload across
  adjacent SIMT entry calls in the same outer launch sequence. `pto.keep`
  records the current workitem's `%value` in logical slot `N`; `pto.resume`
  restores the value for the same logical workitem from logical slot `N`.

```text
for each active workitem:
  keep(slot, value) stores value in that workitem's slot
  resume(slot) returns the value stored in the same workitem's slot
```

- **inputs:** `pto.keep` takes one scalar `%value` of type `T`.
- **outputs:** `pto.resume` returns one scalar value of type `T`.
- **attributes:** `slot` is a non-negative `i64` logical slot identifier in
  the range `[0, 122]`.
- **supported types:** `T` may be any signless integer scalar with bit width up
  to 64 bits, `f16`, `bf16`, or `f32`.
- **constraints and limitations:** Both ops must appear inside functions marked
  with `pto.simt_entry`. A `pto.resume` group must be the first non-constant
  operation group in its SIMT entry. A `pto.keep` group must be the final
  operation group before optional `pto.syncthreads` and `func.return`. Slot
  storage words must not overlap within one `pto.resume` group or one
  `pto.keep` group. A value resumed from a slot should use the same type as the
  value kept into that slot.
- **slot allocation rule:** Users allocate slots explicitly. Values up to 32
  bits and supported floating-point values consume only `slot`. A 64-bit
  integer value consumes `slot` and `slot + 1`; therefore its slot must be even
  and must leave room for the second word. Because slot mapping is explicit, a
  later SIMT entry may resume only the subset of preserved slots that it needs
  without changing the location of those slots.

Example:

```mlir
func.func @stage0(%dst: !pto.ptr<i32, ub>) attributes {pto.simt_entry} {
  %tx = pto.get_tid_x : i32
  %idx = arith.index_castui %tx : i32 to index
  pto.store %tx, %dst[%idx] : !pto.ptr<i32, ub>, i32
  pto.keep %tx {slot = 0 : i64} : i32
  pto.syncthreads
  return
}

func.func @stage1(%dst: !pto.ptr<i32, ub>) attributes {pto.simt_entry} {
  %tx0 = pto.resume {slot = 0 : i64} : i32
  %tx = pto.get_tid_x : i32
  %idx = arith.index_castui %tx : i32 to index
  %sum = arith.addi %tx0, %tx : i32
  pto.store %sum, %dst[%idx] : !pto.ptr<i32, ub>, i32
  return
}
```

---

## Outer Pipeline Synchronization and Ordering

SIMT body execution is sequenced as a function call from the outer kernel. Use
the existing PTO pipeline synchronization ops around the SIMT call when data is
produced or consumed by other pipelines.

```mlir
pto.store_vfsimt_info %dim_z, %dim_y, %dim_x : i32, i32, i32
func.call @body(%ub_out) : (!pto.ptr<i32, ub>) -> ()
pto.set_flag["PIPE_V", "PIPE_MTE3", "EVENT_ID0"]
pto.wait_flag["PIPE_V", "PIPE_MTE3", "EVENT_ID0"]
pto.mte_ub_gm %ub_out, %gm_out, %len
  nburst(%n, %src_stride, %dst_stride)
  : !pto.ptr<i32, ub>, !pto.ptr<i32, gm>, i64, i64, i64, i64
```

For pipeline synchronization semantics, see
[`01-pipeline-sync.md`](01-pipeline-sync.md). Do not use pipeline barriers as a
substitute for lane collectives: vote, shuffle, redux, and atomic ops are the
SIMT-specific mechanisms documented in this chapter.

---

## Complete Minimal Example

```mlir
module attributes {pto.target_arch = "a5",
                   pto.kernel_kind = #pto.kernel_kind<vector>} {
  func.func @simt_store_tid_kernel(%out: !pto.ptr<i32, gm>)
      attributes {pto.aicore} {
    %c0_i64 = arith.constant 0 : i64
    %c32_i64 = arith.constant 32 : i64
    %c128_i64 = arith.constant 128 : i64
    %dim_z = arith.constant 1 : i32
    %dim_y = arith.constant 1 : i32
    %dim_x = arith.constant 32 : i32

    %ub_out = pto.castptr %c0_i64 : i64 -> !pto.ptr<i32, ub>
    pto.store_vfsimt_info %dim_z, %dim_y, %dim_x : i32, i32, i32
    func.call @simt_write(%ub_out) : (!pto.ptr<i32, ub>) -> ()

    pto.set_flag["PIPE_V", "PIPE_MTE3", "EVENT_ID0"]
    pto.wait_flag["PIPE_V", "PIPE_MTE3", "EVENT_ID0"]
    pto.mte_ub_gm %ub_out, %out, %c128_i64
      nburst(%c32_i64, %c128_i64, %c128_i64)
      : !pto.ptr<i32, ub>, !pto.ptr<i32, gm>, i64, i64, i64, i64
    pto.barrier #pto.pipe<PIPE_ALL>
    return
  }

  func.func @simt_write(%dst: !pto.ptr<i32, ub>)
      attributes {pto.simt_entry} {
    %tx = pto.get_tid_x : i32
    %ty = pto.get_tid_y : i32
    %tz = pto.get_tid_z : i32
    %c8_i32 = arith.constant 8 : i32
    %c16_i32 = arith.constant 16 : i32
    %c32_i32 = arith.constant 32 : i32
    %ty_shift = arith.shli %ty, %c8_i32 : i32
    %tz_shift = arith.shli %tz, %c16_i32 : i32
    %xy = arith.ori %tx, %ty_shift : i32
    %xyz = arith.ori %xy, %tz_shift : i32
    %lane_base = arith.muli %ty, %c32_i32 : i32
    %idx_i32 = arith.addi %lane_base, %tx : i32
    %idx = arith.index_castui %idx_i32 : i32 to index
    pto.store %xyz, %dst[%idx] : !pto.ptr<i32, ub>, i32
    return
  }
}
```
