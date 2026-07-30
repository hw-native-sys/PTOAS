# 2. DMA Copy Programming

> **Category:** DMA transfer configuration and execution
> **Pipelines:** MTE2 (GM→UB), MTE3 (UB→GM)

DMA transfers move data between Global Memory (GM) and Unified Buffer (UB). The MTE engines operate asynchronously from the Vector core, requiring explicit sync (see [Pipeline Sync](01-pipeline-sync.md)).

This document describes the public grouped DMA interfaces:

- `pto.mte_gm_ub`
- `pto.mte_ub_gm`
- `pto.mte_ub_ub`
- `pto.mte_ub_l1`

`pto.mte_gm_ub` / `pto.mte_ub_gm` also accept remote GM pointers (`#pto.remote`)
for synchronous cross-rank copies. There is no `mte_gm_gm`.

---

## DMA Transfer Execution

### `pto.mte_gm_ub`

- **syntax:**
```mlir
pto.mte_gm_ub %gm_src, %ub_dst, %l2_cache_ctl, %len_burst
  nburst(%n_burst, %src_stride, %dst_stride)
  [loop(%loop_count, %loop_src_stride, %loop_dst_stride)]*
  [pad(%pad_value[, %left_padding_count, %right_padding_count])]
  : !pto.ptr<T, gm>, !pto.ptr<T, ub>, i64, i64, i64,
    [loop i64, i64, i64,]*
    [pad T[, i64, i64]]
```
- **semantics:** Grouped GM→UB DMA transfer. `nburst(...)` defines the innermost repeated burst transfer, optional `loop(...)` groups add outer repetition levels, and `pad(...)` controls UB row padding.
  `%gm_src` may carry `#pto.remote` for a synchronous remote read; nburst / stride / pad are unchanged.

**Parameter Table:**

| Parameter | Width | Description |
|-----------|-------|-------------|
| `%gm_src` | ptr | GM source pointer (`!pto.ptr<T, gm>` or with `#pto.remote`) |
| `%ub_dst` | ptr | UB destination pointer (`!pto.ptr<T, ub>`, 32B-aligned) |
| `%l2_cache_ctl` | 2 bits | L2 cache allocate control |
| `%len_burst` | 16 bits | Contiguous bytes transferred per burst row |
| `nburst(%n_burst, %src_stride, %dst_stride)` | 16 bits / 40 bits / 21 bits | Required innermost burst group: count, GM source stride, UB destination stride |
| `loop(%loop_count, %loop_src_stride, %loop_dst_stride)` | 21 bits / 40 bits / 21 bits | Optional outer repetition group: count, GM source stride, UB destination stride |
| `pad(%pad_value[, %left_padding_count, %right_padding_count])` | scalar / 8 bits / 8 bits | Optional padding: fill value, optional left padding count, optional right padding count |

**Constraints:**

- `nburst(...)` is always required.
- Each `loop(...)` group must be provided as a complete triple when present.
- `nburst(...)` is the innermost group.
- `loop(...)` groups are ordered from inner to outer.
- The first `loop(...)` group wraps `nburst(...)`.
- Each additional `loop(...)` group wraps all earlier groups.
- `pad(...)` may contain only `%pad_value`; omitted left and right padding counts default to 0.
- If either left or right padding count is provided, both counts must be provided.
- `pad(...)` is independent of the optional `loop(...)` groups.
- A DMA load may use `nburst(...) pad(...)` without any `loop(...)` group.
- Remote `%gm_src` must carry `#pto.remote` (caller computes address from
  `CommDeviceContext.windowsIn[]`); no session or completion-record result.
- Local completion follows existing MTE / pipe sync. Cross-rank visibility (E3) is not implied.

**Example:**

```mlir
pto.mte_gm_ub %gm_in, %ub_out, %cache, %len_burst
  nburst(%rows, %gm_row_stride, %ub_row_stride)
  loop(%tiles, %gm_tile_stride, %ub_tile_stride)
  pad(%pad)
  : !pto.ptr<f16, gm>, !pto.ptr<f16, ub>, i64, i64,
    loop i64, i64, i64, pad f16
```

**Example (remote GM → UB):**

```mlir
%src = pto.castptr %remote_i64 : i64 -> !pto.ptr<f16, gm, #pto.remote>
pto.mte_gm_ub %src, %ub_out, %cache, %len_burst
  nburst(%rows, %gm_row_stride, %ub_row_stride)
  : !pto.ptr<f16, gm, #pto.remote>, !pto.ptr<f16, ub>, i64, i64, i64
```

---

### `pto.mte_ub_gm`

- **syntax:**
```mlir
pto.mte_ub_gm %ub_src, %gm_dst, %len_burst
  nburst(%n_burst, %src_stride, %dst_stride) l2_cache_ctl(%l2_cache_ctl)
  [loop(%loop_count, %loop_src_stride, %loop_dst_stride)]*
  : !pto.ptr<T, ub>, !pto.ptr<T, gm>, i64, i64, i64, i64, i64,
    [loop i64, i64, i64,]*
```
- **semantics:** Grouped UB→GM DMA transfer. `nburst(...)` defines the innermost repeated burst transfer, and optional `loop(...)` groups add outer repetition levels.
  The `l2_cache_ctl(...)` group is optional in textual VPTO IR; when omitted, lowering uses `0`.
  `%gm_dst` may carry `#pto.remote` for a synchronous remote write; nburst / stride are unchanged.

**Parameter Table:**

| Parameter | Width | Description |
|-----------|-------|-------------|
| `%ub_src` | ptr | UB source pointer (`!pto.ptr<T, ub>`, 32B-aligned) |
| `%gm_dst` | ptr | GM destination pointer (`!pto.ptr<T, gm>` or with `#pto.remote`) |
| `%len_burst` | 16 bits | Contiguous bytes transferred per burst row |
| `nburst(%n_burst, %src_stride, %dst_stride)` | 16 bits / 21 bits / 40 bits | Required innermost burst group: count, UB source stride, GM destination stride |
| `l2_cache_ctl(%l2_cache_ctl)` | 4 bits | Optional GM store-side L2 cache control; omitted means `0` |
| `loop(%loop_count, %loop_src_stride, %loop_dst_stride)` | 21 bits / 21 bits / 40 bits | Optional outer repetition group: count, UB source stride, GM destination stride |

**Constraints:**

- `nburst(...)` is always required.
- Each `loop(...)` group must be provided as a complete triple when present.
- `nburst(...)` is the innermost group.
- `loop(...)` groups are ordered from inner to outer.
- The first `loop(...)` group wraps `nburst(...)`.
- Each additional `loop(...)` group wraps all earlier groups.
- Remote `%gm_dst` must carry `#pto.remote` (caller computes address from
  `CommDeviceContext.windowsIn[]`); no session or completion-record result.
- Local completion follows existing MTE / pipe sync. Cross-rank visibility (E3) is not implied. No `mte_*_signal` / `mte_*_counter` fused form.

**Example:**

```mlir
pto.mte_ub_gm %ub_in, %gm_out, %len_burst
  nburst(%rows, %ub_row_stride, %gm_row_stride) l2_cache_ctl(%l2_cache_ctl)
  loop(%tiles, %ub_tile_stride, %gm_tile_stride)
  loop(%batches, %ub_batch_stride, %gm_batch_stride)
  : !pto.ptr<f16, ub>, !pto.ptr<f16, gm>, i64, i64, i64, i64, i64,
    loop i64, i64, i64, loop i64, i64, i64
```

**Example (UB → remote GM):**

```mlir
%dst = pto.castptr %remote_i64 : i64 -> !pto.ptr<f16, gm, #pto.remote>
pto.mte_ub_gm %ub_in, %dst, %len_burst
  nburst(%rows, %ub_row_stride, %gm_row_stride) l2_cache_ctl(%l2_cache_ctl)
  : !pto.ptr<f16, ub>, !pto.ptr<f16, gm, #pto.remote>, i64, i64, i64, i64, i64
```

---

### `pto.mte_ub_ub`

- **syntax:**
```mlir
pto.mte_ub_ub %ub_src, %ub_dst, %len_burst
  nburst(%n_burst, %src_gap, %dst_gap)
  : !pto.ptr<T, ub>, !pto.ptr<T, ub>, i64, i64, i64, i64
```
- **semantics:** Grouped UB→UB copy.

**Parameter Table:**

| Parameter | Width | Description |
|-----------|-------|-------------|
| `%ub_src` | ptr | UB source pointer (`!pto.ptr<T, ub>`, 32B-aligned) |
| `%ub_dst` | ptr | UB destination pointer (`!pto.ptr<T, ub>`, 32B-aligned) |
| `%len_burst` | 16 bits | Burst length in units of 32 bytes |
| `nburst(%n_burst, %src_gap, %dst_gap)` | 16 bits / 16 bits / 16 bits | Required copy burst group: count, source gap, destination gap |

**Constraints:**

- UB source and destination addresses must be 32B-aligned.
- `%len_burst`, `%src_gap`, and `%dst_gap` are encoded in units of 32 bytes.

**Example:**

```mlir
pto.mte_ub_ub %ub_src, %ub_dst, %len32b
  nburst(%rows, %src_gap, %dst_gap)
  : !pto.ptr<i16, ub>, !pto.ptr<i16, ub>, i64, i64, i64, i64
```

---

### `pto.mte_ub_l1`

- **syntax:**
```mlir
pto.mte_ub_l1 %ub_src, %l1_dst, %len_burst
  nburst(%n_burst, %src_gap, %dst_gap)
  : !pto.ptr<T, ub>, !pto.ptr<T, l1>, i64, i64, i64, i64
```
- **semantics:** Grouped UB→L1/CBUF copy.

**Parameter Table:**

| Parameter | Width | Description |
|-----------|-------|-------------|
| `%ub_src` | ptr | UB source pointer (`!pto.ptr<T, ub>`, 32B-aligned) |
| `%l1_dst` | ptr | L1 destination pointer (`!pto.ptr<T, l1>`, 32B-aligned) |
| `%len_burst` | 16 bits | Burst length in units of 32 bytes |
| `nburst(%n_burst, %src_gap, %dst_gap)` | 16 bits / 16 bits / 16 bits | Required copy burst group: count, source gap, destination gap |

**Constraints:**

- UB source and L1 destination addresses must be 32B-aligned.
- `%len_burst`, `%src_gap`, and `%dst_gap` are encoded in units of 32 bytes.

**Example:**

```mlir
pto.mte_ub_l1 %ub_src, %l1_dst, %len32b
  nburst(%rows, %src_gap, %dst_gap)
  : !pto.ptr<i16, ub>, !pto.ptr<i16, l1>, i64, i64, i64, i64
```

---

## Grouped DMA Burst / Stride / Pad Model

This section describes the grouped DMA interfaces in this document:
`pto.mte_gm_ub` and `pto.mte_ub_gm`.

For these grouped DMA ops, the innermost `nburst(...)` group is
**stride-based**: the source and destination stride operands are the
start-to-start byte distance from one burst row to the next row.

### Key Terms

```
burst    = lenBurst contiguous bytes transferred per row
stride   = distance (bytes) from start of row[r] to start of row[r+1]
pad      = ub_stride - lenBurst, padded to the 32B alignment boundary
```

### Alignment Constraints

- **UB addresses** (both source and destination) must be **32-byte aligned**.
- **GM→UB padding**: When `pad(...)` is present on `pto.mte_gm_ub`, each UB row is padded from `lenBurst` up to the **32B-aligned boundary** of `ub_stride` with `pad_val`. This ensures every UB row starts at a 32B-aligned offset.
- **UB→GM de-padding**: MTE3 reads `lenBurst` bytes from each 32B-aligned UB row (skipping any padding that was added during load), writing only valid data to GM. This effectively strips padding on store.

---

## UB Copy Burst / Step Model

This section describes the grouped UB-copy interface in this document:
`pto.mte_ub_ub` and `pto.mte_ub_l1`.

For `pto.mte_ub_ub` and `pto.mte_ub_l1`, each burst copies `len_burst * 32` bytes.

The next burst starts at:

```text
src_next = src_curr + (len_burst + src_gap) * 32 bytes
dst_next = dst_curr + (len_burst + dst_gap) * 32 bytes
```

So `src_gap` and `dst_gap` are gap fields that advance to the next burst
after the copied 32B blocks.

### 2D Diagram: GM→UB (`pto.mte_gm_ub`)

```
GM (source, `!pto.ptr<T, gm>`):

          |<--- src_stride (start-to-start) --->|
          |<- len_burst ->|                     |
Row 0:    [##DATA########]......................|
Row 1:    [##DATA########]......................|
Row 2:    [##DATA########]......................|
          ...
Row N-1:  [##DATA########]

UB (destination, `!pto.ptr<T, ub>`, 32B-aligned):

          |<---------- dst_stride (32B-aligned) ---------->|
          |<- len_burst ->|<- pad (to 32B boundary) ->|    |
Row 0:    [##DATA########][000000 PAD 000000000000000]
Row 1:    [##DATA########][000000 PAD 000000000000000]
Row 2:    [##DATA########][000000 PAD 000000000000000]
          ...
Row N-1:  [##DATA########][000000 PAD 000000000000000]

N = n_burst
stride = start of row[r] to start of row[r+1]
pad    = filled with pad_val to 32B boundary (`pad(...)` present)
[DATA] = valid data transferred by DMA
[PAD]  = pad_val fill (from `pad(...)`)
```

### 2D Diagram: UB→GM (`pto.mte_ub_gm` with GM destination)

```
UB (source, `!pto.ptr<T, ub>`, 32B-aligned start addr):

          |<---------- src_stride (32B-aligned) --------->|
          |<- len_burst ->|<-- pad (ignored on read) -->| |
Row 0:    [##DATA########][000 pad 000000000000000000]
Row 1:    [##DATA########][000 pad 000000000000000000]
Row 2:    [##DATA########][000 pad 000000000000000000]
          ...
Row N-1:  [##DATA########][000 pad 000000000000000000]

GM (destination, `!pto.ptr<T, gm>`):

          |<--- dst_stride (start-to-start) --->|
          |<- len_burst ->|                     |
Row 0:    [##DATA########]......................|
Row 1:    [##DATA########]......................|
Row 2:    [##DATA########]......................|
          ...
Row N-1:  [##DATA########]

N = n_burst
MTE3 reads only len_burst bytes from each UB row (de-padding).
Only len_burst bytes are written to each GM row.
```

---

## Multi-Level Loop Semantics

The full DMA transfer is a nested loop. `nburst(...)` is the innermost group.
If one or more `loop(...)` groups are present, they wrap `nburst(...)` in the
same order they appear in the op: the first `loop(...)` is the innermost outer
group, the second `loop(...)` wraps the first one, and so on.

### GM→UB Full Loop

For a form

```mlir
pto.mte_gm_ub %gm_src, %ub_dst, %l2_cache_ctl, %len_burst
  nburst(%n_burst, %src_stride, %dst_stride)
  loop(%c0, %s0, %d0)
  loop(%c1, %s1, %d1)
  ...
  loop(%cN, %sN, %dN)
  [pad(%pad_value[, %left_padding_count, %right_padding_count])]
```

the transfer is equivalent to:

```c
for (int lN = 0; lN < cN; ++lN) {
  ...
  for (int l1 = 0; l1 < c1; ++l1) {
    for (int l0 = 0; l0 < c0; ++l0) {
      uint8_t *gm_base = gm_src + l0 * s0 + l1 * s1 + ... + lN * sN;
      uint8_t *ub_base = ub_dst + l0 * d0 + l1 * d1 + ... + lN * dN;
      for (int r = 0; r < n_burst; ++r) {
        memcpy(ub_base + r * dst_stride,
               gm_base + r * src_stride,
               len_burst);
        if (pad_enabled)
          memset(ub_base + r * dst_stride + len_burst,
                 pad_val,
                 dst_stride - len_burst);
      }
    }
  }
}
```

If no `loop(...)` group is present, only the innermost `nburst(...)` loop
remains.

### UB→Destination Full Loop

For a form

```mlir
pto.mte_ub_gm %ub_src, %dst, %len_burst
  nburst(%n_burst, %src_stride, %dst_stride)
  l2_cache_ctl(%l2_cache_ctl)
  loop(%c0, %s0, %d0)
  loop(%c1, %s1, %d1)
  ...
  loop(%cN, %sN, %dN)
```

the transfer is equivalent to:

```c
for (int lN = 0; lN < cN; ++lN) {
  ...
  for (int l1 = 0; l1 < c1; ++l1) {
    for (int l0 = 0; l0 < c0; ++l0) {
      uint8_t *ub_base = ub_src + l0 * s0 + l1 * s1 + ... + lN * sN;
      uint8_t *dst_base = dst + l0 * d0 + l1 * d1 + ... + lN * dN;
      for (int r = 0; r < n_burst; ++r) {
        memcpy(dst_base + r * dst_stride,
               ub_base + r * src_stride,
               len_burst);
      }
    }
  }
}
```

If no `loop(...)` group is present, only the innermost `nburst(...)` loop
remains.

---

## Example 1: GM→UB — Load a 32×32 f32 Tile (Simple Case)

Load a 32×32 f32 tile from GM into UB. This matches the `abs_kernel_2d` test case.

```
GM layout (32 × 32 f32, contiguous):

    |<- len_burst = 128B (32 × 4) ->|
    |<- src_stride = 128B --------->|
    +--[#######TILE#######]--+  row 0
    +--[#######TILE#######]--+  row 1
    ...
    +--[#######TILE#######]--+  row 31

UB layout (32 × 32 f32, 32B-aligned, contiguous):

    |<- dst_stride = 128B (32B-aligned) ->|
    +--[#######TILE#######]--+  row 0
    +--[#######TILE#######]--+  row 1
    ...
    +--[#######TILE#######]--+  row 31

    len_burst   = 32 × 4 = 128 bytes
    src_stride  = 128 bytes (contiguous rows)
    dst_stride  = 128 bytes (already 32B-aligned, no padding)
```

```mlir
// Simple 2D load — only nburst(...) is needed
pto.mte_gm_ub %arg0, %ub_in, %c0_i64, %c128_i64
  nburst(%c32_i64, %c128_i64, %c128_i64)
  : !pto.ptr<f32, gm>, !pto.ptr<f32, ub>, i64, i64, i64
```

---

## Example 2: GM→UB — Load a 2D Tile from a Larger Matrix

Load a 64×128 tile (f16) from a 1024×512 matrix in GM into UB.

```
GM layout (1024 × 512 f16):

    col 0          col 128               col 512
    |              |                     |
    +--[###TILE###]+.....................+  row R
    +--[###TILE###]+.....................+  row R+1
    ...
    +--[###TILE###]+.....................+  row R+63

    |<--------- src_stride = 1024B ----------->|
    |<-len_burst=256B->|

    len_burst   = 128 × 2 = 256 bytes (128 f16 elements)
    src_stride  = 512 × 2 = 1024 bytes (start-to-start, full GM row)

UB layout (64 × 128 f16, 32B-aligned, contiguous):

    +--[###TILE###]--+  row 0  (256 bytes, 32B-aligned, no pad)
    +--[###TILE###]--+  row 1
    ...
    +--[###TILE###]--+  row 63

    dst_stride = 256 bytes (= len_burst, already 32B-aligned, no padding)
```

```mlir
pto.mte_gm_ub %gm_ptr, %ub_ptr, %c0_i64, %c256_i64
  nburst(%c64_i64, %c1024_i64, %c256_i64)
  : !pto.ptr<f16, gm>, !pto.ptr<f16, ub>, i64, i64, i64
```

---

## Example 3: GM→UB — Load with Padding

Load 100 valid columns from GM into a 128-wide UB tile (f16). The remaining 28 columns are zero-padded.

```
GM (100 cols valid, contiguous):

    |<-len_burst=200B->|
    |<- src_stride=200B (start-to-start) ->|
    +--[####DATA####]-+  row 0
    +--[####DATA####]-+  row 1
    ...
    +--[####DATA####]-+  row 63

UB (128 cols wide, 32B-aligned, padded):

    |<--------- dst_stride = 256B (32B-aligned) --------->|
    |<-len_burst=200B->|<---- pad = 56B to 32B boundary ->|
    +--[####DATA####]-+[0000000 PAD 0000000000000000000000]+  row 0
    +--[####DATA####]-+[0000000 PAD 0000000000000000000000]+  row 1
    ...
    +--[####DATA####]-+[0000000 PAD 0000000000000000000000]+  row 63

    len_burst   = 100 × 2 = 200 bytes
    src_stride  = 200 bytes (start-to-start, contiguous in GM)
    dst_stride  = 128 × 2 = 256 bytes (32B-aligned tile width in UB)
    pad         = 256 - 200 = 56 bytes (padded to 32B boundary with pad_val)
```

```mlir
%pad = arith.constant 0 : i16
pto.mte_gm_ub %gm_ptr, %ub_ptr, %c0_i64, %c200_i64
  nburst(%c64_i64, %c200_i64, %c256_i64)
  pad(%pad, %c0_i64, %c0_i64)
  : !pto.ptr<f16, gm>, !pto.ptr<f16, ub>, i64, i64, i64, pad i16, i64, i64
```

---

## Example 4: UB→GM — Store a 32×32 f32 Tile (Simple Case)

Store a 32×32 f32 tile from UB back to GM. This matches the `abs_kernel_2d` test case.

```
UB (source, 32B-aligned, 32 × 32 f32):

    |<- src_stride = 128B (32B-aligned) ->|
    |<- len_burst = 128B ->|
    +--[#######TILE#######]---+  row 0
    +--[#######TILE#######]---+  row 1
    ...
    +--[#######TILE#######]---+  row 31

    (no padding here — len_burst == src_stride)

GM (dest, 32 × 32 f32):

    |<- dst_stride = 128B ->|
    |<- len_burst = 128B -->|
    +--[#######TILE#######]---+  row 0
    +--[#######TILE#######]---+  row 1
    ...
    +--[#######TILE#######]---+  row 31
```

```mlir
pto.mte_ub_gm %ub_out, %arg1, %c128_i64
  nburst(%c32_i64, %c128_i64, %c128_i64) l2_cache_ctl(%c0_i64)
  : !pto.ptr<f32, ub>, !pto.ptr<f32, gm>, i64, i64, i64, i64, i64
```

---

## Example 5: UB→GM — Store a 2D Tile Back to a Larger Matrix

Store a 64×128 tile (f16) from UB back to a 1024×512 GM matrix at an offset.

```
UB (source, 32B-aligned, 64 × 128 f16):

    |<- src_stride = 256B (32B-aligned) ->|
    |<- len_burst = 256B ->|
    +--[#####TILE#####]---+  row 0
    +--[#####TILE#####]---+  row 1
    ...
    +--[#####TILE#####]---+  row 63

    (no padding here — len_burst == src_stride)

GM (dest, into 1024 × 512 matrix):

    |<----------- dst_stride = 1024B (start-to-start) --------->|
    |<- len_burst = 256B ->|                                    |
    col 0          col 128                              col 512
    +--[#####TILE#####]---+.............................+  row R
    +--[#####TILE#####]---+.............................+  row R+1
    ...
    +--[#####TILE#####]---+.............................+  row R+63

    MTE3 reads len_burst bytes from each 32B-aligned UB row,
    writes only len_burst bytes per GM row (stride controls row spacing).
```

```mlir
pto.mte_ub_gm %ub_ptr, %gm_ptr, %c256_i64
  nburst(%c64_i64, %c256_i64, %c1024_i64) l2_cache_ctl(%c0_i64)
  : !pto.ptr<f16, ub>, !pto.ptr<f16, gm>, i64, i64, i64, i64, i64
```

---

## Example 6: GM→UB with Multi-Level Loop (Batch of Tiles)

Load 4 batches of 8×128 tiles from a [4, 8, 128] f16 tensor using one outer
`loop(...)` group.

```
GM [4, 8, 128] f16 (contiguous):        UB (4 tiles laid out sequentially):

    batch 0: 8 rows × 256 bytes          [batch 0: 8×128][batch 1: 8×128]
    batch 1: 8 rows × 256 bytes          [batch 2: 8×128][batch 3: 8×128]
    batch 2: 8 rows × 256 bytes
    batch 3: 8 rows × 256 bytes          outer loop src_stride = 2048 bytes (8 × 256)
                                          outer loop dst_stride = 2048 bytes (8 × 256)
    Each batch = 8 × 256 = 2048 bytes     outer loop count = 4 (iterate over batches)
```

```mlir
// One outer loop group over 4 batches
pto.mte_gm_ub %gm_ptr, %ub_ptr, %c0_i64, %c256_i64
  nburst(%c8_i64, %c256_i64, %c256_i64)
  loop(%c4_i64, %c2048_i64, %c2048_i64)
  : !pto.ptr<f16, gm>, !pto.ptr<f16, ub>, i64, i64, i64, loop i64, i64, i64
```

Execution trace:

```
loop iter 0: gm_ptr + 0×2048 → ub_ptr + 0×2048, DMA 8 rows × 256B
loop iter 1: gm_ptr + 1×2048 → ub_ptr + 1×2048, DMA 8 rows × 256B
loop iter 2: gm_ptr + 2×2048 → ub_ptr + 2×2048, DMA 8 rows × 256B
loop iter 3: gm_ptr + 3×2048 → ub_ptr + 3×2048, DMA 8 rows × 256B
```
