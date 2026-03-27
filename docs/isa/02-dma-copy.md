# 2. DMA Copy Programming

> **Category:** DMA transfer configuration and execution
> **Pipelines:** MTE2 (GM→UB), MTE3 (UB→GM)

DMA transfers move data between Global Memory (GM) and Unified Buffer (UB). The MTE engines operate asynchronously from the Vector core, requiring explicit sync (see [Pipeline Sync](01-pipeline-sync.md)).

The MTE2/MTE3 DMA engine executes a **multi-level nested loop** transfer. Before issuing the copy instruction, stride and loop-size registers must be configured.

---

## Loop Stride Configuration (GM→UB)

These ops configure the MTE2 DMA engine's hardware loops for GM→UB transfers. They must be set **before** calling `pto.copy_gm_to_ubuf`.

### `pto.set_loop_size_outtoub`

- **syntax:** `pto.set_loop_size_outtoub %loop1_count, %loop2_count : i64, i64`
- **semantics:** Configure HW loop iteration counts for GM→UB DMA.

**Register encoding:**

```c
set_loop_size_outtoub(loop2 << 21 | loop1);
```

| Bits | Field | Width | Description |
|------|-------|-------|-------------|
| [20:0] | `loop1` | 21 bits | Inner HW loop iteration count |
| [42:21] | `loop2` | 22 bits | Outer HW loop iteration count |

When not using multi-level looping, set both to 1: `set_loop_size_outtoub(1ULL << 21 | 1ULL)`.

---

### `pto.set_loop2_stride_outtoub`

- **syntax:** `pto.set_loop2_stride_outtoub %src_stride, %dst_stride : i64, i64`
- **semantics:** Configure outer loop (loop2) pointer advance for GM→UB DMA.

**Register encoding:**

```c
set_loop2_stride_outtoub(ub_dst_stride << 40 | gm_src_stride);
```

| Bits | Field | Width | Description |
|------|-------|-------|-------------|
| [39:0] | `gm_src_stride` | 40 bits | GM source pointer advance per loop2 iteration (bytes) |
| [60:40] | `ub_dst_stride` | 21 bits | UB destination pointer advance per loop2 iteration (bytes) |

After each loop2 iteration, the DMA engine advances the GM read pointer by `gm_src_stride` and UB write pointer by `ub_dst_stride`.

---

### `pto.set_loop1_stride_outtoub`

- **syntax:** `pto.set_loop1_stride_outtoub %src_stride, %dst_stride : i64, i64`
- **semantics:** Configure inner loop (loop1) pointer advance for GM→UB DMA.

**Register encoding:**

```c
set_loop1_stride_outtoub(ub_dst_stride << 40 | gm_src_stride);
```

| Bits | Field | Width | Description |
|------|-------|-------|-------------|
| [39:0] | `gm_src_stride` | 40 bits | GM source pointer advance per loop1 iteration (bytes) |
| [60:40] | `ub_dst_stride` | 21 bits | UB destination pointer advance per loop1 iteration (bytes) |

---

## Loop Stride Configuration (UB→GM)

These ops configure the MTE3 DMA engine's hardware loops for UB→GM transfers. They must be set **before** calling `pto.copy_ubuf_to_gm`.

Note: UB stride fields are 21 bits (sufficient for 256KB UB address space), GM stride fields are 40 bits (full GM address range).

### `pto.set_loop_size_ubtoout`

- **syntax:** `pto.set_loop_size_ubtoout %loop1_count, %loop2_count : i64, i64`
- **semantics:** Configure HW loop iteration counts for UB→GM DMA.

**Register encoding:**

```c
set_loop_size_ubtoout(loop2 << 21 | loop1);
```

| Bits | Field | Width | Description |
|------|-------|-------|-------------|
| [20:0] | `loop1` | 21 bits | Inner HW loop iteration count |
| [42:21] | `loop2` | 22 bits | Outer HW loop iteration count |

---

### `pto.set_loop2_stride_ubtoout`

- **syntax:** `pto.set_loop2_stride_ubtoout %src_stride, %dst_stride : i64, i64`
- **semantics:** Configure outer loop (loop2) pointer advance for UB→GM DMA.

**Register encoding:**

```c
set_loop2_stride_ubtoout(gm_dst_stride << 21 | ub_src_stride);
```

| Bits | Field | Width | Description |
|------|-------|-------|-------------|
| [20:0] | `ub_src_stride` | 21 bits | UB source pointer advance per loop2 iteration (bytes) |
| [60:21] | `gm_dst_stride` | 40 bits | GM destination pointer advance per loop2 iteration (bytes) |

---

### `pto.set_loop1_stride_ubtoout`

- **syntax:** `pto.set_loop1_stride_ubtoout %src_stride, %dst_stride : i64, i64`
- **semantics:** Configure inner loop (loop1) pointer advance for UB→GM DMA.

**Register encoding:**

```c
set_loop1_stride_ubtoout(gm_dst_stride << 21 | ub_src_stride);
```

| Bits | Field | Width | Description |
|------|-------|-------|-------------|
| [20:0] | `ub_src_stride` | 21 bits | UB source pointer advance per loop1 iteration (bytes) |
| [60:21] | `gm_dst_stride` | 40 bits | GM destination pointer advance per loop1 iteration (bytes) |

---

## DMA Transfer Execution

### `pto.copy_gm_to_ubuf`

- **syntax:**
```mlir
pto.copy_gm_to_ubuf %gm_src, %ub_dst, %valid_rows, %valid_cols,
    %sid, %n_burst, %len_burst, %left_padding, %right_padding,
    %data_select_bit, %l2_cache_ctl, %gm_src_stride, %ub_dst_stride
    : !pto.ptr<T, gm>, !pto.ptr<T, ub>, i64, i64, i64, i64, i64,
      i64, i64, i1, i64, i64, i64
```
- **semantics:** DMA transfer from Global Memory (`!pto.ptr<T, gm>`) to Unified Buffer (`!pto.ptr<T, ub>`).

**Parameters:**

| Parameter | Description |
|-----------|-------------|
| `%gm_src` | GM source pointer (`!pto.ptr<T, gm>`) |
| `%ub_dst` | UB destination pointer (`!pto.ptr<T, ub>`, 32B-aligned) |
| `%valid_rows` | Valid row count |
| `%valid_cols` | Valid column count |
| `%sid` | Stream ID (usually 0) |
| `%n_burst` | Number of burst rows (innermost loop count) |
| `%len_burst` | Contiguous bytes transferred per burst row |
| `%left_padding` | Left padding count (bytes) |
| `%right_padding` | Right padding count (bytes) |
| `%data_select_bit` | Padding / data-select control bit (`i1`) |
| `%l2_cache_ctl` | L2 cache allocate control (TBD — controls whether DMA allocates in L2 cache) |
| `%gm_src_stride` | GM source stride: start-to-start distance between consecutive burst rows (bytes) |
| `%ub_dst_stride` | UB destination stride: start-to-start distance between consecutive burst rows (bytes, 32B-aligned) |

---

### `pto.copy_ubuf_to_gm`

- **syntax:**
```mlir
pto.copy_ubuf_to_gm %ub_src, %gm_dst, %valid_rows, %valid_cols,
    %sid, %n_burst, %len_burst, %reserved, %gm_dst_stride, %ub_src_stride
    : !pto.ptr<T, ub>, !pto.ptr<T, gm>, i64, i64, i64, i64, i64, i64, i64, i64
```
- **semantics:** DMA transfer from Unified Buffer (`!pto.ptr<T, ub>`) to Global Memory (`!pto.ptr<T, gm>`). MTE3 reads only `len_burst` bytes from each UB row (de-padding).

**Parameters:**

| Parameter | Description |
|-----------|-------------|
| `%ub_src` | UB source pointer (`!pto.ptr<T, ub>`, 32B-aligned) |
| `%gm_dst` | GM destination pointer (`!pto.ptr<T, gm>`) |
| `%valid_rows` | Valid row count |
| `%valid_cols` | Valid column count |
| `%sid` | Stream ID (usually 0) |
| `%n_burst` | Number of burst rows |
| `%len_burst` | Contiguous bytes transferred per burst row |
| `%reserved` | Reserved field (set to 0) |
| `%gm_dst_stride` | GM destination stride: start-to-start distance between consecutive burst rows (bytes) |
| `%ub_src_stride` | UB source stride: start-to-start distance between consecutive burst rows (bytes, 32B-aligned) |

---

### `pto.copy_ubuf_to_ubuf`

- **syntax:**
```mlir
pto.copy_ubuf_to_ubuf %source, %dest, %sid, %n_burst, %len_burst, %src_stride, %dst_stride
    : !pto.ptr<T, ub>, !pto.ptr<T, ub>, i64 x5
```
- **semantics:** Copy within Unified Buffer.

**Parameters:**

| Parameter | Description |
|-----------|-------------|
| `%source` | UB source pointer |
| `%dest` | UB destination pointer |
| `%sid` | Stream ID |
| `%n_burst` | Number of bursts |
| `%len_burst` | Length per burst |
| `%src_stride` | Source stride |
| `%dst_stride` | Destination stride |

---

## Burst / Stride / Pad Model

All A5 DMA addresses are **stride-based**: stride is the distance from the start of one row to the start of the next row (`stride >= lenBurst`). There is no separate "gap" parameter.

### Key Terms

```
burst    = lenBurst contiguous bytes transferred per row
stride   = distance (bytes) from start of row[r] to start of row[r+1]
pad      = ub_stride - lenBurst, padded to the 32B alignment boundary
```

### Alignment Constraints

- **UB addresses** (both source and destination) must be **32-byte aligned**.
- **GM→UB padding**: When `data_select_bit = true`, each UB row is padded from `lenBurst` up to the **32B-aligned boundary** of `ub_stride` with `pad_val` (set via `set_mov_pad_val`). This ensures every UB row starts at a 32B-aligned offset.
- **UB→GM de-padding**: MTE3 reads `lenBurst` bytes from each 32B-aligned UB row (skipping any padding that was added during load), writing only valid data to GM. This effectively strips padding on store.

### 2D Diagram: GM→UB (pto.copy_gm_to_ubuf)

```
GM (source, `!pto.ptr<T, gm>`):

          |<--- gm_stride (start-to-start) --->|
          |<- lenBurst ->|                      |
Row 0:    [##DATA########].......................|
Row 1:    [##DATA########].......................|
Row 2:    [##DATA########].......................|
          ...
Row N-1:  [##DATA########]

UB (destination, `!pto.ptr<T, ub>`, 32B-aligned):

          |<---------- ub_stride (32B-aligned) ---------->|
          |<- lenBurst ->|<- pad (to 32B boundary) ->|    |
Row 0:    [##DATA########][000000 PAD 000000000000000]
Row 1:    [##DATA########][000000 PAD 000000000000000]
Row 2:    [##DATA########][000000 PAD 000000000000000]
          ...
Row N-1:  [##DATA########][000000 PAD 000000000000000]

N = n_burst
stride = start of row[r] to start of row[r+1]
pad    = filled with pad_val to 32B boundary (data_select_bit=true)
[DATA] = valid data transferred by DMA
[PAD]  = pad_val fill (set via set_mov_pad_val)
```

### 2D Diagram: UB→GM (pto.copy_ubuf_to_gm)

```
UB (source, `!pto.ptr<T, ub>`, 32B-aligned start addr):

          |<---------- src_stride (32B-aligned) --------->|
          |<- lenBurst ->|<-- pad (ignored on read) -->|  |
Row 0:    [##DATA########][000 pad 000000000000000000]
Row 1:    [##DATA########][000 pad 000000000000000000]
Row 2:    [##DATA########][000 pad 000000000000000000]
          ...
Row N-1:  [##DATA########][000 pad 000000000000000000]

GM (destination, `!pto.ptr<T, gm>`):

          |<--- dst_stride (start-to-start) --->|
          |<- lenBurst ->|                      |
Row 0:    [##DATA########].......................|
Row 1:    [##DATA########].......................|
Row 2:    [##DATA########].......................|
          ...
Row N-1:  [##DATA########]

N = n_burst
MTE3 reads only lenBurst bytes from each UB row (de-padding).
Only lenBurst bytes are written to each GM row.
```

---

## Multi-Level Loop Semantics (C Code)

The full DMA transfer is a nested loop. The HW loop registers (set before the copy) control the outer levels, and the copy instruction parameters control the innermost burst level.

### GM→UB Full Loop

```c
// Register setup (once before copy):
//   loop_size:  [20:0]=loop1(21b), [42:21]=loop2(22b)
//   stride:     [39:0]=gm_src(40b), [60:40]=ub_dst(21b)
set_loop_size_outtoub(loop2 << 21 | loop1);
set_loop2_stride_outtoub(loop2_ub_stride << 40 | loop2_gm_stride);
set_loop1_stride_outtoub(loop1_ub_stride << 40 | loop1_gm_stride);

// C equivalent of what the HW executes:
for (int j = 0; j < loop2; j++) {                      // HW outer loop
    uint8_t *gm1 = gm_src + j * loop2_gm_stride;
    uint8_t *ub1 = ub_dst + j * loop2_ub_stride;

    for (int k = 0; k < loop1; k++) {                  // HW inner loop
        uint8_t *gm2 = gm1 + k * loop1_gm_stride;
        uint8_t *ub2 = ub1 + k * loop1_ub_stride;

        for (int r = 0; r < n_burst; r++) {             // burst engine
            memcpy(ub2 + r * ub_dst_stride,             //   UB dest row
                   gm2 + r * gm_src_stride,             //   GM src row
                   len_burst);                           //   contiguous bytes
            if (data_select_bit)
                memset(ub2 + r * ub_dst_stride + len_burst,
                       pad_val, ub_dst_stride - len_burst);
        }
    }
}
```

### UB→GM Full Loop

```c
// Register setup:
//   loop_size:  [20:0]=loop1(21b), [42:21]=loop2(22b)
//   stride:     [20:0]=ub_src(21b), [60:21]=gm_dst(40b)
set_loop_size_ubtoout(loop2 << 21 | loop1);
set_loop2_stride_ubtoout(loop2_gm_stride << 21 | loop2_ub_stride);
set_loop1_stride_ubtoout(loop1_gm_stride << 21 | loop1_ub_stride);

// C equivalent:
for (int j = 0; j < loop2; j++) {
    uint8_t *ub1 = ub_src + j * loop2_ub_stride;
    uint8_t *gm1 = gm_dst + j * loop2_gm_stride;

    for (int k = 0; k < loop1; k++) {
        uint8_t *ub2 = ub1 + k * loop1_ub_stride;
        uint8_t *gm2 = gm1 + k * loop1_gm_stride;

        for (int r = 0; r < n_burst; r++) {
            memcpy(gm2 + r * gm_dst_stride,             //   GM dest row
                   ub2 + r * ub_src_stride,              //   UB src row
                   len_burst);                           //   contiguous bytes
        }
    }
}
```

---

## Example 1: GM→UB — Load a 2D Tile from a Larger Matrix

Load a 64×128 tile (f16) from a 1024×512 matrix in GM into UB.

```
GM layout (1024 × 512 f16):

    col 0          col 128               col 512
    |              |                     |
    +--[###TILE###]+.....................+  row R
    +--[###TILE###]+.....................+  row R+1
    ...
    +--[###TILE###]+.....................+  row R+63

    |<--------- gm_src_stride = 1024B --------->|
    |<-lenBurst=256B->|

    lenBurst       = 128 × 2 = 256 bytes (128 f16 elements)
    gm_src_stride  = 512 × 2 = 1024 bytes (start-to-start, full GM row)

UB layout (64 × 128 f16, 32B-aligned, contiguous):

    +--[###TILE###]--+  row 0  (256 bytes, 32B-aligned, no pad)
    +--[###TILE###]--+  row 1
    ...
    +--[###TILE###]--+  row 63

    ub_dst_stride = 256 bytes (= lenBurst, already 32B-aligned, no padding)
```

```mlir
// Simple 2D load — no multi-level loops needed
pto.set_loop_size_outtoub %c1_i64, %c1_i64 : i64, i64
pto.set_loop1_stride_outtoub %c0_i64, %c0_i64 : i64, i64
pto.set_loop2_stride_outtoub %c0_i64, %c0_i64 : i64, i64

pto.copy_gm_to_ubuf %gm_ptr, %ub_ptr,
    %c0_i64,       // sid = 0
    %c64_i64,      // n_burst = 64 (64 rows)
    %c256_i64,     // len_burst = 256 bytes per row
    %c0_i64,       // left_padding = 0
    %c0_i64,       // right_padding = 0
    %c0_i64,       // l2_cache_ctl = 0
    %c1024_i64,    // gm_src_stride = 1024 bytes (full matrix row)
    %c256_i64      // ub_dst_stride = 256 bytes (tile row)
    {layout = "nd", data_select_bit = false, ub_pad = false}
    : !pto.ptr<f16, gm>, !pto.ptr<f16, ub>, i64, i64, i64, i64, i64,
      i64, i64, i64
```

---

## Example 2: GM→UB — Load with Padding

Load 100 valid columns from GM into a 128-wide UB tile (f16). The remaining 28 columns are zero-padded.

```
GM (100 cols valid, contiguous):

    |<-lenBurst=200B->|
    |<- gm_src_stride=200B (start-to-start) ->|
    +--[####DATA####]-+  row 0
    +--[####DATA####]-+  row 1
    ...
    +--[####DATA####]-+  row 63

UB (128 cols wide, 32B-aligned, padded):

    |<--------- ub_dst_stride = 256B (32B-aligned) --------->|
    |<-lenBurst=200B->|<---- pad = 56B to 32B boundary ---->|
    +--[####DATA####]-+[0000000 PAD 000000000000000000000000]+  row 0
    +--[####DATA####]-+[0000000 PAD 000000000000000000000000]+  row 1
    ...
    +--[####DATA####]-+[0000000 PAD 000000000000000000000000]+  row 63

    lenBurst       = 100 × 2 = 200 bytes
    gm_src_stride  = 200 bytes (start-to-start, contiguous in GM)
    ub_dst_stride  = 128 × 2 = 256 bytes (32B-aligned tile width in UB)
    pad            = 256 - 200 = 56 bytes (padded to 32B boundary with pad_val)
```

```mlir
pto.set_loop_size_outtoub %c1_i64, %c1_i64 : i64, i64
pto.set_loop1_stride_outtoub %c0_i64, %c0_i64 : i64, i64
pto.set_loop2_stride_outtoub %c0_i64, %c0_i64 : i64, i64

pto.copy_gm_to_ubuf %gm_ptr, %ub_ptr,
    %c0_i64,       // sid = 0
    %c64_i64,      // n_burst = 64
    %c200_i64,     // len_burst = 200 bytes
    %c0_i64,       // left_padding = 0
    %c0_i64,       // right_padding = 0
    %c0_i64,       // l2_cache_ctl = 0
    %c200_i64,     // gm_src_stride = 200 bytes
    %c256_i64      // ub_dst_stride = 256 bytes (32B-aligned)
    {layout = "nd", data_select_bit = true, ub_pad = true}
    : !pto.ptr<f16, gm>, !pto.ptr<f16, ub>, i64, i64, i64, i64, i64,
      i64, i64, i64
```

---

## Example 3: UB→GM — Store a 2D Tile Back to a Larger Matrix

Store a 64×128 tile (f16) from UB back to a 1024×512 GM matrix at an offset.

```
UB (source, 32B-aligned, 64 × 128 f16):

    |<- ub_src_stride = 256B (32B-aligned) ->|
    |<- lenBurst = 256B ->|
    +--[#####TILE#####]---+  row 0
    +--[#####TILE#####]---+  row 1
    ...
    +--[#####TILE#####]---+  row 63

    (no padding here — lenBurst == ub_src_stride)

GM (dest, into 1024 × 512 matrix):

    |<----------- gm_dst_stride = 1024B (start-to-start) --------->|
    |<- lenBurst = 256B ->|                                        |
    col 0          col 128                                 col 512
    +--[#####TILE#####]---+...............................+  row R
    +--[#####TILE#####]---+...............................+  row R+1
    ...
    +--[#####TILE#####]---+...............................+  row R+63

    MTE3 reads lenBurst bytes from each 32B-aligned UB row,
    writes only lenBurst bytes per GM row (stride controls row spacing).
```

```mlir
// Configure MTE3 strides
pto.set_loop_size_ubtoout %c1_i64, %c1_i64 : i64, i64
pto.set_loop1_stride_ubtoout %c0_i64, %c0_i64 : i64, i64
pto.set_loop2_stride_ubtoout %c0_i64, %c0_i64 : i64, i64

pto.copy_ubuf_to_gm %ub_ptr, %gm_ptr,
    %c0_i64,       // sid = 0
    %c64_i64,      // n_burst = 64
    %c256_i64,     // len_burst = 256 bytes
    %c0_i64,       // reserved = 0
    %c1024_i64,    // gm_dst_stride = 1024 bytes (GM row)
    %c256_i64      // ub_src_stride = 256 bytes (UB row)
    {layout = "nd"}
    : !pto.ptr<f16, ub>, !pto.ptr<f16, gm>, i64, i64, i64, i64, i64, i64
```

---

## Example 4: GM→UB with Multi-Level Loop (Batch of Tiles)

Load 4 batches of 8×128 tiles from a [4, 8, 128] f16 tensor using loop1.

```
GM [4, 8, 128] f16 (contiguous):        UB (4 tiles laid out sequentially):

    batch 0: 8 rows × 256 bytes          [batch 0: 8×128][batch 1: 8×128]
    batch 1: 8 rows × 256 bytes          [batch 2: 8×128][batch 3: 8×128]
    batch 2: 8 rows × 256 bytes
    batch 3: 8 rows × 256 bytes          loop1_gm_stride = 2048 bytes (8 × 256)
                                          loop1_ub_stride = 2048 bytes (8 × 256)
    Each batch = 8 × 256 = 2048 bytes     loop1 = 4 (iterate over batches)
```

```mlir
// loop1 = 4 batches, loop2 = 1 (not used)
pto.set_loop_size_outtoub %c4_i64, %c1_i64 : i64, i64

// loop1 stride: advance by one batch (2048 bytes) in both GM and UB
pto.set_loop1_stride_outtoub %c2048_i64, %c2048_i64 : i64, i64
pto.set_loop2_stride_outtoub %c0_i64, %c0_i64 : i64, i64

pto.copy_gm_to_ubuf %gm_ptr, %ub_ptr,
    %c0_i64,       // sid = 0
    %c8_i64,       // n_burst = 8 rows per batch
    %c256_i64,     // len_burst = 256 bytes per row
    %c0_i64, %c0_i64, %c0_i64,
    %c256_i64,     // gm_src_stride = 256 (contiguous rows)
    %c256_i64      // ub_dst_stride = 256 (contiguous rows)
    {layout = "nd", data_select_bit = false, ub_pad = false}
    : !pto.ptr<f16, gm>, !pto.ptr<f16, ub>, i64, i64, i64, i64, i64,
      i64, i64, i64
```

Execution trace:

```
loop1 iter 0: gm_ptr + 0×2048 → ub_ptr + 0×2048, DMA 8 rows × 256B
loop1 iter 1: gm_ptr + 1×2048 → ub_ptr + 1×2048, DMA 8 rows × 256B
loop1 iter 2: gm_ptr + 2×2048 → ub_ptr + 2×2048, DMA 8 rows × 256B
loop1 iter 3: gm_ptr + 3×2048 → ub_ptr + 3×2048, DMA 8 rows × 256B
```

---

## Register Summary

| Register | Direction | Encoding | Field Widths | Purpose |
|----------|-----------|----------|--------------|---------|
| `set_loop_size_outtoub` | GM→UB | `loop2 << 21 \| loop1` | loop1: 21b, loop2: 22b | HW loop iteration counts |
| `set_loop2_stride_outtoub` | GM→UB | `ub_dst << 40 \| gm_src` | gm_src: 40b, ub_dst: 21b | Outer loop pointer advance (bytes) |
| `set_loop1_stride_outtoub` | GM→UB | `ub_dst << 40 \| gm_src` | gm_src: 40b, ub_dst: 21b | Inner loop pointer advance (bytes) |
| `set_loop_size_ubtoout` | UB→GM | `loop2 << 21 \| loop1` | loop1: 21b, loop2: 22b | HW loop iteration counts |
| `set_loop2_stride_ubtoout` | UB→GM | `gm_dst << 21 \| ub_src` | ub_src: 21b, gm_dst: 40b | Outer loop pointer advance (bytes) |
| `set_loop1_stride_ubtoout` | UB→GM | `gm_dst << 21 \| ub_src` | ub_src: 21b, gm_dst: 40b | Inner loop pointer advance (bytes) |
| `set_mov_pad_val` | GM→UB | pad value | — | Padding fill value (with `data_select_bit = true`) |
