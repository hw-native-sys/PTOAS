# 3. Vector Load/Store

> **Category:** UB ↔ Vector Register data movement
> **Pipeline:** PIPE_V (Vector Core)

Vector loads move data from Unified Buffer (UB) to vector registers (`vreg`). Vector stores move data from `vreg` back to UB. All vector compute operates only on `vreg` — UB is the staging area between DMA and compute.

## Common Operand Model

- `%source` / `%dest` is the base address operand in SSA form. The base pointer
  MUST address the Vector tile buffer / UB space.
- `%offset` is the displacement operand in SSA form. The exact encoding is
  instruction-specific, but the effective address and any post-update behavior
  MUST match the selected instruction form.
- `%mask` is the predicate operand for predicated memory families. For memory
  families,
  inactive lanes or inactive blocks MUST NOT issue memory requests unless the
  instruction explicitly documents a different behavior.
- `%result` is the destination vector register value in SSA form.
- `!pto.align` is the SSA carrier for alignment-buffer state used by unaligned
  load/store families. The PTO micro Instruction representation makes that state explicit rather than implicit.

---

## Latency and throughput (A5)

**Cycle-accurate simulator (CA model)** issue→retire timings for vector-side instructions behind this chapter. Values are **simulator** results, **not** guaranteed for silicon.

**SOC:** Tables below are from **Ascend910_9599** CA sim (the pto-isa ST default when **Ascend950PR_9599** is not selected).

**Log `dist:` tokens:** PTO load/store modes lower to **`RV_VLD` / `RV_VLDI` / `RV_VST` / `RV_VSTI`** with a **`dist:`** field on the vector pipes (`RVECLD` / `RVECST`). Some simulator logs typo contiguous load as `dist:NORAML`; treat as **`NORMAL`**.

### Reference op latencies (A5 mnemonics)

| A5 mnemonic | Mode / note | Typical issue→retire (cycles) |
|-------------|-------------|------------------------------|
| `RV_VLD` | `dist:NORMAL` / `NORAML` | **9** |
| `RV_VLDI` | `dist:DINTLV` (dual vreg) | **9** |
| `RV_VST` / `RV_VSTI` | `dist:NORM` | **9** |
| `RV_VGATHER2` | `Dtype: B32` | **27–28** |
| `RV_VGATHERB` | indexed byte gather | **~21** |
| `RV_VSCATTER` | `Dtype: B16` | **~17** |
| `RV_VADD` | F32 between UB-backed ops | **7** |

### `dist:` tokens (issue→retire)

Most **`dist:`** tokens are **9** issue→retire cycles. **`INTLV`** on **`RV_VSTI`** is **12** cycles.

| `dist:` (as in log) | RV op | issue→retire (cycles) |
|---------------------|-------|----------------------|
| `DINTLV` | `RV_VLDI` | **9** |
| `BRC` | `RV_VLD` | **9** |
| `BRC_BLK` | `RV_VLD` | **9** |
| `INTLV` | `RV_VSTI` | **12** |
| `UNPK` | `RV_VLD` | **9** |
| `NORM` | `RV_VSTI` | **9** |
| `PK` | `RV_VSTI` | **9** |
| `NORMAL` / `NORAML` | `RV_VLD` | **9** |

**Note:** PTO intrinsic **`BRC_BLK`** matches the **`BRC_BLK`** `dist:` string on **`RV_VLD`** in simulator logs (block-replicate path; not a plain contiguous copy in the usual tiling use).

**Issue (vector load/store):** `pto.vlds` (**`RV_VLD`**) is **dual-issue capable**: two independent `pto.vlds` can issue **in the same cycle**. **Alternatively**, the hardware can issue **one** `pto.vlds` **and** **one** `pto.vsts` together (**1+1**) in the same cycle. Each cycle is **either** dual **`vlds`** **or** **`vlds` + `vsts` (1+1)**—those two issue modes are mutually exclusive. Sustained throughput still depends on RAW hazards and loop structure.

**Throughput (simulator, pattern-dependent):**

- **`RV_VLD` / `pto.vlds`:** Dual-issue **or** half of a **1+1** with `vsts`, per the rule above.
- **`RV_VST` / `pto.vsts`:** In a **1+1** cycle, pairs with one `vlds`; otherwise typically **one** store per cycle in tight loops.
- **`RV_VGATHER2`:** Much lower than contiguous `RV_VLD` (on the order of **~0.1** ops/cycle in steady-state alongside 27–28-cycle latency).

### PTO `dist` summary (loads)

| PTO `dist` (load) | Latency |
|-------------------|-------------------|
| `NORM` | **9** cycles |
| `UNPK` | **9** cycles |
| `DINTLV` | **9** cycles (`RV_VLDI`) |
| `BRC` | **9** cycles (`RV_VLD`) |
| `BRC_BLK` | **9** cycles as **`dist:BRC_BLK`** on `RV_VLD` |
| `BDINTLV` | **9** cycles |
| `US`, `DS`, `SPLT4CHN`, `SPLT2CHN` | **9** cycles |

### PTO `dist` summary (stores)

| PTO `dist` (store) | Latency |
|--------------------|-------------------|
| `NORM` | **9** cycles (`RV_VSTI`) |
| `PK` | **9** cycles |
| `INTLV` (`pto.vstx2`) | **12** cycles |
| `MRG4CHN`, `MRG2CHN` | **9** cycles |

### Gather, scatter, and special addressing

| PTO op | A5-level | Latency |
|--------|----------|-------------------|
| `pto.vgather2` | `RV_VGATHER2` | **27–28** cycles (pattern-dependent) |
| `pto.vgatherb` | `RV_VGATHERB` | **~21** cycles issue→retire |
| `pto.vgather2_bc` | (broadcast gather) | **27–28** cycles (same as **`pto.vgather2`**) |
| `pto.vscatter` | `RV_VSCATTER` | **~17** cycles for **`Dtype: B16`** |

### Strided loads/stores, unaligned ops, alignment state

Ops such as **`pto.vldas`**, **`pto.vldus`**, **`pto.vsld`**, **`pto.vsldb`**, **`pto.vsst`**, **`pto.vsstb`**, **`pto.vsta`**, **`pto.vstas`**, **`pto.vstar`**, **`pto.vstu`**, **`pto.vstus`**, **`pto.vstur`**: **9** cycles (same vector load/store pipe family as contiguous `RV_VLD` / `RV_VST` unless listed otherwise above).

### Dual-issue vs DMA

DMA **`TLOAD` / `TSTORE`** (global memory ↔ UB) use **MTE** pipes, not `RV_VLD`/`RV_VST`. **MTE2** `MOV_*` latency is not the same as vector `RV_VLD` latency (see `02-dma-copy.md` for GM↔UB movement).

---

## Contiguous Loads

### `pto.vlds`

- **syntax:** `%result = pto.vlds %source[%offset] {dist = "DIST"} : !pto.ptr<T, ub> -> !pto.vreg<NxT>`
- **semantics:** Vector load with distribution mode.
- **inputs:**
  `%source` is the UB base address, `%offset` is the load displacement, and
  `DIST` selects the distribution mode.
- **outputs:**
  `%result` is the loaded vector register value.
- **constraints and limitations:**
  The effective address MUST satisfy the alignment rule of the selected
  distribution mode. `NORM` reads one full vector footprint. Broadcast,
  upsample, downsample, unpack, split-channel, and deinterleave modes change
  how memory bytes are mapped into destination lanes, but they do not change the
  fact that the source is UB memory. PTO surface exposes load `dist` as family
  tokens, and each family only supports the element widths listed below.

**Distribution families:**

| Family | Allowed element widths | C semantics | Latency |
|------|-------------|-------------|-------------|
| `NORM` | width-agnostic | `dst[i] = UB[base + i * sizeof(T)]` | **9** cycles |
| `BRC` | `b8`, `b16`, `b32` | `dst[i] = UB[base]` for all `i` | **9** cycles |
| `US` | `b8`, `b16` | `dst[2*i] = dst[2*i+1] = UB[base + i]` | **9** cycles |
| `DS` | `b8`, `b16` | `dst[i] = UB[base + 2*i]` | **9** cycles |
| `UNPK` | `b8`, `b16`, `b32` | Expand packed source data into wider lanes | **9** cycles |
| `BRC_BLK` | width-agnostic | Block-replicate load path; simulator logs may print `dist:BRC_BLK` | **9** cycles |
| `E2B` | `b16`, `b32` | Load element groups and expand them into byte-oriented lane layout | **9** cycles |
| `UNPK4` | `b8` | Unpack 4-way packed `b8` source groups into destination lanes | **9** cycles |
| `SPLT4CHN` | `b8` | Split 4-channel interleaved source into one channel plane | **9** cycles |
| `SPLT2CHN` | `b8`, `b16` | Split 2-channel interleaved source into one channel plane | **9** cycles |

`pto.vlds` currently covers only single-result load families. Dual-result
deinterleave forms are modeled separately in PTO surface as
[`pto.vldsx2`](#ptovldsx2): `BDINTLV` is the block-deinterleave family, while
`DINTLV` is the element-width-sensitive deinterleave family.

**Example — Contiguous load:**
```mlir
%v = pto.vlds %ub[%offset] {dist = "NORM"} : !pto.ptr<f32, ub> -> !pto.vreg<64xf32>
```

**Example — Broadcast scalar to all lanes:**
```mlir
%v = pto.vlds %ub[%c0] {dist = "BRC"} : !pto.ptr<f32, ub> -> !pto.vreg<64xf32>
```

---

### `pto.vldas`

- **syntax:** `%result = pto.vldas %source : !pto.ptr<T, ub> -> !pto.align`
- **semantics:** Prime alignment buffer for subsequent unaligned load.
- **inputs:**
  `%source` is the UB address whose surrounding aligned block seeds the load
  alignment state.
- **outputs:**
  `%result` is the initialized load-alignment state.
- **constraints and limitations:**
  This op is the required leading operation for a `pto.vldus` stream using the
  same alignment state. The source address itself need not be 32-byte aligned;
  hardware truncates it to the aligned block boundary for the priming load.
- **Latency:** **9** cycles.

---

### `pto.vldus`

- **syntax:** `%result, %align_out = pto.vldus %source, %align : !pto.ptr<T, ub>, !pto.align -> !pto.vreg<NxT>, !pto.align`
- **semantics:** Unaligned load using primed align state.
- **inputs:**
  `%source` is the current UB address and `%align` is the incoming load
  alignment state primed by `pto.vldas` or a prior `pto.vldus`.
- **outputs:**
  `%result` is the assembled vector value and `%align_out` is the updated
  alignment state.
- **constraints and limitations:**
  A matching `pto.vldas` MUST appear before the first dependent `pto.vldus`
  stream in the same vector loop. The installed no-post A5 interface keeps a
  struct-shaped internal return for lowering convenience, but its no-post
  `base` field is not meaningful user-visible state. VPTO therefore hides that
  value and only exposes the updated align carrier. Reusing the original
  `%source` starts a new explicit access point; if the caller wants another
  no-post access, it should compute the next source pointer explicitly and pair
  it with the required align setup.
- **Latency:** **9** cycles.

**Unaligned load pattern:**
```mlir
%align = pto.vldas %ub : !pto.ptr<f32, ub> -> !pto.align
%vec, %align2 = pto.vldus %ub, %align : !pto.ptr<f32, ub>, !pto.align -> !pto.vreg<64xf32>, !pto.align
```

---

### `pto.init_align`

- **syntax:** `%result = pto.init_align : !pto.align`
- **semantics:** Initialize store-side align carrier state.
- **outputs:**
  `%result` is a fresh zero-initialized align carrier for store-side unaligned
  streams such as `pto.vstus`, `pto.vstur`, `pto.vstar`, `pto.vstas`, and
  `pto.pstu`.
- **constraints and limitations:**
  This op is for store-family initialization only. Unaligned load streams still
  start from `pto.vldas`.

---

## Dual Loads (Deinterleave)

### `pto.vldsx2`

- **syntax:** `%low, %high = pto.vldsx2 %source[%offset], "DIST" : !pto.ptr<T, ub>, index -> !pto.vreg<NxT>, !pto.vreg<NxT>`
- **semantics:** Dual load with deinterleave (AoS → SoA conversion).
- **inputs:**
  `%source` is the UB base pointer, `%offset` is the displacement, and `DIST`
  selects a dual-load/deinterleave layout.
- **outputs:**
  `%low` and `%high` are the two destination vectors.
- **constraints and limitations:**
  This family is only legal for interleave/deinterleave style distributions.
  The two outputs form an ordered pair, and that pairing MUST be preserved.
  PTO surface accepts deinterleave families. `BDINTLV` is element-width
  agnostic, while `DINTLV` supports only the element widths listed in the
  table.
- **latency:** `BDINTLV` / `DINTLV` are both **9** cycles.

**Distribution families:**

| Family | Allowed element widths | C semantics | Latency |
|------|-------------|-------------|-------------|
| `BDINTLV` | width-agnostic | Block deinterleave into two destination vectors | **9** cycles |
| `DINTLV` | `b8`, `b16`, `b32` | Deinterleave alternating elements into `%low` / `%high` | **9** cycles |

```c
// DINTLV family on 32-bit elements: deinterleave 32-bit elements
for (int i = 0; i < 64; i++) {
    low[i]  = UB[base + 8*i];       // even elements
    high[i] = UB[base + 8*i + 4];   // odd elements
}
```

**Example — Load interleaved XY pairs into separate X/Y vectors:**
```mlir
%x, %y = pto.vldsx2 %ub[%offset], "DINTLV" : !pto.ptr<f32, ub>, index -> !pto.vreg<64xf32>, !pto.vreg<64xf32>
```

### `pto.vsldb`

- **syntax:** `%result = pto.vsldb %source, %block_stride, %repeat_stride, %mask : !pto.ptr<T, ub>, i16, i16, !pto.mask<G> -> !pto.vreg<NxT>`
- **semantics:** Block-strided load for 2D tile access.
- **inputs:**
  `%source` is the UB base pointer. `%block_stride` and `%repeat_stride` are
  the two 16-bit fields of the hardware control word, and `%mask` controls
  which blocks participate.
- **outputs:**
  `%result` is the loaded vector.
- **constraints and limitations:**
  PTO surface does not expose the packed control word directly. If a block is
  masked off, the corresponding destination block is zeroed and MUST NOT raise
  an address overflow exception for that block.
- **Latency:** **9** cycles.

```c
// Block-strided load on 32-bit elements: one 32B block = 8 lanes.
for (int blk = 0; blk < 8; ++blk) {
    if (pg_b32[blk])
        dst_block[blk] = UB_block[base + repeat_stride + blk * block_stride];
    else
        dst_block[blk] = 0;
}
```

---

## Gather (Indexed) Loads

### `pto.vgather2`

- **syntax:** `%result = pto.vgather2 %source, %offsets, %active_lanes : !pto.ptr<T, ub>, !pto.vreg<NxI>, index -> !pto.vreg<NxT>`
- **semantics:** Indexed gather from UB.
- **inputs:**
  `%source` is the UB base pointer, `%offsets` provides per-lane element
  offsets, and `%active_lanes` bounds how many lanes participate.
- **outputs:**
  `%result` is the gathered vector.
- **constraints and limitations:**
  Only the first `%active_lanes` indices participate. The index element width
  and interpretation MUST match the selected gather form, and each effective
  address must satisfy that form's alignment rules.
- **Latency:** **27–28** cycles per `RV_VGATHER2`; throughput much lower than contiguous `RV_VLD` (see **Latency and throughput (A5)** at the start of this chapter).

```c
for (int i = 0; i < active_lanes; i++)
    dst[i] = UB[base + offsets[i] * sizeof(T)];
```

---

### `pto.vgatherb`

- **syntax:** `%result = pto.vgatherb %source, %offsets, %mask : !pto.ptr<T, ub>, !pto.vreg<NxI>, !pto.mask<b32> -> !pto.vreg<NxT>`
- **semantics:** Block gather load from UB.
- **inputs:**
  `%source` is the UB base pointer, `%offsets` is a `ui32` offset vector, and
  `%mask` is a `b32` predicate over the block-index lanes.
- **outputs:**
  `%result` is the gathered vector.
- **constraints and limitations:**
  This is a 32-byte block gather, not an element gather. `%source` MUST be
  32-byte aligned. Each participating `offsets[i]` is interpreted as a byte
  offset and MUST itself be 32-byte aligned. Only the low `VL/8` bytes of the
  offset vector are semantically valid; the effective block address is
  `block_addr[i] = offsets_u32[i] + base`. If a `b32` predicate position is
  false, the corresponding block does not participate in address coalescing,
  does not raise overflow on that block address, and the destination block is
  zero-filled.
- **Latency:** **~21** cycles issue→retire.

```c
for (int blk = 0; blk < VL / 32; ++blk) {
    if (pg_b32[blk])
        dst_block[blk] = UB_block[base + offsets_u32[blk]];
    else
        dst_block[blk] = 0;
}
```

---

### `pto.vgather2_bc`

- **syntax:** `%result = pto.vgather2_bc %source, %offsets, %mask : !pto.ptr<T, ub>, !pto.vreg<NxI>, !pto.mask<G> -> !pto.vreg<NxT>`
- **semantics:** Gather with broadcast, conditioned by mask.
- **inputs:**
  `%source` is the UB base pointer, `%offsets` contains gather indices, and
  `%mask` gates which lanes participate.
- **outputs:**
  `%result` is the gathered vector.
- **constraints and limitations:**
  This is a backward-compatible family. Masked-off lanes do not participate in
  address coalescing and do not trigger address overflow exceptions; their
  destination lanes are zero-filled. On the current PTO surface, `%offsets`
  uses 32-bit integer elements.
- **Latency:** **27–28** cycles (same as **`pto.vgather2`**).

---

## Contiguous Stores

### `pto.vsts`

- **syntax:** `pto.vsts %value, %dest[%offset], %mask {dist = "DIST"} : !pto.vreg<NxT>, !pto.ptr<T, ub>, !pto.mask<G>`
- **semantics:** Vector store with distribution mode.
- **inputs:**
  `%value` is the source vector, `%dest` is the UB base pointer, `%offset` is
  the displacement, `%mask` selects the active lanes or sub-elements, and
  `DIST` selects the store distribution.
- **outputs:**
  This op has no SSA result; it writes to UB memory.
- **constraints and limitations:**
  The effective destination address MUST satisfy the alignment rule of the
  selected store mode. The single-input `pto.vsts` family covers contiguous
  store, first-element-only store, packed store, and channel-merge store.
  Dual-input interleave store remains in `pto.vstsx2`. PTO surface exposes
  store `dist` as family tokens, and each family only supports the element
  widths listed below.

**Distribution families:**

| Family | Allowed element widths | C semantics | Latency |
|------|-------------|-------------|-------------|
| `NORM` | `b8`, `b16`, `b32` | `UB[base + i] = src[i]` | **9** cycles |
| `1PT` | `b8`, `b16`, `b32` | Only element 0 is written to the destination footprint | **9** cycles |
| `PK` | `b16`, `b32`, `b64` | Pack low half bits of each source element before store | **9** cycles |
| `PK4` | `b32` | Pack low 8 bits of each `b32` element before store | **9** cycles |
| `MRG4CHN` | `b8` | Merge 4 channel planes into an interleaved 4-channel layout | **9** cycles |
| `MRG2CHN` | `b8`, `b16` | Merge 2 channel planes into an interleaved 2-channel layout | **9** cycles |

**Example — Contiguous store:**
```mlir
pto.vsts %v, %ub[%offset], %mask {dist = "NORM"} : !pto.vreg<64xf32>, !pto.ptr<f32, ub>, !pto.mask<G>
```

---

## Dual Stores (Interleave)

### `pto.vstsx2`

- **syntax:** `pto.vstsx2 %low, %high, %dest[%offset], "DIST", %mask : !pto.vreg<NxT>, !pto.vreg<NxT>, !pto.ptr<T, ub>, index, !pto.mask<G>`
- **semantics:** Dual interleaved store (SoA → AoS conversion).
- **inputs:**
  `%low` and `%high` are the two source vectors, `%dest` is the UB base pointer,
  `%offset` is the displacement, `DIST` selects the interleave layout, and
  `%mask` gates the participating elements.
- **outputs:**
  This op has no SSA result; it writes an interleaved stream to UB.
- **constraints and limitations:**
  This family is only legal for interleave distributions. The two source
  vectors form an ordered pair, and the interleave semantics of that pair MUST
  be preserved. PTO surface accepts the `INTLV` family, which only supports the
  element widths listed below.
  be preserved. PTO surface accepts the `INTLV` family, which only supports the
  element widths listed below.
- **latency:** `INTLV` is **12** cycles。

**Distribution families:**

| Family | Allowed element widths | C semantics | Latency |
|------|-------------|-------------|-------------|
| `INTLV` | `b8`, `b16`, `b32` | Interleave `%low` / `%high` into one destination stream | **12** cycles |
| `INTLV` | `b8`, `b16`, `b32` |

```c
// INTLV family on 32-bit elements:
for (int i = 0; i < 64; i++) {
    UB[base + 8*i]     = low[i];
    UB[base + 8*i + 4] = high[i];
}
```

### `pto.vsstb`

- **syntax:** `pto.vsstb %value, %dest, %block_stride, %repeat_stride, %mask : !pto.vreg<NxT>, !pto.ptr<T, ub>, i16, i16, !pto.mask<G>`
- **semantics:** Block-strided store for 2D tile access.
- **inputs:**
  `%value` is the source vector, `%dest` is the UB base pointer,
  `%block_stride` and `%repeat_stride` are the two 16-bit fields of the
  hardware control word, and `%mask` controls block participation.
- **outputs:**
  This op writes UB memory and returns no SSA value.
- **constraints and limitations:**
  PTO surface does not expose the packed control word directly. Masked-off
  blocks MUST NOT issue memory writes.
- **Latency:** **9** cycles.

```c
// Block-strided store on 32-bit elements: one 32B block = 8 lanes.
for (int blk = 0; blk < 8; ++blk) {
    if (pg_b32[blk])
        UB_block[base + repeat_stride + blk * block_stride] = src_block[blk];
}
```

---

## Scatter (Indexed) Stores

### `pto.vscatter`

- **syntax:** `pto.vscatter %value, %dest, %offsets, %active_lanes : !pto.vreg<NxT>, !pto.ptr<T, ub>, !pto.vreg<NxI>, index`
- **semantics:** Indexed scatter to UB.
- **inputs:**
  `%value` is the source vector, `%dest` is the UB base pointer, `%offsets`
  provides per-lane or per-block indices, and `%active_lanes` bounds the active
  requests.
- **outputs:**
  This op writes UB memory and returns no SSA value.
- **constraints and limitations:**
  Only `b8`, `b16`, and `b32` element sizes are supported. The index vector
  must use a supported integer element type and layout for this family.
  Each computed address MUST be element-aligned. If two or more indices alias,
  only one write is guaranteed and the winning lane is implementation-defined.
- **Latency:** **~17** cycles for **`Dtype: B16`**.

```c
for (int i = 0; i < active_lanes; i++)
    UB[base + offsets[i] * sizeof(T)] = src[i];
```

---

## Alignment State Stores

### `pto.vstas`
- **syntax:** `pto.vstas %value, %dest, %offset : !pto.align, !pto.ptr<T, ub>, i32`
- **semantics:** Scalar-register-offset form of alignment-state flush.
- **inputs:**
  `%value` is the pending store-alignment state, `%dest` is the UB base
  pointer, and `%offset` is the scalar-register style displacement.
- **outputs:**
  This op writes buffered tail bytes to UB and returns no SSA value.
- **constraints and limitations:**
  This family flushes pending store-alignment state using an explicit scalar
  offset and keeps the scalar-offset form explicit. The incoming `%value`
  should come from `pto.init_align` or from a prior state-producing unaligned
  store op in the same stream. `%dest` and `%offset` together must identify the
  same logical flush point produced by the immediately preceding stateful
  unaligned-store step on that stream; using an unrelated base/offset pair is
  invalid even if `%value` itself came from the same stream.
- **Latency:** **9** cycles.

---

### `pto.vstar`
- **syntax:** `pto.vstar %value, %dest : !pto.align, !pto.ptr<T, ub>`
- **semantics:** Flush alignment state using the register-update form.
- **inputs:**
  `%value` is the pending store-alignment state and `%dest` is the UB base
  pointer.
- **outputs:**
  This op writes buffered tail bytes to UB and returns no SSA value.
- **constraints and limitations:**
  The implicit update state consumed by this flush MUST correspond to the same
  store stream that produced `%value`. The first store-side state in a stream
  should be created by `pto.init_align`.
- **Latency:** **9** cycles.

---

### `pto.vstar`

- **syntax:** `pto.vstar %value, %dest : !pto.align, !pto.ptr<T, ub>`
- **semantics:** Flush remaining alignment state.
- **inputs:**
  `%value` is the pending alignment/buffer state that still needs to be emitted,
  and `%dest` is the UB destination base pointer.
- **outputs:**
  No SSA result. The effect is a memory-side flush that writes the remaining
  buffered bytes to memory.
- **constraints and limitations:**
  This op terminates an unaligned-store sequence. It MUST be paired with a
  compatible prior state-producing store sequence so that the pending tail state
  is well-defined.
- **Latency:** **9** cycles.

---

## Stateful Store Ops

These ops make reference-updated state explicit as SSA results.

### `pto.vstus`

- **syntax:** `%align_out = pto.vstus %align_in, %offset, %value, %base : !pto.align, i32, !pto.vreg<NxT>, !pto.ptr<T, ub> -> !pto.align`
- **semantics:** No-post unaligned store with scalar offset.
- **inputs:**
  `%align_in` is the incoming store-alignment state, `%offset` is the scalar
  displacement, `%value` is the vector being stored, and `%base` is the UB base
  pointer.
- **outputs:**
  `%align_out` is the updated buffered-tail state.
- **constraints and limitations:**
  This is the scalar-offset stateful form of the unaligned store family. The
  scalar offset width MUST match the selected form, and a later flush op is
  still required. The first `%align_in` in the stream should come from
  `pto.init_align`. This op does not mean "store a full vector starting at
  `%base + %offset`". Instead, `%offset` describes how far the store stream
  advances at this step, and `%align_out` carries any residual tail that could
  not be committed yet. The no-post surface does not expose an updated base
  pointer. A later flush op must therefore use an explicit destination/offset
  pair that identifies the same logical flush point as this `pto.vstus`.
- **Latency:** **9** cycles.

---

### `pto.vstur`

- **syntax:** `%align_out = pto.vstur %align_in, %value, %base, "MODE" : !pto.align, !pto.vreg<NxT>, !pto.ptr<T, ub> -> !pto.align`
- **semantics:** Unaligned store with residual flush and SPR-AR-driven state update.
- **inputs:**
  `%align_in` is the incoming store-alignment state, `%value` is the vector to
  store, `%base` is the UB base pointer, and `MODE` selects whether the
  hardware updates `SPR AR` after the store.
- **outputs:**
  `%align_out` is the updated residual state after the current partial store.
- **constraints and limitations:**
  The effective address is `base + AR`, where `AR` is the hardware SPR state
  carried outside SSA. `POST_UPDATE` means hardware may advance `SPR AR`
  according to the fixed `SPR SQZN` configuration; `NO_POST_UPDATE` preserves
  the current `SPR AR` value. This form exposes only the evolving residual
  align-state in SSA; it does not by itself guarantee that all buffered bytes
  have reached memory. A compatible final flush is still required unless the
  surrounding sequence is known to be complete. Independent sequences typically
  begin from `AR = 0`; if the surrounding program does not already guarantee
  that, the hardware sequence should clear `SPR AR` before the first dependent
  `pto.vstur`. The first `%align_in` in the stream should come from
  `pto.init_align`. `pto.vstur` also consumes the fixed `SPR SQZN` state, so a
  preceding squeeze producer such as `pto.vsqz` / `pto.vusqz` MUST establish
  the byte count before the store. `MODE` MUST be one of `POST_UPDATE` or
  `NO_POST_UPDATE`.
- **Latency:** **9** cycles.
