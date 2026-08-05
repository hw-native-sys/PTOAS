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
| `RV_VST` / `RV_VSTI` | `dist:NORM_B8` / `NORM_B16` / `NORM_B32` | **9** |
| `RV_VGATHER2` | `Dtype: B32` | **27–28** |
| `RV_VGATHERB` | indexed byte gather | **~21** |
| `RV_VSCATTER` | `Dtype: B16` | **~17** |
| `RV_VADD` | F32 between UB-backed ops | **7** |

### `dist:` tokens (issue→retire)

Most **`dist:`** tokens are **9** issue→retire cycles. **`INTLV_B8` / `INTLV_B16` / `INTLV_B32`** on **`RV_VSTI`** are **12** cycles.

| `dist:` (as in log) | RV op | issue→retire (cycles) |
|---------------------|-------|----------------------|
| `DINTLV_B8` / `DINTLV_B16` / `DINTLV_B32` | `RV_VLDI` | **9** |
| `BRC_B8` / `BRC_B16` / `BRC_B32` | `RV_VLD` | **9** |
| `BRC_BLK` | `RV_VLD` | **9** |
| `INTLV_B8` / `INTLV_B16` / `INTLV_B32` | `RV_VSTI` | **12** |
| `UNPK_B8` / `UNPK_B16` / `UNPK_B32` | `RV_VLD` | **9** |
| `NORM_B8` / `NORM_B16` / `NORM_B32` | `RV_VSTI` | **9** |
| `PK_B16` / `PK_B32` / `PK_B64` / `PK4_B32` | `RV_VSTI` | **9** |
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
| `UNPK_B8` / `UNPK_B16` / `UNPK_B32` | **9** cycles |
| `DINTLV_B8` / `DINTLV_B16` / `DINTLV_B32` | **9** cycles (`RV_VLDI`) |
| `BRC_B8` / `BRC_B16` / `BRC_B32` | **9** cycles (`RV_VLD`) |
| `BRC_BLK` | **9** cycles as **`dist:BRC_BLK`** on `RV_VLD` |
| `BDINTLV` | **9** cycles |
| `US_B8` / `US_B16`, `DS_B8` / `DS_B16`, `SPLT4CHN`, `SPLT2CHN_B8` / `SPLT2CHN_B16` | **9** cycles |

### PTO `dist` summary (stores)

| PTO `dist` (store) | Latency |
|--------------------|-------------------|
| `NORM_B8` / `NORM_B16` / `NORM_B32` | **9** cycles (`RV_VSTI`) |
| `PK_B16` / `PK_B32` / `PK_B64` / `PK4_B32` | **9** cycles |
| `INTLV_B8` / `INTLV_B16` / `INTLV_B32` (`pto.vstsx2`) | **12** cycles |
| `MRG4CHN_B8`, `MRG2CHN_B8`, `MRG2CHN_B16` | **9** cycles (surface retained; current A5 hardware still reports them unsupported at validation time) |

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

  Post-update form:

  `%result, %updated_base = pto.vlds %source[%offset] {dist = "DIST"} : !pto.ptr<T, ub> -> !pto.vreg<NxT>, !pto.ptr<T, ub>`
- **semantics:** Vector load with distribution mode.
- **inputs:**
  `%source` is the UB base address, `%offset` is the load displacement, and
  `DIST` selects the distribution mode.
- **outputs:**
  `%result` is the loaded vector register value. In the post-update form,
  `%updated_base` is the base pointer advanced according to `%offset`.
- **constraints and limitations:**
  The effective address MUST satisfy the alignment rule of the selected
  distribution mode. `NORM` reads one full vector footprint. Broadcast,
  upsample, downsample, unpack, split-channel, and deinterleave modes change
  how memory bytes are mapped into destination lanes, but they do not change the
  fact that the source is UB memory. PTO surface exposes load `dist` as family
  tokens, and each family only supports the element widths listed below.
  The optional `%updated_base` result must have the same pointer type as the
  base address operand.

**Distribution families:**

| Family | Allowed element widths | C semantics | Latency |
|------|-------------|-------------|-------------|
| `NORM` | width-agnostic | `dst[i] = UB[base + i * sizeof(T)]` | **9** cycles |
| `BRC_B8` / `BRC_B16` / `BRC_B32` | `b8`, `b16`, `b32` | `dst[i] = UB[base]` for all `i` | **9** cycles |
| `US_B8` / `US_B16` | `b8`, `b16` | `dst[2*i] = dst[2*i+1] = UB[base + i]` | **9** cycles |
| `DS_B8` / `DS_B16` | `b8`, `b16` | `dst[i] = UB[base + 2*i]` | **9** cycles |
| `UNPK_B8` / `UNPK_B16` / `UNPK_B32` | `b8`, `b16`, `b32` | Expand packed source data into wider lanes | **9** cycles |
| `BRC_BLK` | width-agnostic | Block-replicate load path; simulator logs may print `dist:BRC_BLK` | **9** cycles |
| `E2B_B16` / `E2B_B32` | `b16`, `b32` | Load element groups and expand them into byte-oriented lane layout | **9** cycles |
| `UNPK4` | `b8` | Unpack 4-way packed `b8` source groups into destination lanes | **9** cycles |
| `SPLT4CHN` | `b8` | Split 4-channel interleaved source into one channel plane | **9** cycles |
| `SPLT2CHN_B8` / `SPLT2CHN_B16` | `b8`, `b16` | Split 2-channel interleaved source into one channel plane | **9** cycles |

`pto.vlds` currently covers only single-result load families. Dual-result
deinterleave forms are modeled separately in PTO surface as
[`pto.vldsx2`](#ptovldsx2): `BDINTLV` is the block-deinterleave family, while
`DINTLV_B8` / `DINTLV_B16` / `DINTLV_B32` are the element-width-sensitive
deinterleave forms.

**Example — Contiguous load:**
```mlir
%v = pto.vlds %ub[%offset] {dist = "NORM"} : !pto.ptr<f32, ub> -> !pto.vreg<64xf32>
```

**Example — Broadcast scalar to all lanes:**
```mlir
%v = pto.vlds %ub[%c0] {dist = "BRC_B32"} : !pto.ptr<f32, ub> -> !pto.vreg<64xf32>
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
- **post-update syntax:** `%result, %align_out, %base_out = pto.vldus %source, %align, %increment : !pto.ptr<T, ub>, !pto.align, index -> !pto.vreg<NxT>, !pto.align, !pto.ptr<T, ub>`
- **semantics:** Unaligned load using primed align state.
- **inputs:**
  `%source` is the current UB address and `%align` is the incoming load
  alignment state primed by `pto.vldas` or a prior `pto.vldus`. In the
  post-update form, `%increment` is the number of `T` elements by which the
  base advances after this access.
- **outputs:**
  `%result` is the assembled vector value and `%align_out` is the updated
  alignment state. The post-update form additionally returns `%base_out`,
  equivalent to `%source` advanced by `%increment` elements.
- **constraints and limitations:**
  A matching `pto.vldas` MUST appear before the first dependent `pto.vldus`
  stream in the same vector loop. The installed no-post A5 interface keeps a
  struct-shaped internal return for lowering convenience, but its no-post
  `base` field is not meaningful user-visible state. VPTO therefore hides that
  value and only exposes the updated align carrier. Reusing the original
  `%source` starts a new explicit access point; if the caller wants another
  no-post access, it should compute the next source pointer explicitly and pair
  it with the required align setup. The align carrier already has its own SSA
  update chain in both forms; enabling base post-update does not change that
  chain's semantics.
- **Latency:** **9** cycles.

**Unaligned load pattern:**
```mlir
%align = pto.vldas %ub : !pto.ptr<f32, ub> -> !pto.align
%vec, %align2 = pto.vldus %ub, %align : !pto.ptr<f32, ub>, !pto.align -> !pto.vreg<64xf32>, !pto.align
%vec2, %align3, %next = pto.vldus %ub, %align2, %c64 : !pto.ptr<f32, ub>, !pto.align, index -> !pto.vreg<64xf32>, !pto.align, !pto.ptr<f32, ub>
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

- **syntax:** `%low, %high [, %updated_base] = pto.vldsx2 %source[%offset], "DIST" : !pto.ptr<T, ub>, index -> !pto.vreg<NxT>, !pto.vreg<NxT> [, !pto.ptr<T, ub>]`
- **semantics:** Dual load with deinterleave (AoS → SoA conversion).
- **inputs:**
  `%source` is the UB base pointer, `%offset` is the displacement, and `DIST`
  selects a dual-load/deinterleave layout.
- **outputs:**
  `%low` and `%high` are the two destination vectors. If requested,
  `%updated_base` is the source pointer advanced by `%offset`.
- **constraints and limitations:**
  This family is only legal for interleave/deinterleave style distributions.
  The two outputs form an ordered pair, and that pairing MUST be preserved.
  PTO surface accepts deinterleave families. `BDINTLV` is element-width
  agnostic, while `DINTLV_B8` / `DINTLV_B16` / `DINTLV_B32` support only the
  element widths listed in the
  table.
- **latency:** `BDINTLV` / `DINTLV_B8` / `DINTLV_B16` / `DINTLV_B32` are all
  **9** cycles.

**Distribution families:**

| Family | Allowed element widths | C semantics | Latency |
|------|-------------|-------------|-------------|
| `BDINTLV` | width-agnostic | Block deinterleave into two destination vectors | **9** cycles |
| `DINTLV_B8` / `DINTLV_B16` / `DINTLV_B32` | `b8`, `b16`, `b32` | Deinterleave alternating elements into `%low` / `%high` | **9** cycles |

```c
// DINTLV_B32 family on 32-bit elements: deinterleave 32-bit elements
for (int i = 0; i < 64; i++) {
    low[i]  = UB[base + 8*i];       // even elements
    high[i] = UB[base + 8*i + 4];   // odd elements
}
```

**Example — Load interleaved XY pairs into separate X/Y vectors:**
```mlir
%x, %y = pto.vldsx2 %ub[%offset], "DINTLV_B32" : !pto.ptr<f32, ub>, index -> !pto.vreg<64xf32>, !pto.vreg<64xf32>
```

### `pto.vsldb`

- **syntax:** `%result [, %updated_base] = pto.vsldb %source, %block_stride, %repeat_stride, %mask : !pto.ptr<T, ub>, i16, i16, !pto.mask<G> -> !pto.vreg<NxT> [, !pto.ptr<T, ub>]`
- **semantics:** Block-strided load for 2D tile access.
- **inputs:**
  `%source` is the UB base pointer. `%block_stride` and `%repeat_stride` are
  the two 16-bit fields of the hardware control word, and `%mask` controls
  which blocks participate.
- **outputs:**
  `%result` is the loaded vector. If requested, `%updated_base` is the source
  pointer advanced by `%repeat_stride` 32-byte blocks.
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

- **syntax:** `%result = pto.vgather2 %source, %offsets, %mask : !pto.ptr<T, ub>, !pto.vreg<NxI>, !pto.mask<G> -> !pto.vreg<NxT>`
- **semantics:** Indexed gather from UB.
- **inputs:**
  `%source` is the UB base pointer, `%offsets` provides one unsigned element
  index for each logical gather lane, and `%mask` selects those logical
  index/result lanes.
- **outputs:**
  `%result` is the gathered vector.
- **addressing:**
  Each active lane computes its UB byte address from the source element width:

  ```text
  addr[i] = byte_address(%source) + unsigned(%offsets[i]) * sizeof(source element)
  ```

  For `i8/ui8` sources, `sizeof(source element) == 1`; for 16-bit sources it is
  `2`; for 32-bit sources it is `4`. The computed address must be aligned for
  the source element type. For `i8/ui8` sources, each loaded 8-bit payload is
  zero-extended into a 16-bit result lane.
- **semantic pseudocode:**

  ```text
  for i in lanes:
    if mask[i]:
      value = UB_load(source_element_type, addr[i])
      if source_element_type is i8 or ui8:
        result[i] = zero_extend_to_16_bits(value)
      else:
        result[i] = value
    else:
      result[i] = 0
  ```
- **constraints and limitations:**
  Only masked-on indices participate. The index element width
  and interpretation MUST match the selected gather form, and each effective
  address must satisfy that form's alignment rules.
  Supported forms are:
  `i8/ui8 -> i16/ui16` with `!pto.vreg<128xui16>` offsets and
  `!pto.mask<b16>`; `i16/ui16/f16/bf16 -> same type` with
  `!pto.vreg<128xui16>` offsets and `!pto.mask<b16>`; and
  `i32/ui32/f32 -> same type` with `!pto.vreg<64xui32>` offsets and
  `!pto.mask<b32>`. Signless integer offsets are accepted as storage-compatible
  aliases for the unsigned offset register payload.
- **Latency:** **27–28** cycles per `RV_VGATHER2`; throughput much lower than contiguous `RV_VLD` (see **Latency and throughput (A5)** at the start of this chapter).

```c
for (int i = 0; i < N; i++)
    if (mask[i])
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
- **post-update syntax:** `%updated_dest = pto.vsts %value, %dest[%offset], %mask {dist = "DIST"} : !pto.vreg<NxT>, !pto.ptr<T, ub>, !pto.mask<G> -> !pto.ptr<T, ub>`
- **semantics:** Vector store with distribution mode.
- **inputs:**
  `%value` is the source vector, `%dest` is the UB base pointer, `%offset` is
  the displacement, `%mask` is the predicate operand, and
  `DIST` selects the store distribution.
- **outputs:**
  The normal form has no SSA result and writes to UB memory. In the post-update
  form, `%updated_dest` is the destination pointer advanced according to
  `%offset`.
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
| `NORM_B8` / `NORM_B16` / `NORM_B32` | `b8`, `b16`, `b32` | `UB[base + i] = src[i]` | **9** cycles |
| `1PT_B8` / `1PT_B16` / `1PT_B32` | `b8`, `b16`, `b32` | Only element 0 is written to the destination footprint; the predicate register is ignored. | **9** cycles |
| `PK_B16` | `b16` | Pack the source vector, extract the lower half bits of all elements, and only store the active elements. The predicate is interpreted for 16-bit data. | **9** cycles |
| `PK_B32` | `b32` | Pack the source vector, extract the lower half bits of all elements, and only store the active elements. The predicate is interpreted for 32-bit data. | **9** cycles |
| `PK_B64` | `b64` | Pack the source vector, extract the lower half bits of all elements, and only store the active elements. The predicate is interpreted for 64-bit data. | **9** cycles |
| `PK4_B32` | `b32` | Pack the source vector, extract the lower 8 bits of all elements, and only store the active elements. The predicate is interpreted for 32-bit data. | **9** cycles |
| `MRG4CHN_B8` | `b8` | Merge 4 interleaved 8-bit channels within each 32B block; the predicate is interpreted for 32-bit data and applies after channel merge. | **9** cycles |
| `MRG2CHN_B8` / `MRG2CHN_B16` | `b8`, `b16` | Merge 2 interleaved channels within each 32B block; for `MRG2CHN_B8` the predicate is interpreted for 16-bit data, and for `MRG2CHN_B16` it is interpreted for 32-bit data; in both cases it applies after channel merge. | **9** cycles |

**Example — Contiguous store:**
```mlir
pto.vsts %v, %ub[%offset], %mask {dist = "NORM_B32"} : !pto.vreg<64xf32>, !pto.ptr<f32, ub>, !pto.mask<G>
```

---

## Dual Stores (Interleave)

### `pto.vstsx2`

- **syntax:** `pto.vstsx2 %low, %high, %dest[%offset], "DIST", %mask : !pto.vreg<NxT>, !pto.vreg<NxT>, !pto.ptr<T, ub>, index, !pto.mask<G>`
- **semantics:** Dual interleaved store (SoA → AoS conversion).
- **inputs:**
  `%low` and `%high` are the two source vectors, `%dest` is the UB base pointer,
  `%offset` is the displacement, `DIST` selects the interleave layout, and
  `%mask` is the predicate operand.
- **outputs:**
  This op has no SSA result; it writes an interleaved stream to UB.
- **constraints and limitations:**
  This family is only legal for interleave distributions. The two source
  vectors form an ordered pair, and the interleave semantics of that pair MUST
  be preserved. PTO surface accepts the `INTLV` family, which only supports the
  element widths listed below. For all `INTLV_*` distributions, the predicate
  register is ignored.
- **latency:** `INTLV` is **12** cycles。

**Distribution families:**

| Family | Allowed element widths | C semantics | Latency |
|------|-------------|-------------|-------------|
| `INTLV` | `b8`, `b16`, `b32` | Interleave `%low` / `%high` into one destination stream | **12** cycles |

```c
// INTLV family on 32-bit elements:
for (int i = 0; i < 64; i++) {
    UB[base + 8*i]     = low[i];
    UB[base + 8*i + 4] = high[i];
}
```

### `pto.vsstb`

- **syntax:** `pto.vsstb %value, %dest, %block_stride, %repeat_stride, %mask : !pto.vreg<NxT>, !pto.ptr<T, ub>, i16, i16, !pto.mask<G>`
- **post-update syntax:** `%updated_dest = pto.vsstb %value, %dest, %block_stride, %repeat_stride, %mask : !pto.vreg<NxT>, !pto.ptr<T, ub>, i16, i16, !pto.mask<G> -> !pto.ptr<T, ub>`
- **semantics:** Block-strided store for 2D tile access.
- **inputs:**
  `%value` is the source vector, `%dest` is the UB base pointer,
  `%block_stride` and `%repeat_stride` are the two 16-bit fields of the
  hardware control word, and `%mask` controls block participation.
- **outputs:**
  The normal form writes UB memory and returns no SSA value. In the post-update
  form, `%updated_dest` is the destination pointer advanced according to the
  packed stride control word.
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

- **syntax:** `pto.vscatter %value, %dest, %offsets, %mask : !pto.vreg<NxT>, !pto.ptr<T, ub>, !pto.vreg<NxI>, !pto.mask<G>`
- **semantics:** Indexed scatter to UB.
- **inputs:**
  `%value` is the source vector, `%dest` is the UB base pointer, `%offsets`
  provides unsigned element indices relative to `%dest`, and `%mask` selects
  the active requests. The legal type combinations are:

  | Value and destination type | Offset type | Mask type | Requests |
  | --- | --- | --- | --- |
  | `b8` (`!pto.vreg<256xT>`) | `!pto.vreg<128xui16>` or `!pto.vreg<128xi16>` | `!pto.mask<b16>` | 128 |
  | `b16` (`!pto.vreg<128xT>`) | `!pto.vreg<128xui16>` or `!pto.vreg<128xi16>` | `!pto.mask<b16>` | 128 |
  | `b32` (`!pto.vreg<64xT>`) | `!pto.vreg<64xui32>` or `!pto.vreg<64xi32>` | `!pto.mask<b32>` | 64 |

  Signless offset element types have the same unsigned interpretation as the
  corresponding `ui16` or `ui32` type.
- **outputs:**
  This op writes UB memory and returns no SSA value.
- **constraints and limitations:**
  Only `b8`, `b16`, and `b32` element sizes are supported, and `%dest` must have
  the same element type as `%value`. Each destination address is
  `%dest + %offsets[i]` in elements, equivalently
  `byte_address(%dest) + unsigned(%offsets[i]) * sizeof(T)`, and MUST be
  element-aligned. For `b8`, request `i` stores `%value[2*i]`; odd-numbered
  source bytes are ignored. If two or more active indices are equal, only one
  write is guaranteed and the winning request is implementation-defined.
- **Latency:** **~17** cycles for **`Dtype: B16`**.

```c
for (int i = 0; i < num_requests; i++)
    if (mask[i])
        dest[unsigned(offsets[i])] =
            sizeof(T) == 1 ? value[2 * i] : value[i];
```

---

## SPR State Ops

### `pto.sprclr`

- **syntax:** `pto.sprclr "AR"`
- **semantics:** Clear the hardware SPR AR state used by AR-driven unaligned
  store forms.
- **constraints and limitations:**
  Only `"AR"` is supported.

---

### `pto.sprsti`

- **syntax:**
  - normal: `pto.sprsti "AR", %dest[%offset] : !pto.ptr<ui32, ub>, i32`
  - post-update: `%next = pto.sprsti "AR", %dest[%offset] : !pto.ptr<ui32, ub>, i32 -> !pto.ptr<ui32, ub>`
- **semantics:** Store SPR AR to UB using a signed 8-bit immediate offset.
- **inputs:**
  `%dest` is the UB base pointer and `%offset` counts 4-byte words.
- **outputs:**
  The normal form returns no SSA value. The post-update form returns `%next`,
  which advances `%dest` by `4 * %offset` bytes.
- **constraints and limitations:**
  Only `"AR"` is supported. `%dest` must be a `ui32` or signless `i32` UB
  pointer. `%offset` must be a constant signed 8-bit `i32`.

---

### `pto.sprsts`

- **syntax:**
  - normal: `pto.sprsts "AR", %dest[%offset] : !pto.ptr<ui32, ub>, i32`
  - post-update: `%next = pto.sprsts "AR", %dest[%offset] : !pto.ptr<ui32, ub>, i32 -> !pto.ptr<ui32, ub>`
- **semantics:** Store SPR AR to UB using a scalar-register offset.
- **inputs:**
  `%dest` is the UB base pointer and `%offset` is the scalar offset in bytes.
- **outputs:**
  The normal form returns no SSA value. The post-update form returns `%next`,
  which advances `%dest` by `%offset` bytes.
- **constraints and limitations:**
  Only `"AR"` is supported. `%dest` must be a `ui32` or signless `i32` UB
  pointer.

---

## Alignment State Stores

### `pto.vstas`
- **syntax:**
  - normal: `pto.vstas %value, %dest, %offset : !pto.align, !pto.ptr<T, ub>, i32`
  - post-update: `%next = pto.vstas %value, %dest, %offset : !pto.align, !pto.ptr<T, ub>, i32 -> !pto.ptr<T, ub>`
- **semantics:** Scalar-register-offset form of alignment-state flush.
- **inputs:**
  `%value` is the pending store-alignment state, `%dest` is the UB base
  pointer, and `%offset` is the displacement in destination elements.
- **outputs:**
  This op writes buffered tail bytes to UB. The normal form returns no SSA
  value. The post-update form returns `%next`, which advances `%dest` by
  `%offset` destination elements.
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
- **post-update syntax:** `%align_out, %base_out = pto.vstus %align_in, %offset, %value, %base : !pto.align, i32, !pto.vreg<NxT>, !pto.ptr<T, ub> -> !pto.align, !pto.ptr<T, ub>`
- **semantics:** Unaligned store with scalar stream advance and optional base post-update.
- **inputs:**
  `%align_in` is the incoming store-alignment state, `%offset` is the number of
  `T` elements by which the store stream advances, `%value` is the vector being
  stored, and `%base` is the UB base pointer.
- **outputs:**
  `%align_out` is the updated buffered-tail state. In the post-update form,
  `%base_out` is `%base` advanced by `%offset` elements.
- **constraints and limitations:**
  This is the scalar-offset stateful form of the unaligned store family. The
  scalar offset width MUST match the selected form, and a later flush op is
  still required. The first `%align_in` in the stream should come from
  `pto.init_align`. This op does not mean "store a full vector starting at
  `%base + %offset`". Instead, `%offset` describes how far the store stream
  advances at this step, and `%align_out` carries any residual tail that could
  not be committed yet. The no-post surface does not expose an updated base
  pointer, while the post-update form returns that pointer directly. The align
  carrier remains an independent SSA state chain in either form. A later flush
  op must use a destination/offset pair that identifies the same logical flush
  point as this `pto.vstus` stream.
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
