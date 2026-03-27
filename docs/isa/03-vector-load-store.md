# 3. Vector Load/Store

> **Category:** UB ↔ Vector Register data movement
> **Pipeline:** PIPE_V (Vector Core)

Vector loads move data from Unified Buffer (UB) to vector registers (`vreg`). Vector stores move data from `vreg` back to UB. All vector compute operates only on `vreg` — UB is the staging area between DMA and compute.

---

## Contiguous Loads

### `pto.vlds`

- **syntax:** `%result = pto.vlds %source[%offset] {dist = "DIST"} : !pto.ptr<T, ub> -> !pto.vreg<NxT>`
- **semantics:** Vector load with distribution mode.

**Distribution modes:**

| Mode | Description | C Semantics |
|------|-------------|-------------|
| `NORM` | Contiguous 256B load | `dst[i] = UB[base + i * sizeof(T)]` |
| `BRC_B8/B16/B32` | Broadcast single element | `dst[i] = UB[base]` for all i |
| `US_B8/B16` | Upsample (duplicate each element) | `dst[2*i] = dst[2*i+1] = UB[base + i]` |
| `DS_B8/B16` | Downsample (every 2nd element) | `dst[i] = UB[base + 2*i]` |
| `UNPK_B8/B16/B32` | Unpack (zero-extend to wider type) | `dst_i32[i] = (uint32_t)UB_i16[base + 2*i]` |
| `SPLT4CHN_B8` | Split 4-channel (RGBA → R plane) | Extract every 4th byte |
| `SPLT2CHN_B8/B16` | Split 2-channel | Extract every 2nd element |
| `DINTLV_B32` | Deinterleave 32-bit | Even elements only |
| `BLK` | Block load | Blocked access pattern |

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

---

### `pto.vldus`

- **syntax:** `%result, %align_out, %base_out = pto.vldus %source, %align : !pto.ptr<T, ub>, !pto.align -> !pto.vreg<NxT>, !pto.align, !pto.ptr<T, ub>`
- **semantics:** Unaligned load using primed align state.

**Unaligned load pattern:**
```mlir
%align = pto.vldas %ub : !pto.ptr<f32, ub> -> !pto.align
%vec, %align2, %ub2 = pto.vldus %ub, %align : !pto.ptr<f32, ub>, !pto.align -> !pto.vreg<64xf32>, !pto.align, !pto.ptr<f32, ub>
```

---

## Dual Loads (Deinterleave)

### `pto.vldx2`

- **syntax:** `%low, %high = pto.vldx2 %source[%offset], "DIST" : !pto.ptr<T, ub>, index -> !pto.vreg<NxT>, !pto.vreg<NxT>`
- **semantics:** Dual load with deinterleave (AoS → SoA conversion).

**Distribution modes:** `DINTLV_B8`, `DINTLV_B16`, `DINTLV_B32`, `BDINTLV`

```c
// DINTLV_B32: deinterleave 32-bit elements
for (int i = 0; i < 64; i++) {
    low[i]  = UB[base + 8*i];       // even elements
    high[i] = UB[base + 8*i + 4];   // odd elements
}
```

**Example — Load interleaved XY pairs into separate X/Y vectors:**
```mlir
%x, %y = pto.vldx2 %ub[%offset], "DINTLV_B32" : !pto.ptr<f32, ub>, index -> !pto.vreg<64xf32>, !pto.vreg<64xf32>
```

---

## Strided Loads

### `pto.vsld`

- **syntax:** `%result = pto.vsld %source[%offset], "STRIDE" : !pto.ptr<T, ub> -> !pto.vreg<NxT>`
- **semantics:** Strided load with fixed stride pattern.

**Stride modes:** `STRIDE_S3_B16`, `STRIDE_S4_B64`, `STRIDE_S8_B32`, `STRIDE_S2_B64`

---

### `pto.vsldb`

- **syntax:** `%result = pto.vsldb %source, %offset, %mask : !pto.ptr<T, ub>, i32, !pto.mask -> !pto.vreg<NxT>`
- **semantics:** Block-strided load for 2D tile access.

---

## Gather (Indexed) Loads

### `pto.vgather2`

- **syntax:** `%result = pto.vgather2 %source, %offsets, %active_lanes : !pto.ptr<T, ub>, !pto.vreg<NxI>, index -> !pto.vreg<NxT>`
- **semantics:** Indexed gather from UB.

```c
for (int i = 0; i < active_lanes; i++)
    dst[i] = UB[base + offsets[i] * sizeof(T)];
```

---

### `pto.vgatherb`

- **syntax:** `%result = pto.vgatherb %source, %offsets, %active_lanes : !pto.ptr<T, ub>, !pto.vreg<NxI>, index -> !pto.vreg<NxT>`
- **semantics:** Byte-granularity indexed gather from UB.

```c
for (int i = 0; i < active_lanes; i++)
    dst[i] = UB[base + offsets[i]];  // byte-addressed
```

---

### `pto.vgather2_bc`

- **syntax:** `%result = pto.vgather2_bc %source, %offsets, %mask : !pto.ptr<T, ub>, !pto.vreg<NxI>, !pto.mask -> !pto.vreg<NxT>`
- **semantics:** Gather with broadcast, conditioned by mask.

---

## Contiguous Stores

### `pto.vsts`

- **syntax:** `pto.vsts %value, %dest[%offset], %mask {dist = "DIST"} : !pto.vreg<NxT>, !pto.ptr<T, ub>, !pto.mask`
- **semantics:** Vector store with distribution mode.

**Distribution modes:**

| Mode | Description | C Semantics |
|------|-------------|-------------|
| `NORM_B8/B16/B32` | Contiguous store | `UB[base + i] = src[i]` |
| `PK_B16/B32` | Pack/narrowing store | `UB_i16[base + 2*i] = truncate_16(src_i32[i])` |
| `MRG4CHN_B8` | Merge 4 channels (R,G,B,A → RGBA) | Interleave 4 planes |
| `MRG2CHN_B8/B16` | Merge 2 channels | Interleave 2 planes |

**Example — Contiguous store:**
```mlir
pto.vsts %v, %ub[%offset], %mask {dist = "NORM_B32"} : !pto.vreg<64xf32>, !pto.ptr<f32, ub>, !pto.mask
```

---

## Dual Stores (Interleave)

### `pto.vstx2`

- **syntax:** `pto.vstx2 %low, %high, %dest[%offset], "DIST", %mask : !pto.vreg<NxT>, !pto.vreg<NxT>, !pto.ptr<T, ub>, index, !pto.mask`
- **semantics:** Dual interleaved store (SoA → AoS conversion).

**Distribution modes:** `INTLV_B8`, `INTLV_B16`, `INTLV_B32`

```c
// INTLV_B32:
for (int i = 0; i < 64; i++) {
    UB[base + 8*i]     = low[i];
    UB[base + 8*i + 4] = high[i];
}
```

---

## Strided Stores

### `pto.vsst`

- **syntax:** `pto.vsst %value, %dest[%offset], "STRIDE" : !pto.vreg<NxT>, !pto.ptr<T, ub>`
- **semantics:** Strided store with fixed stride pattern.

---

### `pto.vsstb`

- **syntax:** `pto.vsstb %value, %dest, %offset, %mask : !pto.vreg<NxT>, !pto.ptr<T, ub>, i32, !pto.mask`
- **semantics:** Block-strided store for 2D tile access.

---

## Scatter (Indexed) Stores

### `pto.vscatter`

- **syntax:** `pto.vscatter %value, %dest, %offsets, %active_lanes : !pto.vreg<NxT>, !pto.ptr<T, ub>, !pto.vreg<NxI>, index`
- **semantics:** Indexed scatter to UB.

```c
for (int i = 0; i < active_lanes; i++)
    UB[base + offsets[i] * sizeof(T)] = src[i];
```

---

## Alignment State Stores

### `pto.vsta`

- **syntax:** `pto.vsta %value, %dest[%offset] : !pto.align, !pto.ptr<T, ub>, index`
- **semantics:** Flush alignment state to memory.

---

### `pto.vstas`

- **syntax:** `pto.vstas %value, %dest, %offset : !pto.align, !pto.ptr<T, ub>, i32`
- **semantics:** Flush alignment state with scalar offset.

---

### `pto.vstar`

- **syntax:** `pto.vstar %value, %dest : !pto.align, !pto.ptr<T, ub>`
- **semantics:** Flush remaining alignment state.

---

## Stateful Store Ops

These ops make reference-updated state explicit as SSA results.

### `pto.vstu`

- **syntax:** `%align_out, %offset_out = pto.vstu %align_in, %offset_in, %value, %base, "MODE" : !pto.align, index, !pto.vreg<NxT>, !pto.ptr<T, ub> -> !pto.align, index`
- **semantics:** Unaligned store with align + offset state update.

**Mode tokens:** `π_UPDATE`, `NO_POST_UPDATE`

---

### `pto.vstus`

- **syntax:** `%align_out, %base_out = pto.vstus %align_in, %offset, %value, %base, "MODE" : !pto.align, i32, !pto.vreg<NxT>, !pto.ptr<T, ub> -> !pto.align, !pto.ptr<T, ub>`
- **semantics:** Unaligned store with scalar offset and state update.

---

### `pto.vstur`

- **syntax:** `%align_out = pto.vstur %align_in, %value, %base, "MODE" : !pto.align, !pto.vreg<NxT>, !pto.ptr<T, ub> -> !pto.align`
- **semantics:** Unaligned store with residual flush and state update.
