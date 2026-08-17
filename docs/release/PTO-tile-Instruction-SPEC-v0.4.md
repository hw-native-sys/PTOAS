# PTO Tile Instruction SPEC (A5)

- v0.4: Initial PTO Tile Instruction SPEC covering core TileOps

[toc]

---

<a id="tile-01-tile-overview"></a>

## 1. Tile and PTO Tile Instruction Overview

> **Category:** Foundational concepts

This chapter introduces both the tile data model and the **Tile Instruction** surface that operates on it. Read this before any of the per-group Tile Instruction references.

---

### 1.1 What is PTO Tile Instruction

**PTO Tile Instruction** is a high-performance instruction library built on top of [PTO micro Instruction](PTO-micro-Instruction-SPEC-v0.4.md#micro-01-pipeline-sync). Each tile instruction encapsulates a tile-granular pattern — DMA between GM and on-chip buffers, vector arithmetic over a whole tile, reductions, broadcast / expansion, selection, padding — that internally expands to a sequence of micro-instruction primitives (`pto.vlds`, `pto.vsts`, `pto.vadd`, mask ops, sync flags, …).

For the kernel author this means:

- **Author at the tile level.** Use `pto.tload`, `pto.tadd`, `pto.trowsum`, etc., to express tile-granular DMA and compute without writing the underlying vector loop.
- **Drop down to micro instruction when needed.** Inside `pto.vecscope`, `pto.tile_buf_addr` lowers a tile handle to a UB pointer, so handwritten micro-instruction code can read and write the same on-chip data. The mixing pattern is documented in [§1.10](#110-mixing-pto-tile-instruction-and-pto-micro-instruction).
- **Predictable lowering.** Because every Tile Instruction is templated against micro instruction, a kernel that mixes Tile and micro can share scratch tiles, masks, and pipeline events with no representation gap.

The remaining chapters in this document cover the tile data types, pointer / view ops, DMA, compute families, and op-by-op syntax. The semantics below define the storage contract those ops share.

### 1.2 Tile Buffer Model

A **tile** is a bounded, rectangular 2-D sub-region of data that lives in **local on-chip memory** (UB, L0A, L0B, L0C, bias, or scaling buffer) and is consumed or produced by tile-level instructions. A tile is a storage object with an explicit lifetime and an explicit on-chip placement.

Tile Instruction models tiles as **tile buffers** of type `!pto.tile_buf<...>`. A tile buffer records:

- the **memory domain** (`loc`) — where the tile lives on chip;
- the **element type** (`dtype`) — how bits are interpreted;
- the **physical shape** (`rows`, `cols`) — how much storage the tile occupies;
- the **valid region** (`v_row`, `v_col`) — the populated sub-rectangle within the physical tile (may be `?` for runtime-dynamic);
- **layout and fractal** metadata (`blayout`, `slayout`, `fractal`, `pad`) — how elements are arranged in storage.

This differs from a global tensor:

- A `!pto.tensor_view` is a logical descriptor over **global memory (GM)** — shape information, no on-chip residency.
- A `!pto.partition_tensor_view` is a logical sub-window of a tensor view, still in GM.
- A `!pto.tile_buf` is the **local, on-chip** materialization of a partition — data placed in UB / L0 / bias / scaling buffers.

Data flow between these is explicit:

```
!pto.tensor_view  --partition_view-->  !pto.partition_tensor_view  --tload-->  !pto.tile_buf
       (GM)                                      (GM slice)                    (on-chip tile)
```

Placement, lifetime, and reuse affect both correctness and performance. `pto.alloc_tile` makes allocation explicit, and pipeline ordering is expressed through the synchronization primitives described in [`01-pipeline-sync.md`](PTO-micro-Instruction-SPEC-v0.4.md#micro-01-pipeline-sync).

**Explicit buffer lifetime example:**

```mlir
%a0 = pto.alloc_tile : !pto.tile_buf<vec, 16x16xf16>
%a1 = pto.alloc_tile : !pto.tile_buf<vec, 16x16xf16>

pto.tload ins(%pv0 : !pto.partition_tensor_view<16x16xf16>)
          outs(%a0 : !pto.tile_buf<vec, 16x16xf16>)
pto.tload ins(%pv1 : !pto.partition_tensor_view<16x16xf16>)
          outs(%a1 : !pto.tile_buf<vec, 16x16xf16>)
```

### 1.3 Hardware Memory Hierarchy

The Ascend NPU on-chip memory layout that tile buffers map onto:

```
GM (Global Memory)
|- MAT  (L1 Cache)
|  |- LEFT    (L0A — left matrix buffer)
|  |- RIGHT   (L0B — right matrix buffer)
|  |- ACC     (L0C — accumulator)
|  `- BIAS    (bias buffer)
`- VEC  (UB — unified buffer)
```

`loc` on a tile buffer selects one of these domains. The full enum (with mnemonics) is defined in [§2.6 AddressSpace](#26-addressspace); each tile ISA chapter calls out which `loc` domains are legal for the ops it covers.

### 1.4 Instruction Form

Most Tile Instruction ops use an explicit source/destination form. The destination tile buffer is named in `outs(...)` and is updated in place:

```mlir
pto.<op> ins(<src0>, <src1>, ... : <src0_type>, <src1_type>, ...)
         outs(<dst> : <dst_type>)
         [ {optional-attrs} ]
```

- Inputs appear inside `ins(...)` with their types.
- The output tile buffer appears inside `outs(...)`.
- Scalar operands (where applicable) are listed inside `ins(...)` alongside tile operands.
- Optional attributes follow as a trailing `{ ... }` block.

Synchronization, sub-view, and allocation ops may diverge from this pattern (for example `pto.alloc_tile` yields a tile-buffer handle, and `pto.subset` returns a view). Each chapter states the assembly format for its ops.

```mlir
pto.tadd ins(%a, %b : !pto.tile_buf<vec, 16x16xf16>, !pto.tile_buf<vec, 16x16xf16>)
         outs(%c : !pto.tile_buf<vec, 16x16xf16>)
```

### 1.5 Physical Shape vs Valid Region

Every tile buffer has two shape concepts:

- **Physical shape** `(rows, cols)` — the extent of backing storage; static and known when the tile buffer type is declared.
- **Valid region** `(v_row, v_col)` — the populated sub-rectangle; either static or dynamic (`?`).

The physical shape drives layout, fractal alignment, and buffer-size accounting. The valid region drives the iteration domain of compute and DMA ops. **Undefined behavior:** elements outside the valid region are padding — their contents must not be read.

When the valid region is dynamic (`v_row = ?` or `v_col = ?`), it is provided at `pto.alloc_tile` time (or updated later with `pto.set_validshape`). Most Tile Instruction ops use the destination valid region as the iteration domain; a few ops require all operands to share the same valid region.

### 1.6 Pipeline Association

Every Tile Instruction op is associated with a hardware pipeline in the Decoupled Access-Execute architecture:

| Pipeline | Symbol | Typical ops |
|----------|--------|------------|
| DMA inbound | `PIPE_MTE2` | `pto.tload` |
| DMA outbound | `PIPE_MTE3` | `pto.tstore` |
| Vector | `PIPE_V` | `pto.tadd`, `pto.tadds`, `pto.texp`, `pto.tcvt`, and the rest of the vector arithmetic set |
| Scalar | `PIPE_S` | scalar `arith`/`scf` ops interleaved with tile code |

Cross-pipeline data dependencies are ordered explicitly, either via the **Flag/Event** mechanism (`pto.set_flag`/`pto.wait_flag`) or the **Buffer-ID** mechanism (`pto.get_buf`/`pto.rls_buf`). See [`01-pipeline-sync.md`](PTO-micro-Instruction-SPEC-v0.4.md#micro-01-pipeline-sync) for the full semantics.

### 1.7 Scratch Operands and A2/A3 Compatibility

Some Tile Instruction ops carry an extra `%tmp` tile operand whose only purpose is to keep the operand list aligned with the corresponding A2/A3 PTO instruction interface. Examples include `pto.txor` / `pto.txors` ([Chapter 8](#tile-08-bitwise-shift-ops)) and `pto.tsel` / `pto.tsels` ([Chapter 11](#tile-11-selection-ops)).

`%tmp` exists for cross-arch interface compatibility — A5 templates may not materially use it, but it remains in the public op signature so the same Tile IR can be reused across A2/A3 and A5. Treat it as a required operand whose dtype/shape constraints are stated by the individual op page.

### 1.8 Conventions for Chapters 5–13

Unless an op page states otherwise, the chapters that follow assume:

- tile operands use `loc=vec`;
- tile layouts use `blayout=row_major` and `slayout=none_box`;
- valid bounds satisfy `v_row <= rows` and `v_col <= cols`;
- examples use the compact `!pto.tile_buf<loc, RxCxdtype[, valid=...]>` form. Omitted attributes carry their default values: `valid` = physical shape, `blayout=row_major`, `slayout=none_box`, `fractal=512`, `pad=0`.

The op pages call out any deviation from these conventions explicitly.

### 1.9 Minimal End-to-End Example

A minimal tile-level "load, add, store" kernel:

```mlir
// Build the GM view and partition it
%tv = pto.make_tensor_view %gm_ptr, shape = [%m, %n], strides = [%s0, %s1]
        : !pto.tensor_view<?x?xf16>
%pv = pto.partition_view %tv, offsets = [%c0, %c0], sizes = [%c16, %c16]
        : !pto.tensor_view<?x?xf16> -> !pto.partition_tensor_view<16x16xf16>

// Allocate on-chip tile buffers
%a = pto.alloc_tile : !pto.tile_buf<vec, 16x16xf16>
%b = pto.alloc_tile : !pto.tile_buf<vec, 16x16xf16>
%c = pto.alloc_tile : !pto.tile_buf<vec, 16x16xf16>

// DMA-in, compute, DMA-out
pto.tload ins(%pv  : !pto.partition_tensor_view<16x16xf16>) outs(%a : !pto.tile_buf<vec, 16x16xf16>)
pto.tload ins(%pv2 : !pto.partition_tensor_view<16x16xf16>) outs(%b : !pto.tile_buf<vec, 16x16xf16>)
pto.tadd  ins(%a, %b : !pto.tile_buf<vec, 16x16xf16>, !pto.tile_buf<vec, 16x16xf16>)
          outs(%c : !pto.tile_buf<vec, 16x16xf16>)
pto.tstore ins(%c : !pto.tile_buf<vec, 16x16xf16>)
           outs(%pv_out : !pto.partition_tensor_view<16x16xf16>)
```

Synchronization is omitted for clarity; for the real ordering contracts (`pto.set_flag`/`pto.wait_flag`, `pto.get_buf`/`pto.rls_buf`, `pto.pipe_barrier`) see [`01-pipeline-sync.md`](PTO-micro-Instruction-SPEC-v0.4.md#micro-01-pipeline-sync).

<a id="110-mixing-pto-tile-instruction-and-pto-micro-instruction"></a>

### 1.10 Mixing PTO Tile Instruction and PTO micro Instruction

PTO Tile Instruction and PTO micro Instruction can be authored side-by-side in the same kernel. The Tile Instruction surface owns tile placement and GM ↔ on-chip DMA; the micro surface owns vector-register compute inside `pto.vecscope`. The two surfaces meet through `pto.tile_buf_addr`, which converts a tile handle into a UB pointer that vector ops can consume.

This section presents a softmax kernel that uses both surfaces together, then walks through it.

#### Kernel Structure

The kernel follows a fixed shape that all mixed Tile + micro programs share:

1. Build `tensor_view` / `partition_view` descriptors for each GM operand.
2. Use `pto.alloc_tile` to allocate UB tiles with explicit static **size** and **address**.
3. Use `pto.tload` to move data from GM partitions into tiles.
4. Cross the **MTE2 → V** synchronization edge with `pto.set_flag` / `pto.wait_flag`.
5. Open a `pto.vecscope` region. Inside the scope:
   - Use `pto.tile_buf_addr` to lower each tile handle into a `!pto.ptr<..., ub>`.
   - Use `pto.vlds` / `pto.vsts` and the rest of the micro vector ops to read, compute, and write UB.
6. Cross the **V → MTE3** synchronization edge with `pto.set_flag` / `pto.wait_flag`.
7. Use `pto.tstore` to move tiles back to GM.

Two boundary rules govern this layout:

- Tile-domain ops (`pto.tload`, `pto.tstore`, `pto.tadd`, …) **must not appear inside** `pto.vecscope`.
- `pto.tile_buf_addr` is **only legal inside** `pto.vecscope` / `pto.strict_vecscope`.

The kernel also manually drives address allocation (`alloc_tile addr = ...`) and pipeline synchronization. Lowering with `--enable-insert-sync` is therefore disabled, and `--pto-level=level3` is used so that `alloc_tile` accepts an explicit address operand.

#### Kernel Listing

The listing below is an online softmax-update kernel reduced to the structurally interesting parts. Repeated descriptors and the deep online-softmax math are abbreviated with `// ...` so that the Tile / micro / sync boundaries stay visible.

```mlir
module attributes {pto.target_arch = "a5"} {
  func.func @online_softmax_update_kernel_2d(
      %arg0: !pto.ptr<f32, gm>,        // oldmax  (rows x 1)
      %arg1: !pto.ptr<f32, gm>,        // oldsum  (rows x 1)
      %arg2: !pto.ptr<f32, gm>,        // qk      (rows x 128)
      %arg3: !pto.ptr<f32, gm>,        // newmax  (rows x 1)
      %arg4: !pto.ptr<f32, gm>,        // newsum  (rows x 1)
      %arg5: !pto.ptr<f32, gm>,        // expmax  (rows x 1)
      %arg6: !pto.ptr<f32, gm>,        // out     (rows x 128)
      %arg7: i32, %arg8: i32) {        // %arg7 = seq_len, %arg8 = total_rows
    // -------- (1) GM views and partitions --------
    // Eight rows of the qk and out tensors are processed per block.
    %qk_view = pto.make_tensor_view %arg2,
        shape   = [%c1, %c1, %c1, %rows, %c128],
        strides = [%rows_x_128, %rows_x_128, %rows_x_128, %c128, %c1]
        : !pto.tensor_view<?x?x?x?x?xf32>
    %qk_part = pto.partition_view %qk_view,
        offsets = [%c0, %c0, %c0, %row_base, %c0],
        sizes   = [%c1, %c1, %c1, %row_count, %seq]
        : !pto.tensor_view<?x?x?x?x?xf32>
       -> !pto.partition_tensor_view<?x?x?x?x?xf32>
    // ... oldmax/oldsum/newmax/newsum/expmax/out views/partitions analogous ...

    // -------- (2) Tile allocation with static size and explicit UB address --------
    %qk_tile = pto.alloc_tile addr = %c256_i64
                              valid_row = %row_count valid_col = %seq
        : !pto.tile_buf<vec, 8x128xf32, valid=?x?>
    %out_tile = pto.alloc_tile addr = %c8448_i64
                               valid_row = %row_count valid_col = %seq
        : !pto.tile_buf<vec, 8x128xf32, valid=?x?>
    %oldmax_tile = pto.alloc_tile addr = %c0_i64 valid_row = %row_count
        : !pto.tile_buf<vec, 8x1xf32, valid=?x1, blayout=col_major>
    // ... oldsum/newmax/newsum/expmax tiles analogous (each at its own UB addr) ...

    // -------- (3) GM → tile DMA --------
    pto.tload ins(%qk_part     : !pto.partition_tensor_view<?x?x?x?x?xf32>)
              outs(%qk_tile    : !pto.tile_buf<vec, 8x128xf32, valid=?x?>)
    pto.tload ins(%oldmax_part : !pto.partition_tensor_view<?x?x?x?x?xf32>)
              outs(%oldmax_tile: !pto.tile_buf<vec, 8x1xf32, valid=?x1, blayout=col_major>)
    // ... oldsum tload analogous ...

    // -------- (4) MTE2 → V synchronization --------
    pto.set_flag ["PIPE_MTE2", "PIPE_V", "EVENT_ID0"]
    pto.wait_flag["PIPE_MTE2", "PIPE_V", "EVENT_ID0"]

    // -------- (5) Vector region: tile_buf_addr + micro compute --------
    pto.vecscope {
      // Lower tile handles to UB pointers.
      %ub_qk     = pto.tile_buf_addr %qk_tile
          : !pto.tile_buf<vec, 8x128xf32, valid=?x?>
         -> !pto.ptr<f32, ub>
      %ub_out    = pto.tile_buf_addr %out_tile
          : !pto.tile_buf<vec, 8x128xf32, valid=?x?>
         -> !pto.ptr<f32, ub>
      %ub_newmax = pto.tile_buf_addr %newmax_tile
          : !pto.tile_buf<vec, 8x1xf32, valid=?x1, blayout=col_major>
         -> !pto.ptr<f32, ub>
      // ... ub_oldmax / ub_oldsum / ub_newsum / ub_expmax analogous ...

      %active = pto.pset_b32 "PAT_ALL" : !pto.mask<b32>
      %one_mask, %_ = pto.plt_b32 %c1_i32 : i32 -> !pto.mask<b32>, i32

      scf.for %row = %c0 to %row_count step %c1 {
        // Online-softmax max/sum reduction (one row at a time).
        %row_qk     = arith.muli %row, %c128 : index
        %oldmax_bc  = pto.vlds %ub_oldmax[%row]
                          {dist = "BRC_B32"} : !pto.ptr<f32, ub> -> !pto.vreg<64xf32>
        %final_max, %final_sum = scf.for %chunk = %c0 to %c128 step %c64
            iter_args(%running_max = %oldmax_bc, %running_sum = %oldsum_bc)
            -> (!pto.vreg<64xf32>, !pto.vreg<64xf32>) {
          %base = arith.addi %row_qk, %chunk : index
          %vec  = pto.vlds %ub_qk[%base]
                      : !pto.ptr<f32, ub> -> !pto.vreg<64xf32>
          // ... running_max / running_sum update via vcmax / vexpdif / vmul / vadd ...
          scf.yield %merged_max, %merged_sum : !pto.vreg<64xf32>, !pto.vreg<64xf32>
        }

        // Persist the row-local results back to UB.
        pto.vsts %final_max, %ub_newmax[%row], %one_mask
                {dist = "1PT_B32"}
                : !pto.vreg<64xf32>, !pto.ptr<f32, ub>, !pto.mask<b32>

        // Second pass: write softmax output back into the qk tile's UB region.
        scf.for %chunk = %c0 to %c128 step %c64 {
          %base = arith.addi %row_qk, %chunk : index
          %vec  = pto.vlds %ub_qk[%base]
                      : !pto.ptr<f32, ub> -> !pto.vreg<64xf32>
          %exp  = pto.vexpdif %vec, %final_max, %chunk_mask, "ODD"
                      : !pto.vreg<64xf32>, !pto.vreg<64xf32>, !pto.mask<b32>
                     -> !pto.vreg<64xf32>
          %out  = pto.vdiv %exp, %final_sum, %chunk_mask
                      : !pto.vreg<64xf32>, !pto.vreg<64xf32>, !pto.mask<b32>
                     -> !pto.vreg<64xf32>
          pto.vsts %out, %ub_out[%base], %chunk_mask
                      : !pto.vreg<64xf32>, !pto.ptr<f32, ub>, !pto.mask<b32>
        }
      }
    }

    // -------- (6) V → MTE3 synchronization --------
    pto.set_flag ["PIPE_V", "PIPE_MTE3", "EVENT_ID0"]
    pto.wait_flag["PIPE_V", "PIPE_MTE3", "EVENT_ID0"]

    // -------- (7) Tile → GM DMA --------
    pto.tstore ins(%out_tile    : !pto.tile_buf<vec, 8x128xf32, valid=?x?>)
               outs(%out_part   : !pto.partition_tensor_view<?x?x?x?x?xf32>)
    pto.tstore ins(%newmax_tile : !pto.tile_buf<vec, 8x1xf32, valid=?x1, blayout=col_major>)
               outs(%newmax_part: !pto.partition_tensor_view<?x?x?x?x?xf32>)
    // ... newsum/expmax tstore analogous ...

    pto.barrier #pto.pipe<PIPE_ALL>
    return
  }
}
```

#### Code Walkthrough

The seven numbered comments in the listing above mark the seven steps from §Kernel Structure. The notes below highlight what each step contributes to the Tile / micro split.

**(1) GM views and partitions** — pure metadata. `pto.make_tensor_view` records the GM tensor's shape and strides; `pto.partition_view` carves out the per-block sub-window. Neither op moves data, and both stay outside `pto.vecscope`. The 5-D shape is a quirk of this kernel's layout convention; the boundary rules don't depend on rank.

**(2) `pto.alloc_tile` with static size and address** — declares the UB tile handles. The result type fixes the static physical shape (e.g. `8x128xf32`); `addr = %c256_i64` pins the tile to a specific UB byte offset; `valid_row = ...` / `valid_col = ...` carry the runtime valid extents (the `?` markers in `valid=?x?`). Because addresses are hand-assigned, this kernel compiles with `--pto-level=level3` and disables `--enable-insert-sync`.

**(3) `pto.tload`** — copies a GM partition into the UB tile. Runs on `PIPE_MTE2`. Stays in the Tile domain; it cannot appear inside `pto.vecscope`.

**(4) MTE2 → V flag handshake** — DMA inbound and the vector pipeline run asynchronously. The producer/consumer edge between `tload` and the upcoming `vecscope` must be made explicit with `pto.set_flag` / `pto.wait_flag`.

**(5) Vector region** — `pto.vecscope` opens a vector-execution region. The first thing inside is a series of `pto.tile_buf_addr` ops, each lowering a tile handle into a `!pto.ptr<f32, ub>`. From that point on the body is pure micro: `pto.vlds` reads UB into vregs, vector arithmetic / SFU / mask ops compute on vregs, and `pto.vsts` writes vregs back to UB. Tile ops are forbidden inside this region; `pto.tile_buf_addr` is forbidden outside.

**(6) V → MTE3 flag handshake** — mirror of step (4), this time gating the vector results visible to the outbound DMA.

**(7) `pto.tstore`** — writes each UB tile back to its GM partition, completing the round trip. Same Tile-domain rules as `tload`.

#### Where the Tile and Micro Boundaries Sit

| Op | Where it must live | Why |
|----|-------------------|-----|
| `pto.alloc_tile`, `pto.tload`, `pto.tstore`, `pto.tadd`, … (Tile domain) | **Outside** `pto.vecscope` | Tile ops describe tile residency and tile-granular DMA / compute; they have no meaning inside a vector-register region. |
| `pto.vlds`, `pto.vsts`, `pto.vmax`, `pto.vexpdif`, … (micro domain) | **Inside** `pto.vecscope` | These ops produce/consume `!pto.vreg` and `!pto.mask` values that only exist inside a vector region. |
| `pto.tile_buf_addr` | **Inside** `pto.vecscope` only | This is the single sanctioned bridge from a tile handle to a UB pointer; outside vecscope, tile handles must be consumed by Tile ops, not by address extraction. |
| `pto.set_flag` / `pto.wait_flag` (and other sync primitives) | Either side | Sync ops belong to whichever pipeline edge they coordinate; in this kernel they appear at the MTE2 → V and V → MTE3 boundaries. |

In short: keep DMA and tile shape management in Tile-land, keep vreg/mask compute in vecscope, and use `pto.tile_buf_addr` exactly at the boundary.

<a id="tile-02-types-and-attributes"></a>

## 2. Types & Attributes

> **Category:** Type system and attribute vocabulary

This chapter defines the types and attributes used across the Tile Instruction chapters.

---

### 2.1 Element Types

Element types describe the primitive scalar values stored in tiles; by themselves they do not form a value. Common element categories:

- **Integers:** signless — `i1`, `i8`, `i16`, `i32`, `i64`. Signedness is not encoded in the type; it is selected by operation semantics or attributes.
- **Floating-point:** `f16`, `bf16`, `f32`.
- **Index-like:** `index` values appear as scalar operands (offsets, sizes, scalar compares).

Operation-specific constraints:

- Elementwise ops typically require operand and result element types to match.
- Reductions, math ops, and division typically restrict to floating-point or a subset of integer types.
- Bitwise ops require integer element types.
- `pto.tcvt` defines explicit element-type changes under an explicit rounding mode.

Memory layout and address space do not change element-type semantics; they only affect placement and access patterns.

### 2.2 `!pto.ptr<elementType[, memorySpace]>`

A typed pointer. `memorySpace` is optional and defaults to `gm`.

| Parameter | Type | Description |
|-----------|------|-------------|
| `elementType` | element type | Element type pointed to. |
| `memorySpace` | `gm` \| `vec` | Pointer address space (`gm` → global memory, `vec` → UB / vector memory). |

**Syntax:** `!pto.ptr<f16>` or `!pto.ptr<f16, vec>`

Pointer conversions are modeled explicitly with `pto.castptr`. Between two `!pto.ptr` types, casts are only legal when both pointers stay in the same PTO memory space.

### 2.3 `!pto.tensor_view<d0 x d1 x elementType>`

A descriptor for a global-memory tensor. Holds shape information; strides are supplied at `pto.make_tensor_view` construction time. Does not own data.

| Parameter | Type | Description |
|-----------|------|-------------|
| `shape` | `ArrayRef<i64>` | Tensor shape `[d0, d1]` (each dim may be `?`). |
| `elementType` | element type | Element data type. |

**Syntax:** `!pto.tensor_view<1024x512xf16>`

### 2.4 `!pto.partition_tensor_view<d0 x d1 x elementType>`

A logical partition (slice) of a `tensor_view`. Holds shape information for a tile-sized region; strides are inherited from the parent `tensor_view`. Does not own data.

| Parameter | Type | Description |
|-----------|------|-------------|
| `shape` | `ArrayRef<i64>` | Partition shape `[d0, d1]`. |
| `elementType` | element type | Element data type. |

**Syntax:** `!pto.partition_tensor_view<16x16xf16>`

### 2.5 `!pto.tile_buf<loc, RxCxdtype[, valid=v_rxv_c][, blayout=..., slayout=..., fractal=..., pad=...]>`

`pto.tile_buf` represents a local on-chip tile buffer with explicit placement, shape, valid region, and layout/fractal metadata. The textual form is **compact**: only the leading `<loc, RxCxdtype>` triple is mandatory; everything else is omitted when it equals its default.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `loc` | keyword | — | Local memory domain (`vec` / `mat` / `left` / `right` / `acc` / `bias` / `scaling`). |
| `R` × `C` × `dtype` | shape × element type | — | Physical row/column count and element type. |
| `valid` | `v_row x v_col` (each `int64` or `?`) | `R x C` | Valid region. Omitted when equal to physical shape. |
| `blayout` | `BLayout` | `row_major` | Base layout. |
| `slayout` | `SLayout` | `none_box` | Secondary layout. |
| `fractal` | `int32` | `512` | Fractal size. |
| `pad` | `PadValue` enum int | `0` (`null`) | Padding policy/value selector. |

**Examples:**

```mlir
// Default config, valid == physical
!pto.tile_buf<vec, 16x16xf16>

// Dynamic valid region
!pto.tile_buf<vec, 16x16xf16, valid=?x?>

// Non-default config
!pto.tile_buf<vec, 8x8xf32, blayout=col_major, slayout=row_major, fractal=1024, pad=1>
```

`?` denotes a dynamic symbol resolved at runtime (via `pto.alloc_tile` operands or `pto.set_validshape`).

### 2.6 AddressSpace

Defines the physical storage location of a buffer in the Ascend NPU memory hierarchy.

| Value | Int | Mnemonic | Hardware Mapping |
|-------|-----|----------|------------------|
| `Zero` | 0 | `zero` | Default (unspecified). |
| `GM` | 1 | `gm` | Global Memory. |
| `MAT` | 2 | `mat` | L1 Cache. |
| `LEFT` | 3 | `left` | L0A (left matrix buffer). |
| `RIGHT` | 4 | `right` | L0B (right matrix buffer). |
| `ACC` | 5 | `acc` | L0C (accumulator). |
| `VEC` | 6 | `vec` | UB (unified buffer). |
| `BIAS` | 7 | `bias` | Bias buffer. |
| `SCALING` | 8 | `scaling` | Scaling buffer. |

**Attribute syntax:** `loc=<mnemonic>` (for example `loc=vec`).

### 2.7 Tile Buf Config

Composite attribute for tile-buffer layout/fractal/pad.

| Parameter | Type | Description |
|-----------|------|-------------|
| `bLayout` | `BLayoutAttr` | Base layout (RowMajor / ColMajor). |
| `sLayout` | `SLayoutAttr` | Secondary layout (NoneBox / RowMajor / ColMajor). |
| `sFractalSize` | `IntegerAttr (i32)` | Secondary fractal size. |
| `pad` | `PadValueAttr` | Pad value policy. |

**Syntax:** `#pto.tile_buf_config<row_major, none_box, 16, zero>`

**BLayout:**

| Value | Int | Mnemonic |
|-------|-----|----------|
| `RowMajor` | 0 | `row_major` |
| `ColMajor` | 1 | `col_major` |

**SLayout:**

| Value | Int | Mnemonic |
|-------|-----|----------|
| `NoneBox` | 0 | `none_box` |
| `RowMajor` | 1 | `row_major` |
| `ColMajor` | 2 | `col_major` |

**PadValue:**

| Value | Int | Mnemonic |
|-------|-----|----------|
| `Null` | 0 | `null` |
| `Zero` | 1 | `zero` |
| `Max` | 2 | `max` |
| `Min` | 3 | `min` |

### 2.8 Layout

Global tensor layout attribute for `tensor_view` and `partition_tensor_view`. Tile buffers additionally use **Tile Buf Config** (§2.7) to describe physical/fractal layout.

| Value | Int | Mnemonic | Description |
|-------|-----|----------|-------------|
| `ND` | 0 | `nd` | Row-major (Normal-Dimension). |
| `DN` | 1 | `dn` | Column-major (Dimension-Normal). |
| `NZ` | 2 | `nz` | Fractal / blocked layout. |

**Attribute syntax:** `#pto.layout<nd>`

### 2.9 PadMode (for loads)

Padding mode for `pto.tload`.

| Value | Int | Description |
|-------|-----|-------------|
| `PadNull` | 0 | No padding. |
| `PadFirstElem` | 1 | Pad using the first element. |
| `PadValue` | 2 | Pad using a specified value. |

### 2.10 Shared Scalar and Control-Flow Ops

Tile programs commonly interleave `pto` instructions with a small set of supporting ops:

- **`func`** — `func.func`, `func.return`, `func.call`.
- **`arith`** — scalar constants/casts (`arith.constant`, `arith.index_cast`, `arith.bitcast`, `arith.extf`/`truncf`/…), integer/float arithmetic, bitwise/shift, compares/select, extended and min/max ops.
- **`scf`** — `scf.for`, `scf.if`, `scf.yield`; several other structured control-flow forms are lowered through `cf`.

These supporting ops are included here only insofar as tile programs need function structure, scalar computation, and structured control flow; full coverage of those surfaces is out of scope for this reference.

<a id="tile-03-pointer-and-view"></a>

## 3. Pointer & View Operations

> **Category:** Address arithmetic, tensor-view construction, tile-buffer allocation
> **Pipeline:** None (all ops are metadata / view construction; no HW side effect)

These instructions build the address, view, and tile-buffer metadata that later DMA and compute instructions consume. None of them moves data.

---

### `pto.addptr`

- **syntax:**
```mlir
%result = pto.addptr %base, %offset : !pto.ptr<T> -> !pto.ptr<T>
```
- **semantics:** `result = ptr + offset`, with `offset` counted in **elements** (not bytes).

**Parameter Table:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `%base` | `!pto.ptr<T>` | Base pointer. |
| `%offset` | `index` | Element offset. |

**Constraints:**

- Result type must match the input pointer type.
- The op is pure (no side effects).

**Example:**

```mlir
%ptr_off = pto.addptr %base, %offset : !pto.ptr<f32> -> !pto.ptr<f32>
```

---

### `pto.castptr`

- **syntax:**
```mlir
%p_ptr  = pto.castptr %addr : i64 -> !pto.ptr<T, space>
%p_ptr2 = pto.castptr %p_ptr : !pto.ptr<T, space> -> !pto.ptr<T2, space>
%addr2  = pto.castptr %p_ptr : !pto.ptr<T, space> -> i64
```
- **semantics:** Explicit cast between integer addresses and `!pto.ptr`, or between two `!pto.ptr` types.

**Parameter Table:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `input` | integer \| `!pto.ptr<...>` | Source value. |

**Constraints:**

- Integer-to-integer casts are rejected; use normal integer cast ops.
- Descriptor types (`!pto.tensor_view<...>`, `!pto.partition_tensor_view<...>`) are not legal direct inputs; extract an address first.
- Pointer-to-pointer casts are only legal when source and destination stay in the same PTO memory space (`gm` or `vec`).
- The op is pure.

**Example:**

```mlir
%p0 = pto.castptr %addr : i64 -> !pto.ptr<f32, vec>
%p1 = pto.castptr %p0   : !pto.ptr<f32, vec> -> !pto.ptr<i8, vec>
%a2 = pto.castptr %p1   : !pto.ptr<i8, vec>  -> i64
```

---

### `pto.make_tensor_view`

- **syntax:**
```mlir
%tv = pto.make_tensor_view %ptr, shape = [%m, %n], strides = [%s0, %s1]
    : !pto.tensor_view<?x?xT>
```
- **semantics:** Construct a global tensor view from a pointer, declaring the physical base and strides. No allocation, no data movement.

**Parameter Table:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `%ptr` | `AnyType` | Source pointer (must be `!pto.ptr<T>` with element type matching the result). |
| `shape` | `Variadic<Index>` | Dynamic shape dimensions. |
| `strides` | `Variadic<Index>` | Dynamic strides. |
| `layout` | `LayoutAttr` (optional) | `nd` / `dn` / `nz` hint. |

**Constraints:**

- `ptr` element type must match the result element type.
- `shape` and `strides` operand counts must match the tensor_view rank.
- If `layout` is provided with static shapes/strides, it must be consistent with the inferred layout.

**Example:**

```mlir
%tv = pto.make_tensor_view %ptr, shape = [%m, %n], strides = [%s0, %s1]
    : !pto.tensor_view<?x?xf32>
```

---

### `pto.get_tensor_view_dim`

- **syntax:**
```mlir
%dim = pto.get_tensor_view_dim %tv, %idx : !pto.tensor_view<...> -> index
```
- **semantics:** Return the runtime size of dimension `%idx` from a `tensor_view`.

**Parameter Table:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `%tv` | `!pto.tensor_view<...>` | Logical tensor view. |
| `%idx` | `index` | Dimension index (0-based). |

**Example:**

```mlir
%h = pto.get_tensor_view_dim %tv, %c0 : !pto.tensor_view<?x?xf32> -> index
```

---

### `pto.get_tensor_view_stride`

- **syntax:**
```mlir
%stride = pto.get_tensor_view_stride %tv, %idx : !pto.tensor_view<...> -> index
```
- **semantics:** Return the logical stride of dimension `%idx`, measured in **elements** (not bytes).

**Parameter Table:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `%tv` | `!pto.tensor_view<...>` or memref form | Tensor view or its lowered memory-reference form. |
| `%idx` | `index` | Dimension index (0-based). |

**Example:**

```mlir
%s0 = pto.get_tensor_view_stride %tv, %c0 : !pto.tensor_view<?x?xf32> -> index
```

---

### `pto.tensor_view_addr`

- **syntax:**
```mlir
%result = pto.tensor_view_addr %src : !pto.tensor_view<...> -> memref<...>
%result = pto.tensor_view_addr %src : !pto.tensor_view<...> -> !pto.ptr<T, gm>
```
- **semantics:** Extract the underlying address view from a `tensor_view` or `partition_tensor_view`.

**Parameter Table:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `%src` | `!pto.tensor_view<...>` or `!pto.partition_tensor_view<...>` | Source view descriptor. |

**Constraints:**

- The result type must be either the lowered memref view or a GM pointer `!pto.ptr<T, gm>` to the same underlying storage.
- The op is pure and does not move data.

**Example:**

```mlir
%base = pto.tensor_view_addr %tv : !pto.tensor_view<?x?xf32> -> !pto.ptr<f32, gm>
```

`pto.tensor_view_addr` exposes the underlying address represented by the view descriptor. When the result type is a memref, it exposes the lowered view directly. When the result type is `!pto.ptr<..., gm>`, it exposes the same address in pointer form. During compiler-internal lowering, the operand may already be rewritten to a memref form; in that case this op is folded away or rewritten to an equivalent memref-to-ptr cast.

---

### `pto.partition_view`

- **syntax:**
```mlir
%pv = pto.partition_view %tv, offsets = [%o0, %o1], sizes = [%s0, %s1]
    : !pto.tensor_view<...> -> !pto.partition_tensor_view<...>
```
- **semantics:** `result = source[offsets, sizes]` — a logical window on a `tensor_view`. Captures both static and dynamic shapes; does not move data.

**Parameter Table:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `%tv` | `TensorViewType` | Input tensor view. |
| `offsets` | `Variadic<Index>` | Dynamic offsets. |
| `sizes` | `Variadic<Index>` | Dynamic sizes. |

**Constraints:**

- `offsets`/`sizes` counts must match the rank of `source`.

**Example:**

```mlir
%pv = pto.partition_view %tv, offsets = [%off0, %off1], sizes = [%s0, %s1]
    : !pto.tensor_view<1024x512xf16> -> !pto.partition_tensor_view<16x16xf16>
```

---

### `pto.alloc_tile`

- **syntax:**
```mlir
%tb  = pto.alloc_tile : !pto.tile_buf<...>
%tb2 = pto.alloc_tile valid_row = %vr valid_col = %vc : !pto.tile_buf<vec, RxCxT, valid=?x?>
%tb3 = pto.alloc_tile addr = %ad : !pto.tile_buf<...>
```
- **semantics:** Declare the lifetime of a tile buffer. Each call produces an **independent** tile-buffer instance.

**Parameter Table:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `addr` | `Optional<i64>` | Optional start address. If omitted, assigned by the implementation. |
| `valid_row` | `Optional<index>` | Dynamic valid-row count (required when result `v_row = ?`). |
| `valid_col` | `Optional<index>` | Dynamic valid-col count (required when result `v_col = ?`). |

**Constraints:**

- If result `v_row`/`v_col` are dynamic (`?`), the corresponding operands must be present.
- If result `v_row`/`v_col` are static, the corresponding operands must be absent.

**Example:**

```mlir
%tb = pto.alloc_tile : !pto.tile_buf<vec, 16x16xf16>
```

---

### `pto.subset`

- **syntax:**
```mlir
%sub = pto.subset %src[%i, %j] sizes [rows, cols] : !pto.tile_buf<...>
```
- **semantics:** `result = source[offsets]` with static `sizes`. Creates a strided view of a parent tile.

**Parameter Table:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `%src` | `pto.tile_buf` | Parent tile buffer. |
| `offsets` | `Variadic<Index>` | Runtime offsets `[i, j]`. |
| `sizes` | `I64ArrayAttr` | Static shape `[rows, cols]`. |

**Constraints:**

- Boxed-vs-non-boxed behavior is derived from the source's tile config (`blayout`, `slayout`, `fractal`) and element type.
- For non-boxed layouts (`slayout=none_box`), no additional subset-specific structural checks are enforced.
- For boxed layouts:
  - `sizes` must have length 2 and both subset sizes must be positive.
  - Subset sizes must be multiples of the inferred inner boxed shape.
  - `offsets` must have length 2; constant offsets must be non-negative and multiples of the inferred inner boxed shape.
  - Source tile shape must be statically known.
  - For boxed row-major tiles: subset must keep the full source column extent, and the column offset must be the constant `0`.
  - For boxed col-major tiles: subset must keep the full source row extent, and the row offset must be the constant `0`.
- The inferred result reuses the source's element type, address space, and tile config. `valid_shape` is derived from the parent valid shape and constant offsets, or dynamic when offsets are dynamic.

**Example:**

```mlir
%sub = pto.subset %src[%i, %j] sizes [32, 32]
     : !pto.tile_buf<vec, 64x64xf16>
```

---

### `pto.set_validshape`

- **syntax:**
```mlir
pto.set_validshape %src, %valid_row, %valid_col : !pto.tile_buf<vec, RxCxT, valid=?x?>
```
- **semantics:** Update the runtime `v_row`/`v_col` metadata on an existing **dynamic** tile buffer.

**Parameter Table:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `%src` | `pto.tile_buf` | Dynamic rank-2 tile buffer. |
| `%valid_row` | `index` | Runtime valid row count. |
| `%valid_col` | `index` | Runtime valid column count. |

**Constraints:**

- `%src` must be rank-2 and use `v_row = ?` and `v_col = ?` on both dimensions.
- Tile programs use `pto.tile_buf`; memref forms are a lowering artifact and are not part of this surface.
- Constant `valid_row`/`valid_col` must be non-negative and `<=` the tile's static shape bounds.

**Example:**

```mlir
%src = pto.alloc_tile : !pto.tile_buf<vec, 32x32xf16, valid=?x?>
pto.set_validshape %src, %vr, %vc : !pto.tile_buf<vec, 32x32xf16, valid=?x?>
```

---

### `pto.tile_buf_addr`

- **syntax:**
```mlir
%ub_ptr = pto.tile_buf_addr %tile : !pto.tile_buf<...> -> !pto.ptr<T, vec>
%ub_ref = pto.tile_buf_addr %tile : !pto.tile_buf<...> -> memref<...>
```
- **semantics:** Extract the address of a `pto.tile_buf`'s data region. Returns either a typed PTO pointer (`!pto.ptr<T, space>`) or a memref view, depending on the requested result type. Pure op: no data movement, no pipeline activity.

This op is the **boundary between tile-buffer instructions and pointer-based vector instructions**. Inside a `pto.vecscope` body, use `pto.tile_buf_addr` to materialize a vec-space pointer from a tile handle allocated outside the scope; vector load/store ops such as `pto.vlds` and `pto.vsts` then consume that pointer.

**Parameter Table:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `%tile` | `pto.tile_buf` or tile-bound memref | Tile handle whose data-region address is taken. |

**Results:** `!pto.ptr<T, space>` or `memref<...>`. Memref results use the tile's static shape and address space; pointer results use the tile's element type and memory space (e.g. `vec`).

**Constraints:**

- Result must be either a typed PTO pointer or a memref view; no other result types are accepted.
- When a memref result is requested, the lowered form uses the tile's static shape and address space.
- `pto.tile_buf_addr` is **only legal inside `pto.vecscope` / `pto.strict_vecscope`**. Outside a vector scope, tile handles must be consumed by tile-level ops (`pto.tload`, `pto.tstore`, `pto.tadd`, …) rather than by address extraction. Conversely, tile-level ops must **not** appear inside `pto.vecscope`.

**Example (inside `pto.vecscope`):**

```mlir
%tile = pto.alloc_tile addr = %c0_i64 valid_row = %r
  : !pto.tile_buf<vec, 8x128xf32, valid=?x?>

pto.vecscope {
  %ub = pto.tile_buf_addr %tile
    : !pto.tile_buf<vec, 8x128xf32, valid=?x?> -> !pto.ptr<f32, vec>
  // ... vector-scope loads/stores on %ub ...
}
```

See [`03-vector-load-store.md`](PTO-micro-Instruction-SPEC-v0.4.md#micro-03-vector-load-store) for the pointer-based
vector load/store side of this handoff.

<a id="tile-04-dma-data-movement"></a>

## 4. DMA Data Movement

> **Category:** GM↔on-chip DMA for tile buffers
> **Pipelines:** PIPE_MTE2 (GM→UB), PIPE_MTE3 (UB→GM), PIPE_FIX (when source is `loc=acc`)

This chapter documents the public tile DMA instructions `pto.tload` and `pto.tstore`. Other raw scalar load/store helpers are outside the current tile-instruction subset and are not covered here.

---

### `pto.tload`

- **syntax:**
```mlir
pto.tload ins(%src : !pto.partition_tensor_view<...>)
          outs(%dst : !pto.tile_buf<...>)
```
- **semantics:** Physical DMA transfer from a global partition view into a local tile buffer. For each element `(i, j)` in the destination valid region: `dst[i, j] = src[i, j]`.

**Parameter Table:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `src` | `PartitionTensorViewType` | Source partition view. |
| `dst` | `pto.tile_buf` | Destination tile buffer. |

**Constraints:**

- Tile element type ∈ `{i8, i16, i32, i64, f16, bf16, f32}`.
- Destination tile must use `loc=vec`.
- Destination tile element type and source partition element type must have the same bitwidth.
- Runtime: source partition extents and destination valid region must be positive.

**Pipeline:** `PIPE_MTE2`.

**Example:**

```mlir
pto.tload ins(%pv : !pto.partition_tensor_view<16x16xf16>)
          outs(%tb : !pto.tile_buf<vec, 16x16xf16>)
```

---

### `pto.tstore`

- **syntax:**
```mlir
pto.tstore ins(%src : !pto.tile_buf<...>)
           outs(%dst : !pto.partition_tensor_view<...>)
```
- **semantics:** Store a 2-D tile buffer back to a 2-D partition view. For each element `(i, j)` in the source valid region: `dst[i, j] = src[i, j]`.

**Parameter Table:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `src` | `pto.tile_buf` | Source tile buffer. |
| `dst` | `PartitionTensorViewType` | Destination partition view. |

**Constraints:**

- `src` must be `!pto.tile_buf`, `dst` must be `!pto.partition_tensor_view`.
- Static dst shape dims and static src valid-shape dims must be positive.
- `src.loc ∈ {vec, mat, acc}`.
- For `loc=vec` / `loc=mat`: src element type ∈ `{i8, i16, i32, i64, f16, bf16, f32}`; src/dst element bitwidth must match.
- For `loc=acc`:
  - src element type must be `i32` or `f32`.
  - dst element type ∈ `{i32, f32, f16, bf16}`.

**Pipeline:**

- `src.loc=acc` uses **PIPE_FIX**.
- `src.loc=vec` / `src.loc=mat` uses **PIPE_MTE3**.

**Example:**

```mlir
pto.tstore ins(%tb : !pto.tile_buf<vec, 16x16xf16>)
           outs(%pv : !pto.partition_tensor_view<16x16xf16>)
```

<a id="tile-05-vector-arithmetic"></a>

## 5. Vector Arithmetic and Activation Operations

> **Category:** Base tile-local VEC arithmetic
> **Pipeline:** PIPE_V

This chapter documents the TileLib arithmetic families that keep the same output tile shape as their source tiles. These instructions operate on `!pto.tile_buf` values in `loc=vec` and cover tile-tile arithmetic, tile-scalar arithmetic, unary math, and activation ops.

Reduction, partial, bitwise, conversion, broadcast / expansion, selection, and fill / padding families are documented in Chapters 6-12.

---

### 5.1 Binary Tile-Tile Arithmetic

Tile-tile arithmetic families:

| Op | Semantics |
|----|-----------|
| `pto.tadd` | `dst[i, j] = src0[i, j] + src1[i, j]` |
| `pto.tsub` | `dst[i, j] = src0[i, j] - src1[i, j]` |
| `pto.tmul` | `dst[i, j] = src0[i, j] * src1[i, j]` |
| `pto.tdiv` | `dst[i, j] = src0[i, j] / src1[i, j]` |
| `pto.tmax` | `dst[i, j] = max(src0[i, j], src1[i, j])` |
| `pto.tmin` | `dst[i, j] = min(src0[i, j], src1[i, j])` |

#### Common Syntax

```mlir
pto.<op> ins(%src0, %src1 : !pto.tile_buf<...>, !pto.tile_buf<...>)
         outs(%dst : !pto.tile_buf<...>)
```

**Parameter Table:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `src0` | `pto.tile_buf` | First source tile buffer. |
| `src1` | `pto.tile_buf` | Second source tile buffer. |
| `dst` | `pto.tile_buf` | Destination tile buffer. |

**Constraints:**

- `src0`, `src1`, and `dst` must be shape-compatible tile buffers on `loc=vec`.
- The valid region must match across all three tiles.
- Element type legality is target-defined; ops specialize over the tile dtype selected at expansion time.
- `pto.tdiv` uses element-wise division; **undefined behavior** on divide-by-zero.

**Example:**

```mlir
pto.tadd ins(%a, %b : !pto.tile_buf<vec, 16x16xf16>, !pto.tile_buf<vec, 16x16xf16>)
         outs(%c : !pto.tile_buf<vec, 16x16xf16>)
```

---

### 5.2 Tile-Scalar Arithmetic

Tile-scalar families:

| Op | Supported operand form(s) | Semantics |
|----|---------------------------|-----------|
| `pto.tadds` | `tile, scalar` | `dst[i, j] = src[i, j] + scalar` |
| `pto.tsubs` | `tile, scalar` | `dst[i, j] = src[i, j] - scalar` |
| `pto.tmuls` | `tile, scalar` | `dst[i, j] = src[i, j] * scalar` |
| `pto.tdivs` | `tile, scalar` and `scalar, tile` | `dst = src / scalar` or `dst = scalar / src` |
| `pto.tmaxs` | `tile, scalar` | `dst[i, j] = max(src[i, j], scalar)` |
| `pto.tmins` | `tile, scalar` | `dst[i, j] = min(src[i, j], scalar)` |

#### Common Syntax

For `pto.tadds`, `pto.tsubs`, `pto.tmuls`, `pto.tmaxs`, and `pto.tmins`:

```mlir
pto.<op> ins(%src, %scalar : !pto.tile_buf<...>, <scalar_type>)
          outs(%dst : !pto.tile_buf<...>)
```

For `pto.tdivs`:

```mlir
pto.tdivs ins(%src, %scalar : !pto.tile_buf<...>, <scalar_type>)
          outs(%dst : !pto.tile_buf<...>)

pto.tdivs ins(%scalar, %src : <scalar_type>, !pto.tile_buf<...>)
          outs(%dst : !pto.tile_buf<...>)
```

**Parameter Table:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `src` | `pto.tile_buf` | Source tile buffer. |
| `scalar` | signless integer / floating-point scalar | Scalar broadcast across the tile. |
| `dst` | `pto.tile_buf` | Destination tile buffer. |

**Constraints:**

- `src` and `dst` must be shape-compatible `loc=vec` tile buffers.
- The scalar element type must be compatible with the tile element type.
- `pto.tdivs` is the only scalar family with two public operand orders. **Undefined behavior** on divide-by-zero (either `scalar==0` or any `src[i,j]==0` in the `scalar/src` form).

**Example:**

```mlir
pto.tadds ins(%a, %s : !pto.tile_buf<vec, 32x32xf32>, f32)
          outs(%c : !pto.tile_buf<vec, 32x32xf32>)
```

```mlir
pto.tdivs ins(%s, %a : f32, !pto.tile_buf<vec, 32x32xf32>)
          outs(%c : !pto.tile_buf<vec, 32x32xf32>)
```

---

### 5.3 Unary Math

All ops below share the common form:

```mlir
pto.<op> ins(%src : !pto.tile_buf<...>)
         outs(%dst : !pto.tile_buf<...>)
```

| Op | Semantics |
|----|-----------|
| `pto.tabs` | `dst = abs(src)` |
| `pto.tneg` | `dst = -src` |
| `pto.texp` | `dst = exp(src)` |
| `pto.tlog` | `dst = ln(src)` |
| `pto.tsqrt` | `dst = sqrt(src)` |
| `pto.trsqrt` | `dst = 1 / sqrt(src)` |
| `pto.trecip` | `dst = 1 / src` |

**Constraints:**

- `src` and `dst` must have the same valid region.
- These ops are numeric Tile Instruction ops on `loc=vec`.
- **Undefined behavior** on out-of-domain inputs: `tlog(<=0)`, `tsqrt(<0)`, `trsqrt(<=0)`, `trecip(0)`.

**Example:**

```mlir
pto.tabs ins(%a : !pto.tile_buf<vec, 16x16xf16>)
         outs(%c : !pto.tile_buf<vec, 16x16xf16>)
```

---

### 5.4 Activation Operations

Activation family:

| Op | Semantics |
|----|-----------|
| `pto.trelu` | `dst[i, j] = max(0, src[i, j])` |
| `pto.tlrelu` | `dst[i, j] = src[i, j] > 0 ? src[i, j] : slope * src[i, j]` |

#### Common Forms

ReLU:

```mlir
pto.trelu ins(%src : !pto.tile_buf<...>)
          outs(%dst : !pto.tile_buf<...>)
```

Leaky ReLU:

```mlir
pto.tlrelu ins(%src, %slope : !pto.tile_buf<...>, f32)
           outs(%dst : !pto.tile_buf<...>)
```

**Constraints:**

- `src` and `dst` must have the same valid region.
- `pto.trelu` supports `f16`, `f32`, and `i32`.
- `pto.tlrelu` supports `f16` and `f32`, with the slope passed as an `f32` scalar operand.
- Both ops execute on `loc=vec` tiles via the vector pipeline.

**Example:**

```mlir
pto.trelu ins(%src : !pto.tile_buf<vec, 16x64xf32>)
          outs(%dst : !pto.tile_buf<vec, 16x64xf32>)
```

<a id="tile-06-reduction-ops"></a>

## 6. Reduction Operations

> **Category:** Tile-local VEC reductions
> **Pipeline:** PIPE_V

This chapter documents the TileLib reduction families. These ops reduce one or more source dimensions into smaller destination tiles and are organized into row-reduction and column-reduction groups.

---

### 6.1 Row Reductions

Row reductions reduce each row of `%src` into one element stored at `%dst[row, 0]`. The op shape carries a scratch tile operand `%tmp` to keep the operand list aligned with the A2/A3 PTO instruction interface (see [§1.7](#17-scratch-operands-and-a2a3-compatibility)).

#### Common Syntax

```mlir
pto.<op> ins(%src, %tmp : !pto.tile_buf<...>, !pto.tile_buf<...>)
         outs(%dst : !pto.tile_buf<...>)
```

| Op | Semantics |
|----|-----------|
| `pto.trowsum` | `dst[i, 0] = sum_j src[i, j]` |
| `pto.trowprod` | `dst[i, 0] = prod_j src[i, j]` |
| `pto.trowmax` | `dst[i, 0] = max_j src[i, j]` |
| `pto.trowmin` | `dst[i, 0] = min_j src[i, j]` |
| `pto.trowargmax` | `dst[i, 0] = argmax_j src[i, j]` |
| `pto.trowargmin` | `dst[i, 0] = argmin_j src[i, j]` |

**Parameter Table:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `src` | `pto.tile_buf` | Source tile buffer. |
| `tmp` | `pto.tile_buf` | Scratch tile for A2/A3 interface compatibility. |
| `dst` | `pto.tile_buf` | Destination tile storing one result per source row. |

**Constraints:**

- `dst.v_row` should match `src.v_row`.
- `dst.v_col` should be `1`.
- `pto.trowargmax` and `pto.trowargmin` require an integer destination element type for the row-local index result.
- Numeric widening / narrowing inside the reduction is target-defined by the selected template (e.g. `trowsum` may widen `i16` accumulation internally before storing to `dst`).

**Example:**

```mlir
pto.trowsum ins(%src, %tmp : !pto.tile_buf<vec, 16x32xf32>, !pto.tile_buf<vec, 16x32xf32>)
            outs(%dst : !pto.tile_buf<vec, 16x1xf32>)
```

---

### 6.2 Column Reductions

Column reductions reduce each column of `%src` into one element stored at `%dst[0, col]`.

#### Common Syntax

```mlir
pto.<op> ins(%src : !pto.tile_buf<...>)
         outs(%dst : !pto.tile_buf<...>)
```

| Op | Semantics |
|----|-----------|
| `pto.tcolsum` | `dst[0, j] = sum_i src[i, j]` |
| `pto.tcolprod` | `dst[0, j] = prod_i src[i, j]` |
| `pto.tcolmax` | `dst[0, j] = max_i src[i, j]` |
| `pto.tcolmin` | `dst[0, j] = min_i src[i, j]` |

**Constraints:**

- `dst.v_row` should be `1`.
- `dst.v_col` should match `src.v_col`.
- Templates assume prefix-aligned valid regions and row-major VEC tiles.

**Example:**

```mlir
pto.tcolsum ins(%src : !pto.tile_buf<vec, 16x16xf32>)
            outs(%dst : !pto.tile_buf<vec, 1x16xf32>)
```

<a id="tile-07-partial-elementwise"></a>

## 7. Partial Elementwise Operations

> **Category:** Tile-local VEC partial-shape compute
> **Pipeline:** PIPE_V

This chapter documents the TileLib partial elementwise families. These ops combine two tiles whose valid regions may differ, but whose overlap starts at `[0, 0]`.

---

### Common Syntax

```mlir
pto.<op> ins(%src0, %src1 : !pto.tile_buf<...>, !pto.tile_buf<...>)
         outs(%dst : !pto.tile_buf<...>)
```

| Op | Semantics on the overlap region |
|----|----------------------------------|
| `pto.tpartadd` | `dst = src0 + src1` |
| `pto.tpartmul` | `dst = src0 * src1` |
| `pto.tpartmax` | `dst = max(src0, src1)` |
| `pto.tpartmin` | `dst = min(src0, src1)` |

**Constraints:**

- Let `big` ∈ {`src0`, `src1`} be the operand whose valid shape equals `dst.valid_shape`, and `small` be the other operand. Exactly one operand plays each role.
- `small.valid_shape` must be a prefix-aligned sub-rectangle of `dst.valid_shape` (i.e. starting at `[0, 0]`).
- For `pto.tpartadd` and `pto.tpartmul`: outside the overlap (where only `big` covers `dst`), `dst` takes `big`'s value.
- For `pto.tpartmax` and `pto.tpartmin`: A5 templates initialize `dst` with the dtype extremum before merging the operands, so uncovered regions follow the template's pad-extremum behavior.

**Example:**

```mlir
pto.tpartadd ins(%a, %b : !pto.tile_buf<vec, 32x32xf32>,
                          !pto.tile_buf<vec, 32x32xf32, valid=16x32>)
             outs(%dst : !pto.tile_buf<vec, 32x32xf32>)
```

<a id="tile-08-bitwise-shift-ops"></a>

## 8. Bitwise and Shift Operations

> **Category:** Tile-local integer VEC compute
> **Pipeline:** PIPE_V

This chapter documents the integer-only TileLib bitwise and shift families.

---

### 8.1 Unary Bitwise NOT: `pto.tnot`

- **syntax:**
```mlir
pto.tnot ins(%src : !pto.tile_buf<...>)
         outs(%dst : !pto.tile_buf<...>)
```
- **semantics:** `dst = ~src`.

**Constraints:**

- Tile element types must be integer types.
- `src` and `dst` must have the same valid region.

**Example:**

```mlir
pto.tnot ins(%a : !pto.tile_buf<vec, 16x16xi32>)
         outs(%c : !pto.tile_buf<vec, 16x16xi32>)
```

---

### 8.2 Binary Tile-Tile Bitwise and Shift Families

Tile-tile bitwise and shift families:

| Op | Semantics |
|----|-----------|
| `pto.tand` | `dst = src0 & src1` |
| `pto.tor` | `dst = src0 \| src1` |
| `pto.txor` | `dst = src0 ^ src1` |
| `pto.tshl` | `dst = src0 << src1` |
| `pto.tshr` | `dst = src0 >> src1` |

#### Common Forms

For `pto.tand`, `pto.tor`, `pto.tshl`, and `pto.tshr`:

```mlir
pto.<op> ins(%src0, %src1 : !pto.tile_buf<...>, !pto.tile_buf<...>)
          outs(%dst : !pto.tile_buf<...>)
```

`pto.txor` carries an extra scratch tile `%tmp` for A2/A3 interface compatibility (see [§1.7](#17-scratch-operands-and-a2a3-compatibility)):

```mlir
pto.txor ins(%src0, %src1, %tmp : !pto.tile_buf<...>, !pto.tile_buf<...>,
             !pto.tile_buf<...>)
         outs(%dst : !pto.tile_buf<...>)
```

**Constraints:**

- Tile element types must be integer types.
- `src0`, `src1`, and `dst` must have the same valid region.

**Example:**

```mlir
pto.tand ins(%a, %b : !pto.tile_buf<vec, 16x16xi32>, !pto.tile_buf<vec, 16x16xi32>)
         outs(%c : !pto.tile_buf<vec, 16x16xi32>)
```

---

### 8.3 Tile-Scalar Bitwise and Shift Families

Tile-scalar bitwise and shift families:

| Op | Semantics |
|----|-----------|
| `pto.tands` | `dst = src & scalar` |
| `pto.tors` | `dst = src \| scalar` |
| `pto.txors` | `dst = src ^ scalar` |
| `pto.tshls` | `dst = src << scalar` |
| `pto.tshrs` | `dst = src >> scalar` |

#### Common Forms

For `pto.tands`, `pto.tors`, `pto.tshls`, and `pto.tshrs`:

```mlir
pto.<op> ins(%src, %scalar : !pto.tile_buf<...>, <integer_scalar_type>)
          outs(%dst : !pto.tile_buf<...>)
```

`pto.txors` carries an extra scratch tile `%tmp` for A2/A3 interface compatibility:

```mlir
pto.txors ins(%src, %scalar, %tmp : !pto.tile_buf<...>, <integer_scalar_type>,
              !pto.tile_buf<...>)
          outs(%dst : !pto.tile_buf<...>)
```

**Constraints:**

- Tile element types must be integer types.
- `src` and `dst` must have the same valid region.
- The scalar operand must be an integer-compatible shift / bitwise scalar.

**Example:**

```mlir
pto.tands ins(%a, %s : !pto.tile_buf<vec, 16x16xi32>, i32)
          outs(%dst : !pto.tile_buf<vec, 16x16xi32>)
```

<a id="tile-09-type-conversion"></a>

## 9. Type Conversion

> **Category:** Element-wise type conversion
> **Pipeline:** PIPE_V

This chapter documents the element-wise tile conversion instruction `pto.tcvt` and the rounding modes it uses.

---

### `RoundMode`

Rounding modes for `pto.tcvt`.

| Value | Int | Description |
|-------|-----|-------------|
| `NONE` | 0 | No rounding. |
| `RINT` | 1 | Round to nearest integer. |
| `ROUND` | 2 | Round `f16` away from zero. |
| `FLOOR` | 3 | Round toward negative infinity. |
| `CEIL` | 4 | Round toward positive infinity. |
| `TRUNC` | 5 | Truncate toward zero. |
| `ODD` | 6 | Round to odd. |
| `CAST_RINT` | 7 | Cast with round-to-nearest (default). |

**Attribute syntax:** `#pto<round_mode FLOOR>`

---

### `pto.tcvt`

- **syntax:**
```mlir
pto.tcvt ins(%src : !pto.tile_buf<...>)
         outs(%dst : !pto.tile_buf<...>)
         {rmode = #pto<round_mode ...>}
```
- **semantics:** `dst[i, j] = cast(src[i, j], rmode)` element-wise.

**Parameter Table:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `src` | `pto.tile_buf` | Source tile. |
| `dst` | `pto.tile_buf` | Destination tile (different element type). |
| `rmode` | `RoundModeAttr` | Default `CAST_RINT`. |

**Constraints:**

- `src`/`dst` must be shape/valid-region compatible.
- This reference does not define extra legality rules for the `(src, dst)` type pair. **Undefined behavior** on conversion pairs not supported by the target hardware; consult the A2/A3 and A5 hardware specs for legal pairs.

**Example:**

```mlir
pto.tcvt ins(%src : !pto.tile_buf<vec, 16x16xf32>)
         outs(%dst : !pto.tile_buf<vec, 16x16xf16>)
         {rmode = #pto<round_mode FLOOR>}
```

<a id="tile-10-broadcast-and-expansion-ops"></a>

## 10. Broadcast and Expansion Operations

> **Category:** Tile-local VEC broadcast and expansion compute
> **Pipeline:** PIPE_V

This chapter documents the TileLib broadcast, row-expansion, and column-expansion families. These ops populate destination tiles by broadcasting one logical scalar across a larger region — either from a standalone scalar operand, one source value per destination row, or one source value per destination column.

---

### 10.1 Scalar Broadcast: `pto.texpands`

- **syntax:**
```mlir
pto.texpands ins(%scalar : <scalar_type>)
             outs(%dst : !pto.tile_buf<...>)
```
- **semantics:** `dst[i, j] = scalar` for every element inside `dst`'s valid region.

**Parameter Table:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `scalar` | signless integer / floating-point scalar | Scalar value broadcast into the destination tile. |
| `dst` | `pto.tile_buf` | Destination tile buffer. |

**Constraints:**

- The TileLib template is VEC-oriented and fills `dst.valid_shape`.
- The scalar type must be compatible with `dst.dtype`.

**Example:**

```mlir
pto.texpands ins(%scalar : f32)
             outs(%dst : !pto.tile_buf<vec, 16x64xf32>)
```

---

### 10.2 Row-Wise Broadcast: `pto.trowexpand`

- **syntax:**
```mlir
pto.trowexpand ins(%src : !pto.tile_buf<...>)
               outs(%dst : !pto.tile_buf<...>)
```
- **semantics:** `dst[row, col] = src[row, 0]`.

**Parameter Table:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `src` | `pto.tile_buf` | Source tile carrying one logical scalar per destination row. |
| `dst` | `pto.tile_buf` | Destination tile buffer. |

**Constraints:**

- `src` and `dst` must have the same number of valid rows.
- `src` must encode exactly one logical source value per destination row.
- Templates target row-major VEC layouts.

**Example:**

```mlir
pto.trowexpand ins(%src : !pto.tile_buf<vec, 16x1xf32>)
               outs(%dst : !pto.tile_buf<vec, 16x16xf32>)
```

---

### 10.3 Row-Wise Broadcast Arithmetic and Transform Families

The row-expansion family combines a full tile `%src0` with a per-row scalar carrier `%src1`:

| Op | Semantics |
|----|-----------|
| `pto.trowexpandadd` | `dst[row, col] = src0[row, col] + src1[row, 0]` |
| `pto.trowexpandsub` | `dst[row, col] = src0[row, col] - src1[row, 0]` |
| `pto.trowexpandmul` | `dst[row, col] = src0[row, col] * src1[row, 0]` |
| `pto.trowexpanddiv` | `dst[row, col] = src0[row, col] / src1[row, 0]` |
| `pto.trowexpandmax` | `dst[row, col] = max(src0[row, col], src1[row, 0])` |
| `pto.trowexpandmin` | `dst[row, col] = min(src0[row, col], src1[row, 0])` |
| `pto.trowexpandexpdif` | `dst[row, col] = exp(src0[row, col] - src1[row, 0])` |

#### Common Syntax

```mlir
pto.<op> ins(%src0, %src1 : !pto.tile_buf<...>, !pto.tile_buf<...>)
         outs(%dst : !pto.tile_buf<...>)
```

**Parameter Table:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `src0` | `pto.tile_buf` | Main source tile. |
| `src1` | `pto.tile_buf` | Tile carrying one logical scalar per destination row. |
| `dst` | `pto.tile_buf` | Destination tile buffer. |

**Constraints:**

- `src0` and `dst` must be shape/valid-region compatible.
- `src1` must provide one logical scalar per destination row.
- `pto.trowexpandadd` also accepts an optional scratch `%tmp` operand for
  cross-architecture API compatibility.
- Templates target row-major VEC layouts.
- `pto.trowexpanddiv` and `pto.trowexpandexpdif` are floating-point-only.

**Example:**

```mlir
pto.trowexpandadd ins(%src0, %src1 : !pto.tile_buf<vec, 16x128xf32>,
                                     !pto.tile_buf<vec, 16x1xf32, blayout=col_major>)
                  outs(%dst : !pto.tile_buf<vec, 16x128xf32>)
```

```mlir
pto.trowexpandadd ins(%src0, %src1, %tmp : !pto.tile_buf<vec, 16x128xf32>,
                                            !pto.tile_buf<vec, 16x1xf32, blayout=col_major>,
                                            !pto.tile_buf<vec, 16x128xf32>)
                  outs(%dst : !pto.tile_buf<vec, 16x128xf32>)
```

---

### 10.4 Column-Wise Broadcast: `pto.tcolexpand`

- **syntax:**
```mlir
pto.tcolexpand ins(%src : !pto.tile_buf<...>)
               outs(%dst : !pto.tile_buf<...>)
```
- **semantics:** `dst[row, col] = src[0, col]`.

**Parameter Table:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `src` | `pto.tile_buf` | Source tile carrying one logical scalar per destination column. |
| `dst` | `pto.tile_buf` | Destination tile buffer. |

**Constraints:**

- `src` and `dst` must have the same number of valid columns.
- `src` must encode exactly one logical source value per destination column.
- Templates target row-major VEC layouts.

**Example:**

```mlir
pto.tcolexpand ins(%src : !pto.tile_buf<vec, 1x16xf32>)
               outs(%dst : !pto.tile_buf<vec, 16x16xf32>)
```

---

### 10.5 Column-Wise Broadcast Arithmetic and Transform Families

The column-expansion family combines a full tile `%src0` with a per-column scalar carrier `%src1`:

| Op | Semantics |
|----|-----------|
| `pto.tcolexpandadd` | `dst[row, col] = src0[row, col] + src1[0, col]` |
| `pto.tcolexpandsub` | `dst[row, col] = src0[row, col] - src1[0, col]` |
| `pto.tcolexpandmul` | `dst[row, col] = src0[row, col] * src1[0, col]` |
| `pto.tcolexpanddiv` | `dst[row, col] = src0[row, col] / src1[0, col]` |
| `pto.tcolexpandmax` | `dst[row, col] = max(src0[row, col], src1[0, col])` |
| `pto.tcolexpandmin` | `dst[row, col] = min(src0[row, col], src1[0, col])` |
| `pto.tcolexpandexpdif` | `dst[row, col] = exp(src0[row, col] - src1[0, col])` |

#### Common Syntax

```mlir
pto.<op> ins(%src0, %src1 : !pto.tile_buf<...>, !pto.tile_buf<...>)
         outs(%dst : !pto.tile_buf<...>)
```

**Parameter Table:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `src0` | `pto.tile_buf` | Main source tile. |
| `src1` | `pto.tile_buf` | Tile carrying one logical scalar per destination column. |
| `dst` | `pto.tile_buf` | Destination tile buffer. |

**Constraints:**

- `src0` and `dst` must be shape/valid-region compatible.
- `src1` must provide one logical scalar per destination column.
- Templates target row-major VEC layouts.
- `pto.tcolexpanddiv` and `pto.tcolexpandexpdif` are floating-point-only.

**Example:**

```mlir
pto.tcolexpandadd ins(%src0, %src1 : !pto.tile_buf<vec, 16x128xf32>,
                                     !pto.tile_buf<vec, 1x128xf32>)
                  outs(%dst : !pto.tile_buf<vec, 16x128xf32>)
```

<a id="tile-11-selection-ops"></a>

## 11. Selection Operations

> **Category:** Tile-local VEC selection compute
> **Pipeline:** PIPE_V

This chapter documents the TileLib selection families. These ops select between data sources under control of a packed predicate-mask tile.

The mask tile carries packed predicate bytes in UB. Templates load predicate bits directly with predicate-load helpers such as `plds`, then use `vsel` to choose the data path.

`pto.tsel` and `pto.tsels` carry an extra `%tmp` operand for A2/A3 interface compatibility (see [§1.7](#17-scratch-operands-and-a2a3-compatibility)).

---

### 11.1 `pto.tsel`

- **syntax:**
```mlir
pto.tsel ins(%mask, %src0, %src1, %tmp :
             !pto.tile_buf<...>, !pto.tile_buf<...>,
             !pto.tile_buf<...>, !pto.tile_buf<...>)
         outs(%dst : !pto.tile_buf<...>)
```
- **semantics:** `dst[i, j] = mask[i, j] ? src0[i, j] : src1[i, j]`.

**Parameter Table:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `mask` | `pto.tile_buf` | Packed predicate-mask carrier. |
| `src0` | `pto.tile_buf` | Value selected when the predicate bit is true. |
| `src1` | `pto.tile_buf` | Value selected when the predicate bit is false. |
| `tmp` | `pto.tile_buf` | Scratch tile for A2/A3 interface compatibility. |
| `dst` | `pto.tile_buf` | Destination tile buffer. |

**Constraints:**

- `src0`, `src1`, and `dst` must have the same shape and valid region.
- The `tsel` template specializes the mask carrier as an `i8` tile with packed predicate bytes.

**Example:**

```mlir
pto.tsel ins(%mask, %a, %b, %tmp :
             !pto.tile_buf<vec, 16x16xi8>, !pto.tile_buf<vec, 16x16xf16>,
             !pto.tile_buf<vec, 16x16xf16>, !pto.tile_buf<vec, 16x16xf16>)
         outs(%dst : !pto.tile_buf<vec, 16x16xf16>)
```

---

### 11.2 `pto.tsels`

- **syntax:**
```mlir
pto.tsels ins(%mask, %src, %tmp, %scalar :
              !pto.tile_buf<...>, !pto.tile_buf<...>,
              !pto.tile_buf<...>, <scalar_type>)
          outs(%dst : !pto.tile_buf<...>)
```
- **semantics:** `dst[i, j] = mask[i, j] ? src[i, j] : scalar`.

**Parameter Table:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `mask` | `pto.tile_buf` | Packed predicate-mask carrier. |
| `src` | `pto.tile_buf` | Source tile selected when the predicate bit is true. |
| `tmp` | `pto.tile_buf` | Scratch tile for A2/A3 interface compatibility. |
| `scalar` | signless integer / floating-point scalar | Scalar selected when the predicate bit is false. |
| `dst` | `pto.tile_buf` | Destination tile buffer. |

**Constraints:**

- `src` and `dst` must have the same shape and valid region.
- `tsels` accepts packed-mask carrier tiles with `i8`, `i16`, or `i32` element types.

**Example:**

```mlir
pto.tsels ins(%mask, %src, %tmp, %scalar :
              !pto.tile_buf<vec, 16x16xi8>, !pto.tile_buf<vec, 16x16xf16>,
              !pto.tile_buf<vec, 16x16xf16>, f16)
          outs(%dst : !pto.tile_buf<vec, 16x16xf16>)
```

<a id="tile-12-fill-and-padding-ops"></a>

## 12. Fill and Padding Operations

> **Category:** Tile-local fill, pad, and expansion materialization
> **Pipeline:** PIPE_V

This chapter documents the unified TileLib fill / padding operation. It preserves or materializes valid data and then synthesizes the remaining destination region from the destination tile's padding policy.

The destination tile's `pad` / `pad_value` configuration determines which value is written into the synthesized padding or expansion region.

---

### 12.1 `pto.tfillpad`

- **syntax:**
```mlir
pto.tfillpad ins(%src : !pto.tile_buf<...>)
             outs(%dst : !pto.tile_buf<...>)
```
- **semantics:** PTOAS infers normal, in-place, or expand behavior from physical tile shapes and post-PlanMemory addresses. Users do not specify a mode.

**Parameter Table:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `src` | `pto.tile_buf` | Source tile. |
| `dst` | `pto.tile_buf` | Destination tile carrying the pad configuration. |

**Inference Table:**

| Compiler condition | Behavior | PTO-ISA mapping |
|--------------------|----------|-----------------|
| VEC, equal physical shapes, and different or unprovable addresses | Copy valid data, then fill padding. | `TFILLPAD(dst, src)` |
| VEC, equal physical shapes, and identical starting addresses after memory planning | Skip the copy phase and fill padding on shared storage. | `TFILLPAD<pto::TFillPadMode::InPlace>(dst, src)` |
| VEC, destination physical shape is at least the source shape in every dimension and larger in at least one | Copy into the larger destination and fill the expanded region. | `TFILLPAD<pto::TFillPadMode::Expand>(dst, src)` |
| Supported non-VEC form, regardless of address equality | Use the architecture's normal overload. | `TFILLPAD(dst, src)` |

**Constraints:**

- Source and destination element types must be compatible.
- The destination tile must carry a meaningful pad configuration.
- In-place and expand lowering are VEC-only. Normal lowering also supports the homogeneous MAT overload.
- Expand inference compares physical `shape`, not `valid_shape`.
- Equal physical shapes use exact starting-address equality after PlanMemory to select in-place lowering; an unprovable address relationship selects normal lowering.
- MAT always uses Normal lowering, including when source and destination share the same starting address.

**Example:**

```mlir
pto.tfillpad ins(%src : !pto.tile_buf<vec, 8x64xf32, valid=?x?>)
             outs(%dst : !pto.tile_buf<vec, 8x64xf32, pad=1>)

pto.tfillpad ins(%tile : !pto.tile_buf<vec, 32x32xf32, pad=1>)
             outs(%tile : !pto.tile_buf<vec, 32x32xf32, pad=1>)

pto.tfillpad ins(%src_small : !pto.tile_buf<vec, 4x32xf32>)
             outs(%dst_large : !pto.tile_buf<vec, 8x64xf32, pad=1>)
```

<a id="tile-13-mx-quantization"></a>

## 13. MX Quantization and Exponent Layout Conversion

> **Category:** A5 tile-local grouped quantization and exponent packing
> **Pipeline:** PIPE_V

This chapter defines the A5 MX quantization surface. `pto.tquant.mx` produces
the quantized payload plus exponent, maximum, and scaling tiles. A following
three-tile `pto.tmov` can pack the exponent tile from ND or DN order into the ZZ
box order used by Cube operations.

### 13.1 `pto.tquant.mx`

**Syntax:**

```mlir
pto.tquant.mx ins(%src : !pto.tile_buf<...>)
              outs(%dst, %exp, %max, %scaling :
                    !pto.tile_buf<...>, !pto.tile_buf<...>,
                    !pto.tile_buf<...>, !pto.tile_buf<...>)
              {quant_type = #pto<quant_type MXFP8>,
               quantScaleAlg = #pto<quant_scale_alg ocp>,
               grpAxis = #pto<mx_group_axis axis0>,
               interleave = false}
```

`quantScaleAlg`, `grpAxis`, and `interleave` default to `ocp`, `axis1`, and
`false`, respectively.

**Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `src` | vec tile | f32/f16/bf16 source values. |
| `dst` | vec tile | MXFP8 bytes or packed MXFP4 E2M1 pairs. |
| `exp` | vec tile | One shared 8-bit exponent per group. |
| `max` | vec tile | One maximum-magnitude value per group, stored in the source type. |
| `scaling` | vec tile | One reciprocal scale per group, stored in the source type. |
| `quant_type` | `#pto.quant_type` | `MXFP8` or `MXFP4_E2M1`. |
| `quantScaleAlg` | `#pto.quant_scale_alg` | `ocp` or `nv`. |
| `grpAxis` | `#pto.mx_group_axis` | `axis0` groups rows; `axis1` groups columns. |
| `interleave` | bool | Interleaves the axis0 exponent output; false by default. |

For each 32-element group, the operation computes the maximum magnitude,
derives a shared exponent and reciprocal scale according to the selected
algorithm, and quantizes the scaled values. `MXFP8` produces one output byte
per source element. `MXFP4_E2M1` packs two output values per byte.

Let `M=src.valid_rows`, `N=src.valid_cols`, `Ps=src.cols`, and
`G=M*N/32`.

**Valid-shape table:**

| Mode | `exp` | `max` / `scaling` | `dst` |
|------|-------|-------------------|-------|
| axis0 | `[M/32, N]` | `[M/32, N]` | `[M, N]` for MXFP8; `[M, N/2]` for packed MXFP4 |
| axis0 + interleave | `[M/64, 2N]` | `[M/32, N]` | Same as non-interleaved axis0 |
| axis1 canonical | `[M, N/32]` | `[M, N/32]` | `[M, N]` for MXFP8; `[M, N/2]` for packed MXFP4 |
| axis1 legacy flat | `[1, G]` | `[1, G]` | Same as canonical axis1 |

Axis0 requires `M % 32 == 0`; interleave additionally requires
`M % 64 == 0`. Axis1 requires `N % 32 == 0`. Legacy-flat axis1 requires a
tight source (`Ps == N`) and `exp.rows == 1`.

**Physical-layout table:**

| Mode/tile | Contract |
|-----------|----------|
| axis0 MXFP4 `dst` | `dst.cols == Ps/2`; capacity >= `(M-1)*(Ps/2) + N/2` bytes. |
| axis0 `max/scaling` | Physical columns equal `Ps`; capacity >= `(M/32)*Ps` elements. |
| axis0 non-interleaved `exp` | Physical columns equal `Ps`; capacity >= `(M/32)*Ps` elements. |
| axis0 interleaved `exp` | Exact physical shape `[src.rows/64, alignTo(2*Ps, 32)]`; `src.rows % 64 == 0`. |
| axis1 MXFP8 `dst` | `dst.cols == Ps`; capacity >= `(M-1)*Ps + N` bytes. |
| axis1 MXFP4 `dst` | `dst.cols == Ps/2`; capacity >= `(M-1)*(Ps/2) + N/2` bytes. |
| axis1 `max/scaling` | Valid values form a compact allocation prefix. |
| axis1 canonical `exp` | Physical shape covers `[M, N/32]`; capacity >= `(M-1)*exp.cols + N/32`. |
| axis1 flat `exp` | `exp.rows == 1`; capacity >= `G` elements. |

Flat axis1 scaling writes full vector widths. Its required capacity is
`alignTo(G,64)*4` bytes for ordinary f32, `alignTo(2G,64)*4` bytes for the
unrolled f32 path, and `alignTo(G,128)*2` bytes for f16/bf16 OCP. These bounds
are at least 256, 512, and 256 bytes, respectively. The f32 path is unrolled
when `src.rows*src.cols > 1024` and both `src.rows*src.cols` and
`src.valid_rows*src.cols` are multiples of 256.

The pinned A5 revisions reject these combinations:

- axis1 canonical f16/bf16;
- axis1 legacy-flat f16/bf16 with NV scaling;
- axis0 FP16 MXFP4 when `Ps/2` is not a multiple of 32 bytes;
- axis0 FP32 MXFP8 with `interleave=true`;
- padded f16/bf16 source whose final vector-aligned padding store is incomplete.

Padded f16/bf16 source may be zeroed in place, so it has Read+Write effects.
The source allocation must not overlap any output allocation. All relevant
valid and physical dimensions are static.

Four-output lowering uses the grouped form
`TQUANT<grp_axis, MxQuantAlg[, interleave]>`. The `MxQuantAlg` value is selected
from the Cartesian product of `quant_type` and `quantScaleAlg`.

The older `exp_zz + storeMode` fused form remains readable but is deprecated
and restricted to axis1. Use the four-output form followed by §13.2 instead.

**Example:**

```mlir
pto.tquant.mx ins(%src : !pto.tile_buf<vec, 64x64xf32>)
              outs(%dst, %exp, %max, %scaling :
                    !pto.tile_buf<vec, 64x64xi8>,
                    !pto.tile_buf<vec, 2x64xui8>,
                    !pto.tile_buf<vec, 2x64xf32>,
                    !pto.tile_buf<vec, 2x64xf32>)
              {quant_type = #pto<quant_type MXFP8>,
               grpAxis = #pto<mx_group_axis axis0>}
```

### 13.2 `pto.tmov` MX Exponent X-to-ZZ Form

**Syntax:**

```mlir
pto.tmov ins(%exp : !pto.tile_buf<vec, ...>,
             %tmp : !pto.tile_buf<vec, ...>)
         outs(%exp_zz : !pto.tile_buf<vec, ..., slayout=row_major>)
         {grpAxis = #pto<mx_group_axis axis0>}
```

The existing optional operand name remains `fp`. A third tile in
`loc=scaling` still selects FP TMOV. A non-scaling third tile selects X-to-ZZ
and is interpreted as `%tmp`; `grpAxis` is legal only in that form and defaults
to `axis1`.

All three tiles are rank-2 `loc=vec`, use the same type from
`{ui8, !pto.hif8, !pto.f8E8M0}`, and have static source/destination valid and
physical shapes plus a static temporary physical shape. Source layout is
row-major/none-box; destination layout is row-major/row-major. Signed `i8` is
not legal. X-to-ZZ cannot carry `preQuantScalar`, `accToVecMode`, or a
non-default `reluPreMode`.

**Axis0 (DN) mapping:**

```text
dst[col_block, pair, q, delta] = src[2*pair + delta, 16*col_block + q]
```

`src.valid_rows` is even and at least 2, `src.valid_cols` is a multiple of 16,
and `src.cols == src.valid_cols`. Source and destination capacity each cover
`src.valid_rows * src.valid_cols` bytes. The temporary tile is not accessed and
has no minimum capacity beyond a valid static tile. Lowering emits
`TMOV<0>(dst, src, tmp)`.

**Axis1 (ND) mapping:**

```text
dst[row_block, col_pair, row_in_block, pair_lane] =
    src[16*row_block + row_in_block, 2*col_pair + pair_lane]
```

`dst.valid_cols` is even. Source valid values form a compact prefix. Source and
destination capacity each cover
`alignTo(dst.valid_rows,16) * dst.valid_cols` bytes. Temporary capacity covers
`64 + ceil(dst.valid_rows/16) * dst.valid_cols` bytes. Padding rows are zeroed
in the source before packing, so source and temporary have Read+Write effects.
Lowering emits `TMOV(dst, src, tmp)`.

For both axes, source and destination valid element counts match and their byte
ranges do not overlap, including through subviews or reshaped views.

**Example:**

```mlir
pto.tmov ins(%exp : !pto.tile_buf<vec, 2x16xui8>,
             %tmp : !pto.tile_buf<vec, 1x1xui8>)
         outs(%exp_zz : !pto.tile_buf<vec, 2x16xui8, slayout=row_major>)
         {grpAxis = #pto<mx_group_axis axis0>}
```
