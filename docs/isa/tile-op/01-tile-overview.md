# 1. Tile and PTO Tile Instruction Overview

> **Category:** Foundational concepts

This chapter introduces both the tile data model and the **Tile Instruction** surface that operates on it. Read this before any of the per-group Tile Instruction references.

---

## 1.1 What is PTO Tile Instruction

**PTO Tile Instruction** is a high-performance instruction library built on top of [PTO micro Instruction](../micro-isa/01-pipeline-sync.md). Each tile instruction encapsulates a tile-granular pattern — DMA between GM and on-chip buffers, vector arithmetic over a whole tile, reductions, broadcast / expansion, selection, padding — that internally expands to a sequence of micro-instruction primitives (`pto.vlds`, `pto.vsts`, `pto.vadd`, mask ops, sync flags, …).

For the kernel author this means:

- **Author at the tile level.** Use `pto.tload`, `pto.tadd`, `pto.trowsum`, etc., to express tile-granular DMA and compute without writing the underlying vector loop.
- **Drop down to micro instruction when needed.** Inside `pto.vecscope`, `pto.tile_buf_addr` lowers a tile handle to a UB pointer, so handwritten micro-instruction code can read and write the same on-chip data. The mixing pattern is documented in [§1.10](#110-mixing-pto-tile-instruction-and-pto-micro-instruction).
- **Predictable lowering.** Because every Tile Instruction is templated against micro instruction, a kernel that mixes Tile and micro can share scratch tiles, masks, and pipeline events with no representation gap.

The remaining chapters in this document cover the tile data types, pointer / view ops, DMA, compute families, and op-by-op syntax. The semantics below define the storage contract those ops share.

## 1.2 Tile Buffer Model

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

Placement, lifetime, and reuse affect both correctness and performance. `pto.alloc_tile` makes allocation explicit, and pipeline ordering is expressed through the synchronization primitives described in [`01-pipeline-sync.md`](../micro-isa/01-pipeline-sync.md).

**Explicit buffer lifetime example:**

```mlir
%a0 = pto.alloc_tile : !pto.tile_buf<vec, 16x16xf16>
%a1 = pto.alloc_tile : !pto.tile_buf<vec, 16x16xf16>

pto.tload ins(%pv0 : !pto.partition_tensor_view<16x16xf16>)
          outs(%a0 : !pto.tile_buf<vec, 16x16xf16>)
pto.tload ins(%pv1 : !pto.partition_tensor_view<16x16xf16>)
          outs(%a1 : !pto.tile_buf<vec, 16x16xf16>)
```

## 1.3 Hardware Memory Hierarchy

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

`loc` on a tile buffer selects one of these domains. The full enum (with mnemonics) is defined in [§2.6 AddressSpace](02-types-and-attributes.md#26-addressspace); each tile ISA chapter calls out which `loc` domains are legal for the ops it covers.

## 1.4 Instruction Form

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

## 1.5 Physical Shape vs Valid Region

Every tile buffer has two shape concepts:

- **Physical shape** `(rows, cols)` — the extent of backing storage; static and known when the tile buffer type is declared.
- **Valid region** `(v_row, v_col)` — the populated sub-rectangle; either static or dynamic (`?`).

The physical shape drives layout, fractal alignment, and buffer-size accounting. The valid region drives the iteration domain of compute and DMA ops. **Undefined behavior:** elements outside the valid region are padding — their contents must not be read.

When the valid region is dynamic (`v_row = ?` or `v_col = ?`), it is provided at `pto.alloc_tile` time (or updated later with `pto.set_validshape`). Most Tile Instruction ops use the destination valid region as the iteration domain; a few ops require all operands to share the same valid region.

## 1.6 Pipeline Association

Every Tile Instruction op is associated with a hardware pipeline in the Decoupled Access-Execute architecture:

| Pipeline | Symbol | Typical ops |
|----------|--------|------------|
| DMA inbound | `PIPE_MTE2` | `pto.tload` |
| DMA outbound | `PIPE_MTE3` | `pto.tstore` |
| Vector | `PIPE_V` | `pto.tadd`, `pto.tadds`, `pto.texp`, `pto.tcvt`, and the rest of the vector arithmetic set |
| Scalar | `PIPE_S` | scalar `arith`/`scf` ops interleaved with tile code |

Cross-pipeline data dependencies are ordered explicitly, either via the **Flag/Event** mechanism (`pto.set_flag`/`pto.wait_flag`) or the **Buffer-ID** mechanism (`pto.get_buf`/`pto.rls_buf`). See [`01-pipeline-sync.md`](../micro-isa/01-pipeline-sync.md) for the full semantics.

## 1.7 Scratch Operands and A2/A3 Compatibility

Some Tile Instruction ops carry an extra `%tmp` tile operand whose only purpose is to keep the operand list aligned with the corresponding A2/A3 PTO instruction interface. Examples include `pto.txor` / `pto.txors` ([Chapter 8](08-bitwise-shift-ops.md)) and `pto.tsel` / `pto.tsels` ([Chapter 11](11-selection-ops.md)).

`%tmp` exists for cross-arch interface compatibility — A5 templates may not materially use it, but it remains in the public op signature so the same Tile IR can be reused across A2/A3 and A5. Treat it as a required operand whose dtype/shape constraints are stated by the individual op page.

## 1.8 Conventions for Chapters 5–12

Unless an op page states otherwise, the chapters that follow assume:

- tile operands use `loc=vec`;
- tile layouts use `blayout=row_major` and `slayout=none_box`;
- valid bounds satisfy `v_row <= rows` and `v_col <= cols`;
- examples use the compact `!pto.tile_buf<loc, RxCxdtype[, valid=...]>` form. Omitted attributes carry their default values: `valid` = physical shape, `blayout=row_major`, `slayout=none_box`, `fractal=512`, `pad=0`.

The op pages call out any deviation from these conventions explicitly.

## 1.9 Minimal End-to-End Example

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

Synchronization is omitted for clarity; for the real ordering contracts (`pto.set_flag`/`pto.wait_flag`, `pto.get_buf`/`pto.rls_buf`, `pto.pipe_barrier`) see [`01-pipeline-sync.md`](../micro-isa/01-pipeline-sync.md).

<a id="110-mixing-pto-tile-instruction-and-pto-micro-instruction"></a>

## 1.10 Mixing PTO Tile Instruction and PTO micro Instruction

PTO Tile Instruction and PTO micro Instruction can be authored side-by-side in the same kernel. The Tile Instruction surface owns tile placement and GM ↔ on-chip DMA; the micro surface owns vector-register compute inside `pto.vecscope`. The two surfaces meet through `pto.tile_buf_addr`, which converts a tile handle into a UB pointer that vector ops can consume.

This section presents a softmax kernel that uses both surfaces together, then walks through it.

### Kernel Structure

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

### Kernel Listing

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
                          {dist = "BRC_B32"} : !pto.ptr<f32, ub>, index -> !pto.vreg<64xf32>
        %final_max, %final_sum = scf.for %chunk = %c0 to %c128 step %c64
            iter_args(%running_max = %oldmax_bc, %running_sum = %oldsum_bc)
            -> (!pto.vreg<64xf32>, !pto.vreg<64xf32>) {
          %base = arith.addi %row_qk, %chunk : index
          %vec  = pto.vlds %ub_qk[%base]
                      : !pto.ptr<f32, ub>, index -> !pto.vreg<64xf32>
          // ... running_max / running_sum update via vcmax / vexpdif / vmul / vadd ...
          scf.yield %merged_max, %merged_sum : !pto.vreg<64xf32>, !pto.vreg<64xf32>
        }

        // Persist the row-local results back to UB.
        pto.vsts %final_max, %ub_newmax[%row], %one_mask
                {dist = "1PT_B32"}
                : !pto.vreg<64xf32>, !pto.ptr<f32, ub>, index, !pto.mask<b32>

        // Second pass: write softmax output back into the qk tile's UB region.
        scf.for %chunk = %c0 to %c128 step %c64 {
          %base = arith.addi %row_qk, %chunk : index
          %vec  = pto.vlds %ub_qk[%base]
                      : !pto.ptr<f32, ub>, index -> !pto.vreg<64xf32>
          %exp  = pto.vexpdif %vec, %final_max, %chunk_mask, "ODD"
                      : !pto.vreg<64xf32>, !pto.vreg<64xf32>, !pto.mask<b32>
                     -> !pto.vreg<64xf32>
          %out  = pto.vdiv %exp, %final_sum, %chunk_mask
                      : !pto.vreg<64xf32>, !pto.vreg<64xf32>, !pto.mask<b32>
                     -> !pto.vreg<64xf32>
          pto.vsts %out, %ub_out[%base], %chunk_mask
                      : !pto.vreg<64xf32>, !pto.ptr<f32, ub>, index, !pto.mask<b32>
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

### Code Walkthrough

The seven numbered comments in the listing above mark the seven steps from §Kernel Structure. The notes below highlight what each step contributes to the Tile / micro split.

**(1) GM views and partitions** — pure metadata. `pto.make_tensor_view` records the GM tensor's shape and strides; `pto.partition_view` carves out the per-block sub-window. Neither op moves data, and both stay outside `pto.vecscope`. The 5-D shape is a quirk of this kernel's layout convention; the boundary rules don't depend on rank.

**(2) `pto.alloc_tile` with static size and address** — declares the UB tile handles. The result type fixes the static physical shape (e.g. `8x128xf32`); `addr = %c256_i64` pins the tile to a specific UB byte offset; `valid_row = ...` / `valid_col = ...` carry the runtime valid extents (the `?` markers in `valid=?x?`). Because addresses are hand-assigned, this kernel compiles with `--pto-level=level3` and disables `--enable-insert-sync`.

**(3) `pto.tload`** — copies a GM partition into the UB tile. Runs on `PIPE_MTE2`. Stays in the Tile domain; it cannot appear inside `pto.vecscope`.

**(4) MTE2 → V flag handshake** — DMA inbound and the vector pipeline run asynchronously. The producer/consumer edge between `tload` and the upcoming `vecscope` must be made explicit with `pto.set_flag` / `pto.wait_flag`.

**(5) Vector region** — `pto.vecscope` opens a vector-execution region. The first thing inside is a series of `pto.tile_buf_addr` ops, each lowering a tile handle into a `!pto.ptr<f32, ub>`. From that point on the body is pure micro: `pto.vlds` reads UB into vregs, vector arithmetic / SFU / mask ops compute on vregs, and `pto.vsts` writes vregs back to UB. Tile ops are forbidden inside this region; `pto.tile_buf_addr` is forbidden outside.

**(6) V → MTE3 flag handshake** — mirror of step (4), this time gating the vector results visible to the outbound DMA.

**(7) `pto.tstore`** — writes each UB tile back to its GM partition, completing the round trip. Same Tile-domain rules as `tload`.

### Where the Tile and Micro Boundaries Sit

| Op | Where it must live | Why |
|----|-------------------|-----|
| `pto.alloc_tile`, `pto.tload`, `pto.tstore`, `pto.tadd`, … (Tile domain) | **Outside** `pto.vecscope` | Tile ops describe tile residency and tile-granular DMA / compute; they have no meaning inside a vector-register region. |
| `pto.vlds`, `pto.vsts`, `pto.vmax`, `pto.vexpdif`, … (micro domain) | **Inside** `pto.vecscope` | These ops produce/consume `!pto.vreg` and `!pto.mask` values that only exist inside a vector region. |
| `pto.tile_buf_addr` | **Inside** `pto.vecscope` only | This is the single sanctioned bridge from a tile handle to a UB pointer; outside vecscope, tile handles must be consumed by Tile ops, not by address extraction. |
| `pto.set_flag` / `pto.wait_flag` (and other sync primitives) | Either side | Sync ops belong to whichever pipeline edge they coordinate; in this kernel they appear at the MTE2 → V and V → MTE3 boundaries. |

In short: keep DMA and tile shape management in Tile-land, keep vreg/mask compute in vecscope, and use `pto.tile_buf_addr` exactly at the boundary.
