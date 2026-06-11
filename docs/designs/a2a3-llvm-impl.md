# A3 VPTO LLVM Lowering Pipeline

## Overview

The A3 (`dav-c220-vec`) backend reuses the same top-level VPTO emission pipeline
as A5 (`dav-c310-vec`) but replaces the TileLang-template expansion path with a
direct pointer-based lowering of binary tile operations to UB intrinsics.

## Pipeline: Shared vs Divergent

### Shared passes (arch-independent)

```
runPipeline(module, march, diagOS, emit):
  1. VPTOSplitCVModule            — split cube/vector sections into child modules
  2. VPTONormalizeContainer       — canonicalize kernel container shape
  3. VPTOPtrNormalize             — normalize ptr-like values into !pto.ptr
  4. VPTOPtrCastCleanup           — collapse transient ptr-memref-ptr bridge casts
  5. VPTOExpandWrapperOps         — expand DMA/bridge/mad/acc-store wrappers
  6. PTOInferVPTOVecScope         — infer vecscope regions
  7. PrepareVPTOLLVMLowering      — materialize vecscope carrier loops
  8. LowerVPTOOpsPass             — pattern-based op → LLVM intrinsic lowering
  9. LowerVPTOTypesPass           — type conversion (pto.ptr → llvm.ptr)
 10. Arith / SCF / CF / Func → LLVM  (standard MLIR conversion passes)
 11. ReconcileUnrealizedCasts     — cleanup
```

### Divergence: pre-VPTO backend pipeline

```
lowerPTOToVPTOBackend(pm):

  ┌─ A3 ────────────────────────────────────────────────────┐
  │ LowerPTOToUBufOps                                        │
  │   pto.tadd / tsub / tmul / tdiv                         │
  │   → pto.ub.vadd / vsub / vmul / vdiv                    │
  │   + pto.ub.set_mask / set_mask_count / set_mask_norm    │
  │   (pointer-based, !pto.ptr<f32, ub>, no memref)         │
  └─────────────────────────────────────────────────────────┘

  ┌─ A5 ────────────────────────────────────────────────────┐
  │ ExpandTileOp                                             │
  │   → TileLang DSL template expansion                     │
  │ InlineLibCall → FoldTileBufIntrinsics                   │
  │   pto.tadd → pto.vlds / vadd / vsts                     │
  │   (memref-based, vreg values)                            │
  └─────────────────────────────────────────────────────────┘
```

### Divergence inside shared passes

| Pass | A3 (`dav-c220-vec`) | A5 (`dav-c310-vec`) |
|---|---|---|
| **VPTOExpandWrapperOps** | v220 DMA: `nBurst=1`, `lenBurst>>5`, `scf.for` software loops | Standard DMA: dual-config calls, `nBurst` loop pairs |
| **LowerVPTOOpsPass** | UB binary ops → `llvm.hivm.VADD/VSUB/VMUL/VDIV.f32` + mask via `SBITSET1/SBITSET0` on CTRL[56] | Vector ops → `llvm.hivm.ADD/LOAD/STORE` |
| **CopyOp (DMA)** | 1-config: `MOV.OUT.TO.UB.v220` / `MOV.UB.TO.OUT.v220.1` | 2-config: `MOV.OUT.TO.UB.ALIGN.V2.s32.DV` / `MOV.UB.TO.OUT.ALIGN.V2.DV` |
| **makeDeviceEmissionOptions** | `dav-c220-vec`, features: `+ASAN,+ATOMIC,+FFTSBlk,+MOVX8,+MSTX,+MathOp` | `dav-c310-vec`, features: `+ArchV130,+F8e4m3,+F8e5m2,+Fp4e1m2x2,+LDExtRefine` |
| **configureVPTOOpLoweringTarget** | UB ops (`UBVaddOp`, `UBSetMaskOp`, etc.) marked illegal | Not present |

## TADD → ub.vadd: Tile-size lowering examples

### Repeat & stride calculation

```
elementsPerRepeat = 256 / sizeof(element)          // 64 for f32, 128 for f16, 32 for f64
totalElements     = rows × cols
totalRepeats      = totalElements / elementsPerRepeat

blockStrides  = (1, 1, 1)                          // always contiguous
repStride     = elementsPerRepeat / 8              // 8 for f32, 16 for f16, 4 for f64
```

### CCE VADD config layout (i64)

| Bits | Field |
|---|---|
| 7:0 | repeat |
| 15:8 | dst block stride |
| 23:16 | src0 block stride |
| 31:24 | src1 block stride |
| 39:32 | dst repeat stride |
| 47:40 | src0 repeat stride |
| 55:48 | src1 repeat stride |
| 63:56 | simdFlag (=1) |

### Example 1: 1×64 f32 — Direct count mode

Input:
```mlir
%t = pto.alloc_tile : !pto.tile_buf<loc=vec, dtype=f32, rows=1, cols=64, v_row=1, v_col=64>
pto.tadd ins(%src0, %src1 : ...) outs(%dst : ...)
```

Computed: `elementsPerRepeat=64`, `totalElements=64`, `totalRepeats=1`.

Output:
```mlir
%ub_dst = pto.tile_buf_addr %dst : !pto.tile_buf<...> → !pto.ptr<f32, ub>
%ub_s0  = pto.tile_buf_addr %src0 : !pto.tile_buf<...> → !pto.ptr<f32, ub>
%ub_s1  = pto.tile_buf_addr %src1 : !pto.tile_buf<...> → !pto.ptr<f32, ub>
%c1 = arith.constant 1 : i64
%c8 = arith.constant 8 : i64

pto.ub.set_mask_count                            // SBITSET1(ctrl, 56)
pto.ub.vadd %ub_dst, %ub_s0, %ub_s1,             // VADD: repeat=1, all strides=1/1/1/8/8/8
     %c1, %c1, %c1, %c1, %c8, %c8, %c8
pto.ub.set_mask_norm                             // SBITSET0(ctrl, 56)
pto.ub.set_mask %c-1_i64, %c0_i64                // full mask (mask0=-1, mask1=0)
```

After LLVM lowering:
```mlir
llvm.call @llvm.hivm.SBITSET1(%ctrl, %c56)       // enter count mode
llvm.call @llvm.hivm.MOVEMASK(%c0, %c-1)         // mask[0] = -1
llvm.call @llvm.hivm.MOVEMASK(%c1, %c0)          // mask[1] = 0 (64-lane f32)
llvm.call @llvm.hivm.VADD.f32(%ub_dst, %ub_s0, %ub_s1, %config)
llvm.call @llvm.hivm.SBITSET0(%ctrl, %c56)       // exit count mode
```

### Example 2: 4×64 f32 — Multi-row count mode with scf.for

Input:
```mlir
%t = pto.alloc_tile : !pto.tile_buf<loc=vec, dtype=f32, rows=4, cols=64, v_row=4, v_col=64>
pto.tadd ins(%src0, %src1 : ...) outs(%dst : ...)
```

Computed: `headRepeats=4`, `elementsPerRow=64`, `repeat=1` per row.

Output:
```mlir
%c0 = arith.constant 0 : index
%c1 = arith.constant 1 : i64
%c4 = arith.constant 4 : index
%c8 = arith.constant 8 : i64

%ub_dst = pto.tile_buf_addr %dst : !pto.ptr<f32, ub>
%ub_s0  = pto.tile_buf_addr %src0 : !pto.ptr<f32, ub>
%ub_s1  = pto.tile_buf_addr %src1 : !pto.ptr<f32, ub>

%row_dst = pto.addptr %ub_dst, %c0 : !pto.ptr<f32, ub>
%row_s0  = pto.addptr %ub_s0, %c0 : !pto.ptr<f32, ub>
%row_s1  = pto.addptr %ub_s1, %c0 : !pto.ptr<f32, ub>

scf.for %i = %c0 to %c4 step %c1 {
  %off = arith.index_cast %i : index to i64
  %rd = pto.addptr %row_dst, %off : !pto.ptr<f32, ub>
  %r0 = pto.addptr %row_s0, %off : !pto.ptr<f32, ub>
  %r1 = pto.addptr %row_s1, %off : !pto.ptr<f32, ub>

  pto.ub.set_mask_count
  pto.ub.vadd %rd, %r0, %r1, %c1, %c1, %c1, %c1, %c8, %c8, %c8
  pto.ub.set_mask_norm
}
pto.ub.set_mask %c-1_i64, %c0_i64
```

### Example 3: 16×64 f32 — Larger multi-row

Input:
```mlir
%t = pto.alloc_tile : !pto.tile_buf<loc=vec, dtype=f32, rows=16, cols=64, v_row=16, v_col=64>
pto.tadd ins(%src0, %src1 : ...) outs(%dst : ...)
```

Same pattern as 4×64 but `scf.for` iterates 16 times. One `VADD` per row:
```
scf.for %i = 0 to 16 step 1 {
  pto.ub.set_mask_count
  pto.ub.vadd ...   // repeat=1 per row
  pto.ub.set_mask_norm
}
```

### Example 4: 32×32 f32 — Square tile

Input:
```mlir
%t = pto.alloc_tile : !pto.tile_buf<loc=vec, dtype=f32, rows=32, cols=32, v_row=32, v_col=32>
pto.tadd ins(%src0, %src1 : ...) outs(%dst : ...)
```

Computed: `totalElements=1024`, `elementsPerRepeat=64`, `totalRepeats=16`. With `headRepeats=32` rows, each row has `32/64 = 1` repeat → same `scf.for` per-row pattern, 32 iterations.

### Example 5: 4×96 f32, v_col=32 — Multiple repeats per row

Input:
```mlir
%t = pto.alloc_tile : !pto.tile_buf<loc=vec, dtype=f32, rows=4, cols=96, v_row=4, v_col=32>
pto.tadd ins(%src0, %src1 : ...) outs(%dst : ...)
```

Computed: `totalV = 4×32 = 128`, `elementsPerRepeat = 64`, `totalRepeats = 128/64 = 2` per call but `totalV > REPEAT_MAX(255)` triggers count mode.

Output: count mode, 2 repeats per row, 4 rows via scf.for.

### Example 6: 16×1024 f32 — Chunked (repeat > 255)

Input:
```mlir
%t = pto.alloc_tile : !pto.tile_buf<loc=vec, dtype=f32, rows=16, cols=1024, v_row=16, v_col=1024>
pto.tadd ins(%src0, %src1 : ...) outs(%dst : ...)
```

Computed: `totalElements=16384`, `elementsPerRepeat=64`, `totalRepeats=256`, which equals `REPEAT_MAX`. Count mode, no `scf.for` (single VADD with repeat=256):

```mlir
pto.ub.set_mask_count
pto.ub.set_mask %c256_i64, %c0_i64     // mask0 = 256 (count mode: mask count in mask[0])
pto.ub.vadd %dst, %src0, %src1, %c256, %c1, %c1, %c1, %c8, %c8, %c8
pto.ub.set_mask_norm
pto.ub.set_mask %c-1_i64, %c0_i64      // restore full mask
```

### Example 7: A5 — No UB lowering

On A5 targets, `pto.tadd` passes through `LowerPTOToUBufOps` unchanged and is later expanded via `ExpandTileOp` into the TileLang DSL template path:

```mlir
// VPTO IR (after ExpandTileOp + inlining):
pto.vecscope {
  %v0 = pto.vlds %src0_memref : memref<...>
  %v1 = pto.vlds %src1_memref : memref<...>
  %v2 = pto.vadd %v0, %v1 : !pto.vreg<f32>
  pto.vsts %v2, %dst_memref : memref<...>
}
```

No `pto.ub.vadd` or `llvm.hivm.VADD.f32` appears.

### Mask / control-flow modes

| Mode | Condition | Pattern |
|---|---|---|
| **count mode** | `totalV > kRepeatMax` (255) or row-repeat with count | `SBITSET1(ctrl,56)` → `MOVEMASK(0,count)` → `VADD` → `SBITSET0(ctrl,56)` |
| **norm mode (head+tail)** | `totalRepeats ≤ kRepeatMax` | Head VADD → tail mask VADD |
| **norm mode small** | `v_col < 32`, head+tail ≤ max | Same as head+tail but per chunk via `scf.for` |
| **norm mode 1L** | Single row, repeat ≤ max | Direct norm mode VADD |
| **full mask** | After all VADDs | `MOVEMASK(0,-1)` + `MOVEMASK(1,0)` for 64-lane f32 |

### C220 COUNT mode limitation

On `dav-c220-vec`, count mode only supports `repeat=1`. Multi-element count mode
(`repeat > 1`) causes CCE errors. As a result:

- **repeat=1**: direct count mode (no loop)
- **repeat>1, rows=1**: split into per-chunk `scf.for` with repeat=1 each (`modeCount1L`)
- **repeat>1, rows>1**: per-row `scf.for` with repeat=1 each (`modeNormal1L`)
- **repeat=0** (`modeCount1L`): broken on C220, routed to head+tail path

Test cases `1×80` and `256×64` are excluded from on-device tests due to this
limitation; they fall back to the head+tail norm path.

## Full lowering chain: TADD example

```
PTO DSL:          pto.tile.add(a_tile, b_tile, c_tile)
    ↓
PTO IR:           pto.tadd ins(%a, %b) outs(%c)
    ↓ LowerPTOToUBufOps (A3 only)
UB IR:            pto.ub.set_mask_count
                  pto.ub.vadd %dst, %src0, %src1, repeat=1, blkStrides=(1,1,1), repStrides=(8,8,8)
                  pto.ub.set_mask_norm
                  pto.ub.set_mask %c-1, %c0
    ↓ Lower copy ops
Copy IR:          pto.copy_gm_to_ubuf / pto.copy_ubuf_to_gm  (v220 DMA)
    ↓ LowerVPTOOpsPass (A3 patterns)
LLVM IR:          llvm.call @llvm.hivm.SBITSET1(%ctrl, %56)
                  llvm.call @llvm.hivm.MOVEMASK(%0, %-1)
                  llvm.call @llvm.hivm.MOVEMASK(%1, %0)
                  llvm.call @llvm.hivm.VADD.f32(%dst, %src0, %src1, %config)
                  llvm.call @llvm.hivm.SBITSET0(%ctrl, %56)
                  llvm.call @llvm.hivm.MOV.OUT.TO.UB.v220(...)
                  llvm.call @llvm.hivm.MOV.UB.TO.OUT.v220.1(...)
    ↓ bisheng
Binary:           native AICORE binary
```

## Summary table

| Aspect | A3 | A5 |
|---|---|---|
| **Lowering strategy** | Direct pointer-based: `pto.tadd → ub.vadd → llvm.hivm.VADD` | Template expansion: `pto.tadd → TileLang DSL → vlds/vadd/vsts` |
| **Memory model** | `!pto.ptr<f32, ub>` pointers, no memref | `memref` alloc/load/store |
| **DMA** | v220 single-config, software loops | Dual-config, hardware nBurst loops |
| **Mask/control** | `SBITSET1/SBITSET0` on CTRL bit 56 for count mode | `SET_FLAG/WAIT_FLAG` synchronization |
| **Passes added** | `LowerPTOToUBufOps` (1 new pass) | `ExpandTileOp` + `InlineLibCall` + `FoldTileBufIntrinsics` |
| **New ops (6)** | `pto.ub.{vadd,vsub,vmul,vdiv,set_mask_count,set_mask_norm}` | none |
| **Intrinsics** | `llvm.hivm.{VADD,VSUB,VMUL,VDIV}.f32` | `llvm.hivm.{ADD,LOAD,STORE}` |
| **Relevant files** | `LowerPTOToUBufOps.cpp`, `VPTOUbOps.td`, `VPTOUbOps.cpp`, `VPTOLLVMEmitter.cpp`, `VPTOExpandWrapperOps.cpp`, `ObjectEmission.cpp` | `ExpandTileOp.cpp`, `FoldTileBufIntrinsics.cpp`, `VPTOLLVMEmitter.cpp` |
