# A2/A3 VPTO Gather Lowering — Design Notes

Scope: bring tile-level gather (`pto.tgather`, `pto.tgatherb`, and the GM-to-UB
form of `pto.mgather`) to the A2/A3 `vpto` backend so it reaches feature parity
with the EmitC path for gather,
using the flat `pto.ub.*` UB-pointer op layer (the same layer as the
elementwise `ub.vadd` family). The A5 `vreg`/`vgather2` path is **out of
scope** for A2/A3 and must not be used.

## Background: two lowering tiers for gather

PTOAS has two parallel lowering tiers for gather, one per backend family:

| Backend | Gather tier | Primitive | Where |
|---|---|---|---|
| A5 (`vpto`) | vreg | `vgather2` / `vgather2_bc` (vreg operands) | `VPTOLLVMEmitter.cpp:7926` lowers to `llvm.hivm.vgather2.v300.*` |
| A2/A3 (`vpto`) | flat UB pointer (`pto.ub.*`) | raw CCE `vgather` / `vgatherb` (UB pointers) | implemented by this branch |
| Any arch (EmitC) | tile-level CCE call | `TGATHER` / `TGATHERB` opaque calls | `PTOToEmitC.cpp:9772` (tgather), `:9875` (tgatherb) |

The EmitC path already covers gather for A2/A3 (`ptoas --pto-arch=a3 %s` →
bisheng → pto-isa `TGATHER*`). The gap this design closes is the **vpto/A2/A3**
path, which previously had no gather lowering at all.

## Raw CCE primitives for A2/A3 gather

Read from the pto-isa source (`include/pto/npu/a2a3/`). A2/A3 gather is built
on flat `__ubuf__` pointer intrinsics with repeat/stride configuration — the
same shape as the `pto.ub.*` elementwise ops — **not** vreg.

### `vgather` — element-indexed gather (TGather index/mask/compare form)

`npu/a2a3/TGather.hpp` implements the index/mask/compare forms on top of the
raw CCE `vgather`:

```cpp
__ubuf__ T* dstPtr  = (__ubuf__ T*)__cce_get_tile_ptr(dst);
__ubuf__ T* src0Ptr = (__ubuf__ T*)__cce_get_tile_ptr(src0);
...
constexpr unsigned elementsPerRepeat = REPEAT_BYTE / sizeof(T);
unsigned numRepeatPerLine = validCol / elementsPerRepeat;
...
vgather((__ubuf__ uint32_t*)(dstPtr + i * TShape1), ...);  // 32-bit lanes
vgather((__ubuf__ uint16_t*)(dstPtr + i * TShape1), ...);  // 16-bit lanes
```

Three forms live in `TGather.hpp`: index (`TGather<D,S0,S1,Tmp>`), mask
(`TGather<D,S,MaskPattern>`), and compare (`TGather_cmp<...,CmpMode,offset>`).
A2/A3 supports `CmpMode::GT | EQ` only (`TGather.hpp:174`).

### `vgatherb` — byte-offset gather (TGatherB)

`npu/a2a3/TGatherB.hpp` wraps the raw CCE `vgatherb`:

```cpp
// GatherBInstr
vgatherb((__ubuf__ T*)dst, offset /*__ubuf__ uint32_t**/, srcAddr,
         dstRepeatStride, /*deCal=*/1, repeats);
// b8 variant widens to uint16_t lanes
vgatherb((__ubuf__ uint16_t*)dst, offset, srcAddr, dstRepeatStride, 1, repeats);
```

The tile driver (`GatherBlockHead`/`GatherBlockTail`) loops over
`validRow × numRepeatPerLine` with the `REPEAT_MAX` chunking — structurally
identical to `LowerPTOToUBufOps`'s `dispatch` head/tail repeat loop.

### Related A2/A3 operations

- **`MGather` exists on A2/A3** — `npu/a2a3/MGather.hpp` implements GM gather
  loads for row and element coalescing. Its GM-to-UB VPTO lowering is implemented
  as a stacked follow-up using scalar address generation, scalar loads/stores,
  and per-row MTE2 copies rather than the UB-local `vgather`/`vgatherb`
  instructions covered here.
- **No vreg** — A5's `vgather2`/`vgather2_bc` (`npu/a5/TGather.hpp:51,79,108`)
  must not be used on A2/A3.

## EmitC reference (the contract to match)

The EmitC lowering emits arch-agnostic CCE opaque calls that bisheng resolves
to the pto-isa templates above. This is the semantic contract the new vpto
lowering must reproduce:

| Tile op | EmitC call (`PTOToEmitC.cpp`) | Forms |
|---|---|---|
| `pto.tgather` | `emitc::CallOpaqueOp "TGATHER"` (`:9772`) | index `TGATHER(dst,src0,indices,tmp)`; compare `TGATHER<D,S,Tmp,C,CmpMode>(dst,src0,k,tmp,c,offset)`; mask `TGATHER<D,S,MaskPattern::Pxxxx>(dst,src0)` |
| `pto.tgatherb` | `emitc::CallOpaqueOp "TGATHERB"` (`:9875`) | `TGATHERB(dst, src, offsets)` |
| `pto.mgather` | `emitc::CallOpaqueOp "MGATHER<Coalesce[,OOB]>"` (`:3323`) | GM-to-UB Row/Elem lowering is implemented for A2/A3 VPTO; GM-to-L1 remains deferred |

MGATHER semantics: `dst[r,:]=table[idx[r],:]` (Coalesce::Row)
or `dst[i,j]=table[idx[i,j]]` (Coalesce::Elem), with `GatherOOB` modes.

### A2/A3 VPTO MGATHER decomposition

`LowerPTOToUBufOps` lowers GM-to-UB MGATHER without introducing a new UB
intrinsic:

- `Coalesce::Row` loads each `i32` index on the scalar pipe, normalizes it for
  `Undefined`, `Clamp`, `Wrap`, or `Zero`, then issues a row-sized
  `pto.mte_gm_ub`. The scalar-to-MTE2 handoff uses the macro-reserved
  `EVENT_ID0`.
- `Coalesce::Elem` performs scalar indexed GM loads and UB stores for each valid
  destination element, with the same four OOB policies.
- The lowering accepts the A2/A3 payload set
  (`i8/ui8/i16/ui16/i32/ui32/f16/bf16/f32`) with signless `i32` indices and
  row-major, none-box UB tiles. Source and destination element types must match.
- `Coalesce::Row` requires a statically positive effective rank-2 ND table,
  exact source/destination row-width agreement, unit innermost stride, and a
  32-byte-aligned row transfer. `Coalesce::Elem` requires a statically positive,
  dense contiguous table and treats each index as a direct flat element offset.
- Unsupported dynamic, empty, strided Elem, column-major, row-plus-one, scratch,
  and implicit-coalesce forms fail at the A2/A3 VPTO lowering boundary.
- GM-to-L1 MGATHER is rejected explicitly at the A2/A3 VPTO lowering boundary
  until its separate NZ/scratch path is implemented.

## Current `pto.ub.*` layer

Defined in `include/PTO/IR/VPTOUbOps.td`. Before this work it was
**elementwise-only**:

- Binary: `ub.vadd/vsub/vmul/vdiv/vmax/vmin/vand/vor/vaddrelu`
- Unary: `ub.vnot/vabs/vrelu/vexp/vln/vsqrt/vrsqrt`
- Shift/scalar: `ub.vshl/vshr/vmuls/vadds/vmaxs/vmins`
- Misc: `ub.vdup`, `ub.set_mask[_count|_norm]`

These map to `llvm.hivm.*` for the dav-c220-vec (A2/A3) backend, operating on
flat `PTO_BufferType` (`!pto.ptr<..., ub>`) operands with the repeat /
block-stride / repeat-stride packed into i64 config words
(`cce_aicore_intrinsics_3101.h` layout). This branch adds the corresponding
flat-UB gather ops.

## Plan

### 1. New UB gather ops (`include/PTO/IR/VPTOUbOps.td`)

Add flat-UB-pointer gather ops mirroring `PTO_UBufBinaryOp`/`PTO_UBufUnaryOp`
(PTO_BufferType operands + i64 repeat/stride config), shaped to match the raw
CCE signatures:

- `pto.ub.vgather` — element-indexed gather (wraps `vgather`).
- `pto.ub.vgatherb` — byte-offset gather (wraps `vgatherb`):
  `(dst, src/offsets, srcAddr, repeat, strides…)`.

Verifier + `getEffects` follow the existing UB op pattern (`lib/PTO/IR/VPTO.cpp`
UBVaddOp-style). C++ builders generated via tablegen.

### 2. Tile → UB lowering (`lib/PTO/Transforms/LowerPTOToUBufOps.cpp`)

Add `TGatherOp` / `TGatherBOp` walks to the pass (a2/a3 arch guard already in
place at `:221-224`), following the `dispatch` head/tail repeat loop:
- `pto.tgather` (index form) → `pto.ub.vgather`
- `pto.tgatherb` → `pto.ub.vgatherb`
- Mask/compare forms: decompose to compare+`vgather` (defer; index form first).

Reuse the existing helpers (`extractTileShapeInfo`, `addPtr`, mask count/norm,
`TileBufAddrOp` for tile→UB ptr). Match the CCE driver's
`validRow × numRepeatPerLine` chunking.

### 3. UB → LLVM lowering (`lib/PTO/Transforms/VPTOLLVMEmitter.cpp`)

The `llvm.hivm` callees are resolved (from `docs/designs/a2a3-vector-builtins.md`,
probe target `dav-c220-vec`):

| UB op | `llvm.hivm` intrinsic | Intrinsic signature |
|---|---|---|
| `pto.ub.vgather` | `llvm.hivm.VGATHER.b16` / `.b32` | `void (ptr addrspace(6) dst, ptr addrspace(6) src, i64 config)` |
| `pto.ub.vgatherb` | `llvm.hivm.VGATHERB.b16` / `.b32` | `void (ptr addrspace(6) dst, ptr addrspace(6) src, i64 config)` |

Both take **2 UB pointers + 1 packed `i64` config** — the same shape as the
elementwise UB ops. Mirror `LowerUBufBinaryOpPattern` (`VPTOLLVMEmitter.cpp:4734`)
which packs repeat/stride fields into one `i64` via `maskByte`/`shl`/`OrIOp`
and passes the pointer operands directly. Register in the `patterns.add<…>`
block (`:10955`) and declare legality (`:11110`). Keep `VPTOCANN900LLVMEmitter.cpp`
in sync (dormant via the dispatcher's `return false`, but must not rot).

The CCE builtin forms `vgatherb(dst, src, offsetAddr, dstRepeatStride,
dstBlockStride, repeat)` get packed into `(dst_ptr, src_ptr, config)`. The
elementwise config bit-layout is documented in `VPTOLLVMEmitter.cpp` (e.g.
`LowerUBufShiftOpPattern:4885`); the **vgather/vgatherb config bit-layout is
not documented in CANN** ("No `// ->` packing comment"), so it was decoded
empirically from the bisheng-emitted LLVM IR. The `.b16`/`.b32` suffix selects
16-bit vs 32-bit element lanes.

### VGATHERB config bit-layout (decoded from bisheng IR)

Ground truth obtained by probing the real builtin with
`ccec --cce-aicore-arch=dav-c220-vec -xcce -std=c++17 -fenable-matrix \
-emit-llvm -c -I<pto-isa-include> probe.cpp` (no `-o`; multi-output) then
`llvm-dis` on the device `.bc` (`*-cce-hiipu64-hisilicon-cce-dav-c220-vec.bc`).

Emitted call:
```llvm
call void @llvm.hivm.VGATHERB.b32(ptr addrspace(6) %dst, ptr addrspace(6) %offset, i64 %config)
```
- **2nd pointer operand = the offset buffer** (NOT the source data); the source
  data base address is packed *into* `config[31:0]`.
- The `i64` config packs the scalar fields as:
  - `config[31:0]`  — source data address (`offsetAddr`/`srcAddr`)
  - `config[39:32]` — `dstRepeatStride`
  - `config[47:40]` — `dstBlockStride`
  - `config[55:48]` — reserved (0)
  - `config[63:56]` — `repeat`
- Verified with a probe using distinctive values
  (`srcAddr=0x12345678, dstRepeatStride=0xab, dstBlockStride=0xcd, repeat=0xef`)
  which produced `or` chains at exactly the shifts above
  (`0xAB<<32`, `0xCD<<40`, `0xEF<<56`).
- The `llvm.hivm.VGATHERB.b16/.b32.cfg.v220` 6-operand overload also exists
  (fields passed directly, no packing) but the 3-operand packed form is the one
  bisheng emits from the bottom-level `__builtin_cce_vgatherb`.

Note: `llvm.hivm.VGATHER.b16/.b32` (element gather) uses the same packed
`(dst, src/addr-ptr, i64 config)` shape; its layout will be confirmed the same
way when `pto.ub.vgather` is implemented (Slice 2). The cce-mlir repo's
4-argument `VGATHER(dst, src, addr, repeat)` lowering is a simplified,
bisheng-unverified model (drops strides, no suffix, no bisheng test) — not used.

### 4. End-to-end tests (`ptodsl/tests/e2e/test_gather.py`)

PTODSL-authored via `pto.tile.gatherb` (block form), `pto.tile.gather`
(index form), and `pto.tile.mgather` (GM-to-UB Row/Elem), a3/vpto, numpy goldens, reusing the
`ptodsl/tests/e2e/common.py` harness. Run on NPU manually; there is no CI NPU
job for these today.

### 5. Lit tests

- A3 lit pinning `pto.tgather` (index) → `pto.ub.vgather` under
  `pto-lower-to-ubuf-ops` (modeled on `test/lit/vpto/ub/tile_to_ub/elementwise/tadd_a2.pto`).
- Replace the current `tgather_index_a3.pto` (which wrongly expects the A5
  `vgather2`) with the `pto.ub.vgather` expectation.

## Phasing

1. **Slice 1 — DONE**: `pto.ub.vgatherb` (byte-offset, `vgatherb`): op def +
   `tgatherb → ub.vgatherb` tile→UB lowering (with multi-repeat stride from
   `TileShapeInfo`) + `ub.vgatherb → llvm.hivm.VGATHERB.b16/.b32` UB→LLVM
   lowering. Lit-tested at all three levels (tile→UB, UB→LLVM, full-chain).
2. **Slice 2 — DONE**: `pto.ub.vgather` (element, `vgather`): op def +
   UB→LLVM lowering (`→ llvm.hivm.VGATHER.b16/.b32`) + `TGatherOp` index-form
   tile→UB lowering via index→byte-offset decomposition (`ub.vmuls` for byte
   offsets + `ub.vgather` with `srcAddr` packed as `offsetAddr`). Lit-tested
   at all levels.
3. **Slice 3 — PENDING**: mask / compare tgather forms. These require
   compile-time index-buffer generation (expanding mask patterns like P0101 to
   explicit element indices, or compare-conditional index selection), which is
   non-trivial at the PTO IR level. The pto-isa `a2a3/TGather.hpp` handles
   these via template-level loops; the MLIR lowering would need to replicate
   that (e.g., generating a constant index buffer + using the index-form path).
4. **Stacked follow-up — DONE**: A2/A3 `mgather` GM-to-UB Row/Elem lowering,
   including scalar index handling, all four `GatherOOB` modes, per-row MTE2
   copies, scalar element access, the full A2/A3 payload set, and a fail-closed
   GM-to-L1 diagnostic.

## References

- A2/A3 gather implementations: `pto-isa:npu/a2a3/TGather.hpp` (`vgather`),
  `npu/a2a3/TGatherB.hpp` (`vgatherb`), `npu/a2a3/MGather.hpp` (GM gather), and
  `npu/a2a3/TScatter.hpp`.
- A5 vreg gather (contrast): `pto-isa:npu/a5/TGather.hpp` (`vgather2`).
- EmitC contract: `PTOToEmitC.cpp:9772` (tgather), `:9875` (tgatherb),
  `:3323` (mgather).
- UB op layer: `include/PTO/IR/VPTOUbOps.td`; tile→UB dispatch:
  `LowerPTOToUBufOps.cpp`; UB→LLVM: `VPTOLLVMEmitter.cpp`.
- e2e harness: `ptodsl/tests/e2e/common.py`.

## E2E investigation (NPU hardware)

### Sync fix
`TGatherBOp` is a single-phase PIPE_V operation. InsertSync derives its reads
of src+offsets and write of dst from the existing `OpPipeInterface` and
`MemoryEffectsOpInterface`; no dedicated sync macro model is required. The
generic path inserts the MTE2→V set_flag/wait_flag between DMA loads and the
gather.

### Correct `vgatherb` semantics
The NPU result initially looked wrong because the test used scalar-gather
expectations. CCE documents `vgatherb` as a **32B block gather**:

- The `src` operand is the UB buffer containing u32 addresses, not scalar
  per-element offsets.
- Each repeat reads 8 u32 addresses, so compact offset rows must be padded to
  a multiple of 8 entries even when the output row has a partial tail block.
- Each address must be 32B-aligned.
- The instruction copies 8 source blocks of 32B each into dst.

Therefore, for a 1×8 f32 tile with all-zero addresses, the expected result is
the first 32B block `[10,20,30,40,50,60,70,80]`, not repeated `src[0]`.
Scalar arbitrary element gather belongs on the `vgather` path, not `vgatherb`.

The LLVM IR matches the bisheng probe exactly (same intrinsic name, same
operand order, same config packing: srcAddr[31:0], dstRepeatStride[39:32],
dstBlockStride[47:40], repeat[63:56]).

## E2e NPU debugging session (devices 6,7)

### Key findings
1. **InsertSync**: the generic pipe/effects path models TGatherBOp directly.
2. **vgatherb reads compact block addresses**: repeat=255 crashes if the
   compact address buffer is too small, because 255 repeats read 255×8 u32
   addresses and then follow stale address values.
3. **1×8 all-zero output is correct**: repeat=1 copies the first 32B source
   block, which appears as sequential f32 values.
4. **Root cause**: the PTOAS e2e harness and part of the tile→UB lowering used
   scalar-offset semantics. The lowering must advance the offset pointer by
   8 address entries per repeat and use a compact offset row stride of one
   address per 32B output block, padded to 8-entry repeat boundaries.
5. **srcAddr from castptr**: changed from PtrToInt to castptr operand
   trace-back (matches pto-isa Tile metadata approach). Both produce the
   same i64 value for compile-time addresses.

### Next step
Use block-gather e2e tests: compact offset tensors with one 32B-aligned byte
address per output 32B block (padded to 8 entries per repeat), and numpy
goldens that copy whole 32B blocks.

## Final NPU status

Validated on 910B2 devices 6/7 with `ptodsl/tests/e2e/test_gather.py`:

- `tgatherb` block gather: f32 shapes `1x8`, `1x64`, `1x96`, `2x64`,
  `1x128`, `4x64`.
- `tgatherb` block gather: f16 shapes `1x16`, `1x128`, `1x192`.
- `tgather` scalar index gather: f32/f16 shapes `1x64`, `2x64`.

The final scalar `tgather` fix was to keep count-mode mask active across the
`vmuls` and `vgather` pair and pass `srcAddr` as `vgather`'s packed
`offsetAddr`, matching `pto/npu/a2a3/TGather.hpp`. Resetting the mask between
`vmuls` and `vgather` made the hardware read too many address entries and
caused UB out-of-bounds.
