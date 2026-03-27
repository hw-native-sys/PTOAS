# VPTO Spec — Merged Draft (A5)

> **Status:** DRAFT for review
> **Base:** [vpto-spec.md](https://github.com/mouliangyu/PTOAS/blob/feature-vpto-backend/docs/vpto-spec.md) (2026-03-20)
> **Additions from:** [a5_intrinsic_ir.md](../a5_intrinsic/a5_intrinsic_ir.md) v3.2 (2026-03-21)
> **Updated:** 2026-03-27

---

## Part I: Architecture Overview

### Overview

This document defines the Vector PTO (VPTO) Intermediate Representation (IR), a compiler-internal and externally facing specification designed to represent vector compute kernels within the PTO architecture. Much like NVVM provides a robust IR for GPU architectures, VPTO serves as the direct bridge between high-level programming models and the underlying hardware ISA, providing a precise, low-level representation of vector workloads explicitly designed for the Ascend 950 architecture.

#### Position in the Stack and Layer Modeled

VPTO operates as a very low-level intermediate representation within the PTO compiler stack. It is uniquely designed to accurately and comprehensively express all architectural information of the Ascend 950 hardware. It specifically models the bare-metal vector execution layer, making hardware-specific capabilities and constraints, such as exact vector lane configurations, memory space hierarchies, and hardware-specific fusion semantics, fully transparent and controllable.

#### Why External Developers Read or Author VPTO

While the majority of users will interact with the PTO architecture via higher-level frameworks, external developers may need to read or author VPTO IR directly for several key reasons:

- Custom Toolchain Development: build custom compiler frontends or domain-specific languages (DSLs) that target the Ascend 950 architecture with maximum hardware utilization.
- Performance Engineering: inspect the output of high-level compiler passes, verify fine-grained optimization behaviors, and pinpoint performance bottlenecks at the architectural level.
- Micro-Optimization: hand-author highly optimized, critical mathematical kernels using a stable, precise IR when higher-level abstractions cannot achieve the theoretical peak performance of the hardware.

#### Relationship to CCE

VPTO is designed to express the full semantic capabilities of the Compute Cube Engine (CCE), but with significant structural and pipeline advantages for compiler development.

- Bypassing the C/Clang Pipeline: while CCE heavily relies on C/C++ extensions parsed by Clang, VPTO operates entirely independently of the C language frontend. By bypassing Clang AST generation and frontend processing, utilizing VPTO significantly reduces overall compilation time and memory overhead.
- Enhanced IR Verification: because VPTO is a strongly typed, SSA-based (Static Single Assignment) compiler IR rather than a C-wrapper API, it provides a much more rigorous and detailed IR verification process. Structural inconsistencies, invalid memory access patterns, and operand type mismatches are caught immediately with precise, explicit diagnostic feedback, providing developers with much higher visibility into kernel correctness than traditional CCE error reporting.

#### Intended Audience

This document is written for compiler engineers, library writers, and advanced performance architects. We expect the reader to have a working understanding of modern compiler infrastructure, specifically MLIR, the principles of Static Single Assignment (SSA) form, and a deep understanding of the vector-processing capabilities of the Ascend 950 architecture.

### Getting Started

The Vector PTO (VPTO) IR is architected as a performance-critical layer within the compiler stack, specifically designed to exploit the **Decoupled Access-Execute** (DAE) nature of the Ascend 950 hardware.

#### Hardware Pipeline Modeling

The IR is structured to mirror the three primary hardware pipelines of the Ascend 950 architecture. Correct VPTO authoring requires managing the interaction between these asynchronous units:

**MTE2** (Memory Transfer Engine - Inbound): Responsible for moving data from Global Memory (GM) to the Unified Buffer (UB).

**Vector Core** (Computation): The primary engine for executing SIMD operations on data stored in UB.

**MTE3** (Memory Transfer Engine - Outbound): Responsible for moving processed data from UB back to GM.

#### Architecture Detail: Vector Lane (VLane)

The vector register is organized as **8 VLanes** of 32 bytes each. A VLane is the atomic unit for group reduction operations.

```
vreg (256 bytes total):
┌─────────┬─────────┬─────────┬─────┬─────────┬─────────┐
│ VLane 0 │ VLane 1 │ VLane 2 │ ... │ VLane 6 │ VLane 7 │
│   32B   │   32B   │   32B   │     │   32B   │   32B   │
└─────────┴─────────┴─────────┴─────┴─────────┴─────────┘
```

Elements per VLane by data type:

| Data Type | Elements/VLane | Total Elements/vreg |
|-----------|---------------|-------------------|
| i8/u8 | 32 | 256 |
| i16/u16/f16/bf16 | 16 | 128 |
| i32/u32/f32 | 8 | 64 |
| i64/u64 | 4 | 32 |

#### Memory and Synchronization Model

VPTO enforces a strict memory hierarchy. The Unified Buffer (UB) is the only valid operand source for vector compute instructions. Consequently, the architecture of a VPTO program is defined by the explicit management of data movement:

**Address Space Isolation**: The IR uses `!pto.ptr<element-type, space>` to distinguish between GM (`!pto.ptr<T, gm>`) and UB (`!pto.ptr<T, ub>`). The verifier ensures that vector compute operations do not access GM directly; data must first be moved into UB.

**UB Capacity**: The Unified Buffer provides 256KB of on-chip SRAM (also referred to as "vecTile").

**Data Flow**:

```
┌─────────────────────────────────────────────┐
│                 Global Memory (GM)           │
│              (Off-chip HBM/DDR)              │
└─────────────────────┬───────────────────────┘
                      │ DMA (MTE2 inbound / MTE3 outbound)
┌─────────────────────▼───────────────────────┐
│              Unified Buffer (UB)             │
│            (On-chip SRAM, 256KB)             │
└─────────────────────┬───────────────────────┘
                      │ Vector Load/Store (PIPE_V)
┌─────────────────────▼───────────────────────┐
│           Vector Register File (VRF)         │
│     vreg (256B each) + mask (256-bit each)   │
└─────────────────────────────────────────────┘
```

1. **GM → UB**: DMA transfer via MTE2 (`pto.copy_gm_to_ubuf`)
2. **UB → vreg**: Vector Load instructions (`pto.vlds`, `pto.vldx2`, etc.)
3. **vreg → vreg**: Compute instructions (`pto.vadd`, `pto.vmul`, etc.)
4. **vreg → UB**: Vector Store instructions (`pto.vsts`, `pto.vstx2`, etc.)
5. **UB → GM**: DMA transfer via MTE3 (`pto.copy_ubuf_to_gm`)

**Load/Store Access Patterns**:

For UB↔vreg data movement, besides contiguous load/store, the architecture provides rich access pattern support including strided access, pack/unpack, interleave/deinterleave, broadcast, upsample/downsample, channel split/merge, gather/scatter, and squeeze/expand operations. For detailed instruction syntax and distribution modes, refer to the [Vector Load/Store](isa/03-vector-load-store.md) group in the ISA specification.

#### Synchronization Model

The Ascend 950 architecture employs a cluster-based design with a 1:2 ratio of Cube cores to Vector cores. VPTO provides multiple levels of synchronization to manage concurrent execution across pipelines and cores:

**Inter-Core Synchronization (within a cluster):**

Synchronization between cores within the same cluster is achieved via the core sync mechanism using `pto.set_intra_core` and `pto.wait_intra_core` operations. This enables coordination between Cube and Vector cores sharing the same cluster resources.

**Vector Core Pipeline Synchronization:**

Within a single core, multiple pipelines operate asynchronously:

- **MTE2 (PIPE_MTE2)**: DMA copy-in from GM to UB
- **MTE3 (PIPE_MTE3)**: DMA copy-out from UB to GM
- **Vector Compute (PIPE_V)**: Vector ALU operations
- **Scalar (PIPE_S)**: Scalar unit running the kernel program

Pipeline synchronization can be achieved through two mechanisms:

1. **Flag/Event mechanism**: `pto.set_flag` and `pto.wait_flag` operations resolve Read-After-Write (RAW) and Write-After-Read (WAR) hazards between pipelines.

2. **Buffer-ID mechanism**: `pto.get_buf` and `pto.rls_buf` provide finer-grained synchronization through buffer acquisition and release semantics for producer-consumer coordination.

**Intra-Pipeline Memory Barriers (within `__VEC_SCOPE__`):**

Within the vector execution scope, the hardware does not track UB address aliasing between reg↔UB accesses. When UB addresses overlap or alias between vector load/store operations, explicit memory barriers are required:

```c
pto.mem_bar "VV_ALL"      // All prior vector ops complete before subsequent
pto.mem_bar "VST_VLD"     // All prior vector stores visible before subsequent loads
pto.mem_bar "VLD_VST"     // All prior vector loads complete before subsequent stores
```

Without proper barriers, loads may see stale data or stores may be reordered incorrectly.

#### Execution Scopes (__VEC_SCOPE__)

`__VEC_SCOPE__` is the IR-level representation of a Vector Function (VF) launch. In the PTO architecture, it defines the hardware interface between the Scalar Unit and the Vector Thread.

It is not a dedicated `pto` op. In VPTO IR, this scope is modeled as a specialized `scf.for` loop annotated with `llvm.loop.aivector_scope`. This gives the compiler a natural structural boundary for identifying the code block that must be lowered into a discrete VF hardware instruction sequence.

**Scalar-Vector Interface:**

The execution model follows non-blocking fork semantics:

- Scalar invocation: the scalar processor invokes a vector thread by calling a VF. Once the launch command is issued, the scalar unit does not stall and continues executing subsequent instructions in the pipeline.
- Vector execution: after invocation, the vector thread independently fetches and executes the instructions defined within the VF scope.
- Parallelism: this decoupled execution allows the scalar and vector units to run in parallel, so the scalar unit can prepare addresses or manage control flow while the vector unit performs heavy SIMD computation.

**Launch Mechanism And Constraints:**

- Parameter buffering: all arguments required by the VF must be staged in hardware-specific buffers.
- Launch overhead: launching a VF incurs a latency of a few cycles. Very small VFs should account for this overhead because launch cost can rival useful computation time.

**MLIR Representation:**

```mlir
scf.for %dummy = %c0 to %c1 step %c1 {
  %mask = pto.pset_b32 "PAT_ALL" : !pto.mask
  %v = pto.vlds %ub[%lane] : !pto.ptr<f32, ub> -> !pto.vreg<64xf32>
  %abs = pto.vabs %v, %mask : !pto.vreg<64xf32>, !pto.mask -> !pto.vreg<64xf32>
  pto.vsts %abs, %ub_out[%lane], %mask : !pto.vreg<64xf32>, !pto.ptr<f32, ub>, !pto.mask
} {llvm.loop.aivector_scope}
```

### Example: Abs

```mlir
pto.set_loop2_stride_outtoub %c4096_i64, %c4096_i64 : i64, i64
pto.set_loop1_stride_outtoub %c4096_i64, %c4096_i64 : i64, i64
pto.set_loop_size_outtoub %c1_i64, %c1_i64 : i64, i64
pto.copy_gm_to_ubuf %7, %2, %3, %3, %c0_i64, %c32_i64, %4, %c0_i64, %c0_i64,
    %false, %c0_i64, %c128_i64, %c128_i64
    : !pto.ptr<f32, gm>, !pto.ptr<f32, ub>, i64, i64, i64, i64, i64, i64, i64, i1, i64, i64, i64

pto.set_flag["PIPE_MTE2", "PIPE_V", "EVENT_ID0"]
pto.wait_flag["PIPE_MTE2", "PIPE_V", "EVENT_ID0"]

scf.for %dummy = %c0 to %c1 step %c1 {
  scf.for %lane = %c0 to %9 step %c64 {
    %mask = pto.pset_b32 "PAT_ALL" : !pto.mask
    %v = pto.vlds %2[%lane] : !pto.ptr<f32, ub> -> !pto.vreg<64xf32>
    %abs = pto.vabs %v, %mask : !pto.vreg<64xf32>, !pto.mask -> !pto.vreg<64xf32>
    pto.vsts %abs, %8[%lane], %mask : !pto.vreg<64xf32>, !pto.ptr<f32, ub>, !pto.mask
  }
} {llvm.loop.aivector_scope}

pto.set_flag["PIPE_V", "PIPE_MTE3", "EVENT_ID0"]
pto.wait_flag["PIPE_V", "PIPE_MTE3", "EVENT_ID0"]
pto.set_loop_size_ubtoout %c1_i64, %c1_i64 : i64, i64
pto.set_loop1_stride_ubtoout %c4096_i64, %c4096_i64 : i64, i64
pto.set_loop2_stride_ubtoout %c4096_i64, %c4096_i64 : i64, i64
pto.copy_ubuf_to_gm %8, %14, %3, %3, %c0_i64, %c32_i64, %4, %c0_i64, %c128_i64, %c128_i64
    : !pto.ptr<f32, ub>, !pto.ptr<f32, gm>, i64, i64, i64, i64, i64, i64, i64, i64
```

### Scope

This document is the interface specification centered on the `mlir::pto` dialect and the shared MLIR surface used alongside it in VPTO programs.

It only describes:

- operation names
- operand and result lists
- operand and result types
- important attributes
- C-style semantics for each operation

It does not describe lowering strategy.

VPTO source programs are not restricted to `pto` operations alone. In practice they also use shared MLIR dialect ops, most notably the full scalar operation surface of `arith` together with structured control-flow ops from `scf`, to express scalar constants, scalar arithmetic, type conversion, comparisons, and structured control flow around PTO vector or tile regions. These shared-dialect ops are part of the supported VPTO source surface and should be regarded as part of PTO-ISA alongside `pto` dialect operations.

### Shared MLIR Dialects

- `arith`: the full scalar `arith` surface is supported in VPTO programs, covering scalar integer, floating-point, boolean, and `index` operations. In current samples the most common uses are still constants, offset/bounds arithmetic, casts, compares, and selects.
- `scf`: structured control flow used to model counted loops, conditional regions, loop-carried state, and break-like control around PTO compute and data-movement ops.
- Shared dialect ops remain in standard MLIR form so that PTO analyses and backend passes can reason about control flow and scalar state without re-encoding them as PTO-specific instructions.

### Core Types

### Element Types
`vreg<T>`: `!pto.vreg<NxT>` Fixed-width VPTO vector type with total width exactly 256 bytes (2048 bits). `N` is the lane count, `T` is the element type, and `N * bitwidth(T) = 2048`.

| Type | Bits | Description |
|------|------|-------------|
| `i8` / `s8` / `u8` | 8 | Signless/signed/unsigned 8-bit integer |
| `i16` / `s16` / `u16` | 16 | Signless/signed/unsigned 16-bit integer |
| `i32` / `s32` / `u32` | 32 | Signless/signed/unsigned 32-bit integer |
| `i64` / `s64` / `u64` | 64 | Signless/signed/unsigned 64-bit integer |
| `f16` | 16 | IEEE 754 half precision |
| `bf16` | 16 | Brain floating point |
| `f32` | 32 | IEEE 754 single precision |
| `f8e4m3` | 8 | FP8 (4-bit exponent, 3-bit mantissa) |
| `f8e5m2` | 8 | FP8 (5-bit exponent, 2-bit mantissa) |

### Address Space Conventions

VPTO memory operands use `!pto.ptr<element-type, space>`. This specification models the following memory-space attributes:

| Space | Interpretation |
|-------|----------------|
| `gm` | Global Memory (GM), off-chip HBM/DDR storage |
| `ub` | Unified Buffer (UB), on-chip vector buffer |

Typical pointer construction and pointer arithmetic follow the same `!pto.ptr<..., space>` form:

```mlir
%0 = pto.castptr %c0 : i64 -> !pto.ptr<f32, ub>
%1 = pto.addptr %0, %c1024 : !pto.ptr<f32, ub> -> !pto.ptr<f32, ub>
```

### `!pto.ptr<T, space>`

`!pto.ptr<T, space>` is the typed pointer form used for explicit memory operands in VPTO.

- `T` is the element type associated with the pointed-to storage.
- `space` is the memory domain, typically `gm` or `ub` in this specification.
- A `pto.ptr` value carries an address plus its element-type / memory-space interpretation, but it does not carry tensor shape or stride metadata by itself.
- Tensor semantics are introduced separately through view-building operations such as `pto.make_tensor_view`.
- Pointer arithmetic is element-based rather than byte-based.

Typical examples:

- `!pto.ptr<f32, gm>`
- `!pto.ptr<f32, ub>`
- `!pto.ptr<bf16, gm>`

### Pointer Operations

#### `pto.castptr`

- **syntax:** `%result = pto.castptr %addr : i64 -> !pto.ptr<T, space>`
- **semantics:** Reinterpret a scalar address value as a typed PTO pointer in the target memory space.

```c
result = (ptr<T, space>)addr;
```

`pto.castptr` is a pointer-construction operation. It does not perform data movement and does not by itself imply any load/store side effect.

#### `pto.addptr`

- **syntax:** `%result = pto.addptr %ptr, %offset : !pto.ptr<T, space> -> !pto.ptr<T, space>`
- **semantics:** Compute a new pointer by advancing the base pointer by an element offset.

```c
result = ptr + offset;  // offset counted in elements, not bytes
```

`pto.addptr` preserves both the element type `T` and the memory-space tag `space`.

#### Pointer-Based Vector Access Example

The following lowered-style fragment shows how typed PTO pointers flow through pointer construction, pointer arithmetic, structured control flow, and PTO memory ops:

```mlir
%0 = pto.castptr %c0 : i64 -> !pto.ptr<f32, ub>
%1 = pto.addptr %0, %c1024 : !pto.ptr<f32, ub> -> !pto.ptr<f32, ub>
scf.for %arg2 = %c0 to %c1 step %c1 {
  %16 = scf.for %arg3 = %c0 to %11 step %c64 iter_args(%arg4 = %12) -> (i32) {
    %mask, %scalar_out = pto.plt_b32 %arg4 : i32 -> !pto.mask, i32
    %17 = pto.vlds %1[%arg3] : !pto.ptr<f32, ub> -> !pto.vreg<64xf32>
    %18 = pto.vabs %17, %mask : !pto.vreg<64xf32>, !pto.mask -> !pto.vreg<64xf32>
    pto.vsts %18, %10[%arg3], %mask : !pto.vreg<64xf32>, !pto.ptr<f32, ub>, !pto.mask
    scf.yield %scalar_out : i32
  }
} {llvm.loop.aivector_scope}
```

In this pattern, `pto.castptr` materializes a typed UB pointer, `pto.addptr` shifts the base by 1024 `f32` elements, and the subsequent `[%arg3]` indexing on `pto.vlds` / `pto.vsts` applies an additional element offset relative to that base.

### Special Types

#### `!pto.mask`

`!pto.mask` models an A5 predicate register (256-bit), not an integer vector.

**Mask Granularity:**

The mask is 256 bits in length, where each bit controls 1 byte of data. This means mask granularity varies by element type:

| Element Type | Bits/Element | Mask Bits per Element |
|--------------|--------------|----------------------|
| `f32`/`i32` | 32 | 4 bits |
| `f16`/`bf16`/`i16` | 16 | 2 bits |
| `f8`/`i8` | 8 | 1 bit |

**Predication Behavior (Zero-Merge):**

The native hardware predication mode is **ZEROING** — inactive lanes produce zero:

```c
dst[i] = mask[i] ? op(src0[i], src1[i]) : 0    // ZEROING mode
```

```mlir
// Predicated add: inactive lanes produce zero
%mask = pto.pset_b32 "PAT_VL32" : !pto.mask   // first 32 lanes active
%result = pto.vcmp %a, %b, %mask, "lt" : !pto.vreg<64xf32>, !pto.vreg<64xf32>, !pto.mask -> !pto.mask
```

```mlir
// Compare and select: generate mask from comparison, use for conditional select
%mask = pto.vcmp %lhs, %rhs, %seed, "lt" : !pto.vreg<64xf32>, !pto.vreg<64xf32>, !pto.mask -> !pto.mask
%out = pto.vsel %x, %y, %mask : !pto.vreg<64xf32>, !pto.vreg<64xf32>, !pto.mask -> !pto.vreg<64xf32>
```

#### `!pto.align`

`!pto.align` models the A5 vector-align carrier state. It is not payload data.

```mlir
%align = pto.vldas %ub : !pto.ptr<f32, ub> -> !pto.align
%vec, %align_out, %base_out = pto.vldus %ub, %align : !pto.ptr<f32, ub>, !pto.align -> !pto.vreg<64xf32>, !pto.align, !pto.ptr<f32, ub>
```

---

## Part II: Notation Convention

This section defines the MLIR syntax patterns and C-style semantic notation used throughout the ISA reference (Part III).

### MLIR Op Syntax Patterns

All VPTO operations follow standard MLIR syntax. The common patterns are:

**Unary (one vector in, one vector out):**

```mlir
%result = pto.<op> %input : !pto.vreg<NxT> -> !pto.vreg<NxT>
```

**Binary (two vectors in, one vector out):**

```mlir
%result = pto.<op> %lhs, %rhs : !pto.vreg<NxT>, !pto.vreg<NxT> -> !pto.vreg<NxT>
```

**Vec-Scalar (one vector + one scalar in, one vector out):**

```mlir
%result = pto.<op> %input, %scalar : !pto.vreg<NxT>, T -> !pto.vreg<NxT>
```

**Load (memory to register):**

```mlir
%result = pto.vlds %source[%offset] {dist = "DIST"} : !pto.ptr<T, ub> -> !pto.vreg<NxT>
```

**Store (register to memory):**

```mlir
pto.vsts %value, %destination[%offset] {dist = "DIST"} : !pto.vreg<NxT>, !pto.ptr<T, ub>
```

**Dual Load (one load, two results — deinterleave):**

```mlir
%low, %high = pto.vldx2 %source[%offset], "DIST" : !pto.ptr<T, ub>, index -> !pto.vreg<NxT>, !pto.vreg<NxT>
```

**Dual Store (two inputs, one interleaved store):**

```mlir
pto.vstx2 %low, %high, %dest[%offset], "DIST", %mask : !pto.vreg<NxT>, !pto.vreg<NxT>, !pto.ptr<T, ub>, index, !pto.mask
```

**Compare (two vectors + seed mask in, mask out):**

```mlir
%mask = pto.vcmp %src0, %src1, %seed, "CMP_MODE" : !pto.vreg<NxT>, !pto.vreg<NxT>, !pto.mask -> !pto.mask
```

**Conversion (one vector in, different-typed vector out):**

```mlir
%result = pto.vcvt %input {round_mode = "ROUND_R", sat = "RS_ENABLE", part = "PART_EVEN"} : !pto.vreg<NxT0> -> !pto.vreg<MxT1>
```

**Predicate construction:**

```mlir
%mask = pto.pset_b32 "PAT_ALL" : !pto.mask
%tail = pto.pge_b32 "PAT_VL16" : !pto.mask
```

**Sync operations:**

```mlir
pto.set_flag["PIPE_MTE2", "PIPE_V", "EVENT_ID0"]
pto.wait_flag["PIPE_MTE2", "PIPE_V", "EVENT_ID0"]
```

**Pointer construction and arithmetic:**

```mlir
%ptr = pto.castptr %addr : i64 -> !pto.ptr<T, SPACE>
%ptr2 = pto.addptr %ptr, %offset : !pto.ptr<T, SPACE> -> !pto.ptr<T, SPACE>
```

### Shared Dialect Syntax Patterns

VPTO programs may interleave PTO ops with standard MLIR `arith` and `scf` ops.
The examples below emphasize common index-heavy patterns, but `arith` support is not limited to index arithmetic.

**Scalar / index constant:**

```mlir
%c0 = arith.constant 0 : index
%zero = arith.constant 0.0 : f32
```

**Scalar arithmetic (integer / float / boolean-style bitwise):**

```mlir
%sum_i = arith.addi %lhs_i, %rhs_i : i32
%sum_f = arith.addf %lhs_f, %rhs_f : f32
%bits = arith.andi %flags0, %flags1 : i32
```

**Scalar compare and select:**

```mlir
%cond = arith.cmpi eq, %lhs, %rhs : index
%bound = arith.select %cond, %a, %b : index
```

**Counted loop with loop-carried values:**

```mlir
%result = scf.for %iv = %lb to %ub step %step
    iter_args(%acc = %init) -> (index) {
  %next = arith.addi %acc, %iv : index
  scf.yield %next : index
}
```

**Structured conditional region:**

```mlir
%selected = scf.if %cond -> (index) {
  scf.yield %then_value : index
} else {
  scf.yield %else_value : index
}
```

**Structured while loop:**

```mlir
%state:2 = scf.while (%iv = %c0, %alive = %true) : (index, i1) -> (index, i1) {
  %keep_going = arith.cmpi slt, %iv, %limit : index
  scf.condition(%keep_going) %iv, %alive : index, i1
} do {
^bb0(%iv_in: index, %alive_in: i1):
  %iv_next = arith.addi %iv_in, %c1 : index
  scf.yield %iv_next, %alive_in : index, i1
}
```

### C-Style Semantics Convention

For each ISA operation in Part III, semantics are expressed as C code. The convention:

```c
// Vector register contents as arrays:
T dst[N];       // destination
T src0[N];      // first source
T src1[N];      // second source (binary ops)
T scalar;       // scalar operand (vec-scalar ops)
int mask[N];    // per-lane predicate (0 or 1)

// N = lane count determined by type:
//   N = 256 for i8/u8
//   N = 128 for i16/u16/f16/bf16
//   N = 64  for i32/u32/f32
//   N = 32  for i64/u64
```

**Example — pto.vadd semantics:**

```c
for (int i = 0; i < N; i++)
    dst[i] = src0[i] + src1[i];
```

**Example — pto.vcgadd (group reduction per VLane) semantics:**

```c
int K = N / 8;  // elements per VLane
for (int g = 0; g < 8; g++) {
    T sum = 0;
    for (int i = 0; i < K; i++)
        sum += src[g*K + i];
    dst[g*K] = sum;
    for (int i = 1; i < K; i++)
        dst[g*K + i] = 0;
}
```

### Template Placeholder Conventions

| Placeholder | Meaning |
|-------------|---------|
| `"SRC_PIPE"`, `"DST_PIPE"` | Pipeline identifiers: `"PIPE_MTE2"`, `"PIPE_V"`, `"PIPE_MTE3"` |
| `"EVENT_ID"` | Event identifier: `"EVENT_ID0"` etc. |
| `"DIST"` | Distribution mode string (see the relevant load/store ISA group in Part III) |
| `"CMP_MODE"` | Compare predicate: `eq \| ne \| lt \| le \| gt \| ge` |
| `"ROUND_MODE"` | Rounding mode: `ROUND_R \| ROUND_A \| ROUND_F \| ROUND_C \| ROUND_Z` |
| `"SAT_MODE"` | Saturation: `RS_ENABLE \| RS_DISABLE` |
| `"PART_MODE"` | Half selector: `PART_EVEN \| PART_ODD` |
| `"PAT_*"` | Predicate pattern literal |
| `T` | Element type (f32, f16, bf16, i32, i16, i8, etc.) |
| `N` | Lane count (`N * bitwidth(T) = 2048`) |

---

## Part III: ISA Instruction Reference
# Part III: ISA Instruction Reference — Summary

This section provides a categorized overview of all VPTO instructions plus the shared MLIR `arith` and `scf` ops that may appear in VPTO programs. Detailed documentation for each group is available in the linked files.

---

## Instruction Groups

| # | Group | Description | Count | Details |
|---|-------|-------------|-------|---------|
| 1 | [Pipeline Sync](isa/01-pipeline-sync.md) | Intra-core pipeline synchronization | 5 | `pto.set_flag`, `pto.wait_flag`, `pto.pipe_barrier`, `pto.get_buf`, `pto.rls_buf` |
| 2 | [DMA Copy Programming](isa/02-dma-copy.md) | DMA configuration and transfer between GM↔UB | 9 | `pto.set_loop*_stride_*`, `pto.set_loop_size_*`, `pto.copy_gm_to_ubuf`, `pto.copy_ubuf_to_ubuf`, `pto.copy_ubuf_to_gm` |
| 3 | [Vector Load/Store](isa/03-vector-load-store.md) | UB↔vreg data movement with various access patterns | ~20 | `pto.vlds`, `pto.vldx2`, `pto.vgather2`, `pto.vsts`, `pto.vstx2`, `pto.vscatter`, etc. |
| 4 | [Predicate Load/Store](isa/04-predicate-load-store.md) | UB↔mask register movement | 7 | `pto.plds`, `pto.pld`, `pto.pldi`, `pto.psts`, `pto.pst`, `pto.psti`, `pto.pstu` |
| 5 | [Materialization & Predicate Ops](isa/05-materialization-predicate.md) | Scalar broadcast, predicate generation and manipulation | ~17 | `pto.vbr`, `pto.vdup`, `pto.pset_b*`, `pto.pge_b*`, `pto.plt_b*`, `pto.ppack`, `pto.punpack`, `pto.pnot`, `pto.psel`, etc. |
| 6 | [Unary Vector Ops](isa/06-unary-vector-ops.md) | Single-input element-wise operations | 9 | `pto.vabs`, `pto.vexp`, `pto.vln`, `pto.vsqrt`, `pto.vrec`, `pto.vrelu`, `pto.vnot`, `pto.vbcnt`, `pto.vcls` |
| 7 | [Binary Vector Ops](isa/07-binary-vector-ops.md) | Two-input element-wise operations | 13 | `pto.vadd`, `pto.vsub`, `pto.vmul`, `pto.vdiv`, `pto.vmax`, `pto.vmin`, `pto.vand`, `pto.vor`, `pto.vxor`, `pto.vshl`, `pto.vshr`, `pto.vaddc`, `pto.vsubc` |
| 8 | [Vec-Scalar Ops](isa/08-vec-scalar-ops.md) | Vector-scalar operations | 8 | `pto.vadds`, `pto.vmuls`, `pto.vmaxs`, `pto.vmins`, `pto.vlrelu`, `pto.vshls`, `pto.vshrs`, `pto.vaddcs`, `pto.vsubcs` |
| 9 | [Conversion Ops](isa/09-conversion-ops.md) | Type conversion with rounding/saturation control | 2 | `pto.vcvt`, `pto.vtrc` |
| 10 | [Reduction Ops](isa/10-reduction-ops.md) | Vector reductions | 3 | `pto.vcadd`, `pto.vcmax`, `pto.vcmin` |
| 11 | [Compare & Select](isa/11-compare-select.md) | Comparison and conditional selection | 5 | `pto.vcmp`, `pto.vcmps`, `pto.vsel`, `pto.vselr`, `pto.vselrv2` |
| 12 | [Data Rearrangement](isa/12-data-rearrangement.md) | In-register data movement and permutation | 4 | `pto.vintlv`, `pto.vdintlv`, `pto.vintlvv2`, `pto.vdintlvv2` |
| 13 | [DSA/SFU Ops](isa/13-dsa-sfu-ops.md) | Specialized ops, index generation, and sorting helpers | 5 | `pto.vmull`, `pto.vmula`, `pto.vci`, `pto.vbitsort`, `pto.vmrgsort4` |
| 14 | [Arith (Shared MLIR Dialect)](isa/14-shared-arith.md) | Full scalar `arith` surface used around PTO ops; the companion page lists categories and representative examples | all scalar ops | `arith.constant`, `arith.addi`, `arith.addf`, `arith.cmpi`, `arith.cmpf`, `arith.select`, `arith.index_cast`, `arith.extsi`, `arith.trunci`, `arith.andi`, `arith.shli`, etc. |
| 15 | [SCF (Shared MLIR Dialect)](isa/15-shared-scf.md) | Structured loops, branches, and loop-carried state around PTO regions | 5 | `scf.for`, `scf.if`, `scf.while`, `scf.condition`, `scf.yield` |

---

## Quick Reference by Category

### Memory Operations

| Operation | Group | Description |
|-----------|-------|-------------|
| GM→UB DMA | 2 | `pto.copy_gm_to_ubuf` |
| UB→GM DMA | 2 | `pto.copy_ubuf_to_gm` |
| UB→UB Copy | 2 | `pto.copy_ubuf_to_ubuf` |
| Contiguous Load | 3 | `pto.vlds` with `NORM` dist |
| Broadcast Load | 3 | `pto.vlds` with `BRC_*` dist |
| Gather | 3 | `pto.vgather2`, `pto.vgatherb` |
| Contiguous Store | 3 | `pto.vsts` with `NORM_*` dist |
| Scatter | 3 | `pto.vscatter` |

### Compute Operations

| Operation | Group | Description |
|-----------|-------|-------------|
| Element-wise Arithmetic | 6, 7 | `pto.vadd`, `pto.vmul`, `pto.vabs`, etc. |
| Scalar Operations | 8 | `pto.vadds`, `pto.vmuls`, etc. |
| Transcendental | 6 | `pto.vexp`, `pto.vln`, `pto.vsqrt`, etc. |
| Reduction | 10 | `pto.vcadd`, `pto.vcmax`, `pto.vcmin` |
| Comparison | 11 | `pto.vcmp`, `pto.vcmps` |
| Selection | 11 | `pto.vsel`, `pto.vselr` |

### Type & Data Manipulation

| Operation | Group | Description |
|-----------|-------|-------------|
| Type Conversion | 9 | `pto.vcvt` |
| Interleave/Deinterleave | 12 | `pto.vintlv`, `pto.vdintlv` |
| Interleave/Deinterleave | 12 | `pto.vintlv`, `pto.vdintlv`, `pto.vintlvv2`, `pto.vdintlvv2` |

### Synchronization

| Operation | Group | Description |
|-----------|-------|-------------|
| Intra-core Sync | 1 | `pto.set_flag`, `pto.wait_flag` |
| Pipeline Buffer Sync | 1 | `pto.get_buf`, `pto.rls_buf` |

### Scalar & Control Operations

Group 14 covers the full scalar `arith` surface. The rows below list common VPTO patterns rather than an exhaustive partition of `arith` ops.

| Operation | Group | Description |
|-----------|-------|-------------|
| Scalar Constants | 14 | `arith.constant` |
| Scalar Integer / Index Arithmetic | 14 | `arith.addi`, `arith.subi`, `arith.muli`, `arith.divsi`, `arith.remui`, `arith.ceildivsi`, etc. |
| Scalar Floating-Point Arithmetic | 14 | `arith.addf`, `arith.subf`, `arith.mulf`, `arith.divf`, `arith.maximumf`, etc. |
| Scalar Compare & Select | 14 | `arith.cmpi`, `arith.cmpf`, `arith.select` |
| Scalar Casts / Width Changes | 14 | `arith.index_cast`, `arith.index_castui`, `arith.extsi`, `arith.extui`, `arith.trunci`, `arith.sitofp`, etc. |
| Scalar Bitwise / Shift Ops | 14 | `arith.andi`, `arith.ori`, `arith.xori`, `arith.shli`, `arith.shrsi`, `arith.shrui`, etc. |
| Counted Loops | 15 | `scf.for` |
| Conditional Regions | 15 | `scf.if`, `scf.yield` |
| Break-like Structured Loops | 15 | `scf.while`, `scf.condition`, `scf.yield` |

---

## Supported Data Types

| Type | Bits | vreg Lanes | Description |
|------|------|-----------|-------------|
| `i8` / `u8` | 8 | 256 | Signed/unsigned 8-bit integer |
| `i16` / `u16` | 16 | 128 | Signed/unsigned 16-bit integer |
| `f16` | 16 | 128 | IEEE 754 half precision |
| `bf16` | 16 | 128 | Brain floating point |
| `i32` / `u32` | 32 | 64 | Signed/unsigned 32-bit integer |
| `f32` | 32 | 64 | IEEE 754 single precision |
| `i64` / `u64` | 64 | 32 | Signed/unsigned 64-bit integer |

---

## Common Patterns

### Softmax (Numerically Stable)

```mlir
// 1. Find max
%max_vec = pto.vcmax %logits, %mask : !pto.vreg<64xf32>, !pto.mask -> !pto.vreg<64xf32>
pto.vsts %max_vec, %ub_tmp[%c0], %mask : !pto.vreg<64xf32>, !pto.ptr<f32, ub>, !pto.mask
%max_bc = pto.vlds %ub_tmp[%c0] {dist = "BRC_B32"} : !pto.ptr<f32, ub> -> !pto.vreg<64xf32>

// 2. exp(x - max) using fused op
%exp = pto.vexpdiff %logits, %max_bc : !pto.vreg<64xf32>, !pto.vreg<64xf32> -> !pto.vreg<64xf32>

// 3. Sum
%sum = pto.vcadd %exp, %mask : !pto.vreg<64xf32>, !pto.mask -> !pto.vreg<64xf32>
pto.vsts %sum, %ub_tmp[%c0], %mask : !pto.vreg<64xf32>, !pto.ptr<f32, ub>, !pto.mask
%sum_bc = pto.vlds %ub_tmp[%c0] {dist = "BRC_B32"} : !pto.ptr<f32, ub> -> !pto.vreg<64xf32>

// 4. Divide
%softmax = pto.vdiv %exp, %sum_bc, %mask : !pto.vreg<64xf32>, !pto.vreg<64xf32>, !pto.mask -> !pto.vreg<64xf32>
```

### ReLU Variants

```mlir
// Standard ReLU
%relu = pto.vrelu %input, %mask : !pto.vreg<64xf32>, !pto.mask -> !pto.vreg<64xf32>

// Leaky ReLU (scalar alpha)
%lrelu = pto.vlrelu %input, %alpha, %mask : !pto.vreg<64xf32>, f32, !pto.mask -> !pto.vreg<64xf32>

// Parametric ReLU (per-element alpha)
%prelu = pto.vprelu %input, %alpha_vec : !pto.vreg<64xf32>, !pto.vreg<64xf32> -> !pto.vreg<64xf32>

// Fused add + ReLU
%fused = pto.vaddrelu %a, %b : !pto.vreg<64xf32>, !pto.vreg<64xf32> -> !pto.vreg<64xf32>
```

### Data Layout Conversion

```mlir
// AoS → SoA (deinterleave)
%x, %y = pto.vldx2 %ub_xy[%offset], "DINTLV_B32" : !pto.ptr<f32, ub>, index -> !pto.vreg<64xf32>, !pto.vreg<64xf32>

// SoA → AoS (interleave)
pto.vstx2 %x, %y, %ub_xy[%offset], "INTLV_B32", %all_mask : !pto.vreg<64xf32>, !pto.vreg<64xf32>, !pto.ptr<f32, ub>, index, !pto.mask
```

---

*For detailed semantics, C-style pseudocode, and CCE mappings, see the individual group documentation files.*

---

## Appendix A: Change Summary

### Part I Changes

| # | Section | What Changed | Source |
|---|---------|-------------|--------|
| 1 | Overview, Position, Audience, CCE | Kept verbatim | vpto-spec.md |
| 2 | VLane concept (32B) | **ADDED** | a5_intrinsic_ir.md |
| 3 | Register/type system | Follows vpto-spec.md | vpto-spec.md |
| 4 | UB size (256KB) | **ADDED** | a5_intrinsic_ir.md |
| 5 | Data flow diagram | **ADDED** | a5_intrinsic_ir.md + vpto naming |
| 6 | Load/store pattern table | **EXPANDED** | a5_intrinsic_ir.md |
| 7 | Sync: set_flag/wait_flag | Kept | vpto-spec.md |
| 8 | Sync: get_buf/rls_buf | **CLARIFIED** as inter-pipe sync | vpto-spec.md |
| 9 | Sync: mem_bar | **ADDED** for intra-VEC_SCOPE | a5_intrinsic_ir.md |
| 10 | Predication | **ADDED** ZEROING only | a5_intrinsic_ir.md |
| 11 | __VEC_SCOPE__ | Kept verbatim | vpto-spec.md |
| 12 | Element types | **EXPANDED** with FP8/FP4 | a5_intrinsic_ir.md |
| 13 | Load dist tokens | **EXPANDED** (BRC, US, DS, SPLT, UNPK) | a5_intrinsic_ir.md |
| 14 | Store dist tokens | **EXPANDED** (NORM_B*, PK_B*, MRG*) | a5_intrinsic_ir.md |
| 15 | mem_bar tokens | **ADDED** | a5_intrinsic_ir.md |

### Part II Changes

| # | What Changed | Source |
|---|-------------|--------|
| 1 | MLIR syntax patterns | Organized from vpto-spec.md |
| 2 | C-style semantics convention | **NEW** — replaces math notation |
| 3 | VLane-aware reduction example | **NEW** |
| 4 | Template placeholders | Consolidated from vpto-spec.md |

### Part 3A Changes (Sections 1–6)

| # | Section | What Changed | Source |
|---|---------|-------------|--------|
| 1 | Sec 1: Sync | **ADDED** `pto.mem_bar` with C semantics | a5_intrinsic_ir.md |
| 2 | Sec 2-3: Copy | Kept from vpto-spec.md | vpto-spec.md |
| 3 | Sec 4: Loads — vgatherb, vgather2_bc | Kept from vpto-spec.md | vpto-spec.md |
| 4 | Sec 4: Loads — BRC/US/DS/SPLT dist modes | **ADDED** with C semantics | a5_intrinsic_ir.md |
| 5 | Sec 5: vbr | Kept from vpto-spec.md | vpto-spec.md |
| 6 | Sec 5: vdupi | **ADDED** | a5_intrinsic_ir.md |
| 7 | Sec 5: pand, por, pxor, ppack/punpack | **ADDED** | a5_intrinsic_ir.md |
| 8 | Sec 5: pintlv/pdintlv-style predicate movement | **ADDED** | a5_intrinsic_ir.md |
| 9 | Sec 6: vneg, vrsqrt | **ADDED** | a5_intrinsic_ir.md |
| 10 | Sec 6: vcgadd, vcgmax, vcgmin | **ADDED** per-VLane reductions | a5_intrinsic_ir.md |
| 11 | Sec 6: vcpadd | **ADDED** prefix sum | a5_intrinsic_ir.md |
| 12 | Sec 6: vmov | **ADDED** | a5_intrinsic_ir.md |

### Part 3B Changes (Sections 7–10)

| # | Section | What Changed | Source |
|---|---------|-------------|--------|
| 1 | Sec 7: Binary ops | No changes — full 1:1 match | Both |
| 2 | Sec 8: vsubs, vands, vors, vxors | **ADDED** | a5_intrinsic_ir.md |
| 3 | Sec 9: vselrv2 | **REMOVED** (not A5) | — |
| 4 | Sec 9: vprelu | **ADDED** parametric ReLU | a5_intrinsic_ir.md |
| 5 | Sec 10: vintlvv2, vdintlvv2, pdintlv_b8, pintlv_b16 | **REMOVED** (not A5) | — |
| 6 | Sec 10: vslide, vshift, vsqz, vusqz | **ADDED** data movement | a5_intrinsic_ir.md |
| 7 | Sec 10: vperm (was vgather reg) | **ADDED** in-register permute | a5_intrinsic_ir.md |
| 8 | Sec 10: vtranspose | **ADDED** | a5_intrinsic_ir.md |
| 9 | Sec 10: vpack, vsunpack, vzunpack | **ADDED** pack/unpack | a5_intrinsic_ir.md |

### Part 3C Changes (Sections 11–14)

| # | Section | What Changed | Source |
|---|---------|-------------|--------|
| 1 | Sec 11: vcvt | **EXPANDED** — full A5 conversion pairs + width-changing pattern | a5_intrinsic_ir.md |
| 2 | Sec 11: vtrc, vci, vbitsort | Kept from vpto-spec.md | vpto-spec.md |
| 3 | Sec 12: vmull | C semantics + A5 type info added | Both |
| 4 | Sec 12: vmula | Kept from vpto-spec.md | vpto-spec.md |
| 5 | Sec 12: vaddrelu, vsubrelu | **ADDED** fused add/sub+ReLU | a5_intrinsic_ir.md |
| 6 | Sec 12: vaxpy | **ADDED** AXPY | a5_intrinsic_ir.md |
| 7 | Sec 12: vaddreluconv, vmulconv | **ADDED** fused compute+convert | a5_intrinsic_ir.md |
| 8 | Sec 13: vsts dist modes | **EXPANDED** — PK_B*, MRG* C semantics | a5_intrinsic_ir.md |
| 9 | Sec 13-14: All store ops | C semantics added where missing | Both |

## Appendix B: Discussion Points

### Part I

1. **mem_bar as pto op:** Should `pto.mem_bar` be a formal pto dialect op, or is there an existing mechanism?
2. **UB size parameterization:** Is 256KB always fixed, or should spec allow for architecture variants?
3. **Dist token expansion:** The added BRC/US/DS/SPLT/MRG tokens need verifier implementation. Are all confirmed for A5?
4. **MERGING predication:** Intentionally omitted (SW-emulated, perf overhead). Revisit if needed later.

### Part II

1. **Predication in C semantics:** Should every op's C code explicitly show the `if (mask[i])` guard, or assume all-active and note predication separately?
2. **VLane terminology:** Using "VLane" instead of "DataBlock" — confirm this naming is preferred.

### Part 3A

1. **pto.vmov:** May not need a dedicated op if MLIR copy semantics suffice. Confirm if needed.
2. **pto.vdupi:** Is this distinct from `pto.vdup` with an immediate operand, or can `pto.vdup` handle both?
3. **Predicate ops (pand/por/pxor and predicate movement forms):** These need MLIR op definitions and verifier rules. Confirm priority.

### Part 3B

1. **pto.vperm naming:** a5_intrinsic `vgather` (in-register permute) mapped to `pto.vperm`. Confirm naming preference.
2. **pto.vshift naming:** a5_intrinsic `vsld` (single-source slide) mapped to `pto.vshift` to avoid `pto.vsld` collision. Confirm.
3. **Section 10 removals:** 4 interleave ops removed (not on A5). If multi-arch support is needed later, these would need conditional inclusion.

### Part 3C

1. **Fused op naming convention:** `pto.vaddrelu`, `pto.vaddreluconv`, `pto.vmulconv` use long compound names. Should we adopt a shorter convention (e.g., `pto.vfma_relu`)?
2. **vmrgsort4:** Kept from vpto-spec.md but no a5_intrinsic mapping found. Confirm if A5 supports this.
3. **Store dist token completeness:** PK_B16, MRG4CHN_B8, MRG2CHN_B8, MRG2CHN_B16 added. Are there other store distribution modes on A5?
4. **vcvt width-changing pattern:** The even/odd + vor pattern for f32→f16 is the standard compiler lowering. Confirm this is the intended representation in the spec.
5. **Stateful store ops (Section 14):** These are complex with SSA state threading. Are they all needed for A5, or can some be simplified?
