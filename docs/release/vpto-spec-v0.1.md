# PTO micro Instruction Spec — Draft (A5)

- v0.1: Doc Init

[toc]

---

## Part I: Architecture Overview

### Overview

This document defines the PTO micro Instruction, a compiler-internal and externally facing specification designed to represent vector compute kernels within the PTO architecture. Much like NVVM provides a robust IR for GPU architectures, the PTO micro Instruction serves as the direct bridge between high-level programming models and the underlying hardware ISA, providing a precise, low-level representation of vector workloads explicitly designed for the Ascend 950 architecture.

#### Position in the Stack and Layer Modeled

The PTO micro Instruction operates as a very low-level intermediate representation within the PTO compiler stack. It is uniquely designed to accurately and comprehensively express all architectural information of the Ascend 950 hardware. It specifically models the bare-metal vector execution layer, making hardware-specific capabilities and constraints, such as exact vector lane configurations, memory space hierarchies, and hardware-specific fusion semantics, fully transparent and controllable.

#### Why External Developers Read or Author PTO micro Instruction

While the majority of users will interact with the PTO architecture via higher-level frameworks, external developers may need to read or author PTO micro Instruction directly for several key reasons:

- Custom Toolchain Development: build custom compiler frontends or domain-specific languages (DSLs) that target the Ascend 950 architecture with maximum hardware utilization.
- Performance Engineering: inspect the output of high-level compiler passes, verify fine-grained optimization behaviors, and pinpoint performance bottlenecks at the architectural level.
- Micro-Optimization: hand-author highly optimized, critical mathematical kernels using a stable, precise IR when higher-level abstractions cannot achieve the theoretical peak performance of the hardware.

#### Relationship to CCE

The PTO micro Instruction is designed to express the full semantic capabilities of the Compute Cube Engine (CCE), but with significant structural and pipeline advantages for compiler development.

- Bypassing the C/Clang Pipeline: while CCE heavily relies on C/C++ extensions parsed by Clang, the PTO micro Instruction operates entirely independently of the C language frontend. By bypassing Clang AST generation and frontend processing, utilizing the PTO micro Instruction significantly reduces overall compilation time and memory overhead.
- Enhanced IR Verification: because the PTO micro Instruction is a strongly typed, SSA-based (Static Single Assignment) compiler IR rather than a C-wrapper API, it provides a much more rigorous and detailed IR verification process. Structural inconsistencies, invalid memory access patterns, and operand type mismatches are caught immediately with precise, explicit diagnostic feedback, providing developers with much higher visibility into kernel correctness than traditional CCE error reporting.

#### Intended Audience

This document is written for compiler engineers, library writers, and advanced performance architects. We expect the reader to have a working understanding of modern compiler infrastructure, specifically MLIR, the principles of Static Single Assignment (SSA) form, and a deep understanding of the vector-processing capabilities of the Ascend 950 architecture.

### Getting Started

The PTO micro Instruction is architected as a performance-critical layer within the compiler stack, specifically designed to exploit the **Decoupled Access-Execute** (DAE) nature of the Ascend 950 hardware.

#### Hardware Pipeline Modeling

The IR is structured to mirror the three primary hardware pipelines of the Ascend 950 architecture. Correct PTO micro Instruction authoring requires managing the interaction between these asynchronous units:

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

The PTO micro Instruction enforces a strict memory hierarchy. The Unified Buffer (UB) is the only valid operand source for vector compute instructions. Consequently, the architecture of a PTO micro Instruction program is defined by the explicit management of data movement:

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

For UB↔vreg data movement, besides contiguous load/store, the architecture provides rich access pattern support including strided access, pack/unpack, interleave/deinterleave, broadcast, upsample/downsample, channel split/merge, gather/scatter, and squeeze/expand operations. For detailed instruction syntax and distribution modes, refer to the [Vector Load/Store](#isa-03-vector-load-store) group in the ISA specification.

#### Synchronization Model

The Ascend 950 architecture employs a cluster-based design with a 1:2 ratio of Cube cores to Vector cores. The PTO micro Instruction provides multiple levels of synchronization to manage concurrent execution across pipelines and cores:

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

It is not a dedicated `pto` op. In the PTO micro Instruction, this scope is modeled as a specialized `scf.for` loop annotated with `llvm.loop.aivector_scope`. This gives the compiler a natural structural boundary for identifying the code block that must be lowered into a discrete VF hardware instruction sequence.

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

This document is the interface specification centered on the `mlir::pto` dialect and the shared MLIR surface used alongside it in PTO micro Instruction programs.

It only describes:

- operation names
- operand and result lists
- operand and result types
- important attributes
- C-style semantics for each operation

It does not describe lowering strategy.

PTO micro Instruction source programs are not restricted to `pto` operations alone. In practice they also use shared MLIR dialect ops, most notably the full scalar operation surface of `arith` together with structured control-flow ops from `scf`, to express scalar constants, scalar arithmetic, type conversion, comparisons, and structured control flow around PTO vector or tile regions. These shared-dialect ops are part of the supported PTO micro Instruction source surface and should be regarded as part of PTO-ISA alongside `pto` dialect operations.

### Shared MLIR Dialects

- `arith`: the full scalar `arith` surface is supported in PTO micro Instruction programs, covering scalar integer, floating-point, boolean, and `index` operations. In current samples the most common uses are still constants, offset/bounds arithmetic, casts, compares, and selects.
- `scf`: structured control flow used to model counted loops, conditional regions, loop-carried state, and break-like control around PTO compute and data-movement ops.
- Shared dialect ops remain in standard MLIR form so that PTO analyses and backend passes can reason about control flow and scalar state without re-encoding them as PTO-specific instructions.

### Core Types

#### Element Types
`vreg<T>`: `!pto.vreg<NxT>` Fixed-width PTO micro Instruction vector type with total width exactly 256 bytes (2048 bits). `N` is the lane count, `T` is the element type, and `N * bitwidth(T) = 2048`.

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

#### Address Space Conventions

PTO micro Instruction memory operands use `!pto.ptr<element-type, space>`. This specification models the following memory-space attributes:

| Space | Interpretation |
|-------|----------------|
| `gm` | Global Memory (GM), off-chip HBM/DDR storage |
| `ub` | Unified Buffer (UB), on-chip vector buffer |

Typical pointer construction and pointer arithmetic follow the same `!pto.ptr<..., space>` form:

```mlir
%0 = pto.castptr %c0 : i64 -> !pto.ptr<f32, ub>
%1 = pto.addptr %0, %c1024 : !pto.ptr<f32, ub> -> !pto.ptr<f32, ub>
```

#### `!pto.ptr<T, space>`

`!pto.ptr<T, space>` is the typed pointer form used for explicit memory operands in PTO micro Instruction.

- `T` is the element type associated with the pointed-to storage.
- `space` is the memory domain, typically `gm` or `ub` in this specification.
- A `pto.ptr` value carries an address plus its element-type / memory-space interpretation, but it does not carry tensor shape or stride metadata by itself.
- Tensor semantics are introduced separately through view-building operations such as `pto.make_tensor_view`.
- Pointer arithmetic is element-based rather than byte-based.

Typical examples:

- `!pto.ptr<f32, gm>`
- `!pto.ptr<f32, ub>`
- `!pto.ptr<bf16, gm>`

#### Pointer Operations

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

#### Special Types

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

All PTO micro Instruction operations follow standard MLIR syntax. The common patterns are:

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

PTO micro Instruction programs may interleave PTO ops with standard MLIR `arith` and `scf` ops.
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

## Part III: ISA Instruction Reference — Summary

This section provides a categorized overview of all PTO micro Instruction operations plus the shared MLIR `arith` and `scf` ops that may appear in PTO micro Instruction programs. Detailed documentation for each group is included later in this merged document.

---

## Instruction Groups

| # | Group | Description | Count | Details |
|---|-------|-------------|-------|---------|
| 1 | [Pipeline Sync](#isa-01-pipeline-sync) | Intra-core pipeline synchronization | 5 | `pto.set_flag`, `pto.wait_flag`, `pto.pipe_barrier`, `pto.get_buf`, `pto.rls_buf` |
| 2 | [DMA Copy Programming](#isa-02-dma-copy) | DMA configuration and transfer between GM↔UB | 9 | `pto.set_loop*_stride_*`, `pto.set_loop_size_*`, `pto.copy_gm_to_ubuf`, `pto.copy_ubuf_to_ubuf`, `pto.copy_ubuf_to_gm` |
| 3 | [Vector Load/Store](#isa-03-vector-load-store) | UB↔vreg data movement with various access patterns | ~20 | `pto.vlds`, `pto.vldx2`, `pto.vgather2`, `pto.vsts`, `pto.vstx2`, `pto.vscatter`, etc. |
| 4 | [Predicate Load/Store](#isa-04-predicate-load-store) | UB↔mask register movement | 7 | `pto.plds`, `pto.pld`, `pto.pldi`, `pto.psts`, `pto.pst`, `pto.psti`, `pto.pstu` |
| 5 | [Materialization & Predicate Ops](#isa-05-materialization-predicate) | Scalar broadcast, predicate generation and manipulation | ~17 | `pto.vbr`, `pto.vdup`, `pto.pset_b*`, `pto.pge_b*`, `pto.plt_b*`, `pto.ppack`, `pto.punpack`, `pto.pnot`, `pto.psel`, etc. |
| 6 | [Unary Vector Ops](#isa-06-unary-vector-ops) | Single-input element-wise operations | 9 | `pto.vabs`, `pto.vexp`, `pto.vln`, `pto.vsqrt`, `pto.vrec`, `pto.vrelu`, `pto.vnot`, `pto.vbcnt`, `pto.vcls` |
| 7 | [Binary Vector Ops](#isa-07-binary-vector-ops) | Two-input element-wise operations | 13 | `pto.vadd`, `pto.vsub`, `pto.vmul`, `pto.vdiv`, `pto.vmax`, `pto.vmin`, `pto.vand`, `pto.vor`, `pto.vxor`, `pto.vshl`, `pto.vshr`, `pto.vaddc`, `pto.vsubc` |
| 8 | [Vec-Scalar Ops](#isa-08-vec-scalar-ops) | Vector-scalar operations | 8 | `pto.vadds`, `pto.vmuls`, `pto.vmaxs`, `pto.vmins`, `pto.vlrelu`, `pto.vshls`, `pto.vshrs`, `pto.vaddcs`, `pto.vsubcs` |
| 9 | [Conversion Ops](#isa-09-conversion-ops) | Type conversion with rounding/saturation control | 2 | `pto.vcvt`, `pto.vtrc` |
| 10 | [Reduction Ops](#isa-10-reduction-ops) | Vector reductions | 3 | `pto.vcadd`, `pto.vcmax`, `pto.vcmin` |
| 11 | [Compare & Select](#isa-11-compare-select) | Comparison and conditional selection | 5 | `pto.vcmp`, `pto.vcmps`, `pto.vsel`, `pto.vselr`, `pto.vselrv2` |
| 12 | [Data Rearrangement](#isa-12-data-rearrangement) | In-register data movement and permutation | 4 | `pto.vintlv`, `pto.vdintlv`, `pto.vintlvv2`, `pto.vdintlvv2` |
| 13 | [DSA/SFU Ops](#isa-13-dsa-sfu-ops) | Specialized ops, index generation, and sorting helpers | 5 | `pto.vmull`, `pto.vmula`, `pto.vci`, `pto.vbitsort`, `pto.vmrgsort4` |
| 14 | [Arith (Shared MLIR Dialect)](#isa-14-shared-arith) | Full scalar `arith` surface used around PTO ops; the companion page lists categories and representative examples | all scalar ops | `arith.constant`, `arith.addi`, `arith.addf`, `arith.cmpi`, `arith.cmpf`, `arith.select`, `arith.index_cast`, `arith.extsi`, `arith.trunci`, `arith.andi`, `arith.shli`, etc. |
| 15 | [SCF (Shared MLIR Dialect)](#isa-15-shared-scf) | Structured loops, branches, and loop-carried state around PTO regions | 5 | `scf.for`, `scf.if`, `scf.while`, `scf.condition`, `scf.yield` |

---

## Detailed ISA Group Reference

This section inlines the 15 ISA group documents so the architectural overview, notation, summary table, and per-group semantics can be read in a single file.

<a id="isa-01-pipeline-sync"></a>

### 1. Pipeline Synchronization

> **Category:** Synchronization primitives for coordinating pipeline execution
> **Pipelines:** MTE2 (GM→UB), PIPE_V (Vector), MTE3 (UB→GM)

The PTO micro Instruction model operates on the Ascend 950's **Decoupled Access-Execute** architecture. The MTE and Vector pipelines run asynchronously, requiring explicit synchronization to prevent data hazards.

---

#### Intra-Core Pipeline Sync

These ops coordinate data flow between pipelines within a single vector core.

##### `pto.set_flag`

- **syntax:** `pto.set_flag["SRC_PIPE", "DST_PIPE", "EVENT_ID"]`
- **semantics:** Signal event from source pipe to destination pipe.

```c
set_flag(src_pipe, dst_pipe, event_id);
```

**Example:** After MTE2 completes GM→UB transfer, signal Vector pipe:
```mlir
pto.set_flag["PIPE_MTE2", "PIPE_V", "EVENT_ID0"]
```

---

##### `pto.wait_flag`

- **syntax:** `pto.wait_flag["SRC_PIPE", "DST_PIPE", "EVENT_ID"]`
- **semantics:** Block destination pipe until source pipe signals event.

```c
wait_flag(src_pipe, dst_pipe, event_id);
```

**Example:** Vector pipe waits for MTE2 data to arrive:
```mlir
pto.wait_flag["PIPE_MTE2", "PIPE_V", "EVENT_ID0"]
```

---

##### `pto.pipe_barrier`

- **syntax:** `pto.pipe_barrier "PIPE_*"`
- **semantics:** Drain all pending ops in the specified pipe. All previously issued operations on that pipe complete before any subsequent operation begins.

```c
pipe_barrier(pipe);
```

**Pipe identifiers:** `PIPE_MTE2`, `PIPE_V`, `PIPE_MTE3`

**Example:** Two back-to-back `copy_ubuf_to_gm` calls writing to the same GM address. Without a barrier, MTE3 may reorder them and the final GM value is non-deterministic:

```mlir
// Both stores target the same GM address — order matters!
pto.copy_ubuf_to_gm %ub_partial_0, %gm_result, ...
// Without pipe_barrier, MTE3 could execute the second copy before the first
// completes, producing a non-deterministic result at %gm_result.
pto.pipe_barrier "PIPE_MTE3"
// After barrier: first copy is guaranteed complete. Second copy overwrites deterministically.
pto.copy_ubuf_to_gm %ub_partial_1, %gm_result, ...
```

---

##### `pto.get_buf`

- **syntax:** `pto.get_buf "PIPE_*", %buf_id, %mode : i64, i64`
- **semantics:** Acquire buffer slot for inter-pipeline double-buffering coordination.

```c
get_buf(pipe, buf_id, mode);
```

---

##### `pto.rls_buf`

- **syntax:** `pto.rls_buf "PIPE_*", %buf_id, %mode : i64, i64`
- **semantics:** Release buffer slot to allow other pipeline to proceed.

```c
rls_buf(pipe, buf_id, mode);
```

---

##### `pto.mem_bar`

- **syntax:** `pto.mem_bar "BARRIER_TYPE"`
- **semantics:** Intra-vector-pipe memory fence within `__VEC_SCOPE__`. Required when UB addresses alias between vector load/store operations.

```c
mem_bar(barrier_type);
```

**Barrier types:**

| Type | Semantics |
|------|-----------|
| `VV_ALL` | All prior vector ops complete before subsequent |
| `VST_VLD` | All prior vector stores visible before subsequent loads |
| `VLD_VST` | All prior vector loads complete before subsequent stores |

**Example:** Ensure stores are visible before loads to same UB region:
```mlir
pto.vsts %v0, %ub[%c0] : !pto.vreg<64xf32>, !pto.ptr<f32, ub>
pto.mem_bar "VST_VLD"
%v1 = pto.vlds %ub[%c0] : !pto.ptr<f32, ub> -> !pto.vreg<64xf32>
```

---

#### Intra-Core Sync Patterns & Examples

##### Example 1: `set_flag` / `wait_flag` (Explicit Events)

Each cross-pipeline data dependency requires an explicit signal/wait pair. The programmer must manually insert `set_flag` after the producer and `wait_flag` before the consumer.

```mlir
// ─── Stage 1: MTE2 loads data from GM into UB ───
pto.copy_gm_to_ubuf %gm_ptr, %ub_ptr, ...

// MTE2 signals: "UB data is ready for Vector pipe"
pto.set_flag["PIPE_MTE2", "PIPE_V", "EVENT_ID0"]

// ─── Stage 2: Vector pipe consumes UB data ───
// Vector waits until MTE2's signal arrives
pto.wait_flag["PIPE_MTE2", "PIPE_V", "EVENT_ID0"]

scf.for %dummy = %c0 to %c1 step %c1 {
  %v   = pto.vlds %ub_ptr[%lane] : !pto.ptr<f32, ub> -> !pto.vreg<64xf32>
  %mask = pto.pset_b32 "PAT_ALL" : !pto.mask
  %abs = pto.vabs %v, %mask : !pto.vreg<64xf32>, !pto.mask -> !pto.vreg<64xf32>
  pto.vsts %abs, %ub_out[%lane], %mask : !pto.vreg<64xf32>, !pto.ptr<f32, ub>, !pto.mask
} {llvm.loop.aivector_scope}

// Vector signals: "UB output is ready for MTE3"
pto.set_flag["PIPE_V", "PIPE_MTE3", "EVENT_ID0"]

// ─── Stage 3: MTE3 stores result from UB back to GM ───
// MTE3 waits until Vector's signal arrives
pto.wait_flag["PIPE_V", "PIPE_MTE3", "EVENT_ID0"]

pto.copy_ubuf_to_gm %ub_out, %gm_out, ...
```

**Key property:** Every cross-pipeline edge is an explicit `(set_flag, wait_flag)` pair. Simple for straight-line code, but gets verbose in loops (see Example 3).

---

##### Example 2: `get_buf` / `rls_buf` (Resource-Based)

Instead of naming events, each pipeline declares when it **acquires** (`get_buf`) and **releases** (`rls_buf`) a shared UB buffer. Cross-pipeline RAW/WAR dependencies are resolved implicitly by program order — if MTE2 releases `buf_A` and Vector later acquires `buf_A`, the hardware ensures the acquire cannot proceed until the release completes.

```mlir
// ─── Stage 1: MTE2 loads data into UB ───
// MTE2 acquires ub_ptr — blocks if Vector hasn't released it from a prior iteration
pto.get_buf "PIPE_MTE2", %bufid_ub_ptr, %mode : i64, i64
pto.copy_gm_to_ubuf %gm_ptr, %ub_ptr, ...
// MTE2 done writing ub_ptr — release it so Vector can consume
pto.rls_buf "PIPE_MTE2", %bufid_ub_ptr, %mode : i64, i64

// ─── Stage 2: Vector computation ───
// Vector acquires ub_ptr (input) — blocks until MTE2 releases it (RAW: MTE2 write → V read)
pto.get_buf "PIPE_V", %bufid_ub_ptr, %mode : i64, i64
// Vector acquires ub_out (output) — blocks until MTE3 releases it from a prior iteration (WAR: MTE3 read → V write)
pto.get_buf "PIPE_V", %bufid_ub_out, %mode : i64, i64

scf.for %dummy = %c0 to %c1 step %c1 {
  %mask = pto.pset_b32 "PAT_ALL" : !pto.mask
  %v   = pto.vlds %ub_ptr[%lane] : !pto.ptr<f32, ub> -> !pto.vreg<64xf32>
  %abs = pto.vabs %v, %mask : !pto.vreg<64xf32>, !pto.mask -> !pto.vreg<64xf32>
  pto.vsts %abs, %ub_out[%lane], %mask : !pto.vreg<64xf32>, !pto.ptr<f32, ub>, !pto.mask
} {llvm.loop.aivector_scope}

// Vector done reading ub_ptr — release so MTE2 can reuse it in next iteration
pto.rls_buf "PIPE_V", %bufid_ub_ptr, %mode : i64, i64
// Vector done writing ub_out — release so MTE3 can consume
pto.rls_buf "PIPE_V", %bufid_ub_out, %mode : i64, i64

// ─── Stage 3: MTE3 stores result to GM ───
// MTE3 acquires ub_out — blocks until Vector releases it (RAW: V write → MTE3 read)
pto.get_buf "PIPE_MTE3", %bufid_ub_out, %mode : i64, i64
pto.copy_ubuf_to_gm %ub_out, %gm_out, ...
// MTE3 done reading ub_out — release so Vector can reuse it in next iteration
pto.rls_buf "PIPE_MTE3", %bufid_ub_out, %mode : i64, i64
```

**Key property:** No event IDs needed. Dependencies are implicit from program order of `get_buf`/`rls_buf` on the same buffer ID. This becomes much more convenient in multi-iteration loops (see Example 3).

---

##### Example 3: Ping/Pong Double-Buffering Loop

Double-buffering overlaps DMA and compute by using two UB buffers alternately. All three stages (MTE2, Vector, MTE3) appear in the **same iteration** — the hardware pipelines them across iterations because different iterations operate on different buffers (`buf[i%2]`).

###### Event ID scheme (`set_flag` / `wait_flag`)

With 2 ping/pong buffers and 2 pipeline pairs (MTE2↔V, V↔MTE3), `set_flag`/`wait_flag` needs **8 event IDs** = 2 pipe-pairs × 2 buffers × (forward + reverse):

**MTE2 ↔ Vector (input buffers):**

| Event ID | Direction | Purpose |
|----------|-----------|---------|
| `EVT_IN_FWD_0` | MTE2 → V | RAW: buf_in[0] data ready |
| `EVT_IN_FWD_1` | MTE2 → V | RAW: buf_in[1] data ready |
| `EVT_IN_REV_0` | V → MTE2 | WAR: Vector done reading buf_in[0] |
| `EVT_IN_REV_1` | V → MTE2 | WAR: Vector done reading buf_in[1] |

**Vector ↔ MTE3 (output buffers):**

| Event ID | Direction | Purpose |
|----------|-----------|---------|
| `EVT_OUT_FWD_0` | V → MTE3 | RAW: buf_out[0] result ready |
| `EVT_OUT_FWD_1` | V → MTE3 | RAW: buf_out[1] result ready |
| `EVT_OUT_REV_0` | MTE3 → V | WAR: MTE3 done reading buf_out[0] |
| `EVT_OUT_REV_1` | MTE3 → V | WAR: MTE3 done reading buf_out[1] |

###### 3a. `set_flag` / `wait_flag` version

```mlir
// ═══ Pre-loop: prime ALL reverse-dependency signals ═══
// Both input and output buffers start unused. We must pre-send
// reverse-dep signals so the first iteration's wait_flags don't deadlock.
pto.set_flag["PIPE_V",    "PIPE_MTE2", "EVT_IN_REV_0"]   // ◀ PRIME: buf_in[0] "free"
pto.set_flag["PIPE_V",    "PIPE_MTE2", "EVT_IN_REV_1"]   // ◀ PRIME: buf_in[1] "free"
pto.set_flag["PIPE_MTE3", "PIPE_V",    "EVT_OUT_REV_0"]  // ◀ PRIME: buf_out[0] "free"
pto.set_flag["PIPE_MTE3", "PIPE_V",    "EVT_OUT_REV_1"]  // ◀ PRIME: buf_out[1] "free"

scf.for %i = %c0 to %N step %c1 {
  // ── All 3 stages in same iteration, indexed by i%2 ──
  // %pp = i % 2  (ping/pong selector for buffer & event IDs)

  // ── MTE2: load tile[i] into buf_in[i%2] ──
  // WAR: wait until Vector has released buf_in[i%2] from iteration i-2
  pto.wait_flag["PIPE_V", "PIPE_MTE2", "EVT_IN_REV_{pp}"]
  pto.copy_gm_to_ubuf %gm_ptr[%i], %ub_in[%pp], ...
  // RAW: signal Vector that buf_in[i%2] data is ready
  pto.set_flag["PIPE_MTE2", "PIPE_V", "EVT_IN_FWD_{pp}"]

  // ── Vector: compute buf_in[i%2] → buf_out[i%2] ──
  // RAW: wait for MTE2 to finish loading buf_in[i%2]
  pto.wait_flag["PIPE_MTE2", "PIPE_V", "EVT_IN_FWD_{pp}"]
  // WAR: wait for MTE3 to finish reading buf_out[i%2] from iteration i-2
  pto.wait_flag["PIPE_MTE3", "PIPE_V", "EVT_OUT_REV_{pp}"]
  scf.for %dummy = %c0 to %c1 step %c1 {
    %v   = pto.vlds %ub_in[%pp][%lane] : !pto.ptr<f32, ub> -> !pto.vreg<64xf32>
    %mask = pto.pset_b32 "PAT_ALL" : !pto.mask
    %abs = pto.vabs %v, %mask : !pto.vreg<64xf32>, !pto.mask -> !pto.vreg<64xf32>
    pto.vsts %abs, %ub_out[%pp][%lane], %mask : !pto.vreg<64xf32>, !pto.ptr<f32, ub>, !pto.mask
  } {llvm.loop.aivector_scope}
  // WAR: tell MTE2 "done reading buf_in[i%2]"
  pto.set_flag["PIPE_V", "PIPE_MTE2", "EVT_IN_REV_{pp}"]
  // RAW: tell MTE3 "buf_out[i%2] result ready"
  pto.set_flag["PIPE_V", "PIPE_MTE3", "EVT_OUT_FWD_{pp}"]

  // ── MTE3: store result from buf_out[i%2] to GM ──
  // RAW: wait for Vector to finish writing buf_out[i%2]
  pto.wait_flag["PIPE_V", "PIPE_MTE3", "EVT_OUT_FWD_{pp}"]
  pto.copy_ubuf_to_gm %ub_out[%pp], %gm_out[%i], ...
  // WAR: tell Vector "done reading buf_out[i%2]"
  pto.set_flag["PIPE_MTE3", "PIPE_V", "EVT_OUT_REV_{pp}"]
}

// ═══ Post-loop: drain — match every pre-loop prime with a wait ═══
// Each priming set_flag must be paired. The last loop iteration's
// set_flags are consumed by wait_flags that will never fire inside the
// loop (there is no iteration i+2). Drain them here.
pto.wait_flag["PIPE_V",    "PIPE_MTE2", "EVT_IN_REV_{(N-1)%2}"]  // ◀ DRAIN
pto.wait_flag["PIPE_V",    "PIPE_MTE2", "EVT_IN_REV_{(N-2)%2}"]  // ◀ DRAIN
pto.wait_flag["PIPE_MTE3", "PIPE_V",    "EVT_OUT_REV_{(N-1)%2}"] // ◀ DRAIN
pto.wait_flag["PIPE_MTE3", "PIPE_V",    "EVT_OUT_REV_{(N-2)%2}"] // ◀ DRAIN
```

**What `set_flag`/`wait_flag` requires outside the loop:**
- **Before the loop (4 × `set_flag`):** Prime every reverse-dependency event ID — one per buffer per pipe-pair. Without this, the first iteration's `wait_flag` for reverse deps would deadlock (no signal was ever sent).
- **After the loop (4 × `wait_flag`):** Drain the matching reverse-dep signals from the last iterations. Every `set_flag` must be paired with a `wait_flag` — the last loop iterations produce signals that no subsequent iteration consumes, so they must be drained explicitly.

###### 3b. `get_buf` / `rls_buf` version

Same ping/pong double-buffering, but **no pre-loop priming or post-loop draining needed.** Buffer acquire/release semantics handle everything.

```mlir
scf.for %i = %c0 to %N step %c1 {
  // %pp = i % 2  (ping/pong selector)

  // ── MTE2: load tile[i] into buf[i%2] ──
  // Acquires buf[i%2] — on first iteration, buffer is free so proceeds immediately.
  // On later iterations, blocks until Vector releases buf[i%2] (WAR: automatic).
  pto.get_buf %bufid_buf[%pp], "PIPE_MTE2"
  pto.copy_gm_to_ubuf %gm_ptr[%i], %ub_buf[%pp], ...
  pto.rls_buf %bufid_buf[%pp], "PIPE_MTE2"

  // ── Vector: compute on buf[i%2] ──
  // Acquires buf[i%2] — blocks until MTE2 releases it (RAW: automatic)
  pto.get_buf %bufid_buf[%pp], "PIPE_V"
  pto.get_buf %bufid_out[%pp], "PIPE_V"
  scf.for %dummy = %c0 to %c1 step %c1 {
    %v   = pto.vlds %ub_buf[%pp][%lane] : !pto.ptr<f32, ub> -> !pto.vreg<64xf32>
    %mask = pto.pset_b32 "PAT_ALL" : !pto.mask
    %abs = pto.vabs %v, %mask : !pto.vreg<64xf32>, !pto.mask -> !pto.vreg<64xf32>
    pto.vsts %abs, %ub_out[%pp][%lane], %mask : !pto.vreg<64xf32>, !pto.ptr<f32, ub>, !pto.mask
  } {llvm.loop.aivector_scope}
  // Release buf[i%2] — MTE2 can reuse in iteration i+2 (WAR resolved)
  pto.rls_buf %bufid_buf[%pp], "PIPE_V"
  pto.rls_buf %bufid_out[%pp], "PIPE_V"

  // ── MTE3: store result ──
  // Acquires out[i%2] — blocks until Vector releases it (RAW: automatic)
  pto.get_buf %bufid_out[%pp], "PIPE_MTE3"
  pto.copy_ubuf_to_gm %ub_out[%pp], %gm_out[%i], ...
  pto.rls_buf %bufid_out[%pp], "PIPE_MTE3"
}
// No post-loop drain needed — last rls_buf completes the pipeline.
```

**No priming, no draining, no event IDs.** The acquire/release protocol on buffer IDs indexed by `i%2` implicitly resolves all cross-pipeline dependencies:
- **RAW** (MTE2→V): Vector's `get_buf` blocks until MTE2's `rls_buf` on `buf[i%2]`
- **WAR** (V→MTE2): MTE2's `get_buf` in iteration `i+2` blocks until Vector's `rls_buf` in iteration `i` (same buffer)
- **First iteration:** Buffer is initially free, so `get_buf` proceeds without blocking — no priming needed

---

#### Comparison Summary

| Aspect | `set_flag` / `wait_flag` | `get_buf` / `rls_buf` |
|--------|--------------------------|------------------------|
| Dependency model | Explicit event signals | Implicit via buffer acquire/release |
| IDs per pipe-pair | **8** = 2 buffers × 2 dirs × 2 (fwd+rev) | 1 fwd + 1 rev per buffer (shared global pool) |
| Total HW IDs | 8 per pipe-pair, grows with buffers | **32 global** across all pipes |
| Reverse (WAR) deps | Extra `set_flag`/`wait_flag` pair per buffer | Handled automatically |
| Pre-loop setup | `set_flag` to prime each reverse dep | None |
| Post-loop teardown | `wait_flag` to drain all primed signals | None |
| Straight-line code | Simple, clear | Slightly more verbose (bracket each stage) |
| Ping/pong loops | 8 event IDs + 4 prime + 4 drain | Same pattern, no overhead |
| Best used for | Simple pipelines, fine-grained control | Double/multi-buffering, complex loops |

---

#### Inter-Core Sync

> **Note:** Inter-core sync is only needed for **mixed Cube+Vector tasks** where Cube produces data that Vector consumes (or vice versa). **Vec-only tasks can ignore this section entirely.**

These ops coordinate execution across the Cube block and Vector subblocks within a cluster. Each core cluster consists of **1 Cube block : 2 Vector subblocks**, each with its own **SU (Sequencer Unit)** running independent instruction streams.

```
Core Cluster (1:2 ratio)
┌─────────────────────────────────────────────┐
│  ┌──────────────┐    ┌──────────────┐       │
│  │  AIC (Cube)  │    │  AIV0 (Vec)  │       │
│  │  ┌────────┐  │    │  ┌────────┐  │       │
│  │  │   SU   │──┼────┼──│   SU   │  │       │
│  │  └────────┘  │    │  └────────┘  │       │
│  │  CUBE pipe   │    │  MTE2/V/MTE3 │       │
│  │  L0C buffer  │    │  UB (256KB)  │       │
│  └──────────────┘    └──────────────┘       │
│                      ┌──────────────┐       │
│                      │  AIV1 (Vec)  │       │
│                      │  ┌────────┐  │       │
│                      │  │   SU   │  │       │
│                      │  └────────┘  │       │
│                      │  MTE2/V/MTE3 │       │
│                      │  UB (256KB)  │       │
│                      └──────────────┘       │
└─────────────────────────────────────────────┘
```

##### Platform Comparison

| Aspect | A2A3 (Ascend 910) | A5 (Ascend 950) |
|--------|-------------------|-----------------|
| **Signal op** | `set_cross_core` (mode2) | `set_intra_block` |
| **Wait op** | `wait_flag_dev` | `wait_intra_core` |
| **Wait behavior** | SU-level blocking (entire core stalls) | Per-pipeline (only named pipe stalls) |
| **Semaphore pool** | 16 IDs per cluster, 4-bit counter | 16 IDs, but 32-ID address space (see below) |
| **C→V** | **Broadcast**: one `set` reaches both AIV0+AIV1 | **1:1**: separate `set` per subblock required |
| **V→C** | **Reduce**: Cube waits for both subblocks in one `wait` | **1:1**: Cube needs separate `wait` per subblock |

##### A2A3: `set_cross_core` / `wait_flag_dev`

```c
// mode2 broadcast/reduce semantics for 1:2 cluster
set_cross_core(pipe, semaphore_id);   // pipe: VEC/MTE2/CUBE/FIX
wait_flag_dev(semaphore_id);          // SU-level blocking
```

```
C→V Broadcast (one set reaches both):
    AIC ──set_cross_core──┬──> AIV0 sema++
                          └──> AIV1 sema++

V→C Reduce (one wait for both):
    AIV0 ──set_cross_core──┐
                           ├──> AIC wait_flag_dev (blocks until BOTH)
    AIV1 ──set_cross_core──┘
```

##### `pto.set_cross_core`

- **syntax:** `pto.set_cross_core %core_id, %event_id : i64, i64`
- **semantics:** Signal event to another core. Uses **mode2** for 1:2 cluster on A2A3.

##### `pto.wait_flag_dev`

- **syntax:** `pto.wait_flag_dev %core_id, %event_id : i64, i64`
- **semantics:** Wait for event from another core. **SU-level blocking** — entire core stalls.

##### A5: `set_intra_block` / `wait_intra_core`

```c
set_intra_block(trigger_pipe, semaphore_id);
wait_intra_core(wait_pipe, semaphore_id);   // only named pipe stalls
```

**A5 semaphore address space:** The hardware has **16 physical semaphore IDs** but exposes a **32-ID address space** to support 1:1 signaling to each subblock:

| ID Range | Target |
|----------|--------|
| 0–15 | AIV0 (subblock 0) |
| 16–31 (+15 offset) | AIV1 (subblock 1) |

This means C→V requires **separate `set_intra_block` calls** per subblock (no broadcast), and V→C requires **separate `wait_intra_core` calls** per subblock (no hardware reduce).

```
C→V on A5 (1:1, no broadcast — need two sets):
    AIC ──set_intra_block(pipe, sema_id)────> AIV0
    AIC ──set_intra_block(pipe, sema_id+15)──> AIV1

V→C on A5 (1:1, no reduce — need two waits):
    AIV0 ──set_intra_block──> AIC wait_intra_core(pipe, sema_id)
    AIV1 ──set_intra_block──> AIC wait_intra_core(pipe, sema_id+15)  // extra wait
```

##### `pto.set_intra_block`

- **syntax:** `pto.set_intra_block %block_id, %event_id : i64, i64`
- **semantics:** Signal event within a block (A5). Specifies **trigger pipe**. 1:1 per subblock.

##### `pto.wait_intra_core`

- **syntax:** `pto.wait_intra_core %block_id, %event_id : i64, i64`
- **semantics:** Wait for event within block (A5). Specifies **which pipeline should wait** — only that pipe stalls, SU and other pipes continue.

##### Wait Granularity Comparison

```
A2A3 wait_flag_dev (SU-level stall):
    SU ──┬── PIPE_MTE2 ───╳ ALL STALLED
         ├── PIPE_V    ───╳ ALL STALLED
         └── PIPE_MTE3 ───╳ ALL STALLED

A5 wait_intra_core "PIPE_MTE2" (per-pipe stall):
    SU ──┬── PIPE_MTE2 ───╳ STALLED (waiting for Cube)
         ├── PIPE_V    ─── ✓ RUNNING
         └── PIPE_MTE3 ─── ✓ RUNNING
```

---

<a id="isa-02-dma-copy"></a>

### 2. DMA Copy Programming

> **Category:** DMA transfer configuration and execution
> **Pipelines:** MTE2 (GM→UB), MTE3 (UB→GM)

DMA transfers move data between Global Memory (GM) and Unified Buffer (UB). The MTE engines operate asynchronously from the Vector core, requiring explicit sync (see [Pipeline Sync](#isa-01-pipeline-sync)).

The MTE2/MTE3 DMA engine executes a **multi-level nested loop** transfer. Before issuing the copy instruction, stride and loop-size registers must be configured.

---

#### Loop Stride Configuration (GM→UB)

These ops configure the MTE2 DMA engine's hardware loops for GM→UB transfers. They must be set **before** calling `pto.copy_gm_to_ubuf`.

##### `pto.set_loop_size_outtoub`

- **syntax:** `pto.set_loop_size_outtoub %loop1_count, %loop2_count : i64, i64`
- **semantics:** Configure HW loop iteration counts for GM→UB DMA.

**Parameter Table:**

| Parameter | Width | Description |
|-----------|-------|-------------|
| `%loop1_count` | 21 bits | Inner HW loop iteration count |
| `%loop2_count` | 21 bits | Outer HW loop iteration count |

When not using multi-level looping, set both to 1.

---

##### `pto.set_loop2_stride_outtoub`

- **syntax:** `pto.set_loop2_stride_outtoub %src_stride, %dst_stride : i64, i64`
- **semantics:** Configure outer loop (loop2) pointer advance for GM→UB DMA.

**Parameter Table:**

| Parameter | Width | Description |
|-----------|-------|-------------|
| `%src_stride` | 40 bits | GM source pointer advance per loop2 iteration (bytes) |
| `%dst_stride` | 21 bits | UB destination pointer advance per loop2 iteration (bytes) |

After each loop2 iteration, the DMA engine advances the GM read pointer by `%src_stride` and UB write pointer by `%dst_stride`.

---

##### `pto.set_loop1_stride_outtoub`

- **syntax:** `pto.set_loop1_stride_outtoub %src_stride, %dst_stride : i64, i64`
- **semantics:** Configure inner loop (loop1) pointer advance for GM→UB DMA.

**Parameter Table:**

| Parameter | Width | Description |
|-----------|-------|-------------|
| `%src_stride` | 40 bits | GM source pointer advance per loop1 iteration (bytes) |
| `%dst_stride` | 21 bits | UB destination pointer advance per loop1 iteration (bytes) |

---

#### Loop Stride Configuration (UB→GM)

These ops configure the MTE3 DMA engine's hardware loops for UB→GM transfers. They must be set **before** calling `pto.copy_ubuf_to_gm`.

Note: UB stride fields are 21 bits (sufficient for 256KB UB address space), GM stride fields are 40 bits (full GM address range).

##### `pto.set_loop_size_ubtoout`

- **syntax:** `pto.set_loop_size_ubtoout %loop1_count, %loop2_count : i64, i64`
- **semantics:** Configure HW loop iteration counts for UB→GM DMA.

**Parameter Table:**

| Parameter | Width | Description |
|-----------|-------|-------------|
| `%loop1_count` | 21 bits | Inner HW loop iteration count |
| `%loop2_count` | 21 bits | Outer HW loop iteration count |

---

##### `pto.set_loop2_stride_ubtoout`

- **syntax:** `pto.set_loop2_stride_ubtoout %src_stride, %dst_stride : i64, i64`
- **semantics:** Configure outer loop (loop2) pointer advance for UB→GM DMA.

**Parameter Table:**

| Parameter | Width | Description |
|-----------|-------|-------------|
| `%src_stride` | 21 bits | UB source pointer advance per loop2 iteration (bytes) |
| `%dst_stride` | 40 bits | GM destination pointer advance per loop2 iteration (bytes) |

---

##### `pto.set_loop1_stride_ubtoout`

- **syntax:** `pto.set_loop1_stride_ubtoout %src_stride, %dst_stride : i64, i64`
- **semantics:** Configure inner loop (loop1) pointer advance for UB→GM DMA.

**Parameter Table:**

| Parameter | Width | Description |
|-----------|-------|-------------|
| `%src_stride` | 21 bits | UB source pointer advance per loop1 iteration (bytes) |
| `%dst_stride` | 40 bits | GM destination pointer advance per loop1 iteration (bytes) |

---

#### DMA Transfer Execution

##### `pto.copy_gm_to_ubuf`

- **syntax:**
```mlir
pto.copy_gm_to_ubuf %gm_src, %ub_dst,
    %sid, %n_burst, %len_burst, %left_padding, %right_padding,
    %data_select_bit, %l2_cache_ctl, %src_stride, %dst_stride
    : !pto.ptr<T, gm>, !pto.ptr<T, ub>, i64, i64, i64,
      i64, i64, i1, i64, i64, i64
```
- **semantics:** DMA transfer from Global Memory (`!pto.ptr<T, gm>`) to Unified Buffer (`!pto.ptr<T, ub>`).

**Parameters:**

| Parameter | Description |
|-----------|-------------|
| `%gm_src` | GM source pointer (`!pto.ptr<T, gm>`) |
| `%ub_dst` | UB destination pointer (`!pto.ptr<T, ub>`, 32B-aligned) |
| `%sid` | Stream ID (usually 0) |
| `%n_burst` | Number of burst rows (innermost loop count) |
| `%len_burst` | Contiguous bytes transferred per burst row |
| `%left_padding` | Left padding count (bytes) |
| `%right_padding` | Right padding count (bytes) |
| `%data_select_bit` | Padding / data-select control bit (`i1`) |
| `%l2_cache_ctl` | L2 cache allocate control (TBD — controls whether DMA allocates in L2 cache) |
| `%src_stride` | GM source stride: start-to-start distance between consecutive burst rows (bytes) |
| `%dst_stride` | UB destination stride: start-to-start distance between consecutive burst rows (bytes, 32B-aligned) |

---

##### `pto.copy_ubuf_to_gm`

- **syntax:**
```mlir
pto.copy_ubuf_to_gm %ub_src, %gm_dst,
    %sid, %n_burst, %len_burst, %reserved, %dst_stride, %src_stride
    : !pto.ptr<T, ub>, !pto.ptr<T, gm>, i64, i64, i64, i64, i64, i64
```
- **semantics:** DMA transfer from Unified Buffer (`!pto.ptr<T, ub>`) to Global Memory (`!pto.ptr<T, gm>`). MTE3 reads only `len_burst` bytes from each UB row (de-padding).

**Parameters:**

| Parameter | Description |
|-----------|-------------|
| `%ub_src` | UB source pointer (`!pto.ptr<T, ub>`, 32B-aligned) |
| `%gm_dst` | GM destination pointer (`!pto.ptr<T, gm>`) |
| `%sid` | Stream ID (usually 0) |
| `%n_burst` | Number of burst rows |
| `%len_burst` | Contiguous bytes transferred per burst row |
| `%reserved` | Reserved field (set to 0) |
| `%dst_stride` | GM destination stride: start-to-start distance between consecutive burst rows (bytes) |
| `%src_stride` | UB source stride: start-to-start distance between consecutive burst rows (bytes, 32B-aligned) |

---

##### `pto.copy_ubuf_to_ubuf`

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

#### Burst / Stride / Pad Model

All A5 DMA addresses are **stride-based**: stride is the distance from the start of one row to the start of the next row (`stride >= lenBurst`). There is no separate "gap" parameter.

##### Key Terms

```
burst    = lenBurst contiguous bytes transferred per row
stride   = distance (bytes) from start of row[r] to start of row[r+1]
pad      = ub_stride - lenBurst, padded to the 32B alignment boundary
```

##### Alignment Constraints

- **UB addresses** (both source and destination) must be **32-byte aligned**.
- **GM→UB padding**: When `data_select_bit = true`, each UB row is padded from `lenBurst` up to the **32B-aligned boundary** of `ub_stride` with `pad_val` (set via `set_mov_pad_val`). This ensures every UB row starts at a 32B-aligned offset.
- **UB→GM de-padding**: MTE3 reads `lenBurst` bytes from each 32B-aligned UB row (skipping any padding that was added during load), writing only valid data to GM. This effectively strips padding on store.

##### 2D Diagram: GM→UB (pto.copy_gm_to_ubuf)

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
pad    = filled with pad_val to 32B boundary (data_select_bit=true)
[DATA] = valid data transferred by DMA
[PAD]  = pad_val fill (set via set_mov_pad_val)
```

##### 2D Diagram: UB→GM (pto.copy_ubuf_to_gm)

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

#### Multi-Level Loop Semantics (C Code)

The full DMA transfer is a nested loop. The HW loop registers (set before the copy) control the outer levels, and the copy instruction parameters control the innermost burst level.

##### GM→UB Full Loop

```c
// C equivalent of what the HW executes:
for (int j = 0; j < loop2_count; j++) {                // HW outer loop
    uint8_t *gm1 = gm_src + j * loop2_src_stride;
    uint8_t *ub1 = ub_dst + j * loop2_dst_stride;

    for (int k = 0; k < loop1_count; k++) {            // HW inner loop
        uint8_t *gm2 = gm1 + k * loop1_src_stride;
        uint8_t *ub2 = ub1 + k * loop1_dst_stride;

        for (int r = 0; r < n_burst; r++) {            // burst engine
            memcpy(ub2 + r * dst_stride,               //   UB dest row
                   gm2 + r * src_stride,               //   GM src row
                   len_burst);                          //   contiguous bytes
            if (data_select_bit)
                memset(ub2 + r * dst_stride + len_burst,
                       pad_val, dst_stride - len_burst);
        }
    }
}
```

##### UB→GM Full Loop

```c
// C equivalent:
for (int j = 0; j < loop2_count; j++) {
    uint8_t *ub1 = ub_src + j * loop2_src_stride;
    uint8_t *gm1 = gm_dst + j * loop2_dst_stride;

    for (int k = 0; k < loop1_count; k++) {
        uint8_t *ub2 = ub1 + k * loop1_src_stride;
        uint8_t *gm2 = gm1 + k * loop1_dst_stride;

        for (int r = 0; r < n_burst; r++) {
            memcpy(gm2 + r * dst_stride,               //   GM dest row
                   ub2 + r * src_stride,               //   UB src row
                   len_burst);                          //   contiguous bytes
        }
    }
}
```

---

#### Example 1: GM→UB — Load a 32×32 f32 Tile (Simple Case)

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
// Simple 2D load — no multi-level loops needed
pto.set_loop_size_outtoub %c1_i64, %c1_i64 : i64, i64

pto.copy_gm_to_ubuf %arg0, %ub_in,
    %c0_i64,       // sid = 0
    %c32_i64,      // n_burst = 32 (32 rows)
    %c128_i64,     // len_burst = 128 bytes per row
    %c0_i64,       // left_padding = 0
    %c0_i64,       // right_padding = 0
    %false,        // data_select_bit = false
    %c0_i64,       // l2_cache_ctl = 0
    %c128_i64,     // src_stride = 128 bytes
    %c128_i64      // dst_stride = 128 bytes
    : !pto.ptr<f32, gm>, !pto.ptr<f32, ub>, i64, i64, i64,
      i64, i64, i1, i64, i64, i64
```

---

#### Example 2: GM→UB — Load a 2D Tile from a Larger Matrix

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
    %false,        // data_select_bit = false
    %c0_i64,       // l2_cache_ctl = 0
    %c1024_i64,    // src_stride = 1024 bytes (full matrix row)
    %c256_i64      // dst_stride = 256 bytes (tile row)
    : !pto.ptr<f16, gm>, !pto.ptr<f16, ub>, i64, i64, i64,
      i64, i64, i1, i64, i64, i64
```

---

#### Example 3: GM→UB — Load with Padding

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
pto.set_loop_size_outtoub %c1_i64, %c1_i64 : i64, i64
pto.set_loop1_stride_outtoub %c0_i64, %c0_i64 : i64, i64
pto.set_loop2_stride_outtoub %c0_i64, %c0_i64 : i64, i64

pto.copy_gm_to_ubuf %gm_ptr, %ub_ptr,
    %c0_i64,       // sid = 0
    %c64_i64,      // n_burst = 64
    %c200_i64,     // len_burst = 200 bytes
    %c0_i64,       // left_padding = 0
    %c0_i64,       // right_padding = 0
    %true,         // data_select_bit = true (enable padding)
    %c0_i64,       // l2_cache_ctl = 0
    %c200_i64,     // src_stride = 200 bytes
    %c256_i64      // dst_stride = 256 bytes (32B-aligned)
    : !pto.ptr<f16, gm>, !pto.ptr<f16, ub>, i64, i64, i64,
      i64, i64, i1, i64, i64, i64
```

---

#### Example 4: UB→GM — Store a 32×32 f32 Tile (Simple Case)

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
// Configure MTE3 strides
pto.set_loop_size_ubtoout %c1_i64, %c1_i64 : i64, i64

pto.copy_ubuf_to_gm %ub_out, %arg1,
    %c0_i64,       // sid = 0
    %c32_i64,      // n_burst = 32
    %c128_i64,     // len_burst = 128 bytes
    %c0_i64,       // reserved = 0
    %c128_i64,     // dst_stride = 128 bytes
    %c128_i64      // src_stride = 128 bytes
    : !pto.ptr<f32, ub>, !pto.ptr<f32, gm>, i64, i64, i64, i64, i64, i64
```

---

#### Example 5: UB→GM — Store a 2D Tile Back to a Larger Matrix

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
// Configure MTE3 strides
pto.set_loop_size_ubtoout %c1_i64, %c1_i64 : i64, i64
pto.set_loop1_stride_ubtoout %c0_i64, %c0_i64 : i64, i64
pto.set_loop2_stride_ubtoout %c0_i64, %c0_i64 : i64, i64

pto.copy_ubuf_to_gm %ub_ptr, %gm_ptr,
    %c0_i64,       // sid = 0
    %c64_i64,      // n_burst = 64
    %c256_i64,     // len_burst = 256 bytes
    %c0_i64,       // reserved = 0
    %c1024_i64,    // dst_stride = 1024 bytes (GM row)
    %c256_i64      // src_stride = 256 bytes (UB row)
    : !pto.ptr<f16, ub>, !pto.ptr<f16, gm>, i64, i64, i64, i64, i64, i64
```

---

#### Example 6: GM→UB with Multi-Level Loop (Batch of Tiles)

Load 4 batches of 8×128 tiles from a [4, 8, 128] f16 tensor using loop1.

```
GM [4, 8, 128] f16 (contiguous):        UB (4 tiles laid out sequentially):

    batch 0: 8 rows × 256 bytes          [batch 0: 8×128][batch 1: 8×128]
    batch 1: 8 rows × 256 bytes          [batch 2: 8×128][batch 3: 8×128]
    batch 2: 8 rows × 256 bytes
    batch 3: 8 rows × 256 bytes          loop1 src_stride = 2048 bytes (8 × 256)
                                          loop1 dst_stride = 2048 bytes (8 × 256)
    Each batch = 8 × 256 = 2048 bytes     loop1_count = 4 (iterate over batches)
```

```mlir
// loop1_count = 4 batches, loop2_count = 1 (not used)
pto.set_loop_size_outtoub %c4_i64, %c1_i64 : i64, i64

// loop1 stride: advance by one batch (2048 bytes) in both GM and UB
pto.set_loop1_stride_outtoub %c2048_i64, %c2048_i64 : i64, i64
pto.set_loop2_stride_outtoub %c0_i64, %c0_i64 : i64, i64

pto.copy_gm_to_ubuf %gm_ptr, %ub_ptr,
    %c0_i64,       // sid = 0
    %c8_i64,       // n_burst = 8 rows per batch
    %c256_i64,     // len_burst = 256 bytes per row
    %c0_i64,       // left_padding = 0
    %c0_i64,       // right_padding = 0
    %false,        // data_select_bit = false
    %c0_i64,       // l2_cache_ctl = 0
    %c256_i64,     // src_stride = 256 (contiguous rows)
    %c256_i64      // dst_stride = 256 (contiguous rows)
    : !pto.ptr<f16, gm>, !pto.ptr<f16, ub>, i64, i64, i64,
      i64, i64, i1, i64, i64, i64
```

Execution trace:

```
loop1 iter 0: gm_ptr + 0×2048 → ub_ptr + 0×2048, DMA 8 rows × 256B
loop1 iter 1: gm_ptr + 1×2048 → ub_ptr + 1×2048, DMA 8 rows × 256B
loop1 iter 2: gm_ptr + 2×2048 → ub_ptr + 2×2048, DMA 8 rows × 256B
loop1 iter 3: gm_ptr + 3×2048 → ub_ptr + 3×2048, DMA 8 rows × 256B
```

---

<a id="isa-03-vector-load-store"></a>

### 3. Vector Load/Store

> **Category:** UB ↔ Vector Register data movement
> **Pipeline:** PIPE_V (Vector Core)

Vector loads move data from Unified Buffer (UB) to vector registers (`vreg`). Vector stores move data from `vreg` back to UB. All vector compute operates only on `vreg` — UB is the staging area between DMA and compute.

#### Common Operand Model

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

#### Contiguous Loads

##### `pto.vlds`

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
  fact that the source is UB memory.

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

##### `pto.vldas`

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

---

##### `pto.vldus`

- **syntax:** `%result, %align_out, %base_out = pto.vldus %source, %align : !pto.ptr<T, ub>, !pto.align -> !pto.vreg<NxT>, !pto.align, !pto.ptr<T, ub>`
- **semantics:** Unaligned load using primed align state.
- **inputs:**
  `%source` is the current UB address and `%align` is the incoming load
  alignment state primed by `pto.vldas` or a prior `pto.vldus`.
- **outputs:**
  `%result` is the assembled vector value, `%align_out` is the updated alignment
  state, and `%base_out` is the post-update base pointer state exposed in SSA
  form.
- **constraints and limitations:**
  A matching `pto.vldas` MUST appear before the first dependent `pto.vldus`
  stream in the same vector loop. Both the alignment state and the base address
  advance across the stream, and the PTO micro Instruction representation exposes those updates as SSA results.

**Unaligned load pattern:**
```mlir
%align = pto.vldas %ub : !pto.ptr<f32, ub> -> !pto.align
%vec, %align2, %ub2 = pto.vldus %ub, %align : !pto.ptr<f32, ub>, !pto.align -> !pto.vreg<64xf32>, !pto.align, !pto.ptr<f32, ub>
```

---

#### Dual Loads (Deinterleave)

##### `pto.vldx2`

- **syntax:** `%low, %high = pto.vldx2 %source[%offset], "DIST" : !pto.ptr<T, ub>, index -> !pto.vreg<NxT>, !pto.vreg<NxT>`
- **semantics:** Dual load with deinterleave (AoS → SoA conversion).
- **inputs:**
  `%source` is the UB base pointer, `%offset` is the displacement, and `DIST`
  selects a dual-load/deinterleave layout.
- **outputs:**
  `%low` and `%high` are the two destination vectors.
- **constraints and limitations:**
  This family is only legal for interleave/deinterleave style distributions.
  The two outputs form an ordered pair, and that pairing MUST be preserved.

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

#### Strided Loads

##### `pto.vsld`

- **syntax:** `%result = pto.vsld %source[%offset], "STRIDE" : !pto.ptr<T, ub> -> !pto.vreg<NxT>`
- **semantics:** Strided load with fixed stride pattern.
- **inputs:**
  `%source` is the UB base pointer and `%offset` is the displacement encoded
  with the selected fixed stride mode.
- **outputs:**
  `%result` is the loaded vector.
- **constraints and limitations:**
  This is a deprecated compatibility family. The selected stride token
  determines which sub-elements are read from each source block.

**Stride modes:** `STRIDE_S3_B16`, `STRIDE_S4_B64`, `STRIDE_S8_B32`, `STRIDE_S2_B64`

---

##### `pto.vsldb`

- **syntax:** `%result = pto.vsldb %source, %offset, %mask : !pto.ptr<T, ub>, i32, !pto.mask -> !pto.vreg<NxT>`
- **semantics:** Block-strided load for 2D tile access.
- **inputs:**
  `%source` is the UB base pointer, `%offset` is the packed stride/control word,
  and `%mask` controls which blocks participate.
- **outputs:**
  `%result` is the loaded vector.
- **constraints and limitations:**
  `%offset` is not a plain byte displacement; it encodes the block stride and
  repeat pattern. If a block is masked off, the corresponding destination block
  is zeroed and MUST NOT raise an address overflow exception for that block.

---

#### Gather (Indexed) Loads

##### `pto.vgather2`

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

```c
for (int i = 0; i < active_lanes; i++)
    dst[i] = UB[base + offsets[i] * sizeof(T)];
```

---

##### `pto.vgatherb`

- **syntax:** `%result = pto.vgatherb %source, %offsets, %active_lanes : !pto.ptr<T, ub>, !pto.vreg<NxI>, index -> !pto.vreg<NxT>`
- **semantics:** Byte-granularity indexed gather from UB.
- **inputs:**
  `%source` is the UB base pointer, `%offsets` contains per-block byte offsets,
  and `%active_lanes` bounds the number of active gathered blocks.
- **outputs:**
  `%result` is the gathered vector.
- **constraints and limitations:**
  This is a block gather, not a byte-per-lane gather. `%source` MUST be 32-byte
  aligned, each participating offset MUST describe a 32-byte-aligned block, and
  inactive blocks are zero-filled.

```c
for (int i = 0; i < active_lanes; i++)
    dst[i] = UB[base + offsets[i]];  // byte-addressed
```

---

##### `pto.vgather2_bc`

- **syntax:** `%result = pto.vgather2_bc %source, %offsets, %mask : !pto.ptr<T, ub>, !pto.vreg<NxI>, !pto.mask -> !pto.vreg<NxT>`
- **semantics:** Gather with broadcast, conditioned by mask.
- **inputs:**
  `%source` is the UB base pointer, `%offsets` contains gather indices, and
  `%mask` gates which lanes participate.
- **outputs:**
  `%result` is the gathered vector.
- **constraints and limitations:**
  This is a backward-compatible family. Masked-off lanes do not participate in
  address coalescing and do not trigger address overflow exceptions; their
  destination lanes are zero-filled.

---

#### Contiguous Stores

##### `pto.vsts`

- **syntax:** `pto.vsts %value, %dest[%offset], %mask {dist = "DIST"} : !pto.vreg<NxT>, !pto.ptr<T, ub>, !pto.mask`
- **semantics:** Vector store with distribution mode.
- **inputs:**
  `%value` is the source vector, `%dest` is the UB base pointer, `%offset` is
  the displacement, `%mask` selects the active lanes or sub-elements, and
  `DIST` selects the store distribution.
- **outputs:**
  This op has no SSA result; it writes to UB memory.
- **constraints and limitations:**
  The effective destination address MUST satisfy the alignment rule of the
  selected store mode. Narrowing/packing modes may only preserve a subset of the
  source bits. Merge-channel modes reinterpret the source vector as channel
  planes and interleave them on store.

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

#### Dual Stores (Interleave)

##### `pto.vstx2`

- **syntax:** `pto.vstx2 %low, %high, %dest[%offset], "DIST", %mask : !pto.vreg<NxT>, !pto.vreg<NxT>, !pto.ptr<T, ub>, index, !pto.mask`
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
  be preserved.

**Distribution modes:** `INTLV_B8`, `INTLV_B16`, `INTLV_B32`

```c
// INTLV_B32:
for (int i = 0; i < 64; i++) {
    UB[base + 8*i]     = low[i];
    UB[base + 8*i + 4] = high[i];
}
```

---

#### Strided Stores

##### `pto.vsst`

- **syntax:** `pto.vsst %value, %dest[%offset], "STRIDE" : !pto.vreg<NxT>, !pto.ptr<T, ub>`
- **semantics:** Strided store with fixed stride pattern.
- **inputs:**
  `%value` is the source vector, `%dest` is the UB base pointer, and `%offset`
  / `STRIDE` select the fixed strided layout.
- **outputs:**
  This op writes UB memory and returns no SSA value.
- **constraints and limitations:**
  This is a deprecated compatibility family. The stride token, not the vector
  lane number alone, determines which destination elements are written.

---

##### `pto.vsstb`

- **syntax:** `pto.vsstb %value, %dest, %offset, %mask : !pto.vreg<NxT>, !pto.ptr<T, ub>, i32, !pto.mask`
- **semantics:** Block-strided store for 2D tile access.
- **inputs:**
  `%value` is the source vector, `%dest` is the UB base pointer, `%offset` is
  the packed stride/control word, and `%mask` controls block participation.
- **outputs:**
  This op writes UB memory and returns no SSA value.
- **constraints and limitations:**
  `%offset` is a control word, not a plain byte displacement. This is a
  deprecated compatibility family kept for surface coverage.

---

#### Scatter (Indexed) Stores

##### `pto.vscatter`

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

```c
for (int i = 0; i < active_lanes; i++)
    UB[base + offsets[i] * sizeof(T)] = src[i];
```

---

#### Alignment State Stores

##### `pto.vsta`

- **syntax:** `pto.vsta %value, %dest[%offset] : !pto.align, !pto.ptr<T, ub>, index`
- **semantics:** Flush alignment state to memory.
- **inputs:**
  `%value` is the pending store-alignment state, `%dest` is the UB base pointer,
  and `%offset` is the flush displacement.
- **outputs:**
  This op writes buffered tail bytes to UB and returns no SSA value.
- **constraints and limitations:**
  The flush address MUST match the post-updated address expected by the
  preceding unaligned-store stream. After the flush, the corresponding store
  alignment state is consumed.

---

##### `pto.vstas`
- **syntax:** `pto.vstas %value, %dest, %offset : !pto.align, !pto.ptr<T, ub>, i32`
- **semantics:** Scalar-register-offset form of alignment-state flush.
- **inputs:**
  `%value` is the pending store-alignment state, `%dest` is the UB base
  pointer, and `%offset` is the scalar-register style displacement.
- **outputs:**
  This op writes buffered tail bytes to UB and returns no SSA value.
- **constraints and limitations:**
  This family uses the same buffered-tail semantics as `pto.vsta` but keeps the
  scalar-offset form explicit.

---

##### `pto.vstar`
- **syntax:** `pto.vstar %value, %dest : !pto.align, !pto.ptr<T, ub>`
- **semantics:** Flush alignment state using the register-update form.
- **inputs:**
  `%value` is the pending store-alignment state and `%dest` is the UB base
  pointer.
- **outputs:**
  This op writes buffered tail bytes to UB and returns no SSA value.
- **constraints and limitations:**
  The implicit update state consumed by this flush MUST correspond to the same
  store stream that produced `%value`.

---

##### `pto.vstu`
- **syntax:** `%align_out, %base_out = pto.vstu %align_in, %base_in, %value, %dest, %mode : !pto.align, !pto.ptr<T, ub>, !pto.vreg<NxT>, !pto.ptr<T, ub>, index -> !pto.align, !pto.ptr<T, ub>`
- **semantics:** Unaligned store with explicit threaded alignment/base state.
- **inputs:**
  `%align_in` is the incoming store-alignment state, `%base_in` is the current
  stream base, `%value` is the vector to store, `%dest` is the UB base pointer,
  and `%mode` selects the post-update behavior.
- **outputs:**
  `%align_out` is the updated buffered-tail state and `%base_out` is the
  post-update base pointer state.
- **constraints and limitations:**
  This op models a stateful unaligned-store sequence in SSA form. A final
  `pto.vsta` / `pto.vstas` / `pto.vstar` is still required to flush the trailing
  buffered bytes.

---

##### `pto.vstus`
- **syntax:** `%align_out, %base_out = pto.vstus %align_in, %base_in, %value, %dest, %offset : !pto.align, !pto.ptr<T, ub>, !pto.vreg<NxT>, !pto.ptr<T, ub>, i32 -> !pto.align, !pto.ptr<T, ub>`
- **semantics:** Scalar-offset unaligned store with threaded state.
- **inputs:**
  Same roles as `pto.vstu`, but `%offset` is provided explicitly as the scalar
  displacement.
- **outputs:**
  Updated alignment state and base state.
- **constraints and limitations:**
  The same final flush requirement and state-threading constraints as
  `pto.vstu` apply.

---

##### `pto.vstur`
- **syntax:** `%align_out = pto.vstur %align_in, %value, %dest : !pto.align, !pto.vreg<NxT>, !pto.ptr<T, ub> -> !pto.align`
- **semantics:** Register-update unaligned store form.
- **inputs:**
  `%align_in` is the incoming store-alignment state, `%value` is the vector to
  store, and `%dest` is the UB base pointer.
- **outputs:**
  `%align_out` is the updated buffered-tail state.
- **constraints and limitations:**
  This op updates only the residual alignment state. A matching flush op is
  still required to emit the trailing bytes.

- **syntax:** `pto.vstas %value, %dest, %offset : !pto.align, !pto.ptr<T, ub>, i32`
- **semantics:** Flush alignment state with scalar offset.

---

##### `pto.vstar`

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

---

#### Stateful Store Ops

These ops make reference-updated state explicit as SSA results.

##### `pto.vstu`

- **syntax:** `%align_out, %offset_out = pto.vstu %align_in, %offset_in, %value, %base, "MODE" : !pto.align, index, !pto.vreg<NxT>, !pto.ptr<T, ub> -> !pto.align, index`
- **semantics:** Unaligned store with align + offset state update.
- **inputs:**
  `%align_in` is the incoming store-alignment state, `%offset_in` is the current
  logical byte/element displacement, `%value` is the vector being stored, and
  `%base` is the UB base pointer.
- **outputs:**
  `%align_out` is the updated alignment/tail state and `%offset_out` is the
  next offset after applying the selected post-update rule.
- **constraints and limitations:**
  The alignment state MUST be threaded in program order. A terminating flush
  form such as `pto.vstar`/`pto.vstas` is still required to commit the buffered
  tail bytes.

**Mode tokens:** `POST_UPDATE`, `NO_POST_UPDATE`

---

##### `pto.vstus`

- **syntax:** `%align_out, %base_out = pto.vstus %align_in, %offset, %value, %base, "MODE" : !pto.align, i32, !pto.vreg<NxT>, !pto.ptr<T, ub> -> !pto.align, !pto.ptr<T, ub>`
- **semantics:** Unaligned store with scalar offset and state update.
- **inputs:**
  `%align_in` is the incoming store-alignment state, `%offset` is the scalar
  displacement, `%value` is the vector being stored, and `%base` is the UB base
  pointer.
- **outputs:**
  `%align_out` is the updated buffered-tail state and `%base_out` is the next
  base pointer when the lowering chooses a post-update form.
- **constraints and limitations:**
  This is the scalar-offset stateful form of the unaligned store family. The
  scalar offset width and update mode MUST match the selected form, and a later
  flush op is still required.

---

##### `pto.vstur`

- **syntax:** `%align_out = pto.vstur %align_in, %value, %base, "MODE" : !pto.align, !pto.vreg<NxT>, !pto.ptr<T, ub> -> !pto.align`
- **semantics:** Unaligned store with residual flush and state update.
- **inputs:**
  `%align_in` is the incoming store-alignment state, `%value` is the vector to
  store, and `%base` is the UB base pointer.
- **outputs:**
  `%align_out` is the updated residual state after the current partial store.
- **constraints and limitations:**
  This form exposes only the evolving state; it does not by itself guarantee
  that all buffered bytes have reached memory. A compatible final flush is still
  required unless the surrounding sequence is known to be complete.

---

<a id="isa-04-predicate-load-store"></a>

### 4. Predicate Load/Store

> **Category:** UB ↔ Predicate Register data movement
> **Pipeline:** PIPE_V (Vector Core)

Predicate registers (`!pto.mask`) are 256-bit registers that enable per-lane conditional execution. These ops move predicate values between UB and predicate registers.

---

#### Predicate Loads

##### `pto.plds`

- **syntax:** `%result = pto.plds %source[%offset] {dist = "DIST"} : !pto.ptr<T, ub> -> !pto.mask`
- **semantics:** Load predicate register with scalar offset.

**Distribution modes:** `NORM`, `US`, `DS`

**Example:**
```mlir
%mask = pto.plds %ub[%c0] {dist = "NORM"} : !pto.ptr<T, ub> -> !pto.mask
```

---

##### `pto.pld`

- **syntax:** `%result = pto.pld %source[%offset], "DIST" : !pto.ptr<T, ub>, index -> !pto.mask`
- **semantics:** Load predicate register with areg offset.

---

##### `pto.pldi`

- **syntax:** `%result = pto.pldi %source, %offset, "DIST" : !pto.ptr<T, ub>, i32 -> !pto.mask`
- **semantics:** Load predicate register with immediate offset.

---

#### Predicate Stores

##### `pto.psts`

- **syntax:** `pto.psts %value, %dest[%offset] : !pto.mask, !pto.ptr<T, ub>`
- **semantics:** Store predicate register with scalar offset.

**Example:**
```mlir
pto.psts %mask, %ub[%c0] : !pto.mask, !pto.ptr<T, ub>
```

---

##### `pto.pst`

- **syntax:** `pto.pst %value, %dest[%offset], "DIST" : !pto.mask, !pto.ptr<T, ub>, index`
- **semantics:** Store predicate register with areg offset.

**Distribution modes:** `NORM`, `PK`

---

##### `pto.psti`

- **syntax:** `pto.psti %value, %dest, %offset, "DIST" : !pto.mask, !pto.ptr<T, ub>, i32`
- **semantics:** Store predicate register with immediate offset.

---

##### `pto.pstu`

- **syntax:** `%align_out, %base_out = pto.pstu %align_in, %value, %base : !pto.align, !pto.mask, !pto.ptr<T, ub> -> !pto.align, !pto.ptr<T, ub>`
- **semantics:** Predicate unaligned store with align state update.

---

#### Typical Usage Pattern

```mlir
// Generate comparison mask
%mask = pto.vcmp %v0, %v1, %seed, "lt" : !pto.vreg<64xf32>, !pto.vreg<64xf32>, !pto.mask -> !pto.mask

// Store mask to UB for later use
pto.psts %mask, %ub_mask[%c0] : !pto.mask, !pto.ptr<T, ub>

// ... later in another kernel ...

// Load mask from UB
%saved_mask = pto.plds %ub_mask[%c0] {dist = "NORM"} : !pto.ptr<T, ub> -> !pto.mask

// Use for predicated select
%result = pto.vsel %v_true, %v_false, %saved_mask : !pto.vreg<64xf32>, !pto.vreg<64xf32>, !pto.mask -> !pto.vreg<64xf32>
```

---

<a id="isa-05-materialization-predicate"></a>

### 5. Materialization & Predicate Ops

> **Category:** Scalar broadcast, predicate generation and manipulation
> **Pipeline:** PIPE_V (Vector Core)

These ops create vectors from scalar values and manipulate predicate registers.

#### Common Operand Model

- `%value` is the scalar source value in SSA form.
- `%input` is either a source scalar or a source vector depending on the op.
- `%result` is the destination vector register value.
- For 32-bit scalar inputs, the scalar source MUST satisfy the backend's legal
  scalar-source constraints for this family.

---

#### Scalar Materialization

##### `pto.vbr`

- **syntax:** `%result = pto.vbr %value : T -> !pto.vreg<NxT>`
- **semantics:** Broadcast scalar to all vector lanes.
- **inputs:**
  `%value` is the scalar source.
- **outputs:**
  `%result` is a vector whose active lanes all carry `%value`.
- **constraints and limitations:**
  Supported forms are `b8`, `b16`, and `b32`. For `b8`, only the low 8 bits of
  the scalar source are consumed.

```c
for (int i = 0; i < N; i++)
    dst[i] = value;
```

**Example:**
```mlir
%one = pto.vbr %c1_f32 : f32 -> !pto.vreg<64xf32>
```

---

##### `pto.vdup`

- **syntax:** `%result = pto.vdup %input {position = "POSITION"} : T|!pto.vreg<NxT> -> !pto.vreg<NxT>`
- **semantics:** Duplicate scalar or vector element to all lanes.
- **inputs:**
  `%input` supplies the scalar or source-lane value selected by `position`.
- **outputs:**
  `%result` is the duplicated vector.
- **constraints and limitations:**
  `position` selects which source element or scalar position is duplicated. The
  current PTO micro Instruction representation models that selector as an attribute rather than a
  separate operand.

```c
for (int i = 0; i < N; i++)
    dst[i] = input_scalar_or_element;
```

---

#### Predicate Generation

##### `pto.pset_b8` / `pto.pset_b16` / `pto.pset_b32`

- **syntax:** `%result = pto.pset_b32 "PAT_*" : !pto.mask`
- **semantics:** Generate predicate from pattern.

**Patterns:**

| Pattern | Description |
|---------|-------------|
| `PAT_ALL` | All lanes active |
| `PAT_ALLF` | All lanes inactive |
| `PAT_H` | High half active |
| `PAT_Q` | Upper quarter active |
| `PAT_VL1`...`PAT_VL128` | First N lanes active |
| `PAT_M3`, `PAT_M4` | Modular patterns |

**Example — All 64 f32 lanes active:**
```mlir
%all_active = pto.pset_b32 "PAT_ALL" : !pto.mask
```

**Example — First 16 lanes active:**
```mlir
%first_16 = pto.pset_b32 "PAT_VL16" : !pto.mask
```

---

##### `pto.pge_b8` / `pto.pge_b16` / `pto.pge_b32`

- **syntax:** `%result = pto.pge_b32 "PAT_*" : !pto.mask`
- **semantics:** Generate tail mask — first N lanes active.

```c
for (int i = 0; i < TOTAL_LANES; i++)
    mask[i] = (i < len);
```

**Example — Tail mask for remainder loop:**
```mlir
%tail_mask = pto.pge_b32 "PAT_VL8" : !pto.mask

---

##### `pto.plt_b8` / `pto.plt_b16` / `pto.plt_b32`

- **syntax:** `%mask, %scalar_out = pto.plt_b32 %scalar : i32 -> !pto.mask, i32`
- **semantics:** Generate predicate state together with updated scalar state.
```

---

#### Predicate Pack/Unpack

##### `pto.ppack`

- **syntax:** `%result = pto.ppack %input, "PART" : !pto.mask -> !pto.mask`
- **semantics:** Narrowing pack of predicate register.

**Part tokens:** `LOWER`, `HIGHER`

---

##### `pto.punpack`

- **syntax:** `%result = pto.punpack %input, "PART" : !pto.mask -> !pto.mask`
- **semantics:** Widening unpack of predicate register.

---

#### Predicate Logical Ops

##### `pto.pand`

- **syntax:** `%result = pto.pand %src0, %src1, %mask : !pto.mask, !pto.mask, !pto.mask -> !pto.mask`
- **semantics:** Predicate bitwise AND.

```c
for (int i = 0; i < N; i++)
    if (mask[i]) dst[i] = src0[i] & src1[i];
```

---

##### `pto.por`

- **syntax:** `%result = pto.por %src0, %src1, %mask : !pto.mask, !pto.mask, !pto.mask -> !pto.mask`
- **semantics:** Predicate bitwise OR.

```c
for (int i = 0; i < N; i++)
    if (mask[i]) dst[i] = src0[i] | src1[i];
```

---

##### `pto.pxor`

- **syntax:** `%result = pto.pxor %src0, %src1, %mask : !pto.mask, !pto.mask, !pto.mask -> !pto.mask`
- **semantics:** Predicate bitwise XOR.

```c
for (int i = 0; i < N; i++)
    if (mask[i]) dst[i] = src0[i] ^ src1[i];
```

---

##### `pto.pnot`

- **syntax:** `%result = pto.pnot %input, %mask : !pto.mask, !pto.mask -> !pto.mask`
- **semantics:** Predicate bitwise NOT.

```c
for (int i = 0; i < N; i++)
    if (mask[i]) dst[i] = ~src[i];
```

---

##### `pto.psel`

- **syntax:** `%result = pto.psel %src0, %src1, %sel : !pto.mask, !pto.mask, !pto.mask -> !pto.mask`
- **semantics:** Predicate select (mux).

```c
for (int i = 0; i < N; i++)
    dst[i] = sel[i] ? src0[i] : src1[i];
```

---

#### Predicate Movement

##### `pto.ppack`

- **syntax:** `%result = pto.ppack %input, "PART" : !pto.mask -> !pto.mask`
- **semantics:** Narrowing pack of predicate register.

```c
for (int i = 0; i < N; i++)
    if (mask[i]) dst[i] = src[i];
```

---

##### `pto.punpack`

- **syntax:** `%result = pto.punpack %input, "PART" : !pto.mask -> !pto.mask`
- **semantics:** Widening unpack of predicate register.

---

##### `pto.pdintlv_b8`

- **syntax:** `%low, %high = pto.pdintlv_b8 %src0, %src1 : !pto.mask, !pto.mask -> !pto.mask, !pto.mask`
- **semantics:** Predicate deinterleave.

---

##### `pto.pintlv_b16`

- **syntax:** `%low, %high = pto.pintlv_b16 %src0, %src1 : !pto.mask, !pto.mask -> !pto.mask, !pto.mask`
- **semantics:** Predicate interleave.

---

#### Typical Usage

```mlir
// Generate all-active mask for f32 (64 lanes)
%all = pto.pset_b32 "PAT_ALL" : !pto.mask

// Generate tail mask for remainder (last 12 elements)
%tail = pto.pge_b32 "PAT_VL12" : !pto.mask

// Compare and generate mask
%cmp_mask = pto.vcmp %a, %b, %all, "lt" : !pto.vreg<64xf32>, !pto.vreg<64xf32>, !pto.mask -> !pto.mask

// Combine masks: only process tail elements that passed comparison
%combined = pto.pand %cmp_mask, %tail, %all : !pto.mask, !pto.mask, !pto.mask -> !pto.mask

// Use for predicated operation
%result = pto.vsel %true_vals, %false_vals, %combined : !pto.vreg<64xf32>, !pto.vreg<64xf32>, !pto.mask -> !pto.vreg<64xf32>
```

---

<a id="isa-06-unary-vector-ops"></a>

### 6. Unary Vector Ops

> **Category:** Single-input vector operations
> **Pipeline:** PIPE_V (Vector Core)

Element-wise operations that take one vector input and produce one vector output.

#### Common Operand Model

- `%input` is the source vector register value.
- `%mask` is the predicate operand. For this family, inactive lanes follow the
  predication behavior of the selected instruction form: zeroing forms
  zero-fill inactive lanes, while merging forms preserve the destination value.
- `%result` is the destination vector register value. Unless stated otherwise,
  `%result` has the same lane count and element type as `%input`.

---

#### Arithmetic

##### `pto.vabs`

- **syntax:** `%result = pto.vabs %input, %mask : !pto.vreg<NxT>, !pto.mask -> !pto.vreg<NxT>`
- **A5 types:** i8-i32, f16, f32

```c
for (int i = 0; i < N; i++)
    dst[i] = (src[i] < 0) ? -src[i] : src[i];
```

- **inputs:** `%input` supplies the source lanes and `%mask` selects which lanes
  participate.
- **outputs:** `%result` receives the lane-wise absolute values.
- **constraints and limitations:** Source and result types MUST match. Integer
  overflow on the most-negative signed value follows the target-defined
  behavior.

---

##### `pto.vneg`

- **syntax:** `%result = pto.vneg %input, %mask : !pto.vreg<NxT>, !pto.mask -> !pto.vreg<NxT>`
- **A5 types:** i8-i32, f16, f32

```c
for (int i = 0; i < N; i++)
    dst[i] = -src[i];
```

- **inputs:** `%input` is the source vector and `%mask` selects active lanes.
- **outputs:** `%result` is the lane-wise arithmetic negation.
- **constraints and limitations:** Source and result types MUST match.

---

#### Transcendental

##### `pto.vexp`

- **syntax:** `%result = pto.vexp %input, %mask : !pto.vreg<NxT>, !pto.mask -> !pto.vreg<NxT>`
- **A5 types:** f16, f32

```c
for (int i = 0; i < N; i++)
    dst[i] = expf(src[i]);
```

- **inputs:** `%input` is the source vector and `%mask` selects active lanes.
- **outputs:** `%result` holds `exp(input[i])` per active lane.
- **constraints and limitations:** Only floating-point element types are legal.

---

##### `pto.vln`

- **syntax:** `%result = pto.vln %input, %mask : !pto.vreg<NxT>, !pto.mask -> !pto.vreg<NxT>`
- **A5 types:** f16, f32

```c
for (int i = 0; i < N; i++)
    dst[i] = logf(src[i]);
```

- **inputs:** `%input` is the source vector and `%mask` selects active lanes.
- **outputs:** `%result` holds the natural logarithm per active lane.
- **constraints and limitations:** Only floating-point element types are legal.
  For real-number semantics, active inputs SHOULD be strictly positive; non-
  positive inputs follow the target's exception/NaN rules.

---

##### `pto.vsqrt`

- **syntax:** `%result = pto.vsqrt %input, %mask : !pto.vreg<NxT>, !pto.mask -> !pto.vreg<NxT>`
- **A5 types:** f16, f32

```c
for (int i = 0; i < N; i++)
    dst[i] = sqrtf(src[i]);
```

- **inputs:** `%input` is the source vector and `%mask` selects active lanes.
- **outputs:** `%result` holds the square root per active lane.
- **constraints and limitations:** Only floating-point element types are legal.
  Negative active inputs follow the target's exception/NaN rules.

---

##### `pto.vrsqrt`

- **syntax:** `%result = pto.vrsqrt %input, %mask : !pto.vreg<NxT>, !pto.mask -> !pto.vreg<NxT>`
- **A5 types:** f16, f32

```c
for (int i = 0; i < N; i++)
    dst[i] = 1.0f / sqrtf(src[i]);
```

- **inputs:** `%input` is the source vector and `%mask` selects active lanes.
- **outputs:** `%result` holds reciprocal-square-root values per active lane.
- **constraints and limitations:** Only floating-point element types are legal.
  Active inputs containing `+0` or `-0` follow the target's divide-style
  exceptional behavior.

---

##### `pto.vrec`

- **syntax:** `%result = pto.vrec %input, %mask : !pto.vreg<NxT>, !pto.mask -> !pto.vreg<NxT>`
- **A5 types:** f16, f32

```c
for (int i = 0; i < N; i++)
    dst[i] = 1.0f / src[i];
```

- **inputs:** `%input` is the source vector and `%mask` selects active lanes.
- **outputs:** `%result` holds the reciprocal per active lane.
- **constraints and limitations:** Only floating-point element types are legal.
  Active inputs containing `+0` or `-0` follow the target's divide-style
  exceptional behavior.

---

#### Activation

##### `pto.vrelu`

- **syntax:** `%result = pto.vrelu %input, %mask : !pto.vreg<NxT>, !pto.mask -> !pto.vreg<NxT>`
- **A5 types:** f16, f32

```c
for (int i = 0; i < N; i++)
    dst[i] = (src[i] > 0) ? src[i] : 0;
```

- **inputs:** `%input` is the source vector and `%mask` selects active lanes.
- **outputs:** `%result` holds `max(input[i], 0)` per active lane.
- **constraints and limitations:** Only floating-point element types are legal
  on the current A5 surface described here.

---

#### Bitwise

##### `pto.vnot`

- **syntax:** `%result = pto.vnot %input, %mask : !pto.vreg<NxT>, !pto.mask -> !pto.vreg<NxT>`
- **A5 types:** all integer types

```c
for (int i = 0; i < N; i++)
    dst[i] = ~src[i];
```

- **inputs:** `%input` is the source vector and `%mask` selects active lanes.
- **outputs:** `%result` holds the lane-wise bitwise inversion.
- **constraints and limitations:** Integer element types only.

---

##### `pto.vbcnt`

- **syntax:** `%result = pto.vbcnt %input, %mask : !pto.vreg<NxT>, !pto.mask -> !pto.vreg<NxT>`
- **A5 types:** all integer types

```c
for (int i = 0; i < N; i++)
    dst[i] = __builtin_popcount(src[i]);
```

- **inputs:** `%input` is the source vector and `%mask` selects active lanes.
- **outputs:** `%result` holds the population count for each active lane.
- **constraints and limitations:** Integer element types only. The count is
  over the source element width, not over the full vector register.

---

##### `pto.vcls`

- **syntax:** `%result = pto.vcls %input, %mask : !pto.vreg<NxT>, !pto.mask -> !pto.vreg<NxT>`
- **A5 types:** all integer types

```c
for (int i = 0; i < N; i++)
    dst[i] = count_leading_sign_bits(src[i]);
```

- **inputs:** `%input` is the source vector and `%mask` selects active lanes.
- **outputs:** `%result` holds the leading-sign-bit count per active lane.
- **constraints and limitations:** Integer element types only. This operation is
  sign-aware, so signed interpretation matters.

---

#### Movement

##### `pto.vmov`

- **syntax:** `%result = pto.vmov %input, %mask : !pto.vreg<NxT>, !pto.mask -> !pto.vreg<NxT>`
- **semantics:** Vector register copy.

```c
for (int i = 0; i < N; i++)
    dst[i] = src[i];
```

- **inputs:** `%input` is the source vector and `%mask` selects active lanes.
- **outputs:** `%result` is a copy of the source vector.
- **constraints and limitations:** Predicated `pto.vmov` behaves like a masked
  copy, while the unpredicated form behaves like a full-register copy.

---

#### Typical Usage

```mlir
// Softmax numerator: exp(x - max)
%sub = pto.vsub %x, %max_broadcast, %mask : !pto.vreg<64xf32>, !pto.vreg<64xf32>, !pto.mask -> !pto.vreg<64xf32>
%exp = pto.vexp %sub, %mask : !pto.vreg<64xf32>, !pto.mask -> !pto.vreg<64xf32>

// Reciprocal for division
%sum_rcp = pto.vrec %sum, %mask : !pto.vreg<64xf32>, !pto.mask -> !pto.vreg<64xf32>

// ReLU activation
%activated = pto.vrelu %linear_out, %mask : !pto.vreg<64xf32>, !pto.mask -> !pto.vreg<64xf32>
```

---

<a id="isa-07-binary-vector-ops"></a>

### 7. Binary Vector Ops

> **Category:** Two-input vector operations
> **Pipeline:** PIPE_V (Vector Core)

Element-wise operations that take two vector inputs and produce one vector output.

#### Common Operand Model

- `%lhs` and `%rhs` are the two source vector register values.
- `%mask` is the predicate operand `Pg` that gates which lanes participate.
- `%result` is the destination vector register value. Unless explicitly noted,
  it has the same lane count and element type as the inputs.
- Unless explicitly documented otherwise, `%lhs`, `%rhs`, and `%result` MUST
  have matching vector shapes and element types.

---

#### Arithmetic

##### `pto.vadd`

- **syntax:** `%result = pto.vadd %lhs, %rhs, %mask : !pto.vreg<NxT>, !pto.vreg<NxT>, !pto.mask -> !pto.vreg<NxT>`
- **A5 types:** i8-i64, f16, bf16, f32

```c
for (int i = 0; i < N; i++)
    dst[i] = src0[i] + src1[i];
```

- **inputs:** `%lhs` and `%rhs` are added lane-wise; `%mask` selects active
  lanes.
- **outputs:** `%result` is the lane-wise sum.
- **constraints and limitations:** Input and result types MUST match.

---

##### `pto.vsub`

- **syntax:** `%result = pto.vsub %lhs, %rhs, %mask : !pto.vreg<NxT>, !pto.vreg<NxT>, !pto.mask -> !pto.vreg<NxT>`
- **A5 types:** i8-i64, f16, bf16, f32

```c
for (int i = 0; i < N; i++)
    dst[i] = src0[i] - src1[i];
```

- **inputs:** `%lhs` is the minuend, `%rhs` is the subtrahend, and `%mask`
  selects active lanes.
- **outputs:** `%result` is the lane-wise difference.
- **constraints and limitations:** Input and result types MUST match.

---

##### `pto.vmul`

- **syntax:** `%result = pto.vmul %lhs, %rhs, %mask : !pto.vreg<NxT>, !pto.vreg<NxT>, !pto.mask -> !pto.vreg<NxT>`
- **A5 types:** i16-i32, f16, bf16, f32 (**NOT** i8/u8)

```c
for (int i = 0; i < N; i++)
    dst[i] = src0[i] * src1[i];
```

- **inputs:** `%lhs` and `%rhs` are multiplied lane-wise; `%mask` selects
  active lanes.
- **outputs:** `%result` is the lane-wise product.
- **constraints and limitations:** The current A5 profile excludes `i8/u8`
  forms from this surface.

---

##### `pto.vdiv`

- **syntax:** `%result = pto.vdiv %lhs, %rhs, %mask : !pto.vreg<NxT>, !pto.vreg<NxT>, !pto.mask -> !pto.vreg<NxT>`
- **A5 types:** f16, f32 only (no integer division)

```c
for (int i = 0; i < N; i++)
    dst[i] = src0[i] / src1[i];
```

- **inputs:** `%lhs` is the numerator, `%rhs` is the denominator, and `%mask`
  selects active lanes.
- **outputs:** `%result` is the lane-wise quotient.
- **constraints and limitations:** Floating-point element types only. Active
  denominators containing `+0` or `-0` follow the target's exceptional
  behavior.

---

##### `pto.vmax`

- **syntax:** `%result = pto.vmax %lhs, %rhs, %mask : !pto.vreg<NxT>, !pto.vreg<NxT>, !pto.mask -> !pto.vreg<NxT>`
- **A5 types:** i8-i32, f16, bf16, f32

```c
for (int i = 0; i < N; i++)
    dst[i] = (src0[i] > src1[i]) ? src0[i] : src1[i];
```

- **inputs:** `%lhs`, `%rhs`, and `%mask` as above.
- **outputs:** `%result` holds the lane-wise maximum.
- **constraints and limitations:** Input and result types MUST match.

---

##### `pto.vmin`

- **syntax:** `%result = pto.vmin %lhs, %rhs, %mask : !pto.vreg<NxT>, !pto.vreg<NxT>, !pto.mask -> !pto.vreg<NxT>`
- **A5 types:** i8-i32, f16, bf16, f32

```c
for (int i = 0; i < N; i++)
    dst[i] = (src0[i] < src1[i]) ? src0[i] : src1[i];
```

- **inputs:** `%lhs`, `%rhs`, and `%mask` as above.
- **outputs:** `%result` holds the lane-wise minimum.
- **constraints and limitations:** Input and result types MUST match.

---

#### Bitwise

##### `pto.vand`

- **syntax:** `%result = pto.vand %lhs, %rhs, %mask : !pto.vreg<NxT>, !pto.vreg<NxT>, !pto.mask -> !pto.vreg<NxT>`
- **A5 types:** all integer types

```c
for (int i = 0; i < N; i++)
    dst[i] = src0[i] & src1[i];
```

- **inputs:** `%lhs`, `%rhs`, and `%mask` as above.
- **outputs:** `%result` is the lane-wise bitwise AND.
- **constraints and limitations:** Integer element types only.

---

##### `pto.vor`

- **syntax:** `%result = pto.vor %lhs, %rhs, %mask : !pto.vreg<NxT>, !pto.vreg<NxT>, !pto.mask -> !pto.vreg<NxT>`
- **A5 types:** all integer types

```c
for (int i = 0; i < N; i++)
    dst[i] = src0[i] | src1[i];
```

- **inputs:** `%lhs`, `%rhs`, and `%mask` as above.
- **outputs:** `%result` is the lane-wise bitwise OR.
- **constraints and limitations:** Integer element types only.

---

##### `pto.vxor`

- **syntax:** `%result = pto.vxor %lhs, %rhs, %mask : !pto.vreg<NxT>, !pto.vreg<NxT>, !pto.mask -> !pto.vreg<NxT>`
- **A5 types:** all integer types

```c
for (int i = 0; i < N; i++)
    dst[i] = src0[i] ^ src1[i];
```

- **inputs:** `%lhs`, `%rhs`, and `%mask` as above.
- **outputs:** `%result` is the lane-wise bitwise XOR.
- **constraints and limitations:** Integer element types only.

---

#### Shift

##### `pto.vshl`

- **syntax:** `%result = pto.vshl %lhs, %rhs, %mask : !pto.vreg<NxT>, !pto.vreg<NxT>, !pto.mask -> !pto.vreg<NxT>`
- **A5 types:** all integer types

```c
for (int i = 0; i < N; i++)
    dst[i] = src0[i] << src1[i];
```

- **inputs:** `%lhs` supplies the shifted value, `%rhs` supplies the per-lane
  shift amount, and `%mask` selects active lanes.
- **outputs:** `%result` is the shifted vector.
- **constraints and limitations:** Integer element types only. Shift counts
  SHOULD stay within `[0, bitwidth(T) - 1]`; out-of-range behavior is target-
  defined unless the verifier narrows it further.

---

##### `pto.vshr`

- **syntax:** `%result = pto.vshr %lhs, %rhs, %mask : !pto.vreg<NxT>, !pto.vreg<NxT>, !pto.mask -> !pto.vreg<NxT>`
- **A5 types:** all integer types

```c
for (int i = 0; i < N; i++)
    dst[i] = src0[i] >> src1[i];  // arithmetic for signed, logical for unsigned
```

- **inputs:** `%lhs` supplies the shifted value, `%rhs` supplies the per-lane
  shift amount, and `%mask` selects active lanes.
- **outputs:** `%result` is the shifted vector.
- **constraints and limitations:** Integer element types only. Signedness of the
  element type determines arithmetic vs logical behavior.

---

#### Carry Operations

##### `pto.vaddc`

- **syntax:** `%result, %carry = pto.vaddc %lhs, %rhs, %mask : !pto.vreg<NxT>, !pto.vreg<NxT>, !pto.mask -> !pto.vreg<NxT>, !pto.mask`
- **semantics:** Add with carry output.

```c
for (int i = 0; i < N; i++) {
    uint64_t r = (uint64_t)src0[i] + src1[i];
    dst[i] = (T)r;
    carry[i] = (r >> bitwidth);
}
```

- **inputs:** `%lhs` and `%rhs` are added lane-wise and `%mask` selects active
  lanes.
- **outputs:** `%result` is the truncated arithmetic result and `%carry` is the
  carry/overflow predicate per lane.
- **constraints and limitations:** This is a carry-chain integer add family. On
  the current A5 surface, it SHOULD be treated as an unsigned integer
  operation.

---

##### `pto.vsubc`

- **syntax:** `%result, %borrow = pto.vsubc %lhs, %rhs, %mask : !pto.vreg<NxT>, !pto.vreg<NxT>, !pto.mask -> !pto.vreg<NxT>, !pto.mask`
- **semantics:** Subtract with borrow output.

```c
for (int i = 0; i < N; i++) {
    dst[i] = src0[i] - src1[i];
    borrow[i] = (src0[i] < src1[i]);
}
```

- **inputs:** `%lhs` and `%rhs` are subtracted lane-wise and `%mask` selects
  active lanes.
- **outputs:** `%result` is the arithmetic difference and `%borrow` marks lanes
  that borrowed.
- **constraints and limitations:** This operation SHOULD be treated as an
  unsigned 32-bit carry-chain family unless and until the verifier states
  otherwise.

---

#### Typical Usage

```mlir
// Vector addition
%sum = pto.vadd %a, %b, %mask : !pto.vreg<64xf32>, !pto.vreg<64xf32>, !pto.mask -> !pto.vreg<64xf32>

// Element-wise multiply
%prod = pto.vmul %x, %y, %mask : !pto.vreg<64xf32>, !pto.vreg<64xf32>, !pto.mask -> !pto.vreg<64xf32>

// Clamp to range [min, max]
%clamped_low = pto.vmax %input, %min_vec, %mask : !pto.vreg<64xf32>, !pto.vreg<64xf32>, !pto.mask -> !pto.vreg<64xf32>
%clamped = pto.vmin %clamped_low, %max_vec, %mask : !pto.vreg<64xf32>, !pto.vreg<64xf32>, !pto.mask -> !pto.vreg<64xf32>

// Bit manipulation
%masked = pto.vand %data, %bitmask, %mask : !pto.vreg<64xi32>, !pto.vreg<64xi32>, !pto.mask -> !pto.vreg<64xi32>
```

---

<a id="isa-08-vec-scalar-ops"></a>

### 8. Vec-Scalar Ops

> **Category:** Vector-scalar operations
> **Pipeline:** PIPE_V (Vector Core)

Operations that combine a vector with a scalar value, applying the scalar to every lane.

#### Common Operand Model

- `%input` is the source vector register value.
- `%scalar` is the scalar operand in SSA form.
- `%mask` is the predicate operand.
- `%result` is the destination vector register value.
- For 32-bit scalar forms, the scalar source MUST satisfy the backend's legal
  scalar-source constraints for this family.

---

#### Arithmetic

##### `pto.vadds`

- **syntax:** `%result = pto.vadds %input, %scalar, %mask : !pto.vreg<NxT>, T, !pto.mask -> !pto.vreg<NxT>`

```c
for (int i = 0; i < N; i++)
    dst[i] = src[i] + scalar;
```

- **inputs:** `%input` is the source vector, `%scalar` is broadcast logically to
  each active lane, and `%mask` selects active lanes.
- **outputs:** `%result` is the lane-wise sum.
- **constraints and limitations:** Inactive lanes follow the predication
  behavior defined for this family. On the current surface, inactive lanes are
  treated as zeroing lanes.

---

##### `pto.vsubs`

- **syntax:** `%result = pto.vsubs %input, %scalar, %mask : !pto.vreg<NxT>, T, !pto.mask -> !pto.vreg<NxT>`

```c
for (int i = 0; i < N; i++)
    dst[i] = src[i] - scalar;
```

- **inputs:** `%input`, `%scalar`, and `%mask` as above.
- **outputs:** `%result` is the lane-wise difference.
- **constraints and limitations:** Integer or floating-point legality depends on
  the selected type family in lowering.

---

##### `pto.vmuls`

- **syntax:** `%result = pto.vmuls %input, %scalar, %mask : !pto.vreg<NxT>, T, !pto.mask -> !pto.vreg<NxT>`

```c
for (int i = 0; i < N; i++)
    dst[i] = src[i] * scalar;
```

- **inputs:** `%input`, `%scalar`, and `%mask` as above.
- **outputs:** `%result` is the lane-wise product.
- **constraints and limitations:** Supported element types are hardware-family
  specific; the current PTO micro Instruction documentation covers the common numeric cases.

---

##### `pto.vmaxs`

- **syntax:** `%result = pto.vmaxs %input, %scalar, %mask : !pto.vreg<NxT>, T, !pto.mask -> !pto.vreg<NxT>`

```c
for (int i = 0; i < N; i++)
    dst[i] = (src[i] > scalar) ? src[i] : scalar;
```

- **inputs:** `%input`, `%scalar`, and `%mask` as above.
- **outputs:** `%result` is the lane-wise maximum.
- **constraints and limitations:** Input and result types MUST match.

---

##### `pto.vmins`

- **syntax:** `%result = pto.vmins %input, %scalar, %mask : !pto.vreg<NxT>, T, !pto.mask -> !pto.vreg<NxT>`

```c
for (int i = 0; i < N; i++)
    dst[i] = (src[i] < scalar) ? src[i] : scalar;
```

- **inputs:** `%input`, `%scalar`, and `%mask` as above.
- **outputs:** `%result` is the lane-wise minimum.
- **constraints and limitations:** Input and result types MUST match.

---

#### Bitwise

##### `pto.vands`

- **syntax:** `%result = pto.vands %input, %scalar, %mask : !pto.vreg<NxT>, T, !pto.mask -> !pto.vreg<NxT>`

```c
for (int i = 0; i < N; i++)
    dst[i] = src[i] & scalar;
```

- **inputs:** `%input`, `%scalar`, and `%mask` as above.
- **outputs:** `%result` is the lane-wise bitwise AND.
- **constraints and limitations:** Integer element types only.

---

##### `pto.vors`

- **syntax:** `%result = pto.vors %input, %scalar, %mask : !pto.vreg<NxT>, T, !pto.mask -> !pto.vreg<NxT>`

```c
for (int i = 0; i < N; i++)
    dst[i] = src[i] | scalar;
```

- **inputs:** `%input`, `%scalar`, and `%mask` as above.
- **outputs:** `%result` is the lane-wise bitwise OR.
- **constraints and limitations:** Integer element types only.

---

##### `pto.vxors`

- **syntax:** `%result = pto.vxors %input, %scalar, %mask : !pto.vreg<NxT>, T, !pto.mask -> !pto.vreg<NxT>`

```c
for (int i = 0; i < N; i++)
    dst[i] = src[i] ^ scalar;
```

- **inputs:** `%input`, `%scalar`, and `%mask` as above.
- **outputs:** `%result` is the lane-wise bitwise XOR.
- **constraints and limitations:** Integer element types only.

---

#### Shift

##### `pto.vshls`

- **syntax:** `%result = pto.vshls %input, %scalar, %mask : !pto.vreg<NxT>, T, !pto.mask -> !pto.vreg<NxT>`

```c
for (int i = 0; i < N; i++)
    dst[i] = src[i] << scalar;
```

- **inputs:** `%input` is the value vector, `%scalar` is the uniform shift
  amount, and `%mask` selects active lanes.
- **outputs:** `%result` is the shifted vector.
- **constraints and limitations:** Integer element types only. The shift amount
  SHOULD stay within the source element width.

---

##### `pto.vshrs`

- **syntax:** `%result = pto.vshrs %input, %scalar, %mask : !pto.vreg<NxT>, T, !pto.mask -> !pto.vreg<NxT>`

```c
for (int i = 0; i < N; i++)
    dst[i] = src[i] >> scalar;
```

- **inputs:** `%input` is the value vector, `%scalar` is the uniform shift
  amount, and `%mask` selects active lanes.
- **outputs:** `%result` is the shifted vector.
- **constraints and limitations:** Integer element types only.

---

##### `pto.vlrelu`

- **syntax:** `%result = pto.vlrelu %input, %scalar, %mask : !pto.vreg<NxT>, T, !pto.mask -> !pto.vreg<NxT>`

```c
for (int i = 0; i < N; i++)
    dst[i] = (src[i] >= 0) ? src[i] : scalar * src[i];
```

- **inputs:** `%input` is the activation vector, `%scalar` is the leaky slope,
  and `%mask` selects active lanes.
- **outputs:** `%result` is the lane-wise leaky-ReLU result.
- **constraints and limitations:** Only `f16` and `f32` forms are currently
  documented for `pto.vlrelu`.

---

#### Carry Operations

##### `pto.vaddcs`

- **syntax:** `%result, %carry = pto.vaddcs %lhs, %rhs, %carry_in, %mask : !pto.vreg<NxT>, !pto.vreg<NxT>, !pto.mask, !pto.mask -> !pto.vreg<NxT>, !pto.mask`
- **semantics:** Add with carry-in and carry-out.

```c
for (int i = 0; i < N; i++) {
    uint64_t r = (uint64_t)src0[i] + src1[i] + carry_in[i];
    dst[i] = (T)r;
    carry_out[i] = (r >> bitwidth);
}
```

- **inputs:** `%lhs` and `%rhs` are the value vectors, `%carry_in` is the
  incoming carry predicate, and `%mask` selects active lanes.
- **outputs:** `%result` is the arithmetic result and `%carry` is the carry-out
  predicate.
- **constraints and limitations:** This is the scalar-extended carry-chain
  family. Treat it as an unsigned integer operation unless the verifier states a
  wider legal domain.

---

##### `pto.vsubcs`

- **syntax:** `%result, %borrow = pto.vsubcs %lhs, %rhs, %borrow_in, %mask : !pto.vreg<NxT>, !pto.vreg<NxT>, !pto.mask, !pto.mask -> !pto.vreg<NxT>, !pto.mask`
- **semantics:** Subtract with borrow-in and borrow-out.

```c
for (int i = 0; i < N; i++) {
    dst[i] = src0[i] - src1[i] - borrow_in[i];
    borrow_out[i] = (src0[i] < src1[i] + borrow_in[i]);
}
```

- **inputs:** `%lhs` and `%rhs` are the value vectors, `%borrow_in` is the
  incoming borrow predicate, and `%mask` selects active lanes.
- **outputs:** `%result` is the arithmetic result and `%borrow` is the
  borrow-out predicate.
- **constraints and limitations:** This is the scalar-extended borrow-chain
  family and SHOULD be treated as an unsigned integer operation.

---

#### Typical Usage

```mlir
// Add bias to all elements
%biased = pto.vadds %activation, %bias_scalar, %mask : !pto.vreg<64xf32>, f32, !pto.mask -> !pto.vreg<64xf32>

// Scale by constant
%scaled = pto.vmuls %input, %scale, %mask : !pto.vreg<64xf32>, f32, !pto.mask -> !pto.vreg<64xf32>

// Clamp to [0, 255] for uint8 quantization
%clamped_low = pto.vmaxs %input, %c0, %mask : !pto.vreg<64xf32>, f32, !pto.mask -> !pto.vreg<64xf32>
%clamped = pto.vmins %clamped_low, %c255, %mask : !pto.vreg<64xf32>, f32, !pto.mask -> !pto.vreg<64xf32>

// Shift right by fixed amount
%shifted = pto.vshrs %data, %c4, %mask : !pto.vreg<64xi32>, i32, !pto.mask -> !pto.vreg<64xi32>
```

---

<a id="isa-09-conversion-ops"></a>

### 9. Conversion Ops

> **Category:** Type conversion operations
> **Pipeline:** PIPE_V (Vector Core)

Operations that convert between data types (float/int, narrowing/widening).

#### Common Operand Model

- `%input` is the source vector register value.
- `%result` is the destination vector register value.
- `round_mode`, `sat`, and `part` control rounding, saturation, and lane-part
  selection in attribute form.
- The single `pto.vcvt` surface covers float-int, float-float, int-float, and
  int-int conversion families.

---

#### `pto.vci`

- **syntax:** `%result = pto.vci %index {order = "ORDER"} : integer -> !pto.vreg<NxT>`
- **semantics:** Generate a lane-index vector from a scalar seed/index value.
- **inputs:**
  `%index` is the scalar seed or base index.
- **outputs:**
  `%result` is the generated index vector.
- **constraints and limitations:**
  This is an index-generation family, not a numeric conversion. `ORDER` and the
  result element type together determine how indices are generated.

---

#### `pto.vcvt`

- **syntax:** `%result = pto.vcvt %input {round_mode = "ROUND_MODE", sat = "SAT_MODE", part = "PART_MODE"} : !pto.vreg<NxT0> -> !pto.vreg<MxT1>`
- **semantics:** Type conversion between float/int types with rounding control.

```c
for (int i = 0; i < min(N, M); i++)
    dst[i] = convert(src[i], T0, T1, round_mode);
```

- **inputs:**
  `%input` is the source vector; attributes select rounding, saturation, and
  even/odd placement when the conversion changes width.
- **outputs:**
  `%result` is the converted vector.
- **constraints and limitations:**
  Only documented source/destination type pairs are legal. `PART_EVEN` /
  `PART_ODD` is only meaningful for width-changing forms that pack two source
  streams into one destination register.

---

##### Rounding Modes

| Mode | Description |
|------|-------------|
| `ROUND_R` | Round to nearest, ties to even (default) |
| `ROUND_A` | Round away from zero |
| `ROUND_F` | Round toward negative infinity (floor) |
| `ROUND_C` | Round toward positive infinity (ceil) |
| `ROUND_Z` | Round toward zero (truncate) |
| `ROUND_O` | Round to odd |

---

##### Saturation Modes

| Mode | Description |
|------|-------------|
| `RS_ENABLE` | Saturate on overflow |
| `RS_DISABLE` | No saturation (wrap/undefined on overflow) |

---

##### Part Modes (for width-changing conversions)

| Mode | Description |
|------|-------------|
| `PART_EVEN` | Output to even-indexed lanes |
| `PART_ODD` | Output to odd-indexed lanes |

---

##### A5 Supported Conversions

**Float-Float (vcvtff):**
- f32 ↔ f16
- f32 ↔ bf16
- f16 ↔ bf16

**Float-Int (vcvtfi):**
- f16 → i16, f16 → i32
- f32 → i16, f32 → i32
- bf16 → i32

**Int-Float (vcvtif):**
- i16 → f16
- i32 → f32

---

##### Width-Changing Conversion Pattern

For conversions that change width (e.g., f32→f16), use even/odd parts and combine:

```mlir
// Convert two f32 vectors to one f16 vector
%even = pto.vcvt %in0 {round_mode = "ROUND_R", sat = "RS_ENABLE", part = "PART_EVEN"}
    : !pto.vreg<64xf32> -> !pto.vreg<128xf16>
%odd  = pto.vcvt %in1 {round_mode = "ROUND_R", sat = "RS_ENABLE", part = "PART_ODD"}
    : !pto.vreg<64xf32> -> !pto.vreg<128xf16>
%result = pto.vor %even, %odd, %mask : !pto.vreg<128xf16>, !pto.vreg<128xf16>, !pto.mask -> !pto.vreg<128xf16>
```

---

#### `pto.vtrc`

- **syntax:** `%result = pto.vtrc %input, %mask, "ROUND_MODE" : !pto.vreg<NxT>, !pto.mask<BW> -> !pto.vreg<NxT>`
- **semantics:** Truncate/round float to integer-valued float (stays in float type).

```c
for (int i = 0; i < N; i++)
    dst[i] = round_to_int_valued_float(src[i], round_mode);
```

- **inputs:**
  `%input` is the floating-point source vector, `%mask` selects active lanes,
  and `ROUND_MODE` selects the truncation/rounding rule.
- **outputs:**
  `%result` is still a floating-point vector, but each active lane now carries
  an integer-valued floating-point result.
- **constraints and limitations:**
  This op does not change the element type. `T` must be `f16`, `f32`, or
  `bf16`. `ROUND_MODE` must be one of `ROUND_R`, `ROUND_A`, `ROUND_F`,
  `ROUND_C`, or `ROUND_Z`. `BW` must match the element width: `b16` for
  `f16`/`bf16`, `b32` for `f32`.

**Example:**
```mlir
// Round to nearest integer, keep as float
%rounded = pto.vtrc %input, %mask, "ROUND_R" : !pto.vreg<64xf32>, !pto.mask<b32> -> !pto.vreg<64xf32>
// input:  [1.4, 2.6, -1.5, 3.0]
// output: [1.0, 3.0, -2.0, 3.0]
```

---

#### Typical Usage

```mlir
// Quantization: f32 → i8 with saturation
%scaled = pto.vmuls %input, %scale, %mask : !pto.vreg<64xf32>, f32, !pto.mask -> !pto.vreg<64xf32>
%quantized = pto.vcvt %scaled {round_mode = "ROUND_R", sat = "RS_ENABLE"}
    : !pto.vreg<64xf32> -> !pto.vreg<64xi32>
// Then narrow i32 → i8 via pack ops

// Mixed precision: bf16 → f32 for accumulation
%f32_vec = pto.vcvt %bf16_input {round_mode = "ROUND_R"}
    : !pto.vreg<128xbf16> -> !pto.vreg<64xf32>

// Floor for integer division
%floored = pto.vtrc %ratio, %mask, "ROUND_F" : !pto.vreg<64xf32>, !pto.mask<b32> -> !pto.vreg<64xf32>
%int_div = pto.vcvt %floored {round_mode = "ROUND_Z"}
    : !pto.vreg<64xf32> -> !pto.vreg<64xi32>
```

---

<a id="isa-10-reduction-ops"></a>

### 10. Reduction Ops

> **Category:** Vector reduction operations
> **Pipeline:** PIPE_V (Vector Core)

Operations that reduce a vector to a scalar or per-group result.

#### Common Operand Model

- `%input` is the source vector register value.
- `%mask` is the predicate operand `Pg`; inactive lanes do not participate.
- `%result` is the destination vector register value.
- Reduction results are written into the low-significance portion of the
  destination vector and the remaining destination bits are zero-filled.

---

#### Full Vector Reductions

##### `pto.vcadd`

- **syntax:** `%result = pto.vcadd %input, %mask : !pto.vreg<NxT>, !pto.mask -> !pto.vreg<NxT>`
- **A5 types:** i16-i64, f16, f32
- **semantics:** Sum all elements. Result in lane 0, others zeroed.

```c
T sum = 0;
for (int i = 0; i < N; i++)
    sum += src[i];
dst[0] = sum;
for (int i = 1; i < N; i++)
    dst[i] = 0;
```

- **inputs:** `%input` is the source vector and `%mask` selects participating
  lanes.
- **outputs:** `%result` contains the reduction result in its low element(s).
- **constraints and limitations:** Some narrow integer forms may widen the
  internal accumulation or result placement. If all predicate bits are zero, the
  result is zero.

---

##### `pto.vcmax`

- **syntax:** `%result = pto.vcmax %input, %mask : !pto.vreg<NxT>, !pto.mask -> !pto.vreg<NxT>`
- **A5 types:** i16-i32, f16, f32
- **semantics:** Find max element with argmax. Result value + index in lane 0.

```c
T mx = -INF; int idx = 0;
for (int i = 0; i < N; i++)
    if (src[i] > mx) { mx = src[i]; idx = i; }
dst_val[0] = mx;
dst_idx[0] = idx;
```

- **inputs:** `%input` is the source vector and `%mask` selects participating
  lanes.
- **outputs:** `%result` carries the reduction result in the low destination
  positions.
- **constraints and limitations:** This family computes both the extremum and
  location information, but the exact packing of that information into the
  destination vector depends on the chosen form. If all predicate bits are zero,
  the result follows the zero-filled convention.

---

##### `pto.vcmin`

- **syntax:** `%result = pto.vcmin %input, %mask : !pto.vreg<NxT>, !pto.mask -> !pto.vreg<NxT>`
- **A5 types:** i16-i32, f16, f32
- **semantics:** Find min element with argmin. Result value + index in lane 0.

```c
T mn = INF; int idx = 0;
for (int i = 0; i < N; i++)
    if (src[i] < mn) { mn = src[i]; idx = i; }
dst_val[0] = mn;
dst_idx[0] = idx;
```

- **inputs:** `%input` is the source vector and `%mask` selects participating
  lanes.
- **outputs:** `%result` carries the reduction result in the low destination
  positions.
- **constraints and limitations:** As with `pto.vcmax`, the exact value/index
  packing depends on the chosen form and MUST be preserved consistently.

---

#### Per-VLane (Group) Reductions

The vector register is organized as **8 VLanes** of 32 bytes each. Group reductions operate within each VLane independently.

```
vreg layout (f32 example, 64 elements total):
VLane 0: [0..7]   VLane 1: [8..15]  VLane 2: [16..23] VLane 3: [24..31]
VLane 4: [32..39] VLane 5: [40..47] VLane 6: [48..55] VLane 7: [56..63]
```

##### `pto.vcgadd`

- **syntax:** `%result = pto.vcgadd %input, %mask : !pto.vreg<NxT>, !pto.mask -> !pto.vreg<NxT>`
- **A5 types:** i16-i32, f16, f32
- **semantics:** Sum within each VLane. 8 results at indices 0, 8, 16, 24, 32, 40, 48, 56 (for f32).

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
// For f32: results at dst[0], dst[8], dst[16], dst[24], dst[32], dst[40], dst[48], dst[56]
```

- **inputs:** `%input` is the source vector and `%mask` selects participating
  lanes.
- **outputs:** `%result` contains one sum per 32-byte VLane group, written
  contiguously into the low slot of each group.
- **constraints and limitations:** This is a per-32-byte VLane-group reduction.
  Inactive lanes are treated as zero.

---

##### `pto.vcgmax`

- **syntax:** `%result = pto.vcgmax %input, %mask : !pto.vreg<NxT>, !pto.mask -> !pto.vreg<NxT>`
- **A5 types:** i16-i32, f16, f32
- **semantics:** Max within each VLane.

```c
int K = N / 8;
for (int g = 0; g < 8; g++) {
    T mx = -INF;
    for (int i = 0; i < K; i++)
        if (src[g*K + i] > mx) mx = src[g*K + i];
    dst[g*K] = mx;
    for (int i = 1; i < K; i++)
        dst[g*K + i] = 0;
}
```

- **inputs:** `%input` is the source vector and `%mask` selects participating
  lanes.
- **outputs:** `%result` contains one maximum per 32-byte VLane group.
- **constraints and limitations:** Grouping is by hardware 32-byte VLane, not by
  arbitrary software subvector.

---

##### `pto.vcgmin`

- **syntax:** `%result = pto.vcgmin %input, %mask : !pto.vreg<NxT>, !pto.mask -> !pto.vreg<NxT>`
- **A5 types:** i16-i32, f16, f32
- **semantics:** Min within each VLane.

```c
int K = N / 8;
for (int g = 0; g < 8; g++) {
    T mn = INF;
    for (int i = 0; i < K; i++)
        if (src[g*K + i] < mn) mn = src[g*K + i];
    dst[g*K] = mn;
    for (int i = 1; i < K; i++)
        dst[g*K + i] = 0;
}
```

- **inputs:** `%input` is the source vector and `%mask` selects participating
  lanes.
- **outputs:** `%result` contains one minimum per 32-byte VLane group.
- **constraints and limitations:** Grouping is by hardware 32-byte VLane, not by
  arbitrary software subvector.

---

#### Prefix Operations

##### `pto.vcpadd`

- **syntax:** `%result = pto.vcpadd %input, %mask : !pto.vreg<NxT>, !pto.mask -> !pto.vreg<NxT>`
- **A5 types:** f16, f32
- **semantics:** Inclusive prefix sum (scan).

```c
dst[0] = src[0];
for (int i = 1; i < N; i++)
    dst[i] = dst[i-1] + src[i];
```

**Example:**
```c
// input:  [1, 2, 3, 4, 5, ...]
// output: [1, 3, 6, 10, 15, ...]
```

- **inputs:** `%input` is the source vector and `%mask` selects participating
  lanes.
- **outputs:** `%result` is the inclusive prefix-sum vector.
- **constraints and limitations:** Only floating-point element types are
  documented on the current A5 surface here.

---

#### Typical Usage

```mlir
// Softmax: find max for numerical stability
%max_vec = pto.vcmax %logits, %mask : !pto.vreg<64xf32>, !pto.mask -> !pto.vreg<64xf32>
// max is in lane 0, broadcast it
%max_broadcast = pto.vlds %ub_tmp[%c0] {dist = "BRC_B32"} : !pto.ptr<f32, ub> -> !pto.vreg<64xf32>

// Row-wise sum using vcgadd (for 8-row tile)
%row_sums = pto.vcgadd %tile, %mask : !pto.vreg<64xf32>, !pto.mask -> !pto.vreg<64xf32>
// Results at indices 0, 8, 16, 24, 32, 40, 48, 56

// Full vector sum for normalization
%total = pto.vcadd %values, %mask : !pto.vreg<64xf32>, !pto.mask -> !pto.vreg<64xf32>
// total[0] contains the sum

// Prefix sum for cumulative distribution
%cdf = pto.vcpadd %pdf, %mask : !pto.vreg<64xf32>, !pto.mask -> !pto.vreg<64xf32>
```

---

<a id="isa-11-compare-select"></a>

### 11. Compare & Select

> **Category:** Comparison and conditional selection operations
> **Pipeline:** PIPE_V (Vector Core)

Operations that compare vectors and conditionally select elements.

#### Common Operand Model

- `%src0` and `%src1` are source vector operands.
- `%scalar` is the scalar operand for scalar-comparison families.
- `%seed` is the incoming predicate that limits which lanes participate in the
  compare.
- `%result` is either a predicate mask (`vcmp`, `vcmps`) or a vector register
  (`vsel`, `vselr`, `vselrv2`).

---

#### Comparison Operations

##### `pto.vcmp`

- **syntax:** `%result = pto.vcmp %src0, %src1, %seed, "CMP_MODE" : !pto.vreg<NxT>, !pto.vreg<NxT>, !pto.mask -> !pto.mask`
- **semantics:** Element-wise comparison, output predicate mask.

```c
for (int i = 0; i < N; i++)
    if (seed[i])
        dst[i] = (src0[i] CMP src1[i]) ? 1 : 0;
```

**Compare modes:**

| Mode | Operation |
|------|-----------|
| `eq` | Equal (==) |
| `ne` | Not equal (!=) |
| `lt` | Less than (<) |
| `le` | Less than or equal (<=) |
| `gt` | Greater than (>) |
| `ge` | Greater than or equal (>=) |

**Example:**
```mlir
%all_active = pto.pset_b32 "PAT_ALL" : !pto.mask
%lt_mask = pto.vcmp %a, %b, %all_active, "lt" : !pto.vreg<64xf32>, !pto.vreg<64xf32>, !pto.mask -> !pto.mask
// lt_mask[i] = 1 if a[i] < b[i]
```

- **inputs:** `%src0`, `%src1`, and `%seed`; `CMP_MODE` selects the comparison
  predicate.
- **outputs:** `%result` is the generated predicate mask.
- **constraints and limitations:** Only lanes enabled by `%seed` participate.
  Integer and floating-point comparisons follow their own element-type-specific
  comparison rules.

---

##### `pto.vcmps`

- **syntax:** `%result = pto.vcmps %src, %scalar, %seed, "CMP_MODE" : !pto.vreg<NxT>, T, !pto.mask -> !pto.mask`
- **semantics:** Compare vector against scalar.

```c
for (int i = 0; i < N; i++)
    if (seed[i])
        dst[i] = (src[i] CMP scalar) ? 1 : 0;
```

**Example:**
```mlir
%positive_mask = pto.vcmps %values, %c0_f32, %all_active, "gt"
    : !pto.vreg<64xf32>, f32, !pto.mask -> !pto.mask
// positive_mask[i] = 1 if values[i] > 0
```

- **inputs:** `%src` is the vector source, `%scalar` is the scalar comparison
  value, and `%seed` is the incoming predicate.
- **outputs:** `%result` is the generated predicate mask.
- **constraints and limitations:** For 32-bit scalar forms, the scalar source
  MUST satisfy the backend's legal scalar-source constraints for this family.

---

#### Selection Operations

##### `pto.vsel`

- **syntax:** `%result = pto.vsel %src0, %src1, %mask : !pto.vreg<NxT>, !pto.vreg<NxT>, !pto.mask -> !pto.vreg<NxT>`
- **semantics:** Per-lane select based on mask.

```c
for (int i = 0; i < N; i++)
    dst[i] = mask[i] ? src0[i] : src1[i];
```

**Example — Conditional assignment:**
```mlir
// dst = mask ? true_vals : false_vals
%result = pto.vsel %true_vals, %false_vals, %condition
    : !pto.vreg<64xf32>, !pto.vreg<64xf32>, !pto.mask -> !pto.vreg<64xf32>
```

- **inputs:** `%src0` is the true-path vector, `%src1` is the false-path vector,
  and `%mask` selects between them.
- **outputs:** `%result` is the selected vector.
- **constraints and limitations:** Source vectors and result MUST have matching
  vector shapes and element types.

---

##### `pto.vselr`

- **syntax:** `%result = pto.vselr %src0, %src1 : !pto.vreg<NxT>, !pto.vreg<NxT> -> !pto.vreg<NxT>`
- **semantics:** Select with reversed mask semantics.

```c
for (int i = 0; i < N; i++)
    dst[i] = mask[i] ? src1[i] : src0[i];  // reversed from vsel
```

- **inputs:** `%src0` and `%src1` are the source vectors.
- **outputs:** `%result` is the selected vector.
- **constraints and limitations:** This family preserves reversed-select
  semantics. If the concrete lowering uses an implicit predicate source, that
  predicate source MUST be documented by the surrounding IR pattern.

---

##### `pto.vselrv2`

- **syntax:** `%result = pto.vselrv2 %src0, %src1 : !pto.vreg<NxT>, !pto.vreg<NxT> -> !pto.vreg<NxT>`
- **semantics:** Variant select form with the same current two-vector operand shape.
- **inputs:** `%src0` and `%src1` are the source vectors.
- **outputs:** `%result` is the selected vector.
- **constraints and limitations:** This page records the surface shape only.
  Lowering MUST preserve the exact A5 variant semantics selected for this form.

---

#### Typical Usage

```mlir
// Clamp negative values to zero (manual ReLU)
%all = pto.pset_b32 "PAT_ALL" : !pto.mask
%zero = pto.vbr %c0_f32 : f32 -> !pto.vreg<64xf32>
%neg_mask = pto.vcmps %input, %c0_f32, %all, "lt" : !pto.vreg<64xf32>, f32, !pto.mask -> !pto.mask
%clamped = pto.vsel %zero, %input, %neg_mask : !pto.vreg<64xf32>, !pto.vreg<64xf32>, !pto.mask -> !pto.vreg<64xf32>

// Element-wise max via compare+select
%gt_mask = pto.vcmp %a, %b, %all, "gt" : !pto.vreg<64xf32>, !pto.vreg<64xf32>, !pto.mask -> !pto.mask
%max_ab = pto.vsel %a, %b, %gt_mask : !pto.vreg<64xf32>, !pto.vreg<64xf32>, !pto.mask -> !pto.vreg<64xf32>

// Threshold filter
%above_thresh = pto.vcmps %scores, %threshold, %all, "ge" : !pto.vreg<64xf32>, f32, !pto.mask -> !pto.mask
%filtered = pto.vsel %scores, %zero, %above_thresh : !pto.vreg<64xf32>, !pto.vreg<64xf32>, !pto.mask -> !pto.vreg<64xf32>
```

---

#### Compare + Select Pattern

```mlir
// Softmax safe exp: exp(x - max) where x < max returns exp of negative
// but we want to clamp to avoid underflow

%all = pto.pset_b32 "PAT_ALL" : !pto.mask

// 1. Compare against threshold
%too_small = pto.vcmps %x_minus_max, %min_exp_arg, %all, "lt"
    : !pto.vreg<64xf32>, f32, !pto.mask -> !pto.mask

// 2. Clamp values below threshold
%clamped = pto.vsel %min_exp_arg_vec, %x_minus_max, %too_small
    : !pto.vreg<64xf32>, !pto.vreg<64xf32>, !pto.mask -> !pto.vreg<64xf32>

// 3. Safe exp
%exp_result = pto.vexp %clamped, %all : !pto.vreg<64xf32>, !pto.mask -> !pto.vreg<64xf32>
```

---

<a id="isa-12-data-rearrangement"></a>

### 12. Data Rearrangement

> **Category:** In-register data movement and permutation
> **Pipeline:** PIPE_V (Vector Core)

Operations that rearrange data within or between vector registers without memory access.

#### Common Operand Model

- `%lhs` / `%rhs` are source vector register values.
- `%src` is a single source vector register value.
- `%result` is the destination vector register value unless an op explicitly
  returns multiple vectors.
- These families do not access UB directly; they only rearrange register
  contents.

---

#### Interleave / Deinterleave

##### `pto.vintlv`

- **syntax:** `%low, %high = pto.vintlv %lhs, %rhs : !pto.vreg<NxT>, !pto.vreg<NxT> -> !pto.vreg<NxT>, !pto.vreg<NxT>`
- **semantics:** Interleave elements from two sources.

```c
// Interleave: merge even/odd elements from two sources
// low  = {src0[0], src1[0], src0[1], src1[1], ...}
// high = {src0[N/2], src1[N/2], src0[N/2+1], src1[N/2+1], ...}
```

- **inputs:** `%lhs` and `%rhs` are the two source vectors.
- **outputs:** `%low` and `%high` are the two destination vectors.
- **constraints and limitations:** The two outputs form a paired interleave
  result. The PTO micro Instruction representation exposes that pair as two SSA results, and the pair ordering MUST
  be preserved.

---

##### `pto.vdintlv`

- **syntax:** `%low, %high = pto.vdintlv %lhs, %rhs : !pto.vreg<NxT>, !pto.vreg<NxT> -> !pto.vreg<NxT>, !pto.vreg<NxT>`
- **semantics:** Deinterleave elements into even/odd.

```c
// Deinterleave: separate even/odd elements
// low  = {src0[0], src0[2], src0[4], ...}  // even
// high = {src0[1], src0[3], src0[5], ...}  // odd
```

- **inputs:** `%lhs` and `%rhs` represent the interleaved source stream in the
  current PTO micro Instruction representation.
- **outputs:** `%low` and `%high` are the separated destination vectors.
- **constraints and limitations:** The two outputs form the even/odd
  deinterleave result pair, and their ordering MUST be preserved.

---

#### Slide / Shift

##### `pto.vslide`

- **syntax:** `%result = pto.vslide %src0, %src1, %amt : !pto.vreg<NxT>, !pto.vreg<NxT>, i16 -> !pto.vreg<NxT>`
- **semantics:** Concatenate two vectors and extract N-element window at offset.

```c
// Conceptually: tmp[0..2N-1] = {src1, src0}
// dst[i] = tmp[amt + i]
if (amt >= 0)
    for (int i = 0; i < N; i++)
        dst[i] = (i >= amt) ? src0[i - amt] : src1[N - amt + i];
```

**Use case:** Sliding window operations, shift register patterns.

- **inputs:** `%src0` and `%src1` provide the concatenated source window and
  `%amt` selects the extraction offset.
- **outputs:** `%result` is the extracted destination window.
- **constraints and limitations:** `pto.vslide` operates on the logical
  concatenation of `%src1` and `%src0`. The source order and extraction offset
  MUST be preserved exactly.

---

##### `pto.vshift`

- **syntax:** `%result = pto.vshift %src, %amt : !pto.vreg<NxT>, i16 -> !pto.vreg<NxT>`
- **semantics:** Single-source slide (shift with zero fill).

```c
for (int i = 0; i < N; i++)
    dst[i] = (i >= amt) ? src[i - amt] : 0;
```

- **inputs:** `%src` is the source vector and `%amt` is the slide amount.
- **outputs:** `%result` is the shifted vector.
- **constraints and limitations:** This surface represents the single-source
  slide/shift family. Zero-fill versus other fill behavior MUST match the
  selected form.

---

#### Compress / Expand

##### `pto.vsqz`

- **syntax:** `%result = pto.vsqz %src, %mask : !pto.vreg<NxT>, !pto.mask -> !pto.vreg<NxT>`
- **semantics:** Compress — pack active lanes to front.

```c
int j = 0;
for (int i = 0; i < N; i++)
    if (mask[i]) dst[j++] = src[i];
while (j < N) dst[j++] = 0;
```

**Use case:** Sparse data compaction, filtering.

- **inputs:** `%src` is the source vector and `%mask` selects which elements are
  kept.
- **outputs:** `%result` is the compacted vector.
- **constraints and limitations:** This is a reduction-style compaction family.
  Preserved element order MUST match source lane order.

---

##### `pto.vusqz`

- **syntax:** `%result = pto.vusqz %mask : !pto.mask -> !pto.vreg<NxT>`
- **semantics:** Expand — scatter front elements to active positions.

```c
int j = 0;
for (int i = 0; i < N; i++)
    if (mask[i]) dst[i] = src_front[j++];
    else dst[i] = 0;
```

- **inputs:** `%mask` is the expansion/placement predicate.
- **outputs:** `%result` is the expanded vector image.
- **constraints and limitations:** The source-front stream is implicit in the
  current surface. Lane placement for active and inactive positions MUST be
  preserved exactly.

---

#### Permutation

##### `pto.vperm`

- **syntax:** `%result = pto.vperm %src, %index : !pto.vreg<NxT>, !pto.vreg<NxI> -> !pto.vreg<NxT>`
- **semantics:** In-register permute (table lookup). **Not** memory gather.

```c
for (int i = 0; i < N; i++)
    dst[i] = src[index[i] % N];
```

**Note:** This operates on register contents, unlike `pto.vgather2` which reads from UB memory.

- **inputs:** `%src` is the source vector and `%index` supplies per-lane source
  indices.
- **outputs:** `%result` is the permuted vector.
- **constraints and limitations:** This is an in-register permutation family.
  `%index` values outside the legal range follow the wrap/clamp behavior of the
  selected form.

---

##### `pto.vselr`

- **syntax:** `%result = pto.vselr %src0, %src1 : !pto.vreg<NxT>, !pto.vreg<NxI> -> !pto.vreg<NxT>`
- **semantics:** Register select with reversed mask semantics.

```c
for (int i = 0; i < N; i++)
    dst[i] = mask[i] ? src1[i] : src0[i];
```

- **inputs:** `%src0` and `%src1` are source vectors.
- **outputs:** `%result` is the selected vector.
- **constraints and limitations:** This page records the rearrangement use of
  the family; the compare/select page documents the same name from the predicate
  selection perspective.

---

#### Pack / Unpack

##### `pto.vpack`

- **syntax:** `%result = pto.vpack %src0, %src1, %part : !pto.vreg<NxT_wide>, !pto.vreg<NxT_wide>, index -> !pto.vreg<2NxT_narrow>`
- **semantics:** Narrowing pack — two wide vectors to one narrow vector.

```c
// e.g., two vreg<64xi32> → one vreg<128xi16>
for (int i = 0; i < N; i++) {
    dst[i]     = truncate(src0[i]);
    dst[N + i] = truncate(src1[i]);
}
```

- **inputs:** `%src0` and `%src1` are wide source vectors and `%part` selects
  the packing submode.
- **outputs:** `%result` is the packed narrow vector.
- **constraints and limitations:** Packing is a narrowing conversion. Source
  values that do not fit the destination width follow the truncation semantics
  of the selected packing mode.

---

##### `pto.vsunpack`

- **syntax:** `%result = pto.vsunpack %src, %part : !pto.vreg<NxT_narrow>, index -> !pto.vreg<N/2xT_wide>`
- **semantics:** Sign-extending unpack — narrow to wide (half).

```c
// e.g., vreg<128xi16> → vreg<64xi32> (one half)
for (int i = 0; i < N/2; i++)
    dst[i] = sign_extend(src[part_offset + i]);
```

- **inputs:** `%src` is the packed narrow vector and `%part` selects which half
  is unpacked.
- **outputs:** `%result` is the widened vector.
- **constraints and limitations:** This is the sign-extending unpack family.

---

##### `pto.vzunpack`

- **syntax:** `%result = pto.vzunpack %src, %part : !pto.vreg<NxT_narrow>, index -> !pto.vreg<N/2xT_wide>`
- **semantics:** Zero-extending unpack — narrow to wide (half).

```c
for (int i = 0; i < N/2; i++)
    dst[i] = zero_extend(src[part_offset + i]);
```

- **inputs:** `%src` is the packed narrow vector and `%part` selects which half
  is unpacked.
- **outputs:** `%result` is the widened vector.
- **constraints and limitations:** This is the zero-extending unpack family.

---

#### Typical Usage

```mlir
// AoS → SoA conversion using deinterleave
%even, %odd = pto.vdintlv %interleaved0, %interleaved1
    : !pto.vreg<64xf32>, !pto.vreg<64xf32> -> !pto.vreg<64xf32>, !pto.vreg<64xf32>

// Filter: keep only elements passing condition
%pass_mask = pto.vcmps %values, %threshold, %all, "gt"
    : !pto.vreg<64xf32>, f32, !pto.mask -> !pto.mask
%compacted = pto.vsqz %values, %pass_mask
    : !pto.vreg<64xf32>, !pto.mask -> !pto.vreg<64xf32>

// Sliding window sum
%prev_window = pto.vslide %curr, %prev, %c1
    : !pto.vreg<64xf32>, !pto.vreg<64xf32>, i16 -> !pto.vreg<64xf32>
%window_sum = pto.vadd %curr, %prev_window, %all
    : !pto.vreg<64xf32>, !pto.vreg<64xf32>, !pto.mask -> !pto.vreg<64xf32>

// Type narrowing via pack
%packed_i16 = pto.vpack %wide0_i32, %wide1_i32, %c0
    : !pto.vreg<64xi32>, !pto.vreg<64xi32>, index -> !pto.vreg<128xi16>
```

---

#### V2 Interleave Forms

##### `pto.vintlvv2`

- **syntax:** `%result = pto.vintlvv2 %lhs, %rhs, "PART" : !pto.vreg<NxT>, !pto.vreg<NxT> -> !pto.vreg<NxT>`
- **inputs:** `%lhs` and `%rhs` are source vectors and `PART` selects the
  returned half of the V2 interleave result.
- **outputs:** `%result` is the selected interleave half.
- **constraints and limitations:** This op exposes only one half of the V2
  result in SSA form.

##### `pto.vdintlvv2`

- **syntax:** `%result = pto.vdintlvv2 %lhs, %rhs, "PART" : !pto.vreg<NxT>, !pto.vreg<NxT> -> !pto.vreg<NxT>`
- **inputs:** `%lhs` and `%rhs` are source vectors and `PART` selects the
  returned half of the V2 deinterleave result.
- **outputs:** `%result` is the selected deinterleave half.
- **constraints and limitations:** This op exposes only one half of the V2
  result in SSA form.

---

<a id="isa-13-dsa-sfu-ops"></a>

### 13. DSA/SFU Ops

> **Category:** Domain-specific accelerator and special function unit operations
> **Pipeline:** PIPE_V (Vector Core) / SFU

Fused operations, special functions, and UB-to-UB operations that leverage hardware acceleration.

#### Common Operand Model

- `%input`, `%lhs`, `%rhs`, `%acc`, and `%alpha` are source SSA values whose
  roles are called out per instruction.
- `%mask` is the predicate operand `Pg` when present.
- `%result` is the destination SSA value.
- This page mixes three different backend shapes: pure `vreg -> vreg` ops,
  conversion/fusion ops, and UB-to-UB helpers. Each instruction section calls
  out which storage model it uses.

---

#### Fused Activation Ops (vreg→vreg)

##### `pto.vlrelu`

- **syntax:** `%result = pto.vlrelu %input, %alpha, %mask : !pto.vreg<NxT>, T, !pto.mask -> !pto.vreg<NxT>`
- **A5 types:** f16, f32
- **semantics:** Leaky ReLU with scalar alpha.

```c
for (int i = 0; i < N; i++)
    dst[i] = (src[i] >= 0) ? src[i] : alpha * src[i];
```

- **inputs:** `%input` is the activation vector, `%alpha` is the scalar slope,
  and `%mask` selects active lanes.
- **outputs:** `%result` is the leaky-ReLU vector.
- **constraints and limitations:** Only `f16` and `f32` forms are currently
  documented for `pto.vlrelu`.

---

##### `pto.vprelu`

- **syntax:** `%result = pto.vprelu %input, %alpha : !pto.vreg<NxT>, !pto.vreg<NxT> -> !pto.vreg<NxT>`
- **A5 types:** f16, f32
- **semantics:** Parametric ReLU with per-element alpha vector.

```c
for (int i = 0; i < N; i++)
    dst[i] = (src[i] >= 0) ? src[i] : alpha[i] * src[i];
```

- **inputs:** `%input` is the activation vector and `%alpha` is the per-element
  slope vector.
- **outputs:** `%result` is the parametric-ReLU vector.
- **constraints and limitations:** Floating-point element types only on the
  current A5 surface.

---

##### `pto.vexpdiff`

- **syntax:** `%result = pto.vexpdiff %input, %max : !pto.vreg<NxT>, !pto.vreg<NxT> -> !pto.vreg<NxT>`
- **A5 types:** f16, f32
- **semantics:** Fused exp(x - max) for numerically stable softmax.

```c
for (int i = 0; i < N; i++)
    dst[i] = expf(src[i] - max[i]);
```

**Use case:** Softmax numerator computation with numerical stability.

- **inputs:** `%input` is the source vector and `%max` is the broadcasted
  subtraction term.
- **outputs:** `%result` is the fused `exp(input - max)` vector.
- **constraints and limitations:** Floating-point element types only.

---

#### Fused Compute+Convert Ops

##### `pto.vaddrelu`

- **syntax:** `%result = pto.vaddrelu %lhs, %rhs : !pto.vreg<NxT>, !pto.vreg<NxT> -> !pto.vreg<NxT>`
- **A5 types:** f16, f32
- **semantics:** Fused add + ReLU.

```c
for (int i = 0; i < N; i++)
    dst[i] = max(src0[i] + src1[i], 0);
```

- **inputs:** `%lhs` and `%rhs` are the two addends.
- **outputs:** `%result` is the fused add-then-ReLU result.
- **constraints and limitations:** Floating-point element types only on the
  current documented surface.

---

##### `pto.vsubrelu`

- **syntax:** `%result = pto.vsubrelu %lhs, %rhs : !pto.vreg<NxT>, !pto.vreg<NxT> -> !pto.vreg<NxT>`
- **A5 types:** f16, f32
- **semantics:** Fused sub + ReLU.

```c
for (int i = 0; i < N; i++)
    dst[i] = max(src0[i] - src1[i], 0);
```

- **inputs:** `%lhs` is the minuend and `%rhs` is the subtrahend.
- **outputs:** `%result` is the fused sub-then-ReLU result.
- **constraints and limitations:** Floating-point element types only on the
  current documented surface.

---

##### `pto.vaxpy`

- **syntax:** `%result = pto.vaxpy %src0, %src1, %alpha : !pto.vreg<NxT>, !pto.vreg<NxT>, T -> !pto.vreg<NxT>`
- **A5 types:** f16, f32
- **semantics:** AXPY — scalar-vector multiply-add.

```c
for (int i = 0; i < N; i++)
    dst[i] = alpha * src0[i] + src1[i];
```

- **inputs:** `%src0` is the scaled vector, `%src1` is the addend vector, and
  `%alpha` is the scalar multiplier.
- **outputs:** `%result` is the fused AXPY result.
- **constraints and limitations:** Floating-point element types only on the
  current documented surface.

---

##### `pto.vaddreluconv`

- **syntax:** `%result = pto.vaddreluconv %lhs, %rhs : !pto.vreg<NxT0>, !pto.vreg<NxT0> -> !pto.vreg<MxT1>`
- **semantics:** Fused add + ReLU + type conversion (HW fusion).

```c
// f32→f16 variant:
for (int i = 0; i < 64; i++)
    dst_f16[i] = f32_to_f16(max(src0_f32[i] + src1_f32[i], 0));

// f16→i8 variant:
for (int i = 0; i < 128; i++)
    dst_i8[i] = f16_to_i8(max(src0_f16[i] + src1_f16[i], 0));
```

- **inputs:** `%lhs` and `%rhs` are the source vectors.
- **outputs:** `%result` is the fused add/ReLU/convert result.
- **constraints and limitations:** Only backend-supported source/destination
  type pairs are legal. Rounding, saturation, and packing rules follow the
  semantics of this fused operation, not an arbitrary sequence of standalone
  ops.

---

##### `pto.vmulconv`

- **syntax:** `%result = pto.vmulconv %lhs, %rhs : !pto.vreg<NxT0>, !pto.vreg<NxT0> -> !pto.vreg<MxT1>`
- **semantics:** Fused mul + type conversion (HW fusion).

```c
// f16→i8 variant:
for (int i = 0; i < 128; i++)
    dst_i8[i] = f16_to_i8(src0_f16[i] * src1_f16[i]);
```

- **inputs:** `%lhs` and `%rhs` are the source vectors.
- **outputs:** `%result` is the fused mul/convert result.
- **constraints and limitations:** Only backend-supported source/destination
  type pairs are legal.

---

#### Extended Arithmetic

##### `pto.vmull`

- **syntax:** `%low, %high = pto.vmull %lhs, %rhs, %mask : !pto.vreg<NxT>, !pto.vreg<NxT>, !pto.mask -> !pto.vreg<NxT>, !pto.vreg<NxT>`
- **A5 types:** i32/u32 (native 32×32→64 widening multiply)
- **semantics:** Widening multiply with high/low results.

```c
for (int i = 0; i < 64; i++) {
    int64_t r = (int64_t)src0_i32[i] * (int64_t)src1_i32[i];
    dst_lo[i] = (int32_t)(r & 0xFFFFFFFF);
    dst_hi[i] = (int32_t)(r >> 32);
}
```

- **inputs:** `%lhs` and `%rhs` are the source vectors and `%mask` selects
  active lanes.
- **outputs:** `%low` and `%high` expose the widened-product low/high parts.
- **constraints and limitations:** The current documented A5 form is the native
  widening 32x32->64 integer multiply family.

---

##### `pto.vmula`

- **syntax:** `%result = pto.vmula %acc, %lhs, %rhs, %mask : !pto.vreg<NxT>, !pto.vreg<NxT>, !pto.vreg<NxT>, !pto.mask -> !pto.vreg<NxT>`
- **semantics:** Multiply-accumulate.

```c
for (int i = 0; i < N; i++)
    if (mask[i])
        dst[i] = acc[i] + lhs[i] * rhs[i];
```

- **inputs:** `%acc` is the accumulator input, `%lhs` and `%rhs` are the
  multiplicands, and `%mask` selects active lanes.
- **outputs:** `%result` is the multiply-accumulate result.
- **constraints and limitations:** `pto.vmula` is a fused multiply-accumulate
  operation and is not always interchangeable with separate `vmul` plus `vadd`.

---

#### Index Generation

##### `pto.vci`

- **syntax:** `%result = pto.vci %index {order = "ORDER"} : integer -> !pto.vreg<NxT>`
- **semantics:** Generate lane index vector.

```c
for (int i = 0; i < N; i++)
    dst[i] = base_index + i;
```

**Use case:** Generate indices for gather/scatter, argsort, etc.

- **inputs:** `%index` is the scalar seed/base index.
- **outputs:** `%result` is the generated index vector.
- **constraints and limitations:** This page documents the arithmetic/indexing
  use of the family; the conversion page also records the same opcode for
  completeness.

---

#### UB-to-UB Operations

##### `pto.vtranspose`

- **syntax:** `pto.vtranspose %dest, %src, %config : !pto.ptr<T, ub>, !pto.ptr<T, ub>, i64`
- **semantics:** UB-to-UB transpose operation (not vreg-to-vreg).

**Note:** This operates on UB memory directly, not on vector registers.

- **inputs:** `%dest` and `%src` are UB pointers and `%config` is the ISA
  control/config word.
- **outputs:** This op writes UB memory and returns no SSA value.
- **constraints and limitations:** This is not a `vreg -> vreg` op even though
  it lives in the `pto.v*` namespace. Its correctness depends on the control
  word and UB layout contract.

---

#### Sorting Operations

##### `pto.vsort32`

- **syntax:** `pto.vsort32 %dest, %src, %config : !pto.ptr<T, ub>, !pto.ptr<T, ub>, i64`
- **semantics:** Sort 32 elements in UB.
- **inputs:** `%dest` and `%src` are UB pointers and `%config` is the ISA
  control/config word.
- **outputs:** This op writes UB memory and returns no SSA value.
- **constraints and limitations:** This is a UB-to-UB accelerator helper, not a
  pure vector-register op.

---

##### `pto.vmrgsort`

- **syntax:** `pto.vmrgsort4 %dest, %src0, %src1, %src2, %src3, %count, %config : !pto.ptr<T, ub>, !pto.ptr<T, ub> x4, i64, i64`
- **semantics:** Merge-sort 4 pre-sorted input vectors.
- **inputs:** `%dest` is the UB destination, `%src0..%src3` are the four
  pre-sorted UB inputs, `%count` is the number of valid elements, and `%config`
  is the operation control word.
- **outputs:** This op writes UB memory and returns no SSA value.
- **constraints and limitations:** Inputs MUST already be sorted according to
  the sort order encoded by `%config`. This page uses the shorter mnemonic
  `pto.vmrgsort`, while the current implementation summary still refers to
  `pto.vmrgsort4`.

---

#### Current Implementation Surface Summary

- `pto.vmull %lhs, %rhs, %mask : !pto.vreg<NxT>, !pto.vreg<NxT>, !pto.mask -> !pto.vreg<NxT>, !pto.vreg<NxT>`
- `pto.vmula %acc, %lhs, %rhs, %mask : !pto.vreg<NxT>, !pto.vreg<NxT>, !pto.vreg<NxT>, !pto.mask -> !pto.vreg<NxT>`
- `pto.vci %index {order = "ORDER"} : integer -> !pto.vreg<NxT>`
- `pto.vbitsort %dest, %src, %indices, %repeat_times : !pto.ptr<...>, !pto.ptr<...>, !pto.ptr<...>, index`
- `pto.vmrgsort4 %dest, %src0, %src1, %src2, %src3, %count, %config : !pto.ptr<...>, !pto.ptr<...>, !pto.ptr<...>, !pto.ptr<...>, !pto.ptr<...>, i64, i64`

---

#### Typical Usage

```mlir
// Softmax with fused expdiff
%max_broadcast = pto.vlds %ub_max[%c0] {dist = "BRC_B32"} : !pto.ptr<f32, ub> -> !pto.vreg<64xf32>
%exp_stable = pto.vexpdiff %logits, %max_broadcast : !pto.vreg<64xf32>, !pto.vreg<64xf32> -> !pto.vreg<64xf32>

// Leaky ReLU activation
%activated = pto.vlrelu %linear_out, %alpha_scalar, %mask : !pto.vreg<64xf32>, f32, !pto.mask -> !pto.vreg<64xf32>

// Fused residual add + ReLU
%residual = pto.vaddrelu %conv_out, %skip_connection : !pto.vreg<64xf32>, !pto.vreg<64xf32> -> !pto.vreg<64xf32>

// Generate indices for argsort
%indices = pto.vci %c0 {order = "ASC"} : i32 -> !pto.vreg<64xi32>
```

---

<a id="isa-14-shared-arith"></a>

### 14. Arith (Shared MLIR Dialect)

> **Category:** Shared full scalar `arith` surface used around PTO ops
> **Dialect:** `arith`
> **Upstream Reference:** https://mlir.llvm.org/docs/Dialects/ArithOps/

The upstream MLIR `arith` dialect defines primitive arithmetic, comparison, select, and cast operations over signless integer, index, floating-point, and boolean-compatible scalar values. Within PTO micro Instruction code, the full scalar operation surface of `arith` is supported. These ops are used around PTO instructions to build constants, compute offsets and loop bounds, perform general scalar math, derive valid-shape metadata, and form predicates for `scf` control flow.

These ops are part of the documented PTO micro Instruction surface, but they are not PTO ISA instructions.

---

#### Role in PTO micro Instruction Code

- materialize scalar constants used by PTO scalar operands and loop bounds
- compute scalar/index offsets for tensor views, partitioning, and dynamic shapes
- perform general scalar integer and floating-point math outside PTO vector/tile payload operations
- derive scalar predicates that guard `scf.if` or `scf.while`
- apply scalar casts, width changes, bitwise ops, and selects without introducing PTO-specific control ops

Prefer PTO ops for vector or tile payload math. Use `arith` for scalar computation and bookkeeping that surrounds PTO regions.

---

#### Supported Surface

The documented PTO micro Instruction surface supports the full scalar operation surface of upstream `arith`. The upstream `arith` dialect reference remains authoritative for the exhaustive op-by-op syntax and semantics. The categories below summarize how that support is used in PTO micro Instruction code.

| Category | Representative Ops | Typical Use in PTO micro Instruction Code |
|----------|--------------------|------------------|
| Constants | `arith.constant` | integer, floating-point, boolean, and `index` constants |
| Integer / Index Arithmetic | `arith.addi`, `arith.subi`, `arith.muli`, `arith.divsi`, `arith.divui`, `arith.ceildivsi`, `arith.ceildivui`, `arith.floordivsi`, `arith.remsi`, `arith.remui` | offsets, bounds, chunk sizes, scalar math |
| Floating-Point Arithmetic | `arith.addf`, `arith.subf`, `arith.mulf`, `arith.divf`, `arith.negf`, `arith.maximumf`, `arith.minimumf`, `arith.maxnumf`, `arith.minnumf` | scalar math around PTO regions |
| Bitwise / Shift Ops | `arith.andi`, `arith.ori`, `arith.xori`, `arith.shli`, `arith.shrsi`, `arith.shrui` | flags, masks, packed scalar fields |
| Comparisons / Select | `arith.cmpi`, `arith.cmpf`, `arith.select`, `arith.maxsi`, `arith.minui` | predicates, clamps, scalar muxes |
| Casts / Width Changes | `arith.index_cast`, `arith.index_castui`, `arith.extsi`, `arith.extui`, `arith.trunci`, `arith.sitofp`, `arith.uitofp`, `arith.fptosi`, `arith.fptoui`, `arith.extf`, `arith.truncf`, `arith.bitcast` | ABI glue, dynamic-shape plumbing, scalar type adaptation |

---

#### Current PTOAS Coverage

- the current repository examples are still dominated by constants, casts, integer/index arithmetic, compares, and selects because those are the most common surrounding-scalar patterns in existing kernels
- backend-specific tests such as the PTO shared-dialect fixture visibly exercise only a representative subset of `arith` ops in a single path
- the documented PTO micro Instruction source-level contract is nevertheless the full scalar `arith` surface, not just the index-heavy subset that appears most often in current samples

This section therefore uses representative categories and examples instead of pretending that the supported `arith` surface is limited to the currently most common sample patterns.

---

#### Typical Patterns

##### Scalar Setup

```mlir
%c0 = arith.constant 0 : index
%c1 = arith.constant 1 : index
%scale = arith.constant 2.0 : f32
```

##### Dynamic Offset Computation

```mlir
%vrow = arith.index_cast %valid_row : i32 to index
%chunk = arith.muli %row, %c32 : index
%tail = arith.subi %limit, %chunk : index
```

##### General Scalar Arithmetic

```mlir
%sum_i = arith.addi %lhs_i, %rhs_i : i32
%sum_f = arith.addf %lhs_f, %rhs_f : f32
%prod_f = arith.mulf %sum_f, %scale : f32
```

##### Scalar Predicate and Selection

```mlir
%is_first = arith.cmpi eq, %i, %c0 : index
%active = arith.select %is_first, %first_count, %steady_count : index
```

##### Bitwise / Width Adaptation

```mlir
%flags = arith.andi %flags0, %flags1 : i32
%wide = arith.extui %flags : i32 to i64
%shrunk = arith.trunci %wide : i64 to i16
```

---

#### Authoring Guidance

- treat upstream `arith` scalar semantics as the source of truth for supported scalar ops
- keep `arith` values scalar or `index` typed; do not use `arith` as a substitute for PTO vector/tile compute
- use `arith` for general scalar math, scalar comparisons, bitwise operations, and casts around PTO regions, not just for `index` arithmetic
- use `arith.cmpi` / `arith.cmpf` plus `scf.if` / `scf.while` for control flow, not ad hoc control intrinsics
- prefer `arith.index_cast` / `arith.index_castui` at ABI or shape boundaries where `index` is required, but do not read that as a restriction on the rest of scalar `arith`

---

<a id="isa-15-shared-scf"></a>

### 15. SCF (Shared MLIR Dialect)

> **Category:** Shared structured control flow around PTO regions
> **Dialect:** `scf`
> **Upstream Reference:** https://mlir.llvm.org/docs/Dialects/SCFDialect/

The upstream MLIR `scf` dialect defines structured control flow operations with regions, including counted loops, conditional regions, and while-style loops. In PTO micro Instruction code, `scf` is the control shell around PTO ops: it sequences DMA, vector, and tile operations; carries scalar or tile state across iterations; and preserves analyzable control flow for PTO-specific analyses and lowerings.

These ops are part of the documented PTO micro Instruction surface, but they are shared MLIR control-flow constructs rather than PTO ISA instructions.

---

#### Supported Ops

| Op | Role in PTO micro Instruction Code | Notes |
|----|------------------------|-------|
| `scf.for` | counted loops and loop-carried values | also used for `__VEC_SCOPE__` dummy-loop form |
| `scf.if` | structured conditional execution | may yield values or act as side-effect-only branch |
| `scf.yield` | region terminator for `for` / `if` / `while` bodies | carries loop or branch results |
| `scf.while` | break-like or stateful loops | useful for source-level structured control |
| `scf.condition` | loop-continue / loop-exit decision for `scf.while` | placed in the "before" region |

Ops such as `scf.execute_region`, `scf.forall`, or `scf.index_switch` are not part of the documented shared-dialect portion of the PTO micro Instruction surface here.

---

#### Current PTOAS Coverage

- `scf.for`, `scf.if`, and `scf.yield` are directly exercised in the shared-dialect PTO fixture and appear widely across PTO samples
- the `__VEC_SCOPE__` contract in PTO micro Instruction is modeled as a specialized `scf.for` annotated with `llvm.loop.aivector_scope`
- PTO synchronization and memory analyses explicitly reason about `scf.for`, `scf.if`, `scf.yield`, and `scf.while`
- `scf.while` and `scf.condition` appear in control-flow samples and are handled in PTO-to-EmitC control-flow lowering, but they are less broadly exercised than `for` / `if` on all backend paths

---

#### Typical Patterns

##### Counted Loop

```mlir
scf.for %i = %c0 to %c4 step %c1 {
  %offset = arith.muli %i, %c32 : index
  %mask = pto.pset_b32 "PAT_ALL" : !pto.mask
  %v = pto.vlds %ub[%offset] : !pto.ptr<f32, ub> -> !pto.vreg<64xf32>
  %abs = pto.vabs %v, %mask : !pto.vreg<64xf32>, !pto.mask -> !pto.vreg<64xf32>
  pto.vsts %abs, %ub_out[%offset], %mask : !pto.vreg<64xf32>, !pto.ptr<f32, ub>, !pto.mask
}
```

##### Counted Loop with Loop-Carried State

```mlir
%final_alive = scf.for %i = %c0 to %c4 step %c1
    iter_args(%alive = %true) -> (i1) {
  %break_now = arith.cmpi eq, %i, %c2 : index
  %next_alive = scf.if %break_now -> (i1) {
    scf.yield %false : i1
  } else {
    scf.yield %alive : i1
  }
  scf.yield %next_alive : i1
}
```

##### Structured Conditional Region

```mlir
%is_mode_a = arith.cmpi eq, %mode, %c0_i32 : i32
scf.if %is_mode_a {
  pto.tmuls ins(%data, %scale_a : !pto.tile_buf<...>, f32) outs(%data : !pto.tile_buf<...>)
} else {
  pto.tadds ins(%data, %bias_b : !pto.tile_buf<...>, f32) outs(%data : !pto.tile_buf<...>)
}
```

##### While-Style Break Loop

```mlir
%final:2 = scf.while (%i = %c0, %alive = %true) : (index, i1) -> (index, i1) {
  %lt = arith.cmpi slt, %i, %c4 : index
  %go = arith.andi %lt, %alive : i1
  scf.condition(%go) %i, %alive : index, i1
} do {
^bb0(%i2: index, %alive2: i1):
  %next_i = arith.addi %i2, %c1 : index
  scf.yield %next_i, %alive2 : index, i1
}
```

---

#### Authoring Guidance

- use `scf.for` for regular counted loops and loop-carried scalar/tile state
- use `scf.if` for structured branching around PTO regions instead of inventing PTO-specific branch ops
- keep region results explicit with `scf.yield`; this is important for PTO analyses that track carried buffers and aliasing
- use `scf.while` only when a counted loop cannot express the control cleanly; `scf.for` remains the more common and better-exercised form in the current repository
- build branch predicates and loop conditions with `arith` ops, not PTO vector masks, unless the control decision truly comes from a scalarized value

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

Group 14 covers the full scalar `arith` surface. The rows below list common PTO micro Instruction patterns rather than an exhaustive partition of `arith` ops.

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
