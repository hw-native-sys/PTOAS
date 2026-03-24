# VPTO Spec — Merged Draft (A5)

> **Status:** DRAFT for review
> **Base:** [vpto-spec.md](https://github.com/mouliangyu/PTOAS/blob/feature-vpto-backend/docs/vpto-spec.md) (2026-03-20)
> **Additions from:** [a5_intrinsic_ir.md](../a5_intrinsic/a5_intrinsic_ir.md) v3.2 (2026-03-21)
> **Updated:** 2026-03-24

---

## Part I: Architecture Overview

### Overview

This document defines the Vector PTO (VPTO) Intermediate Representation (IR), a
compiler-internal and externally facing specification designed to represent
vector compute kernels within the PTO architecture. Much like NVVM provides a
robust IR for GPU architectures, VPTO serves as the direct bridge between
high-level programming models and the underlying hardware ISA, providing a
precise, low-level representation of vector workloads explicitly designed for
the Ascend 950 architecture.

#### Position in the Stack and Layer Modeled

VPTO operates as a very low-level intermediate representation within the PTO
compiler stack. It is uniquely designed to accurately and comprehensively
express all architectural information of the Ascend 950 hardware. It
specifically models the bare-metal vector execution layer, making
hardware-specific capabilities and constraints, such as exact vector lane
configurations, memory space hierarchies, and hardware-specific fusion
semantics, fully transparent and controllable.

#### Why External Developers Read or Author VPTO

While the majority of users will interact with the PTO architecture via
higher-level frameworks, external developers may need to read or author VPTO IR
directly for several key reasons:

- Custom Toolchain Development:
  build custom compiler frontends or domain-specific languages (DSLs) that
  target the Ascend 950 architecture with maximum hardware utilization.
- Performance Engineering:
  inspect the output of high-level compiler passes, verify fine-grained
  optimization behaviors, and pinpoint performance bottlenecks at the
  architectural level.
- Micro-Optimization:
  hand-author highly optimized, critical mathematical kernels using a stable,
  precise IR when higher-level abstractions cannot achieve the theoretical peak
  performance of the hardware.

#### Relationship to CCE

VPTO is designed to express the full semantic capabilities of the Compute Cube
Engine (CCE), but with significant structural and pipeline advantages for
compiler development.

- Bypassing the C/Clang Pipeline:
  while CCE heavily relies on C/C++ extensions parsed by Clang, VPTO operates
  entirely independently of the C language frontend. By bypassing Clang AST
  generation and frontend processing, utilizing VPTO significantly reduces
  overall compilation time and memory overhead.
- Enhanced IR Verification:
  because VPTO is a strongly typed, SSA-based (Static Single Assignment)
  compiler IR rather than a C-wrapper API, it provides a much more rigorous and
  detailed IR verification process. Structural inconsistencies, invalid memory
  access patterns, and operand type mismatches are caught immediately with
  precise, explicit diagnostic feedback, providing developers with much higher
  visibility into kernel correctness than traditional CCE error reporting.

#### Intended Audience

This document is written for compiler engineers, library writers, and advanced
performance architects. We expect the reader to have a working understanding of
modern compiler infrastructure, specifically MLIR, the principles of Static
Single Assignment (SSA) form, and a deep understanding of the vector-processing
capabilities of the Ascend 950 architecture.

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

**Address Space Isolation**: The IR uses LLVM pointer address spaces to distinguish between GM (`!llvm.ptr<1>`) and UB (`!llvm.ptr<6>`). The verifier ensures that no compute operation attempts to access GM directly.

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
│        (On-chip SRAM, 256KB, AS=6)           │
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

| Pattern | Instructions | Description |
|---------|--------------|-------------|
| Contiguous | `pto.vlds` NORM | Sequential 256B access |
| Strided | `pto.vsld` | Fixed stride pattern access |
| Block-Strided | `pto.vsldb`, `pto.vsstb` | 2D tile access pattern |
| Unaligned | `pto.vldas` + `pto.vldus` | Non-aligned address handling via align state |
| Broadcast | `pto.vlds` BRC_* | Single element → all lanes |
| Upsample | `pto.vlds` US_* | Each element duplicated to 2 lanes |
| Downsample | `pto.vlds` DS_* | Every 2nd element selected |
| Pack/Unpack | `pto.vlds` UNPK_* / `pto.vsts` PK_* | Narrowing/widening on load/store |
| Interleave | `pto.vstx2` INTLV_* / `pto.vldx2` DINTLV_* | AoS ↔ SoA conversion |
| Channel Split/Merge | `pto.vlds` SPLT* / `pto.vsts` MRG* | Multi-channel deinterleave/interleave |
| Gather/Scatter | `pto.vgather2`, `pto.vscatter` | Indirect indexed access |
| Squeeze/Expand | `pto.vsqz`, `pto.vusqz` | Compress/expand by mask |

#### Synchronization Model

VPTO provides two levels of synchronization:

**Inter-Pipeline Synchronization (MTE ↔ Vector):**

Because the MTE and Vector pipelines operate asynchronously, VPTO utilizes a Flag/Event mechanism. Developers must explicitly insert `pto.set_flag` and `pto.wait_flag` operations to resolve Read-After-Write (RAW) and Write-After-Read (WAR) hazards between memory staging and computation.

For enhanced inter-pipeline coordination on Ascend 950, `pto.get_buf` and `pto.rls_buf` provide a finer-grained synchronization mechanism that coordinates pipeline execution through buffer acquisition and release semantics.

**Intra-Pipeline Memory Barriers (within `__VEC_SCOPE__`):**

Within the vector execution scope, the hardware does not track UB address aliasing. When UB addresses overlap or alias between vector load/store operations, explicit memory barriers are required:

```c
pto.mem_bar "VV_ALL"      // All prior vector ops complete before subsequent
pto.mem_bar "VST_VLD"     // All prior vector stores visible before subsequent loads
pto.mem_bar "VLD_VST"     // All prior vector loads complete before subsequent stores
```

Without proper barriers, loads may see stale data or stores may be reordered incorrectly.

#### Predication Model

Vector compute and load/store instructions support **predicated execution** via `!pto.mask`:

```
dst[i] = mask[i] ? op(src0[i], src1[i]) : 0    // ZEROING mode
```

In ZEROING mode, inactive lanes produce zero. This is the native hardware predication mode.

#### Execution Scopes (__VEC_SCOPE__)

`__VEC_SCOPE__` is the IR-level representation of a Vector Function (VF)
launch. In the PTO architecture, it defines the hardware interface between the
Scalar Unit and the Vector Thread.

It is not a dedicated `pto` op. In VPTO IR, this scope is modeled as a
specialized `scf.for` loop annotated with `llvm.loop.aivector_scope`. This
gives the compiler a natural structural boundary for identifying the code block
that must be lowered into a discrete VF hardware instruction sequence.

**Scalar-Vector Interface:**

The execution model follows non-blocking fork semantics:

- Scalar invocation:
  the scalar processor invokes a vector thread by calling a VF. Once the launch
  command is issued, the scalar unit does not stall and continues executing
  subsequent instructions in the pipeline.
- Vector execution:
  after invocation, the vector thread independently fetches and executes the
  instructions defined within the VF scope.
- Parallelism:
  this decoupled execution allows the scalar and vector units to run in
  parallel, so the scalar unit can prepare addresses or manage control flow
  while the vector unit performs heavy SIMD computation.

**Launch Mechanism And Constraints:**

- Parameter buffering:
  all arguments required by the VF must be staged in hardware-specific buffers.
- Launch overhead:
  launching a VF incurs a latency of a few cycles. Very small VFs should
  account for this overhead because launch cost can rival useful computation time.

**MLIR Representation:**

```mlir
scf.for %dummy = %c0 to %c1 step %c1 {
  %v = pto.vlds %ub[%lane] : !llvm.ptr<6> -> !pto.vreg<64xf32>
  %abs = pto.vabs %v : !pto.vreg<64xf32> -> !pto.vreg<64xf32>
  pto.vsts %abs, %ub_out[%lane] : !pto.vreg<64xf32>, !llvm.ptr<6>
} {llvm.loop.aivector_scope}
```

### Example: Abs

```mlir
pto.set_loop2_stride_outtoub %c4096_i64, %c4096_i64 : i64, i64
pto.set_loop1_stride_outtoub %c4096_i64, %c4096_i64 : i64, i64
pto.set_loop_size_outtoub %c1_i64, %c1_i64 : i64, i64
pto.copy_gm_to_ubuf %7, %2, %3, %3, %c0_i64, %c32_i64, %4, %c0_i64, %c0_i64, %c0_i64, %c128_i64, %c128_i64
    {data_select_bit = false, layout = "nd", ub_pad = false}
    : !llvm.ptr<1>, !llvm.ptr<6>, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64

pto.set_flag["PIPE_MTE2", "PIPE_V", "EVENT_ID0"]
pto.wait_flag["PIPE_MTE2", "PIPE_V", "EVENT_ID0"]

scf.for %dummy = %c0 to %c1 step %c1 {
  scf.for %lane = %c0 to %9 step %c64 {
    %v = pto.vlds %2[%lane] : !llvm.ptr<6> -> !pto.vreg<64xf32>
    %abs = pto.vabs %v : !pto.vreg<64xf32> -> !pto.vreg<64xf32>
    pto.vsts %abs, %8[%lane] : !pto.vreg<64xf32>, !llvm.ptr<6>
  }
} {llvm.loop.aivector_scope}

pto.set_flag["PIPE_V", "PIPE_MTE3", "EVENT_ID0"]
pto.wait_flag["PIPE_V", "PIPE_MTE3", "EVENT_ID0"]
pto.set_loop_size_ubtoout %c1_i64, %c1_i64 : i64, i64
pto.set_loop1_stride_ubtoout %c4096_i64, %c4096_i64 : i64, i64
pto.set_loop2_stride_ubtoout %c4096_i64, %c4096_i64 : i64, i64
pto.copy_ubuf_to_gm %8, %14, %3, %3, %c0_i64, %c32_i64, %4, %c0_i64, %c128_i64, %c128_i64
    {layout = "nd"}
    : !llvm.ptr<6>, !llvm.ptr<1>, i64, i64, i64, i64, i64, i64, i64, i64
```

### Scope

This document is the interface specification for the `mlir::pto` dialect.

It only describes:

- operation names
- operand and result lists
- operand and result types
- important attributes
- corresponding CCE builtin or CCE wrapper family
- C-style semantics for each operation

It does not describe lowering strategy.

### Core Types

- `vreg<T>`: `!pto.vreg<NxT>`
  Fixed-width VPTO vector type with total width exactly 256 bytes (2048 bits).
  `N` is the lane count, `T` is the element type, and `N * bitwidth(T) = 2048`.
- `mask`: `!pto.mask`
  Models an A5 predicate register (256-bit). Per-lane enable/disable state.
- `align`: `!pto.align`
  Models the A5 vector-align carrier state for unaligned load/store sequences.
- `buf`: `!llvm.ptr<AS>`
  Buffer-like LLVM pointer type. AS=1 for GM, AS=6 for UB.
- `idx`: `index`
- `i32`: `i32`
- `i64`: `i64`

### Address Space Conventions

| `AS` | PTO mnemonic | Interpretation |
|------|--------------|----------------|
| `0` | `Zero` | Default / unspecified (treated as GM-like) |
| `1` | `GM` | Global Memory (GM) |
| `2` | `MAT` | Matrix / L1-related storage |
| `3` | `LEFT` | Left matrix buffer / L0A |
| `4` | `RIGHT` | Right matrix buffer / L0B |
| `5` | `ACC` | Accumulator / L0C |
| `6` | `VEC` | Unified Buffer (UB) / vector buffer (256KB) |
| `7` | `BIAS` | Bias buffer |
| `8` | `SCALING` | Scaling buffer |

### Element Type Constraints

| Type | Bits | Description |
|------|------|-------------|
| `i8` / `u8` | 8 | Signed/unsigned 8-bit integer |
| `i16` / `u16` | 16 | Signed/unsigned 16-bit integer |
| `i32` / `u32` | 32 | Signed/unsigned 32-bit integer |
| `i64` / `u64` | 64 | Signed/unsigned 64-bit integer |
| `f16` | 16 | IEEE 754 half precision |
| `bf16` | 16 | Brain floating point |
| `f32` | 32 | IEEE 754 single precision |
| `f8e4m3` | 8 | FP8 (4-bit exponent, 3-bit mantissa) |
| `f8e5m2` | 8 | FP8 (5-bit exponent, 2-bit mantissa) |
| `f8e8m0` | 8 | FP8 scale factor (8-bit exponent only) |
| `f4e2m1` | 4 | FP4 (2-bit exponent, 1-bit mantissa) |
| `f4e1m2` | 4 | FP4 (1-bit exponent, 2-bit mantissa) |

Valid `!pto.vreg<NxT>` configurations: `N * bitwidth(T) = 2048`

### Special Types

#### `!pto.mask`

`!pto.mask` models an A5 predicate register, not an integer vector.

- producers:
  `pto.vpset_b8`, `pto.vpset_b16`, `pto.vpset_b32`,
  `pto.vpge_b8`, `pto.vpge_b16`, `pto.vpge_b32`,
  `pto.vplds`, `pto.vpld`, `pto.vpldi`,
  `pto.vcmp`, `pto.vcmps`
- consumers:
  `pto.vsel`,
  `pto.vaddc`, `pto.vsubc`, `pto.vaddcs`, `pto.vsubcs`,
  `pto.vpnot`, `pto.vpsel`,
  `pto.vgather2_bc`,
  `pto.vstx2`, `pto.vsstb`,
  `pto.vpsts`, `pto.vpst`, `pto.vpsti`,
  `pto.vpstu`,
  `pto.vmula`

```mlir
%mask = pto.vcmp %lhs, %rhs, %seed, "lt" : !pto.vreg<64xf32>, !pto.vreg<64xf32>, !pto.mask -> !pto.mask
%out = pto.vsel %x, %y, %mask : !pto.vreg<64xf32>, !pto.vreg<64xf32>, !pto.mask -> !pto.vreg<64xf32>
```

#### `!pto.align`

`!pto.align` models the A5 vector-align carrier state. It is not payload data.

- producers: `pto.vldas`, `pto.vpstu`, `pto.vstu`, `pto.vstus`, `pto.vstur`
- consumers: `pto.vldus`, `pto.vsta`, `pto.vstas`, `pto.vstar`, `pto.vpstu`, `pto.vstu`, `pto.vstus`, `pto.vstur`

```mlir
%align = pto.vldas %ub[%c0] : !llvm.ptr<6> -> !pto.align
%vec = pto.vldus %align, %ub[%c64] : !pto.align, !llvm.ptr<6> -> !pto.vreg<64xf32>
```

### Implemented String Constraints

#### Predicate Patterns

Used by `pto.vpset_b*`, `pto.vpge_b*`:
`PAT_ALL | PAT_VL1 | PAT_VL2 | PAT_VL3 | PAT_VL4 | PAT_VL8 | PAT_VL16 | PAT_VL32 | PAT_VL64 | PAT_VL128 | PAT_M3 | PAT_M4 | PAT_H | PAT_Q | PAT_ALLF`

#### Distribution Tokens

| Op | Allowed Values |
|----|---------------|
| `pto.vlds` | `NORM \| BLK \| DINTLV_B32 \| UNPK_B16 \| BRC_B8 \| BRC_B16 \| BRC_B32 \| US_B8 \| US_B16 \| DS_B8 \| DS_B16 \| SPLT4CHN_B8 \| SPLT2CHN_B8 \| SPLT2CHN_B16 \| UNPK_B8 \| UNPK_B32` |
| `pto.vpld`, `pto.vpldi` | `NORM \| US \| DS` |
| `pto.vpst`, `pto.vpsti` | `NORM \| PK` |
| `pto.vldx2` | `DINTLV_B8 \| DINTLV_B16 \| DINTLV_B32 \| BDINTLV` |
| `pto.vstx2` | `INTLV_B8 \| INTLV_B16 \| INTLV_B32` |
| `pto.vsts` | `NORM_B8 \| NORM_B16 \| NORM_B32 \| PK_B16 \| PK_B32 \| MRG4CHN_B8 \| MRG2CHN_B8 \| MRG2CHN_B16` |

#### Stride Tokens

Used by `pto.vsld`, `pto.vsst`:
`STRIDE_S3_B16 | STRIDE_S4_B64 | STRIDE_S8_B32 | STRIDE_S2_B64 | STRIDE_VSST_S8_B16`

#### Compare Modes

Used by `pto.vcmp`, `pto.vcmps`:
`eq | ne | lt | le | gt | ge`

#### Part Tokens

Used by `pto.vppack`, `pto.vpunpack`:
`LOWER | HIGHER`

#### Mode Tokens

Used by `pto.vmula`:
`MODE_ZEROING | MODE_UNKNOWN | MODE_MERGING`

Used by `pto.vstu`, `pto.vstus`, `pto.vstur`:
`POST_UPDATE | NO_POST_UPDATE`

#### Conversion Control Tokens

- Round mode (`pto.vcvt`): `ROUND_R | ROUND_A | ROUND_F | ROUND_C | ROUND_Z | ROUND_O`
- Saturation (`pto.vcvt`): `RS_ENABLE | RS_DISABLE`
- Part (`pto.vcvt`): `PART_EVEN | PART_ODD`

#### Memory Barrier Types

Used by `pto.mem_bar`:
`VV_ALL | VST_VLD | VLD_VST`

### Correspondence Categories

- `direct builtin`: maps to one CCE builtin family (`__builtin_cce_*`)
- `wrapper family`: corresponds to a CCE wrapper that may dispatch to multiple builtins

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
%result = pto.vlds %source[%offset] {dist = "DIST"} : !llvm.ptr<6> -> !pto.vreg<NxT>
```

**Store (register to memory):**

```mlir
pto.vsts %value, %destination[%offset] {dist = "DIST"} : !pto.vreg<NxT>, !llvm.ptr<6>
```

**Dual Load (one load, two results — deinterleave):**

```mlir
%low, %high = pto.vldx2 %source[%offset], "DIST" : !llvm.ptr<6>, index -> !pto.vreg<NxT>, !pto.vreg<NxT>
```

**Dual Store (two inputs, one interleaved store):**

```mlir
pto.vstx2 %low, %high, %dest[%offset], "DIST", %mask : !pto.vreg<NxT>, !pto.vreg<NxT>, !llvm.ptr<6>, index, !pto.mask
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
%mask = pto.vpset_b32 "PAT_ALL" : !pto.mask
%mask = pto.vpge_b32 "PAT_VL16" : !pto.mask
```

**Sync operations:**

```mlir
pto.set_flag["PIPE_MTE2", "PIPE_V", "EVENT_ID0"]
pto.wait_flag["PIPE_MTE2", "PIPE_V", "EVENT_ID0"]
pto.mem_bar "VV_ALL"
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
| `"DIST"` | Distribution mode string (see String Constraints in Part I) |
| `"CMP_MODE"` | Compare predicate: `eq \| ne \| lt \| le \| gt \| ge` |
| `"ROUND_MODE"` | Rounding mode: `ROUND_R \| ROUND_A \| ROUND_F \| ROUND_C \| ROUND_Z` |
| `"SAT_MODE"` | Saturation: `RS_ENABLE \| RS_DISABLE` |
| `"PART_MODE"` | Half selector: `PART_EVEN \| PART_ODD` |
| `"PAT_*"` | Predicate pattern literal |
| `"MODE"` | Mode selector: `MODE_ZEROING \| MODE_MERGING` |
| `T` | Element type (f32, f16, bf16, i32, i16, i8, etc.) |
| `N` | Lane count (`N * bitwidth(T) = 2048`) |

---

## Part III: ISA Instruction Reference

### 1. Sync And Buffer Control

#### `pto.set_flag`

- syntax: `pto.set_flag["SRC_PIPE", "DST_PIPE", "EVENT_ID"]`
- semantics: Signal event from source pipe to destination pipe.
- CCE: `__builtin_cce_set_flag`

```c
set_flag(src_pipe, dst_pipe, event_id);
```

#### `pto.wait_flag`

- syntax: `pto.wait_flag["SRC_PIPE", "DST_PIPE", "EVENT_ID"]`
- semantics: Block destination pipe until source pipe signals event.
- CCE: `__builtin_cce_wait_flag`

```c
wait_flag(src_pipe, dst_pipe, event_id);
```

#### `pto.pipe_barrier`

- syntax: `pto.pipe_barrier "PIPE_*"`
- semantics: Drain all pending ops in the specified pipe.
- CCE: `__builtin_cce_pipe_barrier`

```c
pipe_barrier(pipe);
```

#### `pto.get_buf`

- syntax: `pto.get_buf "PIPE_*", %buf_id, %mode : i64, i64`
- semantics: Acquire buffer for inter-pipeline synchronization on Ascend 950.
- CCE: `__builtin_cce_get_buf`

```c
get_buf(pipe, buf_id, mode);
```

#### `pto.rls_buf`

- syntax: `pto.rls_buf "PIPE_*", %buf_id, %mode : i64, i64`
- semantics: Release buffer for inter-pipeline synchronization on Ascend 950.
- CCE: `__builtin_cce_rls_buf`

```c
rls_buf(pipe, buf_id, mode);
```

#### `pto.mem_bar`

- syntax: `pto.mem_bar "BARRIER_TYPE"`
- semantics: Intra-vector-pipe memory fence within `__VEC_SCOPE__`. Required when UB addresses alias between vector load/store operations.
- CCE: `__builtin_cce_pipe_barrier` (PIPE_V context)

```c
mem_bar(barrier_type);
// VV_ALL:   all prior vector ops complete before subsequent
// VST_VLD:  all prior stores visible before subsequent loads
// VLD_VST:  all prior loads complete before subsequent stores
```

---

### 2. Copy Programming

#### `pto.set_loop2_stride_outtoub`

- syntax: `pto.set_loop2_stride_outtoub %first, %second : i64, i64`
- semantics: Configure outer loop stride for GM→UB DMA.
- CCE: `__builtin_cce_set_loop2_stride_outtoub`

#### `pto.set_loop1_stride_outtoub`

- syntax: `pto.set_loop1_stride_outtoub %first, %second : i64, i64`
- semantics: Configure inner loop stride for GM→UB DMA.
- CCE: `__builtin_cce_set_loop1_stride_outtoub`

#### `pto.set_loop_size_outtoub`

- syntax: `pto.set_loop_size_outtoub %first, %second : i64, i64`
- semantics: Configure loop iteration count for GM→UB DMA.
- CCE: `__builtin_cce_set_loop_size_outtoub`

#### `pto.set_loop2_stride_ubtoout`

- syntax: `pto.set_loop2_stride_ubtoout %first, %second : i64, i64`
- semantics: Configure outer loop stride for UB→GM DMA.
- CCE: `__builtin_cce_set_loop2_stride_ubtoout`

#### `pto.set_loop1_stride_ubtoout`

- syntax: `pto.set_loop1_stride_ubtoout %first, %second : i64, i64`
- semantics: Configure inner loop stride for UB→GM DMA.
- CCE: `__builtin_cce_set_loop1_stride_ubtoout`

#### `pto.set_loop_size_ubtoout`

- syntax: `pto.set_loop_size_ubtoout %first, %second : i64, i64`
- semantics: Configure loop iteration count for UB→GM DMA.
- CCE: `__builtin_cce_set_loop_size_ubtoout`

---

### 3. Copy Transfers

#### `pto.copy_gm_to_ubuf`

- syntax: `pto.copy_gm_to_ubuf %source, %destination, %valid_rows, %valid_cols, %sid, %n_burst, %len_burst, %left_padding, %right_padding, %l2_cache_ctl, %gm_stride, %ub_stride {layout = "LAYOUT", data_select_bit = true|false, ub_pad = true|false} : !llvm.ptr<1>, !llvm.ptr<6>, i64 x10`
- semantics: DMA transfer from Global Memory to Unified Buffer.
- CCE: `__builtin_cce_copy_gm_to_ubuf_align_v2`

#### `pto.copy_ubuf_to_ubuf`

- syntax: `pto.copy_ubuf_to_ubuf %source, %destination, %sid, %n_burst, %len_burst, %src_stride, %dst_stride : !llvm.ptr<6>, !llvm.ptr<6>, i64 x5`
- semantics: Copy within Unified Buffer.
- CCE: `__builtin_cce_copy_ubuf_to_ubuf`

#### `pto.copy_ubuf_to_gm`

- syntax: `pto.copy_ubuf_to_gm %source, %destination, %valid_rows, %valid_cols, %sid, %n_burst, %len_burst, %reserved, %burst_dst_stride, %burst_src_stride {layout = "LAYOUT"} : !llvm.ptr<6>, !llvm.ptr<1>, i64 x8`
- semantics: DMA transfer from Unified Buffer to Global Memory.
- CCE: `__builtin_cce_copy_ubuf_to_gm_align_v2`

---

### 4. Vector, Predicate And Align Loads

#### `pto.vlds`

- syntax: `%result = pto.vlds %source[%offset] {dist = "DIST"} : !llvm.ptr<6> -> !pto.vreg<NxT>`
- semantics: Vector load with distribution mode.
- CCE: `__builtin_cce_vldsx1_*`

```c
// NORM: contiguous load
for (int i = 0; i < N; i++)
    dst[i] = UB[base + offset + i * sizeof(T)];

// BRC_B32: broadcast single element to all lanes
for (int i = 0; i < N; i++)
    dst[i] = UB[base];

// US_B16: upsample — duplicate each element to 2 lanes
for (int i = 0; i < N/2; i++)
    dst[2*i] = dst[2*i+1] = UB[base + i * sizeof(T)];

// DS_B16: downsample — take every 2nd element
for (int i = 0; i < N; i++)
    dst[i] = UB[base + 2*i * sizeof(T)];

// UNPK_B16: unpack 16-bit → 32-bit (zero-extend)
for (int i = 0; i < 64; i++)
    dst_i32[i] = (uint32_t)UB_i16[base + 2*i];

// SPLT4CHN_B8: split 4-channel interleaved (e.g. RGBA → R plane)
// SPLT2CHN_B8/B16: split 2-channel interleaved
```

#### `pto.vldas`

- syntax: `%result = pto.vldas %source[%offset] : !llvm.ptr<6> -> !pto.align`
- semantics: Prime alignment buffer for subsequent unaligned load.
- CCE: `__builtin_cce_vldas_*`

#### `pto.vldus`

- syntax: `%result = pto.vldus %align, %source[%offset] : !pto.align, !llvm.ptr<6> -> !pto.vreg<NxT>`
- semantics: Unaligned load using primed align state.
- CCE: `__builtin_cce_vldus_*`

#### `pto.vplds`

- syntax: `%result = pto.vplds %source[%offset] {dist = "DIST"} : !llvm.ptr<6> -> !pto.mask`
- semantics: Load predicate register with scalar offset.
- CCE: `__builtin_cce_plds_b8`

#### `pto.vpld`

- syntax: `%result = pto.vpld %source[%offset], "DIST" : !llvm.ptr<6>, index -> !pto.mask`
- semantics: Load predicate register with areg offset.
- CCE: `__builtin_cce_pld_b8`

#### `pto.vpldi`

- syntax: `%result = pto.vpldi %source, %offset, "DIST" : !llvm.ptr<6>, i32 -> !pto.mask`
- semantics: Load predicate register with immediate offset.
- CCE: `__builtin_cce_pldi_b8`

#### `pto.vldx2`

- syntax: `%low, %high = pto.vldx2 %source[%offset], "DIST" : !llvm.ptr<6>, index -> !pto.vreg<NxT>, !pto.vreg<NxT>`
- semantics: Dual load with deinterleave (AoS → SoA).
- CCE: `__builtin_cce_vldx2_*`

```c
// DINTLV_B32: deinterleave 32-bit elements
for (int i = 0; i < 64; i++) {
    low[i]  = UB[base + 8*i];       // even elements
    high[i] = UB[base + 8*i + 4];   // odd elements
}
```

#### `pto.vgather2`

- syntax: `%result = pto.vgather2 %source, %offsets, %active_lanes : !llvm.ptr<6>, !pto.vreg<NxI>, index -> !pto.vreg<NxT>`
- semantics: Indexed gather from UB.
- CCE: `__builtin_cce_vgather2_*`

```c
for (int i = 0; i < active_lanes; i++)
    dst[i] = UB[base + offsets[i] * sizeof(T)];
```

#### `pto.vgatherb`

- syntax: `%result = pto.vgatherb %source, %offsets, %active_lanes : !llvm.ptr<6>, !pto.vreg<NxI>, index -> !pto.vreg<NxT>`
- semantics: Byte-granularity indexed gather from UB.
- CCE: `__builtin_cce_vgatherb_*`

```c
for (int i = 0; i < active_lanes; i++)
    dst[i] = UB[base + offsets[i]];  // byte-addressed
```

#### `pto.vgather2_bc`

- syntax: `%result = pto.vgather2_bc %source, %offsets, %mask : !llvm.ptr<6>, !pto.vreg<NxI>, !pto.mask -> !pto.vreg<NxT>`
- semantics: Gather with broadcast, conditioned by mask.
- CCE: `__builtin_cce_vgather2_bc_*`

#### `pto.vsld`

- syntax: `%result = pto.vsld %source[%offset], "STRIDE" : !llvm.ptr<6> -> !pto.vreg<NxT>`
- semantics: Strided load with fixed stride pattern.
- CCE: `__builtin_cce_vsld_*`

#### `pto.vsldb`

- syntax: `%result = pto.vsldb %source, %offset, %mask : !llvm.ptr<6>, i32, !pto.mask -> !pto.vreg<NxT>`
- semantics: Block-strided load for 2D tile access.
- CCE: `__builtin_cce_vsldb_*`

---

### 5. Materialization And Predicate Construction

#### `pto.vbr`

- syntax: `%result = pto.vbr %value : T -> !pto.vreg<NxT>`
- semantics: Broadcast scalar to all vector lanes (materialization).
- CCE: broadcast/materialization family

```c
for (int i = 0; i < N; i++)
    dst[i] = value;
```

#### `pto.vdup`

- syntax: `%result = pto.vdup %input {position = "POSITION", mode = "MODE"} : T|!pto.vreg<NxT> -> !pto.vreg<NxT>`
- semantics: Duplicate scalar or vector element to all lanes.
- CCE: `__builtin_cce_vdup_*`

```c
for (int i = 0; i < N; i++)
    dst[i] = input_scalar_or_element;
```

#### `pto.vdupi`

- syntax: `%result = pto.vdupi %imm : i32 -> !pto.vreg<NxT>`
- semantics: Broadcast immediate constant to all lanes.
- CCE: immediate broadcast family

```c
for (int i = 0; i < N; i++)
    dst[i] = (T)imm;
```

#### `pto.vpset_b8` / `pto.vpset_b16` / `pto.vpset_b32`

- syntax: `%result = pto.vpset_b32 "PAT_*" : !pto.mask`
- semantics: Generate predicate from pattern.
- CCE: `__builtin_cce_pset_b8/b16/b32`

```c
// PAT_ALL:  all lanes active
// PAT_ALLF: all lanes inactive
// PAT_H:    high half active
// PAT_Q:    upper quarter active
// PAT_VLn:  first n lanes active
```

#### `pto.vpge_b8` / `pto.vpge_b16` / `pto.vpge_b32`

- syntax: `%result = pto.vpge_b32 "PAT_*" : !pto.mask`
- semantics: Generate tail mask — first N lanes active.
- CCE: `__builtin_cce_pge_b8/b16/b32`

```c
for (int i = 0; i < TOTAL_LANES; i++)
    mask[i] = (i < len);
```

#### `pto.vppack`

- syntax: `%result = pto.vppack %input, "PART" : !pto.mask -> !pto.mask`
- semantics: Narrowing pack of predicate register.
- CCE: `ppack(...)`

#### `pto.vpunpack`

- syntax: `%result = pto.vpunpack %input, "PART" : !pto.mask -> !pto.mask`
- semantics: Widening unpack of predicate register.
- CCE: `punpack(...)`

#### `pto.vpand`

- syntax: `%result = pto.vpand %src0, %src1, %mask : !pto.mask, !pto.mask, !pto.mask -> !pto.mask`
- semantics: Predicate bitwise AND.

```c
for (int i = 0; i < N; i++)
    if (mask[i]) dst[i] = src0[i] & src1[i];
```

#### `pto.vpor`

- syntax: `%result = pto.vpor %src0, %src1, %mask : !pto.mask, !pto.mask, !pto.mask -> !pto.mask`
- semantics: Predicate bitwise OR.

```c
for (int i = 0; i < N; i++)
    if (mask[i]) dst[i] = src0[i] | src1[i];
```

#### `pto.vpxor`

- syntax: `%result = pto.vpxor %src0, %src1, %mask : !pto.mask, !pto.mask, !pto.mask -> !pto.mask`
- semantics: Predicate bitwise XOR.

```c
for (int i = 0; i < N; i++)
    if (mask[i]) dst[i] = src0[i] ^ src1[i];
```

#### `pto.vpmov`

- syntax: `%result = pto.vpmov %src, %mask : !pto.mask, !pto.mask -> !pto.mask`
- semantics: Predicate move (copy under mask).

```c
for (int i = 0; i < N; i++)
    if (mask[i]) dst[i] = src[i];
```

#### `pto.vpintlv`

- syntax: `%low, %high = pto.vpintlv %src0, %src1 : !pto.mask, !pto.mask -> !pto.mask, !pto.mask`
- semantics: Predicate interleave.

#### `pto.vpdintlv`

- syntax: `%low, %high = pto.vpdintlv %src0, %src1 : !pto.mask, !pto.mask -> !pto.mask, !pto.mask`
- semantics: Predicate deinterleave.

#### `pto.vpslide`

- syntax: `%result = pto.vpslide %src0, %src1, %amt : !pto.mask, !pto.mask, i16 -> !pto.mask`
- semantics: Predicate slide/shift.

---

### 6. Unary Vector Ops

#### `pto.vabs`

- syntax: `%result = pto.vabs %input : !pto.vreg<NxT> -> !pto.vreg<NxT>`
- CCE: `__builtin_cce_vabs_*`
- A5 types: i8-i32, f16, f32

```c
for (int i = 0; i < N; i++)
    dst[i] = (src[i] < 0) ? -src[i] : src[i];
```

#### `pto.vneg`

- syntax: `%result = pto.vneg %input : !pto.vreg<NxT> -> !pto.vreg<NxT>`
- CCE: `__builtin_cce_vneg_*`
- A5 types: i8-i32, f16, f32

```c
for (int i = 0; i < N; i++)
    dst[i] = -src[i];
```

#### `pto.vexp`

- syntax: `%result = pto.vexp %input : !pto.vreg<NxT> -> !pto.vreg<NxT>`
- CCE: `__builtin_cce_vexp_*`
- A5 types: f16, f32

```c
for (int i = 0; i < N; i++)
    dst[i] = expf(src[i]);
```

#### `pto.vln`

- syntax: `%result = pto.vln %input : !pto.vreg<NxT> -> !pto.vreg<NxT>`
- CCE: `__builtin_cce_vln_*`
- A5 types: f16, f32

```c
for (int i = 0; i < N; i++)
    dst[i] = logf(src[i]);
```

#### `pto.vsqrt`

- syntax: `%result = pto.vsqrt %input : !pto.vreg<NxT> -> !pto.vreg<NxT>`
- CCE: `__builtin_cce_vsqrt_*`
- A5 types: f16, f32

```c
for (int i = 0; i < N; i++)
    dst[i] = sqrtf(src[i]);
```

#### `pto.vrsqrt`

- syntax: `%result = pto.vrsqrt %input : !pto.vreg<NxT> -> !pto.vreg<NxT>`
- CCE: `__builtin_cce_vrsqrt_*`
- A5 types: f16, f32

```c
for (int i = 0; i < N; i++)
    dst[i] = 1.0f / sqrtf(src[i]);
```

#### `pto.vrec`

- syntax: `%result = pto.vrec %input : !pto.vreg<NxT> -> !pto.vreg<NxT>`
- CCE: `__builtin_cce_vrec_*`
- A5 types: f16, f32

```c
for (int i = 0; i < N; i++)
    dst[i] = 1.0f / src[i];
```

#### `pto.vrelu`

- syntax: `%result = pto.vrelu %input : !pto.vreg<NxT> -> !pto.vreg<NxT>`
- CCE: `__builtin_cce_vrelu_*`
- A5 types: f16, f32

```c
for (int i = 0; i < N; i++)
    dst[i] = (src[i] > 0) ? src[i] : 0;
```

#### `pto.vnot`

- syntax: `%result = pto.vnot %input : !pto.vreg<NxT> -> !pto.vreg<NxT>`
- CCE: `__builtin_cce_vnot_*`
- A5 types: all integer types

```c
for (int i = 0; i < N; i++)
    dst[i] = ~src[i];
```

#### `pto.vbcnt`

- syntax: `%result = pto.vbcnt %input : !pto.vreg<NxT> -> !pto.vreg<NxT>`
- CCE: `__builtin_cce_vbcnt_*`
- A5 types: all integer types

```c
for (int i = 0; i < N; i++)
    dst[i] = __builtin_popcount(src[i]);
```

#### `pto.vcls`

- syntax: `%result = pto.vcls %input : !pto.vreg<NxT> -> !pto.vreg<NxT>`
- CCE: `__builtin_cce_vcls_*`
- A5 types: all integer types

```c
for (int i = 0; i < N; i++)
    dst[i] = count_leading_sign_bits(src[i]);
```

#### `pto.vmov`

- syntax: `%result = pto.vmov %input : !pto.vreg<NxT> -> !pto.vreg<NxT>`
- semantics: Vector register copy.

```c
for (int i = 0; i < N; i++)
    dst[i] = src[i];
```

#### `pto.vcadd`

- syntax: `%result = pto.vcadd %input : !pto.vreg<NxT> -> !pto.vreg<NxT>`
- CCE: `__builtin_cce_vcadd_*`
- A5 types: i16-i64, f16, f32
- semantics: Full vector sum reduction.

```c
T sum = 0;
for (int i = 0; i < N; i++)
    sum += src[i];
dst[0] = sum;
for (int i = 1; i < N; i++)
    dst[i] = 0;
```

#### `pto.vcmax`

- syntax: `%result = pto.vcmax %input : !pto.vreg<NxT> -> !pto.vreg<NxT>`
- CCE: `__builtin_cce_vcmax_*`
- A5 types: i16-i32, f16, f32
- semantics: Full vector max with argmax.

```c
T mx = -INF; int idx = 0;
for (int i = 0; i < N; i++)
    if (src[i] > mx) { mx = src[i]; idx = i; }
dst_val[0] = mx;
dst_idx[0] = idx;
```

#### `pto.vcmin`

- syntax: `%result = pto.vcmin %input : !pto.vreg<NxT> -> !pto.vreg<NxT>`
- CCE: `__builtin_cce_vcmin_*`
- A5 types: i16-i32, f16, f32
- semantics: Full vector min with argmin.

```c
T mn = INF; int idx = 0;
for (int i = 0; i < N; i++)
    if (src[i] < mn) { mn = src[i]; idx = i; }
dst_val[0] = mn;
dst_idx[0] = idx;
```

#### `pto.vcgadd`

- syntax: `%result = pto.vcgadd %input : !pto.vreg<NxT> -> !pto.vreg<NxT>`
- CCE: `__builtin_cce_vcgadd_*`
- A5 types: i16-i32, f16, f32
- semantics: Per-VLane (32B group) sum reduction.

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

#### `pto.vcgmax`

- syntax: `%result = pto.vcgmax %input : !pto.vreg<NxT> -> !pto.vreg<NxT>`
- CCE: `__builtin_cce_vcgmax_*`
- A5 types: i16-i32, f16, f32
- semantics: Per-VLane max reduction.

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

#### `pto.vcgmin`

- syntax: `%result = pto.vcgmin %input : !pto.vreg<NxT> -> !pto.vreg<NxT>`
- CCE: `__builtin_cce_vcgmin_*`
- A5 types: i16-i32, f16, f32
- semantics: Per-VLane min reduction.

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

#### `pto.vcpadd`

- syntax: `%result = pto.vcpadd %input : !pto.vreg<NxT> -> !pto.vreg<NxT>`
- A5 types: f16, f32
- semantics: Inclusive prefix sum (scan).

```c
dst[0] = src[0];
for (int i = 1; i < N; i++)
    dst[i] = dst[i-1] + src[i];
```

---

### 7. Binary Vector Ops

#### `pto.vadd`

- syntax: `%result = pto.vadd %lhs, %rhs : !pto.vreg<NxT>, !pto.vreg<NxT> -> !pto.vreg<NxT>`
- CCE: `__builtin_cce_vadd_*`
- A5 types: i8-i64, f16, bf16, f32

```c
for (int i = 0; i < N; i++)
    dst[i] = src0[i] + src1[i];
```

#### `pto.vsub`

- syntax: `%result = pto.vsub %lhs, %rhs : !pto.vreg<NxT>, !pto.vreg<NxT> -> !pto.vreg<NxT>`
- CCE: `__builtin_cce_vsub_*`
- A5 types: i8-i64, f16, bf16, f32

```c
for (int i = 0; i < N; i++)
    dst[i] = src0[i] - src1[i];
```

#### `pto.vmul`

- syntax: `%result = pto.vmul %lhs, %rhs : !pto.vreg<NxT>, !pto.vreg<NxT> -> !pto.vreg<NxT>`
- CCE: `__builtin_cce_vmul_*`
- A5 types: i16-i32, f16, bf16, f32 (**NOT** i8/u8)

```c
for (int i = 0; i < N; i++)
    dst[i] = src0[i] * src1[i];
```

#### `pto.vdiv`

- syntax: `%result = pto.vdiv %lhs, %rhs : !pto.vreg<NxT>, !pto.vreg<NxT> -> !pto.vreg<NxT>`
- CCE: `__builtin_cce_vdiv_*`
- A5 types: f16, f32 only (no integer division)

```c
for (int i = 0; i < N; i++)
    dst[i] = src0[i] / src1[i];
```

#### `pto.vmax`

- syntax: `%result = pto.vmax %lhs, %rhs : !pto.vreg<NxT>, !pto.vreg<NxT> -> !pto.vreg<NxT>`
- CCE: `__builtin_cce_vmax_*`
- A5 types: i8-i32, f16, bf16, f32

```c
for (int i = 0; i < N; i++)
    dst[i] = (src0[i] > src1[i]) ? src0[i] : src1[i];
```

#### `pto.vmin`

- syntax: `%result = pto.vmin %lhs, %rhs : !pto.vreg<NxT>, !pto.vreg<NxT> -> !pto.vreg<NxT>`
- CCE: `__builtin_cce_vmin_*`
- A5 types: i8-i32, f16, bf16, f32

```c
for (int i = 0; i < N; i++)
    dst[i] = (src0[i] < src1[i]) ? src0[i] : src1[i];
```

#### `pto.vand`

- syntax: `%result = pto.vand %lhs, %rhs : !pto.vreg<NxT>, !pto.vreg<NxT> -> !pto.vreg<NxT>`
- CCE: `__builtin_cce_vand_*`
- A5 types: all integer types

```c
for (int i = 0; i < N; i++)
    dst[i] = src0[i] & src1[i];
```

#### `pto.vor`

- syntax: `%result = pto.vor %lhs, %rhs : !pto.vreg<NxT>, !pto.vreg<NxT> -> !pto.vreg<NxT>`
- CCE: `__builtin_cce_vor_*`
- A5 types: all integer types

```c
for (int i = 0; i < N; i++)
    dst[i] = src0[i] | src1[i];
```

#### `pto.vxor`

- syntax: `%result = pto.vxor %lhs, %rhs : !pto.vreg<NxT>, !pto.vreg<NxT> -> !pto.vreg<NxT>`
- CCE: `__builtin_cce_vxor_*`
- A5 types: all integer types

```c
for (int i = 0; i < N; i++)
    dst[i] = src0[i] ^ src1[i];
```

#### `pto.vshl`

- syntax: `%result = pto.vshl %lhs, %rhs : !pto.vreg<NxT>, !pto.vreg<NxT> -> !pto.vreg<NxT>`
- CCE: `__builtin_cce_vshl_*`
- A5 types: all integer types

```c
for (int i = 0; i < N; i++)
    dst[i] = src0[i] << src1[i];
```

#### `pto.vshr`

- syntax: `%result = pto.vshr %lhs, %rhs : !pto.vreg<NxT>, !pto.vreg<NxT> -> !pto.vreg<NxT>`
- CCE: `__builtin_cce_vshr_*`
- A5 types: all integer types

```c
for (int i = 0; i < N; i++)
    dst[i] = src0[i] >> src1[i];  // arithmetic for signed, logical for unsigned
```

---

### 8. Vec-Scalar Ops

#### `pto.vadds`

- syntax: `%result = pto.vadds %input, %scalar : !pto.vreg<NxT>, T -> !pto.vreg<NxT>`
- CCE: `__builtin_cce_vadds_*`

```c
for (int i = 0; i < N; i++)
    dst[i] = src[i] + scalar;
```

#### `pto.vsubs`

- syntax: `%result = pto.vsubs %input, %scalar : !pto.vreg<NxT>, T -> !pto.vreg<NxT>`
- CCE: `__builtin_cce_vsubs_*`

```c
for (int i = 0; i < N; i++)
    dst[i] = src[i] - scalar;
```

#### `pto.vmuls`

- syntax: `%result = pto.vmuls %input, %scalar : !pto.vreg<NxT>, T -> !pto.vreg<NxT>`
- CCE: `__builtin_cce_vmuls_*`

```c
for (int i = 0; i < N; i++)
    dst[i] = src[i] * scalar;
```

#### `pto.vmaxs`

- syntax: `%result = pto.vmaxs %input, %scalar : !pto.vreg<NxT>, T -> !pto.vreg<NxT>`
- CCE: `__builtin_cce_vmaxs_*`

```c
for (int i = 0; i < N; i++)
    dst[i] = (src[i] > scalar) ? src[i] : scalar;
```

#### `pto.vmins`

- syntax: `%result = pto.vmins %input, %scalar : !pto.vreg<NxT>, T -> !pto.vreg<NxT>`
- CCE: `__builtin_cce_vmins_*`

```c
for (int i = 0; i < N; i++)
    dst[i] = (src[i] < scalar) ? src[i] : scalar;
```

#### `pto.vands`

- syntax: `%result = pto.vands %input, %scalar : !pto.vreg<NxT>, T -> !pto.vreg<NxT>`
- CCE: `__builtin_cce_vands_*`

```c
for (int i = 0; i < N; i++)
    dst[i] = src[i] & scalar;
```

#### `pto.vors`

- syntax: `%result = pto.vors %input, %scalar : !pto.vreg<NxT>, T -> !pto.vreg<NxT>`
- CCE: `__builtin_cce_vors_*`

```c
for (int i = 0; i < N; i++)
    dst[i] = src[i] | scalar;
```

#### `pto.vxors`

- syntax: `%result = pto.vxors %input, %scalar : !pto.vreg<NxT>, T -> !pto.vreg<NxT>`
- CCE: `__builtin_cce_vxors_*`

```c
for (int i = 0; i < N; i++)
    dst[i] = src[i] ^ scalar;
```

#### `pto.vlrelu`

- syntax: `%result = pto.vlrelu %input, %alpha : !pto.vreg<NxT>, T -> !pto.vreg<NxT>`
- CCE: `__builtin_cce_vlrelu_*`
- A5 types: f16, f32

```c
for (int i = 0; i < N; i++)
    dst[i] = (src[i] >= 0) ? src[i] : alpha * src[i];
```

#### `pto.vshls`

- syntax: `%result = pto.vshls %input, %scalar : !pto.vreg<NxT>, T -> !pto.vreg<NxT>`
- CCE: `__builtin_cce_vshls_*`

```c
for (int i = 0; i < N; i++)
    dst[i] = src[i] << scalar;
```

#### `pto.vshrs`

- syntax: `%result = pto.vshrs %input, %scalar : !pto.vreg<NxT>, T -> !pto.vreg<NxT>`
- CCE: `__builtin_cce_vshrs_*`

```c
for (int i = 0; i < N; i++)
    dst[i] = src[i] >> scalar;
```

---

### 9. Carry, Compare And Select

#### `pto.vaddc`

- syntax: `%result, %carry = pto.vaddc %lhs, %rhs, %mask : !pto.vreg<NxT>, !pto.vreg<NxT>, !pto.mask -> !pto.vreg<NxT>, !pto.mask`
- CCE: `__builtin_cce_vaddc_*`

```c
for (int i = 0; i < N; i++) {
    uint64_t r = (uint64_t)src0[i] + src1[i];
    dst[i] = (T)r;
    carry[i] = (r >> bitwidth);
}
```

#### `pto.vsubc`

- syntax: `%result, %carry = pto.vsubc %lhs, %rhs, %mask : !pto.vreg<NxT>, !pto.vreg<NxT>, !pto.mask -> !pto.vreg<NxT>, !pto.mask`
- CCE: `__builtin_cce_vsubc_*`

```c
for (int i = 0; i < N; i++) {
    dst[i] = src0[i] - src1[i];
    borrow[i] = (src0[i] < src1[i]);
}
```

#### `pto.vaddcs`

- syntax: `%result, %carry = pto.vaddcs %lhs, %rhs, %carry_in, %mask : !pto.vreg<NxT>, !pto.vreg<NxT>, !pto.mask, !pto.mask -> !pto.vreg<NxT>, !pto.mask`
- CCE: `__builtin_cce_vaddcs_*`

```c
for (int i = 0; i < N; i++) {
    uint64_t r = (uint64_t)src0[i] + src1[i] + carry_in[i];
    dst[i] = (T)r;
    carry_out[i] = (r >> bitwidth);
}
```

#### `pto.vsubcs`

- syntax: `%result, %carry = pto.vsubcs %lhs, %rhs, %carry_in, %mask : !pto.vreg<NxT>, !pto.vreg<NxT>, !pto.mask, !pto.mask -> !pto.vreg<NxT>, !pto.mask`
- CCE: `__builtin_cce_vsubcs_*`

```c
for (int i = 0; i < N; i++) {
    dst[i] = src0[i] - src1[i] - borrow_in[i];
    borrow_out[i] = (src0[i] < src1[i] + borrow_in[i]);
}
```

#### `pto.vsel`

- syntax: `%result = pto.vsel %src0, %src1, %mask : !pto.vreg<NxT>, !pto.vreg<NxT>, !pto.mask -> !pto.vreg<NxT>`
- CCE: `__builtin_cce_vsel_*`

```c
for (int i = 0; i < N; i++)
    dst[i] = mask[i] ? src0[i] : src1[i];
```

#### `pto.vselr`

- syntax: `%result = pto.vselr %src0, %src1 : !pto.vreg<NxT>, !pto.vreg<NxI> -> !pto.vreg<NxT>`
- CCE: `__builtin_cce_vselr_*`

```c
for (int i = 0; i < N; i++)
    dst[i] = mask[i] ? src1[i] : src0[i];  // reversed
```

#### `pto.vcmp`

- syntax: `%result = pto.vcmp %src0, %src1, %seed, "CMP_MODE" : !pto.vreg<NxT>, !pto.vreg<NxT>, !pto.mask -> !pto.mask`
- CCE: `__builtin_cce_vcmp_<op>_*_z`

```c
for (int i = 0; i < N; i++)
    if (seed[i])
        dst[i] = (src0[i] CMP src1[i]) ? 1 : 0;
// CMP_MODE: eq, ne, lt, le, gt, ge
```

#### `pto.vcmps`

- syntax: `%result = pto.vcmps %src, %scalar, %seed, "CMP_MODE" : !pto.vreg<NxT>, T, !pto.mask -> !pto.mask`
- CCE: `__builtin_cce_vcmps_<op>_*_z`

```c
for (int i = 0; i < N; i++)
    if (seed[i])
        dst[i] = (src[i] CMP scalar) ? 1 : 0;
```

#### `pto.vpnot`

- syntax: `%result = pto.vpnot %input, %mask : !pto.mask, !pto.mask -> !pto.mask`
- CCE: `pnot(...)`

```c
for (int i = 0; i < N; i++)
    if (mask[i]) dst[i] = ~src[i];
```

#### `pto.vpsel`

- syntax: `%result = pto.vpsel %src0, %src1, %sel : !pto.mask, !pto.mask, !pto.mask -> !pto.mask`
- CCE: `psel(...)`

```c
for (int i = 0; i < N; i++)
    dst[i] = sel[i] ? src0[i] : src1[i];
```

#### `pto.vprelu`

- syntax: `%result = pto.vprelu %input, %alpha : !pto.vreg<NxT>, !pto.vreg<NxT> -> !pto.vreg<NxT>`
- A5 types: f16, f32
- semantics: Parametric ReLU with per-element alpha vector.

```c
for (int i = 0; i < N; i++)
    dst[i] = (src[i] >= 0) ? src[i] : alpha[i] * src[i];
```

---

### 10. Data Rearrangement And Interleave

#### `pto.vintlv`

- syntax: `%low, %high = pto.vintlv %lhs, %rhs : !pto.vreg<NxT>, !pto.vreg<NxT> -> !pto.vreg<NxT>, !pto.vreg<NxT>`
- CCE: `__builtin_cce_vintlv_*`

```c
// Interleave: merge even/odd elements from two sources
// dst = {src0[0], src1[0], src0[1], src1[1], ...}
```

#### `pto.vdintlv`

- syntax: `%low, %high = pto.vdintlv %lhs, %rhs : !pto.vreg<NxT>, !pto.vreg<NxT> -> !pto.vreg<NxT>, !pto.vreg<NxT>`
- CCE: `__builtin_cce_vdintlv_*`

```c
// Deinterleave: separate even/odd elements
// low  = {src0[0], src0[2], src0[4], ...}  // even
// high = {src0[1], src0[3], src0[5], ...}  // odd
```

#### `pto.vslide`

- syntax: `%result = pto.vslide %src0, %src1, %amt : !pto.vreg<NxT>, !pto.vreg<NxT>, i16 -> !pto.vreg<NxT>`
- semantics: Concatenate two vectors and extract N-element window at offset.

```c
// Conceptually: tmp[0..2N-1] = {src1, src0}
// dst[i] = tmp[amt + i]
if (amt >= 0)
    for (int i = 0; i < N; i++)
        dst[i] = (i >= amt) ? src0[i - amt] : src1[N - amt + i];
```

#### `pto.vshift`

- syntax: `%result = pto.vshift %src, %amt : !pto.vreg<NxT>, i16 -> !pto.vreg<NxT>`
- semantics: Single-source slide (shift with zero fill).

```c
for (int i = 0; i < N; i++)
    dst[i] = (i >= amt) ? src[i - amt] : 0;
```

#### `pto.vsqz`

- syntax: `%result = pto.vsqz %src, %mask : !pto.vreg<NxT>, !pto.mask -> !pto.vreg<NxT>`
- semantics: Compress — pack active lanes to front.

```c
int j = 0;
for (int i = 0; i < N; i++)
    if (mask[i]) dst[j++] = src[i];
while (j < N) dst[j++] = 0;
```

#### `pto.vusqz`

- syntax: `%result = pto.vusqz %mask : !pto.mask -> !pto.vreg<NxT>`
- semantics: Expand — scatter front elements to active positions.

```c
int j = 0;
for (int i = 0; i < N; i++)
    if (mask[i]) dst[i] = src_front[j++];
    else dst[i] = 0;
```

#### `pto.vperm`

- syntax: `%result = pto.vperm %src, %index : !pto.vreg<NxT>, !pto.vreg<NxI> -> !pto.vreg<NxT>`
- semantics: In-register permute (table lookup). **Not** memory gather.

```c
for (int i = 0; i < N; i++)
    dst[i] = src[index[i] % N];
```

#### `pto.vtranspose`

- syntax: `%result = pto.vtranspose %src, %pattern : !pto.vreg<NxT>, i32 -> !pto.vreg<NxT>`
- semantics: In-register transpose with configurable pattern.

#### `pto.vpack`

- syntax: `%result = pto.vpack %src0, %src1, %part : !pto.vreg<NxT_wide>, !pto.vreg<NxT_wide>, index -> !pto.vreg<2NxT_narrow>`
- semantics: Narrowing pack — two wide vectors to one narrow vector.

```c
// e.g., two vreg<64xi32> → one vreg<128xi16>
for (int i = 0; i < N; i++) {
    dst[i]     = truncate(src0[i]);
    dst[N + i] = truncate(src1[i]);
}
```

#### `pto.vsunpack`

- syntax: `%result = pto.vsunpack %src, %part : !pto.vreg<NxT_narrow>, index -> !pto.vreg<N/2xT_wide>`
- semantics: Sign-extending unpack — narrow to wide (half).

```c
// e.g., vreg<128xi16> → vreg<64xi32> (one half)
for (int i = 0; i < N/2; i++)
    dst[i] = sign_extend(src[part_offset + i]);
```

#### `pto.vzunpack`

- syntax: `%result = pto.vzunpack %src, %part : !pto.vreg<NxT_narrow>, index -> !pto.vreg<N/2xT_wide>`
- semantics: Zero-extending unpack — narrow to wide (half).

```c
for (int i = 0; i < N/2; i++)
    dst[i] = zero_extend(src[part_offset + i]);
```

---

### 11. Conversion, Index And Sort

#### `pto.vcvt`

- syntax: `%result = pto.vcvt %input {round_mode = "ROUND_MODE", sat = "SAT_MODE", part = "PART_MODE"} : !pto.vreg<NxT0> -> !pto.vreg<MxT1>`
- CCE: `__builtin_cce_vcvt*`, `__builtin_cce_vcvtfi_*`, `__builtin_cce_vcvtif_*`, `__builtin_cce_vcvtff_*`
- semantics: Type conversion between float/int types with rounding control.

```c
for (int i = 0; i < min(N, M); i++)
    dst[i] = convert(src[i], T0, T1, round_mode);
```

**A5 Float-Float conversions (vcvtff):**
- f32 ↔ f16
- f32 ↔ bf16
- f16 ↔ bf16

**A5 Float-Int conversions (vcvtfi):**
- f16 → i16, f16 → i32
- f32 → i16, f32 → i32
- bf16 → i32

**A5 Int-Float conversions (vcvtif):**
- i16 → f16
- i32 → f32

**Round modes:** `ROUND_R` (nearest-even), `ROUND_A` (away), `ROUND_F` (floor), `ROUND_C` (ceil), `ROUND_Z` (truncate)

**Width-changing conversion pattern (e.g., f32 → f16):**

```mlir
%even = pto.vcvt %in0 {round_mode = "ROUND_R", sat = "RS_ENABLE", part = "PART_EVEN"} : !pto.vreg<64xf32> -> !pto.vreg<128xf16>
%odd  = pto.vcvt %in1 {round_mode = "ROUND_R", sat = "RS_ENABLE", part = "PART_ODD"}  : !pto.vreg<64xf32> -> !pto.vreg<128xf16>
%result = pto.vor %even, %odd : !pto.vreg<128xf16>, !pto.vreg<128xf16> -> !pto.vreg<128xf16>
```

#### `pto.vtrc`

- syntax: `%result = pto.vtrc %input, "ROUND_MODE" : !pto.vreg<NxT> -> !pto.vreg<NxT>`
- CCE: `__builtin_cce_vtrc_*`
- semantics: Truncate/round float to integer-valued float.

```c
for (int i = 0; i < N; i++)
    dst[i] = round_to_int_valued_float(src[i], round_mode);
```

#### `pto.vci`

- syntax: `%result = pto.vci %index {order = "ORDER"} : integer -> !pto.vreg<NxT>`
- CCE: `__builtin_cce_vci_*`
- semantics: Generate lane index vector.

```c
for (int i = 0; i < N; i++)
    dst[i] = base_index + i;
```

#### `pto.vbitsort`

- syntax: `pto.vbitsort %dest, %source, %indices, %repeat : !llvm.ptr<6>, !llvm.ptr<6>, !llvm.ptr<6>, index`
- CCE: `__builtin_cce_vbitsort_*`
- semantics: Bit-level sorting operation.

#### `pto.vmrgsort4`

- syntax: `pto.vmrgsort4 %dest, %src0, %src1, %src2, %src3, %count, %config : !llvm.ptr<6>, !llvm.ptr<6> x4, i64, i64`
- CCE: `__builtin_cce_vmrgsort4_*`
- semantics: Merge-sort 4 pre-sorted input vectors.

---

### 12. Extended Arithmetic

#### `pto.vmull`

- syntax: `%low, %high = pto.vmull %lhs, %rhs, %mask : !pto.vreg<NxT>, !pto.vreg<NxT>, !pto.mask -> !pto.vreg<NxT>, !pto.vreg<NxT>`
- CCE: `__builtin_cce_vmull_*`
- A5 types: i32/u32 (native 32×32→64 widening multiply)

```c
for (int i = 0; i < 64; i++) {
    int64_t r = (int64_t)src0_i32[i] * (int64_t)src1_i32[i];
    dst_lo[i] = (int32_t)(r & 0xFFFFFFFF);
    dst_hi[i] = (int32_t)(r >> 32);
}
```

#### `pto.vmula`

- syntax: `%result = pto.vmula %acc, %lhs, %rhs, %mask {mode = "MODE"} : !pto.vreg<NxT>, !pto.vreg<NxT>, !pto.vreg<NxT>, !pto.mask -> !pto.vreg<NxT>`
- CCE: `__builtin_cce_vmula_*_m`
- semantics: Multiply-accumulate with mode control.

```c
for (int i = 0; i < N; i++)
    if (mask[i])
        dst[i] = acc[i] + lhs[i] * rhs[i];
    else if (mode == MODE_ZEROING)
        dst[i] = 0;
// MODE: MODE_ZEROING | MODE_MERGING
```

#### `pto.vaddrelu`

- syntax: `%result = pto.vaddrelu %lhs, %rhs : !pto.vreg<NxT>, !pto.vreg<NxT> -> !pto.vreg<NxT>`
- A5 types: f16, f32
- semantics: Fused add + ReLU.

```c
for (int i = 0; i < N; i++)
    dst[i] = max(src0[i] + src1[i], 0);
```

#### `pto.vsubrelu`

- syntax: `%result = pto.vsubrelu %lhs, %rhs : !pto.vreg<NxT>, !pto.vreg<NxT> -> !pto.vreg<NxT>`
- A5 types: f16, f32
- semantics: Fused sub + ReLU.

```c
for (int i = 0; i < N; i++)
    dst[i] = max(src0[i] - src1[i], 0);
```

#### `pto.vaxpy`

- syntax: `%result = pto.vaxpy %src0, %src1, %alpha : !pto.vreg<NxT>, !pto.vreg<NxT>, T -> !pto.vreg<NxT>`
- A5 types: f16, f32
- semantics: AXPY — scalar-vector multiply-add.

```c
for (int i = 0; i < N; i++)
    dst[i] = alpha * src0[i] + src1[i];
```

#### `pto.vaddreluconv`

- syntax: `%result = pto.vaddreluconv %lhs, %rhs : !pto.vreg<NxT0>, !pto.vreg<NxT0> -> !pto.vreg<MxT1>`
- semantics: Fused add + ReLU + type conversion (A5-specific HW fusion).

```c
// f32→f16 variant:
for (int i = 0; i < 64; i++)
    dst_f16[i] = f32_to_f16(max(src0_f32[i] + src1_f32[i], 0));

// f16→i8 variant:
for (int i = 0; i < 128; i++)
    dst_i8[i] = f16_to_i8(max(src0_f16[i] + src1_f16[i], 0));
```

#### `pto.vmulconv`

- syntax: `%result = pto.vmulconv %lhs, %rhs : !pto.vreg<NxT0>, !pto.vreg<NxT0> -> !pto.vreg<MxT1>`
- semantics: Fused mul + type conversion (A5-specific HW fusion).

```c
// f16→i8 variant:
for (int i = 0; i < 128; i++)
    dst_i8[i] = f16_to_i8(src0_f16[i] * src1_f16[i]);
```

---

### 13. Stateless Stores

#### `pto.vsts`

- syntax: `pto.vsts %value, %dest[%offset] {dist = "DIST"} : !pto.vreg<NxT>, !llvm.ptr<6>`
- CCE: `__builtin_cce_vstx1_*`, `__builtin_cce_vstsx1_*`
- semantics: Vector store with distribution mode.

```c
// NORM_B32: contiguous store
for (int i = 0; i < N; i++)
    UB[base + offset + i * sizeof(T)] = src[i];

// PK_B16: pack/narrowing store (32→16)
for (int i = 0; i < 64; i++)
    UB_i16[base + 2*i] = truncate_16(src_i32[i]);

// MRG4CHN_B8: merge 4 planes into interleaved (R,G,B,A → RGBA)
// MRG2CHN_B8/B16: merge 2 planes into interleaved
```

#### `pto.vscatter`

- syntax: `pto.vscatter %value, %dest, %offsets, %active_lanes : !pto.vreg<NxT>, !llvm.ptr<6>, !pto.vreg<NxI>, index`
- CCE: `__builtin_cce_vscatter_*`

```c
for (int i = 0; i < active_lanes; i++)
    UB[base + offsets[i] * sizeof(T)] = src[i];
```

#### `pto.vsts_pred`

- syntax: `pto.vsts_pred %value, %dest[%offset], %active_lanes {dist = "DIST"} : !pto.vreg<NxT>, !llvm.ptr<6>, index`
- semantics: Predicated vector store.

#### `pto.vpsts`

- syntax: `pto.vpsts %value, %dest[%offset] : !pto.mask, !llvm.ptr<6>`
- CCE: `__builtin_cce_psts_b8`
- semantics: Store predicate register with scalar offset.

#### `pto.vpst`

- syntax: `pto.vpst %value, %dest[%offset], "DIST" : !pto.mask, !llvm.ptr<6>, index`
- CCE: `__builtin_cce_pst_b8`
- semantics: Store predicate register with areg offset.

#### `pto.vpsti`

- syntax: `pto.vpsti %value, %dest, %offset, "DIST" : !pto.mask, !llvm.ptr<6>, i32`
- CCE: `__builtin_cce_psti_b8`
- semantics: Store predicate register with immediate offset.

#### `pto.vsst`

- syntax: `pto.vsst %value, %dest[%offset], "STRIDE" : !pto.vreg<NxT>, !llvm.ptr<6>`
- CCE: `__builtin_cce_vsst_*`
- semantics: Strided store with fixed stride pattern.

#### `pto.vstx2`

- syntax: `pto.vstx2 %low, %high, %dest[%offset], "DIST", %mask : !pto.vreg<NxT>, !pto.vreg<NxT>, !llvm.ptr<6>, index, !pto.mask`
- CCE: `__builtin_cce_vstx2_*`
- semantics: Dual interleaved store (SoA → AoS).

```c
// INTLV_B32:
for (int i = 0; i < 64; i++) {
    UB[base + 8*i]     = low[i];
    UB[base + 8*i + 4] = high[i];
}
```

#### `pto.vsstb`

- syntax: `pto.vsstb %value, %dest, %offset, %mask : !pto.vreg<NxT>, !llvm.ptr<6>, i32, !pto.mask`
- CCE: `__builtin_cce_vsstb_*`
- semantics: Block-strided store for 2D tile access.

#### `pto.vsta`

- syntax: `pto.vsta %value, %dest[%offset] : !pto.align, !llvm.ptr<6>, index`
- CCE: `__builtin_cce_vsta_*`
- semantics: Flush alignment state to memory.

#### `pto.vstas`

- syntax: `pto.vstas %value, %dest, %offset : !pto.align, !llvm.ptr<6>, i32`
- CCE: `__builtin_cce_vstas_*`
- semantics: Flush alignment state with scalar offset.

#### `pto.vstar`

- syntax: `pto.vstar %value, %dest : !pto.align, !llvm.ptr<6>`
- CCE: `__builtin_cce_vstar_*`
- semantics: Flush remaining alignment state.

---

### 14. Stateful Store Ops

These ops make CCE reference-updated state explicit as SSA results.

#### `pto.vpstu`

- syntax: `%align_out, %base_out = pto.vpstu %align_in, %value, %base : !pto.align, !pto.mask, !llvm.ptr<6> -> !pto.align, !llvm.ptr<6>`
- CCE: `__builtin_cce_pstu_b16`, `__builtin_cce_pstu_b32`
- semantics: Predicate unaligned store with align state update.

#### `pto.vstu`

- syntax: `%align_out, %offset_out = pto.vstu %align_in, %offset_in, %value, %base, "MODE" : !pto.align, index, !pto.vreg<NxT>, !llvm.ptr<6> -> !pto.align, index`
- CCE: `__builtin_cce_vstu_*`
- semantics: Unaligned store with align + offset state update.

#### `pto.vstus`

- syntax: `%align_out, %base_out = pto.vstus %align_in, %offset, %value, %base, "MODE" : !pto.align, i32, !pto.vreg<NxT>, !llvm.ptr<6> -> !pto.align, !llvm.ptr<6>`
- CCE: `__builtin_cce_vstus_*`
- semantics: Unaligned store with scalar offset and state update.

#### `pto.vstur`

- syntax: `%align_out = pto.vstur %align_in, %value, %base, "MODE" : !pto.align, !pto.vreg<NxT>, !llvm.ptr<6> -> !pto.align`
- CCE: `__builtin_cce_vstur_*`
- semantics: Unaligned store with residual flush and state update.

**Stateful store mode tokens:** `POST_UPDATE | NO_POST_UPDATE`

---

*End of Part III — ISA Instruction Reference complete*

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
| 7 | Sec 5: vpand, vpor, vpxor, vpmov | **ADDED** | a5_intrinsic_ir.md |
| 8 | Sec 5: vpintlv, vpdintlv, vpslide | **ADDED** | a5_intrinsic_ir.md |
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
| 5 | Sec 10: vintlvv2, vdintlvv2, vpdintlv_b8, vpintlv_b16 | **REMOVED** (not A5) | — |
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
3. **Predicate ops (vpand/vpor/vpxor/vpmov/vpintlv/vpdintlv/vpslide):** These need MLIR op definitions and verifier rules. Confirm priority.

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
