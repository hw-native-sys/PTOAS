# PTO micro Instruction Spec — Merged Draft (A5)

> **Status:** DRAFT for review
> **Base:** [vpto-spec.md](https://github.com/mouliangyu/PTOAS/blob/feature-vpto-backend/docs/vpto-spec.md) (2026-03-20)
> **Updated:** 2026-05-28

---

## Part I: Architecture Overview

### Overview

This document defines the PTO micro Instruction, a compiler-internal and externally facing specification designed to represent vector compute kernels within the PTO architecture. Much like NVVM provides a robust IR for GPU architectures, the PTO micro Instruction serves as the direct bridge between high-level programming models and the underlying hardware ISA, providing a precise, architecture-aware representation of vector workloads explicitly designed for the Ascend 950 architecture.

#### Position in the Stack and Layer Modeled

The PTO micro Instruction operates as an explicit intermediate representation within the PTO compiler stack. It is designed to accurately express the user-visible architectural information needed for Ascend 950 kernels, including vector lane organization, memory space hierarchy, synchronization, and hardware-specific fusion semantics.

#### PTO Instruction Modes and Compilation Flows

Within the end-to-end PTO software stack, PTO instructions may appear in three closely related authoring or lowering modes:

- **PTO Tile Instruction**: tile-oriented PTO code that serves as a nano-kernel encapsulation of tile instructions, primarily expressing computation and data movement in terms of tile buffers, tile shapes, and tile-local layout.
- **PTO micro Instruction**: vector-execution-oriented PTO code that makes DMA setup, vector registers, masks, synchronization, and `__VEC_SCOPE__` boundaries explicit. This document is centered on this mode.
- **PTO Tile+micro Instruction**: a hybrid PTO form that keeps tile-level orchestration while embedding explicit micro-instruction regions where direct vector-pipeline control is required.

From these PTO instruction forms, the stack can proceed along two main compilation flows:

- **CCE generation flow**: PTO ISA is lowered into a CCE-oriented representation, which is then compiled by the BiSheng toolchain into Ascend device binaries.
- **VPTO flow**: PTO ISA is lowered through the VPTO backend for A5 device code generation. PTOAS organizes the device components and invokes the BiSheng compiler internally to produce the final device artifact.

```text
        High-level frameworks / DSLs / library kernels
                             |
                             v
            +----------------------------------+
            |          PTO ISA layer           |
            |                                  |
            |  (1) PTO Tile Instruction        |
            |  (2) PTO micro Instruction       |
            |  (3) PTO Tile+micro Instruction  |
            +----------------+-----------------+
                             |
              +--------------+--------------+
              |                             |
              v                             v
 +-------------------------+   +-------------------------+
 | Path A: generate CCE    |   | Path B: generate        |
 | (CCE-oriented form)     |   | bytecode                |
 +------------+------------+   +------------+------------+
              |                             |
              v                             v
 +-------------------------+   +-------------------------+
 | BiSheng compiler        |   | BiSheng compiler        |
 | invoked explicitly      |   | invoked inside PTOAS    |
 +------------+------------+   +------------+------------+
              |                             |
              +--------------+--------------+
                             |
                             v
              +-----------------------------+
              |   Ascend device binaries    |
              +-----------------------------+
```

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

#### Authoring VPTO `.pto` Files

A VPTO source file must make the target architecture, launched device function,
and cube/vector placement explicit. The recommended authoring form is a single
outer module with one or more `pto.kernel` functions whose bodies are split by
`pto.section.vector` and `pto.section.cube`. The Vector section describes the
Vector-unit program, and the Cube section describes the Cube-unit program.
Synchronization and communication between the two units are written as normal
operations in the relevant section bodies.

**Common module attributes:**

| Attribute | Attachment site | Required | Meaning |
|-----------|-----------------|----------|---------|
| `pto.target_arch = "a5"` | outer `module` | Recommended in source files | Selects the A5 PTO parser and verifier contract. A command-line `--pto-arch` value overrides the module attribute. |
| `pto.kernel` | `func.func` | Required for externally launched device kernels | Marks the function as a device kernel entry. Helper functions inside the same module do not need this attribute unless they are launched directly. |
| `pto.section.vector` | region inside a `pto.kernel` function | Required for vector-core code in the recommended source form | Contains the Vector program. |
| `pto.section.cube` | region inside a `pto.kernel` function | Required for cube-core code in the recommended source form | Contains the Cube program. |
| `pto.kernel_kind = #pto.kernel_kind<vector>` | normalized kernel `module` | Advanced/frontend-emitted form only | Marks a normalized submodule as vector-core code. |
| `pto.kernel_kind = #pto.kernel_kind<cube>` | normalized kernel `module` | Advanced/frontend-emitted form only | Marks a normalized submodule as cube-core code. |

In this source form, every `pto.kernel` function must contain one or both
sections. A function may contain at most one `pto.section.vector` and at most
one `pto.section.cube`; nested sections are invalid. Values defined outside the
sections may be used by both sections, but values defined inside one section
are local to that section.

For the recommended source form, keep `pto.target_arch` on the outer module,
mark the launched function with `pto.kernel`, and place core-specific code
inside `pto.section.vector` and/or `pto.section.cube`:

```mlir
module attributes {pto.target_arch = "a5"} {
  func.func @mixed_kernel(%a: !pto.ptr<f16, gm>,
                          %b: !pto.ptr<f16, gm>,
                          %out: !pto.ptr<f32, gm>) attributes {pto.kernel} {
    %c0_i64 = arith.constant 0 : i64
    %l1 = pto.castptr %c0_i64 : i64 -> !pto.ptr<f16, l1>
    %ub = pto.castptr %c0_i64 : i64 -> !pto.ptr<f32, ub>

    pto.section.cube {
      // Cube program body.
    }

    pto.section.vector {
      // Vector program body.
    }

    return
  }
}
```

Vector-only and cube-only kernels use the same structure with only the section
they need:

```mlir
module attributes {pto.target_arch = "a5"} {
  func.func @vadd_kernel(%lhs: !pto.ptr<f32, gm>,
                         %rhs: !pto.ptr<f32, gm>,
                         %out: !pto.ptr<f32, gm>) attributes {pto.kernel} {
    %c0_i64 = arith.constant 0 : i64
    %ub = pto.castptr %c0_i64 : i64 -> !pto.ptr<f32, ub>

    pto.section.vector {
      // Vector-core program body.
    }

    return
  }
}
```

Advanced frontends may emit a normalized container directly. This is not the
preferred hand-authored source shape, but it is a valid compiler-facing form:

```mlir
module attributes {pto.target_arch = "a5"} {
  module attributes {pto.kernel_kind = #pto.kernel_kind<vector>} {
    func.func @kernel(%in: !pto.ptr<f32, gm>,
                      %out: !pto.ptr<f32, gm>) attributes {pto.kernel} {
      return
    }
  }
  module attributes {pto.kernel_kind = #pto.kernel_kind<cube>} {
    func.func @kernel(%in: !pto.ptr<f32, gm>,
                      %out: !pto.ptr<f32, gm>) attributes {pto.kernel} {
      return
    }
  }
}
```

At the container top level, only kernel submodules are valid. Each kernel
submodule must carry exactly one `pto.kernel_kind`. Put `pto.target_arch` on
the outer module so all submodules share the same target contract.

**Compilation:**

```bash
ptoas --pto-arch=a5 --pto-backend=vpto kernel.pto -o kernel.o
```

This command emits the final device artifact.

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
| i8/si8/ui8 | 32 | 256 |
| i16/si16/ui16/f16/bf16 | 16 | 128 |
| i32/si32/ui32/f32 | 8 | 64 |
| i64/si64/ui64 | 4 | 32 |

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

1. **GM → UB**: DMA transfer via MTE2 (`pto.mte_gm_ub`)
2. **UB → vreg**: Vector Load instructions (`pto.vlds`, `pto.vldsx2`, etc.)
3. **vreg → vreg**: Compute instructions (`pto.vadd`, `pto.vmul`, etc.)
4. **vreg → UB**: Vector Store instructions (`pto.vsts`, `pto.vstsx2`, etc.)
5. **UB → GM**: DMA transfer via MTE3 (`pto.mte_ub_gm`)

The grouped DMA surface in this specification covers `pto.mte_gm_ub`
(GM→UB), `pto.mte_ub_gm` (UB→GM), and `pto.mte_ub_ub` / `pto.mte_ub_l1`
(UB→UB or UB→CBUF).

**Load/Store Access Patterns**:

For UB↔vreg data movement, besides contiguous load/store, the architecture provides rich access pattern support including strided access, pack/unpack, interleave/deinterleave, broadcast, upsample/downsample, channel split/merge, gather/scatter, and squeeze/expand operations. For detailed instruction syntax and distribution modes, refer to the [Vector Load/Store](isa/micro-isa/03-vector-load-store.md) group in the ISA specification.

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
pto.dcci %gm "ENTIRE_DATA_CACHE", "CACHELINE_OUT" : !pto.ptr<i8, gm>
pto.dsb "ALL"
```

Without proper barriers, loads may see stale data or stores may be reordered incorrectly.

#### Execution Scopes (__VEC_SCOPE__)

`__VEC_SCOPE__` is the IR-level representation of a Vector Function (VF) launch. In the PTO architecture, it defines the hardware interface between the Scalar Unit and the Vector Thread.

In PTO micro Instruction source IR, vector execution scopes are modeled as dedicated region ops. The default form is `pto.vecscope`; when the scope body must reject implicit capture and require explicit region arguments, use `pto.strict_vecscope`.

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
pto.vecscope {
  %mask = pto.pset_b32 "PAT_ALL" : !pto.mask<b32>
  %v = pto.vlds %ub[%lane] : !pto.ptr<f32, ub> -> !pto.vreg<64xf32>
  %abs = pto.vabs %v, %mask : !pto.vreg<64xf32>, !pto.mask<b32> -> !pto.vreg<64xf32>
  pto.vsts %abs, %ub_out[%lane], %mask : !pto.vreg<64xf32>, !pto.ptr<f32, ub>, !pto.mask<b32>
}
```

**Strict MLIR Representation:**

```mlir
pto.strict_vecscope(%ub, %ub_out, %lane) {
^bb0(%in: !pto.ptr<f32, ub>, %out: !pto.ptr<f32, ub>, %iv: index):
  %mask = pto.pset_b32 "PAT_ALL" : !pto.mask<b32>
  %v = pto.vlds %in[%iv] : !pto.ptr<f32, ub> -> !pto.vreg<64xf32>
  %abs = pto.vabs %v, %mask : !pto.vreg<64xf32>, !pto.mask<b32> -> !pto.vreg<64xf32>
  pto.vsts %abs, %out[%iv], %mask : !pto.vreg<64xf32>, !pto.ptr<f32, ub>, !pto.mask<b32>
} : (!pto.ptr<f32, ub>, !pto.ptr<f32, ub>, index) -> ()
```

`pto.strict_vecscope` is the strict form of `pto.vecscope`.

- `pto.vecscope` allows the body to use surrounding SSA values directly.
- `pto.strict_vecscope` requires every external value used by the body to be passed through the op operand list and received as a body block argument.
- `pto.strict_vecscope` rejects implicit capture from the surrounding scope.
- both ops still represent one explicit VPTO vector interval.
- regardless of whether the source form uses `pto.vecscope`,
  `pto.strict_vecscope`, or a lowered carrier loop with
  `llvm.loop.aivector_scope`, every op that produces or consumes `!pto.vreg`,
  `!pto.mask<...>`, or `!pto.align` must be enclosed by exactly one vector
  interval
- nested vector intervals are not part of the legal VPTO surface; ordinary
  nested `scf.for` structure is fine, but one vector interval may not contain
  another vector interval

### Example: VecScope

```mlir
pto.mte_gm_ub %7, %2, %c0_i64, %c128_i64
  nburst(%c32_i64, %c128_i64, %c128_i64)
  : !pto.ptr<f32, gm>, !pto.ptr<f32, ub>, i64, i64, i64

pto.set_flag["PIPE_MTE2", "PIPE_V", "EVENT_ID0"]
pto.wait_flag["PIPE_MTE2", "PIPE_V", "EVENT_ID0"]

pto.vecscope {
  scf.for %lane = %c0 to %9 step %c64 {
    %mask = pto.pset_b32 "PAT_ALL" : !pto.mask<b32>
    %v = pto.vlds %2[%lane] : !pto.ptr<f32, ub> -> !pto.vreg<64xf32>
    %abs = pto.vabs %v, %mask : !pto.vreg<64xf32>, !pto.mask<b32> -> !pto.vreg<64xf32>
    pto.vsts %abs, %8[%lane], %mask : !pto.vreg<64xf32>, !pto.ptr<f32, ub>, !pto.mask<b32>
  }
}

pto.set_flag["PIPE_V", "PIPE_MTE3", "EVENT_ID0"]
pto.wait_flag["PIPE_V", "PIPE_MTE3", "EVENT_ID0"]
pto.mte_ub_gm %8, %14, %c128_i64
  nburst(%c32_i64, %c128_i64, %c128_i64) l2_cache_ctl(%c0_i64)
  : !pto.ptr<f32, ub>, !pto.ptr<f32, gm>, i64, i64, i64, i64, i64
```

### Example: Strict VecScope

```mlir
pto.strict_vecscope(%ub_in, %ub_out, %lane, %remaining) {
^bb0(%in: !pto.ptr<f32, ub>, %out: !pto.ptr<f32, ub>, %iv: index, %rem: i32):
  %mask, %next_remaining = pto.plt_b32 %rem : i32 -> !pto.mask<b32>, i32
  %v = pto.vlds %in[%iv] : !pto.ptr<f32, ub> -> !pto.vreg<64xf32>
  %abs = pto.vabs %v, %mask : !pto.vreg<64xf32>, !pto.mask<b32> -> !pto.vreg<64xf32>
  pto.vsts %abs, %out[%iv], %mask : !pto.vreg<64xf32>, !pto.ptr<f32, ub>, !pto.mask<b32>
} : (!pto.ptr<f32, ub>, !pto.ptr<f32, ub>, index, i32) -> ()
```

Use `pto.strict_vecscope` when the source form should make all vector-scope inputs explicit in the region signature instead of relying on surrounding SSA visibility. The scope op itself only defines the vector-interval boundary and region argument contract.

### Cluster Programming Model

#### Overview

An A5 cluster contains one **Cube block** (AIC) and two **Vector blocks** (AIV0, AIV1). Each
block runs an **independent program** under its own Scalar Unit (SU), with its own issue queues:

| Block | Issue Queues |
|---|---|
| Cube (AIC) | MTE2, MTE1, CUBE, FIXP |
| Vector (AIV) | MTE2, VEC, MTE3 |

There is no implicit synchronization between blocks. All coordination between the Cube and Vector
programs is **explicit**, via the primitives described below.


```
┌─────────────────────────────────────── A5 CLUSTER ───────────────────────────────────────┐
│                                                                                           │
│  ┌─────────────────────┐    ┌─────────────────────┐    ┌─────────────────────┐           │
│  │   CUBE CORE (AIC)   │    │  VECTOR 0 (AIV0)    │    │  VECTOR 1 (AIV1)    │           │
│  │                     │    │   subblock_id = 0   │    │   subblock_id = 1   │           │
│  │  ┌───────────────┐  │    │  ┌───────────────┐  │    │  ┌───────────────┐  │           │
│  │  │  Scalar Unit  │  │    │  │  Scalar Unit  │  │    │  │  Scalar Unit  │  │           │
│  │  │  (SU)         │  │    │  │  (SU)         │  │    │  │  (SU)         │  │           │
│  │  │  runs cube    │  │    │  │  runs vec     │  │    │  │  runs vec     │  │           │
│  │  │  program      │  │    │  │  program      │  │    │  │  program      │  │           │
│  │  └───────────────┘  │    │  └───────────────┘  │    │  └───────────────┘  │           │
│  │   ── Issue Queues ─ │    │   ── Issue Queues ─ │    │   ── Issue Queues ─ │           │
│  │  ┌───────────────┐  │    │  ┌───────────────┐  │    │  ┌───────────────┐  │           │
│  │  │     MTE2      │  │    │  │     MTE2      │  │    │  │     MTE2      │  │           │
│  │  │    GM → L1    │  │    │  │    GM → UB    │  │    │  │    GM → UB    │  │           │
│  │  ├───────────────┤  │    │  ├───────────────┤  │    │  ├───────────────┤  │           │
│  │  │     MTE1      │  │    │  │      VEC      │  │    │  │      VEC      │  │           │
│  │  │   L1 → L0A/B  │  │    │  │  SIMD compute │  │    │  │  SIMD compute │  │           │
│  │  ├───────────────┤  │    │  ├───────────────┤  │    │  ├───────────────┤  │           │
│  │  │     CUBE      │  │    │  │     MTE3      │  │    │  │     MTE3      │  │           │
│  │  │  MMAD (L0C)   │  │    │  │    UB → GM    │  │    │  │    UB → GM    │  │           │
│  │  ├───────────────┤  │    │  └───────────────┘  │    │  └───────────────┘  │           │
│  │  │     FIXP      │  │    │                     │    │                     │           │
│  │  │  L0C → UB     │  │    │                     │    │                     │           │
│  │  │  (fixpipe)    │  │    │                     │    │                     │           │
│  │  └───────────────┘  │    │                     │    │                     │           │
│  └─────────────────────┘    └─────────────────────┘    └─────────────────────┘           │
│                                                                                           │
│  ┌────────────────────── SC (System Controller) ──────────────────────────────────────┐  │
│  │                                                                                     │  │
│  │   32 semaphores · 4-bit counter each · shared for C→V and V→C directions           │  │
│  │                                                                                     │  │
│  │   ┌──────────────────────────────────────────────────────────────────────────────┐ │  │
│  │   │  sema_id 0 –15  │ [ 0][ 1][ 2][ 3][ 4][ 5][ 6][ 7][ 8][ 9][10][11][12][13][14][15] │ │  │
│  │   │                 │                    ↕  C→V / V→C  ↕                         │ │  │
│  │   │                 │              communicate with AIV0 (subblock_id=0)          │ │  │
│  │   ├──────────────────────────────────────────────────────────────────────────────┤ │  │
│  │   │  sema_id 16–31  │ [16][17][18][19][20][21][22][23][24][25][26][27][28][29][30][31] │ │  │
│  │   │                 │                    ↕  C→V / V→C  ↕                         │ │  │
│  │   │                 │              communicate with AIV1 (subblock_id=1)          │ │  │
│  │   └──────────────────────────────────────────────────────────────────────────────┘ │  │
│  │                                                                                     │  │
│  │   → 16 sema_id pairs (0–15) available for 1:2 C:V sync per slot                   │  │
│  │                                                                                     │  │
│  │   set_intra_block(trigger_pipe, sema_id)  ──►  increments semaphore                │  │
│  │   wait_intra_core(wait_pipe,    sema_id)  ──►  stalls pipe until semaphore > 0     │  │
│  │                                                                                     │  │
│  └─────────────────────────────────────────────────────────────────────────────────────┘  │
└───────────────────────────────────────────────────────────────────────────────────────────┘
```

#### Intra-Cluster Synchronization

Within a cluster, the PTO micro ISA provides two levels of synchronization:

**Intra-core pipeline sync** (`pto.set_flag` / `pto.wait_flag`): coordinates the asynchronous
pipelines *within a single block* — for example, ensuring MTE2 completes a GM→UB load before
the VEC pipeline begins computation. This does not cross block boundaries.

**Inter-block sync** (`pto.set_intra_block` / `pto.wait_intra_core`): coordinates between the
Cube block and a Vector block within the same cluster. The sender specifies which **local
pipeline** commits the signal, ensuring the preceding operation on that pipeline has completed
before the signal is issued. The receiver specifies which **local pipeline** should stall until
the signal arrives. This is the fundamental IPC primitive for Cube–Vector cooperation on A5.

For the public `pto.sync.set` / `pto.sync.wait` surface on A5, event IDs are physical semaphore
IDs in the `0-31` range. AIV subblock 0 uses event IDs `0-15`, and AIV subblock 1 uses event
IDs `16-31`. Code that signals or waits for both AIV subblocks should explicitly emit both
the base ID and `base_id + 16`.

> **Note:** `pto.set_cross_core` / `pto.wait_cross_core` operate at **multi-cluster** scope and
> are not used for intra-cluster communication.

#### Intra-Cluster Data Paths

A5 provides dedicated on-chip data paths between the Cube and Vector blocks, bypassing Global
Memory entirely. These are the **recommended high-performance paths** for intra-cluster tile
exchange.

##### C→V: Cube L0C → Vector UB (fixpipe)

The **fixpipe** instruction transfers data directly from Cube's L0C buffer to a Vector block's UB.
Because Cube natively produces results in **NZ fractal layout** and Vector operates on **ND
(row-major) layout**, fixpipe performs the layout conversion in hardware:

```
Cube L0C  (NZ layout)  ──[fixpipe, NZ2ND]──▶  Vector UB  (ND layout)
```

Fixpipe supports a **dual-destination mode**: a single transfer can write to *both* AIV0's UB and
AIV1's UB simultaneously, with the tile split in hardware along either the row axis
(`DualModeSplitM`) or the column axis (`DualModeSplitN`):

| Split | AIV0 receives | AIV1 receives |
|---|---|---|
| Split-M (rows) | Upper `[M/2, N]` in ND | Lower `[M/2, N]` in ND |
| Split-N (cols) | Left `[M, N/2]` in ND | Right `[M, N/2]` in ND |

This 1→2 broadcast with in-hardware tile split is the architectural basis for 1:2
Cube-to-Vector tile distribution.

##### V→C: Vector UB → Cube L1

The reverse path transfers data from a Vector block's UB into Cube's L1 buffer.
A key architectural constraint: Cube's L1 stores tiles in **NZ fractal layout** (e.g.
`K1M1M0K0` — for fp16: `K0=16`, `M0=16`) so they can be loaded into L0A/L0B for MMAD
computation. Since Vector produces tiles in **ND layout**, the layout conversion from ND to NZ
must be applied as part of the V→C transfer:

```
Vector UB  (ND layout)  ──[ND→NZ movement]──▶  Cube L1  (NZ K1M1M0K0)
```

For 1:2 mode, both AIV0 and AIV1 each transfer a sub-tile into Cube's L1. The two sub-tiles are
assembled into a single contiguous NZ Mat tile in L1, ready for use as a LeftTile or RightTile
input to MMAD:

| Split | AIV0 writes to L1 | AIV1 writes to L1 | Assembled in L1 |
|---|---|---|---|
| Split-M (rows) | `[K/2, N]` NZ at base | `[K/2, N]` NZ at offset | Full `[K, N]` NZ Mat tile |
| Split-N (cols) | `[K, N/2]` NZ at base | `[K, N/2]` NZ at offset | Full `[K, N]` NZ Mat tile |

##### Fallback: GM-Staged Transfer

When the local data path is not applicable, data can be exchanged via a **Global Memory staging
buffer**: the producer DMAs data to GM, and the consumer DMAs from GM. This path incurs off-chip
bandwidth cost and higher latency, but serves as a general fallback.

#### Cube Internal Buffer Layout: NZ Fractal Format

All cube unit internal buffers (L1/cbuf, L0A, L0B, L0C) use a **fractal NZ layout** rather than
row-major ND. Understanding this layout is essential when authoring cube data-movement ops.

##### Definition

Given hardware constant `C0 = 32 bytes`, for element type with byte width `E = sizeof(T)`:

- Inner tile width: `K0 = N0 = C0 / E` (e.g. `K0 = 16` for fp16/bf16)
- Inner tile height: `M0 = 16`

NZ re-indexing for a logical `[M, K]` tensor:

```
NZ index: (k1, m1, m0, k0)
  where  k1 = k / K0,  k0 = k % K0
         m1 = m / M0,  m0 = m % M0
Physical layout: K1 x M1 x M0 x K0  (last dimension contiguous)
```

##### Per-buffer NZ Layouts

| Buffer | Logical shape | Physical NZ layout | Notes |
|--------|--------------|-------------------|-------|
| L1 (cbuf) - Tensor A | `[M, K]` | `K1 M1 M0 K0` | Row-major A staged into NZ layout |
| L1 (cbuf) - Tensor B | `[K, N]` | `K1 N1 K0 N0` | Row-major B staged into NZ layout |
| L0A (left operand)   | -        | `K1 M1 M0 K0` | FRACTAL_NZ (A5) / FRACTAL_ZZ (A3): same NZ order as L1 cbuf |
| L0B (right operand)  | -        | `K1 N1 N0 K0` | FRACTAL_ZN: row-major outer, col-major inner (K0 innermost) |
| L0C (accumulator)    | `[M, N]` | `N1 M1 M0 N0` | output of MMAD (FRACTAL_NZ: col-major outer, row-major inner) |

##### Data Flow: GM -> L1 -> L0A/B -> L0C

```
+------------------------------------------------------------------------------+
|              GEMM Data Layout: GM -> L1 (NZ) -> L0A/B -> L0C               |
+------------------------------------------------------------------------------+

STEP 1 - Global Memory (ND, row-major)
--------------------------------------
 Tensor A [M, K]                     Tensor B [K, N]
 (K is the contiguous axis)          (N is the contiguous axis)

  col->  k0  k1  ...  kK-1             col->  n0  n1  ...  nN-1
row|   +--------------------+         row|   +--------------------+
  m0  | a00 a01 ...        |           k0  | b00 b01 ...        |
  m1  | a10 a11 ...        |           k1  | b10 b11 ...        |
  ... |                    |           ... |                    |
  mM-1|                    |          kK-1 |                    |
      +--------------------+               +--------------------+
  Physical: A[m*K + k]                 Physical: B[k*N + n]


STEP 2 - GM -> L1 (cbuf): NDtoNZ fractal repack
-------------------------------------------------
 Use the structured cube load surface to stage row-major A and B into L1 NZ layout.

 A in L1: K1 x M1 x M0 x K0          B in L1: K1 x N1 x K0 x N0
 For each outer block (k1, m1):       For each outer block (k1, n1):
 +----------------------------+       +----------------------------+
 |  M0 rows x K0 cols         |       |  K0 rows x N0 cols         |
 |  (16x16 elems contiguous)  |       |  (16x16 elems contiguous)  |
 |  m0|  k0-> [0 .. K0-1]     |       |  k0|  n0-> [0 .. N0-1]     |
 |   0   [a a a a ...]        |       |   0   [b b b b ...]        |
 |   1   [a a a a ...]        |       |   1   [b b b b ...]        |
 |  ...                       |       |  ...                       |
 |  M0-1 [a a a a ...]        |       |  K0-1 [b b b b ...]        |
 +----------------------------+       +----------------------------+
 Physical: A_nz[k1][m1][m0][k0]       Physical: B_nz[k1][n1][k0][n0]



 NOTE: For GEMM with row-major A/B, stage both operands from GM to L1 as
   logical ND-to-NZ movement. If the source is already in a transposed logical
   layout, express that at the structured load level instead of relying on a
   later interpretation of the same bytes.


STEP 3 - L1 -> L0A / L0B
--------------------------
 L0A: cbuf K1 M1 M0 K0 --mte_l1_l0a-->  L0A K1 M1 M0 K0  (FRACTAL_NZ on A5)
 L0B: cbuf K1 N1 K0 N0 --mte_l1_l0b--> L0B K1 N1 N0 K0  (FRACTAL_ZN, K0 innermost)

 Why transpose at L1->L0B and not at GM->L1?
 --------------------------------------------
 The cube reduction axis is K. L0B requires K innermost (N1 K1 K0 N0)
 so the cube hardware reads all K0 elements per cycle without striding.
 The inner-box transpose is performed as part of the structured right-load
 movement itself; no separate user-visible pass is required.
 Each 512B fractal z-block is permuted as it moves from L1 to L0B.

  L0A tile (cube LEFT port):           L0B tile (cube RIGHT port):
  +---------------------+              +---------------------+
  |  shape: [M0, K0]    |       x      |  shape: [K0, N0]    |
  |  M0 rows, K0 cols   |              |  K0 rows, N0 cols   |
  |  K innermost (fast) |              |  K innermost (fast) |
  +---------------------+              +---------------------+
          |                                      |
          +-----------------+--------------------+
                            |  pto.mad (MMAD)
                            v

STEP 4 - L0C output layout: N1 M1 M0 N0
-----------------------------------------
  For each outer block (n1, m1):
  +------------------------------+
  |  M0 rows x N0 cols           |
  |  = result sub-tile of C[M,N] |
  |  n0->  [0 .. N0-1]           |
  |  m0|  [c c c c ...]          |
  |       [c c c c ...]          |
  +------------------------------+
  Physical: C_nz[n1][m1][m0][n0]  ->  C_nd[m1*M0+m0][n1*N0+n0]

  Writeback: FIXPIPE MTE ops convert the L0C NZ result to the requested
             destination layout and memory space.


Full pipeline summary
----------------------
  GM (ND)          L1/cbuf (NZ)            L0A/B (NZ)          L0C (NZ)    GM (ND)

  A[M,K] --mte_gm_l1_frac/mte_gm_l1--> K1 M1 M0 K0 --mte_l1_l0a-->  K1 M1 M0 K0 -+
                                                               +-MAD-> N1 M1 M0 N0 --> C[M,N]
  B[K,N] --mte_gm_l1_frac/mte_gm_l1--> K1 N1 K0 N0 --mte_l1_l0b--> K1 N1 N0 K0 -+
                                 ^
                      transpose as part of mte_l1_l0b when requested
                      NOT at GM->L1
```



#### Programming Model

The common pattern for Cube–Vector co-programming is a **software pipeline**: the Cube and Vector
programs run a coordinated loop where each iteration the Cube produces a tile and the Vector
consumes it (or vice versa), with explicit `pto.set_intra_block` / `pto.wait_intra_core`
handshakes at each step to maintain correct data ordering.

The PTO micro ISA exposes all the hardware primitives above directly. Higher-level constructs
that simplify this pattern (such as in-order FIFO abstractions) can be implemented as software
libraries on top of these primitives; they are not part of the ISA itself.


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

### BlockDim Query Operations

These ops expose the current kernel instance's execution coordinates to scalar code. They are the PTO-level equivalent of runtime queries such as `GetBlockIdx()` and `GetBlockNum()` in kernel programming models.

Use them when the same kernel body is launched across multiple blocks or subblocks and each execution instance must figure out which slice of the global workload it owns.

A common pattern is:

- split the full input/output tensor into `block_num` disjoint block-sized regions
- let each block compute its own starting offset from `block_idx`
- within one block, further tile the local region and drive the tile loop with ordinary scalar `arith` / `scf` ops

For example, if a tensor is split evenly across 8 blocks and each block handles `block_length = 2048` elements, then block `b` owns the global range `[b * block_length, (b + 1) * block_length)`. The per-block GM base pointer can be formed by adding `block_idx * block_length` elements to the original base pointer.

At the PTO micro Instruction level, these runtime-query ops are pure scalar producers. They do not perform data movement, do not allocate memory, and do not by themselves create tiling or double buffering. Instead, they provide the scalar values used by surrounding address computation and structured control flow.

#### Example: block-level data partitioning

```mlir
%block = pto.get_block_idx
%block_num = pto.get_block_num
%block_len = arith.constant 2048 : index
%base = arith.index_cast %block : i64 to index
%offset = arith.muli %base, %block_len : index
%block_in = pto.addptr %gm_in, %offset : !pto.ptr<f32, gm> -> !pto.ptr<f32, gm>
%block_out = pto.addptr %gm_out, %offset : !pto.ptr<f32, gm> -> !pto.ptr<f32, gm>
```

In this pattern, all blocks execute the same kernel body, but each block sees a different `%block` value and therefore computes a different GM window.

#### `pto.get_block_idx`

- **syntax:** `%block = pto.get_block_idx`
- **result:** `i64`
- **semantics:** Return the current block ID in the range `[0, pto.get_block_num())`.

```c
block = block_idx();
```

#### `pto.get_subblock_idx`

- **syntax:** `%subblock = pto.get_subblock_idx`
- **result:** `i64`
- **semantics:** Return the current subblock ID in the range `[0, pto.get_subblock_num())`.

```c
subblock = subblock_idx();
```

#### `pto.get_block_num`

- **syntax:** `%block_num = pto.get_block_num`
- **result:** `i64`
- **semantics:** Return the total number of launched blocks visible to the current kernel instance.

```c
block_num = block_num();
```

#### `pto.get_subblock_num`

- **syntax:** `%subblock_num = pto.get_subblock_num`
- **result:** `i64`
- **semantics:** Return the total number of visible subblocks for the current execution instance.

```c
subblock_num = subblock_num();
```

#### `pto.store_vfsimt_info`

- **syntax:** `pto.store_vfsimt_info %dim_z, %dim_y, %dim_x : i32, i32, i32`
- **operands:** `i32, i32, i32`
- **semantics:** Configure the SIMT VF launch descriptor consumed by a subsequent SIMT entry invocation. The three operands are the launch dimensions in `z, y, x` order.
- **placement:** This op must appear in the outer non-SIMT caller. It must not appear inside a function marked with `pto.simt_entry`.

```c
store_vfsimt_info(dim_z, dim_y, dim_x);
```

#### `pto.simt_launch`

- **syntax:** `pto.simt_launch @body<<<%dim_x, %dim_y, %dim_z>>>(%arg0, ...) : (arg_types...) -> ()`
- **operands:** `%dim_x`, `%dim_y`, and `%dim_z` are `i32` workitem counts in `x, y, z` launch order. The remaining operands are passed to `@body`.
- **semantics:** Invoke the SIMT entry `@body` for the workitem space described by `%dim_x * %dim_y * %dim_z`. Workitems in `@body` observe thread coordinates through the SIMT query ops.
- **placement:** This op must appear in the outer non-SIMT caller. The callee must be marked with `pto.simt_entry` and must return no values.

```mlir
pto.simt_launch @simt_write<<<%dim_x, %dim_y, %dim_z>>>(%ub_out)
  : (!pto.ptr<i32, ub>) -> ()
```

#### `pto.get_tid_x`

- **syntax:** `%tx = pto.get_tid_x : i32`
- **result:** `i32`
- **semantics:** Return the current SIMT lane X coordinate inside the active VF launch.

```c
tx = get_tid_x();
```

#### `pto.get_tid_y`

- **syntax:** `%ty = pto.get_tid_y : i32`
- **result:** `i32`
- **semantics:** Return the current SIMT lane Y coordinate inside the active VF launch.

```c
ty = get_tid_y();
```

#### `pto.get_tid_z`

- **syntax:** `%tz = pto.get_tid_z : i32`
- **result:** `i32`
- **semantics:** Return the current SIMT lane Z coordinate inside the active VF launch.

```c
tz = get_tid_z();
```

Example:

```mlir
module attributes {pto.target_arch = "a5", pto.kernel_kind = #pto.kernel_kind<vector>} {
  func.func @simt_store_tid_kernel(%out: !pto.ptr<i32, gm>) attributes {pto.kernel} {
    %c0_i64 = arith.constant 0 : i64
    %c32_i64 = arith.constant 32 : i64
    %c128_i64 = arith.constant 128 : i64
    %dim_z = arith.constant 1 : i32
    %dim_y = arith.constant 32 : i32
    %dim_x = arith.constant 32 : i32

    %ub_out = pto.castptr %c0_i64 : i64 -> !pto.ptr<i32, ub>
    pto.store_vfsimt_info %dim_z, %dim_y, %dim_x : i32, i32, i32
    func.call @simt_write(%ub_out) : (!pto.ptr<i32, ub>) -> ()

    pto.set_flag["PIPE_V", "PIPE_MTE3", "EVENT_ID0"]
    pto.wait_flag["PIPE_V", "PIPE_MTE3", "EVENT_ID0"]
    pto.dma_store %ub_out, %out, %c128_i64
      nburst(%c32_i64, %c128_i64, %c128_i64)
      : !pto.ptr<i32, ub>, !pto.ptr<i32, gm>, i64, i64, i64, i64
    return
  }

  func.func @simt_write(%dst: !pto.ptr<i32, ub>) attributes {pto.simt_entry} {
    %tx = pto.get_tid_x : i32
    %ty = pto.get_tid_y : i32
    %tz = pto.get_tid_z : i32
    %c8_i32 = arith.constant 8 : i32
    %c16_i32 = arith.constant 16 : i32
    %c32_i32 = arith.constant 32 : i32
    %ty_shift = arith.shli %ty, %c8_i32 : i32
    %tz_shift = arith.shli %tz, %c16_i32 : i32
    %xy = arith.ori %tx, %ty_shift : i32
    %xyz = arith.ori %xy, %tz_shift : i32
    %lane_base = arith.muli %ty, %c32_i32 : i32
    %tid = arith.addi %lane_base, %tx : i32
    %tid_idx = arith.index_castui %tid : i32 to index
    pto.store %xyz, %dst[%tid_idx] : !pto.ptr<i32, ub>, i32
    return
  }
}
```

Typical usage:

```mlir
%block = pto.get_block_idx
%subblock = pto.get_subblock_idx
%block_num = pto.get_block_num
%subblock_num = pto.get_subblock_num
```

### VMS4 Status Query

#### `pto.get_vms4_sr`

- **syntax:** `%list0, %list1, %list2, %list3 = pto.get_vms4_sr : i16, i16, i16, i16`
- **results:** four `i16` values
- **semantics:** Read `VMS4_SR` and return the finished element counts for
  source lists 0, 1, 2, and 3. After an exhausted `pto.vmrgsort4`, these are
  the per-source-list executed counts.

| Bits | Meaning |
|------|---------|
| `[15:0]` | finished count for source list 0 |
| `[31:16]` | finished count for source list 1 |
| `[47:32]` | finished count for source list 2 |
| `[63:48]` | finished count for source list 3 |

```c
status = VMS4_SR;
list0 = (uint16_t)(status & 0xffff);
list1 = (uint16_t)((status >> 16) & 0xffff);
list2 = (uint16_t)((status >> 32) & 0xffff);
list3 = (uint16_t)((status >> 48) & 0xffff);
```

### Core Types

### Element Types
`vreg<T>`: `!pto.vreg<NxT>` Fixed-width PTO micro Instruction vector type with total width exactly 256 bytes (2048 bits). `N` is the lane count, `T` is the element type, and `N * bitwidth(T) = 2048`.

| Type | Bits | Description |
|------|------|-------------|
| `i8` / `si8` / `ui8` | 8 | Signless/signed/unsigned 8-bit integer |
| `i16` / `si16` / `ui16` | 16 | Signless/signed/unsigned 16-bit integer |
| `i32` / `si32` / `ui32` | 32 | Signless/signed/unsigned 32-bit integer |
| `i64` / `si64` / `ui64` | 64 | Signless/signed/unsigned 64-bit integer |
| `f16` | 16 | IEEE 754 half precision |
| `bf16` | 16 | Brain floating point |
| `f32` | 32 | IEEE 754 single precision |

### Mask Types

`mask<G>`: `!pto.mask<G>` Typed predicate-register view. `G` is one of `b8`, `b16`, `b32` and records the byte-granularity interpretation used by VPTO ops and verifiers.

Typed masks are also the primary legality contract for predicated VPTO code:

- vector ops over `f32`, `i32`, `si32`, and `ui32` consume `!pto.mask<b32>`
- vector ops over `f16`, `bf16`, `i16`, `si16`, and `ui16` consume
  `!pto.mask<b16>`
- vector ops over 8-bit element families consume `!pto.mask<b8>`
- compare families keep seed-mask and result-mask granularity aligned with the
  compared vector family
- carry families keep carry-in, carry-out, and execution-mask granularity
  aligned with the data-vector family
- mask-only ops that do not explicitly change granularity preserve the same `G`

### Address Space Conventions

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

### `!pto.ptr<T, space>`

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

#### `pto.load_scalar`

- **syntax:** `%value = pto.load_scalar %ptr[%offset] : !pto.ptr<T, space> -> T`
- **semantics:** Load one scalar element from a pointer-like operand.

```c
value = ptr[offset];
```

- **inputs:**
  `%ptr` is a typed PTO pointer `!pto.ptr<T, space>`, and `%offset` is an
  `index` displacement counted in elements.
- **outputs:**
  `%value` is the loaded scalar element.
- **constraints and limitations:**
  The result type MUST match the element type of `%ptr`. This op is a scalar
  memory helper; unlike `pto.vlds`, it does not produce a `vreg` result and
  does not participate in vector load `dist` families.

#### `pto.store_scalar`

- **syntax:** `pto.store_scalar %value, %ptr[%offset] : !pto.ptr<T, space>, T`
- **semantics:** Store one scalar element to a pointer-like operand.

```c
ptr[offset] = value;
```

- **inputs:**
  `%value` is the scalar value to store. `%ptr` is a typed PTO pointer
  `!pto.ptr<T, space>`, and `%offset` is an `index` displacement counted in
  elements.
- **constraints and limitations:**
  The stored value type MUST match the element type of `%ptr`. This op is a
  scalar memory helper; unlike `pto.vsts`, it does not consume a mask and does
  not target vector-store `dist` families.

#### `pto.load`

- **syntax:** `%value = pto.load %ptr[%offset] : !pto.ptr<T, space> -> T`
- **semantics:** Load one scalar element from a VPTO pointer-like operand.

```c
value = ptr[offset];
```

- **inputs:**
  `%ptr` is a typed PTO pointer `!pto.ptr<T, space>` or a memref operand that
  will be normalized to a PTO pointer before LLVM emission. `%offset` is an
  `index` displacement counted in elements.
- **outputs:**
  `%value` is the loaded scalar element.
- **constraints and limitations:**
  The result type MUST match the element type of `%ptr`. This is the preferred
  scalar memory op for VPTO/SIMT authoring.

#### `pto.store`

- **syntax:** `pto.store %value, %ptr[%offset] : !pto.ptr<T, space>, T`
- **semantics:** Store one scalar element to a VPTO pointer-like operand.

```c
ptr[offset] = value;
```

- **inputs:**
  `%value` is the scalar value to store. `%ptr` is a typed PTO pointer
  `!pto.ptr<T, space>` or a memref operand that will be normalized to a PTO
  pointer before LLVM emission. `%offset` is an `index` displacement counted in
  elements.
- **constraints and limitations:**
  The stored value type MUST match the element type of `%ptr`. This is the
  preferred scalar memory op for VPTO/SIMT authoring.

#### Pointer-Based Vector Access Example

The following lowered-style fragment shows how typed PTO pointers flow through
pointer construction, pointer arithmetic, structured control flow, and PTO
memory ops. Scalar memory access is expressed on `!pto.ptr<T, space>` in
general, but the common VPTO pattern here is UB-local scalar access alongside
UB vector loads/stores:

```mlir
%0 = pto.castptr %c0 : i64 -> !pto.ptr<f32, ub>
%1 = pto.addptr %0, %c1024 : !pto.ptr<f32, ub> -> !pto.ptr<f32, ub>
pto.vecscope {
  %16 = scf.for %arg3 = %c0 to %11 step %c64 iter_args(%arg4 = %12) -> (i32) {
    %mask, %scalar_out = pto.plt_b32 %arg4 : i32 -> !pto.mask<b32>, i32
    %s = pto.load %1[%c4] : !pto.ptr<f32, ub> -> f32
    pto.store %s, %1[%c8] : !pto.ptr<f32, ub>, f32
    %17 = pto.vlds %1[%arg3] : !pto.ptr<f32, ub> -> !pto.vreg<64xf32>
    %18 = pto.vabs %17, %mask : !pto.vreg<64xf32>, !pto.mask<b32> -> !pto.vreg<64xf32>
    pto.vsts %18, %10[%arg3], %mask : !pto.vreg<64xf32>, !pto.ptr<f32, ub>, !pto.mask<b32>
    scf.yield %scalar_out : i32
  }
}
```

In this pattern, `pto.castptr` materializes a typed UB pointer, `pto.addptr` shifts the base by 1024 `f32` elements, and the subsequent `[%arg3]` indexing on `pto.vlds` / `pto.vsts` applies an additional element offset relative to that base.

### Special Types

#### `!pto.mask<G>`

`!pto.mask<G>` models an A5 predicate register (256-bit) under a typed granularity view, not an integer vector.

`G` is part of the type and MUST be one of:

- `b32`
- `b16`
- `b8`

All three forms describe the same physical 256-bit predicate-register class. The type parameter does not encode how many lanes are currently active. Instead, it records how VPTO interprets the register when matching mask-producing ops, mask-consuming ops, and verifier legality rules.

In the ISA chapters below, this document uses `!pto.mask<G>` as shorthand when a
family is generic over granularity. For op families whose names already encode
the granularity, such as `pset_b32`, `pge_b16`, `plt_b8`,
`pdintlv_b8`, and `pintlv_b16`, examples use the corresponding concrete typed
mask.

**Mask Granularity:**

The predicate register is 256 bits in length, where each bit controls 1 byte of data. `G` therefore describes how many bytes form one logical element slot:

| Mask Type | Bytes / Element Slot | Typical Element Family | Derived Logical Lanes |
|-----------|----------------------|------------------------|-----------------------|
| `!pto.mask<b32>` | 4 | `f32` / `i32` | 64 |
| `!pto.mask<b16>` | 2 | `f16` / `bf16` / `i16` | 128 |
| `!pto.mask<b8>` | 1 | 8-bit element family | 256 |

This is intentionally different from a lane-vector model such as `mask<64xi1>`:

- `!pto.mask<b32>` still denotes a 256-bit predicate register;
- `64` is only the derived logical lane count for the `b32` view;
- value-level patterns such as `PAT_VL32` describe which lanes are active, not a different type.
- `pto.vaddc`, `pto.vsubc`, `pto.vaddcs`, and `pto.vsubcs` use `!pto.mask<G>`
  to carry their per-lane carry results, interpreted with this same
  granularity.

**Predication Behavior (Zero-Merge):**

The native hardware predication mode is **ZEROING** — inactive lanes produce zero:

```c
dst[i] = mask[i] ? op(src0[i], src1[i]) : 0    // ZEROING mode
```

```mlir
// Predicated add: inactive lanes produce zero
%mask = pto.pset_b32 "PAT_VL32" : !pto.mask<b32>   // first 32 logical b32 lanes active
%result = pto.vcmp %a, %b, %mask, "lt" : !pto.vreg<64xf32>, !pto.vreg<64xf32>, !pto.mask<b32> -> !pto.mask<b32>
```

```mlir
// Compare and select: generate mask from comparison, use for conditional select
%mask = pto.vcmp %lhs, %rhs, %seed, "lt" : !pto.vreg<64xf32>, !pto.vreg<64xf32>, !pto.mask<b32> -> !pto.mask<b32>
%out = pto.vsel %x, %y, %mask : !pto.vreg<64xf32>, !pto.vreg<64xf32>, !pto.mask<b32> -> !pto.vreg<64xf32>
```

#### `!pto.align`

`!pto.align` models the A5 vector-align carrier state. It is not payload data.

```mlir
%align = pto.vldas %ub : !pto.ptr<f32, ub> -> !pto.align
%vec, %align_out = pto.vldus %ub, %align : !pto.ptr<f32, ub>, !pto.align -> !pto.vreg<64xf32>, !pto.align

%store_align = pto.init_align : !pto.align
%next_align = pto.vstus %store_align, %offset, %vec, %ub
    : !pto.align, i32, !pto.vreg<64xf32>, !pto.ptr<f32, ub> -> !pto.align
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
%result, %updated_base = pto.vlds %source[%offset] {dist = "DIST"} : !pto.ptr<T, ub> -> !pto.vreg<NxT>, !pto.ptr<T, ub>
```

**Store (register to memory):**

```mlir
pto.vsts %value, %destination[%offset] {dist = "DIST"} : !pto.vreg<NxT>, !pto.ptr<T, ub>
%updated_base = pto.vsts %value, %destination[%offset] {dist = "DIST"} : !pto.vreg<NxT>, !pto.ptr<T, ub> -> !pto.ptr<T, ub>
```

**Dual Load (one load, two results — deinterleave):**

```mlir
%low, %high = pto.vldsx2 %source[%offset], "DIST" : !pto.ptr<T, ub>, index -> !pto.vreg<NxT>, !pto.vreg<NxT>
%low, %high, %updated_base = pto.vldsx2 %source[%offset], "DIST" : !pto.ptr<T, ub>, index -> !pto.vreg<NxT>, !pto.vreg<NxT>, !pto.ptr<T, ub>
```

**Dual Store (two inputs, one interleaved store):**

```mlir
pto.vstsx2 %low, %high, %dest[%offset], "DIST", %mask : !pto.vreg<NxT>, !pto.vreg<NxT>, !pto.ptr<T, ub>, index, !pto.mask<G>
```

The two vectors and destination use one identical element type. The A5 dual
store surface accepts 8/16/32-bit integers, `f16`, `bf16`, supported FP8
formats, and packed FP4 pair formats; it does not accept `f32` or 64-bit
elements.

**Compare (two vectors + seed mask in, mask out):**

```mlir
%mask = pto.vcmp %src0, %src1, %seed, "CMP_MODE" : !pto.vreg<NxT>, !pto.vreg<NxT>, !pto.mask<G> -> !pto.mask<G>
```

**Conversion (one vector in, different-typed vector out):**

```mlir
%result = pto.vcvt %input, %mask {rnd = "R", sat = "SAT", part = "EVEN"} : !pto.vreg<NxT0>, !pto.mask<G> -> !pto.vreg<MxT1>
```

**Predicate construction:**

```mlir
%mask = pto.pset_b32 "PAT_ALL" : !pto.mask<b32>
%tail = pto.pge_b32 "PAT_VL16" : !pto.mask<b32>
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
//   N = 256 for i8/si8/ui8
//   N = 128 for i16/si16/ui16/f16/bf16
//   N = 64  for i32/si32/ui32/f32
//   N = 32  for i64/si64/ui64
```

**Example — pto.vadd semantics:**

```c
for (int i = 0; i < N; i++)
    dst[i] = src0[i] + src1[i];
```

**Example — pto.vcgadd (group reduction per VLane) semantics:**

```c
int groups = 8;
int K = 32 / sizeof(T);  // elements per 32-byte VLane
for (int g = 0; g < 8; g++) {
    T sum = 0;
    for (int i = 0; i < K; i++)
        if (mask[g*K + i])
            sum += src[g*K + i];
    dst[g] = sum;
}
for (int i = groups; i < N; i++)
    dst[i] = 0;
```

For A5 reduction result types:

- `pto.vcadd` widens `i8 -> i16`, `u8 -> u16`, `i16 -> i32`, and `u16 -> u32`,
  with the lane count halved in each widening case.
- `pto.vcadd` keeps the same result type for `f16`, `f32`, `i32`, and `u32`.

### Template Placeholder Conventions

| Placeholder | Meaning |
|-------------|---------|
| `"SRC_PIPE"`, `"DST_PIPE"` | Pipeline identifiers: `"PIPE_MTE2"`, `"PIPE_V"`, `"PIPE_MTE3"` |
| `"EVENT_ID"` | Event identifier: `"EVENT_ID0"` etc. |
| `"DIST"` | Distribution mode string (see the relevant load/store ISA group in Part III) |
| `"CMP_MODE"` | Compare predicate: `eq \| ne \| lt \| le \| gt \| ge` |
| `"RND"` | Rounding mode: `R \| A \| F \| C \| Z \| O` |
| `"SAT"` | Saturation: `SAT \| NOSAT` |
| `"PART"` | Half selector: `EVEN \| ODD` |
| `"PAT_*"` | Predicate pattern literal |
| `T` | Element type (f32, f16, bf16, i32, i16, i8, etc.) |
| `N` | Lane count (`N * bitwidth(T) = 2048`) |

---

## Part III: ISA Instruction Reference
# Part III: ISA Instruction Reference — Summary

This section provides a categorized overview of all PTO micro Instruction operations plus the shared MLIR `arith` and `scf` ops that may appear in PTO micro Instruction programs. Detailed documentation for each group is available in the linked files.

---

## Instruction Groups

| # | Group | Description | Count | Details |
|---|-------|-------------|-------|---------|
| 1 | [Pipeline Sync](isa/micro-isa/01-pipeline-sync.md) | Intra-core pipeline synchronization | 5 | `pto.set_flag`, `pto.wait_flag`, `pto.pipe_barrier`, `pto.get_buf`, `pto.rls_buf` |
| 2 | [DMA Copy Programming](isa/micro-isa/02-dma-copy.md) | Public DMA transfer interface between GM↔UB, UB→UB, and UB→L1 | 4 | `pto.mte_gm_ub`, `pto.mte_ub_gm`, `pto.mte_ub_ub`, `pto.mte_ub_l1` |
| 3 | [Vector Load/Store](isa/micro-isa/03-vector-load-store.md) | UB↔vreg data movement with various access patterns | ~23 | `pto.vlds`, `pto.vldsx2`, `pto.vgather2`, `pto.vsts`, `pto.vstsx2`, `pto.vscatter`, `pto.sprclr`, `pto.sprsti`, `pto.sprsts`, etc. |
| 4 | [Predicate Load/Store](isa/micro-isa/04-predicate-load-store.md) | UB↔mask register movement | 5 | `pto.plds`, `pto.pldi`, `pto.psts`, `pto.psti`, `pto.pstu` |
| 5 | [Materialization & Predicate Ops](isa/micro-isa/05-materialization-predicate.md) | Scalar broadcast, predicate generation and manipulation | ~20 | `pto.vbr`, `pto.vdup`, `pto.pset_b*`, `pto.pge_b*`, `pto.plt_b*`, `pto.pltm_b*`, `pto.ppack`, `pto.punpack`, `pto.pnot`, `pto.psel`, etc. |
| 6 | [Unary Vector Ops](isa/micro-isa/06-unary-vector-ops.md) | Single-input element-wise operations | 7 | `pto.vabs`, `pto.vneg`, `pto.vexp`, `pto.vln`, `pto.vsqrt`, `pto.vrelu`, `pto.vnot` |
| 7 | [Binary Vector Ops](isa/micro-isa/07-binary-vector-ops.md) | Two-input element-wise operations | 14 | `pto.vadd`, `pto.vsub`, `pto.vmul`, `pto.vdiv`, `pto.vmax`, `pto.vmin`, `pto.vmadd`, `pto.vand`, `pto.vor`, `pto.vxor`, `pto.vshl`, `pto.vshr`, `pto.vaddc`, `pto.vsubc` |
| 8 | [Vec-Scalar Ops](isa/micro-isa/08-vec-scalar-ops.md) | Vector-scalar operations | 9 | `pto.vadds`, `pto.vmuls`, `pto.vmaxs`, `pto.vmins`, `pto.vlrelu`, `pto.vshls`, `pto.vshrs`, `pto.vaddcs`, `pto.vsubcs` |
| 9 | [Conversion Ops](isa/micro-isa/09-conversion-ops.md) | Type conversion with rounding/saturation control | 4 | `pto.vcvt`, `pto.vtrc`, `pto.vbitcast`, `pto.pbitcast` |
| 10 | [Reduction Ops](isa/micro-isa/10-reduction-ops.md) | Vector reductions | 11 | `pto.vcadd`, `pto.vcmax`, `pto.vcmin`, `pto.vcbmax`, `pto.vcbmin`, `pto.vcgadd`, `pto.vcgmax`, `pto.vcgmin`, `pto.vcpadd`, `pto.chistv2`, `pto.dhistv2` |
| 11 | [Compare & Select](isa/micro-isa/11-compare-select.md) | Comparison and conditional selection | 4 (+1 not A5) | `pto.vcmp`, `pto.vcmps`, `pto.vsel`, `pto.vselr` (`pto.vselrv2` removed: not A5) |
| 12 | [Data Rearrangement](isa/micro-isa/12-data-rearrangement.md) | In-register data movement and permutation | 2 (+2 not A5) | `pto.vintlv`, `pto.vdintlv` (`pto.vintlvv2`, `pto.vdintlvv2` removed: not A5) |
| 13 | [DSA/SFU Ops](isa/micro-isa/13-dsa-sfu-ops.md) | Specialized ops, index generation, and sorting helpers | 11 | `pto.vlrelu`, `pto.vprelu`, `pto.vexpdif`, `pto.vaxpy`, `pto.vmulscvt`, `pto.vmull`, `pto.vmula`, `pto.vci`, `pto.vbitsort`, `pto.vmrgsort4`, `pto.get_vms4_sr` |
| 14 | [Arith (Shared MLIR Dialect)](isa/micro-isa/14-shared-arith.md) | Full scalar `arith` surface used around PTO ops; the companion page lists categories and representative examples | all scalar ops | `arith.constant`, `arith.addi`, `arith.addf`, `arith.cmpi`, `arith.cmpf`, `arith.select`, `arith.index_cast`, `arith.extsi`, `arith.trunci`, `arith.andi`, `arith.shli`, etc. |
| 15 | [SCF (Shared MLIR Dialect)](isa/micro-isa/15-shared-scf.md) | Structured loops, branches, and loop-carried state around PTO regions | 5 | `scf.for`, `scf.if`, `scf.while`, `scf.condition`, `scf.yield` |
| 16 | [Cube Matrix Multiply](isa/micro-isa/16-cube-matmul.md) | GM↔L1 (`l1`/cbuf) staging, L1 (`l1`)↔UB/BT/FB side moves, L1→L0A/L0B loads, L0C (`l0c`) matmul, and FIXPIPE MTE writeback | 19 | `pto.mte_gm_l1`, `pto.mte_l1_ub`, `pto.mte_gm_l1_frac`, `pto.mte_l1_bt`, `pto.mte_l1_fb`, `pto.mte_l1_l0a`, `pto.mte_l1_l0b`, `pto.mte_l1_l0a_mx`, `pto.mte_l1_l0b_mx`, `pto.mad`, `pto.mad_acc`, `pto.mad_bias`, `pto.mad_mx`, `pto.mad_mx_acc`, `pto.mad_mx_bias`, `pto.mte_l0c_l1`, `pto.mte_l0c_gm`, `pto.mte_l0c_ub` |
| 17 | [SIMT Ops](isa/micro-isa/17-simt.md) | SIMT launch, thread/lane queries, vote/shuffle/redux, scalar memory, atomics, scalar math, conversion, entry synchronization, and state preservation | ~65 | `pto.store_vfsimt_info`, `pto.simt_launch`, `pto.get_tid_x`, `pto.get_laneid`, `pto.vote_*`, `pto.shuffle_*`, `pto.redux_*`, `pto.load`, `pto.store`, `pto.atomic_*`, `pto.convert`, `pto.syncthreads`, `pto.keep`, `pto.resume`, etc. |

---

## Quick Reference by Category

### Memory Operations

| Operation | Group | Description |
|-----------|-------|-------------|
| GM→UB DMA | 2 | `pto.mte_gm_ub` |
| UB→GM DMA | 2 | `pto.mte_ub_gm` |
| UB→UB / UB→L1 copy | 2 | `pto.mte_ub_ub`, `pto.mte_ub_l1` |
| GM→L1 | 16 | `pto.mte_gm_l1`, `pto.mte_gm_l1_frac` |
| L1→UB | 16 | `pto.mte_l1_ub` |
| L1→BT | 16 | `pto.mte_l1_bt` |
| L1→FB | 16 | `pto.mte_l1_fb` |
| L1→L0A / L1→L0B | 16 | `pto.mte_l1_l0a`, `pto.mte_l1_l0b`, `pto.mte_l1_l0a_mx`, `pto.mte_l1_l0b_mx` |
| L0C→L1 / GM / UB (FIXPIPE MTE) | 16 | `pto.mte_l0c_l1`, `pto.mte_l0c_gm`, `pto.mte_l0c_ub` |
| Contiguous Load | 3 | `pto.vlds` with `NORM` dist |
| Broadcast Load | 3 | `pto.vlds` with `BRC` family dist |
| Gather | 3 | `pto.vgather2`, `pto.vgatherb` |
| Contiguous Store | 3 | `pto.vsts` with `NORM_B8` / `NORM_B16` / `NORM_B32` dist |
| Scatter | 3 | `pto.vscatter` |

### Compute Operations

| Operation | Group | Description |
|-----------|-------|-------------|
| Element-wise Arithmetic | 6, 7 | `pto.vadd`, `pto.vmul`, `pto.vabs`, etc. |
| Scalar Operations | 8 | `pto.vadds`, `pto.vmuls`, etc. |
| Transcendental | 6 | `pto.vexp`, `pto.vln`, `pto.vsqrt`, etc. |
| Reduction | 10 | `pto.vcadd`, `pto.vcmax`, `pto.vcmin` |
| Cube matmul family (zero-init / accumulate / bias-init; shared clauses `unit_flag`, `disable_gemv`, `sat`, `tf32_mode`, `n_dir`) | 16 | `pto.mad`, `pto.mad_acc`, `pto.mad_bias`, `pto.mad_mx`, `pto.mad_mx_acc`, `pto.mad_mx_bias` |
| Comparison | 11 | `pto.vcmp`, `pto.vcmps` |
| Selection | 11 | `pto.vsel`, `pto.vselr` |

### Type & Data Manipulation

| Operation | Group | Description |
|-----------|-------|-------------|
| Type Conversion | 9 | `pto.vcvt`, `pto.vbitcast`, `pto.pbitcast` |
| Interleave/Deinterleave | 12 | `pto.vintlv`, `pto.vdintlv` |
| Interleave/Deinterleave (not A5) | 12 | `pto.vintlvv2`, `pto.vdintlvv2` |

### Synchronization

| Operation | Group | Description |
|-----------|-------|-------------|
| Intra-core Sync | 1 | `pto.set_flag`, `pto.wait_flag` |
| Pipeline Buffer Sync | 1 | `pto.get_buf`, `pto.rls_buf` |
| Memory Barrier / Cache Maintenance | 1 | `pto.mem_bar`, `pto.dsb`, `pto.dcci` |

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

### Cube Operation Surface

- `pto.mte_l1_bt`
- `pto.mte_l1_fb`
- `pto.mte_gm_l1`
- `pto.mte_gm_l1_frac`
- `pto.mte_l1_ub`
- `pto.mte_l1_l0a`
- `pto.mte_l1_l0b`
- `pto.mte_l1_l0a_mx`
- `pto.mte_l1_l0b_mx`
- `pto.mad`
- `pto.mad_acc`
- `pto.mad_bias`
- `pto.mad_mx`
- `pto.mad_mx_acc`
- `pto.mad_mx_bias`
- `pto.mte_l0c_l1`
- `pto.mte_l0c_gm`
- `pto.mte_l0c_ub`

---

## Supported Data Types

| Type | Bits | vreg Lanes | Description |
|------|------|-----------|-------------|
| `i8` / `si8` / `ui8` | 8 | 256 | Signless/signed/unsigned 8-bit integer |
| `i16` / `si16` / `ui16` | 16 | 128 | Signless/signed/unsigned 16-bit integer |
| `f16` | 16 | 128 | IEEE 754 half precision |
| `bf16` | 16 | 128 | Brain floating point |
| `i32` / `si32` / `ui32` | 32 | 64 | Signless/signed/unsigned 32-bit integer |
| `f32` | 32 | 64 | IEEE 754 single precision |
| `i64` / `si64` / `ui64` | 64 | 32 | Signless/signed/unsigned 64-bit integer |

---

## Common Patterns

### Softmax (Numerically Stable)

```mlir
// 1. Find max
%max_vec = pto.vcmax %logits, %mask : !pto.vreg<64xf32>, !pto.mask<b32> -> !pto.vreg<64xf32>
pto.vsts %max_vec, %ub_tmp[%c0], %mask : !pto.vreg<64xf32>, !pto.ptr<f32, ub>, !pto.mask<b32>
%max_bc = pto.vlds %ub_tmp[%c0] {dist = "BRC_B32"} : !pto.ptr<f32, ub> -> !pto.vreg<64xf32>

// 2. exp(x - max) using fused op
%exp = pto.vexpdif %logits, %max_bc, %mask, "ODD" : !pto.vreg<64xf32>, !pto.vreg<64xf32>, !pto.mask<b32> -> !pto.vreg<64xf32>

// 3. Sum
%sum = pto.vcadd %exp, %mask : !pto.vreg<64xf32>, !pto.mask<b32> -> !pto.vreg<64xf32>
pto.vsts %sum, %ub_tmp[%c0], %mask : !pto.vreg<64xf32>, !pto.ptr<f32, ub>, !pto.mask<b32>
%sum_bc = pto.vlds %ub_tmp[%c0] {dist = "BRC_B32"} : !pto.ptr<f32, ub> -> !pto.vreg<64xf32>

// 4. Divide
%softmax = pto.vdiv %exp, %sum_bc, %mask : !pto.vreg<64xf32>, !pto.vreg<64xf32>, !pto.mask<b32> -> !pto.vreg<64xf32>
```

### ReLU Variants

```mlir
// Standard ReLU
%relu = pto.vrelu %input, %mask : !pto.vreg<64xf32>, !pto.mask<b32> -> !pto.vreg<64xf32>

// Leaky ReLU (scalar alpha)
%lrelu = pto.vlrelu %input, %alpha, %mask : !pto.vreg<64xf32>, f32, !pto.mask<b32> -> !pto.vreg<64xf32>

// Parametric ReLU (per-element alpha)
%prelu = pto.vprelu %input, %alpha_vec, %mask : !pto.vreg<64xf32>, !pto.vreg<64xf32>, !pto.mask<b32> -> !pto.vreg<64xf32>

```

### Data Layout Conversion

```mlir
// AoS → SoA (deinterleave)
%x, %y = pto.vldsx2 %ub_xy[%offset], "DINTLV_B16" : !pto.ptr<f16, ub>, index -> !pto.vreg<128xf16>, !pto.vreg<128xf16>

// SoA → AoS (interleave)
pto.vstsx2 %x, %y, %ub_xy[%offset], "INTLV_B16", %all_mask : !pto.vreg<128xf16>, !pto.vreg<128xf16>, !pto.ptr<f16, ub>, index, !pto.mask<b16>
```

---

*For detailed semantics, C-style pseudocode, and CCE mappings, see the individual group documentation files.*

---

## Part IV: PTO Tile Instruction

PTO Tile Instruction is a high-performance instruction surface built on top of PTO micro Instruction. Each tile instruction encapsulates a tile-granular pattern — DMA between GM and on-chip buffers, vector arithmetic over a whole tile, reductions, broadcast / expansion, selection, padding — and internally expands to a sequence of micro-instruction primitives (`pto.vlds`, `pto.vsts`, `pto.vadd`, mask ops, sync flags, …).

The full PTO Tile Instruction reference starts from [Tile and PTO Tile Instruction overview](isa/tile-op/01-tile-overview.md). It covers:

- [Tile and PTO Tile Instruction overview](isa/tile-op/01-tile-overview.md) — tile concept, on-chip placement, physical shape vs valid region, conventions
- [Types & Attributes](isa/tile-op/02-types-and-attributes.md) — `!pto.tile_buf`, `!pto.tensor_view`, address spaces, layout, pad
- [Pointer & View](isa/tile-op/03-pointer-and-view.md) — tensor views, partitions, tile allocation, valid-shape updates
- [DMA Data Movement](isa/tile-op/04-dma-data-movement.md) — `pto.tload` / `pto.tstore`
- [Vector Arithmetic](isa/tile-op/05-vector-arithmetic.md) — `pto.tadd / tsub / tmul / tdiv / tmax / tmin`, tile-scalar forms, unary math, activations
- [Reductions](isa/tile-op/06-reduction-ops.md), [Partial Elementwise](isa/tile-op/07-partial-elementwise.md), [Bitwise & Shift](isa/tile-op/08-bitwise-shift-ops.md), [Type Conversion](isa/tile-op/09-type-conversion.md), [Broadcast & Expansion](isa/tile-op/10-broadcast-and-expansion-ops.md), [Selection](isa/tile-op/11-selection-ops.md), [Fill & Padding](isa/tile-op/12-fill-and-padding-ops.md)

For the boundary between Tile Instruction and the micro instruction surface (when to drop into `pto.vecscope` and how `pto.tile_buf_addr` bridges the two), see [Tile and PTO Tile Instruction overview §1.10](isa/tile-op/01-tile-overview.md#110-mixing-pto-tile-instruction-and-pto-micro-instruction).

---

## Appendix: Discussion Points

### Part I

1. **mem_bar as pto op:** Should `pto.mem_bar` be a formal pto dialect op, or is there an existing mechanism?
2. **UB size parameterization:** Is 256KB always fixed, or should spec allow for architecture variants?
3. **MERGING predication:** Intentionally omitted (SW-emulated, perf overhead). Revisit if needed later.

### Part II

1. **Predication in C semantics:** Should every op's C code explicitly show the `if (mask[i])` guard, or assume all-active and note predication separately?
2. **VLane terminology:** Using "VLane" instead of "DataBlock" — confirm this naming is preferred.

### Part 3A

1. **pto.vdupi:** Is this distinct from `pto.vdup` with an immediate operand, or can `pto.vdup` handle both?
2. **Predicate ops (pand/por/pxor and predicate movement forms):** These need MLIR op definitions and verifier rules. Confirm priority.

### Part 3B

1. **Section 10 removals:** 4 interleave ops removed (not on A5). If multi-arch support is needed later, these would need conditional inclusion.

### Part 3C

2. **Store dist family completeness:** `vsts` currently covers `NORM_B8`, `NORM_B16`, `NORM_B32`, `1PT_B8`, `1PT_B16`, `1PT_B32`, `PK_B16`, `PK_B32`, `PK_B64`, `PK4_B32`, `MRG4CHN_B8`, `MRG2CHN_B8`, and `MRG2CHN_B16`, while `vstsx2` covers `INTLV_B8` / `INTLV_B16` / `INTLV_B32`. `MRG4CHN_B8` / `MRG2CHN_B8` / `MRG2CHN_B16` are preserved in the VPTO surface, but the current hardware still reports them as unsupported via verifier warning and they are not expected to validate at runtime on A5 today.
3. **vcvt width-changing pattern:** The even/odd + `vor` pattern for forms such as `f32 -> f16` is the standard compiler lowering. Confirm this is the intended representation in the spec.
4. **Stateful store ops (Section 14):** These are complex with SSA state threading. Are they all needed for A5, or can some be simplified?
