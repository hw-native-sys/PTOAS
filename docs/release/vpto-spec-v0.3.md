# PTO micro Instruction Spec — Draft (A5)

- v0.3: Add runtime block query and vector-interval legality notes; Normalize load/store distribution families; Update get_buf/rls_buf details
- v0.2: Update micro Instruction latency and throughput
- v0.1: Doc Init

[toc]

---

## Part I: Architecture Overview

### Overview

This document defines the PTO micro Instruction, a compiler-internal and externally facing specification designed to represent vector compute kernels within the PTO architecture. Much like NVVM provides a robust IR for GPU architectures, the PTO micro Instruction serves as the direct bridge between high-level programming models and the underlying hardware ISA, providing a precise, low-level representation of vector workloads explicitly designed for the Ascend 950 architecture.

#### Position in the Stack and Layer Modeled

The PTO micro Instruction operates as a very low-level intermediate representation within the PTO compiler stack. It is uniquely designed to accurately and comprehensively express all architectural information of the Ascend 950 hardware. It specifically models the bare-metal vector execution layer, making hardware-specific capabilities and constraints, such as exact vector lane configurations, memory space hierarchies, and hardware-specific fusion semantics, fully transparent and controllable.

#### PTO Instruction Modes and Compilation Flows

Within the end-to-end PTO software stack, PTO instructions may appear in three closely related authoring or lowering modes:

- **PTO Tile Instruction**: tile-oriented PTO code that serves as a nano-kernel encapsulation of Tile operations, primarily expressing computation and data movement in terms of tile buffers, tile shapes, and tile-local layout.
- **PTO micro Instruction**: vector-execution-oriented PTO code that makes DMA setup, vector registers, masks, synchronization, and `__VEC_SCOPE__` boundaries explicit. This document is centered on this mode.
- **PTO Tile+micro Instruction**: a hybrid PTO form that keeps tile-level orchestration while embedding explicit micro-instruction regions where direct vector-pipeline control is required.

From these PTO instruction forms, the stack can proceed along two main compilation flows:

- **CCE generation flow**: PTO ISA is lowered into a CCE-oriented representation, which is then compiled by the BiSheng toolchain into Ascend device binaries.
- **Bytecode generation flow**: PTO ISA is emitted as bytecode, which is then compiled by the BiSheng toolchain into Ascend device binaries.

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
             +------------+------------+
             |                         |
             v                         v
 +-------------------------+   +-------------------------+
 | Path A: generate CCE    |   | Path B: generate        |
 | (CCE-oriented form)     |   | bytecode                |
 +------------+------------+   +------------+------------+
              |                             |
              v                             v
   +-----------------------------------------------+
   |               BiSheng compiler                |
   +---------------------------+-------------------+
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

1. **GM → UB**: DMA transfer via MTE2 (`pto.copy_gm_to_ubuf`)
2. **UB → vreg**: Vector Load instructions (`pto.vlds`, `pto.vldsx2`, etc.)
3. **vreg → vreg**: Compute instructions (`pto.vadd`, `pto.vmul`, etc.)
4. **vreg → UB**: Vector Store instructions (`pto.vsts`, `pto.vstsx2`, etc.)
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
pto.set_loop2_stride_outtoub %c4096_i64, %c4096_i64 : i64, i64
pto.set_loop1_stride_outtoub %c4096_i64, %c4096_i64 : i64, i64
pto.set_loop_size_outtoub %c1_i64, %c1_i64 : i64, i64
pto.copy_gm_to_ubuf %7, %2, %3, %3, %c0_i64, %c32_i64, %4, %c0_i64, %c0_i64,
    %false, %c0_i64, %c128_i64, %c128_i64
    : !pto.ptr<f32, gm>, !pto.ptr<f32, ub>, i64, i64, i64, i64, i64, i64, i64, i1, i64, i64, i64

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
pto.set_loop_size_ubtoout %c1_i64, %c1_i64 : i64, i64
pto.set_loop1_stride_ubtoout %c4096_i64, %c4096_i64 : i64, i64
pto.set_loop2_stride_ubtoout %c4096_i64, %c4096_i64 : i64, i64
pto.copy_ubuf_to_gm %8, %14, %3, %3, %c0_i64, %c32_i64, %4, %c0_i64, %c128_i64, %c128_i64
    : !pto.ptr<f32, ub>, !pto.ptr<f32, gm>, i64, i64, i64, i64, i64, i64, i64, i64
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

Typical usage:

```mlir
%block = pto.get_block_idx
%subblock = pto.get_subblock_idx
%block_num = pto.get_block_num
%subblock_num = pto.get_subblock_num
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
    %s = pto.load_scalar %1[%c4] : !pto.ptr<f32, ub> -> f32
    pto.store_scalar %s, %1[%c8] : !pto.ptr<f32, ub>, f32
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
```

**Store (register to memory):**

```mlir
pto.vsts %value, %destination[%offset] {dist = "DIST"} : !pto.vreg<NxT>, !pto.ptr<T, ub>
```

**Dual Load (one load, two results — deinterleave):**

```mlir
%low, %high = pto.vldsx2 %source[%offset], "DIST" : !pto.ptr<T, ub>, index -> !pto.vreg<NxT>, !pto.vreg<NxT>
```

**Dual Store (two inputs, one interleaved store):**

```mlir
pto.vstsx2 %low, %high, %dest[%offset], "DIST", %mask : !pto.vreg<NxT>, !pto.vreg<NxT>, !pto.ptr<T, ub>, index, !pto.mask<G>
```

**Compare (two vectors + seed mask in, mask out):**

```mlir
%mask = pto.vcmp %src0, %src1, %seed, "CMP_MODE" : !pto.vreg<NxT>, !pto.vreg<NxT>, !pto.mask<G> -> !pto.mask<G>
```

**Conversion (one vector in, different-typed vector out):**

```mlir
%result = pto.vcvt %input {rnd = "R", sat = "SAT", part = "EVEN"} : !pto.vreg<NxT0> -> !pto.vreg<MxT1>
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
| `"RND"` | Rounding mode: `R \| A \| F \| C \| Z \| O` |
| `"SAT"` | Saturation: `SAT \| NOSAT` |
| `"PART"` | Half selector: `EVEN \| ODD` |
| `"PAT_*"` | Predicate pattern literal |
| `T` | Element type (f32, f16, bf16, i32, i16, i8, etc.) |
| `N` | Lane count (`N * bitwidth(T) = 2048`) |

---

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
| `"RND"` | Rounding mode: `R \| A \| F \| C \| Z \| O` |
| `"SAT"` | Saturation: `SAT \| NOSAT` |
| `"PART"` | Half selector: `EVEN \| ODD` |
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
| 3 | [Vector Load/Store](#isa-03-vector-load-store) | UB↔vreg data movement with various access patterns | ~20 | `pto.vlds`, `pto.vldsx2`, `pto.vgather2`, `pto.vsts`, `pto.vstsx2`, `pto.vscatter`, etc. |
| 4 | [Predicate Load/Store](#isa-04-predicate-load-store) | UB↔mask register movement | 5 | `pto.plds`, `pto.pldi`, `pto.psts`, `pto.psti`, `pto.pstu` |
| 5 | [Materialization & Predicate Ops](#isa-05-materialization-predicate) | Scalar broadcast, predicate generation and manipulation | ~17 | `pto.vbr`, `pto.vdup`, `pto.pset_b*`, `pto.pge_b*`, `pto.plt_b*`, `pto.ppack`, `pto.punpack`, `pto.pnot`, `pto.psel`, etc. |
| 6 | [Unary Vector Ops](#isa-06-unary-vector-ops) | Single-input element-wise operations | 6 | `pto.vabs`, `pto.vexp`, `pto.vln`, `pto.vsqrt`, `pto.vrelu`, `pto.vnot` |
| 7 | [Binary Vector Ops](#isa-07-binary-vector-ops) | Two-input element-wise operations | 13 | `pto.vadd`, `pto.vsub`, `pto.vmul`, `pto.vdiv`, `pto.vmax`, `pto.vmin`, `pto.vand`, `pto.vor`, `pto.vxor`, `pto.vshl`, `pto.vshr`, `pto.vaddc`, `pto.vsubc` |
| 8 | [Vec-Scalar Ops](#isa-08-vec-scalar-ops) | Vector-scalar operations | 9 | `pto.vadds`, `pto.vmuls`, `pto.vmaxs`, `pto.vmins`, `pto.vlrelu`, `pto.vshls`, `pto.vshrs`, `pto.vaddcs`, `pto.vsubcs` |
| 9 | [Conversion Ops](#isa-09-conversion-ops) | Type conversion with rounding/saturation control | 2 | `pto.vcvt`, `pto.vtrc` |
| 10 | [Reduction Ops](#isa-10-reduction-ops) | Vector reductions | 7 | `pto.vcadd`, `pto.vcmax`, `pto.vcmin`, `pto.vcgadd`, `pto.vcgmax`, `pto.vcgmin`, `pto.vcpadd` |
| 11 | [Compare & Select](#isa-11-compare-select) | Comparison and conditional selection | 4 (+1 not A5) | `pto.vcmp`, `pto.vcmps`, `pto.vsel`, `pto.vselr` (`pto.vselrv2` removed: not A5) |
| 12 | [Data Rearrangement](#isa-12-data-rearrangement) | In-register data movement and permutation | 2 (+2 not A5) | `pto.vintlv`, `pto.vdintlv` (`pto.vintlvv2`, `pto.vdintlvv2` removed: not A5) |
| 13 | [DSA/SFU Ops](#isa-13-dsa-sfu-ops) | Specialized ops, index generation, and sorting helpers | 9 | `pto.vlrelu`, `pto.vprelu`, `pto.vexpdiff`, `pto.vaxpy`, `pto.vmull`, `pto.vmula`, `pto.vci`, `pto.vbitsort`, `pto.vmrgsort4` |
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

##### Mode Parameter for `get_buf` / `rls_buf`

The `mode` parameter controls how `get_buf` and `rls_buf` interact with pipeline execution and dependency tracking:

| Mode | `get_buf` Behavior | `rls_buf` Behavior | Use Case |
|------|-------------------|-------------------|----------|
| **0** (default) | **Blocking acquire**: waits for all previous `rls_buf` with same `buf_id` from all pipelines (in program order) before the specified pipe can proceed | **Immediate release**: signals completion for only the instructions related to the specified pipe | **Automatic ping/pong dependency** — recommended for double/multi-buffering |
| **1** | **Non-blocking acquire**: does not wait; pipe execution proceeds immediately | **Deferred release**: waits for all instructions across all pipelines with same `buf_id` to retire before signaling | **Backward compatibility** with `set_flag`/`wait_flag` semantics |

**Mode 0 (Default — Recommended):**
- `get_buf`: The specified pipeline blocks until all previous `rls_buf` operations for the same buffer ID (from any pipeline) have completed, respecting program order.
- `rls_buf`: Immediately signals that the specified pipeline has finished using the buffer — only waits for that pipe's related instructions.
- This mode provides **automatic RAW/WAR/WAW dependency resolution** based on buffer ID and program order, making it ideal for ping/pong and N-buffer patterns.

**Mode 1 (Legacy Compatibility):**
- `get_buf`: Does not block — the pipeline proceeds immediately without waiting.
- `rls_buf`: Waits for **all** previous instructions across **all** pipelines with the same buffer ID to retire before signaling release.
- This mode emulates `set_flag`/`wait_flag` behavior and is provided for backward compatibility with existing code patterns.

> **Note:** A5 supports both `set_flag`/`wait_flag` and `get_buf`/`rls_buf` mechanisms. Mode 1 is rarely needed since mode 0 provides a more programmer-friendly approach for buffer-based synchronization.

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

#### Why `get_buf` / `rls_buf` is More Programmer-Friendly

The buffer-based synchronization (`get_buf`/`rls_buf`) provides the **same functional capability** as `set_flag`/`wait_flag` for maintaining correct ordering of RAW/WAR/WAW dependencies across pipelines, but with significant usability advantages:

##### 1. No Manual Priming or Draining

With `set_flag`/`wait_flag`, ping/pong loops require:
- **Pre-loop priming**: 4× `set_flag` to initialize reverse-dependency signals (otherwise first iteration deadlocks)
- **Post-loop draining**: 4× `wait_flag` to consume leftover signals from final iterations

With `get_buf`/`rls_buf`:
- **First iteration**: Buffer is initially free, so `get_buf` proceeds immediately — no priming needed
- **Final iteration**: Last `rls_buf` simply completes — no draining required

##### 2. No Loop Peeling for Complex Dependencies

For non-1:1 producer-consumer ratios (e.g., 1 MTE2 load : N Vector compute slices), `set_flag`/`wait_flag` requires **peeling the set_flag outside the loop**:

```mlir
// set_flag/wait_flag: 1 MTE2 load, 8 Vector computes on slices
// MTE2 loads large tile once
pto.copy_gm_to_ubuf %gm_ptr, %ub_tile, ...
pto.set_flag["PIPE_MTE2", "PIPE_V", "EVT_TILE_READY"]  // ◀ MUST be outside loop

// Vector consumes in 8 slices — but wait_flag can only fire ONCE
pto.wait_flag["PIPE_MTE2", "PIPE_V", "EVT_TILE_READY"] // ◀ MUST peel before loop
scf.for %slice = %c0 to %c8 step %c1 {
  // compute on %ub_tile[%slice]
  // Cannot put wait_flag here — would deadlock on iteration 1+
}
```

With `get_buf`/`rls_buf`, acquire/release can be **inside the loop** — no peeling needed:

```mlir
// get_buf/rls_buf: same 1:8 pattern, acquire/release inside loop works fine
// MTE2 loads large tile
pto.get_buf "PIPE_MTE2", %bufid_tile, %c0 : i64, i64
pto.copy_gm_to_ubuf %gm_ptr, %ub_tile, ...
pto.rls_buf "PIPE_MTE2", %bufid_tile, %c0 : i64, i64

// Vector acquires/releases per slice — all 8 iterations work correctly
scf.for %slice = %c0 to %c8 step %c1 {
  pto.get_buf "PIPE_V", %bufid_tile, %c0 : i64, i64  // iteration 0: blocks until MTE2 done
                                                      // iteration 1-7: proceeds immediately (already acquired)
  // compute on %ub_tile[%slice]
  pto.rls_buf "PIPE_V", %bufid_tile, %c0 : i64, i64
}
// No peeling required — get_buf handles the MTE2→V dependency automatically
```

##### 3. Simpler Mental Model

| Aspect | `set_flag`/`wait_flag` | `get_buf`/`rls_buf` |
|--------|------------------------|---------------------|
| **Dependency tracking** | Manual: track event IDs, signal directions, pair every set with wait | Automatic: buffer ID + program order |
| **Event ID management** | **8 IDs per pipe-pair direction** (HW limit); each buffer occupies 1 ID per direction | **1 buffer ID per shared resource** (HW limit: 32 global); same ID used across all pipelines |
| **Error-prone areas** | Forgetting prime/drain, mismatched IDs, wrong direction | Forgetting release (but compile-time checkable) |

##### Quick Example Comparison

**Problem:** MTE2 loads into `buf[i%2]`, Vector processes, MTE3 stores — standard ping/pong.

**set_flag/wait_flag approach:**
```mlir
// BEFORE loop: prime 4 reverse-dep signals
pto.set_flag["PIPE_V", "PIPE_MTE2", "EVT_IN_REV_0"]
pto.set_flag["PIPE_V", "PIPE_MTE2", "EVT_IN_REV_1"]
pto.set_flag["PIPE_MTE3", "PIPE_V", "EVT_OUT_REV_0"]
pto.set_flag["PIPE_MTE3", "PIPE_V", "EVT_OUT_REV_1"]

scf.for %i = ... {
  // 4 set_flag + 4 wait_flag inside loop
  // Must track 4 IDs: 2 pipe-pair directions × 2 ping/pong buffers
}

// AFTER loop: drain 4 signals
pto.wait_flag["PIPE_V", "PIPE_MTE2", "EVT_IN_REV_0"]
pto.wait_flag["PIPE_V", "PIPE_MTE2", "EVT_IN_REV_1"]
pto.wait_flag["PIPE_MTE3", "PIPE_V", "EVT_OUT_REV_0"]
pto.wait_flag["PIPE_MTE3", "PIPE_V", "EVT_OUT_REV_1"]
```

**get_buf/rls_buf approach:**
```mlir
scf.for %i = ... {
  pto.get_buf %bufid_in[%pp], "PIPE_MTE2"
  // ... MTE2 work ...
  pto.rls_buf %bufid_in[%pp], "PIPE_MTE2"

  pto.get_buf %bufid_in[%pp], "PIPE_V"
  pto.get_buf %bufid_out[%pp], "PIPE_V"
  // ... Vector work ...
  pto.rls_buf %bufid_in[%pp], "PIPE_V"
  pto.rls_buf %bufid_out[%pp], "PIPE_V"

  pto.get_buf %bufid_out[%pp], "PIPE_MTE3"
  // ... MTE3 work ...
  pto.rls_buf %bufid_out[%pp], "PIPE_MTE3"
}
// Done. No prime. No drain. Dependencies resolved by buffer ID + program order.
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

pto.vecscope {
  %v   = pto.vlds %ub_ptr[%lane] : !pto.ptr<f32, ub> -> !pto.vreg<64xf32>
  %mask = pto.pset_b32 "PAT_ALL" : !pto.mask<b32>
  %abs = pto.vabs %v, %mask : !pto.vreg<64xf32>, !pto.mask<b32> -> !pto.vreg<64xf32>
  pto.vsts %abs, %ub_out[%lane], %mask : !pto.vreg<64xf32>, !pto.ptr<f32, ub>, !pto.mask<b32>
}

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
pto.get_buf "PIPE_MTE2", %bufid_ub_ptr, %c0 : i64, i64   // mode=0 (default)
pto.copy_gm_to_ubuf %gm_ptr, %ub_ptr, ...
// MTE2 done writing ub_ptr — release it so Vector can consume
pto.rls_buf "PIPE_MTE2", %bufid_ub_ptr, %c0 : i64, i64

// ─── Stage 2: Vector computation ───
// Vector acquires ub_ptr (input) — blocks until MTE2 releases it (RAW: MTE2 write → V read)
pto.get_buf "PIPE_V", %bufid_ub_ptr, %c0 : i64, i64
// Vector acquires ub_out (output) — blocks until MTE3 releases it from a prior iteration (WAR: MTE3 read → V write)
pto.get_buf "PIPE_V", %bufid_ub_out, %c0 : i64, i64

pto.vecscope {
  %mask = pto.pset_b32 "PAT_ALL" : !pto.mask<b32>
  %v   = pto.vlds %ub_ptr[%lane] : !pto.ptr<f32, ub> -> !pto.vreg<64xf32>
  %abs = pto.vabs %v, %mask : !pto.vreg<64xf32>, !pto.mask<b32> -> !pto.vreg<64xf32>
  pto.vsts %abs, %ub_out[%lane], %mask : !pto.vreg<64xf32>, !pto.ptr<f32, ub>, !pto.mask<b32>
}

// Vector done reading ub_ptr — release so MTE2 can reuse it in next iteration
pto.rls_buf "PIPE_V", %bufid_ub_ptr, %c0 : i64, i64
// Vector done writing ub_out — release so MTE3 can consume
pto.rls_buf "PIPE_V", %bufid_ub_out, %c0 : i64, i64

// ─── Stage 3: MTE3 stores result to GM ───
// MTE3 acquires ub_out — blocks until Vector releases it (RAW: V write → MTE3 read)
pto.get_buf "PIPE_MTE3", %bufid_ub_out, %c0 : i64, i64
pto.copy_ubuf_to_gm %ub_out, %gm_out, ...
// MTE3 done reading ub_out — release so Vector can reuse it in next iteration
pto.rls_buf "PIPE_MTE3", %bufid_ub_out, %c0 : i64, i64
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
    %mask = pto.pset_b32 "PAT_ALL" : !pto.mask<b32>
    %abs = pto.vabs %v, %mask : !pto.vreg<64xf32>, !pto.mask<b32> -> !pto.vreg<64xf32>
    pto.vsts %abs, %ub_out[%pp][%lane], %mask : !pto.vreg<64xf32>, !pto.ptr<f32, ub>, !pto.mask<b32>
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
  pto.get_buf "PIPE_MTE2", %bufid_buf[%pp], %c0 : i64, i64   // mode=0
  pto.copy_gm_to_ubuf %gm_ptr[%i], %ub_buf[%pp], ...
  pto.rls_buf "PIPE_MTE2", %bufid_buf[%pp], %c0 : i64, i64

  // ── Vector: compute on buf[i%2] ──
  // Acquires buf[i%2] — blocks until MTE2 releases it (RAW: automatic)
  pto.get_buf "PIPE_V", %bufid_buf[%pp], %c0 : i64, i64
  pto.get_buf "PIPE_V", %bufid_out[%pp], %c0 : i64, i64
  scf.for %dummy = %c0 to %c1 step %c1 {
    %v   = pto.vlds %ub_buf[%pp][%lane] : !pto.ptr<f32, ub> -> !pto.vreg<64xf32>
    %mask = pto.pset_b32 "PAT_ALL" : !pto.mask<b32>
    %abs = pto.vabs %v, %mask : !pto.vreg<64xf32>, !pto.mask<b32> -> !pto.vreg<64xf32>
    pto.vsts %abs, %ub_out[%pp][%lane], %mask : !pto.vreg<64xf32>, !pto.ptr<f32, ub>, !pto.mask<b32>
  } {llvm.loop.aivector_scope}
  // Release buf[i%2] — MTE2 can reuse in iteration i+2 (WAR resolved)
  pto.rls_buf "PIPE_V", %bufid_buf[%pp], %c0 : i64, i64
  pto.rls_buf "PIPE_V", %bufid_out[%pp], %c0 : i64, i64

  // ── MTE3: store result ──
  // Acquires out[i%2] — blocks until Vector releases it (RAW: automatic)
  pto.get_buf "PIPE_MTE3", %bufid_out[%pp], %c0 : i64, i64
  pto.copy_ubuf_to_gm %ub_out[%pp], %gm_out[%i], ...
  pto.rls_buf "PIPE_MTE3", %bufid_out[%pp], %c0 : i64, i64
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
| IDs per pipe-pair | 2 IDs per buffer: 1 for forward (e.g., MTE2→V) + 1 for reverse (V→MTE2) | **1 ID per buffer** (handles both directions automatically) |
| Total HW IDs | **8 per pipe-pair** (hardware limit) | **32 global** across all pipes |
| Reverse (WAR) deps | Extra `set_flag`/`wait_flag` pair per buffer | Handled automatically |
| Pre-loop setup | `set_flag` to prime each reverse dep | **None** |
| Post-loop teardown | `wait_flag` to drain all primed signals | **None** |
| Loop peeling for complex deps | Required for non-1:1 or nested loops | **Not required** |
| Straight-line code | Simple, clear | Slightly more verbose (bracket each stage) |
| Ping/pong loops | 8 event IDs + 4 prime + 4 drain | Same pattern, **no overhead** |
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

#### Latency and throughput (A5)

**Cycle-accurate simulator (CA model)** issue→retire timings for vector-side instructions behind this chapter. Values are **simulator** results, **not** guaranteed for silicon.

**SOC:** Tables below are from **Ascend910_9599** CA sim (the pto-isa ST default when **Ascend950PR_9599** is not selected).

**Log `dist:` tokens:** PTO load/store modes lower to **`RV_VLD` / `RV_VLDI` / `RV_VST` / `RV_VSTI`** with a **`dist:`** field on the vector pipes (`RVECLD` / `RVECST`). Some simulator logs typo contiguous load as `dist:NORAML`; treat as **`NORMAL`**.

##### Reference op latencies (A5 mnemonics)

| A5 mnemonic | Mode / note | Typical issue→retire (cycles) |
|-------------|-------------|------------------------------|
| `RV_VLD` | `dist:NORMAL` / `NORAML` | **9** |
| `RV_VLDI` | `dist:DINTLV` (dual vreg) | **9** |
| `RV_VST` / `RV_VSTI` | `dist:NORM` | **9** |
| `RV_VGATHER2` | `Dtype: B32` | **27–28** |
| `RV_VGATHERB` | indexed byte gather | **~21** |
| `RV_VSCATTER` | `Dtype: B16` | **~17** |
| `RV_VADD` | F32 between UB-backed ops | **7** |

##### `dist:` tokens (issue→retire)

Most **`dist:`** tokens are **9** issue→retire cycles. **`INTLV`** on **`RV_VSTI`** is **12** cycles.

| `dist:` (as in log) | RV op | issue→retire (cycles) |
|---------------------|-------|----------------------|
| `DINTLV` | `RV_VLDI` | **9** |
| `BRC` | `RV_VLD` | **9** |
| `BRC_BLK` | `RV_VLD` | **9** |
| `INTLV` | `RV_VSTI` | **12** |
| `UNPK` | `RV_VLD` | **9** |
| `NORM` | `RV_VSTI` | **9** |
| `PK` | `RV_VSTI` | **9** |
| `NORMAL` / `NORAML` | `RV_VLD` | **9** |

**Note:** PTO intrinsic **`BRC_BLK`** matches the **`BRC_BLK`** `dist:` string on **`RV_VLD`** in simulator logs (block-replicate path; not a plain contiguous copy in the usual tiling use).

**Issue (vector load/store):** `pto.vlds` (**`RV_VLD`**) is **dual-issue capable**: two independent `pto.vlds` can issue **in the same cycle**. **Alternatively**, the hardware can issue **one** `pto.vlds` **and** **one** `pto.vsts` together (**1+1**) in the same cycle. Each cycle is **either** dual **`vlds`** **or** **`vlds` + `vsts` (1+1)**—those two issue modes are mutually exclusive. Sustained throughput still depends on RAW hazards and loop structure.

**Throughput (simulator, pattern-dependent):**

- **`RV_VLD` / `pto.vlds`:** Dual-issue **or** half of a **1+1** with `vsts`, per the rule above.
- **`RV_VST` / `pto.vsts`:** In a **1+1** cycle, pairs with one `vlds`; otherwise typically **one** store per cycle in tight loops.
- **`RV_VGATHER2`:** Much lower than contiguous `RV_VLD` (on the order of **~0.1** ops/cycle in steady-state alongside 27–28-cycle latency).

##### PTO `dist` summary (loads)

| PTO `dist` (load) | Latency |
|-------------------|-------------------|
| `NORM` | **9** cycles |
| `UNPK` | **9** cycles |
| `DINTLV` | **9** cycles (`RV_VLDI`) |
| `BRC` | **9** cycles (`RV_VLD`) |
| `BRC_BLK` | **9** cycles as **`dist:BRC_BLK`** on `RV_VLD` |
| `BDINTLV` | **9** cycles |
| `US`, `DS`, `SPLT4CHN`, `SPLT2CHN` | **9** cycles |

##### PTO `dist` summary (stores)

| PTO `dist` (store) | Latency |
|--------------------|-------------------|
| `NORM` | **9** cycles (`RV_VSTI`) |
| `PK` | **9** cycles |
| `INTLV` (`pto.vstx2`) | **12** cycles |
| `MRG4CHN`, `MRG2CHN` | **9** cycles |

##### Gather, scatter, and special addressing

| PTO op | A5-level | Latency |
|--------|----------|-------------------|
| `pto.vgather2` | `RV_VGATHER2` | **27–28** cycles (pattern-dependent) |
| `pto.vgatherb` | `RV_VGATHERB` | **~21** cycles issue→retire |
| `pto.vgather2_bc` | (broadcast gather) | **27–28** cycles (same as **`pto.vgather2`**) |
| `pto.vscatter` | `RV_VSCATTER` | **~17** cycles for **`Dtype: B16`** |

##### Strided loads/stores, unaligned ops, alignment state

Ops such as **`pto.vldas`**, **`pto.vldus`**, **`pto.vsld`**, **`pto.vsldb`**, **`pto.vsst`**, **`pto.vsstb`**, **`pto.vsta`**, **`pto.vstas`**, **`pto.vstar`**, **`pto.vstu`**, **`pto.vstus`**, **`pto.vstur`**: **9** cycles (same vector load/store pipe family as contiguous `RV_VLD` / `RV_VST` unless listed otherwise above).

##### Dual-issue vs DMA

DMA **`TLOAD` / `TSTORE`** (global memory ↔ UB) use **MTE** pipes, not `RV_VLD`/`RV_VST`. **MTE2** `MOV_*` latency is not the same as vector `RV_VLD` latency (see `02-dma-copy.md` for GM↔UB movement).

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
  fact that the source is UB memory. PTO surface exposes load `dist` as family
  tokens, and each family only supports the element widths listed below.

**Distribution families:**

| Family | Allowed element widths | C semantics | Latency |
|------|-------------|-------------|-------------|
| `NORM` | width-agnostic | `dst[i] = UB[base + i * sizeof(T)]` | **9** cycles |
| `BRC` | `b8`, `b16`, `b32` | `dst[i] = UB[base]` for all `i` | **9** cycles |
| `US` | `b8`, `b16` | `dst[2*i] = dst[2*i+1] = UB[base + i]` | **9** cycles |
| `DS` | `b8`, `b16` | `dst[i] = UB[base + 2*i]` | **9** cycles |
| `UNPK` | `b8`, `b16`, `b32` | Expand packed source data into wider lanes | **9** cycles |
| `BRC_BLK` | width-agnostic | Block-replicate load path; simulator logs may print `dist:BRC_BLK` | **9** cycles |
| `E2B` | `b16`, `b32` | Load element groups and expand them into byte-oriented lane layout | **9** cycles |
| `UNPK4` | `b8` | Unpack 4-way packed `b8` source groups into destination lanes | **9** cycles |
| `SPLT4CHN` | `b8` | Split 4-channel interleaved source into one channel plane | **9** cycles |
| `SPLT2CHN` | `b8`, `b16` | Split 2-channel interleaved source into one channel plane | **9** cycles |

`pto.vlds` currently covers only single-result load families. Dual-result
deinterleave forms are modeled separately in PTO surface as
[`pto.vldsx2`](#ptovldsx2): `BDINTLV` is the block-deinterleave family, while
`DINTLV` is the element-width-sensitive deinterleave family.

**Example — Contiguous load:**
```mlir
%v = pto.vlds %ub[%offset] {dist = "NORM"} : !pto.ptr<f32, ub> -> !pto.vreg<64xf32>
```

**Example — Broadcast scalar to all lanes:**
```mlir
%v = pto.vlds %ub[%c0] {dist = "BRC"} : !pto.ptr<f32, ub> -> !pto.vreg<64xf32>
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
- **Latency:** **9** cycles.

---

##### `pto.vldus`

- **syntax:** `%result, %align_out = pto.vldus %source, %align : !pto.ptr<T, ub>, !pto.align -> !pto.vreg<NxT>, !pto.align`
- **semantics:** Unaligned load using primed align state.
- **inputs:**
  `%source` is the current UB address and `%align` is the incoming load
  alignment state primed by `pto.vldas` or a prior `pto.vldus`.
- **outputs:**
  `%result` is the assembled vector value and `%align_out` is the updated
  alignment state.
- **constraints and limitations:**
  A matching `pto.vldas` MUST appear before the first dependent `pto.vldus`
  stream in the same vector loop. The installed no-post A5 interface keeps a
  struct-shaped internal return for lowering convenience, but its no-post
  `base` field is not meaningful user-visible state. VPTO therefore hides that
  value and only exposes the updated align carrier. Reusing the original
  `%source` starts a new explicit access point; if the caller wants another
  no-post access, it should compute the next source pointer explicitly and pair
  it with the required align setup.
- **Latency:** **9** cycles.

**Unaligned load pattern:**
```mlir
%align = pto.vldas %ub : !pto.ptr<f32, ub> -> !pto.align
%vec, %align2 = pto.vldus %ub, %align : !pto.ptr<f32, ub>, !pto.align -> !pto.vreg<64xf32>, !pto.align
```

---

##### `pto.init_align`

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

#### Dual Loads (Deinterleave)

##### `pto.vldsx2`

- **syntax:** `%low, %high = pto.vldsx2 %source[%offset], "DIST" : !pto.ptr<T, ub>, index -> !pto.vreg<NxT>, !pto.vreg<NxT>`
- **semantics:** Dual load with deinterleave (AoS → SoA conversion).
- **inputs:**
  `%source` is the UB base pointer, `%offset` is the displacement, and `DIST`
  selects a dual-load/deinterleave layout.
- **outputs:**
  `%low` and `%high` are the two destination vectors.
- **constraints and limitations:**
  This family is only legal for interleave/deinterleave style distributions.
  The two outputs form an ordered pair, and that pairing MUST be preserved.
  PTO surface accepts deinterleave families. `BDINTLV` is element-width
  agnostic, while `DINTLV` supports only the element widths listed in the
  table.
- **latency:** `BDINTLV` / `DINTLV` are both **9** cycles.

**Distribution families:**

| Family | Allowed element widths | C semantics | Latency |
|------|-------------|-------------|-------------|
| `BDINTLV` | width-agnostic | Block deinterleave into two destination vectors | **9** cycles |
| `DINTLV` | `b8`, `b16`, `b32` | Deinterleave alternating elements into `%low` / `%high` | **9** cycles |

```c
// DINTLV family on 32-bit elements: deinterleave 32-bit elements
for (int i = 0; i < 64; i++) {
    low[i]  = UB[base + 8*i];       // even elements
    high[i] = UB[base + 8*i + 4];   // odd elements
}
```

**Example — Load interleaved XY pairs into separate X/Y vectors:**
```mlir
%x, %y = pto.vldsx2 %ub[%offset], "DINTLV" : !pto.ptr<f32, ub>, index -> !pto.vreg<64xf32>, !pto.vreg<64xf32>
```

##### `pto.vsldb`

- **syntax:** `%result = pto.vsldb %source, %block_stride, %repeat_stride, %mask : !pto.ptr<T, ub>, i16, i16, !pto.mask<G> -> !pto.vreg<NxT>`
- **semantics:** Block-strided load for 2D tile access.
- **inputs:**
  `%source` is the UB base pointer. `%block_stride` and `%repeat_stride` are
  the two 16-bit fields of the hardware control word, and `%mask` controls
  which blocks participate.
- **outputs:**
  `%result` is the loaded vector.
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
- **Latency:** **27–28** cycles per `RV_VGATHER2`; throughput much lower than contiguous `RV_VLD` (see **Latency and throughput (A5)** at the start of this chapter).

```c
for (int i = 0; i < active_lanes; i++)
    dst[i] = UB[base + offsets[i] * sizeof(T)];
```

---

##### `pto.vgatherb`

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

##### `pto.vgather2_bc`

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

#### Contiguous Stores

##### `pto.vsts`

- **syntax:** `pto.vsts %value, %dest[%offset], %mask {dist = "DIST"} : !pto.vreg<NxT>, !pto.ptr<T, ub>, !pto.mask<G>`
- **semantics:** Vector store with distribution mode.
- **inputs:**
  `%value` is the source vector, `%dest` is the UB base pointer, `%offset` is
  the displacement, `%mask` selects the active lanes or sub-elements, and
  `DIST` selects the store distribution.
- **outputs:**
  This op has no SSA result; it writes to UB memory.
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
| `NORM` | `b8`, `b16`, `b32` | `UB[base + i] = src[i]` | **9** cycles |
| `1PT` | `b8`, `b16`, `b32` | Only element 0 is written to the destination footprint | **9** cycles |
| `PK` | `b16`, `b32`, `b64` | Pack low half bits of each source element before store | **9** cycles |
| `PK4` | `b32` | Pack low 8 bits of each `b32` element before store | **9** cycles |
| `MRG4CHN` | `b8` | Merge 4 channel planes into an interleaved 4-channel layout | **9** cycles |
| `MRG2CHN` | `b8`, `b16` | Merge 2 channel planes into an interleaved 2-channel layout | **9** cycles |

**Example — Contiguous store:**
```mlir
pto.vsts %v, %ub[%offset], %mask {dist = "NORM"} : !pto.vreg<64xf32>, !pto.ptr<f32, ub>, !pto.mask<G>
```

---

#### Dual Stores (Interleave)

##### `pto.vstsx2`

- **syntax:** `pto.vstsx2 %low, %high, %dest[%offset], "DIST", %mask : !pto.vreg<NxT>, !pto.vreg<NxT>, !pto.ptr<T, ub>, index, !pto.mask<G>`
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
  be preserved. PTO surface accepts the `INTLV` family, which only supports the
  element widths listed below.
  be preserved. PTO surface accepts the `INTLV` family, which only supports the
  element widths listed below.
- **latency:** `INTLV` is **12** cycles。

**Distribution families:**

| Family | Allowed element widths | C semantics | Latency |
|------|-------------|-------------|-------------|
| `INTLV` | `b8`, `b16`, `b32` | Interleave `%low` / `%high` into one destination stream | **12** cycles |
| `INTLV` | `b8`, `b16`, `b32` |

```c
// INTLV family on 32-bit elements:
for (int i = 0; i < 64; i++) {
    UB[base + 8*i]     = low[i];
    UB[base + 8*i + 4] = high[i];
}
```

##### `pto.vsstb`

- **syntax:** `pto.vsstb %value, %dest, %block_stride, %repeat_stride, %mask : !pto.vreg<NxT>, !pto.ptr<T, ub>, i16, i16, !pto.mask<G>`
- **semantics:** Block-strided store for 2D tile access.
- **inputs:**
  `%value` is the source vector, `%dest` is the UB base pointer,
  `%block_stride` and `%repeat_stride` are the two 16-bit fields of the
  hardware control word, and `%mask` controls block participation.
- **outputs:**
  This op writes UB memory and returns no SSA value.
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
- **Latency:** **~17** cycles for **`Dtype: B16`**.

```c
for (int i = 0; i < active_lanes; i++)
    UB[base + offsets[i] * sizeof(T)] = src[i];
```

---

#### Alignment State Stores

##### `pto.vstas`
- **syntax:** `pto.vstas %value, %dest, %offset : !pto.align, !pto.ptr<T, ub>, i32`
- **semantics:** Scalar-register-offset form of alignment-state flush.
- **inputs:**
  `%value` is the pending store-alignment state, `%dest` is the UB base
  pointer, and `%offset` is the scalar-register style displacement.
- **outputs:**
  This op writes buffered tail bytes to UB and returns no SSA value.
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
  store stream that produced `%value`. The first store-side state in a stream
  should be created by `pto.init_align`.
- **Latency:** **9** cycles.

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
- **Latency:** **9** cycles.

---

#### Stateful Store Ops

These ops make reference-updated state explicit as SSA results.

##### `pto.vstus`

- **syntax:** `%align_out = pto.vstus %align_in, %offset, %value, %base : !pto.align, i32, !pto.vreg<NxT>, !pto.ptr<T, ub> -> !pto.align`
- **semantics:** No-post unaligned store with scalar offset.
- **inputs:**
  `%align_in` is the incoming store-alignment state, `%offset` is the scalar
  displacement, `%value` is the vector being stored, and `%base` is the UB base
  pointer.
- **outputs:**
  `%align_out` is the updated buffered-tail state.
- **constraints and limitations:**
  This is the scalar-offset stateful form of the unaligned store family. The
  scalar offset width MUST match the selected form, and a later flush op is
  still required. The first `%align_in` in the stream should come from
  `pto.init_align`. This op does not mean "store a full vector starting at
  `%base + %offset`". Instead, `%offset` describes how far the store stream
  advances at this step, and `%align_out` carries any residual tail that could
  not be committed yet. The no-post surface does not expose an updated base
  pointer. A later flush op must therefore use an explicit destination/offset
  pair that identifies the same logical flush point as this `pto.vstus`.
- **Latency:** **9** cycles.

---

##### `pto.vstur`

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

<a id="isa-04-predicate-load-store"></a>

### 4. Predicate Load/Store

> **Category:** UB ↔ Predicate Register data movement
> **Pipeline:** PIPE_V (Vector Core)

Predicate registers (`!pto.mask<G>`) are 256-bit registers that enable per-lane conditional execution. These ops move predicate values between UB and predicate registers.

In concrete examples, `G` should be chosen to match the consumer family. The
examples below use `b32` when the loaded/stored mask is used with `f32`
vector compares or selects.

The predicate load/store ops documented on this page always use explicit
`base[offset]` addressing. The immediate forms (`pldi`, `psti`) and dynamic
forms (`plds`, `psts`) differ only in how `%offset` is supplied.

---

#### Predicate Loads

##### `pto.plds`

- **syntax:** `%result = pto.plds %source[%offset], "DIST" : !pto.ptr<T, ub>, index -> !pto.mask<G>`
- **semantics:** Load predicate register with runtime offset. This is the
  dynamic-offset form of `pto.pldi`: the predicate payload interpretation is
  the same, but `%offset` is supplied as an SSA `index` instead of a constant
  `index` immediate.
- **DIST:** mandatory string token, one of `NORM`, `US`, `DS`.
  - `NORM`: load a normal packed predicate payload of size `VL/8`.
  - `US`: load a packed predicate payload of size `VL/16`, then duplicate each
    loaded bit once.
  - `DS`: load a packed predicate payload of size `2 * VL/8`, then keep one
    bit out of every two bits.

The loaded payload is a packed predicate image in UB. Consumer ops interpret
the resulting `!pto.mask<G>` according to the mask granularity `G`.
`pto.plds` only
models the explicit `base[offset]` form.

**Example:**
```mlir
%mask = pto.plds %ub[%c0], "NORM" : !pto.ptr<T, ub>, index -> !pto.mask<G>
```

---

##### `pto.pldi`

- **syntax:** `%result = pto.pldi %source[%offset], "DIST" : !pto.ptr<T, ub>, index -> !pto.mask<G>`
- **offset:** must be a constant `index` immediate in PTO surface form.
- **semantics:** Load predicate register with immediate offset.
- **DIST:** mandatory string token, one of `NORM`, `US`, `DS`.
  - `NORM`: load a normal packed predicate payload of size `VL/8`.
  - `US`: load a packed predicate payload of size `VL/16`, then duplicate each
    loaded bit once.
  - `DS`: load a packed predicate payload of size `2 * VL/8`, then keep one
    bit out of every two bits.

Like `pto.plds`, this op reads a packed predicate payload from UB and
materializes it as `!pto.mask<G>`.

---

#### Predicate Stores

##### `pto.psts`

- **syntax:** `pto.psts %value, %dest[%offset], "DIST" : !pto.mask<G>, !pto.ptr<T, ub>, index`
- **semantics:** Store predicate register with runtime offset. This is the
  dynamic-offset form of `pto.psti`: the predicate payload interpretation is
  the same, but `%offset` is supplied as an SSA `index` instead of a constant
  `index` immediate.
- **DIST:** mandatory string token, one of `NORM`, `PK`.
  - `NORM`: store the packed predicate payload into a normal destination space
    of size `VL/8`.
  - `PK`: store the packed predicate payload into a destination space of size
    `VL/16`, keeping one bit out of every two bits.

`pto.psts` stores the packed predicate payload represented by `!pto.mask<G>`.
It only models the explicit `base[offset]` form.

**Example:**
```mlir
pto.psts %mask, %ub[%c0], "NORM" : !pto.mask<G>, !pto.ptr<T, ub>, index
```

---

##### `pto.psti`

- **syntax:** `pto.psti %value, %dest[%offset], "DIST" : !pto.mask<G>, !pto.ptr<T, ub>, index`
- **offset:** must be a constant `index` immediate in PTO surface form.
- **semantics:** Store predicate register with immediate offset.
- **DIST:** mandatory string token, one of `NORM`, `PK`.
  - `NORM`: store the packed predicate payload into a normal destination space
    of size `VL/8`.
  - `PK`: store the packed predicate payload into a destination space of size
    `VL/16`, keeping one bit out of every two bits.

`pto.psti` and `pto.psts` store the packed predicate payload represented by
`!pto.mask<G>`. The surface distinction is only immediate-offset versus
dynamic-offset.

---

##### `pto.pstu`

- **syntax:** `%align_out, %base_out = pto.pstu %align_in, %value, %base : !pto.align, !pto.mask<b16>, !pto.ptr<ui16, ub> -> !pto.align, !pto.ptr<ui16, ub>`
- **syntax:** `%align_out, %base_out = pto.pstu %align_in, %value, %base : !pto.align, !pto.mask<b32>, !pto.ptr<ui32, ub> -> !pto.align, !pto.ptr<ui32, ub>`
- **semantics:** Predicate unaligned store with align/base state update. The base type is fixed by mask granularity: `b16 <-> ui16`, `b32 <-> ui32`.
- **outputs:**
  `%align_out` and `%base_out` are the updated unaligned-store state and are
  intended to be used by a later `pto.pstu` call.
- **constraints and limitations:**
  The first `%align_in` in a predicate unaligned-store stream should come from
  `pto.init_align`.

---

#### Typical Usage Pattern

```mlir
// Generate comparison mask
%mask = pto.vcmp %v0, %v1, %seed, "lt" : !pto.vreg<64xf32>, !pto.vreg<64xf32>, !pto.mask<b32> -> !pto.mask<b32>

// Store mask to UB for later use
pto.psts %mask, %ub_mask[%c0], "NORM" : !pto.mask<b32>, !pto.ptr<T, ub>, index

// ... later in another kernel ...

// Load mask from UB
%saved_mask = pto.plds %ub_mask[%c0], "NORM" : !pto.ptr<T, ub>, index -> !pto.mask<b32>

// Use for predicated select
%result = pto.vsel %v_true, %v_false, %saved_mask : !pto.vreg<64xf32>, !pto.vreg<64xf32>, !pto.mask<b32> -> !pto.vreg<64xf32>
```

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

- **syntax:** `%result = pto.vdup %input, %mask {position = "LOWEST|HIGHEST"} : T|!pto.vreg<NxT>, !pto.mask<G> -> !pto.vreg<NxT>`
- **semantics:** Duplicate scalar or vector element to all lanes.
- **inputs:**
  `%input` supplies the scalar or source-lane value selected by `position`,
  and `%mask` controls the active lanes.
- **outputs:**
  `%result` is the duplicated vector.
- **constraints and limitations:**
  `position` selects which source vector element is duplicated and is only valid
  for vector input. `position` defaults to `LOWEST`.

```c
for (int i = 0; i < N; i++)
    dst[i] = mask[i] ? input_scalar_or_element : 0;
```

---

#### Predicate Generation

##### `pto.pset_b8` / `pto.pset_b16` / `pto.pset_b32`

- **syntax:** `%result = pto.pset_b8 "PATTERN" : !pto.mask<b8>`
- **syntax:** `%result = pto.pset_b16 "PATTERN" : !pto.mask<b16>`
- **syntax:** `%result = pto.pset_b32 "PATTERN" : !pto.mask<b32>`
- **semantics:** Materialize a predicate register from a named pattern token.

**Supported pattern tokens:**

| Pattern | Description |
|---------|-------------|
| `PAT_ALL` | All lanes active |
| `PAT_ALLF` | All lanes inactive |
| `PAT_H` | High half active |
| `PAT_Q` | Upper quarter active |
| `PAT_VL1`...`PAT_VL128` | First N logical lanes active |
| `PAT_M3`, `PAT_M4` | Modular patterns |

`PAT_ALL` is the PTO spelling of the VISA-style all-true predicate pattern.
The other tokens listed above are also concrete installed-toolchain pattern
objects, not PTO-only aliases.

**Example — All 64 f32 lanes active:**
```mlir
%all_active = pto.pset_b32 "PAT_ALL" : !pto.mask<b32>
```

**Example — First 16 lanes active:**
```mlir
%first_16 = pto.pset_b32 "PAT_VL16" : !pto.mask<b32>
```

---

##### `pto.pge_b8` / `pto.pge_b16` / `pto.pge_b32`

- **syntax:** `%result = pto.pge_b8 "PATTERN" : !pto.mask<b8>`
- **syntax:** `%result = pto.pge_b16 "PATTERN" : !pto.mask<b16>`
- **syntax:** `%result = pto.pge_b32 "PATTERN" : !pto.mask<b32>`
- **semantics:** Generate a predicate from a lane-count pattern token. In the
  common tail-mask form, `PAT_VL<N>` marks the first `N` logical lanes active.
- **supported pattern tokens:** `PAT_ALL`, `PAT_ALLF`, `PAT_H`, `PAT_Q`,
  `PAT_VL1`, `PAT_VL2`, `PAT_VL3`, `PAT_VL4`, `PAT_VL8`, `PAT_VL16`,
  `PAT_VL32`, `PAT_VL64`, `PAT_VL128`, `PAT_M3`, `PAT_M4`

```c
for (int i = 0; i < TOTAL_LANES; i++)
    mask[i] = (i < len);
```

**Example — Tail mask for remainder loop:**
```mlir
%tail_mask = pto.pge_b32 "PAT_VL8" : !pto.mask<b32>
```

---

##### `pto.plt_b8` / `pto.plt_b16` / `pto.plt_b32`

- **syntax:** `%mask, %scalar_out = pto.plt_b8 %scalar : i32 -> !pto.mask<b8>, i32`
- **syntax:** `%mask, %scalar_out = pto.plt_b16 %scalar : i32 -> !pto.mask<b16>, i32`
- **syntax:** `%mask, %scalar_out = pto.plt_b32 %scalar : i32 -> !pto.mask<b32>, i32`
- **semantics:** Generate a tail-style predicate from an SSA lane-count value.
  On A5/V300-style toolchains, this family is exposed as a post-update wrapper:
  the predicate result becomes `%mask`, and the wrapper's carry-out scalar state
  is surfaced as `%scalar_out`.
- **inputs:**
  `%scalar` is the incoming lane-count / remaining-count state.
- **outputs:**
  `%mask` is the generated predicate.
  `%scalar_out` is the post-update scalar carry-out from the same `plt` call
  and can be threaded into a subsequent `pto.plt_b*` call in the same chain.

```c
for (int i = 0; i < VL_t; ++i)
    mask[i] = (i < scalar_in);

scalar_out = (scalar_in < VL_t) ? 0 : (scalar_in - VL_t);
```

Where `VL_t` is the logical lane count of the concrete op variant:

- `pto.plt_b8`: `VL_t = 256`
- `pto.plt_b16`: `VL_t = 128`
- `pto.plt_b32`: `VL_t = 64`

---

#### Predicate Pack/Unpack

##### `pto.ppack`

- **syntax:** `%result = pto.ppack %input, "PART" : !pto.mask<G> -> !pto.mask<G>`
- **semantics:** Narrowing pack of predicate register.
- **part tokens:**
  - `LOWER`: pack into the lower half of `%result`; the upper half is zeroed.
  - `HIGHER`: pack into the higher half of `%result`; the lower half is zeroed.

Conceptually, `pto.ppack` keeps one bit out of each adjacent 2-bit group from
`%input`, packs those kept bits into the selected half of `%result`, and fills
the other half with zeros.

```c
// Let VL be the logical lane count of the destination predicate.
// LOWER
for (int i = 0; i < VL / 2; ++i)
    result[i] = input[2 * i];
for (int i = VL / 2; i < VL; ++i)
    result[i] = 0;

// HIGHER
for (int i = 0; i < VL / 2; ++i)
    result[VL / 2 + i] = input[2 * i];
for (int i = 0; i < VL / 2; ++i)
    result[i] = 0;
```

---

##### `pto.punpack`

- **syntax:** `%result = pto.punpack %input, "PART" : !pto.mask<G> -> !pto.mask<G>`
- **semantics:** Widening unpack of predicate register.
- **part tokens:**
  - `LOWER`: unpack from the lower half of `%input`.
  - `HIGHER`: unpack from the higher half of `%input`.

Conceptually, `pto.punpack` reads the selected half of `%input`, zero-extends
each 1-bit predicate element into a 2-bit group in `%result`, and leaves the
expanded image in the full destination predicate register.

```c
// Let VL be the logical lane count of the destination predicate.
// LOWER
for (int i = 0; i < VL / 2; ++i) {
    result[2 * i] = input[i];
    result[2 * i + 1] = 0;
}

// HIGHER
for (int i = 0; i < VL / 2; ++i) {
    result[2 * i] = input[VL / 2 + i];
    result[2 * i + 1] = 0;
}
```

---

#### Predicate Logical Ops

##### `pto.pand`

- **syntax:** `%result = pto.pand %src0, %src1, %mask : !pto.mask<G>, !pto.mask<G>, !pto.mask<G> -> !pto.mask<G>`
- **semantics:** Predicate bitwise AND gated by a governing predicate.

Inactive lanes selected out by `%mask` are zeroed.

```c
for (int i = 0; i < N; i++)
    dst[i] = mask[i] ? (src0[i] & src1[i]) : 0;
```

---

##### `pto.por`

- **syntax:** `%result = pto.por %src0, %src1, %mask : !pto.mask<G>, !pto.mask<G>, !pto.mask<G> -> !pto.mask<G>`
- **semantics:** Predicate bitwise OR gated by a governing predicate.

Inactive lanes selected out by `%mask` are zeroed.

```c
for (int i = 0; i < N; i++)
    dst[i] = mask[i] ? (src0[i] | src1[i]) : 0;
```

---

##### `pto.pxor`

- **syntax:** `%result = pto.pxor %src0, %src1, %mask : !pto.mask<G>, !pto.mask<G>, !pto.mask<G> -> !pto.mask<G>`
- **semantics:** Predicate bitwise XOR gated by a governing predicate.

Inactive lanes selected by `%mask` are zeroed.

```c
for (int i = 0; i < N; i++)
    dst[i] = mask[i] ? (src0[i] ^ src1[i]) : 0;
```

---

##### `pto.pnot`

- **syntax:** `%result = pto.pnot %input, %mask : !pto.mask<G>, !pto.mask<G> -> !pto.mask<G>`
- **semantics:** Predicate bitwise NOT gated by a governing predicate.

Inactive lanes selected by `%mask` are zeroed.

```c
for (int i = 0; i < N; i++)
    dst[i] = mask[i] ? (~src[i]) : 0;
```

---

##### `pto.psel`

- **syntax:** `%result = pto.psel %src0, %src1, %sel : !pto.mask<G>, !pto.mask<G>, !pto.mask<G> -> !pto.mask<G>`
- **semantics:** Predicate select (mux). `%sel` is the governing predicate that
  chooses lanes from `%src0` or `%src1`.

```c
for (int i = 0; i < N; i++)
    dst[i] = sel[i] ? src0[i] : src1[i];
```

---

##### `pto.pdintlv_b8` / `pto.pdintlv_b16` / `pto.pdintlv_b32`

- **syntax:** `%low, %high = pto.pdintlv_b8 %src0, %src1 : !pto.mask<b8>, !pto.mask<b8> -> !pto.mask<b8>, !pto.mask<b8>`
- **syntax:** `%low, %high = pto.pdintlv_b16 %src0, %src1 : !pto.mask<b16>, !pto.mask<b16> -> !pto.mask<b16>, !pto.mask<b16>`
- **syntax:** `%low, %high = pto.pdintlv_b32 %src0, %src1 : !pto.mask<b32>, !pto.mask<b32> -> !pto.mask<b32>, !pto.mask<b32>`
- **semantics:** De-interleave two predicate sources and return the two
  de-interleaved predicate images in the same predicate element family.

---

##### `pto.pintlv_b8` / `pto.pintlv_b16` / `pto.pintlv_b32`

- **syntax:** `%low, %high = pto.pintlv_b8 %src0, %src1 : !pto.mask<b8>, !pto.mask<b8> -> !pto.mask<b8>, !pto.mask<b8>`
- **syntax:** `%low, %high = pto.pintlv_b16 %src0, %src1 : !pto.mask<b16>, !pto.mask<b16> -> !pto.mask<b16>, !pto.mask<b16>`
- **syntax:** `%low, %high = pto.pintlv_b32 %src0, %src1 : !pto.mask<b32>, !pto.mask<b32> -> !pto.mask<b32>, !pto.mask<b32>`
- **semantics:** Interleave two predicate sources and return the two
  resulting predicate images in the same predicate element family.

---

#### Typical Usage

```mlir
// Generate all-active mask for f32 (64 lanes)
%all = pto.pset_b32 "PAT_ALL" : !pto.mask<b32>

// Generate tail mask for remainder (last 12 elements)
%tail = pto.pge_b32 "PAT_VL12" : !pto.mask<b32>

// Compare and generate mask
%cmp_mask = pto.vcmp %a, %b, %all, "lt" : !pto.vreg<64xf32>, !pto.vreg<64xf32>, !pto.mask<b32> -> !pto.mask<b32>

// Combine masks: only process tail elements that passed comparison
%combined = pto.pand %cmp_mask, %tail, %all : !pto.mask<b32>, !pto.mask<b32>, !pto.mask<b32> -> !pto.mask<b32>

// Use for predicated operation
%result = pto.vsel %true_vals, %false_vals, %combined : !pto.vreg<64xf32>, !pto.vreg<64xf32>, !pto.mask<b32> -> !pto.vreg<64xf32>
```

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

#### CA latency (A5, Ascend910_9599 CA)

Cycle-accurate simulator **popped→retire** latency (cycles). **fp16** values use **aclFloat16** in traces where measured. **bf16:** no simple-tile ST coverage on this surface; treat as **—**.

| PTO op | RV (CA) | fp32 | fp16 | bf16 |
|--------|---------|------|------|------|
| `pto.vabs` | `RV_VABS_FP` | **5** | **5** | — |
| `pto.vneg` | `RV_VMULS` | **8** | **8** | — |
| `pto.vexp` | `RV_VEXP` | **16** | **21** | — |
| `pto.vln` | `RV_VLN` | **18** | **23** | — |
| `pto.vsqrt` | `RV_VSQRT` | **17** | **22** | — |
| `pto.vrelu` | `RV_VRELU` | **5** | **5** | — |
| `pto.vnot` | `RV_VNOT` | — | int-only paths | — |
| `pto.vmov` | `RV_VLD` proxy | **9** | **9** | — |

---

#### Arithmetic

##### `pto.vabs`

- **syntax:** `%result = pto.vabs %input, %mask : !pto.vreg<NxT>, !pto.mask<G> -> !pto.vreg<NxT>`
- **A5 types:** i8-i32, f16, f32

```c
for (int i = 0; i < N; i++)
    dst[i] = (src[i] < 0) ? -src[i] : src[i];
```

- **inputs:** `%input` supplies the source lanes and `%mask` selects which lanes
  participate.
- **outputs:** `%result` receives the lane-wise absolute values.
- **constraints and limitations:** Source and result types MUST match. On A5,
  integer overflow follows the ISA default truncation behavior for this family;
  `pto.vabs` is not an explicit saturating op.

---

##### `pto.vneg`

- **syntax:** `%result = pto.vneg %input, %mask : !pto.vreg<NxT>, !pto.mask<G> -> !pto.vreg<NxT>`
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

- **syntax:** `%result = pto.vexp %input, %mask : !pto.vreg<NxT>, !pto.mask<G> -> !pto.vreg<NxT>`
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

- **syntax:** `%result = pto.vln %input, %mask : !pto.vreg<NxT>, !pto.mask<G> -> !pto.vreg<NxT>`
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

- **syntax:** `%result = pto.vsqrt %input, %mask : !pto.vreg<NxT>, !pto.mask<G> -> !pto.vreg<NxT>`
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

#### Activation

##### `pto.vrelu`

- **syntax:** `%result = pto.vrelu %input, %mask : !pto.vreg<NxT>, !pto.mask<G> -> !pto.vreg<NxT>`
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

- **syntax:** `%result = pto.vnot %input, %mask : !pto.vreg<NxT>, !pto.mask<G> -> !pto.vreg<NxT>`
- **A5 types:** all integer types

```c
for (int i = 0; i < N; i++)
    dst[i] = ~src[i];
```

- **inputs:** `%input` is the source vector and `%mask` selects active lanes.
- **outputs:** `%result` holds the lane-wise bitwise inversion.
- **constraints and limitations:** Integer element types only.

---

#### Movement

#### Typical Usage

```mlir
// Softmax numerator: exp(x - max)
%sub = pto.vsub %x, %max_broadcast, %mask : !pto.vreg<64xf32>, !pto.vreg<64xf32>, !pto.mask<G> -> !pto.vreg<64xf32>
%exp = pto.vexp %sub, %mask : !pto.vreg<64xf32>, !pto.mask<G> -> !pto.vreg<64xf32>

// ReLU activation
%activated = pto.vrelu %linear_out, %mask : !pto.vreg<64xf32>, !pto.mask<G> -> !pto.vreg<64xf32>
```

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

#### CA latency (A5, Ascend910_9599 CA)

Cycle-accurate simulator **popped→retire** latency (cycles). **fp16** uses **aclFloat16** in measured traces. **bf16:** — (no dedicated vec tile ST on this surface).

| PTO op | RV (CA) | fp32 | fp16 | bf16 |
|--------|---------|------|------|------|
| `pto.vadd` | `RV_VADD` | **7** | **7** | — |
| `pto.vsub` | `RV_VSUB` | **7** | **7** | — |
| `pto.vmul` | `RV_VMUL` | **8** | **8** | — |
| `pto.vdiv` | `RV_VDIV` | **17** | **22** | — |

---

#### Arithmetic

##### `pto.vadd`

- **syntax:** `%result = pto.vadd %lhs, %rhs, %mask : !pto.vreg<NxT>, !pto.vreg<NxT>, !pto.mask<G> -> !pto.vreg<NxT>`
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

- **syntax:** `%result = pto.vsub %lhs, %rhs, %mask : !pto.vreg<NxT>, !pto.vreg<NxT>, !pto.mask<G> -> !pto.vreg<NxT>`
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

- **syntax:** `%result = pto.vmul %lhs, %rhs, %mask : !pto.vreg<NxT>, !pto.vreg<NxT>, !pto.mask<G> -> !pto.vreg<NxT>`
- **A5 types:** i16-i32, f16, bf16, f32 (**NOT** i8/ui8)

```c
for (int i = 0; i < N; i++)
    dst[i] = src0[i] * src1[i];
```

- **inputs:** `%lhs` and `%rhs` are multiplied lane-wise; `%mask` selects
  active lanes.
- **outputs:** `%result` is the lane-wise product.
- **constraints and limitations:** The current A5 profile excludes `i8/ui8`
  forms from this surface.

---

##### `pto.vdiv`

- **syntax:** `%result = pto.vdiv %lhs, %rhs, %mask : !pto.vreg<NxT>, !pto.vreg<NxT>, !pto.mask<G> -> !pto.vreg<NxT>`
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

- **syntax:** `%result = pto.vmax %lhs, %rhs, %mask : !pto.vreg<NxT>, !pto.vreg<NxT>, !pto.mask<G> -> !pto.vreg<NxT>`
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

- **syntax:** `%result = pto.vmin %lhs, %rhs, %mask : !pto.vreg<NxT>, !pto.vreg<NxT>, !pto.mask<G> -> !pto.vreg<NxT>`
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

- **syntax:** `%result = pto.vand %lhs, %rhs, %mask : !pto.vreg<NxT>, !pto.vreg<NxT>, !pto.mask<G> -> !pto.vreg<NxT>`
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

- **syntax:** `%result = pto.vor %lhs, %rhs, %mask : !pto.vreg<NxT>, !pto.vreg<NxT>, !pto.mask<G> -> !pto.vreg<NxT>`
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

- **syntax:** `%result = pto.vxor %lhs, %rhs, %mask : !pto.vreg<NxT>, !pto.vreg<NxT>, !pto.mask<G> -> !pto.vreg<NxT>`
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

- **syntax:** `%result = pto.vshl %lhs, %rhs, %mask : !pto.vreg<NxT>, !pto.vreg<NxT>, !pto.mask<G> -> !pto.vreg<NxT>`
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

- **syntax:** `%result = pto.vshr %lhs, %rhs, %mask : !pto.vreg<NxT>, !pto.vreg<NxT>, !pto.mask<G> -> !pto.vreg<NxT>`
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

- **syntax:** `%result, %carry = pto.vaddc %lhs, %rhs, %mask : !pto.vreg<NxT>, !pto.vreg<NxT>, !pto.mask<G> -> !pto.vreg<NxT>, !pto.mask<G>`
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
- **A5 types:** `i32`, `si32`, `ui32`
- **constraints and limitations:** This is a carry-chain integer add family. On
  the current A5 surface, only 32-bit integer element types are supported.
  `%mask` and `%carry` therefore use the same typed-mask granularity as the
  data vector family, which on the current documented A5 surface means
  `!pto.mask<b32>`.

---

##### `pto.vsubc`

- **syntax:** `%result, %carry = pto.vsubc %lhs, %rhs, %mask : !pto.vreg<NxT>, !pto.vreg<NxT>, !pto.mask<G> -> !pto.vreg<NxT>, !pto.mask<G>`
- **semantics:** Subtract with per-lane carry output.

```c
for (int i = 0; i < N; i++) {
    dst[i] = src0[i] - src1[i];
    carry[i] = (src0[i] >= src1[i]);
}
```

- **inputs:** `%lhs` and `%rhs` are subtracted lane-wise and `%mask` selects
  active lanes.
- **outputs:** `%result` is the arithmetic difference and `%carry` is the
  per-lane carry predicate. For this subtraction family, active lanes set
  `%carry[i] = 1` when the subtraction completes without borrow, and
  `%carry[i] = 0` when a borrow occurs.
- **A5 types:** `i32`, `si32`, `ui32`
- **constraints and limitations:** This operation is currently restricted to
  the 32-bit integer carry/borrow-chain family. `%mask` and `%carry`
  therefore use the same typed-mask granularity as the data vector family,
  which on the current documented A5 surface means `!pto.mask<b32>`.

---

#### Typical Usage

```mlir
// Vector addition
%sum = pto.vadd %a, %b, %mask : !pto.vreg<64xf32>, !pto.vreg<64xf32>, !pto.mask<G> -> !pto.vreg<64xf32>

// Element-wise multiply
%prod = pto.vmul %x, %y, %mask : !pto.vreg<64xf32>, !pto.vreg<64xf32>, !pto.mask<G> -> !pto.vreg<64xf32>

// Clamp to range [min, max]
%clamped_low = pto.vmax %input, %min_vec, %mask : !pto.vreg<64xf32>, !pto.vreg<64xf32>, !pto.mask<G> -> !pto.vreg<64xf32>
%clamped = pto.vmin %clamped_low, %max_vec, %mask : !pto.vreg<64xf32>, !pto.vreg<64xf32>, !pto.mask<G> -> !pto.vreg<64xf32>

// Bit manipulation
%masked = pto.vand %data, %bitmask, %mask : !pto.vreg<64xi32>, !pto.vreg<64xi32>, !pto.mask<G> -> !pto.vreg<64xi32>
```

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
- For elementwise vec-scalar families whose scalar conceptually matches the
  vector element type (`pto.vadds`, `pto.vmuls`, `pto.vmaxs`,
  `pto.vmins`, `pto.vlrelu`):
  - signed integer vectors accept signed integer scalars with the same width,
    and also accept signless `i<width>`
  - unsigned integer vectors accept unsigned integer scalars with the same
    width, and also accept signless `i<width>`
  - signless integer vectors accept signless `i<width>`
- `pto.vshls` and `pto.vshrs` are not part of that rule; their scalar operand
  is the shift amount and remains fixed to `i16`.

#### CA latency (A5, Ascend910_9599 CA)

Cycle-accurate simulator **popped→retire** latency (cycles). **fp16** uses **aclFloat16** in measured traces. **bf16:** —.

| PTO op | RV (CA) | fp32 | fp16 | bf16 |
|--------|---------|------|------|------|
| `pto.vadds` | `RV_VADDS` | **7** | **7** | — |
| `pto.vmuls` | `RV_VMULS` | **8** | **8** | — |

---

#### Arithmetic

##### `pto.vadds`

- **syntax:** `%result = pto.vadds %input, %scalar, %mask : !pto.vreg<NxT>, T, !pto.mask<G> -> !pto.vreg<NxT>`
- **A5 types:** `si8`, `si16`, `si32`, `ui8`, `ui16`, `ui32`, `f16`, `bf16`, `f32`

```c
for (int i = 0; i < N; i++)
    dst[i] = src[i] + scalar;
```

- **inputs:** `%input` is the source vector, `%scalar` is broadcast logically to
  each lane, and `%mask` selects active lanes.
- **outputs:** `%result` is the lane-wise sum.
- **constraints and limitations:** Input vector element type, scalar type, and
  result vector element type MUST match. For integer vector forms, `%scalar`
  may also use matching-signedness integer or signless `i<width>` with the same
  bit width as the vector element type, so it can be fed directly from `arith`
  constants.

---

##### `pto.vmuls`

- **syntax:** `%result = pto.vmuls %input, %scalar, %mask : !pto.vreg<NxT>, T, !pto.mask<G> -> !pto.vreg<NxT>`

```c
for (int i = 0; i < N; i++)
    dst[i] = src[i] * scalar;
```

- **inputs:** `%input`, `%scalar`, and `%mask` as above.
- **outputs:** `%result` is the lane-wise product.
- **constraints and limitations:** Supported element types are hardware-family
  specific; the current PTO micro Instruction documentation covers the common
  numeric cases. For integer vector forms, `%scalar` may use matching-signedness
  integer or signless `i<width>` with the same bit width as the vector element
  type.

---

##### `pto.vmaxs`

- **syntax:** `%result = pto.vmaxs %input, %scalar, %mask : !pto.vreg<NxT>, T, !pto.mask<G> -> !pto.vreg<NxT>`

```c
for (int i = 0; i < N; i++)
    dst[i] = (src[i] > scalar) ? src[i] : scalar;
```

- **inputs:** `%input`, `%scalar`, and `%mask` as above.
- **outputs:** `%result` is the lane-wise maximum.
- **constraints and limitations:** Input and result types MUST match. For
  integer vector forms, `%scalar` may use matching-signedness integer or
  signless `i<width>` with the same bit width as the vector element type.

---

##### `pto.vmins`

- **syntax:** `%result = pto.vmins %input, %scalar, %mask : !pto.vreg<NxT>, T, !pto.mask<G> -> !pto.vreg<NxT>`

```c
for (int i = 0; i < N; i++)
    dst[i] = (src[i] < scalar) ? src[i] : scalar;
```

- **inputs:** `%input`, `%scalar`, and `%mask` as above.
- **outputs:** `%result` is the lane-wise minimum.
- **constraints and limitations:** Input and result types MUST match. For
  integer vector forms, `%scalar` may use matching-signedness integer or
  signless `i<width>` with the same bit width as the vector element type.

---

#### Shift

##### `pto.vshls`

- **syntax:** `%result = pto.vshls %input, %scalar, %mask : !pto.vreg<NxT>, i16, !pto.mask<G> -> !pto.vreg<NxT>`

```c
for (int i = 0; i < N; i++)
    dst[i] = src[i] << scalar;
```

- **inputs:** `%input` is the value vector, `%scalar` is the uniform `i16` shift
  amount, and `%mask` selects active lanes.
- **outputs:** `%result` is the shifted vector.
- **constraints and limitations:** Integer element types only. The shift amount
  SHOULD stay within the source element width.

---

##### `pto.vshrs`

- **syntax:** `%result = pto.vshrs %input, %scalar, %mask : !pto.vreg<NxT>, i16, !pto.mask<G> -> !pto.vreg<NxT>`

```c
for (int i = 0; i < N; i++)
    dst[i] = src[i] >> scalar;
```

- **inputs:** `%input` is the value vector, `%scalar` is the uniform `i16` shift
  amount, and `%mask` selects active lanes.
- **outputs:** `%result` is the shifted vector.
- **constraints and limitations:** Integer element types only.

---

##### `pto.vlrelu`

- **syntax:** `%result = pto.vlrelu %input, %scalar, %mask : !pto.vreg<NxT>, T, !pto.mask<G> -> !pto.vreg<NxT>`

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

- **syntax:** `%result, %carry = pto.vaddcs %lhs, %rhs, %carry_in, %mask : !pto.vreg<NxT>, !pto.vreg<NxT>, !pto.mask<G>, !pto.mask<G> -> !pto.vreg<NxT>, !pto.mask<G>`
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
- **A5 types:** `i32`, `si32`, `ui32`
- **constraints and limitations:** This is the scalar-extended carry-chain
  family. On the current A5 surface, only 32-bit integer element types are
  supported. `%carry_in`, `%mask`, and `%carry` therefore all use the same
  typed-mask granularity as the data vector family, which on the current
  documented A5 surface means `!pto.mask<b32>`.

---

##### `pto.vsubcs`

- **syntax:** `%result, %carry = pto.vsubcs %lhs, %rhs, %carry_in, %mask : !pto.vreg<NxT>, !pto.vreg<NxT>, !pto.mask<G>, !pto.mask<G> -> !pto.vreg<NxT>, !pto.mask<G>`
- **semantics:** Subtract with carry input and output.

```c
for (int i = 0; i < N; i++) {
    dst[i] = src0[i] - src1[i] - (1 - carry_in[i]);
    carry_out[i] = (src0[i] >= src1[i] + (1 - carry_in[i]));
}
```

- **inputs:** `%lhs` and `%rhs` are the value vectors, `%carry_in` is the
  incoming carry predicate, and `%mask` selects active lanes.
- **outputs:** `%result` is the arithmetic result and `%carry` is the
  carry predicate after the lane-wise subtraction. For this subtraction family,
  active lanes set `%carry[i] = 1` when the subtraction completes without
  borrow, and `%carry[i] = 0` when a borrow occurs.
- **A5 types:** `i32`, `si32`, `ui32`
- **constraints and limitations:** This is the scalar-extended borrow-chain
  family and is currently restricted to 32-bit integer element types.
  `%carry_in`, `%mask`, and `%carry` therefore all use the same typed-mask
  granularity as the data vector family, which on the current documented A5
  surface means `!pto.mask<b32>`.

---

#### Typical Usage

```mlir
// Add bias to all elements
%biased = pto.vadds %activation, %bias_scalar, %mask : !pto.vreg<64xf32>, f32, !pto.mask<G> -> !pto.vreg<64xf32>

// Scale by constant
%scaled = pto.vmuls %input, %scale, %mask : !pto.vreg<64xf32>, f32, !pto.mask<G> -> !pto.vreg<64xf32>

// Clamp to [0, 255] for uint8 quantization
%clamped_low = pto.vmaxs %input, %c0, %mask : !pto.vreg<64xf32>, f32, !pto.mask<G> -> !pto.vreg<64xf32>
%clamped = pto.vmins %clamped_low, %c255, %mask : !pto.vreg<64xf32>, f32, !pto.mask<G> -> !pto.vreg<64xf32>

// Shift right by fixed amount
%shifted = pto.vshrs %data, %c4, %mask : !pto.vreg<64xi32>, i16, !pto.mask<G> -> !pto.vreg<64xi32>
```

<a id="isa-09-conversion-ops"></a>

### 9. Conversion Ops

> **Category:** Type conversion operations
> **Pipeline:** PIPE_V (Vector Core)

Operations that convert between data types (float/int, narrowing/widening).

#### Common Operand Model

- `%input` is the source vector register value.
- `%mask` is the predicate mask that selects active conversion lanes.
- `%result` is the destination vector register value.
- `rnd`, `sat`, and `part` are optional attributes that refine
  conversion behavior when the selected source/destination type pair needs
  rounding, saturation, or lane placement control.
- The single `pto.vcvt` surface covers float-int, float-float, int-float, and
  int-int conversion families.

#### CA latency (A5, Ascend910_9599 CA)

Cycle-accurate simulator **popped→retire** latency (cycles). Only representative traces below; other `pto.vcvt` conversion pairs depend on the RV lowering in the trace.

| PTO op | RV (CA) | Note | Latency |
|--------|---------|------|---------|
| `pto.vcvt` | `RV_VCVT_F2F` | f32→f16 | **7** |
| `pto.vci` | — | no vector `RV_*` in sampled `veccore0` trace | — |

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
  result element type together determine how indices are generated. `%result`
  uses an integer element type, and the scalar `%index` type matches that
  result element type.

---

#### `pto.vcvt`

- **syntax:** `%result = pto.vcvt %input, %mask {rnd = "RND", sat = "SAT", part = "PART"} : !pto.vreg<NxT0>, !pto.mask<G> -> !pto.vreg<MxT1>`
- **semantics:** Type conversion between float/int types with rounding control.

```c
for (int i = 0; i < min(N, M); i++)
    if (mask[i])
        dst[i] = convert(src[i], T0, T1, rnd);
```

- **inputs:**
  `%input` is the source vector, `%mask` selects active lanes, and attributes
  select rounding, saturation, and output placement when the conversion changes
  width or packs into sub-lane positions.
- **outputs:**
  `%result` is the converted vector.
- **constraints and limitations:**
  Only documented source/destination type pairs are legal. All three
  attributes are optional at the surface level, but only the subset meaningful
  to the selected conversion kind should be provided. The execution mask must
  use the typed-mask granularity that matches the source vector family on the
  current surface; there is no `!pto.mask<b64>` form in VPTO.

---

##### Rounding Modes

| Mode | Description |
|------|-------------|
| `R` | Round to nearest, ties to even (default) |
| `A` | Round away from zero |
| `F` | Round toward negative infinity (floor) |
| `C` | Round toward positive infinity (ceil) |
| `Z` | Round toward zero (truncate) |
| `O` | Round to odd |

---

##### Saturation Modes

| Mode | Description |
|------|-------------|
| `SAT` | Saturate on overflow |
| `NOSAT` | No saturation (wrap/undefined on overflow) |

---

##### Part Modes

Use `part` when a width-changing conversion writes only one half of each wider
destination lane group. This is typically used in even/odd placement forms such
as `32 -> 16` or `16 -> 32` style conversions.

| Mode | Description |
|------|-------------|
| `EVEN` | Output to even-indexed lanes |
| `ODD` | Output to odd-indexed lanes |

---

##### Attribute Guidance

- `rnd`
  - Use when the conversion needs an explicit rounding rule, especially for
    float-to-int, float-to-float narrowing, or integer-to-float forms that do
    not map exactly.
- `mask`
  - Use to select which source lanes participate in the conversion. In
    width-changing conversions, `mask` works together with `part` / `pp` to
    determine which logical lane positions are produced.
- `sat`
  - Use when the conversion may overflow the destination range and hardware
    exposes a saturating form.
- `part`
  - Use for width-changing conversions that select the even or odd half of the
    destination packing layout.

###### Float To Int

- `%dst = pto.vcvt %src, %mask {rnd, sat, part} : !pto.vreg<64xf32>, !pto.mask<b32> -> !pto.vreg<32xsi64>`
- `%dst = pto.vcvt %src, %mask {rnd, sat} : !pto.vreg<64xf32>, !pto.mask<b32> -> !pto.vreg<64xsi32>`
- `%dst = pto.vcvt %src, %mask {rnd, sat, part} : !pto.vreg<64xf32>, !pto.mask<b32> -> !pto.vreg<128xsi16>`
- `%dst = pto.vcvt %src, %mask {rnd, part} : !pto.vreg<128xf16>, !pto.mask<b16> -> !pto.vreg<64xsi32>`
- `%dst = pto.vcvt %src, %mask {rnd, sat} : !pto.vreg<128xf16>, !pto.mask<b16> -> !pto.vreg<128xsi16>`
- `%dst = pto.vcvt %src, %mask {rnd, sat, part} : !pto.vreg<128xf16>, !pto.mask<b16> -> !pto.vreg<256xsi8>`
- `%dst = pto.vcvt %src, %mask {rnd, sat, part} : !pto.vreg<128xf16>, !pto.mask<b16> -> !pto.vreg<256xui8>`
- `%dst = pto.vcvt %src, %mask {rnd, sat, part} : !pto.vreg<128xbf16>, !pto.mask<b16> -> !pto.vreg<64xsi32>`

###### Float To Float

- `%dst = pto.vcvt %src, %mask {rnd, sat, part} : !pto.vreg<64xf32>, !pto.mask<b32> -> !pto.vreg<128xf16>`
- `%dst = pto.vcvt %src, %mask {rnd, sat, part} : !pto.vreg<64xf32>, !pto.mask<b32> -> !pto.vreg<128xbf16>`
- `%dst = pto.vcvt %src, %mask {part} : !pto.vreg<128xf16>, !pto.mask<b16> -> !pto.vreg<64xf32>`
- `%dst = pto.vcvt %src, %mask {part} : !pto.vreg<128xbf16>, !pto.mask<b16> -> !pto.vreg<64xf32>`

###### Int To Float

- `%dst = pto.vcvt %src, %mask {part} : !pto.vreg<256xui8>, !pto.mask<b8> -> !pto.vreg<128xf16>`
- `%dst = pto.vcvt %src, %mask {part} : !pto.vreg<256xsi8>, !pto.mask<b8> -> !pto.vreg<128xf16>`
- `%dst = pto.vcvt %src, %mask {rnd} : !pto.vreg<128xsi16>, !pto.mask<b16> -> !pto.vreg<128xf16>`
- `%dst = pto.vcvt %src, %mask {part} : !pto.vreg<128xsi16>, !pto.mask<b16> -> !pto.vreg<64xf32>`
- `%dst = pto.vcvt %src, %mask {rnd} : !pto.vreg<64xsi32>, !pto.mask<b32> -> !pto.vreg<64xf32>`

###### Int To Int

- `%dst = pto.vcvt %src, %mask {part} : !pto.vreg<256xui8>, !pto.mask<b8> -> !pto.vreg<128xui16>`
- `%dst = pto.vcvt %src, %mask {part} : !pto.vreg<256xui8>, !pto.mask<b8> -> !pto.vreg<64xui32>`
- `%dst = pto.vcvt %src, %mask {part} : !pto.vreg<256xsi8>, !pto.mask<b8> -> !pto.vreg<128xsi16>`
- `%dst = pto.vcvt %src, %mask {part} : !pto.vreg<256xsi8>, !pto.mask<b8> -> !pto.vreg<64xsi32>`
- `%dst = pto.vcvt %src, %mask {sat, part} : !pto.vreg<128xui16>, !pto.mask<b16> -> !pto.vreg<256xui8>`
- `%dst = pto.vcvt %src, %mask {part} : !pto.vreg<128xui16>, !pto.mask<b16> -> !pto.vreg<64xui32>`
- `%dst = pto.vcvt %src, %mask {sat, part} : !pto.vreg<128xsi16>, !pto.mask<b16> -> !pto.vreg<256xui8>`
- `%dst = pto.vcvt %src, %mask {part} : !pto.vreg<128xsi16>, !pto.mask<b16> -> !pto.vreg<64xui32>`
- `%dst = pto.vcvt %src, %mask {part} : !pto.vreg<128xsi16>, !pto.mask<b16> -> !pto.vreg<64xsi32>`
- `%dst = pto.vcvt %src, %mask {sat, part} : !pto.vreg<64xui32>, !pto.mask<b32> -> !pto.vreg<256xui8>`
- `%dst = pto.vcvt %src, %mask {sat, part} : !pto.vreg<64xui32>, !pto.mask<b32> -> !pto.vreg<128xui16>`
- `%dst = pto.vcvt %src, %mask {sat, part} : !pto.vreg<64xui32>, !pto.mask<b32> -> !pto.vreg<128xsi16>`
- `%dst = pto.vcvt %src, %mask {sat, part} : !pto.vreg<64xsi32>, !pto.mask<b32> -> !pto.vreg<256xui8>`
- `%dst = pto.vcvt %src, %mask {sat, part} : !pto.vreg<64xsi32>, !pto.mask<b32> -> !pto.vreg<128xui16>`
- `%dst = pto.vcvt %src, %mask {sat, part} : !pto.vreg<64xsi32>, !pto.mask<b32> -> !pto.vreg<128xsi16>`
- `%dst = pto.vcvt %src, %mask {part} : !pto.vreg<64xsi32>, !pto.mask<b32> -> !pto.vreg<32xsi64>`

##### A5 Supported Type Matrix

The table below is only a summary. For exact attribute combinations, use the
per-form entries above as the source of truth.

| `src \ dst` | `ui8` | `si8` | `ui16` | `si16` | `ui32` | `si32` | `si64` | `f16` | `f32` | `bf16` |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `ui8` |  |  | Y |  | Y |  |  | Y |  |  |
| `si8` |  |  |  | Y |  | Y |  | Y |  |  |
| `ui16` | Y |  |  |  | Y |  |  |  |  |  |
| `si16` | Y |  |  |  | Y | Y |  | Y | Y |  |
| `ui32` | Y |  | Y | Y |  |  |  |  |  |  |
| `si32` | Y |  | Y | Y |  |  | Y |  | Y |  |
| `si64` |  |  |  |  |  |  |  |  |  |  |
| `f16` | Y | Y |  | Y |  | Y |  |  | Y |  |
| `f32` |  |  |  | Y |  | Y | Y | Y |  | Y |
| `bf16` |  |  |  |  |  | Y |  |  | Y |  |

---

##### Width-Changing Conversion Pattern

For conversions that change width (e.g., f32→f16), use even/odd parts and combine:

```mlir
// Convert two f32 vectors to one f16 vector
%even = pto.vcvt %in0, %mask {rnd = "R", sat = "SAT", part = "EVEN"}
    : !pto.vreg<64xf32>, !pto.mask<b32> -> !pto.vreg<128xf16>
%odd  = pto.vcvt %in1, %mask {rnd = "R", sat = "SAT", part = "ODD"}
    : !pto.vreg<64xf32>, !pto.mask<b32> -> !pto.vreg<128xf16>
%result = pto.vor %even, %odd, %mask : !pto.vreg<128xf16>, !pto.vreg<128xf16>, !pto.mask<b16> -> !pto.vreg<128xf16>
```

---

#### `pto.vtrc`

- **syntax:** `%result = pto.vtrc %input, %mask, "RND" : !pto.vreg<NxT>, !pto.mask<BW> -> !pto.vreg<NxT>`
- **semantics:** Truncate/round float to integer-valued float (stays in float type).

```c
for (int i = 0; i < N; i++)
    dst[i] = round_to_int_valued_float(src[i], rnd);
```

- **inputs:**
  `%input` is the floating-point source vector, `%mask` selects active lanes,
  and `RND` selects the truncation/rounding rule.
- **outputs:**
  `%result` is still a floating-point vector, but each active lane now carries
  an integer-valued floating-point result.
- **constraints and limitations:**
  This op does not change the element type. `T` must be `f16`, `f32`, or
  `bf16`. `RND` must be one of `R`, `A`, `F`, `C`, or `Z`. `BW` must match the
  element width: `b16` for `f16`/`bf16`, `b32` for `f32`.

**Example:**
```mlir
// Round to nearest integer, keep as float
%rounded = pto.vtrc %input, %mask, "R" : !pto.vreg<64xf32>, !pto.mask<b32> -> !pto.vreg<64xf32>
// input:  [1.4, 2.6, -1.5, 3.0]
// output: [1.0, 3.0, -2.0, 3.0]
```

---

#### Typical Usage

```mlir
// Quantization: f32 → i8 with saturation
%scaled = pto.vmuls %input, %scale, %mask : !pto.vreg<64xf32>, f32, !pto.mask<b32> -> !pto.vreg<64xf32>
%quantized = pto.vcvt %scaled, %mask {rnd = "R", sat = "SAT"}
    : !pto.vreg<64xf32>, !pto.mask<b32> -> !pto.vreg<64xi32>
// Then narrow i32 → i8 via pack ops

// Mixed precision: bf16 → f32 for accumulation
%f32_vec = pto.vcvt %bf16_input, %mask {part = "EVEN"}
    : !pto.vreg<128xbf16>, !pto.mask<b16> -> !pto.vreg<64xf32>

// Floor for integer division
%floored = pto.vtrc %ratio, %mask, "F" : !pto.vreg<64xf32>, !pto.mask<b32> -> !pto.vreg<64xf32>
%int_div = pto.vcvt %floored, %mask {rnd = "Z"}
    : !pto.vreg<64xf32>, !pto.mask<b32> -> !pto.vreg<64xi32>
```

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

- **syntax:** `%result = pto.vcadd %input, %mask : !pto.vreg<NxT>, !pto.mask<G> -> !pto.vreg<NxT>`
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

- **syntax:** `%result = pto.vcmax %input, %mask : !pto.vreg<NxT>, !pto.mask<G> -> !pto.vreg<NxT>`
- **A5 types:** i16-i32, f16, f32
- **semantics:** Find max element with argmax. The lowest destination element
  stores the maximum value, the second-lowest destination element stores the
  index of the first maximum, and all remaining elements are zero-filled.

```c
T mx = -INF; int idx = 0;
for (int i = 0; i < N; i++)
    if (src[i] > mx) { mx = src[i]; idx = i; }
dst[0] = mx;
dst[1] = idx;
for (int i = 2; i < N; i++)
    dst[i] = 0;
```

- **inputs:** `%input` is the source vector and `%mask` selects participating
  lanes.
- **outputs:** `%result[0]` holds the extremum value and `%result[1]` holds the
  index. Other destination elements are zero-filled.
- **constraints and limitations:** If there are multiple maxima, the minimum
  index is written. For floating-point types, inactive lanes are treated as
  `-INF`; if all lanes are inactive, `%result[0]` becomes `-INF`. For integer
  types, inactive lanes are treated as the literal minimum value; if all lanes
  are inactive, `%result[0]` becomes that literal minimum value. The index is
  written into the second destination element slot of the same destination
  vector register.

---

##### `pto.vcmin`

- **syntax:** `%result = pto.vcmin %input, %mask : !pto.vreg<NxT>, !pto.mask<G> -> !pto.vreg<NxT>`
- **A5 types:** i16-i32, f16, f32
- **semantics:** Find min element with argmin. The lowest destination element
  stores the minimum value, the second-lowest destination element stores the
  index of the first minimum, and all remaining elements are zero-filled.

```c
T mn = INF; int idx = 0;
for (int i = 0; i < N; i++)
    if (src[i] < mn) { mn = src[i]; idx = i; }
dst[0] = mn;
dst[1] = idx;
for (int i = 2; i < N; i++)
    dst[i] = 0;
```

- **inputs:** `%input` is the source vector and `%mask` selects participating
  lanes.
- **outputs:** `%result[0]` holds the extremum value and `%result[1]` holds the
  index. Other destination elements are zero-filled.
- **constraints and limitations:** If there are multiple minima, the minimum
  index is written. For floating-point types, inactive lanes are treated as
  `+INF`; if all lanes are inactive, `%result[0]` becomes `+INF`. For integer
  types, inactive lanes are treated as the literal maximum value; if all lanes
  are inactive, `%result[0]` becomes that literal maximum value. The index is
  written into the second destination element slot of the same destination
  vector register.

---

#### Per-VLane (Group) Reductions

The vector register is organized as **8 VLanes** of 32 bytes each. Group reductions operate within each VLane independently.

```
vreg layout (f32 example, 64 elements total):
VLane 0: [0..7]   VLane 1: [8..15]  VLane 2: [16..23] VLane 3: [24..31]
VLane 4: [32..39] VLane 5: [40..47] VLane 6: [48..55] VLane 7: [56..63]
```

##### `pto.vcgadd`

- **syntax:** `%result = pto.vcgadd %input, %mask : !pto.vreg<NxT>, !pto.mask<G> -> !pto.vreg<NxT>`
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

- **syntax:** `%result = pto.vcgmax %input, %mask : !pto.vreg<NxT>, !pto.mask<G> -> !pto.vreg<NxT>`
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

- **syntax:** `%result = pto.vcgmin %input, %mask : !pto.vreg<NxT>, !pto.mask<G> -> !pto.vreg<NxT>`
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

- **syntax:** `%result = pto.vcpadd %input, %mask : !pto.vreg<NxT>, !pto.mask<G> -> !pto.vreg<NxT>`
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
%max_vec = pto.vcmax %logits, %mask : !pto.vreg<64xf32>, !pto.mask<G> -> !pto.vreg<64xf32>
// max is in lane 0, broadcast it
%max_broadcast = pto.vlds %ub_tmp[%c0] {dist = "BRC"} : !pto.ptr<f32, ub> -> !pto.vreg<64xf32>

// Row-wise sum using vcgadd (for 8-row tile)
%row_sums = pto.vcgadd %tile, %mask : !pto.vreg<64xf32>, !pto.mask<G> -> !pto.vreg<64xf32>
// Results at indices 0, 8, 16, 24, 32, 40, 48, 56

// Full vector sum for normalization
%total = pto.vcadd %values, %mask : !pto.vreg<64xf32>, !pto.mask<G> -> !pto.vreg<64xf32>
// total[0] contains the sum

// Prefix sum for cumulative distribution
%cdf = pto.vcpadd %pdf, %mask : !pto.vreg<64xf32>, !pto.mask<G> -> !pto.vreg<64xf32>
```

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

- **syntax:** `%result = pto.vcmp %src0, %src1, %seed, "CMP_MODE" : !pto.vreg<NxT>, !pto.vreg<NxT>, !pto.mask<G> -> !pto.mask<G>`
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
%all_active = pto.pset_b32 "PAT_ALL" : !pto.mask<b32>
%lt_mask = pto.vcmp %a, %b, %all_active, "lt" : !pto.vreg<64xf32>, !pto.vreg<64xf32>, !pto.mask<b32> -> !pto.mask<b32>
// lt_mask[i] = 1 if a[i] < b[i]
```

- **inputs:** `%src0`, `%src1`, and `%seed`; `CMP_MODE` selects the comparison
  predicate.
- **outputs:** `%result` is the generated predicate mask.
- **constraints and limitations:** Only lanes enabled by `%seed` participate.
  Integer and floating-point comparisons follow their own element-type-specific
  comparison rules. `%seed` and `%result` keep the typed-mask granularity that
  matches `%src0` / `%src1`.

---

##### `pto.vcmps`

- **syntax:** `%result = pto.vcmps %src, %scalar, %seed, "CMP_MODE" : !pto.vreg<NxT>, T, !pto.mask<G> -> !pto.mask<G>`
- **semantics:** Compare vector against scalar.

```c
for (int i = 0; i < N; i++)
    if (seed[i])
        dst[i] = (src[i] CMP scalar) ? 1 : 0;
```

**Example:**
```mlir
%positive_mask = pto.vcmps %values, %c0_f32, %all_active, "gt"
    : !pto.vreg<64xf32>, f32, !pto.mask<b32> -> !pto.mask<b32>
// positive_mask[i] = 1 if values[i] > 0
```

- **inputs:** `%src` is the vector source, `%scalar` is the scalar comparison
  value, and `%seed` is the incoming predicate.
- **outputs:** `%result` is the generated predicate mask.
- **constraints and limitations:** For 32-bit scalar forms, the scalar source
  MUST satisfy the backend's legal scalar-source constraints for this family.
  `%seed` and `%result` keep the typed-mask granularity that matches `%src`.

---

#### Selection Operations

##### `pto.vsel`

- **syntax:** `%result = pto.vsel %src0, %src1, %mask : !pto.vreg<NxT>, !pto.vreg<NxT>, !pto.mask<G> -> !pto.vreg<NxT>`
- **semantics:** Per-lane select based on mask.

```c
for (int i = 0; i < N; i++)
    dst[i] = mask[i] ? src0[i] : src1[i];
```

**Example — Conditional assignment:**
```mlir
// dst = mask ? true_vals : false_vals
%result = pto.vsel %true_vals, %false_vals, %condition
    : !pto.vreg<64xf32>, !pto.vreg<64xf32>, !pto.mask<b32> -> !pto.vreg<64xf32>
```

- **inputs:** `%src0` is the true-path vector, `%src1` is the false-path vector,
  and `%mask` selects between them.
- **outputs:** `%result` is the selected vector.
- **constraints and limitations:** Source vectors and result MUST have matching
  vector shapes and element types. `%mask` keeps the typed-mask granularity
  that matches the selected vector family.

---

##### `pto.vselr`

- **syntax:** `%result = pto.vselr %src, %idx : !pto.vreg<NxT>, !pto.vreg<Nxi<width>> -> !pto.vreg<NxT>`
- **semantics:** Lane-select by index vector.

```c
for (int i = 0; i < N; i++)
    dst[i] = src[idx[i]];
```

- **inputs:** `%src` is the source vector. `%idx` is the lane-index vector.
- **outputs:** `%result` is the reordered vector.
- **constraints and limitations:** `%idx` must use integer elements. `%idx`
  must have the same lane count as `%src`, and its integer element width must
  match the bit width of `%src` element type.

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
%all = pto.pset_b32 "PAT_ALL" : !pto.mask<b32>
%zero = pto.vbr %c0_f32 : f32 -> !pto.vreg<64xf32>
%neg_mask = pto.vcmps %input, %c0_f32, %all, "lt" : !pto.vreg<64xf32>, f32, !pto.mask<b32> -> !pto.mask<b32>
%clamped = pto.vsel %zero, %input, %neg_mask : !pto.vreg<64xf32>, !pto.vreg<64xf32>, !pto.mask<b32> -> !pto.vreg<64xf32>

// Element-wise max via compare+select
%gt_mask = pto.vcmp %a, %b, %all, "gt" : !pto.vreg<64xf32>, !pto.vreg<64xf32>, !pto.mask<b32> -> !pto.mask<b32>
%max_ab = pto.vsel %a, %b, %gt_mask : !pto.vreg<64xf32>, !pto.vreg<64xf32>, !pto.mask<b32> -> !pto.vreg<64xf32>

// Threshold filter
%above_thresh = pto.vcmps %scores, %threshold, %all, "ge" : !pto.vreg<64xf32>, f32, !pto.mask<b32> -> !pto.mask<b32>
%filtered = pto.vsel %scores, %zero, %above_thresh : !pto.vreg<64xf32>, !pto.vreg<64xf32>, !pto.mask<b32> -> !pto.vreg<64xf32>
```

---

#### Compare + Select Pattern

```mlir
// Softmax safe exp: exp(x - max) where x < max returns exp of negative
// but we want to clamp to avoid underflow

%all = pto.pset_b32 "PAT_ALL" : !pto.mask<b32>

// 1. Compare against threshold
%too_small = pto.vcmps %x_minus_max, %min_exp_arg, %all, "lt"
    : !pto.vreg<64xf32>, f32, !pto.mask<b32> -> !pto.mask<b32>

// 2. Clamp values below threshold
%clamped = pto.vsel %min_exp_arg_vec, %x_minus_max, %too_small
    : !pto.vreg<64xf32>, !pto.vreg<64xf32>, !pto.mask<b32> -> !pto.vreg<64xf32>

// 3. Safe exp
%exp_result = pto.vexp %clamped, %all : !pto.vreg<64xf32>, !pto.mask<b32> -> !pto.vreg<64xf32>
```

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

#### Compress / Expand

##### `pto.vsqz`

- **syntax:** `%result = pto.vsqz %src, %mask : !pto.vreg<NxT>, !pto.mask<G> -> !pto.vreg<NxT>`
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

- **syntax:** `%result = pto.vusqz %src, %mask : !pto.vreg<NxT>, !pto.mask<G> -> !pto.vreg<NxT>`
- **semantics:** Generate per-lane prefix counts from the governing predicate.

```c
dst[0] = 0;
for (int i = 1; i < N; i++)
    dst[i] = mask[i - 1] ? (dst[i - 1] + 1) : dst[i - 1];
```

- **inputs:** `%mask` is the governing predicate. The current PTO surface keeps
  `%src` in the operand list for interface compatibility, but the observable
  result semantics are determined by `%mask`.
- **outputs:** `%result[i]` equals the number of active lanes in `%mask[0:i)`,
  with `%result[0] = 0`.
- **constraints and limitations:** `T` is currently limited to `si8`, `si16`,
  or `si32`. This operation is a predicate-derived counting/rearrangement
  primitive rather than a value-placement primitive. The final predicate lane
  does not contribute to a later output lane because there is no `dst[N]`.

---

---

##### `pto.vselr`

- **syntax:** `%result = pto.vselr %src, %idx : !pto.vreg<NxT>, !pto.vreg<Nxi<width>> -> !pto.vreg<NxT>`
- **semantics:** Register lane-select with an explicit index vector.

```c
for (int i = 0; i < N; i++)
    dst[i] = src[idx[i]];
```

- **inputs:** `%src` is the source vector. `%idx` is the lane-index vector.
- **outputs:** `%result` is the reordered vector.
- **constraints and limitations:** This page records the rearrangement use of
  the family; the compare/select page documents the same name from the predicate
  selection perspective.

---

#### Pack / Unpack

##### `pto.vpack`

- **syntax:** `%result = pto.vpack %src, "PART" : !pto.vreg<NxT_wide> -> !pto.vreg<2NxT_narrow>`
- **semantics:** Narrow one wide vector and place the narrowed payload into the
  selected half of the result. The other half is filled with zero.

```c
// e.g., vreg<64xi32> → vreg<128xui16>
for (int i = 0; i < N; i++)
    dst[i] = 0;

if (part == LOWER) {
    for (int i = 0; i < N; i++)
        dst[i] = truncate(src[i]);
} else { // HIGHER
    for (int i = 0; i < N; i++)
        dst[N + i] = truncate(src[i]);
}
```

- **inputs:** `%src` is the wide source vector. `"LOWER"` and `"HIGHER"`
  select whether the narrowed payload lands in the lower or upper half.
- **outputs:** `%result` is the packed narrow vector.
- **constraints and limitations:** Packing is a narrowing conversion with
  truncation semantics. Current VPTO surface supports `i32/ui32 -> ui16` and
  `i16/ui16 -> ui8`.

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
    : !pto.vreg<64xf32>, f32, !pto.mask<G> -> !pto.mask<G>
%compacted = pto.vsqz %values, %pass_mask
    : !pto.vreg<64xf32>, !pto.mask<G> -> !pto.vreg<64xf32>

// Type narrowing via pack
%packed_i16 = pto.vpack %wide_i32, "LOWER"
  : !pto.vreg<64xi32> -> !pto.vreg<128xui16>
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

- **syntax:** `%result = pto.vlrelu %input, %alpha, %mask : !pto.vreg<NxT>, T, !pto.mask<G> -> !pto.vreg<NxT>`
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

- **syntax:** `%result = pto.vexpdiff %input, %max, "EVEN|ODD" : !pto.vreg<NxT>, !pto.vreg<NxT> -> !pto.vreg<Mxf32>`
- **A5 types:** input `f16` or `f32`, output `f32`
- **semantics:** Fused exp(x - max) for numerically stable softmax.

```c
for (int i = 0; i < N; i++)
    dst[i] = expf(src[i] - max[i]);
```

**Use case:** Softmax numerator computation with numerical stability.

- **inputs:** `%input` is the source vector and `%max` is the broadcasted
  subtraction term. `%part` selects `EVEN` or `ODD` for the
  underlying hardware contract.
- **outputs:** `%result` is the fused `exp(input - max)` vector with `f32`
  elements.
- **constraints and limitations:** Source vectors must be `f16` or `f32`, the
  result vector must be `f32`, and source/result storage width must match.

---

#### Fused Compute+Convert Ops

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

#### Extended Arithmetic

##### `pto.vmull`

- **syntax:** `%low, %high = pto.vmull %lhs, %rhs, %mask : !pto.vreg<NxT>, !pto.vreg<NxT>, !pto.mask<G> -> !pto.vreg<NxT>, !pto.vreg<NxT>`
- **A5 types:** i32/ui32 (native 32×32→64 widening multiply)
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

- **syntax:** `%result = pto.vmula %acc, %lhs, %rhs, %mask : !pto.vreg<NxT>, !pto.vreg<NxT>, !pto.vreg<NxT>, !pto.mask<G> -> !pto.vreg<NxT>`
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

#### Sorting Operations

##### `pto.vbitsort`

- **syntax:** `pto.vbitsort %dest, %src, %indices, %repeat_times : !pto.ptr<...>, !pto.ptr<...>, !pto.ptr<...>, index`
- **semantics:** Sort 32 region proposals by score and materialize sorted
  proposal records into `%dest`.
- **inputs:** `%dest` is the UB destination buffer. `%src` is the UB score
  buffer. `%indices` is the UB index buffer. `%repeat_times` is the repeat
  count; each repeat processes the next adjacent group of 32 scores and 32
  indices.
- **outputs:** This op writes UB memory and returns no SSA value. Each output
  record occupies 8 bytes: the upper 4 bytes hold the index and the lower
  4 bytes hold the score. For `f16` score forms, the score uses the lower
  2 bytes of that 4-byte score field and the upper 2 bytes are reserved.
- **constraints and limitations:** `%dest`, `%src`, and `%indices` MUST be
  UB-backed pointers and SHOULD satisfy the backend alignment contract expected
  by the A5 `VBS32` instruction. Scores are sorted in descending order, so the
  highest score is written to the lowest destination address. Equal-score ties
  preserve the earlier input proposal first. This is a UB helper, not a pure
  `vreg -> vreg` op.

---

##### `pto.vmrgsort4`

- **syntax:** `pto.vmrgsort4 %dest, %src0, %src1, %src2, %src3, %count, %config : !pto.ptr<T, ub>, !pto.ptr<T, ub>, !pto.ptr<T, ub>, !pto.ptr<T, ub>, !pto.ptr<T, ub>, i64, i64`
- **semantics:** Merge-sort 4 pre-sorted input vectors.
- **inputs:** `%dest` is the UB destination, `%src0..%src3` are the four
  pre-sorted UB inputs, `%count` is the number of valid elements, and `%config`
  is the operation control word.
- **outputs:** This op writes UB memory and returns no SSA value.
- **constraints and limitations:** Inputs MUST already be sorted according to
  the sort order encoded by `%config`.

---

#### Current Implementation Surface Summary

- `pto.vmull %lhs, %rhs, %mask : !pto.vreg<NxT>, !pto.vreg<NxT>, !pto.mask<G> -> !pto.vreg<NxT>, !pto.vreg<NxT>`
- `pto.vmula %acc, %lhs, %rhs, %mask : !pto.vreg<NxT>, !pto.vreg<NxT>, !pto.vreg<NxT>, !pto.mask<G> -> !pto.vreg<NxT>`
- `pto.vci %index {order = "ORDER"} : integer -> !pto.vreg<NxT>`
- `pto.vbitsort %dest, %src, %indices, %repeat_times : !pto.ptr<...>, !pto.ptr<...>, !pto.ptr<...>, index`
- `pto.vmrgsort4 %dest, %src0, %src1, %src2, %src3, %count, %config : !pto.ptr<...>, !pto.ptr<...>, !pto.ptr<...>, !pto.ptr<...>, !pto.ptr<...>, i64, i64`

---

#### Typical Usage

```mlir
// Softmax with fused expdiff
%max_broadcast = pto.vlds %ub_max[%c0] {dist = "BRC"} : !pto.ptr<f32, ub> -> !pto.vreg<64xf32>
%exp_stable = pto.vexpdiff %logits, %max_broadcast : !pto.vreg<64xf32>, !pto.vreg<64xf32> -> !pto.vreg<64xf32>

// Leaky ReLU activation
%activated = pto.vlrelu %linear_out, %alpha_scalar, %mask : !pto.vreg<64xf32>, f32, !pto.mask<G> -> !pto.vreg<64xf32>

// Generate indices for argsort
%indices = pto.vci %c0 {order = "ASC"} : i32 -> !pto.vreg<64xi32>
```

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
| `scf.for` | counted loops and loop-carried values | common structured counted loop form |
| `scf.if` | structured conditional execution | may yield values or act as side-effect-only branch |
| `scf.yield` | region terminator for `for` / `if` / `while` bodies | carries loop or branch results |
| `scf.while` | break-like or stateful loops | useful for source-level structured control |
| `scf.condition` | loop-continue / loop-exit decision for `scf.while` | placed in the "before" region |

Ops such as `scf.execute_region`, `scf.forall`, or `scf.index_switch` are not part of the documented shared-dialect portion of the PTO micro Instruction surface here.

---

#### Current PTOAS Coverage

- `scf.for`, `scf.if`, and `scf.yield` are directly exercised in the shared-dialect PTO fixture and appear widely across PTO samples
- PTO synchronization and memory analyses explicitly reason about `scf.for`, `scf.if`, `scf.yield`, and `scf.while`
- `scf.while` and `scf.condition` appear in control-flow samples and are handled in PTO-to-EmitC control-flow lowering, but they are less broadly exercised than `for` / `if` on all backend paths

---

#### Typical Patterns

##### Counted Loop

```mlir
scf.for %i = %c0 to %c4 step %c1 {
  %offset = arith.muli %i, %c32 : index
  %mask = pto.pset_b32 "PAT_ALL" : !pto.mask<b32>
  %v = pto.vlds %ub[%offset] : !pto.ptr<f32, ub> -> !pto.vreg<64xf32>
  %abs = pto.vabs %v, %mask : !pto.vreg<64xf32>, !pto.mask<b32> -> !pto.vreg<64xf32>
  pto.vsts %abs, %ub_out[%offset], %mask : !pto.vreg<64xf32>, !pto.ptr<f32, ub>, !pto.mask<b32>
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
%max_bc = pto.vlds %ub_tmp[%c0] {dist = "BRC"} : !pto.ptr<f32, ub> -> !pto.vreg<64xf32>

// 2. exp(x - max) using fused op
%exp = pto.vexpdiff %logits, %max_bc, "ODD" : !pto.vreg<64xf32>, !pto.vreg<64xf32> -> !pto.vreg<64xf32>

// 3. Sum
%sum = pto.vcadd %exp, %mask : !pto.vreg<64xf32>, !pto.mask<b32> -> !pto.vreg<64xf32>
pto.vsts %sum, %ub_tmp[%c0], %mask : !pto.vreg<64xf32>, !pto.ptr<f32, ub>, !pto.mask<b32>
%sum_bc = pto.vlds %ub_tmp[%c0] {dist = "BRC"} : !pto.ptr<f32, ub> -> !pto.vreg<64xf32>

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
%prelu = pto.vprelu %input, %alpha_vec : !pto.vreg<64xf32>, !pto.vreg<64xf32> -> !pto.vreg<64xf32>

```

### Data Layout Conversion

```mlir
// AoS → SoA (deinterleave)
%x, %y = pto.vldsx2 %ub_xy[%offset], "DINTLV" : !pto.ptr<f32, ub>, index -> !pto.vreg<64xf32>, !pto.vreg<64xf32>

// SoA → AoS (interleave)
pto.vstsx2 %x, %y, %ub_xy[%offset], "INTLV", %all_mask : !pto.vreg<64xf32>, !pto.vreg<64xf32>, !pto.ptr<f32, ub>, index, !pto.mask<b32>
```

---

---

## Quick Reference by Category

### Memory Operations

| Operation | Group | Description |
|-----------|-------|-------------|
| GM→UB DMA | 2 | `pto.copy_gm_to_ubuf` |
| UB→GM DMA | 2 | `pto.copy_ubuf_to_gm` |
| UB→UB Copy | 2 | `pto.copy_ubuf_to_ubuf` |
| Contiguous Load | 3 | `pto.vlds` with `NORM` dist |
| Broadcast Load | 3 | `pto.vlds` with `BRC` family dist |
| Gather | 3 | `pto.vgather2`, `pto.vgatherb` |
| Contiguous Store | 3 | `pto.vsts` with `NORM` dist |
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
| Interleave/Deinterleave (not A5) | 12 | `pto.vintlvv2`, `pto.vdintlvv2` |

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
