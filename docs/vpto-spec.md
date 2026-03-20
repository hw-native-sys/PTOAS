# VPTO Spec

Updated: 2026-03-21

## Table Of Contents

- [Overview](#overview)
- [Getting Started](#getting-started)
- [Example: Abs](#example-abs)
- [Scope](#scope)
- [ISA Contract](#isa-contract)
- [Core Types](#core-types)
- [Address Space Conventions](#address-space-conventions)
- [Element Type Constraints](#element-type-constraints)
- [Special Types](#special-types)
- [Implemented String Constraints](#implemented-string-constraints)
- [__VEC_SCOPE__](#vec_scope)
- [Correspondence Categories](#correspondence-categories)
- [1. Sync And Buffer Control](#1-sync-and-buffer-control)
- [2. Copy Programming](#2-copy-programming)
- [3. Copy Transfers](#3-copy-transfers)
- [4. Vector, Predicate And Align Loads](#4-vector-predicate-and-align-loads)
- [5. Materialization And Predicate Construction](#5-materialization-and-predicate-construction)
- [6. Unary Vector Ops](#6-unary-vector-ops)
- [7. Binary Vector Ops](#7-binary-vector-ops)
- [8. Vec-Scalar Ops](#8-vec-scalar-ops)
- [9. Carry, Compare And Select](#9-carry-compare-and-select)
- [10. Pairing And Interleave](#10-pairing-and-interleave)
- [11. Conversion, Index And Sort](#11-conversion-index-and-sort)
- [12. Extended Arithmetic](#12-extended-arithmetic)
- [13. Stateless Stores](#13-stateless-stores)
- [14. Stateful Store Ops](#14-stateful-store-ops)

## Overview

This document defines the Vector PTO (VPTO) Intermediate Representation (IR), a
compiler-internal and externally facing specification designed to represent
vector compute kernels within the PTO architecture. Much like NVVM provides a
robust IR for GPU architectures, VPTO serves as the direct bridge between
high-level programming models and the underlying hardware ISA, providing a
precise, low-level representation of vector workloads explicitly designed for
the Ascend 950 architecture.

### PTO Vector ISA Background

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

### Relationship to CCE

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

### Intended Audience

This document is written for compiler engineers, library writers, and advanced
performance architects. We expect the reader to have a working understanding of
modern compiler infrastructure, specifically MLIR, the principles of Static
Single Assignment (SSA) form, and a deep understanding of the vector-processing
capabilities of the Ascend 950 architecture.

## Getting Started

The Vector PTO (VPTO) IR is architected as a performance-critical layer within the compiler stack, specifically designed to exploit the **Decoupled Access-Execute** (DAE) nature of the Ascend 950 hardware.

### Hardware Pipeline Modeling
The IR is structured to mirror the three primary hardware pipelines of the Ascend 950 architecture. Correct VPTO authoring requires managing the interaction between these asynchronous units:

**MTE2** (Memory Transfer Engine - Inbound): Responsible for moving data from Global Memory (GM) to the Unified Buffer (UB).

**Vector Core** (Computation): The primary engine for executing SIMD operations on data stored in UB.

**MTE3** (Memory Transfer Engine - Outbound): Responsible for moving processed data from UB back to GM.

### Memory and Synchronization Model
VPTO enforces a strict memory hierarchy. The Unified Buffer (UB) is the only valid operand source for vector compute instructions. Consequently, the architecture of a VPTO program is defined by the explicit management of data movement:

**Address Space Isolation**: The IR uses LLVM pointer address spaces to distinguish between GM (!llvm.ptr<1>) and UB (!llvm.ptr<6>). The verifier ensures that no compute operation attempts to access GM directly.

**Event-Based Synchronization**: Because the MTE and Vector pipelines operate asynchronously, VPTO utilizes a Flag/Event mechanism. Developers must explicitly insert set_flag and wait_flag operations to resolve Read-After-Write (RAW) and Write-After-Read (WAR) hazards between memory staging and computation.

### Execution Scopes
The IR introduces the concept of the **Vector Function** ({llvm.loop.aivector_scope}). This architectural boundary identifies regions of code where the Vector Core's SIMD capabilities are fully engaged. Inside this scope, the IR provides high-granularity control over vector registers (vreg), predicates (mask), and alignment states (align).

## Example: Abs

Example file:
[build/vpto-doc-abs/Abs/abs-pto.cpp](/data/mouliangyu/projects/github.com/zhangstevenunity/PTOAS/build/vpto-doc-abs/Abs/abs-pto.cpp)

Representative excerpt:

```mlir
pto.vset_loop2_stride_outtoub %c4096_i64, %c4096_i64 : i64, i64
pto.vset_loop1_stride_outtoub %c4096_i64, %c4096_i64 : i64, i64
pto.vset_loop_size_outtoub %c1_i64, %c1_i64 : i64, i64
pto.vcopy_gm_to_ubuf %7, %2, %3, %3, %c0_i64, %c32_i64, %4, %c0_i64, %c0_i64, %c0_i64, %c128_i64, %c128_i64
    {data_select_bit = false, layout = "nd", ub_pad = false}
    : !llvm.ptr<1>, !llvm.ptr<6>, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64

pto.vset_flag["PIPE_MTE2", "PIPE_V", "EVENT_ID0"]
pto.vwait_flag["PIPE_MTE2", "PIPE_V", "EVENT_ID0"]

scf.for %dummy = %c0 to %c1 step %c1 {
  scf.for %lane = %c0 to %9 step %c64 {
    %v = pto.vlds %2[%lane] : !llvm.ptr<6> -> !pto.vreg<64xf32>
    %abs = pto.vabs %v : !pto.vreg<64xf32> -> !pto.vreg<64xf32>
    pto.vsts %abs, %8[%lane] : !pto.vreg<64xf32>, !llvm.ptr<6>
  }
} {llvm.loop.aivector_scope}

pto.vset_flag["PIPE_V", "PIPE_MTE3", "EVENT_ID0"]
pto.vwait_flag["PIPE_V", "PIPE_MTE3", "EVENT_ID0"]
pto.vset_loop_size_ubtoout %c1_i64, %c1_i64 : i64, i64
pto.vset_loop1_stride_ubtoout %c4096_i64, %c4096_i64 : i64, i64
pto.vset_loop2_stride_ubtoout %c4096_i64, %c4096_i64 : i64, i64
pto.vcopy_ubuf_to_gm %8, %14, %3, %3, %c0_i64, %c32_i64, %4, %c0_i64, %c128_i64, %c128_i64
    {layout = "nd"}
    : !llvm.ptr<6>, !llvm.ptr<1>, i64, i64, i64, i64, i64, i64, i64, i64
```

## Scope

This document is the interface specification for the `mlir::pto` dialect.

It only describes:

- operation names
- operand and result lists
- operand and result types
- important attributes
- corresponding CCE builtin or CCE wrapper family

It does not describe lowering strategy.

## ISA Contract

VPTO is an ISA-level contract without an encoding layer. Each op below carries
the architectural behavior of one ISA instruction or ISA family, while making
hidden register updates, predicate flow, and buffer state explicit in SSA form.

Contract policy:

- `- ISA family:` names the architectural instruction family that defines the
  operation semantics.
- `- semantics:` defines the architecturally visible behavior after removing the
  encoding and implicit register-mutation details.
- `- operand roles:` explains the meaning of each SSA operand, result, and
  control token.
- Chapter-level assertions and exceptions below are normative unless an op
  section narrows them further.

Naming note:

- This spec uses `pto.v*` headings for the exposed VPTO contract.
- Some underlying A5VM op definitions omit that extra `v` prefix for control,
  predicate, and copy helpers; the ISA behavior is unchanged.

## Core Types

- `vreg<T>`: `!pto.vreg<NxT>`
  Fixed-width VPTO vector type with total width exactly 256 bytes.
- `mask`: `!pto.mask`
  Opaque predicate-register type. The element granularity is carried by the producing or consuming opcode family (`*_b8`, `*_b16`, `*_b32`), not by a type parameter on `!pto.mask` itself.
- `align`: `!pto.align`
- `buf`: buffer-like LLVM pointer type accepted by the dialect
- `idx`: `index`
- `i32`: `i32`
- `i64`: `i64`

Type parameter conventions used below:

- `!pto.vreg<NxT>`:
  `N` is the lane count, `T` is the element type, and `N * bitwidth(T) = 2048`
- `!llvm.ptr<AS>`:
  `AS` is the LLVM address space number

## Address Space Conventions

The table below defines the address-space interpretation used by this spec and by the current dialect implementation.

| `AS` | PTO mnemonic | Working interpretation in this spec | Status |
|------|--------------|-------------------------------------|--------|
| `0` | `Zero` | Default / unspecified pointer space; treated as GM-like by the current verifier rules | Normative in this spec |
| `1` | `GM` | Global Memory (GM) | Normative in this spec |
| `2` | `MAT` | Matrix / L1-related storage | Normative in this spec |
| `3` | `LEFT` | Left matrix buffer / L0A-related storage | Normative in this spec |
| `4` | `RIGHT` | Right matrix buffer / L0B-related storage | Normative in this spec |
| `5` | `ACC` | Accumulator / L0C-related storage | Normative in this spec |
| `6` | `VEC` | Unified Buffer (UB) / vector buffer | Normative in this spec |
| `7` | `BIAS` | Bias buffer | Normative in this spec |
| `8` | `SCALING` | Scaling buffer | Normative in this spec |

- Current verifier rule: `!llvm.ptr<0>` and `!llvm.ptr<1>` are treated as GM-like, while `!llvm.ptr<6>` is treated as UB-like.
- External authors should keep the raw numeric LLVM address space in IR and use the symbolic names in this table as the explanatory meaning of those numeric values.

## Element Type Constraints

This section defines how placeholders such as `T`, `T0`, `T1`, and `I` should
be read throughout the spec.

- General vector rule:
  `!pto.vreg<NxT>` requires `T` to be an integer or floating-point element
  type, and `N * bitwidth(T) = 2048`.
- `T`:
  General vector element type accepted by the mapped ISA family. In the current tree this means integer lanes and floating-point lanes such as `i8`, `i16`, `i32`, `i64`, `f16`, `bf16`, and `f32`, subject to the narrower legality of each individual op family.
- `T0`, `T1`:
  Source and result element types for conversion ops. Legal pairs are exactly the pairs implemented by the ISA conversion families `VCVTFI`, `VCVTFF`, `VCVTIF`, `VCVTII`, and `VTRC`; VPTO does not treat `pto.vcvt` as an arbitrary bitcast.
- `I`:
  Integer element type used for offsets, indices, lane selectors, and permutation inputs. Gather, scatter, index-generation, and lane-selection ops require integer vectors; scalar offsets use `index`, `i32`, or `i64` exactly as shown in the op syntax.
- Family-specific exceptions:
  Predicate families use `!pto.mask` rather than `!pto.vreg`; `pto.vmull` returns split widened results; stateful store ops thread `!pto.align` and pointer/index state explicitly; and copy-programming ops are configuration side effects rather than value-producing vector instructions.

## Special Types

### `!pto.mask`

`!pto.mask` models an A5 predicate register, not an integer vector.

Mask data-type expression:

- `!pto.mask` is intentionally unparameterized. Predicate granularity is implied by the op family that creates or consumes it, so `pset_b8`, `pset_b16`, and `pset_b32` all return the same abstract mask type while preserving their ISA-level granularity in the op name.

Use it when an operation needs per-lane enable/disable state.

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

Example:

```mlir
%mask = pto.vcmp %lhs, %rhs, %seed, "lt" : !pto.vreg<64xf32>, !pto.vreg<64xf32>, !pto.mask -> !pto.mask
%out = pto.vsel %x, %y, %mask : !pto.vreg<64xf32>, !pto.vreg<64xf32>, !pto.mask -> !pto.vreg<64xf32>
```

### `!pto.align`

`!pto.align` models the A5 vector-align carrier state. It is not payload data.

Use it when an operation needs explicit align-state threading in SSA form.

- producers:
  `pto.vldas`,
  `pto.vpstu`,
  `pto.vstu`,
  `pto.vstus`,
  `pto.vstur`
- consumers:
  `pto.vldus`,
  `pto.vsta`,
  `pto.vstas`,
  `pto.vstar`,
  `pto.vpstu`,
  `pto.vstu`,
  `pto.vstus`,
  `pto.vstur`

Example:

```mlir
%align = pto.vldas %ub[%c0] : !llvm.ptr<6> -> !pto.align
%vec = pto.vldus %align, %ub[%c64] : !pto.align, !llvm.ptr<6> -> !pto.vreg<64xf32>
```

Template placeholder conventions used below:

- `"SRC_PIPE"`, `"DST_PIPE"`:
  string literals such as `"PIPE_MTE2"`, `"PIPE_V"`, `"PIPE_MTE3"`
- `"EVENT_ID"`:
  string literal such as `"EVENT_ID0"`
- `"LAYOUT"`:
  string literal layout selector, for example `"nd"`
- `"DIST"`:
  string literal distribution selector carried by the op
- `"POSITION"`:
  string literal lane-position selector used by `vdup`
- `"MODE"`:
  string literal mode selector used by stateful store / multiply-accumulate ops
- `"ROUND_MODE"`:
  string literal rounding-mode selector
- `"SAT_MODE"`:
  string literal saturation selector
- `"PART_MODE"`:
  string literal half/part selector
- `"ORDER"`:
  string literal order selector used by `vci`
- `"CMP_MODE"`:
  string literal compare predicate selector
- `"PAT_*"`:
  predicate pattern literal accepted by the corresponding predicate op
- `T|!pto.vreg<NxT>`:
  either a scalar `T` or a vector operand `!pto.vreg<NxT>`, matching the op verifier

## Implemented String Constraints

This section records string-valued operands and attributes that are already
checked by the current verifier implementation.

If a token is not listed here, the current dialect usually only requires a
non-empty string or leaves the token unconstrained for now.

### Predicate Patterns

Used by:
`pto.vpset_b8`, `pto.vpset_b16`, `pto.vpset_b32`,
`pto.vpge_b8`, `pto.vpge_b16`, `pto.vpge_b32`

- allowed values:
  `PAT_ALL | PAT_VL1 | PAT_VL2 | PAT_VL3 | PAT_VL4 | PAT_VL8 | PAT_VL16 | PAT_VL32 | PAT_VL64 | PAT_VL128 | PAT_M3 | PAT_M4 | PAT_H | PAT_Q | PAT_ALLF`

### Distribution Tokens

Used by `pto.vlds`:

- allowed values:
  `NORM | BLK | DINTLV_B32 | UNPK_B16`

Used by `pto.vpld`, `pto.vpldi`:

- allowed values:
  `NORM | US | DS`

Used by `pto.vpst`, `pto.vpsti`:

- allowed values:
  `NORM | PK`

Used by `pto.vldx2`:

- allowed values:
  `DINTLV_B8 | DINTLV_B16 | DINTLV_B32 | BDINTLV`

Used by `pto.vstx2`:

- allowed values:
  `INTLV_B8 | INTLV_B16 | INTLV_B32`

### Stride Tokens

Used by `pto.vsld`, `pto.vsst`:

- allowed values:
  `STRIDE_S3_B16 | STRIDE_S4_B64 | STRIDE_S8_B32 | STRIDE_S2_B64 | STRIDE_VSST_S8_B16`

### Compare Modes

Used by `pto.vcmp`, `pto.vcmps`:

- allowed values:
  `eq | ne | lt | le | gt | ge`

### Part Tokens

Used by `pto.vintlvv2`, `pto.vdintlvv2`:

- allowed values:
  `LOWER | HIGHER`

Current restricted subset:

- `pto.vppack`: only `LOWER`
- `pto.vpunpack`: only `LOWER`

### Mode Tokens

Used by `pto.vmula`:

- allowed values:
  `MODE_ZEROING | MODE_UNKNOWN | MODE_MERGING`

Used by `pto.vstu`, `pto.vstus`, `pto.vstur`:

- allowed values:
  `POST_UPDATE | NO_POST_UPDATE`

### Conversion Control Tokens

Used by `pto.vcvt.round_mode`:

- allowed values:
  `ROUND_R | ROUND_A | ROUND_F | ROUND_C | ROUND_Z | ROUND_O`

Used by `pto.vcvt.sat`:

- allowed values:
  `RS_ENABLE | RS_DISABLE`

Used by `pto.vcvt.part`:

- allowed values:
  `PART_EVEN | PART_ODD`

### Not Yet Enumerated In Verifier

The following placeholders appear in syntax templates but are not yet fully
enumerated by the verifier:

- `"LAYOUT"`
- `"POSITION"`
- `"ORDER"`
- `"SRC_PIPE"`, `"DST_PIPE"`, `"EVENT_ID"`

### `LAYOUT`

- Current repo-defined layout spellings are `nd`, `dn`, and `nz`.
- Copy ops preserve the layout token as part of the transfer contract.
- The verifier does not yet exhaustively cross-check every layout-sensitive combination, so producers must only emit layout values that the selected copy helper or backend path actually implements.

### `POSITION`

- `POSITION` selects which `VDUP*` source position is duplicated when the input is a vector.
- The current verifier checks type compatibility but does not enumerate a closed token set, so this field is an implementation-defined token that must be preserved exactly by translators.

### `ORDER`

- `ORDER` selects the lane-index generation order for `VCI`.
- The currently documented token is `INC_ORDER`, which produces monotonic increasing lane indices.
- The current verifier does not enforce a closed enum for this field, so any alternative order token must be introduced together with matching lowering support.

### `SRC_PIPE` / `DST_PIPE`

- Legal pipe names in the current tree are `PIPE_S`, `PIPE_V`, `PIPE_M`, `PIPE_MTE1`, `PIPE_MTE2`, `PIPE_MTE3`, `PIPE_ALL`, `PIPE_MTE4`, `PIPE_MTE5`, `PIPE_V2`, `PIPE_FIX`, `VIRTUAL_PIPE_MTE2_L1A`, and `VIRTUAL_PIPE_MTE2_L1B`.
- `SRC_PIPE` names the producer side of the dependency and `DST_PIPE` names the consumer side.
- A `wait_flag` must use the same source pipe, destination pipe, and event id triplet that the corresponding `set_flag` published.

### `EVENT_ID`

- Legal event identifiers in the current tree are `EVENT_ID0` through `EVENT_ID7`.
- The event id is not meaningful by itself; it is interpreted together with the `(SRC_PIPE, DST_PIPE)` pair.
- Producer and consumer sides must agree on the entire triplet `(SRC_PIPE, DST_PIPE, EVENT_ID)` for synchronization to be well formed.

## Architectural Assertions And Exceptions

The following rules are ISA-level requirements that VPTO preserves even though
it does not encode the original instruction words.

Architectural assertions:

- Vector compute, gather, scatter, predicate-load, and predicate-store families operate on UB-backed storage unless an op section states otherwise.
- Distribution tokens, stride tokens, part selectors, and predicate patterns are semantically significant; changing them changes the operation, not just the encoding.
- Alignment requirements are part of the contract. Violating the distribution-specific address-alignment rule raises an exception.
- Stateful align-register behavior is explicit in VPTO through `!pto.align` values and state-threading results, not by implicit mutation of a hidden architectural register.
- Copy helper families preserve layout, burst geometry, and stride semantics as architecturally visible behavior.

Architectural exceptions:

- UB access overflow raises an exception for UB reads or writes that exceed the permitted UB address range.
- Load and store families raise an exception when ISA-required alignment constraints are violated.
- Memory-to-memory sorter and filter families require their ISA alignment and non-overlap constraints; violating those constraints raises an exception.
- `INF` and `NaN` in source operands raise an exception for the arithmetic, conversion, and sort families in this spec wherever the ISA defines those inputs as exceptional.
- Division by `+0` or `-0` raises an exception for `pto.vdiv` and `pto.vrec`; the same ISA rule also applies to reciprocal-square-root families not currently exposed here.
- Negative input raises an exception for `pto.vln` and `pto.vsqrt`; the same ISA rule also applies to reciprocal-square-root families not currently exposed here.
- Certain conversions from negative source values to unsigned integer destinations, including forms such as `f16 -> u8` and `s32 -> u16/u8`, raise an exception rather than silently wrapping.

## __VEC_SCOPE__

`__VEC_SCOPE__` is not an `pto` op.

It must be represented as:

```mlir
%c0 = arith.constant 0 : index
%c1 = arith.constant 1 : index
scf.for %dummy = %c0 to %c1 step %c1 {
  // vector-scope body
} {llvm.loop.aivector_scope}
```

This is the dialect-level representation of the A5 vector-scope loop.

## Correspondence Categories

- `direct builtin`
  The op maps naturally to one CCE builtin family, usually `__builtin_cce_<name>_*`.
- `wrapper family`
  The op corresponds to a CCE wrapper family, but the wrapper may dispatch to
  multiple builtin spellings depending on type, architecture, or mode.

Builtin naming policy in this document:

- if a visible CCE intrinsic is declared as
  `clang_builtin_alias(__builtin_cce_...)`, the spec lists the builtin name
  explicitly
- if PTO A5 code calls a wrapper function that internally composes several
  intrinsics or builtins, the spec lists both the wrapper name and the visible
  builtin family

## 1. Sync And Buffer Control

### `pto.vset_flag`

- syntax:
  `pto.vset_flag["SRC_PIPE", "DST_PIPE", "EVENT_ID"]`
- operand roles:
  `"SRC_PIPE"` names the producer pipeline, `"DST_PIPE"` names the consumer pipeline, and `"EVENT_ID"` names the event channel being published.
- ISA family:
  `SET_FLAG`
- semantics:
  Publishes `EVENT_ID` from `SRC_PIPE` to `DST_PIPE` so later waits can order the asynchronous pipelines.
- CCE correspondence:
  `set_flag(pipe_t, pipe_t, event_t|uint64_t)`
  `__builtin_cce_set_flag`
  PTO token path:
  `__pto_set_flag`
  `__builtin_cce_tile_set_flag`

### `pto.vwait_flag`

- syntax:
  `pto.vwait_flag["SRC_PIPE", "DST_PIPE", "EVENT_ID"]`
- operand roles:
  `"SRC_PIPE"` names the producer pipeline, `"DST_PIPE"` names the consumer pipeline, and `"EVENT_ID"` names the event channel being waited on.
- ISA family:
  `WAIT_FLAG`
- semantics:
  Stalls the consumer side until the matching `(SRC_PIPE, DST_PIPE, EVENT_ID)` event has been observed.
- CCE correspondence:
  `wait_flag(pipe_t, pipe_t, event_t|uint64_t)`
  `__builtin_cce_wait_flag`
  PTO token path:
  `__pto_wait_flag`
  `__builtin_cce_tile_wait_flag`

### `pto.vpipe_barrier`

- syntax:
  `pto.vpipe_barrier "PIPE_*"`
- operand roles:
  `"PIPE_*"` selects the pipeline whose in-order execution is being fenced.
- ISA family:
  `PIPE_BARRIER`
- semantics:
  Models a same-pipe execution barrier; later operations on `PIPE_*` cannot overtake earlier ones on that pipeline.
- CCE correspondence:
  `pipe_barrier(pipe_t)`
  `__builtin_cce_pipe_barrier`

### `pto.vget_buf`

- syntax:
  `pto.vget_buf "PIPE_*", %buf_id, %mode : i64, i64`
- operand roles:
  `"PIPE_*"` selects the owning pipeline, `%buf_id` is the buffer identifier being requested, and `%mode` carries the hardware acquisition mode.
- ISA family:
  `GET_BUF`
- semantics:
  Models hardware buffer-token acquisition on the selected pipe; the op reserves the identified buffer slot or mode-controlled token before later use.
- CCE correspondence:
  `get_buf(pipe_t, uint8_t|uint64_t, bool)`
  `__builtin_cce_get_buf`

### `pto.vrls_buf`

- syntax:
  `pto.vrls_buf "PIPE_*", %buf_id, %mode : i64, i64`
- operand roles:
  `"PIPE_*"` selects the owning pipeline, `%buf_id` is the buffer identifier being released, and `%mode` carries the hardware release mode.
- ISA family:
  `RLS_BUF`
- semantics:
  Models hardware buffer-token release on the selected pipe; the op returns a previously acquired buffer slot or mode-controlled token to the implementation.
- CCE correspondence:
  `rls_buf(pipe_t, uint8_t|uint64_t, bool)`
  `__builtin_cce_rls_buf`

## 2. Copy Programming

### `pto.vset_loop2_stride_outtoub`

- syntax:
  `pto.vset_loop2_stride_outtoub %first, %second : i64, i64`
- operand roles:
  `%first` and `%second` are the two i64 configuration fields consumed by the corresponding copy-programming register pair.
- ISA family:
  `SET_LOOP2_STRIDE_OUTTOUB`
- semantics:
  Models GM-to-UB copy programming state; sets the outer-loop stride consumed by the next outbound-to-UB transfer sequence.
- CCE correspondence:
  `set_loop2_stride_outtoub(uint64_t)`
  `__builtin_cce_set_loop2_stride_outtoub`

### `pto.vset_loop1_stride_outtoub`

- syntax:
  `pto.vset_loop1_stride_outtoub %first, %second : i64, i64`
- operand roles:
  `%first` and `%second` are the two i64 configuration fields consumed by the corresponding copy-programming register pair.
- ISA family:
  `SET_LOOP1_STRIDE_OUTTOUB`
- semantics:
  Models GM-to-UB copy programming state; sets the inner-loop stride consumed by the next outbound-to-UB transfer sequence.
- CCE correspondence:
  `set_loop1_stride_outtoub(uint64_t)`
  `__builtin_cce_set_loop1_stride_outtoub`

### `pto.vset_loop_size_outtoub`

- syntax:
  `pto.vset_loop_size_outtoub %first, %second : i64, i64`
- operand roles:
  `%first` and `%second` are the two i64 configuration fields consumed by the corresponding copy-programming register pair.
- ISA family:
  `SET_LOOP_SIZE_OUTTOUB`
- semantics:
  Models GM-to-UB copy programming state; sets the loop extents consumed by the next outbound-to-UB transfer sequence.
- CCE correspondence:
  `set_loop_size_outtoub(uint64_t)`
  `__builtin_cce_set_loop_size_outtoub`

### `pto.vset_loop2_stride_ubtoout`

- syntax:
  `pto.vset_loop2_stride_ubtoout %first, %second : i64, i64`
- operand roles:
  `%first` and `%second` are the two i64 configuration fields consumed by the corresponding copy-programming register pair.
- ISA family:
  `SET_LOOP2_STRIDE_UBTOOUT`
- semantics:
  Models UB-to-GM copy programming state; sets the outer-loop stride consumed by the next UB-to-outbound transfer sequence.
- CCE correspondence:
  `set_loop2_stride_ubtoout(uint64_t)`
  `__builtin_cce_set_loop2_stride_ubtoout`

### `pto.vset_loop1_stride_ubtoout`

- syntax:
  `pto.vset_loop1_stride_ubtoout %first, %second : i64, i64`
- operand roles:
  `%first` and `%second` are the two i64 configuration fields consumed by the corresponding copy-programming register pair.
- ISA family:
  `SET_LOOP1_STRIDE_UBTOOUT`
- semantics:
  Models UB-to-GM copy programming state; sets the inner-loop stride consumed by the next UB-to-outbound transfer sequence.
- CCE correspondence:
  `set_loop1_stride_ubtoout(uint64_t)`
  `__builtin_cce_set_loop1_stride_ubtoout`

### `pto.vset_loop_size_ubtoout`

- syntax:
  `pto.vset_loop_size_ubtoout %first, %second : i64, i64`
- operand roles:
  `%first` and `%second` are the two i64 configuration fields consumed by the corresponding copy-programming register pair.
- ISA family:
  `SET_LOOP_SIZE_UBTOOUT`
- semantics:
  Models UB-to-GM copy programming state; sets the loop extents consumed by the next UB-to-outbound transfer sequence.
- CCE correspondence:
  `set_loop_size_ubtoout(uint64_t)`
  `__builtin_cce_set_loop_size_ubtoout`

## 3. Copy Transfers

### `pto.vcopy_gm_to_ubuf`

- syntax:
  `pto.vcopy_gm_to_ubuf %source, %destination, %valid_rows, %valid_cols, %sid, %n_burst, %len_burst, %left_padding_count, %right_padding_count, %l2_cache_ctl, %gm_stride, %ub_stride {layout = "LAYOUT", data_select_bit = true|false, ub_pad = true|false} : !llvm.ptr<AS>, !llvm.ptr<AS>, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64`
- operand roles:
  `%source` is the GM base pointer, `%destination` is the UB base pointer, `%valid_rows` and `%valid_cols` describe the logical tile extent, `%sid` is the stream or source identifier, `%n_burst` and `%len_burst` describe the burst geometry, `%left_padding_count` and `%right_padding_count` describe edge padding, `%l2_cache_ctl` carries cache control bits, `%gm_stride` and `%ub_stride` are per-burst strides, and `"LAYOUT"` records the transfer layout token.
- ISA family:
  `GM->UB copy helper family`
- semantics:
  Copies a 2-D burst tile with optional padding and layout metadata into UB.
- CCE correspondence:
  `copy_gm_to_ubuf(...)`
  PTO A5 path commonly uses `copy_gm_to_ubuf_align_v2(...)`
  `__builtin_cce_copy_gm_to_ubuf_align_v2`
  composed loop intrinsics:
  `__builtin_cce_set_loop2_stride_outtoub`
  `__builtin_cce_set_loop1_stride_outtoub`
  `__builtin_cce_set_loop_size_outtoub`

### `pto.vcopy_ubuf_to_ubuf`

- syntax:
  `pto.vcopy_ubuf_to_ubuf %source, %destination, %sid, %n_burst, %len_burst, %src_stride, %dst_stride : !llvm.ptr<AS>, !llvm.ptr<AS>, i64, i64, i64, i64, i64`
- operand roles:
  `%source` and `%destination` are UB base pointers, `%sid` is the stream identifier, `%n_burst` and `%len_burst` describe the burst geometry, and `%src_stride` and `%dst_stride` are the per-burst source and destination strides.
- ISA family:
  `UB->UB copy helper family`
- semantics:
  Copies data between two UB-backed buffers using the stated burst and stride parameters.
- CCE correspondence:
  `copy_ubuf_to_ubuf(...)`
  `__builtin_cce_copy_ubuf_to_ubuf`

### `pto.vcopy_ubuf_to_gm`

- syntax:
  `pto.vcopy_ubuf_to_gm %source, %destination, %valid_rows, %valid_cols, %sid, %n_burst, %len_burst, %reserved, %burst_dst_stride, %burst_src_stride {layout = "LAYOUT"} : !llvm.ptr<AS>, !llvm.ptr<AS>, i64, i64, i64, i64, i64, i64, i64, i64`
- operand roles:
  `%source` is the UB base pointer, `%destination` is the GM base pointer, `%valid_rows` and `%valid_cols` describe the logical tile extent, `%sid` is the stream identifier, `%n_burst` and `%len_burst` describe the burst geometry, `%reserved` is the ISA-reserved field carried by the helper path, `%burst_dst_stride` and `%burst_src_stride` are the per-burst strides, and `"LAYOUT"` records the transfer layout token.
- ISA family:
  `UB->GM copy helper family`
- semantics:
  Writes a 2-D burst tile from UB back to GM.
- CCE correspondence:
  `copy_ubuf_to_gm(...)`
  PTO A5 path commonly uses `copy_ubuf_to_gm_align_v2(...)`
  `__builtin_cce_copy_ubuf_to_gm_align_v2`
  composed loop intrinsics:
  `__builtin_cce_set_loop2_stride_ubtoout`
  `__builtin_cce_set_loop1_stride_ubtoout`
  `__builtin_cce_set_loop_size_ubtoout`

## 4. Vector, Predicate And Align Loads

ISA assertions for this family:

- These ops read from UB-backed storage; GM-backed sources are not valid vector-load operands in VPTO.
- Alignment requirements are part of the ISA contract and depend on the selected distribution token.
- `pto.vldas` initializes align state for subsequent unaligned accesses; `pto.vldus` assumes a valid align chain for the same logical stream.
- ISA forms that post-update a shared register are represented in VPTO by explicit SSA operands or by state-threading ops rather than by hidden base-pointer mutation.


### `pto.vlds`

- syntax:
  `%result = pto.vlds %source[%offset] {dist = "DIST"} : !llvm.ptr<AS> -> !pto.vreg<NxT>`
- operand roles:
  `%source` is the UB base pointer, `%offset` is the load displacement from that base, `"DIST"` selects the ISA distribution mode, and `%result` is the loaded vector.
- ISA family:
  `VLD` / `VLDS`
- semantics:
  Reads one vector from UB at `source + offset` using the aligned-load form selected by `DIST`. The distribution token determines the lane arrangement and alignment requirement of the access, and any ISA-defined base update is represented outside this op rather than as an implicit side effect.
- CCE correspondence:
  `vld(...)`, `vlds(...)`
  `__builtin_cce_vldsx1_*`
  related extended families:
  `__builtin_cce_vldix1_*`, `__builtin_cce_vldsx1_post_*`

### `pto.vldas`

- syntax:
  `%result = pto.vldas %source[%offset] : !llvm.ptr<AS> -> !pto.align`
- operand roles:
  `%source` is the UB base pointer, `%offset` is the align-initialization displacement, and `%result` is the produced align state.
- ISA family:
  `VLDAS`
- semantics:
  Initializes align state for a subsequent unaligned load stream. The seeded state is derived from the aligned base corresponding to `source + offset`, with the ISA-required low address bits removed.
- CCE correspondence:
  `vldas(...)`
  `__builtin_cce_vldas_*`

### `pto.vldus`

- syntax:
  `%result = pto.vldus %align, %source[%offset] : !pto.align, !llvm.ptr<AS> -> !pto.vreg<NxT>`
- operand roles:
  `%align` is the incoming align state, `%source` is the UB base pointer, `%offset` is the load displacement, and `%result` is the assembled vector result.
- ISA family:
  `VLDUS`
- semantics:
  Loads one unaligned vector by combining `%align` with the aligned data fetched from `source + offset`. `%align` must come from the same logical align stream, and the returned vector is the ISA contiguous lane sequence assembled across the alignment boundary.
- CCE correspondence:
  `vldus(...)`
  `__builtin_cce_vldus_*`, `__builtin_cce_vldus_post_*`

### `pto.vplds`

- syntax:
  `%result = pto.vplds %source[%offset] {dist = "DIST"} : !llvm.ptr<AS> -> !pto.mask`
- operand roles:
  `%source` is the UB base pointer, `%offset` is the load displacement, `"DIST"` selects the predicate-load distribution, and `%result` is the loaded predicate.
- ISA family:
  `PLDS`
- semantics:
  Loads predicate state from UB at `source + offset` using the selected predicate-load distribution.
- CCE correspondence:
  `plds(...)`
  `__builtin_cce_plds_b8`

### `pto.vpld`

- syntax:
  `%result = pto.vpld %source[%offset], "DIST" : !llvm.ptr<AS>, index -> !pto.mask`
- operand roles:
  `%source` is the UB base pointer, `%offset` is the index-style displacement, `"DIST"` selects the predicate-load distribution, and `%result` is the loaded predicate.
- ISA family:
  `PLD`
- semantics:
  Loads predicate state from UB using an explicit index offset and predicate-load distribution token.
- CCE correspondence:
  `pld(...)`
  `__builtin_cce_pld_b8`

### `pto.vpldi`

- syntax:
  `%result = pto.vpldi %source, %offset, "DIST" : !llvm.ptr<AS>, i32 -> !pto.mask`
- operand roles:
  `%source` is the UB base pointer, `%offset` is the immediate-style scalar displacement, `"DIST"` selects the predicate-load distribution, and `%result` is the loaded predicate.
- ISA family:
  `PLDI`
- semantics:
  Loads predicate state from UB using an immediate-style scalar offset and predicate-load distribution token.
- CCE correspondence:
  `pldi(...)`
  `__builtin_cce_pldi_b8`, `__builtin_cce_pldi_post_b8`

### `pto.vldx2`

- syntax:
  `%low, %high = pto.vldx2 %source[%offset], "DIST" : !llvm.ptr<AS>, index -> !pto.vreg<NxT>, !pto.vreg<NxT>`
- operand roles:
  `%source` is the UB base pointer, `%offset` is the displacement, `"DIST"` selects the x2 load distribution token, and `%low` and `%high` are the two produced vector results.
- ISA family:
  `Dual-result aligned-load distributions for VLD variants`
- semantics:
  Reads one UB vector stream at `source + offset` and splits the result into `%low` and `%high` according to the x2 distribution selected by `DIST`. The distribution token defines how lanes are deinterleaved between the two returned vectors.
- CCE correspondence:
  `vld(...)`
  `__builtin_cce_vldx2_*`

### `pto.vgather2`

- syntax:
  `%result = pto.vgather2 %source, %offsets, %active_lanes : !llvm.ptr<AS>, !pto.vreg<NxI>, index -> !pto.vreg<NxT>`
- operand roles:
  `%source` is the UB base pointer, `%offsets` is the per-lane offset vector, `%active_lanes` bounds how many lanes participate, and `%result` is the gathered vector.
- ISA family:
  `VGATHER2`
- semantics:
  For each active lane `i` with `i < active_lanes`, reads the element at UB address `source + offsets[i]` and writes it to result lane `i`.
- CCE correspondence:
  `vgather2(...)`
  `__builtin_cce_vgather2_*`, `__builtin_cce_vgather2_v300_*`

### `pto.vgatherb`

- syntax:
  `%result = pto.vgatherb %source, %offsets, %active_lanes : !llvm.ptr<AS>, !pto.vreg<NxI>, index -> !pto.vreg<NxT>`
- operand roles:
  `%source` is the UB base pointer, `%offsets` is the per-lane offset vector, `%active_lanes` bounds how many lanes participate, and `%result` is the gathered vector.
- ISA family:
  `VGATHERB`
- semantics:
  For each active lane `i` with `i < active_lanes`, reads the byte-granular element at UB address `source + offsets[i]` and writes it to result lane `i`.
- CCE correspondence:
  `vgatherb(...)`
  `__builtin_cce_vgatherb_*`, `__builtin_cce_vgatherb_v300_*`, `__builtin_cce_vgatherb_v310_*`

### `pto.vgather2_bc`

- syntax:
  `%result = pto.vgather2_bc %source, %offsets, %mask : !llvm.ptr<AS>, !pto.vreg<NxI>, !pto.mask -> !pto.vreg<NxT>`
- operand roles:
  `%source` is the UB base pointer, `%offsets` is the per-lane offset vector, `%mask` selects which lanes participate, and `%result` is the gathered vector.
- ISA family:
  `VGATHER2_BC`
- semantics:
  For each lane enabled by `%mask`, reads the element at UB address `source + offsets[i]` and writes it to result lane `i`.
- CCE correspondence:
  `vgather2_bc(...)`
  `__builtin_cce_vgather2_bc_*`

### `pto.vsld`

- syntax:
  `%result = pto.vsld %source[%offset], "STRIDE" : !llvm.ptr<AS> -> !pto.vreg<NxT>`
- operand roles:
  `%source` is the UB base pointer, `%offset` is the displacement, `"STRIDE"` selects the strided-load token, and `%result` is the loaded vector.
- ISA family:
  `VSLD`
- semantics:
  Reads `%result` from UB using the address progression encoded by `STRIDE`. The stride token determines the spacing between consecutive source elements or packed groups.
- CCE correspondence:
  `vsld(...)`
  `__builtin_cce_vsld_*`

### `pto.vsldb`

- syntax:
  `%result = pto.vsldb %source, %offset, %mask : !llvm.ptr<AS>, i32, !pto.mask -> !pto.vreg<NxT>`
- operand roles:
  `%source` is the UB base pointer, `%offset` is the scalar displacement, `%mask` is the predicate control, and `%result` is the loaded vector.
- ISA family:
  `VSLDB`
- semantics:
  Reads a strided vector from UB using the scalar displacement `%offset` and writes only the lanes enabled by `%mask`.
- CCE correspondence:
  `vsldb(...)`
  `__builtin_cce_vsldb_*`, `__builtin_cce_vsldb_post_*`

## 5. Materialization And Predicate Construction

### `pto.vbr`

- syntax:
  `%result = pto.vbr %value : T -> !pto.vreg<NxT>`
- operand roles:
  `%value` is the scalar value broadcast into all lanes and `%result` is the produced vector.
- ISA family:
  `VBR`
- semantics:
  Broadcasts one scalar value across all lanes of the result vector.
- CCE correspondence:
  broadcast/materialization family used by PTO scalar-to-vector expansion

### `pto.vdup`

- syntax:
  `%result = pto.vdup %input {position = "POSITION", mode = "MODE"} : T|!pto.vreg<NxT> -> !pto.vreg<NxT>`
- operand roles:
  `%input` is either the scalar source or the source vector, `"POSITION"` selects the lane position when duplicating from a vector, `"MODE"` carries the duplication mode token, and `%result` is the duplicated vector.
- ISA family:
  `VDUP` / `VDUPS` / `VDUPI` / `VDUPM`
- semantics:
  Duplicates a scalar input or the lane selected by `POSITION` into every lane of `%result`, according to the duplication form selected by `MODE`.
- CCE correspondence:
  `vdup(...)`
  `__builtin_cce_vdup_*`

### `pto.vpset_b8`

- syntax:
  `%result = pto.vpset_b8 "PAT_*" : !pto.mask`
- operand roles:
  `"PAT_*"` selects the ISA predicate pattern and `%result` is the produced predicate register.
- ISA family:
  `PSET`
- semantics:
  Creates a predicate register in 8-bit granularity from the selected `PAT_*` pattern.
- CCE correspondence:
  `pset_b8(...)`
  `__builtin_cce_pset_b8`

### `pto.vpset_b16`

- syntax:
  `%result = pto.vpset_b16 "PAT_*" : !pto.mask`
- operand roles:
  `"PAT_*"` selects the ISA predicate pattern and `%result` is the produced predicate register.
- ISA family:
  `PSET`
- semantics:
  Creates a predicate register in 16-bit granularity from the selected `PAT_*` pattern.
- CCE correspondence:
  `pset_b16(...)`
  `__builtin_cce_pset_b16`

### `pto.vpset_b32`

- syntax:
  `%result = pto.vpset_b32 "PAT_*" : !pto.mask`
- operand roles:
  `"PAT_*"` selects the ISA predicate pattern and `%result` is the produced predicate register.
- ISA family:
  `PSET`
- semantics:
  Creates a predicate register in 32-bit granularity from the selected `PAT_*` pattern.
- CCE correspondence:
  `pset_b32(...)`
  `__builtin_cce_pset_b32`

### `pto.vpge_b8`

- syntax:
  `%result = pto.vpge_b8 "PAT_*" : !pto.mask`
- operand roles:
  `"PAT_*"` selects the ISA predicate pattern and `%result` is the produced predicate register.
- ISA family:
  `PGE`
- semantics:
  Creates an 8-bit-granularity prefix predicate from the selected `PAT_*` pattern.
- CCE correspondence:
  `pge_b8(...)`
  `__builtin_cce_pge_b8`

### `pto.vpge_b16`

- syntax:
  `%result = pto.vpge_b16 "PAT_*" : !pto.mask`
- operand roles:
  `"PAT_*"` selects the ISA predicate pattern and `%result` is the produced predicate register.
- ISA family:
  `PGE`
- semantics:
  Creates a 16-bit-granularity prefix predicate from the selected `PAT_*` pattern.
- CCE correspondence:
  `pge_b16(...)`
  `__builtin_cce_pge_b16`

### `pto.vpge_b32`

- syntax:
  `%result = pto.vpge_b32 "PAT_*" : !pto.mask`
- operand roles:
  `"PAT_*"` selects the ISA predicate pattern and `%result` is the produced predicate register.
- ISA family:
  `PGE`
- semantics:
  Creates a 32-bit-granularity prefix predicate from the selected `PAT_*` pattern.
- CCE correspondence:
  `pge_b32(...)`
  `__builtin_cce_pge_b32`

### `pto.vppack`

- syntax:
  `%result = pto.vppack %input, "PART" : !pto.mask -> !pto.mask`
- operand roles:
  `%input` is the source predicate, `"PART"` selects which packed half is addressed, and `%result` is the transformed predicate.
- ISA family:
  `PPACK`
- semantics:
  Compresses the predicate lanes selected by `PART` into the packed predicate representation returned in `%result`.
- CCE correspondence:
  `ppack(...)`

### `pto.vpunpack`

- syntax:
  `%result = pto.vpunpack %input, "PART" : !pto.mask -> !pto.mask`
- operand roles:
  `%input` is the source predicate, `"PART"` selects which packed half is addressed, and `%result` is the transformed predicate.
- ISA family:
  `PUNPACK`
- semantics:
  Expands the packed predicate lanes selected by `PART` into the unpacked predicate representation returned in `%result`.
- CCE correspondence:
  `punpack(...)`

## 6. Unary Vector Ops

### `pto.vabs`

- syntax:
  `%result = pto.vabs %input : !pto.vreg<NxT> -> !pto.vreg<NxT>`
- operand roles:
  `%input` is the source vector and `%result` is the transformed vector result.
- ISA family:
  `VABS`
- semantics:
  Applies lane-wise absolute value to the input vector.
- CCE correspondence:
  `vabs(...)`
  `__builtin_cce_vabs_*`

### `pto.vexp`

- syntax:
  `%result = pto.vexp %input : !pto.vreg<NxT> -> !pto.vreg<NxT>`
- operand roles:
  `%input` is the source vector and `%result` is the transformed vector result.
- ISA family:
  `VEXP`
- semantics:
  Applies lane-wise exponential to the input vector.
- CCE correspondence:
  `vexp(...)`
  `__builtin_cce_vexp_*`

### `pto.vln`

- syntax:
  `%result = pto.vln %input : !pto.vreg<NxT> -> !pto.vreg<NxT>`
- operand roles:
  `%input` is the source vector and `%result` is the transformed vector result.
- ISA family:
  `VLN`
- semantics:
  Applies lane-wise natural logarithm to the input vector.
- CCE correspondence:
  `vln(...)`
  `__builtin_cce_vln_*`

### `pto.vsqrt`

- syntax:
  `%result = pto.vsqrt %input : !pto.vreg<NxT> -> !pto.vreg<NxT>`
- operand roles:
  `%input` is the source vector and `%result` is the transformed vector result.
- ISA family:
  `VSQRT`
- semantics:
  Applies lane-wise square root to the input vector.
- CCE correspondence:
  `vsqrt(...)`
  `__builtin_cce_vsqrt_*`

### `pto.vrec`

- syntax:
  `%result = pto.vrec %input : !pto.vreg<NxT> -> !pto.vreg<NxT>`
- operand roles:
  `%input` is the source vector and `%result` is the transformed vector result.
- ISA family:
  `VREC`
- semantics:
  Applies lane-wise reciprocal to the input vector.
- CCE correspondence:
  `vrec(...)`
  `__builtin_cce_vrec_*`

### `pto.vrelu`

- syntax:
  `%result = pto.vrelu %input : !pto.vreg<NxT> -> !pto.vreg<NxT>`
- operand roles:
  `%input` is the source vector and `%result` is the transformed vector result.
- ISA family:
  `VRELU`
- semantics:
  Applies lane-wise rectified-linear activation to the input vector.
- CCE correspondence:
  `vrelu(...)`
  `__builtin_cce_vrelu_*`

### `pto.vnot`

- syntax:
  `%result = pto.vnot %input : !pto.vreg<NxT> -> !pto.vreg<NxT>`
- operand roles:
  `%input` is the source vector and `%result` is the transformed vector result.
- ISA family:
  `VNOT`
- semantics:
  Applies lane-wise bitwise logical inversion to the input vector.
- CCE correspondence:
  `vnot(...)`
  `__builtin_cce_vnot_*`

### `pto.vcadd`

- syntax:
  `%result = pto.vcadd %input : !pto.vreg<NxT> -> !pto.vreg<NxT>`
- operand roles:
  `%input` is the source vector and `%result` is the transformed vector result.
- ISA family:
  `VCADD`
- semantics:
  Performs reduction-add within each ISA reduction group and returns the vector-shaped partial results.
- CCE correspondence:
  `vcadd(...)`
  `__builtin_cce_vcadd_*`

### `pto.vcmax`

- syntax:
  `%result = pto.vcmax %input : !pto.vreg<NxT> -> !pto.vreg<NxT>`
- operand roles:
  `%input` is the source vector and `%result` is the transformed vector result.
- ISA family:
  `VCMAX`
- semantics:
  Performs reduction-max within each ISA reduction group and returns the vector-shaped partial results.
- CCE correspondence:
  `vcmax(...)`
  `__builtin_cce_vcmax_*`

### `pto.vcmin`

- syntax:
  `%result = pto.vcmin %input : !pto.vreg<NxT> -> !pto.vreg<NxT>`
- operand roles:
  `%input` is the source vector and `%result` is the transformed vector result.
- ISA family:
  `VCMIN`
- semantics:
  Performs reduction-min within each ISA reduction group and returns the vector-shaped partial results.
- CCE correspondence:
  `vcmin(...)`
  `__builtin_cce_vcmin_*`

### `pto.vbcnt`

- syntax:
  `%result = pto.vbcnt %input : !pto.vreg<NxT> -> !pto.vreg<NxT>`
- operand roles:
  `%input` is the source vector and `%result` is the transformed vector result.
- ISA family:
  `VBCNT`
- semantics:
  Computes the lane-wise bit population count.
- CCE correspondence:
  `vbcnt(...)`
  `__builtin_cce_vbcnt_*`

### `pto.vcls`

- syntax:
  `%result = pto.vcls %input : !pto.vreg<NxT> -> !pto.vreg<NxT>`
- operand roles:
  `%input` is the source vector and `%result` is the transformed vector result.
- ISA family:
  `VCLS`
- semantics:
  Computes the lane-wise count of leading sign bits.
- CCE correspondence:
  `vcls(...)`
  `__builtin_cce_vcls_*`

## 7. Binary Vector Ops

### `pto.vadd`

- syntax:
  `%result = pto.vadd %lhs, %rhs : !pto.vreg<NxT>, !pto.vreg<NxT> -> !pto.vreg<NxT>`
- operand roles:
  `%lhs` is the first source vector, `%rhs` is the second source vector, and `%result` is the computed vector result.
- ISA family:
  `VADD`
- semantics:
  Computes lane-wise addition of `%lhs` and `%rhs`.
- CCE correspondence:
  `vadd(...)`
  `__builtin_cce_vadd_*`

### `pto.vsub`

- syntax:
  `%result = pto.vsub %lhs, %rhs : !pto.vreg<NxT>, !pto.vreg<NxT> -> !pto.vreg<NxT>`
- operand roles:
  `%lhs` is the first source vector, `%rhs` is the second source vector, and `%result` is the computed vector result.
- ISA family:
  `VSUB`
- semantics:
  Computes lane-wise subtraction of `%rhs` from `%lhs`.
- CCE correspondence:
  `vsub(...)`
  `__builtin_cce_vsub_*`

### `pto.vmul`

- syntax:
  `%result = pto.vmul %lhs, %rhs : !pto.vreg<NxT>, !pto.vreg<NxT> -> !pto.vreg<NxT>`
- operand roles:
  `%lhs` is the first source vector, `%rhs` is the second source vector, and `%result` is the computed vector result.
- ISA family:
  `VMUL`
- semantics:
  Computes lane-wise multiplication of `%lhs` and `%rhs`.
- CCE correspondence:
  `vmul(...)`
  `__builtin_cce_vmul_*`

### `pto.vdiv`

- syntax:
  `%result = pto.vdiv %lhs, %rhs : !pto.vreg<NxT>, !pto.vreg<NxT> -> !pto.vreg<NxT>`
- operand roles:
  `%lhs` is the first source vector, `%rhs` is the second source vector, and `%result` is the computed vector result.
- ISA family:
  `VDIV`
- semantics:
  Computes lane-wise division of `%lhs` by `%rhs`.
- CCE correspondence:
  `vdiv(...)`
  `__builtin_cce_vdiv_*`

### `pto.vmax`

- syntax:
  `%result = pto.vmax %lhs, %rhs : !pto.vreg<NxT>, !pto.vreg<NxT> -> !pto.vreg<NxT>`
- operand roles:
  `%lhs` is the first source vector, `%rhs` is the second source vector, and `%result` is the computed vector result.
- ISA family:
  `VMAX`
- semantics:
  Computes the lane-wise maximum of `%lhs` and `%rhs`.
- CCE correspondence:
  `vmax(...)`
  `__builtin_cce_vmax_*`

### `pto.vmin`

- syntax:
  `%result = pto.vmin %lhs, %rhs : !pto.vreg<NxT>, !pto.vreg<NxT> -> !pto.vreg<NxT>`
- operand roles:
  `%lhs` is the first source vector, `%rhs` is the second source vector, and `%result` is the computed vector result.
- ISA family:
  `VMIN`
- semantics:
  Computes the lane-wise minimum of `%lhs` and `%rhs`.
- CCE correspondence:
  `vmin(...)`
  `__builtin_cce_vmin_*`

### `pto.vand`

- syntax:
  `%result = pto.vand %lhs, %rhs : !pto.vreg<NxT>, !pto.vreg<NxT> -> !pto.vreg<NxT>`
- operand roles:
  `%lhs` is the first source vector, `%rhs` is the second source vector, and `%result` is the computed vector result.
- ISA family:
  `VAND`
- semantics:
  Computes lane-wise bitwise AND of `%lhs` and `%rhs`.
- CCE correspondence:
  `vand(...)`
  `__builtin_cce_vand_*`

### `pto.vor`

- syntax:
  `%result = pto.vor %lhs, %rhs : !pto.vreg<NxT>, !pto.vreg<NxT> -> !pto.vreg<NxT>`
- operand roles:
  `%lhs` is the first source vector, `%rhs` is the second source vector, and `%result` is the computed vector result.
- ISA family:
  `VOR`
- semantics:
  Computes lane-wise bitwise OR of `%lhs` and `%rhs`.
- CCE correspondence:
  `vor(...)`
  `__builtin_cce_vor_*`

### `pto.vxor`

- syntax:
  `%result = pto.vxor %lhs, %rhs : !pto.vreg<NxT>, !pto.vreg<NxT> -> !pto.vreg<NxT>`
- operand roles:
  `%lhs` is the first source vector, `%rhs` is the second source vector, and `%result` is the computed vector result.
- ISA family:
  `VXOR`
- semantics:
  Computes lane-wise bitwise XOR of `%lhs` and `%rhs`.
- CCE correspondence:
  `vxor(...)`
  `__builtin_cce_vxor_*`

### `pto.vshl`

- syntax:
  `%result = pto.vshl %lhs, %rhs : !pto.vreg<NxT>, !pto.vreg<NxT> -> !pto.vreg<NxT>`
- operand roles:
  `%lhs` is the first source vector, `%rhs` is the second source vector, and `%result` is the computed vector result.
- ISA family:
  `VSHL`
- semantics:
  Shifts each lane of `%lhs` left by the amount carried in the corresponding `%rhs` lane.
- CCE correspondence:
  `vshl(...)`
  `__builtin_cce_vshl_*`

### `pto.vshr`

- syntax:
  `%result = pto.vshr %lhs, %rhs : !pto.vreg<NxT>, !pto.vreg<NxT> -> !pto.vreg<NxT>`
- operand roles:
  `%lhs` is the first source vector, `%rhs` is the second source vector, and `%result` is the computed vector result.
- ISA family:
  `VSHR`
- semantics:
  Shifts each lane of `%lhs` right by the amount carried in the corresponding `%rhs` lane.
- CCE correspondence:
  `vshr(...)`
  `__builtin_cce_vshr_*`

## 8. Vec-Scalar Ops

### `pto.vmuls`

- syntax:
  `%result = pto.vmuls %input, %scalar : !pto.vreg<NxT>, T -> !pto.vreg<NxT>`
- operand roles:
  `%input` is the source vector, `%scalar` is the scalar operand applied to every lane, and `%result` is the computed vector result.
- ISA family:
  `VMULS`
- semantics:
  Multiplies each input lane by the scalar operand.
- CCE correspondence:
  `vmuls(...)`
  `__builtin_cce_vmuls_*`

### `pto.vadds`

- syntax:
  `%result = pto.vadds %input, %scalar : !pto.vreg<NxT>, T -> !pto.vreg<NxT>`
- operand roles:
  `%input` is the source vector, `%scalar` is the scalar operand applied to every lane, and `%result` is the computed vector result.
- ISA family:
  `VADDS`
- semantics:
  Adds the scalar operand to each input lane.
- CCE correspondence:
  `vadds(...)`
  `__builtin_cce_vadds_*`

### `pto.vmaxs`

- syntax:
  `%result = pto.vmaxs %input, %scalar : !pto.vreg<NxT>, T -> !pto.vreg<NxT>`
- operand roles:
  `%input` is the source vector, `%scalar` is the scalar operand applied to every lane, and `%result` is the computed vector result.
- ISA family:
  `VMAXS`
- semantics:
  Computes the lane-wise maximum of the input vector and the scalar operand.
- CCE correspondence:
  `vmaxs(...)`
  `__builtin_cce_vmaxs_*`

### `pto.vmins`

- syntax:
  `%result = pto.vmins %input, %scalar : !pto.vreg<NxT>, T -> !pto.vreg<NxT>`
- operand roles:
  `%input` is the source vector, `%scalar` is the scalar operand applied to every lane, and `%result` is the computed vector result.
- ISA family:
  `VMINS`
- semantics:
  Computes the lane-wise minimum of the input vector and the scalar operand.
- CCE correspondence:
  `vmins(...)`
  `__builtin_cce_vmins_*`

### `pto.vlrelu`

- syntax:
  `%result = pto.vlrelu %input, %scalar : !pto.vreg<NxT>, T -> !pto.vreg<NxT>`
- operand roles:
  `%input` is the source vector, `%scalar` is the scalar operand applied to every lane, and `%result` is the computed vector result.
- ISA family:
  `VLRELU`
- semantics:
  Applies a leaky-ReLU style lane-wise transform using the scalar slope operand.
- CCE correspondence:
  `vlrelu(...)`
  `__builtin_cce_vlrelu_*`

### `pto.vshls`

- syntax:
  `%result = pto.vshls %input, %scalar : !pto.vreg<NxT>, T -> !pto.vreg<NxT>`
- operand roles:
  `%input` is the source vector, `%scalar` is the scalar operand applied to every lane, and `%result` is the computed vector result.
- ISA family:
  `VSHLS`
- semantics:
  Shifts each input lane left by the scalar shift amount.
- CCE correspondence:
  `vshls(...)`
  `__builtin_cce_vshls_*`

### `pto.vshrs`

- syntax:
  `%result = pto.vshrs %input, %scalar : !pto.vreg<NxT>, T -> !pto.vreg<NxT>`
- operand roles:
  `%input` is the source vector, `%scalar` is the scalar operand applied to every lane, and `%result` is the computed vector result.
- ISA family:
  `VSHRS`
- semantics:
  Shifts each input lane right by the scalar shift amount.
- CCE correspondence:
  `vshrs(...)`
  `__builtin_cce_vshrs_*`

## 9. Carry, Compare And Select

ISA assertions for this family:

- Predicate-gated arithmetic uses the supplied predicate as the active-lane mask.
- Comparison results are predicates, not integer vectors.
- Comparison families use zeroing semantics for inactive destination lanes.


### `pto.vaddc`

- syntax:
  `%result, %carry = pto.vaddc %lhs, %rhs, %mask : !pto.vreg<NxT>, !pto.vreg<NxT>, !pto.mask -> !pto.vreg<NxT>, !pto.mask`
- operand roles:
  `%lhs` and `%rhs` are the source vectors, `%mask` is the predicate control, `%result` is the arithmetic result, and `%carry` is the produced carry or borrow predicate.
- ISA family:
  `VADDC`
- semantics:
  For each lane enabled by `%mask`, adds `%lhs` and `%rhs`, writes the arithmetic result to `%result`, and writes the lane carry-out bit to `%carry`.
- CCE correspondence:
  `vaddc(...)`
  `__builtin_cce_vaddc_*`

### `pto.vsubc`

- syntax:
  `%result, %carry = pto.vsubc %lhs, %rhs, %mask : !pto.vreg<NxT>, !pto.vreg<NxT>, !pto.mask -> !pto.vreg<NxT>, !pto.mask`
- operand roles:
  `%lhs` and `%rhs` are the source vectors, `%mask` is the predicate control, `%result` is the arithmetic result, and `%carry` is the produced carry or borrow predicate.
- ISA family:
  `VSUBC`
- semantics:
  For each lane enabled by `%mask`, subtracts `%rhs` from `%lhs`, writes the arithmetic result to `%result`, and writes the lane carry-or-borrow bit to `%carry`.
- CCE correspondence:
  `vsubc(...)`
  `__builtin_cce_vsubc_*`

### `pto.vaddcs`

- syntax:
  `%result, %carry = pto.vaddcs %lhs, %rhs, %carry_in, %mask : !pto.vreg<NxT>, !pto.vreg<NxT>, !pto.mask, !pto.mask -> !pto.vreg<NxT>, !pto.mask`
- operand roles:
  `%lhs` and `%rhs` are the source vectors, `%carry_in` is the incoming carry or borrow predicate, `%mask` is the predicate control, `%result` is the arithmetic result, and `%carry` is the updated carry or borrow predicate.
- ISA family:
  `VADDCS`
- semantics:
  For each lane enabled by `%mask`, adds `%lhs`, `%rhs`, and the carry-in bit from `%carry_in`, writes the arithmetic result to `%result`, and writes the successor carry bit to `%carry`.
- CCE correspondence:
  `vaddcs(...)`
  `__builtin_cce_vaddcs_*`

### `pto.vsubcs`

- syntax:
  `%result, %carry = pto.vsubcs %lhs, %rhs, %carry_in, %mask : !pto.vreg<NxT>, !pto.vreg<NxT>, !pto.mask, !pto.mask -> !pto.vreg<NxT>, !pto.mask`
- operand roles:
  `%lhs` and `%rhs` are the source vectors, `%carry_in` is the incoming carry or borrow predicate, `%mask` is the predicate control, `%result` is the arithmetic result, and `%carry` is the updated carry or borrow predicate.
- ISA family:
  `VSUBCS`
- semantics:
  For each lane enabled by `%mask`, subtracts `%rhs` and the carry-or-borrow bit from `%carry_in` from `%lhs`, writes the arithmetic result to `%result`, and writes the successor carry-or-borrow bit to `%carry`.
- CCE correspondence:
  `vsubcs(...)`
  `__builtin_cce_vsubcs_*`

### `pto.vsel`

- syntax:
  `%result = pto.vsel %src0, %src1, %mask : !pto.vreg<NxT>, !pto.vreg<NxT>, !pto.mask -> !pto.vreg<NxT>`
- operand roles:
  `%src0` and `%src1` are the candidate source vectors, `%mask` selects which source each lane takes, and `%result` is the selected vector.
- ISA family:
  `VSEL`
- semantics:
  Selects per lane between `%src0` and `%src1` under the control predicate `%mask`.
- CCE correspondence:
  `vsel(...)`
  `__builtin_cce_vsel_*`

### `pto.vselr`

- syntax:
  `%result = pto.vselr %src0, %src1 : !pto.vreg<NxT>, !pto.vreg<NxI> -> !pto.vreg<NxT>`
- operand roles:
  `%src0` is the data vector, `%src1` is the integer lane-selector vector, and `%result` is the selected or permuted output vector.
- ISA family:
  `VSELR`
- semantics:
  Selects or permutes lanes from `%src0` using the lane indices carried in `%src1`.
- CCE correspondence:
  `vselr(...)`
  `__builtin_cce_vselr_*`

### `pto.vselrv2`

- syntax:
  `%result = pto.vselrv2 %src0, %src1 : !pto.vreg<NxT>, !pto.vreg<NxI> -> !pto.vreg<NxT>`
- operand roles:
  `%src0` is the data vector, `%src1` is the integer lane-selector vector, and `%result` is the selected or permuted output vector.
- ISA family:
  `VSELR v2`
- semantics:
  Selects or permutes lanes from `%src0` using the lane indices carried in `%src1`.
- CCE correspondence:
  `vselrv2(...)`
  `__builtin_cce_vselrv2_*`

### `pto.vcmp`

- syntax:
  `%result = pto.vcmp %src0, %src1, %mask, "CMP_MODE" : !pto.vreg<NxT>, !pto.vreg<NxT>, !pto.mask -> !pto.mask`
- operand roles:
  `%src0` and `%src1` are the values being compared, `%mask` is the seed predicate or enable mask, `"CMP_MODE"` selects the comparison relation, and `%result` is the produced predicate.
- ISA family:
  `VCMP`
- semantics:
  For each lane enabled by `%mask`, compares `%src0` and `%src1` using `CMP_MODE` and writes the boolean result into `%result`. Lanes disabled by `%mask` are cleared to zero in the returned predicate.
- CCE correspondence:
  `vcmp(...)`
  `__builtin_cce_vcmp_<op>_*_z`

### `pto.vcmps`

- syntax:
  `%result = pto.vcmps %src, %scalar, %mask, "CMP_MODE" : !pto.vreg<NxT>, T, !pto.mask -> !pto.mask`
- operand roles:
  `%src` is the vector input, `%scalar` is the scalar comparison value, `%mask` is the seed predicate or enable mask, `"CMP_MODE"` selects the comparison relation, and `%result` is the produced predicate.
- ISA family:
  `VCMPS`
- semantics:
  For each lane enabled by `%mask`, compares `%src` against `%scalar` using `CMP_MODE` and writes the boolean result into `%result`. Lanes disabled by `%mask` are cleared to zero in the returned predicate.
- CCE correspondence:
  `vcmps(...)`
  `__builtin_cce_vcmps_<op>_*_z`

### `pto.vpnot`

- syntax:
  `%result = pto.vpnot %input, %mask : !pto.mask, !pto.mask -> !pto.mask`
- operand roles:
  `%input` is the source predicate, `%mask` is the predicate control, and `%result` is the inverted predicate result.
- ISA family:
  `PNOT`
- semantics:
  For each lane enabled by `%mask`, inverts the corresponding bit of `%input` and writes the result bit to `%result`.
- CCE correspondence:
  `pnot(...)`

### `pto.vpsel`

- syntax:
  `%result = pto.vpsel %src0, %src1, %mask : !pto.mask, !pto.mask, !pto.mask -> !pto.mask`
- operand roles:
  `%src0` and `%src1` are the candidate source predicates, `%mask` selects which predicate each bit takes, and `%result` is the selected predicate.
- ISA family:
  `PSEL`
- semantics:
  Selects each result predicate bit from `%src0` or `%src1` under the control of `%mask`.
- CCE correspondence:
  `psel(...)`

## 10. Pairing And Interleave

### `pto.vpdintlv_b8`

- syntax:
  `%low, %high = pto.vpdintlv_b8 %lhs, %rhs : !pto.mask, !pto.mask -> !pto.mask, !pto.mask`
- operand roles:
  `%lhs` and `%rhs` are the two source predicates, and `%low` plus `%high` are the two predicate results produced by the interleave or deinterleave split.
- ISA family:
  `PDINTLV`
- semantics:
  Deinterleaves predicate data into low and high predicate results.
- CCE correspondence:
  predicate interleave/deinterleave family

### `pto.vpintlv_b16`

- syntax:
  `%low, %high = pto.vpintlv_b16 %lhs, %rhs : !pto.mask, !pto.mask -> !pto.mask, !pto.mask`
- operand roles:
  `%lhs` and `%rhs` are the two source predicates, and `%low` plus `%high` are the two predicate results produced by the interleave or deinterleave split.
- ISA family:
  `PINTLV`
- semantics:
  Interleaves predicate data into low and high predicate results.
- CCE correspondence:
  predicate interleave/deinterleave family

### `pto.vintlv`

- syntax:
  `%low, %high = pto.vintlv %lhs, %rhs : !pto.vreg<NxT>, !pto.vreg<NxT> -> !pto.vreg<NxT>, !pto.vreg<NxT>`
- operand roles:
  `%lhs` and `%rhs` are the two source vectors, and `%low` plus `%high` are the two vector results produced by the interleave or deinterleave split.
- ISA family:
  `VINTLV`
- semantics:
  Interleaves lanes from `%lhs` and `%rhs` into one combined lane stream and returns the low half in `%low` and the high half in `%high`.
- CCE correspondence:
  `vintlv(...)`
  `__builtin_cce_vintlv_*`

### `pto.vdintlv`

- syntax:
  `%low, %high = pto.vdintlv %lhs, %rhs : !pto.vreg<NxT>, !pto.vreg<NxT> -> !pto.vreg<NxT>, !pto.vreg<NxT>`
- operand roles:
  `%lhs` and `%rhs` are the two source vectors, and `%low` plus `%high` are the two vector results produced by the interleave or deinterleave split.
- ISA family:
  `VDINTLV`
- semantics:
  Deinterleaves the combined lane streams in `%lhs` and `%rhs` and returns the low deinterleaved half in `%low` and the high deinterleaved half in `%high`.
- CCE correspondence:
  `vdintlv(...)`
  `__builtin_cce_vdintlv_*`

### `pto.vintlvv2`

- syntax:
  `%result = pto.vintlvv2 %lhs, %rhs, "PART" : !pto.vreg<NxT>, !pto.vreg<NxT> -> !pto.vreg<NxT>`
- operand roles:
  `%lhs` and `%rhs` are the two source vectors, `"PART"` selects which half of the interleaved lane stream is returned, and `%result` is the selected vector result.
- ISA family:
  `VINTLV v2`
- semantics:
  Interleaves `%lhs` and `%rhs` into one combined lane stream and returns only the half selected by `PART`.
- CCE correspondence:
  `vintlvv2(...)`
  `__builtin_cce_vintlvv2_*`

### `pto.vdintlvv2`

- syntax:
  `%result = pto.vdintlvv2 %lhs, %rhs, "PART" : !pto.vreg<NxT>, !pto.vreg<NxT> -> !pto.vreg<NxT>`
- operand roles:
  `%lhs` and `%rhs` are the two source vectors, `"PART"` selects which half of the deinterleaved lane stream is returned, and `%result` is the selected vector result.
- ISA family:
  `VDINTLV v2`
- semantics:
  Deinterleaves `%lhs` and `%rhs` into one logical lane stream and returns only the half selected by `PART`.
- CCE correspondence:
  `vdintlvv2(...)`
  `__builtin_cce_vdintlvv2_*`

## 11. Conversion, Index And Sort

ISA assertions for this family:

- For width-changing conversions, predication is applied to input lanes and is composed with part-selection controls such as even/odd or packed-part selectors.
- Narrowing conversions place results into the selected destination part and zero the remaining part of the widened slot.
- Widening conversions read only the selected source part; the unselected part is architecturally ignored.
- Sort families operate on UB-resident proposal data and preserve the ISA tie-break rule that lower original indices win on equal scores.


### `pto.vtrc`

- syntax:
  `%result = pto.vtrc %input, "ROUND_MODE" : !pto.vreg<NxT> -> !pto.vreg<NxT>`
- operand roles:
  `%input` is the source vector, `"ROUND_MODE"` selects the rounding behavior, and `%result` is the converted vector result.
- ISA family:
  `VTRC`
- semantics:
  Rounds each input lane according to `ROUND_MODE` and returns the truncated converted result. The rounding mode is part of the ISA-visible semantics, not a lowering hint.
- CCE correspondence:
  `vtrc(...)`
  `__builtin_cce_vtrc_*`

### `pto.vcvt`

- syntax:
  `%result = pto.vcvt %input {round_mode = "ROUND_MODE", sat = "SAT_MODE", part = "PART_MODE"} : !pto.vreg<NxT0> -> !pto.vreg<NxT1>`
- operand roles:
  `%input` is the source vector, `"ROUND_MODE"` selects the rounding behavior, `"SAT_MODE"` selects saturation or truncation behavior, `"PART_MODE"` selects the even or odd conversion part when required by the ISA form, and `%result` is the converted vector result.
- ISA family:
  `VCVTFI` / `VCVTFF` / `VCVTIF` / `VCVTII`
- semantics:
  Converts `%input` lane-wise according to the source type, destination type, rounding rule, saturation rule, and part-selection rule encoded by the op form. Width-changing forms consume only the selected source part or produce only the selected destination part exactly as required by the ISA conversion family.
- CCE correspondence:
  `vcvt(...)`
  builtin families:
  `__builtin_cce_vcvt*`, `__builtin_cce_vcvtfi_*`, `__builtin_cce_vcvtif_*`, `__builtin_cce_vcvtii_*`, `__builtin_cce_vcvtff_*`

### `pto.vci`

- syntax:
  `%result = pto.vci %index {order = "ORDER"} : integer -> !pto.vreg<NxT>`
- operand roles:
  `%index` is the scalar seed or base index value, `"ORDER"` selects the lane-index ordering policy, and `%result` is the generated integer index vector.
- ISA family:
  `VCI`
- semantics:
  Materializes lane indices from the scalar seed value using the selected ordering policy.
- CCE correspondence:
  `vci(...)`
  `__builtin_cce_vci_*`

### `pto.vbitsort`

- syntax:
  `pto.vbitsort %destination, %source, %indices, %repeat_times : !llvm.ptr<AS>, !llvm.ptr<AS>, !llvm.ptr<AS>, index`
- operand roles:
  `%destination` is the UB output buffer, `%source` is the UB score buffer, `%indices` is the UB index buffer, and `%repeat_times` is the repeat count for consecutive sort invocations.
- ISA family:
  `VBS32`
- semantics:
  Sorts the proposal records named by `%source` and `%indices` and writes the ordered result stream to `%destination`. `repeat_times` controls how many consecutive sort iterations are performed, equal scores are ordered by lower original proposal index first, and the destination region must not overlap the source regions.
- CCE correspondence:
  `vbitsort(...)`
  `__builtin_cce_vbitsort_*`

### `pto.vmrgsort4`

- syntax:
  `pto.vmrgsort4 %destination, %source0, %source1, %source2, %source3, %count, %config : !llvm.ptr<AS>, !llvm.ptr<AS>, !llvm.ptr<AS>, !llvm.ptr<AS>, !llvm.ptr<AS>, i64, i64`
- operand roles:
  `%destination` is the UB output buffer, `%source0` through `%source3` are the four UB input list bases, `%count` is the total work or encoded list-count payload, and `%config` is the ISA merge-sort configuration word.
- ISA family:
  `VMS4v2`
- semantics:
  Merges four sorted proposal lists from UB into one sorted output stream. On equal scores, entries from the lower-numbered input list win; `%config` controls repeat and exhausted-input behavior; and source and destination regions must satisfy the ISA non-overlap rules.
- CCE correspondence:
  `vmrgsort4(...)`
  `__builtin_cce_vmrgsort4_*`

## 12. Extended Arithmetic

### `pto.vmull`

- syntax:
  `%low, %high = pto.vmull %lhs, %rhs, %mask : !pto.vreg<NxT>, !pto.vreg<NxT>, !pto.mask -> !pto.vreg<NxT>, !pto.vreg<NxT>`
- operand roles:
  `%lhs` and `%rhs` are the source vectors, `%mask` is the predicate control, and `%low` plus `%high` are the split widened product results.
- ISA family:
  `VMULL`
- semantics:
  Multiplies lanes under `%mask` and returns the widened product split across low and high result vectors.
- CCE correspondence:
  `vmull(...)`
  `__builtin_cce_vmull_*`

### `pto.vmula`

- syntax:
  `%result = pto.vmula %acc, %lhs, %rhs, %mask {mode = "MODE"} : !pto.vreg<NxT>, !pto.vreg<NxT>, !pto.vreg<NxT>, !pto.mask -> !pto.vreg<NxT>`
- operand roles:
  `%acc` is the accumulator input, `%lhs` and `%rhs` are the multiplicands, `%mask` is the predicate control, `"MODE"` selects merging or zeroing behavior, and `%result` is the accumulated vector result.
- ISA family:
  `VMULA`
- semantics:
  Performs masked vector multiply-accumulate into `%acc`, with `mode` controlling the merge or zeroing behavior.
- CCE correspondence:
  `vmula(...)`
  `__builtin_cce_vmula_*_m`

## 13. Stateless Stores

ISA assertions for this family:

- These ops write to UB-backed storage and must satisfy the distribution-specific destination-alignment rules defined by the ISA family.
- ISA forms that can post-update a shared register are represented here only as addressed stores; hidden pointer mutation is not part of these stateless forms.
- Predicate-store data is architecturally `b8`, regardless of the scalar element type used by surrounding vector code.


### `pto.vsts`

- syntax:
  `pto.vsts %value, %destination[%offset] {dist = "DIST"} : !pto.vreg<NxT>, !llvm.ptr<AS>`
- operand roles:
  `%value` is the vector being stored, `%destination` is the UB base pointer, `%offset` is the store displacement, and `"DIST"` selects the ISA store distribution mode.
- ISA family:
  `VST` / `VSTI` / `VSTS`
- semantics:
  Stores `%value` to UB at `destination + offset` using the store form selected by `DIST`. The distribution token determines the destination lane layout and alignment requirement of the access.
- CCE correspondence:
  `vst(...)`, `vsts(...)`
  `__builtin_cce_vstx1_*`, `__builtin_cce_vstsx1_*`

### `pto.vscatter`

- syntax:
  `pto.vscatter %value, %destination, %offsets, %active_lanes : !pto.vreg<NxT>, !llvm.ptr<AS>, !pto.vreg<NxI>, index`
- operand roles:
  `%value` is the vector being scattered, `%destination` is the UB base pointer, `%offsets` is the per-lane offset vector, and `%active_lanes` bounds how many lanes participate.
- ISA family:
  `VSCATTER`
- semantics:
  Scatters active vector lanes to UB addresses derived from `%destination` and `%offsets`.
- CCE correspondence:
  `vscatter(...)`
  `__builtin_cce_vscatter_*`

### `pto.vsts_pred`

- syntax:
  `pto.vsts_pred %value, %destination[%offset], %active_lanes {dist = "DIST"} : !pto.vreg<NxT>, !llvm.ptr<AS>, index`
- operand roles:
  `%value` is the vector being stored, `%destination` is the UB base pointer, `%offset` is the store displacement, `%active_lanes` bounds the active prefix, and `"DIST"` selects the ISA store distribution mode.
- ISA family:
  `Predicated vector-store helper family`
- semantics:
  Stores only the active prefix of `%value` selected by `%active_lanes`, using the layout and alignment rules implied by `DIST`. Lanes outside the active prefix do not update memory.
- CCE correspondence:
  predicated vector store family

### `pto.vpsts`

- syntax:
  `pto.vpsts %value, %destination[%offset] : !pto.mask, !llvm.ptr<AS>`
- operand roles:
  `%value` is the predicate being stored, `%destination` is the UB base pointer, and `%offset` is the store displacement.
- ISA family:
  `PSTS`
- semantics:
  Stores predicate state to UB at `destination + offset`.
- CCE correspondence:
  `psts(...)`
  `__builtin_cce_psts_b8`, `__builtin_cce_psts_post_b8`

### `pto.vpst`

- syntax:
  `pto.vpst %value, %destination[%offset], "DIST" : !pto.mask, !llvm.ptr<AS>, index`
- operand roles:
  `%value` is the predicate being stored, `%destination` is the UB base pointer, `%offset` is the store displacement, and `"DIST"` selects the predicate-store distribution token.
- ISA family:
  `PST`
- semantics:
  Stores predicate state to UB using an explicit index offset and predicate-store distribution token.
- CCE correspondence:
  `pst(...)`
  `__builtin_cce_pst_b8`

### `pto.vpsti`

- syntax:
  `pto.vpsti %value, %destination, %offset, "DIST" : !pto.mask, !llvm.ptr<AS>, i32`
- operand roles:
  `%value` is the predicate being stored, `%destination` is the UB base pointer, `%offset` is the scalar displacement, and `"DIST"` selects the predicate-store distribution token.
- ISA family:
  `PSTI`
- semantics:
  Stores predicate state to UB using an immediate-style scalar offset and predicate-store distribution token.
- CCE correspondence:
  `psti(...)`
  `__builtin_cce_psti_b8`, `__builtin_cce_psti_post_b8`

### `pto.vsst`

- syntax:
  `pto.vsst %value, %destination[%offset], "STRIDE" : !pto.vreg<NxT>, !llvm.ptr<AS>`
- operand roles:
  `%value` is the vector being stored, `%destination` is the UB base pointer, `%offset` is the store displacement, and `"STRIDE"` selects the ISA strided-store token.
- ISA family:
  `VSST`
- semantics:
  Stores vector data to UB using a stride token rather than a regular contiguous distribution.
- CCE correspondence:
  `vsst(...)`
  `__builtin_cce_vsst_*`

### `pto.vstx2`

- syntax:
  `pto.vstx2 %low, %high, %destination[%offset], "DIST", %mask : !pto.vreg<NxT>, !pto.vreg<NxT>, !llvm.ptr<AS>, index, !pto.mask`
- operand roles:
  `%low` and `%high` are the two source vectors being stored, `%destination` is the UB base pointer, `%offset` is the store displacement, `"DIST"` selects the x2 store distribution token, and `%mask` is the predicate control.
- ISA family:
  `VST x2`
- semantics:
  Stores `%low` and `%high` to UB as one x2 store operation. `DIST` determines how the two source vectors are interleaved into memory, and `%mask` gates which lanes update memory.
- CCE correspondence:
  `vst(...)`
  `__builtin_cce_vstx2_*`

### `pto.vsstb`

- syntax:
  `pto.vsstb %value, %destination, %offset, %mask : !pto.vreg<NxT>, !llvm.ptr<AS>, i32, !pto.mask`
- operand roles:
  `%value` is the vector being stored, `%destination` is the UB base pointer, `%offset` is the scalar displacement, and `%mask` is the predicate control.
- ISA family:
  `VSSTB`
- semantics:
  Performs a masked strided vector store using a scalar offset and predicate mask.
- CCE correspondence:
  `vsstb(...)`
  `__builtin_cce_vsstb_*`, `__builtin_cce_vsstb_post_*`

### `pto.vsta`

- syntax:
  `pto.vsta %value, %destination[%offset] : !pto.align, !llvm.ptr<AS>, index`
- operand roles:
  `%value` is the align payload being stored, `%destination` is the UB base pointer, and `%offset` is the store displacement.
- ISA family:
  `VSTA`
- semantics:
  Stores align-state payload to UB at `destination + offset`.
- CCE correspondence:
  `vsta(...)`
  `__builtin_cce_vsta_*`

### `pto.vstas`

- syntax:
  `pto.vstas %value, %destination, %offset : !pto.align, !llvm.ptr<AS>, i32`
- operand roles:
  `%value` is the align payload being stored, `%destination` is the UB base pointer, and `%offset` is the scalar displacement.
- ISA family:
  `VSTAS`
- semantics:
  Stores align-state payload to UB using a scalar offset form.
- CCE correspondence:
  `vstas(...)`
  `__builtin_cce_vstas_*`, `__builtin_cce_vstas_post_*`

### `pto.vstar`

- syntax:
  `pto.vstar %value, %destination : !pto.align, !llvm.ptr<AS>`
- operand roles:
  `%value` is the align payload being stored and `%destination` is the base pointer used by the register-update store form.
- ISA family:
  `VSTAR`
- semantics:
  Stores align-state payload using the base pointer carried directly by `%destination`.
- CCE correspondence:
  `vstar(...)`
  `__builtin_cce_vstar_*`

## 14. Stateful Store Ops

ISA assertions for this family:

- These ops expose the ISA's hidden align-register and address-update effects as explicit SSA results.
- `"MODE"` determines whether the underlying ISA performs post-update or preserves the incoming base state.
- Correct programs thread the returned align or base state into the next dependent stateful store on the same logical stream.


These ops make ISA reference-updated state explicit as SSA results.

### `pto.vpstu`

- syntax:
  `%align_out, %base_out = pto.vpstu %align_in, %value, %base : !pto.align, !pto.mask, !llvm.ptr<AS> -> !pto.align, !llvm.ptr<AS>`
- operand roles:
  `%align_in` is the incoming align state, `%value` is the predicate being stored, `%base` is the current base pointer, `%align_out` is the updated align state, and `%base_out` is the updated base pointer.
- ISA family:
  `PSTU`
- semantics:
  Stores predicate data through the stateful predicate-store form and returns the align state and base pointer state after the store. The returned state is the exact architected successor state for the next dependent store in the same stream.
- CCE correspondence:
  `pstu(...)`
  `__builtin_cce_pstu_b16`, `__builtin_cce_pstu_b32`

### `pto.vstu`

- syntax:
  `%align_out, %offset_out = pto.vstu %align_in, %offset_in, %value, %base, "MODE" : !pto.align, index, !pto.vreg<NxT>, !llvm.ptr<AS> -> !pto.align, index`
- operand roles:
  `%align_in` is the incoming align state, `%offset_in` is the current index displacement, `%value` is the vector being stored, `%base` is the current base pointer, `"MODE"` selects post-update behavior, `%align_out` is the updated align state, and `%offset_out` is the updated index displacement.
- ISA family:
  `VSTU`
- semantics:
  Stores `%value` through the stateful unaligned-store form addressed by `%base` and `%offset_in`, then returns the successor align state and successor index displacement. If `MODE` is `POST_UPDATE`, `%offset_out` is the ISA-updated displacement; if `MODE` is `NO_POST_UPDATE`, `%offset_out` preserves the incoming displacement.
- CCE correspondence:
  `vstu(...)`
  `__builtin_cce_vstu_*`

### `pto.vstus`

- syntax:
  `%align_out, %base_out = pto.vstus %align_in, %offset, %value, %base, "MODE" : !pto.align, i32, !pto.vreg<NxT>, !llvm.ptr<AS> -> !pto.align, !llvm.ptr<AS>`
- operand roles:
  `%align_in` is the incoming align state, `%offset` is the scalar displacement, `%value` is the vector being stored, `%base` is the current base pointer, `"MODE"` selects post-update behavior, `%align_out` is the updated align state, and `%base_out` is the updated base pointer.
- ISA family:
  `VSTUS`
- semantics:
  Stores `%value` through the scalar-offset stateful unaligned-store form and returns the successor align state and successor base pointer. If `MODE` is `POST_UPDATE`, `%base_out` is the ISA-updated base; if `MODE` is `NO_POST_UPDATE`, `%base_out` preserves the incoming base.
- CCE correspondence:
  `vstus(...)`
  `__builtin_cce_vstus_*`, `__builtin_cce_vstus_post_*`

### `pto.vstur`

- syntax:
  `%align_out = pto.vstur %align_in, %value, %base, "MODE" : !pto.align, !pto.vreg<NxT>, !llvm.ptr<AS> -> !pto.align`
- operand roles:
  `%align_in` is the incoming align state, `%value` is the vector being stored, `%base` is the current base pointer, `"MODE"` selects post-update behavior, and `%align_out` is the updated align state.
- ISA family:
  `VSTUR`
- semantics:
  Stores `%value` through the register-update stateful unaligned-store form and returns the successor align state. Any architected base update is controlled by `MODE` and is not hidden from surrounding VPTO state threading.
- CCE correspondence:
  `vstur(...)`
  `__builtin_cce_vstur_*`

### Chained Usage Example

Stateful store ops make the implicit ISA update chain explicit in SSA form.
A typical sequence starts from an align-producing load-side op such as
`pto.vldas`, then threads the returned align or base values through each store.

```mlir
%align0 = pto.vldas %src[%c0] : !llvm.ptr<6> -> !pto.align
%align1, %offset1 = pto.vstu %align0, %c0, %value0, %dst, "POST_UPDATE"
    : !pto.align, index, !pto.vreg<64xf32>, !llvm.ptr<6> -> !pto.align, index
%align2, %base1 = pto.vstus %align1, %c32_i32, %value1, %dst, "POST_UPDATE"
    : !pto.align, i32, !pto.vreg<64xf32>, !llvm.ptr<6> -> !pto.align, !llvm.ptr<6>
%align3 = pto.vstur %align2, %value2, %base1, "NO_POST_UPDATE"
    : !pto.align, !pto.vreg<64xf32>, !llvm.ptr<6> -> !pto.align
```

In this form, VPTO makes the ordering and the address-state evolution visible to
verification and later lowering passes instead of leaving them as hidden side
effects on an implicit alignment register or base pointer.
