# VPTO Spec

Updated: 2026-03-20

## Table Of Contents

- [Overview](#overview)
- [Getting Started](#getting-started)
- [Example: Abs](#example-abs)
- [Scope](#scope)
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

## Core Types

- `vreg<T>`: `!pto.vreg<NxT>`
  Fixed-width VPTO vector type with total width exactly 256 bytes.
- `mask`: `!pto.mask`
  `TODO(user): extend this type entry to describe how the mask data type is represented in VPTO syntax and semantics.`
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

The table below captures the current working interpretation of `!llvm.ptr<AS>`
in this document. These meanings are based on PTO address-space enums plus the
current verifier behavior, and should be treated as `TO BE CONFIRMED` before
external publication.

| `AS` | PTO mnemonic | Working interpretation in this spec | Status |
|------|--------------|-------------------------------------|--------|
| `0` | `Zero` | Default / unspecified pointer space; currently treated as GM-like by the verifier | To be confirmed |
| `1` | `GM` | Global Memory (GM) | To be confirmed |
| `2` | `MAT` | Matrix / L1-related storage | To be confirmed |
| `3` | `LEFT` | Left matrix buffer / L0A-related storage | To be confirmed |
| `4` | `RIGHT` | Right matrix buffer / L0B-related storage | To be confirmed |
| `5` | `ACC` | Accumulator / L0C-related storage | To be confirmed |
| `6` | `VEC` | Unified Buffer (UB) / vector buffer | To be confirmed |
| `7` | `BIAS` | Bias buffer | To be confirmed |
| `8` | `SCALING` | Scaling buffer | To be confirmed |

- Current verifier rule of thumb: `!llvm.ptr<0>` and `!llvm.ptr<1>` are usually
  treated as GM-like, while `!llvm.ptr<6>` is treated as UB-like.
- `TODO(user): confirm whether external users should rely on raw numeric address
  spaces, symbolic names, or both.`

## Element Type Constraints

This section defines how placeholders such as `T`, `T0`, `T1`, and `I` should
be read throughout the spec.

- General vector rule:
  `!pto.vreg<NxT>` requires `T` to be an integer or floating-point element
  type, and `N * bitwidth(T) = 2048`.
- `T`:
  `TODO(user): summarize the intended element-type set for general arithmetic,
  logical, and load/store ops.`
- `T0`, `T1`:
  `TODO(user): list the intended legal source/result type pairs for conversion
  ops such as pto.vcvt.`
- `I`:
  `TODO(user): summarize which integer element widths are intended for offsets,
  indices, lane selectors, and permutation inputs.`
- Family-specific exceptions:
  `TODO(user): capture any op-family-specific restrictions or implementation
  subsets here.`

## Special Types

### `!pto.mask`

`!pto.mask` models an A5 predicate register, not an integer vector.

Mask data-type expression:

- `TODO(user): define how !pto.mask should carry or reference its data type; implementation pending.`

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

- `TODO(user): enumerate the legal layout literals accepted by copy ops and any
  layout-sensitive constraints.`

### `POSITION`

- `TODO(user): enumerate the legal lane-position tokens used by vdup.`

### `ORDER`

- `TODO(user): enumerate the legal order tokens used by vci.`

### `SRC_PIPE` / `DST_PIPE`

- `TODO(user): enumerate the legal pipeline names and any directionality rules
  for set_flag / wait_flag.`

### `EVENT_ID`

- `TODO(user): enumerate the legal event identifiers and any architectural
  limits or pairing rules.`

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
- semantics:
  TODO(user): add one-line semantics for external developers.
- CCE correspondence:
  `set_flag(pipe_t, pipe_t, event_t|uint64_t)`
  `__builtin_cce_set_flag`
  PTO token path:
  `__pto_set_flag`
  `__builtin_cce_tile_set_flag`

### `pto.vwait_flag`

- syntax:
  `pto.vwait_flag["SRC_PIPE", "DST_PIPE", "EVENT_ID"]`
- semantics:
  TODO(user): add one-line semantics for external developers.
- CCE correspondence:
  `wait_flag(pipe_t, pipe_t, event_t|uint64_t)`
  `__builtin_cce_wait_flag`
  PTO token path:
  `__pto_wait_flag`
  `__builtin_cce_tile_wait_flag`

### `pto.vpipe_barrier`

- syntax:
  `pto.vpipe_barrier "PIPE_*"`
- semantics:
  TODO(user): add one-line semantics for external developers.
- CCE correspondence:
  `pipe_barrier(pipe_t)`
  `__builtin_cce_pipe_barrier`

### `pto.vget_buf`

- syntax:
  `pto.vget_buf "PIPE_*", %buf_id, %mode : i64, i64`
- semantics:
  TODO(user): add one-line semantics for external developers.
- CCE correspondence:
  `get_buf(pipe_t, uint8_t|uint64_t, bool)`
  `__builtin_cce_get_buf`

### `pto.vrls_buf`

- syntax:
  `pto.vrls_buf "PIPE_*", %buf_id, %mode : i64, i64`
- semantics:
  TODO(user): add one-line semantics for external developers.
- CCE correspondence:
  `rls_buf(pipe_t, uint8_t|uint64_t, bool)`
  `__builtin_cce_rls_buf`

## 2. Copy Programming

### `pto.vset_loop2_stride_outtoub`

- syntax:
  `pto.vset_loop2_stride_outtoub %first, %second : i64, i64`
- semantics:
  TODO(user): add one-line semantics for external developers.
- CCE correspondence:
  `set_loop2_stride_outtoub(uint64_t)`
  `__builtin_cce_set_loop2_stride_outtoub`

### `pto.vset_loop1_stride_outtoub`

- syntax:
  `pto.vset_loop1_stride_outtoub %first, %second : i64, i64`
- semantics:
  TODO(user): add one-line semantics for external developers.
- CCE correspondence:
  `set_loop1_stride_outtoub(uint64_t)`
  `__builtin_cce_set_loop1_stride_outtoub`

### `pto.vset_loop_size_outtoub`

- syntax:
  `pto.vset_loop_size_outtoub %first, %second : i64, i64`
- semantics:
  TODO(user): add one-line semantics for external developers.
- CCE correspondence:
  `set_loop_size_outtoub(uint64_t)`
  `__builtin_cce_set_loop_size_outtoub`

### `pto.vset_loop2_stride_ubtoout`

- syntax:
  `pto.vset_loop2_stride_ubtoout %first, %second : i64, i64`
- semantics:
  TODO(user): add one-line semantics for external developers.
- CCE correspondence:
  `set_loop2_stride_ubtoout(uint64_t)`
  `__builtin_cce_set_loop2_stride_ubtoout`

### `pto.vset_loop1_stride_ubtoout`

- syntax:
  `pto.vset_loop1_stride_ubtoout %first, %second : i64, i64`
- semantics:
  TODO(user): add one-line semantics for external developers.
- CCE correspondence:
  `set_loop1_stride_ubtoout(uint64_t)`
  `__builtin_cce_set_loop1_stride_ubtoout`

### `pto.vset_loop_size_ubtoout`

- syntax:
  `pto.vset_loop_size_ubtoout %first, %second : i64, i64`
- semantics:
  TODO(user): add one-line semantics for external developers.
- CCE correspondence:
  `set_loop_size_ubtoout(uint64_t)`
  `__builtin_cce_set_loop_size_ubtoout`

## 3. Copy Transfers

### `pto.vcopy_gm_to_ubuf`

- syntax:
  `pto.vcopy_gm_to_ubuf %source, %destination, %valid_rows, %valid_cols, %sid, %n_burst, %len_burst, %left_padding_count, %right_padding_count, %l2_cache_ctl, %gm_stride, %ub_stride {layout = "LAYOUT", data_select_bit = true|false, ub_pad = true|false} : !llvm.ptr<AS>, !llvm.ptr<AS>, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64`
- semantics:
  TODO(user): add one-line semantics for external developers.
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
- semantics:
  TODO(user): add one-line semantics for external developers.
- CCE correspondence:
  `copy_ubuf_to_ubuf(...)`
  `__builtin_cce_copy_ubuf_to_ubuf`

### `pto.vcopy_ubuf_to_gm`

- syntax:
  `pto.vcopy_ubuf_to_gm %source, %destination, %valid_rows, %valid_cols, %sid, %n_burst, %len_burst, %reserved, %burst_dst_stride, %burst_src_stride {layout = "LAYOUT"} : !llvm.ptr<AS>, !llvm.ptr<AS>, i64, i64, i64, i64, i64, i64, i64, i64`
- semantics:
  TODO(user): add one-line semantics for external developers.
- CCE correspondence:
  `copy_ubuf_to_gm(...)`
  PTO A5 path commonly uses `copy_ubuf_to_gm_align_v2(...)`
  `__builtin_cce_copy_ubuf_to_gm_align_v2`
  composed loop intrinsics:
  `__builtin_cce_set_loop2_stride_ubtoout`
  `__builtin_cce_set_loop1_stride_ubtoout`
  `__builtin_cce_set_loop_size_ubtoout`

## 4. Vector, Predicate And Align Loads

### `pto.vlds`

- syntax:
  `%result = pto.vlds %source[%offset] {dist = "DIST"} : !llvm.ptr<AS> -> !pto.vreg<NxT>`
- semantics:
  TODO(user): add one-line semantics for external developers.
- CCE correspondence:
  `vld(...)`, `vlds(...)`
  `__builtin_cce_vldsx1_*`
  related extended families:
  `__builtin_cce_vldix1_*`, `__builtin_cce_vldsx1_post_*`

### `pto.vldas`

- syntax:
  `%result = pto.vldas %source[%offset] : !llvm.ptr<AS> -> !pto.align`
- semantics:
  TODO(user): add one-line semantics for external developers.
- CCE correspondence:
  `vldas(...)`
  `__builtin_cce_vldas_*`

### `pto.vldus`

- syntax:
  `%result = pto.vldus %align, %source[%offset] : !pto.align, !llvm.ptr<AS> -> !pto.vreg<NxT>`
- semantics:
  TODO(user): add one-line semantics for external developers.
- CCE correspondence:
  `vldus(...)`
  `__builtin_cce_vldus_*`, `__builtin_cce_vldus_post_*`

### `pto.vplds`

- syntax:
  `%result = pto.vplds %source[%offset] {dist = "DIST"} : !llvm.ptr<AS> -> !pto.mask`
- semantics:
  TODO(user): add one-line semantics for external developers.
- CCE correspondence:
  `plds(...)`
  `__builtin_cce_plds_b8`

### `pto.vpld`

- syntax:
  `%result = pto.vpld %source[%offset], "DIST" : !llvm.ptr<AS>, index -> !pto.mask`
- semantics:
  TODO(user): add one-line semantics for external developers.
- CCE correspondence:
  `pld(...)`
  `__builtin_cce_pld_b8`

### `pto.vpldi`

- syntax:
  `%result = pto.vpldi %source, %offset, "DIST" : !llvm.ptr<AS>, i32 -> !pto.mask`
- semantics:
  TODO(user): add one-line semantics for external developers.
- CCE correspondence:
  `pldi(...)`
  `__builtin_cce_pldi_b8`, `__builtin_cce_pldi_post_b8`

### `pto.vldx2`

- syntax:
  `%low, %high = pto.vldx2 %source[%offset], "DIST" : !llvm.ptr<AS>, index -> !pto.vreg<NxT>, !pto.vreg<NxT>`
- semantics:
  TODO(user): add one-line semantics for external developers.
- CCE correspondence:
  `vld(...)`
  `__builtin_cce_vldx2_*`

### `pto.vgather2`

- syntax:
  `%result = pto.vgather2 %source, %offsets, %active_lanes : !llvm.ptr<AS>, !pto.vreg<NxI>, index -> !pto.vreg<NxT>`
- semantics:
  TODO(user): add one-line semantics for external developers.
- CCE correspondence:
  `vgather2(...)`
  `__builtin_cce_vgather2_*`, `__builtin_cce_vgather2_v300_*`

### `pto.vgatherb`

- syntax:
  `%result = pto.vgatherb %source, %offsets, %active_lanes : !llvm.ptr<AS>, !pto.vreg<NxI>, index -> !pto.vreg<NxT>`
- semantics:
  TODO(user): add one-line semantics for external developers.
- CCE correspondence:
  `vgatherb(...)`
  `__builtin_cce_vgatherb_*`, `__builtin_cce_vgatherb_v300_*`, `__builtin_cce_vgatherb_v310_*`

### `pto.vgather2_bc`

- syntax:
  `%result = pto.vgather2_bc %source, %offsets, %mask : !llvm.ptr<AS>, !pto.vreg<NxI>, !pto.mask -> !pto.vreg<NxT>`
- semantics:
  TODO(user): add one-line semantics for external developers.
- CCE correspondence:
  `vgather2_bc(...)`
  `__builtin_cce_vgather2_bc_*`

### `pto.vsld`

- syntax:
  `%result = pto.vsld %source[%offset], "STRIDE" : !llvm.ptr<AS> -> !pto.vreg<NxT>`
- semantics:
  TODO(user): add one-line semantics for external developers.
- CCE correspondence:
  `vsld(...)`
  `__builtin_cce_vsld_*`

### `pto.vsldb`

- syntax:
  `%result = pto.vsldb %source, %offset, %mask : !llvm.ptr<AS>, i32, !pto.mask -> !pto.vreg<NxT>`
- semantics:
  TODO(user): add one-line semantics for external developers.
- CCE correspondence:
  `vsldb(...)`
  `__builtin_cce_vsldb_*`, `__builtin_cce_vsldb_post_*`

## 5. Materialization And Predicate Construction

### `pto.vbr`

- syntax:
  `%result = pto.vbr %value : T -> !pto.vreg<NxT>`
- semantics:
  TODO(user): add one-line semantics for external developers.
- CCE correspondence:
  broadcast/materialization family used by PTO scalar-to-vector expansion

### `pto.vdup`

- syntax:
  `%result = pto.vdup %input {position = "POSITION", mode = "MODE"} : T|!pto.vreg<NxT> -> !pto.vreg<NxT>`
- semantics:
  TODO(user): add one-line semantics for external developers.
- CCE correspondence:
  `vdup(...)`
  `__builtin_cce_vdup_*`

### `pto.vpset_b8`

- syntax:
  `%result = pto.vpset_b8 "PAT_*" : !pto.mask`
- semantics:
  TODO(user): add one-line semantics for external developers.
- CCE correspondence:
  `pset_b8(...)`
  `__builtin_cce_pset_b8`

### `pto.vpset_b16`

- syntax:
  `%result = pto.vpset_b16 "PAT_*" : !pto.mask`
- semantics:
  TODO(user): add one-line semantics for external developers.
- CCE correspondence:
  `pset_b16(...)`
  `__builtin_cce_pset_b16`

### `pto.vpset_b32`

- syntax:
  `%result = pto.vpset_b32 "PAT_*" : !pto.mask`
- semantics:
  TODO(user): add one-line semantics for external developers.
- CCE correspondence:
  `pset_b32(...)`
  `__builtin_cce_pset_b32`

### `pto.vpge_b8`

- syntax:
  `%result = pto.vpge_b8 "PAT_*" : !pto.mask`
- semantics:
  TODO(user): add one-line semantics for external developers.
- CCE correspondence:
  `pge_b8(...)`
  `__builtin_cce_pge_b8`

### `pto.vpge_b16`

- syntax:
  `%result = pto.vpge_b16 "PAT_*" : !pto.mask`
- semantics:
  TODO(user): add one-line semantics for external developers.
- CCE correspondence:
  `pge_b16(...)`
  `__builtin_cce_pge_b16`

### `pto.vpge_b32`

- syntax:
  `%result = pto.vpge_b32 "PAT_*" : !pto.mask`
- semantics:
  TODO(user): add one-line semantics for external developers.
- CCE correspondence:
  `pge_b32(...)`
  `__builtin_cce_pge_b32`

### `pto.vppack`

- syntax:
  `%result = pto.vppack %input, "PART" : !pto.mask -> !pto.mask`
- semantics:
  TODO(user): add one-line semantics for external developers.
- CCE correspondence:
  `ppack(...)`

### `pto.vpunpack`

- syntax:
  `%result = pto.vpunpack %input, "PART" : !pto.mask -> !pto.mask`
- semantics:
  TODO(user): add one-line semantics for external developers.
- CCE correspondence:
  `punpack(...)`

## 6. Unary Vector Ops

### `pto.vabs`

- syntax:
  `%result = pto.vabs %input : !pto.vreg<NxT> -> !pto.vreg<NxT>`
- semantics:
  TODO(user): add one-line semantics for external developers.
- CCE correspondence:
  `vabs(...)`
  `__builtin_cce_vabs_*`

### `pto.vexp`

- syntax:
  `%result = pto.vexp %input : !pto.vreg<NxT> -> !pto.vreg<NxT>`
- semantics:
  TODO(user): add one-line semantics for external developers.
- CCE correspondence:
  `vexp(...)`
  `__builtin_cce_vexp_*`

### `pto.vln`

- syntax:
  `%result = pto.vln %input : !pto.vreg<NxT> -> !pto.vreg<NxT>`
- semantics:
  TODO(user): add one-line semantics for external developers.
- CCE correspondence:
  `vln(...)`
  `__builtin_cce_vln_*`

### `pto.vsqrt`

- syntax:
  `%result = pto.vsqrt %input : !pto.vreg<NxT> -> !pto.vreg<NxT>`
- semantics:
  TODO(user): add one-line semantics for external developers.
- CCE correspondence:
  `vsqrt(...)`
  `__builtin_cce_vsqrt_*`

### `pto.vrec`

- syntax:
  `%result = pto.vrec %input : !pto.vreg<NxT> -> !pto.vreg<NxT>`
- semantics:
  TODO(user): add one-line semantics for external developers.
- CCE correspondence:
  `vrec(...)`
  `__builtin_cce_vrec_*`

### `pto.vrelu`

- syntax:
  `%result = pto.vrelu %input : !pto.vreg<NxT> -> !pto.vreg<NxT>`
- semantics:
  TODO(user): add one-line semantics for external developers.
- CCE correspondence:
  `vrelu(...)`
  `__builtin_cce_vrelu_*`

### `pto.vnot`

- syntax:
  `%result = pto.vnot %input : !pto.vreg<NxT> -> !pto.vreg<NxT>`
- semantics:
  TODO(user): add one-line semantics for external developers.
- CCE correspondence:
  `vnot(...)`
  `__builtin_cce_vnot_*`

### `pto.vcadd`

- syntax:
  `%result = pto.vcadd %input : !pto.vreg<NxT> -> !pto.vreg<NxT>`
- semantics:
  TODO(user): add one-line semantics for external developers.
- CCE correspondence:
  `vcadd(...)`
  `__builtin_cce_vcadd_*`

### `pto.vcmax`

- syntax:
  `%result = pto.vcmax %input : !pto.vreg<NxT> -> !pto.vreg<NxT>`
- semantics:
  TODO(user): add one-line semantics for external developers.
- CCE correspondence:
  `vcmax(...)`
  `__builtin_cce_vcmax_*`

### `pto.vcmin`

- syntax:
  `%result = pto.vcmin %input : !pto.vreg<NxT> -> !pto.vreg<NxT>`
- semantics:
  TODO(user): add one-line semantics for external developers.
- CCE correspondence:
  `vcmin(...)`
  `__builtin_cce_vcmin_*`

### `pto.vbcnt`

- syntax:
  `%result = pto.vbcnt %input : !pto.vreg<NxT> -> !pto.vreg<NxT>`
- semantics:
  TODO(user): add one-line semantics for external developers.
- CCE correspondence:
  `vbcnt(...)`
  `__builtin_cce_vbcnt_*`

### `pto.vcls`

- syntax:
  `%result = pto.vcls %input : !pto.vreg<NxT> -> !pto.vreg<NxT>`
- semantics:
  TODO(user): add one-line semantics for external developers.
- CCE correspondence:
  `vcls(...)`
  `__builtin_cce_vcls_*`

## 7. Binary Vector Ops

### `pto.vadd`

- syntax:
  `%result = pto.vadd %lhs, %rhs : !pto.vreg<NxT>, !pto.vreg<NxT> -> !pto.vreg<NxT>`
- semantics:
  TODO(user): add one-line semantics for external developers.
- CCE correspondence:
  `vadd(...)`
  `__builtin_cce_vadd_*`

### `pto.vsub`

- syntax:
  `%result = pto.vsub %lhs, %rhs : !pto.vreg<NxT>, !pto.vreg<NxT> -> !pto.vreg<NxT>`
- semantics:
  TODO(user): add one-line semantics for external developers.
- CCE correspondence:
  `vsub(...)`
  `__builtin_cce_vsub_*`

### `pto.vmul`

- syntax:
  `%result = pto.vmul %lhs, %rhs : !pto.vreg<NxT>, !pto.vreg<NxT> -> !pto.vreg<NxT>`
- semantics:
  TODO(user): add one-line semantics for external developers.
- CCE correspondence:
  `vmul(...)`
  `__builtin_cce_vmul_*`

### `pto.vdiv`

- syntax:
  `%result = pto.vdiv %lhs, %rhs : !pto.vreg<NxT>, !pto.vreg<NxT> -> !pto.vreg<NxT>`
- semantics:
  TODO(user): add one-line semantics for external developers.
- CCE correspondence:
  `vdiv(...)`
  `__builtin_cce_vdiv_*`

### `pto.vmax`

- syntax:
  `%result = pto.vmax %lhs, %rhs : !pto.vreg<NxT>, !pto.vreg<NxT> -> !pto.vreg<NxT>`
- semantics:
  TODO(user): add one-line semantics for external developers.
- CCE correspondence:
  `vmax(...)`
  `__builtin_cce_vmax_*`

### `pto.vmin`

- syntax:
  `%result = pto.vmin %lhs, %rhs : !pto.vreg<NxT>, !pto.vreg<NxT> -> !pto.vreg<NxT>`
- semantics:
  TODO(user): add one-line semantics for external developers.
- CCE correspondence:
  `vmin(...)`
  `__builtin_cce_vmin_*`

### `pto.vand`

- syntax:
  `%result = pto.vand %lhs, %rhs : !pto.vreg<NxT>, !pto.vreg<NxT> -> !pto.vreg<NxT>`
- semantics:
  TODO(user): add one-line semantics for external developers.
- CCE correspondence:
  `vand(...)`
  `__builtin_cce_vand_*`

### `pto.vor`

- syntax:
  `%result = pto.vor %lhs, %rhs : !pto.vreg<NxT>, !pto.vreg<NxT> -> !pto.vreg<NxT>`
- semantics:
  TODO(user): add one-line semantics for external developers.
- CCE correspondence:
  `vor(...)`
  `__builtin_cce_vor_*`

### `pto.vxor`

- syntax:
  `%result = pto.vxor %lhs, %rhs : !pto.vreg<NxT>, !pto.vreg<NxT> -> !pto.vreg<NxT>`
- semantics:
  TODO(user): add one-line semantics for external developers.
- CCE correspondence:
  `vxor(...)`
  `__builtin_cce_vxor_*`

### `pto.vshl`

- syntax:
  `%result = pto.vshl %lhs, %rhs : !pto.vreg<NxT>, !pto.vreg<NxT> -> !pto.vreg<NxT>`
- semantics:
  TODO(user): add one-line semantics for external developers.
- CCE correspondence:
  `vshl(...)`
  `__builtin_cce_vshl_*`

### `pto.vshr`

- syntax:
  `%result = pto.vshr %lhs, %rhs : !pto.vreg<NxT>, !pto.vreg<NxT> -> !pto.vreg<NxT>`
- semantics:
  TODO(user): add one-line semantics for external developers.
- CCE correspondence:
  `vshr(...)`
  `__builtin_cce_vshr_*`

## 8. Vec-Scalar Ops

### `pto.vmuls`

- syntax:
  `%result = pto.vmuls %input, %scalar : !pto.vreg<NxT>, T -> !pto.vreg<NxT>`
- semantics:
  TODO(user): add one-line semantics for external developers.
- CCE correspondence:
  `vmuls(...)`
  `__builtin_cce_vmuls_*`

### `pto.vadds`

- syntax:
  `%result = pto.vadds %input, %scalar : !pto.vreg<NxT>, T -> !pto.vreg<NxT>`
- semantics:
  TODO(user): add one-line semantics for external developers.
- CCE correspondence:
  `vadds(...)`
  `__builtin_cce_vadds_*`

### `pto.vmaxs`

- syntax:
  `%result = pto.vmaxs %input, %scalar : !pto.vreg<NxT>, T -> !pto.vreg<NxT>`
- semantics:
  TODO(user): add one-line semantics for external developers.
- CCE correspondence:
  `vmaxs(...)`
  `__builtin_cce_vmaxs_*`

### `pto.vmins`

- syntax:
  `%result = pto.vmins %input, %scalar : !pto.vreg<NxT>, T -> !pto.vreg<NxT>`
- semantics:
  TODO(user): add one-line semantics for external developers.
- CCE correspondence:
  `vmins(...)`
  `__builtin_cce_vmins_*`

### `pto.vlrelu`

- syntax:
  `%result = pto.vlrelu %input, %scalar : !pto.vreg<NxT>, T -> !pto.vreg<NxT>`
- semantics:
  TODO(user): add one-line semantics for external developers.
- CCE correspondence:
  `vlrelu(...)`
  `__builtin_cce_vlrelu_*`

### `pto.vshls`

- syntax:
  `%result = pto.vshls %input, %scalar : !pto.vreg<NxT>, T -> !pto.vreg<NxT>`
- semantics:
  TODO(user): add one-line semantics for external developers.
- CCE correspondence:
  `vshls(...)`
  `__builtin_cce_vshls_*`

### `pto.vshrs`

- syntax:
  `%result = pto.vshrs %input, %scalar : !pto.vreg<NxT>, T -> !pto.vreg<NxT>`
- semantics:
  TODO(user): add one-line semantics for external developers.
- CCE correspondence:
  `vshrs(...)`
  `__builtin_cce_vshrs_*`

## 9. Carry, Compare And Select

### `pto.vaddc`

- syntax:
  `%result, %carry = pto.vaddc %lhs, %rhs, %mask : !pto.vreg<NxT>, !pto.vreg<NxT>, !pto.mask -> !pto.vreg<NxT>, !pto.mask`
- semantics:
  TODO(user): add one-line semantics for external developers.
- CCE correspondence:
  `vaddc(...)`
  `__builtin_cce_vaddc_*`

### `pto.vsubc`

- syntax:
  `%result, %carry = pto.vsubc %lhs, %rhs, %mask : !pto.vreg<NxT>, !pto.vreg<NxT>, !pto.mask -> !pto.vreg<NxT>, !pto.mask`
- semantics:
  TODO(user): add one-line semantics for external developers.
- CCE correspondence:
  `vsubc(...)`
  `__builtin_cce_vsubc_*`

### `pto.vaddcs`

- syntax:
  `%result, %carry = pto.vaddcs %lhs, %rhs, %carry_in, %mask : !pto.vreg<NxT>, !pto.vreg<NxT>, !pto.mask, !pto.mask -> !pto.vreg<NxT>, !pto.mask`
- semantics:
  TODO(user): add one-line semantics for external developers.
- CCE correspondence:
  `vaddcs(...)`
  `__builtin_cce_vaddcs_*`

### `pto.vsubcs`

- syntax:
  `%result, %carry = pto.vsubcs %lhs, %rhs, %carry_in, %mask : !pto.vreg<NxT>, !pto.vreg<NxT>, !pto.mask, !pto.mask -> !pto.vreg<NxT>, !pto.mask`
- semantics:
  TODO(user): add one-line semantics for external developers.
- CCE correspondence:
  `vsubcs(...)`
  `__builtin_cce_vsubcs_*`

### `pto.vsel`

- syntax:
  `%result = pto.vsel %src0, %src1, %mask : !pto.vreg<NxT>, !pto.vreg<NxT>, !pto.mask -> !pto.vreg<NxT>`
- semantics:
  TODO(user): add one-line semantics for external developers.
- CCE correspondence:
  `vsel(...)`
  `__builtin_cce_vsel_*`

### `pto.vselr`

- syntax:
  `%result = pto.vselr %src0, %src1 : !pto.vreg<NxT>, !pto.vreg<NxI> -> !pto.vreg<NxT>`
- semantics:
  TODO(user): add one-line semantics for external developers.
- CCE correspondence:
  `vselr(...)`
  `__builtin_cce_vselr_*`

### `pto.vselrv2`

- syntax:
  `%result = pto.vselrv2 %src0, %src1 : !pto.vreg<NxT>, !pto.vreg<NxI> -> !pto.vreg<NxT>`
- semantics:
  TODO(user): add one-line semantics for external developers.
- CCE correspondence:
  `vselrv2(...)`
  `__builtin_cce_vselrv2_*`

### `pto.vcmp`

- syntax:
  `%result = pto.vcmp %src0, %src1, %mask, "CMP_MODE" : !pto.vreg<NxT>, !pto.vreg<NxT>, !pto.mask -> !pto.mask`
- semantics:
  TODO(user): add one-line semantics for external developers.
- CCE correspondence:
  `vcmp(...)`
  `__builtin_cce_vcmp_<op>_*_z`

### `pto.vcmps`

- syntax:
  `%result = pto.vcmps %src, %scalar, %mask, "CMP_MODE" : !pto.vreg<NxT>, T, !pto.mask -> !pto.mask`
- semantics:
  TODO(user): add one-line semantics for external developers.
- CCE correspondence:
  `vcmps(...)`
  `__builtin_cce_vcmps_<op>_*_z`

### `pto.vpnot`

- syntax:
  `%result = pto.vpnot %input, %mask : !pto.mask, !pto.mask -> !pto.mask`
- semantics:
  TODO(user): add one-line semantics for external developers.
- CCE correspondence:
  `pnot(...)`

### `pto.vpsel`

- syntax:
  `%result = pto.vpsel %src0, %src1, %mask : !pto.mask, !pto.mask, !pto.mask -> !pto.mask`
- semantics:
  TODO(user): add one-line semantics for external developers.
- CCE correspondence:
  `psel(...)`

## 10. Pairing And Interleave

### `pto.vpdintlv_b8`

- syntax:
  `%low, %high = pto.vpdintlv_b8 %lhs, %rhs : !pto.mask, !pto.mask -> !pto.mask, !pto.mask`
- semantics:
  TODO(user): add one-line semantics for external developers.
- CCE correspondence:
  predicate interleave/deinterleave family

### `pto.vpintlv_b16`

- syntax:
  `%low, %high = pto.vpintlv_b16 %lhs, %rhs : !pto.mask, !pto.mask -> !pto.mask, !pto.mask`
- semantics:
  TODO(user): add one-line semantics for external developers.
- CCE correspondence:
  predicate interleave/deinterleave family

### `pto.vintlv`

- syntax:
  `%low, %high = pto.vintlv %lhs, %rhs : !pto.vreg<NxT>, !pto.vreg<NxT> -> !pto.vreg<NxT>, !pto.vreg<NxT>`
- semantics:
  TODO(user): add one-line semantics for external developers.
- CCE correspondence:
  `vintlv(...)`
  `__builtin_cce_vintlv_*`

### `pto.vdintlv`

- syntax:
  `%low, %high = pto.vdintlv %lhs, %rhs : !pto.vreg<NxT>, !pto.vreg<NxT> -> !pto.vreg<NxT>, !pto.vreg<NxT>`
- semantics:
  TODO(user): add one-line semantics for external developers.
- CCE correspondence:
  `vdintlv(...)`
  `__builtin_cce_vdintlv_*`

### `pto.vintlvv2`

- syntax:
  `%result = pto.vintlvv2 %lhs, %rhs, "PART" : !pto.vreg<NxT>, !pto.vreg<NxT> -> !pto.vreg<NxT>`
- semantics:
  TODO(user): add one-line semantics for external developers.
- CCE correspondence:
  `vintlvv2(...)`
  `__builtin_cce_vintlvv2_*`

### `pto.vdintlvv2`

- syntax:
  `%result = pto.vdintlvv2 %lhs, %rhs, "PART" : !pto.vreg<NxT>, !pto.vreg<NxT> -> !pto.vreg<NxT>`
- semantics:
  TODO(user): add one-line semantics for external developers.
- CCE correspondence:
  `vdintlvv2(...)`
  `__builtin_cce_vdintlvv2_*`

## 11. Conversion, Index And Sort

### `pto.vtrc`

- syntax:
  `%result = pto.vtrc %input, "ROUND_MODE" : !pto.vreg<NxT> -> !pto.vreg<NxT>`
- semantics:
  TODO(user): add one-line semantics for external developers.
- CCE correspondence:
  `vtrc(...)`
  `__builtin_cce_vtrc_*`

### `pto.vcvt`

- syntax:
  `%result = pto.vcvt %input {round_mode = "ROUND_MODE", sat = "SAT_MODE", part = "PART_MODE"} : !pto.vreg<NxT0> -> !pto.vreg<NxT1>`
- semantics:
  TODO(user): add one-line semantics for external developers.
- CCE correspondence:
  `vcvt(...)`
  builtin families:
  `__builtin_cce_vcvt*`, `__builtin_cce_vcvtfi_*`, `__builtin_cce_vcvtif_*`, `__builtin_cce_vcvtii_*`, `__builtin_cce_vcvtff_*`

### `pto.vci`

- syntax:
  `%result = pto.vci %index {order = "ORDER"} : integer -> !pto.vreg<NxT>`
- semantics:
  TODO(user): add one-line semantics for external developers.
- CCE correspondence:
  `vci(...)`
  `__builtin_cce_vci_*`

### `pto.vbitsort`

- syntax:
  `pto.vbitsort %destination, %source, %indices, %repeat_times : !llvm.ptr<AS>, !llvm.ptr<AS>, !llvm.ptr<AS>, index`
- semantics:
  TODO(user): add one-line semantics for external developers.
- CCE correspondence:
  `vbitsort(...)`
  `__builtin_cce_vbitsort_*`

### `pto.vmrgsort4`

- syntax:
  `pto.vmrgsort4 %destination, %source0, %source1, %source2, %source3, %count, %config : !llvm.ptr<AS>, !llvm.ptr<AS>, !llvm.ptr<AS>, !llvm.ptr<AS>, !llvm.ptr<AS>, i64, i64`
- semantics:
  TODO(user): add one-line semantics for external developers.
- CCE correspondence:
  `vmrgsort4(...)`
  `__builtin_cce_vmrgsort4_*`

## 12. Extended Arithmetic

### `pto.vmull`

- syntax:
  `%low, %high = pto.vmull %lhs, %rhs, %mask : !pto.vreg<NxT>, !pto.vreg<NxT>, !pto.mask -> !pto.vreg<NxT>, !pto.vreg<NxT>`
- semantics:
  TODO(user): add one-line semantics for external developers.
- CCE correspondence:
  `vmull(...)`
  `__builtin_cce_vmull_*`

### `pto.vmula`

- syntax:
  `%result = pto.vmula %acc, %lhs, %rhs, %mask {mode = "MODE"} : !pto.vreg<NxT>, !pto.vreg<NxT>, !pto.vreg<NxT>, !pto.mask -> !pto.vreg<NxT>`
- semantics:
  TODO(user): add one-line semantics for external developers.
- CCE correspondence:
  `vmula(...)`
  `__builtin_cce_vmula_*_m`

## 13. Stateless Stores

### `pto.vsts`

- syntax:
  `pto.vsts %value, %destination[%offset] {dist = "DIST"} : !pto.vreg<NxT>, !llvm.ptr<AS>`
- semantics:
  TODO(user): add one-line semantics for external developers.
- CCE correspondence:
  `vst(...)`, `vsts(...)`
  `__builtin_cce_vstx1_*`, `__builtin_cce_vstsx1_*`

### `pto.vscatter`

- syntax:
  `pto.vscatter %value, %destination, %offsets, %active_lanes : !pto.vreg<NxT>, !llvm.ptr<AS>, !pto.vreg<NxI>, index`
- semantics:
  TODO(user): add one-line semantics for external developers.
- CCE correspondence:
  `vscatter(...)`
  `__builtin_cce_vscatter_*`

### `pto.vsts_pred`

- syntax:
  `pto.vsts_pred %value, %destination[%offset], %active_lanes {dist = "DIST"} : !pto.vreg<NxT>, !llvm.ptr<AS>, index`
- semantics:
  TODO(user): add one-line semantics for external developers.
- CCE correspondence:
  predicated vector store family

### `pto.vpsts`

- syntax:
  `pto.vpsts %value, %destination[%offset] : !pto.mask, !llvm.ptr<AS>`
- semantics:
  TODO(user): add one-line semantics for external developers.
- CCE correspondence:
  `psts(...)`
  `__builtin_cce_psts_b8`, `__builtin_cce_psts_post_b8`

### `pto.vpst`

- syntax:
  `pto.vpst %value, %destination[%offset], "DIST" : !pto.mask, !llvm.ptr<AS>, index`
- semantics:
  TODO(user): add one-line semantics for external developers.
- CCE correspondence:
  `pst(...)`
  `__builtin_cce_pst_b8`

### `pto.vpsti`

- syntax:
  `pto.vpsti %value, %destination, %offset, "DIST" : !pto.mask, !llvm.ptr<AS>, i32`
- semantics:
  TODO(user): add one-line semantics for external developers.
- CCE correspondence:
  `psti(...)`
  `__builtin_cce_psti_b8`, `__builtin_cce_psti_post_b8`

### `pto.vsst`

- syntax:
  `pto.vsst %value, %destination[%offset], "STRIDE" : !pto.vreg<NxT>, !llvm.ptr<AS>`
- semantics:
  TODO(user): add one-line semantics for external developers.
- CCE correspondence:
  `vsst(...)`
  `__builtin_cce_vsst_*`

### `pto.vstx2`

- syntax:
  `pto.vstx2 %low, %high, %destination[%offset], "DIST", %mask : !pto.vreg<NxT>, !pto.vreg<NxT>, !llvm.ptr<AS>, index, !pto.mask`
- semantics:
  TODO(user): add one-line semantics for external developers.
- CCE correspondence:
  `vst(...)`
  `__builtin_cce_vstx2_*`

### `pto.vsstb`

- syntax:
  `pto.vsstb %value, %destination, %offset, %mask : !pto.vreg<NxT>, !llvm.ptr<AS>, i32, !pto.mask`
- semantics:
  TODO(user): add one-line semantics for external developers.
- CCE correspondence:
  `vsstb(...)`
  `__builtin_cce_vsstb_*`, `__builtin_cce_vsstb_post_*`

### `pto.vsta`

- syntax:
  `pto.vsta %value, %destination[%offset] : !pto.align, !llvm.ptr<AS>, index`
- semantics:
  TODO(user): add one-line semantics for external developers.
- CCE correspondence:
  `vsta(...)`
  `__builtin_cce_vsta_*`

### `pto.vstas`

- syntax:
  `pto.vstas %value, %destination, %offset : !pto.align, !llvm.ptr<AS>, i32`
- semantics:
  TODO(user): add one-line semantics for external developers.
- CCE correspondence:
  `vstas(...)`
  `__builtin_cce_vstas_*`, `__builtin_cce_vstas_post_*`

### `pto.vstar`

- syntax:
  `pto.vstar %value, %destination : !pto.align, !llvm.ptr<AS>`
- semantics:
  TODO(user): add one-line semantics for external developers.
- CCE correspondence:
  `vstar(...)`
  `__builtin_cce_vstar_*`

## 14. Stateful Store Ops

These ops make CCE reference-updated state explicit as SSA results.

### `pto.vpstu`

- syntax:
  `%align_out, %base_out = pto.vpstu %align_in, %value, %base : !pto.align, !pto.mask, !llvm.ptr<AS> -> !pto.align, !llvm.ptr<AS>`
- semantics:
  TODO(user): add one-line semantics for external developers.
- CCE correspondence:
  `pstu(...)`
  `__builtin_cce_pstu_b16`, `__builtin_cce_pstu_b32`

### `pto.vstu`

- syntax:
  `%align_out, %offset_out = pto.vstu %align_in, %offset_in, %value, %base, "MODE" : !pto.align, index, !pto.vreg<NxT>, !llvm.ptr<AS> -> !pto.align, index`
- semantics:
  TODO(user): add one-line semantics for external developers.
- CCE correspondence:
  `vstu(...)`
  `__builtin_cce_vstu_*`

### `pto.vstus`

- syntax:
  `%align_out, %base_out = pto.vstus %align_in, %offset, %value, %base, "MODE" : !pto.align, i32, !pto.vreg<NxT>, !llvm.ptr<AS> -> !pto.align, !llvm.ptr<AS>`
- semantics:
  TODO(user): add one-line semantics for external developers.
- CCE correspondence:
  `vstus(...)`
  `__builtin_cce_vstus_*`, `__builtin_cce_vstus_post_*`

### `pto.vstur`

- syntax:
  `%align_out = pto.vstur %align_in, %value, %base, "MODE" : !pto.align, !pto.vreg<NxT>, !llvm.ptr<AS> -> !pto.align`
- semantics:
  TODO(user): add one-line semantics for external developers.
- CCE correspondence:
  `vstur(...)`
  `__builtin_cce_vstur_*`

### Chained Usage Example

This subsection is intentionally reserved for a full end-to-end stateful-store
example.

- `TODO(user): add a complete chained example that threads %align_out,
  %base_out, and %offset_out across multiple stateful store ops.`
- `TODO(user): show how the stateful-store chain interacts with vldas / vldus
  and with surrounding vector-scope structure.`

```mlir
// TODO(user): replace this skeleton with a complete chained stateful-store example.
%align0 = ...
%value0 = ...
%base0 = ...

// %align1, %offset1 = pto.vstu ...
// %align2, %base1 = pto.vstus ...
// %align3 = pto.vstur ...
```
