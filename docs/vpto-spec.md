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

This section is a placeholder for the external-developer-facing introduction.

- PTO vector ISA background:
  `TODO(user): explain where VPTO fits in the PTO stack, which layer it models, and why an external developer would read or author this IR directly.`
- Relationship to CCE:
  `TODO(user): compare VPTO ops with CCE builtins / wrapper APIs, including what is preserved, renamed, or made explicit in SSA form.`
- Intended audience and non-goals:
  `TODO(user): explain what this document expects from external developers and what remains compiler-internal.`

## Getting Started

This section is a scaffold for a future newcomer-oriented walkthrough.

1. Start from buffer-like LLVM pointers and identify the relevant address spaces for GM, UB, and any other storage involved.
2. Use copy-programming and copy-transfer ops to stage data into UB when the workflow requires it.
3. Enter `__VEC_SCOPE__` for vector execution.
4. Use load / compute / store ops inside the vector-scope body.
5. Use sync ops and copy-back ops to publish results.

- `TODO(user): add a fuller end-to-end walkthrough for first-time external developers.`

## Example: Abs

Example file:
[build/vpto-doc-abs/Abs/abs-pto.cpp](/data/mouliangyu/projects/github.com/zhangstevenunity/PTOAS/build/vpto-doc-abs/Abs/abs-pto.cpp)

Representative excerpt:

```mlir
vpto.set_loop2_stride_outtoub %c4096_i64, %c4096_i64 : i64, i64
vpto.set_loop1_stride_outtoub %c4096_i64, %c4096_i64 : i64, i64
vpto.set_loop_size_outtoub %c1_i64, %c1_i64 : i64, i64
vpto.copy_gm_to_ubuf %7, %2, %3, %3, %c0_i64, %c32_i64, %4, %c0_i64, %c0_i64, %c0_i64, %c128_i64, %c128_i64
    {data_select_bit = false, layout = "nd", ub_pad = false}
    : !llvm.ptr<1>, !llvm.ptr<6>, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64

vpto.set_flag["PIPE_MTE2", "PIPE_V", "EVENT_ID0"]
vpto.wait_flag["PIPE_MTE2", "PIPE_V", "EVENT_ID0"]

scf.for %dummy = %c0 to %c1 step %c1 {
  scf.for %lane = %c0 to %9 step %c64 {
    %v = vpto.vlds %2[%lane] : !llvm.ptr<6> -> !vpto.vec<64xf32>
    %abs = vpto.vabs %v : !vpto.vec<64xf32> -> !vpto.vec<64xf32>
    vpto.vsts %abs, %8[%lane] : !vpto.vec<64xf32>, !llvm.ptr<6>
  }
} {llvm.loop.aivector_scope}

vpto.set_flag["PIPE_V", "PIPE_MTE3", "EVENT_ID0"]
vpto.wait_flag["PIPE_V", "PIPE_MTE3", "EVENT_ID0"]
vpto.set_loop_size_ubtoout %c1_i64, %c1_i64 : i64, i64
vpto.set_loop1_stride_ubtoout %c4096_i64, %c4096_i64 : i64, i64
vpto.set_loop2_stride_ubtoout %c4096_i64, %c4096_i64 : i64, i64
vpto.copy_ubuf_to_gm %8, %14, %3, %3, %c0_i64, %c32_i64, %4, %c0_i64, %c128_i64, %c128_i64
    {layout = "nd"}
    : !llvm.ptr<6>, !llvm.ptr<1>, i64, i64, i64, i64, i64, i64, i64, i64
```

## Scope

This document is the interface specification for the `mlir::vpto` dialect.

It only describes:

- operation names
- operand and result lists
- operand and result types
- important attributes
- corresponding CCE builtin or CCE wrapper family

It does not describe lowering strategy.

## Core Types

- `vec<T>`: `!vpto.vec<NxT>`
  Fixed-width VPTO vector type with total width exactly 256 bytes.
- `mask`: `!vpto.mask`
- `align`: `!vpto.align`
- `buf`: buffer-like LLVM pointer type accepted by the dialect
- `idx`: `index`
- `i32`: `i32`
- `i64`: `i64`

Type parameter conventions used below:

- `!vpto.vec<NxT>`:
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
  `!vpto.vec<NxT>` requires `T` to be an integer or floating-point element
  type, and `N * bitwidth(T) = 2048`.
- `T`:
  `TODO(user): summarize the intended element-type set for general arithmetic,
  logical, and load/store ops.`
- `T0`, `T1`:
  `TODO(user): list the intended legal source/result type pairs for conversion
  ops such as vpto.vcvt.`
- `I`:
  `TODO(user): summarize which integer element widths are intended for offsets,
  indices, lane selectors, and permutation inputs.`
- Family-specific exceptions:
  `TODO(user): capture any op-family-specific restrictions or implementation
  subsets here.`

## Special Types

### `!vpto.mask`

`!vpto.mask` models an A5 predicate register, not an integer vector.

Use it when an operation needs per-lane enable/disable state.

- producers:
  `vpto.pset_b8`, `vpto.pset_b16`, `vpto.pset_b32`,
  `vpto.pge_b8`, `vpto.pge_b16`, `vpto.pge_b32`,
  `vpto.plds`, `vpto.pld`, `vpto.pldi`,
  `vpto.vcmp`, `vpto.vcmps`
- consumers:
  `vpto.vsel`,
  `vpto.vaddc`, `vpto.vsubc`, `vpto.vaddcs`, `vpto.vsubcs`,
  `vpto.pnot`, `vpto.psel`,
  `vpto.vgather2_bc`,
  `vpto.vstx2`, `vpto.vsstb`,
  `vpto.psts`, `vpto.pst`, `vpto.psti`,
  `vpto.pstu`,
  `vpto.vmula`

Example:

```mlir
%mask = vpto.vcmp %lhs, %rhs, %seed, "lt" : !vpto.vec<64xf32>, !vpto.vec<64xf32>, !vpto.mask -> !vpto.mask
%out = vpto.vsel %x, %y, %mask : !vpto.vec<64xf32>, !vpto.vec<64xf32>, !vpto.mask -> !vpto.vec<64xf32>
```

### `!vpto.align`

`!vpto.align` models the A5 vector-align carrier state. It is not payload data.

Use it when an operation needs explicit align-state threading in SSA form.

- producers:
  `vpto.vldas`,
  `vpto.pstu`,
  `vpto.vstu`,
  `vpto.vstus`,
  `vpto.vstur`
- consumers:
  `vpto.vldus`,
  `vpto.vsta`,
  `vpto.vstas`,
  `vpto.vstar`,
  `vpto.pstu`,
  `vpto.vstu`,
  `vpto.vstus`,
  `vpto.vstur`

Example:

```mlir
%align = vpto.vldas %ub[%c0] : !llvm.ptr<6> -> !vpto.align
%vec = vpto.vldus %align, %ub[%c64] : !vpto.align, !llvm.ptr<6> -> !vpto.vec<64xf32>
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
- `T|!vpto.vec<NxT>`:
  either a scalar `T` or a vector operand `!vpto.vec<NxT>`, matching the op verifier

## Implemented String Constraints

This section records string-valued operands and attributes that are already
checked by the current verifier implementation.

If a token is not listed here, the current dialect usually only requires a
non-empty string or leaves the token unconstrained for now.

### Predicate Patterns

Used by:
`vpto.pset_b8`, `vpto.pset_b16`, `vpto.pset_b32`,
`vpto.pge_b8`, `vpto.pge_b16`, `vpto.pge_b32`

- allowed values:
  `PAT_ALL | PAT_VL1 | PAT_VL2 | PAT_VL3 | PAT_VL4 | PAT_VL8 | PAT_VL16 | PAT_VL32 | PAT_VL64 | PAT_VL128 | PAT_M3 | PAT_M4 | PAT_H | PAT_Q | PAT_ALLF`

### Distribution Tokens

Used by `vpto.vlds`:

- allowed values:
  `NORM | BLK | DINTLV_B32 | UNPK_B16`

Used by `vpto.pld`, `vpto.pldi`:

- allowed values:
  `NORM | US | DS`

Used by `vpto.pst`, `vpto.psti`:

- allowed values:
  `NORM | PK`

Used by `vpto.vldx2`:

- allowed values:
  `DINTLV_B8 | DINTLV_B16 | DINTLV_B32 | BDINTLV`

Used by `vpto.vstx2`:

- allowed values:
  `INTLV_B8 | INTLV_B16 | INTLV_B32`

### Stride Tokens

Used by `vpto.vsld`, `vpto.vsst`:

- allowed values:
  `STRIDE_S3_B16 | STRIDE_S4_B64 | STRIDE_S8_B32 | STRIDE_S2_B64 | STRIDE_VSST_S8_B16`

### Compare Modes

Used by `vpto.vcmp`, `vpto.vcmps`:

- allowed values:
  `eq | ne | lt | le | gt | ge`

### Part Tokens

Used by `vpto.vintlvv2`, `vpto.vdintlvv2`:

- allowed values:
  `LOWER | HIGHER`

Current restricted subset:

- `vpto.ppack`: only `LOWER`
- `vpto.punpack`: only `LOWER`

### Mode Tokens

Used by `vpto.vmula`:

- allowed values:
  `MODE_ZEROING | MODE_UNKNOWN | MODE_MERGING`

Used by `vpto.vstu`, `vpto.vstus`, `vpto.vstur`:

- allowed values:
  `POST_UPDATE | NO_POST_UPDATE`

### Conversion Control Tokens

Used by `vpto.vcvt.round_mode`:

- allowed values:
  `ROUND_R | ROUND_A | ROUND_F | ROUND_C | ROUND_Z | ROUND_O`

Used by `vpto.vcvt.sat`:

- allowed values:
  `RS_ENABLE | RS_DISABLE`

Used by `vpto.vcvt.part`:

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

`__VEC_SCOPE__` is not an `vpto` op.

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

### `vpto.set_flag`

- syntax:
  `vpto.set_flag["SRC_PIPE", "DST_PIPE", "EVENT_ID"]`
- semantics:
  TODO(user): add one-line semantics for external developers.
- CCE correspondence:
  `set_flag(pipe_t, pipe_t, event_t|uint64_t)`
  `__builtin_cce_set_flag`
  PTO token path:
  `__pto_set_flag`
  `__builtin_cce_tile_set_flag`

### `vpto.wait_flag`

- syntax:
  `vpto.wait_flag["SRC_PIPE", "DST_PIPE", "EVENT_ID"]`
- semantics:
  TODO(user): add one-line semantics for external developers.
- CCE correspondence:
  `wait_flag(pipe_t, pipe_t, event_t|uint64_t)`
  `__builtin_cce_wait_flag`
  PTO token path:
  `__pto_wait_flag`
  `__builtin_cce_tile_wait_flag`

### `vpto.pipe_barrier`

- syntax:
  `vpto.pipe_barrier "PIPE_*"`
- semantics:
  TODO(user): add one-line semantics for external developers.
- CCE correspondence:
  `pipe_barrier(pipe_t)`
  `__builtin_cce_pipe_barrier`

### `vpto.get_buf`

- syntax:
  `vpto.get_buf "PIPE_*", %buf_id, %mode : i64, i64`
- semantics:
  TODO(user): add one-line semantics for external developers.
- CCE correspondence:
  `get_buf(pipe_t, uint8_t|uint64_t, bool)`
  `__builtin_cce_get_buf`

### `vpto.rls_buf`

- syntax:
  `vpto.rls_buf "PIPE_*", %buf_id, %mode : i64, i64`
- semantics:
  TODO(user): add one-line semantics for external developers.
- CCE correspondence:
  `rls_buf(pipe_t, uint8_t|uint64_t, bool)`
  `__builtin_cce_rls_buf`

## 2. Copy Programming

### `vpto.set_loop2_stride_outtoub`

- syntax:
  `vpto.set_loop2_stride_outtoub %first, %second : i64, i64`
- semantics:
  TODO(user): add one-line semantics for external developers.
- CCE correspondence:
  `set_loop2_stride_outtoub(uint64_t)`
  `__builtin_cce_set_loop2_stride_outtoub`

### `vpto.set_loop1_stride_outtoub`

- syntax:
  `vpto.set_loop1_stride_outtoub %first, %second : i64, i64`
- semantics:
  TODO(user): add one-line semantics for external developers.
- CCE correspondence:
  `set_loop1_stride_outtoub(uint64_t)`
  `__builtin_cce_set_loop1_stride_outtoub`

### `vpto.set_loop_size_outtoub`

- syntax:
  `vpto.set_loop_size_outtoub %first, %second : i64, i64`
- semantics:
  TODO(user): add one-line semantics for external developers.
- CCE correspondence:
  `set_loop_size_outtoub(uint64_t)`
  `__builtin_cce_set_loop_size_outtoub`

### `vpto.set_loop2_stride_ubtoout`

- syntax:
  `vpto.set_loop2_stride_ubtoout %first, %second : i64, i64`
- semantics:
  TODO(user): add one-line semantics for external developers.
- CCE correspondence:
  `set_loop2_stride_ubtoout(uint64_t)`
  `__builtin_cce_set_loop2_stride_ubtoout`

### `vpto.set_loop1_stride_ubtoout`

- syntax:
  `vpto.set_loop1_stride_ubtoout %first, %second : i64, i64`
- semantics:
  TODO(user): add one-line semantics for external developers.
- CCE correspondence:
  `set_loop1_stride_ubtoout(uint64_t)`
  `__builtin_cce_set_loop1_stride_ubtoout`

### `vpto.set_loop_size_ubtoout`

- syntax:
  `vpto.set_loop_size_ubtoout %first, %second : i64, i64`
- semantics:
  TODO(user): add one-line semantics for external developers.
- CCE correspondence:
  `set_loop_size_ubtoout(uint64_t)`
  `__builtin_cce_set_loop_size_ubtoout`

## 3. Copy Transfers

### `vpto.copy_gm_to_ubuf`

- syntax:
  `vpto.copy_gm_to_ubuf %source, %destination, %valid_rows, %valid_cols, %sid, %n_burst, %len_burst, %left_padding_count, %right_padding_count, %l2_cache_ctl, %gm_stride, %ub_stride {layout = "LAYOUT", data_select_bit = true|false, ub_pad = true|false} : !llvm.ptr<AS>, !llvm.ptr<AS>, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64`
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

### `vpto.copy_ubuf_to_ubuf`

- syntax:
  `vpto.copy_ubuf_to_ubuf %source, %destination, %sid, %n_burst, %len_burst, %src_stride, %dst_stride : !llvm.ptr<AS>, !llvm.ptr<AS>, i64, i64, i64, i64, i64`
- semantics:
  TODO(user): add one-line semantics for external developers.
- CCE correspondence:
  `copy_ubuf_to_ubuf(...)`
  `__builtin_cce_copy_ubuf_to_ubuf`

### `vpto.copy_ubuf_to_gm`

- syntax:
  `vpto.copy_ubuf_to_gm %source, %destination, %valid_rows, %valid_cols, %sid, %n_burst, %len_burst, %reserved, %burst_dst_stride, %burst_src_stride {layout = "LAYOUT"} : !llvm.ptr<AS>, !llvm.ptr<AS>, i64, i64, i64, i64, i64, i64, i64, i64`
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

### `vpto.vlds`

- syntax:
  `%result = vpto.vlds %source[%offset] {dist = "DIST"} : !llvm.ptr<AS> -> !vpto.vec<NxT>`
- semantics:
  TODO(user): add one-line semantics for external developers.
- CCE correspondence:
  `vld(...)`, `vlds(...)`
  `__builtin_cce_vldsx1_*`
  related extended families:
  `__builtin_cce_vldix1_*`, `__builtin_cce_vldsx1_post_*`

### `vpto.vldas`

- syntax:
  `%result = vpto.vldas %source[%offset] : !llvm.ptr<AS> -> !vpto.align`
- semantics:
  TODO(user): add one-line semantics for external developers.
- CCE correspondence:
  `vldas(...)`
  `__builtin_cce_vldas_*`

### `vpto.vldus`

- syntax:
  `%result = vpto.vldus %align, %source[%offset] : !vpto.align, !llvm.ptr<AS> -> !vpto.vec<NxT>`
- semantics:
  TODO(user): add one-line semantics for external developers.
- CCE correspondence:
  `vldus(...)`
  `__builtin_cce_vldus_*`, `__builtin_cce_vldus_post_*`

### `vpto.plds`

- syntax:
  `%result = vpto.plds %source[%offset] {dist = "DIST"} : !llvm.ptr<AS> -> !vpto.mask`
- semantics:
  TODO(user): add one-line semantics for external developers.
- CCE correspondence:
  `plds(...)`
  `__builtin_cce_plds_b8`

### `vpto.pld`

- syntax:
  `%result = vpto.pld %source[%offset], "DIST" : !llvm.ptr<AS>, index -> !vpto.mask`
- semantics:
  TODO(user): add one-line semantics for external developers.
- CCE correspondence:
  `pld(...)`
  `__builtin_cce_pld_b8`

### `vpto.pldi`

- syntax:
  `%result = vpto.pldi %source, %offset, "DIST" : !llvm.ptr<AS>, i32 -> !vpto.mask`
- semantics:
  TODO(user): add one-line semantics for external developers.
- CCE correspondence:
  `pldi(...)`
  `__builtin_cce_pldi_b8`, `__builtin_cce_pldi_post_b8`

### `vpto.vldx2`

- syntax:
  `%low, %high = vpto.vldx2 %source[%offset], "DIST" : !llvm.ptr<AS>, index -> !vpto.vec<NxT>, !vpto.vec<NxT>`
- semantics:
  TODO(user): add one-line semantics for external developers.
- CCE correspondence:
  `vld(...)`
  `__builtin_cce_vldx2_*`

### `vpto.vgather2`

- syntax:
  `%result = vpto.vgather2 %source, %offsets, %active_lanes : !llvm.ptr<AS>, !vpto.vec<NxI>, index -> !vpto.vec<NxT>`
- semantics:
  TODO(user): add one-line semantics for external developers.
- CCE correspondence:
  `vgather2(...)`
  `__builtin_cce_vgather2_*`, `__builtin_cce_vgather2_v300_*`

### `vpto.vgatherb`

- syntax:
  `%result = vpto.vgatherb %source, %offsets, %active_lanes : !llvm.ptr<AS>, !vpto.vec<NxI>, index -> !vpto.vec<NxT>`
- semantics:
  TODO(user): add one-line semantics for external developers.
- CCE correspondence:
  `vgatherb(...)`
  `__builtin_cce_vgatherb_*`, `__builtin_cce_vgatherb_v300_*`, `__builtin_cce_vgatherb_v310_*`

### `vpto.vgather2_bc`

- syntax:
  `%result = vpto.vgather2_bc %source, %offsets, %mask : !llvm.ptr<AS>, !vpto.vec<NxI>, !vpto.mask -> !vpto.vec<NxT>`
- semantics:
  TODO(user): add one-line semantics for external developers.
- CCE correspondence:
  `vgather2_bc(...)`
  `__builtin_cce_vgather2_bc_*`

### `vpto.vsld`

- syntax:
  `%result = vpto.vsld %source[%offset], "STRIDE" : !llvm.ptr<AS> -> !vpto.vec<NxT>`
- semantics:
  TODO(user): add one-line semantics for external developers.
- CCE correspondence:
  `vsld(...)`
  `__builtin_cce_vsld_*`

### `vpto.vsldb`

- syntax:
  `%result = vpto.vsldb %source, %offset, %mask : !llvm.ptr<AS>, i32, !vpto.mask -> !vpto.vec<NxT>`
- semantics:
  TODO(user): add one-line semantics for external developers.
- CCE correspondence:
  `vsldb(...)`
  `__builtin_cce_vsldb_*`, `__builtin_cce_vsldb_post_*`

## 5. Materialization And Predicate Construction

### `vpto.vbr`

- syntax:
  `%result = vpto.vbr %value : T -> !vpto.vec<NxT>`
- semantics:
  TODO(user): add one-line semantics for external developers.
- CCE correspondence:
  broadcast/materialization family used by PTO scalar-to-vector expansion

### `vpto.vdup`

- syntax:
  `%result = vpto.vdup %input {position = "POSITION", mode = "MODE"} : T|!vpto.vec<NxT> -> !vpto.vec<NxT>`
- semantics:
  TODO(user): add one-line semantics for external developers.
- CCE correspondence:
  `vdup(...)`
  `__builtin_cce_vdup_*`

### `vpto.pset_b8`

- syntax:
  `%result = vpto.pset_b8 "PAT_*" : !vpto.mask`
- semantics:
  TODO(user): add one-line semantics for external developers.
- CCE correspondence:
  `pset_b8(...)`
  `__builtin_cce_pset_b8`

### `vpto.pset_b16`

- syntax:
  `%result = vpto.pset_b16 "PAT_*" : !vpto.mask`
- semantics:
  TODO(user): add one-line semantics for external developers.
- CCE correspondence:
  `pset_b16(...)`
  `__builtin_cce_pset_b16`

### `vpto.pset_b32`

- syntax:
  `%result = vpto.pset_b32 "PAT_*" : !vpto.mask`
- semantics:
  TODO(user): add one-line semantics for external developers.
- CCE correspondence:
  `pset_b32(...)`
  `__builtin_cce_pset_b32`

### `vpto.pge_b8`

- syntax:
  `%result = vpto.pge_b8 "PAT_*" : !vpto.mask`
- semantics:
  TODO(user): add one-line semantics for external developers.
- CCE correspondence:
  `pge_b8(...)`
  `__builtin_cce_pge_b8`

### `vpto.pge_b16`

- syntax:
  `%result = vpto.pge_b16 "PAT_*" : !vpto.mask`
- semantics:
  TODO(user): add one-line semantics for external developers.
- CCE correspondence:
  `pge_b16(...)`
  `__builtin_cce_pge_b16`

### `vpto.pge_b32`

- syntax:
  `%result = vpto.pge_b32 "PAT_*" : !vpto.mask`
- semantics:
  TODO(user): add one-line semantics for external developers.
- CCE correspondence:
  `pge_b32(...)`
  `__builtin_cce_pge_b32`

### `vpto.ppack`

- syntax:
  `%result = vpto.ppack %input, "PART" : !vpto.mask -> !vpto.mask`
- semantics:
  TODO(user): add one-line semantics for external developers.
- CCE correspondence:
  `ppack(...)`

### `vpto.punpack`

- syntax:
  `%result = vpto.punpack %input, "PART" : !vpto.mask -> !vpto.mask`
- semantics:
  TODO(user): add one-line semantics for external developers.
- CCE correspondence:
  `punpack(...)`

## 6. Unary Vector Ops

### `vpto.vabs`

- syntax:
  `%result = vpto.vabs %input : !vpto.vec<NxT> -> !vpto.vec<NxT>`
- semantics:
  TODO(user): add one-line semantics for external developers.
- CCE correspondence:
  `vabs(...)`
  `__builtin_cce_vabs_*`

### `vpto.vexp`

- syntax:
  `%result = vpto.vexp %input : !vpto.vec<NxT> -> !vpto.vec<NxT>`
- semantics:
  TODO(user): add one-line semantics for external developers.
- CCE correspondence:
  `vexp(...)`
  `__builtin_cce_vexp_*`

### `vpto.vln`

- syntax:
  `%result = vpto.vln %input : !vpto.vec<NxT> -> !vpto.vec<NxT>`
- semantics:
  TODO(user): add one-line semantics for external developers.
- CCE correspondence:
  `vln(...)`
  `__builtin_cce_vln_*`

### `vpto.vsqrt`

- syntax:
  `%result = vpto.vsqrt %input : !vpto.vec<NxT> -> !vpto.vec<NxT>`
- semantics:
  TODO(user): add one-line semantics for external developers.
- CCE correspondence:
  `vsqrt(...)`
  `__builtin_cce_vsqrt_*`

### `vpto.vrec`

- syntax:
  `%result = vpto.vrec %input : !vpto.vec<NxT> -> !vpto.vec<NxT>`
- semantics:
  TODO(user): add one-line semantics for external developers.
- CCE correspondence:
  `vrec(...)`
  `__builtin_cce_vrec_*`

### `vpto.vrelu`

- syntax:
  `%result = vpto.vrelu %input : !vpto.vec<NxT> -> !vpto.vec<NxT>`
- semantics:
  TODO(user): add one-line semantics for external developers.
- CCE correspondence:
  `vrelu(...)`
  `__builtin_cce_vrelu_*`

### `vpto.vnot`

- syntax:
  `%result = vpto.vnot %input : !vpto.vec<NxT> -> !vpto.vec<NxT>`
- semantics:
  TODO(user): add one-line semantics for external developers.
- CCE correspondence:
  `vnot(...)`
  `__builtin_cce_vnot_*`

### `vpto.vcadd`

- syntax:
  `%result = vpto.vcadd %input : !vpto.vec<NxT> -> !vpto.vec<NxT>`
- semantics:
  TODO(user): add one-line semantics for external developers.
- CCE correspondence:
  `vcadd(...)`
  `__builtin_cce_vcadd_*`

### `vpto.vcmax`

- syntax:
  `%result = vpto.vcmax %input : !vpto.vec<NxT> -> !vpto.vec<NxT>`
- semantics:
  TODO(user): add one-line semantics for external developers.
- CCE correspondence:
  `vcmax(...)`
  `__builtin_cce_vcmax_*`

### `vpto.vcmin`

- syntax:
  `%result = vpto.vcmin %input : !vpto.vec<NxT> -> !vpto.vec<NxT>`
- semantics:
  TODO(user): add one-line semantics for external developers.
- CCE correspondence:
  `vcmin(...)`
  `__builtin_cce_vcmin_*`

### `vpto.vbcnt`

- syntax:
  `%result = vpto.vbcnt %input : !vpto.vec<NxT> -> !vpto.vec<NxT>`
- semantics:
  TODO(user): add one-line semantics for external developers.
- CCE correspondence:
  `vbcnt(...)`
  `__builtin_cce_vbcnt_*`

### `vpto.vcls`

- syntax:
  `%result = vpto.vcls %input : !vpto.vec<NxT> -> !vpto.vec<NxT>`
- semantics:
  TODO(user): add one-line semantics for external developers.
- CCE correspondence:
  `vcls(...)`
  `__builtin_cce_vcls_*`

## 7. Binary Vector Ops

### `vpto.vadd`

- syntax:
  `%result = vpto.vadd %lhs, %rhs : !vpto.vec<NxT>, !vpto.vec<NxT> -> !vpto.vec<NxT>`
- semantics:
  TODO(user): add one-line semantics for external developers.
- CCE correspondence:
  `vadd(...)`
  `__builtin_cce_vadd_*`

### `vpto.vsub`

- syntax:
  `%result = vpto.vsub %lhs, %rhs : !vpto.vec<NxT>, !vpto.vec<NxT> -> !vpto.vec<NxT>`
- semantics:
  TODO(user): add one-line semantics for external developers.
- CCE correspondence:
  `vsub(...)`
  `__builtin_cce_vsub_*`

### `vpto.vmul`

- syntax:
  `%result = vpto.vmul %lhs, %rhs : !vpto.vec<NxT>, !vpto.vec<NxT> -> !vpto.vec<NxT>`
- semantics:
  TODO(user): add one-line semantics for external developers.
- CCE correspondence:
  `vmul(...)`
  `__builtin_cce_vmul_*`

### `vpto.vdiv`

- syntax:
  `%result = vpto.vdiv %lhs, %rhs : !vpto.vec<NxT>, !vpto.vec<NxT> -> !vpto.vec<NxT>`
- semantics:
  TODO(user): add one-line semantics for external developers.
- CCE correspondence:
  `vdiv(...)`
  `__builtin_cce_vdiv_*`

### `vpto.vmax`

- syntax:
  `%result = vpto.vmax %lhs, %rhs : !vpto.vec<NxT>, !vpto.vec<NxT> -> !vpto.vec<NxT>`
- semantics:
  TODO(user): add one-line semantics for external developers.
- CCE correspondence:
  `vmax(...)`
  `__builtin_cce_vmax_*`

### `vpto.vmin`

- syntax:
  `%result = vpto.vmin %lhs, %rhs : !vpto.vec<NxT>, !vpto.vec<NxT> -> !vpto.vec<NxT>`
- semantics:
  TODO(user): add one-line semantics for external developers.
- CCE correspondence:
  `vmin(...)`
  `__builtin_cce_vmin_*`

### `vpto.vand`

- syntax:
  `%result = vpto.vand %lhs, %rhs : !vpto.vec<NxT>, !vpto.vec<NxT> -> !vpto.vec<NxT>`
- semantics:
  TODO(user): add one-line semantics for external developers.
- CCE correspondence:
  `vand(...)`
  `__builtin_cce_vand_*`

### `vpto.vor`

- syntax:
  `%result = vpto.vor %lhs, %rhs : !vpto.vec<NxT>, !vpto.vec<NxT> -> !vpto.vec<NxT>`
- semantics:
  TODO(user): add one-line semantics for external developers.
- CCE correspondence:
  `vor(...)`
  `__builtin_cce_vor_*`

### `vpto.vxor`

- syntax:
  `%result = vpto.vxor %lhs, %rhs : !vpto.vec<NxT>, !vpto.vec<NxT> -> !vpto.vec<NxT>`
- semantics:
  TODO(user): add one-line semantics for external developers.
- CCE correspondence:
  `vxor(...)`
  `__builtin_cce_vxor_*`

### `vpto.vshl`

- syntax:
  `%result = vpto.vshl %lhs, %rhs : !vpto.vec<NxT>, !vpto.vec<NxT> -> !vpto.vec<NxT>`
- semantics:
  TODO(user): add one-line semantics for external developers.
- CCE correspondence:
  `vshl(...)`
  `__builtin_cce_vshl_*`

### `vpto.vshr`

- syntax:
  `%result = vpto.vshr %lhs, %rhs : !vpto.vec<NxT>, !vpto.vec<NxT> -> !vpto.vec<NxT>`
- semantics:
  TODO(user): add one-line semantics for external developers.
- CCE correspondence:
  `vshr(...)`
  `__builtin_cce_vshr_*`

## 8. Vec-Scalar Ops

### `vpto.vmuls`

- syntax:
  `%result = vpto.vmuls %input, %scalar : !vpto.vec<NxT>, T -> !vpto.vec<NxT>`
- semantics:
  TODO(user): add one-line semantics for external developers.
- CCE correspondence:
  `vmuls(...)`
  `__builtin_cce_vmuls_*`

### `vpto.vadds`

- syntax:
  `%result = vpto.vadds %input, %scalar : !vpto.vec<NxT>, T -> !vpto.vec<NxT>`
- semantics:
  TODO(user): add one-line semantics for external developers.
- CCE correspondence:
  `vadds(...)`
  `__builtin_cce_vadds_*`

### `vpto.vmaxs`

- syntax:
  `%result = vpto.vmaxs %input, %scalar : !vpto.vec<NxT>, T -> !vpto.vec<NxT>`
- semantics:
  TODO(user): add one-line semantics for external developers.
- CCE correspondence:
  `vmaxs(...)`
  `__builtin_cce_vmaxs_*`

### `vpto.vmins`

- syntax:
  `%result = vpto.vmins %input, %scalar : !vpto.vec<NxT>, T -> !vpto.vec<NxT>`
- semantics:
  TODO(user): add one-line semantics for external developers.
- CCE correspondence:
  `vmins(...)`
  `__builtin_cce_vmins_*`

### `vpto.vlrelu`

- syntax:
  `%result = vpto.vlrelu %input, %scalar : !vpto.vec<NxT>, T -> !vpto.vec<NxT>`
- semantics:
  TODO(user): add one-line semantics for external developers.
- CCE correspondence:
  `vlrelu(...)`
  `__builtin_cce_vlrelu_*`

### `vpto.vshls`

- syntax:
  `%result = vpto.vshls %input, %scalar : !vpto.vec<NxT>, T -> !vpto.vec<NxT>`
- semantics:
  TODO(user): add one-line semantics for external developers.
- CCE correspondence:
  `vshls(...)`
  `__builtin_cce_vshls_*`

### `vpto.vshrs`

- syntax:
  `%result = vpto.vshrs %input, %scalar : !vpto.vec<NxT>, T -> !vpto.vec<NxT>`
- semantics:
  TODO(user): add one-line semantics for external developers.
- CCE correspondence:
  `vshrs(...)`
  `__builtin_cce_vshrs_*`

## 9. Carry, Compare And Select

### `vpto.vaddc`

- syntax:
  `%result, %carry = vpto.vaddc %lhs, %rhs, %mask : !vpto.vec<NxT>, !vpto.vec<NxT>, !vpto.mask -> !vpto.vec<NxT>, !vpto.mask`
- semantics:
  TODO(user): add one-line semantics for external developers.
- CCE correspondence:
  `vaddc(...)`
  `__builtin_cce_vaddc_*`

### `vpto.vsubc`

- syntax:
  `%result, %carry = vpto.vsubc %lhs, %rhs, %mask : !vpto.vec<NxT>, !vpto.vec<NxT>, !vpto.mask -> !vpto.vec<NxT>, !vpto.mask`
- semantics:
  TODO(user): add one-line semantics for external developers.
- CCE correspondence:
  `vsubc(...)`
  `__builtin_cce_vsubc_*`

### `vpto.vaddcs`

- syntax:
  `%result, %carry = vpto.vaddcs %lhs, %rhs, %carry_in, %mask : !vpto.vec<NxT>, !vpto.vec<NxT>, !vpto.mask, !vpto.mask -> !vpto.vec<NxT>, !vpto.mask`
- semantics:
  TODO(user): add one-line semantics for external developers.
- CCE correspondence:
  `vaddcs(...)`
  `__builtin_cce_vaddcs_*`

### `vpto.vsubcs`

- syntax:
  `%result, %carry = vpto.vsubcs %lhs, %rhs, %carry_in, %mask : !vpto.vec<NxT>, !vpto.vec<NxT>, !vpto.mask, !vpto.mask -> !vpto.vec<NxT>, !vpto.mask`
- semantics:
  TODO(user): add one-line semantics for external developers.
- CCE correspondence:
  `vsubcs(...)`
  `__builtin_cce_vsubcs_*`

### `vpto.vsel`

- syntax:
  `%result = vpto.vsel %src0, %src1, %mask : !vpto.vec<NxT>, !vpto.vec<NxT>, !vpto.mask -> !vpto.vec<NxT>`
- semantics:
  TODO(user): add one-line semantics for external developers.
- CCE correspondence:
  `vsel(...)`
  `__builtin_cce_vsel_*`

### `vpto.vselr`

- syntax:
  `%result = vpto.vselr %src0, %src1 : !vpto.vec<NxT>, !vpto.vec<NxI> -> !vpto.vec<NxT>`
- semantics:
  TODO(user): add one-line semantics for external developers.
- CCE correspondence:
  `vselr(...)`
  `__builtin_cce_vselr_*`

### `vpto.vselrv2`

- syntax:
  `%result = vpto.vselrv2 %src0, %src1 : !vpto.vec<NxT>, !vpto.vec<NxI> -> !vpto.vec<NxT>`
- semantics:
  TODO(user): add one-line semantics for external developers.
- CCE correspondence:
  `vselrv2(...)`
  `__builtin_cce_vselrv2_*`

### `vpto.vcmp`

- syntax:
  `%result = vpto.vcmp %src0, %src1, %mask, "CMP_MODE" : !vpto.vec<NxT>, !vpto.vec<NxT>, !vpto.mask -> !vpto.mask`
- semantics:
  TODO(user): add one-line semantics for external developers.
- CCE correspondence:
  `vcmp(...)`
  `__builtin_cce_vcmp_<op>_*_z`

### `vpto.vcmps`

- syntax:
  `%result = vpto.vcmps %src, %scalar, %mask, "CMP_MODE" : !vpto.vec<NxT>, T, !vpto.mask -> !vpto.mask`
- semantics:
  TODO(user): add one-line semantics for external developers.
- CCE correspondence:
  `vcmps(...)`
  `__builtin_cce_vcmps_<op>_*_z`

### `vpto.pnot`

- syntax:
  `%result = vpto.pnot %input, %mask : !vpto.mask, !vpto.mask -> !vpto.mask`
- semantics:
  TODO(user): add one-line semantics for external developers.
- CCE correspondence:
  `pnot(...)`

### `vpto.psel`

- syntax:
  `%result = vpto.psel %src0, %src1, %mask : !vpto.mask, !vpto.mask, !vpto.mask -> !vpto.mask`
- semantics:
  TODO(user): add one-line semantics for external developers.
- CCE correspondence:
  `psel(...)`

## 10. Pairing And Interleave

### `vpto.pdintlv_b8`

- syntax:
  `%low, %high = vpto.pdintlv_b8 %lhs, %rhs : !vpto.mask, !vpto.mask -> !vpto.mask, !vpto.mask`
- semantics:
  TODO(user): add one-line semantics for external developers.
- CCE correspondence:
  predicate interleave/deinterleave family

### `vpto.pintlv_b16`

- syntax:
  `%low, %high = vpto.pintlv_b16 %lhs, %rhs : !vpto.mask, !vpto.mask -> !vpto.mask, !vpto.mask`
- semantics:
  TODO(user): add one-line semantics for external developers.
- CCE correspondence:
  predicate interleave/deinterleave family

### `vpto.vintlv`

- syntax:
  `%low, %high = vpto.vintlv %lhs, %rhs : !vpto.vec<NxT>, !vpto.vec<NxT> -> !vpto.vec<NxT>, !vpto.vec<NxT>`
- semantics:
  TODO(user): add one-line semantics for external developers.
- CCE correspondence:
  `vintlv(...)`
  `__builtin_cce_vintlv_*`

### `vpto.vdintlv`

- syntax:
  `%low, %high = vpto.vdintlv %lhs, %rhs : !vpto.vec<NxT>, !vpto.vec<NxT> -> !vpto.vec<NxT>, !vpto.vec<NxT>`
- semantics:
  TODO(user): add one-line semantics for external developers.
- CCE correspondence:
  `vdintlv(...)`
  `__builtin_cce_vdintlv_*`

### `vpto.vintlvv2`

- syntax:
  `%result = vpto.vintlvv2 %lhs, %rhs, "PART" : !vpto.vec<NxT>, !vpto.vec<NxT> -> !vpto.vec<NxT>`
- semantics:
  TODO(user): add one-line semantics for external developers.
- CCE correspondence:
  `vintlvv2(...)`
  `__builtin_cce_vintlvv2_*`

### `vpto.vdintlvv2`

- syntax:
  `%result = vpto.vdintlvv2 %lhs, %rhs, "PART" : !vpto.vec<NxT>, !vpto.vec<NxT> -> !vpto.vec<NxT>`
- semantics:
  TODO(user): add one-line semantics for external developers.
- CCE correspondence:
  `vdintlvv2(...)`
  `__builtin_cce_vdintlvv2_*`

## 11. Conversion, Index And Sort

### `vpto.vtrc`

- syntax:
  `%result = vpto.vtrc %input, "ROUND_MODE" : !vpto.vec<NxT> -> !vpto.vec<NxT>`
- semantics:
  TODO(user): add one-line semantics for external developers.
- CCE correspondence:
  `vtrc(...)`
  `__builtin_cce_vtrc_*`

### `vpto.vcvt`

- syntax:
  `%result = vpto.vcvt %input {round_mode = "ROUND_MODE", sat = "SAT_MODE", part = "PART_MODE"} : !vpto.vec<NxT0> -> !vpto.vec<NxT1>`
- semantics:
  TODO(user): add one-line semantics for external developers.
- CCE correspondence:
  `vcvt(...)`
  builtin families:
  `__builtin_cce_vcvt*`, `__builtin_cce_vcvtfi_*`, `__builtin_cce_vcvtif_*`, `__builtin_cce_vcvtii_*`, `__builtin_cce_vcvtff_*`

### `vpto.vci`

- syntax:
  `%result = vpto.vci %index {order = "ORDER"} : integer -> !vpto.vec<NxT>`
- semantics:
  TODO(user): add one-line semantics for external developers.
- CCE correspondence:
  `vci(...)`
  `__builtin_cce_vci_*`

### `vpto.vbitsort`

- syntax:
  `vpto.vbitsort %destination, %source, %indices, %repeat_times : !llvm.ptr<AS>, !llvm.ptr<AS>, !llvm.ptr<AS>, index`
- semantics:
  TODO(user): add one-line semantics for external developers.
- CCE correspondence:
  `vbitsort(...)`
  `__builtin_cce_vbitsort_*`

### `vpto.vmrgsort4`

- syntax:
  `vpto.vmrgsort4 %destination, %source0, %source1, %source2, %source3, %count, %config : !llvm.ptr<AS>, !llvm.ptr<AS>, !llvm.ptr<AS>, !llvm.ptr<AS>, !llvm.ptr<AS>, i64, i64`
- semantics:
  TODO(user): add one-line semantics for external developers.
- CCE correspondence:
  `vmrgsort4(...)`
  `__builtin_cce_vmrgsort4_*`

## 12. Extended Arithmetic

### `vpto.vmull`

- syntax:
  `%low, %high = vpto.vmull %lhs, %rhs, %mask : !vpto.vec<NxT>, !vpto.vec<NxT>, !vpto.mask -> !vpto.vec<NxT>, !vpto.vec<NxT>`
- semantics:
  TODO(user): add one-line semantics for external developers.
- CCE correspondence:
  `vmull(...)`
  `__builtin_cce_vmull_*`

### `vpto.vmula`

- syntax:
  `%result = vpto.vmula %acc, %lhs, %rhs, %mask {mode = "MODE"} : !vpto.vec<NxT>, !vpto.vec<NxT>, !vpto.vec<NxT>, !vpto.mask -> !vpto.vec<NxT>`
- semantics:
  TODO(user): add one-line semantics for external developers.
- CCE correspondence:
  `vmula(...)`
  `__builtin_cce_vmula_*_m`

## 13. Stateless Stores

### `vpto.vsts`

- syntax:
  `vpto.vsts %value, %destination[%offset] {dist = "DIST"} : !vpto.vec<NxT>, !llvm.ptr<AS>`
- semantics:
  TODO(user): add one-line semantics for external developers.
- CCE correspondence:
  `vst(...)`, `vsts(...)`
  `__builtin_cce_vstx1_*`, `__builtin_cce_vstsx1_*`

### `vpto.vscatter`

- syntax:
  `vpto.vscatter %value, %destination, %offsets, %active_lanes : !vpto.vec<NxT>, !llvm.ptr<AS>, !vpto.vec<NxI>, index`
- semantics:
  TODO(user): add one-line semantics for external developers.
- CCE correspondence:
  `vscatter(...)`
  `__builtin_cce_vscatter_*`

### `vpto.vsts_pred`

- syntax:
  `vpto.vsts_pred %value, %destination[%offset], %active_lanes {dist = "DIST"} : !vpto.vec<NxT>, !llvm.ptr<AS>, index`
- semantics:
  TODO(user): add one-line semantics for external developers.
- CCE correspondence:
  predicated vector store family

### `vpto.psts`

- syntax:
  `vpto.psts %value, %destination[%offset] : !vpto.mask, !llvm.ptr<AS>`
- semantics:
  TODO(user): add one-line semantics for external developers.
- CCE correspondence:
  `psts(...)`
  `__builtin_cce_psts_b8`, `__builtin_cce_psts_post_b8`

### `vpto.pst`

- syntax:
  `vpto.pst %value, %destination[%offset], "DIST" : !vpto.mask, !llvm.ptr<AS>, index`
- semantics:
  TODO(user): add one-line semantics for external developers.
- CCE correspondence:
  `pst(...)`
  `__builtin_cce_pst_b8`

### `vpto.psti`

- syntax:
  `vpto.psti %value, %destination, %offset, "DIST" : !vpto.mask, !llvm.ptr<AS>, i32`
- semantics:
  TODO(user): add one-line semantics for external developers.
- CCE correspondence:
  `psti(...)`
  `__builtin_cce_psti_b8`, `__builtin_cce_psti_post_b8`

### `vpto.vsst`

- syntax:
  `vpto.vsst %value, %destination[%offset], "STRIDE" : !vpto.vec<NxT>, !llvm.ptr<AS>`
- semantics:
  TODO(user): add one-line semantics for external developers.
- CCE correspondence:
  `vsst(...)`
  `__builtin_cce_vsst_*`

### `vpto.vstx2`

- syntax:
  `vpto.vstx2 %low, %high, %destination[%offset], "DIST", %mask : !vpto.vec<NxT>, !vpto.vec<NxT>, !llvm.ptr<AS>, index, !vpto.mask`
- semantics:
  TODO(user): add one-line semantics for external developers.
- CCE correspondence:
  `vst(...)`
  `__builtin_cce_vstx2_*`

### `vpto.vsstb`

- syntax:
  `vpto.vsstb %value, %destination, %offset, %mask : !vpto.vec<NxT>, !llvm.ptr<AS>, i32, !vpto.mask`
- semantics:
  TODO(user): add one-line semantics for external developers.
- CCE correspondence:
  `vsstb(...)`
  `__builtin_cce_vsstb_*`, `__builtin_cce_vsstb_post_*`

### `vpto.vsta`

- syntax:
  `vpto.vsta %value, %destination[%offset] : !vpto.align, !llvm.ptr<AS>, index`
- semantics:
  TODO(user): add one-line semantics for external developers.
- CCE correspondence:
  `vsta(...)`
  `__builtin_cce_vsta_*`

### `vpto.vstas`

- syntax:
  `vpto.vstas %value, %destination, %offset : !vpto.align, !llvm.ptr<AS>, i32`
- semantics:
  TODO(user): add one-line semantics for external developers.
- CCE correspondence:
  `vstas(...)`
  `__builtin_cce_vstas_*`, `__builtin_cce_vstas_post_*`

### `vpto.vstar`

- syntax:
  `vpto.vstar %value, %destination : !vpto.align, !llvm.ptr<AS>`
- semantics:
  TODO(user): add one-line semantics for external developers.
- CCE correspondence:
  `vstar(...)`
  `__builtin_cce_vstar_*`

## 14. Stateful Store Ops

These ops make CCE reference-updated state explicit as SSA results.

### `vpto.pstu`

- syntax:
  `%align_out, %base_out = vpto.pstu %align_in, %value, %base : !vpto.align, !vpto.mask, !llvm.ptr<AS> -> !vpto.align, !llvm.ptr<AS>`
- semantics:
  TODO(user): add one-line semantics for external developers.
- CCE correspondence:
  `pstu(...)`
  `__builtin_cce_pstu_b16`, `__builtin_cce_pstu_b32`

### `vpto.vstu`

- syntax:
  `%align_out, %offset_out = vpto.vstu %align_in, %offset_in, %value, %base, "MODE" : !vpto.align, index, !vpto.vec<NxT>, !llvm.ptr<AS> -> !vpto.align, index`
- semantics:
  TODO(user): add one-line semantics for external developers.
- CCE correspondence:
  `vstu(...)`
  `__builtin_cce_vstu_*`

### `vpto.vstus`

- syntax:
  `%align_out, %base_out = vpto.vstus %align_in, %offset, %value, %base, "MODE" : !vpto.align, i32, !vpto.vec<NxT>, !llvm.ptr<AS> -> !vpto.align, !llvm.ptr<AS>`
- semantics:
  TODO(user): add one-line semantics for external developers.
- CCE correspondence:
  `vstus(...)`
  `__builtin_cce_vstus_*`, `__builtin_cce_vstus_post_*`

### `vpto.vstur`

- syntax:
  `%align_out = vpto.vstur %align_in, %value, %base, "MODE" : !vpto.align, !vpto.vec<NxT>, !llvm.ptr<AS> -> !vpto.align`
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

// %align1, %offset1 = vpto.vstu ...
// %align2, %base1 = vpto.vstus ...
// %align3 = vpto.vstur ...
```
