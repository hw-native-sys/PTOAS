# VPTO Spec

Updated: 2026-03-21

## Table Of Contents

- [Overview](#overview)
- [A5 VecCore Profile](#a5-veccore-profile)
- [Architectural State](#architectural-state)
- [Core Types](#core-types)
- [Address Space Conventions](#address-space-conventions)
- [Element Type Constraints](#element-type-constraints)
- [Special Types](#special-types)
- [Tokens And Control Fields](#tokens-and-control-fields)
- [A5 Instruction Inventory](#a5-instruction-inventory)
- [1. Sync And Resource Control](#1-sync-and-resource-control)
- [2. Vector Thread And Loop Control](#2-vector-thread-and-loop-control)
- [3. Vector, Predicate, And Align Loads](#3-vector-predicate-and-align-loads)
- [4. Vector, Predicate, And Align Stores](#4-vector-predicate-and-align-stores)
- [5. Reduction, Conversion, And Sort](#5-reduction-conversion-and-sort)

## Overview

This document defines the VPTO ISA contract for the A5 VecCore profile used by
PTOAS. VPTO preserves ISA-visible behavior while removing binary encodings and
making selected hidden state explicit in SSA form.

## A5 VecCore Profile

This profile is limited to the A5 VecCore instruction and register surface. It
models the vector core itself, the scalar-side control that launches or
configures vector execution, and the Vector tile buffer storage seen by VecCore
load, store, sort, and histogram instructions.

Enabled A5 extensions in this profile: `Basic Set`, `int8 add Extension`, `int16 add Extension`, `int16 mul Extension`, `int16 redux Extension`, `int redux Extension`, `F32 Extension`, `FMIX Extension`, `S64 Extension`, `BF16 Extension`, `Sort F32 Extension`, `HIST Extension`, `S64 CVT Extension`, `BF16 CVT Extension`, `HiF8 CVT Extension`, `FP8 CVT Extension`, `FP4 CVT Extension`.

The programmer's model has three structural domains:

- Main-scalar control domain: launches vector-thread work, pushes parameter
  buffers, configures loop topology, configures address generation, and stores
  or clears selected SPR state.
- Vector-thread execution domain: executes VecCore instructions over vector
  registers, predicate state, align state, shared-register inputs, and selected
  SPR state.
- Vector tile buffer domain: holds the byte-addressed storage consumed and
  produced by VecCore loads, stores, gather, sort, and histogram families.

PTOAS compiles mixed tiles, vectors, and scalars by preserving that structural
boundary. `pto.t*` tile ops and `!pto.tile_buf<loc=vec,...>` describe tiled
views of the same Vector tile buffer storage that `pto.v*` families address
through `!llvm.ptr<6>`. Scalar and shared-register state remain architectural
state, not compiler-internal decoration.

## Architectural State

The A5 VecCore profile in this document uses the following architectural state:

- Vector registers: carried in VPTO as `!pto.vreg<NxT>` values.
- Predicate registers: carried in VPTO as `!pto.mask` values.
- Align state: carried in VPTO as `!pto.align` values for unaligned load and
  store streams.
- Shared registers and scalar registers: carried in VPTO as integer or pointer
  SSA operands whose width matches the consuming ISA family.
- Address-register and SPR state: modeled as explicit operands, attributes, or
  store/clear effects rather than hidden compiler temporaries.
- Vector tile buffer storage: addressed through `!llvm.ptr<6>` and shared with
  the tile-world `loc=vec` storage model.

Core SPRs listed for this profile:
`BLOCKDIM`, `BLOCKID`, `CLOCK`, `COREID`, `FFTS_THREAD_DIM`, `FFTS_THREAD_ID`, `LANEID`, `LANEMASK_EQ`, `LANEMASK_GE`, `LANEMASK_GT`, `LANEMASK_LE`, `LANEMASK_LT`, `SUBBLOCKDIM`, `SUBBLOCKID`, `SYS_VA_BASE`, `VECCOREID`, `VTHREADDIM.x`, `VTHREADDIM.y`, `VTHREADDIM.z`, `VTHREADID.x`, `VTHREADID.y`, `VTHREADID.z`, `PC`, `WARP_STACK_SIZE`

Vector SPRs listed for this profile:
`VPC`, `VL`, `SQZN`, `AR`, `VMS4_SR`, `CTRL`, `PAD_CNT_NDDMA`, `PAD_VAL_NDDMA`, `LOOP0_STRIDE_NDDMA`, `LOOP1_STRIDE_NDDMA`, `LOOP2_STRIDE_NDDMA`, `LOOP3_STRIDE_NDDMA`, `LOOP4_STRIDE_NDDMA`

## Core Types

- `vreg<T>`: `!pto.vreg<NxT>`
  Fixed-width VPTO vector type with total width exactly 256 bytes.
- `mask`: `!pto.mask`
  Opaque predicate-register type. Predicate granularity is defined by the
  consuming or producing ISA family, not by a type parameter.
- `align`: `!pto.align`
  Opaque align-state carrier for unaligned VecCore load and store streams.
- `buf`: buffer-like LLVM pointer type accepted by the dialect.
- `idx`: `index`
- `i32`: `i32`
- `i64`: `i64`

VPTO is an ISA-level contract without an encoding layer. Each `pto.v*` op below
preserves the visible ISA behavior of one A5 VecCore instruction or instruction
family, with only encoding details removed and with hidden state updates made
explicit where SSA needs them.

## Address Space Conventions

| `AS` | PTO mnemonic | Working interpretation in this spec |
|------|--------------|-------------------------------------|
| `0` | `Zero` | Default or unspecified pointer space; treated as GM-like by the current verifier |
| `1` | `GM` | Global Memory |
| `6` | `VEC` | Vector tile buffer |

This profile uses `!llvm.ptr<6>` for Vector tile buffer storage. A `loc=vec`
`!pto.tile_buf` value is a tile-shaped view of the same storage, not a separate
memory system.

## Element Type Constraints

- `!pto.vreg<NxT>` requires `N * bitwidth(T) = 2048`.
- `T` denotes the element type accepted by the named ISA family. In this A5
  profile that includes integer and floating-point lane types such as `i8`,
  `i16`, `i32`, `i64`, `f16`, `bf16`, and `f32`, subject to each op family's
  narrower legality rules.
- Conversion families use exact source and destination pairings from the A5
  profile inventory. `pto.vcvt` is not a free-form bitcast.
- Offset and index vectors use integer element types.
- Reduction families may place a scalar reduction result in the low lane of a
  vector-shaped carrier and zero-fill the remaining lanes.

## Special Types

### `!pto.mask`

`!pto.mask` models an A5 predicate register. It is not an integer vector.
Predicate granularity comes from the instruction family. In this profile it is
produced by predicate-load families and consumed by predicate-store,
stateful-store, reduction-compression, and histogram families.

### `!pto.align`

`!pto.align` models the A5 align-state carrier used by `VLDAS`, `VLDU` /
`VLDUS` / `VLDUI`, `VSTA` / `VSTAI` / `VSTAS`, `VSTU` / `VSTUI`, `VSTUS`,
`VSTUR`, and `VSTAR`. It is explicit in VPTO so the unaligned stream state is
visible to verification and lowering.

## Tokens And Control Fields

This profile uses the following control tokens and fields in op syntax:

- `SRC_PIPE`, `DST_PIPE`, `EVENT_ID`: synchronization channel selectors used by
  `pto.vset_flag` and `pto.vwait_flag`.
- `DIST`: distribution selector used by aligned vector load and store families
  and predicate load and store families.
- `MODE`: post-update or family submode selector used by stateful stores.
- `ROUND_MODE`, `SAT_MODE`, `PART_MODE`: conversion control fields used by
  `pto.vcvt`.
- `layer`, `last`, `target_mode`, and `dispatch`: loop and fetch-control fields
  that preserve the scalar-side VecCore launch contract.

## A5 Instruction Inventory

The A5 VecCore profile in this document is limited to the ISA instructions and special registers listed below. Every instruction not enumerated here is outside this profile.

### Vector Control And Sync

- Basic Set: `MOVEMASK`, `MOVEVA`, `LDVA`, `PUSH_PB`, `VFI`, `VF`, `VF_SIMT`, `VFI_RU`, `VF_RU`, `RELEASE_PBID`, `MOV_UB_TO_UB`, `MOV_VSPR`, `VLOOPv2`, `VAG`, `VTRANSPOSE` (`b16`), `VBS32` (`f16`), `VMS4v2` (`f16`), `VNCHWCONV` (`b8`, `b16`, `b32`)
- Sort F32 Extension: `VBS32` (`f32`), `VMS4v2` (`f32`)
- Block Sync Extension: `WAIT_FLAG_DEV_V`, `WAIT_FLAG_DEVI_V`, `SET_CROSS_CORE_V`
- BUFID Extension: `GET_BUF_V`, `GET_BUFI_V`, `RLS_BUF_V`, `RLS_BUFI_V`
- MIX-MODE Extension: `SET_INTRA_BLOCK_V`, `SET_INTRA_BLOCKI_V`, `WAIT_INTRA_BLOCK_V`, `WAIT_INTRA_BLOCKI_V`
- DFX Extension: `DFX_REGION_V`

### Scalar Control And Memory Inventory

- control_flow: `BRANCH_IND`, `BRANCH`, `CALL`, `CALLI`, `CALL_IND`, `END_DVG`, `END`, `RET`, `START_DVG`
- misc: `BREV`, `CLZ`, `MOV`, `MOVI`, `P2R`, `POPC`, `PRMT`, `R2P`, `S2R`, `S2RL`, `SEL`
- sync_and_barrier: `BAR.THREAD_BLOCK`, `JOIN`, `NOP`, `MEMBAR`
- stores_and_cache: `DCCI`, `ST`, `STG`, `STK`, `STS`
- warp_collective: `REDUX`, `SHFL`, `VOTE.ballot`, `VOTE`

### Inner-Scalar Control Inventory

- control: `SJUMP`, `SJUMPI`, `SNOP`, `SEND`, `FORK`

### Vector Load Inventory

- Supported mnemonics: `VLD`, `VLDS`, `VLDI`, `VLDA`, `VLDAS`, `VLDU`, `VLDUS`, `VLDUI`, `VSLDB`, `PLD`, `PLDS`, `PLDI`, `VGATHERB`

### Vector Store Inventory

- Supported mnemonics: `VST`, `VSTI`, `VSTS`, `VSTA`, `VSTAI`, `VSTAS`, `VSTU`, `VSTUI`, `VSTUS`, `PST`, `PSTS`, `PSTI`, `VSTUR`, `VSTAR`, `SPRSTI`, `SPRSTS`, `SPRCLR`

### Vector Move And Predicate Logic Inventory

- No additional A5 VecCore instructions are listed in this category.

### Vector Arithmetic Inventory

- No additional A5 VecCore instructions are listed in this category.

### Vector Reduction Inventory

- `VSQZ`: `b8`, `b16`, `b32`
- `VUSQZ`: `b8`, `b16`, `b32`
- `VCADD`: `u16`, `s16`, `u32`, `s32`, `f16`, `f32`
- `VCMAX`: `u16`, `s16`, `u32`, `s32`, `f16`, `f32`
- `VCMIN`: `u16`, `s16`, `u32`, `s32`, `f16`, `f32`
- `VCGADD`: `u16`, `s16`, `u32`, `s32`, `f16`, `f32`
- `VCGMAX`: `u16`, `s16`, `u32`, `s32`, `f16`, `f32`
- `VCGMIN`: `u16`, `s16`, `u32`, `s32`, `f16`, `f32`
- `VCPADD`: `f16`, `f32`
- `DHISTv2`: `base`
- `CHISTv2`: `base`

### Vector Convert Inventory

- `VCVTFI`: `f322s16`, `f322s32`, `f322s64`, `f162u8`, `f162s8`, `f162s16`, `f162s32`, `f162s4`, `bf162s32`
- `VCVTFF`: `f322f16`, `f322bf16`, `f322HiF8`, `f322e4m3`, `f322e5m2`, `f162f32`, `f162bf16`, `f162HiF8`, `bf162f32`, `bf162f16`, `HiF82f32`, `HiF82f16`, `e4m32f32`, `e5m22f32`
- `VCVTFF2`: `e2m12bf16`, `e1m22bf16`, `bf162e2m1`, `bf162e1m2`
- `VCVTIF`: `s42f16`, `s42bf16`, `u82f16`, `s82f16`, `s162f32`, `s162f16`, `s322f32`, `s642f32`
- `VCVTII`: `u82u16`, `u82u32`, `s82s16`, `s82s32`, `u162u8`, `u162u32`, `s162u8`, `s162u32`, `s162s32`, `u322u8`, `u322u16`, `u322s16`, `s322u8`, `s322u16`, `s322s16`, `s322s64`, `s642s32`, `s42s16`, `s162s4`

Instructions that appear in the inventory but do not yet have a standalone `pto.v*` heading below are still part of the A5 ISA profile. This document records them in the inventory and keeps the detailed SSA contract only for the currently surfaced VPTO families.

## 1. Sync And Resource Control

These ops execute in the scalar-control domain and affect later VecCore
execution through ordering, resource ownership, or parameter-buffer state.

### `pto.vset_flag`

- covered ISA mnemonics:
  `SET_CROSS_CORE_V`, `SET_INTRA_BLOCK_V`, `SET_INTRA_BLOCKI_V`
- syntax:
  `pto.vset_flag["SRC_PIPE", "DST_PIPE", "EVENT_ID"]`
- operand roles:
  `"SRC_PIPE"` names the producer pipeline or source side of the flag,
  `"DST_PIPE"` names the consumer pipeline or destination side of the flag,
  and `"EVENT_ID"` names the architecturally visible event channel.
- data types:
  `SRC_PIPE`, `DST_PIPE`, and `EVENT_ID` are symbolic ISA selectors carried as
  string attributes in VPTO.
- ISA family:
  `SET_CROSS_CORE_V` / `SET_INTRA_BLOCK_V` / `SET_INTRA_BLOCKI_V`
- semantics:
  Publishes one synchronization event from `SRC_PIPE` to `DST_PIPE`. The event
  becomes visible to the matching wait instruction on the named channel and
  orders the corresponding producer and consumer domains exactly as the ISA
  flag form specifies. Immediate and register-configured forms are both carried
  by the same VPTO event triple; VPTO removes the encoding difference but not
  the synchronization effect.

### `pto.vwait_flag`

- covered ISA mnemonics:
  `WAIT_FLAG_DEV_V`, `WAIT_FLAG_DEVI_V`, `WAIT_INTRA_BLOCK_V`, `WAIT_INTRA_BLOCKI_V`
- syntax:
  `pto.vwait_flag["SRC_PIPE", "DST_PIPE", "EVENT_ID"]`
- operand roles:
  `"SRC_PIPE"` names the producer side being observed, `"DST_PIPE"` names the
  consumer side that is stalled, and `"EVENT_ID"` selects the architecturally
  visible event channel.
- data types:
  `SRC_PIPE`, `DST_PIPE`, and `EVENT_ID` are symbolic ISA selectors carried as
  string attributes in VPTO.
- ISA family:
  `WAIT_FLAG_DEV_V` / `WAIT_FLAG_DEVI_V` / `WAIT_INTRA_BLOCK_V` / `WAIT_INTRA_BLOCKI_V`
- semantics:
  Blocks the consumer side until the matching event published for the same
  `(SRC_PIPE, DST_PIPE, EVENT_ID)` tuple is observed. The wait does not retire
  early, and it preserves the ISA ordering relation between the producer and
  consumer domains. Immediate and register-configured wait forms differ only in
  encoding; the visible synchronization contract is the same.

### `pto.vget_buf`

- covered ISA mnemonics:
  `GET_BUF_V`, `GET_BUFI_V`
- syntax:
  `pto.vget_buf "PIPE_*", %buf_id, %mode : i64, i64`
- operand roles:
  `"PIPE_*"` selects the owning pipeline, `%buf_id` is the requested buffer
  identifier or immediate-selected slot, and `%mode` carries the ISA-defined
  acquisition mode bits.
- ISA family:
  `GET_BUF_V` / `GET_BUFI_V`
- semantics:
  Acquires the named VecCore buffer resource on the selected pipeline. The
  operation reserves the ISA-defined slot or token selected by `%buf_id` and
  `%mode`, and later instructions may rely on that reservation until the
  matching release completes.

### `pto.vrls_buf`

- covered ISA mnemonics:
  `RLS_BUF_V`, `RLS_BUFI_V`
- syntax:
  `pto.vrls_buf "PIPE_*", %buf_id, %mode : i64, i64`
- operand roles:
  `"PIPE_*"` selects the owning pipeline, `%buf_id` is the buffer identifier
  or immediate-selected slot being released, and `%mode` carries the ISA-defined
  release mode bits.
- ISA family:
  `RLS_BUF_V` / `RLS_BUFI_V`
- semantics:
  Releases the previously held VecCore buffer resource selected by `%buf_id`
  and `%mode` on the chosen pipeline. After retirement, later acquisition ops
  may reuse that resource exactly as the ISA permits.

## 2. Vector Thread And Loop Control

These ops launch vector-thread work, define loop topology, configure address
progression, or store and clear architecturally visible SPR state.

### `pto.vthread_fetch`

- covered ISA mnemonics:
  `VFI`, `VF`, `VF_SIMT`, `VFI_RU`, `VF_RU`
- syntax:
  `pto.vthread_fetch %target, %instr_count {target_mode = "pc_rel|absolute", dispatch = "scalar|simt", pbid = %pbid?} : T, i16`
- operand roles:
  `%target` is either the relative vector-PC delta or the absolute vector-PC
  start address, `%instr_count` is the unsigned instruction count of the fetched
  vector body, `dispatch` selects ordinary vector-thread execution versus the
  SIMT dispatch form, and `pbid` names the parameter-buffer resource attached to
  RU forms.
- data types:
  `%target` is a signed 16-bit PC-relative displacement for `VFI`-family forms
  and a scalar absolute address for `VF`-family forms. `%instr_count` is an
  unsigned 16-bit count.
- semantics:
  Starts one vector-thread fetch window. For `target_mode = "pc_rel"`, the
  fetched vector PC equals `sign_extend(target * 4) + PC_of_main_scalar`, where
  `PC_of_main_scalar` is the address of the first 4-byte slot of the fetch
  instruction. For `target_mode = "absolute"`, the fetched vector PC equals
  `%target`. RU forms additionally bind the fetch to `PBID`. `dispatch = "simt"` preserves the ISA SIMT launch behavior of `VF_SIMT`.
- assertions and exceptions:
  One fetch op launches exactly one vector body. `%instr_count` MUST match the
  ISA-visible body length that follows the fetch target.

### `pto.vparam_buffer`

- covered ISA mnemonics:
  `PUSH_PB`, `RELEASE_PBID`
- syntax:
  `pto.vparam_buffer "push" %xd, %xn, %xm, %xt : T, T, T, T`
  `pto.vparam_buffer "release" %pbid : T`
- operand roles:
  `%xd`, `%xn`, `%xm`, and `%xt` are the four scalar words pushed into the
  parameter buffer in order; `%pbid` is the parameter-buffer identifier being
  released.
- data types:
  Push consumes four scalar register-width words and writes one 256-bit PB
  entry. Release consumes one scalar resource identifier.
- semantics:
  `push` writes `{%xt, %xm, %xn, %xd}` into the next vector-thread parameter
  buffer entry. In a multi-push parameter sequence, the lowest 32 bits of the
  first pushed word are the SREG update bitmap; each set bit names one pair of
  16-bit SREGs to initialize, and the reserved bitmap bits for the special SREG
  slots, including the low pair and the final two pairs, MUST remain clear.
  Additional pushes extend the same PB slot and do not carry a new bitmap.
  `release` returns a previously used parameter-buffer identifier so software
  may reuse that slot after the retained-parameter sequence is finished.

### `pto.vloop`

- covered ISA mnemonics:
  `VLOOPv2`
- syntax:
  `pto.vloop %count, %instr_count {layer = i4, last = true|false} : T, i16`
- operand roles:
  `%count` is the unsigned loop trip count, `%instr_count` is the instruction
  count of the loop body, and `layer` plus `last` describe the ISA nesting
  topology for this loop header.
- data types:
  `%count` is a 16-bit unsigned loop count. `%instr_count` is a non-zero 16-bit
  unsigned body length.
- semantics:
  Begins one counted hardware vector loop at the selected loop layer. If `%count`
  is zero, the loop body is skipped. Otherwise the vector thread executes the
  next `%instr_count` instructions `%count` times using the ISA loop nesting and
  loop-boundary rules of `VLOOPv2`.
- assertions and exceptions:
  `%instr_count` MUST be non-zero. The `layer` / `last` fields MUST encode a
  valid nesting topology. The bound register used by the loop MUST NOT be
  post-updated inside that loop body. There MUST NOT be any `SJUMP`, `SJUMPI`,
  or `SEND` instruction inside the loop body.

### `pto.vaddr_gen`

- covered ISA mnemonics:
  `VAG`
- syntax:
  `pto.vaddr_gen %ad, %i1_stride, %i2_stride, %i3_stride, %i4_stride : T, T, T, T, T`
- operand roles:
  `%ad` is the destination address-register state, and the four stride operands
  are the loop-layer contributions for `i1` through `i4`.
- data types:
  The stride operands are unsigned shared-register values interpreted in address
  units for the corresponding loop layer.
- semantics:
  Defines the affine address evolution used by vector load/store instructions:
  `Ad = const1*i1 + const2*i2 + const3*i3 + const4*i4`, omitting absent outer
  layers. When multiple declarations target the same address register, the last
  declaration overwrites earlier ones.
- assertions and exceptions:
  `VAG` MUST be outside `VLOOPv2` bodies. Its source shared registers are drawn
  from the architecturally fixed `S0..S31` set and preserve the ISA even-ID
  rule for 32-bit address contributions. When the address register is also used
  by post-update load/store forms, the resulting address stream MUST remain
  consecutive across loop boundaries.

### `pto.vspr_store`

- covered ISA mnemonics:
  `SPRSTI`, `SPRSTS`, `SPRCLR`
- syntax:
  `pto.vspr_store "store_imm" %spr, %base, %offset {post_update = true|false}`
  `pto.vspr_store "store_reg" %spr, %base, %offset_reg {post_update = true|false}`
  `pto.vspr_store "clear" %spr`
- operand roles:
  `%spr` is the special register being stored or cleared, `%base` is the base
  address shared register, `%offset` is the signed immediate displacement,
  `%offset_reg` is the signed byte displacement from a shared register, and
  `post_update` selects whether the base register is updated after the store.
- data types:
  `%base` is a shared-register address value, `%offset` is a signed 8-bit
  immediate scaled by the stored-SPR width, and `%offset_reg` is a signed
  32-bit byte displacement.
- semantics:
  Stores the selected SPR to memory at the architecturally addressed location or
  clears the SPR to zero. If `post_update` is set, the base register itself is
  used as the store address and then advanced by the scaled immediate or by the
  register offset.
- assertions and exceptions:
  The access address MUST satisfy the alignment requirement of the named SPR.
  Current ISA text only admits `AR` as the stored or cleared SPR in this family.

### `pto.vmove_ub`

- covered ISA mnemonics:
  `MOV_UB_TO_UB`
- syntax:
  `pto.vmove_ub %dst, %src, %config : T, T, T`
- operand roles:
  `%dst` and `%src` are Vector tile buffer addresses, and `%config` is the ISA-defined control
  word that determines the transfer submode.
- data types:
  `%dst` and `%src` are Vector tile buffer address operands. `%config` is a family-specific
  scalar control word.
- semantics:
  Performs the Vector-tile-buffer-to-Vector-tile-buffer movement defined by the configuration word. VPTO treats
  the addressing, element grouping, and submode fields as part of the
  architectural contract of `%config`; no encoding bits are elided.
- assertions and exceptions:
  This family remains configuration-driven. Any frontend that lowers to this
  VPTO family MUST preserve the full control word rather than rewriting it into
  a narrower helper sequence.

## 3. Vector, Predicate, And Align Loads

All loads in this chapter address the Vector tile buffer through `!llvm.ptr<6>`.
The selected load form determines alignment, distribution, and whether the load
produces vector, predicate, or align state.

### `pto.vlds`

- covered ISA mnemonics:
  `VLD`, `VLDS`, `VLDI`
- syntax:
  `%result = pto.vlds %source[%offset] {dist = "DIST"} : !llvm.ptr<AS> -> !pto.vreg<NxT>`
- operand roles:
  `%source` is the Vector tile buffer base pointer, `%offset` is the load displacement from that base, `"DIST"` selects the ISA distribution mode, and `%result` is the loaded vector.
- ISA family:
  `VLD` / `VLDS` / `VLDI`
- semantics:
  Let `addr = source + offset`. `pto.vlds` performs the aligned load form
  selected by `DIST`. `NORM` reads one full vector from `addr`. Broadcast and
  unpack forms read the ISA-defined smaller source footprint and expand it into
  the destination lane layout. Any dual-destination distribution is outside this A5 profile. `addr` MUST satisfy the alignment rule of the selected
  distribution token.
- CCE correspondence:
  `vld(...)`, `vlds(...)`
  `__builtin_cce_vldsx1_*`
  related extended families:
  `__builtin_cce_vldix1_*`, `__builtin_cce_vldsx1_post_*`

### `pto.vldas`

- covered ISA mnemonics:
  `VLDA`, `VLDAS`
- syntax:
  `%result = pto.vldas %source[%offset] : !llvm.ptr<AS> -> !pto.align`
- operand roles:
  `%source` is the Vector tile buffer base pointer, `%offset` is the align-initialization displacement, and `%result` is the produced align state.
- ISA family:
  `VLDA` / `VLDAS`
- semantics:
  Let `addr = source + offset`. `pto.vldas` reads the 32-byte block at
  `floor(addr / 32) * 32`, records the low address bits of `addr`, and produces
  the align carrier required by a subsequent unaligned load stream. `addr`
  itself need not be 32-byte aligned.
- CCE correspondence:
  `vldas(...)`
  `__builtin_cce_vldas_*`

### `pto.vldus`

- covered ISA mnemonics:
  `VLDU`, `VLDUS`, `VLDUI`
- syntax:
  `%result = pto.vldus %align, %source[%offset] : !pto.align, !llvm.ptr<AS> -> !pto.vreg<NxT>`
- operand roles:
  `%align` is the incoming align state, `%source` is the Vector tile buffer base pointer, `%offset` is the load displacement, and `%result` is the assembled vector result.
- ISA family:
  `VLDU` / `VLDUS` / `VLDUI`
- semantics:
  Let `addr = source + offset` and let `aligned_tmp = ceil(addr / 32) * 32`.
  `pto.vldus` forms the returned vector by concatenating the bytes in `%align`
  that cover `[addr, aligned_tmp)` with the bytes fetched from the newly loaded
  aligned vector that cover `[aligned_tmp, aligned_tmp + VL - (aligned_tmp -
  addr))`. If `addr` is already 32-byte aligned, the result is the newly loaded
  aligned vector itself. `%align` MUST have been produced by a matching
  `pto.vldas` stream before the first dependent `pto.vldus`.
- CCE correspondence:
  `vldus(...)`
  `__builtin_cce_vldus_*`, `__builtin_cce_vldus_post_*`

### `pto.vplds`

- covered ISA mnemonics:
  `PLDS`
- syntax:
  `%result = pto.vplds %source[%offset] {dist = "DIST"} : !llvm.ptr<AS> -> !pto.mask`
- operand roles:
  `%source` is the Vector tile buffer base pointer, `%offset` is the load displacement, `"DIST"` selects the predicate-load distribution, and `%result` is the loaded predicate.
- ISA family:
  `PLDS`
- semantics:
  Loads predicate state from `source + offset` using the selected predicate
  distribution. `DIST = "NORM"` loads `VL/8` bytes directly, `DIST = "US"`
  loads `VL/16` bytes and duplicates each loaded bit twice, and `DIST = "DS"`
  loads `2*VL/8` bytes and keeps every other bit. The effective address MUST
  satisfy the alignment rule of the selected distribution.
- CCE correspondence:
  `plds(...)`
  `__builtin_cce_plds_b8`

### `pto.vpld`

- covered ISA mnemonics:
  `PLD`
- syntax:
  `%result = pto.vpld %source[%offset], "DIST" : !llvm.ptr<AS>, index -> !pto.mask`
- operand roles:
  `%source` is the Vector tile buffer base pointer, `%offset` is the index-style displacement, `"DIST"` selects the predicate-load distribution, and `%result` is the loaded predicate.
- ISA family:
  `PLD`
- semantics:
  Loads predicate state from `source + offset` using the selected predicate-load
  distribution token. `NORM`, `US`, and `DS` preserve the same bit-level
  layouts and alignment rules as the corresponding `PLDS` forms.
- CCE correspondence:
  `pld(...)`
  `__builtin_cce_pld_b8`

### `pto.vpldi`

- covered ISA mnemonics:
  `PLDI`
- syntax:
  `%result = pto.vpldi %source, %offset, "DIST" : !llvm.ptr<AS>, i32 -> !pto.mask`
- operand roles:
  `%source` is the Vector tile buffer base pointer, `%offset` is the immediate-style scalar displacement, `"DIST"` selects the predicate-load distribution, and `%result` is the loaded predicate.
- ISA family:
  `PLDI`
- semantics:
  Loads predicate state from `source + offset` using the immediate-offset
  predicate-load form. The offset is scaled by the alignment size of the chosen
  distribution token exactly as in the ISA immediate form.
- CCE correspondence:
  `pldi(...)`
  `__builtin_cce_pldi_b8`, `__builtin_cce_pldi_post_b8`

### `pto.vgatherb`

- covered ISA mnemonics:
  `VGATHERB`
- syntax:
  `%result = pto.vgatherb %source, %offsets, %active_lanes : !llvm.ptr<AS>, !pto.vreg<NxI>, index -> !pto.vreg<NxT>`
- operand roles:
  `%source` is the Vector tile buffer base pointer, `%offsets` is the per-lane offset vector, `%active_lanes` bounds how many lanes participate, and `%result` is the gathered vector.
- ISA family:
  `VGATHERB`
- semantics:
  `pto.vgatherb` is the block-gather form, not a byte-element gather. For each
  active block `i < active_lanes`, it computes `block_addr[i] = source +
  offsets[i]`, where each offset is a 32-bit byte offset that MUST be 32-byte
  aligned, and loads one 32-byte block from `block_addr[i]` into block `i` of
  the destination vector. `%source` MUST be 32-byte aligned. Inactive blocks do
  not issue memory requests and their destination block is zeroed.
- CCE correspondence:
  `vgatherb(...)`
  `__builtin_cce_vgatherb_*`, `__builtin_cce_vgatherb_v300_*`, `__builtin_cce_vgatherb_v310_*`

### `pto.vsldb`

- covered ISA mnemonics:
  `VSLDB`
- syntax:
  `%result = pto.vsldb %source, %offset, %mask : !llvm.ptr<AS>, i32, !pto.mask -> !pto.vreg<NxT>`
- operand roles:
  `%source` is the Vector tile buffer base pointer, `%offset` is the scalar displacement, `%mask` is the predicate control, and `%result` is the loaded vector.
- ISA family:
  `VSLDB`
- semantics:
  Interprets `%offset` as the packed block-stride configuration word whose
  upper 16 bits are the block stride and whose lower 16 bits are the repeat
  stride. The op loads `VL_BLK` 32-byte blocks. If any bit in the governing
  32-bit predicate slice of a block is 1, the whole block is loaded. If that
  predicate slice is all 0, the block load is suppressed, the destination block
  is zeroed, and no overflow exception is raised for that block address.
- CCE correspondence:
  `vsldb(...)`
  `__builtin_cce_vsldb_*`, `__builtin_cce_vsldb_post_*`

## 4. Vector, Predicate, And Align Stores

All stores in this chapter write the Vector tile buffer through `!llvm.ptr<6>`.
Stateless forms address one explicit destination, while stateful forms make the
ISA align or post-update chain explicit in SSA form.

### `pto.vsts`

- covered ISA mnemonics:
  `VST`, `VSTI`, `VSTS`
- syntax:
  `pto.vsts %value, %destination[%offset] {dist = "DIST"} : !pto.vreg<NxT>, !llvm.ptr<AS>`
- operand roles:
  `%value` is the vector being stored, `%destination` is the Vector tile buffer base pointer, `%offset` is the store displacement, and `"DIST"` selects the ISA store distribution mode.
- ISA family:
  `VST` / `VSTI` / `VSTS`
- semantics:
  Stores `%value` to `destination + offset` using the store form selected by
  `DIST`. `DIST` determines the stored element width, lane layout, packing or
  channel-merge behavior, and the required destination alignment. Interleaving two-source store forms are outside this A5 profile.
- CCE correspondence:
  `vst(...)`, `vsts(...)`
  `__builtin_cce_vstx1_*`, `__builtin_cce_vstsx1_*`

### `pto.vpsts`

- covered ISA mnemonics:
  `PSTS`
- syntax:
  `pto.vpsts %value, %destination[%offset] : !pto.mask, !llvm.ptr<AS>`
- operand roles:
  `%value` is the predicate being stored, `%destination` is the Vector tile buffer base pointer, and `%offset` is the store displacement.
- ISA family:
  `PSTS`
- semantics:
  Stores predicate state to `destination + offset`. The stored predicate data
  type is always `b8`, regardless of the surrounding vector element type. The
  effective address MUST satisfy the alignment rule of the predicate-store
  family.
- CCE correspondence:
  `psts(...)`
  `__builtin_cce_psts_b8`, `__builtin_cce_psts_post_b8`

### `pto.vpst`

- covered ISA mnemonics:
  `PST`
- syntax:
  `pto.vpst %value, %destination[%offset], "DIST" : !pto.mask, !llvm.ptr<AS>, index`
- operand roles:
  `%value` is the predicate being stored, `%destination` is the Vector tile buffer base pointer, `%offset` is the store displacement, and `"DIST"` selects the predicate-store distribution token.
- ISA family:
  `PST`
- semantics:
  Stores predicate state to `destination + offset` using the predicate-store
  distribution token. `DIST = "NORM"` stores the full `VL/8` predicate image.
  `DIST = "PK"` packs the source predicate by keeping every other bit and
  stores `VL/16` bytes.
- CCE correspondence:
  `pst(...)`
  `__builtin_cce_pst_b8`

### `pto.vpsti`

- covered ISA mnemonics:
  `PSTI`
- syntax:
  `pto.vpsti %value, %destination, %offset, "DIST" : !pto.mask, !llvm.ptr<AS>, i32`
- operand roles:
  `%value` is the predicate being stored, `%destination` is the Vector tile buffer base pointer, `%offset` is the scalar displacement, and `"DIST"` selects the predicate-store distribution token.
- ISA family:
  `PSTI`
- semantics:
  Stores predicate state using the immediate-offset predicate-store form. The
  offset is scaled by the alignment size of the chosen distribution token
  exactly as in the ISA immediate form.
- CCE correspondence:
  `psti(...)`
  `__builtin_cce_psti_b8`, `__builtin_cce_psti_post_b8`

### `pto.vsta`

- covered ISA mnemonics:
  `VSTA`, `VSTAI`
- syntax:
  `pto.vsta %value, %destination[%offset] : !pto.align, !llvm.ptr<AS>, index`
- operand roles:
  `%value` is the align payload being stored, `%destination` is the Vector tile buffer base pointer, and `%offset` is the store displacement.
- ISA family:
  `VSTA` / `VSTAI`
- semantics:
  Flushes the valid tail bytes buffered in `%value` to the aligned Vector tile
  buffer address determined by `dst_addr = destination + offset` and
  `aligned_addr = floor(dst_addr / 32) * 32`. The flush address MUST equal the
  post-updated address of the last dependent unaligned-store stream that wrote
  `%value`. After the flush, the align flag is cleared.
- CCE correspondence:
  `vsta(...)`
  `__builtin_cce_vsta_*`

### `pto.vstas`

- covered ISA mnemonics:
  `VSTAS`
- syntax:
  `pto.vstas %value, %destination, %offset : !pto.align, !llvm.ptr<AS>, i32`
- operand roles:
  `%value` is the align payload being stored, `%destination` is the Vector tile buffer base pointer, and `%offset` is the scalar displacement.
- ISA family:
  `VSTAS`
- semantics:
  Performs the same buffered-tail flush as `pto.vsta`, but with the scalar
  register offset form of the addressed flush.
- CCE correspondence:
  `vstas(...)`
  `__builtin_cce_vstas_*`, `__builtin_cce_vstas_post_*`

### `pto.vstar`

- covered ISA mnemonics:
  `VSTAR`
- syntax:
  `pto.vstar %value, %destination : !pto.align, !llvm.ptr<AS>`
- operand roles:
  `%value` is the align payload being stored and `%destination` is the base pointer used by the register-update store form.
- ISA family:
  `VSTAR`
- semantics:
  Flushes the buffered tail bytes in `%value` using the base-plus-`AR`
  addressing form of the ISA. The address implied by the live `AR` SPR MUST
  equal the post-updated address of the last dependent `pto.vstur` stream.
  After the flush, the align flag is cleared.
- CCE correspondence:
  `vstar(...)`
  `__builtin_cce_vstar_*`

### `pto.vstu`

- covered ISA mnemonics:
  `VSTU`, `VSTUI`
- syntax:
  `%align_out, %offset_out = pto.vstu %align_in, %offset_in, %value, %base, "MODE" : !pto.align, index, !pto.vreg<NxT>, !llvm.ptr<AS> -> !pto.align, index`
- operand roles:
  `%align_in` is the incoming align state, `%offset_in` is the current index displacement, `%value` is the vector being stored, `%base` is the current base pointer, `"MODE"` selects post-update behavior, `%align_out` is the updated align state, and `%offset_out` is the updated index displacement.
- ISA family:
  `VSTU` / `VSTUI`
- semantics:
  Stores `%value` through the stateful unaligned-store form addressed by
  `dst_addr = base + offset_in`. The store merges any valid prefix bytes held in
  `%align_in` with the new vector data, commits every byte that reaches an
  aligned 32-byte store boundary, and returns the residual suffix in
  `%align_out`. If `MODE` is `POST_UPDATE`, `%offset_out` is the ISA successor
  displacement after advancing by one vector length; otherwise `%offset_out`
  preserves `%offset_in`.
- CCE correspondence:
  `vstu(...)`
  `__builtin_cce_vstu_*`

### `pto.vstus`

- covered ISA mnemonics:
  `VSTUS`
- syntax:
  `%align_out, %base_out = pto.vstus %align_in, %offset, %value, %base, "MODE" : !pto.align, i32, !pto.vreg<NxT>, !llvm.ptr<AS> -> !pto.align, !llvm.ptr<AS>`
- operand roles:
  `%align_in` is the incoming align state, `%offset` is the variable byte count
  to store and the post-update distance of the ISA form, `%value` is the vector
  being stored, `%base` is the current base pointer, `"MODE"` selects
  post-update behavior, `%align_out` is the updated align state, and
  `%base_out` is the updated base pointer.
- ISA family:
  `VSTUS`
- semantics:
  Stores only the least-significant `%offset` bytes of `%value` through the
  variable-size unaligned-store form. Bytes that do not yet complete a 32-byte
  aligned destination block remain buffered in `%align_out`; aligned destination
  bytes are committed immediately. `%base_out` is the ISA successor base when
  `MODE` requests post-update; otherwise it preserves the incoming base.
- CCE correspondence:
  `vstus(...)`
  `__builtin_cce_vstus_*`, `__builtin_cce_vstus_post_*`

### `pto.vstur`

- covered ISA mnemonics:
  `VSTUR`
- syntax:
  `%align_out = pto.vstur %align_in, %value, %base, "MODE" : !pto.align, !pto.vreg<NxT>, !llvm.ptr<AS> -> !pto.align`
- operand roles:
  `%align_in` is the incoming align state, `%value` is the vector being stored, `%base` is the current base pointer, `"MODE"` selects post-update behavior, and `%align_out` is the updated align state.
- ISA family:
  `VSTUR`
- semantics:
  Stores a variable-size suffix of `%value` through the register-update
  unaligned-store form. The effective address is `base + AR`, the stored byte
  count is the live value of `SQZN`, and `%align_out` carries the residual
  buffered tail after committing every full 32-byte aligned destination block.
  If `MODE` requests post-update, the live `AR` SPR is advanced by `SQZN`; if
  not, `AR` is preserved.
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

## 5. Reduction, Conversion, And Sort

This chapter covers the A5 VecCore profile's supported reduction,
compression, conversion, sort, and histogram families.

### `pto.vcadd`

- covered ISA mnemonics:
  `VCADD`
- syntax:
  `%result = pto.vcadd %input : !pto.vreg<NxT> -> !pto.vreg<NxT>`
- operand roles:
  `%input` is the source vector and `%result` is the vector-shaped reduction
  result carrier.
- data types:
  `u16`, `s16`, `u32`, `s32`, `f16`, `f32`
- ISA family:
  `VCADD`
- semantics:
  Reduces all active source elements to one sum and writes that reduced value to
  the lowest destination lane. Every remaining destination lane is zero-filled.
  Inactive lanes contribute `0` for integer types and `+0` for floating-point
  types. Frontends MUST preserve the ISA reduction width of the low lane; for
  example, 16-bit integer inputs reduce into the ISA 32-bit result width.

### `pto.vcmax`

- covered ISA mnemonics:
  `VCMAX`
- syntax:
  `%result = pto.vcmax %input : !pto.vreg<NxT> -> !pto.vreg<NxT>`
- operand roles:
  `%input` is the source vector and `%result` is the vector-shaped reduction
  result carrier.
- data types:
  `u16`, `s16`, `u32`, `s32`, `f16`, `f32`
- ISA family:
  `VCMAX`
- semantics:
  Reduces all active source elements to one maximum value and writes that value
  to the lowest destination lane. Every remaining destination lane is zero-
  filled. Inactive lanes contribute the ISA identity value for max: the literal
  minimum integer value for integer types and `-inf` for floating-point types.

### `pto.vcmin`

- covered ISA mnemonics:
  `VCMIN`
- syntax:
  `%result = pto.vcmin %input : !pto.vreg<NxT> -> !pto.vreg<NxT>`
- operand roles:
  `%input` is the source vector and `%result` is the vector-shaped reduction
  result carrier.
- data types:
  `u16`, `s16`, `u32`, `s32`, `f16`, `f32`
- ISA family:
  `VCMIN`
- semantics:
  Reduces all active source elements to one minimum value and writes that value
  to the lowest destination lane. Every remaining destination lane is zero-
  filled. Inactive lanes contribute the ISA identity value for min: the literal
  maximum integer value for integer types and `+inf` for floating-point types.

### `pto.vreduce_group`

- covered ISA mnemonics:
  `VCGADD`, `VCGMAX`, `VCGMIN`, `VCPADD`
- syntax:
  `pto.vreduce_group %dst, %src, %pg {mode = "add32B|max32B|min32B|pair_add"}`
- operand roles:
  `%dst` is the shortened-vector reduction result, `%src` is the source vector,
  and `%pg` selects the active input lanes.
- data types:
  `add32B`, `max32B`, and `min32B` accept `u8`, `s8`, `u16`, `s16`, `u32`,
  `s32`, `f16`, or `f32`. `pair_add` accepts `f16` or `f32`.
- semantics:
  `add32B` reduces all active elements within each 32-byte block and packs the
  block results contiguously into the low lanes of `%dst`. `max32B` and `min32B`
  compute the maximum or minimum of each block with the same block-wise packing.
  `pair_add` reduces every adjacent pair of lanes and writes the pairwise sums
  into the lower half of `%dst`.
- assertions and exceptions:
  Inactive lanes contribute `0` to additive reductions, `-inf` or the literal
  minimum to max reductions, and `+inf` or the literal maximum to min
  reductions. If all lanes of an additive reduction are inactive, the reduced
  value is `0` or `+0`.

### `pto.vsqueeze`

- covered ISA mnemonics:
  `VSQZ`, `VUSQZ`
- syntax:
  `pto.vsqueeze %dst, %src?, %pg {mode = "squeeze|unsqueeze", element_type = "b8|b16|b32", store_hint = true|false, sqzn = "SQZN"?}`
- operand roles:
  `%dst` is the squeezed or unsqueezed vector result, `%src` is the input vector
  for `squeeze`, `%pg` is the governing predicate, `store_hint` carries the
  ISA's `#st` hint, and `sqzn` names the SPR receiving the surviving-byte count.
- data types:
  `squeeze` and `unsqueeze` use `b8`, `b16`, or `b32` lane interpretation.
- semantics:
  `squeeze` compacts the active `%src` elements toward the least-significant end
  of the logical result sequence, writes the packed sequence into `%dst`, and
  zero-fills the remaining lanes. It writes the number of surviving bytes into
  `SQZN`. `unsqueeze` computes the prefix count of active predicate bits, with
  `dst[0] = 0` and `dst[i] = dst[i-1] + 1` when `pg[i-1]` is true.
- assertions and exceptions:
  When `store_hint` is set, the current `SQZN` value is queued for a later
  `VSTUR`; the total number of queued hints MUST match the total number of
  consuming stateful stores, and a `VSTUR` MUST separate consecutive squeezed
  results with `store_hint = true` to avoid hardware deadlock.

### `pto.vcvt`

- covered ISA mnemonics:
  `VCVTFI`, `VCVTFF`, `VCVTFF2`, `VCVTIF`, `VCVTII`
- syntax:
  `%result = pto.vcvt %input {round_mode = "ROUND_MODE", sat = "SAT_MODE", part = "PART_MODE"} : !pto.vreg<NxT0> -> !pto.vreg<NxT1>`
- operand roles:
  `%input` is the source vector, `"ROUND_MODE"` selects the rounding behavior, `"SAT_MODE"` selects saturation or truncation behavior, `"PART_MODE"` selects the even or odd conversion part when required by the ISA form, and `%result` is the converted vector result.
- ISA family:
  `VCVTFI` / `VCVTFF` / `VCVTFF2` / `VCVTIF` / `VCVTII`
- semantics:
  Converts `%input` lane-wise according to the source type, destination type, rounding rule, saturation rule, and part-selection rule encoded by the op form. Width-changing forms consume only the selected source part or produce only the selected destination part exactly as required by the ISA conversion family.
  For conversions from wider lanes to narrower lanes, the selected destination
  part receives the converted result and the unselected part is zero-filled. For
  conversions from narrower lanes to wider lanes, only the selected input part
  is consumed. Saturating signed-to-unsigned forms preserve the ISA special
  case that negative `s16 -> u32` inputs saturate to zero.
- CCE correspondence:
  `vcvt(...)`
  builtin families:
  `__builtin_cce_vcvt*`, `__builtin_cce_vcvtfi_*`, `__builtin_cce_vcvtif_*`, `__builtin_cce_vcvtii_*`, `__builtin_cce_vcvtff_*`

### `pto.vbitsort`

- covered ISA mnemonics:
  `VBS32`
- syntax:
  `pto.vbitsort %destination, %source, %indices, %repeat_times : !llvm.ptr<AS>, !llvm.ptr<AS>, !llvm.ptr<AS>, index`
- operand roles:
  `%destination` is the Vector tile buffer output buffer, `%source` is the Vector tile buffer score buffer, `%indices` is the Vector tile buffer index buffer, and `%repeat_times` is the repeat count for consecutive sort invocations.
- ISA family:
  `VBS32`
- semantics:
  Sorts 32 proposals per iteration by score and writes the ordered proposal
  structures to `%destination`, with the highest score at the lowest address.
  `%source` supplies the score stream and `%indices` supplies the index stream;
  the ISA combines them into one 8-byte `{index, score}` structure per sorted
  proposal, with the index in the upper 4 bytes. When two scores are equal, the
  proposal with the lower original index wins. `%destination`, `%source`, and
  `%indices` MUST be 32-byte aligned. `repeat_times = 0` performs no execution.
- CCE correspondence:
  `vbitsort(...)`
  `__builtin_cce_vbitsort_*`

### `pto.vmrgsort4`

- covered ISA mnemonics:
  `VMS4v2`
- syntax:
  `pto.vmrgsort4 %destination, %source0, %source1, %source2, %source3, %count, %config : !llvm.ptr<AS>, !llvm.ptr<AS>, !llvm.ptr<AS>, !llvm.ptr<AS>, !llvm.ptr<AS>, i64, i64`
- operand roles:
  `%destination` is the Vector tile buffer output buffer, `%source0` through `%source3` are the four Vector tile buffer input list bases, `%count` is the total work or encoded list-count payload, and `%config` is the ISA merge-sort configuration word.
- ISA family:
  `VMS4v2`
- semantics:
  Merges four sorted proposal lists from the Vector tile buffer into one sorted
  output stream. The four source bases may be discrete, but each individual
  input list MUST be continuous in the Vector tile buffer. On equal scores,
  entries from the lower-numbered input list win. `%count` and `%config` carry
  the ISA list-count and repeat-mode configuration, including the repeat-mode
  restrictions that all four lists be continuous and have equal list lengths.
  Source and destination regions MUST not overlap.
- CCE correspondence:
  `vmrgsort4(...)`
  `__builtin_cce_vmrgsort4_*`

### `pto.vhistogram`

- covered ISA mnemonics:
  `DHISTv2`, `CHISTv2`
- syntax:
  `pto.vhistogram %dst, %src, %pg, #bin {mode = "distribution|cumulative"}`
- operand roles:
  `%dst` is the destination vector of `u16` bin accumulators, `%src` is the
  source vector of `u8` samples, `%pg` selects active input elements, and `#bin`
  chooses which contiguous range of the 256 total histogram bins is represented
  by the destination vector.
- data types:
  `%src` is `u8`, `%dst` is `u16`, and `#bin` is a 4-bit unsigned immediate.
- semantics:
  `distribution` accumulates the per-bin counts for the selected bin range.
  `cumulative` accumulates the cumulative histogram for the same range. `%pg`
  filters which input elements contribute, but it does not mask destination
  writes: every destination bin accumulator is updated.
- assertions and exceptions:
  `#bin` MUST be less than `256 / VL_16` for the active vector length.
