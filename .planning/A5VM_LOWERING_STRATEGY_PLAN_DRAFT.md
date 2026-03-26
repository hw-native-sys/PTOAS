# A5VM Lowering Strategy Plan Draft

## Summary

This draft keeps the original strategy intent intact and extends the formal
scope only where installed PTO behavior has been confirmed.

The plan governs `pto -> a5vm` lowering for:

- unary elementwise
- binary elementwise
- row-reduce
- rowexpand
- rowexpand-derived binary families
- scalar-op families

The only legal strategy values remain:

- `post-update`
- `no-post-update`

These are semantic implementation branches, not backend convenience knobs. The
caller passes `strategy` explicitly to `lowerXXX(...)`. `PTOToA5VMPass` is only
one default caller and provides a global default via a pass option.

This draft explicitly adds support or formal decomposition for:

- unary `2D`
- binary `2D`
- row-reduce
- rowexpand
- rowexpand-derived binary family alignment rules
- scalar-op family decomposition where installed PTO evidence is sufficient

The extension does not weaken the original constraints:

- all lowering must still align to installed PTO helpers one-to-one
- all alignment is still judged at full-helper, statement-by-statement
  granularity
- statement-by-statement alignment is a global hard requirement for every op
  family covered by this plan, not a row-reduce-only special rule
- if behavior cannot yet be aligned, the gap remains explicit and no guessed
  implementation is allowed

## Public Interfaces

- Keep `enum class A5VMLoweringStrategy { PostUpdate, NoPostUpdate };` in
  `include/PTO/Transforms/A5VMLowering.h`.
- Keep explicit `strategy` parameters on strategy-enabled `lowerXXX(...)`
  APIs.
- Strategy-enabled entrypoints currently formalized in this draft are:

  - Unary
    - `lowerTABS`
    - `lowerTEXP`
    - `lowerTLOG`
    - `lowerTSQRT`
    - `lowerTRECIP`
    - `lowerTRELU`
    - `lowerTNOT`

  - Binary
    - `lowerTADD`
    - `lowerTSUB`
    - `lowerTMUL`
    - `lowerTDIV`
    - `lowerTMAX`
    - `lowerTMIN`
    - `lowerTAND`
    - `lowerTOR`
    - `lowerTXOR`

  - Row-reduce
    - `lowerTRowSum`
    - `lowerTRowMax`
    - `lowerTRowMin`

- Shared helpers participating in this plan are:

  - `buildUnaryVecScope(..., A5VMLoweringStrategy strategy)`
  - `buildBinaryVecScope(..., A5VMLoweringStrategy strategy)`
  - `buildRowReduceVecScope(..., A5VMLoweringStrategy strategy)`

Additional families also covered by this draft, but not all as
strategy-enabled families, are:

- rowexpand
- rowexpand-derived binary families
- scalar-op families

No query-table or implicit lowering context is introduced. The plan remains
centered on explicit `lowerXXX(..., strategy)` calls.

Global alignment rule:

- once PTO helper or PTO helper branch selection is determined, the selected
  target must be implemented by full-helper, statement-by-statement alignment
- this rule applies equally to unary, binary, row-reduce, rowexpand, and any
  later-admitted scalar / rowexpand-derived family once its installed PTO
  target helper has been identified
- `1D/2D` and `post-update/no-post-update` only decide which installed PTO
  helper or helper branch is the alignment target; they do not relax the
  alignment bar
- no op family may be declared one-to-one implemented until its installed PTO
  target helper or helper branch has been closed by an evidence path
- frontend-expanded `.ccei`, emitted LLVM artifacts, and testcase build flags
  are verification evidence for the installed PTO target; they do not replace
  installed PTO as the semantic baseline

## Strategy Semantics

### Unary

Installed PTO provides:

- `TUnaryOps_1D_PostUpdate`
- `TUnaryOps_1D_NoPostUpdate`
- `TUnaryOps_2D`

So the formal rule is:

- unary `1D`
  - `post-update` must align one-to-one with `TUnaryOps_1D_PostUpdate`
  - `no-post-update` must align one-to-one with `TUnaryOps_1D_NoPostUpdate`
- unary `2D`
  - must align one-to-one with `TUnaryOps_2D`

Important clarification:
unary `2D` does not distinguish strategy in installed PTO. Therefore:

- `strategy` remains part of the public API for unary lowering as a whole
- but for unary `2D`, it does not create a new semantic split
- when PTO selects `TUnaryOps_2D`, lowering must directly align to that helper
- when PTO selects unary `2D`, lowering must use that same unified `2D`
  implementation regardless of whether the caller passed `post-update` or
  `no-post-update`
- it is forbidden to derive two unary `2D` lowering branches from `strategy`
- if the caller-selected strategy cannot be proven equivalent to that single
  PTO baseline, the issue must remain explicit instead of being guessed away

### Binary

Installed PTO provides:

- `TBinOps_1D_PostUpdate`
- `TBinOps_1D_NoPostUpdate`
- `TBinOps_2D_PostUpdate`
- `TBinOps_2D_NoPostUpdate`

So the formal rule is:

- binary `1D`
  - `post-update` must align one-to-one with `TBinOps_1D_PostUpdate`
  - `no-post-update` must align one-to-one with `TBinOps_1D_NoPostUpdate`
- binary `2D`
  - `post-update` must align one-to-one with `TBinOps_2D_PostUpdate`
  - `no-post-update` must align one-to-one with `TBinOps_2D_NoPostUpdate`

### Row-Reduce

Installed PTO row-reduce is defined in `TRowReduce.hpp` and centered on:

- `TRowReduceImpl<ReduceOp, TileDataOut, TileDataIn, elementsPerRepeat>(..., version)`

with two concrete implementation branches inside the helper body:

- `version == VFIMPL_2D_NO_POST_UPDATE`
  - explicit indexed `vlds`
  - explicit indexed `vsts`
- `else`
  - row base computed as `row_ptr = srcPtr + i * TileDataIn::RowStride`
  - `vlds(..., POST_UPDATE)`
  - `vsts(..., POST_UPDATE)`

The row-reduce op frontends are:

- `TRowSum`
- `TRowMax`
- `TRowMin`

and the reduction kernels are defined by:

- `ROWSUM`
- `ROWMAX`
- `ROWMIN`

Therefore the formal row-reduce rule is:

- `TRowSum`
  - implementation must align one-to-one with PTO
    `TRowReduceImpl<ROWSUM,...>`
- `TRowMax`
  - implementation must align one-to-one with PTO
    `TRowReduceImpl<ROWMAX,...>`
- `TRowMin`
  - implementation must align one-to-one with PTO
    `TRowReduceImpl<ROWMIN,...>`

Important clarification:
for row-reduce, the one-to-one correspondence target is the full concrete
statement sequence inside `TRowReduceImpl`, not a guessed pair of independently
named helpers and not merely a matched high-level skeleton.

For row-reduce strategy naming:

- the `no-post-update` branch is currently justified by the explicit
  `VFIMPL_2D_NO_POST_UPDATE` branch in installed PTO
- the `post-update` branch is the `else` branch of the same installed
  `TRowReduceImpl` helper, and must align one-to-one with that exact branch

The row-reduce strategy-to-branch mapping is fixed:

- `A5VMLoweringStrategy::NoPostUpdate`
  - maps to the `version == VFIMPL_2D_NO_POST_UPDATE` branch inside
    `TRowReduceImpl`
- `A5VMLoweringStrategy::PostUpdate`
  - maps to the `else` branch inside `TRowReduceImpl`

The evidence path for treating the row-reduce `else` branch as the
`post-update` strategy target is:

- installed PTO `TRowReduce.hpp`
  - the `else` branch performs `vlds(..., POST_UPDATE)` and
    `vsts(..., POST_UPDATE)`
- frontend-expanded `.ccei`
  - the same branch remains `vlds(..., POST_UPDATE)` and
    `vsts(..., POST_UPDATE)` after frontend expansion
- emitted LLVM artifacts
  - the same branch lowers to `llvm.hivm.vldsx1.post.*` and
    `llvm.hivm.vstsx1.post.*`
  - the returned pointer values are loop-carried, proving that state
    advancement is performed by post-update intrinsic results rather than by
    guessed explicit offset arithmetic

So row-reduce strategy alignment must be expressed as:

- `no-post-update`
  - align to the `version == VFIMPL_2D_NO_POST_UPDATE` branch inside
    `TRowReduceImpl`
- `post-update`
  - align to the `else` branch inside `TRowReduceImpl`

It is forbidden to rewrite this as two guessed repo-local helper families or as
an approximate high-level simulation of row-wise reduction behavior.

### Rowexpand

Installed PTO `TRowExpand.hpp` provides one frontend op with two concrete
helper targets:

- `TRowExpandInstr_NoPostUpdate`
- `TRowExpandInstr_PostUpdate`

The version-to-helper mapping is fixed:

- `VFIMPL_1D_NO_POST_UPDATE` and `VFIMPL_2D_NO_POST_UPDATE`
  - map to `TRowExpandInstr_NoPostUpdate`
- `VFIMPL_1D_POST_UPDATE`, `VFIMPL_2D_POST_UPDATE`, and `VFIMPL_DEFAULT`
  - map to `TRowExpandInstr_PostUpdate`

So the rowexpand strategy mapping is:

- `A5VMLoweringStrategy::NoPostUpdate`
  - align to `TRowExpandInstr_NoPostUpdate`
- `A5VMLoweringStrategy::PostUpdate`
  - align to `TRowExpandInstr_PostUpdate`

Important clarification:
for `TRowExpand`, the installed PTO version names still mention `1D/2D`, but
the concrete implementation target is only this helper pair. Therefore `1D/2D`
does not introduce additional rowexpand helper families beyond the two
NoPostUpdate / PostUpdate implementations above.

### Rowexpand-Derived Binary Families

Installed PTO rowexpand-derived binary ops are built on `TRowExpandBinOp.hpp`
and currently expose these concrete helper bodies:

- `TRowExpandBinOps_2D_NoPostUpdate`
- `TRowExpandBinOps_2D_NoPostUpdate2`
- `TRowExpandBinOps_2D_PostUpdate`

But the installed selector `RowExpandBinaryInstr(...)` currently dispatches only
to the two NoPostUpdate helpers:

- when the broadcast tile type is row-major
  - align to `TRowExpandBinOps_2D_NoPostUpdate2`
- otherwise
  - align to `TRowExpandBinOps_2D_NoPostUpdate`

The installed selector does not currently dispatch to
`TRowExpandBinOps_2D_PostUpdate`.

So the rule for this family is:

- `TRowExpandAdd`
- `TRowExpandSub`
- `TRowExpandMul`
- `TRowExpandDiv`
- `TRowExpandMax`
- `TRowExpandMin`

must currently align to the installed selector-reachable helper body for the
matching operand-layout case.

Evidence closure:

- frontend-expanded `.ccei` must be checked against the same selector body
- if testcase build flags or frontend artifacts expand a different helper body,
  that is a divergence to be explained or fixed; it does not retarget lowering
  away from the installed PTO selector-reachable helper body

Important clarification:

- this family is now semantically closed as a selector-reachable helper family
- it is not a strategy-governed dual-branch family
- the mere presence of `TRowExpandBinOps_2D_PostUpdate` in the header is not
  enough to claim a real strategy split
- no repo-local PostUpdate branch may be invented here unless installed PTO
  selector logic changes or installed frontend/compiler evidence proves that
  branch is part of the installed contract

### Scalar-Op Families

The scalar-op area is now split by installed PTO evidence into:

- one installed-A5 helper family that is closed
- one installed-A5 special case (`TEXPANDS`) that is closed
- two groups that are not yet closed as installed-A5 helper families

#### Scalar Family A: `TBinSOp`-Backed Scalar Families

Installed PTO `TBinSOp.hpp` defines the common helper family:

- `TBinSOps_1D_NoPostUpdate`
- `TBinSOps_1D_PostUpdate`
- `TBinSOps_2D_NoPostUpdate`
- `TBinSOps_2D_PostUpdate`

and the dispatch enum:

- `BinSOpsIMPL_1D_NO_POST_UPDATE`
- `BinSOpsIMPL_2D_NO_POST_UPDATE`
- `BinSOpsIMPL_1D_POST_UPDATE`
- `BinSOpsIMPL_2D_POST_UPDATE`

The following ops are governed by this helper family in installed PTO:

- `TADDS`
- `TSUBS`
- `TMULS`
- `TMAXS`
- `TMINS`
- `TLRELU`

and `TDIVS` is mostly governed by the same helper family, with an important
installed-PTO exception:

- for `int16_t`, installed `TDIVS` takes the explicit scalar-loop fallback
  `TDivs_naive` / `TSDiv_naive`
- for the non-fallback types, it dispatches through `BinaryInstr<...>` and the
  `TBinSOp.hpp` helper family

Frontend-expanded `.ccei` should be checked against the same `TBinSOp.hpp`
dispatch and helper bodies. A mismatch must remain explicit rather than
changing the installed PTO alignment target.

Important clarification:

- for scalar ops, helper names containing `PostUpdate` do not automatically
  prove a `.post` LLVM/HIVM contract
- the alignment target is the installed PTO helper body first
- any claim that a scalar helper is truly PostUpdate-aligned at the LLVM/HIVM
  boundary still requires explicit frontend/compiler evidence of returned
  advancement state

#### Scalar Family B: `TEXPANDS`

`TEXPANDS` is closed by the installed PTO header and then checked against
frontend/compiler artifacts.

Evidence closure:

- installed PTO `TExpandS.hpp` defines the alignment target
- frontend-expanded `.ccei` and emitted LLVM must be checked against that same
  installed helper body
- in installed PTO, vector `TEXPANDS` routes through
  `BinaryInstr<ExpandSOp<...>>(dstPtr, dstPtr, scalar, ...)`
- mat `TEXPANDS` remains a separate explicit path in the same installed header

So the current rule is:

- `TEXPANDS` must align to the installed PTO helper body
- if testcase build flags or `.ccei` show expansion through a conflicting
  non-installed header body, that is a pipeline divergence to be fixed or
  tracked explicitly, not a reason to retarget lowering

#### Scalar Family C: Bitwise Scalar Families

The following bitwise scalar front-door ops exist in `pto/common/pto_instr.hpp`:

- `TANDS`
- `TORS`
- `TXORS`

But installed A5 does not currently expose dedicated `a5/TAndS.hpp`,
`a5/TOrS.hpp`, or `a5/TXorS.hpp` helper headers, and `pto_instr_impl.hpp`
under `REGISTER_BASE` does not include such A5 helper files.

So the current rule is:

- these ops are not yet closed as installed-A5 helper families
- no repo-local lowering may claim one-to-one installed-A5 alignment for them
  until an installed A5 helper baseline is located and traced
- CPU implementations or non-installed traces may be useful clues, but they do
  not close the installed-A5 contract

#### Scalar Family D: No Closed A5 NPU Baseline Yet

The following scalar families do not currently have a closed A5 NPU baseline:

- `TADDSC`
- `TSUBSC`

Evidence closure here is negative rather than missing:

- installed Ascend headers expose the common front-door `MAP_INSTR_IMPL`
  declarations
- only CPU implementations were located
- no A5-specific helper implementation was located in installed PTO
- the frontend trace shows calls to `TADDSC_IMPL` / `TSUBSC_IMPL`, but no
  closed A5 helper definition was located from the current evidence chain

So the current rule is:

- these two ops are not allowed into formal A5 one-to-one scope yet
- no repo-local A5 lowering may claim one-to-one alignment for them until an
  actual A5 helper baseline is located and traced

## 1D vs 2D Selection Rule

`1D/2D` is not a third strategy. It is a PTO-driven helper selection rule.

### Unary 1D vs 2D

Installed PTO `TUnaryOp.hpp` selects as follows:

- use `1D` when valid columns are full-width for the participating tiles
- use `2D` when valid columns are not full-width

More precisely:

- if static tile metadata proves `ValidCol == Cols`, use `1D`
- otherwise, if runtime `validCol == DstTile::Cols == SrcTile::Cols`, use `1D`
- otherwise, use `2D`

### Binary 1D vs 2D

Installed PTO `TBinOp.hpp` selects based on both:

- whether valid columns are full-width
- whether src0/src1/dst tile shape / row-stride relationship matches the PTO
  same-shape linear path

So binary must use `2D` whenever either is true:

- `validCols` is not full-width
- src0/src1/dst row stride or tile shape forces PTO into the `2D` helper family

### Row-Reduce

Row-reduce is not treated as `1D` linear elementwise lowering. Its semantic
baseline is the row-structured PTO helper `TRowReduceImpl`, with:

- outer row loop
- inner repeat loop over columns
- accumulator initialization
- reduction / accumulation
- final row store

## What “Strictly Align” Means

Strict alignment still means full-helper alignment, not just result equality
and not just matching a few core ops.

Every strategy-enabled helper must be checked across these four layers:

- Initialization state
  - `repeatTimes`
  - `sReg` / `count`
  - initial offset / ptr / scalar / accumulator
- Loop skeleton
  - row loop / chunk loop / repeat loop
  - induction variable progression
  - loop-carried state organization
- Per-iteration body
  - `CreatePredicate<T>(...)`
  - `vlds`
  - compute / reduce op
  - `vsts` or final row result materialization
- State advancement
  - `sReg` advancement
  - `src/dst ptr` advancement
  - `offset` advancement
  - accumulator / destination carry state
  - which state is advanced by intrinsic return versus explicit arithmetic

Additional mandatory checks:

- Unary `2D`
  - row traversal order
  - row-base computation
  - per-row chunk restart
  - tail predicate at end-of-row
- Binary `2D`
  - row traversal order
  - row-base computation for each operand
  - row-stride-specific offset progression
  - tail predicate at end-of-row
- Row-reduce
  - `ReduceOp::InitVal`
  - `ReduceOp::Reduce`
  - `ReduceOp::Accumulate`
  - `destItems` / destination predicate setup
  - final row-store progression

These four layers are only an organization for the comparison. They do not
reduce the required alignment granularity.

The real requirement is:

- every strategy-enabled helper must be aligned statement-by-statement against
  the installed PTO helper it claims to implement
- each statement in PTO must have a corresponding lowering statement or
  statement group with the same role and ordering constraints
- if only the high-level loop shape matches but concrete statements do not,
  the implementation is not considered aligned

No part of row-reduce is exempt from the original strong constraint. It must
also be implemented one-to-one at full-helper, statement-by-statement level.

## Evidence Rule For PostUpdate State Advancement

`post-update` advancement cannot be justified by “implicit semantics”.
Implementation and validation must provide an explicit evidence path.

- For predicate-driving scalar advancement:
  - trace from installed PTO `CreatePredicate<T>(...)` to the builtin wrapper
  - provide compiler-facing evidence such as `{mask, scalar_out}` or visible
    `extractvalue ..., 1`
- For source/destination pointer advancement:
  - do not claim pointer advancement without evidence
  - acceptable evidence is one of:
    - installed wrapper / builtin declaration returns updated pointer/base
    - frontend artifact contains a `.post` intrinsic returning `{vec, ptr}` or
      `ptr`
- For row-reduce destination progression:
  - show whether final-store post-update returns next destination pointer or PTO
    advances it explicitly outside the intrinsic
- For rowexpand:
  - show whether source and destination advancement is performed by explicit
    pointer arithmetic or by post-update intrinsic return values
- For `TBinSOp`-backed scalar families:
  - do not infer LLVM/HIVM `.post` semantics from the installed helper name
    alone; close that claim only with frontend/compiler artifact evidence
- For unary `2D`
  - because installed PTO only exposes `TUnaryOps_2D`, do not infer a hidden
    strategy split unless there is direct evidence
- Without evidence, no A5VM or LLVM shape may be labeled PostUpdate-aligned

This rule applies equally to unary, binary, `2D`, row-reduce, rowexpand, and
any scalar family whose helper-level baseline has already been closed.

## Implementation Changes

- Keep the `pto-to-a5vm` pass option:
  - `--a5vm-lowering-strategy=post-update|no-post-update`
  - default: `post-update`
- In `PTOToA5VM.cpp`
  - continue parsing the option once
  - continue passing `strategy` explicitly to all strategy-enabled `lowerXXX`
- In `PTOToA5VMLowering.cpp`

  - `buildUnaryVecScope`
    - must implement PTO-aligned `1D/2D` selection
    - `1D` must split into PostUpdate / NoPostUpdate
    - `2D` must align to installed `TUnaryOps_2D`
    - once PTO selects unary `2D`, lowering must use that same unified `2D`
      implementation regardless of `strategy`
    - if strategy materially conflicts with that single PTO baseline, keep the
      issue explicit instead of guessing a branch

  - `buildBinaryVecScope`
    - must implement PTO-aligned `1D/2D` selection
    - `1D` must split into PostUpdate / NoPostUpdate
    - `2D` must split into `TBinOps_2D_PostUpdate` /
      `TBinOps_2D_NoPostUpdate`

  - `buildRowReduceVecScope`
    - becomes a first-class draft item
    - must be implemented by explicit full-helper, statement-by-statement
      alignment to installed PTO `TRowReduceImpl`
    - `TRowSum`, `TRowMax`, `TRowMin` must each map one-to-one through the PTO
      helper statement sequence
    - the row-reduce strategy targets are already fixed by installed PTO:
      `version == VFIMPL_2D_NO_POST_UPDATE` and the `else` branch
    - if repo lowering does not align to those targets, preserve that
      implementation mismatch explicitly instead of weakening the target or
      guessing a replacement

  - `lowerTRowExpand`
    - becomes a first-class formalized strategy item
    - must align one-to-one with installed `TRowExpandInstr_NoPostUpdate` or
      `TRowExpandInstr_PostUpdate`
    - must treat installed `VFIMPL_1D_*` and `VFIMPL_2D_*` rowexpand versions
      only as aliases for that helper pair, not as extra helper families

  - rowexpand-derived binary lowering entrypoints such as
    `lowerTRowExpandAdd` / `lowerTRowExpandSub` / `lowerTRowExpandMul` /
    `lowerTRowExpandDiv` / `lowerTRowExpandMax` / `lowerTRowExpandMin`
    - remain tracked against the installed selector-reachable helper body
    - must not invent a repo-local PostUpdate branch until the installed PTO
      contract is proven beyond the currently reachable NoPostUpdate helpers

  - closed scalar lowering entrypoints such as
    `lowerTAddS` / `lowerTSubS` / `lowerTMulS` / `lowerTDivS` / `lowerTMaxS` /
    `lowerTMinS` / `lowerTExpandS`
    - must align first to the installed helper body that can actually be
      evidenced for each closed scalar family
    - must preserve any unresolved mismatch explicitly when installed-header and
      frontend/compiler evidence diverge

  - unclosed scalar families such as `TANDS` / `TORS` / `TXORS` / `TADDSC` /
    `TSUBSC`
    - are not implementation targets for one-to-one A5 lowering in this draft
    - must remain explicitly unresolved until an installed A5 helper baseline
      is located and traced

- Remove any mixed-shape implementation:
  - PostUpdate predicate/scalar with NoPostUpdate addressing is forbidden
  - PostUpdate row-store behavior with NoPostUpdate source traversal is
    forbidden
  - if behavior does not align cleanly, do not guess

## A5VM / LLVM Boundary

The original boundary rule remains fully intact.

- strategy is chosen only in `pto -> a5vm`
- `a5vm -> llvm` must faithfully export A5VM `mode` and state-advancement
  contract
- `VldsOp(mode=POST_UPDATE)` must export to the matching `.post` intrinsic and
  preserve returned advancement state
- `VstsOp(mode=POST_UPDATE)` must export to the matching `.post` intrinsic and
  preserve returned advancement state
- `NO_POST_UPDATE` must export to normal `vldsx1/vstsx1`
- if A5VM is already `POST_UPDATE` but LLVM export still drops advancement
  state, it is not strictly aligned

No temporary guessed contract is allowed here. If the current A5VM op model
cannot represent the required returned state, that mismatch remains an explicit
open issue until the contract is made real.

## Scope

Formal strategy-governed scope:

- Unary elementwise
  - `TABS`
  - `TEXP`
  - `TLOG`
  - `TSQRT`
  - `TRECIP`
  - `TRELU`
  - `TNOT`
  - with explicit support for both `1D` and `2D`
- Binary elementwise
  - `TADD`
  - `TSUB`
  - `TMUL`
  - `TDIV`
  - `TMAX`
  - `TMIN`
  - `TAND`
  - `TOR`
  - `TXOR`
  - with explicit support for both `1D` and `2D`
- Row-reduce
  - `TRowSum`
  - `TRowMax`
  - `TRowMin`
- Rowexpand
  - `TRowExpand`

Additional formally closed helper-alignment scope:

- Rowexpand-derived binary families
  - `TRowExpandAdd`
  - `TRowExpandSub`
  - `TRowExpandMul`
  - `TRowExpandDiv`
  - `TRowExpandMax`
  - `TRowExpandMin`
  - note: closed as selector-reachable helper alignment, not as a dual-strategy
    family
- Scalar Family A: `TBinSOp`-backed
  - `TADDS`
  - `TSUBS`
  - `TMULS`
  - `TDIVS`
  - `TMAXS`
  - `TMINS`
  - `TLRELU`
- Scalar Family B: `TEXPANDS`

Tracked scope still pending baseline closure:

- Scalar Family C
  - `TANDS`
  - `TORS`
  - `TXORS`
- Scalar Family D
  - `TADDSC`
  - `TSUBSC`

Still out of scope:

- any op family whose installed PTO dual-branch behavior has not been confirmed

## Validation Rules

Every formalized family in scope must have validation rules tied to installed
PTO and real frontend/compiler artifacts.

Required comparison coverage:

- Unary
  - one `1D` case aligned to `TUnaryOps_1D_PostUpdate`
  - one `1D` case aligned to `TUnaryOps_1D_NoPostUpdate`
  - one `2D` case aligned to `TUnaryOps_2D`
- Binary
  - one `1D` PostUpdate case aligned to `TBinOps_1D_PostUpdate`
  - one `1D` NoPostUpdate case aligned to `TBinOps_1D_NoPostUpdate`
  - one `2D` PostUpdate case aligned to `TBinOps_2D_PostUpdate`
  - one `2D` NoPostUpdate case aligned to `TBinOps_2D_NoPostUpdate`
- Row-reduce
  - full coverage is required; row-reduce validation is not allowed to rely on
    symmetry assumptions between reduce ops
  - `rowsum`
    - one case aligned to the
      `version == VFIMPL_2D_NO_POST_UPDATE` branch of
      `TRowReduceImpl<ROWSUM,...>`
    - one case aligned to the `else` branch of
      `TRowReduceImpl<ROWSUM,...>`
  - `rowmax`
    - one case aligned to the
      `version == VFIMPL_2D_NO_POST_UPDATE` branch of
      `TRowReduceImpl<ROWMAX,...>`
    - one case aligned to the `else` branch of
      `TRowReduceImpl<ROWMAX,...>`
  - `rowmin`
    - one case aligned to the
      `version == VFIMPL_2D_NO_POST_UPDATE` branch of
      `TRowReduceImpl<ROWMIN,...>`
    - one case aligned to the `else` branch of
      `TRowReduceImpl<ROWMIN,...>`
- Rowexpand
  - one case aligned to `TRowExpandInstr_NoPostUpdate`
  - one case aligned to `TRowExpandInstr_PostUpdate`
- Rowexpand-derived binary families
  - validation is defined against the currently selector-reachable helper body
    for the exercised layout case
  - one row-major broadcast case aligned to
    `TRowExpandBinOps_2D_NoPostUpdate2`
  - one non-row-major broadcast case aligned to
    `TRowExpandBinOps_2D_NoPostUpdate`
  - no case may be labeled PostUpdate-aligned unless a selector or
    installed frontend/compiler evidence path proves that
    `TRowExpandBinOps_2D_PostUpdate` is part of the installed contract
- Scalar-op families
  - Scalar Family A: `TBinSOp`-backed ops
    - validation is defined against the installed PTO helper body selected by
      `VFImplKind`
    - helper-name `PostUpdate` is not by itself enough to claim `.post`
      LLVM/HIVM alignment
    - `TDIVS` validation must additionally state whether the exercised case hit
      the `TBinSOp` path or the scalar-loop fallback
  - Scalar Family B: `TEXPANDS`
    - validation is defined against the installed PTO helper body
    - build flags and `.ccei` are used only to verify that the compiled path
      does not diverge from that installed target
  - Scalar Family C: bitwise scalar composition
    - remain tracked but unvalidated because no closed installed-A5 helper
      baseline is currently available
  - Scalar Family D
    - remain tracked but unvalidated because no closed A5 helper baseline is
      currently available

Each comparison report must explicitly state:

- whether initialization state aligns
- whether loop skeleton aligns
- whether per-iteration body aligns
- whether state advancement has an evidence path
- whether any mixed-shape behavior remains
- whether any unresolved semantic gap remains

If a gap remains, the implementation must keep that gap explicit instead of
papering it over. A statement-by-statement mismatch is enough to keep the
status unresolved.

Case availability rule:

- validation must first check whether the repo already contains a stable case
  that can be shown to hit the target PTO helper or helper branch
- if such a case exists, that validation is required and cannot be skipped
- if such a case does not currently exist, this draft does not require adding a
  new case in the same step
- missing case availability does not reduce the coverage target and does not
  relax the statement-by-statement alignment requirement
- when a target cannot be validated only because no suitable existing case is
  available, its status must remain explicitly marked as unvalidated due to
  missing case
- a missing case may not be substituted by a different op, a different
  strategy branch, or a merely result-equal case that does not prove the target
  helper / branch was exercised

## Test Plan

- A5VM IR inspection
  - `--a5vm-print-ir`
  - verify `POST_UPDATE` vs `NO_POST_UPDATE` on strategy-enabled `vlds/vsts`
  - verify unary `1D` cases lower as PTO linear shape
  - verify unary `2D` cases lower as PTO `TUnaryOps_2D` shape
  - verify binary `2D` cases lower as PTO `TBinOps_2D_*` shape
  - verify row-reduce uses PTO-aligned row loop + repeat loop structure
- LLVM export inspection
  - `--a5vm-emit-hivm-llvm`
  - under `post-update`, `.post` intrinsics appear
  - under `no-post-update`, normal `vldsx1/vstsx1` remain
  - verify advanced pointer contract is preserved or remains explicit as an
    open issue
- Compile validation
  - `bisheng -c -x ir` for representative strategy-enabled kernels, including
    `PyPTOIRParser/paged_attention_example_kernel_online_update`
- Runtime validation
  - host NPU validation for representative cases
- Regression set
  - unary
    - `Abs`
    - `Exp`
    - one explicit unary `2D` case
  - binary
    - `Add`
    - `Sub`
    - `Mul`
    - `Max`
    - one explicit binary `2D` case
  - row-reduce
    - `Rowsum`
    - `Rowmax`
    - `Rowmin`
  - one chained case involving reduce input or reduce output

## Assumptions

- Installed PTO A5 headers remain the semantic baseline.
- Real frontend/compiler artifacts are verification evidence for that baseline,
  not a replacement baseline.
- The implementation model remains `lowerXXX(..., strategy)`, not query-table
  dispatch.
- Unary `2D` currently has a single installed PTO helper baseline:
  `TUnaryOps_2D`.
- For unary `2D`, the same installed PTO helper must be used regardless of the
  caller-selected strategy, unless later evidence proves an actual PTO-level
  semantic split.
- Binary `2D` currently has explicit installed PTO PostUpdate / NoPostUpdate
  helper branches.
- Row-reduce is now formally in scope and must satisfy the same strong
  helper-level alignment rule as unary and binary.
- Where installed PTO evidence is incomplete, the draft requires preserving the
  mismatch as an explicit open issue rather than guessing an implementation.
