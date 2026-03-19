# A5VM Sample Expansion Plan

## Goal

Expand the `a5vm` backend from the current `Abs`-only state to cover a broad
set of `test/samples` cases by using the real backend seam IR as the source of
truth.

This plan explicitly does **not** use the highest-level PTO ops found in the
Python samples as the implementation list. The implementation list must come
from the shared IR right before backend branching, i.e. the effective input to
the `emitc` backend and the `a5vm` backend.

## Scope Rules

1. The support matrix is built from the shared pre-backend IR, not from sample
   source code.
2. For each PTO op family to support:
   - first inspect how `PTOToEmitC.cpp` lowers it;
   - then identify the corresponding PTO library API/helper;
   - then inspect the A5 PTO library implementation;
   - then choose the backend representation:
     - use `a5vm` only for semantics that correspond to CCE builtins or
       hardware-facing operations;
     - use `llvm` dialect or other LLVM-lowerable dialects for ordinary
       C/C++-style semantics.
3. `__VEC_SCOPE__` must be preserved as a loop structure, not as a free-form
   marker attribute on arbitrary ops.
   Required shape:
   `for (int dummay = 0; dummay <= 0; dummay += 1)`.
   In IR terms, generate a carrier loop equivalent to a single-iteration loop
   and attach `llvm.loop.aivector_scope` to that loop.
3. If an existing `a5vm` lowering conflicts with the PTO library behavior, fix
   the existing lowering instead of layering more special cases on top.

## Temporary Exclusions

The first expansion wave excludes matrix-family samples and related kernels.
These are deferred until the vector/data-movement path is stable.

Excluded PTO op families:

- `TMatmulOp`
- `TMatmulAccOp`
- `TMatmulBiasOp`
- `TMatmulMxOp`
- `TMatmulMxAccOp`
- `TMatmulMxBiasOp`
- `TGemvOp`
- `TGemvAccOp`
- `TGemvBiasOp`

Excluded sample areas include, at minimum:

- `test/samples/MatMul`
- `test/samples/Matmul_transpose`
- `test/samples/Gemv`
- any `Sync` or `PyPTOIRParser` sample whose seam IR contains the excluded op
  families above

## Execution Steps

### Step 1: Add Seam IR Capture

Add a `ptoas` debug option that prints or writes the shared IR after
`addSharedPreBackendPasses(...)` and before backend-specific lowering begins.

Requirements:

- the seam IR must be obtainable for both `--pto-backend=emitc` and
  `--pto-backend=a5vm`
- the dump must reflect the true backend input layer
- the dump format must be stable enough to script against

### Step 2: Build the Real Implementation List

Run `test/samples` in batches, skipping excluded matrix-family cases, and
collect seam IR.

For each sample:

- record whether it reaches the seam successfully
- record the PTO ops that survive to the seam
- group samples by shared PTO op families

Deliverable:

- a seam-IR-derived support matrix, not a source-derived guess list

### Step 3: Create an Op Family Matrix

For each seam PTO op family:

- locate the corresponding `emitc` lowering pattern
- identify the PTO helper/API/function it emits
- locate the A5 PTO library implementation
- classify the lowering target as:
  - `a5vm`
  - `llvm`
  - shared `arith/scf/memref` scaffolding

Suggested columns:

- seam PTO op
- sample coverage
- `emitc` lowering entry
- PTO helper/API
- A5 implementation file
- backend representation choice
- status
- notes

### Step 4: Implement High-Frequency Non-Matrix Families First

Prioritize families that unlock the largest number of non-matrix samples.
Expected early buckets:

- vector unary/binary/scalar families
- move/cast/extract/value ops
- addptr/bitcast/reshape/subset-related address/view semantics
- synchronization families already visible at seam
- data-movement extensions adjacent to `TLoad`/`TStore`

### Step 5: Batch Regression

After each family lands:

- rebuild `ptoas`
- rerun the relevant sample subset
- inspect emitted A5VM text
- verify newly introduced operations follow the PTO library behavior and the
  backend representation rule above

### Step 6: Revisit Deferred Families

Once the vector/data-movement path is stable, re-enable matrix-family samples
and repeat the same seam-first process.

## Current Baseline

Current known-good baseline:

- `Abs` lowers through:
  - LLVM pointer materialization for non-builtin tile/address semantics
  - `a5vm.copy_gm_to_ubuf`
  - `a5vm.vlds`
  - `a5vm.vabs`
  - `a5vm.vsts`
  - `a5vm.copy_ubuf_to_gm`
  - `a5vm.set_flag`
  - `a5vm.wait_flag`
  - `a5vm.pipe_barrier`
- vector scope is represented by a dedicated single-iteration carrier loop
  equivalent to:
  `for (int dummay = 0; dummay <= 0; dummay += 1)`
  with `llvm.loop.aivector_scope` attached to that loop

## Current Execution Status

Status as of 2026-03-19:

- Step 1 is complete.
- `ptoas` now exposes shared seam capture through:
  - `--pto-print-seam-ir`
  - `--pto-seam-ir-file=<path>`
- The seam is emitted after the shared pre-backend pipeline and before backend
  lowering.
- The pass pipeline was split so the seam seen by `--pto-backend=emitc` and
  `--pto-backend=a5vm` is identical.
- Direct A5VM input intentionally rejects seam capture because there is no
  shared PTO seam in that mode.

Wave 1 seam inventory is recorded under:

- `build/seam-wave1/summary.tsv`

Wave 1 coverage focused on high-frequency non-matrix families:

- unary/binary/scalar vector samples
- address/view-adjacent samples: `AddPtr`, `Reshape`, `Subset`
- extraction / scalar / layout / movement samples
- vector addition variants

Current wave-1 seam op families observed:

- `pto.tload`
- `pto.tstore`
- `pto.tabs`
- `pto.tadd`
- `pto.taddc`
- `pto.tadds`
- `pto.taddsc`
- `pto.tand`
- `pto.tands`
- `pto.tcmp`
- `pto.tcmps`
- `pto.tcvt`
- `pto.tdiv`
- `pto.tdivs`
- `pto.texp`
- `pto.textract`
- `pto.tgetval`
- `pto.tlog`
- `pto.tlrelu`
- `pto.tmax`
- `pto.tmaxs`
- `pto.tmin`
- `pto.tmins`
- `pto.tmov`
- `pto.tmul`
- `pto.tmuls`
- `pto.tneg`
- `pto.tnot`
- `pto.tor`
- `pto.tors`
- `pto.trecip`
- `pto.trelu`
- `pto.trem`
- `pto.trems`
- `pto.trsqrt`
- `pto.tsel`
- `pto.tsels`
- `pto.tsetval`
- `pto.tshl`
- `pto.tshls`
- `pto.tshr`
- `pto.tshrs`
- `pto.tsqrt`
- `pto.tsub`
- `pto.tsubc`
- `pto.tsubs`
- `pto.tsubsc`
- `pto.txor`
- `pto.txors`
- `pto.view_semantics`
- `pto.load_scalar`
- `pto.store_scalar`
- `pto.addptr_trace`

Wave-1 exceptions requiring care:

- `Extract/extract.py` seam also contains `pto.tgemv`, so it belongs to the
  deferred matrix-family bucket for actual lowering work.
- Several invalid/dynamic-negative samples collapse to minimal seam output and
  should not be treated as feature coverage.

## Update Rule

This file should be updated whenever one of these changes:

- the seam capture mechanism
- the temporary exclusion list
- the op family prioritization
- the backend representation policy
- the support matrix status
