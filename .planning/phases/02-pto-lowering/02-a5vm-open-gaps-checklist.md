# A5VM Open Gaps Checklist

Updated: 2026-03-20

Purpose:

- Record the currently confirmed failure points for the `a5vm` backend.
- Keep dynamic-shape gaps, missing lowerings, and deferred-domain work separate.
- Provide a stable checklist so implementation can proceed item by item without losing scope.

## Priority Order

- [x] P0. Broaden `TLOAD/TSTORE` copy-family lowering so dynamic and tail copies follow PTO A5 behavior instead of only the current narrow vec ND2ND happy path.
- [x] P0. Relax and correct unary/binary vec valid-shape contract checks so they match PTO A5 helper behavior instead of requiring today’s over-strict source/destination equality.
- [x] P1. Add valid-region fallback derivation when seam IR no longer carries explicit `bind_tile` valid dims but the structured tile-driven seam still determines them.
- [x] P1. Lower synchronization-family ops that are still left as PTO ops at the backend seam.
- [ ] P2. Revisit deferred `ACC`/`MAT` domains and matrix-family samples after vector/data-movement paths are stable.

## Confirmed Active Gaps

### 1. Dynamic / Tail Copy-Family Gap

- [x] `lowerTLOAD` must support dynamic or tail copy scenarios that currently fail with:
  `requires PTO-compatible vec ND2ND copy_gm_to_ubuf arguments`
- [x] `lowerTSTORE` must support dynamic or tail copy scenarios that currently fail with:
  `requires PTO-compatible vec ND2ND copy_ubuf_to_gm arguments`
- [x] Copy-family transfer derivation must recover PTO-relevant shape/stride decisions from the seam value graph, not only from the current static full-tile case.
- [x] Dynamic transfer operands must stay on the existing unified `a5vm.copy_gm_to_ubuf` / `a5vm.copy_ubuf_to_gm` interface. No static-vs-dynamic duplicate ops.
- [x] `len_burst`, loop programming, and stride decisions must accept runtime valid-shape participation where PTO A5 behavior requires it.

Status note:

- Closed in current branch by refactoring `lowerTLOAD` / `lowerTSTORE` around recovered tensor-view shape/stride/offset state plus explicit `a5vm.set_loop*` programming ops.
- Verified by:
  - `test/samples/Abs/abs.py` via `./test/samples/runop.sh -t Abs`
  - `build/a5vm-failure-scan/out-sync/Sync/add_double_dynamic-pto-ir.pto`
- End-to-end acceptance coverage:
  - `test/samples/run_a5vm_acceptance_checks.sh`
  - script assertions:
    - `./test/samples/runop.sh -t Abs` must compile `Abs` and emit `a5vm.copy_gm_to_ubuf`, `a5vm.vabs`, `llvm.loop.aivector_scope`, `a5vm.copy_ubuf_to_gm`
    - `./test/samples/runop.sh -t Sync` must compile `Sync/add_double_dynamic.py` and emit `a5vm.set_loop2_stride_outtoub`, `a5vm.copy_gm_to_ubuf`, `a5vm.vadd`, `a5vm.copy_ubuf_to_gm`
- `test/samples/Sync/test_dynamic_valid_shape.pto` now passes `TLOAD/TSTORE`; the remaining failure is the separate unary valid-shape contract on `pto.trelu`.
- Static phase-2 fixtures were updated to the new operand-based copy op form, but full `FileCheck` execution is currently blocked on this machine because `FileCheck` is not installed.

Representative failing samples:

- `test/samples/Sync/add_double_dynamic.py`
- `test/samples/Sync/test_dynamic_valid_shape.py`

Representative current failure text:

- `'pto.tload' op requires PTO-compatible vec ND2ND copy_gm_to_ubuf arguments`
- `'pto.tstore' op requires PTO-compatible vec ND2ND copy_ubuf_to_gm arguments`

### 2. Unary / Binary Vec Valid-Shape Contract Gap

- [x] `TRELU` and other unary vec lowering must not require a source/destination valid-region relation that is stricter than PTO A5 helper behavior.
- [x] Binary vec lowering must keep execution extent and valid-region reasoning aligned with PTO A5 decision structure, not merely with the easiest static contract.
- [x] Contract extraction and prechecks must be re-audited for:
  - source valid region
  - destination valid region
  - actual loop execution extent
  - which side drives tail behavior

Representative failing sample:

- `test/samples/Sync/test_dynamic_valid_shape.py`

Representative current failure text:

- `'pto.trelu' op relu lowering requires matching source and destination valid region`

Status note:

- Closed for the current vec seam by changing generic unary execution extent
  to follow the destination valid region and by relaxing prechecks from
  `src == dst` to `dst <= src` when both sides are statically known.
- This preserves the PTO A5-compatible decision that the destination-side
  execution region drives the vector loop/tail behavior.

Verified with:

- `./test/samples/runop.sh -t Sync`
- `./test/samples/run_a5vm_acceptance_checks.sh`

### 3. Missing Valid-Dim Fallback Gap

- [x] When seam IR no longer contains explicit `bind_tile` runtime valid dims, lowering must still derive valid rows/cols from shaped memref/tile information when PTO behavior allows it.
- [x] Vec lowering must not fail solely because valid dims are absent as explicit SSA operands if equivalent information is already present in type/shape form.
- [x] Higher-rank vec-shaped forms need an explicit rule for how they collapse into A5 vec execution extent.

Status note:

- Closed for the current structured tile-driven seam by adding shape-backed
  valid fallback only when the carrier can still be traced back to tile
  metadata (`tile_buf`, tile config, or explicit valid dims):
  0-D -> `1x1`, 1-D -> `1xN`, rank>=2 -> `prod(prefix_dims) x last_dim`.
- Runtime values are materialized from `memref.dim` only for those
  structured-tile-backed carriers. Pure memref-only samples without `tile_buf`
  semantics are intentionally out of scope and must not become accepted by
  this fallback.

Verified with:

- `./test/samples/runop.sh -t Sync`
- `./test/samples/run_a5vm_acceptance_checks.sh`

### 4. Missing Sync-Family Lowerings

- [x] Add lowering for `pto.barrier_sync[...]` to the correct backend representation, aligned with PTO/EmitC behavior.
- [x] Add lowering for `pto.get_buf`.
- [x] Add lowering for `pto.rls_buf`.
- [x] Confirm whether these lower to new `a5vm` ops or to LLVM-lowerable helper form by first checking `PTOToEmitC.cpp` and the PTO A5 implementation.

Status note:

- `pto.barrier_sync[...]` was already closed by the shared `LoweringSyncToPipe` pass before A5VM lowering:
  - `TLOAD -> PIPE_MTE2`
  - `TSTORE_VEC -> PIPE_MTE3`
  - `TVEC -> PIPE_V`, then erased on A5 by arch legalization
- A5VM backend now lowers:
  - `pto.barrier` -> `a5vm.pipe_barrier`
  - `pto.get_buf` -> `a5vm.get_buf`
  - `pto.rls_buf` -> `a5vm.rls_buf`
- `get_buf/rls_buf` reuse the shared PTO `SyncOpType -> PIPE` mapping and lower to hardware-facing A5VM ops, matching the EmitC/PTO intent.
- End-to-end acceptance coverage now includes:
  - `test/samples/Sync/test_a5_buf_sync.py`
  - `test/samples/run_a5vm_acceptance_checks.sh`

Representative failing samples:

- `test/samples/Sync/test_barrier_sync.py`
- `test/samples/Sync/test_a5_buf_sync.pto`

Representative current failure text:

- `missing pipe_barrier(PIPE_MTE2) lowering for barrier_sync[TLOAD]`
- `missing get_buf/rls_buf lowering`

Closed end-to-end acceptance:

- [x] `test/samples/Sync/test_barrier_sync.py` lowers through `pto.barrier` to `a5vm.pipe_barrier`.
- [x] `test/samples/Sync/test_a5_buf_sync.py` lowers to `a5vm.get_buf` / `a5vm.rls_buf`.

## Deferred Domains

These are not accidental regressions. They remain intentionally incomplete until the vector/data-movement path is stable.

### 5. Deferred `MAT` / `ACC` Domain Work

- [ ] Implement `TSTORE ACC` lowering.
- [ ] Implement `TSTORE MAT` lowering.
- [ ] Audit whether `TLOAD` needs parallel `ACC` / `MAT` branch completion beyond current vec-first coverage.

Representative failing samples:

- `test/samples/Partition5D/partition5d.py`
- `test/samples/Partition5D/partition5d_dynamic.py`

Representative current failure text:

- `TSTORE MAT lowering TODO for a5vm backend`

### 6. Deferred Matrix-Family Work

- [ ] Re-enable matrix-family seam buckets after vector/data-movement stabilization.
- [ ] Implement matrix-family lowerings by the same seam-first workflow.

Representative failing samples:

- `test/samples/Sync/matmul.py`
- `test/samples/Sync/tmatmulk_autosync.py`
- `test/samples/Sync/tmatmulk_autosync_a5.py`

Notes:

- These remain excluded by the current sample-expansion plan.

### 7. Backlog Of Remaining PTO Op Lowerings At The A5VM Seam

Purpose:

- Track every PTO op family that still survives into emitted `a5vm` output.
- Bind every gap to at least one `test/samples` acceptance case.
- Prevent sample-expansion work from losing uncovered families during iteration.

Rules:

- Every closure must be accepted with the listed `test/samples` case via `./test/samples/runop.sh`.
- A family is not closed merely because one sample compiles; the emitted `-pto.cpp` must no longer contain the listed `pto.*` op for the target sample.
- Every new closure must also rerun the previously closed sample-acceptance set. Do not trade one family’s closure for regressions in earlier families.
- If fixing item `N` breaks item `N-1`, item `N` is not considered closed. Regression-free closure is mandatory.
- Matrix-multiply family work remains excluded even if related ops appear in the scan.
- `MAT/cbuf` copy-family work is currently deferred; samples blocked only by that path stay open but are not immediate priority.
- Support scope is limited to seam IR still driven by structured `tile_buf`
  semantics. Pure memref-only samples are not valid A5VM backend targets for
  this phase, even if they happen to contain vec-shaped memrefs.
- A sample only counts as in-scope when the relevant vec op can still be traced
  to `tile_buf` / `alloc_tile` / `bind_tile` structure at the backend seam.

Explicit exclusions:

- `test/samples/Sync/test_inject_sync_intra_pipe_barrier.py`
  Reason:
  pure memref vec operands without structured `tile_buf` drivers; must remain
  unsupported for the A5VM backend.

#### 7.1 Reduce / Expand Family

- [x] `pto.trowmax`
  Acceptance samples:
  `test/samples/Rowmax/rowmax.py`
  `test/samples/PyPTOIRParser/paged_attention_example_kernel_softmax_prepare.py`
  Status note:
  Closed for the current row-reduce seam by lowering to PTO A5-style `__VEC_SCOPE__ + vbr + vlds + vcmax + vmax + vsts`.
  Verified with:
  `./test/samples/runop.sh -t Rowmax`
  `./test/samples/run_a5vm_acceptance_checks.sh`
- [x] `pto.trowmin`
  Acceptance samples:
  `test/samples/Rowmin/rowmin.py`
  Status note:
  Closed for the current row-reduce seam by lowering to PTO A5-style `__VEC_SCOPE__ + vbr + vlds + vcmin + vmin + vsts`.
  Verified with:
  `./test/samples/runop.sh -t Rowmin`
  `./test/samples/run_a5vm_acceptance_checks.sh`
- [x] `pto.trowsum`
  Acceptance samples:
  `test/samples/Rowsum/rowsum.py`
  `test/samples/PyPTOIRParser/paged_attention_example_kernel_softmax_prepare.py`
  Status note:
  Closed for the current row-reduce seam by lowering to PTO A5-style `__VEC_SCOPE__ + vbr + vlds + vcadd + vadd + vsts`.
  Verified with:
  `./test/samples/runop.sh -t Rowsum`
  `./test/samples/run_a5vm_acceptance_checks.sh`
- [x] `pto.tcolmax`
  Acceptance samples:
  `test/samples/Colmax/colmax.py`
  Status note:
  Closed for the current col-reduce seam by lowering to PTO A5-style `__VEC_SCOPE__ + vlds + vmax + vsts`.
  Verified with:
  `./test/samples/runop.sh -t Colmax`
  `./test/samples/run_a5vm_acceptance_checks.sh`
- [x] `pto.tcolmin`
  Acceptance samples:
  `test/samples/Colmin/colmin.py`
  Status note:
  Closed for the current col-reduce seam by lowering to PTO A5-style `__VEC_SCOPE__ + vlds + vmin + vsts`.
  Verified with:
  `./test/samples/runop.sh -t Colmin`
  `./test/samples/run_a5vm_acceptance_checks.sh`
- [x] `pto.tcolsum`
  Acceptance samples:
  `test/samples/Colsum/colsum.py`
  Status note:
  Closed for the current seam for both shared and binary-tmp sample forms by lowering to A5VM vector loads/adds/stores under `llvm.loop.aivector_scope`.
  Verified with:
  `./test/samples/runop.sh -t Colsum`
  `./test/samples/run_a5vm_acceptance_checks.sh`
- [x] `pto.tpartadd`
  Acceptance samples:
  `test/samples/Partadd/partadd.py`
  Status note:
  Closed for the current seam by following PTO A5 `TPartAdd` branch structure and lowering to `__VEC_SCOPE__ + vlds/vsts/vadd`.
  Verified with:
  `./test/samples/runop.sh -t Partadd`
  `./test/samples/run_a5vm_acceptance_checks.sh`
- [x] `pto.tpartmax`
  Acceptance samples:
  `test/samples/Partmax/partmax.py`
  Status note:
  Closed for the current seam by following PTO A5 `pad + copy src0 + combine src1` structure and lowering to `__VEC_SCOPE__ + vlds/vsts/vmax`.
  Verified with:
  `./test/samples/runop.sh -t Partmax`
  `./test/samples/run_a5vm_acceptance_checks.sh`
- [x] `pto.tpartmin`
  Acceptance samples:
  `test/samples/Partmin/partmin.py`
  Status note:
  Closed for the current seam by following PTO A5 `pad + copy src0 + combine src1` structure and lowering to `__VEC_SCOPE__ + vlds/vsts/vmin`.
  Verified with:
  `./test/samples/runop.sh -t Partmin`
  `./test/samples/run_a5vm_acceptance_checks.sh`
- [x] `pto.tcolexpand`
  Acceptance samples:
  `test/samples/Colexpand/colexpand.py`
  Status note:
  Closed for the current seam by lowering the source first-row column vector to repeated row stores under `llvm.loop.aivector_scope`.
  Verified with:
  `./test/samples/runop.sh -t Colexpand`
  `./test/samples/run_a5vm_acceptance_checks.sh`
- [x] `pto.trowexpand`
  Acceptance samples:
  `test/samples/Rowexpand/rowexpand.py`
  Status note:
  Closed for the current seam by following PTO A5 `TRowExpand` current sample path and lowering to `__VEC_SCOPE__ + vlds + vdup + vsts`.
  Verified with:
  `./test/samples/runop.sh -t Rowexpand`
  `./test/samples/run_a5vm_acceptance_checks.sh`
- [x] `pto.trowexpanddiv`
  Acceptance samples:
  `test/samples/Rowexpanddiv/rowexpanddiv.py`
  `test/samples/PyPTOIRParser/paged_attention_example_kernel_online_update.py`
  Status note:
  Closed for the current PTO A5-compatible seam by following
  `TRowExpandDiv.hpp` / `TRowExpandBinOp.hpp`:
  row-major `dst`, one side matching `dst`, expand-side constrained to
  `row_major validCol == 32 / sizeof(T)` or `col_major validCol == 1`,
  lowered to `__VEC_SCOPE__ + (vldas/vldus/vdup | vlds[BLK]) + vlds + vdiv + vsts`.
  Verified with:
  `./test/samples/runop.sh -t PyPTOIRParser`
  `./test/samples/run_a5vm_acceptance_checks.sh`
- [x] `pto.trowexpandmul`
  Acceptance samples:
  `test/samples/Rowexpandmul/rowexpandmul.py`
  `test/samples/PyPTOIRParser/paged_attention_example_kernel_online_update.py`
  Status note:
  Closed for the current PTO A5-compatible seam by following
  `TRowExpandMul.hpp` / `TRowExpandBinOp.hpp`:
  row-major `dst`, one side matching `dst`, expand-side constrained to
  `row_major validCol == 32 / sizeof(T)` or `col_major validCol == 1`,
  lowered to `__VEC_SCOPE__ + (vldas/vldus/vdup | vlds[BLK]) + vlds + vmul + vsts`.
  Verified with:
  `./test/samples/runop.sh -t PyPTOIRParser`
  `./test/samples/run_a5vm_acceptance_checks.sh`
- [x] `pto.trowexpandsub`
  Acceptance samples:
  `test/samples/Rowexpandsub/rowexpandsub.py`
  `test/samples/PyPTOIRParser/paged_attention_example_kernel_softmax_prepare.py`
  Status note:
  Closed for the current `softmax_prepare` seam where `src0`/`dst` share the
  row-major tile type and the expand-side operand is `col_major, validCol == 1`,
  following PTO A5 `TRowExpandSub.hpp` / `TRowExpandBinOp.hpp` as
  `__VEC_SCOPE__ + vldas + vldus + vdup + vlds + vsub + vsts`.
  The standalone `Rowexpandsub` sample remains blocked under strict PTO A5
  shape-contract rules if its seam input does not satisfy the required expand-side shape.
  Verified with:
  `./test/samples/runop.sh -t PyPTOIRParser`
  `./test/samples/run_a5vm_acceptance_checks.sh`
- [x] `pto.texpands`
  Acceptance samples:
  `test/samples/Expands/expand.py`
  `test/samples/Expands/expands.py`
  Status note:
  Closed for the current seam by following PTO A5 `TExpandS` and lowering to `__VEC_SCOPE__ + vdup + vsts`.
  Verified with:
  `./test/samples/runop.sh -t Expands`
  `./test/samples/run_a5vm_acceptance_checks.sh`

#### 7.2 Scalar-Operand Vec Variants

- [x] `pto.tadds`
  Acceptance samples:
  `test/samples/Adds/adds.py`
  `test/samples/DataMovement/dataMovement.py`
  Status note:
  Closed for the current scalar-operand vec seam by following PTO A5 `TAddS`
  and lowering to `__VEC_SCOPE__ + vlds + vadds + vsts`.
  Verified with:
  `./test/samples/runop.sh -t Adds`
  `./test/samples/runop.sh -t DataMovement`
  `./test/samples/run_a5vm_acceptance_checks.sh`
- [x] `pto.taddc`
  Acceptance samples:
  `test/samples/Addc/addc.py`
  Status note:
  Closed for the current structured vec seam by following the PTO/EmitC
  decomposition `TADD + TADD`; the emitted `runop.sh` artifact now lowers all
  the way to two final `a5vm.vadd` regions and no longer leaves residual
  `pto.taddc`.
  Verified with:
  `./test/samples/runop.sh -t Addc`
- [x] `pto.taddsc`
  Acceptance samples:
  `test/samples/Addsc/addsc.py`
  `test/samples/TileScalar/tileScalar.py`
  Status note:
  Closed for the current structured vec seam by following the PTO/EmitC
  decomposition `TADDS + TADD`; the emitted `runop.sh` artifact now lowers to
  `a5vm.vadds + a5vm.vadd` and no longer leaves residual `pto.taddsc`.
  Verified with:
  `./test/samples/runop.sh -t Addsc`
  `./test/samples/runop.sh -t TileScalar`
- [x] `pto.tsubs`
  Acceptance samples:
  `test/samples/Subs/subs.py`
  `test/samples/Subset/subset_tsubs.py`
  Status note:
  Closed for the current scalar-operand vec seam by following PTO A5 `TSubS`
  and lowering to `__VEC_SCOPE__ + (0 - scalar) + vadds + vsts`.
  Verified with:
  `./test/samples/runop.sh -t Subs`
  `./test/samples/run_a5vm_acceptance_checks.sh`
- [x] `pto.tsubc`
  Acceptance samples:
  `test/samples/Subc/subc.py`
  Status note:
  Closed for the current structured vec seam by following the PTO/EmitC
  decomposition `TSUB + TADD`; the emitted `runop.sh` artifact now lowers to
  `a5vm.vsub + a5vm.vadd` and no longer leaves residual `pto.tsubc`.
  Verified with:
  `./test/samples/runop.sh -t Subc`
- [x] `pto.tsubsc`
  Acceptance samples:
  `test/samples/Subsc/subsc.py`
  Status note:
  Closed for the current structured vec seam by following the PTO/EmitC
  decomposition `TSUBS + TADD`; the emitted `runop.sh` artifact now lowers to
  `a5vm.vadds + a5vm.vadd` and no longer leaves residual `pto.tsubsc`.
  Verified with:
  `./test/samples/runop.sh -t Subsc`
- [x] `pto.tmuls`
  Acceptance samples:
  `test/samples/Muls/muls.py`
  `test/samples/PyPTOIRParser/paged_attention_example_kernel_softmax_prepare.py`
  Status note:
  Closed for the current scalar-operand vec seam by following PTO A5 `TMulS`
  shape/type contract and lowering to `__VEC_SCOPE__ + vlds + vmuls + vsts`.
  Verified with:
  `./test/samples/runop.sh -t Muls`
  `./test/samples/runop.sh -t PyPTOIRParser`
  `./test/samples/run_a5vm_acceptance_checks.sh`
- [x] `pto.tdivs`
  Acceptance samples:
  `test/samples/Divs/divs.py`
  `test/samples/Divs2/divs2.py`
  Status note:
  Closed for the current A5 seam by following PTO A5 `TDivS` floating-point
  branch structure: the current emitted seam normalizes the sample pair to the
  tile/scalar path, which lowers to `__VEC_SCOPE__ + vlds + (1/scalar) + vmuls + vsts`.
  Verified with:
  `./test/samples/runop.sh -t Divs`
  `./test/samples/runop.sh -t Divs2`
  `./test/samples/run_a5vm_acceptance_checks.sh`

#### 7.3 Logical / Compare / Select Family

- [x] `pto.tand`
  Acceptance samples:
  `test/samples/And/and.py`
  `test/samples/Elementwise/elementwise.py`
  Status note:
  Closed for the current structured vec seam by following PTO A5 `TBinOp`
  binary tile skeleton: `__VEC_SCOPE__ + vlds + vand + vsts/vsts_pred`,
  with the vector loop driven by destination valid shape.
  Verified with:
  `./test/samples/runop.sh -t And`
- [x] `pto.tands`
  Acceptance samples:
  `test/samples/Ands/ands.py`
  Status note:
  Closed for the current structured vec seam by following PTO A5 `TBinSOp`
  scalar-tile skeleton: `__VEC_SCOPE__ + vdup + vlds + vand + vsts/vsts_pred`.
  Verified with:
  `./test/samples/runop.sh -t Ands`
- [x] `pto.tor`
  Acceptance samples:
  `test/samples/Or/or.py`
  Status note:
  Closed for the current structured vec seam by following PTO A5 `TBinOp`
  binary tile skeleton with `a5vm.vor`.
  Verified with:
  `./test/samples/runop.sh -t Or`
- [x] `pto.tors`
  Acceptance samples:
  `test/samples/Ors/ors.py`
  Status note:
  Closed for the current structured vec seam by following PTO A5 `TBinSOp`
  scalar-tile skeleton with `a5vm.vor`.
  Verified with:
  `./test/samples/runop.sh -t Ors`
- [x] `pto.txor`
  Acceptance samples:
  `test/samples/Xor/xor.py`
  Status note:
  Closed for the current structured vec seam by following PTO A5 `TBinOp`
  binary tile skeleton with `a5vm.vxor`.
  Verified with:
  `./test/samples/runop.sh -t Xor`
- [x] `pto.txors`
  Acceptance samples:
  `test/samples/Xors/xors.py`
  Status note:
  Closed for the current structured vec seam by following PTO A5 `TBinSOp`
  scalar-tile skeleton with `a5vm.vxor`.
  Verified with:
  `./test/samples/runop.sh -t Xors`
- [x] `pto.tcmp`
  Acceptance samples:
  `test/samples/Cmp/cmp.py`
  Status note:
  Closed for the current f32-to-ui8 packed-predicate seam by following PTO A5
  `TCMP_IMPL` 32-byte path with `vlds + vcmp + pdintlv_b8 + psts` under
  `__VEC_SCOPE__`.
  Verified with:
  `./test/samples/runop.sh -t Cmp`
  `./test/samples/run_a5vm_acceptance_checks.sh`
- [x] `pto.tcmps`
  Acceptance samples:
  `test/samples/Cmps/cmps.py`
  Status note:
  Closed for the current f32-to-ui8 packed-predicate seam by following PTO A5
  `TCMPS_IMPL` 32-byte path with `vlds + vcmps + pdintlv_b8 + psts` under
  `__VEC_SCOPE__`.
  Verified with:
  `./test/samples/runop.sh -t Cmps`
  `./test/samples/run_a5vm_acceptance_checks.sh`
- [x] `pto.tsel`
  Acceptance samples:
  `test/samples/Sel/sel.py`
  `test/samples/Sel/sel_head.py`
  Status note:
  Closed for the current `TSel_b32` seam by following PTO A5 `TSelHead +
  TSelTail`, lowering `plds + pintlv_b16/punpack + vsel + predicated vsts`
  under `__VEC_SCOPE__`.
  Verified with:
  `./test/samples/runop.sh -t Sel`
  `./test/samples/run_a5vm_acceptance_checks.sh`
- [x] `pto.tsels`
  Acceptance samples:
  `test/samples/Sels/sels.py`
  Status note:
  Closed for the current no-pad vector seam by following PTO A5 `TSELS_IMPL`,
  materializing the global predicate with `a5vm.pset_b8` and selecting with
  `a5vm.vsel` under `__VEC_SCOPE__`.
  Verified with:
  `./test/samples/runop.sh -t Sels`
  `./test/samples/run_a5vm_acceptance_checks.sh`

#### 7.4 Unary / Math / Conversion Family

- [x] `pto.tneg`
  Acceptance samples:
  `test/samples/Neg/neg.py`
  Status note:
  Closed by following PTO A5 `TNEG_IMPL`, which delegates to `TMULS_IMPL(dst, src, -1)`
  instead of introducing a dedicated negate builtin. Lowered to the existing scalar
  unary vec path as `__VEC_SCOPE__ + vlds + vmuls(-1) + vsts`.
  Verified with:
  `./test/samples/runop.sh -t Neg`
  `./test/samples/run_a5vm_acceptance_checks.sh`
- [x] `pto.tlrelu`
  Acceptance samples:
  `test/samples/Lrelu/lrelu.py`
  Status note:
  Closed for the current scalar-slope vec seam by following PTO A5 `TLRELU`
  intent and lowering to `__VEC_SCOPE__ + vlds + vlrelu + vsts`.
  Verified with:
  `./test/samples/runop.sh -t Lrelu`
  `./test/samples/run_a5vm_acceptance_checks.sh`
- [ ] `pto.tprelu`
  Acceptance samples:
  `test/samples/Prelu/prelu.py`
  Status note:
  Skipped for the current phase by user direction. Do not implement until this
  family is explicitly re-opened.
- [x] `pto.trsqrt`
  Acceptance samples:
  `test/samples/Rsqrt/rsqrt.py`
  Status note:
  Closed by following PTO A5 `TRSQRT_IMPL`, which materializes a vector of ones
  and lowers to `vdup(1.0) + vsqrt + vdiv + vsts` under `__VEC_SCOPE__`.
  Verified with:
  `./test/samples/runop.sh -t Rsqrt`
  `./test/samples/run_a5vm_acceptance_checks.sh`
- [x] `pto.tcvt`
  Acceptance samples:
  `test/samples/Cvt/cvt.py`
  `test/samples/PyPTOIRParser/paged_attention_example_kernel_softmax_prepare.py`
  `test/samples/Reshape/bitcast_inplace_cvt.py`
  Status note:
  Closed for the currently proven PTO A5 seam paths:
  `f32 -> f32` via `vtrc`, as used by `test/samples/Cvt/cvt.py`;
  `f32 -> bf16` via `vlds + vcvt(PART_ODD/PART_EVEN, RS_ENABLE) + vor + vsts`;
  `bf16 -> f32` via `vlds(UNPK_B16) + vcvt(PART_EVEN) + vsts`,
  as used by `test/samples/PyPTOIRParser/paged_attention_example_kernel_softmax_prepare.py`.
  Wider dtype-conversion branches from `TCvt.hpp` still remain open until a
  corresponding non-matrix `test/samples` seam is brought under acceptance.
  Verified with:
  `./test/samples/runop.sh -t Cvt`
  `./test/samples/runop.sh -t PyPTOIRParser`
  `./test/samples/run_a5vm_acceptance_checks.sh`
- [x] `pto.tci`
  Acceptance samples:
  `test/samples/Ci/ci.py`
  Status note:
  Closed for the current A5 seam by following PTO A5 `TCI`, which is a
  software loop over valid columns rather than a dedicated vector builtin path.
  Lowered to `scf.for + arith + llvm.store`, followed by existing `TSTORE`
  copy-family lowering.
  Verified with:
  `./test/samples/runop.sh -t Ci`
  `./test/samples/run_a5vm_acceptance_checks.sh`
- [x] `pto.tmax`
  Acceptance samples:
  `test/samples/Max/max.py`
  Status note:
  Closed for the current binary vec seam by following PTO A5 `TMax`
  and lowering to `__VEC_SCOPE__ + vlds + vmax + vsts`.
  Verified with:
  `./test/samples/runop.sh -t Max`
  `./test/samples/run_a5vm_acceptance_checks.sh`
- [x] `pto.tmaxs`
  Acceptance samples:
  `test/samples/Maxs/maxs.py`
  Status note:
  Closed for the current scalar-operand vec seam by following PTO A5 `TMaxS`
  and lowering to `__VEC_SCOPE__ + vlds + vmaxs + vsts`.
  Verified with:
  `./test/samples/runop.sh -t Maxs`
  `./test/samples/run_a5vm_acceptance_checks.sh`
- [x] `pto.tmin`
  Acceptance samples:
  `test/samples/Min/min.py`
  Status note:
  Closed for the current binary vec seam by following PTO A5 `TMin`
  and lowering to `__VEC_SCOPE__ + vlds + vmin + vsts`.
  Verified with:
  `./test/samples/runop.sh -t Min`
  `./test/samples/run_a5vm_acceptance_checks.sh`
- [x] `pto.tmins`
  Acceptance samples:
  `test/samples/Mins/mins.py`
  Status note:
  Closed for the current scalar-operand vec seam by following PTO A5 `TMins`
  and lowering to `__VEC_SCOPE__ + vlds + vmins + vsts`.
  Verified with:
  `./test/samples/runop.sh -t Mins`
  `./test/samples/run_a5vm_acceptance_checks.sh`
- [ ] `pto.trem`
  Acceptance samples:
  `test/samples/Rem/rem.py`
  Status note:
  Skipped for the current phase. PTO common entry exists, but no corresponding
  A5 implementation was found under `pto/npu/a5`, so this backend must not
  guess semantics.
- [ ] `pto.trems`
  Acceptance samples:
  `test/samples/Rems/rems.py`
  Status note:
  Skipped for the current phase. PTO common entry exists, but no corresponding
  A5 implementation was found under `pto/npu/a5`, so this backend must not
  guess semantics.
- [ ] `pto.tshl`
  Acceptance samples:
  `test/samples/Shl/shl.py`
  Status note:
  Skipped for the current phase. PTO common entry exists, but no corresponding
  A5 implementation was found under `pto/npu/a5`, so this backend must not
  guess semantics.
- [ ] `pto.tshls`
  Acceptance samples:
  `test/samples/Shls/shls.py`
  Status note:
  Skipped for the current phase. PTO common entry exists, but no corresponding
  A5 implementation was found under `pto/npu/a5`, so this backend must not
  guess semantics.
- [ ] `pto.tshr`
  Acceptance samples:
  `test/samples/Shr/shr.py`
  Status note:
  Skipped for the current phase. PTO common entry exists, but no corresponding
  A5 implementation was found under `pto/npu/a5`, so this backend must not
  guess semantics.
- [ ] `pto.tshrs`
  Acceptance samples:
  `test/samples/Shrs/shrs.py`
  Status note:
  Skipped for the current phase. PTO common entry exists, but no corresponding
  A5 implementation was found under `pto/npu/a5`, so this backend must not
  guess semantics.

#### 7.5 Data Movement / Indexed / Permute Family

- [x] `pto.tgather`
  Acceptance samples:
  `test/samples/Gather/gather.py`
- [x] `pto.tgatherb`
  Acceptance samples:
  `test/samples/Gatherb/gatherb.py`
- [x] `pto.tscatter`
  Acceptance samples:
  `test/samples/Scatter/scatter.py`
- [ ] `pto.mgather`
  Acceptance samples:
  `test/samples/Mgather/mgather.py`
  Status note:
  Skipped for the current phase. This is a global-memory gather seam rather
  than the current structured tile_buf-driven backend scope.
- [ ] `pto.mscatter`
  Acceptance samples:
  `test/samples/Mscatter/mscatter.py`
  Status note:
  Skipped for the current phase. This is a global-memory scatter seam rather
  than the current structured tile_buf-driven backend scope.
- [x] `pto.ttrans`
  Acceptance samples:
  `test/samples/Trans/trans.py`
  Status note:
  Closed for the current b32 row-major seam by following PTO A5 `TTRANS`
  index-generation and gather structure, lowering to
  `__VEC_SCOPE__ + vci + vmuls + vadds + vgather2 + vsts`.
  Verified with:
  `./test/samples/runop.sh -t Trans`
  `./test/samples/run_a5vm_acceptance_checks.sh`
- [x] `pto.tsort`
  Acceptance samples:
  `test/samples/Sort32/sort32.py`
  Status note:
  Closed for the current vec row-major seam by following PTO A5 `TSort32`
  structure, lowering each row to `a5vm.vbitsort` over UB-resident source,
  value-dst, and index-dst tiles without leaving PTO scaffolding in the
  emitted text.
  Verified with:
  `./test/samples/runop.sh -t Sort32`
  `./test/samples/run_a5vm_acceptance_checks.sh`
- [x] `pto.tmrgsort`
  Acceptance samples:
  `test/samples/Mrgsort/mrgsort.py`
  `test/samples/Mrgsort/mrgsort_a5.py`
  `test/samples/Mrgsort/mrgsort_format2.py`
  Status note:
  Closed for the current vec row-major seam by following PTO A5 `TMrgSort`
  format decisions, lowering to `a5vm.vmrgsort4` and using
  `a5vm.copy_ubuf_to_ubuf` for the format2 tmp-to-dst materialization path.
  Verified with:
  `./test/samples/runop.sh -t Mrgsort`
  `./test/samples/run_a5vm_acceptance_checks.sh`

#### 7.6 Padding / Tile Utility / Debug Family

- [x] `pto.tfillpad`
  Acceptance samples:
  `test/samples/Fillpad/fillpad.py`
  Status note:
  Closed for the current vec row-major seam by following PTO A5 `TFILLPAD`
  copy-then-pad structure, lowering to `__VEC_SCOPE__ + vlds + vsts_pred`.
  Verified with:
  `./test/samples/runop.sh -t Fillpad`
  `./test/samples/run_a5vm_acceptance_checks.sh`
- [x] `pto.tfillpad_expand`
  Acceptance samples:
  `test/samples/Fillpad/fillpad_expand.py`
  Status note:
  Closed for the current vec row-major expand seam by following PTO A5
  `TFILLPAD_EXPAND` structure, lowering to
  `__VEC_SCOPE__ + vlds + vdup + vsts_pred`.
  Verified with:
  `./test/samples/runop.sh -t Fillpad`
  `./test/samples/run_a5vm_acceptance_checks.sh`
- [ ] `pto.tgetval`
  Acceptance samples:
  `test/samples/TileSetGetValue/tileSetGetValue.py`
  Status note:
  Skipped for the current phase by user direction. Do not implement until this
  family is explicitly re-opened.
- [ ] `pto.tsetval`
  Acceptance samples:
  `test/samples/TileSetGetValue/tileSetGetValue.py`
  Status note:
  Skipped for the current phase by user direction. Do not implement until this
  family is explicitly re-opened.
- [ ] `pto.tprint`
  Acceptance samples:
  `test/samples/Print/print.py`
  Status note:
  Skipped for the current phase by user direction. Do not implement until this
  family is explicitly re-opened.
- [ ] `pto.trap`
  Acceptance samples:
  `test/samples/Trap/trap.py`
  Status note:
  Skipped for the current phase by user direction. Do not implement until this
  family is explicitly re-opened.

#### 7.7 Scaffold Residue That Also Needs Elimination

- [x] `pto.pointer_cast` must not remain in final A5VM backend output except where an explicit LLVM-lowerable pointer expression is intentionally preserved.
  Acceptance samples:
  `test/samples/Abs/abs.py`
  `test/samples/Reshape/reshape.py`
  `test/samples/Rowmax/rowmax.py`
  `test/samples/PyPTOIRParser/paged_attention_example_kernel_online_update.py`
  Status note:
  Closed for current final `--pto-backend=a5vm` text output: scaffold pointer
  carriers are rewritten into `memref.reinterpret_cast`,
  `memref.extract_aligned_pointer_as_index`, and `llvm.inttoptr/ptrtoint`
  before emission, and acceptance now scans all generated final `*.cpp`
  outputs for residual `pto.pointer_cast`.
  Verified with:
  `./test/samples/run_a5vm_acceptance_checks.sh`
- [x] `pto.bind_tile` must not remain in final A5VM backend output once the corresponding PTO operation family is lowered.
  Acceptance samples:
  `test/samples/Abs/abs.py`
  `test/samples/Reshape/reshape.py`
  `test/samples/Rowmax/rowmax.py`
  `test/samples/PyPTOIRParser/paged_attention_example_kernel_online_update.py`
  Status note:
  Closed for current final `--pto-backend=a5vm` text output: tile scaffold is
  consumed during lowering and acceptance now scans all generated final
  `*.cpp` outputs for residual `pto.bind_tile`.
  Verified with:
  `./test/samples/run_a5vm_acceptance_checks.sh`

## Near-Term Execution Checklist

- [x] Fix dynamic/tail `TLOAD` and `TSTORE`.
- [x] Fix unary/binary vec valid-shape contract handling.
- [x] Add valid-dim fallback derivation for shaped seam values.
- [x] Implement `barrier_sync` lowering.
- [x] Implement `get_buf/rls_buf` lowering.
- [ ] Burn down section `7. Backlog Of Remaining PTO Op Lowerings At The A5VM Seam` family by family, each with `test/samples` acceptance.
- [x] After each family closure, rerun the accumulated acceptance set so earlier closed families stay green.
- [ ] Re-run `Sync`, `Partition5D`, and current phase-2 checks after each closure.
