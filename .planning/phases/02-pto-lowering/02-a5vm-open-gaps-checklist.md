# A5VM Open Gaps Checklist

Updated: 2026-03-19

Purpose:

- Record the currently confirmed failure points for the `a5vm` backend.
- Keep dynamic-shape gaps, missing lowerings, and deferred-domain work separate.
- Provide a stable checklist so implementation can proceed item by item without losing scope.

## Priority Order

- [x] P0. Broaden `TLOAD/TSTORE` copy-family lowering so dynamic and tail copies follow PTO A5 behavior instead of only the current narrow vec ND2ND happy path.
- [ ] P0. Relax and correct unary/binary vec valid-shape contract checks so they match PTO A5 helper behavior instead of requiring today’s over-strict source/destination equality.
- [ ] P1. Add valid-region fallback derivation when seam IR no longer carries explicit `bind_tile` valid dims but the shaped memref/tile form still determines them.
- [ ] P1. Lower synchronization-family ops that are still left as PTO ops at the backend seam.
- [ ] P2. Revisit deferred `ACC`/`MAT` domains and matrix-family samples after vector/data-movement paths are stable.

## Confirmed Active Gaps

### 1. Dynamic / Tail Copy-Family Gap

- [ ] `lowerTLOAD` must support dynamic or tail copy scenarios that currently fail with:
  `requires PTO-compatible vec ND2ND copy_gm_to_ubuf arguments`
- [ ] `lowerTSTORE` must support dynamic or tail copy scenarios that currently fail with:
  `requires PTO-compatible vec ND2ND copy_ubuf_to_gm arguments`
- [ ] Copy-family transfer derivation must recover PTO-relevant shape/stride decisions from the seam value graph, not only from the current static full-tile case.
- [ ] Dynamic transfer operands must stay on the existing unified `a5vm.copy_gm_to_ubuf` / `a5vm.copy_ubuf_to_gm` interface. No static-vs-dynamic duplicate ops.
- [ ] `len_burst`, loop programming, and stride decisions must accept runtime valid-shape participation where PTO A5 behavior requires it.

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

- [ ] `TRELU` and other unary vec lowering must not require a source/destination valid-region relation that is stricter than PTO A5 helper behavior.
- [ ] Binary vec lowering must keep execution extent and valid-region reasoning aligned with PTO A5 decision structure, not merely with the easiest static contract.
- [ ] Contract extraction and prechecks must be re-audited for:
  - source valid region
  - destination valid region
  - actual loop execution extent
  - which side drives tail behavior

Representative failing sample:

- `test/samples/Sync/test_dynamic_valid_shape.py`

Representative current failure text:

- `'pto.trelu' op relu lowering requires matching source and destination valid region`

Planned end-to-end acceptance once fixed:

- [ ] Add `test/samples/Sync/test_dynamic_valid_shape.py` to `test/samples/run_a5vm_acceptance_checks.sh` as a required passing sample after unary valid-shape contract lowering is corrected.

### 3. Missing Valid-Dim Fallback Gap

- [ ] When seam IR no longer contains explicit `bind_tile` runtime valid dims, lowering must still derive valid rows/cols from shaped memref/tile information when PTO behavior allows it.
- [ ] Vec lowering must not fail solely because valid dims are absent as explicit SSA operands if equivalent information is already present in type/shape form.
- [ ] Higher-rank vec-shaped forms need an explicit rule for how they collapse into A5 vec execution extent.

Representative failing sample:

- `test/samples/Sync/test_inject_sync_intra_pipe_barrier.py`

Representative current failure text:

- `binary lowering requires valid rows and cols`

Planned end-to-end acceptance once fixed:

- [ ] Add `test/samples/Sync/test_inject_sync_intra_pipe_barrier.py` to the sample acceptance script after valid-dim fallback is implemented.

### 4. Missing Sync-Family Lowerings

- [ ] Add lowering for `pto.barrier_sync[...]` to the correct backend representation, aligned with PTO/EmitC behavior.
- [ ] Add lowering for `pto.get_buf`.
- [ ] Add lowering for `pto.rls_buf`.
- [ ] Confirm whether these lower to new `a5vm` ops or to LLVM-lowerable helper form by first checking `PTOToEmitC.cpp` and the PTO A5 implementation.

Representative failing samples:

- `test/samples/Sync/test_barrier_sync.py`
- `test/samples/Sync/test_a5_buf_sync.pto`

Representative current failure text:

- `missing pipe_barrier(PIPE_MTE2) lowering for barrier_sync[TLOAD]`
- `missing get_buf/rls_buf lowering`

Planned end-to-end acceptance once fixed:

- [ ] Add `test/samples/Sync/test_barrier_sync.py` as a required passing sample after `barrier_sync` lowering lands.
- [ ] Add `test/samples/Sync/test_a5_buf_sync.py` as a required passing sample after `get_buf/rls_buf` lowering lands.

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

## Near-Term Execution Checklist

- [x] Fix dynamic/tail `TLOAD` and `TSTORE`.
- [ ] Fix unary/binary vec valid-shape contract handling.
- [ ] Add valid-dim fallback derivation for shaped seam values.
- [ ] Implement `barrier_sync` lowering.
- [ ] Implement `get_buf/rls_buf` lowering.
- [ ] Re-run `Sync`, `Partition5D`, and current phase-2 checks after each closure.
