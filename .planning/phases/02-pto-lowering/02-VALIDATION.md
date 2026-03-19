---
phase: 02
slug: pto-lowering
status: ready
nyquist_compliant: true
wave_0_complete: true
created: 2026-03-19
---

# Phase 02 — Validation Strategy

> Per-phase validation contract for feedback sampling during execution.

---

## Test Infrastructure

| Property | Value |
|----------|-------|
| **Framework** | other — MLIR `RUN:` + `FileCheck` tests plus direct phase runner and build-tree smoke checks |
| **Config file** | none committed in source tree; direct `RUN:` lines plus `test/phase2/run_phase2_checks.sh` |
| **Quick run command** | `bash test/phase2/run_phase2_checks.sh` |
| **Full suite command** | `bash test/phase2/run_phase2_checks.sh` |
| **Estimated runtime** | ~20 seconds |

The committed runner resolves `filecheck_bin` deterministically before running fixtures. Search order: `FileCheck`, `FileCheck-19`, then `/usr/lib/llvm-19/bin/FileCheck`. If no candidate exists, `bash test/phase2/run_phase2_checks.sh` exits early with a checked-path error instead of a shell `command not found` failure.

---

## Sampling Rate

- **After every task commit:** Run `bash test/phase2/run_phase2_checks.sh`
- **After every plan wave:** Run `bash test/phase2/run_phase2_checks.sh`
- **Before `$gsd-verify-work`:** Full suite must be green
- **Max feedback latency:** 20 seconds

---

## Per-Task Verification Map

| Task ID | Plan | Wave | Requirement | Test Type | Automated Command | File Exists | Status |
|---------|------|------|-------------|-----------|-------------------|-------------|--------|
| 02-01-01 | 01 | 0 | PTO-01 | integration | `./build/tools/ptoas/ptoas --pto-backend=a5vm --a5vm-print-ir test/phase2/tload_copy_family_shape.mlir -o /dev/null 2>&1 | FileCheck test/phase2/tload_copy_family_shape.mlir` | ❌ W0 | ⬜ pending |
| 02-01-02 | 01 | 0 | PTO-02 | integration | `./build/tools/ptoas/ptoas --pto-backend=a5vm --a5vm-print-ir test/phase2/tabs_abs_loop_shape.mlir -o /dev/null 2>&1 | FileCheck test/phase2/tabs_abs_loop_shape.mlir` | ❌ W0 | ⬜ pending |
| 02-01-03 | 01 | 0 | PTO-02 | integration | `./build/tools/ptoas/ptoas --pto-backend=a5vm --a5vm-print-ir test/phase2/tabs_precheck_a5.mlir -o /dev/null 2>&1 | FileCheck test/phase2/tabs_precheck_a5.mlir` | ❌ W0 | ⬜ pending |
| 02-01-04 | 01 | 0 | PTO-03 | integration | `./build/tools/ptoas/ptoas --pto-backend=a5vm --a5vm-print-ir test/phase2/tstore_copy_family_shape.mlir -o /dev/null 2>&1 | FileCheck test/phase2/tstore_copy_family_shape.mlir` | ❌ W0 | ⬜ pending |
| 02-01-05 | 01 | 0 | PTO-03,PTO-04 | integration | `./build/tools/ptoas/ptoas --pto-backend=a5vm --a5vm-print-ir test/phase2/tstore_domain_todos.mlir -o /dev/null 2>&1 | FileCheck test/phase2/tstore_domain_todos.mlir` | ❌ W0 | ⬜ pending |
| 02-01-06 | 01 | 0 | PTO-04 | integration | `./build/tools/ptoas/ptoas --pto-backend=a5vm --a5vm-print-ir test/phase2/pto_backend_a5vm_wiring.mlir -o /dev/null 2>&1 | FileCheck test/phase2/pto_backend_a5vm_wiring.mlir` | ❌ W0 | ⬜ pending |

*Status: ⬜ pending · ✅ green · ❌ red · ⚠️ flaky*

---

## Wave 0 Requirements

- [ ] `test/phase2/tload_copy_family_shape.mlir` — verify `TLOAD` lowers through copy-family A5VM structure, preserving layout, shape, strides, valid region, pad/init fields, partition trace, and out-to-UB loop programming
- [ ] `test/phase2/tabs_abs_loop_shape.mlir` — verify `TABS` lowers through `__VEC_SCOPE__` plus inner software loop with ordered `a5vm.vlds`, `a5vm.vabs`, and `a5vm.vsts`
- [ ] `test/phase2/tabs_precheck_a5.mlir` — verify `TABS` rejects unsupported non-A5/vector/layout/shape/type cases before creating any A5VM IR
- [ ] `test/phase2/tstore_copy_family_shape.mlir` — verify `TSTORE` lowers through UB-to-GM copy-family structure with destination layout/shape/strides, valid region, trace info, and UB-to-out loop programming
- [ ] `test/phase2/tstore_domain_todos.mlir` — verify explicit `VEC` / `ACC` / `MAT` branch behavior and dedicated TODO diagnostics for unsupported domains
- [ ] `test/phase2/pto_backend_a5vm_wiring.mlir` — verify `--pto-backend=a5vm` runs PTO-to-A5VM lowering before final backend emission and does not route through obsolete EmitC/PTO pseudo-lowering
- [ ] `test/phase2/run_phase2_checks.sh` — direct invocation path for the corrected Phase 2 suite

---

## Manual-Only Verifications

| Behavior | Requirement | Why Manual | Test Instructions |
|----------|-------------|------------|-------------------|
| PTO-library correspondence remains readable in code | PTO-04 | code readability of the template-to-lowering correspondence is partly structural | Inspect `lowerTLOAD`, `lowerTABS`, and `lowerTSTORE`, confirm the first-order A5 PTO branches and loop structure are recognizable |
| Loop-programming representation is truthful without over-encoding final hardware detail | PTO-01,PTO-03 | exact adequacy of loop-programming IR is partly a design judgment | Inspect the chosen A5VM representation for `set_loop*_outtoub` and `set_loop*_ubtoout`, confirm it preserves the needed semantics without pretending to be final hardware text |

---

## Validation Sign-Off

- [x] All tasks have `<automated>` verify or Wave 0 dependencies
- [x] Sampling continuity: no 3 consecutive tasks without automated verify
- [x] Wave 0 covers all MISSING references
- [x] No watch-mode flags
- [x] Feedback latency < 45s
- [x] `nyquist_compliant: true` set in frontmatter

**Approval:** ready for execution planning
