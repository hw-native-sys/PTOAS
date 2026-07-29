# store_pad8 — VMI pad-8 reduce store repro

Self-contained cannsim microbench for PTOAS gap 03: after a cross-lane
`vcadd`, CCE stores one f32 scalar per group with `vsts ... ONEPT_B32`.
Product VMI kernels often broadcast the reduced value to 8 lanes (`vbrc`)
and store under an 8-wide mask.

This repro isolates that epilogue and compares **kernel formulations** on a
built PTOAS pin (branch `zjw/store_pad_issue`, tag lineage **vmi-v0.1.3**).

## What to compare

| Path | Store shape | Notes |
|------|-------------|-------|
| CCE (`cce/`) | ONEPT / 1 scalar | Baseline — see `fixtures/reference_asc_cce.asc` |
| VMI pad-8 (`vmi/store_pad8_vmi.py`) | pad-8 + mask8 | Product workaround |
| VMI mask1 (`vmi/store_mask1_vmi.py`) | compact `[N]` + mask1 | Masked 1-lane store |
| VMI **group1** (`vmi/store_group1_vmi.py`) | `vcadd(group=1)` + `vstore(group=1)` | Compact score → point store / `1PT_B32` |
| VMI desired (`fixtures/desired_vmi.*`) | mask-1 after `vcadd` | Compile-only single-tile fixture |
| VMI target MI (`fixtures/target_mi.pto`) | `dist = "1PT_B32"` | Hand-written MI; bisheng may still crash |

**Best VMI path:** `group1`. It lowers to `vsts ... {dist = "1PT_B32"}` and
matches CCE’s `vcadd` + scalar store shape in cannsim (see table below).

## Layout

```
store_pad8/
  cce/          CCE ONEPT kernel + ctypes launcher
  vmi/          VMI pad-8 / mask1 / group1 kernels (ptodsl)
  common/       torch runtime, golden ref, build helper, top launcher
  fixtures/     IR dumps and reference ASC from gap 03
  test/         correctness test
  scripts/      env, cannsim runners, check_desired, run_three_way
```

## Prerequisites

- CANN 9.0 simulator (`cannsim`, `bisheng`)
- Built PTOAS from this repo (`./quick_install.sh` with
  `-DPTOAS_ENABLE_WERROR=OFF` if needed for older gcc). LLVM/MLIR comes from
  the docker image. Install the produced wheel so `ptoas` on `PATH` is the
  wheel launcher (not only `build/tools/ptoas`).
- `torch_npu` for cannsim host launch

Push remote hint: https://github.com/learning-chip/PTOAS.git

## Run

From this directory (`docs/repro/store_pad8/`):

```bash
source scripts/env.sh

# Four-way: CCE + VMI pad8 / mask1 / group1
STORE_PAD8_CASE=large ./scripts/run_three_way.sh
STORE_PAD8_CASE=small ./scripts/run_three_way.sh

# Or one backend
STORE_PAD8_CASE=large ./scripts/run_cannsim.sh cce
STORE_PAD8_VARIANT=group1 STORE_PAD8_CASE=large \
  ./scripts/run_sim.sh test/test_store_pad8.py sim_outputs/store_pad8_large_vmi_group1

./scripts/check_desired.sh
```

Correctness only (no cannsim):

```bash
source scripts/env.sh
TLVF_VMI_BACKEND=cce STORE_PAD8_CASE=large python3 test/test_store_pad8.py
TLVF_VMI_BACKEND=vmi STORE_PAD8_VARIANT=group1 STORE_PAD8_CASE=large python3 test/test_store_pad8.py
```

## Measured RVEC (Ascend950 cannsim, 2026-07-29)

Built PTOAS from this branch (wheel `ptoas 0.52`). All paths PASS
(`maxDiff≈1e-6`).

### N_ACC=20 (LARGE)

| Backend | RVEC span | vs CCE | Top ops (approx.) |
|---------|----------:|-------:|-------------------|
| CCE ONEPT | **63** | 1.00× | `RV_VLDI=20, RV_VCADD=20, RV_VSTI=20, RV_PSET=1` |
| VMI **group1** | **59** | **0.94×** | `RV_VCADD=20, RV_VSTI=20, RV_VLDI=16, RV_VLDS=4, RV_PSET=3` |
| VMI mask1 | **66** | 1.05× | `RV_VCADD=20, RV_VADD=20, RV_VSTS=17, RV_VLDI=16, RV_VLDS=4` |
| VMI pad8 | **99** | 1.57× | `RV_VCADD=20, RV_VADD=20, RV_VDUP=20, RV_VSTI=20, RV_VLDI=16` |

### N_ACC=4 (SMALL)

| Backend | RVEC span | vs CCE | Top ops (approx.) |
|---------|----------:|-------:|-------------------|
| CCE ONEPT | **38** | 1.00× | `RV_VLDI=4, RV_VCADD=4, RV_VSTI=4, RV_PSET=1` |
| VMI **group1** | **43** | 1.13× | `RV_VLDI=4, RV_VCADD=4, RV_VSTI=4, RV_PSET=3, RV_VDUP=1` |
| VMI mask1 | **43** | 1.13× | `RV_VLDI=4, RV_VCADD=4, RV_VADD=4, RV_PSET=3, RV_VSTS=3` |
| VMI pad8 | **53** | 1.39× | `RV_PSET=4, RV_VLDI=4, RV_VCADD=4, RV_VADD=4, RV_VDUP=4, RV_VSTI=4` |

**Read:** `vcadd(group=1)` + `vstore(group=1, stride=1)` is the VMI form that
gets ONEPT/`1PT_B32` lowering. At N_ACC=20 it matches (slightly beats) CCE
RVEC. Pad-8 and mask1 remain useful as slower baselines. Older PyPI
`ptoas 0.51` without a built tree overstated the pad-8 tax (~281 RVEC).

## Done when

Product kernels use the group1 / compact-score store (or equivalent ONEPT
lowering) instead of pad-8. This repro already shows the fast path on the
built pin above.

## Pin

- PTOAS: this branch `zjw/store_pad_issue` (from tag **vmi-v0.1.3**), built +
  wheel-installed (`ptoas 0.52` in the measured run)
- `PTOAS_ROOT` defaults to the repo root (`../../..` from this repro directory)
