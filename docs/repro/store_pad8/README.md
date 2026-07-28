# store_pad8 — VMI pad-8 reduce store repro

Self-contained cannsim microbench for PTOAS gap 03: after a cross-lane
`vcadd`, CCE stores one f32 scalar per group with `vsts ... ONEPT_B32`.
Product VMI kernels often broadcast the reduced value to 8 lanes (`vbrc`)
and store under an 8-wide mask — extra vector work and UB traffic on every
accumulator row.

This repro isolates that epilogue pattern and splits **kernel formulation**
(pad-8 vs compact mask1) from **compiler lowering** (still not CCE-quality
ONEPT / `1PT_B32` on tag **vmi-v0.1.3**).

## What breaks

| Path | Store shape | Notes |
|------|-------------|-------|
| CCE (`cce/`) | ONEPT / 1 scalar | Baseline — see `fixtures/reference_asc_cce.asc` |
| VMI pad-8 (`vmi/store_pad8_vmi.py`) | pad-8 + mask8 | Product workaround; cannsim-runnable |
| VMI mask1 (`vmi/store_mask1_vmi.py`) | compact `[N]` + mask1 | Same math; cannsim-runnable (`STORE_PAD8_VARIANT=mask1`) |
| VMI desired (`fixtures/desired_vmi.*`) | mask-1 after `vcadd` | Compile-only single-tile fixture |
| VMI target MI (`fixtures/target_mi.pto`) | `dist = "1PT_B32"` | ptoas LLVM IR OK; **bisheng crashes** on emitted HIVM |

`mask1` shows VMI can express and run a compact 1-lane store. It is still
much slower than CCE because lowering keeps a fat `vdup`/`vadd` + masked
`vsts` path, not ONEPT. True parity needs `1PT_B32` / ONEPT through bisheng.

## Layout

```
store_pad8/
  cce/          CCE ONEPT kernel + ctypes launcher
  vmi/          VMI pad-8 and mask1 kernels (ptodsl)
  common/       torch runtime, golden ref, build helper, top launcher
  fixtures/     IR dumps and reference ASC from gap 03
  test/         correctness test
  scripts/      env, cannsim runners, check_desired, run_three_way
```

## Prerequisites

- CANN 9.0 simulator (`cannsim`, `bisheng`)
- Built PTOAS at tag **vmi-v0.1.3** (branch `zjw/store_pad_issue`)
- `torch_npu` for cannsim host launch

Push remote hint: https://github.com/learning-chip/PTOAS.git

## Run

From this directory (`docs/repro/store_pad8/`):

```bash
source scripts/env.sh

# Three-way: CCE ONEPT + VMI pad8 + VMI mask1 (default N_ACC=20)
STORE_PAD8_CASE=large ./scripts/run_three_way.sh
STORE_PAD8_CASE=small ./scripts/run_three_way.sh

# Or one backend
STORE_PAD8_CASE=large ./scripts/run_cannsim.sh cce
STORE_PAD8_CASE=large ./scripts/run_cannsim.sh vmi          # pad8
STORE_PAD8_VARIANT=mask1 STORE_PAD8_CASE=large \
  ./scripts/run_sim.sh test/test_store_pad8.py sim_outputs/store_pad8_large_vmi_mask1

# Record desired/target compile attempts
./scripts/check_desired.sh
```

Correctness only (no cannsim):

```bash
source scripts/env.sh
TLVF_VMI_BACKEND=cce STORE_PAD8_CASE=large python3 test/test_store_pad8.py
TLVF_VMI_BACKEND=vmi STORE_PAD8_VARIANT=pad8 STORE_PAD8_CASE=large python3 test/test_store_pad8.py
TLVF_VMI_BACKEND=vmi STORE_PAD8_VARIANT=mask1 STORE_PAD8_CASE=large python3 test/test_store_pad8.py
```

## Measured RVEC (Ascend950 cannsim, 2026-07-28)

All three paths PASS correctness (`maxDiff≈1e-6`).

### N_ACC=20 (LARGE)

| Backend | RVEC span | vs CCE | Top ops (approx.) |
|---------|----------:|-------:|-------------------|
| CCE ONEPT | **63** | 1.00× | `RV_VLDI=20, RV_VCADD=20, RV_VSTI=20, RV_PSET=1` |
| VMI pad8 | **281** | 4.46× | `RV_VCADD=20, RV_VADD=20, RV_VDUP=20, RV_VSTI=20, RV_VLDI=16, RV_PSET=4` |
| VMI mask1 | **267** | 4.24× | `RV_VCADD=20, RV_VADD=20, RV_VSTS=17, RV_VLDI=16, RV_VLDS=4, RV_PSET=3` |

### N_ACC=4 (SMALL)

| Backend | RVEC span | vs CCE | Top ops (approx.) |
|---------|----------:|-------:|-------------------|
| CCE ONEPT | **38** | 1.00× | `RV_VLDI=4, RV_VCADD=4, RV_VSTI=4, RV_PSET=1` |
| VMI pad8 | **83** | 2.18× | `RV_PSET=4, RV_VLDI=4, RV_VCADD=4, RV_VADD=4, RV_VDUP=4, RV_VSTI=4` |
| VMI mask1 | **75** | 1.97× | `RV_VLDI=4, RV_VCADD=4, RV_VADD=4, RV_PSET=3, RV_VSTS=3, RV_PLT=1` |

**Read:** dropping pad-8 for compact mask1 barely helps (~5% at N=20). Most of
the gap vs CCE is left in PTOAS lowering of the serial 1-lane store, not in
the `vbrc×8` formulation alone. Simulator may log `check_addr_aligned` on
some mask1 `RV_VSTS` ops; results still match golden.

## Done when

VMI can emit and run `vcadd` + ONEPT/1PT scalar store — same instruction
shape as CCE and `fixtures/target_mi.pto`. Until bisheng accepts the 1PT
HIVM path, mask1 is a useful formulation check but not a performance fix.

## Pin

- PTOAS tag: **vmi-v0.1.3**
- Branch: `zjw/store_pad_issue`
- `PTOAS_ROOT` defaults to the repo root (`../../..` from this repro directory).
