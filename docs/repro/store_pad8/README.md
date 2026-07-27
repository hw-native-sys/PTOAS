# store_pad8 — VMI pad-8 reduce store repro

Self-contained cannsim microbench for PTOAS gap 03: after a cross-lane
`vcadd`, CCE stores one f32 scalar per group with `vsts ... ONEPT_B32`.
VMI on tag **vmi-v0.1.3** has no working scalar store, so kernels broadcast
the reduced value to 8 lanes (`vbrc`) and store under an 8-wide mask — extra
vector work and UB traffic on every accumulator row.

This repro isolates that epilogue pattern. Fixing it lets VMI match CCE on
small-tile paths (e.g. MHC post_bwd with small tiles). Large-tile main loops
may still need separate follow-up work.

## What breaks

| Path | Store shape | Notes |
|------|-------------|-------|
| CCE (`cce/`) | ONEPT / 1 scalar | Baseline — see `fixtures/reference_asc_cce.asc` |
| VMI current (`vmi/`) | pad-8 + mask8 | Workaround used in production kernels |
| VMI desired (`fixtures/desired_vmi.*`) | mask-1 after `vcadd` | Lowers in IR but kernels still pad in practice |
| VMI target MI (`fixtures/target_mi.pto`) | `dist = "1PT_B32"` | ptoas LLVM IR OK; **bisheng crashes** on emitted HIVM |

Run `./scripts/check_desired.sh` to record compile results for desired/target
fixtures.

## Layout

```
store_pad8/
  cce/          CCE ONEPT kernel + ctypes launcher
  vmi/          VMI pad-8 kernel (ptodsl)
  common/       torch runtime, golden ref, build helper, top launcher
  fixtures/     IR dumps and reference ASC from gap 03
  test/         correctness test
  scripts/      env, cannsim runners, check_desired
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

# Correctness + RVEC (default N_ACC=20 LARGE)
STORE_PAD8_CASE=large ./scripts/run_cannsim.sh cce
STORE_PAD8_CASE=large ./scripts/run_cannsim.sh vmi

# Optional smoke (N_ACC=4)
STORE_PAD8_CASE=small ./scripts/run_cannsim.sh cce
STORE_PAD8_CASE=small ./scripts/run_cannsim.sh vmi

# Record desired/target compile attempts
./scripts/check_desired.sh
```

Correctness only (no cannsim):

```bash
source scripts/env.sh
TLVF_VMI_BACKEND=cce STORE_PAD8_CASE=large python3 test/test_store_pad8.py
TLVF_VMI_BACKEND=vmi STORE_PAD8_CASE=large python3 test/test_store_pad8.py
```

## Expected RVEC (N_ACC=20, Ascend950)

| Backend | RVEC span | Top ops (approx.) |
|---------|----------:|-------------------|
| CCE | **63** | `RV_VLDI=20, RV_VCADD=20, RV_VSTI=20, RV_PSET=1` |
| VMI | **281** | `RV_VCADD=20, RV_VADD=20, RV_VDUP=20, RV_VSTI=20, RV_VLDI=16, RV_PSET=4` |

Ratio **4.46×** (281 / 63) — almost entirely from the pad-8 broadcast+store per accumulator.

Measured on Ascend950 cannsim (2026-07-27): both backends PASS correctness for N_ACC=20 (`maxDiff≈1.43e-06`).

## Done when

VMI can emit and run `vcadd` + ONEPT/1PT scalar store without pad-8 — same
instruction shape as CCE and `fixtures/target_mi.pto`. Until bisheng accepts
the 1PT HIVM path, production kernels keep the pad-8 workaround.

## Pin

- PTOAS tag: **vmi-v0.1.3**
- Branch: `zjw/store_pad_issue`
- `PTOAS_ROOT` defaults to the repo root (`../../..` from this repro directory).
