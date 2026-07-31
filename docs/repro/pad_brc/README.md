# Compact recip expand — pad-32 + `brc` vs in-register `vbrc`

Quantize kernels need to expand a few per-strip reciprocal / inverse values
across a wide vector strip. AscendC keeps those values in VF locals. VMI today
works only by storing into a **32-wide padded** UB slot and reloading with
`dist_mode="brc"`. A compact in-register `vbrc {group=G}` expand does not
legalize.

This package is a self-contained feature request and compile repro for PTOAS.
Branch: `pad_brc`. Details: [`FEATURE_REQUEST.md`](FEATURE_REQUEST.md).

## Algorithm (representative)

**G=4 f32 scalars → expand across 128 lanes.**  
G=8 bf16 → 256 lanes is the same pattern (same pad-32 + `brc` tax).

| Fixture | Role | Expected on current best build |
|---------|------|-------------------------|
| `fixtures/reference_asc_cce.asc` | Working AscendC baseline (reg-local inverse) | **PASS** (`bisheng`) |
| `fixtures/current_pad_brc_vmi.pto` | Working VMI: pad-32 store + `brc` reload | **PASS** (VMI→VPTO) |
| `fixtures/desired_compact_vmi.pto` | Desired VMI: in-register `vbrc {group=4}` | **FAIL** (`group_broadcast` unsupported) |

DSL twins: `current_pad_brc_vmi.py`, `desired_compact_vmi.py`.

## Layout

```text
pad_brc/
  FEATURE_REQUEST.md
  README.md
  fixtures/
  scripts/
  outputs/          # produced by check script (gitignored)
```

## Prerequisites

- CANN toolkit with `bisheng` and Ascend headers (`ASCEND_HOME_PATH`)
- This PTOAS tree built so `pto-test-opt` and `ptoas` are on `PATH`
  (or set `PTOAS_ROOT` to a built checkout)

## Run

From this directory (`docs/repro/pad_brc/`):

```bash
source scripts/env.sh
./scripts/check_pad_brc.sh
```

Results: `outputs/check_pad_brc/compile_results.txt`.

## Recorded check (example)

On a built PTOAS tree with `pto-test-opt` + CANN `bisheng`:

| Fixture | Result |
|---------|--------|
| `current_pad_brc_vmi.pto` | PASS |
| `desired_compact_vmi.pto` | FAIL — `VMI-UNSUPPORTED` / `group_broadcast` |
| `reference_asc_cce.asc` | PASS |

## How to read the result

- If **desired** compiles successfully, the criteria in
  [`FEATURE_REQUEST.md`](FEATURE_REQUEST.md) are met.
- **Current** pad+`brc` should keep working until kernels can move to the
  compact shape.
