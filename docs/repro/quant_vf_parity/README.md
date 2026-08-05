# AscendC vs VMI residuals (block-idx VF emit, dequant buffering, FP32 density)

After bf16 abs and group-4 broadcast landed on `zjw/bf16_vmi_fixes`, three
residuals remain. This package states each one as an AscendC program that works
beside a nearly equivalent VMI/PTO program that is slower, wrong, or fails to
emit. Details: [`FEATURE_REQUEST.md`](FEATURE_REQUEST.md).

## Fixture map

| Fixture | Ask | Role | Expected today |
|---------|-----|------|----------------|
| `reference_asc_block_idx_vf.asc` | 1 | AscendC block-idx guard | PASS (`bisheng`) |
| `current_vmi_block_idx_vf.pto` | 1 | Minimized block-idx → VF | FAIL ptoas→bisheng expand |
| `current_vmi_multibuf_expand_full.pto` | 1 | Full dump of the fail | FAIL expand (same class) |
| `reference_asc_dequant_dblbuf.asc` | 2 | AscendC dequant VF | PASS (`bisheng`) |
| `broken_vmi_dequant_dblbuf.pto` | 2 | Wide-strip VMI VF | PASS lower; device mismatch recorded |
| `working_vmi_dequant_narrow.pto` | 2 | Narrow-strip VMI VF | PASS lower; device matched AscendC |
| `reference_asc_fp32_strip_amax.asc` | 3 | AscendC FP32 strip amax | PASS (`bisheng`) |
| `current_vmi_fp32_strip_amax.pto` | 3 | VMI `group=1` 1PT path | PASS lower |

Notes: [`fixtures/RECORDED_DEVICE_NOTES.md`](fixtures/RECORDED_DEVICE_NOTES.md),
[`fixtures/emit_compare_note.md`](fixtures/emit_compare_note.md).

## Layout

```text
quant_vf_parity/
  FEATURE_REQUEST.md
  README.md
  PINS.md
  fixtures/
  scripts/            # env + check_vmi_asc_residual
  historical/         # old product-harness scripts (not entrypoints)
  outputs/            # gitignored
```

## Prerequisites

- CANN toolkit with `bisheng` (`ASCEND_HOME_PATH`)
- This PTOAS tree built (`pto-test-opt` / `ptoas` on `PATH`, or `PTOAS_ROOT`)

## Run

From this directory (`docs/repro/quant_vf_parity/`):

```bash
source scripts/env.sh
./scripts/check_vmi_asc_residual.sh
```

Results: `outputs/check_vmi_asc_residual/compile_results.txt`.

## How to read the result

- **Ask 1:** AscendC PASS and VMI block-idx VF emit FAIL (expand error) means the
  request is still open.
- **Ask 2:** Both VF bodies should lower; device mismatch for the wide schedule
  is documented in `RECORDED_DEVICE_NOTES.md`.
- **Ask 3:** Lower PASS alone does not close the ask; see `emit_compare_note.md`.
