# Remaining quant gaps: AscendC vs VMI (after PR #1145)

On top of [PR #1145](https://github.com/hw-native-sys/PTOAS/pull/1145)
(bf16 abs and group-4 broadcast), three issues remain. This package states each
one as a working AscendC program next to a nearly equivalent VMI/PTO program
that fails to emit, gets wrong results, or is slower. Full write-up:
[`FEATURE_REQUEST.md`](FEATURE_REQUEST.md).

## Fixture map

| Fixture | Feature request | Role | Expected today |
|---------|-----------------|------|----------------|
| `reference_asc_block_idx_vf.asc` | 1 | AscendC: block index decides if vector work runs | PASS (`bisheng`) |
| `current_vmi_block_idx_vf.pto` | 1 | Smallest VMI that still hits the expand error | FAIL ptoas→bisheng expand |
| `current_vmi_multibuf_expand_full.pto` | 1 | Larger dump of the same failure | FAIL expand (same class) |
| `reference_asc_dequant_dblbuf.asc` | 2 | AscendC dequant vector body | PASS (`bisheng`) |
| `broken_vmi_dequant_dblbuf.pto` | 2 | Wide-strip VMI vector body | PASS lower; on-device mismatch recorded |
| `working_vmi_dequant_narrow.pto` | 2 | Narrow-strip VMI vector body | PASS lower; matched AscendC on device |
| `reference_asc_fp32_strip_amax.asc` | 3 | AscendC FP32 strip abs-max | PASS (`bisheng`) |
| `current_vmi_fp32_strip_amax.pto` | 3 | VMI FP32 abs + `vcmax` (`group = 1`) | PASS lower |

More detail: [`fixtures/RECORDED_DEVICE_NOTES.md`](fixtures/RECORDED_DEVICE_NOTES.md),
[`fixtures/emit_compare_note.md`](fixtures/emit_compare_note.md).

## Layout

```text
quant_vf_parity/
  FEATURE_REQUEST.md
  README.md
  PINS.md
  fixtures/
  scripts/            # env + check_vmi_asc_residual
  historical/         # older AscendC/VMI fixtures (not entrypoints)
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

- **Feature request 1:** AscendC PASS and VMI emit FAIL with the expand error
  means the request is still open.
- **Feature request 2:** Both vector bodies should lower. The wide schedule’s
  on-device mismatch vs AscendC is recorded in `RECORDED_DEVICE_NOTES.md`.
- **Feature request 3:** Lower PASS only shows the VMI program is legal. Closing
  this as a PTOAS change still needs an AscendC-vs-VMI emit comparison in
  `emit_compare_note.md` (see FEATURE_REQUEST.md).
