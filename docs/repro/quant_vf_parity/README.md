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
| `reference_asc_fp32_block_quant.asc` | 3 | AscendC full FP32 block-quant (primary; MicroAPI) | PASS (`bisheng`) |
| `current_vmi_fp32_block_quant_*.ptodsl.py` | 3 | VMI full kernel (primary; on-device µs) | Compiles; ~0.86× vs AscendC @8192 |
| `fp32_block_quant_artifact/` | 3 | Prebuilt AscendC `.so` for host bench | Used by device script |
| `reference_asc_fp32_strip_amax.asc` | 3 | AscendC strip abs-max (**VF fragment only**) | PASS (`bisheng`) |
| `current_vmi_fp32_strip_amax.pto` | 3 | VMI strip abs + `vcmax(group=1)` (**fragment**) | PASS lower |

Performance table and msopprof pipe insights:
[`fixtures/PERF_FINDINGS.md`](fixtures/PERF_FINDINGS.md).
Strip emit note (optional): [`fixtures/emit_compare_note.md`](fixtures/emit_compare_note.md).
Feature request 2 device notes: [`fixtures/RECORDED_DEVICE_NOTES.md`](fixtures/RECORDED_DEVICE_NOTES.md).

## Layout

```text
quant_vf_parity/
  FEATURE_REQUEST.md
  README.md
  PINS.md
  fixtures/
  scripts/            # env, check, fp32 block-quant device + msopprof
  historical/         # older AscendC/VMI fixtures (not entrypoints)
  outputs/            # gitignored
```

## Prerequisites

- CANN toolkit with `bisheng` (`ASCEND_HOME_PATH`)
- This PTOAS tree built (`pto-test-opt` / `ptoas` on `PATH`, or `PTOAS_ROOT`)
- For on-device FR3: NPU + `torch` / `torch_npu` (or ACL); no product `PYTHONPATH`

## Run

From this directory (`docs/repro/quant_vf_parity/`):

```bash
source scripts/env.sh
./scripts/check_vmi_asc_residual.sh
```

Results: `outputs/check_vmi_asc_residual/compile_results.txt`.

Full FP32 block-quant correctness + µs (when NPU available):

```bash
PTOAS_ROOT=/path/to/PTOAS NPU_DEVICE=1 ./scripts/run_fp32_block_quant_device.sh
./scripts/run_msopprof_fp32_block_quant.sh both
```

## How to read the result

- **Feature request 1:** AscendC PASS and VMI emit FAIL with the expand error
  means the request is still open.
- **Feature request 2:** Both vector bodies should lower. The wide schedule’s
  on-device mismatch vs AscendC is recorded in `RECORDED_DEVICE_NOTES.md`.
- **Feature request 3:** Primary evidence is the **full** kernel +
  `PERF_FINDINGS.md` (~0.86× at 8192×2048; near parity at 512). Strip fixtures
  only prove the reduce path compiles; they do not reproduce the gap. Closing
  this needs VMI parity at 8192 or an emit/pipe fix justified by those findings.
