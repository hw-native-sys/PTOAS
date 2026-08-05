# Remaining quant gaps: AscendC vs VMI (after PR #1145)

On top of [PR #1145](https://github.com/hw-native-sys/PTOAS/pull/1145)
(bf16 abs and group-4 broadcast), **two issues remain open**. A third
(wall-clock on FP32 block-quant) is **solved — not a blocker now** and is kept
only as backlog. Full write-up: [`FEATURE_REQUEST.md`](FEATURE_REQUEST.md).

## Status

| Priority | Request | Status |
|----------|---------|--------|
| **Open — primary** | 1 — `get_block_idx` bound cannot control a vector body | **Blocker** (emit expand fails) |
| **Open** | 2 — wide double-buffered dequant ≠ AscendC | **Still open** (numeric mismatch) |
| Backlog | former 3 — FP32 double-buffered block-quant µs | **Solved; not a blocker. Kept as backlog.** |

## Fixture map

| Fixture | Request | Role | Expected today |
|---------|---------|------|----------------|
| `reference_asc_block_idx_vf.asc` | **1 (open)** | AscendC: block index decides if vector work runs | PASS (`bisheng`) |
| `current_vmi_block_idx_vf.pto` | **1 (open)** | Smallest VMI that still hits the expand error | FAIL ptoas→bisheng expand |
| `current_vmi_multibuf_expand_full.pto` | **1 (open)** | Larger dump of the same failure | FAIL expand (same class) |
| `reference_asc_dequant_dblbuf.asc` | **2 (open)** | AscendC dequant vector body | PASS (`bisheng`) |
| `broken_vmi_dequant_dblbuf.pto` | **2 (open)** | Wide-strip VMI vector body | PASS lower; on-device mismatch recorded |
| `working_vmi_dequant_narrow.pto` | **2 (open)** | Narrow-strip VMI vector body | PASS lower; matched AscendC on device |
| `reference_asc_fp32_block_quant.asc` | backlog | AscendC full FP32 block-quant | PASS; wall ~parity — **not a blocker** |
| `current_vmi_fp32_block_quant_*.ptodsl.py` | backlog | VMI full kernel (serial abs-max) | Compiles; ~0.98× vs AscendC @8192 |
| `fp32_block_quant_artifact/libfp32_block_quant.so` | backlog | AscendC ctypes launch lib | Built by `build_fp32_block_quant_asc.sh` |
| `reference_asc_fp32_strip_amax.asc` | backlog | AscendC strip abs-max (VF fragment only) | PASS (`bisheng`) |
| `current_vmi_fp32_strip_amax.pto` | backlog | VMI strip abs + `vcmax(group=1)` (fragment) | PASS lower |

Open Ask 2 device notes:
[`fixtures/RECORDED_DEVICE_NOTES.md`](fixtures/RECORDED_DEVICE_NOTES.md).

Backlog FP32 perf table (solved wall gap):
[`fixtures/PERF_FINDINGS.md`](fixtures/PERF_FINDINGS.md).
Strip emit note (optional): [`fixtures/emit_compare_note.md`](fixtures/emit_compare_note.md).

## Layout

```text
quant_vf_parity/
  FEATURE_REQUEST.md
  README.md
  PINS.md
  fixtures/
  scripts/            # env, check, fp32 block-quant device + msopprof (backlog)
  historical/         # older AscendC/VMI fixtures (not entrypoints)
  outputs/            # gitignored
```

## Prerequisites

- CANN toolkit with `bisheng` (`ASCEND_HOME_PATH`)
- This PTOAS tree built (`pto-test-opt` / `ptoas` on `PATH`, or `PTOAS_ROOT`)
- For optional backlog FP32 device runs: NPU + `torch` / `torch_npu` (or ACL)

## Run

From this directory (`docs/repro/quant_vf_parity/`):

```bash
source scripts/env.sh
./scripts/check_vmi_asc_residual.sh
```

Results: `outputs/check_vmi_asc_residual/compile_results.txt`.

Optional — backlog FP32 block-quant (not required to track open asks):

```bash
PTOAS_ROOT=/path/to/PTOAS NPU_DEVICE=1 ./scripts/run_fp32_block_quant_device.sh
./scripts/run_msopprof_fp32_block_quant.sh both
```

## How to read the result

- **Feature request 1 (primary open):** AscendC PASS and VMI emit FAIL with the
  expand error means the request is still open — this is the main blocker.
- **Feature request 2 (open):** Both vector bodies should lower. The wide
  schedule’s on-device mismatch vs AscendC is in `RECORDED_DEVICE_NOTES.md`.
- **Former request 3 (backlog):** Issue solved; not a blocker now. Wall-clock
  ~0.98× at 8192×2048 after serial abs-max. Fixtures kept for regression /
  history only — see `PERF_FINDINGS.md`.
