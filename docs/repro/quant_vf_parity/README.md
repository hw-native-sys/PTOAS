# Remaining quant gaps: AscendC vs VMI (after PR #1145)

On top of [PR #1145](https://github.com/hw-native-sys/PTOAS/pull/1145)
(bf16 abs and group-4 broadcast), **two issues remain open**. A third
(wall-clock on FP32 block-quant) is **solved — not a blocker now** and lives
under [`backlog/`](backlog/). Full write-up:
[`FEATURE_REQUEST.md`](FEATURE_REQUEST.md).

## Status

| Priority | Request | Status |
|----------|---------|--------|
| **Open — primary** | 1 — `get_block_idx` bound cannot control a vector body | **Blocker** (emit expand fails) |
| **Open** | 2 — wide double-buffered dequant ≠ AscendC | **Still open** (numeric mismatch) |
| Backlog | former 3 — FP32 double-buffered block-quant µs | **Solved; not a blocker.** See [`backlog/`](backlog/). |

## Fixture map (`fixtures/` — open blockers only)

| Fixture | Request | Role | Expected today |
|---------|---------|------|----------------|
| `reference_asc_block_idx_vf.asc` | **1** | AscendC: block index decides if vector work runs | PASS (`bisheng`) |
| `current_vmi_block_idx_vf.pto` | **1** | Smallest VMI that still hits the expand error | FAIL ptoas→bisheng expand |
| `current_vmi_multibuf_expand_full.pto` | **1** | Larger dump of the same failure | FAIL expand (same class) |
| `reference_asc_dequant_dblbuf.asc` | **2** | AscendC dequant vector body | PASS (`bisheng`) |
| `broken_vmi_dequant_dblbuf.pto` | **2** | Wide-strip VMI vector body | PASS lower; on-device mismatch recorded |
| `working_vmi_dequant_narrow.pto` | **2** | Narrow-strip VMI vector body | PASS lower; matched AscendC on device |
| `broken_vmi_dequant_dblbuf.ptodsl.py` | **2** | Full wide double-buffer dump | on-device mismatch vs AscendC |
| `working_vmi_dequant_narrow.ptodsl.py` | **2** | Narrow control dump | matched AscendC on device |

Device notes: [`fixtures/RECORDED_DEVICE_NOTES.md`](fixtures/RECORDED_DEVICE_NOTES.md).

## Layout

```text
quant_vf_parity/
  FEATURE_REQUEST.md
  README.md
  fixtures/           # Ask 1 + Ask 2 only
  scripts/            # env + compile check for open asks
  backlog/            # solved FR3 — not a blocker; fixtures + device scripts
  historical/         # older provenance (not entrypoints)
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

Optional backlog (solved wall gap — not required for open asks):

```bash
PTOAS_ROOT=/path/to/PTOAS NPU_DEVICE=1 ./backlog/scripts/run_fp32_block_quant_device.sh
./backlog/scripts/run_msopprof_fp32_block_quant.sh both
```

## How to read the result

- **Feature request 1 (primary open):** AscendC PASS and VMI emit FAIL with the
  expand error means the request is still open — this is the main blocker.
- **Feature request 2 (open):** Both vector bodies should lower. The wide
  schedule’s on-device mismatch vs AscendC is in `RECORDED_DEVICE_NOTES.md`.
- **Former request 3:** Issue solved; not a blocker now. Parked under
  `backlog/` — see `backlog/fixtures/PERF_FINDINGS.md`.
