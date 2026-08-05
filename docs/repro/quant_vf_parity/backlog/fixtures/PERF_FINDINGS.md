# Backlog — former feature request 3 (solved; not a blocker)

**Issue solved. Not a blocker now. Kept as backlog** for regression and history.

FP32 → e4m3 block quantize with two on-chip faces (tile 32×512).
AscendC: `reference_asc_fp32_block_quant.asc` + ctypes
`fp32_block_quant_artifact/libfp32_block_quant.so` (`call_fp32_block_quant`).
VMI: `current_vmi_fp32_block_quant_{M}x{N}.ptodsl.py` (known-best: pair /
abs-max / quantize rows all Python `range()` → runtime `scf.for`).

Measured 2026-08-05, device 1, 72 vector cores, CANN 9.1.0-beta.3.
Host: [`../scripts/bench_fp32_block_quant.py`](../scripts/bench_fp32_block_quant.py)
(`--timer event`).
Profiler: `msprof op --aic-metrics=Default,PipeUtilization --kernel-name=…`.

## Wall-clock (host timer)

Ratio = AscendC_us / VMI_us (≥1 means VMI faster). Host NPU Event under-reads
absolute µs; use msopprof for absolute times.

| Shape | AscendC us (host) | VMI us (host) | AscendC/VMI | Correctness |
|-------|------------------:|--------------:|------------:|-------------|
| 512×2048 | 17.2 | 17.5 | **0.98** | sf/out mismatch = 0 |
| 8192×2048 | 18.1 | 19.8 | **0.92** | sf/out mismatch = 0 |

## msopprof Task Duration (8192×2048)

| Side | Kernel name | Task Duration (us) |
|------|-------------|-------------------:|
| AscendC | `fp32_block_quant_kernel` | **27.12** |
| VMI | `fp32_block_quant_8192x2048_mix_aiv` | **27.66** |

AscendC/VMI = **0.98**. Serial abs-max closed the prior ~0.88–0.90 gap (unrolled
abs-max VMI was ~30.5–31 µs).

## Product zero-copy cross-check (Table B stack)

| Shape | AscendC us | VMI us | AscendC/VMI |
|-------|-----------:|-------:|------------:|
| 512×2048 | 3.7 | 3.7 | **1.00** |
| 8192×2048 | 27.2 | 27.8 | **0.98** |

## PipeUtilization (mean over 72 cores, 8192×2048)

| Metric | AscendC | VMI | Note |
|--------|--------:|----:|------|
| aiv_time (us) | 25.1 | 25.6 | Near parity on core |
| aiv_vec_ratio | 0.338 | **0.464** | VMI still denser in vector window |
| aiv_mte2_ratio | 0.939 | 0.926 | Both heavily overlap GM→UB copy |
| aiv_mte3_ratio | **0.285** | 0.193 | AscendC still overlaps store more |
| aiv_scalar_ratio | 0.037 | 0.033 | Similar after serial abs-max |

## Insights

1. **Abs-max loop kind was the wall-clock limiter.** Explicit unroll of the
   32-row abs-max strip cost ~11% vs serial `range`/`scf.for` at 8192×2048.
   AscendC already used serial row loops.

2. **`range` ≠ `pto.static_range` here.** `static_range` unrolls at AST-rewrite
   time; over-applying it recreates a huge slow IR. Known-best keeps Python
   `range()` for pair / abs-max / quantize.

3. **Residual (small).** msopprof still shows higher VMI `aiv_vec_ratio` and
   lower `aiv_mte3_ratio` while finishing within ~2% of AscendC — emit/pipe
   polish, not a functional bug (mismatches = 0).

## How to reproduce

```bash
# From package root (docs/repro/quant_vf_parity/)
PTOAS_ROOT=/path/to/PTOAS NPU_DEVICE=1 ./backlog/scripts/run_fp32_block_quant_device.sh
./backlog/scripts/run_msopprof_fp32_block_quant.sh both
```
