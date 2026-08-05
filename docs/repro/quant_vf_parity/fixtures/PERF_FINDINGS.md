# Feature request 3 — on-device performance findings

FP32 → e4m3 block quantize with two on-chip faces (tile 32×512).
AscendC: `reference_asc_fp32_block_quant.asc` + `fp32_block_quant_artifact/executable.so`.
VMI: `current_vmi_fp32_block_quant_{M}x{N}.ptodsl.py`.

Measured 2026-08-05, device 1, 72 vector cores, CANN 9.1.0-beta.3.
Host: [`scripts/bench_fp32_block_quant.py`](../scripts/bench_fp32_block_quant.py).
Profiler: `msprof op --aic-metrics=Default,PipeUtilization --kernel-name=…`.

## Wall-clock (host timer)

Ratio = AscendC_us / VMI_us (≥1 means VMI faster). Host ffts parse can under-read
absolute µs; use msopprof Task Duration for absolute times.

| Shape | AscendC us (host) | VMI us (host) | AscendC/VMI | Correctness |
|-------|------------------:|--------------:|------------:|-------------|
| 512×2048 | 3.1 | 3.2 | **0.97** | sf/out mismatch = 0 |
| 8192×2048 | 14.6 | 17.2 | **0.85** | sf/out mismatch = 0 |

## msopprof Task Duration (8192×2048)

| Side | Kernel name | Task Duration (us) |
|------|-------------|-------------------:|
| AscendC | symbol in prebuilt `.so` (filter: `per_block_cast_kernel`) | **26.97** |
| VMI | `fp32_block_quant_8192x2048_mix_aiv` | **31.52** |

AscendC/VMI = **0.86**. Matches the known large-shape gap (~0.88 in product benches).

## PipeUtilization (mean over 72 cores, 8192×2048)

| Metric | AscendC | VMI | Note |
|--------|--------:|----:|------|
| aiv_time (us) | 25.3 | 28.0 | VMI longer on core |
| aiv_vec_ratio | 0.335 | **0.465** | VMI spends more of the window in vector |
| aiv_mte2_ratio | 0.938 | 0.883 | Both heavily overlap GM→UB copy |
| aiv_mte3_ratio | **0.280** | 0.132 | AscendC stores overlap more with compute |
| aiv_scalar_ratio | 0.037 | **0.085** | VMI more scalar tax |

Both runs report: *MTE2 bandwidth utilization lower than 80% when active.*

## MemoryUB (mean, 8192×2048)

| Metric | AscendC | VMI |
|--------|--------:|----:|
| UB read BW (vector) GB/s | 46.1 | **63.4** |
| UB write BW (vector) GB/s | 36.5 | 32.9 |
| UB write BW (GM) GB/s | 34.3 | 31.0 |

## Insights

1. **Same algorithm, large-shape gap.** At 512×2048 both sides are within noise (~0.97). At 8192×2048 VMI is ~14–17% slower (0.85–0.86×). This is not a functional bug (mismatches = 0).

2. **VMI does more vector and scalar work per core-time.** Higher `aiv_vec_ratio` and `aiv_scalar_ratio`, plus higher UB read bandwidth from the vector unit, fit a denser AscendC MicroAPI strip (paired abs-max / scale store) versus VMI’s dual `vcmax(group=1)` + `vsel` recip expand + per-row reload.

3. **AscendC overlaps store (MTE3) better.** Roughly 2× higher `aiv_mte3_ratio` while finishing sooner — copy/store pipeline is healthier on the AscendC double-buffer emit.

4. **Not a missing legalization of `group=1`.** The one-element reduce/store path already compiles (see strip fixtures). The residual is emit / schedule density for the full double-buffered kernel.

5. **What a PTOAS fix should target.** Reduce redundant UB reloads / scalar around the 64-lane strip, and/or improve MTE3 overlap of the VMI double-buffer emit so VMI’s pipe picture looks closer to AscendC at 8192×2048.

## How to reproduce

```bash
# Wall-clock + correctness
PTOAS_ROOT=/path/to/PTOAS NPU_DEVICE=1 ./scripts/run_fp32_block_quant_device.sh

# msopprof (filter out torch.randn with --kernel-name)
./scripts/run_msopprof_fp32_block_quant.sh both
```
