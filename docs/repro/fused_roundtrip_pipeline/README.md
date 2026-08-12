# Persistent BF16 FP8 round-trip: CCE versus VMI

This is a standalone 72-core workload, not a one-tile timing control.  Both
kernels process the same contiguous BF16 tensor (`8192 x 2048`) with the same
three-stage persistent schedule and the same launch grid.  The CCE body keeps
the vector pipeline and DMA overlap explicit; the VMI body keeps the complete
group reduction, scale encoding, FP8 conversion, reverse scaling, and DMA
pipeline in [`full_roundtrip_vmi.py`](fixtures/full_roundtrip_vmi.py).

Run on CANN 9.1 with one device variable:

```bash
conda activate cann91_dev
source /home/jzhuang/cann_installed/9.1.0-beta.3/cann/set_env.sh
ACL_DEVICE_ID=0 bash check.sh compile
ACL_DEVICE_ID=0 bash check.sh benchmark
```

The benchmark first checks finite output and CCE/VMI agreement within one FP8
quantization step. It reports a cold-L2 event median as a launch-path sanity
check and uses device-only FFTS time as the accepted comparison. The profiler
aggregates the AIC records across 30 repetitions, matching the production
benchmark instead of selecting the slowest individual launch.

| Verified device-0 measurement | CCE us | VMI us | CCE/VMI |
|---|---:|---:|---:|
| Cold-L2 event median | 31.034 | 31.795 | 0.9761 |
| Device-only FFTS, 30 repetitions | 21.786 | 27.296 | 0.7981 |

The event result is launch-floor dominated and is diagnostic only. The FFTS
result reproduces the production-sized VMI performance gap with the same
72-core persistent operation and is the accepted issue measurement.
