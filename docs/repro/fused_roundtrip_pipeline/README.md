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
quantization step.  It also emits both batched events and raw profiler records.
The profiler record parser is intentionally marked `msprof_debug_reps`, rather
than an acceptance result: the currently observed CCE record is implausibly
short for the full tensor traffic and must be reconciled against the complete
kernel timeline before publishing a ratio.  This guard prevents the report
from repeating the earlier launch-floor error in the opposite direction.
