# FP8 rescale: CCE versus composed VMI

This standalone A5 benchmark retains the complete production-sized
`8064 x 7168` operation.  Both paths read packed FP8 with packed per-32-value
input scales and write FP8 with per-32-row float output scales.

The CCE control is one 72-core GM-to-UB-to-GM kernel.  The equivalent VMI path
uses its actual two-stage composition: eight 1024-row, 72-core unpack launches
followed by one full-tensor 72-core requant launch.  The host code is ctypes
only; no framework launcher is required.

Run with the one device selector:

```bash
conda activate cann91_dev
source /home/jzhuang/cann_installed/9.1.0-beta.3/cann-9.1.0-beta.3/set_env.sh
ACL_DEVICE_ID=0 bash check.sh compile
ACL_DEVICE_ID=0 bash check.sh benchmark
```

The benchmark checks CCE/VMI output and scale agreement before timing.  The
retained padded final strip currently reports `VMI_TAIL_MISMATCH` for rows
`7168:8064`; the first `7168` rows and their scales agree with CCE. This is a
functional blocker in the same VMI composition, not a reason to omit the tail
from the workload.  It then reports a cold-L2 event
median for both full operations and the primary device-only FFTS result. FFTS
uses the same AIC-preferred record rule as the production benchmark and sums
both VMI stages, so Python/ctypes launch time is not mistaken for kernel time.

| Verified device-0 measurement | CCE us | VMI us | CCE/VMI |
|---|---:|---:|---:|
| Cold-L2 event median, full operation | 46.933 | 176.122 | 0.2665 |
| Device-only FFTS, full operation | 45.255 | 141.459 | 0.3199 |

The FFTS result is the report's primary ratio: it excludes Python/ctypes launch
time and sums every device operation in the isolated CCE or VMI callable. This
includes the VMI composition's pad/copy device operations, which are part of
its observed end-to-end penalty.
