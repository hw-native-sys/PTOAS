# FP8 rescale: CCE versus composed VMI

This standalone A5 benchmark retains the complete production-sized
`8064 x 7168` operation.  Both paths read packed FP8 with packed per-32-value
input scales and write FP8 with per-32-row float output scales.

The CCE control is one 72-core GM-to-UB-to-GM kernel.  The equivalent VMI path
uses its actual two-stage composition: eight 1024-row, 72-core unpack launches
followed by one full-tensor 72-core requant launch.  The host code is ctypes
only; no framework launcher is required.

Requirements: CANN `9.1.0-beta.3`, the pinned PTODSL/PTOAS installation, and
PyTorch with NPU support. Source that CANN release first, or set `CANN_ENV` to
its `set_env.sh`; set `CONDA_ENV` only when the active environment is not the
default `cann91_dev`.

Run with one device selector:

```bash
source /path/to/cann-9.1.0-beta.3/set_env.sh
ACL_DEVICE_ID=0 bash check.sh compile
ACL_DEVICE_ID=0 bash check.sh benchmark
```

The benchmark checks CCE/VMI output and scale agreement over the defined
`8064`-row extent before timing. The final 896-row strip retains the same
static 1024-row launch as the other VMI strips; its inactive rows are
zero-padded and excluded from the public result. It then reports a cold-L2
event median for both full operations and the primary device-only FFTS result.
FFTS
uses the same AIC-preferred record rule as the production benchmark and sums
both VMI stages, so Python/ctypes launch time is not mistaken for kernel time.

| Verified device-0 measurement | CCE us | VMI us | CCE/VMI |
|---|---:|---:|---:|
| Cold-L2 event median, full operation | 47.041 | 302.386 | 0.1556 |
| Device-only FFTS, full operation | 45.430 | 145.072 | 0.3132 |

The FFTS result is the report's primary ratio: it excludes Python/ctypes launch
time and sums every device operation in the isolated CCE or VMI callable. This
includes the VMI composition's pad/copy device operations, which are part of
its observed end-to-end penalty.
