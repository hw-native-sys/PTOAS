# Dequantize/reduce/requantize needs a fused lowering

> **Scope:** this package retains the production-shaped `8064x7168` extent and
> 72-core CCE schedule.  The checked-in VMI body is one 128-row low-level tile;
> the standalone host launcher composes 63 such tiles over the same extent.

This standalone A5 package contains complete GM-to-UB-to-GM kernels for an FP8
requantization chain: load FP8 and grouped input scales, widen and dequantize,
compute new group maxima, rescale, convert back to FP8, and store FP8 plus the
new scales. `fixtures/requant_vmi.pto` is stock PTOAS/VMI input and
`fixtures/reference_device.asc` with `reference_launch.cpp` is a direct A5
CCE peer with a minimal ctypes stream ABI.

Use `bash check.sh compile`, `bash check.sh benchmark`, or `bash check.sh`.
With `ACL_DEVICE_ID`, benchmark mode builds and launches both sides through
`torch_npu`. Both paths use one 72-core launch per event interval, with identical cache
flush and synchronization policy.

| Verified device-0 event median | CCE us | VMI us | CCE/VMI |
|---|---:|---:|---:|
| production-shaped FP8 requant (`8064x7168`, CCE 72 cores; VMI 63 tiles) | pending device recovery | pending device recovery | pending |

The ratio is intentionally left pending until the large composed launch is
captured on-device.  The CCE body is the reachable fast control; the VMI value
must include all 63 tile launches before it can be compared fairly.

The requested lowering must first compile this f8-to-f32 broadcast/reduce/f8
sequence, then keep input-scale broadcast and output amax reduction in
registers without an intermediate UB round trip.
