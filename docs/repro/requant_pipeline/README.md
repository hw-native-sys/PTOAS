# Dequantize/reduce/requantize needs a fused lowering

> **Scope:** this package retains the production-shaped `128x7168` schedule,
> 72-core launch grid, and double-buffered CCE/VMI device bodies.

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
| production-shaped FP8 requant (`128x7168`, 72 cores) | 4.974 | 8.251 | 0.6028 |

The measured ratio is in the same range as the private production regression;
the CCE body is the reachable fast control, while the equivalent VMI body is
slower under the same device-only timing method.

The requested lowering must first compile this f8-to-f32 broadcast/reduce/f8
sequence, then keep input-scale broadcast and output amax reduction in
registers without an intermediate UB round trip.
