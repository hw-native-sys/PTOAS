# Dequantize/reduce/requantize needs a fused lowering

This standalone A5 package contains complete GM-to-UB-to-GM kernels for an FP8
requantization chain: load FP8 and grouped input scales, widen and dequantize,
compute new group maxima, rescale, convert back to FP8, and store FP8 plus the
new scales. `fixtures/requant_vmi.pto` is stock PTOAS/VMI input and
`fixtures/reference_device.asc` with `reference_launch.cpp` is a direct A5
CCE peer with a minimal ctypes stream ABI.

Use `bash check.sh compile`, `bash check.sh benchmark`, or `bash check.sh`.
With `ACL_DEVICE_ID`, benchmark mode builds and launches both sides through
`torch_npu`. The fair control enqueues 64 identical 256-value launches between each event pair, synchronizes once, and reports per-launch medians for both sides.

| Verified device-0 event median | CCE us | VMI us | CCE/VMI |
|---|---:|---:|---:|
| one 256-value FP8 requant work item (64-launch batch) | 14.682 | 14.502 | 1.0124 |

The compact VMI body is a low-level work-item control; earlier full-shape numbers were invalid Python launch-loop timings. Some runs also expose a PTOAS device-LLVM frontend crash.

The requested lowering must first compile this f8-to-f32 broadcast/reduce/f8
sequence, then keep input-scale broadcast and output amax reduction in
registers without an intermediate UB round trip.
