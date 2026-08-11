# Dequantize/reduce/requantize needs a fused lowering

This standalone A5 package contains complete GM-to-UB-to-GM kernels for an FP8
requantization chain: load FP8 and grouped input scales, widen and dequantize,
compute new group maxima, rescale, convert back to FP8, and store FP8 plus the
new scales. `fixtures/requant_vmi.pto` is stock PTOAS/VMI input and
`fixtures/reference_device.asc` with `reference_launch.cpp` is a direct A5
CCE peer with a minimal ctypes stream ABI.

Use `bash check.sh compile`, `bash check.sh benchmark`, or `bash check.sh`.
With `ACL_DEVICE_ID`, benchmark mode builds and launches both sides through
`torch_npu`. A verified device-0 run uses a common 128x7168 output extent,
passes both host goldens, and measures 32.960 us CCE versus 50553.814 us VMI.

| Verified device-0 event median | CCE us | VMI us | CCE/VMI |
|---|---:|---:|---:|
| 128x7168 FP8 requant work extent | 32.960 | 50553.814 | 0.0007 |

The compact VMI body is tiled across the full output extent. Some runs also
expose a PTOAS device-LLVM frontend crash; a successful compile gives the live
timing above, while the CCE baseline remains independently launchable.

The requested lowering must first compile this f8-to-f32 broadcast/reduce/f8
sequence, then keep input-scale broadcast and output amax reduction in
registers without an intermediate UB round trip.
