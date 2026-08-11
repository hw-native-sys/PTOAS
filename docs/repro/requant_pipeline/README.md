# Dequantize/reduce/requantize needs a fused lowering

This standalone A5 package contains complete GM-to-UB-to-GM kernels for an FP8
requantization chain: load FP8 and grouped input scales, widen and dequantize,
compute new group maxima, rescale, convert back to FP8, and store FP8 plus the
new scales. `fixtures/requant_vmi.pto` is stock PTOAS/VMI input and
`fixtures/reference_device.asc` with `reference_launch.cpp` is a direct A5
CCE peer with a minimal ctypes stream ABI.

Use `bash check.sh compile`, `bash check.sh benchmark`, or `bash check.sh`.
With `ACL_DEVICE_ID`, benchmark mode builds and launches both sides through
`torch_npu`. The direct CCE side was verified on device 1 at 29.542 us for a
128x7168 all-zero FP8 input and passed an exact all-zero output golden.

The current VMI fixture is itself the blocker: PTOAS reaches device LLVM and
the CANN compiler frontend exits 139. This is a reproducible VMI failure, not
a missing launch dependency; the CCE baseline is still compiled, launched,
timed, and host-golden checked independently.

The requested lowering must first compile this f8-to-f32 broadcast/reduce/f8
sequence, then keep input-scale broadcast and output amax reduction in
registers without an intermediate UB round trip.
