# Dequantize/reduce/requantize needs a fused lowering

This standalone A5 package contains complete GM-to-UB-to-GM kernels for an FP8
requantization chain: load FP8 and grouped input scales, widen and dequantize,
compute new group maxima, rescale, convert back to FP8, and store FP8 plus the
new scales. `fixtures/requant_vmi.pto` is stock PTOAS/VMI input and
`fixtures/reference_cce.cpp` is a direct register-resident CCE peer.

Use `bash check.sh compile`, `bash check.sh benchmark`, or `bash check.sh`.
With `LIVE_DEVICE` and `ACL_DEVICE_ID`, benchmark mode builds and launches the
VMI fixture through `torch_npu` and checks its finite scale output. The compile
check verifies the reduction, multiply, conversion, and complete DMA envelope.
The table is historical pinned A5 event medians:

| Representative case | ASC us | VMI us | ASC/VMI |
|---|---:|---:|---:|
| small FP8, group 32 | 4.1403 | 7.1908 | 0.5758 |
| small FP8, group 128 | 4.4532 | 11.3552 | 0.3922 |
| large FP8, group 32 | 46.7131 | 140.8352 | 0.3317 |

Parity is `ASC_us / VMI_us >= 0.98`. The requested lowering keeps input-scale
broadcast and output amax reduction in registers, reuses predicates/layouts,
and introduces no intermediate UB round trip.
