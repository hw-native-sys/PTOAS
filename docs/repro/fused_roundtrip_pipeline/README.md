# Fused grouped quantize/dequantize: fair one-launch control

> **Scope warning:** this public fixture is a 256-value low-level control, not
> a faithful full-shape performance reproduction. The private study's gap
> comes from persistent multi-core tiling and must be reproduced with that
> schedule before using this report as performance evidence.

The reproducer quantizes a BF16 tile to FP8 using eight group scales and
immediately dequantizes it. `fixtures/fused_roundtrip_vmi.pto` includes GM DMA,
group reduction, scale expansion, FP8 conversion, reverse scaling, and output
DMA. `fixtures/reference_cce.cpp` is the complete direct CCE control.

Use `bash check.sh compile`, `bash check.sh benchmark`, or `bash check.sh`.
Set `ACL_DEVICE_ID` to run the live PTO/VMI launch and exact
one-value round-trip check through `torch_npu`.

| Verified device-0 event median | CCE us | VMI us | CCE/VMI |
|---|---:|---:|---:|
| one 256-value BF16 round trip | 32.661 | 32.875 | 0.9935 |

The live control uses one 256-value work item and one device launch per side.
Earlier multi-row figures were invalid host-loop timings. Absolute values vary
with load; the threshold is `CCE_us / VMI_us >= 0.98`. Requested behavior is to
keep maxima, scales, packed FP8 values, and predicates live across the entire
round trip without a store/reload or layout materialization.
