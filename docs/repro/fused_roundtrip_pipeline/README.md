# Fused grouped quantize/dequantize is slower than direct CCE

The reproducer quantizes a BF16 tile to FP8 using eight group scales and
immediately dequantizes it. `fixtures/fused_roundtrip_vmi.pto` includes GM DMA,
group reduction, scale expansion, FP8 conversion, reverse scaling, and output
DMA. `fixtures/reference_cce.cpp` is the complete direct CCE control.

Use `bash check.sh compile`, `bash check.sh benchmark`, or `bash check.sh`.

| Representative case | ASC us | VMI us | ASC/VMI |
|---|---:|---:|---:|
| small FP32 | 8.3281 | 8.7894 | 0.9475 |
| large BF16 | 22.0386 | 27.3305 | 0.8064 |
| wide BF16 | 7.2110 | 8.0451 | 0.8963 |

These are pinned A5 event-timed medians. Absolute values vary with load; the
acceptance threshold is `ASC_us / VMI_us >= 0.98`. Requested behavior is to
keep maxima, scales, packed FP8 values, and predicates live across the entire
round trip without a store/reload or layout materialization.
