# Fused grouped quantize–dequantize is slower than the direct sequence

## Summary

The reduced fixture quantizes BF16 data to FP8 with eight scales and immediately
dequantizes it. This is a single vector body with no application semantics. VMI
expresses the dataflow naturally, but the generated pipeline does not match the
throughput of a direct CCE implementation on several small and large grids.

## Reproduce

```bash
bash docs/repro/fused_roundtrip_pipeline/check.sh
```

For device measurement, place the body in a persistent GM/UB loop, validate the
BF16 output against `fixtures/reference_cce.cpp`, clear L2 between measurements,
and compare event-timed medians.

## Requested behavior

Keep grouped reduction, reciprocal broadcast, FP8 conversion, reverse scale
broadcast, and dequantization in registers. Avoid redundant FP8 UB store/reload,
layout materialization, and predicate setup. The target is at least 98% of the
direct CCE reference throughput for the same tiling.
