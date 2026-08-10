# Dequantize–reduce–requantize pipeline needs a fused lowering

## Summary

This case covers a low-level conversion pipeline: FP8 values are expanded with
grouped input scales, reduced to new per-group maxima, rescaled, and converted
back to FP8. The equivalent direct CCE implementation pipelines scale decode,
vector arithmetic, reduction, and packed stores. Current VMI code is correct
but remains far slower, especially when the input and output scale formats
differ.

## Reproduce

```bash
bash docs/repro/requant_pipeline/check.sh
```

The check lowers the framework-free PTO fixture and reports the number of
vector loads/stores and barriers in the resulting VPTO. For on-device A/B, use
the same persistent loop and buffers for the VMI body and the direct sequence
outlined in `fixtures/reference_cce.cpp`.

## Requested behavior

Lower this whole chain without intermediate UB round trips and reach at least
98% of the direct CCE implementation. Grouped input-scale broadcast and output
amax reduction should coexist in registers, and the compiler should reuse
predicates/layout conversions across adjacent vector chunks.
