# Grouped scale pipeline is substantially slower than the direct vector sequence

## Summary

This standalone case reduces a common vector pattern: load 256 BF16 values,
compute eight independent maxima, broadcast eight scale values back over the
256 lanes, multiply, convert to FP8, and store. The VMI spelling is compact and
correct, but end-to-end kernels dominated by this pattern remain well below the
throughput of an equivalent direct CCE implementation.

The fixture contains no framework dependency. It is a plain PTO/VMI module and
the reference is a small CCE-style instruction sketch.

## Reproduce

```bash
bash docs/repro/grouped_scale_pipeline/check.sh
```

The check lowers `fixtures/grouped_scale_vmi.pto` to VPTO and confirms that the
group reduction and group broadcast survive as the expected low-level vector
operations. On A5, wrap the same body in a persistent GM-to-UB/UB-to-GM loop and
compare it with `fixtures/reference_cce.cpp` using identical buffers and event
timing.

## Observed behavior

Across both small latency-bound tiles and large bandwidth-bound grids, kernels
whose hot body is this pattern remain measurably behind the direct sequence.
Padding or changing the outer grid helps occupancy but does not close the gap.

## Requested behavior

Please provide a lowering/scheduling path whose generated code reaches at least
98% of the direct CCE reference throughput. In particular, grouped reduction
results should remain in registers through broadcast and conversion, with no
avoidable layout materialization, UB spill/reload, or redundant predicate work.
