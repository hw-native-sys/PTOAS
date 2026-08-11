# Grouped reduce/scale/convert pipeline is slower through VMI

This is a framework-free A5 reproducer for a vector pattern that reduces 256
BF16 lanes into eight group maxima, expands the eight values, scales the input,
and converts it to FP8.  Both sides are complete GM-to-UB-to-GM kernels:

- `fixtures/grouped_scale_vmi.pto` is compiled by stock PTOAS/VMI.
- `fixtures/reference_cce.cpp` is the direct CCE peer and keeps the grouped
  values in vector registers through E2B expansion and packed conversion.

Run `bash check.sh compile` to compile both sides and inspect the emitted VPTO.
Run `bash check.sh benchmark` to print the pinned A5 event medians that selected
this reduction, or `bash check.sh` for both.  Generated files go under
`outputs/`.

| Representative case | ASC us | VMI us | ASC/VMI |
|---|---:|---:|---:|
| small FP32, groups of 32 | 5.5633 | 6.2189 | 0.8946 |
| ragged BF16, groups of 32 | 34.6239 | 67.5086 | 0.5129 |
| large BF16, groups of 32 | 507.5570 | 1810.7665 | 0.2803 |

The numbers are medians from the pinned A5 environment and are data rather
than synthetic estimates. Device load can move absolute time; parity is defined
as `ASC_us / VMI_us >= 0.98`. The reduced 256-lane fixture is intentionally
small enough for compiler review while retaining the same reduction,
broadcast, and conversion chain.

Requested behavior: retain group results and predicates across broadcast and
conversion, avoid layout materialization and UB spill/reload, and reach at
least 98% of the direct CCE throughput.
