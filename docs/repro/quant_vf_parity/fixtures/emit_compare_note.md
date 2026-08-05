# Feature request 3 — AscendC vs VMI emit comparison (FP32 strip abs-max)

## What to compare

| Side | Fixture | Role in the comparison |
|------|---------|------------------------|
| AscendC | `reference_asc_fp32_strip_amax.asc` | Dense MicroAPI reference (two loads, abs, `vmax`, store) |
| VMI | `current_vmi_fp32_strip_amax.pto` | Same algorithm as VMI: abs + `vcmax` with `group = 1` |

## Status today

- AscendC fixture compiles with `bisheng`.
- VMI fixture lowers with `pto-test-opt` (check-script pass list).
- A side-by-side dump of **AscendC generated code** vs **VMI→VPTO generated
  code** has not yet shown clear extra memory traffic or layout reshapes on the
  VMI side. Until that is written here, feature request 3 is an open
  performance investigation, not a proven PTOAS legalization bug.

## How to fill this note

1. Lower `current_vmi_fp32_strip_amax.pto` (check-script passes, or wrap the
   body in `pto.vecscope` and run `ptoas --emit-vpto`).
2. Compile `reference_asc_fp32_strip_amax.asc` and inspect its vector / memory
   sequence.
3. List below any extra VMI moves, spills, or layout reshapes relative to
   AscendC. Only then ask for a concrete PTOAS code change in
   `FEATURE_REQUEST.md`.

## Findings

_(none yet)_
