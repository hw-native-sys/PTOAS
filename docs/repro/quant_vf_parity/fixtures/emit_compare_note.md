# Ask 3 — emit compare note (FP32 strip amax)

## Status

`current_vmi_fp32_strip_amax.pto` lowers (`vcmax`/`vstore` with `group = 1` →
`1PT_B32`). AscendC baseline `reference_asc_fp32_strip_amax.asc` compiles with
`bisheng`.

A side-by-side emit dump (AscendC assembly / MicroAPI vs VMI→VPTO) has **not**
yet shown a clear redundant MTE or layout tax that PTOAS must fix. Until that
evidence exists, Ask 3 is **not** a PTOAS legalization bug — it may be ISA /
schedule density on the AscendC side.

## How to extend this note

1. Lower with the check-script pass list (`pto-test-opt` → VPTO), or wrap the
   body in `pto.vecscope` and use `ptoas --emit-vpto`.
2. Compile `reference_asc_fp32_strip_amax.asc` and inspect the vector / MTE
   sequence.
3. List any extra VMI moves, spills, or layout reshapes here. Only then promote
   Ask 3 to a concrete PTOAS change request in `FEATURE_REQUEST.md`.
