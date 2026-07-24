# 03 — Direct scalar reduce store (`1PT_B32`)

## Algorithm need

Residual-mix kernels (MHC-style) accumulate a tile and reduce to **one f32
scalar per group** (mix gradient / residual coefficient). That compact result
must be stored densely to UB/GM for the next stage.

## Current failure

Practical VMI authoring today pads the reduce result to **8 slots** (or
broadcasts with `vbrc` sized for 8) before store, because a true slots=1 /
single-scalar store path is unreliable or rejected. The pad tax shows up as
extra UB traffic and prevents a clean `1PT_B32` store.

## Desired PTOAS behavior

`vcadd` / `group_reduce_addf` producing `V<1×f32>` or slots=1 compact vectors
must lower to a direct `pto.vsts {dist = "1PT_B32"}` (or equivalent) with no
mandatory pad-to-8 store.
