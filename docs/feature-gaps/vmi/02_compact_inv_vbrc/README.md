# 02 — Compact inverse `vbrc` (no pad + `BRC`)

## Algorithm need

Quantization kernels compute per-group reciprocal scale factors, then broadcast
each inverse across the lanes of its group before multiplying the activation
tile. Authors want a **compact** inverse vector (one slot per group) fanned
out with `pto.vmi.vbrc` / a `BRC` load.

## Current failure

VPTO lowering expects reciprocal scales to be materialized as a **padded
full-chunk** layout and reloaded with `dist_mode = "brc"`. Compact inverse
`vbrc` (e.g. 8 valid slots) does not lower cleanly, so kernels pad UB and pay
extra stores plus a broadcast reload.

## Desired PTOAS behavior

Lower compact inverse `vbrc` (or a slots-aligned `vload {dist_mode = "brc"}`)
directly to `BRC_B*` without requiring pad-to-full-chunk waste in UB.
