# Feature request — compact recip / inverse expand without pad-32 + `brc`

Audience: PTOAS maintainers.

## Problem

Quantize-style kernels compute a small number of per-strip reciprocal (or inverse)
values, then multiply them across a wide data strip.

On AscendC this stays in vector registers: a few `inverse[]` locals are reused
for the strip multiply. No padded UB store is required.

On VMI today, the working path must:

1. group-store those values into a **32-element padded** UB slot, then
2. reload with `vload(..., dist_mode="brc", group=G)`.

A compact in-register expand (`vbrc` with `group=G` from a G-lane group-slot
source onto the strip) does **not** legalize. That forces extra UB traffic on
every strip.

Representative shape in this package: **G=4 f32 → expand across 128 lanes**.
The same algorithm appears as G=8 bf16 → 256 lanes.

## What AscendC already does

See [`fixtures/reference_asc_cce.asc`](fixtures/reference_asc_cce.asc).

After forming inverse values, they live in VF locals (`vector_f32 inverse[...]`)
and are used directly in the strip multiply. Compiles with `bisheng`.

## What VMI does today (works, not ideal)

See [`fixtures/current_pad_brc_vmi.pto`](fixtures/current_pad_brc_vmi.pto)
(and the DSL twin `.py`).

- `vstore(..., group=4)` into a pad-32 layout
- `vload(..., dist_mode="brc", group=4)` back to 128 lanes
- then `vmul`

This **lowers** to VPTO on the current tip. It is the workaround, not the
desired shape.

## What we need

See [`fixtures/desired_compact_vmi.pto`](fixtures/desired_compact_vmi.pto).

Allow expanding a G-lane group-slot recip/inverse vector across the strip
**without** a pad-to-32 store and padded `dist_mode="brc"` reload — for example
legalizing:

```text
vbrc %recip {group = G} : Gxf32 (group_slots) -> stripxf32
```

so the strip multiply can use the expanded vector in-register (matching the
AscendC local-inverse pattern).

## Criteria

1. [`desired_compact_vmi.pto`](fixtures/desired_compact_vmi.pto) lowers with
   `pto-test-opt` VMI→VPTO (no residual `group_broadcast` /
   `VMI-UNSUPPORTED`).
2. Until then, [`current_pad_brc_vmi.pto`](fixtures/current_pad_brc_vmi.pto)
   remains a valid working path (no forced breakage of the pad+`brc` shape).

## Non-goals

- SoftPipeline stage count or scheduling policy
- Unrelated integer widen / cast legalization bugs
- Simulator-only cycle claims as the sole proof (device bandwidth and
  compile/lower evidence in this package are enough for the request)
