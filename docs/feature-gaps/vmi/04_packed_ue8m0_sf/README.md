# 04 — Packed UE8M0 scale factors (`pack_factor=2`)

## Algorithm need

Per-token MXFP8 quantization stores UE8M0 scale factors with **`pack_factor=2`**:
two consecutive `ui8` exponents share one `ui16` element in GM/UB. This halves
SF bandwidth versus an unpacked `ui8` tensor.

## Current failure

VMI/PTODSL paths that try to pack adjacent UE8M0 bytes (reinterpret / element
pack into `ui16`) do not lower reliably. Authors keep **unpacked `ui8` SF
only**, doubling scale-factor traffic on the memory pipe.

## Desired PTOAS behavior

Provide a first-class pack path so a logical `ui8` UE8M0 vector lowers through
`PK` (or equivalent) into packed `ui16` storage with `pack_factor=2`, and the
inverse unpack on load.
