# 01 — Scalar GM dcache-bypass store/load

## Algorithm need

Per-block quantization over groups of shape **32×32** produces one f32 scale
factor (SF) per group. After the scale is computed in UB, that single scalar
must be written to GM (and later re-read) so consumers see a dense SF tensor.

## Current failure

PTOAS does not expose a scalar GM path with dcache bypass for these tiny
writes/reads. Authors stage the SF in UB and issue a bulk MTE
(`pto.copy_ubuf_to_gm` / `pto.copy_gm_to_ubuf`) even when `len_burst` is only a
few bytes. That adds MTE setup, alignment padding, and pipeline sync out of
proportion to the payload.

## Desired PTOAS behavior

Allow VMI/PTODSL to store or load a scalar (or compact slots=1 vector) directly
against `!pto.ptr<T, gm>` with an explicit **dcache-bypass** attribute, lowering
to a single GM scalar path without a bulk MTE burst.
