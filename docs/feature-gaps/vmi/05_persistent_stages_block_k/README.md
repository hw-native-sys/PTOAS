# 05 — Persistent multi-buffer stages and `block_k > 512`

## Algorithm need

Long-K quantization / cast-back vector kernels run as **Persistent** loops over
the K dimension. Authors want multi-buffer software pipeline stages (double /
triple buffer) and tile sizes with **`block_k > 512`** so MTE and vector pipes
stay overlapped on large hidden dimensions.

## Current failure

Practical PTOAS acceptance for these Persistent VMI modules is capped at
**`stages = 1`** and **`block_k ≤ 512`**. Raising either is rejected or
miscompiles, so kernels cannot express a legal multi-buffer Persistent schedule.

## Desired PTOAS behavior

Accept Persistent vector modules with `stages >= 2` and `block_k > 512`, and
lower them to a correct multi-buffer VPTO schedule (distinct UB stage buffers +
pipeline sync) without forcing the single-stage / K≤512 workaround.
