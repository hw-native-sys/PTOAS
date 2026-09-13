# Withdrawn (old F) — Persistent last-wave remainder

**Not a live PTOAS blocker.** The 7 fused per_channel TMA-col
large-shape payload+SF mismatches were a scalar `size=1` unpack, not
last-wave lowering.

PR91 replaced that unpack with ASC-form `vgather` + in-register `<<7`.
On `pr_89_90_91_rebase` those 7 rows are bitwise (`error=None`): one
ready (8064×2560 **1.056**), six TODO(perf) 0.82–0.96.

`desired_vmi.ptodsl.py` is still a 1×T remainder-wave sketch. Production
kernels no longer emit it. Do not keep F on that micro-repro alone.

Original recording: PTOAS `b465f26b` / CANN 9.2.0 / TileLang `5038c468`.
Isolated fused 512×7168 / 8064×2048 mismatch is in
[`../per_channel_tma_in_large_shape/recorded.log`](../per_channel_tma_in_large_shape/recorded.log)
(stale).

One leftover ACL 507035 (per_channel TMA-col 8064×768) was not in this
set and is unisolated — not a PTOAS issue on that evidence.
