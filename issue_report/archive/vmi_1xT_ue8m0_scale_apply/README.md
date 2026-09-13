# Withdrawn (old A) — UE8M0 1×T extract + scale apply

**Not a live PTOAS blocker.** TileKernels-vmi kernels no longer need
this IR for the Nightly configs that used to prove the hole.

PR90: `(32,1)` packed SF uses `dintlv` + one vector decode per
row-block, TMA occupancy / decode-once. Isolated representative
`cast_back` e4m3→fp32 TMA npt=32 512×128 (this dir’s `recorded.log`
507035) now runs bitwise, ratio **1.139**.

PR91: TMA-col in-SF uses `vgather` + in-register byte select (`<<7`),
not a scalar `size=1` unpack. ASC’s pattern is `vgather2`, not 1×T
`vload`.

Keep `desired_vmi.ptodsl.py` as history of the legal compact extract
PTOAS *could* lower. Do not treat remaining unisolated `cast_back`
507035, canonical `(1,32)` TODO(perf), or per_token row-rescale
compose as proof that this file still blocks a port.

Original recording: PTOAS `b465f26b` / CANN 9.2.0 / TileLang `5038c468`.

## High-level configs (stale recordings)

| Config | Then | Now (`pr_89_90_91_rebase`) |
|---|---|---|
| cast_back e4m3→fp32 TMA npt=32 H=128 | ACL 507035 | bitwise **1.139** (isolated) |
| per_channel TMA-in compose (cast_back TMA npt=1) | payload mismatch | canonical `(1,32)` still TODO(perf); not a PTOAS issue |
| per_token row-major rescale compose | SF off-by-one | kernel compose leftover; not remasured as PTOAS |

Nightly leftover snapshot: [`../cast_back_e4m3_fp32_tma_npt32_h128/`](../cast_back_e4m3_fp32_tma_npt32_h128/).
