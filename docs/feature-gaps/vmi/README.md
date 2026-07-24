# VMI / PTODSL feature gaps

Minimal reproducers for PTOAS capabilities that VMI/PTODSL authors need when
writing high-throughput **quantization** and **residual-mix** style vector
kernels. Each subdirectory documents one gap with:

| File | Role |
|---|---|
| `README.md` | Algorithm need, current failure, desired PTOAS behavior |
| `buggy_vmi.pto` | Compiling-today style workaround IR (illustrative) |
| `desired_vmi.pto` | Idiomatic VMI using the missing feature (may not lower today) |
| `target_mi.pto` | Desired VPTO / `pto.mi` shape after lowering |

These samples are documentation fixtures, not lit tests (no `// RUN:` lines).

## Gaps

| # | Gap | Why it matters |
|---|---|---|
| 01 | [Scalar GM dcache bypass](01_scalar_gm_dcache_bypass/) | Per-block quantization writes one f32 scale per 32×32 group to GM. Forcing bulk MTE for a few bytes burns bandwidth and sync. |
| 02 | [Compact inverse `vbrc`](02_compact_inv_vbrc/) | Quant kernels fan reciprocal scales across lanes. Padding to a full chunk before `dist_mode=brc` wastes UB and ops. |
| 03 | [Direct scalar reduce store](03_direct_scalar_reduce_store/) | Residual-mix kernels reduce a tile to one f32 scalar per group. Padding stores to 8 slots pollutes bandwidth and masks `1PT_B32`. |
| 04 | [Packed UE8M0 scale factors](04_packed_ue8m0_sf/) | Per-token MXFP8 uses UE8M0 scales with `pack_factor=2`. Unpacked `ui8` doubles SF traffic vs packed `ui16`. |
| 05 | [Persistent stages / `block_k`](05_persistent_stages_block_k/) | Persistent vector loops need multi-buffer stages and `block_k > 512` for long-K quant/cast tiles. Caps leave the pipe underfed. |

Closing these gaps improves authoring productivity (fewer workarounds in
PTODSL) and end-to-end performance of quantization + residual-mix kernels
without changing their algorithm.
