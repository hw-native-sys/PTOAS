# Archive — leftover-class snapshots and withdrawn issues

These directories are **not** live PTOAS issues. They keep recorded
logs, dumps, and pointer READMEs.

Live issues are **B** and **D** one level up. **A, C, E, F** were
withdrawn after TileKernels-vmi PRs 89–91 and `quant_missing_impl_0913`.

| Archive dir | Status |
|---|---|
| `vmi_1xT_ue8m0_scale_apply` (old **A**) | withdrawn — kernel workaround (PR90/PR91) |
| `vmi_fp8_vbrc_clear_tail` (old **C**) | withdrawn — i8 zero + `vinterpret` (0913); fused M=8001 bitwise at H=3072/16384 |
| `vmi_ue8m0_pack_from_fp32_amax` (old **E**) | withdrawn — fp32 pack + per_token `sf_only` packed bitwise (0913) |
| `vmi_persistent_last_wave` (old **F**) | withdrawn — 7 fused TMA-col mismatches now bitwise after PR91 gather |
| `cast_back_e4m3_fp32_tma_npt32_h128` | kernel-fixed on isolate (PR90 remasure bitwise 1.139) |
| `cast_back_row_npt1` | kernel-fixed PR78 (bitwise, TODO(perf)) |
| `cast_back_tma_npt1_h2048`, `cast_back_e4m3_fp32_tma_npt1`, `cast_back_e2m1_fp32_tma_npt32_h2048` | kernel-fixed 32B slots (`benchfix_0911`); bitwise, TODO(perf) |
| `per_channel_tma_in_large_shape` | fused half kernel-fixed PR91 (bitwise); compose was A, also withdrawn |
| `per_token_rescale_row_sf` | kernel compose leftover; closed bitwise 0913 |
| `per_block_h384` | withdrawn **B** — dual `V<64>` bitwise 0913 |
| `per_token_h128_h384` | unpacked compact withdrawn; **packed** still live **[B](../vmi_compact_v128_residual/)** |
| `per_token_fp4_rescale_m8001` | withdrawn **C** |
| `per_token_tma_unpacked`, `per_token_fp4_unpacked` | live **[D](../vmi_unpacked_float_sf_move/)** |
| `per_channel_rescale_unpacked_in` | withdrawn **D** — compose bitwise 0913 |
| `per_token_sf_only_packed`, `per_block_sf_only_packed_fp32` | withdrawn **E** |
