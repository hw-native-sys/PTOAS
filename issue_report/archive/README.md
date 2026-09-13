# Archive — leftover-class snapshots and withdrawn issues

These directories are **not** live PTOAS issues. They keep recorded
logs, dumps, and pointer READMEs.

Live issues are **B–E** one level up. **A** and **F** were withdrawn
after TileKernels-vmi PRs 89–91: those Nightly configs have kernel
workarounds (`dintlv` / decode-once / TMA-col `vgather`).

| Archive dir | Status |
|---|---|
| `vmi_1xT_ue8m0_scale_apply` (old **A**) | withdrawn — kernel workaround (PR90/PR91); desired 1×T IR is not a current Nightly blocker |
| `vmi_persistent_last_wave` (old **F**) | withdrawn — 7 fused TMA-col mismatches now bitwise after PR91 gather |
| `cast_back_e4m3_fp32_tma_npt32_h128` | kernel-fixed on isolate (PR90 remasure bitwise 1.139); was the A representative 507035 |
| `cast_back_row_npt1` | kernel-fixed PR78 (bitwise, TODO(perf)) |
| `cast_back_tma_npt1_h2048`, `cast_back_e4m3_fp32_tma_npt1`, `cast_back_e2m1_fp32_tma_npt32_h2048` | kernel-fixed 32B slots (`benchfix_0911`); bitwise, TODO(perf) |
| `per_channel_tma_in_large_shape` | fused half kernel-fixed PR91 (bitwise); compose was A, also withdrawn |
| `per_token_rescale_row_sf` | kernel compose leftover; not a live PTOAS issue |
| `per_token_h128_h384`, `per_block_h384` | live **[B](../vmi_compact_v128_residual/)** |
| `per_token_fp4_rescale_m8001` | live **[C](../vmi_fp8_vbrc_clear_tail/)** |
| `per_token_tma_unpacked`, `per_token_fp4_unpacked`, `per_channel_rescale_unpacked_in` | live **[D](../vmi_unpacked_float_sf_move/)** |
| `per_token_sf_only_packed`, `per_block_sf_only_packed_fp32` | live **[E](../vmi_ue8m0_pack_from_fp32_amax/)** |
