# Archive — leftover-class snapshots

These directories are **not** PTOAS issues. They are the old 1:1 Nightly leftover-class folders (recorded logs, dumps, pointer READMEs).

The live issues are A–F one level up:

| Archive dir | Issue |
|---|---|
| `cast_back_*` except `cast_back_row_npt1`, `cast_back_tma_npt1_h2048`, `cast_back_e4m3_fp32_tma_npt1`, `cast_back_e2m1_fp32_tma_npt32_h2048`, `per_token_rescale_row_sf`, `per_channel_tma_in_large_shape` (compose) | [A](../vmi_1xT_ue8m0_scale_apply/) |
| `cast_back_row_npt1` | kernel-fixed by TileKernels-vmi PR78; not a PTOAS issue (TODO(perf) still open) |
| `cast_back_tma_npt1_h2048`, `cast_back_e4m3_fp32_tma_npt1`, `cast_back_e2m1_fp32_tma_npt32_h2048` | kernel-fixed TMA/non-canonical 32B slots (`benchfix_0911` / `5b4bb1f3d`); bitwise, **TODO(perf)** still open |
| `per_token_h128_h384`, `per_block_h384` | [B](../vmi_compact_v128_residual/) |
| `per_token_fp4_rescale_m8001` | [C](../vmi_fp8_vbrc_clear_tail/) |
| `per_token_tma_unpacked`, `per_token_fp4_unpacked`, `per_channel_rescale_unpacked_in` | [D](../vmi_unpacked_float_sf_move/) |
| `per_token_sf_only_packed`, `per_block_sf_only_packed_fp32` | [E](../vmi_ue8m0_pack_from_fp32_amax/) |
| `per_channel_tma_in_large_shape` (fused) | [F](../vmi_persistent_last_wave/) |
