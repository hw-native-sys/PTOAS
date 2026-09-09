# Split: compose → A, fused last-wave → F

- **Compose** (cast_back TMA npt=1) is **[vmi_1xT_ue8m0_scale_apply](../../vmi_1xT_ue8m0_scale_apply/)** (issue A).
- **Fused** remainder / periodic wave is **[vmi_persistent_last_wave](../../vmi_persistent_last_wave/)** (issue F).

Representative config: per_channel e4m3 rescale TMA-col input SF, 512×7168 / 8064×2048. `recorded.log` is kept here.
