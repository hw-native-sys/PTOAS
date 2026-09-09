# Moved to issue A

Same low-level bug as **[vmi_1xT_ue8m0_scale_apply](../../vmi_1xT_ue8m0_scale_apply/)**.

Representative config: cast_back row npt=1, e2m1/e4m3→bf16/fp32 @ 512×2048. Recorded: payload mismatch (or 507035 in a dirty session). `recorded.log` is kept here.
