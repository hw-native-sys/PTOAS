# Moved to issue A

Compose calls `cast_back` then bf16 per_token (`_compose_per_token_rescale`). Same low-level bug as **[vmi_1xT_ue8m0_scale_apply](../vmi_1xT_ue8m0_scale_apply/)**.

Recorded: fused path rejects; compose SF off-by-one vs ASC fused. `recorded.log` is kept here.
