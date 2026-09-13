# Withdrawn from issue E

per_block fp32 `sf_only`+packed is **kernel-closed** on
`quant_missing_impl_0913` (bitwise vs ASC; TODO(perf)). Snapshot:
[`vmi_ue8m0_pack_from_fp32_amax`](../vmi_ue8m0_pack_from_fp32_amax/).

Recorded originally: launch OK, 1024-byte SF mismatch.
