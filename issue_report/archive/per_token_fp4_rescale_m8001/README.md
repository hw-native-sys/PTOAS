# Withdrawn from issue C

Fused rescale M=8001 is **kernel-closed** on `quant_missing_impl_0913`
(i8 `vbrc(0)` + `vinterpret` to fp8/fp4, no host-pad). Snapshot of the
old compiler hole: [`vmi_fp8_vbrc_clear_tail`](../vmi_fp8_vbrc_clear_tail/).

Recorded originally: `vbrc(f8e4m3(0))` TypeError; `T.clear` e4m3 Bad bit-width.
