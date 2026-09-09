# Moved to issue C

Same low-level bug as **[vmi_fp8_vbrc_clear_tail](../vmi_fp8_vbrc_clear_tail/)**: fp8 `vbrc` / `T.clear` for an in-kernel ceildiv tail (no host pad).

Recorded: `vbrc(f8e4m3(0))` TypeError; `T.clear` e4m3 Bad bit-width. `recorded.log` is kept here.
