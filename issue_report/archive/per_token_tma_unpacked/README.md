# Still live issue D

Same low-level bug as **[vmi_unpacked_float_sf_move](../../vmi_unpacked_float_sf_move/)**:
on-device unpacked float SF `vstore` (TMA-col is compiler layout, not a host
permute). Production IR now emits; isolated remasure is a **48-byte SF
mismatch** vs ASC (not a packed-only assert).

`recorded.log` here is the old assert. Current evidence is in the live D dir.
