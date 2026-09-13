# Still live issue D

Same low-level bug as **[vmi_unpacked_float_sf_move](../../vmi_unpacked_float_sf_move/)**:
unpacked e2m1 payload. Production still dies in the TileLang frontend:
packed FP4 `vstore` at 32 lanes is illegal (`1,2,4,8,64,128,256`).

`recorded.log` here is the old packed-UE8M0 assert. Current evidence is
`fp4_unpacked.err.txt` in the live D dir.
