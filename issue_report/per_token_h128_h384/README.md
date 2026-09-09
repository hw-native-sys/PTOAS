# Moved to issue B

Same low-level bug as **[vmi_compact_v128_residual](../vmi_compact_v128_residual/)**: legal `V<128×T>` strip (ASC `hidden % 128`).

Recorded: assert `hidden % 256 == 0`. `recorded.log` is kept here.
