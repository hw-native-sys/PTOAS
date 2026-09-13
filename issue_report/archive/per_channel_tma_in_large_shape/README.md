# Kernel-fixed fused leftover (old A compose / F fused)

per_channel e4m3 rescale TMA-col in-SF, 512×7168 / 8064×2048 and the
other five large-tile INFO rows.

- **Fused** payload+SF mismatch: closed by PR91 `vgather` (bitwise;
  1 ready, 6 TODO(perf)). Old report:
  [`vmi_persistent_last_wave`](../vmi_persistent_last_wave/).
- **Compose** (cast_back TMA npt=1): old **A**, also withdrawn. Canonical
  `(1,32)` leftover is kernel TODO(perf).

`recorded.log` is the old mismatch. Not a live PTOAS issue.
