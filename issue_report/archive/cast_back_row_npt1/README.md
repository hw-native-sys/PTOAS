# Kernel-fixed — not issue A

cast_back **row** npt=1 (e2m1/e4m3→bf16/fp32 @ 512×2048) is **not** a PTOAS blocker.

TileKernels-vmi PR78 (32B-align packed scale slots + `default_stages=1` on the tail schedule) makes these four rows bitwise vs ASC. They are TODO(perf) (ratios 0.36–0.87 on CANN 9.2.0, `BENCH_REP=5`). Historical mismatch log stays here.

TMA npt=1 mismatch and isolated ACL 507035 remain **[issue A](../../vmi_1xT_ue8m0_scale_apply/)**.
