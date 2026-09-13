# Kernel-fixed — not issue A

cast_back e2m1→fp32 TMA npt=32 @ 512×2048 is **not** a PTOAS blocker.

TileKernels-vmi TMA/non-canonical 32B scale slots (`benchfix_0911` / Codex `5b4bb1f3d`) make this row bitwise vs ASC. It is **TODO(perf)** (this-host ratio ~0.079, `BENCH_REP=20`). Correctness is closed; the slow ratio is still an issue to solve. Historical mismatch log stays here.

The H=128 fp32 TMA representative is kernel-fixed on isolate (PR90, bitwise 1.139). Old issue **A** is withdrawn ([`vmi_1xT_ue8m0_scale_apply`](../vmi_1xT_ue8m0_scale_apply/)). Unisolated 507035 is not a PTOAS issue.
