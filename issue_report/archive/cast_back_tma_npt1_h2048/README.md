# Kernel-fixed — not issue A

cast_back **TMA** npt=1 H=2048 (e2m1/e4m3→bf16) is **not** a PTOAS blocker.

TileKernels-vmi TMA/non-canonical 32B scale slots (`benchfix_0911` / Codex `5b4bb1f3d`) make these rows bitwise vs ASC. They are **TODO(perf)** (this-host ratios ~0.030–0.033, `BENCH_REP=20`). Correctness is closed; the slow ratio is still an issue to solve. Historical mismatch log stays here.

Remaining isolated ACL 507035 and the H=128 fp32 TMA representative stay **[issue A](../../vmi_1xT_ue8m0_scale_apply/)**.
