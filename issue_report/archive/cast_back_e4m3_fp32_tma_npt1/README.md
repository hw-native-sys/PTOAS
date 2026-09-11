# Kernel-fixed — not issue A

cast_back e4m3→fp32 **TMA** npt=1 is **not** a PTOAS blocker.

TileKernels-vmi TMA/non-canonical 32B scale slots (`benchfix_0911` / Codex `5b4bb1f3d`) make this row bitwise vs ASC. It is **TODO(perf)** (this-host ratio ~0.096, `BENCH_REP=20`). Correctness is closed; the slow ratio is still an issue to solve. Historical mismatch / 507035 log stays here.

Remaining isolated ACL 507035 and the H=128 fp32 TMA representative stay **[issue A](../../vmi_1xT_ue8m0_scale_apply/)**.
