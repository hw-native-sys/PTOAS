# Recorded device results

- PTOAS source commit: `e1c4fd9a`
- Includes related load-side fix: `823ec908`
- CANN: `9.1.0.beta3`
- Python environment: `cann91_dev`
- Device: Ascend 950DT, device 0
- Date: 2026-08-08 (Asia/Shanghai)

Run commands:

```bash
./scripts/check_compile.sh
./scripts/run_device.sh 0
```

## Compile results

All compile checks passed:

- Native ASC/CCE source and launch wrapper built and linked into a shared
  library with Bisheng.
- Desired compact VMI fixture emitted PTODSL MLIR and lowered through PTOAS to
  an object.
- Padded VMI workaround emitted PTODSL MLIR and lowered through PTOAS to an
  object.
- Emitted VPTO reproduces the compile-side defect: the explicit scalar load is
  `BRC_B32`, but `group=1` lowers to a dense `pto.vsts` with the all-lanes mask
  and no `1PT_B32` distribution.

The VMI device launch used the installed editable package/build for the same
source commit (`e1c4fd9a`).

## Isolated device results

| Process | Result |
|---|---|
| Native ASC/CCE baseline | **PASS**, output matches `input * 1.25 + 1.0` |
| Desired VMI `group=1, stride=1` | **FAIL**, exact device error `507035` at `torch.npu.synchronize()` |
| 64-lane padded VMI + SIMT extraction | **PASS**, output matches reference |

Exact desired-case error:

```text
RuntimeError: npuSynchronizeDevice: ... NPU function error:
device error type 3, error code is 507035
[ERROR] ... ERR00100 PTA call acl api failed
```

This confirms that commit `823ec908` fixes the related one-slot load-side
alignment issue but not the broadcast-result-to-compact-store layout
transition. The dense-store mislowering explains the device fault. The
arithmetic and regular 64-lane store path are healthy.
